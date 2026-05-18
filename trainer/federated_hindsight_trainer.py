from __future__ import annotations

"""Federated training utilities built on top of the hindsight training stack.

This implementation keeps one persistent worker per client so that each
client keeps its own environment, replay buffer, and trainer alive across
rounds. The main process only coordinates:
- broadcasting the current global weights
- requesting one local round from each client
- averaging returned weights with FedAvg

That avoids re-loading the road network and re-creating `RealTrafficEnv`
for every round.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import copy
import os
import multiprocessing as mp
from collections import deque
from multiprocessing.connection import Connection

import torch

from agents.hindsight_dqn_agent import HindsightDQNAgent
from env.real_env import RealTrafficEnv
from trainer.trainer import HindsightTrainer


@dataclass
class FederatedClientConfig:
    client_name: str
    graphml_file: str
    num_stations: int
    num_evs: int
    num_chargers_per_station: int = 8
    respawn_after_full_charge: bool = True
    seed: int = 42
    max_nodes: int = 1_000_000
    cache_dir: str | None = None
    network_variant: str = "station_only"
    use_action_mask: bool = True
    batch_size: int = 64
    steps_per_episode: int = 100


@dataclass
class FederatedClient:
    config: FederatedClientConfig
    env: RealTrafficEnv
    agent: HindsightDQNAgent
    trainer: HindsightTrainer


@dataclass
class ClientRoundResult:
    client_name: str
    state_dict: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    memory: list


@dataclass
class WorkerResponse:
    client_name: str
    state_dict: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    memory: list


@dataclass
class WorkerCommand:
    kind: str
    payload: dict | None = None


class _PersistentClientWorker(mp.Process):
    def __init__(self, cfg: FederatedClientConfig, conn: Connection):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.conn = conn

    def _build_runtime(self):
        env = RealTrafficEnv(
            graphml_file=self.cfg.graphml_file,
            num_stations=self.cfg.num_stations,
            num_evs=self.cfg.num_evs,
            num_chargers_per_station=self.cfg.num_chargers_per_station,
            max_nodes=self.cfg.max_nodes,
            cache_dir=self.cfg.cache_dir,
            seed=self.cfg.seed,
            respawn_after_full_charge=self.cfg.respawn_after_full_charge,
            client_name=self.cfg.client_name,
        )
        station_node_ids = [s.traffic_node_id for s in env.stations]
        agent = HindsightDQNAgent(
            num_features=18,
            num_actions=self.cfg.num_stations,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=env.num_nodes,
            network_variant=self.cfg.network_variant,
            use_action_mask=self.cfg.use_action_mask,
        )
        trainer = HindsightTrainer(env, agent)
        return env, agent, trainer

    @staticmethod
    def _serialize_memory(memory: deque) -> list:
        return list(memory)

    @staticmethod
    def _deserialize_memory(memory_list: list | None, maxlen: int) -> deque:
        if memory_list is None:
            return deque(maxlen=maxlen)
        return deque(memory_list, maxlen=maxlen)

    @staticmethod
    def _cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def _run_local_round(
        self,
        env: RealTrafficEnv,
        agent: HindsightDQNAgent,
        trainer: HindsightTrainer,
        batch_size: int,
        steps_per_episode: int,
        initial_memory: list | None,
    ) -> WorkerResponse:
        agent.memory = self._deserialize_memory(initial_memory, agent.memory.maxlen)
        env.reset()
        trainer.pending.clear()
        trainer._current_step = 0

        ep_trip: list[float] = []
        ep_queue: list[float] = []
        ep_fee: list[float] = []
        ep_reward: list[float] = []

        for _step in range(steps_per_episode):
            done, info = trainer.step_episode()

            for entry in info.get("completed", []):
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                reward = -(0.3 * trip + 0.5 * queue + 0.03 * fee)
                ep_trip.append(trip)
                ep_queue.append(queue)
                ep_fee.append(fee)
                ep_reward.append(reward)

            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

            if done:
                break

        metrics = {
            "episodes": 1.0,
            "memory_size": float(len(agent.memory)),
            "avg_trip_h": float(sum(ep_trip) / len(ep_trip)) if ep_trip else 0.0,
            "avg_queue_h": float(sum(ep_queue) / len(ep_queue)) if ep_queue else 0.0,
            "avg_fee": float(sum(ep_fee) / len(ep_fee)) if ep_fee else 0.0,
            "avg_reward": float(sum(ep_reward) / len(ep_reward)) if ep_reward else 0.0,
        }
        return WorkerResponse(
            client_name=self.cfg.client_name,
            state_dict=self._cpu_state_dict(agent.policy_net.state_dict()),
            metrics=metrics,
            memory=self._serialize_memory(agent.memory),
        )

    def run(self):
        env = agent = trainer = None
        try:
            env, agent, trainer = self._build_runtime()
            while True:
                cmd: WorkerCommand = self.conn.recv()
                if cmd.kind == "stop":
                    self.conn.send(None)
                    break
                if cmd.kind == "round":
                    payload = cmd.payload or {}
                    if "global_state" in payload and payload["global_state"] is not None:
                        agent.policy_net.load_state_dict(payload["global_state"], strict=False)
                        agent.target_net.load_state_dict(agent.policy_net.state_dict())
                    resp = self._run_local_round(
                        env=env,
                        agent=agent,
                        trainer=trainer,
                        batch_size=int(payload.get("batch_size", 64)),
                        steps_per_episode=int(payload.get("steps_per_episode", 100)),
                        initial_memory=payload.get("memory"),
                    )
                    self.conn.send(resp)
                    continue
                raise ValueError(f"Unknown command kind: {cmd.kind!r}")
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                del trainer
                del agent
                del env


class FederatedHindsightTrainer:
    """Minimal FedAvg trainer for multiple client-specific EMA power grids."""

    def __init__(self, client_configs: Sequence[FederatedClientConfig], parallel: bool = True):
        if not client_configs:
            raise ValueError("client_configs must not be empty")
        self.client_configs = list(client_configs)
        self.parallel = bool(parallel)
        self.clients: list[FederatedClient] = [self._build_client(cfg) for cfg in self.client_configs]
        self.client_memory_cache: dict[str, list] = {cfg.client_name: [] for cfg in self.client_configs}
        self._workers: dict[str, _PersistentClientWorker] = {}
        self._worker_conns: dict[str, Connection] = {}
        if self.parallel and len(self.client_configs) > 1:
            self._start_workers()

    def _build_client(self, cfg: FederatedClientConfig) -> FederatedClient:
        env = RealTrafficEnv(
            graphml_file=cfg.graphml_file,
            num_stations=cfg.num_stations,
            num_evs=cfg.num_evs,
            num_chargers_per_station=cfg.num_chargers_per_station,
            max_nodes=cfg.max_nodes,
            cache_dir=cfg.cache_dir,
            seed=cfg.seed,
            respawn_after_full_charge=cfg.respawn_after_full_charge,
            client_name=cfg.client_name,
        )
        station_node_ids = [s.traffic_node_id for s in env.stations]
        agent = HindsightDQNAgent(
            num_features=18,
            num_actions=cfg.num_stations,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=env.num_nodes,
            network_variant=cfg.network_variant,
            use_action_mask=cfg.use_action_mask,
        )
        trainer = HindsightTrainer(env, agent)
        return FederatedClient(cfg, env, agent, trainer)

    def _start_workers(self) -> None:
        ctx = mp.get_context("spawn")
        for cfg in self.client_configs:
            parent_conn, child_conn = ctx.Pipe()
            worker = _PersistentClientWorker(cfg, child_conn)
            worker.start()
            self._workers[cfg.client_name] = worker
            self._worker_conns[cfg.client_name] = parent_conn

    def _stop_workers(self) -> None:
        for name, conn in list(self._worker_conns.items()):
            try:
                conn.send(WorkerCommand(kind="stop"))
                _ = conn.recv()
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        for worker in list(self._workers.values()):
            try:
                worker.join(timeout=5)
            except Exception:
                pass
        self._workers.clear()
        self._worker_conns.clear()

    def __del__(self):
        try:
            self._stop_workers()
        except Exception:
            pass

    @staticmethod
    def _fedavg_state_dict(state_dicts: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not state_dicts:
            raise ValueError("state_dicts must not be empty")
        avg_state = {}
        keys = state_dicts[0].keys()
        for key in keys:
            tensors = [sd[key].detach().cpu() for sd in state_dicts]
            first = tensors[0]
            if first.dtype.is_floating_point or first.dtype.is_complex:
                avg_state[key] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                avg_state[key] = first.clone()
        return avg_state

    def _sync_global_weights_to_clients(self, global_state: Dict[str, torch.Tensor]) -> None:
        for client in self.clients:
            client.agent.policy_net.load_state_dict(global_state, strict=False)
            client.agent.target_net.load_state_dict(client.agent.policy_net.state_dict())

    def _train_round_serial(self) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Dict[str, float]]]:
        client_states = []
        metrics: Dict[str, Dict[str, float]] = {}
        for client in self.clients:
            result = self._run_local_round_local_process(client)
            client_states.append(result.state_dict)
            metrics[result.client_name] = result.metrics
            self.client_memory_cache[result.client_name] = result.memory
            client.agent.memory = deque(result.memory, maxlen=client.agent.memory.maxlen)
        return client_states, metrics

    def _run_local_round_local_process(self, client: FederatedClient) -> ClientRoundResult:
        # Reuse the same code path as the persistent worker, but in-process.
        worker = _PersistentClientWorker(client.config, conn=None)  # type: ignore[arg-type]
        env = client.env
        agent = client.agent
        trainer = client.trainer
        agent.memory = deque(self.client_memory_cache.get(client.config.client_name, []), maxlen=agent.memory.maxlen)
        env.reset()
        trainer.pending.clear()
        trainer._current_step = 0

        ep_trip: list[float] = []
        ep_queue: list[float] = []
        ep_fee: list[float] = []
        ep_reward: list[float] = []

        for _step in range(client.config.steps_per_episode):
            done, info = trainer.step_episode()
            for entry in info.get("completed", []):
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                reward = -(0.3 * trip + 0.5 * queue + 0.03 * fee)
                ep_trip.append(trip)
                ep_queue.append(queue)
                ep_fee.append(fee)
                ep_reward.append(reward)
            if len(agent.memory) >= client.config.batch_size:
                agent.replay(client.config.batch_size)
            if done:
                break

        metrics = {
            "episodes": 1.0,
            "memory_size": float(len(agent.memory)),
            "avg_trip_h": float(sum(ep_trip) / len(ep_trip)) if ep_trip else 0.0,
            "avg_queue_h": float(sum(ep_queue) / len(ep_queue)) if ep_queue else 0.0,
            "avg_fee": float(sum(ep_fee) / len(ep_fee)) if ep_fee else 0.0,
            "avg_reward": float(sum(ep_reward) / len(ep_reward)) if ep_reward else 0.0,
        }
        state_dict = {key: value.detach().cpu().clone() for key, value in agent.policy_net.state_dict().items()}
        return ClientRoundResult(client.config.client_name, state_dict, metrics, list(agent.memory))

    def _train_round_parallel(self) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Dict[str, float]]]:
        # Persistent workers are already running; send them a round command and wait.
        client_states: list[Dict[str, torch.Tensor]] = []
        metrics: Dict[str, Dict[str, float]] = {}
        for client in self.clients:
            conn = self._worker_conns[client.config.client_name]
            conn.send(
                WorkerCommand(
                    kind="round",
                    payload={
                        "global_state": client.agent.policy_net.state_dict(),
                        "memory": self.client_memory_cache.get(client.config.client_name, []),
                        "batch_size": client.config.batch_size,
                        "steps_per_episode": client.config.steps_per_episode,
                    },
                )
            )
        for client in self.clients:
            conn = self._worker_conns[client.config.client_name]
            result: WorkerResponse = conn.recv()
            client_states.append(result.state_dict)
            metrics[result.client_name] = result.metrics
            self.client_memory_cache[result.client_name] = result.memory
            client.agent.memory = deque(result.memory, maxlen=client.agent.memory.maxlen)
        return client_states, metrics

    def train_round(self) -> Dict[str, Dict[str, float]]:
        if self.parallel and len(self.client_configs) > 1 and self._worker_conns:
            client_states, metrics = self._train_round_parallel()
        else:
            client_states, metrics = self._train_round_serial()

        global_state = self._fedavg_state_dict(client_states)
        self._sync_global_weights_to_clients(global_state)
        return metrics

    def save_global_models(self, save_dir: str, round_idx: int | None = None) -> None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"round{round_idx}" if round_idx is not None else "final"
        for client in self.clients:
            path = os.path.join(save_dir, f"{client.config.client_name}_{suffix}.pth")
            client.agent.save_model(path)

    def client_names(self) -> List[str]:
        return [client.config.client_name for client in self.clients]
