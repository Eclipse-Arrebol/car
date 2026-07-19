"""Federated hindsight training for station-only EV charging selection.

Each client owns a RealTrafficEnv with its own grid variant and UE traffic
scale. Local training uses HindsightTrainer; the server aggregates policy
network parameters with FedAvg.
"""

import argparse
import copy
import os
import sys
import time
from collections import OrderedDict

import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from agents.hindsight_dqn_agent import HindsightDQNAgent
from env.grid_variants import ALL_GRID_VARIANTS
from env.real_env import RealTrafficEnv
from trainer.trainer import HindsightTrainer, compute_hindsight_reward


DEFAULT_CLIENT_SPECS = "old_city:1.3,new_city:1.0,suburb:0.7"


class FederatedHindsightAgent(HindsightDQNAgent):
    """Hindsight agent with round sample counting and FedAvg helpers."""

    def __init__(self, client_id, *args, fedprox_mu=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.num_samples_this_round = 0
        self.fedprox_mu = float(fedprox_mu)
        self._global_ref = None  # FedProx proximal anchor: {param_name: tensor}

    def store_transition(self, state, action, reward, next_state, action_mask=None, done=False):
        super().store_transition(
            state,
            action,
            reward,
            next_state,
            action_mask=action_mask,
            done=done,
        )
        self.num_samples_this_round += 1

    def reset_round_counter(self):
        self.num_samples_this_round = 0

    def get_model_params(self):
        return copy.deepcopy(self.policy_net.state_dict())

    def load_global_params(self, global_state_dict, is_shared=None):
        """Load global params into the local model.

        is_shared(key)->bool: if given, only keys for which it returns True are
        overwritten by the global value; the rest keep the client's own params
        (this is how FedRep keeps a personalized head). Default (None) overwrites
        all keys = standard FedAvg/FedProx.
        """
        local_state = self.policy_net.state_dict()
        new_state = OrderedDict()
        for key, value in local_state.items():
            if key == "station_node_ids":
                new_state[key] = value
            elif key in global_state_dict and (is_shared is None or is_shared(key)):
                new_state[key] = global_state_dict[key]
            else:
                new_state[key] = value  # keep personalized / non-shared param
        self.policy_net.load_state_dict(new_state, strict=False)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def set_global_reference(self):
        """Snapshot the just-loaded global params as the FedProx proximal anchor."""
        if self.fedprox_mu > 0:
            self._global_ref = {n: p.detach().clone()
                                for n, p in self.policy_net.named_parameters()}

    def _apply_gradients(self, loss, batch_size=None):
        """FedProx: add proximal gradient mu*(w - w_global) before the optimizer step."""
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.fedprox_mu > 0 and self._global_ref is not None:
            for name, p in self.policy_net.named_parameters():
                if p.grad is not None and name in self._global_ref:
                    p.grad.add_(self.fedprox_mu * (p.detach() - self._global_ref[name]))
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()


class FedAvgServer:
    """Federated server supporting three methods (selected by `method`):

    - fedavg  : aggregate & distribute ALL params -> one global model.
    - fedprox : same aggregation as fedavg; heterogeneity handled by the clients'
                proximal term (set fedprox_mu on the agents).
    - fedrep  : aggregate & distribute ONLY the shared representation
                (encoder.*); each client keeps its own personalized head, so the
                "models" live in the clients (saved per-client), not the global.
    """

    def __init__(self, global_agent, method="fedavg", aggregation_momentum=1.0):
        self.global_agent = global_agent
        self.method = method
        self.aggregation_momentum = float(aggregation_momentum)

    def _is_shared(self, key):
        if self.method == "fedrep":
            return key.startswith("encoder.")  # share perception, personalize head
        return True

    def state_dict(self):
        return copy.deepcopy(self.global_agent.policy_net.state_dict())

    def distribute(self, clients):
        global_state = self.state_dict()
        shared = self._is_shared if self.method == "fedrep" else None
        for client in clients:
            client.load_global_params(global_state, is_shared=shared)

    def aggregate(self, clients):
        client_states = [client.get_model_params() for client in clients]
        weights = [max(1, int(client.num_samples_this_round)) for client in clients]
        total_weight = sum(weights)
        if not client_states or total_weight <= 0:
            return weights

        current = self.global_agent.policy_net.state_dict()
        aggregated = OrderedDict()
        for key in current:
            # keep non-float / id buffers, and (FedRep) personal head params as-is
            if (key == "station_node_ids" or not torch.is_floating_point(current[key])
                    or not self._is_shared(key)):
                aggregated[key] = current[key]
                continue
            avg = sum(
                client_states[i][key].float() * (weights[i] / total_weight)
                for i in range(len(client_states))
            )
            if self.aggregation_momentum < 1.0:
                avg = (
                    (1.0 - self.aggregation_momentum) * current[key].float()
                    + self.aggregation_momentum * avg
                )
            aggregated[key] = avg.to(current[key].dtype)

        self.global_agent.policy_net.load_state_dict(aggregated, strict=False)
        self.global_agent.target_net.load_state_dict(self.global_agent.policy_net.state_dict())
        for client in clients:
            client.reset_round_counter()
        return weights

    def save(self, path, epsilon):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_net": self.global_agent.policy_net.state_dict(),
                "epsilon": float(epsilon),
            },
            path,
        )
        print(f"[FedServer] saved global model: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-episodes", type=int, default=2)
    parser.add_argument("--steps-per-episode", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-steps-per-step", type=int, default=1)

    parser.add_argument("--num-evs", type=int, default=40)
    parser.add_argument("--num-stations", type=int, default=4)
    parser.add_argument("--num-chargers-per-station", type=int, default=8)
    parser.add_argument("--respawn", action="store_true", default=True)
    parser.add_argument("--no-respawn", dest="respawn", action="store_false")

    parser.add_argument(
        "--client-specs",
        type=str,
        default=DEFAULT_CLIENT_SPECS,
        help='Comma list such as "old_city:1.3,new_city:1.0,suburb:0.7"',
    )
    parser.add_argument("--graphml-file", type=str, default=os.path.join("map_outputs", "ema", "ema.graphml"))
    parser.add_argument("--cache-dir", type=str, default=os.path.join("map_outputs", "ema_cache"))
    parser.add_argument("--no-ue-background", action="store_true", default=False)
    parser.add_argument("--ue-net-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_net.tntp"))
    parser.add_argument("--ue-trips-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"))
    parser.add_argument("--ue-max-iter", type=int, default=800)
    parser.add_argument("--ue-tol", type=float, default=1e-4)
    parser.add_argument("--ue-verbose", action="store_true", default=False)

    parser.add_argument("--network", choices=["original", "lightweight", "station_only", "station_attn"], default="station_only")
    parser.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=False)
    parser.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")

    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--epsilon-decay", type=float, default=0.994,
                        help="Per-decay epsilon multiplier (decays once/round here; "
                             "lower it for from-scratch runs to anneal within budget)")
    parser.add_argument("--epsilon-decay-per-round", action="store_true", default=False)
    parser.add_argument("--aggregation-momentum", type=float, default=1.0)
    parser.add_argument("--fed-method", choices=["fedavg", "fedprox", "fedrep"],
                        default="fedavg",
                        help="fedavg: one global model; fedprox: + proximal term; "
                             "fedrep: share encoder, personalize head (per-client models)")
    parser.add_argument("--fedprox-mu", type=float, default=0.01,
                        help="FedProx proximal coefficient (only used when "
                             "--fed-method fedprox)")
    parser.add_argument("--save-dir", type=str, default="checkpoints_fed_hindsight")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_client_specs(spec_text):
    specs = []
    for raw in spec_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            variant, scale_text = raw.split(":", 1)
            ue_scale = float(scale_text)
        else:
            variant, ue_scale = raw, 1.0
        variant = variant.strip()
        if variant not in ALL_GRID_VARIANTS:
            raise ValueError(f"Unknown grid variant in --client-specs: {variant!r}")
        specs.append({"grid_variant": variant, "ue_scale": float(ue_scale)})
    if not specs:
        raise ValueError("--client-specs produced no clients")
    return specs


def build_env(args, spec, client_idx):
    env_kw = dict(
        graphml_file=args.graphml_file,
        num_stations=args.num_stations,
        num_evs=args.num_evs,
        num_chargers_per_station=args.num_chargers_per_station,
        max_nodes=1_000_000,
        cache_dir=args.cache_dir,
        seed=args.seed,
        respawn_after_full_charge=args.respawn,
        grid_variant=spec["grid_variant"],
    )
    if not args.no_ue_background and os.path.isfile(args.ue_net_tntp) and os.path.isfile(args.ue_trips_tntp):
        env_kw["background_ue_net_tntp"] = os.path.abspath(args.ue_net_tntp)
        env_kw["background_ue_trips_tntp"] = os.path.abspath(args.ue_trips_tntp)
        env_kw["background_ue_max_iter"] = int(args.ue_max_iter)
        env_kw["background_ue_tol"] = float(args.ue_tol)
        env_kw["background_ue_scale"] = float(spec["ue_scale"])
        env_kw["background_ue_verbose"] = bool(args.ue_verbose)
    elif not args.no_ue_background:
        print(
            f"[client {client_idx}] warn: UE TNTP missing "
            f"(net={os.path.isfile(args.ue_net_tntp)}, trips={os.path.isfile(args.ue_trips_tntp)}); "
            "using heuristic background"
        )
    return RealTrafficEnv(**env_kw)


def build_agent(args, client_id, env):
    station_node_ids = [station.traffic_node_id for station in env.stations]
    agent = FederatedHindsightAgent(
        client_id,
        num_features=19,
        num_actions=args.num_stations,
        station_node_ids=station_node_ids,
        num_nodes_per_graph=env.num_nodes,
        network_variant=args.network,
        use_action_mask=args.use_action_mask,
        epsilon_decay=args.epsilon_decay,
        fedprox_mu=(args.fedprox_mu if args.fed_method == "fedprox" else 0.0),
    )
    agent.epsilon = float(args.epsilon)
    return agent


def average(values):
    return sum(values) / len(values) if values else 0.0


def run_local_training(args, client_info, round_idx):
    env = client_info["env"]
    agent = client_info["agent"]
    trainer = client_info["trainer"]
    spec = client_info["spec"]

    trips, queues, fees, rewards = [], [], [], []
    abandoned = 0
    steps_run = 0

    for _ in range(args.local_episodes):
        env.reset()
        trainer.pending.clear()
        trainer._current_step = 0

        for _ in range(args.steps_per_episode):
            done, info = trainer.step_episode()
            steps_run += 1

            for entry in info.get("charge_started", []):
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                reward = compute_hindsight_reward(trip, queue, fee)
                trips.append(trip)
                queues.append(queue)
                fees.append(fee)
                rewards.append(reward)

            abandoned += len(info.get("abandoned", []))

            if len(agent.memory) >= args.batch_size:
                for _ in range(max(1, int(args.replay_steps_per_step))):
                    agent.replay(args.batch_size)

            if done:
                break

        if not args.epsilon_decay_per_round:
            agent.decay_epsilon()

    print(
        f"[round {round_idx}] client={client_info['name']} "
        f"grid={spec['grid_variant']} ue_scale={spec['ue_scale']} "
        f"steps={steps_run} samples={agent.num_samples_this_round} "
        f"buffer={len(agent.memory)} pending={len(trainer.pending)} "
        f"epsilon={agent.epsilon:.3f} abandoned={abandoned} "
        f"avg_trip={average(trips):.4f}h avg_queue={average(queues):.4f}h "
        f"avg_fee={average(fees):.4f} avg_reward={average(rewards):.4f}"
    )
    return {
        "trip": trips,
        "queue": queues,
        "fee": fees,
        "reward": rewards,
        "abandoned": abandoned,
    }


def main():
    args = parse_args()
    specs = parse_client_specs(args.client_specs)
    os.makedirs(args.save_dir, exist_ok=True)

    print(
        f"[setup] rounds={args.rounds} local_episodes={args.local_episodes} "
        f"steps={args.steps_per_episode} batch={args.batch_size} "
        f"num_evs={args.num_evs} stations={args.num_stations} "
        f"chargers_per_station={args.num_chargers_per_station} respawn={args.respawn} "
        f"network={args.network} use_action_mask={args.use_action_mask}"
    )
    print(f"[setup] client_specs={specs}")

    clients = []
    for idx, spec in enumerate(specs):
        env = build_env(args, spec, idx)
        agent = build_agent(args, idx, env)
        trainer = HindsightTrainer(env, agent)
        clients.append(
            {
                "name": f"client{idx}",
                "spec": spec,
                "env": env,
                "agent": agent,
                "trainer": trainer,
            }
        )
        print(
            f"[setup] client{idx}: grid={spec['grid_variant']} ue_scale={spec['ue_scale']} "
            f"station_node_ids={[s.traffic_node_id for s in env.stations]}"
        )

    global_agent = build_agent(args, "global", clients[0]["env"])
    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise FileNotFoundError(f"--load-model not found: {args.load_model}")
        global_agent.load_model(args.load_model)
        global_agent.epsilon = float(args.epsilon)
        print(f"[setup] loaded warm-start model: {args.load_model}")

    server = FedAvgServer(global_agent, method=args.fed_method,
                          aggregation_momentum=args.aggregation_momentum)
    print(f"[setup] fed_method={args.fed_method}"
          + (f" fedprox_mu={args.fedprox_mu}" if args.fed_method == "fedprox" else ""))
    t0 = time.time()

    for round_idx in range(1, args.rounds + 1):
        server.distribute([client["agent"] for client in clients])
        for client in clients:
            client["agent"].epsilon = global_agent.epsilon
            client["agent"].reset_round_counter()
            client["agent"].set_global_reference()  # FedProx proximal anchor (no-op if mu=0)

        round_trip, round_queue, round_fee, round_reward = [], [], [], []
        round_abandoned = 0
        for client in clients:
            metrics = run_local_training(args, client, round_idx)
            round_trip.extend(metrics["trip"])
            round_queue.extend(metrics["queue"])
            round_fee.extend(metrics["fee"])
            round_reward.extend(metrics["reward"])
            round_abandoned += metrics["abandoned"]

        weights = server.aggregate([client["agent"] for client in clients])
        if args.epsilon_decay_per_round:
            global_agent.decay_epsilon()
        else:
            eps_values = [client["agent"].epsilon for client in clients]
            global_agent.epsilon = min(eps_values) if eps_values else global_agent.epsilon

        elapsed = time.time() - t0
        print(
            f"[round {round_idx}/{args.rounds}] aggregate weights={weights} "
            f"epsilon={global_agent.epsilon:.3f} abandoned={round_abandoned} "
            f"avg_trip={average(round_trip):.4f}h avg_queue={average(round_queue):.4f}h "
            f"avg_fee={average(round_fee):.4f} avg_reward={average(round_reward):.4f} "
            f"elapsed={elapsed:.1f}s"
        )

        if round_idx % args.save_every == 0:
            path = os.path.join(args.save_dir, f"global_round{round_idx}.pth")
            server.save(path, epsilon=global_agent.epsilon)

    final_path = os.path.join(args.save_dir, "global_final.pth")
    server.save(final_path, epsilon=global_agent.epsilon)
    print(f"[done] final global model -> {final_path}")

    if args.fed_method == "fedrep":
        # FedRep "models" are the personalized clients (shared encoder + own head),
        # not the global. Save each client's full model keyed by its city.
        os.makedirs(args.save_dir, exist_ok=True)
        for client in clients:
            city = client["spec"]["grid_variant"]
            cpath = os.path.join(args.save_dir, f"{city}_final.pth")
            torch.save({"policy_net": client["agent"].policy_net.state_dict(),
                        "epsilon": float(global_agent.epsilon)}, cpath)
            print(f"[done] FedRep personalized model ({city}) -> {cpath}")

    print(f"[done] total elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
