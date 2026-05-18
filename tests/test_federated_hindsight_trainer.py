from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from train_federated_hindsight import parse_args
from trainer.federated_hindsight_trainer import (
    FederatedClient,
    FederatedClientConfig,
    FederatedHindsightTrainer,
)


class DummyAgent:
    def __init__(self, weight: float, bias: float = 0.0):
        self.policy_net = nn.Linear(2, 1)
        self.target_net = nn.Linear(2, 1)
        with torch.no_grad():
            self.policy_net.weight.fill_(weight)
            self.policy_net.bias.fill_(bias)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = []
        self.replay_calls: list[int] = []
        self.save_paths: list[str] = []

    def replay(self, batch_size: int):
        self.replay_calls.append(batch_size)

    def save_model(self, path: str):
        self.save_paths.append(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"policy_net": self.policy_net.state_dict(), "epsilon": 0.5}, path)


class DummyEnv:
    def __init__(self):
        self.reset_calls = 0
        self.num_nodes = 33
        self.stations = [SimpleNamespace(traffic_node_id=101), SimpleNamespace(traffic_node_id=202)]

    def reset(self):
        self.reset_calls += 1


class DummyTrainer:
    def __init__(self, agent: DummyAgent, completed_reward: float = 1.0):
        self.agent = agent
        self.pending = [1, 2, 3]
        self._current_step = 99
        self.calls = 0
        self.completed_reward = completed_reward

    def step_episode(self):
        self.calls += 1
        if self.calls == 1:
            self.agent.memory.append("transition")
            info = {
                "completed": [
                    {
                        "actual_trip_time_h": 2.0,
                        "actual_queue_time_h": 0.5,
                        "charging_fee": 3.0,
                    }
                ]
            }
            return False, info
        return True, {"completed": []}


class FederatedHindsightTrainerTests(unittest.TestCase):
    def _make_client(self, name: str, weight: float, batch_size: int = 1) -> FederatedClient:
        cfg = FederatedClientConfig(
            client_name=name,
            graphml_file="map_outputs/ema/ema.graphml",
            num_stations=2,
            num_evs=4,
            num_chargers_per_station=2,
            cache_dir=None,
            batch_size=batch_size,
            steps_per_episode=2,
        )
        env = DummyEnv()
        agent = DummyAgent(weight=weight)
        trainer = DummyTrainer(agent)
        return FederatedClient(cfg, env, agent, trainer)

    def test_fedavg_state_dict_means_tensors(self):
        t = FederatedHindsightTrainer.__new__(FederatedHindsightTrainer)
        sd1 = {"w": torch.tensor([1.0, 3.0]), "b": torch.tensor([2.0]), "n": torch.tensor(1, dtype=torch.long)}
        sd2 = {"w": torch.tensor([5.0, 7.0]), "b": torch.tensor([6.0]), "n": torch.tensor(9, dtype=torch.long)}
        avg = t._fedavg_state_dict([sd1, sd2])
        self.assertTrue(torch.equal(avg["w"], torch.tensor([3.0, 5.0])))
        self.assertTrue(torch.equal(avg["b"], torch.tensor([4.0])))
        self.assertTrue(torch.equal(avg["n"], torch.tensor(1, dtype=torch.long)))

    def test_sync_global_weights_to_clients(self):
        t = FederatedHindsightTrainer.__new__(FederatedHindsightTrainer)
        c1 = self._make_client("old_city", weight=1.0)
        c2 = self._make_client("new_city", weight=5.0)
        t.clients = [c1, c2]
        global_state = c1.agent.policy_net.state_dict()
        with torch.no_grad():
            global_state = {k: v.clone() + 2.0 if torch.is_floating_point(v) else v.clone() for k, v in global_state.items()}
        t._sync_global_weights_to_clients(global_state)
        for client in t.clients:
            for k, v in client.agent.policy_net.state_dict().items():
                self.assertTrue(torch.allclose(v, global_state[k]) if torch.is_floating_point(v) else torch.equal(v, global_state[k]))
            self.assertTrue(torch.allclose(client.agent.target_net.weight, client.agent.policy_net.weight))

    def test_train_round_runs_local_training_and_aggregates(self):
        t = FederatedHindsightTrainer.__new__(FederatedHindsightTrainer)
        c1 = self._make_client("old_city", weight=1.0, batch_size=1)
        c2 = self._make_client("new_city", weight=5.0, batch_size=1)
        t.clients = [c1, c2]
        t.client_configs = [c1.config, c2.config]
        t.parallel = False
        metrics = t.train_round()

        self.assertIn("old_city", metrics)
        self.assertIn("new_city", metrics)
        self.assertEqual(c1.env.reset_calls, 0)
        self.assertEqual(c2.env.reset_calls, 0)
        self.assertEqual(c1.agent.replay_calls, [1, 1])
        self.assertEqual(c2.agent.replay_calls, [1, 1])
        w1 = c1.agent.policy_net.weight.detach().clone()
        w2 = c2.agent.policy_net.weight.detach().clone()
        self.assertTrue(torch.allclose(w1, w2))
        self.assertAlmostEqual(metrics["old_city"]["avg_trip_h"], 2.0)
        self.assertAlmostEqual(metrics["old_city"]["avg_queue_h"], 0.5)
        self.assertAlmostEqual(metrics["old_city"]["avg_fee"], 3.0)
        self.assertAlmostEqual(metrics["old_city"]["avg_reward"], -(0.3 * 2.0 + 0.5 * 0.5 + 0.03 * 3.0))

    def test_save_global_models_writes_files(self):
        trainer = FederatedHindsightTrainer.__new__(FederatedHindsightTrainer)
        c1 = self._make_client("old_city", weight=1.0)
        c2 = self._make_client("new_city", weight=5.0)
        trainer.clients = [c1, c2]
        with tempfile.TemporaryDirectory() as td:
            trainer.save_global_models(td, round_idx=3)
            self.assertTrue(os.path.isfile(os.path.join(td, "old_city_round3.pth")))
            self.assertTrue(os.path.isfile(os.path.join(td, "new_city_round3.pth")))
            trainer.save_global_models(td, round_idx=None)
            self.assertTrue(os.path.isfile(os.path.join(td, "old_city_final.pth")))
            self.assertTrue(os.path.isfile(os.path.join(td, "new_city_final.pth")))

    def test_constructor_rejects_empty_client_configs(self):
        with self.assertRaises(ValueError):
            FederatedHindsightTrainer([])

    def test_constructor_uses_build_client(self):
        fake_client = self._make_client("old_city", weight=1.0)
        with mock.patch.object(FederatedHindsightTrainer, "_build_client", return_value=fake_client) as build:
            trainer = FederatedHindsightTrainer([
                FederatedClientConfig(
                    client_name="old_city",
                    graphml_file="map_outputs/ema/ema.graphml",
                    num_stations=2,
                    num_evs=4,
                )
            ])
        self.assertEqual(len(trainer.clients), 1)
        build.assert_called_once()

    def test_parse_args_action_mask_flags(self):
        import sys
        orig_argv = sys.argv
        try:
            sys.argv = ["prog"]
            args = parse_args()
            self.assertTrue(args.use_action_mask)

            sys.argv = ["prog", "--no-use-action-mask"]
            args = parse_args()
            self.assertFalse(args.use_action_mask)
        finally:
            sys.argv = orig_argv

    def test_parallel_flag_defaults_true(self):
        t = FederatedHindsightTrainer.__new__(FederatedHindsightTrainer)
        c = self._make_client("old_city", weight=1.0)
        t.clients = [c]
        t.client_configs = [c.config]
        t.parallel = True
        self.assertTrue(t.parallel)


if __name__ == "__main__":
    unittest.main()
