"""
Evaluate a trained hindsight model on the EMA road network.

Metrics:
- average trip time
- average queue time
- average charging fee

This script does not train. It only runs episodes and reports summary statistics.
"""

import argparse
import os
import random
import sys
import time

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Windows console cp1252 compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from agents.hindsight_dqn_agent import HindsightDQNAgent
from env.real_env import RealTrafficEnv
from trainer.trainer import HindsightTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate hindsight training on EMA map")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps-per-episode", type=int, default=100)
    p.add_argument("--num-evs", type=int, default=10)
    p.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("checkpoints_hindsight", "model_final.pth"),
        help="Path to a trained model checkpoint.",
    )
    p.add_argument(
        "--graphml-file",
        type=str,
        default=os.path.join("map_outputs", "ema", "ema.graphml"),
        help="EMA graphml file path.",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join("map_outputs", "ema_cache"),
        help="Cache dir for loading road network.",
    )
    p.add_argument("--respawn", action="store_true", default=True)
    p.add_argument("--no-respawn", dest="respawn", action="store_false")
    return p.parse_args()


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _run_policy(args, policy_name, policy_fn):
    env = RealTrafficEnv(
        graphml_file=args.graphml_file,
        num_stations=2,
        num_evs=args.num_evs,
        max_nodes=1_000_000,
        cache_dir=args.cache_dir,
        seed=42,
        respawn_after_full_charge=args.respawn,
    )

    all_trip, all_queue, all_fee, all_reward = [], [], [], []
    t0 = time.time()

    print(f"\n[policy={policy_name}] episodes={args.episodes} steps={args.steps_per_episode} num_evs={args.num_evs}")
    for episode in range(args.episodes):
        env.reset()
        ep_trip, ep_queue, ep_fee, ep_reward = [], [], [], []
        steps_run = 0

        for _ in range(args.steps_per_episode):
            done, info = _step_with_policy(env, policy_fn)
            steps_run += 1

            for entry in info.get("completed", []):
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                reward = -(0.3 * trip + 0.5 * queue + 0.03 * fee)
                ep_trip.append(trip)
                ep_queue.append(queue)
                ep_fee.append(fee)
                ep_reward.append(reward)

            if done:
                break

        all_trip.extend(ep_trip)
        all_queue.extend(ep_queue)
        all_fee.extend(ep_fee)
        all_reward.extend(ep_reward)

        print(
            f"[policy={policy_name}] ep {episode + 1}/{args.episodes} steps={steps_run} "
            f"completed={len(ep_trip)} avg_trip={_avg(ep_trip):.4f}h "
            f"avg_queue={_avg(ep_queue):.4f}h avg_fee={_avg(ep_fee):.4f} "
            f"avg_reward={_avg(ep_reward):.4f}"
        )

    elapsed = time.time() - t0
    summary = {
        "policy": policy_name,
        "episodes": args.episodes,
        "elapsed": elapsed,
        "completed": len(all_trip),
        "avg_trip": _avg(all_trip),
        "avg_queue": _avg(all_queue),
        "avg_fee": _avg(all_fee),
        "avg_reward": _avg(all_reward),
    }
    print(
        f"[policy={policy_name}] summary completed={summary['completed']} "
        f"avg_trip={summary['avg_trip']:.4f}h avg_queue={summary['avg_queue']:.4f}h "
        f"avg_fee={summary['avg_fee']:.4f} avg_reward={summary['avg_reward']:.4f} "
        f"elapsed={summary['elapsed']:.1f}s"
    )
    return summary


def _step_with_policy(env, policy_fn):
    urgent_evs = env.get_pending_decision_evs()
    actions = {}
    for ev in urgent_evs:
        state = env.get_graph_state_for_ev(ev)
        action_mask = env.get_action_mask(ev)
        actions[ev.id] = policy_fn(env, ev, state, action_mask)
    return env.step(actions)


def main():
    args = parse_args()

    model = HindsightDQNAgent(
        num_features=18,
        num_actions=2,
        station_node_ids=None,
        num_nodes_per_graph=9,
    )
    if os.path.isfile(args.model_path):
        model.load_model(args.model_path)
        print(f"[setup] loaded model: {args.model_path}")
    else:
        print(f"[setup] model not found, evaluating with random init: {args.model_path}")

    print(f"[setup] graphml_file={args.graphml_file}")

    def random_policy(env, ev, state, action_mask):
        valid = action_mask.squeeze().nonzero(as_tuple=True)[0].tolist()
        return int(random.choice(valid))

    def shortest_path_policy(env, ev, state, action_mask):
        best_action = None
        best_time = None
        for station in env.stations:
            if not action_mask[0, station.id].item():
                continue
            metrics = env.estimate_action_metrics(ev, station.id)
            trip_time = metrics.get("trip_time_h", float("inf"))
            if best_time is None or trip_time < best_time:
                best_time = trip_time
                best_action = station.id
        return int(best_action if best_action is not None else 0)

    def model_greedy_policy(env, ev, state, action_mask):
        with torch.no_grad():
            q_values = model.policy_net(
                state.to(model.device),
                action_mask=action_mask.to(model.device),
                action_type='t0',
            )
            return int(q_values.argmax().item())

    summaries = []
    summaries.append(_run_policy(args, "random", random_policy))
    summaries.append(_run_policy(args, "shortest_path", shortest_path_policy))
    summaries.append(_run_policy(args, "model_greedy", model_greedy_policy))

    print("\n" + "=" * 60)
    print("Evaluation comparison")
    print("=" * 60)
    for s in summaries:
        print(
            f"[{s['policy']}] avg_trip={s['avg_trip']:.4f}h "
            f"avg_queue={s['avg_queue']:.4f}h avg_fee={s['avg_fee']:.4f} "
            f"avg_reward={s['avg_reward']:.4f} completed={s['completed']} "
            f"elapsed={s['elapsed']:.1f}s"
        )


if __name__ == "__main__":
    main()
