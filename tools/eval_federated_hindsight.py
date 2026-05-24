"""Evaluate federated hindsight checkpoints on the three client grids.

This script is a lightweight companion to `train_federated_hindsight.py` and
`tools/debug_station_bias.py`.

What it does:
- builds each client grid (`old_city`, `new_city`, `suburb`)
- loads the matching checkpoint if it exists
- compares three policies on each client:
  - random
  - shortest_path
  - model_greedy
- prints episode-level and aggregate metrics

The script is intentionally verbose so it can double as a diagnostic tool
for station bias / queue pressure / reward collapse.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import Counter, defaultdict
from statistics import mean

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from agents.hindsight_dqn_agent import HindsightDQNAgent
from env.real_env import RealTrafficEnv
from train_federated_hindsight import DEFAULT_CLIENTS, DEFAULT_CACHE_DIR, DEFAULT_GRAPHML


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate federated hindsight checkpoints")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps-per-episode", type=int, default=100)
    p.add_argument("--num-evs", type=int, default=80)
    p.add_argument("--num-stations", type=int, default=4)
    p.add_argument("--num-chargers-per-station", type=int, default=8)
    p.add_argument("--graphml-file", type=str, default=DEFAULT_GRAPHML)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--clients", nargs="+", default=DEFAULT_CLIENTS, choices=DEFAULT_CLIENTS)
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_federated_hindsight",
        help="Directory containing <client>_final.pth or <client>_roundN.pth checkpoints.",
    )
    p.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="final",
        help="Checkpoint suffix, e.g. final or round100.",
    )
    p.add_argument(
        "--network",
        type=str,
        default="station_only",
        choices=["original", "lightweight", "station_only"],
    )
    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=True)
    mask_group.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")
    p.add_argument("--respawn", action="store_true", default=True)
    p.add_argument("--no-respawn", dest="respawn", action="store_false")
    p.add_argument("--no-ue-background", action="store_true", default=False)
    p.add_argument("--ue-net-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_net.tntp"))
    p.add_argument("--ue-trips-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"))
    p.add_argument("--ue-max-iter", type=int, default=800)
    p.add_argument("--ue-tol", type=float, default=1e-4)
    p.add_argument("--ue-scale", type=float, default=1.0)
    p.add_argument("--ue-verbose", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _env_kwargs(args, client_name: str) -> dict:
    kw = dict(
        graphml_file=args.graphml_file,
        num_stations=args.num_stations,
        num_evs=args.num_evs,
        num_chargers_per_station=args.num_chargers_per_station,
        max_nodes=1_000_000,
        cache_dir=args.cache_dir,
        seed=args.seed,
        respawn_after_full_charge=args.respawn,
        client_name=client_name,
    )
    if not args.no_ue_background:
        net_p = args.ue_net_tntp
        trip_p = args.ue_trips_tntp
        if os.path.isfile(net_p) and os.path.isfile(trip_p):
            kw["background_ue_net_tntp"] = os.path.abspath(net_p)
            kw["background_ue_trips_tntp"] = os.path.abspath(trip_p)
            kw["background_ue_max_iter"] = int(args.ue_max_iter)
            kw["background_ue_tol"] = float(args.ue_tol)
            kw["background_ue_scale"] = float(args.ue_scale)
            kw["background_ue_verbose"] = bool(args.ue_verbose)
    return kw


def _avg(values):
    return mean(values) if values else 0.0


def _make_env(args, client_name: str) -> RealTrafficEnv:
    return RealTrafficEnv(**_env_kwargs(args, client_name))


def _load_model(args, env: RealTrafficEnv, checkpoint_path: str):
    station_node_ids = [s.traffic_node_id for s in env.stations]
    model = HindsightDQNAgent(
        num_features=18,
        num_actions=len(env.stations),
        station_node_ids=station_node_ids,
        num_nodes_per_graph=env.num_nodes,
        network_variant=args.network,
        use_action_mask=args.use_action_mask,
    )
    if os.path.isfile(checkpoint_path):
        model.load_model(checkpoint_path)
    return model


def _run_policy(args, env: RealTrafficEnv, policy_name: str, policy_fn):
    choice_counter = Counter()
    station_trip = defaultdict(list)
    station_queue = defaultdict(list)
    station_fee = defaultdict(list)
    all_trip, all_queue, all_fee, all_reward = [], [], [], []
    t0 = time.time()

    print(
        f"\n[client={env.power_grid.client_name}] policy={policy_name} "
        f"episodes={args.episodes} steps={args.steps_per_episode} "
        f"num_evs={args.num_evs} use_action_mask={args.use_action_mask}"
    )
    print(f"[client={env.power_grid.client_name}] station_nodes={[s.traffic_node_id for s in env.stations]}")

    for episode in range(args.episodes):
        env.reset()
        ep_trip, ep_queue, ep_fee, ep_reward = [], [], [], []
        steps_run = 0

        for _ in range(args.steps_per_episode):
            urgent_evs = env.get_pending_decision_evs()
            actions = {}
            for ev in urgent_evs:
                state = env.get_graph_state_for_ev(ev)
                action_mask = env.get_action_mask(ev)
                action = policy_fn(env, ev, state, action_mask)
                actions[ev.id] = action
                choice_counter[action] += 1

            _obs, _reward, done, info = env.step(actions)
            steps_run += 1

            for entry in info.get("completed", []):
                ev_id = entry.get("ev_id")
                action = None
                for ev in env.evs:
                    if ev.id == ev_id:
                        action = getattr(ev, "target_station_idx", None)
                        break
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                reward = -(0.3 * trip + 0.5 * queue + 0.03 * fee)
                ep_trip.append(trip)
                ep_queue.append(queue)
                ep_fee.append(fee)
                ep_reward.append(reward)
                if action is not None:
                    station_trip[action].append(trip)
                    station_queue[action].append(queue)
                    station_fee[action].append(fee)

            if done:
                break

        all_trip.extend(ep_trip)
        all_queue.extend(ep_queue)
        all_fee.extend(ep_fee)
        all_reward.extend(ep_reward)
        print(
            f"[client={env.power_grid.client_name}] ep {episode + 1}/{args.episodes} "
            f"steps={steps_run} completed={len(ep_trip)} avg_trip={_avg(ep_trip):.4f}h "
            f"avg_queue={_avg(ep_queue):.4f}h avg_fee={_avg(ep_fee):.4f} "
            f"avg_reward={_avg(ep_reward):.4f}"
        )

    elapsed = time.time() - t0
    print(f"\n[client={env.power_grid.client_name}] policy={policy_name} choice_count={dict(choice_counter)}")
    for sid in sorted(set(choice_counter.keys()) | set(station_trip.keys()) | set(station_queue.keys()) | set(station_fee.keys())):
        print(
            f"[client={env.power_grid.client_name}] station={sid} "
            f"choices={choice_counter.get(sid, 0)} "
            f"avg_trip={_avg(station_trip[sid]):.4f}h "
            f"avg_queue={_avg(station_queue[sid]):.4f}h "
            f"avg_fee={_avg(station_fee[sid]):.4f}"
        )

    summary = {
        "policy": policy_name,
        "episodes": args.episodes,
        "elapsed": elapsed,
        "completed": len(all_trip),
        "avg_trip": _avg(all_trip),
        "avg_queue": _avg(all_queue),
        "avg_fee": _avg(all_fee),
        "avg_reward": _avg(all_reward),
        "choice_count": dict(choice_counter),
    }
    print(
        f"[client={env.power_grid.client_name}] policy={policy_name} summary completed={summary['completed']} "
        f"avg_trip={summary['avg_trip']:.4f}h avg_queue={summary['avg_queue']:.4f}h "
        f"avg_fee={summary['avg_fee']:.4f} avg_reward={summary['avg_reward']:.4f} "
        f"elapsed={summary['elapsed']:.1f}s"
    )
    return summary


def main():
    args = parse_args()
    print(
        f"[setup] checkpoint_dir={args.checkpoint_dir} checkpoint_suffix={args.checkpoint_suffix} "
        f"clients={args.clients} network={args.network} use_action_mask={args.use_action_mask}"
    )

    summaries = []
    for client_name in args.clients:
        env = _make_env(args, client_name)
        checkpoint = os.path.join(args.checkpoint_dir, f"{client_name}_{args.checkpoint_suffix}.pth")
        model = _load_model(args, env, checkpoint)
        if os.path.isfile(checkpoint):
            print(f"[setup] loaded checkpoint for {client_name}: {checkpoint}")
        else:
            print(f"[setup] checkpoint not found for {client_name}, using random init: {checkpoint}")

        def random_policy(_env, _ev, _state, action_mask):
            valid = action_mask.squeeze().nonzero(as_tuple=True)[0].tolist()
            return int(random.choice(valid))

        def shortest_path_policy(env_, ev, _state, action_mask):
            best_action = None
            best_time = None
            for station in env_.stations:
                if not action_mask[0, station.id].item():
                    continue
                metrics = env_.estimate_action_metrics(ev, station.id)
                trip_time = metrics.get("trip_time_h", float("inf"))
                if best_time is None or trip_time < best_time:
                    best_time = trip_time
                    best_action = station.id
            return int(best_action if best_action is not None else 0)

        def model_greedy_policy(_env, _ev, state, action_mask):
            with torch.no_grad():
                q_values = model.policy_net(
                    state.to(model.device),
                    action_mask=action_mask.to(model.device) if args.use_action_mask else None,
                    action_type="t0",
                )
                return int(q_values.argmax().item())

        summaries.append((client_name, _run_policy(args, env, "random", random_policy)))
        summaries.append((client_name, _run_policy(args, env, "shortest_path", shortest_path_policy)))
        summaries.append((client_name, _run_policy(args, env, "model_greedy", model_greedy_policy)))

    print("\n" + "=" * 80)
    print("Evaluation comparison")
    print("=" * 80)
    for client_name, s in summaries:
        print(
            f"[{client_name} / {s['policy']}] avg_trip={s['avg_trip']:.4f}h "
            f"avg_queue={s['avg_queue']:.4f}h avg_fee={s['avg_fee']:.4f} "
            f"avg_reward={s['avg_reward']:.4f} completed={s['completed']} "
            f"elapsed={s['elapsed']:.1f}s choice_count={s['choice_count']}"
        )


if __name__ == "__main__":
    main()
