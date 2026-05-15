"""诊断模型是否偏向某个站点的评估脚本。

用途:
- 统计每个 policy / 模型对各充电站的选择次数与 trip / queue / fee
- 快速判断模型是否学成了“偏站”策略

环境与 ``train_hindsight.py`` 默认对齐：``num_stations`` / ``num_evs`` / ``num_chargers_per_station``、
UE 背景、``ChargingStation`` 功率默认值（单桩/站总功率在 ``charging_station.py``）。
"""

import argparse
import os
import random
import sys
import time
from collections import Counter, defaultdict

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
from train_hindsight import (
    TRAIN_DEFAULT_NUM_CHARGERS_PER_STATION,
    TRAIN_DEFAULT_NUM_EVS,
    TRAIN_DEFAULT_NUM_STATIONS,
)


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose station selection bias")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps-per-episode", type=int, default=100)
    p.add_argument("--num-evs", type=int, default=TRAIN_DEFAULT_NUM_EVS)
    p.add_argument("--num-stations", type=int, default=TRAIN_DEFAULT_NUM_STATIONS)
    p.add_argument(
        "--num-chargers-per-station",
        type=int,
        default=TRAIN_DEFAULT_NUM_CHARGERS_PER_STATION,
        help="Each ChargingStation.num_chargers (same default as train_hindsight)",
    )
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
    p.add_argument(
        "--network",
        type=str,
        default="station_only",
        choices=["original", "lightweight", "station_only"],
        help="Network variant used when loading the checkpoint.",
    )
    p.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=True)
    p.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")
    p.add_argument(
        "--no-ue-background",
        action="store_true",
        default=False,
        help="Use heuristic background_edge_base_flows instead of UE (default: UE on)",
    )
    p.add_argument(
        "--ue-net-tntp",
        type=str,
        default=os.path.join("map_outputs", "ema", "EMA_net.tntp"),
        help="TNTP net file for UE baseline",
    )
    p.add_argument(
        "--ue-trips-tntp",
        type=str,
        default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"),
        help="TNTP trips (OD) file for UE baseline",
    )
    p.add_argument("--ue-max-iter", type=int, default=800, help="Frank–Wolfe max iterations")
    p.add_argument("--ue-tol", type=float, default=1e-4, help="FW relative-gap tolerance")
    p.add_argument("--ue-scale", type=float, default=1.0, help="Scale UE edge flows after solve")
    p.add_argument(
        "--ue-verbose",
        action="store_true",
        default=False,
        help="Print Frank–Wolfe UE iteration logs (default: off)",
    )
    return p.parse_args()


def _real_env_kwargs(args) -> dict:
    """Shared RealTrafficEnv kwargs; UE background when TNTP files exist (unless --no-ue-background)."""
    kw: dict = dict(
        graphml_file=args.graphml_file,
        num_stations=args.num_stations,
        num_evs=args.num_evs,
        num_chargers_per_station=args.num_chargers_per_station,
        max_nodes=1_000_000,
        cache_dir=args.cache_dir,
        seed=42,
        respawn_after_full_charge=args.respawn,
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
        else:
            print(
                f"[setup] warn: UE TNTP not found (net={os.path.isfile(net_p)}, "
                f"trips={os.path.isfile(trip_p)}), using heuristic background_edge_base_flows"
            )
    return kw


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _run_policy(args, policy_name, policy_fn):
    env = RealTrafficEnv(**_real_env_kwargs(args))

    choice_counter = Counter()
    station_trip = defaultdict(list)
    station_queue = defaultdict(list)
    station_fee = defaultdict(list)
    all_trip, all_queue, all_fee, all_reward = [], [], [], []
    t0 = time.time()

    print(
        f"\n[policy={policy_name}] episodes={args.episodes} steps={args.steps_per_episode} "
        f"num_evs={args.num_evs} use_action_mask={args.use_action_mask}"
    )
    print(f"[policy={policy_name}] station_nodes={[s.traffic_node_id for s in env.stations]}")

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
                # completed entries don’t carry action; infer from current env records if possible
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
            f"[policy={policy_name}] ep {episode + 1}/{args.episodes} steps={steps_run} "
            f"completed={len(ep_trip)} avg_trip={_avg(ep_trip):.4f}h "
            f"avg_queue={_avg(ep_queue):.4f}h avg_fee={_avg(ep_fee):.4f} "
            f"avg_reward={_avg(ep_reward):.4f}"
        )

    elapsed = time.time() - t0
    print(f"\n[policy={policy_name}] choice_count={dict(choice_counter)}")
    for sid in sorted(set(choice_counter.keys()) | set(station_trip.keys()) | set(station_queue.keys()) | set(station_fee.keys())):
        print(
            f"[policy={policy_name}] station={sid} "
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
        f"[policy={policy_name}] summary completed={summary['completed']} "
        f"avg_trip={summary['avg_trip']:.4f}h avg_queue={summary['avg_queue']:.4f}h "
        f"avg_fee={summary['avg_fee']:.4f} avg_reward={summary['avg_reward']:.4f} "
        f"elapsed={summary['elapsed']:.1f}s"
    )
    return summary


def main():
    args = parse_args()

    if not args.no_ue_background:
        net_p, trip_p = args.ue_net_tntp, args.ue_trips_tntp
        if os.path.isfile(net_p) and os.path.isfile(trip_p):
            print(
                f"[setup] UE background baseline: net={net_p} trips={trip_p} "
                f"max_iter={args.ue_max_iter} tol={args.ue_tol} scale={args.ue_scale}"
            )
        else:
            print("[setup] UE files missing; env will use heuristic baseline (see warn on first construct)")
    else:
        print("[setup] heuristic background (--no-ue-background)")

    _probe = RealTrafficEnv(**_real_env_kwargs(args))
    station_node_ids = [s.traffic_node_id for s in _probe.stations]
    num_actions = len(_probe.stations)
    num_nodes = _probe.num_nodes
    model = HindsightDQNAgent(
        num_features=18,
        num_actions=num_actions,
        station_node_ids=station_node_ids,
        num_nodes_per_graph=num_nodes,
        network_variant=args.network,
        use_action_mask=args.use_action_mask,
    )
    if os.path.isfile(args.model_path):
        model.load_model(args.model_path)
        print(f"[setup] loaded model: {args.model_path}")
    else:
        print(f"[setup] model not found, evaluating with random init: {args.model_path}")

    print(
        f"[setup] graphml_file={args.graphml_file} network={args.network} "
        f"num_stations={args.num_stations} num_evs={args.num_evs} "
        f"chargers_per_station={args.num_chargers_per_station} "
        f"use_action_mask={args.use_action_mask} ue_background={not args.no_ue_background}"
    )

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
                action_mask=action_mask.to(model.device) if args.use_action_mask else None,
                action_type="t0",
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
            f"elapsed={s['elapsed']:.1f}s choice_count={s['choice_count']}"
        )


if __name__ == "__main__":
    main()
