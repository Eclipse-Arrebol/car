"""Evaluate a hindsight station-selection checkpoint without training."""

import argparse
import json
import os
import random
import sys
import time

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
from trainer.trainer import (
    NORM_FEE,
    NORM_TRIP,
    NORM_WAIT,
    compute_hindsight_reward,
)


DEFAULT_NUM_EVS = 40
DEFAULT_NUM_STATIONS = 4
DEFAULT_NUM_CHARGERS_PER_STATION = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps-per-episode", type=int, default=144)
    parser.add_argument("--num-evs", type=int, default=DEFAULT_NUM_EVS)
    parser.add_argument("--num-stations", type=int, default=DEFAULT_NUM_STATIONS)
    parser.add_argument(
        "--num-chargers-per-station",
        type=int,
        default=DEFAULT_NUM_CHARGERS_PER_STATION,
    )
    parser.add_argument("--respawn", action="store_true", default=True)
    parser.add_argument("--no-respawn", dest="respawn", action="store_false")
    parser.add_argument("--strategy", choices=["model", "random", "shortest", "greedy"], default="model")
    parser.add_argument("--compare-baselines", action="store_true", default=False)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--network", choices=["original", "lightweight", "station_only", "station_attn"], default="station_only")
    parser.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=False)
    parser.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=os.path.join("evaluation", "hindsight_results"))

    parser.add_argument("--graphml-file", type=str, default=os.path.join("map_outputs", "ema", "ema.graphml"))
    parser.add_argument("--cache-dir", type=str, default=os.path.join("map_outputs", "ema_cache"))
    parser.add_argument("--no-ue-background", action="store_true", default=False)
    parser.add_argument("--ue-net-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_net.tntp"))
    parser.add_argument("--ue-trips-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"))
    parser.add_argument("--ue-max-iter", type=int, default=800)
    parser.add_argument("--ue-tol", type=float, default=1e-4)
    parser.add_argument("--ue-scale", type=float, default=1.0)
    parser.add_argument("--ue-verbose", action="store_true", default=False)
    parser.add_argument(
        "--grid-variant",
        type=str,
        default="ieee33",
        choices=ALL_GRID_VARIANTS,
        help="Power-grid scenario: ieee33, old_city, new_city, or suburb",
    )
    return parser.parse_args()


def _build_env(args, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    env_kw = dict(
        graphml_file=args.graphml_file,
        num_stations=args.num_stations,
        num_evs=args.num_evs,
        num_chargers_per_station=args.num_chargers_per_station,
        max_nodes=1_000_000,
        cache_dir=args.cache_dir,
        seed=seed,
        respawn_after_full_charge=args.respawn,
        grid_variant=args.grid_variant,
    )
    if not args.no_ue_background:
        if os.path.isfile(args.ue_net_tntp) and os.path.isfile(args.ue_trips_tntp):
            env_kw["background_ue_net_tntp"] = os.path.abspath(args.ue_net_tntp)
            env_kw["background_ue_trips_tntp"] = os.path.abspath(args.ue_trips_tntp)
            env_kw["background_ue_max_iter"] = int(args.ue_max_iter)
            env_kw["background_ue_tol"] = float(args.ue_tol)
            env_kw["background_ue_scale"] = float(args.ue_scale)
            env_kw["background_ue_verbose"] = bool(args.ue_verbose)
    return RealTrafficEnv(**env_kw)


def _build_agent(args, env):
    station_node_ids = [station.traffic_node_id for station in env.stations]
    agent = HindsightDQNAgent(
        num_features=19,
        num_actions=args.num_stations,
        station_node_ids=station_node_ids,
        num_nodes_per_graph=env.num_nodes,
        network_variant=args.network,
        use_action_mask=args.use_action_mask,
    )
    agent.load_model(args.model)
    agent.epsilon = float(args.epsilon)
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent


def _masked_valid_actions(env, ev, pending_counts, use_action_mask):
    if not use_action_mask:
        return list(range(len(env.stations)))
    action_mask = env.get_action_mask(ev, pending_counts=pending_counts)
    valid = action_mask.squeeze().nonzero(as_tuple=True)[0].tolist()
    return valid if valid else list(range(len(env.stations)))


def _select_random(env, ev, pending_counts, use_action_mask):
    valid = _masked_valid_actions(env, ev, pending_counts, use_action_mask)
    return random.choice(valid)


def _select_greedy(env, ev, pending_counts, use_action_mask):
    valid = _masked_valid_actions(env, ev, pending_counts, use_action_mask)
    best_action = valid[0]
    best_cost = float("inf")
    for action in valid:
        metrics = env.estimate_action_metrics(ev, action, pending_counts=pending_counts)
        trip = metrics.get("trip_time_h", 24.0)
        queue = metrics.get("queue_time_h", 24.0)
        fee = metrics.get("charge_cost", 1e6)
        cost = (
            0.4 * (float(trip) / NORM_TRIP)
            + 0.4 * (float(queue) / NORM_WAIT)
            + 0.2 * (float(fee) / NORM_FEE)
        )
        if cost < best_cost:
            best_cost = cost
            best_action = action
    return best_action


def _select_shortest(env, ev, pending_counts, use_action_mask):
    valid = _masked_valid_actions(env, ev, pending_counts, use_action_mask)
    best_action = valid[0]
    best_trip = float("inf")
    for action in valid:
        metrics = env.estimate_action_metrics(ev, action, pending_counts=pending_counts)
        trip = float(metrics.get("trip_time_h", 24.0))
        if trip < best_trip:
            best_trip = trip
            best_action = action
    return best_action


def _select_model(agent, env, ev, pending_counts, use_action_mask):
    state = env.get_graph_state_for_ev(ev, pending_counts=pending_counts)
    action_mask = None
    if use_action_mask:
        action_mask = env.get_action_mask(ev, pending_counts=pending_counts)
    return int(agent.select_action(state, action_mask=action_mask))


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _summarize_episode(strategy, ep_idx, episodes, steps_run, action_counts, trips, queues, fees, rewards, abandoned, env, elapsed):
    print(
        f"[{strategy} ep {ep_idx}/{episodes}] "
        f"steps={steps_run} events={len(rewards)} abandoned={abandoned} "
        f"avg_trip={_avg(trips):.4f}h avg_queue={_avg(queues):.4f}h "
        f"avg_fee={_avg(fees):.4f} avg_reward={_avg(rewards):.4f} "
        f"end_queue={sum(len(s.queue) for s in env.stations)} "
        f"end_charging={sum(len(s.connected_evs) for s in env.stations)} "
        f"actions={action_counts} elapsed={elapsed:.1f}s"
    )


def evaluate_strategy(args, strategy):
    first_env = _build_env(args, args.seed)
    agent = _build_agent(args, first_env) if strategy == "model" else None
    all_trips, all_queues, all_fees, all_rewards = [], [], [], []
    episode_reports = []
    total_abandoned = 0
    total_action_counts = [0 for _ in range(args.num_stations)]
    t0 = time.time()

    for ep in range(args.episodes):
        env = first_env if ep == 0 else _build_env(args, args.seed + ep)
        env.reset()
        steps_run = 0
        ep_trips, ep_queues, ep_fees, ep_rewards = [], [], [], []
        ep_abandoned = 0
        ep_action_counts = [0 for _ in range(args.num_stations)]

        for _ in range(args.steps_per_episode):
            urgent_evs = env.get_pending_decision_evs()
            pending_counts = {station.id: 0 for station in env.stations}
            actions = {}

            for ev in urgent_evs:
                if strategy == "model":
                    action = _select_model(agent, env, ev, pending_counts, args.use_action_mask)
                elif strategy == "shortest":
                    action = _select_shortest(env, ev, pending_counts, args.use_action_mask)
                elif strategy == "greedy":
                    action = _select_greedy(env, ev, pending_counts, args.use_action_mask)
                else:
                    action = _select_random(env, ev, pending_counts, args.use_action_mask)

                actions[ev.id] = int(action)
                if 0 <= int(action) < args.num_stations:
                    pending_counts[int(action)] = pending_counts.get(int(action), 0) + 1
                    ep_action_counts[int(action)] += 1
                    total_action_counts[int(action)] += 1

            _, _, done, info = env.step(actions)
            steps_run += 1

            for entry in info.get("charge_started", []):
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                reward = compute_hindsight_reward(trip, queue, fee)
                ep_trips.append(trip)
                ep_queues.append(queue)
                ep_fees.append(fee)
                ep_rewards.append(reward)
                all_trips.append(trip)
                all_queues.append(queue)
                all_fees.append(fee)
                all_rewards.append(reward)

            ep_abandoned += len(info.get("abandoned", []))
            if done:
                break

        total_abandoned += ep_abandoned
        elapsed = time.time() - t0
        _summarize_episode(
            strategy,
            ep + 1,
            args.episodes,
            steps_run,
            ep_action_counts,
            ep_trips,
            ep_queues,
            ep_fees,
            ep_rewards,
            ep_abandoned,
            env,
            elapsed,
        )
        episode_reports.append(
            {
                "episode": ep + 1,
                "events": len(ep_rewards),
                "abandoned": ep_abandoned,
                "avg_trip": _avg(ep_trips),
                "avg_queue": _avg(ep_queues),
                "avg_fee": _avg(ep_fees),
                "avg_reward": _avg(ep_rewards),
                "action_counts": ep_action_counts,
            }
        )

    summary = {
        "strategy": strategy,
        "episodes": args.episodes,
        "steps_per_episode": args.steps_per_episode,
        "num_evs": args.num_evs,
        "num_stations": args.num_stations,
        "num_chargers_per_station": args.num_chargers_per_station,
        "respawn": args.respawn,
        "grid_variant": args.grid_variant,
        "use_action_mask": args.use_action_mask,
        "epsilon": args.epsilon if strategy == "model" else None,
        "events": len(all_rewards),
        "abandoned": total_abandoned,
        "avg_trip": _avg(all_trips),
        "avg_queue": _avg(all_queues),
        "avg_fee": _avg(all_fees),
        "avg_reward": _avg(all_rewards),
        "action_counts": total_action_counts,
    }
    print(
        f"[summary:{strategy}] events={summary['events']} abandoned={summary['abandoned']} "
        f"avg_trip={summary['avg_trip']:.4f}h avg_queue={summary['avg_queue']:.4f}h "
        f"avg_fee={summary['avg_fee']:.4f} avg_reward={summary['avg_reward']:.4f} "
        f"actions={summary['action_counts']}"
    )
    return {"summary": summary, "episodes": episode_reports}


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"--model not found: {args.model}")

    os.makedirs(args.save_dir, exist_ok=True)
    strategies = ["random", "shortest", "greedy", "model"] if args.compare_baselines else [args.strategy]
    results = {}

    print(
        f"[setup] model={args.model} episodes={args.episodes} steps={args.steps_per_episode} "
        f"num_evs={args.num_evs} stations={args.num_stations} "
        f"chargers_per_station={args.num_chargers_per_station} respawn={args.respawn} "
        f"grid_variant={args.grid_variant} "
        f"use_action_mask={args.use_action_mask} epsilon={args.epsilon}"
    )
    for strategy in strategies:
        results[strategy] = evaluate_strategy(args, strategy)

    save_path = os.path.join(args.save_dir, "hindsight_evaluation.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] saved evaluation -> {save_path}")


if __name__ == "__main__":
    main()
