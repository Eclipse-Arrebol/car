"""Oracle ideal-arrival-wait upper-bound experiment.

Runs the SAME trained checkpoint in two inference modes on identical scenarios /
seeds, the only difference being which state-builder feeds the model:

  - baseline : env.get_graph_state_for_ev          (current instantaneous mean
               field / pending_counts, i.e. spatial diversion only)
  - oracle   : env.get_graph_state_for_ev_oracle   (cheating: feat 11/15 replaced
               with the ground-truth ideal arrival wait from a multi-server FIFO
               occupancy simulation)

If oracle barely beats baseline, dynamic occupancy modelling is not worth the
engineering effort. No training, no network change, no dimension change.
"""

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
from reward_profiles import weights_for
from trainer.trainer import compute_hindsight_reward


DEFAULT_NUM_EVS = 40
DEFAULT_NUM_STATIONS = 4
DEFAULT_NUM_CHARGERS_PER_STATION = 8

MODES = ("baseline", "oracle")


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
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--network",
        choices=["original", "lightweight", "station_only", "station_attn"],
        default="station_only",
    )
    parser.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=False)
    parser.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join("evaluation", "oracle_wait"),
    )

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
    if getattr(args, "station_node_ids", None):
        env_kw["station_node_ids"] = list(args.station_node_ids)
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


def _select_model(agent, env, ev, pending_counts, use_action_mask, mode):
    if mode == "oracle":
        state = env.get_graph_state_for_ev_oracle(ev, pending_counts=pending_counts)
    else:
        state = env.get_graph_state_for_ev(ev, pending_counts=pending_counts)
    action_mask = None
    if use_action_mask:
        action_mask = env.get_action_mask(ev, pending_counts=pending_counts)
    return int(agent.select_action(state, action_mask=action_mask))


# --------------------------- metrics helpers ---------------------------------
def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _variance(values):
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return sum((v - mu) ** 2 for v in values) / len(values)


def _std(values):
    return _variance(values) ** 0.5


def _gini(values):
    """Gini coefficient of a non-negative distribution (0 = perfectly even)."""
    vals = [max(0.0, float(v)) for v in values]
    n = len(vals)
    total = sum(vals)
    if n == 0 or total <= 1e-12:
        return 0.0
    vals.sort()
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


# ------------------------------ run one mode ---------------------------------
def run_mode(args, agent, mode):
    """Run all episodes for one inference mode and aggregate metrics."""
    all_trips, all_queues, all_fees, all_rewards = [], [], [], []
    total_abandoned = 0
    action_counts = [0 for _ in range(args.num_stations)]
    # time-integrated realized load per station (for load-distribution gini/std)
    station_load_integral = [0.0 for _ in range(args.num_stations)]
    min_voltages, voltage_excursions, voltage_violations = [], [], []
    # per-city reward weights (uniform unless HETERO_REWARD=1)
    _rw = weights_for(getattr(args, "grid_variant", None))
    t0 = time.time()

    for ep in range(args.episodes):
        # Same seed per episode as the other mode -> identical initial conditions.
        env = _build_env(args, args.seed + ep)
        env.reset()

        for _ in range(args.steps_per_episode):
            urgent_evs = env.get_pending_decision_evs()
            pending_counts = {station.id: 0 for station in env.stations}
            actions = {}
            for ev in urgent_evs:
                action = _select_model(
                    agent, env, ev, pending_counts, args.use_action_mask, mode
                )
                actions[ev.id] = int(action)
                if 0 <= int(action) < args.num_stations:
                    pending_counts[int(action)] += 1
                    action_counts[int(action)] += 1

            _, _, done, info = env.step(actions)

            for entry in info.get("charge_started", []):
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                all_trips.append(trip)
                all_queues.append(queue)
                all_fees.append(fee)
                all_rewards.append(compute_hindsight_reward(trip, queue, fee, *_rw))

            total_abandoned += len(info.get("abandoned", []))

            grid_loads = info.get("grid_loads", {})
            for i, station in enumerate(env.stations):
                station_load_integral[i] += float(
                    grid_loads.get(station.power_node_id, station.last_total_load)
                )
            if "min_voltage_pu" in info:
                min_voltages.append(float(info["min_voltage_pu"]))
            if "voltage_excursion" in info:
                voltage_excursions.append(float(info["voltage_excursion"]))
            if "voltage_violations" in info:
                voltage_violations.append(float(info["voltage_violations"]))

            if done:
                break

        print(
            f"[{mode} ep {ep + 1}/{args.episodes}] "
            f"events={len(all_rewards)} abandoned={total_abandoned} "
            f"avg_queue={_mean(all_queues):.4f}h avg_reward={_mean(all_rewards):.4f} "
            f"elapsed={time.time() - t0:.1f}s"
        )

    summary = {
        "mode": mode,
        "events": len(all_rewards),
        "abandoned": total_abandoned,
        "avg_trip_h": _mean(all_trips),
        "avg_queue_h": _mean(all_queues),
        "max_queue_h": max(all_queues) if all_queues else 0.0,
        "var_queue_h": _variance(all_queues),
        "std_queue_h": _std(all_queues),
        "avg_fee": _mean(all_fees),
        "avg_reward": _mean(all_rewards),
        "load_gini": _gini(station_load_integral),
        "load_std": _std(station_load_integral),
        "action_counts": action_counts,
        "action_gini": _gini(action_counts),
        "avg_min_voltage_pu": _mean(min_voltages),
        "avg_voltage_excursion": _mean(voltage_excursions),
        "avg_voltage_violations": _mean(voltage_violations),
    }
    return summary


# ------------------------------- reporting -----------------------------------
def _pct_change(base, new, lower_is_better):
    if base is None or new is None:
        return None
    if abs(base) < 1e-12:
        return None
    delta = (new - base) / abs(base) * 100.0
    # express as "improvement %": positive = oracle is better
    return delta if not lower_is_better else -delta


def print_report(baseline, oracle):
    # key: (label, lower_is_better)
    rows = [
        ("avg_queue_h", "avg queue (h)", True),
        ("max_queue_h", "max queue (h)", True),
        ("std_queue_h", "queue std (h)", True),
        ("var_queue_h", "queue var", True),
        ("avg_trip_h", "avg trip (h)", True),
        ("avg_fee", "avg fee", True),
        ("avg_reward", "avg reward", False),
        ("abandoned", "abandoned", True),
        ("load_gini", "load gini", True),
        ("load_std", "load std", True),
        ("action_gini", "action gini", True),
        ("avg_min_voltage_pu", "min voltage (pu)", False),
        ("avg_voltage_excursion", "voltage excursion", True),
        ("avg_voltage_violations", "voltage violations", True),
    ]

    print("\n" + "=" * 78)
    print("ORACLE IDEAL-ARRIVAL-WAIT  —  baseline vs oracle (same weights/seeds)")
    print("=" * 78)
    print(f"{'metric':<22}{'baseline':>15}{'oracle':>15}{'improvement':>14}")
    print("-" * 78)
    improvements = {}
    for key, label, lower_is_better in rows:
        b = baseline.get(key)
        o = oracle.get(key)
        imp = _pct_change(b, o, lower_is_better)
        improvements[key] = imp
        imp_str = "    n/a" if imp is None else f"{imp:+12.2f}%"
        print(f"{label:<22}{b:>15.4f}{o:>15.4f}{imp_str:>14}")
    print("-" * 78)
    print(f"{'events':<22}{baseline['events']:>15}{oracle['events']:>15}")
    print(f"baseline actions: {baseline['action_counts']}")
    print(f"oracle   actions: {oracle['action_counts']}")
    print("=" * 78)

    # short verdict
    q = improvements.get("avg_queue_h")
    r = improvements.get("avg_reward")
    print("\nSummary (positive % = oracle better):")
    if q is not None:
        print(f"  - avg queue improved by {q:+.2f}%")
    if r is not None:
        print(f"  - avg reward improved by {r:+.2f}%")
    ranked = sorted(
        ((k, v) for k, v in improvements.items() if v is not None),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if ranked:
        top = ", ".join(f"{k} ({v:+.1f}%)" for k, v in ranked[:3])
        print(f"  - biggest gains: {top}")
    return improvements


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"--model not found: {args.model}")
    os.makedirs(args.save_dir, exist_ok=True)

    print(
        f"[setup] model={args.model} episodes={args.episodes} steps={args.steps_per_episode} "
        f"num_evs={args.num_evs} stations={args.num_stations} "
        f"chargers_per_station={args.num_chargers_per_station} respawn={args.respawn} "
        f"grid_variant={args.grid_variant} use_action_mask={args.use_action_mask} "
        f"epsilon={args.epsilon} seed={args.seed}"
    )

    # one shared agent (same weights for both modes)
    ref_env = _build_env(args, args.seed)
    agent = _build_agent(args, ref_env)

    results = {}
    for mode in MODES:
        results[mode] = run_mode(args, agent, mode)

    improvements = print_report(results["baseline"], results["oracle"])

    out = {
        "config": {
            "model": args.model,
            "episodes": args.episodes,
            "steps_per_episode": args.steps_per_episode,
            "num_evs": args.num_evs,
            "num_stations": args.num_stations,
            "num_chargers_per_station": args.num_chargers_per_station,
            "grid_variant": args.grid_variant,
            "ue_scale": args.ue_scale,
            "use_action_mask": args.use_action_mask,
            "epsilon": args.epsilon,
            "seed": args.seed,
        },
        "baseline": results["baseline"],
        "oracle": results["oracle"],
        "improvement_pct": improvements,
    }
    save_path = os.path.join(args.save_dir, "oracle_wait_evaluation.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[done] saved -> {save_path}")


if __name__ == "__main__":
    main()
