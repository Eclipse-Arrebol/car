"""A/B/C decision-mechanism ablation (inference-time switch, same weights).

Tests whether decision-layer spatial diversion (C: pending_counts injected into
state at decision time) beats no-foresight paradigms, using ONE shared trained
checkpoint:

  A  pure concurrent      : every EV in the tick is scored from the frozen
                            pre-tick state, pending_counts NOT updated.
  B  sequential, no couple: sequential decision, state WITHOUT pending_counts
                            (passed as None), no accumulation in the loop.
  C  ours (current)       : sequential + pending_counts injected into state
                            (the later EV sees how many peers already picked
                            each station this tick).

Note on A vs B: in this codebase all actions of a tick are collected first and
applied together in env.step(); the env (queue / connected / mean-field) is
FROZEN during the decision loop, and the ONLY intra-tick coupling channel is
pending_counts. So with foresight off, A and B see identical states and produce
identical actions by construction. We still run both for completeness; the real
contrast is {A,B} (no foresight) vs C (foresight).

D (reward-layer congestion penalty) is NOT here: it differs in the TRAINING
reward, not inference, so it needs a retrain. The existing reward already
contains a realized-queue penalty, so "baseline checkpoint under mode B" is a
reasonable approximation of D's trend.

Single-point fairness logic (env build, seed, metric definitions) is reused
verbatim from evaluate_oracle_wait; only the per-EV state builder is switched.
"""

import argparse
import json
import os
import sys
import time

import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from evaluate_oracle_wait import (
    _build_agent,
    _build_env,
    _gini,
    _mean,
    _pct_change,
    _std,
    _variance,
)
from trainer.trainer import compute_hindsight_reward

MODES = ("A", "B", "C")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps-per-episode", type=int, default=144)
    parser.add_argument("--num-evs", type=int, default=40)
    parser.add_argument("--num-stations", type=int, default=4)
    parser.add_argument("--num-chargers-per-station", type=int, default=8)
    parser.add_argument("--respawn", action="store_true", default=True)
    parser.add_argument("--no-respawn", dest="respawn", action="store_false")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument(
        "--network",
        choices=["original", "lightweight", "station_only", "station_attn"],
        default="station_only",
    )
    # A/B/C all run with no action mask (decision coupling under test is the
    # state channel, not the mask); kept fixed for a clean comparison.
    parser.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=False)
    parser.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=os.path.join("evaluation", "decision_ablation"))

    parser.add_argument("--graphml-file", type=str, default=os.path.join("map_outputs", "ema", "ema.graphml"))
    parser.add_argument("--cache-dir", type=str, default=os.path.join("map_outputs", "ema_cache"))
    parser.add_argument("--no-ue-background", action="store_true", default=False)
    parser.add_argument("--ue-net-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_net.tntp"))
    parser.add_argument("--ue-trips-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"))
    parser.add_argument("--ue-max-iter", type=int, default=800)
    parser.add_argument("--ue-tol", type=float, default=1e-4)
    parser.add_argument("--ue-scale", type=float, default=1.3)
    parser.add_argument("--ue-verbose", action="store_true", default=False)
    parser.add_argument("--grid-variant", type=str, default="old_city")
    return parser.parse_args()


def _select_action(agent, env, ev, foresight_counts, mode, use_action_mask):
    """Build state per decision mode and pick an action.

    Only the state-construction differs across modes (the variable under test);
    everything else (weights, argmax) is identical.
    """
    if mode == "C":
        state = env.get_graph_state_for_ev(ev, pending_counts=foresight_counts)
    else:  # A and B: no decision-time foresight
        state = env.get_graph_state_for_ev(ev, pending_counts=None)

    action_mask = None
    if use_action_mask:
        # C couples the mask too; A/B leave it uncoupled.
        pc = foresight_counts if mode == "C" else None
        action_mask = env.get_action_mask(ev, pending_counts=pc)
    return int(agent.select_action(state, action_mask=action_mask))


def run_mode(args, agent, mode):
    all_trips, all_queues, all_fees, all_rewards = [], [], [], []
    total_abandoned = 0
    action_counts = [0 for _ in range(args.num_stations)]
    station_load_integral = [0.0 for _ in range(args.num_stations)]
    min_voltages, voltage_excursions, voltage_violations = [], [], []
    t0 = time.time()

    for ep in range(args.episodes):
        env = _build_env(args, args.seed + ep)
        env.reset()

        for _ in range(args.steps_per_episode):
            urgent_evs = env.get_pending_decision_evs()
            # foresight_counts is accumulated ONLY in mode C; A/B leave it empty.
            foresight_counts = {station.id: 0 for station in env.stations}
            actions = {}
            for ev in urgent_evs:
                action = _select_action(
                    agent, env, ev, foresight_counts, mode, args.use_action_mask
                )
                actions[ev.id] = int(action)
                if 0 <= int(action) < args.num_stations:
                    action_counts[int(action)] += 1
                    if mode == "C":
                        foresight_counts[int(action)] += 1

            _, _, done, info = env.step(actions)

            for entry in info.get("charge_started", []):
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                all_trips.append(trip)
                all_queues.append(queue)
                all_fees.append(fee)
                all_rewards.append(compute_hindsight_reward(trip, queue, fee))

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
            f"[{mode} ep {ep + 1}/{args.episodes}] events={len(all_rewards)} "
            f"abandoned={total_abandoned} avg_queue={_mean(all_queues):.4f}h "
            f"avg_reward={_mean(all_rewards):.4f} elapsed={time.time() - t0:.1f}s"
        )

    events = len(all_rewards)
    denom = events + total_abandoned
    summary = {
        "mode": mode,
        "events": events,
        "abandoned": total_abandoned,
        "abandon_rate": (total_abandoned / denom) if denom > 0 else 0.0,
        "avg_trip_h": _mean(all_trips),
        "avg_queue_h": _mean(all_queues),
        "max_queue_h": max(all_queues) if all_queues else 0.0,
        "var_queue_h": _variance(all_queues),
        "std_queue_h": _std(all_queues),
        "avg_fee": _mean(all_fees),
        "avg_reward": _mean(all_rewards),
        "load_gini": _gini(station_load_integral),
        "action_gini": _gini(action_counts),
        "action_counts": action_counts,
        "avg_min_voltage_pu": _mean(min_voltages),
        "avg_voltage_excursion": _mean(voltage_excursions),
        "avg_voltage_violations": _mean(voltage_violations),
    }
    return summary


# improvement direction per metric (lower_is_better)
METRIC_ROWS = [
    ("avg_queue_h", "avg queue (h)", True),
    ("var_queue_h", "queue var", True),
    ("std_queue_h", "queue std (h)", True),
    ("max_queue_h", "max queue (h)", True),
    ("avg_reward", "avg reward", False),
    ("avg_trip_h", "avg trip (h)", True),
    ("avg_fee", "avg fee", True),
    ("abandon_rate", "abandon rate", True),
    ("load_gini", "load gini", True),
    ("action_gini", "action gini", True),
    ("avg_min_voltage_pu", "min voltage (pu)", False),
    ("avg_voltage_excursion", "voltage excursion", True),
    ("avg_voltage_violations", "voltage violations", True),
]


def print_report(res):
    A, B, C = res["A"], res["B"], res["C"]
    print("\n" + "=" * 92)
    print("DECISION-MECHANISM ABLATION  —  A / B / C  (same weights, seeds, scenario)")
    print("(C-vs-A, C-vs-B: positive % = C better)")
    print("=" * 92)
    print(f"{'metric':<20}{'A concur':>13}{'B seq/D~':>13}{'C ours':>13}{'C vs A':>12}{'C vs B':>12}")
    print("-" * 92)
    improvements = {}
    for key, label, lower in METRIC_ROWS:
        ca = _pct_change(A.get(key), C.get(key), lower)
        cb = _pct_change(B.get(key), C.get(key), lower)
        improvements[key] = {"C_vs_A": ca, "C_vs_B": cb}
        ca_s = "   n/a" if ca is None else f"{ca:+9.2f}%"
        cb_s = "   n/a" if cb is None else f"{cb:+9.2f}%"
        print(
            f"{label:<20}{A.get(key, 0):>13.4f}{B.get(key, 0):>13.4f}"
            f"{C.get(key, 0):>13.4f}{ca_s:>12}{cb_s:>12}"
        )
    print("-" * 92)
    print(f"{'events':<20}{A['events']:>13}{B['events']:>13}{C['events']:>13}")
    print(f"A actions: {A['action_counts']}")
    print(f"B actions: {B['action_counts']}")
    print(f"C actions: {C['action_counts']}")
    print("=" * 92)

    ab_identical = A["action_counts"] == B["action_counts"]
    print(
        "\nA == B (action counts identical): "
        + ("YES — expected; foresight off => same frozen state => same actions."
           if ab_identical
           else "NO — they diverged.")
    )
    rq = improvements["avg_reward"]
    qq = improvements["avg_queue_h"]
    print("C improvement (positive = C better):")
    if rq["C_vs_B"] is not None:
        print(f"  - reward: C vs B(~D) {rq['C_vs_B']:+.2f}%, C vs A {rq['C_vs_A']:+.2f}%")
    if qq["C_vs_B"] is not None:
        print(f"  - avg queue: C vs B(~D) {qq['C_vs_B']:+.2f}%, C vs A {qq['C_vs_A']:+.2f}%")
    print("=" * 92)
    return improvements


def main():
    args = parse_args()
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"--model not found: {args.model}")
    os.makedirs(args.save_dir, exist_ok=True)

    print(
        f"[setup] model={args.model} episodes={args.episodes} steps={args.steps_per_episode} "
        f"num_evs={args.num_evs} stations={args.num_stations} "
        f"chargers_per_station={args.num_chargers_per_station} grid={args.grid_variant} "
        f"ue_scale={args.ue_scale} epsilon={args.epsilon} seed={args.seed}"
    )

    ref_env = _build_env(args, args.seed)
    agent = _build_agent(args, ref_env)

    res = {}
    for mode in MODES:
        res[mode] = run_mode(args, agent, mode)

    improvements = print_report(res)

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
            "epsilon": args.epsilon,
            "seed": args.seed,
        },
        "A": res["A"],
        "B": res["B"],
        "C": res["C"],
        "improvement_pct": improvements,
    }
    save_path = os.path.join(args.save_dir, "decision_ablation.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[done] saved -> {save_path}")


if __name__ == "__main__":
    main()
