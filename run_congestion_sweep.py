"""Congestion-intensity sweep around the oracle ideal-arrival-wait experiment.

Hypothesis under test:
    "Time-staggering (temporal) coordination only pays off once the system is
     actually queue-constrained."

Method: keep EV count and everything else fixed, fix the number of stations
(so the action space / geography / checkpoint stay identical), and sweep ONLY
chargers-per-station to shrink total charging capacity from loose to severe.
Each tier runs baseline vs oracle through the UNCHANGED single-point fairness
logic imported from evaluate_oracle_wait (same weights, same per-episode seed,
the only difference being whether the model sees baseline or oracle state).

This wrapper adds an outer loop + aggregation only. It does not touch the
baseline/oracle comparison logic.
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

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
    _pct_change,
    run_mode,
)


# Fixed stations -> action dim 4 matches the checkpoint; sweep chargers only.
DEFAULT_TIERS = [
    {"num_stations": 4, "num_chargers_per_station": 8},   # total 32 (loose, = prior run)
    {"num_stations": 4, "num_chargers_per_station": 4},   # total 16
    {"num_stations": 4, "num_chargers_per_station": 2},   # total 8
    {"num_stations": 4, "num_chargers_per_station": 1},   # total 4  (severe)
]

# improvement direction per metric (lower_is_better)
METRIC_DIRECTION = {
    "avg_queue_h": True,
    "max_queue_h": True,
    "std_queue_h": True,
    "var_queue_h": True,
    "avg_trip_h": True,
    "avg_fee": True,
    "avg_reward": False,
    "abandoned": True,
    "load_gini": True,
    "load_std": True,
    "action_gini": True,
    "avg_min_voltage_pu": False,
    "avg_voltage_excursion": True,
    "avg_voltage_violations": True,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        default=os.path.join("checkpoints_fed_hindsight_40ev", "global_final.pth"),
    )
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps-per-episode", type=int, default=144)
    p.add_argument("--num-evs", type=int, default=40)
    p.add_argument("--grid-variant", type=str, default="old_city")
    p.add_argument("--ue-scale", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default=os.path.join("evaluation", "congestion_sweep"))
    return p.parse_args()


def make_single_point_args(sweep, tier):
    """Build the argparse-style namespace expected by evaluate_oracle_wait helpers.

    Mirrors evaluate_oracle_wait.parse_args() defaults; only the swept fields and
    the shared fixed config differ.
    """
    return SimpleNamespace(
        model=sweep.model,
        episodes=sweep.episodes,
        steps_per_episode=sweep.steps_per_episode,
        num_evs=sweep.num_evs,
        num_stations=tier["num_stations"],
        num_chargers_per_station=tier["num_chargers_per_station"],
        respawn=True,
        epsilon=0.0,
        network="station_only",
        use_action_mask=False,
        seed=sweep.seed,
        save_dir=sweep.save_dir,
        graphml_file=os.path.join("map_outputs", "ema", "ema.graphml"),
        cache_dir=os.path.join("map_outputs", "ema_cache"),
        no_ue_background=False,
        ue_net_tntp=os.path.join("map_outputs", "ema", "EMA_net.tntp"),
        ue_trips_tntp=os.path.join("map_outputs", "ema", "EMA_trips.tntp"),
        ue_max_iter=800,
        ue_tol=1e-4,
        ue_scale=sweep.ue_scale,
        ue_verbose=False,
        grid_variant=sweep.grid_variant,
    )


def _abandon_rate(summary):
    denom = summary["events"] + summary["abandoned"]
    return (summary["abandoned"] / denom) if denom > 0 else 0.0


def run_tier(sweep, tier):
    args = make_single_point_args(sweep, tier)
    total_chargers = tier["num_stations"] * tier["num_chargers_per_station"]
    label = f"{tier['num_stations']}st x {tier['num_chargers_per_station']}ch (={total_chargers})"
    print("\n" + "#" * 78)
    print(f"# TIER {label}  | evs={args.num_evs} grid={args.grid_variant} ue={args.ue_scale}")
    print("#" * 78)

    # One shared agent (same weights) for both modes in this tier.
    ref_env = _build_env(args, args.seed)
    agent = _build_agent(args, ref_env)

    baseline = run_mode(args, agent, "baseline")
    oracle = run_mode(args, agent, "oracle")

    improvements = {}
    for key, lower_is_better in METRIC_DIRECTION.items():
        improvements[key] = _pct_change(
            baseline.get(key), oracle.get(key), lower_is_better
        )

    return {
        "label": label,
        "num_stations": tier["num_stations"],
        "num_chargers_per_station": tier["num_chargers_per_station"],
        "total_chargers": total_chargers,
        "baseline": baseline,
        "oracle": oracle,
        "improvement_pct": improvements,
        "diagnostics": {
            "baseline_avg_queue_min": baseline["avg_queue_h"] * 60.0,
            "oracle_avg_queue_min": oracle["avg_queue_h"] * 60.0,
            "baseline_abandon_rate": _abandon_rate(baseline),
            "oracle_abandon_rate": _abandon_rate(oracle),
            "reward_improvement_pct": improvements["avg_reward"],
            "queue_mean_improvement_pct": improvements["avg_queue_h"],
            "queue_var_improvement_pct": improvements["var_queue_h"],
        },
    }


def _fmt_pct(v):
    return "   n/a" if v is None else f"{v:+7.2f}%"


def print_summary(tiers_results):
    # ordered loose -> tight (descending total chargers)
    rows = sorted(tiers_results, key=lambda r: r["total_chargers"], reverse=True)

    print("\n" + "=" * 96)
    print("CONGESTION SWEEP  —  oracle (temporal upper bound) vs baseline")
    print("(positive % = oracle better; baseline queue/abandon = the congestion axis)")
    print("=" * 96)
    header = (
        f"{'tier (total ch)':<20}{'base queue(min)':>16}{'base abandon%':>15}"
        f"{'reward +%':>12}{'queue mean +%':>15}{'queue var +%':>14}"
    )
    print(header)
    print("-" * 96)
    for r in rows:
        d = r["diagnostics"]
        print(
            f"{r['label']:<20}"
            f"{d['baseline_avg_queue_min']:>16.2f}"
            f"{d['baseline_abandon_rate'] * 100:>14.1f}%"
            f"{_fmt_pct(d['reward_improvement_pct']):>12}"
            f"{_fmt_pct(d['queue_mean_improvement_pct']):>15}"
            f"{_fmt_pct(d['queue_var_improvement_pct']):>14}"
        )
    print("=" * 96)

    # monotonicity verdict on reward improvement vs tightening congestion
    seq = [
        (r["total_chargers"], r["diagnostics"]["reward_improvement_pct"])
        for r in rows
        if r["diagnostics"]["reward_improvement_pct"] is not None
    ]
    verdict_lines = []
    if len(seq) >= 2:
        vals = [v for _, v in seq]
        monotonic = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
        tightest = rows[-1]
        tr = tightest["diagnostics"]["reward_improvement_pct"]
        verdict_lines.append(
            f"reward improvement as congestion tightens: {[round(v, 2) for v in vals]}"
        )
        verdict_lines.append(
            "  -> monotonically increasing with congestion"
            if monotonic
            else "  -> NOT monotonic"
        )
        if tr is not None:
            magnitude = "two-digit (>=10%)" if abs(tr) >= 10 else "single-digit (<10%)"
            verdict_lines.append(
                f"  -> tightest tier ({tightest['label']}) reward improvement = "
                f"{tr:+.2f}%  [{magnitude}]"
            )

    # abandon caveat
    caveats = [
        r["label"]
        for r in rows
        if r["diagnostics"]["baseline_abandon_rate"] > 0.10
    ]
    if caveats:
        verdict_lines.append(
            "CAVEAT: high abandon rate (>10%) at tiers "
            + ", ".join(caveats)
            + " — cars cannot enter rather than queueing; "
            "treat queue metrics there with care (abandon != queue improvement)."
        )

    print("\n".join(verdict_lines))
    print("=" * 96)


def main():
    sweep = parse_args()
    if not os.path.isfile(sweep.model):
        raise FileNotFoundError(f"--model not found: {sweep.model}")
    os.makedirs(sweep.save_dir, exist_ok=True)

    print(
        f"[sweep] model={sweep.model} evs={sweep.num_evs} grid={sweep.grid_variant} "
        f"ue_scale={sweep.ue_scale} episodes={sweep.episodes} "
        f"steps={sweep.steps_per_episode} seed={sweep.seed}"
    )
    print(f"[sweep] tiers: {[t for t in DEFAULT_TIERS]}")

    t0 = time.time()
    tiers_results = []
    for tier in DEFAULT_TIERS:
        tiers_results.append(run_tier(sweep, tier))
        print(f"[sweep] elapsed so far: {time.time() - t0:.1f}s")

    print_summary(tiers_results)

    out = {
        "config": {
            "model": sweep.model,
            "num_evs": sweep.num_evs,
            "grid_variant": sweep.grid_variant,
            "ue_scale": sweep.ue_scale,
            "episodes": sweep.episodes,
            "steps_per_episode": sweep.steps_per_episode,
            "seed": sweep.seed,
            "tiers": DEFAULT_TIERS,
        },
        "tiers": tiers_results,
    }
    save_path = os.path.join(sweep.save_dir, "congestion_sweep.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[done] total {time.time() - t0:.1f}s  saved -> {save_path}")


if __name__ == "__main__":
    main()
