"""Evaluate one or more checkpoints across the 3 city scenarios.

Used for the Local / Centralized / Federated paradigm comparison. Reuses the
single-point fair-evaluation logic from evaluate_oracle_wait (run_mode "baseline"
= the standard C decision), so metric definitions / seeds / scenarios are
identical to every other experiment in this project.

Each model is evaluated on every city; the aggregator (aggregate_paradigms.py)
later picks the right cells (Local uses the diagonal: each city's own model).

Usage:
    python eval_paradigms.py --out evaluation/paradigm/federated.json \
        --models federated_r10=checkpoints_fed_hindsight_40ev/global_round10.pth \
                 federated_final=checkpoints_fed_hindsight_40ev/global_final.pth
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from evaluate_oracle_wait import _build_agent, _build_env, run_mode

DEFAULT_CITIES = "old_city:1.3,new_city:1.0,suburb:0.7"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="list of label=path pairs")
    p.add_argument("--cities", type=str, default=DEFAULT_CITIES)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps-per-episode", type=int, default=144)
    p.add_argument("--num-evs", type=int, default=40)
    p.add_argument("--num-stations", type=int, default=4)
    p.add_argument("--num-chargers-per-station", type=int, default=8)
    p.add_argument("--network", choices=["original", "lightweight", "station_only", "station_attn"], default="station_only")
    p.add_argument("--epsilon", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", default=False,
                   help="load an existing --out and skip labels already evaluated "
                        "(continue after a killed run without redoing finished cells)")
    return p.parse_args()


def parse_cities(spec_text):
    cities = []
    for raw in spec_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        variant, scale = raw.split(":", 1)
        cities.append({"grid_variant": variant.strip(), "ue_scale": float(scale)})
    return cities


def make_eval_args(base, model_path, city):
    return SimpleNamespace(
        model=model_path,
        episodes=base.episodes,
        steps_per_episode=base.steps_per_episode,
        num_evs=base.num_evs,
        num_stations=base.num_stations,
        num_chargers_per_station=base.num_chargers_per_station,
        respawn=True,
        epsilon=base.epsilon,
        network=base.network,
        use_action_mask=False,
        seed=base.seed,
        save_dir=os.path.dirname(base.out) or ".",
        graphml_file=os.path.join("map_outputs", "ema", "ema.graphml"),
        cache_dir=os.path.join("map_outputs", "ema_cache"),
        no_ue_background=False,
        ue_net_tntp=os.path.join("map_outputs", "ema", "EMA_net.tntp"),
        ue_trips_tntp=os.path.join("map_outputs", "ema", "EMA_trips.tntp"),
        ue_max_iter=800,
        ue_tol=1e-4,
        ue_scale=city["ue_scale"],
        ue_verbose=False,
        grid_variant=city["grid_variant"],
    )


def main():
    base = parse_args()
    cities = parse_cities(base.cities)
    os.makedirs(os.path.dirname(base.out) or ".", exist_ok=True)

    pairs = []
    for item in base.models:
        if "=" not in item:
            raise ValueError(f"--models entry must be label=path: {item!r}")
        label, path = item.split("=", 1)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found for {label}: {path}")
        pairs.append((label, path))

    def _flush(results_so_far):
        out = {
            "config": {
                "cities": cities,
                "episodes": base.episodes,
                "steps_per_episode": base.steps_per_episode,
                "num_evs": base.num_evs,
                "num_stations": base.num_stations,
                "num_chargers_per_station": base.num_chargers_per_station,
                "epsilon": base.epsilon,
                "seed": base.seed,
            },
            "results": results_so_far,
        }
        tmp = base.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.replace(tmp, base.out)

    results = {}
    if base.resume and os.path.isfile(base.out):
        try:
            prev = json.load(open(base.out, encoding="utf-8"))
            results = prev.get("results", {})
            print(f"[resume] loaded {len(results)} completed labels from {base.out}: "
                  f"{list(results)}")
        except (ValueError, OSError) as e:
            print(f"[resume] could not load {base.out} ({e}); starting fresh")

    for label, path in pairs:
        if label in results:
            print(f"### skip {label} (already in {base.out})")
            continue
        per_city = {}
        for city in cities:
            args = make_eval_args(base, path, city)
            print(f"\n### eval {label} on {city['grid_variant']} (ue={city['ue_scale']})")
            env = _build_env(args, args.seed)
            agent = _build_agent(args, env)
            summary = run_mode(args, agent, "baseline")
            denom = summary["events"] + summary["abandoned"]
            summary["abandon_rate"] = (summary["abandoned"] / denom) if denom > 0 else 0.0
            per_city[city["grid_variant"]] = summary

        avg_keys = [
            "avg_reward", "avg_queue_h", "avg_trip_h", "avg_fee",
            "load_gini", "action_gini", "abandon_rate", "events",
        ]
        average = {
            k: sum(per_city[c["grid_variant"]][k] for c in cities) / len(cities)
            for k in avg_keys
        }
        results[label] = {"model": path, "per_city": per_city, "average": average}
        _flush(results)  # incremental: survive a mid-run death
        print(
            f"[{label}] AVG reward={average['avg_reward']:.4f} "
            f"queue={average['avg_queue_h']:.4f}h trip={average['avg_trip_h']:.4f}h "
            f"abandon={average['abandon_rate']*100:.1f}% load_gini={average['load_gini']:.4f} "
            f"(flushed -> {base.out})"
        )

    print(f"\n[done] saved -> {base.out}")


if __name__ == "__main__":
    main()
