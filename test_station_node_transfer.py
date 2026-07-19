"""Cheap (eval-only) test: does a feature-based model transfer across station
placements? If yes, swapping charging-station nodes per client will NOT create
the policy heterogeneity federated needs.

Take local_new (trained on new_city with the default station nodes) and evaluate
it on the SAME new_city grid but with different station-node placements:
  - default    : the auto-selected nodes it was trained on
  - clustered  : 4 mutually-close nodes (changes congestion/spatial dynamics most)
  - spread_alt : a different spread-out set

If reward stays ~flat across placements -> policy is node-invariant (node swap
insufficient). If it collapses on the new placements -> node placement induces
heterogeneity (worth a heterogeneous retrain).
"""

import sys
from types import SimpleNamespace

import networkx as nx

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from evaluate_oracle_wait import _build_agent, _build_env, run_mode

MODEL = "checkpoints_local_new_city\\model_final.pth"
CITY = {"grid_variant": "new_city", "ue_scale": 1.0}
SEED = 42


def base_args(station_node_ids=None):
    return SimpleNamespace(
        model=MODEL, episodes=10, steps_per_episode=144, num_evs=40,
        num_stations=4, num_chargers_per_station=8, respawn=True, epsilon=0.0,
        network="station_only", use_action_mask=False, seed=SEED,
        save_dir="evaluation/station_transfer",
        graphml_file="map_outputs/ema/ema.graphml", cache_dir="map_outputs/ema_cache",
        no_ue_background=False,
        ue_net_tntp="map_outputs/ema/EMA_net.tntp",
        ue_trips_tntp="map_outputs/ema/EMA_trips.tntp",
        ue_max_iter=800, ue_tol=1e-4, ue_scale=CITY["ue_scale"], ue_verbose=False,
        grid_variant=CITY["grid_variant"],
        station_node_ids=station_node_ids,
    )


def _pos(node_positions, n):
    p = node_positions[n]
    return (float(p[0]), float(p[1]))


def pick_placements(env):
    g = env.traffic_graph
    pos = env.node_positions
    default = list(env.station_node_ids)
    nodes = [n for n in g.nodes() if n in pos]

    def connected_set(cand):
        for a in cand:
            for b in cand:
                if a != b and not nx.has_path(g, a, b):
                    return False
        return True

    # clustered: a center + its 3 nearest neighbors by euclidean distance
    best = None
    for c in nodes:
        cx, cy = _pos(pos, c)
        nearest = sorted(
            (n for n in nodes if n != c),
            key=lambda n: (_pos(pos, n)[0] - cx) ** 2 + (_pos(pos, n)[1] - cy) ** 2,
        )[:3]
        cand = [c] + nearest
        if len(set(cand)) == 4 and connected_set(cand) and sorted(cand) != sorted(default):
            best = cand
            break
    clustered = best if best else default

    # spread_alt: 4 distinct non-default, far-apart nodes (greedy farthest-point)
    pool = [n for n in nodes if n not in default]
    spread = [pool[0]]
    while len(spread) < 4 and len(spread) < len(pool):
        far = max(
            (n for n in pool if n not in spread),
            key=lambda n: min(
                (_pos(pos, n)[0] - _pos(pos, s)[0]) ** 2 + (_pos(pos, n)[1] - _pos(pos, s)[1]) ** 2
                for s in spread
            ),
        )
        spread.append(far)
    spread_alt = spread if connected_set(spread) else default

    return {"default": default, "clustered": clustered, "spread_alt": spread_alt}


def main():
    ref = _build_env(base_args(), SEED)
    placements = pick_placements(ref)
    print(f"[setup] model={MODEL} city={CITY['grid_variant']}")
    for name, nodes in placements.items():
        print(f"  {name:10s} -> {nodes}")

    results = {}
    for name, nodes in placements.items():
        args = base_args(station_node_ids=nodes)
        env = _build_env(args, SEED)
        agent = _build_agent(args, env)
        summary = run_mode(args, agent, "baseline")
        results[name] = summary
        print(
            f"[{name}] stations={list(env.station_node_ids)} "
            f"reward={summary['avg_reward']:.4f} queue={summary['avg_queue_h']:.4f}h "
            f"trip={summary['avg_trip_h']:.4f}h load_gini={summary['load_gini']:.4f}"
        )

    base = results["default"]["avg_reward"]
    print("\n=== transfer across station placements (local_new on new_city) ===")
    for name in ("default", "clustered", "spread_alt"):
        r = results[name]["avg_reward"]
        delta = (r - base) / abs(base) * 100.0
        print(f"  {name:10s} reward={r:.4f}  vs default: {delta:+.2f}%")
    print(
        "\nReading: small deltas => policy node-invariant (node swap insufficient). "
        "Large negative deltas => node placement induces heterogeneity."
    )


if __name__ == "__main__":
    main()
