"""绘制某条边在 0-144 step 的背景流强度曲线（RealTrafficEnv 离线路网，与训练主路径一致）。"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import matplotlib.pyplot as plt

from env.background_traffic import build_base_background_flows, build_daily_profile
from env.real_env import RealTrafficEnv


def parse_edge(text: str) -> tuple[int, int]:
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("edge must be formatted as 'u,v'")
    try:
        u = int(parts[0].strip())
        v = int(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("edge nodes must be integers") from exc
    return u, v


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot background traffic strength for one edge (offline RealTrafficEnv)")
    parser.add_argument(
        "--edge",
        type=str,
        default=None,
        help="edge as 'u,v'; default: lexicographically first edge in the graph",
    )
    parser.add_argument("--steps", type=int, default=144, help="steps per day")
    parser.add_argument("--output", type=str, default="background_traffic_edge.png", help="output image path")
    parser.add_argument("--max-nodes", type=int, default=16, help="offline synthetic graph max nodes")
    parser.add_argument("--seed", type=int, default=0, help="graph / station sampling seed")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="directory for offline network pickle cache (default: temporary directory)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="dbg_bg_traffic_")
    if args.cache_dir is None:
        print(f"[debug_background_traffic] temp cache_dir={cache_dir}")

    env = RealTrafficEnv(
        offline=True,
        num_evs=0,
        num_stations=2,
        max_nodes=args.max_nodes,
        seed=args.seed,
        cache_dir=cache_dir,
        respawn_after_full_charge=False,
    )

    profile = build_daily_profile(args.steps)
    base_flows = build_base_background_flows(env.traffic_graph.edges(), env.traffic_graph.nodes())

    if args.edge:
        edge = parse_edge(args.edge)
    else:
        edge = next(iter(sorted(env.traffic_graph.edges())))

    if edge not in base_flows and (edge[1], edge[0]) in base_flows:
        edge = (edge[1], edge[0])

    base_flow = base_flows.get(edge)
    if base_flow is None:
        available = ", ".join(f"{u}-{v}" for u, v in sorted(base_flows.keys()))
        raise SystemExit(f"edge {edge} not found in graph. available edges: {available}")

    background_flow = [base_flow * profile[t % args.steps] for t in range(args.steps + 1)]
    steps = list(range(args.steps + 1))

    plt.figure(figsize=(10, 4.5))
    plt.plot(steps, background_flow, linewidth=2.5)
    plt.xlabel("Step")
    plt.ylabel("Background flow (veh/h)")
    plt.title(f"Background traffic strength for edge {edge[0]}-{edge[1]} (offline RealTrafficEnv)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
