"""对比「无背景交通流」与「有背景流」下边级 BPR 通行时间（及可选最短路总时长）。

用法示例（离线路网，秒级；**小玩具图 t0 极短，BPR 比值可能失真**，仅作连通性冒烟）::

    python tools/compare_travel_time_background.py

**推荐**：用 EMA 子图 graphml（与训练一致），量级更可信::

    python tools/compare_travel_time_background.py --graphml map_outputs/ema/ema.graphml --cache-dir map_outputs/ema_cache

EMA + UE 背景（较慢，会跑 Frank–Wolfe）::

    python tools/compare_travel_time_background.py --graphml map_outputs/ema/ema.graphml \\
        --cache-dir map_outputs/ema_cache \\
        --ue-net map_outputs/ema/EMA_net.tntp --ue-trips map_outputs/ema/EMA_trips.tntp

仅看日周期中某一步（默认取 profile 峰值步）::

    python tools/compare_travel_time_background.py --step 72

打印有背景下 BPR 比值最高的几条边::

    python tools/compare_travel_time_background.py --graphml map_outputs/ema/ema.graphml --cache-dir map_outputs/ema_cache --top-edges 5
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
from typing import Callable, Sequence

import networkx as nx

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from env.real_env import RealTrafficEnv  # noqa: E402


def _min_t_h_t0(env, u: int, v: int, add_vehicle: float = 0.0) -> tuple[float, float] | None:
    row = _best_profile_row(env, u, v, add_vehicle=add_vehicle)
    if row is None:
        return None
    _length_m, _speed_kph, t_h, t0_h, _x_flow, _cap = row
    return float(t_h), float(t0_h)


def _best_profile_row(
    env, u: int, v: int, add_vehicle: float = 0.0
) -> tuple[float, float, float, float, float, float] | None:
    """返回 BPR 最紧的一条 profile: length_m, speed_kph, t_h, t0_h, x_flow, capacity_vehph。"""
    dyn = env._dynamic_profiles(u, v, add_vehicle=add_vehicle)
    if not dyn:
        return None
    best = min(dyn, key=lambda x: x[2])
    return (
        float(best[0]),
        float(best[1]),
        float(best[2]),
        float(best[3]),
        float(best[4]),
        float(best[5]),
    )


def _edge_ratio_stats(env, add_vehicle: float = 0.0) -> dict[str, float]:
    ratios: list[float] = []
    for u, v in env.traffic_graph.edges():
        got = _min_t_h_t0(env, u, v, add_vehicle=add_vehicle)
        if got is None:
            continue
        t_h, t0 = got
        if t0 <= 0:
            continue
        ratios.append(t_h / t0)
    if not ratios:
        return {"n": 0.0}
    ratios.sort()
    n = len(ratios)

    def pct(p: float) -> float:
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return ratios[i]

    return {
        "n": float(n),
        "mean": float(statistics.mean(ratios)),
        "median": float(statistics.median(ratios)),
        "p90": float(pct(0.90)),
        "p99": float(pct(0.99)),
        "max": float(ratios[-1]),
    }


def _make_weight(env) -> Callable[[int, int, dict], float]:
    def w(u: int, v: int, edge_data: dict) -> float:
        return float(env._travel_time_weight(u, v, edge_data))

    return w


def _od_path_hours(env, pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int, float]]:
    w = _make_weight(env)
    out: list[tuple[int, int, float]] = []
    for s, t in pairs:
        if s == t:
            continue
        try:
            path = nx.shortest_path(env.traffic_graph, source=s, target=t, weight=w)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            ed = env.traffic_graph.get_edge_data(u, v, default={})
            total += w(u, v, ed)
        out.append((s, t, total))
    return out


def _summarize_ratios(label: str, stats: dict[str, float]) -> None:
    if stats.get("n", 0) <= 0:
        print(f"{label}: (无边可统计)")
        return
    print(
        f"{label}: n={int(stats['n'])}  "
        f"t_h/t0  mean={stats['mean']:.4f}  median={stats['median']:.4f}  "
        f"p90={stats['p90']:.4f}  p99={stats['p99']:.4f}  max={stats['max']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare edge travel time (BPR) with vs without background traffic."
    )
    parser.add_argument("--graphml", type=str, default=None, help="optional EMA graphml path")
    parser.add_argument("--cache-dir", type=str, default=None, help="cache dir for graph load")
    parser.add_argument("--offline", action="store_true", help="use offline synthetic graph (ignored if --graphml set)")
    parser.add_argument("--max-nodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-stations", type=int, default=2)
    parser.add_argument("--step", type=int, default=None, help="time_step for background (0..steps_per_day-1)")
    parser.add_argument(
        "--ue-net",
        type=str,
        default=None,
        help="TNTP net path for UE baseline (with --ue-trips); only used with --graphml",
    )
    parser.add_argument("--ue-trips", type=str, default=None, help="TNTP trips path for UE baseline")
    parser.add_argument("--ue-max-iter", type=int, default=200)
    parser.add_argument("--ue-scale", type=float, default=1.0)
    parser.add_argument(
        "--od-samples",
        type=int,
        default=6,
        help="number of random OD pairs for shortest-path total time (0 to skip)",
    )
    parser.add_argument(
        "--top-edges",
        type=int,
        default=0,
        metavar="K",
        help="print top K edges by t_h/t0 under current background (0=skip)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="cmp_bg_travel_")
    if args.cache_dir is None:
        print(f"[compare_travel_time_background] temp cache_dir={cache_dir}")

    ue_net = args.ue_net
    ue_trips = args.ue_trips
    if (ue_net is None) ^ (ue_trips is None):
        raise SystemExit("--ue-net and --ue-trips must be passed together or both omitted.")

    if args.graphml:
        env = RealTrafficEnv(
            graphml_file=args.graphml,
            num_stations=args.num_stations,
            num_evs=0,
            max_nodes=args.max_nodes,
            cache_dir=cache_dir,
            seed=args.seed,
            offline=False,
            respawn_after_full_charge=False,
            background_ue_net_tntp=ue_net,
            background_ue_trips_tntp=ue_trips,
            background_ue_max_iter=args.ue_max_iter,
            background_ue_scale=args.ue_scale,
        )
    else:
        env = RealTrafficEnv(
            offline=True,
            num_evs=0,
            num_stations=args.num_stations,
            max_nodes=args.max_nodes,
            seed=args.seed,
            cache_dir=cache_dir,
            respawn_after_full_charge=False,
        )

    profile = list(env.background_daily_profile)
    peak_step = max(range(len(profile)), key=lambda i: profile[i])
    step = args.step if args.step is not None else peak_step
    step = step % env.steps_per_day

    saved_base = dict(env.background_edge_base_flows)

    # 无背景：基线与当步流均置空（与 test_bpr_congestion 中「关背景」一致）
    env.background_edge_base_flows = {}
    env.background_edge_flows = {}
    env.edge_active_counts = {}
    env.time_step = step
    stats_off = _edge_ratio_stats(env, add_vehicle=0.0)

    env.background_edge_base_flows = saved_base
    env.time_step = step
    env.update_background_traffic()
    env.edge_active_counts = {}
    stats_on = _edge_ratio_stats(env, add_vehicle=0.0)

    print("=== 边级 BPR（无车上边计数，add_vehicle=0）===")
    print(f"路网: graphml={args.graphml!r}  offline={not args.graphml}  |E|={env.traffic_graph.number_of_edges()}")
    print(f"考察 time_step={step}（日 profile≈{profile[step]:.4f}；峰值步={peak_step}）")
    if ue_net:
        print(f"UE 背景: net={ue_net!r} trips={ue_trips!r} scale={args.ue_scale} max_iter={args.ue_max_iter}")
    _summarize_ratios("无背景流（t_h/t0 应≈1）", stats_off)
    _summarize_ratios("有背景流（t_h/t0）", stats_on)
    if stats_off.get("n", 0) > 0 and stats_on.get("n", 0) > 0:
        print(
            f"Δmedian(有-无) = {stats_on['median'] - stats_off['median']:.4f}  "
            f"相对升幅 ≈ {(stats_on['median'] / max(stats_off['median'], 1e-9) - 1.0) * 100:.2f}%"
        )
        if stats_on["p99"] > 5 * max(stats_on["median"], 1e-9):
            print(
                "提示：p99 >> median 时，背景流/BPR 延迟主要集中在少数边上；"
                "OD 最短路若不经这些边，总时长变化可能仍很小。"
            )

    if args.top_edges > 0:
        ranked: list[tuple[float, int, int, float, float, float, float]] = []
        for u, v in env.traffic_graph.edges():
            row = _best_profile_row(env, u, v, add_vehicle=0.0)
            if row is None:
                continue
            _lm, _sk, t_h, t0_h, x_flow, cap = row
            if t0_h <= 0:
                continue
            ratio = t_h / t0_h
            ranked.append((ratio, u, v, t_h, t0_h, x_flow, cap))
        ranked.sort(key=lambda item: item[0], reverse=True)
        k = min(args.top_edges, len(ranked))
        print()
        print(f"=== 有背景下 t_h/t0 最高的 {k} 条边（add_vehicle=0）===")
        for i, (ratio, u, v, t_h, t0_h, x_flow, cap) in enumerate(ranked[:k], start=1):
            print(
                f"  {i}. ({u},{v})  t_h/t0={ratio:.4f}  t_h={t_h:.6f}h  t0={t0_h:.6f}h  "
                f"x_flow={x_flow:.2f}  cap={cap:.1f} veh/h"
            )

    if args.od_samples > 0:
        nodes = list(env.traffic_graph.nodes())
        rng = __import__("random").Random(args.seed)
        pairs: list[tuple[int, int]] = []
        for _ in range(args.od_samples * 4):
            s = rng.choice(nodes)
            t = rng.choice(nodes)
            if s != t and (s, t) not in pairs:
                pairs.append((s, t))
            if len(pairs) >= args.od_samples:
                break

        env.background_edge_base_flows = {}
        env.background_edge_flows = {}
        env.edge_active_counts = {}
        env.time_step = step
        env._path_cache_step = {}
        od_off = _od_path_hours(env, pairs)

        env.background_edge_base_flows = saved_base
        env.time_step = step
        env.update_background_traffic()
        env.edge_active_counts = {}
        env._path_cache_step = {}
        od_on = _od_path_hours(env, pairs)

        by_key_off = {(a, b): h for a, b, h in od_off}
        by_key_on = {(a, b): h for a, b, h in od_on}
        print()
        print(f"=== 最短路总时长（权重与 env 一致，含每条边 +1 车的边际项），采样 {len(pairs)} 对 OD ===")
        for s, t in pairs:
            ho = by_key_off.get((s, t))
            hn = by_key_on.get((s, t))
            if ho is None or hn is None:
                continue
            rel = (hn / ho - 1.0) * 100.0 if ho > 0 else float("nan")
            print(f"  {s}->{t}: 无背景 {ho:.4f} h  |  有背景 {hn:.4f} h  |  相对 {rel:+.2f}%")


if __name__ == "__main__":
    main()
