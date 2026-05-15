#!/usr/bin/env python3
"""解析 EMA_net.tntp（TNTP）并绘制有向网络拓扑图；可选叠加仿真快照的边需求 ``x_flow`` 或 BPR ``t/t0`` 上色。"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

ROOT = Path(__file__).resolve().parent.parent


def parse_ema_net_tntp(path: Path) -> tuple[nx.DiGraph, int]:
    """返回 (有向图, first_thru_node)。纯 TNTP 拓扑图用；拥堵模式以 graphml/缓存为准（节点编号与 TNTP 不一定一致）。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_meta = True
    G = nx.DiGraph()
    meta_re = re.compile(r"^<([^>]+)>\s*(.*)$")
    first_thru_node = 1

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if in_meta:
            m = meta_re.match(line)
            if m:
                key = m.group(1).strip().upper()
                val = m.group(2).strip()
                if key == "FIRST THRU NODE":
                    try:
                        first_thru_node = int(float(val.split()[0]))
                    except (ValueError, IndexError):
                        pass
                if key == "END OF METADATA":
                    in_meta = False
            continue

        if line.startswith("~"):
            continue

        parts = [p.strip().rstrip(";") for p in line.split() if p.strip()]
        if len(parts) < 2:
            continue
        try:
            u = int(float(parts[0]))
            v = int(float(parts[1]))
        except ValueError:
            continue
        G.add_edge(u, v)

    return G, first_thru_node


def compute_layout(G: nx.DiGraph, *, spring_threshold: int = 400, seed: int = 42) -> dict:
    n = G.number_of_nodes()
    if n > spring_threshold:
        return nx.spring_layout(G, seed=seed, k=None, iterations=200)
    try:
        return nx.kamada_kawai_layout(G, weight=None)
    except Exception:
        return nx.spring_layout(G, seed=seed, iterations=200)


def _canonical_pair(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def compute_bpr_ratios_by_edge(env: Any) -> dict[tuple[int, int], float]:
    """无向边 (min(u,v), max(u,v)) -> t_BPR / t0（当前步流 + 背景流，不含边际车）。"""
    out: dict[tuple[int, int], float] = {}
    for u, v in env.traffic_graph.edges():
        dyn = env._dynamic_profiles(u, v, add_vehicle=0.0)
        if not dyn:
            continue
        _lm, _sk, t_h, t0_h, _xf, _cap = min(dyn, key=lambda it: it[2])
        out[_canonical_pair(u, v)] = float(t_h) / max(float(t0_h), 1e-12)
    return out


def compute_x_flow_by_edge(env: Any) -> dict[tuple[int, int], float]:
    """与 ``_dynamic_profiles(..., add_vehicle=0)`` 一致的总需求：车上边计数 + 背景流标量。"""
    out: dict[tuple[int, int], float] = {}
    for u, v in env.traffic_graph.edges():
        xf = float(env._edge_flow(u, v)) + float(env._background_flow(u, v))
        out[_canonical_pair(u, v)] = xf
    return out


def run_congestion_simulation(
    graphml: Path,
    cache_dir: Path,
    *,
    num_evs: int,
    sim_steps: int,
    seed: int,
    snapshot_step: int | None = None,
    background_flow_mult: float = 1.0,
    verbose_background: bool = False,
    edge_metric: str = "x_flow",
    ue_net_tntp: Path | None = None,
    ue_trips_tntp: Path | None = None,
    use_ue_background: bool = True,
    ue_max_iter: int = 800,
) -> tuple[dict[tuple[int, int], float], Any]:
    """与 train_hindsight 一致的 EMA 加载；随机 T0 调度 + step 后按 ``edge_metric`` 取样边上图示量。

    ``snapshot_step``：reset 之后执行 ``env.step`` 的次数再取样；未指定时用 ``sim_steps``。
    ``env.time_step`` 每步 +1，对 144 取模影响背景流强度（``update_background_traffic`` 在 ``TrafficPowerEnv.step`` 开头调用）。

    ``background_flow_mult``：在仿真开始前把 ``background_edge_base_flows`` 整体放大（仅影响本脚本跑出的图，与训练默认 1.0 一致）。

    ``edge_metric``：``x_flow``（默认）或 ``bpr``（t/t0）。

    默认 ``use_ue_background=True``：用 TNTP + Frank–Wolfe UE 生成 ``background_edge_base_flows``（路径缺省
    ``map_outputs/ema/EMA_net.tntp`` 与 ``EMA_trips.tntp``，可由 ``ue_net_tntp`` / ``ue_trips_tntp`` 覆盖）。
    ``use_ue_background=False`` 则回退节点启发式基线。
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from env.real_env import RealTrafficEnv

    random.seed(seed)
    np.random.seed(seed)

    n_roll = int(snapshot_step) if snapshot_step is not None else int(sim_steps)
    if n_roll < 1:
        raise ValueError("snapshot_step / sim_steps must be >= 1")

    env_kw: dict[str, Any] = dict(
        graphml_file=str(graphml),
        num_stations=2,
        num_evs=num_evs,
        max_nodes=1_000_000,
        cache_dir=str(cache_dir),
        seed=seed,
        respawn_after_full_charge=True,
    )
    if use_ue_background:
        net_p = Path(ue_net_tntp) if ue_net_tntp is not None else ROOT / "map_outputs" / "ema" / "EMA_net.tntp"
        trip_p = Path(ue_trips_tntp) if ue_trips_tntp is not None else ROOT / "map_outputs" / "ema" / "EMA_trips.tntp"
        if net_p.is_file() and trip_p.is_file():
            env_kw["background_ue_net_tntp"] = str(net_p.resolve())
            env_kw["background_ue_trips_tntp"] = str(trip_p.resolve())
            env_kw["background_ue_max_iter"] = int(ue_max_iter)
            env_kw["background_ue_verbose"] = bool(verbose_background)
            print(
                f"[congestion] UE background baseline: net={net_p} trips={trip_p} max_iter={ue_max_iter}"
            )
        else:
            print(
                f"[congestion] warn: UE TNTP not found (net={net_p.is_file()}, trips={trip_p.is_file()}), "
                f"using heuristic background_edge_base_flows"
            )

    env = RealTrafficEnv(**env_kw)
    env.reset()
    m = float(background_flow_mult)
    if m <= 0:
        raise ValueError("background_flow_mult must be > 0")
    if m != 1.0:
        for k in list(env.background_edge_base_flows.keys()):
            env.background_edge_base_flows[k] = float(env.background_edge_base_flows[k]) * m
    # 与训练一致：每步 step() 内会 update；此处再刷一次，使 time_step==0 时 profile*base 已写入 background_edge_flows
    env.update_background_traffic()

    if verbose_background:
        bases = list(env.background_edge_base_flows.values())
        cur = list(env.background_edge_flows.values()) if env.background_edge_flows else []
        ev_tot = 0.0
        bg_tot = 0.0
        n_e = 0
        for u, v in env.traffic_graph.edges():
            ev_tot += float(env._edge_flow(u, v))
            bg_tot += float(env._background_flow(u, v))
            n_e += 1
        print(
            f"[congestion] background_flow_mult={m} | "
            f"mean(base)={float(np.mean(bases)):.3g} mean(profiled_bg)={float(np.mean(cur)) if cur else 0:.3g} | "
            f"snapshot mean EV x_flow/edge={ev_tot/max(n_e,1):.4g} mean bg x_flow/edge={bg_tot/max(n_e,1):.4g}"
        )

    rng = random.Random(seed)
    for _ in range(n_roll):
        actions: dict[int, int] = {}
        for ev in env.evs:
            if ev.status == "IDLE" and rng.random() < 0.55:
                actions[ev.id] = rng.randrange(len(env.stations))
        env.step(actions)
    if edge_metric == "bpr":
        return compute_bpr_ratios_by_edge(env), env
    if edge_metric == "x_flow":
        return compute_x_flow_by_edge(env), env
    raise ValueError(f"edge_metric must be 'x_flow' or 'bpr', got {edge_metric!r}")


def draw_network(
    G: nx.DiGraph,
    pos: dict,
    out_path: Path,
    *,
    edge_values: list[float | None] | None = None,
    value_vmax: float | None = None,
    edge_value_metric: str = "x_flow",
    equal_aspect: bool = False,
    congestion_auto_color: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=200)
    edgelist = list(G.edges())
    if edge_values is None:
        _ns = 55
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=_ns,
            node_color="#4C72B0",
            edgecolors="white",
            linewidths=0.5,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrows=True,
            arrowsize=4,
            edge_color="#9AA0A6",
            width=0.5,
            alpha=0.6,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.05",
            min_source_margin=3,
            min_target_margin=3,
            node_size=_ns,
        )
    else:
        _ns = 18
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=_ns,
            node_color="#B0B0B0",
            edgecolors="none",
            linewidths=0.0,
        )
        valid = [r for r in edge_values if r is not None]
        metric = (edge_value_metric or "x_flow").lower()
        if metric == "x_flow":
            if not valid:
                norm = plt.Normalize(vmin=0.0, vmax=1.0)
                cbar_label = "x_flow (EV on-edge + background)"
            elif value_vmax is not None:
                norm = plt.Normalize(vmin=0.0, vmax=float(value_vmax))
                cbar_label = "x_flow (EV on-edge + background)"
            else:
                vf = [max(0.0, float(x)) for x in valid]
                lo, hi = min(vf), max(vf)
                spread = hi - lo
                if congestion_auto_color and spread < 1e-9:
                    norm = plt.Normalize(vmin=0.0, vmax=max(hi, 1e-6))
                    cbar_label = "x_flow (EV on-edge + background)"
                elif congestion_auto_color and spread < 0.05 * max(hi, 1.0):
                    pad = max(spread * 0.2, 1e-6)
                    norm = plt.Normalize(vmin=0.0, vmax=hi + pad)
                    cbar_label = "x_flow (auto vmax, small spread)"
                else:
                    vmax_m = float(max(hi * 1.02, np.percentile(vf, 98), 1e-6))
                    norm = plt.Normalize(vmin=0.0, vmax=vmax_m)
                    cbar_label = "x_flow (EV on-edge + background)"
            _magma_base = plt.get_cmap("magma")
            cmap = mpl.colors.ListedColormap(_magma_base(np.linspace(0.15, 1.0, 256)))
        elif metric == "bpr":
            cmap = plt.get_cmap("coolwarm")
            if not valid:
                norm = plt.Normalize(vmin=1.0, vmax=1.2)
                cbar_label = "BPR travel time / free-flow t0"
            elif value_vmax is not None:
                norm = plt.Normalize(vmin=1.0, vmax=float(value_vmax))
                cbar_label = "BPR travel time / free-flow t0"
            else:
                vf = [float(x) for x in valid]
                lo, hi = min(vf), max(vf)
                spread = hi - lo
                if congestion_auto_color and spread < 0.05:
                    pad = max(spread * 0.15, 1e-7)
                    vmin_c = lo - pad
                    vmax_c = hi + pad
                    if abs(vmax_c - vmin_c) < 1e-12:
                        vmax_c = vmin_c + 1e-6
                    norm = plt.Normalize(vmin=vmin_c, vmax=vmax_c)
                    cbar_label = "BPR t/t0 (auto span)"
                else:
                    vmax = float(max(1.15, min(3.0, np.percentile(valid, 95))))
                    norm = plt.Normalize(vmin=1.0, vmax=vmax)
                    cbar_label = "BPR travel time / free-flow t0"
        else:
            raise ValueError(f"edge_value_metric must be 'x_flow' or 'bpr', got {edge_value_metric!r}")

        edge_colors: list[tuple[float, float, float, float]] = []
        edge_widths: list[float] = []
        for r in edge_values:
            if r is None:
                edge_colors.append((0.65, 0.65, 0.65, 0.15))
                edge_widths.append(0.3)
            else:
                t = float(norm(float(r)))
                t = max(0.0, min(1.0, t))
                rgba = cmap(t)
                a = 0.2 + 0.75 * t
                edge_colors.append(
                    (float(rgba[0]), float(rgba[1]), float(rgba[2]), a)
                )
                edge_widths.append(0.3 + 2.2 * t)

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=edgelist,
            arrows=False,
            edge_color=edge_colors,
            width=edge_widths,
            node_size=_ns,
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
        cbar.set_label(cbar_label)
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(labelsize=6)

    if equal_aspect:
        ax.set_aspect("equal")
    ax.axis("off")
    _save_kw: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.02}
    stem_path = out_path.parent / out_path.stem
    fig.savefig(stem_path.with_suffix(".pdf"), **_save_kw)
    fig.savefig(stem_path.with_suffix(".png"), dpi=600, **_save_kw)
    plt.close(fig)


def main() -> None:
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass

    p = argparse.ArgumentParser(description="Draw EMA network from EMA_net.tntp")
    p.add_argument(
        "--input",
        type=Path,
        default=ROOT / "map_outputs" / "ema" / "EMA_net.tntp",
        help="Path to EMA_net.tntp（拓扑解析；拥堵模式下亦作为 UE 背景用的 net TNTP，可用 --trips-tntp 配 OD）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "map_outputs" / "ema" / "EMA_network.png",
        help="Output basename: writes <stem>.png (600 dpi) and <stem>.pdf",
    )
    p.add_argument(
        "--congestion",
        action="store_true",
        help="Draw graphml/cache topology with edge snapshot coloring (needs osmnx + torch)",
    )
    p.add_argument(
        "--graphml",
        type=Path,
        default=ROOT / "map_outputs" / "ema" / "ema.graphml",
        help="Road graph for congestion mode (with --congestion); topology + coords from cache",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "map_outputs" / "ema_cache",
        help="EMA pickle cache dir (with --congestion)",
    )
    p.add_argument("--num-evs", type=int, default=45, help="EV count for congestion simulation")
    p.add_argument(
        "--sim-steps",
        type=int,
        default=60,
        help="reset 后执行 env.step 的次数再取拥堵快照（未指定 --snapshot-step 时使用）",
    )
    p.add_argument(
        "--snapshot-step",
        type=int,
        default=None,
        help="覆盖 --sim-steps：恰好在第 N 次 step 后取样；N mod 144 影响背景交通曲线",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for simulation")
    p.add_argument(
        "--congestion-edge-metric",
        type=str,
        default="x_flow",
        choices=["x_flow", "bpr"],
        help="拥堵上色：x_flow=车上边+背景（默认）；bpr=t/t0",
    )
    p.add_argument(
        "--ratio-vmax",
        type=float,
        default=None,
        dest="value_vmax",
        help="色条 vmax：x_flow 时为流量上限；bpr 时为 t/t0 上限（不指定则自动）",
    )
    p.add_argument(
        "--background-flow-mult",
        type=float,
        default=1.0,
        help="放大 background_edge_base_flows（仅本脚本可视化；训练默认 1.0）。EMA 容量大时背景相对小，可试 5–30 仅用于出图",
    )
    p.add_argument(
        "--trips-tntp",
        type=Path,
        default=ROOT / "map_outputs" / "ema" / "EMA_trips.tntp",
        help="EMA_trips.tntp（拥堵模式 UE 背景 OD；与 --input 成对）",
    )
    p.add_argument(
        "--no-ue-background",
        action="store_true",
        help="拥堵模式不用 UE，改用节点启发式 background_edge_base_flows",
    )
    p.add_argument(
        "--ue-max-iter",
        type=int,
        default=800,
        help="Frank–Wolfe UE 最大迭代次数（默认 800，易达 1e-4 量级 gap）",
    )
    p.add_argument(
        "--verbose-background",
        action="store_true",
        help="打印背景基线 / profile 后背景 / 边上 EV 流均值，确认背景参与 BPR",
    )
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.congestion:
        if not args.graphml.is_file():
            raise SystemExit(f"graphml not found: {args.graphml}")
        args.cache_dir.mkdir(parents=True, exist_ok=True)

        G_tntp, _ = parse_ema_net_tntp(args.input)
        val_by_uv, env = run_congestion_simulation(
            args.graphml,
            args.cache_dir,
            num_evs=args.num_evs,
            sim_steps=args.sim_steps,
            seed=args.seed,
            snapshot_step=args.snapshot_step,
            background_flow_mult=args.background_flow_mult,
            verbose_background=args.verbose_background,
            edge_metric=args.congestion_edge_metric,
            ue_net_tntp=args.input,
            ue_trips_tntp=args.trips_tntp,
            use_ue_background=not args.no_ue_background,
            ue_max_iter=args.ue_max_iter,
        )
        n_snap = args.snapshot_step if args.snapshot_step is not None else args.sim_steps
        if G_tntp.number_of_nodes() != env.num_nodes:
            print(
                f"[warn] TNTP nodes={G_tntp.number_of_nodes()} vs env nodes={env.num_nodes} "
                f"(拥堵图使用 graphml 拓扑，与 TNTP 节点编号可能不一致)"
            )

        G_draw = nx.DiGraph()
        for u, v in env.traffic_graph.edges():
            G_draw.add_edge(u, v)
            G_draw.add_edge(v, u)

        pos = {
            n: (float(env.node_positions[n][0]), float(env.node_positions[n][1]))
            for n in G_draw.nodes()
        }
        edge_values = [val_by_uv[_canonical_pair(u, v)] for u, v in G_draw.edges()]
        vv = [float(x) for x in edge_values if x is not None]
        if vv:
            if args.congestion_edge_metric == "x_flow":
                print(
                    f"[congestion] x_flow min={min(vv):.6g} max={max(vv):.6g} "
                    f"(_edge_flow + _background_flow per undirected edge)"
                )
            else:
                print(
                    f"[congestion] t/t0 min={min(vv):.6g} max={max(vv):.6g} "
                    f"(BPR uses background_edge_flows + EV edge counts)"
                )
        print(
            f"[congestion] nodes={env.num_nodes}, directed_edges={G_draw.number_of_edges()}, "
            f"undirected_edges={env.traffic_graph.number_of_edges()}, "
            f"env.time_step={env.time_step} (snapshot after {n_snap} step calls)"
        )
        draw_network(
            G_draw,
            pos,
            args.output,
            edge_values=edge_values,
            value_vmax=args.value_vmax,
            edge_value_metric=args.congestion_edge_metric,
            equal_aspect=True,
            congestion_auto_color=True,
        )
    else:
        G, first_thru = parse_ema_net_tntp(args.input)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"nodes={n_nodes}, edges={n_edges}, first_thru_node={first_thru}")
        pos = compute_layout(G)
        draw_network(G, pos, args.output)

    _stem = args.output.parent / args.output.stem
    print(f"saved: {_stem.with_suffix('.pdf')}, {_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
