#!/usr/bin/env python3
"""
从 TNTP 路网（如 EMA_net.tntp）生成与 `RealTrafficEnv` / `osm_loader` 字段兼容的 GraphML。

节点编号为 0..N-1，与 TNTP 交通节点 id 对应关系：``tntp_id = node + <FIRST THRU NODE>``。

生成后请删除旧的 ``map_outputs/ema_cache/local_ema_*.pkl``，否则仍会命中旧缓存。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

import networkx as nx

from env.tntp_loader import parse_tntp_net, tntp_net_to_graphml_graph


def _stringify_graphml_attrs(G: nx.Graph) -> None:
    """OSMnx 读 graphml 时期望字符串属性；写盘前全部转 str（加载后由 osm_loader 再转回 float）。"""
    for _, d in G.nodes(data=True):
        for k, v in list(d.items()):
            d[k] = str(v)
    for _, _, d in G.edges(data=True):
        for k, v in list(d.items()):
            d[k] = str(v)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert TNTP net file to GraphML (node ids align with TNTP + first_thru).",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=_root / "map_outputs" / "ema" / "EMA_net.tntp",
        help="TNTP net file path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_root / "map_outputs" / "ema" / "ema.graphml",
        help="Output GraphML path",
    )
    p.add_argument("--layout-seed", type=int, default=42, help="spring_layout seed for synthetic x,y")
    p.add_argument(
        "--backup",
        action="store_true",
        help="If output exists, copy to output.with_suffix(.bak.graphml) before overwrite",
    )
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")

    first_thru, num_nodes, arcs = parse_tntp_net(str(args.input))
    G = tntp_net_to_graphml_graph(
        str(args.input),
        layout_seed=args.layout_seed,
    )
    _stringify_graphml_attrs(G)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backup and args.output.is_file():
        bak = args.output.with_suffix(".bak.graphml")
        shutil.copy2(args.output, bak)
        print(f"[backup] {args.output} -> {bak}")

    nx.write_graphml(G, str(args.output), infer_numeric_types=True)

    cache_glob = list((_root / "map_outputs" / "ema_cache").glob("local_ema*.pkl"))
    print(
        f"[ok] wrote {args.output} | nodes={G.number_of_nodes()} edges={G.number_of_edges()} | "
        f"first_thru={first_thru} n_tntp_nodes={num_nodes} n_arcs={len(arcs)}"
    )
    print(
        "[hint] 节点 i 对应 TNTP 交通节点 id = i + first_thru；"
        "若使用新图，请核对 config/stations.json 中的站点索引，并删除 ema_cache 下旧 pkl 后重载："
    )
    if cache_glob:
        for c in sorted(cache_glob):
            print(f"         del or archive: {c}")
    else:
        print("         (未找到 local_ema*.pkl)")


if __name__ == "__main__":
    main()
