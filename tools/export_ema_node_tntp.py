"""从 `ema.graphml` 导出 `EMA_node.tntp`（与 `EMA_net.tntp` 节点编号 1..N 对齐）。"""
from __future__ import annotations

import argparse
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from env.ema_node_tntp import write_ema_node_tntp_from_graphml


def main() -> None:
    p = argparse.ArgumentParser(description="Export EMA_node.tntp from ema.graphml")
    p.add_argument(
        "--graphml",
        type=str,
        default=os.path.join("map_outputs", "ema", "ema.graphml"),
    )
    p.add_argument(
        "--output",
        type=str,
        default=os.path.join("map_outputs", "ema", "EMA_node.tntp"),
    )
    args = p.parse_args()
    write_ema_node_tntp_from_graphml(args.graphml, args.output)
    print(f"[export_ema_node_tntp] wrote {args.output}")


if __name__ == "__main__":
    main()
