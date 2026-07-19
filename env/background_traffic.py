"""Background traffic helpers shared by grid and real traffic environments."""

from __future__ import annotations

import math
import os
from typing import Iterable, Mapping, Optional

BACKGROUND_EDGE_BASE_SCALE = 1.0


def build_daily_profile(steps_per_day: int = 144) -> list[float]:
    """Return a bounded daily traffic profile with morning and evening peaks."""
    steps = max(1, int(steps_per_day))
    values: list[float] = []
    for step in range(steps):
        hour = 24.0 * step / steps
        morning = math.exp(-0.5 * ((hour - 8.0) / 1.7) ** 2)
        evening = math.exp(-0.5 * ((hour - 17.5) / 2.0) ** 2)
        midday = 0.25 * math.exp(-0.5 * ((hour - 13.0) / 4.0) ** 2)
        night_floor = 0.12
        values.append(night_floor + 0.65 * morning + 0.75 * evening + midday)

    peak = max(values) if values else 1.0
    return [max(1e-6, min(1.0, value / peak)) for value in values]


def _node_score(node, nodes_by_id: Mapping) -> float:
    attrs = nodes_by_id.get(node, {})
    if isinstance(attrs, Mapping):
        try:
            x = float(attrs.get("x"))
            y = float(attrs.get("y"))
            return abs(x) + abs(y)
        except (TypeError, ValueError):
            pass
    try:
        return float(node)
    except (TypeError, ValueError):
        return float(abs(hash(node)) % 1000)


def _heuristic_base_background_flows(edges: Iterable, nodes: Iterable) -> dict[tuple, float]:
    node_items = list(nodes.items()) if hasattr(nodes, "items") else [(n, {}) for n in nodes]
    nodes_by_id = {node: attrs for node, attrs in node_items}
    scores = [_node_score(node, nodes_by_id) for node, _attrs in node_items]
    lo = min(scores) if scores else 0.0
    hi = max(scores) if scores else 1.0
    span = max(1e-9, hi - lo)

    flows: dict[tuple, float] = {}
    for edge in edges:
        if len(edge) < 2:
            continue
        u, v = edge[0], edge[1]
        su = (_node_score(u, nodes_by_id) - lo) / span
        sv = (_node_score(v, nodes_by_id) - lo) / span
        centrality = 1.0 - abs(((su + sv) * 0.5) - 0.5)
        directional_mix = 0.5 + 0.5 * abs(su - sv)
        flows[(u, v)] = BACKGROUND_EDGE_BASE_SCALE * (0.5 + 2.5 * centrality * directional_mix)
    return flows


def build_base_background_flows(
    edges: Iterable,
    nodes: Iterable,
    *,
    net_tntp_path: Optional[str] = None,
    trips_tntp_path: Optional[str] = None,
    ue_scale: float = 1.0,
    ue_max_iter: int = 100,
    ue_tol: float = 1e-4,
    ue_verbose: bool = False,
) -> dict[tuple, float]:
    """Build per-edge background flow baselines.

    If the optional UE assignment implementation and TNTP files are available,
    use them. Otherwise fall back to a deterministic edge-position heuristic.
    """
    edge_list = list(edges)
    heuristic = _heuristic_base_background_flows(edge_list, nodes)

    if not (net_tntp_path and trips_tntp_path):
        return heuristic
    if not (os.path.isfile(net_tntp_path) and os.path.isfile(trips_tntp_path)):
        return heuristic

    try:
        from env.background.ue_assignment import compute_ue_background_flows
    except Exception:
        return heuristic

    try:
        ue_flows = compute_ue_background_flows(
            net_tntp_path,
            trips_tntp_path,
            max_iter=int(ue_max_iter),
            tol=float(ue_tol),
            verbose=bool(ue_verbose),
        )
    except Exception:
        return heuristic

    if not ue_flows:
        return heuristic

    scaled = {}
    for edge in edge_list:
        if len(edge) < 2:
            continue
        u, v = edge[0], edge[1]
        value = ue_flows.get((u, v), ue_flows.get((v, u), heuristic.get((u, v), 0.0)))
        scaled[(u, v)] = float(value) * float(ue_scale)
    return scaled
