import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .osm_loader import _select_station_nodes

MILES_TO_METERS = 1609.34
MPH_TO_KPH = 1.609344


@dataclass(frozen=True)
class TntpArc:
    """一条 TNTP 有向 link（节点编号为文件中的原始 id）。"""

    u_tntp: int
    v_tntp: int
    capacity: float
    length_miles: float
    free_flow_time_h: float


def parse_tntp_net(net_file: str) -> tuple[int, int, list[TntpArc]]:
    """
    解析标准 TNTP 元数据 + link 表（跳过 <...> 与表头 ~ 行）。
    返回 (first_thru_node, number_of_nodes, arcs)。
    """
    path = Path(net_file)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    in_meta = True
    meta_re = re.compile(r"^<([^>]+)>\s*(.*)$")
    first_thru = 1
    num_nodes = 0
    arcs: list[TntpArc] = []

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
                        first_thru = int(float(val.split()[0]))
                    except (ValueError, IndexError):
                        pass
                if key == "NUMBER OF NODES":
                    try:
                        num_nodes = int(float(val.split()[0]))
                    except (ValueError, IndexError):
                        pass
                if key == "END OF METADATA":
                    in_meta = False
            continue

        if line.startswith("~"):
            continue

        parts = [p.strip().rstrip(";") for p in line.split() if p.strip()]
        if len(parts) < 5:
            continue
        try:
            u = int(float(parts[0]))
            v = int(float(parts[1]))
            capacity = float(parts[2])
            length_miles = float(parts[3])
            free_flow_time_h = float(parts[4])
        except ValueError:
            continue
        arcs.append(
            TntpArc(
                u_tntp=u,
                v_tntp=v,
                capacity=capacity,
                length_miles=length_miles,
                free_flow_time_h=free_flow_time_h,
            )
        )

    if num_nodes <= 0:
        seen = set()
        for a in arcs:
            seen.add(a.u_tntp)
            seen.add(a.v_tntp)
        if seen:
            num_nodes = max(seen) - first_thru + 1
    return first_thru, num_nodes, arcs


def _merge_undirected_attrs(bundle: list[TntpArc]) -> dict:
    """双向或多记录合并为一条无向边的属性（与 `TrafficPowerEnv._edge_profiles_from_data` 字段对齐）。"""
    lens_m: list[float] = []
    caps: list[float] = []
    t0s: list[float] = []
    for ar in bundle:
        lm = max(1.0, ar.length_miles * MILES_TO_METERS)
        lens_m.append(lm)
        caps.append(ar.capacity)
        t0s.append(max(1e-9, ar.free_flow_time_h))
    length_m = sum(lens_m) / len(lens_m)
    capacity = sum(caps) / len(caps)
    t0_h = sum(t0s) / len(t0s)
    speed_mphs = [ar.length_miles / max(ar.free_flow_time_h, 1e-9) for ar in bundle]
    speed_kph = max(1.0, (sum(speed_mphs) / len(speed_mphs)) * MPH_TO_KPH)
    return {
        "length": float(length_m),
        "speed_kph": float(speed_kph),
        "capacity": float(capacity),
        "weight": float(t0_h),
    }


def tntp_net_to_graphml_graph(
    net_file: str,
    *,
    layout_seed: int = 42,
    lon_bounds: Tuple[float, float] = (-71.25, -70.88),
    lat_bounds: Tuple[float, float] = (42.12, 42.52),
) -> nx.Graph:
    """
    由 TNTP 构造与训练侧兼容的无向 `nx.Graph`：
    - 节点为连续整数 ``0 .. N-1``，与 TNTP 交通节点 id 满足 ``tntp_id = node + first_thru_node``；
    - 双向 link 合并为一条无向边，``length``(m)、``speed_kph``、``capacity``、``weight``(h) 为双向均值；
    - ``x,y`` 为基于拓扑的 spring 布局再线性映射到给定经纬度范围（仅示意，非测绘坐标）。
    """
    first_thru, num_nodes, arcs = parse_tntp_net(net_file)
    if num_nodes <= 0:
        raise ValueError("无法从 TNTP 推断节点数，请检查元数据 <NUMBER OF NODES>")

    bundles: dict[tuple[int, int], list[TntpArc]] = defaultdict(list)
    for ar in arcs:
        eu = ar.u_tntp - first_thru
        ev = ar.v_tntp - first_thru
        if not (0 <= eu < num_nodes and 0 <= ev < num_nodes):
            continue
        a, b = (eu, ev) if eu <= ev else (ev, eu)
        bundles[(a, b)].append(ar)

    G = nx.Graph()
    G.graph["crs"] = "epsg:4326"
    for i in range(num_nodes):
        G.add_node(i)

    for (u, v), bundle in bundles.items():
        G.add_edge(u, v, **_merge_undirected_attrs(bundle))

    if not nx.is_connected(G):
        raise ValueError("TNTP 无向化后不连通，无法写出单一 graphml；请检查输入文件")

    pos = nx.spring_layout(G, seed=layout_seed, iterations=200)
    lon0, lon1 = lon_bounds
    lat0, lat1 = lat_bounds
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    xmin, xmax = min(xs), max(xs) or 1.0
    ymin, ymax = min(ys), max(ys) or 1.0

    def _scale(t: float, lo: float, hi: float, tmin: float, tmax: float) -> float:
        if abs(tmax - tmin) < 1e-12:
            return (lo + hi) / 2.0
        return lo + (t - tmin) / (tmax - tmin) * (hi - lo)

    for n in G.nodes():
        x, y = pos[n]
        lon = _scale(x, lon0, lon1, xmin, xmax)
        lat = _scale(y, lat0, lat1, ymin, ymax)
        G.nodes[n]["x"] = float(lon)
        G.nodes[n]["y"] = float(lat)

    return G


def load_tntp_network(
    net_file: str,
    num_stations: int = 3,
    seed: int = 42,
    station_node_ids: Optional[List[int]] = None,
) -> Tuple[nx.Graph, list, dict]:
    G = nx.DiGraph()

    with open(net_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_started = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("~") and not data_started:
            data_started = True
            continue
        if line.startswith("<") or line.startswith("~"):
            continue
        if not data_started:
            continue

        parts = line.split(";")[0].split()
        if len(parts) < 8:
            continue

        u = int(parts[0])
        v = int(parts[1])
        capacity = float(parts[2])
        length_miles = float(parts[3])
        free_flow_time_h = float(parts[4])

        length_m = max(1.0, length_miles * MILES_TO_METERS)
        speed_kph = max(1.0, (length_miles / max(1e-6, free_flow_time_h)) * MILES_TO_METERS)

        G.add_node(u)
        G.add_node(v)

        G.add_edge(u, v,
                   length=length_m,
                   speed_kph=speed_kph,
                   capacity=capacity,
                   weight=free_flow_time_h)

    G_undirected = G.to_undirected()

    if not nx.is_connected(G_undirected):
        components = sorted(nx.connected_components(G_undirected), key=len, reverse=True)
        largest = G_undirected.subgraph(components[0]).copy()
    else:
        largest = G_undirected

    largest = nx.convert_node_labels_to_integers(largest)

    np.random.seed(seed)
    random.seed(seed)

    if station_node_ids is not None:
        missing = [n for n in station_node_ids if n not in largest.nodes()]
        if missing:
            raise ValueError(f"station_node_ids 不在图中: {missing}")
        stations = station_node_ids[:num_stations]
    else:
        stations = _select_station_nodes(largest, num_stations, seed)

    pos_layout = nx.spring_layout(largest, seed=seed, k=1.5)
    positions = {}
    for node, (x, y) in pos_layout.items():
        lon = 114.2 + (x + 1) / 2 * 0.3
        lat = 30.4 + (y + 1) / 2 * 0.3
        largest.nodes[node]["x"] = str(lon)
        largest.nodes[node]["y"] = str(lat)
        positions[node] = (lon, lat)

    print(f"[TNTP] nodes={largest.number_of_nodes()}, edges={largest.number_of_edges()}, "
          f"stations={stations}")

    return largest, stations, positions
