"""EMA TNTP 网络上的用户均衡(UE)交通分配 — Frank–Wolfe 算法。

解析 ``*_net.tntp`` / ``*_trips.tntp``，在 BPR 路阻下求边均衡流量，供背景流空间基线使用。
"""
from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class UeEdgeAttrs:
    u: int
    v: int
    capacity: float
    length: float
    fft: float
    alpha: float
    beta: float


def parse_ue_net_tntp(path: Path) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """解析 TNTP net：有向边及 capacity / fft / BPR alpha / beta。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    meta_re = re.compile(r"^<([^>]+)>\s*(.*)$")
    in_meta = True
    edgelist: list[tuple[int, int]] = []
    caps: list[float] = []
    lengths: list[float] = []
    ffts: list[float] = []
    alphas: list[float] = []
    betas: list[float] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_meta:
            m = meta_re.match(line)
            if m and m.group(1).strip().upper() == "END OF METADATA":
                in_meta = False
            continue
        if line.startswith("~"):
            continue
        parts = [p.strip().rstrip(";") for p in line.split() if p.strip()]
        if len(parts) < 7:
            continue
        try:
            u = int(float(parts[0]))
            v = int(float(parts[1]))
            cap = float(parts[2])
            length = float(parts[3])
            fft = float(parts[4])
            alpha = float(parts[5])
            beta = float(parts[6])
        except ValueError:
            continue
        edgelist.append((u, v))
        caps.append(max(cap, 1e-6))
        lengths.append(length)
        ffts.append(max(fft, 1e-12))
        alphas.append(max(alpha, 1e-12))
        betas.append(max(beta, 1.0))

    if not edgelist:
        raise ValueError(f"no directed links parsed from {path}")
    return (
        edgelist,
        np.asarray(caps, dtype=np.float64),
        np.asarray(lengths, dtype=np.float64),
        np.asarray(ffts, dtype=np.float64),
        np.asarray(alphas, dtype=np.float64),
        np.asarray(betas, dtype=np.float64),
    )


def parse_ue_trips_tntp(path: Path) -> dict[tuple[int, int], float]:
    """解析标准 TNTP trips：``Origin <i>`` 块内 ``j : value;`` 为 OD 需求。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    origin_re = re.compile(r"^Origin\s+(\d+)\s*$", re.IGNORECASE)
    pair_re = re.compile(r"(\d+)\s*:\s*([\d.eE+-]+)\s*;")
    demand: dict[tuple[int, int], float] = {}
    current_o: int | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        mo = origin_re.match(line)
        if mo:
            current_o = int(mo.group(1))
            continue
        if current_o is None:
            continue
        for m in pair_re.finditer(line):
            d = int(m.group(1))
            val = float(m.group(2))
            if d == current_o or val <= 0.0:
                continue
            demand[(current_o, d)] = demand.get((current_o, d), 0.0) + val
    return demand


def _bpr_time(x: np.ndarray, fft: np.ndarray, cap: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    r = np.clip(x / cap, 0.0, 50.0)
    return fft * (1.0 + alpha * np.power(r, beta))


def _dphi(
    lam: float,
    x: np.ndarray,
    d: np.ndarray,
    fft: np.ndarray,
    cap: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> float:
    """Beckmann 线目标对 λ 的导数: Σ d_e · t_e(x_e + λ d_e)。"""
    w = x + lam * d
    t = _bpr_time(w, fft, cap, alpha, beta)
    return float(np.dot(d, t))


def _line_search_lambda(
    x: np.ndarray,
    y: np.ndarray,
    fft: np.ndarray,
    cap: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    *,
    bisect_iters: int = 80,
) -> float:
    d = y - x
    g0 = _dphi(0.0, x, d, fft, cap, alpha, beta)
    if g0 >= -1e-14:
        return 0.0
    g1 = _dphi(1.0, x, d, fft, cap, alpha, beta)
    if g1 <= 1e-14:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        gm = _dphi(mid, x, d, fft, cap, alpha, beta)
        if gm > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _aggregate_demand_by_origin(demand: dict[tuple[int, int], float]) -> dict[int, list[tuple[int, float]]]:
    by_o: dict[int, list[tuple[int, float]]] = {}
    for (o, d), v in demand.items():
        if v <= 0.0:
            continue
        by_o.setdefault(o, []).append((d, float(v)))
    return by_o


def _dijkstra_dist_pred(
    n_nodes: int,
    adj: list[list[tuple[int, float, int]]],
    source: int,
) -> tuple[list[float], list[int]]:
    """返回 dist[1..n_nodes], pred[1..n_nodes]（1-based 下标与 TNTP 节点 id 对齐）。"""
    inf = math.inf
    dist = [inf] * (n_nodes + 1)
    pred = [-1] * (n_nodes + 1)
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist[u] + 1e-15:
            continue
        for v, w, _ei in adj[u]:
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                pred[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, pred


def _aon_directional_flows(
    n_edges: int,
    n_nodes: int,
    edgelist: list[tuple[int, int]],
    edge_times: np.ndarray,
    by_origin: dict[int, list[tuple[int, float]]],
) -> np.ndarray:
    """给定各边阻抗，全有全无：按 OD 将流量加到最短路有向边上。"""
    adj: list[list[tuple[int, float, int]]] = [[] for _ in range(n_nodes + 1)]
    for ei, (u, v) in enumerate(edgelist):
        adj[u].append((v, float(edge_times[ei]), ei))

    y = np.zeros(n_edges, dtype=np.float64)
    for r, dest_list in by_origin.items():
        if r < 1 or r > n_nodes:
            continue
        _, pred = _dijkstra_dist_pred(n_nodes, adj, r)
        for s, dem in dest_list:
            if s < 1 or s > n_nodes or dem <= 0.0:
                continue
            if pred[s] < 0 and s != r:
                continue
            cur = s
            while cur != r and pred[cur] >= 0:
                p = pred[cur]
                ei = None
                for vv, _w, eidx in adj[p]:
                    if vv == cur:
                        ei = eidx
                        break
                if ei is None:
                    break
                y[ei] += dem
                cur = p
    return y


def _max_node_id(edgelist: list[tuple[int, int]]) -> int:
    m = 0
    for u, v in edgelist:
        m = max(m, u, v)
    return m


def compute_ue_background_flows(
    net_tntp_path: str | Path,
    trips_tntp_path: str | Path,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
) -> dict[tuple[int, int], float]:
    """Frank–Wolfe UE：返回有向边均衡流量 ``{(u,v): x}``（与 TNTP 节点编号一致）。

    纯 Frank–Wolfe 在大 OD 网络上接近 ``tol`` 可能需要数百次迭代；EMA 上可把 ``max_iter`` 提到 600–900。

    ``verbose``：为 True 时打印迭代 ``relative_gap`` 与收敛摘要；训练脚本默认关闭。
    """
    net_path = Path(net_tntp_path)
    trip_path = Path(trips_tntp_path)
    edgelist, cap, _length, fft, alpha, beta = parse_ue_net_tntp(net_path)
    demand = parse_ue_trips_tntp(trip_path)
    if not demand:
        raise ValueError(f"no positive OD demand parsed from {trip_path}")

    n_edges = len(edgelist)
    n_nodes = max(
        _max_node_id(edgelist),
        max(max(o, d) for (o, d) in demand),
    )
    by_origin = _aggregate_demand_by_origin(demand)

    x = np.zeros(n_edges, dtype=np.float64)
    t_free = _bpr_time(x, fft, cap, alpha, beta)
    x = _aon_directional_flows(n_edges, n_nodes, edgelist, t_free, by_origin)

    last_it = max_iter
    final_rel_gap = float("nan")
    for it in range(1, max_iter + 1):
        t = _bpr_time(x, fft, cap, alpha, beta)
        y = _aon_directional_flows(n_edges, n_nodes, edgelist, t, by_origin)
        d = y - x
        denom = max(float(np.dot(t, x)), 1e-12)
        rel_gap = abs(float(np.dot(t, d))) / denom
        final_rel_gap = rel_gap
        if verbose and (it % 10 == 0 or it == 1 or it == max_iter):
            print(f"[UE-FW] iter={it} relative_gap={rel_gap:.6e}")
        if rel_gap < tol:
            if verbose:
                print(f"[UE-FW] converged at iter={it} relative_gap={rel_gap:.6e}")
            last_it = it
            break
        lam = _line_search_lambda(x, y, fft, cap, alpha, beta)
        x = x + lam * d
    else:
        last_it = max_iter
        t = _bpr_time(x, fft, cap, alpha, beta)
        y = _aon_directional_flows(n_edges, n_nodes, edgelist, t, by_origin)
        d = y - x
        denom = max(float(np.dot(t, x)), 1e-12)
        rel_gap = abs(float(np.dot(t, d))) / denom
        final_rel_gap = rel_gap
        if verbose:
            print(f"[UE-FW] max_iter={max_iter} reached, relative_gap={rel_gap:.6e}")

    if verbose:
        print(f"[UE-FW] total_iterations={last_it} final_relative_gap={final_rel_gap:.6e}")
    return {edgelist[i]: float(x[i]) for i in range(n_edges)}


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[2]
    _net = _ROOT / "map_outputs" / "ema" / "EMA_net.tntp"
    _trips = _ROOT / "map_outputs" / "ema" / "EMA_trips.tntp"
    # 默认 API max_iter=100 在 EMA 上往往达不到 1e-4；自检用更大上限验证收敛。
    flows = compute_ue_background_flows(_net, _trips, max_iter=800, tol=1e-4, verbose=True)
    xs = np.asarray(list(flows.values()), dtype=np.float64)
    xs = xs[xs > 1e-9]
    print(f"[UE-FW] nonzero edges={len(xs)} / {len(flows)}")
    if len(xs):
        print(f"[UE-FW] flow max={float(np.max(xs)):.6g} median={float(np.median(xs)):.6g}")
