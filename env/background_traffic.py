"""背景交通流：144-step 日周期的聚合流量模型。

``build_daily_profile`` 的日周期曲线保留不变。

``edge_base_background_flow`` / 原启发式 ``build_base_background_flows``（无 TNTP 参数时）为按节点编号到「中心」的档位，
仅适合小网格 ``TrafficPowerEnv``。

**EMA 等 TNTP 路网**：调用 ``build_base_background_flows(..., net_tntp_path=..., trips_tntp_path=...)``
（或在 ``RealTrafficEnv`` 上设置 ``background_ue_net_tntp`` / ``background_ue_trips_tntp``）时，
会在内部调用 ``background/ue_assignment.py`` 的 UE 求解，用均衡边流替代启发式。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Tuple


Edge = Tuple[int, int]

# 边上背景流基线总倍数（相对原 15/35/60/90 档位）；再乘日 profile 与 `update_background_traffic` 里 ±5% 正弦扰动。
BACKGROUND_EDGE_BASE_SCALE = 4.0


def build_daily_profile(steps_per_day: int = 144, profile_name: str = "base") -> list[float]:
    """生成 144-step 日周期系数：早晚高峰用高斯峰叠加。"""
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be positive")

    profile_name = (profile_name or "base").strip().lower()
    params_by_profile = {
        "base": (0.35, 0.85, 0.95, 0.08, 0.09, 0.28, 0.72),
        "old_city": (0.45, 1.05, 1.10, 0.055, 0.065, 0.27, 0.71),
        "new_city": (0.32, 0.65, 1.20, 0.08, 0.075, 0.30, 0.75),
        "suburb": (0.25, 0.65, 0.70, 0.12, 0.13, 0.31, 0.70),
    }
    if profile_name not in params_by_profile:
        raise ValueError(f"Unknown daily traffic profile: {profile_name!r}")

    base, morning_amp, evening_amp, morning_sigma, evening_sigma, morning_at, evening_at = (
        params_by_profile[profile_name]
    )
    profile: list[float] = []
    sigma_morning = steps_per_day * morning_sigma
    sigma_evening = steps_per_day * evening_sigma
    morning_peak = steps_per_day * morning_at
    evening_peak = steps_per_day * evening_at

    for t in range(steps_per_day):
        morning = math.exp(-0.5 * ((t - morning_peak) / sigma_morning) ** 2)
        evening = math.exp(-0.5 * ((t - evening_peak) / sigma_evening) ** 2)
        profile.append(base + morning_amp * morning + evening_amp * evening)

    peak = max(profile)
    if peak <= 0:
        return [1.0 for _ in range(steps_per_day)]
    return [float(v / peak) for v in profile]


def edge_base_background_flow(u: int, v: int, graph_nodes: Iterable[int]) -> float:
    """给网格边一个稳定的基础背景流强度。"""
    nodes = set(graph_nodes)
    center = 4 if 4 in nodes else (min(nodes) + max(nodes)) / 2
    dist = (abs(u - center) + abs(v - center)) / 2.0
    if 4 in nodes and (u == 4 or v == 4):
        tier = 90.0
    elif dist <= 1.0:
        tier = 60.0
    elif dist <= 2.0:
        tier = 35.0
    else:
        tier = 15.0
    return float(tier * BACKGROUND_EDGE_BASE_SCALE)


def build_base_background_flows(
    edges: Iterable[Edge],
    graph_nodes: Iterable[int],
    *,
    net_tntp_path: str | Path | None = None,
    trips_tntp_path: str | Path | None = None,
    ue_scale: float = 1.0,
    ue_max_iter: int = 100,
    ue_tol: float = 1e-4,
    ue_verbose: bool = False,
) -> Dict[Edge, float]:
    """为每条边生成固定背景流基线。

    - 若 **同时** 传入 ``net_tntp_path`` 与 ``trips_tntp_path``：内部调用 UE（``ue_assignment.compute_ue_background_flows``），
      再按 ``edges`` 的 ``(u,v)`` 对齐（先试 ``(u,v)``，否则 ``(v,u)``，缺省为 0）。
    - 否则：沿用 ``edge_base_background_flow`` 启发式（小网格）。
    """
    edges_list = [tuple(edge) for edge in edges]
    if net_tntp_path is not None or trips_tntp_path is not None:
        if net_tntp_path is None or trips_tntp_path is None:
            raise ValueError(
                "net_tntp_path and trips_tntp_path must both be set (or both omitted) for UE background"
            )
        ue = build_base_background_flows_ue(
            net_tntp_path,
            trips_tntp_path,
            scale=ue_scale,
            max_iter=ue_max_iter,
            tol=ue_tol,
            verbose=ue_verbose,
        )
        out: Dict[Edge, float] = {}
        for u, v in edges_list:
            val = ue.get((u, v))
            if val is None:
                val = ue.get((v, u), 0.0)
            out[(u, v)] = float(val)
        return out
    sc = float(ue_scale)
    return {
        (u, v): edge_base_background_flow(u, v, graph_nodes) * sc
        for u, v in edges_list
    }


def build_base_background_flows_ue(
    net_tntp_path: str | Path,
    trips_tntp_path: str | Path,
    *,
    scale: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
) -> Dict[Edge, float]:
    """UE 均衡有向边流量 × ``scale``，作为 ``background_edge_base_flows`` 的空间基线（TNTP 节点编号）。"""
    from .background.ue_assignment import compute_ue_background_flows

    flows = compute_ue_background_flows(
        net_tntp_path, trips_tntp_path, max_iter=max_iter, tol=tol, verbose=verbose
    )
    sc = float(scale)
    return {e: float(v) * sc for e, v in flows.items()}
