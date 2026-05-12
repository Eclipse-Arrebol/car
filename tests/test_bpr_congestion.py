"""BPR 拥堵公式：单位一致性与对 active flow 的响应（pytest）。"""
from typing import Tuple

import pytest

from env.base_env import TrafficPowerEnv


@pytest.fixture
def bpr_env():
    return TrafficPowerEnv(num_evs=1, respawn_after_full_charge=False)


def _t_h_from_dynamic(env: TrafficPowerEnv, u: int, v: int, n_active: int) -> Tuple[float, float]:
    env.edge_active_counts = {}
    if n_active > 0:
        env.edge_active_counts[(u, v)] = n_active
    dyn = env._dynamic_profiles(u, v, add_vehicle=0.0)
    assert dyn, "_dynamic_profiles 应返回至少一条 profile"
    row = dyn[0]
    t_h = row[2]
    t0_h = row[3]
    return t_h, t0_h


def test_bpr_responds_to_active_flow(bpr_env):
    env = bpr_env
    u, v = 0, 1
    assert (u, v) in env.traffic_graph.edges()

    t_low, t0_h = _t_h_from_dynamic(env, u, v, 0)
    assert t0_h > 0
    assert abs(t_low - t0_h) / t0_h <= 0.001, "零流时 t_h 应与 t0_h 一致（≤0.1% 误差）"

    t_mid, _ = _t_h_from_dynamic(env, u, v, 20)
    assert t_mid > t_low * 1.02, "中等负载下 BPR 时间应至少比自由流高 2%"

    t_high, _ = _t_h_from_dynamic(env, u, v, 80)
    assert t_high > t_mid * 1.5, "高负载下 BPR 时间应显著高于中等负载"


def test_bpr_monotonic_in_flow(bpr_env):
    env = bpr_env
    t0_h = 0.1
    c = 1000.0
    xs = [0, 11, 22, 33, 44, 55, 66, 77, 88, 100]
    times = [env._bpr_time_h(t0_h, float(x), c) for x in xs]
    for a, b in zip(times, times[1:]):
        assert b >= a - 1e-12, "BPR 输出应对流量单调非减"


def test_bpr_unit_consistency(bpr_env):
    env = bpr_env
    expected = 0.1009375
    got = env._bpr_time_h(0.1, 50.0, 1000.0)
    assert abs(got - expected) <= 1e-6, f"手算 t={expected}, 得到 {got}"
