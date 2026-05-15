"""背景交通流的单元测试（RealTrafficEnv 离线路网，与训练主路径一致）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from env.background_traffic import build_base_background_flows, build_daily_profile  # noqa: E402
from env.real_env import RealTrafficEnv  # noqa: E402


@pytest.fixture(scope="module")
def bg_offline_cache_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp("bg_traffic_offline_cache"))


def _make_offline_real_env(cache_dir: str, seed: int = 0, max_nodes: int = 16) -> RealTrafficEnv:
    return RealTrafficEnv(
        offline=True,
        num_evs=0,
        num_stations=2,
        max_nodes=max_nodes,
        seed=seed,
        cache_dir=cache_dir,
        respawn_after_full_charge=False,
    )


def test_daily_profile_is_periodic_and_bounded() -> None:
    profile = build_daily_profile(144)
    assert len(profile) == 144
    assert all(v > 0 for v in profile)
    assert max(profile) <= 1.0
    assert abs(profile[0] - profile[143]) < 0.2
    assert abs(profile[0] - profile[144 % 144]) < 1e-9


def test_daily_profile_has_morning_and_evening_peaks() -> None:
    profile = build_daily_profile(144)
    morning_peak = max(profile[30:50])
    evening_peak = max(profile[95:115])
    night_low = min(profile[0:20])
    assert morning_peak > night_low
    assert evening_peak > night_low


def test_base_background_flows_depend_on_edge_position(bg_offline_cache_dir: str) -> None:
    env = _make_offline_real_env(bg_offline_cache_dir)
    flows = build_base_background_flows(env.traffic_graph.edges(), env.traffic_graph.nodes())
    assert flows
    assert len(flows) == env.traffic_graph.number_of_edges()
    vals = sorted(flows.values())
    assert vals[-1] > vals[0], "不同边应对应有差异化的背景流基线"


def test_background_flow_updates_enter_bpr_chain(bg_offline_cache_dir: str) -> None:
    env = _make_offline_real_env(bg_offline_cache_dir)
    env.reset()
    u, v = next(iter(env.traffic_graph.edges()))
    env.time_step = 0
    env.update_background_traffic()
    flow0 = env._background_flow(u, v)
    env.time_step = 72
    env.update_background_traffic()
    flow72 = env._background_flow(u, v)
    assert flow0 >= 0
    assert flow72 >= 0
    assert flow72 != flow0
