"""训练规模：80 EV、4 充电站、每站 8 充电桩 — 与 train_hindsight 默认契约对齐。"""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def test_train_hindsight_default_scale_constants():
    """与 docs/HANDOFF 扩展实验一致：默认 80 车、4 站、每站 8 桩。"""
    import train_hindsight as th

    assert getattr(th, "TRAIN_DEFAULT_NUM_EVS", None) == 80
    assert getattr(th, "TRAIN_DEFAULT_NUM_STATIONS", None) == 4
    assert getattr(th, "TRAIN_DEFAULT_NUM_CHARGERS_PER_STATION", None) == 8


def test_parse_args_defaults_match_scale():
    import train_hindsight as th

    old = sys.argv
    try:
        sys.argv = ["train_hindsight.py"]
        args = th.parse_args()
    finally:
        sys.argv = old

    assert args.num_evs == 80
    assert args.num_stations == 4
    assert args.num_chargers_per_station == 8


@pytest.fixture
def offline_cache(tmp_path):
    return str(tmp_path / "scale_cache")


def test_real_traffic_env_80_evs_4_stations_8_chargers_offline_smoke(offline_cache):
    from env.real_env import RealTrafficEnv

    env = RealTrafficEnv(
        offline=True,
        num_stations=4,
        num_evs=80,
        max_nodes=32,
        seed=7,
        cache_dir=offline_cache,
        respawn_after_full_charge=True,
        num_chargers_per_station=8,
    )
    assert len(env.stations) == 4
    assert all(s.num_chargers == 8 for s in env.stations)
    assert len(env.evs) == 80

    env.reset()
    for _ in range(8):
        env.step({})


def test_hindsight_agent_matches_station_count(offline_cache):
    """Hindsight 智能体动作数与站点数一致，图节点数与路网一致。"""
    from agents.hindsight_dqn_agent import HindsightDQNAgent
    from env.real_env import RealTrafficEnv

    env = RealTrafficEnv(
        offline=True,
        num_stations=4,
        num_evs=20,
        max_nodes=24,
        seed=3,
        cache_dir=offline_cache,
        respawn_after_full_charge=False,
        num_chargers_per_station=8,
    )
    env.reset()
    station_node_ids = [s.traffic_node_id for s in env.stations]
    agent = HindsightDQNAgent(
        num_features=18,
        num_actions=4,
        station_node_ids=station_node_ids,
        num_nodes_per_graph=env.num_nodes,
        network_variant="station_only",
        use_action_mask=True,
    )
    assert agent.num_actions == 4
    data = env.get_graph_state()
    mask = env.get_action_mask(env.evs[0])
    _ = agent.select_action(data, action_mask=mask)
