"""充满电后应清理 EV 路上状态字段（TDD）。"""
import pytest

from env.charging_station import ChargingStation
from env.entities import EV


@pytest.fixture
def st() -> ChargingStation:
    return ChargingStation(0, 0, "Grid_A", respawn_after_full_charge=False)


@pytest.fixture
def ev() -> EV:
    return EV(0, 1)


def _trigger_full(st: ChargingStation, ev: EV, *, soc: float = 99.0) -> None:
    ev.soc = soc
    ev.status = "CHARGING"
    st.queue = []
    st.connected_evs = [ev]
    st.step(tou_multiplier=1.0, price_noise=0.0, step_duration_h=0.0, lmp=None)


def test_path_cleared_after_charge_complete(st: ChargingStation, ev: EV) -> None:
    ev.path = [1, 2, 3, 4, 5]
    ev.last_traversed_nodes = [7, 8]
    _trigger_full(st, ev, soc=96.0)
    assert ev.path == []
    assert ev.last_traversed_nodes == []


def test_edge_state_cleared_after_charge_complete(st: ChargingStation, ev: EV) -> None:
    ev.current_edge_from = 10
    ev.current_edge_target = 20
    ev.remaining_edge_time_h = 0.5
    ev.current_edge_speed_kph = 55.0
    ev.current_edge_length_m = 200.0
    _trigger_full(st, ev, soc=96.0)
    assert ev.current_edge_from is None
    assert ev.current_edge_target is None
    assert ev.remaining_edge_time_h == 0.0
    assert ev.current_edge_speed_kph == 0.0
    assert ev.current_edge_length_m == 0.0


def test_target_station_cleared_after_charge_complete(st: ChargingStation, ev: EV) -> None:
    ev.target_station_idx = 0
    ev.assigned_station = "stub_station_ref"
    _trigger_full(st, ev, soc=96.0)
    assert ev.target_station_idx is None
    assert ev.assigned_station is None
