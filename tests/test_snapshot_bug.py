"""复现 / 锁定 charging_station 充满电 snapshot 与累计字段语义（TDD）。"""
import pytest

from env.charging_station import ChargingStation
from env.entities import EV


@pytest.fixture
def station_no_respawn() -> ChargingStation:
    return ChargingStation(
        station_id=0,
        traffic_node_id=0,
        power_node_id="Grid_A",
        respawn_after_full_charge=False,
    )


@pytest.fixture
def ev0() -> EV:
    return EV(0, 1)


def _trigger_full_charge(
    station: ChargingStation,
    ev: EV,
    *,
    fee: float,
    wait_h: float,
    travel_h: float,
    soc: float = 99.0,
    step_duration_h: float = 0.0,
) -> None:
    """让 EV 在本步内满足 soc>=95 并完成充满分支；step_duration_h=0 避免额外计费/加 SOC。"""
    ev.total_fee_paid = fee
    ev.wait_time_h = wait_h
    ev.travel_time_h = travel_h
    ev.soc = soc
    ev.status = "CHARGING"
    station.queue = []
    station.connected_evs = [ev]
    station.step(
        tou_multiplier=1.0,
        price_noise=0.0,
        step_duration_h=step_duration_h,
        lmp=None,
    )


def test_snapshot_is_session_only_not_cumulative(
    station_no_respawn: ChargingStation, ev0: EV
) -> None:
    """第二次充满时 snapshot 应仅为第二次会话累计，不应混入第一次。"""
    st = station_no_respawn
    ev = ev0

    _trigger_full_charge(st, ev, fee=10.0, wait_h=1.0, travel_h=0.5)
    assert ev.charge_fee_snapshot == 10.0
    assert ev.charge_queue_time_h == 1.0
    assert ev.charge_travel_time_h == 0.5

    ev.soc = 50.0
    ev.total_fee_paid += 5.0
    ev.wait_time_h += 0.3
    ev.travel_time_h += 0.2
    _trigger_full_charge(
        st,
        ev,
        fee=ev.total_fee_paid,
        wait_h=ev.wait_time_h,
        travel_h=ev.travel_time_h,
    )

    assert ev.charge_fee_snapshot == 5.0
    assert ev.charge_queue_time_h == 0.3
    assert ev.charge_travel_time_h == 0.2


def test_cumulative_fields_reset_after_charge_complete(
    station_no_respawn: ChargingStation, ev0: EV
) -> None:
    """充满后累计字段应归零（本次会话语义）；charge_sessions 已增加。"""
    st = station_no_respawn
    ev = ev0
    _trigger_full_charge(st, ev, fee=3.0, wait_h=0.4, travel_h=0.1)

    assert ev.travel_time_h == 0.0
    assert ev.wait_time_h == 0.0
    assert ev.total_fee_paid == 0.0
    assert ev.total_energy_charged == 0.0
    assert ev.charge_sessions >= 1
