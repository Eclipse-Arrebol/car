"""车辆充满电后重生逻辑的 TDD 测试。"""
import random

from env.charging_station import ChargingStation
from env.entities import EV


def _fill_and_step(st: ChargingStation, ev: EV, *, soc: float = 96.0) -> None:
    ev.soc = soc
    ev.status = "CHARGING"
    st.queue = []
    st.connected_evs = [ev]
    st.step(tou_multiplier=1.0, price_noise=0.0, step_duration_h=0.0, lmp=3.5)


def test_full_charge_should_respawn_to_new_node(monkeypatch) -> None:
    """充满电后应随机重生到新的节点，而不是停留在原节点。"""
    st = ChargingStation(0, 0, "Grid_A", respawn_after_full_charge=True)
    st.legal_respawn_nodes = [3, 4, 5]
    ev = EV(0, 7)

    # 让未来实现中的随机重生结果可预测。
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 15.0)

    old_node = ev.curr_node
    _fill_and_step(st, ev, soc=96.0)

    assert ev.charge_sessions == 1
    assert ev.curr_node != old_node
    assert ev.curr_node in {3, 4, 5}
    assert ev.status == "IDLE"
    assert 10.0 <= ev.soc <= 20.0


def test_full_charge_should_snapshot_session_counters_before_clear(monkeypatch) -> None:
    """充满电时应先写 snapshot，再清空会话累计字段。"""
    st = ChargingStation(0, 0, "Grid_A", respawn_after_full_charge=True)
    ev = EV(0, 7)
    ev.wait_time_h = 1.25
    ev.travel_time_h = 2.5
    ev.total_fee_paid = 12.75
    ev.total_energy_charged = 8.0

    monkeypatch.setattr(random, "choice", lambda seq: seq[-1])
    monkeypatch.setattr(random, "uniform", lambda a, b: 30.0)

    _fill_and_step(st, ev, soc=96.0)

    assert ev.charge_fee_snapshot == 12.75
    assert ev.charge_queue_time_h == 1.25
    assert ev.charge_travel_time_h == 2.5
    assert ev.total_fee_paid == 0.0
    assert ev.wait_time_h == 0.0
    assert ev.travel_time_h == 0.0
    assert ev.total_energy_charged == 0.0


def test_full_charge_should_clear_old_trip_state_before_respawn(monkeypatch) -> None:
    """重生前应先清空旧行程状态，避免带着上一单的路径继续跑。"""
    st = ChargingStation(0, 0, "Grid_A", respawn_after_full_charge=True)
    ev = EV(0, 7)
    ev.path = [8, 9, 10]
    ev.last_traversed_nodes = [6, 5]
    ev.current_edge_from = 7
    ev.current_edge_target = 8
    ev.remaining_edge_time_h = 0.75
    ev.current_edge_speed_kph = 40.0
    ev.current_edge_length_m = 300.0
    ev.target_station_idx = 2
    ev.assigned_station = "station_ref"

    monkeypatch.setattr(random, "choice", lambda seq: seq[-1])
    monkeypatch.setattr(random, "uniform", lambda a, b: 30.0)

    _fill_and_step(st, ev, soc=96.0)

    assert ev.path == []
    assert ev.last_traversed_nodes == []
    assert ev.current_edge_from is None
    assert ev.current_edge_target is None
    assert ev.remaining_edge_time_h == 0.0
    assert ev.current_edge_speed_kph == 0.0
    assert ev.current_edge_length_m == 0.0
    assert ev.target_station_idx is None
    assert ev.assigned_station is None


def test_full_charge_should_respawn_with_low_soc_and_trigger_again(monkeypatch) -> None:
    """重生后应回到低电量区间，并能再次触发充电决策链路。"""
    st = ChargingStation(0, 0, "Grid_A", respawn_after_full_charge=True)
    ev = EV(0, 7)

    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 15.0)

    _fill_and_step(st, ev, soc=96.0)

    assert 10.0 <= ev.soc <= 20.0
    assert ev.low_soc_triggered is False
    assert ev.charge_decision_pending is False
    assert ev.status == "IDLE"


def test_full_charge_should_keep_environment_steppable(monkeypatch) -> None:
    """重生后环境仍应保持可 step，不应破坏后续仿真。"""
    st = ChargingStation(0, 0, "Grid_A", respawn_after_full_charge=True)
    ev = EV(0, 7)

    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random, "uniform", lambda a, b: 15.0)

    _fill_and_step(st, ev, soc=96.0)
    st.step(tou_multiplier=1.0, price_noise=0.0, step_duration_h=0.0, lmp=3.5)

    assert ev.status in {"IDLE", "CHARGING"}
    assert ev.curr_node is not None


def test_multiple_respawns_remain_stable(monkeypatch) -> None:
    """连续重生多次时，状态仍应稳定且合法。"""
    st = ChargingStation(0, 0, "Grid_A", respawn_after_full_charge=True)
    st.legal_respawn_nodes = [11, 12, 13, 14]
    ev = EV(0, 7)

    nodes = iter([11, 12, 13, 14])
    monkeypatch.setattr(random, "choice", lambda seq: next(nodes))
    monkeypatch.setattr(random, "uniform", lambda a, b: 15.0)

    _fill_and_step(st, ev, soc=96.0)
    second_node = ev.curr_node
    _fill_and_step(st, ev, soc=96.0)
    third_node = ev.curr_node

    assert ev.charge_sessions == 2
    assert second_node == 11
    assert third_node == 12
    assert ev.curr_node in {11, 12}
    assert ev.path == []
    assert ev.target_station_idx is None
    assert ev.assigned_station is None
