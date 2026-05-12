"""ChargingStation 单元测试（隔离路网/电网，仅测站内逻辑）。"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.charging_station import ChargingStation  # noqa: E402


class _StubEV:
    """满足 ChargingStation 估算与 step 所需字段的最小 EV。"""

    def __init__(self, ev_id=1, soc=50.0, target_soc=95.0):
        self.id = ev_id
        self.soc = soc
        self.target_soc = target_soc
        self.battery_capacity_kwh = 60.0
        self.charge_efficiency = 0.92
        self.status = "WAITING"
        self.charge_started_count = 0
        self.wait_time_h = 0.0
        self.travel_time_h = 0.0
        self.total_fee_paid = 0.0
        self.total_energy_charged = 0.0
        self.charge_sessions = 0
        self.low_soc_triggered = True
        self.charge_decision_pending = True
        self.remaining_replans = 0


class TestChargingStationInit(unittest.TestCase):
    def test_init_core_fields(self):
        st = ChargingStation(7, 10, 20, num_chargers=3, max_charger_power=22.0, max_grid_power=55.0)
        self.assertEqual(st.id, 7)
        self.assertEqual(st.traffic_node_id, 10)
        self.assertEqual(st.power_node_id, 20)
        self.assertEqual(st.num_chargers, 3)
        self.assertEqual(st.max_charger_power, 22.0)
        self.assertEqual(st.max_grid_power, 55.0)
        self.assertEqual(st.queue, [])
        self.assertEqual(st.connected_evs, [])


class TestChargingStationPrice(unittest.TestCase):
    def test_update_price_without_lmp(self):
        st = ChargingStation(0, 1, 2)
        p = st.update_price(tou_multiplier=1.0, price_noise=0.0, lmp=None)
        # energy = base_price * tou * (1+noise) = 1.0；拥塞 0；+ service_markup
        self.assertAlmostEqual(p, 1.15)
        st.queue.append(_StubEV())
        st.connected_evs.append(_StubEV(2))
        p2 = st.update_price(tou_multiplier=1.0, price_noise=0.0, lmp=None)
        self.assertAlmostEqual(p2, 1.0 + 0.15 + 0.08 * 2)


class TestChargingStationOptimize(unittest.TestCase):
    def test_optimize_power_scales_when_over_grid_cap(self):
        st = ChargingStation(0, 1, 2, num_chargers=4, max_charger_power=20.0, max_grid_power=30.0)
        ev1 = _StubEV(1, soc=50.0)
        ev2 = _StubEV(2, soc=50.0)
        st.connected_evs = [ev1, ev2]
        alloc = st.optimize_power()
        self.assertAlmostEqual(alloc[1], 15.0)
        self.assertAlmostEqual(alloc[2], 15.0)
        self.assertAlmostEqual(st.last_total_load, 30.0)

    def test_optimize_power_empty(self):
        st = ChargingStation(0, 1, 2)
        self.assertEqual(st.optimize_power(), {})
        self.assertEqual(st.last_total_load, 0.0)


class TestChargingStationStep(unittest.TestCase):
    def test_step_moves_queue_to_charging(self):
        st = ChargingStation(0, 1, 2, num_chargers=2)
        ev = _StubEV(1, soc=30.0)
        st.queue = [ev]
        st.current_price = 1.0
        st.step(step_duration_h=0.01, tou_multiplier=1.0, price_noise=0.0, lmp=None)
        self.assertEqual(st.queue, [])
        self.assertEqual(len(st.connected_evs), 1)
        self.assertEqual(ev.status, "CHARGING")
        self.assertEqual(ev.charge_started_count, 1)


class TestChargingStationEstimates(unittest.TestCase):
    def test_estimate_charge_time_zero_when_at_target(self):
        st = ChargingStation(0, 1, 2)
        ev = _StubEV(1, soc=95.0, target_soc=95.0)
        self.assertEqual(st.estimate_charge_time_hours(ev), 0.0)

    def test_update_arrival_prediction_ema(self):
        st = ChargingStation(0, 1, 2)
        st.predicted_arrivals = 10.0
        v = st.update_arrival_prediction(0.0)
        self.assertAlmostEqual(v, 7.0)


if __name__ == "__main__":
    unittest.main()
