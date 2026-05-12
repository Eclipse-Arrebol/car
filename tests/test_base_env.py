"""TrafficPowerEnv 可隔离逻辑单测（不跑整步 step、不测 t2 动作链）。"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.base_env import TrafficPowerEnv  # noqa: E402
from env.entities import EV  # noqa: E402


class TestShouldRequestChargeDecision(unittest.TestCase):
    def setUp(self):
        self.env = TrafficPowerEnv(num_evs=1, respawn_after_full_charge=False)
        self.env.charge_trigger_soc = 30.0
        self.ev = self.env.evs[0]

    def test_not_idle_returns_false(self):
        self.ev.status = "CHARGING"
        self.ev.charge_decision_pending = True
        self.assertFalse(self.env.should_request_charge_decision(self.ev))

    def test_idle_pending_true_returns_true(self):
        self.ev.status = "IDLE"
        self.ev.soc = 10.0
        self.ev.charge_decision_pending = True
        self.assertTrue(self.env.should_request_charge_decision(self.ev))

    def test_high_soc_clears_low_soc_flag(self):
        self.ev.status = "IDLE"
        self.ev.soc = 50.0
        self.ev.charge_decision_pending = False
        self.ev.low_soc_triggered = True
        self.assertFalse(self.env.should_request_charge_decision(self.ev))
        self.assertFalse(self.ev.low_soc_triggered)

    def test_first_low_soc_sets_pending_and_returns_true(self):
        self.ev.status = "IDLE"
        self.ev.soc = 20.0
        self.ev.charge_decision_pending = False
        self.ev.low_soc_triggered = False
        self.assertTrue(self.env.should_request_charge_decision(self.ev))
        self.assertTrue(self.ev.low_soc_triggered)
        self.assertTrue(self.ev.charge_decision_pending)

    def test_low_soc_already_seen_without_pending_returns_false(self):
        self.ev.status = "IDLE"
        self.ev.soc = 20.0
        self.ev.charge_decision_pending = False
        self.ev.low_soc_triggered = True
        self.assertFalse(self.env.should_request_charge_decision(self.ev))


class TestFindEvById(unittest.TestCase):
    def test_find_existing(self):
        env = TrafficPowerEnv(num_evs=2, respawn_after_full_charge=False)
        self.assertIs(env._find_ev_by_id(0), env.evs[0])
        self.assertIs(env._find_ev_by_id(1), env.evs[1])

    def test_find_missing_raises(self):
        env = TrafficPowerEnv(num_evs=1, respawn_after_full_charge=False)
        with self.assertRaises(ValueError):
            env._find_ev_by_id(999)


class TestTrafficPowerEnvStatics(unittest.TestCase):
    def test_parse_speed_kph_string_digits(self):
        self.assertEqual(TrafficPowerEnv._parse_speed_kph("60 km/h"), 60.0)

    def test_parse_speed_graphml_m_per_h_mislabeled_as_kph(self):
        """EMA 等 graphml 中常见 36836～119172 实为 m/h，须折成 km/h。"""
        self.assertAlmostEqual(TrafficPowerEnv._parse_speed_kph(36835.85398520663), 36.83585398520663)
        self.assertAlmostEqual(TrafficPowerEnv._parse_speed_kph(108883.0151488114), 108.8830151488114)
        self.assertEqual(TrafficPowerEnv._parse_speed_kph(80.0), 80.0)

    def test_parse_lanes_count_list_first(self):
        self.assertEqual(TrafficPowerEnv._parse_lanes_count(["2", "1"]), 2.0)

    def test_infer_capacity_motorway(self):
        self.assertEqual(TrafficPowerEnv._infer_capacity_per_lane("motorway_link"), 2200.0)

    def test_parse_capacity_from_explicit(self):
        cap = TrafficPowerEnv._parse_capacity_vehph({"capacity": "1800", "lanes": 1, "highway": "primary"})
        self.assertEqual(cap, 1800.0)

    def test_parse_capacity_fallback_lanes_times_highway(self):
        cap = TrafficPowerEnv._parse_capacity_vehph({"lanes": 2, "highway": "primary"})
        self.assertEqual(cap, 2.0 * 1800.0)


class TestBprTime(unittest.TestCase):
    def test_zero_flow_equals_free_flow(self):
        env = TrafficPowerEnv(num_evs=0, respawn_after_full_charge=False)
        t0 = 0.5
        t = env._bpr_time_h(t0, x_flow=0.0, c_capacity=1000.0)
        self.assertAlmostEqual(t, t0)

    def test_bpr_monotone_in_flow(self):
        env = TrafficPowerEnv(num_evs=0, respawn_after_full_charge=False)
        t0 = 0.5
        c = 600.0
        t_low = env._bpr_time_h(t0, 10.0, c)
        t_high = env._bpr_time_h(t0, 500.0, c)
        self.assertGreater(t_high, t_low)


if __name__ == "__main__":
    unittest.main()
