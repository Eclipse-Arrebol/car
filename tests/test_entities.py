"""EV 类单元测试（见 docs/HANDOFF.md：不测 t2 遗留字段的业务语义）。"""
import random
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.entities import EV  # noqa: E402


class _MockEnv:
    """仅实现 EV.move 所需的 enter_edge / leave_edge。"""

    def __init__(self, length_m: float = 1000.0, speed_kph: float = 60.0, travel_time_h: float = 1.0):
        self._length_m = length_m
        self._speed_kph = speed_kph
        self._travel_time_h = travel_time_h
        self.enter_calls: list[tuple[int, int]] = []
        self.leave_calls: list[tuple[int, int]] = []

    def enter_edge(self, frm, to):
        self.enter_calls.append((frm, to))
        return (self._length_m, self._speed_kph, self._travel_time_h)

    def leave_edge(self, frm, to):
        self.leave_calls.append((frm, to))


class TestEVInit(unittest.TestCase):
    def test_ev_init_core_fields(self):
        random.seed(42)
        try:
            ev = EV(7, 99)
        finally:
            random.seed()

        self.assertEqual(ev.id, 7)
        self.assertEqual(ev.curr_node, 99)
        self.assertIsNone(ev.target_station_idx)
        self.assertEqual(ev.status, "IDLE")
        self.assertEqual(ev.path, [])
        self.assertEqual(ev.travel_time_h, 0.0)
        self.assertEqual(ev.wait_time_h, 0.0)
        self.assertEqual(ev.total_fee_paid, 0.0)
        self.assertEqual(ev.charge_sessions, 0)
        self.assertGreaterEqual(ev.soc, 20.0)
        self.assertLessEqual(ev.soc, 50.0)


class TestEVMove(unittest.TestCase):
    def test_move_noop_when_not_moving_to_charge(self):
        ev = EV(1, 10)
        ev.status = "IDLE"
        env = _MockEnv()
        before = ev.travel_time_h
        ev.move(env, step_hours=1.0)
        self.assertEqual(ev.travel_time_h, before)
        self.assertEqual(ev.last_traversed_nodes, [])
        self.assertEqual(env.enter_calls, [])

    def test_move_one_edge_updates_node_and_travel_time(self):
        ev = EV(1, 1)
        ev.status = "MOVING_TO_CHARGE"
        ev.path = [2]
        env = _MockEnv(length_m=1000.0, speed_kph=60.0, travel_time_h=1.0)
        ev.move(env, step_hours=1.0)
        self.assertEqual(ev.curr_node, 2)
        self.assertEqual(ev.path, [])
        self.assertEqual(ev.travel_time_h, 1.0)
        self.assertEqual(env.enter_calls, [(1, 2)])
        self.assertEqual(env.leave_calls, [(1, 2)])

    def test_move_partial_step_no_leave_edge(self):
        ev = EV(1, 1)
        ev.status = "MOVING_TO_CHARGE"
        ev.path = [2]
        env = _MockEnv(travel_time_h=1.0)
        ev.move(env, step_hours=0.3)
        self.assertEqual(ev.curr_node, 1)
        self.assertEqual(ev.travel_time_h, 0.3)
        self.assertAlmostEqual(ev.remaining_edge_time_h, 0.7)
        self.assertEqual(env.enter_calls, [(1, 2)])
        self.assertEqual(env.leave_calls, [])

    def test_soc_clamped_to_zero_after_large_drain(self):
        ev = EV(1, 1)
        ev.status = "MOVING_TO_CHARGE"
        ev.path = [2]
        ev.soc = 10.0
        # 100 km 等效边，走完一步后 SoC 下降约 30%，低于 0 时应钳到 0
        env = _MockEnv(length_m=100_000.0, speed_kph=60.0, travel_time_h=1.0)
        ev.move(env, step_hours=1.0)
        self.assertEqual(ev.soc, 0.0)


if __name__ == "__main__":
    unittest.main()
