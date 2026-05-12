"""PPPowerGrid33（IEEE33 + pandapower）轻量单测：不依赖真实环境步进。"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.power_grid_pp import IEEE33_STATION_BUSES, PPPowerGrid33  # noqa: E402


class TestPPPowerGrid33Resolve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = PPPowerGrid33(compute_thevenin=False)

    def test_resolve_bus_number_int_and_bus_string(self):
        g = self.grid
        self.assertEqual(g._resolve_bus_number(18), 18)
        self.assertEqual(g._resolve_bus_number("Bus_18"), 18)

    def test_resolve_bus_number_power_node_alias(self):
        g = self.grid
        self.assertEqual(g._resolve_bus_number("Bus_6"), IEEE33_STATION_BUSES[0])

    def test_get_station_power_node(self):
        self.assertEqual(self.grid.get_station_power_node(0), "Bus_6")

    def test_get_last_bus_voltage_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.grid.get_last_bus_voltage("NotABus")


class TestPPPowerGrid33Optimize(unittest.TestCase):
    def test_optimize_power_is_passthrough(self):
        g = PPPowerGrid33(compute_thevenin=False)
        req = {"Bus_6": 123.4, "Bus_18": 50.0}
        self.assertEqual(g.optimize_power(req), req)


class TestPPPowerGrid33PowerFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = PPPowerGrid33(compute_thevenin=False)

    def test_run_power_flow_base_loads_voltages_shape(self):
        g = self.grid
        v = g.run_power_flow({})
        self.assertEqual(len(v), 33)
        for i in range(1, 34):
            self.assertIn(f"Bus_{i}", v)
            vm = v[f"Bus_{i}"]
            self.assertGreater(vm, 0.85)
            self.assertLess(vm, 1.15)

    def test_run_power_flow_with_extra_load(self):
        g = self.grid
        g.run_power_flow({"Bus_18": 400.0})
        self.assertGreaterEqual(g.total_loss, 0.0)
        self.assertTrue(g.line_losses)
        self.assertIsInstance(g.get_last_bus_voltage("Bus_18"), float)


if __name__ == "__main__":
    unittest.main()
