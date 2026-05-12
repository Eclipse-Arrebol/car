"""RealTrafficEnv 单测：offline 合成路网，不依赖 OSM 下载与本地 graphml。"""
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# osm_loader 命中缓存时会 print 中文；Windows cp1252 控制台会炸，测试里统一 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from torch_geometric.data import Data  # noqa: E402

from env.power_grid_pp import IEEE33_STATION_BUSES, PPPowerGrid33  # noqa: E402
from env.real_env import RealTrafficEnv, _safe_path_display  # noqa: E402


class TestSafePathDisplay(unittest.TestCase):
    def test_returns_relative_when_under_root(self):
        rel = _safe_path_display(str(_ROOT / "tests" / "test_real_env.py"))
        self.assertIn("test_real_env.py", rel)
        self.assertFalse(os.path.isabs(rel) or rel == str(_ROOT))


class TestRealTrafficEnvOffline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls._cache_dir = cls._td.name

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def setUp(self):
        random.seed(7)

    def test_offline_init_wires_stations_to_ieee_buses(self):
        env = RealTrafficEnv(
            offline=True,
            num_evs=2,
            num_stations=2,
            max_nodes=18,
            seed=99,
            cache_dir=self._cache_dir,
            respawn_after_full_charge=False,
        )
        self.assertIsInstance(env.power_grid, PPPowerGrid33)
        self.assertEqual(len(env.stations), 2)
        self.assertEqual(len(env.station_node_ids), 2)
        for i, st in enumerate(env.stations):
            self.assertEqual(st.id, i)
            self.assertEqual(st.traffic_node_id, env.station_node_ids[i])
            self.assertEqual(st.power_node_id, env.power_grid.get_station_power_node(i))
            self.assertEqual(st.power_bus_idx, IEEE33_STATION_BUSES[i])

    def test_reset_returns_graph_data_same_num_nodes(self):
        env = RealTrafficEnv(
            offline=True,
            num_evs=3,
            num_stations=2,
            max_nodes=16,
            seed=1,
            cache_dir=self._cache_dir,
        )
        n = env.num_nodes
        out = env.reset()
        self.assertIsInstance(out, Data)
        self.assertEqual(out.x.shape[0], n)
        self.assertEqual(len(env.evs), 3)

    def test_step_empty_actions_returns_tuple(self):
        env = RealTrafficEnv(
            offline=True,
            num_evs=2,
            num_stations=2,
            max_nodes=14,
            seed=2,
            cache_dir=self._cache_dir,
        )
        env.reset()
        obs, reward, done, info = env.step({})
        self.assertIsInstance(obs, Data)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)
        self.assertIn("bus_voltages", info)


if __name__ == "__main__":
    unittest.main()
