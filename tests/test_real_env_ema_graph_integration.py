"""
层级 1：EMA 完整路网 + RealTrafficEnv 能跑通（仅 T1.1–T1.4，不再加用例）。

T1.1 路网加载 | T1.2 reset | T1.3 单步 step | T1.4 50 步不崩
需 osmnx；缺 ema.graphml 时 skip。Windows 控制台 UTF-8 避免 osm_loader 中文 print 报错。
"""
import random
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import networkx as nx  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from env.osm_loader import HAS_OSMNX, load_road_network_from_file  # noqa: E402
from env.real_env import RealTrafficEnv  # noqa: E402

EMA_GRAPHML = _ROOT / "map_outputs" / "ema" / "ema.graphml"

_LEGAL_EV_STATUSES = frozenset(
    {"IDLE", "MOVING_TO_CHARGE", "WAITING", "CHARGING"}
)


@unittest.skipUnless(EMA_GRAPHML.is_file(), f"未找到 EMA 路网: {EMA_GRAPHML}")
@unittest.skipUnless(HAS_OSMNX, "未安装 osmnx，无法从 graphml 加载路网")
class TestLevel1EmaRoadNetwork(unittest.TestCase):
    """层级 1：仅 T1.1–T1.4。"""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls._cache_dir = cls._td.name

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_T1_1_road_network_load(self):
        """加载 EMA 完整路网；节点/边数；站在图内；无孤立子图（单连通分量）。"""
        graph, station_nodes, _positions = load_road_network_from_file(
            str(EMA_GRAPHML),
            num_stations=2,
            max_nodes=1_000_000,
            seed=42,
            cache_dir=self._cache_dir,
        )
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        self.assertGreater(n, 0, "节点数量为 0")
        self.assertGreater(m, 0, "边数量为 0")
        for sid in station_nodes:
            self.assertIn(sid, graph.nodes(), f"station 节点 {sid} 不在路网里")
        self.assertEqual(
            nx.number_connected_components(graph),
            1,
            "路网非单连通分量（存在与主网断开的孤立子图）",
        )

    def test_T1_2_reset_full(self):
        """50 EV；reset 后 curr_node 合法；path 语义合法；stations 初始化完整。"""
        random.seed(42)
        env = RealTrafficEnv(
            graphml_file=str(EMA_GRAPHML),
            num_stations=2,
            num_evs=50,
            max_nodes=1_000_000,
            seed=42,
            cache_dir=self._cache_dir,
            respawn_after_full_charge=False,
        )
        g = env.traffic_graph
        obs = env.reset()
        self.assertIsInstance(obs, Data)
        self.assertEqual(len(env.evs), 50)
        self.assertEqual(len(env.stations), 2)
        for st in env.stations:
            self.assertIn(st.traffic_node_id, g.nodes())
            self.assertIsNotNone(st.power_node_id)
        for ev in env.evs:
            self.assertIn(ev.curr_node, g.nodes(), "EV curr_node 不在路网节点集合里")
            self.assertIsInstance(ev.path, list)
            if ev.status == "IDLE":
                self.assertEqual(ev.path, [], "IDLE 下 path 合法为空列表")
            else:
                self.assertIsInstance(ev.path, list)

    def test_T1_3_one_step(self):
        """reset 后执行一次 step；info 结构；EV 无非法状态。"""
        random.seed(43)
        env = RealTrafficEnv(
            graphml_file=str(EMA_GRAPHML),
            num_stations=2,
            num_evs=50,
            max_nodes=1_000_000,
            seed=43,
            cache_dir=self._cache_dir,
            respawn_after_full_charge=False,
        )
        env.reset()
        _obs, _reward, _done, info = env.step({})
        for key in ("completed", "abandoned", "pending_t0", "pending_t2", "bus_voltages"):
            self.assertIn(key, info, f"info 缺少 key: {key}")
        for ev in env.evs:
            self.assertIn(
                ev.status,
                _LEGAL_EV_STATUSES,
                f"EV {ev.id} 非法状态: {ev.status!r}",
            )

    def test_T1_4_short_episode_50_steps(self):
        """50 步 step 不崩；结束后环境仍一致、EV 数不变、状态仍合法。"""
        random.seed(44)
        env = RealTrafficEnv(
            graphml_file=str(EMA_GRAPHML),
            num_stations=2,
            num_evs=50,
            max_nodes=1_000_000,
            seed=44,
            cache_dir=self._cache_dir,
            respawn_after_full_charge=False,
        )
        env.reset()
        for _ in range(50):
            env.step({})
        self.assertEqual(len(env.evs), 50)
        for ev in env.evs:
            self.assertIn(ev.status, _LEGAL_EV_STATUSES)
        out = env.get_graph_state()
        self.assertIsInstance(out, Data)


if __name__ == "__main__":
    unittest.main()
