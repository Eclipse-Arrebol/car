"""GraphQNetwork / FeatureEncoder 形状与数值冒烟测试。"""
import math
import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.network import FeatureEncoder, GraphQNetwork  # noqa: E402


def _tiny_grid_data(num_nodes: int = 9, num_edges: int = 24) -> Data:
    """9 节点近似 3×3 栅格：足够跑 GAT，边索引任意连通即可。"""
    torch.manual_seed(0)
    x = torch.randn(num_nodes, 18)
    row = torch.randint(0, num_nodes, (num_edges,))
    col = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = torch.rand(num_edges, 2)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class TestFeatureEncoder(unittest.TestCase):
    def test_output_shape_and_finite(self):
        enc = FeatureEncoder(out_dim=64)
        x = torch.randn(11, 18)
        y = enc(x)
        self.assertEqual(y.shape, (11, 64))
        self.assertTrue(torch.isfinite(y).all())


class TestGraphQNetwork(unittest.TestCase):
    def setUp(self):
        self.data = _tiny_grid_data()
        self.net = GraphQNetwork(
            num_features=18,
            num_actions=2,
            station_node_ids=[0, 8],
            num_nodes_per_graph=9,
            num_edge_features=2,
        )

    def test_forward_t0_shape_and_finite(self):
        q = self.net(self.data, action_mask=None, action_type="t0")
        self.assertEqual(q.shape, (1, 2))
        self.assertTrue(torch.isfinite(q).all())
        qd = q.detach()
        self.assertFalse(any(math.isnan(float(qd[0, i])) for i in range(2)))

    def test_forward_t2_shape(self):
        q = self.net(self.data, action_mask=None, action_type="t2")
        self.assertEqual(q.shape, (1, 2))
        self.assertTrue(torch.isfinite(q).all())

    def test_action_mask_masks_invalid(self):
        mask = torch.tensor([[True, False]], dtype=torch.bool)
        q = self.net(self.data, action_mask=mask, action_type="t0")
        self.assertEqual(q[0, 1].item(), -1e8)

    def test_unknown_action_type_raises(self):
        with self.assertRaises(ValueError):
            self.net(self.data, action_type="unknown")


if __name__ == "__main__":
    unittest.main()
