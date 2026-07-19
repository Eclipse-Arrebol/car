"""站点节点专用的轻量 Q 网络。

设计目标:
- 避开大量普通路网节点上的 0 特征噪声
- 只使用对选站决策最直接的站点节点信息
- 保留与当前 EV 相关的 SOC 作为上下文

适合当前只优化 triptime / queuetime / fee 的场景。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


STATION_FEATURE_IDXS = [
    3,   # price / lmp
    5,   # station load ratio
    10,  # trip_time
    11,  # service_time
    15,  # queue_wait_ratio
    17,  # spare chargers ratio
    18,  # real-time mean field
]

EV_FEATURE_IDXS = [8]  # SOC


class StationFeatureEncoder(nn.Module):
    """只编码站点节点上的关键决策特征。"""

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.station_enc = nn.Sequential(
            nn.Linear(len(STATION_FEATURE_IDXS), 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )
        self.ev_enc = nn.Sequential(
            nn.Linear(len(EV_FEATURE_IDXS), 8),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(out_dim + 8, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, station_feat: torch.Tensor, ev_feat: torch.Tensor) -> torch.Tensor:
        station_h = self.station_enc(station_feat)
        ev_h = self.ev_enc(ev_feat)
        return self.fusion(torch.cat([station_h, ev_h], dim=-1))


class StationOnlyGraphQNetwork(nn.Module):
    """站点专用 Q 网络。

    输入:
      - data.x: [num_nodes, 19]
      - data.station_node_ids: 可选, 若未提供则使用初始化时传入的 station_node_ids

    输出:
      - [batch_size, num_actions]
    """

    def __init__(
        self,
        num_features,
        num_actions,
        station_node_ids: Sequence[int] | None = None,
        num_nodes_per_graph: int = 9,
        num_edge_features: int = 2,
        use_action_mask: bool = True,
    ):
        super().__init__()
        if station_node_ids is None:
            station_node_ids = [0, 8]

        self.register_buffer(
            "station_node_ids",
            torch.tensor(list(station_node_ids), dtype=torch.long),
        )
        self.num_actions = num_actions
        self.num_nodes_per_graph = num_nodes_per_graph
        self.num_edge_features = num_edge_features
        self.use_action_mask = use_action_mask

        self.encoder = StationFeatureEncoder(out_dim=32)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _get_batch(self, data, x: torch.Tensor) -> torch.Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

    def _find_ev_node(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= 8:
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        soc_col = x[:, 8]
        if torch.any(soc_col > 0):
            return int(torch.argmax(soc_col).item())
        return 0

    def forward(self, data, action_mask=None, action_type: str = "t0"):
        if action_type != "t0":
            raise ValueError("StationOnlyGraphQNetwork only supports action_type='t0'")

        x = data.x
        required_dim = max(max(STATION_FEATURE_IDXS), max(EV_FEATURE_IDXS)) + 1
        if x.shape[1] < required_dim:
            x = F.pad(x, (0, required_dim - x.shape[1]))
        batch = self._get_batch(data, x)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        if batch_size == 0:
            batch_size = 1

        station_node_ids = self.station_node_ids.to(x.device)
        q_values = []

        for b in range(batch_size):
            node_offset = b * self.num_nodes_per_graph
            ev_node_idx = node_offset + self._find_ev_node(x[node_offset: node_offset + self.num_nodes_per_graph])
            ev_feat = x[ev_node_idx, EV_FEATURE_IDXS].unsqueeze(0)

            station_hs = []
            for node_id in station_node_ids.tolist():
                global_idx = node_offset + node_id
                if global_idx >= x.shape[0]:
                    station_hs.append(None)
                    continue
                station_feat = x[global_idx, STATION_FEATURE_IDXS].unsqueeze(0)
                h = self.encoder(station_feat, ev_feat)
                station_hs.append(h.squeeze(0))

            valid_hs = [h for h in station_hs if h is not None]
            if valid_hs:
                station_embeddings = torch.stack(valid_hs, dim=0)
                global_ctx = station_embeddings.mean(dim=0)
            else:
                global_ctx = torch.zeros(32, device=x.device, dtype=x.dtype)

            station_qs = []
            for h in station_hs:
                if h is None:
                    station_qs.append(torch.tensor(-1e8, device=x.device))
                    continue
                combined = torch.cat([h, global_ctx], dim=-1)
                q = self.head(combined).squeeze(-1)
                station_qs.append(q.squeeze(0))
            q_values.append(torch.stack(station_qs, dim=0))

        q_values = torch.stack(q_values, dim=0)

        if self.use_action_mask and action_mask is not None:
            action_mask = action_mask.to(q_values.device)
            q_values = q_values.masked_fill(~action_mask.bool(), -1e8)

        return q_values
