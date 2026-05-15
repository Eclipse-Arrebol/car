import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


# 轻量版特征分组索引（面向 triptime / queuetime / fee 三目标）
# 时间组：inv_dist(9), travel_time(10), service_time(11)
_IDX_TIME = [9, 10, 11]
# 排队组：queue(2), connected(4), pred_arrivals(14,15,16,17)
_IDX_QUEUE = [2, 4, 14, 15, 16, 17]
# 费用组：price(3)
_IDX_FEE = [3]


class LightweightFeatureEncoder(nn.Module):
    """轻量版特征编码器。

    只保留和当前三目标强相关的输入：
      - triptime: 时间相关
      - queuetime: 队列相关
      - fee: 费用相关

    目标是减少噪声特征，提升收敛稳定性。
    """

    def __init__(self, out_dim: int = 48):
        super().__init__()
        self.time_enc = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        self.queue_enc = nn.Sequential(nn.Linear(6, 16), nn.ReLU())
        self.fee_enc = nn.Sequential(nn.Linear(1, 16), nn.ReLU())

        self.fusion = nn.Sequential(
            nn.Linear(16 * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_feat = self.time_enc(x[:, _IDX_TIME])
        queue_feat = self.queue_enc(x[:, _IDX_QUEUE])
        fee_feat = self.fee_enc(x[:, _IDX_FEE])
        return self.fusion(torch.cat([time_feat, queue_feat, fee_feat], dim=-1))


class LightweightGraphQNetwork(nn.Module):
    """轻量版 GNN Q 网络。

    设计目标：
      1. 更少输入噪声
      2. 更直接对齐 triptime / queuetime / fee
      3. 保留 station-level 选站能力

    说明：
      - 主训练路径只建议使用 action_type='t0'
      - action_type='t2' 仅为旧实验/旧样本兼容保留
    """

    def __init__(
        self,
        num_features,
        num_actions,
        station_node_ids=None,
        num_nodes_per_graph=9,
        num_edge_features=2,
    ):
        super().__init__()

        if station_node_ids is None:
            station_node_ids = [0, 8]

        self.register_buffer(
            "station_node_ids",
            torch.tensor(station_node_ids, dtype=torch.long),
        )
        self.num_nodes_per_graph = num_nodes_per_graph
        self.num_actions = num_actions
        self.num_station_nodes = len(station_node_ids)

        self.feature_encoder = LightweightFeatureEncoder(out_dim=48)
        self.conv1 = GATv2Conv(
            in_channels=48,
            out_channels=32,
            heads=2,
            concat=False,
            edge_dim=num_edge_features,
        )
        self.conv2 = GATv2Conv(
            in_channels=32,
            out_channels=48,
            heads=2,
            concat=False,
            edge_dim=num_edge_features,
        )

        self.value_fc = nn.Sequential(
            nn.Linear(48 + 48, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.t0_advantage_fc = nn.Sequential(
            nn.Linear(48 + 48, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )
        self.t2_advantage_fc = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        nn.init.zeros_(self.t2_advantage_fc[-1].weight)
        nn.init.zeros_(self.t2_advantage_fc[-1].bias)

    def forward(self, data, action_mask=None, action_type="t0"):
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, "edge_attr", None)

        x = self.feature_encoder(x)
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        global_ctx = global_mean_pool(x, batch)
        batch_size = global_ctx.shape[0]
        global_combined = torch.cat([global_ctx, global_ctx], dim=1)
        value = self.value_fc(global_combined)

        if action_type == "t0":
            advantages = []
            for node_id in self.station_node_ids:
                indices = torch.arange(batch_size, device=x.device) * self.num_nodes_per_graph + node_id
                station_emb = x[indices]
                combined = torch.cat([station_emb, global_ctx], dim=1)
                advantages.append(self.t0_advantage_fc(combined))
            advantages = torch.cat(advantages, dim=1)
            q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        elif action_type == "t2":
            q_values = self.t2_advantage_fc(global_ctx)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        if action_mask is not None:
            action_mask = action_mask.to(q_values.device)
            q_values = q_values.masked_fill(~action_mask.bool(), -1e8)

        return q_values
