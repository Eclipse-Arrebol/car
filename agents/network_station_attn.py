"""站间注意力版 Q 网络（station_attn）。

与 `network_station_only.StationOnlyGraphQNetwork` 是一组**受控对照**:
- 消费完全相同的输入特征(同一份 state、同样的 STATION_FEATURE_IDXS / EV_FEATURE_IDXS),
- 唯一的结构差异:把 mean-pool 这个**钝的跨站交互**换成**站间多头自注意力**,
  让每个站点的表示能显式地和其它站点做相对比较("A 站在变挤 → 把这辆分流到 B")。
- 顺带把原版的双重 Python for 循环改成**全向量化**,推理快很多,便于做并发/规模扫描。

设计动机见与导师讨论:mean-pool 是当前协调优势(贡献2 +1.7%)的一个结构性天花板,
注意力给网络真正的站间关系推理能力;即便不一定提升,也构成"全局上下文机制"的
干净消融对照。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# 与 station_only 保持逐位一致,确保只比较"如何融合站间信息",而非"看哪些特征"。
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


class StationAttnGraphQNetwork(nn.Module):
    """站间自注意力 Q 网络。

    输入/输出与 StationOnlyGraphQNetwork 完全一致:
      - data.x: [num_nodes, >=19]
      - 输出:    [batch_size, num_actions]

    可直接替换 station_only 用于训练/评估(network_variant='station_attn')。
    注意:权重结构与 station_only 不同,**无法复用其 warm-start**。
    """

    def __init__(
        self,
        num_features,
        num_actions,
        station_node_ids: Sequence[int] | None = None,
        num_nodes_per_graph: int = 9,
        num_edge_features: int = 2,
        use_action_mask: bool = True,
        embed_dim: int = 32,
        num_heads: int = 4,
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
        self.embed_dim = embed_dim

        d = embed_dim
        self.station_enc = nn.Sequential(
            nn.Linear(len(STATION_FEATURE_IDXS), 32),
            nn.ReLU(),
            nn.Linear(32, d),
            nn.ReLU(),
        )
        self.ev_enc = nn.Sequential(
            nn.Linear(len(EV_FEATURE_IDXS), 8),
            nn.ReLU(),
        )
        # 把 EV 上下文融进每个站点 token
        self.fusion = nn.Sequential(
            nn.Linear(d + 8, d),
            nn.LayerNorm(d),
            nn.ReLU(),
        )
        # 站间多头自注意力:核心改动,替代 mean-pool
        self.attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d)
        # 打分头:每站 [自身上下文 | 注意力后的全局上下文]
        self.head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

    def _get_batch(self, data, x: torch.Tensor) -> torch.Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            return data.batch
        return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

    def forward(self, data, action_mask=None, action_type: str = "t0"):
        if action_type != "t0":
            raise ValueError("StationAttnGraphQNetwork only supports action_type='t0'")

        x = data.x
        required_dim = max(max(STATION_FEATURE_IDXS), max(EV_FEATURE_IDXS)) + 1
        if x.shape[1] < required_dim:
            x = F.pad(x, (0, required_dim - x.shape[1]))

        batch = self._get_batch(data, x)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        if batch_size == 0:
            batch_size = 1

        N = self.num_nodes_per_graph
        if x.shape[0] != batch_size * N:
            # 兼容不规则批:用图内节点数反推(假设每图节点数一致)
            N = x.shape[0] // batch_size
        xg = x.view(batch_size, N, -1)  # [B, N, F]
        device = x.device

        # 自车:每图内 SOC 列(idx 8)argmax,与 station_only 的 _find_ev_node 语义一致
        soc = xg[:, :, 8]                                   # [B, N]
        ev_idx = soc.argmax(dim=1)                          # [B]
        ev_nodes = xg[torch.arange(batch_size, device=device), ev_idx]  # [B, F]
        ev_feat = ev_nodes[:, EV_FEATURE_IDXS]              # [B, |EV|]

        # 站点特征
        station_ids = self.station_node_ids.to(device)
        valid_id_mask = station_ids < N                     # 防越界(与原版 None 占位对应)
        safe_ids = station_ids.clamp(max=N - 1)
        S = station_ids.numel()
        station_nodes = xg[:, safe_ids, :]                  # [B, S, F]
        station_feat = station_nodes[:, :, STATION_FEATURE_IDXS]  # [B, S, 7]

        # 编码 + 融合 EV 上下文
        station_h = self.station_enc(station_feat)          # [B, S, d]
        ev_h = self.ev_enc(ev_feat)                         # [B, 8]
        ev_b = ev_h.unsqueeze(1).expand(-1, S, -1)          # [B, S, 8]
        fused = self.fusion(torch.cat([station_h, ev_b], dim=-1))  # [B, S, d]

        # 组装 key_padding_mask(True=忽略):越界站点 + 被动作掩码屏蔽的站点
        key_padding = (~valid_id_mask).unsqueeze(0).expand(batch_size, -1).clone()  # [B, S]
        if self.use_action_mask and action_mask is not None:
            am = action_mask.to(device).bool()
            if am.dim() == 1:
                am = am.unsqueeze(0)
            am = am[:, :S]
            key_padding = key_padding | (~am)
        # 某行全屏蔽会让注意力输出 NaN:临时放开,输出阶段再统一屏蔽
        all_masked = key_padding.all(dim=1)
        if all_masked.any():
            key_padding[all_masked] = False

        # 站间自注意力 + 残差归一
        attn_out, _ = self.attn(fused, fused, fused, key_padding_mask=key_padding)
        attended = self.attn_norm(fused + attn_out)         # [B, S, d]

        # 全局上下文:对有效站点做掩码均值
        valid = (~key_padding).float().unsqueeze(-1)        # [B, S, 1]
        global_ctx = (attended * valid).sum(1) / valid.sum(1).clamp(min=1.0)  # [B, d]
        global_ctx = global_ctx.unsqueeze(1).expand(-1, S, -1)               # [B, S, d]

        combined = torch.cat([attended, global_ctx], dim=-1)  # [B, S, 2d]
        q_values = self.head(combined).squeeze(-1)            # [B, S]

        # 越界站点钉成 -1e8(对齐原版 None 占位)
        invalid_id = (~valid_id_mask).unsqueeze(0).expand(batch_size, -1)
        q_values = q_values.masked_fill(invalid_id, -1e8)

        # 宽度补齐到 num_actions(一般 S == num_actions)
        if q_values.shape[1] < self.num_actions:
            pad = q_values.new_full(
                (batch_size, self.num_actions - q_values.shape[1]), -1e8
            )
            q_values = torch.cat([q_values, pad], dim=1)

        if self.use_action_mask and action_mask is not None:
            am = action_mask.to(q_values.device).bool()
            if am.dim() == 1:
                am = am.unsqueeze(0)
            q_values = q_values.masked_fill(~am, -1e8)

        return q_values
