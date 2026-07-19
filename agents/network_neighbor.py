import torch
import torch.nn as nn


class NeighborQNetwork(nn.Module):
    """Per-neighbor Q network for padded local navigation observations."""

    def __init__(
        self,
        num_features,
        num_actions,
        station_node_ids=None,
        num_nodes_per_graph=9,
        use_action_mask=True,
        hidden_dim=64,
        ev_state_dim=2,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.use_action_mask = bool(use_action_mask)
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(int(num_features), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.ev_encoder = nn.Sequential(
            nn.Linear(ev_state_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data, action_mask=None, action_type="t0"):
        if action_type != "t0":
            raise ValueError("NeighborQNetwork only supports action_type='t0'")
        if not isinstance(data, dict):
            raise TypeError("NeighborQNetwork expects dict observations")

        neighbor_features = data["neighbor_features"].float()
        ev_state = data["ev_state"].float()
        if neighbor_features.dim() == 2:
            neighbor_features = neighbor_features.unsqueeze(0)
        if ev_state.dim() == 1:
            ev_state = ev_state.unsqueeze(0)

        batch_size, max_neighbors, _ = neighbor_features.shape
        neighbor_emb = self.neighbor_encoder(neighbor_features)
        ev_emb = self.ev_encoder(ev_state).unsqueeze(1).expand(-1, max_neighbors, -1)

        if action_mask is None and "n_valid_neighbors" in data:
            n_valid = data["n_valid_neighbors"]
            if not torch.is_tensor(n_valid):
                n_valid = torch.as_tensor(n_valid, device=neighbor_features.device)
            n_valid = n_valid.to(neighbor_features.device).long().view(-1)
            idx = torch.arange(max_neighbors, device=neighbor_features.device).unsqueeze(0)
            action_mask = idx < n_valid.unsqueeze(1)

        if action_mask is not None:
            ctx_mask = action_mask.to(neighbor_features.device).bool()
            mask_float = ctx_mask.float().unsqueeze(-1)
            global_ctx = (neighbor_emb * mask_float).sum(1) / mask_float.sum(1).clamp(min=1.0)
        else:
            global_ctx = neighbor_emb.mean(1)

        global_ctx = global_ctx.unsqueeze(1).expand(-1, max_neighbors, -1)
        combined = torch.cat([neighbor_emb, ev_emb, global_ctx], dim=-1)
        q_values = self.score_head(combined).squeeze(-1)

        if action_mask is not None:
            action_mask = action_mask.to(q_values.device).bool()
            q_values = q_values.masked_fill(~action_mask, -torch.inf)

        return q_values
