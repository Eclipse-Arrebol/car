"""
专用于 hindsight 训练的 DQN 智能体。

这个类从 train.py 里的 DQNAgent 拆分出来，保留原有行为，
但不再让训练入口依赖 train.py。
"""

import os

import torch

from agents.dqn_base import DQNBase


class HindsightDQNAgent(DQNBase):
    """面向 hindsight 训练场景的 Double DQN 智能体。"""

    def __init__(self, num_features, num_actions,
                 station_node_ids=None, num_nodes_per_graph=9,
                 network_variant="station_only",
                 use_action_mask=True,
                 epsilon_decay=0.994, epsilon_min=0.05):
        super().__init__(
            num_features, num_actions,
            station_node_ids=station_node_ids,
            num_nodes_per_graph=num_nodes_per_graph,
            memory_size=50000,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            network_variant=network_variant,
            use_action_mask=use_action_mask,
        )

    def replay(self, batch_size):
        """经验回放训练（封装基类 train_on_batch）。"""
        self.train_on_batch(batch_size)

    def save_model(self, path="checkpoints/trained_dqn.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"模型已保存: {path}")

    def load_model(self, path="checkpoints/trained_dqn.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = dict(checkpoint['policy_net'])

        ckpt_station_ids = state_dict.pop('station_node_ids', None)
        if ckpt_station_ids is not None:
            cur_station_ids = self.policy_net.state_dict().get('station_node_ids')
            if cur_station_ids is not None and not torch.equal(ckpt_station_ids, cur_station_ids):
                print(
                    "[load_model] 忽略 checkpoint 中的 station_node_ids，"
                    f"使用当前环境站点: {cur_station_ids.tolist()}"
                )

        cur_state = self.policy_net.state_dict()
        for key in ("conv1.lin_l.weight", "conv1.lin_r.weight"):
            if key in state_dict and key in cur_state:
                old_w, new_w = state_dict[key], cur_state[key]
                if (old_w.shape != new_w.shape and old_w.ndim == 2 and new_w.ndim == 2
                        and old_w.shape[0] == new_w.shape[0]
                        and old_w.shape[1] < new_w.shape[1]):
                    upgraded = new_w.clone().zero_()
                    upgraded[:, :old_w.shape[1]] = old_w
                    state_dict[key] = upgraded
                    print(f"[load_model] 升级旧版输入层权重: {key} "
                          f"{tuple(old_w.shape)} -> {tuple(new_w.shape)}")

        missing, unexpected = self.policy_net.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[load_model] 旧版 checkpoint，缺少 key（使用默认值）: {missing}")
        if unexpected:
            print(f"[load_model] checkpoint 中有多余 key（已忽略）: {unexpected}")
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = checkpoint.get('epsilon', 0.05)
        print(f"模型已加载: {path} (epsilon={self.epsilon:.3f})")
