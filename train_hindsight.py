"""
Hindsight Contextual Bandit 训练入口。

设计契约(见 HANDOFF_v3.md):
- 单步 hindsight bandit:done 恒 True,next_state 恒 None
- Reward 在 EV 充满电那一步用 snapshot 算(trainer 内部完成)
- Abandon 直接 drop pending,不入 buffer
- trainer.step_episode() 是唯一入口,严禁外部 store_transition / 手动算 reward

与旧 train.py 的差异:
- 不再从 env.step 返回值取 reward(hindsight 范式下外部看不到 reward)
- replay 频率简化为"每步 + buffer 够大"
- 增加周期 save(旧版只末尾 save 一次)
"""

import argparse
import os
import sys
import time
import torch

# Windows 控制台 cp1252 兼容(旧 agent.save_model 内部有中文 print)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass  # Python < 3.7 或非 TTY 环境,降级跳过

from env.real_env import RealTrafficEnv
from train import DQNAgent  # 临时方案: 复用 train.py 里的现成实现
from trainer.trainer import HindsightTrainer


def parse_args():
    p = argparse.ArgumentParser()
    # Training scale (legacy hard-coded values exposed as CLI)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--steps-per-episode", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)

    # Env arguments exposed per Step 0 decisions
    p.add_argument("--num-evs", type=int, default=10)
    p.add_argument("--respawn", action="store_true", default=True,
                   help="Respawn EV after full charge (default True)")
    p.add_argument("--no-respawn", dest="respawn", action="store_false")

    # Save strategy
    p.add_argument("--save-dir", type=str, default="checkpoints_hindsight")
    p.add_argument("--save-every", type=int, default=50,
                   help="Save a snapshot every N episodes")

    # Logging
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()

    env = RealTrafficEnv(
        graphml_file=os.path.join("map_outputs", "ema", "ema.graphml"),
        num_stations=2,
        num_evs=args.num_evs,
        max_nodes=1_000_000,
        cache_dir=os.path.join("map_outputs", "ema_cache"),
        seed=42,
        respawn_after_full_charge=args.respawn,
    )
    agent = DQNAgent(
        num_features=18,
        num_actions=2,
        station_node_ids=None,
        num_nodes_per_graph=9,
    )
    trainer = HindsightTrainer(env, agent)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[setup] episodes={args.episodes} steps={args.steps_per_episode} "
          f"batch={args.batch_size} num_evs={args.num_evs} respawn={args.respawn}")
    print(f"[setup] save_dir={args.save_dir} save_every={args.save_every}")

    t0 = time.time()
    for episode in range(args.episodes):
        env.reset()
        trainer.pending.clear()
        trainer._current_step = 0

        steps_run = 0
        for step in range(args.steps_per_episode):
            done = trainer.step_episode()
            steps_run += 1

            if len(agent.memory) >= args.batch_size:
                agent.replay(args.batch_size)

            if done:
                break

        if episode % args.log_every == 0:
            epsilon = getattr(agent, "epsilon", None)
            eps_str = f"{epsilon:.3f}" if isinstance(epsilon, (int, float)) else "N/A"
            elapsed = time.time() - t0
            print(f"[ep {episode+1}/{args.episodes}] "
                  f"steps={steps_run} "
                  f"buffer={len(agent.memory)} "
                  f"pending={len(trainer.pending)} "
                  f"epsilon={eps_str} "
                  f"elapsed={elapsed:.1f}s")

        if (episode + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f"model_ep{episode+1}.pth")
            agent.save_model(path)
            print(f"  [save] {path}")

    final_path = os.path.join(args.save_dir, "model_final.pth")
    agent.save_model(final_path)
    print(f"[done] final model -> {final_path}")
    print(f"[done] total elapsed = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
