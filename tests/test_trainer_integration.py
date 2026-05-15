"""
trainer + 真 agent 端到端 smoke test。

目的:防止 mock drift —— mock 测试只能覆盖 trainer 内部逻辑,
不能覆盖 trainer 跟真 agent 的接口契约。这个文件用真 DQNAgent 和真 env
跑短训练,确保 store_transition / replay 链路通畅。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.base_env import TrafficPowerEnv
from train import DQNAgent
from trainer.trainer import HindsightTrainer


def _ensure_low_soc_idle_all_evs(env: TrafficPowerEnv) -> None:
    """保证车辆会进入 get_pending_decision_evs（与 test_trainer_core 契约一致）。"""
    for ev in env.evs:
        ev.status = "IDLE"
        ev.soc = 25.0
        ev.low_soc_triggered = False
        ev.charge_decision_pending = False


def _make_real_components(num_evs=10):
    # 与 train_hindsight 默认每站桩数对齐，缩短排队、便于短步数内攒满 buffer
    env = TrafficPowerEnv(
        num_evs=num_evs,
        respawn_after_full_charge=True,
        num_chargers_per_station=8,
    )
    agent = DQNAgent(
        num_features=18,
        num_actions=2,
        station_node_ids=None,
        num_nodes_per_graph=9,
    )
    trainer = HindsightTrainer(env, agent)
    return env, agent, trainer


def test_trainer_step_episode_with_real_agent_no_kwarg_error():
    """trainer.step_episode 调用真 agent.store_transition 不应报 TypeError。"""
    env, agent, trainer = _make_real_components(num_evs=10)
    env.reset()
    _ensure_low_soc_idle_all_evs(env)
    for _ in range(50):
        trainer.step_episode()


def test_replay_works_with_hindsight_buffer_entries():
    """buffer 里 next_state=None 的样本应能正常 replay,不报 AttributeError。

    每站 8 桩时多车并行会触发电网上限降额，单步 SOC 增速变慢；200 仿真步内
    通常可攒够若干条 completed，用较小 batch 做 smoke 即可。
    """
    env, agent, trainer = _make_real_components(num_evs=10)
    env.reset()
    _ensure_low_soc_idle_all_evs(env)
    for _ in range(200):
        trainer.step_episode()
        if len(agent.memory) >= 4:
            agent.replay(4)
            return
    pytest.fail("200 step 内 buffer 没攒到 4 个样本 — 检查 env / 充电逻辑是否回退")


def test_buffer_entries_have_done_true():
    """hindsight 范式下,每个入 buffer 的样本 done 都应该是 True。"""
    env, agent, trainer = _make_real_components(num_evs=2)
    env.reset()
    _ensure_low_soc_idle_all_evs(env)
    for _ in range(100):
        trainer.step_episode()
        if len(agent.memory) >= 1:
            sample = agent.memory[0]
            assert len(sample) >= 6, f"tuple 长度 {len(sample)},缺 done 字段"
            assert sample[5] is True, f"hindsight 样本 done 应为 True,实际 {sample[5]}"
            assert sample[3] is None, f"hindsight 样本 next_state 应为 None,实际类型 {type(sample[3])}"
            return
    pytest.fail("100 step 内 buffer 没攒到样本")
