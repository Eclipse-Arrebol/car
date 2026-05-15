"""HindsightTrainer 核心行为：强断言 + 骨架未实现（预期 FAILED = NotImplementedError）。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.base_env import TrafficPowerEnv
from trainer.trainer import HindsightTrainer


def _ensure_low_soc_idle_pending(env: TrafficPowerEnv) -> None:
    """不修改 env/ 源码；通过 EV 公开属性保证 get_pending_decision_evs 非空。"""
    for ev in env.evs:
        ev.status = "IDLE"
        ev.soc = 25.0
        ev.low_soc_triggered = False
        ev.charge_decision_pending = False


class _MockAgent:
    """
    与 agents/dqn_base.DQNAgent.store_transition 签名对齐；
    额外接受 **kwargs，便于 Step 2 写入 ev_id / session_idx 等供本文件断言。
    """

    def __init__(self):
        self.memory = []
        self.transitions = []

    def select_action(self, state, action_mask=None, action_type="t0"):
        """占位：与 policy 选站接口形状一致，本测试可不调用。"""
        return 0

    def store_transition(self, state, action, reward, next_state, action_mask=None, **kwargs):
        rec = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "action_mask": action_mask,
        }
        rec.update(kwargs)
        self.transitions.append(rec)
        self.memory.append(rec)


def test_pending_key_uses_charge_sessions_for_isolation():
    """同一 EV 完成两次充电, buffer 里两条 transition 独立。"""
    env = TrafficPowerEnv(num_evs=1)
    env.reset()
    # EV 初始 SOC 随机；若 >= charge_trigger_soc(30) 则永不进入待决策队列，step 不会 dispatch
    _ensure_low_soc_idle_pending(env)
    agent = _MockAgent()
    trainer = HindsightTrainer(env, agent)

    for _ in range(800):
        trainer.step_episode()
        if len(agent.transitions) >= 2:
            break

    assert len(agent.transitions) >= 2, (
        f"期望至少 2 条 transition, 实际 {len(agent.transitions)}"
    )

    ev = env.evs[0]
    assert agent.transitions[0].get("ev_id") is None
    assert agent.transitions[1].get("ev_id") is None

    t1, t2 = agent.transitions[0], agent.transitions[1]
    assert t1["state"] is not t2["state"], "两条 transition 的 state 不能是同一引用"

    assert t1["reward"] != t2["reward"], "两次充电 hindsight reward 应可区分（非覆盖同一条）"

    for sid in (0, 1):
        assert (ev.id, sid) not in trainer.pending, (
            f"会话 sid={sid} 完成后不应仍挂在 pending（若 Step 2 使用其它 session 编号规则请同步改断言）"
        )


def test_reward_uses_completed_snapshot_fields():
    """on_completed 算出的 reward 公式正确。"""
    # ------------------------------------------------------------------
    # Step 2 接口设计提醒（与 env 真实语义相关，本测试不裁决实现）:
    # pending key 一般为 (ev_id, session_idx)，其中 session_idx 常取 dispatch 时
    # ev.charge_sessions。若仅 mock completed、未跑 env 充满逻辑，则 EV 上
    # charge_sessions 仍为 0；若 on_completed 用「charge_sessions - 1」取 pending
    # 会得到 -1，取不到条目。
    # 可选方向:
    #   (a) on_completed(entry, session_idx_at_dispatch) 由调用方显式传入
    #   (b) on_completed 假定 entry 对应「刚完成」的会话，从 EV 当前 charge_sessions
    #       反推 dispatch 时的 session_idx（需与 dispatch 写入规则一致）
    # Step 2 实现时择一并保持与 step_episode 内调用一致即可。
    # ------------------------------------------------------------------
    env = TrafficPowerEnv(num_evs=1)
    agent = _MockAgent()
    trainer = HindsightTrainer(env, agent)

    env.reset()
    _ensure_low_soc_idle_pending(env)
    pending_evs = env.get_pending_decision_evs()
    assert len(pending_evs) >= 1, "需要至少一辆 IDLE 低 SOC EV"
    ev = pending_evs[0]
    state = env.get_graph_state_for_ev(ev)
    mask = env.get_action_mask(ev)
    trainer.on_dispatch(ev, state, 0, mask)

    entry = {
        "ev_id": ev.id,
        "actual_trip_time_h": 1.0,
        "actual_queue_time_h": 2.0,
        "charging_fee": 100.0,
    }
    expected = -(0.3 * 1.0 + 0.5 * 2.0 + 0.03 * 100.0)

    trainer.on_completed(entry, session_idx_override=0)

    assert len(agent.transitions) == 1
    actual_reward = agent.transitions[0]["reward"]
    assert abs(actual_reward - expected) < 1e-9, (
        f"reward 期望 {expected}, 实际 {actual_reward}"
    )


def test_transition_enters_buffer_only_at_completed_step():
    """transition 入 buffer 的时机 == completed 那一步, 不能提前。"""
    env = TrafficPowerEnv(num_evs=1)
    env.reset()
    _ensure_low_soc_idle_pending(env)
    agent = _MockAgent()
    trainer = HindsightTrainer(env, agent)

    pending_evs = env.get_pending_decision_evs()
    assert len(pending_evs) >= 1
    ev = pending_evs[0]
    state = env.get_graph_state_for_ev(ev)
    mask = env.get_action_mask(ev)

    pending_before = len(trainer.pending)
    buffer_before = len(agent.transitions)

    trainer.on_dispatch(ev, state, 0, mask)

    assert len(trainer.pending) == pending_before + 1, "dispatch 必须新增 pending"
    assert len(agent.transitions) == buffer_before, "dispatch 不能立刻入 buffer"

    entry = {
        "ev_id": ev.id,
        "actual_trip_time_h": 0.5,
        "actual_queue_time_h": 0.3,
        "charging_fee": 50.0,
    }
    trainer.on_completed(entry, session_idx_override=0)

    assert len(trainer.pending) == pending_before, "completed 后 pending 必须减回去"
    assert len(agent.transitions) == buffer_before + 1, "completed 后 buffer 必须 +1"


def test_abandoned_ev_drops_pending_without_buffer_growth():
    """场景:dispatch 一辆 EV, 模拟 abandon, 断言 pending 清空 + buffer 不变。"""
    env = TrafficPowerEnv(num_evs=1)
    env.reset()
    _ensure_low_soc_idle_pending(env)
    agent = _MockAgent()
    trainer = HindsightTrainer(env, agent)

    pending_evs = env.get_pending_decision_evs()
    assert len(pending_evs) >= 1
    ev = pending_evs[0]
    state = env.get_graph_state_for_ev(ev)
    mask = env.get_action_mask(ev)

    trainer.on_dispatch(ev, state, 0, mask)

    assert len(trainer.pending) == 1
    assert len(agent.transitions) == 0

    abandon_entry = {
        "ev_id": ev.id,
        "reason": "queue_full",
    }

    trainer.on_abandoned(abandon_entry, session_idx_override=0)

    assert len(trainer.pending) == 0, "abandon 后 pending 必须清空"
    assert len(agent.transitions) == 0, "abandon 不能入 buffer"


def test_abandoned_with_no_pending_does_not_raise():
    """
    场景:on_abandoned 收到一个不在 pending 里的 ev_id, 不应抛错。
    理由:abandon 可能发生在 trainer 没记录 dispatch 的 EV 上。
    """
    env = TrafficPowerEnv(num_evs=1)
    env.reset()
    agent = _MockAgent()
    trainer = HindsightTrainer(env, agent)

    assert len(trainer.pending) == 0

    abandon_entry = {
        "ev_id": 999,
        "reason": "timeout",
    }

    trainer.on_abandoned(abandon_entry, session_idx_override=0)

    assert len(trainer.pending) == 0
    assert len(agent.transitions) == 0


def test_abandoned_does_not_affect_other_sessions_of_same_ev():
    """
    场景:同一 EV 的两个不同 session_idx 的 pending 同时存在,
    abandon 其中一个, 另一个不受影响。
    """
    env = TrafficPowerEnv(num_evs=1)
    env.reset()
    _ensure_low_soc_idle_pending(env)
    agent = _MockAgent()
    trainer = HindsightTrainer(env, agent)

    pending_evs = env.get_pending_decision_evs()
    assert len(pending_evs) >= 1
    ev = pending_evs[0]
    state = env.get_graph_state_for_ev(ev)
    mask = env.get_action_mask(ev)

    trainer.on_dispatch(ev, state, 0, mask)

    from trainer.trainer import PendingEntry

    trainer.pending[(ev.id, 1)] = PendingEntry(
        state=state,
        action=1,
        action_mask=mask,
        dispatch_step=0,
        ev_id=ev.id,
        session_idx=1,
    )

    assert len(trainer.pending) == 2, "需要先有两个 pending 才能测隔离"

    abandon_entry = {
        "ev_id": ev.id,
        "reason": "queue_full",
    }
    trainer.on_abandoned(abandon_entry, session_idx_override=0)

    assert (ev.id, 0) not in trainer.pending, "session 0 应被 drop"
    assert (ev.id, 1) in trainer.pending, "session 1 不应被影响"
    assert len(agent.transitions) == 0, "abandon 不入 buffer"
