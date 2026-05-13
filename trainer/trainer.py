from typing import Dict, Optional, Tuple


class PendingEntry:
    def __init__(self, state, action, action_mask, dispatch_step, ev_id, session_idx):
        self.state = state
        self.action = action
        self.action_mask = action_mask
        self.dispatch_step = dispatch_step
        self.ev_id = ev_id
        self.session_idx = session_idx


class HindsightTrainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.pending: Dict[Tuple[int, int], PendingEntry] = {}
        self._current_step = 0

    def _find_ev_by_id(self, ev_id):
        """从 env.evs 反查 EV 对象。"""
        for ev in self.env.evs:
            if ev.id == ev_id:
                return ev
        return None

    def on_dispatch(self, ev, state, action, action_mask):
        """
        Dispatch 时调用: 写 pending, 不入 buffer。
        session_idx = ev.charge_sessions 当下的快照值。
        """
        session_idx = ev.charge_sessions
        key = (ev.id, session_idx)
        if key in self.pending:
            raise RuntimeError(
                f"重复 dispatch: ev_id={ev.id}, session_idx={session_idx} 已在 pending 中"
            )
        self.pending[key] = PendingEntry(
            state=state,
            action=action,
            action_mask=action_mask,
            dispatch_step=self._current_step,
            ev_id=ev.id,
            session_idx=session_idx,
        )

    def on_completed(self, entry, session_idx_override: Optional[int] = None):
        """
        Completed 时调用: 取 pending, 算 hindsight reward, 入 buffer。

        entry: info["completed"] 的单条 dict, 含 ev_id / actual_trip_time_h /
               actual_queue_time_h / charging_fee
        session_idx_override: 测试场景显式指定 session_idx;
                             生产场景留 None, 自动从 env.evs 反查
        """
        ev_id = entry["ev_id"]

        if session_idx_override is not None:
            session_idx = session_idx_override
        else:
            ev = self._find_ev_by_id(ev_id)
            if ev is None:
                raise RuntimeError(f"on_completed: 在 env.evs 找不到 ev_id={ev_id}")
            session_idx = ev.charge_sessions - 1

        key = (ev_id, session_idx)
        if key not in self.pending:
            raise RuntimeError(
                f"on_completed: pending 中找不到 key={key};"
                f" 可能 dispatch 未记录或 session_idx 推断错误"
            )
        pending_entry = self.pending.pop(key)

        trip = entry["actual_trip_time_h"]
        queue = entry["actual_queue_time_h"]
        fee = entry["charging_fee"]
        reward = -(0.3 * trip + 0.5 * queue + 0.03 * fee)

        self.agent.store_transition(
            pending_entry.state,
            pending_entry.action,
            reward,
            None,
            action_mask=pending_entry.action_mask,
            done=True,
        )

    def on_abandoned(self, entry, session_idx_override: Optional[int] = None):
        """
        Abandoned 时调用: 直接 drop pending, 不入 buffer。

        注意: abandon 不会自增 ev.charge_sessions, 所以反查时不需要 -1。
        """
        ev_id = entry["ev_id"]

        if session_idx_override is not None:
            session_idx = session_idx_override
        else:
            ev = self._find_ev_by_id(ev_id)
            if ev is None:
                return
            session_idx = ev.charge_sessions

        key = (ev_id, session_idx)
        self.pending.pop(key, None)

    def step_episode(self):
        """
        跑环境的一步:
        1. 拿 urgent_evs
        2. 对每个 ev: 选 action, 写 pending
        3. env.step(actions)
        4. 处理 completed / abandoned
        5. 返回 done
        """
        urgent_evs = self.env.get_pending_decision_evs()
        actions = {}

        for ev in urgent_evs:
            state = self.env.get_graph_state_for_ev(ev)
            action_mask = self.env.get_action_mask(ev)
            action = self.agent.select_action(state, action_mask=action_mask)
            actions[ev.id] = action
            self.on_dispatch(ev, state, action, action_mask)

        _, _, done, info = self.env.step(actions)

        for entry in info.get("completed", []):
            self.on_completed(entry)

        for entry in info.get("abandoned", []):
            self.on_abandoned(entry)

        self._current_step += 1
        return done, info
