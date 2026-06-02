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
    def __init__(self, env, agent, constraint_config: Optional[dict] = None):
        self.env = env
        self.agent = agent
        self.pending: Dict[Tuple[int, int], PendingEntry] = {}
        self._current_step = 0
        self.constraint_config = dict(constraint_config or {})
        self.constraint_mode = self.constraint_config.get("mode", "off")
        self.lambda_lf = float(self.constraint_config.get("lambda_lf", 0.0))
        self.lambda_v = float(self.constraint_config.get("lambda_v", 0.0))
        self.completed_rewards_this_step = []
        self.reset_episode_constraints()

    def reset_episode_constraints(self):
        self._prev_power_kw = 0.0
        self._episode_lf_norm = 0.0
        self._episode_voltage_risk_norm = 0.0
        self._episode_steps = 0
        self._episode_peak_power_kw = 0.0
        self._episode_network_loss_kwh = 0.0
        self._episode_min_voltage_pu = 1.0
        self._last_grid_penalty = 0.0

    def _constraint_enabled(self) -> bool:
        return self.constraint_mode == "lagrangian"

    def _user_reward(self, trip: float, queue: float, fee: float) -> float:
        return -(
            float(self.constraint_config.get("trip_weight", 0.3)) * trip
            + float(self.constraint_config.get("queue_weight", 0.5)) * queue
            + float(self.constraint_config.get("fee_weight", 0.03)) * fee
        )

    def _observe_grid_constraints(self, info: dict) -> float:
        if not self._constraint_enabled():
            self._last_grid_penalty = 0.0
            return 0.0

        power_kw = float(info.get("realized_power", 0.0))
        line_loss_kw = float(info.get("line_losses", 0.0))
        min_voltage_pu = float(info.get("min_voltage_pu", 1.0))
        step_duration_h = float(info.get("step_duration_h", getattr(self.env, "step_duration_h", 1.0 / 6.0)))

        lf_raw = (power_kw - self._prev_power_kw) ** 2
        self._prev_power_kw = power_kw

        v_min_threshold = float(self.constraint_config.get("voltage_min_pu", 0.95))
        voltage_risk_raw = max(0.0, v_min_threshold - min_voltage_pu)

        lf_scale = max(1e-9, float(self.constraint_config.get("lf_scale", 1_000_000.0)))
        voltage_scale = max(1e-9, float(self.constraint_config.get("voltage_scale", 0.01)))
        lf_norm = lf_raw / lf_scale
        voltage_risk_norm = voltage_risk_raw / voltage_scale

        self._episode_lf_norm += lf_norm
        self._episode_voltage_risk_norm += voltage_risk_norm
        self._episode_steps += 1
        self._episode_peak_power_kw = max(self._episode_peak_power_kw, power_kw)
        self._episode_network_loss_kwh += line_loss_kw * step_duration_h
        self._episode_min_voltage_pu = min(self._episode_min_voltage_pu, min_voltage_pu)

        self._last_grid_penalty = self.lambda_lf * lf_norm + self.lambda_v * voltage_risk_norm
        return self._last_grid_penalty

    def update_lagrange_multipliers(self) -> dict:
        steps = max(1, self._episode_steps)
        avg_lf = self._episode_lf_norm / steps
        avg_voltage_risk = self._episode_voltage_risk_norm / steps
        lf_limit = float(self.constraint_config.get("lf_limit", 0.05))
        voltage_risk_limit = float(self.constraint_config.get("voltage_risk_limit", 0.0))

        if self._constraint_enabled():
            eta_lf = float(self.constraint_config.get("lambda_lf_lr", 0.05))
            eta_v = float(self.constraint_config.get("lambda_v_lr", 0.05))
            self.lambda_lf = max(0.0, self.lambda_lf + eta_lf * (avg_lf - lf_limit))
            self.lambda_v = max(0.0, self.lambda_v + eta_v * (avg_voltage_risk - voltage_risk_limit))

        return {
            "constraint_mode": self.constraint_mode,
            "lambda_lf": self.lambda_lf,
            "lambda_v": self.lambda_v,
            "avg_lf_norm": avg_lf,
            "avg_voltage_risk_norm": avg_voltage_risk,
            "lf_limit": lf_limit,
            "voltage_risk_limit": voltage_risk_limit,
            "peak_power_kw": self._episode_peak_power_kw,
            "network_loss_kwh": self._episode_network_loss_kwh,
            "min_voltage_pu": self._episode_min_voltage_pu,
        }

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

    def on_completed(
        self,
        entry,
        session_idx_override: Optional[int] = None,
        grid_penalty_share: float = 0.0,
    ):
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
        reward = self._user_reward(trip, queue, fee) - float(grid_penalty_share)
        self.completed_rewards_this_step.append(reward)

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
        self.completed_rewards_this_step = []
        grid_penalty = self._observe_grid_constraints(info)
        completed_entries = list(info.get("completed", []))
        grid_penalty_share = grid_penalty / max(1, len(completed_entries))

        for entry in completed_entries:
            self.on_completed(entry, grid_penalty_share=grid_penalty_share)

        for entry in info.get("abandoned", []):
            self.on_abandoned(entry)

        self._current_step += 1
        return done, info
