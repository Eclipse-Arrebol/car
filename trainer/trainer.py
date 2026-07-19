from typing import Dict, Optional, Tuple

from reward_profiles import weights_for


NORM_TRIP = 0.58
NORM_WAIT = 0.17
NORM_QUEUE = NORM_WAIT
NORM_FEE = 65.0


def compute_hindsight_reward(trip, queue, fee, w_trip=0.4, w_queue=0.4, w_fee=0.2):
    return -(
        w_trip * (float(trip) / NORM_TRIP) +
        w_queue * (float(queue) / NORM_WAIT) +
        w_fee * (float(fee) / NORM_FEE)
    )


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
        for ev in self.env.evs:
            if ev.id == ev_id:
                return ev
        return None

    def on_dispatch(self, ev, state, action, action_mask):
        session_idx = ev.charge_sessions
        key = (ev.id, session_idx)
        if key in self.pending:
            raise RuntimeError(
                f"duplicate dispatch: ev_id={ev.id}, session_idx={session_idx}"
            )
        self.pending[key] = PendingEntry(
            state=state,
            action=action,
            action_mask=action_mask,
            dispatch_step=self._current_step,
            ev_id=ev.id,
            session_idx=session_idx,
        )

    def on_charge_started(self, entry, session_idx_override: Optional[int] = None):
        ev_id = entry["ev_id"]
        ev = self._find_ev_by_id(ev_id)

        if session_idx_override is not None:
            session_idx = session_idx_override
        else:
            if ev is None:
                raise RuntimeError(f"on_charge_started: ev_id={ev_id} not found")
            session_idx = ev.charge_sessions

        key = (ev_id, session_idx)
        if key not in self.pending:
            if ev is not None:
                ev.ready_to_settle = False
            return
        pending_entry = self.pending.pop(key)

        trip = entry["actual_trip_time_h"]
        queue = entry["actual_queue_time_h"]
        fee = entry["charging_fee"]
        # per-client reward profile (keyed by city); uniform unless HETERO_REWARD=1
        w = weights_for(getattr(self.env, "grid_variant", None))
        reward = compute_hindsight_reward(trip, queue, fee, *w)
        state = pending_entry.state
        state_for_buffer = state.clone() if hasattr(state, "clone") else state

        self.agent.store_transition(
            state_for_buffer,
            pending_entry.action,
            reward,
            None,
            action_mask=pending_entry.action_mask,
            done=True,
        )
        if ev is not None:
            ev.ready_to_settle = False

    def on_completed(self, entry, session_idx_override: Optional[int] = None):
        return None

    def on_abandoned(self, entry, session_idx_override: Optional[int] = None):
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
        urgent_evs = self.env.get_pending_decision_evs()
        actions = {}
        pending_counts = {s.id: 0 for s in self.env.stations}

        for ev in urgent_evs:
            state = self.env.get_graph_state_for_ev(ev, pending_counts=pending_counts)
            action_mask = self.env.get_action_mask(ev, pending_counts=pending_counts)
            action = self.agent.select_action(state, action_mask=action_mask)
            actions[ev.id] = action
            self.on_dispatch(ev, state, action, action_mask)
            pending_counts[action] = pending_counts.get(action, 0) + 1

        _, _, done, info = self.env.step(actions)

        for entry in info.get("charge_started", []):
            self.on_charge_started(entry)

        for entry in info.get("abandoned", []):
            self.on_abandoned(entry)

        self._current_step += 1
        return done, info
