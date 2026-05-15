"""训练/决策时打印轻量网络输入特征的调试脚本。

用途:
- 观察 `LightweightGraphQNetwork` 实际吃到的决策特征是否符合预期
- 检查 triptime / queuetime / fee 相关输入是否过多、是否存在明显异常值

默认只跑少量步骤并打印每个 urgent EV 的图状态摘要。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from env.real_env import RealTrafficEnv
from agents.hindsight_dqn_agent import HindsightDQNAgent
from trainer.trainer import HindsightTrainer


TIME_IDX = [9, 10, 11]
QUEUE_IDX = [2, 4, 14, 15, 16, 17]
FEE_IDX = [3]
SOC_IDX = 8
TRACKED_IDXS = sorted(set(TIME_IDX + QUEUE_IDX + FEE_IDX + [SOC_IDX]))


def _fmt_vals(x: torch.Tensor, idxs: Iterable[int]) -> str:
    parts = []
    for idx in idxs:
        if idx < x.numel():
            parts.append(f"{idx}={float(x[idx]):.4f}")
    return ", ".join(parts)


def _nonzero_nodes(x: torch.Tensor, threshold: float = 1e-9):
    rows = []
    if x.ndim != 2:
        return rows
    for i in range(x.shape[0]):
        row = x[i]
        if torch.any(torch.abs(row) > threshold):
            rows.append(i)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--num-evs", type=int, default=10)
    p.add_argument("--respawn", action="store_true", default=True)
    p.add_argument("--no-respawn", dest="respawn", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    env = RealTrafficEnv(
        graphml_file=os.path.join("map_outputs", "ema", "ema.graphml"),
        num_stations=2,
        num_evs=args.num_evs,
        max_nodes=1_000_000,
        cache_dir=os.path.join("map_outputs", "ema_cache"),
        seed=args.seed,
        respawn_after_full_charge=args.respawn,
    )
    agent = HindsightDQNAgent(
        num_features=18,
        num_actions=2,
        station_node_ids=None,
        num_nodes_per_graph=9,
    )
    trainer = HindsightTrainer(env, agent)

    env.reset()
    trainer.pending.clear()
    trainer._current_step = 0

    print("[debug] start")
    print(f"[debug] steps={args.steps} num_evs={args.num_evs} respawn={args.respawn} seed={args.seed}")
    print(f"[debug] network={agent.policy_net.__class__.__name__}")
    print(f"[debug] time_idx={TIME_IDX} queue_idx={QUEUE_IDX} fee_idx={FEE_IDX}")

    for step in range(args.steps):
        urgent_evs = env.get_pending_decision_evs()
        print(f"\n[step {step}] urgent_evs={len(urgent_evs)}")

        actions = {}
        decisions = {}
        for ev in urgent_evs:
            state = env.get_graph_state_for_ev(ev)
            x = state.x if hasattr(state, "x") else state
            print(
                f"  ev_id={ev.id} soc={getattr(ev, 'soc', None)} "
                f"status={getattr(ev, 'status', None)} target_station_idx={getattr(ev, 'target_station_idx', None)} "
                f"assigned_station={getattr(ev, 'assigned_station', None)}"
            )
            if hasattr(x, "shape"):
                print(f"    x.shape={tuple(x.shape)}")
                if x.ndim == 2 and x.shape[0] > 0:
                    station_node_ids = [s.traffic_node_id for s in env.stations]
                    interested_nodes = []
                    interested_nodes.append(ev.curr_node)
                    interested_nodes.extend(station_node_ids)
                    interested_nodes.extend(_nonzero_nodes(x))
                    seen = set()
                    ordered_nodes = []
                    for node_idx in interested_nodes:
                        if node_idx not in seen and 0 <= node_idx < x.shape[0]:
                            ordered_nodes.append(node_idx)
                            seen.add(node_idx)
                    print(f"    station_node_ids={station_node_ids}")
                    print(f"    nonzero_nodes={_nonzero_nodes(x)[:20]}{' ...' if len(_nonzero_nodes(x)) > 20 else ''}")
                    for node_idx in ordered_nodes[:15]:
                        node_type = []
                        if node_idx in station_node_ids:
                            node_type.append("station")
                        if node_idx == ev.curr_node:
                            node_type.append("ev")
                        tag = "/".join(node_type) if node_type else "other"
                        print(
                            f"    node[{node_idx}]({tag}): "
                            f"{_fmt_vals(x[node_idx], TRACKED_IDXS)}"
                        )

            action_mask = env.get_action_mask(ev)
            print(f"    action_mask={action_mask.tolist() if hasattr(action_mask, 'tolist') else action_mask}")

            with torch.no_grad():
                q = agent.policy_net(state, action_mask=action_mask, action_type="t0")
            action = int(q.argmax(dim=1).item())
            actions[ev.id] = action
            decisions[ev.id] = {
                "action": action,
                "q_values": q.detach().cpu().tolist(),
            }
            print(f"    q_values={q.detach().cpu().tolist()} action={action}")

        _, _, done, info = env.step(actions)
        for ev in urgent_evs:
            print(
                f"    after_step ev_id={ev.id} target_station_idx={getattr(ev, 'target_station_idx', None)} "
                f"assigned_station={getattr(ev, 'assigned_station', None)} station={getattr(ev, 'station', None)} "
                f"path_len={len(getattr(ev, 'path', []) or [])} status={getattr(ev, 'status', None)} "
                f"decision={decisions.get(ev.id)}"
            )
        print(
            f"  completed={len(info.get('completed', []))} "
            f"abandoned={len(info.get('abandoned', []))} done={done}"
        )

        for entry in info.get("completed", []):
            print(
                "    completed: "
                f"ev_id={entry.get('ev_id')} "
                f"trip={entry.get('actual_trip_time_h')} "
                f"queue={entry.get('actual_queue_time_h')} "
                f"fee={entry.get('charging_fee')}"
            )

        if done:
            print("[debug] env done")
            break

    print("[debug] finish")


if __name__ == "__main__":
    main()
