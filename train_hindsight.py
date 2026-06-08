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
from env.traffic_profiles import TRAFFIC_PROFILE_CHOICES
from agents.hindsight_dqn_agent import HindsightDQNAgent
from trainer.trainer import HindsightTrainer

# 默认训练规模（与 tests/test_hindsight_train_scale.py 契约一致）
TRAIN_DEFAULT_NUM_EVS = 80
TRAIN_DEFAULT_NUM_STATIONS = 4
TRAIN_DEFAULT_NUM_CHARGERS_PER_STATION = 8


def parse_args():
    p = argparse.ArgumentParser()
    # Training scale (legacy hard-coded values exposed as CLI)
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--steps-per-episode", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)

    # Env arguments exposed per Step 0 decisions
    p.add_argument("--num-evs", type=int, default=TRAIN_DEFAULT_NUM_EVS)
    p.add_argument("--num-stations", type=int, default=TRAIN_DEFAULT_NUM_STATIONS)
    p.add_argument(
        "--num-chargers-per-station",
        type=int,
        default=TRAIN_DEFAULT_NUM_CHARGERS_PER_STATION,
        help="Each ChargingStation.num_chargers (default 8)",
    )
    p.add_argument("--respawn", action="store_true", default=True,
                   help="Respawn EV after full charge (default True)")
    p.add_argument("--no-respawn", dest="respawn", action="store_false")
    p.add_argument(
        "--traffic-profile",
        type=str,
        default="base",
        choices=TRAFFIC_PROFILE_CHOICES,
        help="Regional traffic style: base, old_city, new_city, or suburb.",
    )

    # UE background (TNTP Frank–Wolfe); default on when both files exist
    p.add_argument(
        "--no-ue-background",
        action="store_true",
        default=False,
        help="Use heuristic background_edge_base_flows instead of UE (default: UE on)",
    )
    p.add_argument(
        "--ue-net-tntp",
        type=str,
        default=os.path.join("map_outputs", "ema", "EMA_net.tntp"),
        help="TNTP net file for UE baseline",
    )
    p.add_argument(
        "--ue-trips-tntp",
        type=str,
        default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"),
        help="TNTP trips (OD) file for UE baseline",
    )
    p.add_argument("--ue-max-iter", type=int, default=800, help="Frank–Wolfe max iterations")
    p.add_argument("--ue-tol", type=float, default=1e-4, help="FW relative-gap tolerance")
    p.add_argument("--ue-scale", type=float, default=1.0, help="Scale UE edge flows after solve")
    p.add_argument(
        "--ue-verbose",
        action="store_true",
        default=False,
        help="Print Frank–Wolfe UE iteration logs (default: off)",
    )

    # Network / mask switch
    p.add_argument("--network", type=str, default="station_only",
                   choices=["original", "lightweight", "station_only"],
                   help="Choose Q-network variant")
    p.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=True,
                   help="Enable action mask inside Q network (default True)")
    p.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")

    # Save strategy
    p.add_argument("--save-dir", type=str, default="checkpoints_hindsight")
    p.add_argument("--save-every", type=int, default=50,
                   help="Save a snapshot every N episodes")

    # Constrained RL: user objective + grid constraints via Lagrangian multipliers
    p.add_argument(
        "--constraint-mode",
        type=str,
        default="off",
        choices=["off", "lagrangian"],
        help="Enable grid-constrained hindsight reward with adaptive Lagrangian penalties.",
    )
    p.add_argument("--lambda-lf-init", type=float, default=0.0,
                   help="Initial multiplier for load fluctuation constraint.")
    p.add_argument("--lambda-v-init", type=float, default=0.0,
                   help="Initial multiplier for voltage-risk constraint.")
    p.add_argument("--lambda-lf-lr", type=float, default=0.05,
                   help="Learning rate for load fluctuation multiplier.")
    p.add_argument("--lambda-v-lr", type=float, default=0.05,
                   help="Learning rate for voltage-risk multiplier.")
    p.add_argument("--lf-scale", type=float, default=1_000_000.0,
                   help="Normalization scale for (P_t - P_{t-1})^2 in kW^2.")
    p.add_argument("--lf-limit", type=float, default=0.05,
                   help="Average normalized load fluctuation constraint threshold.")
    p.add_argument("--voltage-min-pu", type=float, default=0.95,
                   help="Minimum acceptable bus voltage in pu.")
    p.add_argument("--voltage-scale", type=float, default=0.01,
                   help="Normalization scale for voltage risk max(0, voltage_min - Vmin).")
    p.add_argument("--voltage-risk-limit", type=float, default=0.0,
                   help="Average normalized voltage risk constraint threshold.")
    p.add_argument("--trip-weight", type=float, default=0.3)
    p.add_argument("--queue-weight", type=float, default=0.5)
    p.add_argument("--fee-weight", type=float, default=0.03)

    # Logging
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def _constraint_config_from_args(args) -> dict:
    return {
        "mode": args.constraint_mode,
        "lambda_lf": args.lambda_lf_init,
        "lambda_v": args.lambda_v_init,
        "lambda_lf_lr": args.lambda_lf_lr,
        "lambda_v_lr": args.lambda_v_lr,
        "lf_scale": args.lf_scale,
        "lf_limit": args.lf_limit,
        "voltage_min_pu": args.voltage_min_pu,
        "voltage_scale": args.voltage_scale,
        "voltage_risk_limit": args.voltage_risk_limit,
        "trip_weight": args.trip_weight,
        "queue_weight": args.queue_weight,
        "fee_weight": args.fee_weight,
    }


def main():
    args = parse_args()

    env_kw: dict = dict(
        graphml_file=os.path.join("map_outputs", "ema", "ema.graphml"),
        num_stations=args.num_stations,
        num_evs=args.num_evs,
        num_chargers_per_station=args.num_chargers_per_station,
        max_nodes=1_000_000,
        cache_dir=os.path.join("map_outputs", "ema_cache"),
        seed=42,
        respawn_after_full_charge=args.respawn,
        traffic_profile=args.traffic_profile,
    )
    if not args.no_ue_background:
        net_p = args.ue_net_tntp
        trip_p = args.ue_trips_tntp
        if os.path.isfile(net_p) and os.path.isfile(trip_p):
            env_kw["background_ue_net_tntp"] = os.path.abspath(net_p)
            env_kw["background_ue_trips_tntp"] = os.path.abspath(trip_p)
            env_kw["background_ue_max_iter"] = int(args.ue_max_iter)
            env_kw["background_ue_tol"] = float(args.ue_tol)
            env_kw["background_ue_scale"] = float(args.ue_scale)
            env_kw["background_ue_verbose"] = bool(args.ue_verbose)
            print(
                f"[setup] UE background baseline: net={net_p} trips={trip_p} "
                f"max_iter={args.ue_max_iter} tol={args.ue_tol} "
                f"base_scale={args.ue_scale} traffic_profile={args.traffic_profile}"
            )
        else:
            print(
                f"[setup] warn: UE TNTP not found (net={os.path.isfile(net_p)}, "
                f"trips={os.path.isfile(trip_p)}), using heuristic background_edge_base_flows"
            )
    else:
        print("[setup] heuristic background (--no-ue-background)")

    env = RealTrafficEnv(**env_kw)
    station_node_ids = [s.traffic_node_id for s in env.stations]
    print(f"[setup] station_node_ids={station_node_ids}")
    agent = HindsightDQNAgent(
        num_features=18,
        num_actions=args.num_stations,
        station_node_ids=station_node_ids,
        num_nodes_per_graph=env.num_nodes,
        network_variant=args.network,
        use_action_mask=args.use_action_mask,
    )
    trainer = HindsightTrainer(env, agent, constraint_config=_constraint_config_from_args(args))

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[setup] episodes={args.episodes} steps={args.steps_per_episode} "
          f"batch={args.batch_size} num_evs={args.num_evs} num_stations={args.num_stations} "
          f"chargers_per_station={args.num_chargers_per_station} respawn={args.respawn} "
          f"traffic_profile={args.traffic_profile} "
          f"ue_background={not args.no_ue_background} "
          f"network={args.network} use_action_mask={args.use_action_mask} "
          f"full_no_mask_experiment={not args.use_action_mask} "
          f"constraint_mode={args.constraint_mode}")
    if args.constraint_mode != "off":
        print(
            f"[setup] constraints: lf_limit={args.lf_limit} lf_scale={args.lf_scale} "
            f"voltage_min={args.voltage_min_pu} voltage_risk_limit={args.voltage_risk_limit} "
            f"lambda_lf_init={args.lambda_lf_init} lambda_v_init={args.lambda_v_init} "
            f"lambda_lr=({args.lambda_lf_lr},{args.lambda_v_lr})"
        )
    print(f"[setup] save_dir={args.save_dir} save_every={args.save_every}")

    t0 = time.time()
    for episode in range(args.episodes):
        env.reset()
        trainer.pending.clear()
        trainer._current_step = 0
        trainer.reset_episode_constraints()

        steps_run = 0
        ep_trip, ep_queue, ep_fee, ep_reward = [], [], [], []
        for step in range(args.steps_per_episode):
            done, info = trainer.step_episode()
            steps_run += 1

            for entry in info.get("completed", []):
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                ep_trip.append(trip)
                ep_queue.append(queue)
                ep_fee.append(fee)
            ep_reward.extend(trainer.completed_rewards_this_step)

            if len(agent.memory) >= args.batch_size:
                agent.replay(args.batch_size)

            if done:
                break

        constraint_stats = trainer.update_lagrange_multipliers()

        if episode % args.log_every == 0:
            epsilon = getattr(agent, "epsilon", None)
            eps_str = f"{epsilon:.3f}" if isinstance(epsilon, (int, float)) else "N/A"
            elapsed = time.time() - t0

            def _avg(vals):
                return sum(vals) / len(vals) if vals else 0.0

            print(f"[ep {episode+1}/{args.episodes}] "
                  f"steps={steps_run} "
                  f"buffer={len(agent.memory)} "
                  f"pending={len(trainer.pending)} "
                  f"epsilon={eps_str} "
                  f"elapsed={elapsed:.1f}s")
            print(
                f"[ep {episode+1}/{args.episodes}] avg_trip={_avg(ep_trip):.4f}h "
                f"avg_queue={_avg(ep_queue):.4f}h "
                f"avg_fee={_avg(ep_fee):.4f} "
                f"avg_reward={_avg(ep_reward):.4f}"
            )
            if args.constraint_mode != "off":
                print(
                    f"[ep {episode+1}/{args.episodes}] "
                    f"lambda_lf={constraint_stats['lambda_lf']:.4f} "
                    f"lambda_v={constraint_stats['lambda_v']:.4f} "
                    f"avg_lf={constraint_stats['avg_lf_norm']:.5f}/"
                    f"{constraint_stats['lf_limit']:.5f} "
                    f"avg_vrisk={constraint_stats['avg_voltage_risk_norm']:.5f}/"
                    f"{constraint_stats['voltage_risk_limit']:.5f} "
                    f"peak_load={constraint_stats['peak_power_kw']:.2f}kW "
                    f"loss={constraint_stats['network_loss_kwh']:.4f}kWh "
                    f"vmin={constraint_stats['min_voltage_pu']:.4f}pu"
                )

        agent.decay_epsilon()

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
