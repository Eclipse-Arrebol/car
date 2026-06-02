"""Evaluate federated hindsight checkpoints on the three client grids.

This script is a companion to ``train_federated_hindsight.py`` and is meant
for paper-facing evaluation:
- same episode seeds across policies for fair paired comparison
- strict checkpoint loading by default
- per-episode CSV, aggregate CSV, and JSON metadata outputs
- station choice diagnostics for bias/queue-pressure checks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from agents.hindsight_dqn_agent import HindsightDQNAgent
from env.real_env import RealTrafficEnv
from train_federated_hindsight import DEFAULT_CLIENTS, DEFAULT_CACHE_DIR, DEFAULT_GRAPHML


POLICY_NAMES = ("random", "shortest_path", "model_greedy")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate federated hindsight checkpoints")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps-per-episode", type=int, default=100)
    p.add_argument("--num-evs", type=int, default=80)
    p.add_argument("--num-stations", type=int, default=4)
    p.add_argument("--num-chargers-per-station", type=int, default=8)
    p.add_argument("--graphml-file", type=str, default=DEFAULT_GRAPHML)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--clients", nargs="+", default=DEFAULT_CLIENTS, choices=DEFAULT_CLIENTS)
    p.add_argument(
        "--policies",
        nargs="+",
        default=list(POLICY_NAMES),
        choices=POLICY_NAMES,
        help="Policies to evaluate.",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_federated_hindsight",
        help="Directory containing <client>_final.pth or <client>_roundN.pth checkpoints.",
    )
    p.add_argument(
        "--checkpoint-suffix",
        type=str,
        default="final",
        help="Checkpoint suffix, e.g. final or round100.",
    )
    p.add_argument(
        "--allow-missing-checkpoint",
        action="store_true",
        default=False,
        help="Evaluate model_greedy with random initialized weights if a checkpoint is missing.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for summary.csv, episodes.csv, and metadata.json.",
    )
    p.add_argument(
        "--network",
        type=str,
        default="station_only",
        choices=["original", "lightweight", "station_only"],
    )
    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=True)
    mask_group.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")
    p.add_argument("--respawn", action="store_true", default=True)
    p.add_argument("--no-respawn", dest="respawn", action="store_false")
    p.add_argument("--no-ue-background", action="store_true", default=False)
    p.add_argument("--ue-net-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_net.tntp"))
    p.add_argument("--ue-trips-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"))
    p.add_argument("--ue-max-iter", type=int, default=800)
    p.add_argument("--ue-tol", type=float, default=1e-4)
    p.add_argument("--ue-scale", type=float, default=1.0)
    p.add_argument("--ue-verbose", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _env_kwargs(args, client_name: str, seed: int | None = None) -> dict:
    kw = dict(
        graphml_file=args.graphml_file,
        num_stations=args.num_stations,
        num_evs=args.num_evs,
        num_chargers_per_station=args.num_chargers_per_station,
        max_nodes=1_000_000,
        cache_dir=args.cache_dir,
        seed=args.seed if seed is None else int(seed),
        respawn_after_full_charge=args.respawn,
        client_name=client_name,
    )
    if not args.no_ue_background:
        net_p = args.ue_net_tntp
        trip_p = args.ue_trips_tntp
        if os.path.isfile(net_p) and os.path.isfile(trip_p):
            kw["background_ue_net_tntp"] = os.path.abspath(net_p)
            kw["background_ue_trips_tntp"] = os.path.abspath(trip_p)
            kw["background_ue_max_iter"] = int(args.ue_max_iter)
            kw["background_ue_tol"] = float(args.ue_tol)
            kw["background_ue_scale"] = float(args.ue_scale)
            kw["background_ue_verbose"] = bool(args.ue_verbose)
        else:
            print(
                f"[setup] warn: UE TNTP not found (net={os.path.isfile(net_p)}, "
                f"trips={os.path.isfile(trip_p)}), using heuristic background_edge_base_flows"
            )
    return kw


def _make_env(args, client_name: str, seed: int | None = None) -> RealTrafficEnv:
    _set_seed(args.seed if seed is None else int(seed))
    return RealTrafficEnv(**_env_kwargs(args, client_name, seed=seed))


def _avg(values):
    return mean(values) if values else 0.0


def _stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    avg = mean(values) if values else 0.0
    sd = stdev(values) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": avg,
        "std": sd,
        "sem": sem,
        "ci95": 1.96 * sem,
    }


def _episode_seed(args, episode: int) -> int:
    return int(args.seed) + episode


def _resolve_output_dir(args) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.checkpoint_dir) / f"eval_{args.checkpoint_suffix}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_model(args, env: RealTrafficEnv, checkpoint_path: str):
    station_node_ids = [s.traffic_node_id for s in env.stations]
    model = HindsightDQNAgent(
        num_features=18,
        num_actions=len(env.stations),
        station_node_ids=station_node_ids,
        num_nodes_per_graph=env.num_nodes,
        network_variant=args.network,
        use_action_mask=args.use_action_mask,
    )
    if os.path.isfile(checkpoint_path):
        model.load_model(checkpoint_path)
    elif not args.allow_missing_checkpoint:
        raise FileNotFoundError(
            f"Checkpoint not found for model_greedy: {checkpoint_path}. "
            "Pass --allow-missing-checkpoint only for debugging."
        )
    else:
        print(f"[setup] checkpoint not found, using random init: {checkpoint_path}")
    model.policy_net.eval()
    model.target_net.eval()
    return model


def _completed_station_id(env: RealTrafficEnv, entry: dict) -> int | None:
    if "station_id" in entry:
        return int(entry["station_id"])
    ev_id = entry.get("ev_id")
    for ev in env.evs:
        if ev.id == ev_id:
            station_id = getattr(ev, "charge_station_id", None)
            return int(station_id) if station_id is not None else None
    return None


def _valid_actions(action_mask, num_stations: int) -> list[int]:
    valid = action_mask.squeeze().nonzero(as_tuple=True)[0].tolist()
    return [int(v) for v in valid] if valid else list(range(num_stations))


def _run_policy(args, client_name: str, policy_name: str, policy_fn):
    env = _make_env(args, client_name, seed=args.seed)
    choice_counter = Counter()
    station_completed = Counter()
    station_trip = defaultdict(list)
    station_queue = defaultdict(list)
    station_fee = defaultdict(list)
    all_trip, all_queue, all_fee, all_reward = [], [], [], []
    all_step_power, all_step_loss, all_step_min_voltage = [], [], []
    episode_rows: list[dict] = []
    t0 = time.time()

    print(
        f"\n[client={env.power_grid.client_name}] policy={policy_name} "
        f"episodes={args.episodes} steps={args.steps_per_episode} "
        f"num_evs={args.num_evs} use_action_mask={args.use_action_mask}"
    )
    print(f"[client={env.power_grid.client_name}] station_nodes={[s.traffic_node_id for s in env.stations]}")

    for episode in range(args.episodes):
        seed = _episode_seed(args, episode)
        _set_seed(seed)
        env.reset()
        ep_trip, ep_queue, ep_fee, ep_reward = [], [], [], []
        ep_power, ep_loss, ep_min_voltage = [], [], []
        steps_run = 0
        prev_power = 0.0
        load_fluctuation = 0.0

        for _ in range(args.steps_per_episode):
            urgent_evs = env.get_pending_decision_evs()
            actions = {}
            for ev in urgent_evs:
                state = env.get_graph_state_for_ev(ev)
                action_mask = env.get_action_mask(ev)
                action = policy_fn(env, ev, state, action_mask)
                actions[ev.id] = int(action)
                choice_counter[int(action)] += 1

            _obs, _reward, done, info = env.step(actions)
            steps_run += 1
            power_kw = float(info.get("realized_power", 0.0))
            loss_kw = float(info.get("line_losses", 0.0))
            min_voltage_pu = float(info.get("min_voltage_pu", 1.0))
            load_fluctuation += (power_kw - prev_power) ** 2
            prev_power = power_kw
            ep_power.append(power_kw)
            ep_loss.append(loss_kw)
            ep_min_voltage.append(min_voltage_pu)

            for entry in info.get("completed", []):
                station_id = _completed_station_id(env, entry)
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                reward = -(0.3 * trip + 0.5 * queue + 0.03 * fee)
                ep_trip.append(trip)
                ep_queue.append(queue)
                ep_fee.append(fee)
                ep_reward.append(reward)
                if station_id is not None:
                    station_completed[station_id] += 1
                    station_trip[station_id].append(trip)
                    station_queue[station_id].append(queue)
                    station_fee[station_id].append(fee)

            if done:
                break

        all_trip.extend(ep_trip)
        all_queue.extend(ep_queue)
        all_fee.extend(ep_fee)
        all_reward.extend(ep_reward)
        all_step_power.extend(ep_power)
        all_step_loss.extend(ep_loss)
        all_step_min_voltage.extend(ep_min_voltage)
        step_duration_h = float(getattr(env, "step_duration_h", 1.0 / 6.0))
        ep_row = {
            "client": client_name,
            "policy": policy_name,
            "episode": episode + 1,
            "seed": seed,
            "steps": steps_run,
            "completed": len(ep_trip),
            "avg_trip_h": _avg(ep_trip),
            "avg_queue_h": _avg(ep_queue),
            "avg_fee": _avg(ep_fee),
            "avg_reward": _avg(ep_reward),
            "avg_weighted_cost": -_avg(ep_reward),
            "peak_load_kw": max(ep_power) if ep_power else 0.0,
            "load_fluctuation_kw2": load_fluctuation,
            "avg_line_loss_kw": _avg(ep_loss),
            "network_loss_kwh": sum(ep_loss) * step_duration_h,
            "min_voltage_pu": min(ep_min_voltage) if ep_min_voltage else 1.0,
        }
        episode_rows.append(ep_row)
        print(
            f"[client={env.power_grid.client_name}] ep {episode + 1}/{args.episodes} "
            f"seed={seed} steps={steps_run} completed={len(ep_trip)} "
            f"avg_trip={ep_row['avg_trip_h']:.4f}h avg_queue={ep_row['avg_queue_h']:.4f}h "
            f"avg_fee={ep_row['avg_fee']:.4f} avg_cost={ep_row['avg_weighted_cost']:.4f} "
            f"peak_load={ep_row['peak_load_kw']:.2f}kW "
            f"loss={ep_row['network_loss_kwh']:.4f}kWh "
            f"vmin={ep_row['min_voltage_pu']:.4f}pu"
        )

    elapsed = time.time() - t0
    print(f"\n[client={env.power_grid.client_name}] policy={policy_name} choice_count={dict(choice_counter)}")
    for sid in sorted(set(choice_counter.keys()) | set(station_completed.keys())):
        print(
            f"[client={env.power_grid.client_name}] station={sid} "
            f"choices={choice_counter.get(sid, 0)} completed={station_completed.get(sid, 0)} "
            f"avg_trip={_avg(station_trip[sid]):.4f}h "
            f"avg_queue={_avg(station_queue[sid]):.4f}h "
            f"avg_fee={_avg(station_fee[sid]):.4f}"
        )

    completed_per_episode = [int(row["completed"]) for row in episode_rows]
    ep_trip_means = [float(row["avg_trip_h"]) for row in episode_rows if int(row["completed"]) > 0]
    ep_queue_means = [float(row["avg_queue_h"]) for row in episode_rows if int(row["completed"]) > 0]
    ep_fee_means = [float(row["avg_fee"]) for row in episode_rows if int(row["completed"]) > 0]
    ep_cost_means = [float(row["avg_weighted_cost"]) for row in episode_rows if int(row["completed"]) > 0]
    ep_peak_loads = [float(row["peak_load_kw"]) for row in episode_rows]
    ep_load_fluctuations = [float(row["load_fluctuation_kw2"]) for row in episode_rows]
    ep_network_losses = [float(row["network_loss_kwh"]) for row in episode_rows]
    ep_min_voltages = [float(row["min_voltage_pu"]) for row in episode_rows]

    trip_stats = _stats(ep_trip_means)
    queue_stats = _stats(ep_queue_means)
    fee_stats = _stats(ep_fee_means)
    cost_stats = _stats(ep_cost_means)
    completed_stats = _stats([float(v) for v in completed_per_episode])
    peak_load_stats = _stats(ep_peak_loads)
    load_fluctuation_stats = _stats(ep_load_fluctuations)
    network_loss_stats = _stats(ep_network_losses)
    min_voltage_stats = _stats(ep_min_voltages)

    summary = {
        "client": client_name,
        "policy": policy_name,
        "episodes": args.episodes,
        "steps_per_episode": args.steps_per_episode,
        "elapsed_s": elapsed,
        "completed_total": len(all_trip),
        "completed_per_episode_mean": completed_stats["mean"],
        "completed_per_episode_std": completed_stats["std"],
        "session_avg_trip_h": _avg(all_trip),
        "session_avg_queue_h": _avg(all_queue),
        "session_avg_fee": _avg(all_fee),
        "session_avg_reward": _avg(all_reward),
        "session_avg_weighted_cost": -_avg(all_reward),
        "session_peak_load_kw": max(all_step_power) if all_step_power else 0.0,
        "session_avg_line_loss_kw": _avg(all_step_loss),
        "session_min_voltage_pu": min(all_step_min_voltage) if all_step_min_voltage else 1.0,
        "episode_avg_trip_h_mean": trip_stats["mean"],
        "episode_avg_trip_h_std": trip_stats["std"],
        "episode_avg_trip_h_sem": trip_stats["sem"],
        "episode_avg_trip_h_ci95": trip_stats["ci95"],
        "episode_avg_queue_h_mean": queue_stats["mean"],
        "episode_avg_queue_h_std": queue_stats["std"],
        "episode_avg_queue_h_sem": queue_stats["sem"],
        "episode_avg_queue_h_ci95": queue_stats["ci95"],
        "episode_avg_fee_mean": fee_stats["mean"],
        "episode_avg_fee_std": fee_stats["std"],
        "episode_avg_fee_sem": fee_stats["sem"],
        "episode_avg_fee_ci95": fee_stats["ci95"],
        "episode_avg_weighted_cost_mean": cost_stats["mean"],
        "episode_avg_weighted_cost_std": cost_stats["std"],
        "episode_avg_weighted_cost_sem": cost_stats["sem"],
        "episode_avg_weighted_cost_ci95": cost_stats["ci95"],
        "episode_peak_load_kw_mean": peak_load_stats["mean"],
        "episode_peak_load_kw_std": peak_load_stats["std"],
        "episode_peak_load_kw_sem": peak_load_stats["sem"],
        "episode_peak_load_kw_ci95": peak_load_stats["ci95"],
        "episode_load_fluctuation_kw2_mean": load_fluctuation_stats["mean"],
        "episode_load_fluctuation_kw2_std": load_fluctuation_stats["std"],
        "episode_load_fluctuation_kw2_sem": load_fluctuation_stats["sem"],
        "episode_load_fluctuation_kw2_ci95": load_fluctuation_stats["ci95"],
        "episode_network_loss_kwh_mean": network_loss_stats["mean"],
        "episode_network_loss_kwh_std": network_loss_stats["std"],
        "episode_network_loss_kwh_sem": network_loss_stats["sem"],
        "episode_network_loss_kwh_ci95": network_loss_stats["ci95"],
        "episode_min_voltage_pu_mean": min_voltage_stats["mean"],
        "episode_min_voltage_pu_std": min_voltage_stats["std"],
        "episode_min_voltage_pu_sem": min_voltage_stats["sem"],
        "episode_min_voltage_pu_ci95": min_voltage_stats["ci95"],
        "choice_count": dict(choice_counter),
        "completed_by_station": dict(station_completed),
        "station_metrics": {
            str(sid): {
                "choices": int(choice_counter.get(sid, 0)),
                "completed": int(station_completed.get(sid, 0)),
                "avg_trip_h": _avg(station_trip[sid]),
                "avg_queue_h": _avg(station_queue[sid]),
                "avg_fee": _avg(station_fee[sid]),
            }
            for sid in sorted(set(choice_counter.keys()) | set(station_completed.keys()))
        },
    }
    print(
        f"[client={env.power_grid.client_name}] policy={policy_name} summary "
        f"completed={summary['completed_total']} "
        f"trip={summary['episode_avg_trip_h_mean']:.4f}+/-{summary['episode_avg_trip_h_ci95']:.4f}h "
        f"queue={summary['episode_avg_queue_h_mean']:.4f}+/-{summary['episode_avg_queue_h_ci95']:.4f}h "
        f"fee={summary['episode_avg_fee_mean']:.4f}+/-{summary['episode_avg_fee_ci95']:.4f} "
        f"cost={summary['episode_avg_weighted_cost_mean']:.4f}+/-{summary['episode_avg_weighted_cost_ci95']:.4f} "
        f"peak_load={summary['episode_peak_load_kw_mean']:.2f}+/-{summary['episode_peak_load_kw_ci95']:.2f}kW "
        f"loss={summary['episode_network_loss_kwh_mean']:.4f}+/-{summary['episode_network_loss_kwh_ci95']:.4f}kWh "
        f"vmin={summary['episode_min_voltage_pu_mean']:.4f}+/-{summary['episode_min_voltage_pu_ci95']:.4f}pu "
        f"elapsed={summary['elapsed_s']:.1f}s"
    )
    return summary, episode_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_outputs(args, out_dir: Path, summaries: list[dict], episode_rows: list[dict]) -> None:
    summary_fields = [
        "client",
        "policy",
        "episodes",
        "steps_per_episode",
        "elapsed_s",
        "completed_total",
        "completed_per_episode_mean",
        "completed_per_episode_std",
        "session_avg_trip_h",
        "session_avg_queue_h",
        "session_avg_fee",
        "session_avg_weighted_cost",
        "session_peak_load_kw",
        "session_avg_line_loss_kw",
        "session_min_voltage_pu",
        "episode_avg_trip_h_mean",
        "episode_avg_trip_h_std",
        "episode_avg_trip_h_sem",
        "episode_avg_trip_h_ci95",
        "episode_avg_queue_h_mean",
        "episode_avg_queue_h_std",
        "episode_avg_queue_h_sem",
        "episode_avg_queue_h_ci95",
        "episode_avg_fee_mean",
        "episode_avg_fee_std",
        "episode_avg_fee_sem",
        "episode_avg_fee_ci95",
        "episode_avg_weighted_cost_mean",
        "episode_avg_weighted_cost_std",
        "episode_avg_weighted_cost_sem",
        "episode_avg_weighted_cost_ci95",
        "episode_peak_load_kw_mean",
        "episode_peak_load_kw_std",
        "episode_peak_load_kw_sem",
        "episode_peak_load_kw_ci95",
        "episode_load_fluctuation_kw2_mean",
        "episode_load_fluctuation_kw2_std",
        "episode_load_fluctuation_kw2_sem",
        "episode_load_fluctuation_kw2_ci95",
        "episode_network_loss_kwh_mean",
        "episode_network_loss_kwh_std",
        "episode_network_loss_kwh_sem",
        "episode_network_loss_kwh_ci95",
        "episode_min_voltage_pu_mean",
        "episode_min_voltage_pu_std",
        "episode_min_voltage_pu_sem",
        "episode_min_voltage_pu_ci95",
    ]
    episode_fields = [
        "client",
        "policy",
        "episode",
        "seed",
        "steps",
        "completed",
        "avg_trip_h",
        "avg_queue_h",
        "avg_fee",
        "avg_reward",
        "avg_weighted_cost",
        "peak_load_kw",
        "load_fluctuation_kw2",
        "avg_line_loss_kw",
        "network_loss_kwh",
        "min_voltage_pu",
    ]
    _write_csv(out_dir / "summary.csv", summaries, summary_fields)
    _write_csv(out_dir / "episodes.csv", episode_rows, episode_fields)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "summaries": summaries,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    out_dir = _resolve_output_dir(args)
    _set_seed(args.seed)
    print(
        f"[setup] checkpoint_dir={args.checkpoint_dir} checkpoint_suffix={args.checkpoint_suffix} "
        f"clients={args.clients} policies={args.policies} network={args.network} "
        f"use_action_mask={args.use_action_mask} output_dir={out_dir}"
    )
    if not args.no_ue_background:
        print(
            f"[setup] UE background requested: net={args.ue_net_tntp} trips={args.ue_trips_tntp} "
            f"max_iter={args.ue_max_iter} tol={args.ue_tol} scale={args.ue_scale}"
        )
    else:
        print("[setup] heuristic background (--no-ue-background)")

    summaries: list[dict] = []
    episode_rows: list[dict] = []

    for client_name in args.clients:
        probe_env = _make_env(args, client_name, seed=args.seed)
        checkpoint = os.path.join(args.checkpoint_dir, f"{client_name}_{args.checkpoint_suffix}.pth")
        model = None
        if "model_greedy" in args.policies:
            model = _load_model(args, probe_env, checkpoint)
            if os.path.isfile(checkpoint):
                print(f"[setup] loaded checkpoint for {client_name}: {checkpoint}")

        def random_policy(env, _ev, _state, action_mask):
            return int(random.choice(_valid_actions(action_mask, len(env.stations))))

        def shortest_path_policy(env_, ev, _state, action_mask):
            best_action = None
            best_time = None
            for station in env_.stations:
                if station.id not in _valid_actions(action_mask, len(env_.stations)):
                    continue
                metrics = env_.estimate_action_metrics(ev, station.id)
                trip_time = metrics.get("trip_time_h", float("inf"))
                if best_time is None or trip_time < best_time:
                    best_time = trip_time
                    best_action = station.id
            return int(best_action if best_action is not None else 0)

        def model_greedy_policy(_env, _ev, state, action_mask):
            assert model is not None
            with torch.no_grad():
                q_values = model.policy_net(
                    state.to(model.device),
                    action_mask=action_mask.to(model.device) if args.use_action_mask else None,
                    action_type="t0",
                )
                return int(q_values.argmax().item())

        policy_map = {
            "random": random_policy,
            "shortest_path": shortest_path_policy,
            "model_greedy": model_greedy_policy,
        }
        for policy_name in args.policies:
            summary, rows = _run_policy(args, client_name, policy_name, policy_map[policy_name])
            summaries.append(summary)
            episode_rows.extend(rows)
            _write_outputs(args, out_dir, summaries, episode_rows)

    print("\n" + "=" * 80)
    print("Evaluation comparison")
    print("=" * 80)
    for s in summaries:
        print(
            f"[{s['client']} / {s['policy']}] "
            f"trip={s['episode_avg_trip_h_mean']:.4f}+/-{s['episode_avg_trip_h_ci95']:.4f}h "
            f"queue={s['episode_avg_queue_h_mean']:.4f}+/-{s['episode_avg_queue_h_ci95']:.4f}h "
            f"fee={s['episode_avg_fee_mean']:.4f}+/-{s['episode_avg_fee_ci95']:.4f} "
            f"cost={s['episode_avg_weighted_cost_mean']:.4f}+/-{s['episode_avg_weighted_cost_ci95']:.4f} "
            f"peak_load={s['episode_peak_load_kw_mean']:.2f}+/-{s['episode_peak_load_kw_ci95']:.2f}kW "
            f"LF={s['episode_load_fluctuation_kw2_mean']:.2f}+/-{s['episode_load_fluctuation_kw2_ci95']:.2f} "
            f"NL={s['episode_network_loss_kwh_mean']:.4f}+/-{s['episode_network_loss_kwh_ci95']:.4f}kWh "
            f"Vmin={s['episode_min_voltage_pu_mean']:.4f}+/-{s['episode_min_voltage_pu_ci95']:.4f}pu "
            f"completed={s['completed_total']} elapsed={s['elapsed_s']:.1f}s"
        )
    print(f"[done] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
