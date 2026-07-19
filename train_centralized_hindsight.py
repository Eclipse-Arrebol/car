"""Centralized hindsight training (privacy-broken upper-bound reference).

Same recipe as train_federated_hindsight.py EXCEPT there is ONE shared agent and
ONE shared replay buffer; the three city environments are visited round-robin and
all their transitions train the same agent. No FedAvg, no parameter aggregation.

Budget is aligned with the federated run:
    rounds x clients x local_episodes x steps  (default 50 x 3 x 2 x 144 = 43,200)
i.e. identical total env interactions and identical per-city exposure, so the only
difference vs federated is "data pooled into one model" vs "FedAvg of local models".

Network / reward / hyper-params are unchanged. epsilon is decayed twice per round
to track the federated global-epsilon trajectory (~2 decays/round).
"""

import argparse
import os
import sys
import time

import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from agents.hindsight_dqn_agent import HindsightDQNAgent
from env.grid_variants import ALL_GRID_VARIANTS
from env.real_env import RealTrafficEnv
from trainer.trainer import HindsightTrainer, compute_hindsight_reward

DEFAULT_CLIENT_SPECS = "old_city:1.3,new_city:1.0,suburb:0.7"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-episodes", type=int, default=2)
    p.add_argument("--steps-per-episode", type=int, default=144)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-steps-per-step", type=int, default=1)

    p.add_argument("--num-evs", type=int, default=40)
    p.add_argument("--num-stations", type=int, default=4)
    p.add_argument("--num-chargers-per-station", type=int, default=8)
    p.add_argument("--respawn", action="store_true", default=True)
    p.add_argument("--no-respawn", dest="respawn", action="store_false")

    p.add_argument("--client-specs", type=str, default=DEFAULT_CLIENT_SPECS)
    p.add_argument("--graphml-file", type=str, default=os.path.join("map_outputs", "ema", "ema.graphml"))
    p.add_argument("--cache-dir", type=str, default=os.path.join("map_outputs", "ema_cache"))
    p.add_argument("--no-ue-background", action="store_true", default=False)
    p.add_argument("--ue-net-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_net.tntp"))
    p.add_argument("--ue-trips-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"))
    p.add_argument("--ue-max-iter", type=int, default=800)
    p.add_argument("--ue-tol", type=float, default=1e-4)
    p.add_argument("--ue-verbose", action="store_true", default=False)

    p.add_argument("--network", choices=["original", "lightweight", "station_only", "station_attn"], default="station_only")
    p.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=False)
    p.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")

    p.add_argument("--load-model", type=str, default=None)
    p.add_argument("--epsilon", type=float, default=0.4)
    p.add_argument("--epsilon-decay", type=float, default=0.994,
                   help="Per-decay epsilon multiplier (decays ~2x/round here; "
                        "lower it for from-scratch runs to anneal within budget)")
    p.add_argument("--save-dir", type=str, default="checkpoints_centralized")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_client_specs(spec_text):
    specs = []
    for raw in spec_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            variant, scale_text = raw.split(":", 1)
            ue_scale = float(scale_text)
        else:
            variant, ue_scale = raw, 1.0
        variant = variant.strip()
        if variant not in ALL_GRID_VARIANTS:
            raise ValueError(f"Unknown grid variant in --client-specs: {variant!r}")
        specs.append({"grid_variant": variant, "ue_scale": float(ue_scale)})
    if not specs:
        raise ValueError("--client-specs produced no clients")
    return specs


def build_env(args, spec):
    env_kw = dict(
        graphml_file=args.graphml_file,
        num_stations=args.num_stations,
        num_evs=args.num_evs,
        num_chargers_per_station=args.num_chargers_per_station,
        max_nodes=1_000_000,
        cache_dir=args.cache_dir,
        seed=args.seed,
        respawn_after_full_charge=args.respawn,
        grid_variant=spec["grid_variant"],
    )
    if not args.no_ue_background and os.path.isfile(args.ue_net_tntp) and os.path.isfile(args.ue_trips_tntp):
        env_kw["background_ue_net_tntp"] = os.path.abspath(args.ue_net_tntp)
        env_kw["background_ue_trips_tntp"] = os.path.abspath(args.ue_trips_tntp)
        env_kw["background_ue_max_iter"] = int(args.ue_max_iter)
        env_kw["background_ue_tol"] = float(args.ue_tol)
        env_kw["background_ue_scale"] = float(spec["ue_scale"])
        env_kw["background_ue_verbose"] = bool(args.ue_verbose)
    return RealTrafficEnv(**env_kw)


def average(values):
    return sum(values) / len(values) if values else 0.0


def run_city_episodes(args, trainer, agent, episodes):
    """Run `episodes` episodes on one city env, training the shared agent."""
    env = trainer.env
    trips, queues, fees, rewards = [], [], [], []
    abandoned = 0
    for _ in range(episodes):
        env.reset()
        trainer.pending.clear()
        trainer._current_step = 0
        for _ in range(args.steps_per_episode):
            done, info = trainer.step_episode()
            for entry in info.get("charge_started", []):
                trip = float(entry.get("actual_trip_time_h", 0.0))
                queue = float(entry.get("actual_queue_time_h", 0.0))
                fee = float(entry.get("charging_fee", 0.0))
                trips.append(trip)
                queues.append(queue)
                fees.append(fee)
                rewards.append(compute_hindsight_reward(trip, queue, fee))
            abandoned += len(info.get("abandoned", []))
            if len(agent.memory) >= args.batch_size:
                for _ in range(max(1, int(args.replay_steps_per_step))):
                    agent.replay(args.batch_size)
            if done:
                break
    return trips, queues, fees, rewards, abandoned


def main():
    args = parse_args()
    specs = parse_client_specs(args.client_specs)
    os.makedirs(args.save_dir, exist_ok=True)

    print(
        f"[setup-centralized] rounds={args.rounds} local_episodes={args.local_episodes} "
        f"steps={args.steps_per_episode} batch={args.batch_size} num_evs={args.num_evs} "
        f"stations={args.num_stations} chargers_per_station={args.num_chargers_per_station} "
        f"network={args.network} use_action_mask={args.use_action_mask} seed={args.seed}"
    )
    print(f"[setup-centralized] client_specs={specs}")
    total_steps = args.rounds * len(specs) * args.local_episodes * args.steps_per_episode
    print(f"[setup-centralized] total env steps budget = {total_steps}")

    # ONE shared agent, built on the first env's geometry.
    envs = [build_env(args, spec) for spec in specs]
    station_node_ids = [s.traffic_node_id for s in envs[0].stations]
    agent = HindsightDQNAgent(
        num_features=19,
        num_actions=args.num_stations,
        station_node_ids=station_node_ids,
        num_nodes_per_graph=envs[0].num_nodes,
        network_variant=args.network,
        use_action_mask=args.use_action_mask,
        epsilon_decay=args.epsilon_decay,
    )
    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise FileNotFoundError(f"--load-model not found: {args.load_model}")
        agent.load_model(args.load_model)
        print(f"[setup-centralized] warm-start: {args.load_model}")
    agent.epsilon = float(args.epsilon)

    # One trainer per env, all sharing the single agent.
    trainers = [HindsightTrainer(env, agent) for env in envs]
    for spec, env in zip(specs, envs):
        print(
            f"[setup-centralized] city={spec['grid_variant']} ue_scale={spec['ue_scale']} "
            f"station_node_ids={[s.traffic_node_id for s in env.stations]}"
        )

    t0 = time.time()
    for round_idx in range(1, args.rounds + 1):
        r_trip, r_queue, r_fee, r_reward = [], [], [], []
        r_abandoned = 0
        for spec, trainer in zip(specs, trainers):
            trips, queues, fees, rewards, abandoned = run_city_episodes(
                args, trainer, agent, args.local_episodes
            )
            r_trip.extend(trips)
            r_queue.extend(queues)
            r_fee.extend(fees)
            r_reward.extend(rewards)
            r_abandoned += abandoned

        # ~2 decays/round to track federated global-epsilon trajectory.
        agent.decay_epsilon()
        agent.decay_epsilon()

        print(
            f"[c-round {round_idx}/{args.rounds}] epsilon={agent.epsilon:.3f} "
            f"buffer={len(agent.memory)} abandoned={r_abandoned} "
            f"avg_trip={average(r_trip):.4f}h avg_queue={average(r_queue):.4f}h "
            f"avg_fee={average(r_fee):.4f} avg_reward={average(r_reward):.4f} "
            f"elapsed={time.time() - t0:.1f}s"
        )

        if round_idx % args.save_every == 0:
            path = os.path.join(args.save_dir, f"central_round{round_idx}.pth")
            agent.save_model(path)
            print(f"  [save] {path}")

    final_path = os.path.join(args.save_dir, "central_final.pth")
    agent.save_model(final_path)
    print(f"[done] centralized final -> {final_path}")
    print(f"[done] total elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
