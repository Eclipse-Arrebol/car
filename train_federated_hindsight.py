"""Federated hindsight training entrypoint.

This is a minimal bridge from the single-client hindsight pipeline to a
client-specific EMA/IEEE33 federated setup.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from trainer.federated_hindsight_trainer import (
    FederatedClientConfig,
    FederatedHindsightTrainer,
)


if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


DEFAULT_GRAPHML = os.path.join("map_outputs", "ema", "ema.graphml")
DEFAULT_CACHE_DIR = os.path.join("map_outputs", "ema_cache")
DEFAULT_CLIENTS = ["old_city", "new_city", "suburb"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--steps-per-episode", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-evs", type=int, default=80)
    p.add_argument("--num-stations", type=int, default=4)
    p.add_argument("--num-chargers-per-station", type=int, default=8)
    p.add_argument("--respawn", action="store_true", default=True)
    p.add_argument("--no-respawn", dest="respawn", action="store_false")
    p.add_argument("--graphml-file", type=str, default=DEFAULT_GRAPHML)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--clients",
        nargs="+",
        default=DEFAULT_CLIENTS,
        choices=DEFAULT_CLIENTS,
        help="Client names to include in federated training",
    )
    p.add_argument("--save-dir", type=str, default="checkpoints_federated_hindsight")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--network", type=str, default="station_only",
                   choices=["original", "lightweight", "station_only"])
    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=True,
                            help="Enable action mask in the policy network (default: on)")
    mask_group.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false",
                            help="Disable action mask in the policy network")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    client_configs = []
    for i, client_name in enumerate(args.clients):
        client_configs.append(
            FederatedClientConfig(
                client_name=client_name,
                graphml_file=args.graphml_file,
                num_stations=args.num_stations,
                num_evs=args.num_evs,
                num_chargers_per_station=args.num_chargers_per_station,
                respawn_after_full_charge=args.respawn,
                seed=args.seed + i,
                cache_dir=args.cache_dir,
                network_variant=args.network,
                use_action_mask=args.use_action_mask,
                batch_size=args.batch_size,
                steps_per_episode=args.steps_per_episode,
            )
        )

    trainer = FederatedHindsightTrainer(client_configs)
    os.makedirs(args.save_dir, exist_ok=True)

    print(
        f"[setup] rounds={args.rounds} clients={trainer.client_names()} "
        f"num_evs={args.num_evs} num_stations={args.num_stations} "
        f"chargers_per_station={args.num_chargers_per_station} respawn={args.respawn} "
        f"network={args.network} use_action_mask={args.use_action_mask}"
    )
    print(f"[setup] graphml={args.graphml_file} cache_dir={args.cache_dir}")

    t0 = time.time()
    for round_idx in range(args.rounds):
        metrics = trainer.train_round()
        elapsed = time.time() - t0
        print(f"[round {round_idx + 1}/{args.rounds}] elapsed={elapsed:.1f}s")
        for client_name, stat in metrics.items():
            print(
                f"  - {client_name}: memory={stat['memory_size']:.0f} "
                f"avg_trip={stat['avg_trip_h']:.4f}h avg_queue={stat['avg_queue_h']:.4f}h "
                f"avg_fee={stat['avg_fee']:.4f} avg_reward={stat['avg_reward']:.4f}"
            )
        if (round_idx + 1) % args.save_every == 0:
            trainer.save_global_models(args.save_dir, round_idx=round_idx + 1)

    trainer.save_global_models(args.save_dir, round_idx=None)
    print(f"[done] total elapsed = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
