"""Personalized FL comparison under the submitted-aware C-state.

Run from the project root, e.g. G:\\car_charge\\car_charge.

Methods:
  Local       : one model per city, no sharing.
  Centralized : one shared model trained on all city data.
  FedAvg      : full-model aggregation.
  FedProx     : full-model aggregation with proximal local update.
  FedSetRL    : FedRep-style shared encoder + personalized city head.

The environment state fixes the station-only 7 decision features used by the
network and the C-style same-tick submitted heading feature:
  h_s = (n_heading_s + z_s) / max(1, sum_r n_heading_r + sum_r z_r)
without using z_s to correct waiting/service-time features.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import torch


sys.path.insert(0, os.getcwd())

from agents.hindsight_dqn_agent import HindsightDQNAgent
from env.grid_variants import ALL_GRID_VARIANTS
from env.real_env import RealTrafficEnv
from reward_profiles import weights_for
from trainer.trainer import compute_hindsight_reward
from train_federated_hindsight import FedAvgServer, FederatedHindsightAgent


DEFAULT_CLIENT_SPECS = "old_city:1.3,new_city:1.0,suburb:0.7"
DEFAULT_CITY_SCALE = {"old_city": 1.3, "new_city": 1.0, "suburb": 0.7}


if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


class SubmittedAwareCEnv(RealTrafficEnv):
    """C-state: submitted-aware heading feature, no z correction to wait/service."""

    def get_graph_state_for_ev(self, ev, pending_counts=None):
        data = self.get_graph_state()
        data.x[ev.curr_node, 8] = ev.soc / 100.0

        pending_total = sum(pending_counts.values()) if pending_counts else 0
        heading_total = sum(self.evs_heading_to.values())
        denom = max(1, heading_total + pending_total)

        for station in self.stations:
            metrics = self._estimate_ev_station_metrics(ev, station, pending_counts=None)
            node_idx = station.traffic_node_id
            data.x[node_idx, 9] = 1.0 / (1.0 + metrics["trip_time_h"])
            data.x[node_idx, 10] = metrics["trip_time_h"]
            data.x[node_idx, 11] = metrics["service_time_h"]
            data.x[node_idx, 12] = metrics["generalized_cost"] / 100.0
            data.x[node_idx, 15] = min(
                2.0,
                metrics["queue_time_h"] / max(1e-6, station.max_wait_time_h),
            )
            data.x[node_idx, 16] = (
                len(station.queue) + len(station.connected_evs) + station.predicted_arrivals
            ) / max(1.0, station.max_queue_len + station.num_chargers)
            data.x[node_idx, 17] = float(
                max(0, station.num_chargers - len(station.connected_evs))
                / station.num_chargers
            )
            pending = 0 if pending_counts is None else pending_counts.get(station.id, 0)
            data.x[node_idx, 18] = (
                self.evs_heading_to.get(station.id, 0) + pending
            ) / denom

        return data


def mean(values):
    return sum(values) / len(values) if values else 0.0


def variance(values):
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / len(values)


def quantile(values, q):
    if not values:
        return 0.0
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def cvar_upper(values, q):
    if not values:
        return 0.0
    threshold = quantile(values, q)
    return mean([v for v in values if v >= threshold])


def gini(values):
    vals = [max(0.0, float(v)) for v in values]
    n = len(vals)
    total = sum(vals)
    if n == 0 or total <= 1e-12:
        return 0.0
    vals.sort()
    weighted = sum(i * value for i, value in enumerate(vals, start=1))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def parse_client_specs(text):
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        city, scale = raw.split(":", 1) if ":" in raw else (raw, "1.0")
        city = city.strip()
        if city not in ALL_GRID_VARIANTS:
            raise ValueError(f"Unknown city/grid variant: {city}")
        out.append({"city": city, "ue_scale": float(scale)})
    if not out:
        raise ValueError("--client-specs produced no cities")
    return out


def build_env(args, city, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    env_kw = dict(
        graphml_file=args.graphml_file,
        num_stations=args.num_stations,
        num_evs=args.num_evs,
        num_chargers_per_station=args.num_chargers_per_station,
        max_nodes=1_000_000,
        cache_dir=args.cache_dir,
        seed=seed,
        respawn_after_full_charge=args.respawn,
        grid_variant=city["city"],
    )
    if (
        not args.no_ue_background
        and os.path.isfile(args.ue_net_tntp)
        and os.path.isfile(args.ue_trips_tntp)
    ):
        env_kw["background_ue_net_tntp"] = os.path.abspath(args.ue_net_tntp)
        env_kw["background_ue_trips_tntp"] = os.path.abspath(args.ue_trips_tntp)
        env_kw["background_ue_max_iter"] = int(args.ue_max_iter)
        env_kw["background_ue_tol"] = float(args.ue_tol)
        env_kw["background_ue_scale"] = float(city["ue_scale"])
    return SubmittedAwareCEnv(**env_kw)


def build_agent(args, env, federated=False, client_id="local", fedprox_mu=0.0):
    kwargs = dict(
        num_features=19,
        num_actions=args.num_stations,
        station_node_ids=[s.traffic_node_id for s in env.stations],
        num_nodes_per_graph=env.num_nodes,
        network_variant=args.network,
        use_action_mask=args.use_action_mask,
        epsilon_decay=args.epsilon_decay,
    )
    if federated:
        agent = FederatedHindsightAgent(client_id, fedprox_mu=fedprox_mu, **kwargs)
    else:
        agent = HindsightDQNAgent(**kwargs)
    agent.epsilon = float(args.epsilon)
    return agent


def run_episode_train(args, env, agent, city_name):
    env.reset()
    pending = {}
    trips, queues, fees, rewards = [], [], [], []
    abandoned = 0

    for _ in range(args.steps_per_episode):
        urgent_evs = env.get_pending_decision_evs()
        pending_counts = {s.id: 0 for s in env.stations}
        actions = {}

        for ev in urgent_evs:
            state = env.get_graph_state_for_ev(ev, pending_counts=pending_counts)
            action_mask = (
                env.get_action_mask(ev, pending_counts=pending_counts)
                if args.use_action_mask
                else None
            )
            action = int(agent.select_action(state, action_mask=action_mask))
            actions[ev.id] = action
            pending[(ev.id, ev.charge_sessions)] = (state, action, action_mask)
            if 0 <= action < args.num_stations:
                pending_counts[action] = pending_counts.get(action, 0) + 1

        _, _, done, info = env.step(actions)

        for entry in info.get("charge_started", []):
            ev_id = entry["ev_id"]
            session = None
            for ev in env.evs:
                if ev.id == ev_id:
                    session = ev.charge_sessions
                    break
            stored = pending.pop((ev_id, session), None)
            trip = float(entry.get("actual_trip_time_h", 0.0))
            queue = float(entry.get("actual_queue_time_h", 0.0))
            fee = float(entry.get("charging_fee", 0.0))
            reward = compute_hindsight_reward(trip, queue, fee, *weights_for(city_name))
            trips.append(trip)
            queues.append(queue)
            fees.append(fee)
            rewards.append(reward)
            if stored is not None:
                state, action, action_mask = stored
                state_for_buffer = state.clone() if hasattr(state, "clone") else state
                agent.store_transition(
                    state_for_buffer,
                    action,
                    reward,
                    None,
                    action_mask=action_mask,
                    done=True,
                )

        for entry in info.get("abandoned", []):
            ev_id = entry.get("ev_id")
            for key in list(pending.keys()):
                if key[0] == ev_id:
                    pending.pop(key, None)
            abandoned += 1

        if len(agent.memory) >= args.batch_size:
            for _ in range(max(1, int(args.replay_steps_per_step))):
                agent.replay(args.batch_size)
        if done:
            break

    return {"trip": trips, "queue": queues, "fee": fees, "reward": rewards, "abandoned": abandoned}


def train_local(args):
    city = {"city": args.city, "ue_scale": args.ue_scale}
    torch.manual_seed(args.seed)
    env = build_env(args, city, args.seed)
    agent = build_agent(args, env, federated=False)
    os.makedirs(args.save_dir, exist_ok=True)

    print(
        f"[setup-local-c] city={args.city} load={args.num_evs} episodes={args.episodes} "
        f"seed={args.seed} network={args.network} state=C save_dir={args.save_dir}"
    )
    t0 = time.time()
    for ep in range(1, args.episodes + 1):
        m = run_episode_train(args, env, agent, args.city)
        agent.decay_epsilon()
        print(
            f"[local-c ep {ep}/{args.episodes}] city={args.city} epsilon={agent.epsilon:.3f} "
            f"events={len(m['reward'])} abandoned={m['abandoned']} "
            f"avg_queue={mean(m['queue']):.4f}h avg_fee={mean(m['fee']):.2f} "
            f"avg_reward={mean(m['reward']):.4f} elapsed={time.time() - t0:.1f}s"
        )
        if args.save_every > 0 and ep % args.save_every == 0:
            path = os.path.join(args.save_dir, f"model_ep{ep}.pth")
            agent.save_model(path)
            print(f"[save] {path}")

    final_path = os.path.join(args.save_dir, "model_final.pth")
    agent.save_model(final_path)
    print(f"[done] local final -> {final_path}")


def train_centralized(args):
    cities = parse_client_specs(args.client_specs)
    torch.manual_seed(args.seed)
    envs = [build_env(args, city, args.seed) for city in cities]
    agent = build_agent(args, envs[0], federated=False)
    os.makedirs(args.save_dir, exist_ok=True)

    print(
        f"[setup-central-c] rounds={args.rounds} local_episodes={args.local_episodes} "
        f"load={args.num_evs} seed={args.seed} state=C save_dir={args.save_dir}"
    )
    t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        all_trip, all_queue, all_fee, all_reward = [], [], [], []
        abandoned = 0
        for city, env in zip(cities, envs):
            for _ in range(args.local_episodes):
                m = run_episode_train(args, env, agent, city["city"])
                all_trip.extend(m["trip"])
                all_queue.extend(m["queue"])
                all_fee.extend(m["fee"])
                all_reward.extend(m["reward"])
                abandoned += m["abandoned"]
        agent.decay_epsilon()
        print(
            f"[central-c round {rnd}/{args.rounds}] epsilon={agent.epsilon:.3f} "
            f"events={len(all_reward)} abandoned={abandoned} avg_queue={mean(all_queue):.4f}h "
            f"avg_fee={mean(all_fee):.2f} avg_reward={mean(all_reward):.4f} "
            f"elapsed={time.time() - t0:.1f}s"
        )
        if args.save_every > 0 and rnd % args.save_every == 0:
            path = os.path.join(args.save_dir, f"central_round{rnd}.pth")
            agent.save_model(path)
            print(f"[save] {path}")

    final_path = os.path.join(args.save_dir, "central_final.pth")
    agent.save_model(final_path)
    print(f"[done] centralized final -> {final_path}")


def train_federated(args):
    cities = parse_client_specs(args.client_specs)
    method = "fedrep" if args.fed_method == "fedsetrl" else args.fed_method
    torch.manual_seed(args.seed)
    clients = []
    os.makedirs(args.save_dir, exist_ok=True)

    for idx, city in enumerate(cities):
        env = build_env(args, city, args.seed)
        agent = build_agent(
            args,
            env,
            federated=True,
            client_id=idx,
            fedprox_mu=(args.fedprox_mu if args.fed_method == "fedprox" else 0.0),
        )
        clients.append({"city": city, "env": env, "agent": agent})
        print(
            f"[setup-fed-c] client{idx} city={city['city']} ue_scale={city['ue_scale']} "
            f"station_nodes={[s.traffic_node_id for s in env.stations]}"
        )

    global_agent = build_agent(args, clients[0]["env"], federated=True, client_id="global")
    server = FedAvgServer(global_agent, method=method, aggregation_momentum=args.aggregation_momentum)
    print(
        f"[setup-fed-c] method={args.fed_method} internal={method} rounds={args.rounds} "
        f"local_episodes={args.local_episodes} load={args.num_evs} seed={args.seed} state=C"
    )
    t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        server.distribute([c["agent"] for c in clients])
        for c in clients:
            c["agent"].epsilon = global_agent.epsilon
            c["agent"].reset_round_counter()
            c["agent"].set_global_reference()

        all_queue, all_fee, all_reward = [], [], []
        abandoned = 0
        for c in clients:
            for _ in range(args.local_episodes):
                m = run_episode_train(args, c["env"], c["agent"], c["city"]["city"])
                all_queue.extend(m["queue"])
                all_fee.extend(m["fee"])
                all_reward.extend(m["reward"])
                abandoned += m["abandoned"]

        weights = server.aggregate([c["agent"] for c in clients])
        global_agent.decay_epsilon()
        print(
            f"[fed-c round {rnd}/{args.rounds}] method={args.fed_method} weights={weights} "
            f"epsilon={global_agent.epsilon:.3f} events={len(all_reward)} abandoned={abandoned} "
            f"avg_queue={mean(all_queue):.4f}h avg_fee={mean(all_fee):.2f} "
            f"avg_reward={mean(all_reward):.4f} elapsed={time.time() - t0:.1f}s"
        )
        if args.save_every > 0 and rnd % args.save_every == 0:
            server.save(os.path.join(args.save_dir, f"global_round{rnd}.pth"), global_agent.epsilon)

    server.save(os.path.join(args.save_dir, "global_final.pth"), global_agent.epsilon)
    if args.fed_method == "fedsetrl":
        for c in clients:
            city_name = c["city"]["city"]
            path = os.path.join(args.save_dir, f"{city_name}_final.pth")
            torch.save(
                {"policy_net": c["agent"].policy_net.state_dict(), "epsilon": float(global_agent.epsilon)},
                path,
            )
            print(f"[done] FedSetRL personalized model ({city_name}) -> {path}")
    print(f"[done] federated final dir -> {args.save_dir}")


def build_eval_agent(args, env, model_path):
    agent = HindsightDQNAgent(
        num_features=19,
        num_actions=args.num_stations,
        station_node_ids=[s.traffic_node_id for s in env.stations],
        num_nodes_per_graph=env.num_nodes,
        network_variant=args.network,
        use_action_mask=args.use_action_mask,
    )
    agent.load_model(model_path)
    agent.epsilon = 0.0
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent


def run_episode_eval(args, city, eval_seed, model_path):
    env = build_env(args, city, eval_seed)
    agent = build_eval_agent(args, env, model_path)
    env.reset()
    reward_weights = weights_for(city["city"])
    trips, queues, fees, rewards = [], [], [], []
    abandoned = 0
    requests = 0
    action_counts = [0 for _ in range(args.num_stations)]
    station_load_integral = [0.0 for _ in range(args.num_stations)]

    for _ in range(args.steps_per_episode):
        urgent_evs = env.get_pending_decision_evs()
        requests += len(urgent_evs)
        pending_counts = {s.id: 0 for s in env.stations}
        actions = {}
        for ev in urgent_evs:
            state = env.get_graph_state_for_ev(ev, pending_counts=pending_counts)
            action = int(agent.select_action(state, action_mask=None))
            actions[ev.id] = action
            if 0 <= action < args.num_stations:
                pending_counts[action] = pending_counts.get(action, 0) + 1
                action_counts[action] += 1

        _, _, done, info = env.step(actions)
        for entry in info.get("charge_started", []):
            trip = float(entry.get("actual_trip_time_h", 0.0))
            queue = float(entry.get("actual_queue_time_h", 0.0))
            fee = float(entry.get("charging_fee", 0.0))
            trips.append(trip)
            queues.append(queue)
            fees.append(fee)
            rewards.append(compute_hindsight_reward(trip, queue, fee, *reward_weights))
        abandoned += len(info.get("abandoned", []))

        grid_loads = info.get("grid_loads", {})
        for i, station in enumerate(env.stations):
            station_load_integral[i] += float(grid_loads.get(station.power_node_id, station.last_total_load))
        if done:
            break

    events = len(rewards)
    denom = events + abandoned
    return {
        "events": events,
        "requests": requests,
        "abandoned": abandoned,
        "abandon_rate": abandoned / denom if denom else 0.0,
        "served_per_tick": events / float(args.steps_per_episode),
        "served_per_request": events / float(requests) if requests else 0.0,
        "avg_reward": mean(rewards),
        "median_reward": quantile(rewards, 0.5),
        "avg_trip_h": mean(trips),
        "avg_queue_h": mean(queues),
        "median_queue_h": quantile(queues, 0.5),
        "p90_queue_h": quantile(queues, 0.90),
        "p95_queue_h": quantile(queues, 0.95),
        "p99_queue_h": quantile(queues, 0.99),
        "max_queue_h": max(queues) if queues else 0.0,
        "cvar95_queue_h": cvar_upper(queues, 0.95),
        "var_queue_h": variance(queues),
        "avg_fee": mean(fees),
        "load_gini": gini(station_load_integral),
        "action_gini": gini(action_counts),
    }


def parse_model_specs(specs):
    parsed = []
    for item in specs:
        method, city, path = item.split(":", 2)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        parsed.append({"method": method, "city": city, "path": path})
    return parsed


def existing_eval_keys(path):
    if not os.path.isfile(path):
        return set()
    keys = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keys.add((row["method"], row["city"], row["load"], row["train_seed"], row["eval_seed"]))
    return keys


def evaluate(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cities = parse_client_specs(args.client_specs)
    specs = parse_model_specs(args.model_specs)
    done = existing_eval_keys(args.out) if args.resume else set()
    fieldnames = [
        "method", "state_variant", "city", "load", "train_seed", "eval_seed", "model_path",
        "events", "requests", "abandoned", "abandon_rate", "served_per_tick", "served_per_request",
        "avg_reward", "median_reward", "avg_trip_h", "avg_queue_h", "median_queue_h",
        "p90_queue_h", "p95_queue_h", "p99_queue_h", "max_queue_h", "cvar95_queue_h",
        "var_queue_h", "avg_fee", "load_gini", "action_gini",
    ]
    write_header = not os.path.isfile(args.out) or not args.resume
    mode = "a" if args.resume else "w"
    with open(args.out, mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        jobs = []
        for spec in specs:
            for city in cities:
                if spec["city"] == "*" or spec["city"] == city["city"]:
                    jobs.append((spec["method"], city, spec["path"]))
        t0 = time.time()
        for method, city, path in jobs:
            for eval_seed in args.eval_seeds:
                key = (method, city["city"], str(args.num_evs), str(args.train_seed), str(eval_seed))
                if key in done:
                    print(f"[skip] {key}")
                    continue
                print(
                    f"[eval-c] method={method} city={city['city']} load={args.num_evs} "
                    f"train_seed={args.train_seed} eval_seed={eval_seed}"
                )
                metrics = run_episode_eval(args, city, eval_seed, path)
                row = {
                    "method": method,
                    "state_variant": "C",
                    "city": city["city"],
                    "load": args.num_evs,
                    "train_seed": args.train_seed,
                    "eval_seed": eval_seed,
                    "model_path": path,
                }
                row.update(metrics)
                writer.writerow(row)
                f.flush()
                print(
                    f"[done] reward={metrics['avg_reward']:.4f} trip={metrics['avg_trip_h']:.4f}h "
                    f"queue={metrics['avg_queue_h']:.4f}h fee={metrics['avg_fee']:.2f} "
                    f"served={metrics['served_per_request']:.4f} elapsed={time.time() - t0:.1f}s"
                )


def summarize(args):
    rows = []
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for key in ["load", "train_seed", "eval_seed", "events", "requests", "abandoned"]:
                row[key] = int(float(row[key]))
            for key in [
                "abandon_rate", "served_per_request", "avg_reward", "median_reward",
                "avg_trip_h", "avg_queue_h", "p95_queue_h", "p99_queue_h",
                "cvar95_queue_h", "avg_fee", "load_gini", "action_gini",
            ]:
                row[key] = float(row[key])
            rows.append(row)

    groups = {}
    for row in rows:
        groups.setdefault((row["load"], row["city"], row["method"]), []).append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "load", "city", "method", "n", "avg_reward", "median_reward",
        "avg_trip_min", "avg_queue_min", "avg_fee", "p95_queue_min",
        "p99_queue_min", "cvar95_queue_min", "served_request_pct",
        "abandon_pct", "load_gini", "action_gini",
    ]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(groups):
            grp = groups[key]
            writer.writerow(
                {
                    "load": key[0],
                    "city": key[1],
                    "method": key[2],
                    "n": len(grp),
                    "avg_reward": f"{mean([r['avg_reward'] for r in grp]):.6f}",
                    "median_reward": f"{quantile([r['avg_reward'] for r in grp], 0.5):.6f}",
                    "avg_trip_min": f"{mean([r['avg_trip_h'] for r in grp]) * 60:.4f}",
                    "avg_queue_min": f"{mean([r['avg_queue_h'] for r in grp]) * 60:.4f}",
                    "avg_fee": f"{mean([r['avg_fee'] for r in grp]):.4f}",
                    "p95_queue_min": f"{mean([r['p95_queue_h'] for r in grp]) * 60:.4f}",
                    "p99_queue_min": f"{mean([r['p99_queue_h'] for r in grp]) * 60:.4f}",
                    "cvar95_queue_min": f"{mean([r['cvar95_queue_h'] for r in grp]) * 60:.4f}",
                    "served_request_pct": f"{mean([r['served_per_request'] for r in grp]) * 100:.4f}",
                    "abandon_pct": f"{mean([r['abandon_rate'] for r in grp]) * 100:.4f}",
                    "load_gini": f"{mean([r['load_gini'] for r in grp]):.6f}",
                    "action_gini": f"{mean([r['action_gini'] for r in grp]):.6f}",
                }
            )
    print(f"[summary] wrote {args.out}")


def add_common_args(p):
    p.add_argument("--num-evs", type=int, default=60)
    p.add_argument("--num-stations", type=int, default=4)
    p.add_argument("--num-chargers-per-station", type=int, default=8)
    p.add_argument("--steps-per-episode", type=int, default=144)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-steps-per-step", type=int, default=1)
    p.add_argument("--respawn", action="store_true", default=True)
    p.add_argument("--no-respawn", dest="respawn", action="store_false")
    p.add_argument("--graphml-file", default=os.path.join("map_outputs", "ema", "ema.graphml"))
    p.add_argument("--cache-dir", default=os.path.join("map_outputs", "ema_cache"))
    p.add_argument("--no-ue-background", action="store_true", default=False)
    p.add_argument("--ue-net-tntp", default=os.path.join("map_outputs", "ema", "EMA_net.tntp"))
    p.add_argument("--ue-trips-tntp", default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"))
    p.add_argument("--ue-max-iter", type=int, default=800)
    p.add_argument("--ue-tol", type=float, default=1e-4)
    p.add_argument("--network", choices=["original", "lightweight", "station_only", "station_attn"], default="station_only")
    p.add_argument("--use-action-mask", dest="use_action_mask", action="store_true", default=False)
    p.add_argument("--no-use-action-mask", dest="use_action_mask", action="store_false")
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=float, default=0.98)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=101)


def parse_args():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="cmd", required=True)

    p_local = sub.add_parser("train-local")
    add_common_args(p_local)
    p_local.add_argument("--city", choices=ALL_GRID_VARIANTS, required=True)
    p_local.add_argument("--ue-scale", type=float, default=None)
    p_local.add_argument("--episodes", type=int, default=100)
    p_local.add_argument("--save-dir", required=True)

    p_central = sub.add_parser("train-centralized")
    add_common_args(p_central)
    p_central.add_argument("--rounds", type=int, default=100)
    p_central.add_argument("--local-episodes", type=int, default=1)
    p_central.add_argument("--client-specs", default=DEFAULT_CLIENT_SPECS)
    p_central.add_argument("--save-dir", required=True)

    p_fed = sub.add_parser("train-fed")
    add_common_args(p_fed)
    p_fed.add_argument("--rounds", type=int, default=100)
    p_fed.add_argument("--local-episodes", type=int, default=1)
    p_fed.add_argument("--client-specs", default=DEFAULT_CLIENT_SPECS)
    p_fed.add_argument("--fed-method", choices=["fedavg", "fedprox", "fedsetrl"], required=True)
    p_fed.add_argument("--fedprox-mu", type=float, default=0.1)
    p_fed.add_argument("--aggregation-momentum", type=float, default=1.0)
    p_fed.add_argument("--save-dir", required=True)

    p_eval = sub.add_parser("eval")
    add_common_args(p_eval)
    p_eval.add_argument("--client-specs", default=DEFAULT_CLIENT_SPECS)
    p_eval.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    p_eval.add_argument("--train-seed", type=int, required=True)
    p_eval.add_argument("--model-specs", nargs="+", required=True)
    p_eval.add_argument("--out", required=True)
    p_eval.add_argument("--resume", action="store_true", default=False)
    p_eval.set_defaults(epsilon=0.0, epsilon_decay=1.0)

    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--csv", required=True)
    p_sum.add_argument("--out", required=True)

    args = root.parse_args()
    if getattr(args, "ue_scale", None) is None and hasattr(args, "city"):
        args.ue_scale = DEFAULT_CITY_SCALE.get(args.city, 1.0)
    return args


def main():
    args = parse_args()
    if args.cmd == "train-local":
        train_local(args)
    elif args.cmd == "train-centralized":
        train_centralized(args)
    elif args.cmd == "train-fed":
        train_federated(args)
    elif args.cmd == "eval":
        evaluate(args)
    elif args.cmd == "summarize":
        summarize(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
