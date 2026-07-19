"""Sequential training pipeline for the paradigm comparison.

Trains, in order, on top of the SAME warm-start with the SAME hyper-params:
  1. Local old_city   (i-budget: 100 ep x 144 = 14,400 steps)
  2. Local new_city   (i-budget)
  3. Local suburb     (i-budget)
  4. Centralized      (50 rounds x 3 cities x 2 ep x 144 = 43,200 steps)

Federated is NOT retrained (reuse checkpoints_fed_hindsight_40ev/global_final.pth).

Robustness for multi-day background runs:
  - each job streams to its own log file
  - JOB-LEVEL RESUME: a job whose final checkpoint already exists is skipped, so
    re-launching the pipeline after a death continues from the next job. (Within a
    job, train_*.py still save periodic snapshots for manual finer resume.)

Usage:
    python run_paradigm_training.py            # real run (~12h)
    python run_paradigm_training.py --smoke     # tiny wiring check (seconds)
"""

import argparse
import os
import subprocess
import sys
import time

WARMSTART = os.path.join("checkpoints_40ev_32piles_continue", "model_final.pth")
PY = sys.executable
LOG_DIR = os.path.join("evaluation", "paradigm_train_logs")


def _latest_snapshot(save_dir, prefix):
    """Return (path, N) of the highest-numbered snapshot {prefix}{N}.pth, or (None, 0)."""
    best, best_n = None, 0
    if not os.path.isdir(save_dir):
        return best, best_n
    for fn in os.listdir(save_dir):
        if fn.startswith(prefix) and fn.endswith(".pth"):
            num = fn[len(prefix):-4]
            if num.isdigit() and int(num) > best_n:
                best_n, best = int(num), os.path.join(save_dir, fn)
    return best, best_n


def local_job(city, ue_scale, episodes, sfx):
    save_dir = f"checkpoints{sfx}_local_{city}"
    return {
        "name": f"local_{city}", "kind": "local", "save_dir": save_dir,
        "target": episodes, "city": city, "ue_scale": ue_scale,
        "final": os.path.join(save_dir, "model_final.pth"),
        "snap_prefix": "model_ep",
    }


def centralized_job(rounds, sfx):
    save_dir = f"checkpoints{sfx}_centralized"
    return {
        "name": "centralized", "kind": "central", "save_dir": save_dir,
        "target": rounds,
        "final": os.path.join(save_dir, "central_final.pth"),
        "snap_prefix": "central_round",
    }


def federated_job(rounds, sfx):
    save_dir = f"checkpoints{sfx}_federated"
    return {
        "name": "federated", "kind": "federated", "save_dir": save_dir,
        "target": rounds,
        "final": os.path.join(save_dir, "global_final.pth"),
        "snap_prefix": "global_round",
    }


def build_cmd(job, save_every, network="station_only", from_scratch=False,
              epsilon_decay=None):
    """Build the train command, resuming from the latest snapshot if one exists.

    Resume from a snapshot continues the saved epsilon schedule (we drop the
    --epsilon override so the loaded epsilon is kept) and only runs the remaining
    episodes/rounds. A fresh start uses the shared warm-start + epsilon 0.3.

    from_scratch: omit the warm-start on a fresh job and start epsilon at 1.0.
    Required for architectures (e.g. station_attn) whose weights are incompatible
    with the station_only warm-start; resume from this variant's own snapshots
    still works normally.
    """
    snap, n = _latest_snapshot(job["save_dir"], job["snap_prefix"])
    fresh = snap is None
    remaining = job["target"] - n
    if fresh:
        eps_flags = ["--epsilon", "1.0" if from_scratch else "0.3"]
    else:
        eps_flags = []  # resume keeps loaded epsilon
    load_flags = []
    if not (fresh and from_scratch):
        load = WARMSTART if fresh else snap
        load_flags = ["--load-model", load]
    decay_flags = ["--epsilon-decay", str(epsilon_decay)] if epsilon_decay else []
    base = [
        "--num-evs", "40", "--num-stations", "4", "--num-chargers-per-station", "8",
        "--no-use-action-mask", *load_flags, "--save-dir", job["save_dir"],
        "--batch-size", "64", "--steps-per-episode", "144",
        "--network", network, *eps_flags, *decay_flags,
    ]
    if job["kind"] == "local":
        cmd = [PY, "train_hindsight.py", "--episodes", str(remaining),
               "--grid-variant", job["city"], "--ue-scale", str(job["ue_scale"]),
               "--save-every", str(save_every), *base]
    elif job["kind"] == "central":
        cmd = [PY, "train_centralized_hindsight.py", "--rounds", str(remaining),
               "--local-episodes", "2", "--client-specs",
               "old_city:1.3,new_city:1.0,suburb:0.7", "--save-every", str(save_every), *base]
    else:  # federated
        cmd = [PY, "train_federated_hindsight.py", "--rounds", str(remaining),
               "--local-episodes", "2", "--client-specs",
               "old_city:1.3,new_city:1.0,suburb:0.7", "--save-every", str(save_every), *base]
    return cmd, fresh, n, remaining


def build_jobs(smoke, hetero, network="station_only",
               local_eps_override=None, rounds_override=None, tag=""):
    if smoke:
        local_eps, rounds = 1, 1
    else:
        local_eps = local_eps_override if local_eps_override else 100
        rounds = rounds_override if rounds_override else 50
    net_sfx = "" if network == "station_only" else f"_{network}"
    sfx = net_sfx + tag + ("_hetero" if hetero else "")
    jobs = [
        local_job("old_city", 1.3, local_eps, sfx),
        local_job("new_city", 1.0, local_eps, sfx),
        local_job("suburb", 0.7, local_eps, sfx),
        centralized_job(rounds, sfx),
    ]
    # Federated must be (re)trained whenever we can't reuse the existing
    # station_only homogeneous global_final.pth: i.e. heterogeneous rewards, a
    # different architecture, OR a tagged run (e.g. from-scratch control).
    if hetero or network != "station_only" or tag:
        jobs.append(federated_job(rounds, sfx))
    return jobs


def _keep_awake_on():
    """Tell Windows to stay awake while the pipeline runs (idle-sleep guard)."""
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        print("[pipeline] keep-awake ON (ES_CONTINUOUS|ES_SYSTEM_REQUIRED)")
    except Exception as e:
        print(f"[pipeline] keep-awake unavailable: {e}")


def _keep_awake_off():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--force", action="store_true", default=False,
                    help="re-run even if final checkpoint exists")
    ap.add_argument("--hetero", action="store_true", default=False,
                    help="per-city heterogeneous rewards (sets HETERO_REWARD=1, "
                         "_hetero checkpoint dirs, adds federated retrain)")
    ap.add_argument("--network", default="station_only",
                    choices=["original", "lightweight", "station_only", "station_attn"],
                    help="Q-network variant; non-default uses _{network} checkpoint dirs")
    ap.add_argument("--from-scratch", action="store_true", default=False,
                    help="skip the station_only warm-start (required for incompatible "
                         "architectures e.g. station_attn); resume still uses own snapshots")
    ap.add_argument("--local-episodes", type=int, default=None,
                    help="override per-local-job episodes (default 100; from-scratch "
                         "needs more, e.g. 240)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="override centralized/federated rounds (default 50; from-scratch "
                         "needs more, e.g. 120)")
    ap.add_argument("--epsilon-decay", type=float, default=None,
                    help="epsilon decay passed to all jobs; pick so epsilon reaches "
                         "min within budget (local decays 1x/ep, central/fed ~2x/round)")
    ap.add_argument("--tag", type=str, default="",
                    help="extra checkpoint/log dir suffix to namespace a run, e.g. "
                         "_scratch for a from-scratch station_only control")
    args = ap.parse_args()

    if (not args.from_scratch) and (not os.path.isfile(WARMSTART)):
        raise FileNotFoundError(f"warm-start not found: {WARMSTART}")
    os.makedirs(LOG_DIR, exist_ok=True)
    _keep_awake_on()

    child_env = os.environ.copy()
    if args.hetero:
        child_env["HETERO_REWARD"] = "1"

    jobs = build_jobs(args.smoke, args.hetero, args.network,
                      args.local_episodes, args.rounds, args.tag)
    tag = ("smoke" if args.smoke else "full") + ("+hetero" if args.hetero else "")
    tag += f"+{args.network}" if args.network != "station_only" else ""
    tag += f"+{args.tag}" if args.tag else ""
    tag += "+scratch" if args.from_scratch else ""
    warm = "(from-scratch)" if args.from_scratch else WARMSTART
    print(f"[pipeline] {tag} run, {len(jobs)} jobs, warm-start={warm}, "
          f"HETERO_REWARD={child_env.get('HETERO_REWARD', '0')}")

    t0 = time.time()
    for i, job in enumerate(jobs, 1):
        if (not args.force) and os.path.isfile(job["final"]):
            print(f"[pipeline] ({i}/{len(jobs)}) SKIP {job['name']} (exists: {job['final']})")
            continue
        save_every = 20 if job["kind"] == "local" else 10
        cmd, fresh, done_n, remaining = build_cmd(
            job, save_every, network=args.network, from_scratch=args.from_scratch,
            epsilon_decay=args.epsilon_decay)
        if remaining <= 0:
            print(f"[pipeline] ({i}/{len(jobs)}) SKIP {job['name']} (snapshot at target {done_n})")
            continue
        mode = "FRESH" if fresh else f"RESUME from {done_n} ({remaining} left)"
        net_sfx = "" if args.network == "station_only" else f"_{args.network}"
        sfx = net_sfx + args.tag + ("_hetero" if args.hetero else "") + ("_smoke" if args.smoke else "")
        log_path = os.path.join(LOG_DIR, f"{job['name']}{sfx}.log")
        print(f"[pipeline] ({i}/{len(jobs)}) START {job['name']} [{mode}] -> log {log_path}")
        print(f"           cmd: {' '.join(cmd)}")
        jt = time.time()
        # append (not truncate) so resume keeps prior log history
        with open(log_path, "a", encoding="utf-8") as lf:
            ret = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=child_env)
        if ret.returncode != 0:
            print(f"[pipeline] FAILED {job['name']} (exit {ret.returncode}); see {log_path}. Stopping.")
            sys.exit(ret.returncode)
        if not os.path.isfile(job["final"]):
            print(f"[pipeline] WARN {job['name']} finished but no final checkpoint at {job['final']}")
        print(f"[pipeline] ({i}/{len(jobs)}) DONE {job['name']} in {time.time() - jt:.1f}s")

    _keep_awake_off()
    print(f"[pipeline] ALL DONE in {time.time() - t0:.1f}s")
    print("[pipeline] next: evaluate full cross-city matrix with eval_paradigms.py")


if __name__ == "__main__":
    main()
