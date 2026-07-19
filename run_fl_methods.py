"""贡献1 个性化联邦对比:按顺序训练 naive FedAvg / FedProx / FedRep。

三个 job 共用同一 warm-start、同 HETERO_REWARD=1、同预算(120 轮),只差联邦方法,
确保对比公平:
  1. fedavg   -> checkpoints_hetero_fedavg120   (足量重训的干净退化基线)
  2. fedprox  -> checkpoints_hetero_fedprox      (单模型 + 近端项,预期仍修不好)
  3. fedrep   -> checkpoints_hetero_fedrep       (共享 encoder + 个性化 head,解法)

稳健性:
  - 每个 job 串流到独立日志;
  - JOB 级跳过:final checkpoint 已存在则跳过该 job(死掉后重跑同命令即从下一个继续);
  - 不做 job 内续训(train_federated_hindsight 从 warm-start 跑满 --rounds)。

用法:
    python run_fl_methods.py            # 正式跑(~3 个 job)
    python run_fl_methods.py --smoke    # 极小规模接线检查(分钟级)
"""

import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
WARMSTART = os.path.join("checkpoints_40ev_32piles_continue", "model_final.pth")
LOG_DIR = os.path.join("evaluation", "paradigm_train_logs")
CLIENT_SPECS = "old_city:1.3,new_city:1.0,suburb:0.7"


def build_jobs(smoke):
    rounds = 1 if smoke else 120
    common = dict(rounds=rounds)
    return [
        {"name": "fedavg", "method": "fedavg", "mu": None,
         "save_dir": "checkpoints_hetero_fedavg120", **common},
        {"name": "fedprox", "method": "fedprox", "mu": 0.1,
         "save_dir": "checkpoints_hetero_fedprox", **common},
        {"name": "fedrep", "method": "fedrep", "mu": None,
         "save_dir": "checkpoints_hetero_fedrep", **common},
    ]


def build_cmd(job, smoke):
    cmd = [
        PY, "train_federated_hindsight.py",
        "--fed-method", job["method"],
        "--load-model", WARMSTART,
        "--client-specs", CLIENT_SPECS,
        "--rounds", str(job["rounds"]),
        "--local-episodes", "1" if smoke else "2",
        "--num-evs", "10" if smoke else "40",
        "--num-stations", "4", "--num-chargers-per-station", "8",
        "--no-use-action-mask",
        "--epsilon", "0.3", "--epsilon-decay", "0.985",
        "--batch-size", "64",
        "--steps-per-episode", "20" if smoke else "144",
        "--save-every", "20",
        "--save-dir", job["save_dir"],
    ]
    if job["method"] == "fedprox":
        cmd += ["--fedprox-mu", str(job["mu"])]
    if smoke:
        cmd += ["--no-ue-background"]
    return cmd


def _keep_awake_on():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[fl] keep-awake ON")
    except Exception as e:
        print(f"[fl] keep-awake unavailable: {e}")


def _keep_awake_off():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", default=False)
    ap.add_argument("--force", action="store_true", default=False,
                    help="re-run even if final checkpoint exists")
    args = ap.parse_args()

    if not os.path.isfile(WARMSTART):
        raise FileNotFoundError(f"warm-start not found: {WARMSTART}")
    os.makedirs(LOG_DIR, exist_ok=True)
    _keep_awake_on()

    child_env = os.environ.copy()
    child_env["HETERO_REWARD"] = "1"  # 所有 job 全程异构 reward

    jobs = build_jobs(args.smoke)
    print(f"[fl] {'smoke' if args.smoke else 'full'} run, {len(jobs)} jobs, "
          f"warm-start={WARMSTART}, HETERO_REWARD=1")

    t0 = time.time()
    for i, job in enumerate(jobs, 1):
        final = os.path.join(job["save_dir"], "global_final.pth")
        if (not args.force) and os.path.isfile(final):
            print(f"[fl] ({i}/{len(jobs)}) SKIP {job['name']} (exists: {final})")
            continue
        sfx = "_smoke" if args.smoke else ""
        log_path = os.path.join(LOG_DIR, f"{job['name']}_hetero{sfx}.log")
        cmd = build_cmd(job, args.smoke)
        print(f"[fl] ({i}/{len(jobs)}) START {job['name']} -> log {log_path}")
        print(f"     cmd: {' '.join(cmd)}")
        jt = time.time()
        with open(log_path, "a", encoding="utf-8") as lf:
            ret = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=child_env)
        if ret.returncode != 0:
            print(f"[fl] FAILED {job['name']} (exit {ret.returncode}); see {log_path}. Stopping.")
            _keep_awake_off()
            sys.exit(ret.returncode)
        print(f"[fl] ({i}/{len(jobs)}) DONE {job['name']} in {time.time() - jt:.1f}s")

    _keep_awake_off()
    print(f"[fl] ALL DONE in {time.time() - t0:.1f}s")
    print("[fl] next: cross-city matrix with eval_paradigms.py "
          "(fedavg120/fedprox global_final.pth; fedrep per-city {city}_final.pth)")


if __name__ == "__main__":
    main()
