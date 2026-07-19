"""
Step 3 阶段 2: env 性能 + LMP 信号传导诊断。

验证三件事:
1. OPF (pp.runopp) 真的被调用了几次?成功几次?平均耗时多少?
2. LMP 缓存命中率和更新频率
3. buffer 里 reward 的 fee 项方差 — 验证 LMP 信号是否真的进入 reward

不修改任何生产代码,所有探针走 monkey-patch。
"""

import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Windows cp1252 兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from env.real_env import RealTrafficEnv
from env import power_grid_pp
from train import DQNAgent
from trainer.trainer import (
    HindsightTrainer,
    NORM_FEE,
    NORM_QUEUE,
    NORM_TRIP,
    compute_hindsight_reward,
)
import pandapower as pp


# ============================================================
# 探针 1: pp.runopp 计数 + 计时 + 异常率
# ============================================================
_runopp_stats = {"calls": 0, "exceptions": 0, "total_time_s": 0.0, "last_exc": None}
_orig_runopp = pp.runopp


def _patched_runopp(*args, **kwargs):
    _runopp_stats["calls"] += 1
    t0 = time.perf_counter()
    try:
        result = _orig_runopp(*args, **kwargs)
        _runopp_stats["total_time_s"] += time.perf_counter() - t0
        return result
    except Exception as e:
        _runopp_stats["exceptions"] += 1
        _runopp_stats["total_time_s"] += time.perf_counter() - t0
        _runopp_stats["last_exc"] = repr(e)[:200]
        raise


pp.runopp = _patched_runopp


# ============================================================
# 探针 2: get_lmp 返回值统计 (None 率 + lmp 值分布)
# ============================================================
_lmp_stats = {"calls": 0, "returned_none": 0, "lmp_values": []}
_orig_get_lmp = power_grid_pp.PPPowerGrid33.get_lmp


def _patched_get_lmp(self):
    _lmp_stats["calls"] += 1
    result = _orig_get_lmp(self)
    if result is None:
        _lmp_stats["returned_none"] += 1
    else:
        # 记录所有 station bus 的 LMP 值,后面看分布
        for bus_num, lmp_val in result.items():
            _lmp_stats["lmp_values"].append(lmp_val)
    return result


power_grid_pp.PPPowerGrid33.get_lmp = _patched_get_lmp


# ============================================================
# 探针 3: agent.store_transition 捕获 reward 分量
# ============================================================
# 由于 reward 是 trainer 内部算的,我们改从 info["completed"] 直接抓 fee
_completed_log = []  # list of dict: {trip, queue, fee, reward_estimate}

_orig_step = RealTrafficEnv.step


def _patched_step(self, actions):
    obs, reward, done, info = _orig_step(self, actions)
    for entry in info.get("charge_started", []):
        trip = entry.get("actual_trip_time_h", 0.0)
        queue = entry.get("actual_queue_time_h", 0.0)
        fee = entry.get("charging_fee", 0.0)
        # 镜像 trainer 的 reward 公式 (v3 § 二锁定)
        reward_estimate = compute_hindsight_reward(trip, queue, fee)
        _completed_log.append({
            "trip": trip, "queue": queue, "fee": fee,
            "trip_term": 0.4 * (trip / NORM_TRIP),
            "queue_term": 0.4 * (queue / NORM_QUEUE),
            "fee_term": 0.2 * (fee / NORM_FEE),
            "reward": reward_estimate,
        })
    return obs, reward, done, info


RealTrafficEnv.step = _patched_step


# ============================================================
# 主程序
# ============================================================
def main():
    env = RealTrafficEnv(
        graphml_file=os.path.join("map_outputs", "ema", "ema.graphml"),
        num_stations=2,
        num_evs=10,
        max_nodes=1_000_000,
        cache_dir=os.path.join("map_outputs", "ema_cache"),
        seed=42,
        respawn_after_full_charge=True,
    )
    agent = DQNAgent(num_features=18, num_actions=2,
                     station_node_ids=None, num_nodes_per_graph=9)
    trainer = HindsightTrainer(env, agent)

    EPISODES = 3
    STEPS = 100

    t_start = time.perf_counter()
    for ep in range(EPISODES):
        env.reset()
        trainer.pending.clear()
        trainer._current_step = 0
        for _ in range(STEPS):
            trainer.step_episode()
    total_time_s = time.perf_counter() - t_start

    # ========== 报告 ==========
    print("\n" + "="*60)
    print("Step 3 阶段 2 诊断报告")
    print("="*60)

    print(f"\n[overall] {EPISODES} episodes x {STEPS} steps = {EPISODES*STEPS} env.step calls")
    print(f"[overall] total elapsed: {total_time_s:.3f}s")
    print(f"[overall] per env.step:  {total_time_s/(EPISODES*STEPS)*1000:.2f} ms")

    print(f"\n[OPF] pp.runopp calls:      {_runopp_stats['calls']}")
    print(f"[OPF] pp.runopp exceptions: {_runopp_stats['exceptions']}")
    if _runopp_stats['calls'] > 0:
        success_rate = (_runopp_stats['calls'] - _runopp_stats['exceptions']) / _runopp_stats['calls']
        print(f"[OPF] success rate:         {success_rate*100:.1f}%")
        print(f"[OPF] total time:           {_runopp_stats['total_time_s']*1000:.1f} ms")
        print(f"[OPF] per-call time:        {_runopp_stats['total_time_s']/_runopp_stats['calls']*1000:.2f} ms")
    if _runopp_stats['last_exc']:
        print(f"[OPF] last exception:       {_runopp_stats['last_exc']}")

    print(f"\n[LMP] get_lmp calls:        {_lmp_stats['calls']}")
    print(f"[LMP] returned None:        {_lmp_stats['returned_none']}")
    if _lmp_stats['lmp_values']:
        vals = _lmp_stats['lmp_values']
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean)**2 for v in vals) / n
        vmin, vmax = min(vals), max(vals)
        print(f"[LMP] value samples:        {n}")
        print(f"[LMP] mean: {mean:.6f}  std: {var**0.5:.6f}  min: {vmin:.6f}  max: {vmax:.6f}")
    else:
        print("[LMP] no LMP values recorded — get_lmp 从未成功返回非 None")

    print(f"\n[completed] total EVs completed: {len(_completed_log)}")
    if _completed_log:
        n = len(_completed_log)
        for key in ("trip", "queue", "fee", "trip_term", "queue_term", "fee_term", "reward"):
            vals = [c[key] for c in _completed_log]
            mean = sum(vals) / n
            var = sum((v - mean)**2 for v in vals) / n
            print(f"[completed] {key:12s} mean={mean:10.4f}  std={var**0.5:10.4f}  min={min(vals):10.4f}  max={max(vals):10.4f}")

    # 关键判读
    print("\n" + "-"*60)
    print("关键诊断")
    print("-"*60)
    if _runopp_stats['exceptions'] / max(1, _runopp_stats['calls']) > 0.5:
        print("⚠ OPF 异常率 > 50%,LMP 信号链断了")
    if _completed_log:
        fee_vals = [c["fee_term"] for c in _completed_log]
        queue_vals = [c["queue_term"] for c in _completed_log]
        fee_std = (sum((v - sum(fee_vals)/len(fee_vals))**2 for v in fee_vals) / len(fee_vals))**0.5
        queue_std = (sum((v - sum(queue_vals)/len(queue_vals))**2 for v in queue_vals) / len(queue_vals))**0.5
        print(f"reward 分量 std 对比:")
        print(f"  fee_term std:   {fee_std:.4f}")
        print(f"  queue_term std: {queue_std:.4f}")
        if fee_std < 0.01:
            print("⚠ fee_term std < 0.01,LMP 在 reward 里几乎没信号 — 可能等价于常数")
        if fee_std > 0 and queue_std > 0:
            print(f"  fee/queue 信号比: {fee_std/queue_std*100:.2f}%")


if __name__ == "__main__":
    main()
