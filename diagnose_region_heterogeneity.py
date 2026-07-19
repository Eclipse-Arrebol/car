"""区域异构性预检(model-free,不训练,几分钟)。

核心问题:每个城市内部,各充电站在 trip / queue / fee 上有没有真实 trade-off,
使得**不同的 reward 画像会理性地选出不同的站**?
- 若会(画像之间常常分歧)→ HETERO_REWARD 有机会让最优策略真分化 → 那 ~15h 异构
  训练值得跑。
- 若几乎不会(画像基本选同一个站)→ 无论 reward 怎么改,最优选择都不变 → 异构训练
  大概率仍打平,应省下 15h、转别的路线。

做法:对每城用固定的"理性同质策略"(argmin 默认权重)驱动 env;在每个 EV 决策点,
用 env._estimate_ev_station_metrics 取各站决策时的 (trip_time_h, queue_time_h,
charge_cost),再用 trainer 的精确归一化 + reward_profiles 的四个画像,记录每个画像
会选哪个站、以及各站的离散度。纯诊断,不依赖任何训练好的模型。
"""

import argparse
import json
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from evaluate_oracle_wait import _build_env
from trainer.trainer import NORM_TRIP, NORM_WAIT, NORM_FEE
from reward_profiles import DEFAULT_W, _HETERO

# 被测画像:默认 + 三城异构(与 reward_profiles.py 一致)
PROFILES = {
    "default(0.4/0.4/0.2)": DEFAULT_W,
    "old/queue-averse": _HETERO["old_city"],
    "new/fee-averse": _HETERO["new_city"],
    "suburb/trip-averse": _HETERO["suburb"],
}
HETERO_NAMES = ["old/queue-averse", "new/fee-averse", "suburb/trip-averse"]

CITIES = [("old_city", 1.3), ("new_city", 1.0), ("suburb", 0.7)]


def _cost(w, trip, queue, fee):
    """单站在权重 w 下的归一化广义成本(越小越优),与 hindsight reward 同口径。"""
    return w[0] * trip / NORM_TRIP + w[1] * queue / NORM_WAIT + w[2] * fee / NORM_FEE


def _argmin(values):
    return min(range(len(values)), key=lambda i: values[i])


def _cv(values):
    """变异系数 std/|mean|;mean≈0 时返回 0(该维度各站无差异)。"""
    n = len(values)
    if n == 0:
        return 0.0
    mu = sum(values) / n
    if abs(mu) < 1e-9:
        return 0.0
    var = sum((v - mu) ** 2 for v in values) / n
    return (var ** 0.5) / abs(mu)


def run_city(args, grid_variant, ue_scale):
    args.grid_variant = grid_variant
    args.ue_scale = ue_scale

    n_dec = 0
    disagree_all = 0                       # 三个异构画像未全选同一站
    flip_vs_default = {name: 0 for name in HETERO_NAMES}
    pair_flip = {"trip~queue": 0, "trip~fee": 0, "queue~fee": 0}
    cv_sum = {"trip": 0.0, "queue": 0.0, "fee": 0.0}

    for ep in range(args.episodes):
        env = _build_env(args, args.seed + ep)
        env.reset()
        for _ in range(args.steps_per_episode):
            urgent = env.get_pending_decision_evs()
            actions = {}
            for ev in urgent:
                m = [env._estimate_ev_station_metrics(ev, s) for s in env.stations]
                trips = [x["trip_time_h"] for x in m]
                queues = [x["queue_time_h"] for x in m]
                fees = [x["charge_cost"] for x in m]

                choice = {}
                for name, w in PROFILES.items():
                    costs = [_cost(w, trips[i], queues[i], fees[i]) for i in range(len(m))]
                    choice[name] = _argmin(costs)

                n_dec += 1
                ch_def = choice["default(0.4/0.4/0.2)"]
                ch_t = choice["suburb/trip-averse"]
                ch_q = choice["old/queue-averse"]
                ch_f = choice["new/fee-averse"]
                if not (ch_t == ch_q == ch_f):
                    disagree_all += 1
                for name in HETERO_NAMES:
                    if choice[name] != ch_def:
                        flip_vs_default[name] += 1
                pair_flip["trip~queue"] += int(ch_t != ch_q)
                pair_flip["trip~fee"] += int(ch_t != ch_f)
                pair_flip["queue~fee"] += int(ch_q != ch_f)
                cv_sum["trip"] += _cv(trips)
                cv_sum["queue"] += _cv(queues)
                cv_sum["fee"] += _cv(fees)

                actions[ev.id] = ch_def  # 用理性同质策略推进仿真
            _, _, done, _ = env.step(actions)
            if done:
                break
        print(f"  [{grid_variant} ep {ep+1}/{args.episodes}] decisions={n_dec}")

    d = max(1, n_dec)
    return {
        "grid_variant": grid_variant,
        "ue_scale": ue_scale,
        "decisions": n_dec,
        "disagree_all_rate": disagree_all / d,
        "flip_vs_default": {k: v / d for k, v in flip_vs_default.items()},
        "pairwise_flip_rate": {k: v / d for k, v in pair_flip.items()},
        "station_cv": {k: cv_sum[k] / d for k in cv_sum},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps-per-episode", type=int, default=144)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-evs", type=int, default=40)
    p.add_argument("--num-stations", type=int, default=4)
    p.add_argument("--num-chargers-per-station", type=int, default=8)
    p.add_argument("--respawn", action="store_true", default=True)
    p.add_argument("--graphml-file", type=str, default=os.path.join("map_outputs", "ema", "ema.graphml"))
    p.add_argument("--cache-dir", type=str, default=os.path.join("map_outputs", "ema_cache"))
    p.add_argument("--no-ue-background", action="store_true", default=False)
    p.add_argument("--ue-net-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_net.tntp"))
    p.add_argument("--ue-trips-tntp", type=str, default=os.path.join("map_outputs", "ema", "EMA_trips.tntp"))
    p.add_argument("--ue-max-iter", type=int, default=800)
    p.add_argument("--ue-tol", type=float, default=1e-4)
    p.add_argument("--ue-verbose", action="store_true", default=False)
    p.add_argument("--out", type=str, default=os.path.join("evaluation", "region_heterogeneity", "diagnosis.json"))
    args = p.parse_args()

    results = []
    for grid_variant, ue_scale in CITIES:
        print(f"=== {grid_variant} (ue_scale={ue_scale}) ===")
        results.append(run_city(args, grid_variant, ue_scale))

    # ---- 汇总表 ----
    print("\n" + "=" * 78)
    print("区域异构性诊断 — 决策层 trade-off 是否足以让不同 reward 画像选出不同的站")
    print("=" * 78)
    print(f"{'city':12s} {'decisions':>9s} {'画像全分歧%':>11s} "
          f"{'trip~queue%':>11s} {'trip~fee%':>10s} {'queue~fee%':>11s}")
    for r in results:
        pf = r["pairwise_flip_rate"]
        print(f"{r['grid_variant']:12s} {r['decisions']:>9d} "
              f"{r['disagree_all_rate']*100:>10.1f} "
              f"{pf['trip~queue']*100:>10.1f} {pf['trip~fee']*100:>9.1f} "
              f"{pf['queue~fee']*100:>10.1f}")

    print(f"\n{'city':12s} {'CV_trip':>8s} {'CV_queue':>9s} {'CV_fee':>8s}   "
          f"(站间离散度;越小越说明该维度各站可互换)")
    for r in results:
        cv = r["station_cv"]
        print(f"{r['grid_variant']:12s} {cv['trip']:>8.3f} {cv['queue']:>9.3f} {cv['fee']:>8.3f}")

    # ---- 判读 ----
    avg_disagree = sum(r["disagree_all_rate"] for r in results) / len(results)
    print("\n" + "-" * 78)
    print(f"三城平均「画像全分歧率」= {avg_disagree*100:.1f}%")
    if avg_disagree >= 0.15:
        verdict = "强 trade-off:不同目标常选不同站 → HETERO_REWARD 有望分化策略,15h 值得跑。"
    elif avg_disagree >= 0.05:
        verdict = "弱-中 trade-off:边界情形,建议先小规模异构冒烟确认非对角线会动,再决定。"
    else:
        verdict = ("trade-off 极弱:各站近似可互换,改 reward 画像也难改最优选择 → 异构训练"
                   "大概率仍打平,建议省下 15h,转『环境注入空间异构』或『泛化/隐私价值』路线。")
    print(f"判读:{verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"results": results, "avg_disagree_all_rate": avg_disagree},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[done] saved -> {args.out}")


if __name__ == "__main__":
    main()
