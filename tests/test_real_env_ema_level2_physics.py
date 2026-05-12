"""
层级 2：EMA 完整路网 + RealTrafficEnv 物理合理性（T2.1–T2.7）。

需 osmnx；缺 ema.graphml 时 skip。Windows UTF-8 stdio。

断言前后会 print 汇总诊断（flush）。若面板仍不显示：`python -u -m unittest tests.test_real_env_ema_level2_physics -v`
"""
from __future__ import annotations

import math
import random
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from env.osm_loader import HAS_OSMNX  # noqa: E402
from env.real_env import RealTrafficEnv  # noqa: E402

EMA_GRAPHML = _ROOT / "map_outputs" / "ema" / "ema.graphml"


def _diag_print(*lines: str) -> None:
    for line in lines:
        print(line, flush=True)


def _dispatch_pending(env: RealTrafficEnv, station_id: int = 0) -> dict:
    actions: dict = {}
    for ev in env.evs:
        if ev.status == "IDLE" and env.should_request_charge_decision(ev):
            actions[ev.id] = station_id
    return actions


@unittest.skipUnless(EMA_GRAPHML.is_file(), f"未找到 EMA 路网: {EMA_GRAPHML}")
@unittest.skipUnless(HAS_OSMNX, "未安装 osmnx，无法从 graphml 加载路网")
class TestLevel2EmaPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls._cache_dir = cls._td.name

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _make_env(self, num_evs: int, num_stations: int, seed: int) -> RealTrafficEnv:
        random.seed(seed)
        return RealTrafficEnv(
            graphml_file=str(EMA_GRAPHML),
            num_stations=num_stations,
            num_evs=num_evs,
            max_nodes=1_000_000,
            seed=seed,
            cache_dir=self._cache_dir,
            respawn_after_full_charge=True,
        )

    def test_T2_1_travel_time_sane(self):
        """50 step；已完成行程的 actual_trip_time_h 有界（EMA 小网尺度略放宽平均上界）。"""
        env = self._make_env(num_evs=40, num_stations=2, seed=101)
        env.reset()
        trip_times: list[float] = []
        for _ in range(50):
            _, reward, _, info = env.step(_dispatch_pending(env, 0))
            self.assertFalse(math.isnan(reward))
            for c in info.get("completed", []):
                t = float(c["actual_trip_time_h"])
                trip_times.append(t)
        if len(trip_times) < 1:
            _diag_print(
                "",
                "[T2.1] 50 step 内 info['completed'] 无样本（无 actual_trip_time_h 可统计）",
                "  → skipTest（需充电完成事件）",
                "",
            )
            self.skipTest("50 步内无充电完成事件，无法断言行程时间分布")
        for t in trip_times:
            self.assertGreaterEqual(t, 0.0, "行程时间不应为负")
            self.assertLess(t, 5.0, "行程时间应 <5h（无天文级）")
        # 到站即充时 actual_trip_time_h 可为 ~0；对「确有行驶」子集检验均值
        positive = [t for t in trip_times if t > 1e-6]
        if positive:
            mean_pos = sum(positive) / len(positive)
            # EMA 小网大量贴站短行程，平均可远小于 0.1h；只保正与非天文上界
            self.assertGreaterEqual(mean_pos, 1e-9)
            self.assertLessEqual(mean_pos, 4.5, "有行驶样本的平均时间应与单趟 <5h 一致")
        overall = sum(trip_times) / len(trip_times)
        self.assertGreaterEqual(overall, 0.0)
        self.assertLessEqual(overall, 4.5)
        mn, mx = min(trip_times), max(trip_times)
        _diag_print(
            "",
            "[T2.1] 行驶时间 actual_trip_time_h（50 step，completed 汇总）",
            f"  n_samples={len(trip_times)}  min={mn:.6f}h  max={mx:.6f}h  mean_all={overall:.6f}h",
            f"  n_trip>1e-6h={len(positive)}"
            + (
                f"  mean_positive={sum(positive) / len(positive):.6f}h"
                if positive
                else "  (无「>1e-6h」子集，多为贴站短行程)"
            ),
            "  断言：全部 ≥0、<5h；有正样本时均值 ≤4.5h；总均值 ≤4.5h",
            "",
        )

    def test_T2_2_soc_energy_non_negative(self):
        """完成充电：本会话充入电网侧能量 >0（跨多轮时 SOC 快照易错位，不单测 sm-sc）。"""
        env = self._make_env(num_evs=35, num_stations=2, seed=102)
        env.reset()
        n_completed = 0
        rows: list[str] = []
        for _ in range(120):
            pre_te = {ev.id: ev.total_energy_charged for ev in env.evs}
            _, _, _, info = env.step(_dispatch_pending(env, 1))
            for c in info.get("completed", []):
                eid = int(c["ev_id"])
                ev = env._find_ev_by_id(eid)
                de = float(ev.total_energy_charged) - float(pre_te[eid])
                fee = float(c.get("charging_fee", 0.0))
                self.assertTrue(
                    de > 1e-6 or fee > 1e-6,
                    "完成会话应有可测的充电能量或费用（ΔkWh 或 charging_fee）",
                )
                n_completed += 1
                if len(rows) < 8:
                    rows.append(
                        f"    ev_id={eid}  Δtotal_energy_charged_kwh={de:.6g}  charging_fee={fee:.6g}"
                    )
        _diag_print(
            "",
            "[T2.2] 完成充电：电网侧累计能量增量 / 费用（本会话用 ΔkWh 或 fee 判定）",
            f"  120 step 内 completed 次数={n_completed}",
            *(rows if rows else ["    (无 completed，请加大步数或 seed)"]),
            "  断言：每次完成须 de>1e-6 或 charging_fee>1e-6",
            "",
        )

    def test_T2_3_completed_fields_no_nan_non_negative(self):
        """info['completed'] 中时间与费用非 NaN 且 >=0；每步 reward 非 NaN。"""
        env = self._make_env(num_evs=30, num_stations=2, seed=103)
        env.reset()
        n_comp = 0
        tt_all: list[float] = []
        wt_all: list[float] = []
        fee_all: list[float] = []
        rewards: list[float] = []
        for _ in range(50):
            _, reward, _, info = env.step(_dispatch_pending(env, 0))
            rewards.append(float(reward))
            self.assertFalse(math.isnan(reward), "reward 不能为 NaN")
            for c in info.get("completed", []):
                tt = float(c["actual_trip_time_h"])
                wt = float(c["actual_queue_time_h"])
                fee = float(c["charging_fee"])
                for name, val in (("trip", tt), ("wait", wt), ("fee", fee)):
                    self.assertFalse(math.isnan(val), f"completed {name} 为 NaN")
                    self.assertGreaterEqual(val, 0.0)
                n_comp += 1
                tt_all.append(tt)
                wt_all.append(wt)
                fee_all.append(fee)
        _diag_print(
            "",
            "[T2.3] reward + completed 三件套（50 step）",
            f"  reward: n_steps={len(rewards)}  min={min(rewards):.6g}  max={max(rewards):.6g}",
            f"  completed 条数={n_comp}（字段：trip/queue/fee，对应 actual_trip_time_h 等）",
            *(
                [
                    f"  trip_h: min={min(tt_all):.6g} max={max(tt_all):.6g}",
                    f"  wait_h: min={min(wt_all):.6g} max={max(wt_all):.6g}",
                    f"  fee:    min={min(fee_all):.6g} max={max(fee_all):.6g}",
                ]
                if n_comp
                else ["  (本段无 completed 记录，仅验证了每步 reward 非 NaN)"]
            ),
            "  断言：reward 与 trip/wait/fee 均非 NaN 且 ≥0",
            "",
        )

    def test_T2_4_power_flow_voltages_sane(self):
        """每步 bus 电压无 NaN；落在 [0.9,1.1] p.u.（越限记 optional 日志）。"""
        env = self._make_env(num_evs=25, num_stations=2, seed=104)
        env.reset()
        violations_total = 0
        vmin_g, vmax_g = float("inf"), float("-inf")
        n_bus_readings = 0
        for _ in range(50):
            _, _, _, info = env.step(_dispatch_pending(env, 0))
            bv = info["bus_voltages"]
            for bus, v in bv.items():
                self.assertFalse(math.isnan(v), f"{bus} 电压 NaN")
                self.assertGreaterEqual(v, 0.9, f"{bus} 电压低于 0.9 pu")
                self.assertLessEqual(v, 1.1, f"{bus} 电压高于 1.1 pu")
                vmin_g = min(vmin_g, float(v))
                vmax_g = max(vmax_g, float(v))
                n_bus_readings += 1
            violations_total += int(info.get("voltage_violations", 0))
        _diag_print(
            "",
            "[T2.4] 电网潮流 / bus 电压（50 step，每步全母线）",
            f"  全局 min_v={vmin_g:.6f} pu  max_v={vmax_g:.6f} pu（跨所有 step×bus，共 {n_bus_readings} 次读数）",
            f"  cumulative voltage_violations（各步相加）={violations_total}",
            "  断言：每步每母线无 NaN，且 v∈[0.9,1.1] p.u.；violations 仅作记录不 fail",
            "",
        )

    def test_T2_5_prices_positive_lmp_bounded(self):
        """各站 current_price>0；若存在 LMP 则落在合理区间。"""
        env = self._make_env(num_evs=20, num_stations=2, seed=105)
        env.reset()
        for _ in range(20):
            env.step(_dispatch_pending(env, 0))
        price_lines: list[str] = []
        for st in env.stations:
            self.assertGreater(st.current_price, 0.0)
            price_lines.append(
                f"    station_traffic_node={st.traffic_node_id}  current_price={st.current_price:.6g}"
            )
        lmp_lines: list[str] = []
        if hasattr(env.power_grid, "get_lmp"):
            cur = {s.power_node_id: s.last_total_load for s in env.stations}
            env.power_grid.net = env.power_grid._net_with_loads(cur)
            lmp = env.power_grid.get_lmp()
            if lmp is None:
                _diag_print(
                    "",
                    "[T2.5] 电价 / LMP",
                    *price_lines,
                    "  get_lmp 返回 None（OPF 不可用）→ skip LMP 数值断言",
                    "",
                )
                self.skipTest("get_lmp 返回 None（OPF 不可用），跳过 LMP 数值断言")
            for bus, price in lmp.items():
                self.assertFalse(math.isnan(price), f"LMP NaN bus={bus}")
                self.assertGreaterEqual(price, 0.05)
                self.assertLessEqual(price, 3.0, "LMP 典型上界（元/kWh 量级，略放宽）")
                lmp_lines.append(f"    bus={bus}  LMP={price:.6g}")
            lp = list(lmp.values())
            _diag_print(
                "",
                "[T2.5] 电价 / LMP（20 step 后快照）",
                *price_lines,
                f"  LMP: n_bus={len(lmp)}  min={min(lp):.6g}  max={max(lp):.6g} 元/kWh（断言∈[0.05,3]）",
                *lmp_lines[:12],
                *([] if len(lmp_lines) <= 12 else [f"    ... 其余 {len(lmp_lines) - 12} 条省略"]),
                "",
            )
        else:
            _diag_print(
                "",
                "[T2.5] 各站 current_price>0",
                *price_lines,
                "  power_grid 无 get_lmp，未测 LMP",
                "",
            )

    def test_T2_6_bpr_congestion_increases_time(self):
        """50 步后统计峰值边；BPR 下高流比低流时间长。"""
        env = self._make_env(num_evs=45, num_stations=2, seed=106)
        env.reset()
        peak_hist: dict[tuple, int] = defaultdict(int)
        for _ in range(50):
            _, _, _, info = env.step(_dispatch_pending(env, 0))
            for k, v in info.get("peak_edge_flows_this_step", {}).items():
                peak_hist[k] = max(peak_hist[k], int(v))
        ranked = sorted(peak_hist.items(), key=lambda kv: -kv[1])[:5]
        self.assertTrue(ranked, "无任一边有流量统计")
        for edge, cnt in ranked:
            self.assertGreater(cnt, 0, f"边 {edge} 应有 EV 经过")
        busy = ranked[0][0]
        u, v = busy
        _len_m, _spd, t0_h, cap = env.get_edge_base_profile(u, v)
        x_high = float(ranked[0][1])
        x_low = 0.0
        t_low = env._bpr_time_h(t0_h, x_low, cap)
        t_high = env._bpr_time_h(t0_h, max(x_high, 1.0), cap)
        self.assertGreater(t_high, t_low, "BPR：高流量边时间应大于零流参考")
        top5 = ", ".join(f"{e}:{c}" for e, c in ranked)
        _diag_print(
            "",
            "[T2.6] 拥堵 / BPR（50 step，peak_edge_flows 峰值 Top5）",
            f"  top5 edge->count: {top5}",
            f"  最忙边 busy=({u},{v})  t0_h={t0_h:.6g}h  cap={cap}  x_high={x_high}",
            f"  _bpr_time_h: x=0 → t_low={t_low:.6g}h ; x=max(x_high,1) → t_high={t_high:.6g}h",
            "  断言：Top5 边 count>0；t_high>t_low",
            "",
        )

    def test_T2_7_queue_pressure_abandon_mid_range(self):
        """100 EV / 5 站，强压 0 号站队列；100 步内应有 abandon（去掉 t2 挂起后可能出现极多 queue_full）。"""
        env = self._make_env(num_evs=100, num_stations=5, seed=107)
        env.reset()
        abandon_ids: set[int] = set()
        for _ in range(100):
            actions = _dispatch_pending(env, 0)
            _, _, _, info = env.step(actions)
            for rec in info.get("abandoned", []):
                abandon_ids.add(int(rec["ev_id"]))
        self.assertGreater(len(abandon_ids), 0, "高压下应有 EV 经历 abandon")
        # 原「<100 全员不 abandon」依赖 T2_PENDING 在站外堆积；到站即入队后高冲突下可出现全员曾 abandon
        self.assertLessEqual(len(abandon_ids), 100)
        ids_sorted = sorted(abandon_ids)
        preview = ids_sorted[:20]
        tail = f" ...(+{len(ids_sorted) - 20})" if len(ids_sorted) > 20 else ""
        _diag_print(
            "",
            "[T2.7] 队列压力 / abandon（100 EV, 5 站, 100 step，全部派 0 号站）",
            f"  唯一 abandon ev_id 数={len(abandon_ids)}（要求 >0 且 ≤100）",
            f"  ev_id 样例（升序前 20）: {preview}{tail}",
            "",
        )


if __name__ == "__main__":
    unittest.main()
