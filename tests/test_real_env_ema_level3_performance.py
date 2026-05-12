"""
层级 3：EMA + RealTrafficEnv 性能与时间预算（T3.1–T3.3）。

T3.1 单 step 平均耗时 | T3.2 cProfile 热点（top cumulative）| T3.3 步进内存增长

需 osmnx；缺 ema.graphml 时 skip。T3.3 需 psutil，未安装则 skip。Windows UTF-8 stdio。

诊断信息用 print(..., flush=True) 打到 stdout。若 IDE 测试面板仍只显示通过/失败，可在终端执行：
`python -u -m unittest tests.test_real_env_ema_level3_performance -v`
（`-u` 无缓冲）或设置环境变量 `PYTHONUNBUFFERED=1`。
"""
from __future__ import annotations

import cProfile
import io
import pstats
import random
import sys
import tempfile
import time
import unittest
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

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

from env.osm_loader import HAS_OSMNX  # noqa: E402
from env.real_env import RealTrafficEnv  # noqa: E402

EMA_GRAPHML = _ROOT / "map_outputs" / "ema" / "ema.graphml"

# 经验阈值（HANDOFF / 需求）：>2s/步 视为不可接受，须优化后再收紧断言
PER_STEP_MAX_SEC = 2.0
MEM_GROWTH_MAX_MB = 100.0  # 200 step 内相对 reset 后基线的增长


def _diag_print(*lines: str) -> None:
    """写到 stdout 并 flush；若面板仍不显示，请在终端运行：python -m unittest … -v。"""
    for line in lines:
        print(line, flush=True)


def _per_step_bucket_label(per_step: float) -> str:
    if per_step < 0.05:
        return "档位：<0.05s（很好，可敞开跑）"
    if per_step < 0.5:
        return "档位：[0.05, 0.5)s（可接受）"
    if per_step < 2.0:
        return "档位：[0.5, 2.0)s（警惕，大 episode 会很慢）"
    return "档位：≥2.0s（须 profile，否则训练周期不可接受）"


@unittest.skipUnless(EMA_GRAPHML.is_file(), f"未找到 EMA 路网: {EMA_GRAPHML}")
@unittest.skipUnless(HAS_OSMNX, "未安装 osmnx，无法从 graphml 加载路网")
class TestLevel3EmaPerformance(unittest.TestCase):
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
            respawn_after_full_charge=False,
        )

    def test_T3_1_avg_step_time(self):
        """100 次 step 平均耗时；>2s/步 失败（须 profile）。"""
        env = self._make_env(num_evs=50, num_stations=2, seed=301)
        env.reset()
        t0 = time.perf_counter()
        for _ in range(100):
            env.step({})
        elapsed = time.perf_counter() - t0
        per_step = elapsed / 100.0
        _diag_print(
            "",
            "[T3.1] 单 step 耗时（100 次 step，不含 reset）",
            f"  total_elapsed={elapsed:.3f}s  avg_per_step={per_step:.3f}s",
            f"  断言上限: {PER_STEP_MAX_SEC}s/步",
            f"  {_per_step_bucket_label(per_step)}",
            "",
        )
        self.assertLess(
            per_step,
            PER_STEP_MAX_SEC,
            msg=(
                f"avg per step={per_step:.3f}s 超过 {PER_STEP_MAX_SEC}s，"
                "训练周期不可接受；请 profile（见 T3.2）并检查 runpp / shortest_path / BPR 缓存"
            ),
        )

    def test_T3_2_cprofile_top_cumulative(self):
        """20 步 cProfile；抓取 top 15 cumulative，确保统计可读（人工看瓶颈）。"""
        env = self._make_env(num_evs=50, num_stations=2, seed=302)
        env.reset()
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(20):
            env.step({})
        profiler.disable()
        buf = io.StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(15)
        out = buf.getvalue()
        _diag_print(
            "",
            "[T3.2] cProfile 20 步 — top 15 cumulative（常见瓶颈：runpp、shortest_path、BPR）",
            "----- pstats begin -----",
            out.rstrip(),
            "----- pstats end -----",
            "",
        )
        self.assertGreater(len(out), 200, "pstats 输出过短，可能未采集到样本")
        # 典型表头含 ncalls / tottime / cumtime
        self.assertIn("cumtime", out)

    @unittest.skipUnless(psutil is not None, "未安装 psutil，跳过 T3.3：pip install psutil")
    def test_T3_3_memory_growth_over_steps(self):
        """reset 后基线 RSS；200 step 内增长 <100MB（常见泄漏：path_cache / 历史未清）。"""
        proc = psutil.Process()
        env = self._make_env(num_evs=50, num_stations=2, seed=303)
        env.reset()
        baseline_mb = proc.memory_info().rss / (1024 * 1024)
        _diag_print(
            "",
            "[T3.3] 内存（RSS）相对 reset 后基线；每 50 step 打一行",
            f"  post_reset_baseline={baseline_mb:.1f} MB",
            f"  阈值：每检查点相对基线增长 < {MEM_GROWTH_MAX_MB:.0f} MB",
            "",
        )
        for i in range(200):
            env.step({})
            if i % 50 == 49:
                cur_mb = proc.memory_info().rss / (1024 * 1024)
                delta = cur_mb - baseline_mb
                _diag_print(
                    f"  step {i + 1}: rss={cur_mb:.1f} MB "
                    f"(delta_vs_post_reset={delta:+.1f} MB)"
                )
                self.assertLess(
                    delta,
                    MEM_GROWTH_MAX_MB,
                    msg=(
                        f"step {i + 1}: rss={cur_mb:.1f} MB, delta_vs_post_reset={delta:+.1f} MB "
                        f"(阈值 {MEM_GROWTH_MAX_MB} MB)"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
