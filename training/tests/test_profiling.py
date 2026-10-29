"""诊断开销开关、嵌套计时和无同步事件收割的 CPU 验收。"""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from training.haojie.profiling import Profile, TraceWindow
from training.haojie.runtime import pipeline


class ProfilingTests(unittest.TestCase):
    def test_disabled_never_reads_clock_or_creates_cuda_event(self):
        with (
            patch.dict(os.environ, {"HAOJIE_PROFILE": "0"}),
            patch(
                "training.haojie.profiling.time.perf_counter",
                side_effect=AssertionError("关闭诊断时读取了墙钟"),
            ),
            patch(
                "training.haojie.profiling.time.thread_time",
                side_effect=AssertionError("关闭诊断时读取了线程时钟"),
            ),
            patch(
                "training.haojie.profiling.torch.cuda.Event",
                side_effect=AssertionError("关闭诊断时创建了 CUDA Event"),
            ),
        ):
            profile = Profile()
            self.assertFalse(profile.select("encode", 32, 4))
            profile.observe("batch", 16)
            with profile.stage("encode", cuda=True):
                pass
            report = profile.report()
        self.assertEqual(report["host_stages"], {})
        self.assertEqual(report["stream_stages"], {})
        self.assertEqual(report["sampling"], {})
        self.assertEqual(report["pending_events"], 0)
        self.assertEqual(report["observations"], {})

    def test_batch_observations_and_delayed_trace(self):
        with patch.dict(os.environ, {"HAOJIE_PROFILE": "1"}):
            profile = Profile()
            for value in (2, 32, 33):
                profile.observe("batch", value)
            self.assertEqual(
                profile.report()["observations"]["batch"],
                {
                    "count": 3,
                    "sum": 67,
                    "max": 33,
                    "histogram": {"2": 1, "32": 1, "64": 1},
                },
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "delayed.json"
            with (
                patch.dict(
                    os.environ,
                    {"HAOJIE_TRACE": str(path), "HAOJIE_TRACE_START_SECONDS": "20"},
                ),
                patch(
                    "training.haojie.profiling.time.perf_counter",
                    side_effect=[0, 19, 20],
                ),
                patch("training.haojie.profiling.torch.profiler.profile"),
            ):
                trace = TraceWindow(Profile(), "cpu")
                trace.step({})
                trace.step({})
                self.assertIsNone(trace.profiler)
                trace.step({})
                trace.complete({"updates": 1})
                trace.close()
            meta = json.loads(
                path.with_suffix(".meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["start"]["iteration"], 3)
            self.assertEqual(meta["completed_loops"], 1)

    def test_nested_exclusive_times_do_not_count_children_twice(self):
        with (
            patch.dict(os.environ, {"HAOJIE_PROFILE": "1"}),
            patch(
                "training.haojie.profiling.time.perf_counter",
                side_effect=[0, 2, 3, 5, 8, 10],
            ),
            patch(
                "training.haojie.profiling.time.thread_time",
                side_effect=[0, 1, 2, 3, 5, 6],
            ),
        ):
            profile = Profile()
            with profile.stage("outer"):
                with profile.stage("middle"):
                    with profile.stage("inner"):
                        pass
            stages = profile.report()["host_stages"]
        for name, wall, exclusive_wall, cpu, exclusive_cpu in (
            ("outer", 10, 4, 6, 2),
            ("middle", 6, 4, 4, 3),
            ("inner", 2, 2, 1, 1),
        ):
            self.assertEqual(
                stages[name],
                {
                    "count": 1,
                    "inclusive_wall_seconds": wall,
                    "exclusive_wall_seconds": exclusive_wall,
                    "inclusive_thread_cpu_seconds": cpu,
                    "exclusive_thread_cpu_seconds": exclusive_cpu,
                },
            )
        self.assertEqual(sum(s["exclusive_wall_seconds"] for s in stages.values()), 10)
        self.assertEqual(
            sum(s["exclusive_thread_cpu_seconds"] for s in stages.values()), 6
        )

    def test_cuda_events_wait_for_query_without_synchronizing(self):
        start, end = Mock(), Mock()
        start.elapsed_time.return_value = 12.5
        end.query.side_effect = [False, True]
        for event in (start, end):
            event.synchronize.side_effect = AssertionError("诊断调用了事件同步")
        with (
            patch.dict(os.environ, {"HAOJIE_PROFILE": "1"}),
            patch(
                "training.haojie.profiling.torch.cuda.Event", side_effect=[start, end]
            ),
            patch(
                "training.haojie.profiling.torch.cuda.synchronize",
                side_effect=AssertionError("诊断调用了设备同步"),
            ),
        ):
            profile = Profile()
            with profile.stage("encode", cuda=True):
                pass
            waiting = profile.report()
            self.assertEqual(waiting["pending_events"], 1)
            self.assertEqual(waiting["stream_stages"], {})
            start.elapsed_time.assert_not_called()
            completed = profile.report()
            self.assertEqual(completed["pending_events"], 0)
            self.assertEqual(
                completed["stream_stages"]["encode"],
                {"count": 1, "stream_elapsed_seconds": 0.0125},
            )
            self.assertEqual(profile.report(), completed)
        start.record.assert_called_once_with()
        end.record.assert_called_once_with()
        self.assertEqual(end.query.call_count, 2)
        start.elapsed_time.assert_called_once_with(end)

    def test_trace_counts_completed_loops_and_keeps_final_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            for complete_count in (0, 1, 32):
                with self.subTest(complete_count=complete_count):
                    path = Path(directory) / f"trace-{complete_count}.json"
                    with (
                        patch.dict(
                            os.environ,
                            {"HAOJIE_PROFILE": "0", "HAOJIE_TRACE": str(path)},
                        ),
                        patch("training.haojie.profiling.torch.profiler.profile"),
                    ):
                        trace = TraceWindow(Profile(), "cpu")
                        for _ in range(63):
                            trace.step({"updates": 0})
                        trace.step({"updates": 0})
                        for index in range(complete_count):
                            if index:
                                trace.step({"updates": index})
                            trace.complete({"updates": index + 1})
                        if complete_count == 32:
                            trace.step({"updates": 32})
                        else:
                            # 零个完整循环模拟首圈中途异常；最终计数也须保留。
                            trace.close({"updates": 7 if complete_count == 0 else 1})
                    metadata = json.loads(
                        path.with_suffix(".meta.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(metadata["completed_loops"], complete_count)
                    self.assertEqual(metadata["partial"], complete_count < 32)
                    self.assertEqual(
                        metadata["last_completed_iteration"],
                        63 + complete_count if complete_count else None,
                    )
                    self.assertEqual(
                        metadata["incomplete_iteration"],
                        64 if complete_count == 0 else None,
                    )
                    self.assertEqual(
                        metadata["end"]["stats"]["updates"],
                        7 if complete_count == 0 else complete_count,
                    )

    def test_run_discards_previous_profile_counters(self):
        trainer = SimpleNamespace(
            stats={
                "updates": 9,
                "profile_gpu": {"old": True},
                "profile_native": {"old": True},
                "profile_coordinator": {"old": True},
            },
            model=Mock(),
            optimizer=Mock(),
            config=object(),
            published=9,
            pending_versions={},
            weights=Mock(),
            live_versions=Mock(return_value={9}),
            pool=SimpleNamespace(entries=[]),
            directory="unused",
        )
        # 在创建 GPU 所有者前停止，只检查运行入口的恢复清理。
        with patch.object(
            pipeline, "Executor", side_effect=RuntimeError("检查到此为止")
        ):
            with self.assertRaisesRegex(RuntimeError, "检查到此为止"):
                pipeline.run(trainer)
        self.assertEqual(trainer.stats, {"updates": 9})


if __name__ == "__main__":
    unittest.main()
