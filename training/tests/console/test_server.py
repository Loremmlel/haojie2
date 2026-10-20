"""服务器配额、旧配置升级、安全点和有限资源监测。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from haojie_training.console.config import defaults, resources, upgrade, validate
from haojie_training.console.controller import Controller
from haojie_training.console.storage import Store


class ServerTests(unittest.TestCase):
    def test_notebook_and_bootstrap_syntax(self):
        import ast
        import asyncio
        import html
        import json

        repo = Path(__file__).resolve().parents[3]
        notebook = json.loads((repo / "notebooks/haojie_colab.ipynb").read_text(encoding="utf-8"))
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code), 6)
        for cell in code:
            ast.parse("".join(cell["source"]))
            self.assertEqual(cell["outputs"], [])
        ast.parse((repo / "scripts/colab/bootstrap.py").read_text(encoding="utf-8"))
        # 执行监控单元的一次刷新，核对公开状态字段，而不是只检查语法。
        view = Mock()

        async def stop_display(_):
            raise asyncio.CancelledError

        scope = {
            "display": lambda *args, **kwargs: view,
            "print": Mock(),
            "HTML": str,
            "html": html,
            "json": json,
            "asyncio": SimpleNamespace(sleep=stop_display, CancelledError=asyncio.CancelledError),
            "api": lambda: {"resources": {"cpu_percent": 25}},
        }
        monitor = compile(
            "".join(code[-2]["source"]), "monitor-cell", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        )
        asyncio.run(eval(monitor, scope))
        self.assertEqual(scope["shown"]["resources"], {"cpu_percent": 25})
        view.update.assert_called_once()
        scope["print"].assert_called_once()

    def test_parent_cpu_and_memory_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            child = parent / "worker"
            child.mkdir()
            (parent / "cpu.max").write_text("200000 100000")
            (child / "cpu.max").write_text("400000 100000")
            (parent / "memory.max").write_text(str(4 * 2**30))
            (parent / "memory.current").write_text(str(3 * 2**30))
            (child / "memory.max").write_text(str(2 * 2**30))
            (child / "memory.current").write_text(str(256 * 2**20))
            with (
                patch.object(resources, "_cgroups", return_value=(child, parent)),
                patch.object(resources.os, "cpu_count", return_value=64),
                patch.object(
                    resources.os, "sched_getaffinity", return_value=set(range(8)), create=True
                ),
                patch.object(
                    resources.psutil,
                    "virtual_memory",
                    return_value=SimpleNamespace(total=64 * 2**30, available=48 * 2**30),
                ),
            ):
                self.assertEqual(resources.cpu_capacity(), 2)
                self.assertEqual(resources.torch_threads("cpu"), 2)
                self.assertEqual(resources.available_memory(), 2**30)
                profile = resources.server_defaults("cuda")
                validate(profile)
                self.assertEqual(profile["environments"], 4)
                self.assertEqual(profile["games_per_round"], 8)
                self.assertEqual(profile["pool_mib"], 256)

    def test_cpu_threads_respect_quota_explicit_override_and_auto_device(self):
        for capacity, expected in ((0.5, 1), (3.9, 3), (16, 4)):
            with (
                self.subTest(capacity=capacity),
                patch.object(resources, "cpu_capacity", return_value=capacity),
            ):
                self.assertEqual(resources.torch_threads("cpu"), expected)
                self.assertEqual(resources.torch_threads("cpu", 8), 8)
                self.assertEqual(resources.torch_threads("cuda", 2), 2)
        with self.assertRaisesRegex(ValueError, "非负"):
            resources.torch_threads("cpu", -1)
        for cuda, xpu, expected in ((False, False, 3), (True, False, 1), (True, True, 1)):
            with (
                self.subTest(cuda=cuda, xpu=xpu),
                patch.object(resources, "cpu_capacity", return_value=3.9),
                patch.object(torch.cuda, "is_available", return_value=cuda),
                patch.object(torch.xpu, "is_available", return_value=xpu),
            ):
                self.assertEqual(resources.torch_threads(), expected)

    def test_clean_page_cache_can_be_reclaimed_but_dirty_cache_cannot(self):
        for v2 in (True, False):
            with self.subTest(v2=v2), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / ("memory.max" if v2 else "memory.limit_in_bytes")).write_text(
                    str(8 * 2**30)
                )
                (root / ("memory.current" if v2 else "memory.usage_in_bytes")).write_text(
                    str(8 * 2**30 - 256 * 2**20)
                )
                prefix = "" if v2 else "total_"
                stats = root / "memory.stat"
                stats.write_text(
                    f"{prefix}inactive_file {4 * 2**30}\n"
                    f"{'file_dirty' if v2 else 'total_dirty'} {512 * 2**20}\n"
                    f"{'file_writeback' if v2 else 'total_writeback'} {256 * 2**20}\n"
                )
                with patch.object(resources, "_cgroups", return_value=(root,)):
                    self.assertEqual(resources.available_memory(), 3584 * 2**20)
                    stats.write_text("")
                    self.assertEqual(resources.available_memory(), 256 * 2**20)

    def test_old_recovery_options_and_server_ranges(self):
        original = defaults()
        legacy = {
            k: v
            for k, v in original.items()
            if k not in {"inference_batch_size", "batch_wait_ms", "sampling_precision"}
        }
        self.assertEqual(upgrade(legacy), original)
        validate(
            {
                **original,
                "environments": 128,
                "games_per_round": 256,
                "batch_size": 128,
                "pool_mib": 32768,
                "batch_wait_ms": 0.5,
            }
        )
        with self.assertRaises(ValueError):
            validate({**original, "batch_wait_ms": float("nan")})
        with self.assertRaises(ValueError):
            upgrade({**legacy, "unexpected": 1})

    def test_snapshot_reports_initialization_failure_and_closed_service(self):
        with tempfile.TemporaryDirectory() as folder:
            controller = Controller(
                Path(folder) / "haojie-engine", Store(folder), device="cpu", tiny=True
            )
            try:
                future = controller.snapshot(lambda _: self.fail("初始化失败时不应冻结快照"))
                with self.assertRaises(FileNotFoundError):
                    future.result(timeout=5)
            finally:
                controller.close()
            with self.assertRaisesRegex(RuntimeError, "已关闭"):
                controller.snapshot(lambda _: None).result(timeout=1)

    def test_monitor_preserves_other_metrics_without_process_access(self):
        c = SimpleNamespace(device=torch.device("cpu"), shutdown=Mock(), telemetry={})
        c.shutdown.wait.side_effect = [False, True]
        with (
            patch.object(resources.psutil, "Process", side_effect=resources.psutil.AccessDenied()),
            patch.object(resources.time, "monotonic", return_value=10.0),
        ):
            resources.monitor(c)
        self.assertFalse(c.telemetry["process_tree_available"])
        self.assertIsNone(c.telemetry["cpu_cores"])
        self.assertGreater(c.telemetry["cpu_capacity"], 0)
        self.assertGreaterEqual(c.telemetry["available_ram"], 0)

    def test_monitor_zero_elapsed_preserves_memory_and_recovers_next_sample(self):
        c = SimpleNamespace(device=torch.device("cpu"), shutdown=Mock(), telemetry={})
        baseline = {(1, 0): {"pid": 1, "name": "parent", "cpu": 1, "reaped_cpu": 0, "rss": 64}}
        next_sample = {(1, 0): {"pid": 1, "name": "parent", "cpu": 2, "reaped_cpu": 0, "rss": 128}}
        snapshots = []

        def wait(_):
            if c.telemetry:
                snapshots.append(c.telemetry.copy())
            return len(snapshots) == 2

        c.shutdown.wait.side_effect = wait
        with (
            patch.object(resources, "_processes", side_effect=[baseline, baseline, next_sample]),
            patch.object(resources.time, "monotonic", side_effect=[10.0, 10.0, 12.0]),
            patch.object(resources, "cpu_capacity", return_value=2),
        ):
            resources.monitor(c)
        self.assertTrue(snapshots[0]["process_tree_available"])
        self.assertEqual(snapshots[0]["rss"], 64)
        self.assertIsNone(snapshots[0]["cpu_cores"])
        self.assertIsNone(snapshots[0]["cpu_percent"])
        self.assertIsNone(snapshots[0]["processes"][0]["cpu_cores"])
        self.assertEqual(snapshots[1]["rss"], 128)
        self.assertEqual(snapshots[1]["cpu_cores"], 0.5)
        self.assertEqual(snapshots[1]["cpu_percent"], 25)

    def test_monitor_includes_reaped_worker_cpu(self):
        c = SimpleNamespace(device=torch.device("cpu"), shutdown=Mock(), telemetry={})
        c.shutdown.wait.side_effect = [False, True]
        before = {
            (1, 0): {"pid": 1, "name": "parent", "cpu": 1, "reaped_cpu": 5, "rss": 1},
            (2, 0): {"pid": 2, "name": "worker", "cpu": 2, "reaped_cpu": 0, "rss": 1},
        }
        after = {(1, 0): {"pid": 1, "name": "parent", "cpu": 1.5, "reaped_cpu": 8, "rss": 1}}
        with (
            patch.object(resources, "_processes", side_effect=[before, after]),
            patch.object(resources.time, "monotonic", side_effect=[10.0, 12.0]),
            patch.object(resources, "cpu_capacity", return_value=2),
        ):
            resources.monitor(c)
        self.assertEqual(c.telemetry["cpu_cores"], 0.75)
        self.assertEqual(c.telemetry["cpu_percent"], 37.5)
        self.assertEqual(c.telemetry["workers"], 0)
