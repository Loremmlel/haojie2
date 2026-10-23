"""服务器配额、旧配置升级、安全点和有限资源监测。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from haojie_training.console import config
from haojie_training.console.config import defaults, resources, upgrade, validate
from haojie_training.console.controller import Controller
from haojie_training.console.storage import Store


class ServerTests(unittest.TestCase):
    def test_allocator_default_and_user_environment_without_torch(self):
        variables = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        training = Path(__file__).resolve().parents[2]
        code = (
            "import json, os, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import haojie_training\n"
            "assert 'torch' not in sys.modules\n"
            "print(json.dumps({key: os.environ.get(key) for key in sys.argv[2:]}))\n"
        )
        for supplied in (
            {},
            {variables[0]: "expandable_segments:False"},
            {variables[1]: "max_split_size_mb:128"},
            {variables[0]: "backend:cudaMallocAsync", variables[1]: "expandable_segments:False"},
            {variables[0]: ""},
            {variables[1]: ""},
        ):
            with self.subTest(environment=supplied):
                environment = {
                    key: value for key, value in os.environ.items() if key not in variables
                }
                environment.update(supplied)
                result = subprocess.run(
                    [sys.executable, "-S", "-c", code, str(training), *variables],
                    env=environment,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                expected = {key: supplied.get(key) for key in variables}
                if not supplied:
                    expected[variables[0]] = "expandable_segments:True"
                self.assertEqual(json.loads(result.stdout), expected)

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
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device"),
                patch.object(torch.cuda, "is_bf16_supported", return_value=True),
            ):
                self.assertEqual(resources.cpu_capacity(), 2)
                self.assertEqual(resources.torch_threads("cpu"), 2)
                self.assertEqual(resources.available_memory(), 2**30)
                profile = defaults("cuda")
                validate(profile)
                self.assertEqual(profile["environments"], 1)
                self.assertEqual(profile["games_per_round"], 2)
                self.assertEqual(profile["pool_mib"], 256)
                self.assertNotIn("sampling_precision", profile)

    def test_shared_defaults_scale_and_new_experiments_keep_explicit_settings(self):
        with (
            patch.object(config, "cpu_capacity", return_value=40),
            patch.object(config, "available_memory", return_value=64 * 2**30),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device"),
            patch.object(torch.cuda, "is_bf16_supported", return_value=True),
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(Controller, "_loop"),
            tempfile.TemporaryDirectory() as folder,
        ):
            automatic = defaults()
            self.assertEqual(automatic["environments"], 80)
            self.assertEqual(automatic["games_per_round"], 160)
            self.assertEqual(automatic["inference_batch_size"], 32)
            self.assertEqual(automatic["batch_size"], 8)
            c = Controller(Path(folder) / "unused-engine", Store(folder), device="cpu")
            try:
                self.assertEqual(c.config, defaults("cpu"))
                self.assertEqual(c.config["environments"], 40)
                c.command(
                    "new-experiment",
                    {"device": "cuda:1", "config": {"classic_percent": 100, "batch_size": 3}},
                )
                self.assertEqual(c.new_request[1], torch.device("cuda:1"))
                self.assertEqual(
                    c.new_request[2], {**automatic, "classic_percent": 100, "batch_size": 3}
                )
                validate(c.new_request[2])
            finally:
                c.close()
            with patch.object(config, "cpu_capacity", return_value=128):
                self.assertEqual(defaults("cuda")["environments"], 128)
            with patch.object(config, "resolve_device", return_value=torch.device("xpu")):
                self.assertEqual(defaults()["environments"], 80)

    def test_initialization_uses_single_numeric_trainer_and_current_thread_policy(self):
        c = Controller.__new__(Controller)
        c.device = torch.device("cuda:1")
        c.requested_threads = 0
        c.trainer = Mock()
        c._device = Mock()
        with (
            patch.object(torch, "get_num_threads", return_value=4),
            patch.object(torch, "set_num_threads") as set_threads,
            patch("haojie_training.console.config.resources.resolve_device", return_value=c.device),
        ):
            c._initialize()
            set_threads.assert_called_once_with(1)
            c.requested_threads = 3
            c._initialize()
            set_threads.assert_called_with(3)
            c.requested_threads = None
            set_threads.reset_mock()
            c._initialize()
            set_threads.assert_not_called()

        c.trainer, c.imported, c.tiny = None, None, True
        c.engine, c.store, c.config = "unused", Mock(), defaults("cpu")
        c.store.names.return_value = []

        with (
            patch("haojie_training.console.controller.Client") as client,
            patch("haojie_training.console.config.resources.resolve_device", return_value=c.device),
            patch("haojie_training.console.controller.metadata", return_value={}),
            patch("haojie_training.console.controller.PolicyValueNet"),
            patch(
                "haojie_training.console.controller.Trainer", side_effect=RuntimeError("done")
            ) as trainer,
            patch.object(torch, "manual_seed"),
        ):
            client.return_value.__enter__.return_value.ready = {"capabilities": ["mc-context-v2"]}
            with self.assertRaisesRegex(RuntimeError, "done"):
                c._initialize()
            self.assertNotIn("precision", trainer.call_args.kwargs)
            self.assertNotIn("optimizer", trainer.call_args.kwargs)

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
        original = {**defaults(), "inference_batch_size": 32}
        for precision in ("fp32", "bf16", "fp16"):
            self.assertEqual(upgrade({**original, "sampling_precision": precision}), original)
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
