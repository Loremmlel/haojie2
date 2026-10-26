"""边界与控制测试；原生零并发宿主验证保存回执和测评启动，不执行学习。"""

import ctypes
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from .control import EvaluationControl
from .run import schedule, summarize, run
from .teacher import Teacher
from training.colab.panel import Panel


class EvaluationTests(unittest.TestCase):
    def test_running_native_save_receipt_and_evaluation(self):
        import torch

        root = Path(__file__).resolve().parents[2]
        binary = (
            root
            / "artifacts/cpp-training-build"
            / ("Release/haojie-train.exe" if os.name == "nt" else "haojie-train")
        )
        libraries = [
            Path(torch.__file__).parent / "lib",
            root / "artifacts/native-target/release",
        ]
        environment = os.environ.copy()
        environment["PATH"] = (
            os.pathsep.join(map(str, libraries))
            + os.pathsep
            + environment.get("PATH", "")
        )
        with (
            tempfile.TemporaryDirectory(prefix="haojie-save-eval-") as temporary,
            patch.dict(os.environ, environment),
        ):
            directory = Path(temporary) / "run"
            with open(Path(temporary) / "native.log", "w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        str(binary),
                        "run",
                        "--directory",
                        str(directory),
                        "--device",
                        "cpu",
                        "--concurrency",
                        "0",
                        "--workers",
                        "1",
                        "--threads",
                        "1",
                        "--seconds",
                        "45",
                        "--save-seconds",
                        "0",
                    ],
                    stdout=log,
                    stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                panel = Panel(directory, process, {"device": "cpu"})
                control = EvaluationControl(
                    directory, process, [str(binary)], sys.executable, "cpu"
                )

                def wait_for(predicate):
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        self.assertIsNone(process.poll(), "原生宿主提前退出")
                        if predicate():
                            return
                        time.sleep(0.1)
                    self.fail(f"真实保存/测评未完成：{control.status()}")

                try:
                    wait_for(lambda: panel.snapshot()["can_control"])
                    self.assertTrue(panel.call(panel.identity, "save")["save_pending"])
                    wait_for(lambda: not panel.snapshot()["save_pending"])
                    self.assertTrue((directory / "checkpoint.pt").exists())
                    self.assertFalse((directory / "window.json").exists())
                    control.request()
                    wait_for(
                        lambda: (
                            (control.status()["report"] or {}).get("phase") == "running"
                        )
                    )
                    report = control.status()["report"]
                    self.assertEqual(report["rules"], "classic")
                    self.assertEqual(len(report["games"]), 6)
                    self.assertTrue(
                        all(g["rules"] == "classic" for g in report["games"])
                    )
                    self.assertFalse(control.pending)
                    self.assertIsNone(process.poll())
                finally:
                    control.cancel()
                    child = control.child
                    if child:
                        child.wait(timeout=30)
                    control.close()
                    if directory.exists():
                        (directory / "stop").write_text("1", encoding="utf-8")
                    process.wait(timeout=30)
                self.assertEqual(process.returncode, 0)

    def test_schedule_and_unfinished_denominator(self):
        for rules in ("classic", "shrine"):
            plan = schedule(2, rules)
            self.assertEqual(len(plan), 12)
            for level in ("easy", "medium", "hard"):
                rows = [g for g in plan if g["difficulty"] == level]
                self.assertEqual({g["side"] for g in rows}, {1, 2})
                self.assertEqual({g["rules"] for g in rows}, {rules})
                for seed in {g["seed"] for g in rows}:
                    self.assertEqual(
                        sorted(g["side"] for g in rows if g["seed"] == seed), [1, 2]
                    )
        plan[0]["result"] = "win"
        plan[3]["result"] = "unfinished"
        result = summarize(plan)
        self.assertEqual(result["easy"]["win_rate"], 1)
        self.assertIsNone(result["hard"]["win_rate"])

    def test_cancelled_and_timeout_games_remain_unknown(self):
        native = Mock()
        native.receive.return_value = {"updates": 42, "rules": "classic"}
        native.request.side_effect = InterruptedError()
        native.inflight = False
        teacher = Mock()
        teacher.manifest = {"sha256": "fixture"}
        teacher.budget = {}
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("training.evaluation.run.Native", return_value=native),
            patch("training.evaluation.run.Teacher", return_value=teacher),
        ):
            report = run([], Path("fixture"), "cpu", Path(temporary))
            self.assertEqual(len(report["games"]), 6)
            self.assertTrue(all(g["result"] == "unfinished" for g in report["games"]))
            self.assertTrue(all(g["reason"] == "time-budget" for g in report["games"]))
            self.assertTrue(
                all(v["win_rate"] is None for v in report["summary"].values())
            )
            native.close.assert_called_once()

    def test_runner_uses_frozen_checkpoint_rules(self):
        for rules in ("classic", "shrine"):
            with self.subTest(rules=rules), tempfile.TemporaryDirectory() as temporary:
                native = Mock()
                native.receive.return_value = {"updates": 42, "rules": rules}
                native.request.return_value = {"winner": 1, "commands": 3}
                teacher = Mock(manifest={"sha256": "fixture"}, budget={})
                with (
                    patch("training.evaluation.run.Native", return_value=native),
                    patch("training.evaluation.run.Teacher", return_value=teacher),
                ):
                    report = run([], Path("fixture"), "cpu", Path(temporary))
                self.assertEqual(report["phase"], "finished")
                self.assertEqual(report["rules"], rules)
                self.assertEqual(len(native.request.call_args_list), 6)
                self.assertTrue(
                    all(
                        call.args[0].split()[-1] == str(int(rules == "shrine"))
                        for call in native.request.call_args_list
                    )
                )
                self.assertTrue(
                    all(
                        counts["win"] == counts["loss"] == 1
                        for counts in report["summary"].values()
                    )
                )

    def test_unknown_checkpoint_rules_do_not_start_games(self):
        native = Mock()
        native.receive.return_value = {"updates": 42}
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("training.evaluation.run.Native", return_value=native),
            patch("training.evaluation.run.Teacher") as teacher,
        ):
            report = run([], Path("fixture"), "cpu", Path(temporary))
        self.assertEqual(report["phase"], "failed")
        self.assertIn("训练模式", report["error"])
        self.assertEqual(report["games"], [])
        native.request.assert_not_called()
        teacher.assert_not_called()

    def test_background_schedule_without_browser_and_cancel(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(EvaluationControl, "launch") as launch,
        ):
            control = EvaluationControl(
                temporary, Mock(poll=Mock(return_value=None)), [], "python", "cpu"
            )
            try:
                control.configure({"minutes": 1, "pairs": 1, "seconds": 10})
                control.next_due = 0
                deadline = time.monotonic() + 3
                while not launch.called and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(launch.called)
                with self.assertRaises(ValueError):
                    control.configure({"minutes": True, "pairs": 1, "seconds": 10})
                control.cancel()
                self.assertFalse(control.pending)
            finally:
                control.close()

    def test_real_teacher_public_bridge(self):
        root = Path(__file__).resolve().parents[2]
        library = (
            root
            / "artifacts/native-target/release"
            / ("haojie_native.dll" if os.name == "nt" else "libhaojie_native.so")
        )
        api = ctypes.CDLL(str(library))
        api.hj_actor_create.restype = ctypes.c_void_p
        api.hj_actor_call.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        api.hj_actor_close.argtypes = [ctypes.c_void_p]
        actor = api.hj_actor_create()

        def call(op, data):
            pointer, size = ctypes.c_void_p(), ctypes.c_size_t()
            code = api.hj_actor_call(
                actor, op, data, len(data), ctypes.byref(pointer), ctypes.byref(size)
            )
            output = ctypes.string_at(pointer, size.value)
            self.assertEqual(code, 0, output.decode(errors="replace"))
            return output

        try:
            for game, (level, shrine) in enumerate(
                (("easy", 0), ("medium", 1), ("hard", 1)), 1
            ):
                call(0, struct.pack("<QIII", game, 73127, 1, shrine))
                teacher = Teacher(level)
                try:
                    for _ in range(2):
                        state = json.loads(call(7, struct.pack("<Q", game)))
                        self.assertNotIn("rng", state["observation"])
                        self.assertNotIn("seed", state["observation"])
                        action = teacher.next(state, lambda: False)
                        call(8, struct.pack("<Q", game) + json.dumps(action).encode())
                finally:
                    teacher.close()
        finally:
            api.hj_actor_close(actor)


if __name__ == "__main__":
    unittest.main()
