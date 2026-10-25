"""边界与控制测试；不启动真实训练或完整棋力测评。"""

import ctypes
import json
import os
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from .control import EvaluationControl
from .run import schedule, summarize, run
from .teacher import Teacher


class EvaluationTests(unittest.TestCase):
    def test_schedule_and_unfinished_denominator(self):
        plan = schedule(2)
        self.assertEqual(len(plan), 24)
        for level in ("easy", "medium", "hard"):
            rows = [g for g in plan if g["difficulty"] == level]
            self.assertEqual({g["side"] for g in rows}, {1, 2})
            self.assertEqual({g["rules"] for g in rows}, {"classic", "shrine"})
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
        native.receive.return_value = {"updates": 42}
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
            self.assertEqual(len(report["games"]), 12)
            self.assertTrue(all(g["result"] == "unfinished" for g in report["games"]))
            self.assertTrue(all(g["reason"] == "time-budget" for g in report["games"]))
            self.assertTrue(
                all(v["win_rate"] is None for v in report["summary"].values())
            )
            native.close.assert_called_once()

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
