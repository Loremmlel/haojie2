"""长跑契约：产物复用、防篡改、失败保留、筛选拒收和Windows进程树回收。"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from haojie_training.data import load_dataset, synthetic_batch
from haojie_training.model import ModelConfig

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/training/search/continuous"
sys.path.insert(0, str(SCRIPTS))
from dataset import combine, save_split  # noqa: E402
from run import choose  # noqa: E402
from runtime import Stages, read  # noqa: E402


class ContinuousTests(unittest.TestCase):
    def test_history_pool_preserves_parameter_prefixes_and_masks_value(self):
        config = ModelConfig.tiny()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specification = {}
            for split in ("train", "validation"):
                folder = root / split
                folder.mkdir()
                batch = synthetic_batch(config, 2, 3, 4)
                batch["value"] = torch.tensor([1.0, 0.0])
                batch["value_mask"] = torch.tensor([True, False])
                records = [
                    {
                        "group": split,
                        "game_id": split,
                        "index": 0,
                        "step": step,
                        "actor": 1,
                        "command": "deploy",
                        "stage": stage,
                    }
                    for step, stage in enumerate(("action", "point"))
                ]
                metadata = {
                    "ruleset": "test",
                    "encoding": "test",
                    "schema": {},
                    "source_sha256": "same",
                    "synthetic": True,
                }
                save_split(folder, "decisions", [batch], metadata, records)
                specification[split] = [str(folder / "decisions.pt")]
            output = root / "combined"
            output.mkdir()
            report = combine(specification, output)
            for split in ("train", "validation"):
                data, metadata = load_dataset(output / f"{split}.pt", config)
                self.assertEqual([r["step"] for r in metadata["records"]], [0, 1])
                self.assertEqual(report["splits"][split]["examples"], 2)
                self.assertEqual(report["splits"][split]["value_labels"], 1)
                torch.testing.assert_close(data["value_mask"], torch.tensor([True, False]))

    def test_completed_stage_reuses_outputs_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = Stages(root, (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), 0)

            def build(out):
                return [
                    sys.executable,
                    "-c",
                    "from pathlib import Path;Path('value').write_text('done')".replace(
                        "'value'", repr(str(out / "value"))
                    ),
                ], ["value"]

            path = stages.run("first", {"seed": 1}, build)
            before = (path / "value").stat().st_mtime_ns
            self.assertEqual(
                stages.run("first", {"seed": 1}, lambda _: self.fail("不应再次启动")), path
            )
            self.assertEqual((path / "value").stat().st_mtime_ns, before)
            with self.assertRaisesRegex(ValueError, "配置变化"):
                stages.run("first", {"seed": 2}, build)
            (path / "value").write_text("modified")
            with self.assertRaisesRegex(ValueError, "被修改"):
                stages.run("first", {"seed": 1}, build)

    def test_failed_attempt_is_preserved_and_only_that_stage_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            stages = Stages(
                directory, (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), 0
            )

            def build(out):
                code = (
                    f"from pathlib import Path;Path({str(out / 'value')!r}).write_text('partial')"
                )
                if out.name == "attempt-00":
                    code += ";raise SystemExit(7)"
                return [sys.executable, "-c", code], ["value"]

            path = stages.run("sample", "fixed", build)
            self.assertEqual(path.name, "attempt-01")
            self.assertTrue((path.parent / "attempt-00/failure.json").exists())
            self.assertEqual(read(path.parent / "done.json")["attempt"], "attempt-01")

    def test_truncation_or_teacher_regression_prevents_promotion(self):
        def result(winner=1, primary=1, terminated=True):
            return {
                "winner": winner,
                "primary": primary,
                "terminated": terminated,
                "truncated": not terminated,
            }

        winning = [result()] * 8
        self.assertTrue(choose(winning, 2)["accepted"])
        self.assertFalse(choose([result(terminated=False), *winning[1:]], 2)["accepted"])
        self.assertFalse(
            choose([*winning[:4], result(winner=2), result(winner=2), *winning[6:]], 2)["accepted"]
        )
        self.assertFalse(choose(winning[:3], 2)["accepted"])
        self.assertFalse(
            choose([*winning[:4], *([result(winner=2)] * 4)], 2)["accepted"],
            "候选和父代都输给教师，不能仅靠打败弱父代晋升",
        )
        self.assertIsNone(
            choose([result(terminated=False), *winning[1:]], 2)["candidate_vs_parent_score"]
        )

    @unittest.skipUnless(os.name == "nt", "本机Windows进程树契约")
    def test_killed_host_reaps_child_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child.txt"
            lock = Path(directory) / "run.lock"
            code = (
                f"import sys,subprocess,time;sys.path.insert(0,{str(SCRIPTS)!r});"
                "from runtime import contain_process_tree,lock_run;"
                "from pathlib import Path;job=contain_process_tree();"
                f"lock=lock_run(Path({str(lock)!r}));"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],"
                "creationflags=subprocess.CREATE_NO_WINDOW);"
                f"Path({str(marker)!r}).write_text(str(child.pid));time.sleep(120)"
            )
            host = subprocess.Popen(
                [sys.executable, "-c", code], creationflags=subprocess.CREATE_NO_WINDOW
            )
            try:
                for _ in range(100):
                    if marker.exists():
                        break
                    time.sleep(0.1)
                self.assertTrue(marker.exists())
                pid = int(marker.read_text())
                host.kill()
                host.wait(timeout=10)
                import ctypes
                from ctypes import wintypes as w

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.restype = w.HANDLE
                kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
                kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
                kernel.CloseHandle.argtypes = [w.HANDLE]
                for _ in range(100):
                    handle = kernel.OpenProcess(0x1000, False, pid)
                    if not handle:
                        break
                    status = w.DWORD()
                    kernel.GetExitCodeProcess(handle, ctypes.byref(status))
                    kernel.CloseHandle(handle)
                    if status.value != 259:
                        break
                    time.sleep(0.1)
                else:
                    self.fail("强杀宿主后遗留子进程")
                from runtime import lock_run

                # Windows强制退出后的文件锁回收可能晚于进程句柄变为已退出。
                for _ in range(100):
                    try:
                        renewed = lock_run(lock)
                        renewed.close()
                        break
                    except RuntimeError:
                        time.sleep(0.05)
                else:
                    self.fail("强杀宿主后独占锁未释放")
            finally:
                if host.poll() is None:
                    host.kill()
                    host.wait()


if __name__ == "__main__":
    unittest.main()
