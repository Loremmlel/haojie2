"""真实控制进程：自动保存、连续缩放、停止以及未完成会话恢复。"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path


class Control(unittest.TestCase):
    def test_autosave_resize_stop_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "run"
            command = [
                sys.executable,
                "-m",
                "haojie_training.online",
                "run",
                "--device",
                "cpu",
                "--directory",
                str(directory),
                "--concurrency",
                "2",
                "--workers",
                "2",
                "--save-seconds",
                "0.2",
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
            try:
                deadline = time.monotonic() + 30
                while not (directory / "checkpoint.pt").exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        self.fail(f"自动保存未完成，进程状态 {process.poll()}")
                    time.sleep(0.05)
                control = json.loads((directory / "control.json").read_text())

                def request(action, **values):
                    message = urllib.request.Request(
                        f"http://127.0.0.1:{control['port']}/",
                        data=json.dumps(dict(command=action, **values)).encode(),
                        headers={"Authorization": "Bearer " + control["token"]},
                    )
                    with urllib.request.urlopen(message, timeout=30) as response:
                        return json.load(response)

                for target in (4, 1, 4, 1):
                    result = request("resize", concurrency=target)
                    self.assertEqual(result["target_concurrency"], target)
                before = request("stop")
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr.decode())
                self.assertFalse((directory / "owner.lock").exists())
                restored = Path(folder) / "restored"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "haojie_training.online",
                        "run",
                        "--device",
                        "cpu",
                        "--directory",
                        str(restored),
                        "--resume",
                        str(directory / "checkpoint.pt"),
                        "--seconds",
                        "0.5",
                        "--save-seconds",
                        "0",
                    ],
                    capture_output=True,
                    timeout=30,
                    env={**os.environ, "PYTHONUTF8": "1"},
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                after = json.loads((restored / "saved.json").read_text())
                self.assertGreaterEqual(after["commands"], before["commands"])
                self.assertGreaterEqual(after["consumed_microsteps"], before["consumed_microsteps"])
                self.assertEqual(after["target_concurrency"], 1)
                self.assertGreater(after["inflight"], 0)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
