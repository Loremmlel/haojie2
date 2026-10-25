"""面板使用临时状态与进程替身验证，不启动训练。"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from training.colab.panel import Panel


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="haojie-panel-check-")
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        self.process = Mock()
        self.process.poll.return_value = None
        self.panel = Panel(self.run, self.process, {"batch": 512, "concurrency": 8})
        self.panel.resources = Mock()
        self.panel.resources.sample.return_value = {"cpu": 40, "gpu": 70}

    def status(self, seconds=10, **changes):
        self.run.mkdir(exist_ok=True)
        metrics = {
            "seconds": seconds,
            "updates": 1,
            "loss_update": 1,
            "latest_loss": 0.25,
            "inference_nodes": 1000,
            "commands": 300,
            "consumed_within_lag": 512,
            "ready_steps": 128,
            "consumed_lag_max": 8,
            **changes,
        }
        (self.run / "status.json").write_text(json.dumps(metrics), encoding="utf-8")
        self.panel.sampled = None
        return self.panel.call(self.panel.identity)

    def test_observation_rates_and_native_loss(self):
        first = self.status()
        self.assertNotIn("consume", first["history"][-1])
        second = self.status(
            15,
            inference_nodes=2000,
            commands=400,
            consumed_within_lag=1024,
            latest_loss=None,
        )
        self.assertEqual(second["history"][-1]["actor"], 200)
        self.assertEqual(second["history"][-1]["commands"], 20)
        self.assertEqual(second["history"][-1]["consume"], 102.4)
        self.assertIsNone(second["history"][-1]["loss"])
        self.assertEqual(sorted(p.name for p in self.run.iterdir()), ["status.json"])

    def test_control_acknowledges_request_then_actual_save(self):
        self.status()
        for value in (-1, 129, True, 1.5, "8"):
            self.assertIn(
                "error", self.panel.call(self.panel.identity, "resize", value)
            )
        self.assertFalse((self.run / "resize").exists())
        self.panel.call(self.panel.identity, "resize", 0)
        self.assertEqual((self.run / "resize").read_text(), "0")
        requested = self.panel.call(self.panel.identity, "save")
        self.assertTrue(requested["save_pending"])
        self.assertIsNone(requested["saved_at"])
        self.assertIn("error", self.panel.call(self.panel.identity, "save"))
        (self.run / "saved.json").write_text("{}")
        self.assertFalse(self.panel.snapshot()["save_pending"])
        stopped = self.panel.call(self.panel.identity, "stop")
        self.assertEqual(stopped["phase"], "stopping")
        self.assertEqual((self.run / "stop").read_text(), "1")
        self.process.poll.return_value = 0
        self.assertEqual(self.panel.snapshot()["phase"], "stopped")
        self.process.wait.assert_not_called()

    def test_stop_during_initialization_cancels_evaluation(self):
        self.run.mkdir()
        self.panel.evaluation = Mock()
        self.panel.evaluation.settings = {"minutes": 30, "pairs": 1, "seconds": 120}
        self.panel.evaluation.status.return_value = {}
        self.assertEqual(self.panel.snapshot()["phase"], "initializing")
        stopped = self.panel.call(self.panel.identity, "stop")
        self.assertEqual(stopped["phase"], "stopping")
        self.panel.evaluation.cancel.assert_called_once()
        self.panel.evaluation.configure.assert_called_once_with(
            {"minutes": 0, "pairs": 1, "seconds": 120}
        )

    def test_empty_stale_failure_and_old_identity(self):
        panel = Panel(self.run)
        panel.resources = self.panel.resources
        self.assertEqual(panel.snapshot()["phase"], "not_started")
        self.assertFalse(self.run.exists())
        self.status()
        self.assertEqual(self.panel.call("old", "stop"), {"closed": True})
        self.assertFalse((self.run / "stop").exists())
        os.utime(self.run / "status.json", (1, 1))
        self.assertEqual(self.panel.snapshot()["phase"], "stale")
        self.assertIn("error", self.panel.call(self.panel.identity, "resize", 8))
        self.process.poll.return_value = -6
        self.run.with_suffix(".log").write_text("native failure")
        failed = self.panel.snapshot()
        self.assertEqual(failed["phase"], "failed")
        self.assertIn("native failure", failed["log"])

    def test_history_is_bounded_and_restart_does_not_join_series(self):
        for index in range(370):
            self.status(index + 1)
        self.assertEqual(len(self.panel.history), 360)
        self.status(0.5)
        self.assertEqual(len(self.panel.history), 1)

    def test_notebook_click_to_resume_and_start(self):
        notebook = json.loads(
            Path(__file__).with_name("train.ipynb").read_text(encoding="utf-8")
        )
        cells = {
            c["id"]: "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "code"
        }
        root = Path(self.temporary.name)
        old = root / "old" / "checkpoint.pt"
        latest = root / "latest" / "checkpoint.pt"
        for checkpoint, stamp in ((old, 1), (latest, 2)):
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"unchanged checkpoint")
            os.utime(checkpoint, (stamp, stamp))
        widgets = Mock()
        widgets.Dropdown.side_effect = lambda **args: SimpleNamespace(
            value=args["options"][0][1], disabled=False, observe=Mock()
        )
        widgets.Output.side_effect = MagicMock
        fake = Mock()
        fake.check_output.return_value = '{"concurrency":8,"batch":512}'
        fake.Popen.return_value.poll.return_value = None
        namespace = {
            "Path": Mock(return_value=root),
            "SOURCE": Path(__file__).resolve().parents[2],
            "subprocess": fake,
            "json": json,
            "datetime": datetime,
            "timezone": timezone,
            "NATIVE": ["native-test"],
        }
        with (
            patch.dict("sys.modules", {
                "google.colab": Mock(),
                "ipywidgets": widgets,
                "IPython.display": Mock(),
            }),
            patch("importlib.reload"),
            patch("training.colab.panel.show") as show,
            patch("builtins.print"),
        ):
            exec(cells["configuration"], namespace)
            self.assertEqual(namespace["ARGS"], ["--device", "cuda"])
            # 下拉选旧档与新训练均重算配置，不创建进程。
            picker = namespace["checkpoint_picker"]
            for selection in (str(old), "", str(latest)):
                picker.value = selection
                namespace["prepare_training"]()
                self.assertEqual("--resume" in namespace["ARGS"], bool(selection))
            exec(cells["start"], namespace)
            fake.Popen.assert_not_called()
            button = namespace["start_button"]
            button.on_click.call_args.args[0](button)
            fake.Popen.assert_called_once()
            self.assertEqual(fake.Popen.call_args.args[0][-2:], ["--resume", str(latest)])
            self.assertNotEqual(namespace["RUN"], latest.parent)
            self.assertTrue(picker.disabled)
            show.assert_called_once()
            namespace["start_training"](button)
            exec(cells["start"], namespace)
            fake.Popen.assert_called_once()
            self.assertEqual(show.call_count, 2)
            # 停止后重新扫描，不能用禁用的旧选择器再次启动。
            namespace["process"].poll.return_value = 0
            exec(cells["start"], namespace)
            fake.Popen.assert_called_once()
            for checkpoint in (old, latest):
                self.assertEqual(checkpoint.read_bytes(), b"unchanged checkpoint")
            old.unlink()
            latest.unlink()
            exec(cells["configuration"], namespace)
            self.assertEqual(namespace["checkpoint_picker"].value, "")
            self.assertEqual(namespace["ARGS"], ["--device", "cuda"])

    def test_panel_is_embedded(self):
        markup = self.panel.render()
        self.assertIn("haojie.native_panel", markup)
        self.assertIn("var(--paper)", markup)
        self.assertNotIn("__PANEL_ID__", markup)
        self.assertNotIn("https://", markup)


if __name__ == "__main__":
    unittest.main()
