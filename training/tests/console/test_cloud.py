"""真实小模型闭环与本地模拟 Drive：恢复优化器，拒绝半份归档和活跃导入。"""

import copy
import errno
import gzip
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from haojie_training.cloud.backup import Backups, freeze, import_root
from haojie_training.cloud.session import Session, wait_for
from haojie_training.console.config import defaults
from haojie_training.console.controller import identity
from haojie_training.console.storage import FORMAT, Lease, Store
from haojie_training.data import synthetic_batch
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.runtime import Trainer


def stored(folder):
    torch.set_num_threads(1)
    store = Store(folder)
    lease = Lease(store)
    lease.close()
    trainer = Trainer(PolicyValueNet(ModelConfig.tiny()), torch.device("cpu"))
    trainer.step(synthetic_batch(trainer.model.config, size=2, entities=4, actions=2))
    payload = {
        "format": FORMAT,
        "trainer": trainer.checkpoint({}),
        "config": defaults(),
        "counts": {"games": 2},
        "random": random.Random(9).getstate(),
        "saved_at": time.time(),
    }
    store.save_tensor("recovery", trainer.updates, payload, 2)
    return SimpleNamespace(
        root_store=store,
        store=store,
        experiment="legacy",
        saved_step=trainer.updates,
        saved_at=payload["saved_at"],
        device=torch.device("cpu"),
    )


class CloudTests(unittest.TestCase):
    def test_failed_upload_retains_previous_backup_and_foreign_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c = stored(root / "training")
            stage = root / "stage"
            stage.mkdir()
            info = freeze(c, stage)
            backups = Backups(root / "drive", keep=2)
            foreign = backups.root / "my-old-model.pt"
            foreign.write_bytes(b"original")
            original = backups.publish(stage, info, root)

            def fail(source, target, **_):
                target.write(source.read(32))
                raise OSError("模拟 Drive 上传中断")

            with patch("haojie_training.cloud.backup.shutil.copyfileobj", side_effect=fail):
                with self.assertRaisesRegex(OSError, "上传中断"):
                    backups.publish(stage, info, root)
            self.assertEqual(len(backups.records()), 1)
            self.assertEqual(str(backups.records()[0][0]), original["manifest"])
            self.assertEqual(foreign.read_bytes(), b"original")
            self.assertFalse(list(backups.root.glob("*.partial")))
            restored = backups.restore(root / "restored")
            self.assertEqual(restored["updates"], 1)
            self.assertTrue((root / "restored/service.lock").exists())

    def test_corruption_fallback_and_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c = stored(root / "training")
            stage = root / "stage"
            stage.mkdir()
            info = freeze(c, stage)
            backups = Backups(root / "drive", keep=2)
            first = backups.publish(stage, info, root)
            second = backups.publish(stage, info, root)
            third = backups.publish(stage, info, root)
            self.assertFalse(Path(first["manifest"]).exists())
            self.assertEqual(len(backups.records()), 2)
            broken = json.loads(Path(third["manifest"]).read_text())
            (backups.root / broken["archive"]).write_bytes(b"incomplete")
            (backups.root / "interrupted.json.partial").write_bytes(b"partial")
            messages = []
            result = backups.restore(root / "restored", log=messages.append)
            self.assertEqual(result["manifest"], second["manifest"])
            self.assertEqual(len(messages), 1)
            with self.assertRaisesRegex(ValueError, "目标必须为空"):
                backups.restore(root / "restored")
            malicious = backups.records()[0][1]
            malicious["files"]["../escaped"] = {"bytes": 1, "sha256": "x"}
            Path(second["manifest"]).write_text(json.dumps(malicious))
            with self.assertRaisesRegex(ValueError, "没有通过完整性校验"):
                backups.restore(root / "unsafe", selected=second["manifest"])
            self.assertFalse((root / "escaped").exists())

    def test_invalid_completion_metadata_never_publishes_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            stage = root / "stage"
            stage.mkdir()
            info = freeze(stored(root / "training"), stage)
            backups = Backups(root / "drive")
            published = backups.publish(stage, info, root)
            manifest = Path(published["manifest"])
            record = json.loads(manifest.read_text())
            record["checkpoint"] = None
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "没有通过完整性校验"):
                backups.restore(root / "target")
            self.assertFalse((root / "target").exists())
            record["created_at"] = float("inf")
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "存在备份清单"):
                backups.restore(root / "target")
            self.assertFalse((root / "target").exists())

    def test_import_preserves_source_and_rejects_running_service(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c = stored(root / "source")
            unknown = c.store.root / "author-original.txt"
            unknown.write_text("原件", encoding="utf-8")
            lease = Lease(c.store)
            try:
                with self.assertRaisesRegex(RuntimeError, "已有训练服务"):
                    import_root(c.store.root, root / "active-copy")
            finally:
                lease.close()
            import_root(c.store.root, root / "copy")
            self.assertEqual(unknown.read_text(encoding="utf-8"), "原件")
            self.assertFalse((root / "copy/author-original.txt").exists())
            self.assertEqual(
                (root / "copy/recovery/0000000000000001.pt").read_bytes(),
                (c.store.root / "recovery/0000000000000001.pt").read_bytes(),
            )
            if os.name != "nt":
                with patch("fcntl.flock", side_effect=OSError(errno.EOPNOTSUPP, "无锁挂载")):
                    info = import_root(c.store.root, root / "fuse-copy")
                self.assertIn("unsupported", info["source_lock"])
            import_root(c.store.root / "recovery/0000000000000001.pt", root / "single")
            self.assertTrue((root / "single/service.lock").is_file())

    def test_snapshot_survives_training_file_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c = stored(root / "source")
            stage = root / "stage"
            stage.mkdir()
            freeze(c, stage)
            name = c.store.names("recovery")[-1]
            original = (stage / name).read_bytes()
            c.store.write(name, b"replaced after snapshot")
            self.assertEqual((stage / name).read_bytes(), original)

    def test_cli_api_and_graceful_final_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            connection = root / "connection.json"
            with (root / "service.log").open("w+") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "haojie_training.cloud",
                        "--engine",
                        os.environ["HAOJIE_NATIVE"],
                        "--root",
                        str(root / "local"),
                        "--backup-dir",
                        str(root / "drive"),
                        "--connection-file",
                        str(connection),
                        "--device",
                        "cpu",
                        "--torch-threads",
                        "2",
                        "--tiny",
                        "--paused",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    deadline = time.monotonic() + 60
                    while not connection.exists() and time.monotonic() < deadline:
                        if process.poll() is not None:
                            log.seek(0)
                            self.fail(log.read())
                        time.sleep(0.1)
                    address = json.loads(connection.read_text())["url"]

                    def api(path, body=None):
                        request = urllib.request.Request(
                            address + path,
                            data=json.dumps(body).encode() if body else None,
                            headers={
                                "Content-Type": "application/json",
                                "Origin": address,
                                "X-Session-Token": token if body else "",
                            },
                        )
                        with urllib.request.urlopen(request, timeout=10) as response:
                            return json.load(response)

                    token = api("/api/session")["token"]
                    status = api("/api/status")
                    self.assertEqual(status["state"], "paused")
                    self.assertEqual(status["updates"], 0)
                    self.assertIn("cpu_capacity", status["hardware"])
                    api("/api/command", {"action": "backup"})
                    while time.monotonic() < deadline:
                        status = api("/api/status")
                        if status["backup"]["last"]:
                            break
                        time.sleep(0.1)
                    self.assertTrue(Path(status["backup"]["last"]["manifest"]).is_file())
                    api("/api/command", {"action": "shutdown"})
                    self.assertEqual(process.wait(timeout=60), 0)
                    self.assertTrue(Backups(root / "drive").restore(root / "restored"))
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=60)

    def test_real_training_backup_restart_and_optimizer_continuation(self):
        torch.set_num_threads(1)
        engine = os.environ["HAOJIE_NATIVE"]
        repo = Path(__file__).resolve().parents[3]
        starts = json.loads(
            gzip.decompress((repo / "tests/fixtures/native/late-starts.json.gz").read_bytes())
        )[:2]
        environments = int(os.environ.get("HAOJIE_TEST_ENVIRONMENTS", "4"))
        settings = {
            "environments": environments,
            "games_per_round": environments,
            "inference_batch_size": min(8, environments),
            "updates_per_round": 2,
            "batch_size": 2,
            "pool_mib": 16,
            "max_commands": 500,
            "max_plies": 60,
            "history_every": 2,
            "eval_games": 100000,
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            session = Session(
                engine,
                root / "local",
                device="cpu",
                settings=settings,
                backup_dir=root / "drive",
                tiny=True,
                starts=starts,
            )
            restored = None
            try:
                c = session.controller
                peak = 0
                original = c._progress

                def progress(metrics):
                    nonlocal peak
                    peak = max(peak, len(metrics["active"]))
                    original(metrics)

                c._progress = progress
                session.command("start")
                wait_for(c, lambda: c.trainer.updates >= 2, max(120, environments * 15))
                session.command("pause")
                wait_for(c, lambda: c.state == "paused")
                saved_counts = copy.deepcopy(c.counts)
                saved_config = copy.deepcopy(c.config)
                model_hash = identity(c.trainer.model)
                optimizer = copy.deepcopy(c.trainer.optimizer.state_dict())
                updates = c.trainer.updates
                result = session.backup.request().result(timeout=60)
                self.assertEqual(result["updates"], updates)
                self.assertEqual(peak, environments)
                perf = c.performance()
                self.assertGreater(perf["sampling"]["inference_requests"], 0)
                self.assertLessEqual(perf["last_round"]["mean_batch"], min(8, environments))
                self.assertGreater(saved_counts["games"], 0)
                self.assertGreater(saved_counts["retained_samples"], 0)
                self.assertIsNone(c.pool)
                restored = Session(
                    engine,
                    root / "restored",
                    device="cpu",
                    backup_dir=root / "drive",
                    tiny=True,
                    starts=starts,
                )
                other = restored.controller
                self.assertEqual(other.trainer.updates, updates)
                self.assertEqual(other.counts, saved_counts)
                self.assertEqual(other.config, saved_config)
                self.assertEqual(identity(other.trainer.model), model_hash)
                actual = other.trainer.optimizer.state_dict()
                self.assertEqual(actual["param_groups"], optimizer["param_groups"])
                for key, state in optimizer["state"].items():
                    for field, value in state.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(
                                actual["state"][key][field], value, rtol=0, atol=0
                            )
                        else:
                            self.assertEqual(actual["state"][key][field], value)
                self.assertFalse(other.samples.rows)
                self.assertTrue(other.store.names("baseline"))
                self.assertEqual(other.store.names("history"), c.store.names("history"))
                restored.command("start")
                wait_for(
                    other, lambda: other.trainer.updates > updates, max(120, environments * 15)
                )
                print(
                    json.dumps(
                        {
                            "environments": environments,
                            "peak_active": peak,
                            "updates": updates,
                            "sampling": perf["sampling"],
                        }
                    )
                )
            finally:
                if restored is not None:
                    self.assertEqual(restored.close(), [])
                self.assertEqual(session.close(), [])
            self.assertFalse(list(root.glob("haojie-backup-*")))
            self.assertFalse(list(root.glob("haojie-restore-*")))
