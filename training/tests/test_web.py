"""本机控制协议、配置约束及并发重分配的功能验收。"""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from training.haojie.runtime.config import Config
from training.haojie.runtime.trainer import Trainer
from training.haojie.web.controller import Controller
from training.haojie.web.server import server
from training.haojie.web.settings import defaults, schema, validate


class WebTests(unittest.TestCase):
    def test_complete_schema_and_invalid_controls(self):
        self.assertEqual(
            {f["key"] for f in schema()} | {"device"}, set(asdict(Config()))
        )
        for changes in (
            {"workers": 1.5},
            {"batch": float("nan")},
            {"workers": True},
            {"teacher_fraction": 0.8, "history_fraction": 0.8},
            {"workers": 128, "games_per_worker": 128},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate(changes)
        with self.assertRaisesRegex(ValueError, "仅能在新建"):
            validate({"seed": 99}, defaults())
        self.assertEqual(
            validate({"learning_rate": 0.001}, defaults())["learning_rate"], 0.001
        )

    def test_loopback_protocol_revision_and_path_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Controller(directory)
            http = server(control, 0)
            thread = threading.Thread(target=http.serve_forever)
            thread.start()
            url = f"http://127.0.0.1:{http.server_port}"
            try:
                with urlopen(url + "/api/state") as response:
                    self.assertEqual(json.load(response)["state"], "idle")
                for origin, expected in (
                    ("https://untrusted.invalid", 403),
                    (url, 400),
                ):
                    request = Request(
                        url + "/api/control",
                        data=json.dumps(
                            {
                                "action": "start",
                                "name": "../escape",
                                "revision": 0,
                                "session": control.session,
                            }
                        ).encode(),
                        headers={
                            "Content-Type": "application/json",
                            "X-Haojie-Control": "1",
                            "Origin": origin,
                        },
                    )
                    with self.assertRaises(HTTPError) as error:
                        urlopen(request)
                    self.assertEqual(error.exception.code, expected)
                with self.assertRaisesRegex(ValueError, "其他页面"):
                    control.control(
                        "start", "run", revision=10, session=control.session
                    )
                with self.assertRaisesRegex(ValueError, "其他页面"):
                    control.control(
                        "start", "run", revision=0, session="previous-service"
                    )
                self.assertIsNone(control.process)
            finally:
                http.shutdown()
                http.server_close()
                thread.join()
                control.close()

    def test_resize_keeps_every_game_and_applies_optimizer_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                device="cpu",
                warmup_games=0,
                workers=2,
                games_per_worker=2,
                simulations=1,
                fast_simulations=1,
                save_seconds=0,
            )
            (Path(directory) / "STOP").touch()
            trainer = Trainer(directory, config)
            trainer.run()
            saved = torch.load(Path(directory) / "checkpoint.pt", weights_only=False)
            original = {
                g["game"]["id"] for rows in saved["workers"].values() for g in rows
            }
            self.assertEqual(len(original), 4)
            config.workers = config.games_per_worker = 1
            config.learning_rate = 0.001
            small = Trainer(directory, config, resume=True)
            self.assertTrue(all(g["lr"] == 0.001 for g in small.optimizer.param_groups))
            small.run()
            saved = torch.load(Path(directory) / "checkpoint.pt", weights_only=False)
            self.assertEqual(
                {g["game"]["id"] for rows in saved["workers"].values() for g in rows},
                original,
            )
            self.assertEqual(len(saved["workers"][-1]), 3)
            config.workers, config.games_per_worker = 2, 3
            large = Trainer(directory, config, resume=True)
            large.run()
            saved = torch.load(Path(directory) / "checkpoint.pt", weights_only=False)
            self.assertNotIn(-1, saved["workers"])
            games = [
                g["game"]["id"] for rows in saved["workers"].values() for g in rows
            ]
            self.assertEqual(len(games), len(set(games)))
            self.assertEqual(len(games), 6)
            self.assertTrue(original <= set(games))


if __name__ == "__main__":
    unittest.main()
