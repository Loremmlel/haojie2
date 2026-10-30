"""本机控制协议与配置约束；不启动训练或加载原生规则库。"""

from dataclasses import asdict
import json
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from training.haojie.runtime.config import Config
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


if __name__ == "__main__":
    unittest.main()
