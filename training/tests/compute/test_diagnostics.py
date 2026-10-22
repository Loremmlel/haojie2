"""主线程与输入线程的计时分开，避免用跨线程时钟拼出无效累计时间。"""

import threading
import unittest
from pathlib import Path

from haojie_training.diagnostics import profiling
from haojie_training.model import ModelConfig
from haojie_training.native import client

from .test_packed import examples


class DiagnosticsTests(unittest.TestCase):
    def test_decode_cpu_is_attributed_to_its_own_thread(self):
        row = examples(ModelConfig.tiny())[1]
        keys = (
            "entities",
            "globals",
            "candidates",
            "kinds",
            "sources",
            "targets",
            "entity_mask",
            "candidate_mask",
        )
        data = b"".join(row[key].numpy().tobytes() for key in keys)
        report = {}
        errors = []

        def decode():
            try:
                client.decode(data, len(row["entities"]), len(row["sources"]), _arrays=True)
            except BaseException as error:
                errors.append(error)

        with profiling("cpu", Path("unused"), report):
            decode()
            thread = threading.Thread(target=decode)
            thread.start()
            thread.join()
        self.assertFalse(errors)
        stages = report["profile_stages_inclusive"]
        for prefix in ("main:", "readers:"):
            stage = stages[prefix + "haojie_training.native.client.decode"]
            self.assertEqual(stage["calls"], 1)
            self.assertGreaterEqual(stage["cpu"], 0)
            self.assertGreaterEqual(stage["wall"], stage["cpu"] * 0.9)
