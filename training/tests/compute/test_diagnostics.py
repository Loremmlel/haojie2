"""主线程与输入线程的计时分开，避免用跨线程时钟拼出无效累计时间。"""

import threading
import unittest
from copy import deepcopy
from pathlib import Path

from haojie_training.diagnostics import profiling, sample_fingerprint
from haojie_training.model import ModelConfig
from haojie_training.native import client

from .test_packed import examples


class DiagnosticsTests(unittest.TestCase):
    def test_fixed_sample_fingerprint_ignores_arrival_time_but_keeps_choices_and_rewards(self):
        rows = examples(ModelConfig.tiny())[:2]
        for index, row in enumerate(rows):
            row["_meta"] = {"game": {"task": index}, "entered": 1.0}
            row["value"] = row["globals"][0].clone()
        reordered = deepcopy(rows[::-1])
        for row in reordered:
            row["_meta"]["entered"] = 2.0
        expected = sample_fingerprint(rows)
        self.assertEqual(sample_fingerprint(reordered), expected)
        reordered[0]["targets"][0] = -1
        self.assertNotEqual(sample_fingerprint(reordered), expected)
        changed_reward = deepcopy(rows)
        changed_reward[0]["value"].add_(1)
        self.assertNotEqual(sample_fingerprint(changed_reward), expected)

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
        for prefix in ("main:", "workers:"):
            stage = stages[prefix + "haojie_training.native.client.decode"]
            self.assertEqual(stage["calls"], 1)
            self.assertGreaterEqual(stage["cpu"], 0)
            self.assertGreaterEqual(stage["wall"], stage["cpu"] * 0.9)
