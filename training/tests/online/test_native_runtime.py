"""真实规则、采样 RNG 与根内回溯在保存边界前后一致。"""

import json
import unittest

import torch

from haojie_training.online.native import NativeFailure, Worker, frames, packet


class NativeRuntime(unittest.TestCase):
    def test_deferred_inference_does_not_resend_or_resample(self):
        worker = Worker()
        try:
            worker.submit(0, packet([(1, 71, 1, 0, False), (2, 72, 2, 0, False)]))
            current = frames(worker.wait())
            worker.submit(0, packet())
            self.assertEqual(frames(worker.wait()), [])
            frame = current[0]
            worker.submit(0, packet(replies=[(frame["game"], frame["node"]["mask"].bfloat16())]))
            advanced = frames(worker.wait())
            self.assertEqual([f["game"] for f in advanced], [frame["game"]])
            worker.submit(1)
            saved = json.loads(worker.wait())
            self.assertEqual(saved[1]["microsteps"], 0)
            self.assertTrue(saved[1]["waiting"])
        finally:
            worker.close()

    def test_every_random_attempt_records_actual_masked_probability(self):
        import math

        worker = Worker()
        try:
            worker.submit(0, packet([(1, 73127, 1, 0, False)]))
            current = frames(worker.wait())
            selections = 0
            for _ in range(200):
                offered = {}
                for frame in current:
                    if frame["kind"] == 0:
                        mask = frame["node"]["mask"]
                        probs = torch.arange(1, len(mask) + 1, dtype=torch.bfloat16) * mask
                        offered[frame["game"]] = probs
                worker.submit(0, packet(replies=offered.items()))
                current = frames(worker.wait())
                for frame in current:
                    if frame["selected"] >= 0:
                        probabilities = offered[frame["game"]].double()
                        expected = math.log(
                            float(probabilities[frame["selected"]] / probabilities.sum())
                        )
                        self.assertAlmostEqual(frame["logp"], expected, places=12)
                        selections += 1
            self.assertEqual(current[0]["microsteps"], selections)
            self.assertGreater(current[0]["backtracks"], 0)
        finally:
            worker.close()

    def test_failure_preserves_diagnostic_state_and_rng(self):
        worker = Worker()
        try:
            worker.submit(0, packet([(1, 71, 1, 0, False)]))
            frame = frames(worker.wait())[0]
            worker.submit(1)
            before = json.loads(worker.wait())[0]
            worker.submit(
                0,
                packet(
                    replies=[
                        (
                            1,
                            torch.full_like(
                                frame["node"]["mask"], float("nan"), dtype=torch.bfloat16
                            ),
                        )
                    ]
                ),
            )
            with self.assertRaises(NativeFailure) as caught:
                worker.wait()
            diagnostic = json.loads(caught.exception.payload)
            self.assertEqual(diagnostic["error"], "invalid probability support")
            self.assertEqual(diagnostic["games"][0], before)
        finally:
            worker.close()

    def test_history_is_contiguous_and_transmitted_once(self):
        worker = Worker()
        ends = {}
        transmitted = 0
        try:
            worker.submit(0, packet([(1, 73127, 1, 0, False)]))
            for _ in range(120):
                current = frames(worker.wait())
                replies = []
                for frame in current:
                    if frame["kind"] != 0:
                        continue
                    node = frame["node"]
                    self.assertEqual(node["history_start"], ends.get(node["root"], 0))
                    count = len(node["history"]["events"])
                    self.assertEqual(node["history_end"] - node["history_start"], count)
                    ends[node["root"]] = node["history_end"]
                    transmitted += count
                    replies.append((frame["game"], node["mask"].bfloat16()))
                worker.submit(0, packet(replies=replies))
            self.assertGreater(transmitted, 0)
            self.assertEqual(transmitted, sum(ends.values()))
            worker.wait()
            worker.submit(1)
            for game in json.loads(worker.wait()):
                if game["search"] is not None and game["waiting"]:
                    self.assertEqual(game["search"]["pending_events"], [])
                    self.assertGreaterEqual(game["search"]["sent_events"], 0)
        finally:
            worker.close()

    def test_resume_does_not_resample(self):
        a, b = Worker(), Worker()
        try:
            a.submit(0, packet([(1, 73127, 1, 0, False), (2, 73128, 2, 0, True)]))
            current = frames(a.wait())
            for _ in range(35):
                replies = [
                    (f["game"], f["node"]["mask"].bfloat16()) for f in current if f["kind"] == 0
                ]
                a.submit(0, packet(replies=replies))
                current = frames(a.wait())
            a.submit(1)
            saved = a.wait()
            b.submit(2, saved)
            b.wait()
            for _ in range(100):
                replies = [
                    (f["game"], f["node"]["mask"].bfloat16()) for f in current if f["kind"] == 0
                ]
                a.submit(0, packet(replies=replies))
                b.submit(0, packet(replies=replies))
                current, resumed = frames(a.wait()), frames(b.wait())
                self.assertEqual(
                    [(f["selected"], f["logp"], f["root"], f["commands"]) for f in current],
                    [(f["selected"], f["logp"], f["root"], f["commands"]) for f in resumed],
                )
            a.submit(1)
            b.submit(1)
            self.assertEqual(json.loads(a.wait()), json.loads(b.wait()))
        finally:
            a.close()
            b.close()


if __name__ == "__main__":
    unittest.main()
