"""纯 Rust/Python 信任边界与检查点接线测试；HAOJIE_NATIVE 指向交付包构建的二进制。"""

import os
import tempfile
import unittest
from pathlib import Path

import torch

from haojie_training.native.client import Client
from haojie_training.native.pipeline import initialize, model_from, prepare, sample


@unittest.skipUnless(os.environ.get("HAOJIE_NATIVE"), "需显式设置 HAOJIE_NATIVE")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.engine = os.environ["HAOJIE_NATIVE"]

    def test_invalid_inputs_cancel_stale_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder, Client(self.engine) as client:
            root = Path(folder)
            request = {
                "op": "sample",
                "record": str(root / "game.jsonl"),
                "start": {"seed": 71, "rules": "classic"},
                "model": "a" * 64,
                "samplerSeed": 91,
                "maxCommands": 10,
                "maxPlies": 10,
            }
            for change in (
                {"initialState": {}},
                {"model": "bad"},
                {"start": {"seed": 1, "rules": "old"}},
                {"start": {"seed": 1, "rules": "classic", "state": {}}},
                *(
                    {"start": {"seed": 1, "rules": "classic", "prelude": [played]}}
                    for played in (
                        {"actor": 3, "command": {"type": "summon"}},
                        {"actor": 2, "command": {"type": "summon"}},
                        {"actor": 1, "command": {"type": "summon", "secret": 1}},
                        {"actor": 1, "command": {"type": "summon", "x": 0}},
                        {"actor": 1, "command": {"type": "summon", "ability": "undefined"}},
                    )
                ),
            ):
                client.send({**request, **change})
                with self.assertRaises(ValueError):
                    client.receive()
                self.assertFalse((root / "game.jsonl.partial").exists())
            client.send(request)
            infer = client.receive()
            self.assertEqual(infer["type"], "infer")
            self.assertEqual(
                set(infer["input"]),
                {
                    "entities",
                    "globals",
                    "candidates",
                    "kinds",
                    "sources",
                    "targets",
                    "entity_mask",
                    "candidate_mask",
                },
            )
            snapshot = infer["input"]["entities"].clone()
            client.send({"id": infer["id"], "model": infer["model"], "cancel": True})
            done = client.receive()
            self.assertEqual(done["outcome"]["reason"], "cancelled")
            prefix_commands = done["outcome"]["commands"]
            self.assertIsNone(done["outcome"]["returns"])
            self.assertTrue(torch.equal(snapshot, infer["input"]["entities"]))
            client.send({"op": "audit", "record": request["record"], "encode": False})
            self.assertTrue(client.receive()["report"]["complete"])
            raw = (root / "game.jsonl").read_text()
            bad = root / "bad.jsonl"
            bad.write_text(raw.replace('"seed":71', '"seed":72'))
            client.send({"op": "audit", "record": str(bad), "encode": False})
            with self.assertRaisesRegex(ValueError, "hash"):
                client.receive()
            partial = root / "prefix.jsonl"
            partial.write_text(raw.splitlines()[0] + "\n")
            client.send({"op": "audit", "record": str(partial), "encode": False})
            result = client.receive()["report"]
            self.assertFalse(result["complete"])
            self.assertIsNone(result["outcome"]["returns"])
            partial.write_text(raw.splitlines()[0] + '\n{"unfinished":')
            client.send({"op": "audit", "record": str(partial), "encode": False})
            self.assertTrue(client.receive()["report"]["incompleteTail"])
            bad.write_bytes(b"\x1f\x8bcorrupt-gzip\n")
            client.send({"op": "audit", "record": str(bad), "encode": False})
            with self.assertRaisesRegex(ValueError, "corrupt"):
                client.receive()
            bad.write_bytes(b" " * (16 * 1024 * 1024 + 1))
            client.send({"op": "audit", "record": str(bad), "encode": False})
            with self.assertRaisesRegex(ValueError, "exceeds limit"):
                client.receive()
            request["record"] = str(root / "stale.jsonl")
            client.send(request)
            infer = client.receive()
            client.send(
                {"id": infer["id"], "model": "b" * 64, "logits": [0.0] * infer["candidates"]}
            )
            done = client.receive()
            self.assertEqual(done["outcome"]["reason"], "error")
            self.assertIn("stale", done["error"])
            self.assertEqual(done["outcome"]["commands"], prefix_commands)

    def test_checkpoint_rules_and_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "initial.pt"
            initialize(self.engine, path, tiny=True)
            with Client(self.engine) as client:
                model, digest = model_from(path, client.ready)
                payload = torch.load(path, weights_only=True)
                self.assertTrue(
                    all(torch.equal(v, model.state_dict()[k]) for k, v in payload["model"].items())
                )
                self.assertEqual(len(digest), 64)
                payload["metadata"]["rules_package_sha256"] = "old"
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "哈希"):
                    model_from(path, client.ready)

    def test_changed_budget_cannot_duplicate_a_sampling_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "initial.pt"
            initialize(self.engine, checkpoint, tiny=True)
            records = []
            for i, (seed, plies) in enumerate(((71, 10), (72, 10), (71, 20))):
                output = root / f"sample-{i}"
                sample(
                    self.engine,
                    checkpoint,
                    [{"seed": seed, "rules": "classic"}],
                    output,
                    commands=4,
                    plies=plies,
                )
                records.extend(output.glob("*.jsonl"))
            with self.assertRaisesRegex(ValueError, "重复对局"):
                prepare(self.engine, records, root / "data")
            self.assertFalse((root / "data").exists())


if __name__ == "__main__":
    unittest.main()
