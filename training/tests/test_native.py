"""纯 Rust/Python 信任边界与检查点接线测试；HAOJIE_NATIVE 指向交付包构建的二进制。"""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.native.client import Client
from haojie_training.native.model import metadata, model_from
from haojie_training.runtime import Trainer


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if not os.environ.get("HAOJIE_NATIVE"):
            raise RuntimeError("原生验收必须显式设置 HAOJIE_NATIVE，不能跳过")
        cls.engine = str(Path(os.environ["HAOJIE_NATIVE"]).resolve(strict=True))

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
            self.assertEqual(
                done["debug"]["start"],
                {"prelude": [], **request["start"]},
            )
            self.assertEqual(done["debug"]["commands"], prefix_commands)
            self.assertIn("state", done["debug"])
            self.assertNotIn("debug", infer)

    def test_checkpoint_rules_and_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "initial.pt"
            with Client(self.engine) as client:
                Trainer(PolicyValueNet(ModelConfig.tiny()), torch.device("cpu")).save(
                    path, metadata(client.ready)
                )
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

    def test_eof_during_frame_is_distinct_from_corruption(self):
        for payload, kind in (
            (b'{"type":"infer"', RuntimeError),
            (b'{"bytes":666,"entities":1,"candidates":1}\nabc', RuntimeError),
            (b'{"bytes":10,"entities":1,"candidates":1}\nabc', ValueError),
            (b"{bad json}\n", ValueError),
        ):
            client = Client.__new__(Client)
            client._input_refs = False
            client.frame_limit = 256 * 1024 * 1024
            client.process = SimpleNamespace(stdout=io.BytesIO(payload))
            results = []
            client._publish = results.append
            client._read()
            self.assertIsInstance(results[-1], kind)
            if kind is RuntimeError:
                self.assertEqual(str(results[-1]), "原生进程已退出")

    def test_large_valid_frame_is_distinct_from_resource_and_protocol_errors(self):
        entities = 8200
        body = (
            torch.zeros(entities * 64 + 32 + 64).numpy().tobytes()
            + torch.zeros(entities, dtype=torch.int64).numpy().tobytes()
            + torch.full((2,), -1, dtype=torch.int64).numpy().tobytes()
            + bytes([1]) * (entities + 1)
        )
        self.assertGreater(len(body), 2 * 1024 * 1024)
        header = (
            json.dumps(
                {"type": "infer", "bytes": len(body), "entities": entities, "candidates": 1}
            ).encode()
            + b"\n"
        )
        for limit, kind in ((4 * 1024 * 1024, dict), (1024 * 1024, MemoryError)):
            client = Client.__new__(Client)
            client._input_refs = False
            client.frame_limit = limit
            client.read_seconds = client.decode_seconds = client.tensor_bytes = 0
            client.process = SimpleNamespace(stdout=io.BytesIO(header + body))
            result = []
            client._publish = result.append
            client._read()
            self.assertIsInstance(result[0], kind)
            if kind is dict:
                self.assertEqual(result[0]["input"]["entities"].shape[0], entities)


if __name__ == "__main__":
    unittest.main()
