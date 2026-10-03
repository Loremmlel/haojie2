"""内容及审核器身份、单遍等价和核验后替换来源的回归，不依赖本机历史产物。"""
import copy
import hashlib
import importlib
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from haojie_training.data import load_dataset, select_batch
from haojie_training.model import ModelConfig
from haojie_training.native.client import Client
from haojie_training.native.pipeline import audit, initialize, prepare, sample

audit_module = importlib.import_module("haojie_training.native.audit")


class NativeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if not os.environ.get("HAOJIE_NATIVE"):
            raise RuntimeError("原生审核测试必须显式设置 HAOJIE_NATIVE")
        cls.engine = str(Path(os.environ["HAOJIE_NATIVE"]).resolve(strict=True))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        model = self.root / "initial.pt"
        initialize(self.engine, model, tiny=True)
        folder = self.root / "sample"
        sample(self.engine, model, [{"seed": 71, "rules": "classic"},
                                   {"seed": 72, "rules": "classic"}], folder, commands=6, plies=10)
        self.paths = sorted(folder.glob("*.jsonl"))
        self.results = [audit(self.engine, p) for p in self.paths]

    def assert_data_equal(self, a, b):
        for split in ("train", "validation"):
            left, _ = load_dataset(a / (split + ".pt"), ModelConfig.tiny())
            right, _ = load_dataset(b / (split + ".pt"), ModelConfig.tiny())
            self.assertEqual(len(left["value"]), len(right["value"]))
            indices = torch.arange(len(left["value"]))
            left, right = select_batch(left, indices), select_batch(right, indices)
            self.assertEqual(set(left), set(right))
            self.assertTrue(all(torch.equal(left[k], right[k]) for k in left))

    def test_single_pass_equals_strict_prepare_without_replay_or_input_aliasing(self):
        frozen = [{k: v.clone() for k, v in r.inputs[0][0].items()} for r in self.results]
        original = Client.send
        operations = []

        def send(client, message):
            operations.append(message["op"])
            return original(client, message)

        with patch.object(Client, "send", send):
            report = prepare(self.engine, self.paths, self.root / "single", audited=self.results)
            self.assertNotIn("audit", operations)
            prepare(self.engine, self.paths, self.root / "strict")
            self.assertEqual(operations.count("audit"), len(self.paths))
        self.assert_data_equal(self.root / "single", self.root / "strict")
        self.assertEqual(report["metadata"]["audit"], {
            **self.results[0].report["auditor"], "binary_sha256": self.results[0].engine_sha256})
        for path, result, before, source in zip(self.paths, self.results, frozen, report["metadata"]["sources"]):
            self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(all(torch.equal(before[k], result.inputs[0][0][k]) for k in before))

    def test_changed_content_tail_and_auditor_invalidate_reuse(self):
        path = self.paths[0]
        raw = path.read_bytes()
        stamp = path.stat()
        changed = raw.replace(b'"seed":71', b'"seed":73', 1)
        self.assertNotEqual(changed, raw)
        path.write_bytes(changed)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "内容已经变化"):
            prepare(self.engine, self.paths, self.root / "bad", audited=self.results)
        with self.assertRaisesRegex(ValueError, "hash"):
            audit(self.engine, path)
        self.assertFalse((self.root / "bad").exists())
        path.write_bytes(raw)
        for key in ("version", "build", "rulesHash", "schema"):
            report = copy.deepcopy(self.results[0].report)
            report["auditor"][key] = "mismatch"
            with self.assertRaisesRegex(ValueError, "身份不匹配"):
                prepare(self.engine, self.paths, self.root / "bad", audited=[replace(self.results[0], report=report), self.results[1]])
        with self.assertRaisesRegex(ValueError, "实际二进制"):
            prepare(self.engine, self.paths, self.root / "bad", audited=[replace(self.results[0], engine_sha256="0" * 64), self.results[1]])
        prefix = b"".join(raw.splitlines(keepends=True)[:-1]) + b'{"unfinished":'
        path.write_bytes(prefix)
        partial = audit(self.engine, path)
        self.assertFalse(partial.report["complete"])
        self.assertTrue(partial.report["incompleteTail"])
        self.assertIsNone(partial.report["outcome"]["returns"])
        self.assertEqual(partial.report["inputSha256"], hashlib.sha256(prefix).hexdigest())
        with self.assertRaisesRegex(ValueError, "内容已经变化"):
            prepare(self.engine, self.paths, self.root / "bad", audited=self.results)
        prepare(self.engine, self.paths, self.root / "prefix", audited=[partial, self.results[1]])
        data, _ = load_dataset(self.root / "prefix/validation.pt", ModelConfig.tiny())
        self.assertFalse(data["value_mask"].any())
        path.write_bytes(prefix + b"\n")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            audit(self.engine, path)

    def test_source_replaced_after_validation_uses_confirmed_owned_features(self):
        prepare(self.engine, self.paths, self.root / "reference", audited=self.results)
        original = audit_module.validate_audited

        def replaced(executable, paths, results):
            original(executable, paths, results)
            paths[0].write_bytes(b"replaced after content verification\n")

        with patch("haojie_training.native.pipeline.validate_audited", replaced):
            report = prepare(self.engine, self.paths, self.root / "confirmed", audited=self.results)
        self.assert_data_equal(self.root / "reference", self.root / "confirmed")
        self.assertEqual(report["metadata"]["sources"][0]["sha256"], self.results[0].report["inputSha256"])
        self.assertNotEqual(hashlib.sha256(self.paths[0].read_bytes()).hexdigest(), self.results[0].report["inputSha256"])


if __name__ == "__main__":
    unittest.main()
