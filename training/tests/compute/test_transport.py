"""真实进程共享帧的所有权、二进制评分等价与取消；短命令预算仅用于协议验收。"""

import gzip
import hashlib
import json
import os
import queue
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from haojie_training.native import client as client_module
from haojie_training.native.client import Client, decode
from haojie_training.native.model import example
from haojie_training.native.packed import PackedPolicy, prepare_row
from haojie_training.native.transport import FrameBuffer, InputProcesses, SharedInput, worker_count

from .test_packed import examples


class InputTests(unittest.TestCase):
    def test_shared_ring_retires_growth_instead_of_accumulating_descriptors(self):
        frames = FrameBuffer()
        try:
            for power in range(4, 18):
                view = frames.acquire(2**power)
                _, lease = frames.retain()
                frames.release(lease)
                view.release()
                self.assertLessEqual(len(frames.segments), 2)
            self.assertEqual(len(frames.retired), 12)
        finally:
            frames.close()

    def test_native_validation_keeps_numpy_boundaries(self):
        if client_module.native_io() is None:
            self.skipTest("需要已构建 C++ 输入校验")
        from haojie_training.model import ModelConfig

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
        source = examples(ModelConfig())[0]
        for key, index, value in (
            (None, None, None),
            ("entities", (0, 1), float("nan")),
            ("globals", 0, float("inf")),
            ("candidates", (0, 0), -float("inf")),
            ("kinds", 0, -1),
            ("kinds", 0, 256),
            ("candidate_mask", slice(None), False),
            ("sources", 0, -2),
            ("sources", 0, 31),
            ("sources", 0, 17),
            ("targets", 0, 31),
            ("targets", 0, 17),
        ):
            row = {k: v.clone() for k, v in source.items()}
            if key is not None:
                row[key][index] = value
            frame = b"".join(row[k].numpy().tobytes() for k in keys)
            result = []
            for native in (client_module.native_io, lambda: None):
                with patch.object(client_module, "native_io", native):
                    try:
                        actual = decode(
                            frame, len(row["entities"]), len(row["sources"]), _arrays=True
                        )
                        result.append(None)
                        for field in keys:
                            np.testing.assert_array_equal(actual[field], row[field].numpy())
                    except ValueError as error:
                        result.append(str(error))
            self.assertEqual(result[0], result[1], (key, index, value))

    def test_native_bulk_reply_releases_gil_and_preserves_bytes(self):
        if client_module.native_io() is None:
            self.skipTest("需要已构建 C++ 批量回传")
        reader, writer = os.pipe()
        stream = os.fdopen(writer, "wb", buffering=0)
        client = SimpleNamespace(
            ready={"capabilities": ["logits-f32-v1"]},
            process=SimpleNamespace(stdin=stream),
        )
        scores = np.linspace(-4, 4, 100000, dtype=np.float32)
        response = {"id": 7, "model": '带引号"的版本', "logits": scores}
        received = bytearray()

        def receive():
            while chunk := os.read(reader, 65536):
                received.extend(chunk)

        thread = threading.Thread(target=receive)
        thread.start()
        try:
            client_module.send_many([(client, response)])
        finally:
            stream.close()
            thread.join(timeout=5)
            os.close(reader)
        self.assertFalse(thread.is_alive())
        header, payload = received.split(b"\n", 1)
        self.assertEqual(
            json.loads(header), {"id": 7, "model": response["model"], "logitsBytes": scores.nbytes}
        )
        self.assertEqual(payload, scores.tobytes())

    def test_native_bulk_reply_rejects_nonfinite_before_writing(self):
        if client_module.native_io() is None:
            self.skipTest("需要已构建 C++ 批量回传")
        reader, writer = os.pipe()
        with os.fdopen(writer, "wb", buffering=0) as stream:
            client = SimpleNamespace(
                ready={"capabilities": ["logits-f32-v1"]},
                process=SimpleNamespace(stdin=stream),
            )
            with self.assertRaisesRegex(ValueError, "模型输出非有限"):
                client_module.send_many(
                    [
                        (client, {"id": 1, "logits": np.zeros(1, dtype=np.float32)}),
                        (client, {"id": 2, "logits": np.array([np.nan], dtype=np.float32)}),
                    ]
                )
        self.assertEqual(os.read(reader, 1), b"")
        os.close(reader)

    def test_two_frame_ring_waits_for_all_references_before_overwrite(self):
        frames = FrameBuffer()
        first = frames.acquire(8)
        first[:] = b"original"
        _, inference = frames.retain()
        _, example_lease = frames.retain()
        second = frames.acquire(8)
        _, other = frames.retain()
        started, acquired = threading.Event(), threading.Event()

        def overwrite():
            started.set()
            value = frames.acquire(8)
            value[:] = b"replaced"
            value.release()
            acquired.set()

        thread = threading.Thread(target=overwrite)
        try:
            thread.start()
            self.assertTrue(started.wait(2))
            frames.release(inference)
            self.assertFalse(acquired.wait(0.05))
            self.assertEqual(bytes(first), b"original")
            frames.release(example_lease)
            self.assertTrue(acquired.wait(2))
            self.assertEqual(bytes(first), b"replaced")
            frames.release(other)
        finally:
            frames.stop()
            thread.join(timeout=2)
            first.release()
            second.release()
            frames.close()

    def test_prepared_indices_and_retained_sample_own_their_memory(self):
        from types import SimpleNamespace

        from haojie_training.model import ModelConfig

        rows = examples(ModelConfig())
        arrays = [{key: value.numpy() for key, value in row.items()} for row in rows]
        shared = [SharedInput(row, prepare_row(row)) for row in arrays]
        packed = PackedPolicy(SimpleNamespace(config=ModelConfig()), {}, False)
        before, layout = packed.batch(arrays)
        before = {key: value.clone() for key, value in before.items()}
        actual, actual_layout = packed.batch(shared)
        self.assertEqual(layout, actual_layout)
        for key in before:
            torch.testing.assert_close(before[key], actual[key], atol=0, rtol=0)
        retained = example(shared[0])
        snapshot = retained["entities"].clone()
        shared[0]["entities"].fill(0)
        torch.testing.assert_close(retained["entities"], snapshot, atol=0, rtol=0)

    def test_numpy_decode_boundary_and_external_buffer_ownership(self):
        from haojie_training.model import ModelConfig

        row = examples(ModelConfig())[1]
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
        frame = bytearray(b"".join(row[key].numpy().tobytes() for key in keys))
        output = decode(frame, len(row["entities"]), len(row["sources"]), _arrays=True)
        deferred = decode(
            frame, len(row["entities"]), len(row["sources"]), _arrays=True, _defer=True
        )
        self.assertIsNone(deferred._values)
        self.assertFalse(deferred.validated)
        deferred.ensure_validated()
        self.assertTrue(deferred.validated)
        frame[:] = bytes(len(frame))
        for key in keys:
            np.testing.assert_array_equal(output[key], row[key].numpy())
            np.testing.assert_array_equal(deferred[key], row[key].numpy())
        for arrays in (False, True):
            with self.assertRaisesRegex(ValueError, "可训练候选"):
                decode(frame, len(row["entities"]), len(row["sources"]), _arrays=arrays)

    def test_workers_use_actual_resources_and_respect_explicit_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(worker_count(30, "cuda"), 0)
        with (
            patch.dict(os.environ, {"HAOJIE_HOST_WORKERS": "auto"}, clear=True),
            patch(
                "haojie_training.console.config.resources.cpu_capacity", return_value=12
            ) as cores,
            patch(
                "haojie_training.console.config.resources.available_memory", return_value=16 * 2**30
            ) as memory,
        ):
            self.assertEqual(worker_count(30), 0)
            self.assertEqual(worker_count(30, "cuda"), 4)
            self.assertEqual(worker_count(3, "cuda"), 0)
            cores.return_value = 44
            self.assertEqual(worker_count(88, "cuda"), 14)
            cores.return_value = 1.5
            self.assertEqual(worker_count(30, "cuda"), 0)
            cores.return_value = 12
            memory.return_value = 2 * 2**30
            self.assertEqual(worker_count(30, "cuda"), 0)
        with patch.dict(os.environ, {"HAOJIE_HOST_WORKERS": "2"}):
            self.assertEqual(worker_count(30), 2)
            self.assertEqual(worker_count(1), 1)
        with patch.dict(os.environ, {"HAOJIE_HOST_WORKERS": "0"}):
            self.assertEqual(worker_count(30, "cuda"), 0)
        with patch.dict(os.environ, {"HAOJIE_HOST_WORKERS": "17"}):
            with self.assertRaises(ValueError):
                worker_count(30)


@unittest.skipUnless(os.environ.get("HAOJIE_NATIVE"), "需要真实 Rust 引擎")
class ProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = os.environ["HAOJIE_NATIVE"]
        root = Path(__file__).resolve().parents[3]
        fixture = root / "tests/fixtures/native/late-starts.json.gz"
        cls.start = json.loads(
            gzip.decompress(fixture.read_bytes())
            if fixture.exists()
            else (root / "fixtures/late-starts.json").read_bytes()
        )[0]

    def request(self):
        return {
            "op": "sample",
            "start": self.start,
            "model": "a" * 64,
            "samplerSeed": 91,
            "maxCommands": 4,
            "maxPlies": 1000,
            "memory": True,
            "mcContext": True,
            "inputRefs": True,
        }

    def run_protocol(self, client, receive, *, binary, native_batch=False):
        client.send(self.request())
        digest, count = hashlib.sha256(), 0
        retained, snapshot = None, None
        while True:
            message = receive()
            if message["type"] == "done":
                self.assertFalse(message["outcome"]["terminated"])
                self.assertIsNone(message["outcome"]["returns"])
                break
            for key, value in sorted(message["input"].items()):
                digest.update(key.encode())
                digest.update(value.tobytes())
            if message["type"] == "infer":
                if isinstance(message["input"], SharedInput):
                    if client.owner.prepare_inputs:
                        expected = prepare_row(message["input"])
                        for actual, reference in zip(message["input"].packing, expected):
                            np.testing.assert_array_equal(actual, reference)
                    else:
                        self.assertIsNone(message["input"].packing)
                scores = np.linspace(-3, 3, message["candidates"], dtype=np.float32)
                response = {
                    "id": message["id"],
                    "model": message["model"],
                    "logits": scores if binary else scores.tolist(),
                }
                if native_batch:
                    client_module.send_many([(client, response)])
                else:
                    client.send(response)
                count += 1
            else:
                digest.update(str(message["selected"]).encode())
                if retained is None:
                    retained = example(message["input"], message["selected"])
                    snapshot = retained["entities"].clone()
            client.release(message)
        torch.testing.assert_close(retained["entities"], snapshot, atol=0, rtol=0)
        return digest.hexdigest(), count

    def test_shared_workers_match_json_trajectory_and_release_buffers(self):
        with Client(self.engine, arrays=True) as client:
            expected = self.run_protocol(client, client.receive, binary=False)
        with Client(self.engine, arrays=True) as client:
            self.assertEqual(
                self.run_protocol(client, client.receive, binary=True, native_batch=True), expected
            )
        events = queue.Queue(maxsize=4)
        for prepare_inputs in (False, True):
            owner = InputProcesses(
                self.engine, 2, 2, events, 4 * 2**20, prepare_inputs=prepare_inputs
            )
            try:
                clients = [owner.client(slot, 0) for slot in range(2)]
                for slot, client in enumerate(clients):

                    def receive():
                        identity, raw = events.get(timeout=30)
                        self.assertEqual(identity, (slot, 0))
                        return client.checked(raw)

                    self.assertEqual(self.run_protocol(client, receive, binary=True), expected)
                self.assertGreater(len(owner.segments), 0)
                self.assertLessEqual(len(owner.segments), 4)
            finally:
                owner.close()
            self.assertFalse(owner.segments)
            self.assertTrue(all(not process.is_alive() for process in owner.workers))

    def test_inflight_cancel_and_resource_restart(self):
        events = queue.Queue(maxsize=2)
        owner = InputProcesses(self.engine, 1, 1, events, 1024)
        try:
            client = owner.client(0, 0)
            client.send(self.request())
            identity, error = events.get(timeout=30)
            self.assertEqual(identity, (0, 0))
            self.assertIsInstance(error, MemoryError)
            client.close()
            replacement = owner.client(0, 1)
            self.assertNotEqual(replacement.process.pid, client.process.pid)
        finally:
            owner.close()

    def test_malformed_binary_logits_fail_without_terminal_labels(self):
        for bad in ("size", "nan", "both"):
            with self.subTest(bad=bad), Client(self.engine) as client:
                client.send(self.request())
                message = client.receive()
                count = message["candidates"]
                header = {"id": message["id"], "model": message["model"], "logitsBytes": count * 4}
                payload = np.full(count, np.nan, dtype="<f4").tobytes()
                if bad == "size":
                    header["logitsBytes"] += 1
                    payload = b""
                elif bad == "both":
                    header["logits"] = [0.0] * count
                    payload = b""
                client.process.stdin.write(json.dumps(header).encode() + b"\n" + payload)
                client.process.stdin.flush()
                done = client.receive()
                self.assertEqual(done["outcome"]["reason"], "error")
                self.assertIsNone(done["outcome"]["returns"])
                self.assertIn("invalid model logits", done["error"])

    def test_inflight_frame_stays_owned_until_release(self):
        owner = InputProcesses(self.engine, 1, 1, queue.Queue(maxsize=2), 4 * 2**20)
        try:
            client = owner.client(0, 0)
            client.send(self.request())
            identity, raw = owner.events.get(timeout=30)
            message = client.checked(raw)
            self.assertEqual(message["type"], "infer")
            client.send({"id": message["id"], "model": message["model"], "cancel": True})
            snapshot = message["input"]["entities"].copy()
            _, done = owner.events.get(timeout=30)
            np.testing.assert_array_equal(message["input"]["entities"], snapshot)
            client.release(message)
            self.assertEqual(done["outcome"]["reason"], "cancelled")
            self.assertIsNone(done["outcome"]["returns"])
        finally:
            owner.close()


if __name__ == "__main__":
    unittest.main()
