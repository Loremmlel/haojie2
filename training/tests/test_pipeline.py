"""原生池与冻结 Python 参考差分；只验正确性，不在测试阶段比较性能。"""

import copy
import ctypes
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from training.haojie.native import Environment
from training.haojie.native_pool import NativePool
from training.haojie.runtime.config import Config
from training.haojie.runtime.trainer import Trainer
from training.haojie.runtime.storage import GameBuffer
from training.tests.test_training import bundle


def reference_search():
    """参考只从已提交版本读取，不保留另一条生产搜索路径。"""
    name = "haojie_search_reference_7cc4ba9"
    if name not in sys.modules:
        root = Path(__file__).resolve().parents[2]
        source = subprocess.check_output(
            ["git", "show", "7cc4ba9:training/haojie/search.py"],
            cwd=root,
            text=True,
            encoding="utf-8",
        )
        module = types.ModuleType(name)
        module.__file__ = "git:7cc4ba9:training/haojie/search.py"
        sys.modules[name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    return sys.modules[name]


def feedback(op, count):
    # 使用 BF16 可精确表示的非均匀反馈，覆盖价值和条件概率。
    values = (
        np.array([0.125, 0, 0, 0, 0], np.float32)
        if op == "encode"
        else ((np.arange(count, dtype=np.float32) % 7 - 3) / 16)
    )
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def decoded(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def oracle_command(environment, rng, config, game):
    deep = rng.random() < config.full_fraction
    generator = reference_search().search(
        environment,
        rng,
        simulations=config.simulations if deep else config.fast_simulations,
        depth=config.depth,
        noise=game["opponent"] != "evaluation",
    )
    requests, reply = [], None
    while True:
        try:
            request = next(generator) if reply is None else generator.send(reply)
        except StopIteration as result:
            root, policy, version = result.value
            break
        bits = feedback(request.op, len(request.arrays[0]))
        requests.append(
            (request.op, request.context, tuple(a.copy() for a in request.arrays), bits)
        )
        reply = decoded(bits), 17
    if config.temperature == 0:
        chosen = int(np.argmax(policy))
    else:
        probability = np.power(policy.astype(np.float64), 1 / config.temperature)
        probability /= probability.sum()
        chosen = int(rng.choice(len(probability), p=probability))
    info = environment.call("apply", command=root.edges[chosen].command)
    decisions = root.supervision(policy) if deep else []
    return {
        "requests": requests,
        "info": info,
        "root": root,
        "policy": policy,
        "sample": (
            (
                root.bundle,
                decisions,
                root.info["actor"],
                root.info["ply"],
                version,
            )
            if decisions
            else None
        ),
        "rng": copy.deepcopy(rng.bit_generator.state),
    }


def next_record(pool, check=None):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        records = pool.take_records()
        if records:
            if len(records) != 1:
                raise AssertionError("该验收每次只推进一局")
            return records[0]
        batch = pool.take_requests(wait_ms=2)
        if batch is None:
            continue
        try:
            outputs = [
                check(r) if check else feedback(r.op, len(r.arrays[0]))
                for r in batch.requests
            ]
            batch.reply(outputs, [17] * len(outputs))
        except BaseException:
            # 保留差分首错，同时完成原生批次的所有权交接。
            batch.reply(
                [feedback(r.op, len(r.arrays[0])) for r in batch.requests],
                [17] * len(batch.requests),
            )
            raise
        finally:
            batch.close()
    raise TimeoutError("原生搜索未提交完整命令")


def paused_snapshot(pool):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if pool.barrier()["ready"]:
            return pool.snapshot()
        time.sleep(0.001)
    raise TimeoutError("原生池暂停屏障未完成")


class PipelineTests(unittest.TestCase):
    def test_native_threads_and_independent_lane_ownership(self):
        config = Config(
            workers=1,
            games_per_worker=2,
            warmup_games=0,
            simulations=1,
            fast_simulations=1,
            full_fraction=1,
            max_commands=1,
        )
        with Environment(91) as environment:
            environment.call("apply", command={"type": "summon", "ultimate": False})
            environment.call("apply", command={"type": "summon", "ultimate": False})
            environment.call("apply", command={"type": "begin"})
            initial = environment.call("snapshot")
        with NativePool(config) as pool:
            tids = pool.status()["threads"]
            self.assertEqual(len(tids), 1)
            if os.name == "nt":
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenThread.argtypes = (
                    ctypes.c_uint32,
                    ctypes.c_int,
                    ctypes.c_uint32,
                )
                kernel.OpenThread.restype = ctypes.c_void_p
                kernel.GetProcessIdOfThread.argtypes = (ctypes.c_void_p,)
                kernel.GetProcessIdOfThread.restype = ctypes.c_uint32
                kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
                for tid in tids:
                    handle = kernel.OpenThread(0x0800, 0, tid)
                    self.assertTrue(handle)
                    try:
                        self.assertEqual(
                            kernel.GetProcessIdOfThread(handle), os.getpid()
                        )
                    finally:
                        kernel.CloseHandle(handle)
            for lane in range(2):
                game = {
                    "id": lane,
                    "seed": 91,
                    "learner": 1,
                    "opponent": "self",
                    "difficulty": "easy",
                    "ply_limit": 0,
                }
                restored = {
                    "game": game,
                    "environment": initial,
                    "rng": np.random.default_rng(91 ^ 0xA53113).bit_generator.state,
                    "root_version": None,
                    "teacher": None,
                    "steps": 0,
                }
                pool.assign(lane, game, restored)
            first = None
            deadline = time.monotonic() + 5
            while first is None and time.monotonic() < deadline:
                first = pool.take_requests(max_batch=1, wait_ms=10)
            self.assertIsNotNone(first)
            saved = tuple(a.copy() for a in first.requests[0].arrays)
            # 一个 GPU 批次只有一个租约；尚未回复时，原生线程仍须推进另一局。
            self.assertEqual(pool.status()["pending_requests"], 2)
            for actual, expected in zip(first.requests[0].arrays, saved):
                np.testing.assert_array_equal(actual, expected)
            first.reply(
                [feedback(r.op, len(r.arrays[0])) for r in first.requests], [17]
            )
            first.close()
            records = []
            deadline = time.monotonic() + 10
            while len(records) < 2 and time.monotonic() < deadline:
                records.extend(pool.take_records())
                if len(records) == 2:
                    break
                batch = pool.take_requests(max_batch=1, wait_ms=10)
                if batch:
                    try:
                        batch.reply(
                            [feedback(r.op, len(r.arrays[0])) for r in batch.requests],
                            [17],
                        )
                    finally:
                        batch.close()
            self.assertEqual({r["lane"] for r in records}, {0, 1})
            record = records[0]
            copied = tuple(a.copy() for a in record["sample"]["bundle"])
        for actual, expected in zip(record["sample"]["bundle"], copied):
            np.testing.assert_array_equal(actual, expected)

    def assert_sample(self, actual, expected):
        if expected is None:
            self.assertIsNone(actual)
            return
        state, decisions, actor, ply, version = expected
        self.assertEqual(
            (actual["actor"], actual["ply"], actual["version"]), (actor, ply, version)
        )
        for left, right in zip(actual["bundle"], state):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(len(actual["decisions"]), len(decisions))
        for left, right in zip(actual["decisions"], decisions):
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_allclose(left[1], right[1], rtol=1e-6, atol=1e-7)
            self.assertAlmostEqual(left[2], right[2], places=6)

    def test_native_commands_rng_and_targets_match_reference(self):
        for fraction in (0.0, 1.0):
            with self.subTest(full_fraction=fraction):
                config = Config(
                    workers=2,
                    games_per_worker=1,
                    warmup_games=0,
                    full_fraction=fraction,
                )
                game = {
                    "id": 0,
                    "seed": 91,
                    "learner": 1,
                    "opponent": "self",
                    "difficulty": "easy",
                    "ply_limit": 0,
                }
                rng = np.random.default_rng(game["seed"] ^ 0xA53113)
                count = 0
                with (
                    Environment(game["seed"]) as environment,
                    NativePool(config, audit=True) as pool,
                ):
                    pool.assign(0, game)
                    for step in range(10):
                        expected = oracle_command(environment, rng, config, game)
                        pending = iter(expected["requests"])
                        consumed = 0

                        def check(request):
                            nonlocal consumed, count
                            op, key, arrays, bits = next(pending)
                            self.assertEqual(
                                (request.op, request.key),
                                (op, key),
                                f"step={step} request={consumed} fraction={fraction}",
                            )
                            self.assertEqual(len(request.arrays), len(arrays))
                            for actual, reference in zip(request.arrays, arrays):
                                np.testing.assert_array_equal(actual, reference)
                            consumed += 1
                            count += 1
                            return bits

                        record = next_record(pool, check)
                        self.assertEqual(consumed, len(expected["requests"]))
                        self.assertEqual(record["command"], expected["info"]["command"])
                        for key in ("actor", "ply", "winner", "metrics", "key"):
                            self.assertEqual(record["info"][key], expected["info"][key])
                        self.assert_sample(record.get("sample"), expected["sample"])
                        audit, root = record["audit"], expected["root"]
                        self.assertEqual(audit["rng_after"], expected["rng"])
                        self.assertEqual(audit["graph_nodes"], root.graph_nodes)
                        self.assertEqual(audit["transpositions"], root.transpositions)
                        np.testing.assert_array_equal(
                            audit["policy"], expected["policy"]
                        )
                        self.assertEqual(len(audit["edges"]), len(root.edges))
                        for actual, edge in zip(audit["edges"], root.edges):
                            self.assertEqual(actual["command"], edge.command)
                            self.assertEqual(actual["visits"], edge.visits)
                            self.assertAlmostEqual(
                                actual["prior"], edge.prior, places=6
                            )
                            self.assertAlmostEqual(
                                actual["total"], edge.total, places=6
                            )
                        if step != 9:
                            pool.ack_records([0])
                    pool.pause()
                    pool.ack_records([0])
                    self.assertEqual(
                        paused_snapshot(pool)[0]["environment"],
                        environment.call("snapshot"),
                    )
                    self.assertGreater(count, 0)

    def test_ply_limit_preserves_assignment(self):
        config = Config(workers=1, games_per_worker=2, warmup_games=0, max_ply=10)
        with Environment(31) as environment:
            snapshot = environment.call("snapshot")
        snapshot["state"].update(ply=50, phase="play", summonSlots=0)
        with NativePool(config) as pool:
            for lane, limit in enumerate((50, 0)):
                game = {
                    "id": lane,
                    "seed": 31,
                    "learner": 1,
                    "opponent": "self",
                    "difficulty": "easy",
                    "ply_limit": limit,
                }
                restored = {
                    "game": game,
                    "environment": snapshot,
                    "rng": np.random.default_rng(31 ^ 0xA53113).bit_generator.state,
                    "root_version": None,
                    "teacher": None,
                    "steps": 0,
                }
                pool.assign(lane, game, restored)
            records = []
            deadline = time.monotonic() + 10
            while len(records) < 2 and time.monotonic() < deadline:
                records.extend(pool.take_records(wait_ms=10))
            self.assertEqual(len(records), 2)
            short, long = sorted(records, key=lambda r: r["lane"])
            self.assertTrue(short["done"])
            self.assertEqual(short["truncation"], "ply_limit")
            self.assertIsNone(short["info"]["winner"])
            self.assertEqual(short["info"]["ply"], 51)
            self.assertFalse(long["done"])
            self.assertEqual([r["game"]["ply_limit"] for r in (short, long)], [50, 0])
            pool.pause()
            pool.ack_records([r["lane"] for r in records])
            snapshots = paused_snapshot(pool)
            self.assertEqual([s["game"]["ply_limit"] for s in snapshots], [0])

    def test_pause_inside_decode_replays_original_root(self):
        for opponent in ("self", "teacher"):
            with self.subTest(opponent=opponent):
                self.assert_decode_resume(opponent)

    def assert_decode_resume(self, opponent):
        config = Config(
            workers=1,
            games_per_worker=1,
            simulations=4,
            fast_simulations=4,
            warmup_games=0,
        )
        game = {
            "id": 0,
            "seed": 91,
            "learner": 2,
            "opponent": opponent,
            "difficulty": "easy",
            "ply_limit": 0,
        }
        with NativePool(config, audit=True) as original:
            original.assign(0, game)
            deadline, saved = time.monotonic() + 20, None
            while time.monotonic() < deadline:
                for record in original.take_records():
                    original.ack_records([record["lane"]])
                batch = original.take_requests(wait_ms=2)
                if batch is None:
                    continue
                try:
                    is_decode = any(r.op == "decode" for r in batch.requests)
                    if is_decode:
                        original.pause()
                    batch.reply(
                        [feedback(r.op, len(r.arrays[0])) for r in batch.requests],
                        [17] * len(batch.requests),
                    )
                finally:
                    batch.close()
                if is_decode:
                    saved = paused_snapshot(original)[0]
                    break
            self.assertIsNotNone(saved)
            self.assertEqual(saved["root_version"], 17)
            if opponent == "teacher":
                self.assertIsNotNone(saved["teacher"])
            original.resume()
            expected = next_record(original)
            with NativePool(config, audit=True) as restored:
                restored.assign(0, game, saved)
                actual = next_record(restored)
                self.assertEqual(actual["command"], expected["command"])
                self.assertEqual(actual["info"], expected["info"])
                self.assertEqual(
                    actual["audit"]["rng_after"], expected["audit"]["rng_after"]
                )
                np.testing.assert_array_equal(
                    actual["audit"]["policy"], expected["audit"]["policy"]
                )
                for pool in (original, restored):
                    pool.pause()
                    pool.ack_records([0])
                self.assertEqual(paused_snapshot(original), paused_snapshot(restored))

    def test_phase_barrier_checkpoint_and_idle_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                device="cpu",
                workers=2,
                games_per_worker=1,
                warmup_games=1,
                warmup_updates=2,
                batch=2,
                updates=2,
                seconds=30,
                save_seconds=0,
            )
            trainer = Trainer(directory, config)
            writer = GameBuffer({"id": 999, "opponent": "demo"})
            writer.sample(
                bundle(),
                [
                    (
                        np.zeros((2, 704), np.float32),
                        np.array([0.25, 0.75], np.float32),
                        1.0,
                    )
                ],
                1,
                1,
                -1,
            )
            writer.transition(
                {"ply": 3, "winner": 1, "metrics": [0.1] * 4}, {"type": "end"}
            )
            trainer.replay.add(writer.flush())
            trainer.phase = "imitation"
            trainer.stats["demonstrations"] = 1
            trainer.run()
            self.assertEqual(trainer.imitation_updates, 2)
            self.assertEqual(trainer.stats["updates"], 2)
            self.assertEqual(trainer.phase, "self")
            self.assertEqual(trainer.published, 2)
            restored = Trainer(directory, config, resume=True)
            self.assertEqual(restored.imitation_updates, 2)
            self.assertEqual(restored.stats["trained_samples"], 4)
            for state in restored.optimizer.state.values():
                self.assertEqual(state["step"].dtype, torch.int64)
                self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
                self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
            (Path(directory) / "STOP").touch()
            restored.run()
            self.assertEqual(restored.stats["updates"], 2)


if __name__ == "__main__":
    unittest.main()
