"""真实长短对局跨采样窗口、更新和评测继续；固定权重下逐样本核对有限全集。"""

import gzip
import hashlib
import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from haojie_training.console.controller import Controller
from haojie_training.console.evaluation import Evaluation
from haojie_training.console.sampling import MemoryJobs
from haojie_training.console.storage import Store
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.native.execution import PolicyInference
from haojie_training.native.pool import Pool


class ManualController(Controller):
    def _loop(self):
        pass


@contextmanager
def limited_tasks(limit):
    original = MemoryJobs.claim

    def claim(jobs):
        if jobs.evaluation or jobs.c.counts["tasks"] < limit:
            return original(jobs)
        return None

    with patch.object(MemoryJobs, "claim", claim):
        yield


class HistoryLeaseTests(unittest.TestCase):
    def test_live_history_cap_and_current_route_ownership(self):
        current = Mock(precision="bf16", stats=object())
        frozen = Mock()
        frozen.eval.return_value = frozen
        c = SimpleNamespace(
            config={
                "method": "decomposed-mc-q-v2",
                "temperature": 0.25,
                "max_commands": 20000,
                "max_plies": 1000,
                "games_per_round": 2,
                "history_size": 4,
                "history_percent": 50,
            },
            pool=SimpleNamespace(inference=current, slots={}),
            version="a" * 64,
            generation=0,
            device="cpu",
            trainer=SimpleNamespace(model=Mock(config=object()), updates=0),
            store=SimpleNamespace(names=Mock(return_value=["history/new.pt"]), root=Path("unused")),
        )
        with patch(
            "haojie_training.console.sampling.PolicyInference",
            side_effect=lambda model, *args, **kwargs: Mock(model=model),
        ):
            jobs = MemoryJobs(c, c.version, (frozen, c.version), continuous=True)
            # 初始摘要相同的历史权重仍有独立路由，更新当前模型不会借用其执行器。
            self.assertNotEqual(jobs.current, c.version)
            self.assertIs(jobs.inferences[jobs.current], current)
            self.assertIs(jobs.inferences[c.version].model, frozen)
            for name in "bcd":
                jobs.inferences[name * 64] = Mock(model=object())
            c.pool.slots = {
                i: {"models": [jobs.current, name * 64]} for i, name in enumerate("abcd")
            }
            with patch("haojie_training.console.controller.torch.load") as load:
                self.assertEqual(Controller._historical(c, jobs)[1], "a" * 64)
                load.assert_not_called()
                c.store.names.assert_not_called()
            expired = jobs.inferences["b" * 64]
            del c.pool.slots[1]
            new_model = Mock()
            new_model.to.return_value = new_model.eval.return_value = new_model
            with (
                patch(
                    "haojie_training.console.controller.torch.load",
                    return_value={"version": "e" * 64, "model": {}},
                ),
                patch("haojie_training.console.controller.PolicyValueNet", return_value=new_model),
            ):
                historical = Controller._historical(c, jobs)
            expired.close.assert_called_once()
            self.assertNotIn("b" * 64, jobs.inferences)
            jobs.next_window(c.version, historical)
            self.assertIs(jobs.inferences["e" * 64].model, new_model)
            self.assertEqual(len(jobs.inferences), 5)
            self.assertIs(jobs.inferences[jobs.current], current)


class RootLeaseTests(unittest.TestCase):
    def test_root_snapshot_survives_update_and_publication_is_bounded(self):
        torch.set_num_threads(1)
        model = PolicyValueNet(ModelConfig.tiny())
        first = PolicyInference(deepcopy(model).eval().requires_grad_(False), "cpu")
        c = SimpleNamespace(
            config={
                "method": "decomposed-mc-q-v2",
                "temperature": 0.25,
                "max_commands": 20000,
                "max_plies": 1000,
                "games_per_round": 2,
            },
            pool=SimpleNamespace(inference=first, slots={}),
            generation=0,
            device="cpu",
            trainer=SimpleNamespace(model=model, updates=0),
        )
        jobs = MemoryJobs(c, "a" * 64, continuous=True)
        task = {"models": [jobs.current] * 2, "collector": Mock()}
        c.pool.slots[0] = task
        message = {"model": jobs.current, "commandIndex": 0, "actor": 1}
        jobs.bind(task, message)
        before = next(first.model.parameters()).clone()
        try:
            for index, version in enumerate("bcd", 1):
                with torch.no_grad():
                    next(model.parameters()).add_(0.25)
                c.generation, c.trainer.updates = index, index * 32
                jobs.next_window(version * 64, None)
                following = {**message, "commandIndex": index}
                slot = {"models": [jobs.current] * 2, "collector": Mock()}
                c.pool.slots[index] = slot
                jobs.bind(slot, following)
                self.assertIsNot(jobs.inference_for(following), first)
                self.assertFalse(
                    any(p.requires_grad for p in jobs.inference_for(following).model.parameters())
                )
                continuation = {key: message[key] for key in ("model", "commandIndex", "actor")}
                jobs.bind(task, continuation)
                self.assertIs(jobs.inference_for(continuation), first)
                self.assertEqual(continuation["_behavior"], (0, 1, "a" * 64, 0))
                torch.testing.assert_close(next(first.model.parameters()), before, atol=0, rtol=0)
            c.trainer.updates = 128
            jobs.next_window("e" * 64, None)
            self.assertEqual(jobs.version, "d" * 64)
            self.assertEqual(
                jobs.publication_metrics(),
                {
                    "published_model": "d" * 64,
                    "root_snapshots": 4,
                    "coalesced_publications": 1,
                    "oldest_root_update_lag": 128,
                },
            )
            c.pool.slots.clear()
            jobs.next_window("e" * 64, None)
            self.assertEqual(jobs.version, "e" * 64)
            self.assertLessEqual(len(jobs.root_versions), 2)
        finally:
            jobs.close()


@unittest.skipUnless(os.environ.get("HAOJIE_NATIVE"), "需要真实 Rust 引擎")
class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.engine = os.environ["HAOJIE_NATIVE"]
        root = Path(__file__).resolve().parents[3]
        fixture = root / "tests/fixtures/native/late-starts.json.gz"
        starts = json.loads(
            gzip.decompress(fixture.read_bytes())
            if fixture.exists()
            else (root / "fixtures/late-starts.json").read_bytes()
        )
        cls.starts = [starts[0], starts[1], starts[0]]

    @contextmanager
    def controller(self, continuous=True):
        with tempfile.TemporaryDirectory() as folder:
            c = ManualController(
                self.engine, Store(folder), device="cpu", tiny=True, starts=self.starts, threads=1
            )
            c.thread.join()
            c.continuous_selfplay = continuous
            c.config.update(
                environments=2,
                games_per_round=2,
                updates_per_round=1,
                history_percent=0,
                pool_mib=128,
                eval_games=100000,
            )
            c.want_run = True
            try:
                c._initialize()
                yield c
            finally:
                c._close_pool()
                c.close()

    def test_long_game_crosses_real_update_and_keeps_behavior_labels(self):
        with self.controller() as c, limited_tasks(3):
            c._sample()
            self.assertEqual(c.counts["games"], 1)
            carried = {task["id"] for task in c.pool.slots.values()}
            self.assertTrue(carried)
            self.assertTrue(
                all(row["_meta"]["game"]["task"] not in carried for row, *_ in c.samples.rows)
            )
            old_version, pool = c.version, c.pool
            c._update()
            self.assertEqual(c.trainer.updates, 1)
            self.assertNotEqual(c.version, old_version)
            for _ in range(3):
                if c.counts["games"] == 3:
                    break
                c._sample()
            self.assertIs(c.pool, pool)
            self.assertEqual(c.counts["games"], 3)
            self.assertEqual(c.counts["truncated"], 0)
            self.assertEqual(c.counts["discarded"], 0)
            rows = [row for row, *_ in c.samples.rows if row["_meta"]["game"]["task"] in carried]
            self.assertTrue(rows)
            versions = {row["_meta"]["behavior_version"] for row in rows}
            self.assertIn(c.version, versions)
            self.assertLessEqual(versions, {old_version, c.version})
            self.assertIn(1, {row["_meta"]["behavior_generation"] for row in rows})
            self.assertTrue(all(bool(row["value_mask"]) for row in rows))
            self.assertFalse(c.pool.slots)

    def test_evaluation_preserves_parked_training_games(self):
        with self.controller() as c, limited_tasks(3):
            c._sample()
            pool, jobs = c.pool, c.sampling_jobs
            self.assertTrue(pool.slots)
            task_objects = dict(pool.slots)
            progress = {i: task["trajectory"].seen for i, task in pool.slots.items()}
            pids = [client.process.pid for client in pool.clients]
            evaluation = Evaluation(c, c._weights())
            evaluation.plan = [
                {
                    "difficulty": "baseline",
                    "start": self.starts[0],
                    "side": 1,
                    "sampler_seed": 91,
                    "result": "pending",
                }
            ]
            report = evaluation.slice()
            self.assertIsNotNone(report)
            self.assertEqual(report["results"]["baseline"]["n"], 1)
            self.assertIs(c.pool, pool)
            self.assertIs(c.sampling_jobs, jobs)
            self.assertEqual([client.process.pid for client in pool.clients], pids)
            self.assertEqual(set(pool.slots), set(task_objects))
            for i, task in pool.slots.items():
                self.assertIs(task, task_objects[i])
                self.assertEqual(task["trajectory"].seen, progress[i])
            self.assertEqual(c.counts["discarded"], 0)
            c._close_pool()
            self.assertEqual(c.counts["discarded"], len(task_objects))
            c._close_pool()
            self.assertEqual(c.counts["discarded"], len(task_objects))

    def test_shared_inputs_keep_long_game_across_updates(self):
        with patch.dict(os.environ, {"HAOJIE_HOST_WORKERS": "2"}):
            self.test_long_game_crosses_real_update_and_keeps_behavior_labels()

    def test_empty_truncated_window_returns_without_learning(self):
        with self.controller() as c:
            c.config.update(max_commands=1)
            c._sample()
            self.assertGreaterEqual(c.counts["truncated"], 2)
            self.assertEqual(c.counts["games"], 0)
            self.assertFalse(c.samples.rows)
            c._update()
            self.assertEqual(c.trainer.updates, 0)

    def test_resume_consumes_retained_samples_before_sampling(self):
        with self.controller() as c:
            c._sample()
            c._close_pool()
            self.assertTrue(c.samples.rows)
            update = c._update

            def stop_after_update():
                update()
                c.shutdown.set()

            def unexpected_sample():
                c.shutdown.set()
                raise AssertionError("不应先等待采样")

            with (
                patch.object(c, "_sample", side_effect=unexpected_sample) as sample,
                patch.object(c, "_update", side_effect=stop_after_update),
            ):
                Controller._loop(c)
            sample.assert_not_called()
            self.assertIsNone(c.error)
            self.assertEqual(c.trainer.updates, 1)

    def test_runtime_concurrency_keeps_games_and_retires_only_after_terminal(self):
        with self.controller() as c:
            c._sample()
            c._update()
            old_pool, trainer, optimizer = c.pool, c.trainer, c.trainer.optimizer
            carried = {
                task["id"]: (slot, old_pool.clients[slot].process.pid, task["began"])
                for slot, task in old_pool.slots.items()
            }
            discarded = c.counts["discarded"]
            updates = trainer.updates
            retained = c.counts["retained_samples"]
            weights = [parameter.detach().clone() for parameter in trainer.model.parameters()]
            states = {
                parameter: {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in state.items()
                }
                for parameter, state in optimizer.state.items()
            }
            completed = {}
            original = MemoryJobs.finish

            def finish(jobs, task, message):
                completed[task["id"]] = message["outcome"]
                return original(jobs, task, message)

            def settle(target):
                for _ in range(40):
                    c._sample()
                    self.assertIs(c.pool, old_pool)
                    self.assertEqual(c.counts["discarded"], discarded)
                    if old_pool.environments == target:
                        return
                self.fail(f"并发没有收敛到 {target}")

            self.assertGreater(retained, 0)
            self.assertTrue(carried)
            c.command("concurrency", {"environments": 3})
            self.assertIs(c.pool, old_pool)
            self.assertEqual(
                c.status()["concurrency"],
                {
                    "requested": 3,
                    "actual": 2,
                    "pending": True,
                },
            )
            for invalid in (0, 129, True, 1.5):
                with self.assertRaises(ValueError):
                    c.command("concurrency", {"environments": invalid})
            with self.assertRaises(ValueError):
                c.command("concurrency", {"environments": 3, "batch_size": 1})
            for state in ("starting", "pausing"):
                c.state = state
                with self.assertRaisesRegex(ValueError, "请等待"):
                    c.command("concurrency", {"environments": 4})
            c.state = "sampling"
            with self.assertRaisesRegex(ValueError, "先暂停"):
                c.command("settings", {"batch_size": 1})
            with patch.object(MemoryJobs, "finish", finish):
                settle(3)
                active = {task["id"]: slot for slot, task in old_pool.slots.items()}
                for task_id, (slot, pid, began) in carried.items():
                    if task_id in active:
                        self.assertEqual(active[task_id], slot)
                        self.assertEqual(old_pool.clients[slot].process.pid, pid)
                        self.assertEqual(old_pool.slots[slot]["began"], began)
                    else:
                        self.assertTrue(completed[task_id]["terminated"])
                c.command("concurrency", {"environments": 1})
                c._sample()
                c.command("concurrency", {"environments": 4})
                settle(4)
                # 连续改目标只服从最新值，已有局仍由原池和原生进程承接。
                for target in (1, 4, 3):
                    c.command("concurrency", {"environments": target})
                settle(3)
                c.command("concurrency", {"environments": 1})
                settle(1)
            self.assertFalse(old_pool.closed)
            self.assertIs(c.trainer, trainer)
            self.assertIs(c.trainer.optimizer, optimizer)
            self.assertEqual(trainer.updates, updates)
            self.assertEqual(c.counts["discarded"], discarded)
            self.assertGreaterEqual(c.counts["retained_samples"], retained)
            for actual, expected in zip(trainer.model.parameters(), weights, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for parameter, state in states.items():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(
                            optimizer.state[parameter][key], value, atol=0, rtol=0
                        )
                    else:
                        self.assertEqual(optimizer.state[parameter][key], value)
            self.assertEqual(
                c.status()["concurrency"],
                {
                    "requested": 1,
                    "actual": 1,
                    "pending": False,
                },
            )
            c._update()
            self.assertGreater(trainer.updates, updates)
            self.assertEqual(c.pool.environments, 1)
            self.assertTrue(c.want_run)
            self.assertEqual(c.config["games_per_round"], 2)
            c._save()
            recovery = torch.load(
                c.store.root / c.store.names("recovery")[-1],
                weights_only=True,
            )
            self.assertEqual(recovery["config"]["environments"], 1)

    def test_running_resize_does_not_wait_for_new_client_handshake(self):
        with self.controller() as c:
            started, proceed, creating, release = (threading.Event() for _ in range(4))
            initial, errors = {}, []
            progress, create = c._progress, Pool._start_client

            def snapshot(metrics):
                if not started.is_set():
                    initial.update(
                        (task["id"], c.pool.clients[slot].process.pid)
                        for slot, task in c.pool.slots.items()
                    )
                    started.set()
                    if not proceed.wait(10):
                        raise TimeoutError("测试未发布新并发目标")
                return progress(metrics)

            def start_client(pool, slot):
                client = create(pool, slot)
                creating.set()
                if not release.wait(10):
                    client.close()
                    raise TimeoutError("测试未解除新槽握手等待")
                return client

            def sample():
                try:
                    c._sample()
                except BaseException as error:
                    errors.append(error)

            with (
                patch.object(c, "_progress", snapshot),
                patch.object(Pool, "_start_client", start_client),
            ):
                thread = threading.Thread(target=sample)
                thread.start()
                try:
                    self.assertTrue(started.wait(10))
                    c.command("concurrency", {"environments": 3})
                    proceed.set()
                    self.assertTrue(creating.wait(10))
                    # 新槽尚未交付，已有两个真实 Rust 对局仍能完成首个学习窗口。
                    thread.join(10)
                    self.assertFalse(thread.is_alive())
                    self.assertFalse(errors)
                    self.assertGreater(c.counts["games"], 0)
                    self.assertEqual(c.counts["discarded"], 0)
                    self.assertIsNotNone(c.pool.starting)
                    self.assertEqual(c.pool.environments, 2)
                    for slot, task in c.pool.slots.items():
                        self.assertEqual(c.pool.clients[slot].process.pid, initial[task["id"]])
                    c.command("concurrency", {"environments": 1})
                    release.set()
                    for _ in range(40):
                        c._sample()
                        if c.pool.environments == 1 and c.pool.starting is None:
                            break
                    self.assertEqual(c.pool.environments, 1)
                    self.assertIsNone(c.pool.starting)
                    self.assertEqual(c.counts["discarded"], 0)
                finally:
                    proceed.set()
                    release.set()
                    thread.join(10)

    def test_fixed_weights_complete_identical_finite_task_set(self):
        summaries = []
        for continuous in (False, True):
            with self.controller(continuous) as c, limited_tasks(3):
                for _ in range(3):
                    if c.counts["games"] == 3:
                        break
                    c._sample()
                    self.assertEqual(c.counts["truncated"], 0)
                    self.assertEqual(c.counts["discarded"], 0)
                    if not continuous:
                        self.assertIsNone(c.sampling_jobs)
                        self.assertEqual(c.runtime["historical_models"], 0)
                self.assertEqual(c.counts["games"], 3)
                self.assertEqual(c.counts["tasks"], 3)
                self.assertFalse(c.pool.slots)
                samples = []
                for row, *_ in c.samples.rows:
                    meta = row["_meta"]
                    digest = hashlib.sha256()
                    for key in sorted(k for k in row if not k.startswith("_")):
                        value = row[key].contiguous()
                        digest.update(f"{key}:{value.dtype}:{list(value.shape)}".encode())
                        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
                    samples.append((meta["game"]["task"], meta["decision"], digest.hexdigest()))
                self.assertTrue(samples)
                summaries.append((c.counts["seen_samples"], sorted(samples)))
        self.assertEqual(summaries[0], summaries[1])


if __name__ == "__main__":
    unittest.main()
