"""主线程与输入线程的计时分开，避免用跨线程时钟拼出无效累计时间。"""

import threading
import time
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from haojie_training.console.config.resources import _cpu_total
from haojie_training.console.memory import SamplePool, Trajectory
from haojie_training.diagnostics import MeasuredJobs, profiling, sample_fingerprint
from haojie_training.model import ModelConfig
from haojie_training.native import client

from .test_packed import examples


class DiagnosticsTests(unittest.TestCase):
    def test_cpu_work_transfers_from_live_children_to_reaped_parent_once(self):
        before = {
            1: dict(cpu=5, reaped_cpu=1),
            2: dict(cpu=2, reaped_cpu=3),
            3: dict(cpu=4, reaped_cpu=0),
        }
        child_reaped = {1: dict(cpu=5.5, reaped_cpu=1), 2: dict(cpu=2.5, reaped_cpu=8)}
        all_reaped = {1: dict(cpu=6, reaped_cpu=12)}
        # 采样期间新增 CPU 为父进程 1 秒、输入进程 1 秒、引擎 1 秒。
        self.assertEqual(_cpu_total(child_reaped) - _cpu_total(before), 2)
        self.assertEqual(_cpu_total(all_reaped) - _cpu_total(before), 3)

    def test_closed_work_keeps_terminal_truncated_and_unknown_separate(self):
        counts = dict.fromkeys(
            (
                "games",
                "retained_samples",
                "truncated",
                "discarded",
                "seen_samples",
                "oversize_samples",
            ),
            0,
        )
        controller = SimpleNamespace(counts=counts, samples=SamplePool(2**20), closed_tasks=[])
        jobs = MeasuredJobs.__new__(MeasuredJobs)
        jobs.c, jobs.evaluation, jobs.finished = controller, False, 0
        row = examples(ModelConfig.tiny())[1]
        tasks = []
        for task_id, requests in enumerate((7, 11, 13), 1):
            trace = Trajectory(task_id)
            trace.add(
                dict(
                    input=row,
                    selected=0,
                    actor=1,
                    step=0,
                    stage="target",
                    bytes=sum(value.numel() * value.element_size() for value in row.values()),
                    candidates=len(row["sources"]),
                )
            )
            tasks.append(
                dict(
                    id=task_id,
                    requests=requests,
                    began=time.perf_counter(),
                    trajectory=trace,
                    context=dict(rules="classic", matchup="self"),
                )
            )
        states = [task["trajectory"].rng.getstate() for task in tasks]
        jobs.finish(tasks[0], dict(outcome=dict(terminated=True, returns={"1": 1, "2": -1})))
        jobs.finish(tasks[1], dict(outcome=dict(terminated=False, returns=None)))
        jobs.unfinished(tasks[2])
        self.assertEqual([counts[key] for key in ("games", "truncated", "discarded")], [1, 1, 1])
        self.assertEqual(counts["retained_samples"], 1)
        self.assertEqual(len(controller.samples.rows), 1)
        self.assertEqual(controller.samples.rows[0][0]["value"].item(), 1)
        self.assertEqual(
            [row["status"] for row in controller.closed_tasks], ["terminal", "truncated", "unknown"]
        )
        self.assertEqual([row["requests"] for row in controller.closed_tasks], [7, 11, 13])
        self.assertEqual([row["retained_samples"] for row in controller.closed_tasks], [1, 0, 0])
        self.assertEqual([row["returns"] for row in controller.closed_tasks[1:]], [None, None])
        self.assertEqual(states, [task["trajectory"].rng.getstate() for task in tasks])

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
