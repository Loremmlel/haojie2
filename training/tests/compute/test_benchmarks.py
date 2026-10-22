"""基准只比较完整匹配负载；自然截断与捕获开销须明确区分。"""

import copy
import unittest
from unittest.mock import patch

import torch

from scripts.training.benchmarks.compare import compare, validate_round
from scripts.training.benchmarks.cycle import RequestCapture, capture_samples, hardware_identity
from scripts.training.benchmarks.replay import compare_cases


def report():
    return {
        "format": "haojie-cycle-benchmark-v2",
        "complete": True,
        "measurement_eligible": True,
        "network": "entity-transformer-v2",
        "torch": "test",
        "python": "test",
        "parameters": 10,
        "model_config": {"width": 2},
        "initial_model_sha256": "initial",
        "config": {"games_per_round": 2, "updates_per_round": 1, "batch_size": 8},
        "starts_sha256": None,
        "hardware_identity": {"cpu_model": "test", "cpu_capacity": 2},
        "execution": {"device": "cpu", "update_precision": "fp32"},
        "options": {
            "start_indices": None,
            "rounds": 1,
            "ordered_updates": True,
            "threads": 1,
            "device": "cpu",
        },
        "rounds": [
            {
                "round": 0,
                "samples_hash": "samples",
                "updates": 1,
                "samples_consumed": 8,
                "cycle_seconds": 10,
                "sample_seconds": 8,
                "update_seconds": 1,
                "save_seconds": 1,
                "counts": {
                    "tasks": 2,
                    "games": 1,
                    "truncated": 1,
                    "discarded": 0,
                    "retained_samples": 16,
                },
                "tasks": [
                    {
                        "task": 1,
                        "outcome": {"terminated": True, "reason": "terminal"},
                        "final_hash": "terminal",
                    },
                    {
                        "task": 2,
                        "outcome": {"terminated": False, "reason": "commands"},
                        "final_hash": "unknown",
                    },
                ],
            }
        ],
    }


class BenchmarkTests(unittest.TestCase):
    def test_matching_natural_truncation_is_reported_and_partial_work_is_rejected(self):
        before, after = report(), report()
        after["rounds"][0]["cycle_seconds"] = 5
        result = compare(before, after)[0]
        self.assertEqual(result["time_reduction_percent"], 50)
        self.assertEqual(result["truncated"], 1)
        for edit in (
            lambda r: r["counts"].update(tasks=1),
            lambda r: r["counts"].update(discarded=1),
            lambda r: r["counts"].update(games=0),
            lambda r: r["tasks"].pop(),
            lambda r: r["tasks"][1].update(task=1),
            lambda r: r["tasks"][1].update(final_hash=None),
            lambda r: r["tasks"][1]["outcome"].update(reason="cancelled"),
            lambda r: r["tasks"][1]["outcome"].update(reason="error"),
            lambda r: r.update(updates=0),
        ):
            row = copy.deepcopy(before["rounds"][0])
            edit(row)
            with self.assertRaises(ValueError):
                validate_round(row, 2, 1)

    def test_device_hardware_capture_and_old_formats_cannot_mix(self):
        for edit in (
            lambda r: r["hardware_identity"].update(cpu_capacity=4),
            lambda r: r["execution"].update(device="cuda:0"),
            lambda r: r["execution"].update(update_precision="bf16"),
            lambda r: r.update(initial_model_sha256="different"),
            lambda r: r.update(measurement_eligible=False),
            lambda r: r.update(format="haojie-cycle-benchmark-v1"),
            lambda r: r["options"].update(rounds=2),
        ):
            after = report()
            edit(after)
            with self.assertRaises(ValueError):
                compare(report(), after)

    def test_capture_keeps_owned_public_inputs_and_bounded_shape_coverage(self):
        capture = RequestCapture({"entities", "candidates"}, limit=5)
        row = {"entities": torch.ones(8, 2), "candidates": torch.ones(4, 2), "private": "excluded"}
        capture.add([{"input": row}] * 10)
        self.assertEqual(len(capture.rows), 4)
        other = {**row, "entities": torch.ones(100, 2)}
        capture.add([{"input": other}] * 3)
        self.assertEqual(len(capture.rows), 5)
        self.assertEqual(set(capture.rows[0]), {"entities", "candidates"})
        row["entities"].zero_()
        self.assertEqual(capture.rows[0]["entities"].sum().item(), 16)
        row["value"] = torch.tensor(0.5)
        samples = capture_samples([(row, 1, 0)], {"entities", "value"})
        self.assertEqual(set(samples[0]), {"entities", "value"})
        row["value"].zero_()
        self.assertEqual(samples[0]["value"].item(), 0.5)
        arrays = {key: row[key].numpy() for key in ("entities", "candidates")}
        numpy_capture = RequestCapture({"entities", "candidates"})
        numpy_capture.add([{"input": arrays}])
        arrays["candidates"].fill(0)
        self.assertEqual(numpy_capture.rows[0]["candidates"].sum().item(), 8)

    def test_cpu_thread_tuning_requires_explicit_comparison_and_keeps_work_checks(self):
        before, after = report(), report()
        after["options"]["threads"] = 4
        with self.assertRaisesRegex(ValueError, "threads"):
            compare(before, after)
        result = compare(before, after, allow_thread_change=True)[0]
        self.assertEqual(result["execution_changes"], {"threads": {"before": 1, "after": 4}})
        after["rounds"][0]["samples_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "samples_hash"):
            compare(before, after, allow_thread_change=True)
        after["rounds"][0]["samples_hash"] = "samples"
        after["execution"]["update_precision"] = "bf16"
        with self.assertRaisesRegex(ValueError, "execution"):
            compare(before, after, allow_thread_change=True)

    def test_equal_sample_sets_without_ordered_consumption_are_not_matched_work(self):
        before, after = report(), report()
        before["options"]["ordered_updates"] = False
        after["options"]["ordered_updates"] = False
        with self.assertRaisesRegex(ValueError, "ordered-updates"):
            compare(before, after)

    def test_network_change_is_explicit_and_same_network_remains_strict(self):
        before, after = report(), report()
        after["rounds"][0]["samples_hash"] = "different"
        with self.assertRaisesRegex(ValueError, "samples_hash"):
            compare(before, after, allow_network_change=True)
        after["network"] = "entity-transformer-v3"
        with self.assertRaisesRegex(ValueError, "allow-network-change"):
            compare(before, after)
        after["rounds"][0]["counts"].update(games=2, truncated=0, retained_samples=18)
        after["rounds"][0]["tasks"][1]["outcome"].update(terminated=True, reason="terminal")
        after["rounds"][0]["tasks"][1]["final_hash"] = "another-terminal"
        result = compare(before, after, allow_network_change=True)[0]
        self.assertFalse(result["matched_work"])
        self.assertNotIn("time_reduction_percent", result)
        self.assertEqual(result["after"]["samples_hash"], "different")
        before.pop("network")
        with self.assertRaisesRegex(ValueError, "allow-network-change"):
            compare(before, after, allow_network_change=True)

    def test_architecture_comparison_keeps_budget_consumption_and_completion_gates(self):
        for edit in (
            lambda r: r["config"].update(batch_size=4),
            lambda r: r.update(initial_model_sha256="different"),
            lambda r: r.update(measurement_eligible=False),
            lambda r: r["rounds"][0].update(updates=0),
            lambda r: r["rounds"][0].update(samples_consumed=7),
            lambda r: r["rounds"][0]["tasks"][0].update(task=3),
            lambda r: r["rounds"][0]["counts"].update(discarded=1),
            lambda r: r["rounds"][0]["tasks"][0]["outcome"].update(reason="error"),
        ):
            before, after = report(), report()
            after["network"] = "entity-transformer-v3"
            edit(after)
            with self.assertRaises(ValueError):
                compare(before, after, allow_network_change=True)
        before, after = report(), report()
        after["network"] = "entity-transformer-v3"
        for side in (before, after):
            side["rounds"][0]["samples_consumed"] = 7
        with self.assertRaisesRegex(ValueError, "完整且相同"):
            compare(before, after, allow_network_change=True)

    def test_fixed_replay_requires_actual_initial_weights_inputs_and_updates(self):
        before = {
            "network": "v2",
            "source_sha256": "source",
            "torch": "test",
            "python": "test",
            "hardware_identity": {},
            "model_config": {},
            "parameters": 10,
            "initial_model_sha256": "weights",
            "execution": {},
            "options": {"updates": 3},
            "requests_sha256": "requests",
            "samples_sha256": "samples",
            "training_batch_sha256": "batch",
            "work": {"updates": 3},
            "timed_updates": 3,
            "warmup_updates": 1,
            "weight_max_change": 0.1,
            "health": {
                "finite_gradients": True,
                "finite_parameters": True,
                "finite_optimizer": True,
            },
            "seconds": {"training_b8": 10},
        }
        after = copy.deepcopy(before)
        after["seconds"]["training_b8"] = 5
        self.assertEqual(
            compare_cases([before], [after])["training_b8"]["time_reduction_percent"], 50
        )
        for edit in (
            lambda r: r.update(initial_model_sha256="other"),
            lambda r: r.update(requests_sha256="other"),
            lambda r: r.update(training_batch_sha256="other"),
            lambda r: r.update(timed_updates=2),
            lambda r: r.update(warmup_updates=0),
            lambda r: r.update(weight_max_change=0),
            lambda r: r["health"].update(finite_gradients=False),
        ):
            candidate = copy.deepcopy(after)
            edit(candidate)
            with self.assertRaises(ValueError):
                compare_cases([before], [candidate])
        after["source_sha256"] = "another-source"
        with self.assertRaisesRegex(ValueError, "同侧"):
            compare_cases([before, before], [before, after])

    def test_dynamic_memory_is_not_hardware_identity(self):
        hardware = {"logical_cpus": 8, "cpu_capacity": 2, "gpus": [], "ram_available": 100}
        with patch("scripts.training.benchmarks.cycle.sys.platform", "test"):
            first = hardware_identity(hardware)
            hardware["ram_available"] = 50
            self.assertEqual(first, hardware_identity(hardware))
