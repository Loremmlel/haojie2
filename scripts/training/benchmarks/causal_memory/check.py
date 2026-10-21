"""小型行为检查：同结构重算/缓存一致、记忆失效、当前读出和可微参考，不训练。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "training"))

import torch  # noqa: E402
from model import CausalMemoryPolicy  # noqa: E402

from haojie_training.data import validate_inputs  # noqa: E402
from haojie_training.model import ModelConfig, PolicyValueNet  # noqa: E402

CONFIG = ModelConfig(width=64, layers=4, heads=4, ffn=256)
BASE = 6


def example(journals, prefix=2):
    generator = torch.Generator().manual_seed(173)
    base = torch.randn(BASE, CONFIG.entity_features, generator=generator) * 0.1
    base[:, 0] = 2 / 32
    base[[1, 4], 0] = 5 / 32
    globals_ = torch.randn(CONFIG.global_features, generator=generator) * 0.1
    globals_[31] = float(prefix > 0)
    # 追加日志保持先前记录逐字段不变；包含成功、失败及回溯结果。
    journal = torch.zeros(journals, CONFIG.entity_features)
    if journals:
        positions = torch.arange(journals)
        journal[:, 0] = positions / 4096
        journal[:, 1] = positions.remainder(4) / 256
        journal[:, 2] = positions.remainder(5)
        journal[:, 3] = torch.tensor([1, -1, -2])[positions.remainder(3)]
    action = torch.full((prefix, CONFIG.entity_features), prefix / 100)
    kinds = torch.tensor([10, 11, 12, 13, 14, 15] + [240] * prefix + [250] * journals)
    valid = torch.ones(len(kinds), dtype=torch.bool)
    valid[2] = False
    row = {
        "entities": torch.cat((base, action, journal)),
        "kinds": kinds,
        "entity_mask": valid,
        "globals": globals_,
        "candidates": torch.randn(5, CONFIG.action_features, generator=generator) * 0.1,
        "sources": torch.tensor([0, 1, -1, 0, 2]),
        "targets": torch.tensor(
            [BASE if prefix else 3, 4, -1, BASE + prefix if journals else 3, 2]
        ),
        "candidate_mask": torch.tensor([True, True, True, True, False]),
    }
    validate_inputs({key: value.unsqueeze(0) for key, value in row.items()}, CONFIG)
    return row


class CacheChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def model(self, scope):
        torch.manual_seed(29)
        return CausalMemoryPolicy(PolicyValueNet(CONFIG).eval(), query_scope=scope)

    def compare(self, runner, row, root=0, hit=False, atol=2e-6, rtol=2e-5):
        with torch.inference_mode():
            expected = runner.forward_reference(row, BASE)
            actual = runner.forward_cached(row, BASE, root)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
        self.assertEqual(runner.last_event["cache_hit"], hit)
        self.assertEqual(
            runner.last_event["memory_tokens"], BASE + int((row["kinds"] == 250).sum())
        )
        return actual

    def test_append_dynamic_changes_and_invalidations(self):
        for scope in ("active", "readout"):
            with self.subTest(scope=scope):
                runner = self.model(scope)
                row = example(0)
                retained = self.compare(runner, row)
                retained_copy = tuple(value.clone() for value in retained)
                for count in (1, 4, 19):
                    old = [(key.clone(), value.clone()) for key, value in runner._kv]
                    self.compare(runner, example(count), hit=True)
                    for previous, current in zip(old, runner._kv):
                        for left, right in zip(previous, current):
                            # 新日志不能反向改变固定局面或旧日志的深层 K/V。
                            torch.testing.assert_close(
                                left, right[:, :, : left.shape[2]], atol=0, rtol=0
                            )
                self.assertEqual(runner.stats["appended_journal_tokens"], 19)
                self.assertEqual(runner.stats["root_rebuilds"], 1)
                for prefix in (3, 0, 2):
                    row = example(19, prefix)
                    self.compare(runner, row, hit=True)
                    self.assertEqual(runner.last_event["appended_journal_tokens"], 0)
                original = {key: value.clone() for key, value in row.items()}
                baseline = self.compare(runner, row, hit=True)
                # 无效候选不能提升亡者或日志；当前读出仍需要所有有效引用。
                row["sources"][-1], row["targets"][-1] = 4, BASE + 2 + 1
                masked = self.compare(runner, row, hit=True)
                for left, right in zip(baseline, masked):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                row = original
                self.compare(runner, row, root=1)
                row["entities"][0, 7] += 1
                self.compare(runner, row, root=1)
                row["globals"][0] += 1
                self.compare(runner, row, root=1)
                row["entities"][BASE + 2, 3] -= 1
                self.compare(runner, row, root=1)
                self.compare(runner, example(4), root=1)
                with torch.no_grad():
                    runner.model.blocks[0].qkv.weight.mul_(0.9)
                self.compare(runner, example(4), root=1)
                for left, right in zip(retained, retained_copy):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                self.assertEqual(runner.stats["requests"], 15)
                self.assertEqual(runner.stats["root_rebuilds"], 7)

    def test_current_readout_and_differentiable_reference(self):
        for scope in ("active", "readout"):
            with self.subTest(scope=scope):
                runner = self.model(scope)
                row = example(24)
                before = self.compare(runner, row)
                # 旧日志不是动态查询，仍必须影响全局与候选的当前判断。
                row["entities"][BASE + 2 + 2, 3] -= 10
                after = self.compare(runner, row)
                self.assertGreater((before[0][:4] - after[0][:4]).abs().max().item(), 1e-6)
                self.assertGreater((before[1] - after[1]).abs().item(), 1e-6)
                row["entities"].requires_grad_(True)
                logits, value = runner.forward_reference(row, BASE)
                loss = logits[:4].square().mean() + value.square()
                loss.backward()
                self.assertGreater(row["entities"].grad[BASE + 2 + 2].abs().sum().item(), 0)
                self.assertEqual(row["entities"].grad[2].abs().sum().item(), 0)
                for parameter in runner.model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertEqual(runner.stats["requests"], 2)

    def test_precision_change_rebuilds_and_snapshot_ownership(self):
        runner = self.model("active")
        row = example(4)
        self.compare(runner, row)
        saved = {key: value.clone() for key, value in row.items()}
        # 缓存必须独占原始输入快照；调用者复用缓冲后同根也要检测变化。
        row["entities"][0, 1] += 2
        self.assertFalse(torch.equal(runner._cache["entities"], row["entities"][:BASE]))
        self.compare(runner, row)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.compare(runner, row, atol=3e-3, rtol=3e-3)
            self.compare(runner, row, hit=True, atol=3e-3, rtol=3e-3)
        self.compare(runner, row)
        runner.reset_cache()
        self.compare(runner, saved)
        for key in row:
            if key != "entities":
                torch.testing.assert_close(row[key], saved[key], atol=0, rtol=0)
        self.assertEqual(runner.stats["root_rebuilds"], 5)


if __name__ == "__main__":
    unittest.main()
