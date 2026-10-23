"""新协议的概率、bootstrap、席位连接与优化器提交边界。"""

import copy
import unittest

import torch

from haojie_training.online.learner import Learner, vtrace
from haojie_training.online.model import Config, Network, node_batch, root_batch
from haojie_training.online.optimizer import AdamW
from haojie_training.online.staging import Staging
from haojie_training.online.trajectory import Trajectories


def example(key=(1, 1), events=0):
    root = dict(
        entities=torch.randn(3, 64, dtype=torch.bfloat16),
        kinds=torch.tensor([1, 2, 3]),
        globals=torch.randn(32, dtype=torch.bfloat16),
    )
    node = dict(
        root=key,
        prefix=torch.zeros(1, 64, dtype=torch.bfloat16),
        history={"events": torch.randn(events, 68, dtype=torch.bfloat16)},
        history_end=events,
        candidates=torch.randn(3, 64, dtype=torch.bfloat16),
        sources=torch.tensor([0, 1, -1]),
        targets=torch.tensor([1, 2, -1]),
        mask=torch.tensor([True, False, True]),
    )
    return root, node


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)

    def test_vtrace_terminal_and_truncated(self):
        values = torch.tensor([0.25, 0.5], dtype=torch.bfloat16, requires_grad=True)
        logp = torch.tensor([-0.5, -0.5], dtype=torch.bfloat16, requires_grad=True)
        rewards = torch.tensor([0, 1], dtype=torch.bfloat16)
        target, advantage = vtrace(
            rewards, torch.tensor([1, 0]), values, values.new_tensor(9), logp, logp
        )
        torch.testing.assert_close(target, values.new_tensor([1, 1]))
        self.assertFalse(target.requires_grad or advantage.requires_grad)
        target, _ = vtrace(
            rewards * 0, torch.tensor([1, 1]), values, values.new_tensor(0.75), logp, logp
        )
        torch.testing.assert_close(target, values.new_tensor([0.75, 0.75]))

    def test_same_seat_pending_and_real_terminal(self):
        _, node = example()
        traces = Trajectories(2)
        traces.selected(1, node, 0, -0.5, 8)
        self.assertEqual(len(traces.ready), 0)
        traces.successor(1, node)
        traces.selected(1, node, 2, -0.5, 9)
        traces.successor(1, node)
        self.assertIs(traces.ready[0]["bootstrap"], node)
        traces.selected(1, node, 0, -0.5, 9)
        traces.successor(1, reward=-1)
        terminal = traces.ready[1]
        self.assertIsNone(terminal["bootstrap"])
        self.assertEqual(terminal["steps"][-1]["discount"], 0)
        self.assertEqual(terminal["steps"][-1]["reward"], -1)
        traces.commit([0, 1])
        self.assertEqual(traces.acknowledged, 1)

    def test_mask_history_rebuild_and_encoder_gradient(self):
        model = Network(Config(width=32, layers=1, heads=4, ffn=64, history=16))
        root, node = example(events=5)
        encoded = model.encode(*root_batch([root], "cpu"))
        p, pm, e, lengths, c, s, t, mask = node_batch([node], "cpu")
        history = model.histories(e, lengths)
        first = model.histories(e[:, :2], torch.tensor([2]))
        rest = model.histories(e[:, 2:], torch.tensor([3]), first)
        torch.testing.assert_close(rest, history, atol=0.008, rtol=0.02)
        logits, value = model.decode(encoded, p, pm, history, c, s, t, mask)
        probs = logits.softmax(-1)
        self.assertEqual(float(probs[0, 1].detach()), 0)
        ratio = (
            logits.log_softmax(-1)[0, 0]
            - (probs[0, 0].double() / probs[0].double().sum()).log().bfloat16()
        ).exp()
        self.assertAlmostEqual(float(ratio.detach()), 1, delta=0.016)
        (-logits.log_softmax(-1)[0, 0] + value.square().mean()).backward()
        self.assertGreater(float(model.entities.weight.grad.abs().sum()), 0)
        self.assertTrue(
            all(
                p.dtype == torch.bfloat16 and p.grad.dtype == torch.bfloat16
                for p in model.parameters()
            )
        )

    def test_microbatch_has_same_sample_weight(self):
        model = Network(Config(width=32, layers=1, heads=4, ffn=64, history=16))
        root, node = example(events=2)
        steps = [
            dict(node=node, selected=0, logp=-0.7, reward=0.0, discount=1.0),
            dict(node=node, selected=2, logp=-0.7, reward=1.0, discount=0.0),
        ]
        blocks = [dict(id=0, steps=steps, bootstrap=None)]
        a, b = Learner(copy.deepcopy(model)), Learner(copy.deepcopy(model))
        loss_a = a.update(blocks, {node["root"]: root}, 1)
        loss_b = b.update(blocks, {node["root"]: root}, 8)
        self.assertAlmostEqual(loss_a, loss_b, delta=0.02)
        self.assertEqual(a.last_root_count, 1)
        self.assertEqual(a.last_decode_batches, [1, 1])
        for pa, pb in zip(a.model.parameters(), b.model.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, atol=0.012, rtol=0.04)

    def test_segmented_history_preserves_state_and_gradient(self):
        model = Network(Config(width=32, layers=1, heads=4, ffn=64, history=16))
        other = copy.deepcopy(model)
        _, long = example(events=37)
        short = {**long, "history_end": 3}
        empty = {**long, "history_end": 0}
        nodes = [long, short, empty]
        _, _, events, lengths, *_ = node_batch(nodes, "cpu")
        expected = model.histories(events, lengths)
        actual = other.node_histories(nodes, "cpu", Staging("cpu"), chunk_size=8)
        torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.02)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for left, right in zip(model.history.parameters(), other.history.parameters()):
            torch.testing.assert_close(left.grad, right.grad, atol=0.02, rtol=0.04)
        self.assertEqual(float(actual[2].detach().abs().sum()), 0)

    def test_nonfinite_does_not_commit(self):
        p = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.bfloat16))
        optimizer = AdamW([p])
        p.grad = torch.tensor([float("nan")], dtype=torch.bfloat16)
        with self.assertRaises(FloatingPointError):
            optimizer.step()
        self.assertEqual(float(p.detach()), 0.5)
        self.assertEqual(len(optimizer.state), 0)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False),
        "需要真实原生 BF16 CUDA",
    )
    def test_long_root_history_bounds_cuda_workspace(self):
        model = Network().cuda()
        model.flatten_history()
        _, node = example(events=8192)
        nodes = [{**node, "history_end": end} for end in (1, 64, 1024, 8192)]
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with torch.autograd.graph.save_on_cpu():
            history = model.node_histories(nodes, "cuda", Staging(torch.device("cuda")))
        history.square().sum().backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.history.parameters()))
        self.assertLess(torch.cuda.max_memory_allocated() - before, 256 * 1024**2)


if __name__ == "__main__":
    unittest.main()
