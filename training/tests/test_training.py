"""验收分层概率、图复用、原生 BF16、真实/软目标边界和阶段恢复。"""

import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch

from training.haojie.native import Environment
from training.haojie.model import Network, collate_states, collate_options
from training.haojie.opponents import Opponents
from training.haojie.runtime.storage import GameWriter, Replay, atomic_json
from training.haojie.runtime.config import Config
from training.haojie.runtime.trainer import Trainer
from training.haojie.runtime.targets import Calibration


def bundle(n=3):
    return (
        np.zeros((n, 128), np.float32),
        np.zeros((0, 128), np.float32),
        np.zeros(0, np.float32),
        np.zeros((n, n, 8), np.float32),
    )


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_public_rng_restore_and_compact_records(self):
        with Environment(713) as env:
            original = env.call("snapshot")
            info, before = env.encode()
            self.assertFalse(
                set(env.call("observe")) & {"rng", "seed", "log", "events"}
            )
            for seed in range(10, 20):
                env.call(
                    "sample-command",
                    id=0,
                    command={"type": "summon", "ultimate": False},
                    seed=seed,
                )
            self.assertEqual(original, env.call("snapshot"))
            first = env.call("apply", command={"type": "summon", "ultimate": False})
            env.call("restore", snapshot=original)
            _, again = env.encode()
            for a, b in zip(before, again):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(
                first, env.call("apply", command={"type": "summon", "ultimate": False})
            )
            snapshot = env.call("snapshot")
            snapshot["state"]["deaths"] = [
                {"id": f"dead-{i}", "kind": 1, "owner": 1, "ply": 1, "revived": False}
                for i in range(50)
            ]
            env.call("restore", snapshot=snapshot)
            _, compact = env.encode()
            self.assertEqual(len(compact[0]), 4)
            self.assertEqual(len(compact[1]), 50)

    def test_bf16_geometry_and_shared_encoder(self):
        net = Network()
        self.assertTrue(4_000_000 <= net.description()["parameters"] <= 10_000_000)
        s, r, o, g = bundle(3)
        s[1, 0] = 2
        s[1, 12] = 0.2
        s[2, 0] = 1
        s[2, 12] = 0.4
        r = np.zeros((2, 128), np.float32)
        r[:, 0] = 5
        r[:, 12] = [0.2, 0.3]
        o = np.array([1, 1], np.float32)
        g[0, 1, 0] = 0.5
        g[1, 2, 4] = 1
        b = (s, r, o, g)
        options = np.zeros((2, 704), np.float32)
        options[:, 1] = 1
        options[:, 2] = 2
        calls = []
        hook = net.blocks[0].register_forward_hook(lambda *args: calls.append(1))
        x, value, aux = net.encode(*collate_states([b], "cpu"))
        logits = net.decode(x, *collate_options([options], "cpu"))
        net.decode(x, *collate_options([options[:1]], "cpu"))
        self.assertEqual(len(calls), 1)
        hook.remove()
        loss = (
            -logits.log_softmax(-1)[0, 0] + value.square().mean() + aux.square().mean()
        )
        loss.backward()
        self.assertGreater(float(net.relations.weight.grad.abs().sum()), 0)
        self.assertGreater(float(net.records[0].weight.grad.abs().sum()), 0)
        self.assertTrue(
            all(
                p.dtype == torch.bfloat16
                and (
                    p.grad is None
                    or p.grad.dtype == torch.bfloat16
                    and torch.isfinite(p.grad).all()
                )
                for p in net.parameters()
            )
        )
        options[0, [1, 2, 42]] = [257, 299, 259]
        self.assertEqual(
            collate_options([options], "cpu")[2][0, 0].tolist(), [257, 299, 259]
        )

    def test_learning_bf16_and_optimizer_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(device="cpu", batch=2, warmup_games=0)
            trainer = Trainer(directory, config)
            state = bundle()
            options = np.zeros((2, 704), np.float32)
            target = np.array([0.3, 0.7], np.float32)
            request = SimpleNamespace(op="encode", context="state", arrays=state)
            bits = trainer.infer([(0, request, (0, 0))])[0]
            self.assertEqual(bits.dtype, np.uint16)
            self.assertEqual(bits.nbytes, 10)
            self.assertTrue(
                torch.isfinite(torch.from_numpy(bits).view(torch.bfloat16)).all()
            )
            self.assertEqual(
                len(
                    trainer.infer(
                        [
                            (
                                0,
                                SimpleNamespace(
                                    op="decode", context="state", arrays=(options,)
                                ),
                                (0, 0),
                            )
                        ]
                    )[0]
                ),
                2,
            )
            writer = GameWriter(directory, {"id": 0, "opponent": "self"})
            writer.sample(state, [(options, target, 1.0)], 1, 1, 0)
            writer.transition(
                {"ply": 1, "winner": None, "metrics": [0.0] * 4}, {"type": "end"}
            )
            writer.transition(
                {"ply": 3, "winner": 1, "metrics": [0.1] * 4}, {"type": "attack"}
            )
            trainer.replay.add(writer.flush())
            trainer.credit = 4
            before = trainer.model.value[2].weight.detach().clone()
            trainer.learn()
            trainer.learn()
            self.assertFalse(torch.equal(before, trainer.model.value[2].weight))
            trainer.pool.add(0, 1)
            trainer.publish()
            self.assertFalse(
                torch.equal(
                    trainer.inference_model(0).value[2].weight,
                    trainer.inference_model(2).value[2].weight,
                )
            )
            trainer.checkpoint({})
            restored = Trainer(directory, config, resume=True)
            restored.credit = 2
            restored.learn()
            for state in restored.optimizer.state.values():
                self.assertEqual(state["step"].dtype, torch.int64)
                self.assertIn(int(state["step"]), (0, 3))
                self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
                self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
            self.assertEqual(
                int(restored.optimizer.state[restored.model.value[2].weight]["step"]), 3
            )

    def test_unknown_soft_and_terminal_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = GameWriter(directory, {"id": 0, "opponent": "self"})
            decisions = [
                (np.zeros((2, 704), np.float32), np.array([0.3, 0.7], np.float32), 1.0)
            ]
            writer.sample(bundle(), decisions, 2, 1, 0)
            writer.transition(
                {"ply": 1, "winner": None, "metrics": [0.0] * 4}, {"type": "end"}
            )
            replay = Replay()
            replay.add(writer.flush())
            rng = np.random.default_rng(1)
            self.assertTrue(all(b[3] == 0 for b in replay.batch(8, rng)))
            writer.metadata["soft_target"] = {
                "actor": 1,
                "value": 0.6,
                "weight": 0.25,
                "kind": "bootstrap",
            }
            writer.flush()
            soft = replay.batch(8, rng)
            self.assertTrue(
                all(b[2] == -0.6 and b[3] == 0.25 and not b[6] and b[7] for b in soft)
            )
            self.assertIsNone(writer.metadata["winner"])
            writer.transition(
                {"ply": 3, "winner": 1, "metrics": [0.1] * 4}, {"type": "attack"}
            )
            writer.flush()
            self.assertTrue(
                all(
                    b[2] == -1 and b[3] == 1 and b[6] and not b[7]
                    for b in replay.batch(8, rng)
                )
            )

    def test_calibration_heldout_and_native_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration = Calibration()
            self.assertIsNone(calibration.predict([1, 0, 0, 0], 1))
            path = Path(directory) / "game.json"
            for i in range(10):
                sign = 1 if i % 2 else -1
                atomic_json(
                    path,
                    {
                        "winner": 1 if sign > 0 else 2,
                        "game": {"id": i, "opponent": "self"},
                        "timeline": [[1, sign, 0.2 * sign, sign, 0.1 * sign]],
                    },
                )
                calibration.observe(path, 5)
            self.assertIsNotNone(calibration.weights)
            self.assertEqual(calibration.weights.dtype, torch.bfloat16)
            self.assertEqual(calibration.validation["heldout_games"], 2)
            self.assertGreater(calibration.predict([1, 0.2, 1, 0.1], 1), 0)
            self.assertAlmostEqual(
                calibration.predict([1, 0.2, 1, 0.1], 2),
                -calibration.predict([1, 0.2, 1, 0.1], 1),
            )
            trainer = Trainer(
                Path(directory) / "run", Config(device="cpu", warmup_games=0)
            )
            trainer.stats["selfplay_terminal"] = trainer.config.bootstrap_after_games
            writer = GameWriter(
                trainer.directory, {"id": 0, "seed": 91, "opponent": "self"}
            )
            writer.metadata["status"] = "truncated"
            writer.flush()
            with Environment(91) as env:
                atomic_json(writer.path / "environment.json", env.call("snapshot"))
                info = env.call("info")
            event = {
                "game": writer.game,
                "path": str(writer.path / "game.json"),
                "actor": info["actor"],
                "metrics": info["metrics"],
            }
            trainer.boundary_target(event)
            meta = json.loads((writer.path / "game.json").read_text())
            self.assertIsNone(meta["winner"])
            self.assertEqual(meta["soft_target"]["kind"], "bootstrap")
            self.assertLess(abs(meta["soft_target"]["value"]), 1)

    def test_pretraining_barrier_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(device="cpu", warmup_games=1, warmup_updates=2, batch=2)
            trainer = Trainer(directory, config)
            game = trainer.game()
            self.assertEqual(game["opponent"], "demo")
            self.assertEqual(game["ply_limit"], 0)
            self.assertIsNone(trainer.game())
            writer = GameWriter(directory, game)
            writer.sample(
                bundle(),
                [
                    (
                        np.zeros((2, 704), np.float32),
                        np.array([1.0, 0.0], np.float32),
                        1.0,
                    )
                ],
                1,
                1,
                -1,
            )
            writer.transition(
                {"ply": 1, "winner": 1, "metrics": [0.0] * 4}, {"type": "end"}
            )
            trainer.event({"type": "data", "path": writer.flush()})
            trainer.learn()
            self.assertEqual(trainer.stats["updates"], 0)
            trainer.event(
                {
                    "type": "done",
                    "game": game,
                    "winner": 1,
                    "path": str(writer.path / "game.json"),
                }
            )
            self.assertEqual(trainer.phase, "imitation")
            self.assertIsNone(trainer.game())
            trainer.learn()
            trainer.checkpoint({})
            restored = Trainer(directory, config, resume=True)
            self.assertEqual(restored.imitation_updates, 1)
            self.assertIsNone(restored.game())
            restored.learn()
            self.assertEqual(restored.phase, "self")
            self.assertEqual(restored.imitation_updates, 2)
            self.assertNotEqual(restored.game()["opponent"], "demo")

    def test_ply_limit_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                directory, Config(device="cpu", warmup_games=0, full_game_fraction=0)
            )
            self.assertEqual(trainer.game()["ply_limit"], 50)
            trainer.config.full_game_fraction = 1
            self.assertEqual(trainer.game()["ply_limit"], 0)

    def test_old_models_remain_reachable(self):
        pool = Opponents()
        for i in range(1, 100):
            pool.add(i, i * 64)
        rng = np.random.default_rng(7)
        choices = {pool.choose(rng) for _ in range(1000)}
        self.assertIn(1, choices)
        self.assertTrue(choices & set(range(92, 100)))


if __name__ == "__main__":
    unittest.main()
