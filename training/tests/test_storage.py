"""内存回放、历史权重和 v3 恢复点的完整生命周期验收。"""

import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from training.haojie.runtime.config import Config
from training.haojie.runtime.gpu import Executor
from training.haojie.runtime.storage import GameBuffer, Replay
from training.haojie.runtime.trainer import Trainer
from training.tests.bench_pipeline import equal_tree
from training.tests.test_training import bundle


def sample(writer, step=1, winner=None):
    writer.sample(
        bundle(),
        [(np.zeros((2, 704), np.float32), np.array([0.25, 0.75], np.float32), 1.0)],
        1,
        step,
        0,
    )
    writer.transition(
        {"ply": step, "winner": winner, "metrics": [step / 10] * 4}, {"type": "end"}
    )
    return writer.flush()


class StorageTests(unittest.TestCase):
    def test_memory_sampling_and_releases_evicted_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            replay = Replay(capacity=2)
            for number in range(2):
                game = {"id": number, "opponent": "self"}
                writer = GameBuffer(game)
                for step in (1, 3):
                    replay.add(sample(writer, step, 1 if step == 3 else None))
            rng = np.random.default_rng(82)
            with patch(
                "numpy.load", side_effect=AssertionError("内存采样不读文件")
            ), patch.object(
                Path, "read_text", side_effect=AssertionError("元数据不读文件")
            ):
                actual = replay.batch(32, rng)
            self.assertEqual(len(actual), 32)
            self.assertTrue(all(row[2] == 1 and row[3] == 1 for row in actual))
            self.assertEqual(list(Path(directory).iterdir()), [])
            del actual
            first = replay.games[0]
            array = weakref.ref(first["chunks"][0]["states"])
            replay.add(sample(GameBuffer({"id": 2, "opponent": "self"})))
            gc.collect()
            self.assertIsNone(array())
            self.assertEqual(first["chunks"], [])
            limit = replay.games[-1]["bytes"]
            bounded = Replay(games=replay.games, max_bytes=limit)
            self.assertEqual(bounded.bytes, limit)
            self.assertEqual(len(bounded.games), 1)
            with self.assertRaisesRegex(ValueError, "单个回放片段"):
                Replay(games=bounded.games, max_bytes=1)

    def test_only_history_files_and_frozen_root_survive_cache_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(directory, Config(device="cpu", warmup_games=0))
            executor = Executor(
                directory,
                trainer.config,
                {
                    "model": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": None,
                    "stats": trainer.stats,
                    "published": 0,
                    "history": [],
                    "weights": trainer.weights.states,
                },
            )
            self.assertEqual(list(trainer.models.iterdir()), [])
            original = executor.inference_model(0).value[2].weight.detach().clone()
            executor.resolve((0, 0), {"root": (1, 0), "model": "current"})
            for version in range(1, 5):
                with torch.no_grad():
                    executor.model.value[2].weight.add_(0.125)
                executor.publish(version)
                executor.inference_model(version)
            self.assertEqual(set(executor.weights.states), {0, 4})
            torch.testing.assert_close(
                executor.inference_model(0).value[2].weight, original, rtol=0, atol=0
            )
            executor.weights.archive(0)
            self.assertEqual(
                [p.name for p in trainer.models.iterdir()], ["model-00000000.pt"]
            )
            executor.release((0, 0))
            executor.weights.cache_bytes = 1
            executor.prune_models()
            self.assertNotIn(0, executor.weights.states)
            executor.inference_models.clear()
            torch.testing.assert_close(
                executor.inference_model(0).value[2].weight, original, rtol=0, atol=0
            )
            self.assertEqual(set(executor.snapshot()["weights"]), {4})
            self.assertTrue(
                all(v.dtype == torch.bfloat16 for v in executor.weights.get(4).values())
            )

    def test_v3_atomic_save_cleanup_and_reject_other_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(device="cpu", warmup_games=0, workers=1, games_per_worker=1)
            trainer = Trainer(directory, config)
            writer = GameBuffer({"id": 0, "opponent": "self"})
            trainer.event({"type": "data", "data": sample(writer, 3, 1)})
            trainer.pool.add(0, 64)
            trainer.archive(0)
            trainer.checkpoint({})
            torch.save(trainer.model.state_dict(), trainer.models / "model-00000099.pt")
            marker = Path(directory) / "user-note.txt"
            marker.write_text("保留用户文件", encoding="utf-8")
            restored = Trainer(directory, config, resume=True)
            restored.credit += 1
            expected = restored.replay.batch(8, np.random.default_rng(23))
            with patch(
                "training.haojie.runtime.trainer.torch.save",
                side_effect=OSError("模拟保存失败"),
            ):
                with self.assertRaises(OSError):
                    restored.checkpoint({})
            self.assertEqual(
                torch.load(Path(directory) / "checkpoint.pt", weights_only=False)[
                    "credit"
                ],
                trainer.credit,
            )
            restored.checkpoint({})
            self.assertEqual(
                sorted(p.name for p in trainer.models.iterdir()), ["model-00000000.pt"]
            )
            self.assertEqual(marker.read_text(encoding="utf-8"), "保留用户文件")
            resumed = Trainer(directory, config, resume=True)
            equal_tree(resumed.replay.batch(8, np.random.default_rng(23)), expected)
            self.assertEqual(resumed.credit, restored.credit)
            self.assertEqual(
                resumed.rng.bit_generator.state, restored.rng.bit_generator.state
            )
            torch.save({"format": "unsupported"}, Path(directory) / "checkpoint.pt")
            with self.assertRaisesRegex(ValueError, "只接受"):
                Trainer(directory, config, resume=True)

    def test_shared_active_buffer_resume_does_not_duplicate_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(device="cpu", warmup_games=0, workers=1, games_per_worker=1)
            trainer = Trainer(directory, config)
            writer = GameBuffer(trainer.game())
            trainer.event({"type": "data", "data": sample(writer)})
            workers = {
                0: [
                    {
                        "lane": 0,
                        "game": writer.game,
                        "root_version": 0,
                        "buffer": writer.data,
                    }
                ]
            }
            trainer.checkpoint(workers)
            restored = Trainer(directory, config, resume=True)
            data = restored.saved_workers[0][0]["buffer"]
            self.assertIs(data, restored.replay.games[0])
            self.assertEqual(restored.credit, 4)
            restored.event({"type": "data", "data": data})
            self.assertEqual(restored.credit, 4)
            self.assertEqual(
                {p.name for p in Path(directory).iterdir()}, {"models", "checkpoint.pt"}
            )


if __name__ == "__main__":
    unittest.main()
