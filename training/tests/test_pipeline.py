"""异步协议验收：空闲 actor 不阻塞、根版本固定、共享学习与快照一致。"""

import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.haojie.runtime.config import Config
from training.haojie.runtime.gpu import Remote, serve
from training.haojie.runtime.ipc import Slot, pack, unpack
from training.haojie.runtime.trainer import Trainer
from training.haojie.runtime.storage import GameWriter
from training.tests.test_training import bundle


class PipelineTests(unittest.TestCase):
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
            writer = GameWriter(directory, {"id": 999, "opponent": "demo"})
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
            # 立即停止也要让空闲 actor、GPU 和所有共享槽正常关闭。
            (Path(directory) / "STOP").touch()
            restored.run()
            self.assertEqual(restored.stats["updates"], 2)

    def test_independent_requests_versions_and_learning(self):
        ctx = mp.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            config = Config(device="cpu", warmup_games=0, batch=2)
            trainer = Trainer(directory, config)
            parent, child = ctx.Pipe()
            pairs = [ctx.Pipe() for _ in range(2)]
            slot = Slot(size=256 * 1024)
            remote = Remote(parent, lambda: {0})
            payload = {
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": None,
                "stats": trainer.stats,
                "published": 0,
                "retained": [0],
            }
            process = ctx.Process(
                target=serve,
                args=(child, [p[0] for p in pairs], directory, config, payload),
            )
            process.start()
            child.close()
            for gpu, _ in pairs:
                gpu.close()
            try:
                self.assertTrue(parent.poll(30))
                self.assertEqual(remote.receive()["op"], "ready")

                def request(root, kind="encode"):
                    arrays = (
                        bundle()
                        if kind == "encode"
                        else (np.zeros((2, 704), np.float32),)
                    )
                    pairs[0][1].send(
                        {
                            "op": "request",
                            "lane": 0,
                            "slot": slot.name,
                            "shapes": pack(slot, arrays),
                            "model": "current",
                            "root": root,
                            "kind": kind,
                            "context": "state",
                        }
                    )
                    # 第二个 actor 不提交请求，也必须收到第一局的结果。
                    self.assertTrue(pairs[0][1].poll(10))
                    answer = pairs[0][1].recv()
                    self.assertEqual(answer["op"], "answer")
                    result = unpack(slot, answer["shapes"], copy=True)[0]
                    self.assertEqual(result.dtype, np.uint16)
                    return answer["version"]

                self.assertEqual(request("0:0"), 0)
                sample = (
                    bundle(),
                    [
                        (
                            np.zeros((2, 704), np.float32),
                            np.array([0.25, 0.75], np.float32),
                            1.0,
                        )
                    ],
                    1.0,
                    1.0,
                    np.zeros(4, np.float32),
                    True,
                    True,
                    False,
                )
                remote.learn([sample, sample])
                self.assertTrue(parent.poll(10))
                remote.receive()
                learned = remote.completed.popleft()
                self.assertEqual(learned["stats"]["trained_samples"], 2)
                self.assertEqual(learned["stats"]["value_samples"], 2)
                remote.publish(1)
                self.assertEqual(request("0:0", "decode"), 0)
                self.assertEqual(request("0:1"), 1)
                state = remote.call({"op": "snapshot"})["state"]
                self.assertEqual(state["stats"]["updates"], 1)
                self.assertTrue(
                    all(v.dtype == torch.bfloat16 for v in state["model"].values())
                )
                for values in state["optimizer"]["state"].values():
                    self.assertEqual(values["step"].dtype, torch.int64)
                    self.assertEqual(values["exp_avg"].dtype, torch.bfloat16)
                    self.assertEqual(values["exp_avg_sq"].dtype, torch.bfloat16)
                self.assertTrue((Path(directory) / "models/model-00000000.pt").exists())
            finally:
                if process.is_alive():
                    parent.send({"op": "stop"})
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join()
                for _, actor in pairs:
                    actor.close()
                remote.close()
                slot.close()
            self.assertEqual(process.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
