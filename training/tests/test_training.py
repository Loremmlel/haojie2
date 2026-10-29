"""覆盖搜索视角、公开随机边界、真实标签与恢复，而非仅检查参数更新。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.haojie.native import Environment
from training.haojie.model import Network, collate
from training.haojie.search import search
from training.haojie.opponents import Teacher, Opponents
from training.haojie.runtime.storage import GameWriter, Replay
from training.haojie.runtime.config import Config
from training.haojie.runtime.worker import Lane


class SearchEnvironment:
    def prepare(self, position, *_):
        return {"id": position, "actor": 1, "ply": 1}, np.zeros((3, 128), np.float32), np.zeros((2, 576), np.float32)

    def call(self, op, **args):
        # 子节点仍由同一玩家操作；错误的逐层翻号会选择输棋动作。
        if args["id"] == 0:
            return {"id": args["action"] + 1, "actor": 1, "winner": None, "key": str(args["action"])}
        return {"actor": 2, "winner": 1 if args["id"] == 1 else 2}


class TrainingTests(unittest.TestCase):
    def test_search_same_player_and_terminal_perspective(self):
        generator = search(SearchEnvironment(), np.random.default_rng(3), simulations=60, candidates=2, depth=4, noise=False)
        answer = None
        while True:
            try:
                request = next(generator) if answer is None else generator.send(answer)
                answer = (np.zeros(len(request[1]), np.float32), 0., np.zeros(4), 17)
            except StopIteration as result:
                root, policy, version = result.value
                self.assertEqual(version, 17)
                self.assertGreater(policy[0], .8)
                self.assertGreater(root.total[0], 0)
                break

    def test_public_rng_and_restore(self):
        with Environment(713) as env:
            original = env.call("snapshot")
            info, states, actions = env.prepare(0, 5, 8)
            self.assertFalse(set(env.call("observe")) & {"rng", "seed", "log", "events"})
            for seed in range(10, 20):
                env.call("sample", id=0, action=0, seed=seed)
            self.assertEqual(original, env.call("snapshot"))
            first = env.call("step", action=0)
            env.call("restore", snapshot=original)
            _, again_states, again_actions = env.prepare(0, 5, 8)
            np.testing.assert_array_equal(states, again_states)
            np.testing.assert_array_equal(actions, again_actions)
            self.assertEqual(first, env.call("step", action=0))

    def test_padding_and_finite_training(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        net = Network()
        self.assertEqual(net.description()["parameters"], 5338950)
        with Environment(99) as env:
            _, s, a = env.prepare(0, 11, 8)
            env.call("step", action=0)
            _, s2, a2 = env.prepare(0, 12, 8)
        one = net(*collate([(s, a)], "cpu"))
        padded = net(*collate([(s, a), (s2, a2)], "cpu"))
        torch.testing.assert_close(one[0][0], padded[0][0, :len(a)], atol=.004, rtol=.016)
        torch.testing.assert_close(one[1][0], padded[1][0], atol=.004, rtol=.016)
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in padded))
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in net.parameters()))
        loss = padded[1].square().mean() + padded[2].square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in net.parameters()))
        self.assertTrue(all(p.grad is None or p.grad.dtype == torch.bfloat16 for p in net.parameters()))
        many = np.zeros((300,128), np.float32)
        referenced = np.zeros((1,576), np.float32)
        referenced[0,1:3] = [257,299]
        tensors = collate([(many,referenced)], "cpu")
        self.assertEqual(tensors[5][0,0].tolist(), [257,299])
        self.assertTrue(all(torch.isfinite(t).all() for t in net(*tensors)))

    def test_bf16_optimizer_and_ipc(self):
        from training.haojie.runtime.trainer import Trainer
        from training.haojie.runtime.ipc import Slot, pack, unpack, search_values
        with tempfile.TemporaryDirectory() as directory, Environment(99) as env:
            trainer = Trainer(directory, Config(device="cpu", batch=2))
            _, states, actions = env.prepare(0, 11, 8)
            with torch.no_grad():
                expected = trainer.model(*collate([(states,actions)], "cpu"))
            output = trainer.infer([(0,states,actions)])[0]
            frozen_value = trainer.inference_model(0).value[2].weight.detach().clone()
            slot = Slot(size=4096)
            try:
                result = unpack(slot,pack(slot,[output]),copy=True)[0]
                self.assertEqual(result.dtype, np.uint16)
                self.assertEqual(result.nbytes, (len(actions)+5)*2)
                np.testing.assert_array_equal(search_values(result)[:len(actions)], expected[0][0].tolist())
            finally: slot.close()
            writer = GameWriter(directory, {"id":0,"opponent":"self"})
            policy = np.full(len(actions),1/len(actions),np.float32)
            writer.sample(states,actions,policy,1,1,0)
            writer.transition({"ply":1,"winner":None,"metrics":[0.]*4},{"type":"end"})
            writer.transition({"ply":3,"winner":1,"metrics":[.1]*4},{"type":"attack"})
            trainer.replay.add(writer.flush())
            trainer.credit = 4
            before = trainer.model.value[2].weight.detach().clone()
            trainer.learn()
            trainer.learn()
            self.assertFalse(torch.equal(before,trainer.model.value[2].weight))
            for state in trainer.optimizer.state.values():
                self.assertEqual(state["step"].dtype,torch.int64)
                self.assertEqual(int(state["step"]),2)
                self.assertEqual(state["exp_avg"].dtype,torch.bfloat16)
                self.assertEqual(state["exp_avg_sq"].dtype,torch.bfloat16)
            trainer.pool.add(0,1)
            trainer.publish()
            trainer.config.warmup_games = 0
            trainer.config.teacher_fraction = 0
            trainer.config.history_fraction = 1
            self.assertEqual(trainer.game()["opponent"],"history")
            self.assertTrue(torch.equal(frozen_value,trainer.inference_model(0).value[2].weight))
            self.assertFalse(torch.equal(frozen_value,trainer.inference_model(2).value[2].weight))
            self.assertEqual(len(trainer.infer([(0,states,actions),(2,states,actions)])),2)
            trainer.checkpoint({})
            restored = Trainer(directory,Config(device="cpu",batch=2),resume=True)
            self.assertTrue(torch.equal(trainer.model.value[2].weight,restored.model.value[2].weight))
            restored.credit = 2
            restored.learn()
            self.assertEqual(restored.stats["updates"],3)
            for state in restored.optimizer.state.values():
                self.assertEqual(state["step"].dtype,torch.int64)
                self.assertEqual(int(state["step"]),3)
                self.assertEqual(state["exp_avg"].dtype,torch.bfloat16)
                self.assertEqual(state["exp_avg_sq"].dtype,torch.bfloat16)

    def test_unknown_labels_and_terminal_link(self):
        with tempfile.TemporaryDirectory() as directory:
            game = {"id": 0, "opponent": "self"}
            writer = GameWriter(directory, game)
            writer.sample(np.zeros((3,128),np.float32), np.zeros((2,576),np.float32), np.array([.3,.7],np.float32), 1, 1, 0)
            writer.transition({"ply": 1, "winner": None, "metrics": [0.,0.,0.,0.]}, {"type":"end"})
            replay = Replay()
            replay.add(writer.flush())
            self.assertFalse(any(b[4] for b in replay.batch(8, np.random.default_rng(1))))
            writer.transition({"ply": 3, "winner": 1, "metrics": [1.,.1,.2,0.]}, {"type":"attack"})
            writer.flush()
            batch = replay.batch(8, np.random.default_rng(1))
            self.assertTrue(all(b[4] and b[3] == 1. and b[6] for b in batch))
            writer.metadata["game"]["opponent"] = "teacher"
            writer.flush()
            self.assertFalse(any(b[4] for b in replay.batch(8, np.random.default_rng(1))))

    def test_teacher_and_lane_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(candidates=8, chunk_commands=8)
            game = {"id": 0, "seed": 91, "learner": 1, "opponent": "demo", "difficulty": "easy"}
            original = Lane(directory, config, game)
            try:
                for _ in range(12): original.advance()
                snapshot = copy.deepcopy(original.snapshot())
                original.advance()
                expected = original.environment.call("snapshot")
                restored = Lane(directory, config, game, snapshot)
                try:
                    restored.advance()
                    self.assertEqual(expected, restored.environment.call("snapshot"))
                finally: restored.close()
            finally: original.close()

    def test_teacher_opponent_resume_during_neural_search(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(candidates=8,simulations=2,fast_simulations=2)
            game = {"id":0,"seed":91,"learner":2,"opponent":"teacher","difficulty":"easy"}
            original = Lane(directory,config,game)
            restored = None
            try:
                for _ in range(100):
                    request,_ = original.advance()
                    if request is not None: break
                else: self.fail("没有进入神经网络搜索")
                snapshot = copy.deepcopy(original.snapshot())
                self.assertIsNotNone(snapshot["teacher"])
                restored = Lane(directory,config,game,snapshot)
                for lane in (original,restored):
                    before = lane.writer.metadata["steps"]
                    request = lane.request
                    while lane.writer.metadata["steps"] == before:
                        answer = (np.zeros(len(request[1]),np.float32),0.,np.zeros(4),0) if request is not None else None
                        request,_ = lane.advance(answer)
                self.assertEqual(original.environment.call("snapshot"),restored.environment.call("snapshot"))
                self.assertEqual(original.teacher.snapshot(),restored.teacher.snapshot())
            finally:
                original.close()
                if restored: restored.close()

    def test_old_models_remain_reachable(self):
        pool = Opponents()
        for i in range(1, 100): pool.add(i, i * 64)
        rng = np.random.default_rng(7)
        choices = {pool.choose(rng) for _ in range(1000)}
        self.assertIn(1, choices)
        self.assertTrue(choices & set(range(92, 100)))


if __name__ == "__main__": unittest.main()
