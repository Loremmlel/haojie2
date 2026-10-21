"""采样窗口保留长局：真实调度器跨根暂停、恢复、取消与任务身份回归。"""

import threading
import unittest

from . import test_async


class PoolYieldTests(unittest.TestCase):
    def setUp(self):
        # 复用已有设备模拟；不重复执行它的测试，只替换原生管道中的对局消息。
        self.f = test_async.PoolAsyncTests()
        self.f.setUp()
        for task in self.f.pool.slots.values():
            task["startup_model"] = "a"

    def request(self, request_id, command, actor=1):
        message = self.f.request(1)
        message.update(id=request_id, commandIndex=command, actor=actor)
        return message

    def test_long_game_resumes_after_old_root_finishes_with_new_weights(self):
        f = self.f
        remaining = [self.request(3, 0), self.request(4, 1), self.request(5, 1, 2)]

        def respond(message):
            f.sent.append((1, 0, message))
            f.add(1, remaining.pop(0) if remaining else {"type": "done", "model": "a"})

        f.pool.clients[1].send = respond
        first = f.request(0)
        first.update(commandIndex=0, actor=1)
        f.add(0, first)
        f.add(1, self.request(2, 0))
        f.pool.run(f.jobs, keep_open=True, yield_when=lambda: bool(f.committed))
        self.assertEqual(f.committed, [10])
        self.assertFalse(f.unfinished or f.runner.active)
        self.assertEqual([row[2]["id"] for row in f.sent], [1, 2, 3])
        self.assertEqual(f.pool.slots[1]["last_root"], (0, 1))
        self.assertEqual(f.pool.parked[1][2]["id"], 4)

        # 模拟训练/评测停顿超过原生超时，恢复后不将停顿计入响应延迟。
        f.pool.paused_at -= 1000
        f.pool.slots[1]["last_activity"] -= 1000
        f.pool.parked[1][2]["received_at"] -= 1000
        f.pool.model_hash, f.runner.offset = "new-round", 1.0
        f.pool.run(f.jobs, keep_open=True, yield_when=lambda: len(f.committed) == 2)
        self.assertEqual(f.committed, [10, 11])
        self.assertFalse(f.unfinished or f.runner.active or f.pool.slots or f.pool.parked)
        self.assertEqual(f.pool.yield_count, 2)
        self.assertLess(max(f.pool.responses), 10)
        old, new = f.sent[2][2], f.sent[3][2]
        self.assertGreater(new["logits"][0], old["logits"][0])
        self.assertEqual([row[2]["id"] for row in f.sent], [1, 2, 3, 4, 5])

    def test_same_batch_done_parks_new_actor_root_before_inference(self):
        f = self.f
        f.pool.slots[1]["last_root"] = (0, 1)
        # 同一命令序号的回合外 Pass 与己方决策属于不同根。
        f.add(1, self.request(2, 0, 2))
        f.add(0, {"type": "done", "model": "a"})
        f.pool.run(f.jobs, keep_open=True, yield_when=lambda: bool(f.committed))
        self.assertEqual(f.committed, [10])
        self.assertFalse(f.sent or f.unfinished or f.runner.active)
        self.assertEqual(f.pool.parked[1][2]["actor"], 2)

        event = threading.Event()
        event.set()
        f.pool.run(f.jobs, keep_open=True, cancel_event=event, yield_when=lambda: False)
        self.assertEqual(f.unfinished, [11])
        self.assertFalse(f.pool.slots or f.pool.parked or f.sent)
        self.assertIn(("pool-close",), f.log)

    def test_yield_requires_resident_pool_and_preserves_first_unserved_root(self):
        f = self.f
        with self.assertRaisesRegex(ValueError, "keep_open"):
            f.pool.run(f.jobs, yield_when=lambda: True)
        f.add(0, self.request(1, 0))
        f.add(1, self.request(2, 0))
        f.pool.run(f.jobs, keep_open=True, yield_when=lambda: True)
        self.assertFalse(f.sent or f.committed or f.unfinished)
        self.assertEqual(set(f.pool.parked), {0, 1})
        self.assertTrue(all("last_root" not in task for task in f.pool.slots.values()))


if __name__ == "__main__":
    unittest.main()
