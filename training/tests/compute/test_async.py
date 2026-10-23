"""双缓冲的所有权、实际 Pool 排空边界与 CUDA 同结果回归。"""

import os
import queue
import threading
import time
import unittest
from collections import deque
from concurrent.futures import Future
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import torch

from haojie_training.console.sampling import MemoryJobs
from haojie_training.data import INPUT_KEYS, synthetic_batch
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.native.execution import (
    ForwardWorkers,
    InferenceStats,
    InferenceTicket,
    PolicyInference,
)
from haojie_training.native.pool import Pool, _group_size
from haojie_training.runtime import cpu_state


class DeviceQueue(queue.Queue):
    """正时长的空队列等待推进模拟设备；不使用宿主 sleep 或超时退出测试。"""

    def __init__(self, advance):
        super().__init__()
        self.advance = advance

    def get(self, block=True, timeout=None):
        if timeout is not None and timeout > 0 and self.empty():
            self.advance()
            return super().get(block=False)
        return super().get(block, timeout)


class FakeInference:
    """只替代设备计算；请求路由、回包、MC 变换和 Pool.run 均使用生产实现。"""

    supports_async = supports_packed_batches = True

    def __init__(self, log, offset=0):
        self.log, self.offset, self.active = log, offset, []
        self.on_submit = lambda: None

    def __call__(self, rows):
        return torch.tensor(
            [[row["value"] + self.offset, 0.0] for row in rows], dtype=torch.bfloat16
        )

    def submit(self, rows, *, slots):
        if len(self.active) == 2 or any(
            set(slots).intersection(ticket.slots) for ticket in self.active
        ):
            raise AssertionError("超过双组上限或重复槽位")
        completed = threading.Event()
        ticket = InferenceTicket(
            len(self.active),
            tuple(slots),
            SimpleNamespace(query=completed.is_set, synchronize=completed.set),
            self(rows),
            (rows,),
        )
        self.active.append(ticket)
        self.log.append(("submit", tuple(slots)))
        self.on_submit()
        return ticket

    def finish(self, ticket):
        ticket.event.synchronize()
        self.active.remove(ticket)
        ticket.owners = ()
        self.log.append(("finish", ticket.slots))
        return ticket.logits

    def close(self):
        for ticket in self.active:
            ticket.event.synchronize()
            ticket.owners = ()
        self.active.clear()
        self.log.append(("drain",))


class GroupingTests(unittest.TestCase):
    def test_packed_group_preserves_capacity_and_request_order(self):
        original = [(i, i, {"entities": n}) for i, n in enumerate((1, 2, 125, 125, 4000))]
        for maximum in (1, 2, 32):
            with self.subTest(maximum=maximum):
                pending = list(original)
                actual = []
                while pending:
                    count = _group_size(pending, maximum, packed=True)
                    self.assertEqual(count, min(len(pending), maximum))
                    actual.extend(pending[:count])
                    pending = pending[count:]
                self.assertTrue(all(a is b for a, b in zip(actual, original, strict=True)))

    def test_length_boundary_capacity_and_order(self):
        for lengths, maximum, expected in (
            ([], 32, []),
            ([1, 1, 2, 3], 32, [[1, 1], [2], [3]]),
            ([100, 124, 125, 126, 157, 158], 32, [[100, 124, 125], [126, 157], [158]]),
            ([4097], 32, [[4097]]),
            ([100] * 5, 2, [[100, 100], [100, 100], [100]]),
            ([100, 125, 126], 1, [[100], [125], [126]]),
        ):
            with self.subTest(lengths=lengths, maximum=maximum):
                original = [(i, i, {"entities": n}) for i, n in enumerate(lengths)]
                pending = list(original)
                groups = []
                while pending:
                    count = _group_size(pending, maximum)
                    self.assertGreater(count, 0)
                    self.assertLessEqual(count, maximum)
                    groups.append(pending[:count])
                    pending = pending[count:]
                self.assertEqual(
                    [[item[2]["entities"] for item in group] for group in groups], expected
                )
                # 相同长度的请求也按原身份核对，不能以长度相同掩盖重复、漏项或换序。
                actual = [item for group in groups for item in group]
                self.assertEqual(actual, original)
                self.assertTrue(all(a is b for a, b in zip(actual, original, strict=True)))


class PoolAsyncTests(unittest.TestCase):
    def setUp(self):
        self.log, self.sent, self.committed, self.unfinished, self.errors = [], [], [], [], []
        self.runner = FakeInference(self.log)
        self.pool = Pool.__new__(Pool)
        self.pool.device = torch.device("cpu")
        self.pool.input_processes = None
        self.pool.started = time.perf_counter()
        self.pool.active_seconds = self.pool.refills = self.pool.refill_seconds = 0
        self.pool.finished = self.pool.queue_seconds = self.pool.peak_queue = 0
        self.pool.requests, self.pool.last_ids = 0, [0, 0]
        self.pool.stop_at = self.pool.first_done = None
        self.pool.model_hash, self.pool.wait, self.pool.max_batch_size = "a", 0, 1
        self.pool.send_seconds, self.pool.responses = 0.0, deque()
        self.pool.restarts = self.pool.read_seconds = self.pool.decode_seconds = 0
        self.pool.tensor_bytes, self.pool.ready = 0, {}
        self.pool.generations = [0, 0]
        self.pool.events = DeviceQueue(self.complete_inferences)
        self.pool.slots = {i: {"id": 10 + i, "models": ["a", "b"], "requests": 0} for i in range(2)}
        self.pool.metrics = lambda: {"finished": self.pool.finished}
        self.pool.close = lambda: self.log.append(("pool-close",))
        self.pool.inference = self.runner
        self.pool.clients = [self.client(i) for i in range(2)]
        self.pool.new_client = self.client
        self.jobs = MemoryJobs.__new__(MemoryJobs)
        self.jobs.c = SimpleNamespace(config={"method": "decomposed-mc-q-v2"})
        self.jobs.temperature = 0.75
        self.jobs.inferences = {"a": self.runner, "b": FakeInference(self.log, 0.25)}
        self.jobs.stop, self.jobs.claim = lambda: False, lambda: None
        self.jobs.finish = lambda task, message: self.committed.append(task["id"])
        self.jobs.unfinished = lambda task: self.unfinished.append(task["id"])
        self.jobs.resource_error = self.errors.append

    def complete_inferences(self):
        tickets = [
            ticket for inference in self.jobs.inferences.values() for ticket in inference.active
        ]
        self.assertTrue(tickets, "假设备没有在途计算，无法推进空队列等待")
        for ticket in tickets:
            ticket.event.synchronize()

    def client(self, slot):
        generation = self.pool.generations[slot]

        def send(response):
            self.sent.append((slot, generation, response))
            self.pool.events.put(((slot, generation), {"type": "done", "model": "a"}))

        return SimpleNamespace(
            send=send,
            checked=lambda raw: raw,
            release=lambda message: None,
            close=lambda: None,
            ready={},
            read_seconds=0,
            decode_seconds=0,
            tensor_bytes=0,
        )

    def request(self, slot, model="a"):
        return {
            "type": "infer",
            "id": slot + 1,
            "model": model,
            "entities": 10,
            "candidates": 1,
            "input": {"value": 0.1 * (slot + 1)},
            "received_at": time.perf_counter(),
        }

    def add(self, slot, message):
        self.pool.events.put(((slot, self.pool.generations[slot]), message))

    def test_two_groups_and_shared_mc_transform(self):
        for slot in range(2):
            self.add(slot, self.request(slot))
        expected = self.jobs.infer([self.request(0), self.request(1)])
        self.pool.run(self.jobs, keep_open=True)
        self.assertEqual(self.log[:2], [("submit", ((0, 0),)), ("submit", ((1, 0),))])
        self.assertEqual(self.committed, [10, 11])
        self.assertFalse(self.unfinished)
        for row, (slot, generation, response) in enumerate(self.sent):
            self.assertEqual((slot, generation), (row, 0))
            self.assertEqual(
                response["logits"].tolist(), expected[row, :1].view(torch.uint16).tolist()
            )
        self.assertFalse(self.runner.active)

    def test_request_arriving_during_gpu_wait_submits_before_finish(self):
        waits = []
        finish = self.runner.finish

        def advance():
            waits.append(True)
            if len(waits) == 1:
                self.assertFalse(self.runner.active[0].ready())
                self.add(1, self.request(1))
            else:
                self.complete_inferences()

        def ready_finish(ticket):
            self.assertTrue(ticket.ready(), "晚到帧处理前阻塞回收了未完成票据")
            return finish(ticket)

        self.pool.events.advance, self.runner.finish = advance, ready_finish
        self.add(0, self.request(0))
        self.pool.run(self.jobs, keep_open=True)
        self.assertEqual([row[0] for row in self.log[:2]], ["submit", "submit"])
        self.assertEqual(self.committed, [10, 11])
        self.assertFalse(self.unfinished or self.runner.active)

    def test_empty_queue_polls_until_device_completes(self):
        self.pool.clients = self.pool.clients[:1]
        self.pool.slots = {0: self.pool.slots[0]}
        waits = []
        finish = self.runner.finish

        def advance():
            waits.append(True)
            if len(waits) == 2:
                self.complete_inferences()
            self.assertLessEqual(len(waits), 2, "完成事件未使采样循环继续")

        def ready_finish(ticket):
            self.assertEqual(len(waits), 2)
            self.assertTrue(ticket.ready())
            return finish(ticket)

        self.pool.events.advance, self.runner.finish = advance, ready_finish
        self.add(0, self.request(0))
        self.pool.run(self.jobs, keep_open=True)
        self.assertEqual(self.committed, [10])
        self.assertFalse(self.unfinished or self.runner.active)

    def test_completed_second_stream_returns_before_slow_first_stream(self):
        def advance():
            self.assertTrue(self.runner.active)
            self.runner.active[-1].event.synchronize()

        self.pool.events.advance = advance
        for slot in range(2):
            self.add(slot, self.request(slot))
        self.pool.run(self.jobs, keep_open=True)
        self.assertEqual(self.committed, [11, 10])
        self.assertEqual([slot for slot, _, _ in self.sent], [1, 0])
        self.assertFalse(self.unfinished or self.runner.active)

    def test_background_completion_wakes_idle_queue_without_millisecond_polling(self):
        self.pool.background_launch = True
        submit, get = self.runner.submit, self.pool.events.get
        waits = []

        def queued(*args, **kwargs):
            ticket = submit(*args, **kwargs)
            ticket.launch = Future()
            return ticket

        def advance():
            ticket = self.runner.active[-1]
            ticket.event.synchronize()
            ticket.launch.set_result(None)

        def receive(*args, **kwargs):
            if self.pool.events.empty() and kwargs.get("timeout", 0) > 0:
                waits.append(kwargs["timeout"])
            return get(*args, **kwargs)

        self.runner.submit = queued
        self.pool.events.advance, self.pool.events.get = advance, receive
        for slot in range(2):
            self.add(slot, self.request(slot))
        self.pool.run(self.jobs, keep_open=True)
        self.assertTrue(waits)
        self.assertEqual(set(waits), {0.25})
        self.assertEqual(self.committed, [11, 10])
        self.assertFalse(self.unfinished or self.runner.active)

    def test_cuda_waits_for_event_when_all_slots_already_wait_for_model(self):
        self.pool.device = torch.device("cuda")
        self.pool.clients = self.pool.clients[:1]
        self.pool.slots = {0: self.pool.slots[0]}
        self.pool.events.advance = lambda: self.fail("所有槽均等模型时仍轮询空输入队列")
        self.add(0, self.request(0))
        self.pool.run(self.jobs, keep_open=True)
        self.assertEqual(self.committed, [10])
        self.assertFalse(self.unfinished or self.runner.active)

    def test_cancel_during_queue_wait_discards_pending(self):
        event = threading.Event()
        self.pool.events.advance = event.set
        self.add(0, self.request(0))
        self.add(1, self.request(1, "b"))
        self.pool.run(self.jobs, cancel_event=event, keep_open=True)
        self.assertTrue(event.is_set())
        self.assertFalse(self.sent or self.committed)
        self.assertFalse(any(runner.active for runner in self.jobs.inferences.values()))
        self.assertEqual(self.unfinished, [10, 11])
        self.assertIn(("pool-close",), self.log)

    def test_cancel_closes_pending_without_committing_samples(self):
        event = threading.Event()
        self.runner.on_submit = event.set
        self.add(0, self.request(0))
        self.pool.run(self.jobs, cancel_event=event, keep_open=True)
        self.assertFalse(self.runner.active)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.committed, [])
        self.assertEqual(self.unfinished, [10, 11])
        self.assertIn(("pool-close",), self.log)

    def test_cancel_during_finish_sends_cancel_and_discards_other_ticket(self):
        event = threading.Event()
        original = self.runner.finish

        def finish(ticket):
            output = original(ticket)
            event.set()
            return output

        self.runner.finish = finish
        # 模拟取消回包发出后，工作进程尚未返回终止帧。
        self.pool.clients[0].send = lambda response: self.sent.append((0, 0, response))
        for slot in range(2):
            self.add(slot, self.request(slot))
        self.pool.run(self.jobs, cancel_event=event, keep_open=True)
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(self.sent[0][2]["cancel"])
        self.assertNotIn("logits", self.sent[0][2])
        self.assertEqual(self.committed, [])
        self.assertEqual(self.unfinished, [10, 11])
        self.assertFalse(self.runner.active)

    def test_inflight_eof_keeps_existing_memory_task_failure_boundary(self):
        self.runner.on_submit = lambda: self.add(0, RuntimeError("原生进程已退出"))
        self.add(0, self.request(0))
        with self.assertRaisesRegex(RuntimeError, "内存任务工作进程退出"):
            self.pool.run(self.jobs, keep_open=True)
        self.assertFalse(self.runner.active)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.committed, [])
        self.assertEqual(self.unfinished, [10, 11])

    def test_inflight_resource_error_discards_only_failed_slot(self):
        self.pool.max_batch_size = 2
        self.pool.read_seconds, self.pool.decode_seconds, self.pool.tensor_bytes = 1, 2, 3
        self.pool.restarts = 4
        self.pool.clients[0].read_seconds = 10
        self.pool.clients[0].decode_seconds = 20
        self.pool.clients[0].tensor_bytes = 30

        def fail_slot():
            self.add(0, MemoryError("超出帧上限"))
            self.add(0, RuntimeError("原生进程已退出"))

        self.runner.on_submit = fail_slot
        for slot in range(2):
            self.add(slot, self.request(slot))
        self.pool.run(self.jobs, keep_open=True)
        self.assertEqual(self.errors, ["超出帧上限"])
        self.assertEqual(self.pool.generations, [1, 0])
        self.assertEqual(
            (self.pool.read_seconds, self.pool.decode_seconds, self.pool.tensor_bytes),
            (11, 22, 33),
        )
        self.assertEqual(self.pool.restarts, 5)
        self.assertEqual([(slot, generation) for slot, generation, _ in self.sent], [(1, 0)])
        self.assertEqual(self.committed, [11])
        self.assertEqual(self.unfinished, [10])
        self.assertFalse(self.runner.active)

    def test_model_groups_fill_batches_and_alternate_without_draining(self):
        for maximum in (1, 2, 8):
            with self.subTest(maximum=maximum):
                if maximum != 1:
                    self.setUp()
                self.pool.max_batch_size = maximum
                self.pool.generations = [0] * 6
                self.pool.last_ids = [0] * 6
                self.pool.clients = [self.client(i) for i in range(6)]
                self.pool.slots = {
                    i: {"id": 10 + i, "models": ["a", "b"], "requests": 0} for i in range(6)
                }
                requests = [self.request(i, "b" if i % 2 else "a") for i in range(6)]
                for slot, request in enumerate(requests):
                    self.add(slot, request)
                expected = self.jobs.infer(requests)
                self.pool.run(self.jobs, keep_open=True)
                submitted = [item[1] for item in self.log if item[0] == "submit"]
                expected_groups = {
                    1: [((i, 0),) for i in range(6)],
                    2: [((0, 0), (2, 0)), ((1, 0), (3, 0)), ((4, 0),), ((5, 0),)],
                    8: [((0, 0), (2, 0), (4, 0)), ((1, 0), (3, 0), (5, 0))],
                }
                self.assertEqual(submitted, expected_groups[maximum])
                self.assertEqual([item[0] for item in self.log[:2]], ["submit", "submit"])
                self.assertEqual(sorted(self.committed), list(range(10, 16)))
                self.assertFalse(self.unfinished)
                self.assertFalse(any(runner.active for runner in self.jobs.inferences.values()))
                for slot, _, response in self.sent:
                    self.assertEqual(
                        response["logits"].tolist(), expected[slot, :1].view(torch.uint16).tolist()
                    )

    def test_snapshot_drains_live_tickets(self):
        calls = []
        clock = [time.perf_counter()]

        def snapshot(_):
            self.assertFalse(self.runner.active)
            calls.append(True)

        self.add(0, self.request(0))
        self.add(1, self.request(1))
        self.runner.on_submit = lambda: clock.__setitem__(0, clock[0] + 6)
        with patch("haojie_training.native.pool.time.perf_counter", side_effect=lambda: clock[0]):
            self.pool.run(self.jobs, snapshot=snapshot, keep_open=True)
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(self.committed, [10, 11])
        self.assertFalse(self.runner.active)

    def test_failed_submission_drains_pending_before_closing_pool(self):
        self.add(0, self.request(0))
        self.add(1, self.request(1, "b"))
        historical = self.jobs.inferences["b"]
        original = historical.submit

        def submit(*args, **kwargs):
            original(*args, **kwargs)
            raise ValueError("模拟提交失败")

        historical.submit = submit
        with self.assertRaisesRegex(ValueError, "模拟提交失败"):
            self.pool.run(self.jobs, keep_open=True)
        self.assertFalse(any(runner.active for runner in self.jobs.inferences.values()))
        self.assertEqual(self.committed, [])
        self.assertEqual(self.unfinished, [10, 11])
        self.assertEqual(self.log[-1], ("pool-close",))

    def test_unexpected_generation_change_rejects_old_response(self):
        self.add(0, self.request(0))
        self.runner.on_submit = lambda: self.pool.generations.__setitem__(0, 1)
        with self.assertRaisesRegex(ValueError, "进程代次"):
            self.pool.run(self.jobs, keep_open=True)
        self.assertFalse(self.sent or self.committed or self.runner.active)

    def test_completed_logits_reject_mixed_model_or_wrong_rows(self):
        with self.assertRaises(ValueError):
            self.jobs.infer([self.request(0), self.request(1, "b")], completed=torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            self.jobs.infer([self.request(0)], completed=torch.zeros(2, 2))

    def test_reply_thread_allows_next_gpu_batch_and_releases_owned_results(self):
        self.pool.async_replies = True
        self.pool.events = queue.Queue()
        self.pool.generations.append(0)
        self.pool.last_ids.append(0)
        self.pool.clients.append(self.client(2))
        self.pool.slots[2] = {"id": 12, "models": ["a"], "requests": 0}
        third_submitted = threading.Event()
        send_threads = []
        original = self.pool.clients[0].send

        def submit():
            for ticket in self.runner.active:
                ticket.event.synchronize()
            if sum(row[0] == "submit" for row in self.log) == 3:
                third_submitted.set()

        def send(response):
            send_threads.append(threading.current_thread())
            if not third_submitted.wait(3):
                raise AssertionError("回传阻塞了下一批 GPU 提交")
            self.assertTrue(torch.isfinite(torch.from_numpy(response["logits"])).all())
            original(response)

        self.runner.on_submit = submit
        self.pool.clients[0].send = send
        for slot in range(3):
            self.add(slot, self.request(slot))
        self.pool.run(self.jobs, keep_open=True)
        self.assertEqual(sorted(self.committed), [10, 11, 12])
        self.assertEqual(len(self.sent), 3)
        self.assertTrue(send_threads)
        self.assertTrue(all(not thread.is_alive() for thread in send_threads))
        self.assertFalse(self.unfinished or self.runner.active)

    def test_reply_thread_failure_propagates_before_pool_can_resume(self):
        self.pool.async_replies = True
        self.pool.events = queue.Queue()
        self.runner.on_submit = lambda: [
            ticket.event.synchronize() for ticket in self.runner.active
        ]
        self.pool.clients[0].send = lambda _: (_ for _ in ()).throw(BrokenPipeError())
        self.add(0, self.request(0))
        with self.assertRaisesRegex(RuntimeError, "工作进程退出"):
            self.pool.run(self.jobs, keep_open=True)
        self.assertFalse(self.runner.active or self.committed)
        self.assertEqual(self.unfinished, [10, 11])
        self.assertEqual(self.log[-1], ("pool-close",))


class AsyncOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_ticket_leases_and_result_ownership_without_cuda(self):
        runner = PolicyInference(PolicyValueNet(ModelConfig.tiny()), "cpu")
        self.assertFalse(runner.supports_async)
        with self.assertRaisesRegex(ValueError, "同步"):
            runner.submit([{}], slots=[(0, 0)])
        runner.packed = object()
        event = SimpleNamespace(synchronize=lambda: None)
        first = InferenceTicket(0, ((0, 0),), event, torch.tensor([[2.0]]), (object(),))
        second = InferenceTicket(1, ((1, 0),), event, torch.tensor([[7.0]]), (object(),))
        runner._tickets, runner._slots = [first, second], {(0, 0), (1, 0)}
        runner.stats.inflight = 2
        with self.assertRaisesRegex(ValueError, "在途"):
            runner.submit([{}], slots=[(0, 0)])
        with self.assertRaisesRegex(RuntimeError, "两组"):
            runner.submit([{}], slots=[(2, 0)])
        with self.assertRaisesRegex(RuntimeError, "回收"):
            runner([{}])
        output = runner.finish(first)
        with self.assertRaisesRegex(ValueError, "重复回收"):
            runner.finish(first)
        runner.close()
        self.assertEqual(output.item(), 2)
        self.assertFalse(first.owners or second.owners or runner._slots)
        self.assertEqual(runner.stats.async_discarded, 1)
        self.assertEqual(runner.stats.inflight, 0)

    def test_background_failure_propagates_and_releases_cpu_leases(self):
        runner = PolicyInference(PolicyValueNet(ModelConfig.tiny()), "cpu")
        failure = Future()
        failure.set_exception(ValueError("后台发射失败"))
        ticket = InferenceTicket(
            0,
            ((0, 0),),
            SimpleNamespace(query=lambda: self.fail("读取了未记录的事件")),
            torch.zeros(1, 1),
            (object(),),
            launch=failure,
        )
        runner._tickets, runner._slots = [ticket, None], {(0, 0)}
        runner.stats.inflight = 1
        with self.assertRaisesRegex(ValueError, "后台发射失败"):
            ticket.ready()
        with self.assertRaisesRegex(ValueError, "后台发射失败"):
            runner.close()
        self.assertFalse(ticket.owners or ticket.launch or runner._slots)
        self.assertEqual(runner.stats.inflight, 0)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and os.environ.get("HAOJIE_ATEN") == "1",
        "需要 CUDA BF16 和 ATen C++ 发射器",
    )
    def test_background_streams_refresh_weights_only_after_every_window_drains(self):
        self._background_streams_refresh_weights()

    def _background_streams_refresh_weights(self):
        torch.manual_seed(121)
        model = PolicyValueNet().cuda().eval()
        groups = [
            [
                {
                    key: value[0]
                    for key, value in synthetic_batch(
                        model.config, 1, entities, 4, seed=entities
                    ).items()
                    if key in INPUT_KEYS
                }
            ]
            for entities in (5, 9, 13)
        ]
        runner = PolicyInference(model, "cuda")
        previous = None
        launcher = ForwardWorkers(torch.device("cuda"))
        try:
            for window in range(2):
                if window:
                    with torch.no_grad():
                        model.blocks[0].qkv.weight.mul_(0.75)
                        model.policy[-1].bias.add_(0.1)
                with torch.no_grad():
                    expected = [runner(rows) for rows in groups]
                    runner.set_launcher(launcher)
                    first = runner.submit(groups[0], slots=[(0, window)])
                    second = runner.submit(groups[1], slots=[(1, window)])
                    self.assertIsNone(first.launch)
                    self.assertIsNotNone(second.launch)
                    retained = runner.finish(first)
                    third = runner.submit(groups[2], slots=[(2, window)])
                    for ticket, reference in ((second, expected[1]), (third, expected[2])):
                        ticket.launch.result(timeout=10)
                        torch.testing.assert_close(
                            runner.finish(ticket), reference, atol=3e-3, rtol=3e-3
                        )
                    torch.testing.assert_close(retained, expected[0], atol=3e-3, rtol=3e-3)
                    if previous is not None:
                        self.assertFalse(torch.equal(retained, previous))
                    previous = retained.clone()
                    runner.close()
        finally:
            runner.close()
            launcher.close()
        self.assertTrue(all(not thread.is_alive() for thread in launcher.threads))
        self.assertEqual(runner.stats.background_calls, 4)
        self.assertGreater(runner.stats.background_launch_seconds, 0)
        self.assertEqual(runner.stats.inflight, 0)
        self.assertFalse(torch.is_autocast_enabled("cuda"))

    def test_failed_device_submission_synchronizes_before_reuse(self):
        runner = PolicyInference(PolicyValueNet(ModelConfig.tiny()), "cpu")
        runner.packed = object()
        synchronized = []
        first = InferenceTicket(
            0,
            ((0, 0),),
            SimpleNamespace(synchronize=lambda: synchronized.append("first")),
            torch.zeros(1, 1),
            (object(),),
        )
        runner._tickets, runner._slots = [first, None], {(0, 0)}
        runner.stats.inflight = 1
        stream = SimpleNamespace(synchronize=lambda: synchronized.append("failed"))
        with (
            patch.object(runner, "_prepare", return_value=({}, None, None)),
            patch.object(runner, "_forward", side_effect=ValueError("提交设备计算失败")),
            patch("torch.cuda.current_stream", return_value=stream),
            self.assertRaisesRegex(ValueError, "提交设备计算失败"),
        ):
            runner.submit([{}], slots=[(1, 0)])
        self.assertEqual(synchronized, ["failed"])
        self.assertEqual(runner._tickets, [first, None])
        self.assertEqual(runner._slots, {(0, 0)})
        self.assertTrue(first.owners)
        runner.close()
        self.assertEqual(synchronized, ["failed", "first"])
        self.assertFalse(first.owners or runner._slots)

    def test_nonfinite_result_releases_only_finished_ticket(self):
        runner = PolicyInference(PolicyValueNet(ModelConfig.tiny()), "cpu")
        synchronized = []
        tickets = [
            InferenceTicket(
                i,
                ((i, 0),),
                SimpleNamespace(synchronize=lambda i=i: synchronized.append(i)),
                torch.tensor([[value]]),
                (object(),),
            )
            for i, value in enumerate((float("nan"), 1.0))
        ]
        first, second = tickets
        runner._tickets, runner._slots = tickets, {(0, 0), (1, 0)}
        runner.stats.inflight = 2
        with self.assertRaisesRegex(ValueError, "非有限"):
            runner.finish(first)
        self.assertEqual(synchronized, [0])
        self.assertFalse(first.owners)
        self.assertTrue(second.owners)
        self.assertEqual(runner._slots, {(1, 0)})
        runner.close()
        self.assertEqual(synchronized, [0, 1])
        self.assertFalse(second.owners or runner._slots)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "CUDA BF16 双缓冲需要支持的 GPU 执行器",
    )
    def test_cuda_bf16_outputs_timing_update_and_checkpoint_after_drain(self):
        self._cuda_same_outputs_timing_and_parameter_update_after_drain()

    def _cuda_same_outputs_timing_and_parameter_update_after_drain(self):
        torch.manual_seed(93)
        model = PolicyValueNet().cuda().eval()
        stats = InferenceStats()
        runner, reference = (
            PolicyInference(model, "cuda", stats=stats),
            PolicyInference(model, "cuda"),
        )
        if not runner.supports_async:
            self.skipTest("本 Torch/CUDA 后端沿用同步推理")
        groups = [
            [
                {
                    key: value[0]
                    for key, value in synthetic_batch(
                        model.config, 1, entities, 4, seed=entities
                    ).items()
                    if key in INPUT_KEYS
                }
            ]
            for entities in (5, 9, 13)
        ]
        expected = [reference(rows) for rows in groups]
        # 第256批的设备事件在 finish 时读取，不提前同步阻断下一组主机准备。
        stats.calls = 255
        first = runner.submit(groups[0], slots=[(0, 0)])
        second = runner.submit(groups[1], slots=[(1, 0)])
        output = runner.finish(first)
        torch.testing.assert_close(output, expected[0], atol=0, rtol=0)
        third = runner.submit(groups[2], slots=[(2, 0)])
        torch.testing.assert_close(runner.finish(second), expected[1], atol=0, rtol=0)
        self.assertEqual(stats.cuda_timing["batches"], 1)
        runner.close()
        self.assertFalse(third.owners)
        self.assertEqual(stats.async_discarded, 1)
        self.assertGreater(stats.async_overlaps, 0)
        saved = BytesIO()
        torch.save(cpu_state(model.state_dict()), saved)
        with torch.no_grad():
            model.policy[-1].weight.mul_(0.5)
        updated = runner.finish(runner.submit(groups[0], slots=[(0, 1)]))
        torch.testing.assert_close(updated, reference(groups[0]), atol=0, rtol=0)
        torch.testing.assert_close(output, expected[0], atol=0, rtol=0)
        self.assertFalse(torch.equal(updated, output))
        # 排空后保存、更新、恢复同一模型；两个执行器都必须读取恢复后的参数。
        runner.close()
        saved.seek(0)
        model.load_state_dict(torch.load(saved, weights_only=True))
        restored = runner.finish(runner.submit(groups[0], slots=[(0, 2)]))
        torch.testing.assert_close(restored, expected[0], atol=0, rtol=0)
        torch.testing.assert_close(restored, reference(groups[0]), atol=0, rtol=0)
        historical = PolicyInference(model, "cuda", stats=stats)
        overlaps = stats.async_overlaps
        first = runner.submit(groups[0], slots=[(0, 3)])
        second = historical.submit(groups[1], slots=[(1, 3)])
        self.assertEqual(stats.async_overlaps, overlaps + 1)
        torch.testing.assert_close(historical.finish(second), reference(groups[1]), atol=0, rtol=0)
        runner.finish(first)
        self.assertEqual(stats.inflight, 0)


if __name__ == "__main__":
    unittest.main()
