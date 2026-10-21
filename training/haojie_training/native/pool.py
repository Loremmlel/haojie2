"""共享就绪调度器：模型/进程驻留，任务与槽位分离，每槽最多一个在途请求。"""

import queue
import time
from collections import deque
from contextlib import ExitStack

from ..runtime import resolve_device
from .client import Client


def _group_size(pending, maximum, *, packed=False):
    """取排序后的连续前缀；紧凑执行器无需限制长度比，普通执行器保留1.25倍上限。"""
    if packed:
        return min(len(pending), maximum)
    limit = pending[0][2]["entities"] * 5 // 4
    count = next(
        (j for j, (_, _, message) in enumerate(pending) if message["entities"] > limit),
        len(pending),
    )
    return min(count, maximum)


class Pool:
    def __init__(
        self,
        executable,
        environments,
        device="cuda",
        batch_wait_ms=0,
        *,
        inference,
        max_batch_size=None,
        frame_limit=256 * 1024 * 1024,
    ):
        if not 1 <= environments <= 128 or not 0 <= batch_wait_ms <= 10:
            raise ValueError("环境数须为1至128，组批等待须为0至10毫秒")
        if max_batch_size is not None and (
            type(max_batch_size) is not int or not 1 <= max_batch_size <= 256
        ):
            raise ValueError("推理批量上限必须为1至256的整数")
        self.max_batch_size = max_batch_size or environments
        self.started = time.perf_counter()
        self.frame_limit = frame_limit
        self.events = queue.Queue(maxsize=2 * environments)
        self.executable, self.clients, self.slots = executable, [], {}
        self.generations = [0] * environments
        self.last_ids = [0] * environments
        self.device = resolve_device(str(device))
        self.wait = batch_wait_ms / 1000
        self.responses = deque(maxlen=4096)
        self.queue_seconds = self.send_seconds = self.active_seconds = 0.0
        self.refill_seconds = self.refills = self.restarts = self.peak_queue = 0
        self.read_seconds = self.decode_seconds = self.tensor_bytes = 0
        self.assigned = self.finished = self.requests = 0
        self.first_done = self.stop_at = None
        try:
            for i in range(environments):
                self.clients.append(self.new_client(i))
            self.ready = self.clients[0].ready
            if any(c.ready != self.ready for c in self.clients):
                raise ValueError("工作进程版本不一致")
            self.inference, self.model_hash = inference
            self.precision = self.inference.precision
            self.startup_seconds = time.perf_counter() - self.started
        except BaseException:
            self.close()
            raise

    def new_client(self, i):
        return Client(
            self.executable,
            events=self.events,
            identity=(i, self.generations[i]),
            frame_limit=self.frame_limit,
        )

    def close(self):
        try:
            inference = getattr(self, "inference", None)
            if inference is not None and hasattr(inference, "close"):
                inference.close()
        finally:
            for client in self.clients:
                client.close()
            # Reader 已停止，清空队列持有的帧；不把历史张量留到下一次运行。
            while True:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    break

    def metrics(self):
        return {
            "model": self.model_hash,
            "device": str(self.device),
            "precision": self.precision,
            "environments": len(self.clients),
            "batch_wait_ms": self.wait * 1000,
            **self.inference.stats.snapshot(),
            "max_batch_size": self.max_batch_size,
            "requests": self.requests,
            "queue_seconds": self.queue_seconds,
            "send_seconds": self.send_seconds,
            "tensor_read_seconds": self.read_seconds + sum(c.read_seconds for c in self.clients),
            "decode_validate_seconds": self.decode_seconds
            + sum(c.decode_seconds for c in self.clients),
            "tensor_bytes": self.tensor_bytes + sum(c.tensor_bytes for c in self.clients),
            "response_p95_seconds": sorted(self.responses)[int((len(self.responses) - 1) * 0.95)]
            if self.responses
            else 0,
            "response_window": len(self.responses),
            "response_window_capacity": 4096,
            "queue_capacity": self.events.maxsize,
            "peak_queue": self.peak_queue,
            "active_slot_seconds": self.active_seconds,
            "refills": self.refills,
            "refill_seconds": self.refill_seconds,
            "restarts": self.restarts,
            "assigned": self.assigned,
            "finished": self.finished,
            "startup_seconds": self.startup_seconds,
            "first_done_seconds": self.first_done,
            "stop_admission_seconds": self.stop_at,
            "seconds": time.perf_counter() - self.started,
            "active": [
                {
                    "slot": i,
                    "pid": self.clients[i].process.pid,
                    "task": t["id"],
                    "attempt": t.get("attempt", 1),
                    "requests": t["requests"],
                    "commands": t.get("commands"),
                    "ply": t.get("ply"),
                    "seconds": time.perf_counter() - t["began"],
                }
                for i, t in self.slots.items()
            ],
        }

    def run(
        self,
        jobs,
        *,
        snapshot=None,
        cancel_event=None,
        keep_open=False,
    ):
        """内存任务仅接纳完整终局；请求按独占管道、进程代次与递增 ID 校验归属。"""
        previous = time.perf_counter()
        previous_active = 0
        next_snapshot = previous
        stopping = False
        cancel = False
        waiting = deque()
        active_inferences = set()
        inference_cleanup = ExitStack()
        discarded_slots = set()
        failed = True

        def cancelled():
            nonlocal cancel
            cancel = cancel or bool(cancel_event and cancel_event.is_set())
            return cancel

        def respond(group, logits, leases=None):
            cancelled()
            for row, (i, task_id, message) in enumerate(group):
                if leases is not None:
                    if leases[row] in discarded_slots:
                        continue  # 资源失败已丢弃此局；同组其他槽仍可正常完成。
                    if leases[row] != (i, self.generations[i]):
                        raise ValueError("响应进程代次已改变")
                if self.slots.get(i, {}).get("id") != task_id:
                    raise ValueError("响应归属已改变")
                sent = time.perf_counter()
                response = {"id": message["id"], "model": message["model"]}
                response.update(
                    {"cancel": True}
                    if cancel
                    else {"logits": logits[row, : message["candidates"]].tolist()}
                )
                try:
                    self.clients[i].send(response)
                except BrokenPipeError as error:
                    raise RuntimeError("内存任务工作进程退出；丢弃未完成局") from error
                self.send_seconds += time.perf_counter() - sent
                self.responses.append(time.perf_counter() - message["received_at"])

        def receive():
            group, inference, ticket = waiting.popleft()
            logits = inference.finish(ticket)
            if not cancelled():
                if any(jobs.inferences[m["model"]] is not inference for _, _, m in group):
                    raise ValueError("模型执行器在推理未排空时被替换")
                logits = jobs.infer([m for _, _, m in group], completed=logits)
            respond(group, logits, ticket.slots)

        def drain():
            while waiting:
                receive()

        def assign(i):
            task = jobs.claim()
            if task is None:
                return False
            task.update(
                began=time.perf_counter(),
                began_wall=time.time(),
                last_activity=time.perf_counter(),
                requests=0,
            )
            self.slots[i] = task
            self.assigned += 1
            request = {
                "op": "sample",
                "start": task["start"],
                "model": self.model_hash,
                "samplerSeed": task["sampler_seed"],
                "maxCommands": jobs.commands,
                "maxPlies": jobs.plies,
                "memory": True,
                "models": task["models"],
                "mcContext": jobs.mc_context,
            }
            if request["mcContext"] and "mc-input-ref-v1" in self.ready.get("capabilities", []):
                request["inputRefs"] = True
            if task.get("teacher"):
                request["teacher"] = task["teacher"]
            self.clients[i].send(request)
            return True

        try:
            while True:
                if cancel_event and cancel_event.is_set():
                    cancel = True
                    break
                now = time.perf_counter()
                self.active_seconds += (now - previous) * previous_active
                previous = now
                if jobs.stop():
                    stopping = True
                    if self.stop_at is None:
                        self.stop_at = now - self.started
                if not stopping:
                    for i in range(len(self.clients)):
                        if i not in self.slots:
                            before = time.perf_counter()
                            if assign(i) and self.finished:
                                self.refills += 1
                                self.refill_seconds += time.perf_counter() - before
                if not self.slots:
                    break
                previous_active = len(self.slots)
                if snapshot and now >= next_snapshot:
                    drain()
                    snapshot(self.metrics())
                    next_snapshot = now + 5
                # 设备执行期间继续接收新帧；队列暂空不阻塞回收尚未完成的票据。
                if waiting and (cancelled() or waiting[0][2].ready()):
                    receive()
                before = time.perf_counter()
                try:
                    messages = [self.events.get(timeout=0.001 if waiting else 0.25)]
                except queue.Empty:
                    self.queue_seconds += time.perf_counter() - before
                    if any(
                        now - t["last_activity"] > 900
                        for t in self.slots.values()
                        if "last_activity" in t
                    ):
                        raise TimeoutError("原生请求超过保护时间")
                    continue
                until = time.perf_counter() + self.wait
                self.peak_queue = max(self.peak_queue, 1 + self.events.qsize())
                # MC 每槽通常先发 example 再发 infer；控制消息不能占掉推理批量名额。
                # 仅多排空已就绪消息，不增加等待时间；上限与有界队列一致。
                while len(messages) < 2 * len(self.slots):
                    try:
                        messages.append(
                            self.events.get(timeout=max(0, until - time.perf_counter()))
                        )
                    except queue.Empty:
                        break
                self.queue_seconds += time.perf_counter() - before
                pending = []
                message = None
                for (i, generation), raw in messages:
                    if generation != self.generations[i]:
                        continue  # 已关闭管道的 EOF；不能发送到新进程。
                    if i not in self.slots:
                        raise ValueError("空闲槽位收到旧局响应")
                    if isinstance(raw, MemoryError):
                        discarded_slots.add((i, generation))
                        pending = [item for item in pending if item[0] != i]
                        jobs.unfinished(self.slots.pop(i))
                        jobs.resource_error(str(raw))
                        client = self.clients[i]
                        client.close()
                        self.generations[i] += 1
                        self.last_ids[i] = 0
                        self.clients[i] = self.new_client(i)
                        # 替换成功后合并已停止的 Reader；重建失败仍由旧 Client 计账。
                        self.read_seconds += client.read_seconds
                        self.decode_seconds += client.decode_seconds
                        self.tensor_bytes += client.tensor_bytes
                        self.restarts += 1
                        del client
                        continue
                    if isinstance(raw, RuntimeError) and str(raw) == "原生进程已退出":
                        raise RuntimeError("内存任务工作进程退出；丢弃未完成局") from raw
                    message = self.clients[i].checked(raw)
                    task = self.slots[i]
                    task["last_activity"] = time.perf_counter()
                    if message["type"] == "done" and message.get("model") == self.model_hash:
                        self.slots.pop(i)
                        self.finished += 1
                        if self.first_done is None:
                            self.first_done = time.perf_counter() - self.started
                        jobs.finish(task, message)
                        if message.get("error") and message["outcome"]["reason"] != "cancelled":
                            raise RuntimeError(message["error"])
                    elif message["type"] == "example":
                        jobs.example(task, message)
                    elif message["type"] == "teacher":
                        task["commands"] = message.get("commandIndex", task.get("commands"))
                        task["ply"] = message.get("ply", task.get("ply"))
                        command = jobs.teacher_next(message)
                        self.clients[i].send({"id": message["id"], "command": command})
                    elif message["type"] == "infer" and message.get("model") in task["models"]:
                        if type(message.get("id")) is not int or message["id"] <= self.last_ids[i]:
                            raise ValueError("过期或重复的模型请求")
                        self.last_ids[i] = message["id"]
                        task["commands"] = message.get("commandIndex")
                        task["ply"] = message.get("ply")
                        task["requests"] += 1
                        self.requests += 1
                        pending.append((i, task["id"], message))
                    else:
                        raise ValueError("未知或过期模型请求")
                # 模型按首次就绪顺序轮流取一批，桶内按长度组批；同槽仍只有一个请求。
                by_model = {}
                for item in pending:
                    by_model.setdefault(item[2]["model"], []).append(item)
                if pending:
                    del item
                groups = deque(by_model.values())
                for pending in groups:
                    pending.sort(key=lambda item: item[2]["entities"])
                while groups:
                    pending = groups.popleft()
                    inference = jobs.inferences[pending[0][2]["model"]]
                    count = _group_size(
                        pending, self.max_batch_size, packed=inference.supports_packed_batches
                    )
                    group = pending[:count]
                    if count < len(pending):
                        groups.append(pending[count:])
                    if not cancelled() and getattr(inference, "supports_async", False):
                        if inference not in active_inferences:
                            active_inferences.add(inference)
                            inference_cleanup.callback(inference.close)
                        if len(waiting) == 2:
                            receive()
                        if not cancelled():
                            ticket = inference.submit(
                                [m["input"] for _, _, m in group],
                                slots=[(i, self.generations[i]) for i, _, _ in group],
                            )
                            waiting.append((group, inference, ticket))
                            del group
                            continue
                    # 不支持异步的后端沿用同步路由；当前/历史模型无需在切换时排空。
                    drain()
                    logits = None if cancelled() else jobs.infer([m for _, _, m in group])
                    respond(group, logits)
                    # 局部变量不能将上一批张量保留到下一局或排空阶段。
                    del logits, group
                del messages, pending, by_model, groups, raw, message
            failed = False
        finally:
            try:
                inference_cleanup.close()
            finally:
                waiting.clear()
                for task in self.slots.values():
                    jobs.unfinished(task)
                self.slots.clear()
                if not keep_open or cancel or failed:
                    self.close()
        return self.metrics()
