"""共享就绪调度器：模型/进程驻留，任务与槽位分离，每槽最多一个在途请求。"""

import queue
import time
from collections import deque

from ..runtime import resolve_device
from .client import Client
from .execution import PolicyInference


class Pool:
    def __init__(
        self,
        executable,
        checkpoint,
        environments,
        device="cuda",
        precision="fp32",
        batch_wait_ms=0,
        *,
        inference=None,
    ):
        if not 1 <= environments <= 128 or not 0 <= batch_wait_ms <= 10:
            raise ValueError("环境数须为1至128，组批等待须为0至10毫秒")
        self.started = time.perf_counter()
        self.events = queue.Queue(maxsize=2 * environments)
        self.executable, self.clients, self.slots = executable, [], {}
        self.generations = [0] * environments
        self.last_ids = [0] * environments
        self.device, self.precision = resolve_device(str(device)), precision
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
            if inference is None:
                from .pipeline import model_from

                model, self.model_hash = model_from(checkpoint, self.ready, self.device)
                self.inference = PolicyInference(model, self.device, precision)
            else:
                self.inference, self.model_hash = inference
            self.startup_seconds = time.perf_counter() - self.started
        except BaseException:
            self.close()
            raise

    def new_client(self, i):
        return Client(self.executable, events=self.events, identity=(i, self.generations[i]))

    def close(self):
        for client in self.clients:
            client.close()
        # Reader 已停止，清空队列持有的帧；不把历史张量留到下一次运行。
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                break

    def metrics(self):
        infer = self.inference
        return {
            "model": self.model_hash,
            "device": str(self.device),
            "precision": self.precision,
            "environments": len(self.clients),
            "batch_wait_ms": self.wait * 1000,
            "forwards": infer.calls,
            "requests": self.requests,
            "inference_seconds": infer.execution_seconds,
            "collation_seconds": infer.collation_seconds,
            "queue_seconds": self.queue_seconds,
            "send_seconds": self.send_seconds,
            "tensor_read_seconds": self.read_seconds + sum(c.read_seconds for c in self.clients),
            "decode_validate_seconds": self.decode_seconds
            + sum(c.decode_seconds for c in self.clients),
            "tensor_bytes": self.tensor_bytes + sum(c.tensor_bytes for c in self.clients),
            "mean_batch": infer.requests / max(1, infer.calls),
            "batch_sizes": dict(infer.batch_sizes),
            "entity_fill": infer.entities / max(1, infer.entity_slots),
            "candidate_fill": infer.candidates / max(1, infer.candidate_slots),
            "cuda_timing_sample": infer.cuda_timing,
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

    def run(self, jobs, *, seconds=None, drain_seconds=120, stop_file=None, snapshot=None):
        """jobs 负责耐久任务事务；错误立即停止。deadline 包含加载、填满和排空。

        请求通过独占管道绑定当前任务，并校验进程代次与跨局递增 ID。仅 EOF/断管
        可重建槽位，单次运行最多两次；规则、张量或协议错误不重试。
        """
        deadline = self.started + seconds if seconds else float("inf")
        previous = time.perf_counter()
        previous_active = 0
        next_snapshot = previous
        stopping = False
        cancel = False
        interrupted = False

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
            self.clients[i].send(
                {
                    "op": "sample",
                    "record": str(task["record"]),
                    "start": task["start"],
                    "model": self.model_hash,
                    "samplerSeed": task["sampler_seed"],
                    "maxCommands": jobs.commands,
                    "maxPlies": jobs.plies,
                }
            )
            return True

        def restart(i, error):
            task = self.slots.pop(i)
            # 进程可能在发布完成文件后、送出回执前退出；先核对已发布文件。
            if task["record"].exists():
                with Client(self.executable) as verifier:
                    verifier.send({"op": "audit", "record": str(task["record"]), "encode": False})
                    audited = verifier.receive()["report"]
                jobs.finish(
                    task,
                    {
                        "outcome": audited["outcome"],
                        "finalHash": audited["finalHash"],
                        "recovered_audit": audited,
                    },
                )
                self.finished += 1
                if audited["outcome"]["reason"] == "error":
                    raise RuntimeError("退出进程留下实现错误记录")
            else:
                jobs.failed(task, str(error))
            if self.restarts >= 2:
                raise RuntimeError("原生槽位连续退出，已停止重建") from error
            old = self.clients[i]
            old.close()
            self.read_seconds += old.read_seconds
            self.decode_seconds += old.decode_seconds
            self.tensor_bytes += old.tensor_bytes
            self.generations[i] += 1
            self.last_ids[i] = 0
            self.clients[i] = self.new_client(i)
            if self.clients[i].ready != self.ready:
                raise ValueError("重建进程版本不一致")
            self.restarts += 1

        try:
            while True:
                now = time.perf_counter()
                self.active_seconds += (now - previous) * previous_active
                previous = now
                if stop_file and stop_file.exists() and not stopping:
                    deadline = min(deadline, now + drain_seconds)
                if (
                    jobs.stop()
                    or interrupted
                    or (stop_file and stop_file.exists())
                    or now >= deadline - drain_seconds
                ):
                    stopping = True
                    if self.stop_at is None:
                        self.stop_at = now - self.started
                cancel = cancel or now >= deadline - min(5, seconds / 10 if seconds else 5)
                if now >= deadline:
                    break
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
                    snapshot(self.metrics())
                    next_snapshot = now + 5
                before = time.perf_counter()
                try:
                    messages = [self.events.get(timeout=min(0.25, max(0.001, deadline - now)))]
                except queue.Empty:
                    self.queue_seconds += time.perf_counter() - before
                    if any(
                        now - t["last_activity"] > 900
                        for t in self.slots.values()
                        if "last_activity" in t
                    ):
                        raise TimeoutError("原生请求超过保护时间")
                    continue
                except KeyboardInterrupt:
                    if interrupted:
                        cancel = True
                    interrupted = True
                    deadline = min(deadline, time.perf_counter() + drain_seconds)
                    continue
                until = min(deadline, time.perf_counter() + self.wait)
                self.peak_queue = max(self.peak_queue, 1 + self.events.qsize())
                while len(messages) < len(self.slots):
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
                    if isinstance(raw, RuntimeError) and str(raw) == "原生进程已退出":
                        restart(i, raw)
                        continue
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
                    elif message["type"] == "infer" and message.get("model") == self.model_hash:
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
                pending.sort(key=lambda item: item[2]["entities"])
                while pending:
                    limit = pending[0][2]["entities"] * 2
                    count = next(
                        (j for j, (_, _, m) in enumerate(pending) if m["entities"] > limit),
                        len(pending),
                    )
                    group, pending = pending[:count], pending[count:]
                    cancel = cancel or time.perf_counter() >= deadline - min(
                        5, seconds / 10 if seconds else 5
                    )
                    logits = None if cancel else self.inference([m["input"] for _, _, m in group])
                    for row, (i, task_id, message) in enumerate(group):
                        if self.slots[i]["id"] != task_id:
                            raise ValueError("响应归属已改变")
                        sent = time.perf_counter()
                        response = {"id": message["id"], "model": self.model_hash}
                        response.update(
                            {"cancel": True}
                            if cancel
                            else {"logits": logits[row, : message["candidates"]].tolist()}
                        )
                        try:
                            self.clients[i].send(response)
                        except BrokenPipeError as error:
                            restart(i, error)
                        self.send_seconds += time.perf_counter() - sent
                        self.responses.append(time.perf_counter() - message["received_at"])
                    # 局部变量不能将上一批张量保留到下一局或排空阶段。
                    del logits, group
                del messages, pending, raw, message
        finally:
            for task in self.slots.values():
                jobs.unfinished(task)
            self.close()
        return self.metrics()
