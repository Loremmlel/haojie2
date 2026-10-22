"""共享就绪调度器：模型/进程驻留，任务与槽位分离，每槽最多一个在途请求。"""

import queue
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import torch

from ..runtime import resolve_device
from . import client as client_module
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
        host_workers=None,
    ):
        if not 1 <= environments <= 128 or not 0 <= batch_wait_ms <= 10:
            raise ValueError("环境数须为1至128，组批等待须为0至10毫秒")
        if max_batch_size is not None and (
            type(max_batch_size) is not int or not 1 <= max_batch_size <= 256
        ):
            raise ValueError("推理批量上限必须为1至256的整数")
        self.max_batch_size = max_batch_size or environments
        if host_workers is not None and (
            type(host_workers) is not int or not 0 <= host_workers <= min(16, environments)
        ):
            raise ValueError("CPU 输入进程数必须为 0 至 min(16, 环境数) 的整数")
        self.started = time.perf_counter()
        self.frame_limit = frame_limit
        self.events = queue.Queue(maxsize=2 * environments)
        self.executable, self.clients, self.slots = executable, [], {}
        self.generations = [0] * environments
        self.last_ids = [0] * environments
        self.device = resolve_device(str(device))
        self.async_replies = self.device.type == "cuda"
        self.wait = batch_wait_ms / 1000
        self.responses = deque(maxlen=4096)
        self.queue_seconds = self.send_seconds = self.active_seconds = 0.0
        self.refill_seconds = self.refills = self.restarts = self.peak_queue = 0
        self.read_seconds = self.decode_seconds = self.tensor_bytes = 0
        self.assigned = self.finished = self.requests = 0
        self.first_done = self.stop_at = None
        self.parked = {}
        self.paused_at = self.running_since = None
        self.sampling_seconds = self.yield_count = 0
        self.closed = False
        self.input_processes = None
        self.defer_validation = (
            getattr(getattr(inference[0], "packed", None), "operator", None) is not None
        )
        self.python_switch_ms = sys.getswitchinterval() * 1000
        try:
            from .transport import InputProcesses, worker_count

            workers = (
                worker_count(environments, self.device) if host_workers is None else host_workers
            )
            if workers:
                self.input_processes = InputProcesses(
                    executable,
                    environments,
                    workers,
                    self.events,
                    frame_limit,
                    prepare_inputs=getattr(getattr(inference[0], "packed", None), "operator", None)
                    is None,
                    defer_validation=self.defer_validation,
                )
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
        if self.input_processes is not None:
            return self.input_processes.client(i, self.generations[i])
        return Client(
            self.executable,
            events=self.events,
            identity=(i, self.generations[i]),
            frame_limit=self.frame_limit,
            arrays=True,
            defer_validation=self.defer_validation,
        )

    def close(self):
        try:
            inference = getattr(self, "inference", None)
            if inference is not None and hasattr(inference, "close"):
                inference.close()
        finally:
            if self.input_processes is not None:
                self.input_processes.close()
            else:
                for client in self.clients:
                    client.close()
            self.closed = True
            self.slots.clear()
            self.parked.clear()
            # Reader 已停止，清空队列持有的帧；不把历史张量留到下一次运行。
            while True:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    break

    def metrics(self):
        responses = tuple(self.responses)
        return {
            "model": self.model_hash,
            "device": str(self.device),
            "precision": self.precision,
            "dispatch_backend": "aten-cpp"
            if getattr(getattr(self.inference, "packed", None), "operator", None)
            else "python",
            "input_layout": "raw-frames" if self.defer_validation else "numpy",
            "python_switch_ms": self.python_switch_ms,
            "response_workers": int(self.async_replies),
            "policy_layout": "compact-candidates" if self.defer_validation else "padded-candidates",
            "environments": len(self.clients),
            "host_workers": len(self.input_processes.workers) if self.input_processes else 0,
            "host_worker_pids": [p.pid for p in self.input_processes.workers]
            if self.input_processes
            else [],
            "shared_input_segments": len(self.input_processes.segments)
            if self.input_processes
            else 0,
            "peak_shared_input_segments": self.input_processes.peak_segments
            if self.input_processes
            else 0,
            "host_prepare_seconds": sum(getattr(c, "prepare_seconds", 0) for c in self.clients),
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
            "response_p95_seconds": sorted(responses)[int((len(responses) - 1) * 0.95)]
            if responses
            else 0,
            "response_window": len(responses),
            "response_window_capacity": 4096,
            "queue_capacity": self.events.maxsize,
            "peak_queue": self.peak_queue,
            "active_slot_seconds": self.active_seconds,
            "sampling_seconds": self.sampling_seconds
            + (time.perf_counter() - self.running_since if self.running_since is not None else 0),
            "yield_count": self.yield_count,
            "parked": len(self.parked),
            "closed": self.closed,
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
                    "parked": i in self.parked,
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
        yield_when=None,
    ):
        """仅接纳真实终局；可在各槽下一决策根前让出，保留原生局面和未回复请求。

        yield_when 只在 keep_open 时可用；回调仅判断窗口是否结束。让出前排空
        当前根的推理，调用者可更新权重，再以同一任务管理器继续。取消或错误
        仍丢弃未结束局，逐项调用 unfinished 后关闭，不把暂停伪装成终局。
        """
        if yield_when is not None and not keep_open:
            raise ValueError("保留在途局的采样窗口要求 keep_open=True")
        previous = time.perf_counter()
        # 独立测试可注入已有槽位；正常池由构造器初始化这些计数。
        self.parked = getattr(self, "parked", {})
        self.yield_count = getattr(self, "yield_count", 0)
        self.sampling_seconds = getattr(self, "sampling_seconds", 0)
        if getattr(self, "paused_at", None) is not None:
            paused = previous - self.paused_at
            for task in self.slots.values():
                if "last_activity" in task:
                    task["last_activity"] += paused
            for _, _, message in self.parked.values():
                message["received_at"] += paused
        self.paused_at, self.running_since = None, previous
        previous_active = 0
        next_snapshot = previous
        stopping = yielding = yielded = False
        cancel = False
        waiting = deque()
        ready = []
        active_inferences = set()
        inference_cleanup = ExitStack()
        sending = deque()
        sender = (
            inference_cleanup.enter_context(
                ThreadPoolExecutor(1, thread_name_prefix="haojie-reply")
            )
            if getattr(self, "async_replies", False)
            else None
        )
        discarded_slots = set()
        failed = True

        def flush_responses():
            while sending:
                sending.popleft().result()

        def send(responses, clients):
            sent = time.perf_counter()
            try:
                if self.input_processes is not None:
                    self.input_processes.respond(responses)
                else:
                    client_module.send_many(
                        [(client, response) for client, (_, response, _) in zip(clients, responses)]
                    )
                    for client, (_, _, message) in zip(clients, responses):
                        client.release(message)
            except BrokenPipeError as error:
                raise RuntimeError("内存任务工作进程退出；丢弃未完成局") from error
            self.send_seconds += time.perf_counter() - sent
            for _, _, message in responses:
                self.responses.append(time.perf_counter() - message["received_at"])

        def check_yield():
            nonlocal yielding, stopping
            if not yielding and yield_when is not None and yield_when():
                yielding = stopping = True
            return yielding

        def cancelled():
            nonlocal cancel
            cancel = cancel or bool(cancel_event and cancel_event.is_set())
            return cancel

        def respond(group, logits, leases=None):
            cancelled()
            # 每批只构造一次 NumPy 视图；原生客户端协商二进制回传，旧执行器沿用列表。
            scores = logits.numpy() if logits is not None else None
            responses = []
            for row, (i, task_id, message) in enumerate(group):
                if leases is not None:
                    if leases[row] in discarded_slots:
                        continue  # 资源失败已丢弃此局；同组其他槽仍可正常完成。
                    if leases[row] != (i, self.generations[i]):
                        raise ValueError("响应进程代次已改变")
                if self.slots.get(i, {}).get("id") != task_id:
                    raise ValueError("响应归属已改变")
                response = {"id": message["id"], "model": message["model"]}
                response.update(
                    {"cancel": True} if cancel else {"logits": scores[row, : message["candidates"]]}
                )
                responses.append((i, response, message))
            clients = [self.clients[i] for i, _, _ in responses]
            if sender is None:
                send(responses, clients)
            else:
                # 每批持有评分和原帧，最多排队两批；回传不占用 GPU 发射线程。
                while len(sending) >= 2:
                    sending.popleft().result()
                sending.append(sender.submit(send, responses, clients))

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
            flush_responses()

        def assign(i):
            task = jobs.claim()
            if task is None:
                return False
            task.update(
                began=time.perf_counter(),
                began_wall=time.time(),
                last_activity=time.perf_counter(),
                requests=0,
                startup_model=self.model_hash,
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

        def dispatch(pending):
            # 模型按首次就绪顺序轮流取一批，桶内按长度组批；同槽仍只有一个请求。
            by_model = {}
            for item in pending:
                i, _, message = item
                by_model.setdefault(message["model"], []).append(item)
            groups = deque(by_model.values())
            for pending in groups:
                pending.sort(key=lambda item: item[2]["entities"])
            while groups:
                # 两组都在途时继续收集就绪请求；槽释放后跨消息轮次合批，
                # 不先冻结一个小批次再同步等待旧票据。
                if len(waiting) == 2:
                    return [item for group in groups for item in group]
                pending = groups.popleft()
                inference = jobs.inferences[pending[0][2]["model"]]
                count = _group_size(
                    pending, self.max_batch_size, packed=inference.supports_packed_batches
                )
                group = pending[:count]
                if yield_when is not None:
                    for i, _, message in group:
                        self.slots[i]["last_root"] = (
                            message["commandIndex"],
                            message["actor"],
                        )
                if count < len(pending):
                    groups.append(pending[count:])
                if not cancelled() and getattr(inference, "supports_async", False):
                    if inference not in active_inferences:
                        active_inferences.add(inference)
                        inference_cleanup.callback(inference.close)
                    if not cancelled():
                        ticket = inference.submit(
                            [m["input"] for _, _, m in group],
                            slots=[(i, self.generations[i]) for i, _, _ in group],
                        )
                        waiting.append((group, inference, ticket))
                        continue
                # 不支持异步的后端沿用同步路由；当前/历史模型无需在切换时排空。
                drain()
                logits = None if cancelled() else jobs.infer([m for _, _, m in group])
                respond(group, logits)
            return []

        # 仅延长 AMP 权重缓存寿命，不在调度器开启低精度；各模型仍独立选择精度。
        # 窗口退出前先排空推理，再清空缓存，下一次更新后的权重不会读到旧转换值。
        inference_cleanup.enter_context(torch.autocast(self.device.type, enabled=False))
        inference_cleanup.callback(torch.clear_autocast_cache)
        if getattr(self, "defer_validation", False) and len(self.clients) >= 32:
            # 高频管道线程会拖延 GPU 发射线程重新取得 GIL；仅在采样窗口收紧间隔。
            previous_interval = sys.getswitchinterval()
            self.python_switch_ms = min(previous_interval, 0.0005) * 1000
            sys.setswitchinterval(self.python_switch_ms / 1000)
            inference_cleanup.callback(sys.setswitchinterval, previous_interval)
        try:
            if self.parked and not cancelled() and not check_yield():
                pending = list(self.parked.values())
                self.parked.clear()
                previous_active = len(self.slots)
                ready = dispatch(pending)
                del pending
            while True:
                while sending and sending[0].done():
                    sending.popleft().result()
                if cancel_event and cancel_event.is_set():
                    cancel = True
                    break
                now = time.perf_counter()
                self.active_seconds += (now - previous) * previous_active
                previous = now
                check_yield()
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
                if yielding and len(self.parked) == len(self.slots):
                    drain()
                    yielded = True
                    break
                if not self.slots:
                    break
                previous_active = len(self.slots) - len(self.parked)
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
                    if any(
                        now - t["last_activity"] > 900
                        for i, t in self.slots.items()
                        if i not in self.parked and "last_activity" in t
                    ):
                        raise TimeoutError("原生请求超过保护时间")
                    if not ready:
                        self.queue_seconds += time.perf_counter() - before
                        continue
                    # 已就绪请求也能单独推进，不能依赖下一帧唤醒才发射。
                    messages = []
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
                raw = message = None
                for (i, generation), raw in messages:
                    if generation != self.generations[i]:
                        continue  # 已关闭管道的 EOF；不能发送到新进程。
                    if i not in self.slots:
                        raise ValueError("空闲槽位收到旧局响应")
                    if isinstance(raw, MemoryError):
                        flush_responses()
                        discarded_slots.add((i, generation))
                        pending = [item for item in pending if item[0] != i]
                        self.parked.pop(i, None)
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
                    if message["type"] == "done" and message.get("model") == task.get(
                        "startup_model", self.model_hash
                    ):
                        self.slots.pop(i)
                        self.finished += 1
                        if self.first_done is None:
                            self.first_done = time.perf_counter() - self.started
                        jobs.finish(task, message)
                        if message.get("error") and message["outcome"]["reason"] != "cancelled":
                            raise RuntimeError(message["error"])
                        self.clients[i].release(message)
                    elif message["type"] == "example":
                        jobs.example(task, message)
                        self.clients[i].release(message)
                    elif message["type"] == "teacher":
                        task["commands"] = message.get("commandIndex", task.get("commands"))
                        task["ply"] = message.get("ply", task.get("ply"))
                        command = jobs.teacher_next(message)
                        self.clients[i].send({"id": message["id"], "command": command})
                        self.clients[i].release(message)
                    elif message["type"] == "infer" and message.get("model") in task["models"]:
                        if type(message.get("id")) is not int or message["id"] <= self.last_ids[i]:
                            raise ValueError("过期或重复的模型请求")
                        if yield_when is not None and (
                            type(message.get("commandIndex")) is not int
                            or message["commandIndex"] < 0
                            or type(message.get("actor")) is not int
                            or message["actor"] not in (1, 2)
                        ):
                            raise ValueError("采样窗口请求缺少合法决策根身份")
                        self.last_ids[i] = message["id"]
                        task["commands"] = message.get("commandIndex")
                        task["ply"] = message.get("ply")
                        task["requests"] += 1
                        self.requests += 1
                        pending.append((i, task["id"], message))
                    else:
                        raise ValueError("未知或过期模型请求")
                # 累积尚未发射的输入，同一批的 done 可使窗口达标。
                pending = ready + pending
                ready = []
                # 统一判断后才决定新根是否执行。
                if check_yield():
                    continuing = []
                    for item in pending:
                        i, _, message = item
                        if self.slots[i].get("last_root") == (
                            message["commandIndex"],
                            message["actor"],
                        ):
                            continuing.append(item)
                        else:
                            self.parked[i] = item
                    pending = continuing
                ready = dispatch(pending)
                del messages, pending, raw, message
            failed = False
        finally:
            try:
                try:
                    flush_responses()
                finally:
                    inference_cleanup.close()
            except BaseException:
                failed = True
                raise
            finally:
                waiting.clear()
                ready.clear()
                ended = time.perf_counter()
                self.sampling_seconds += ended - self.running_since
                self.running_since = None
                if yielded and not cancel and not failed:
                    self.yield_count += 1
                    self.paused_at = ended
                else:
                    for task in self.slots.values():
                        jobs.unfinished(task)
                    self.slots.clear()
                    self.parked.clear()
                if not keep_open or cancel or failed:
                    self.close()
        return self.metrics()
