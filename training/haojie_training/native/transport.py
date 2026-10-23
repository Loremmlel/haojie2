"""CPU 输入进程分摊协议与索引准备；共享槽在显式归还前不可覆盖，GPU 仍由主进程独占。"""

import asyncio
import math
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace

import numpy as np
import torch

from .client import ArrayInput, Client, array_views, send_many
from .packed import prepare_row


class ReadLoop:
    """Linux 管道共用一个事件循环，避免每个环境的阻塞读线程争抢 GIL。"""

    def __init__(self):
        self.loop, self.tasks = asyncio.new_event_loop(), {}
        self.thread = threading.Thread(target=self.run, name="haojie-input", daemon=True)
        self.thread.start()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.loop.close()

    def add(self, client):
        async def register():
            self.tasks[client] = asyncio.create_task(client._read_async())

        asyncio.run_coroutine_threadsafe(register(), self.loop).result(timeout=10)

    def remove(self, client):
        async def unregister():
            task = self.tasks.pop(client, None)
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        asyncio.run_coroutine_threadsafe(unregister(), self.loop).result(timeout=10)

    def close(self):
        if not self.loop.is_closed():
            for client in list(self.tasks):
                self.remove(client)
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2)


def worker_count(environments, device="cpu"):
    """显式 auto 按 CPU/RAM 配额分摊输入；默认沿用本机实测较快的线程路径。"""
    requested = os.environ.get("HAOJIE_HOST_WORKERS", "0")
    if requested == "auto":
        if torch.device(device).type != "cuda":
            return 0
        from ..console.config.resources import available_memory, cpu_capacity

        cores = math.floor(cpu_capacity())
        memory = max(0, available_memory() - 2 * 2**30) // (512 * 2**20)
        return min(16, environments // 4, cores // 3 if cores >= 4 else 0, memory)
    count = int(requested)
    if not 0 <= count <= 16:
        raise ValueError("HAOJIE_HOST_WORKERS 必须为 auto 或 0 至 16")
    return min(environments, count)


class FrameBuffer:
    """按实际帧增长；旧容量只在槽归还后替换，未映射部分不预占物理 RAM。"""

    def __init__(self):
        self.current = None
        self.segments = []
        self.retired = []
        self.ring = [None, None]
        self.index, self.serial = 0, 0
        self.references, self.leases = {}, {}
        self.condition = threading.Condition()
        self.stopped = False

    def acquire(self, size):
        with self.condition:
            segment = self.ring[self.index]
            while segment is not None and self.references.get(segment.name, 0):
                self.condition.wait(0.1)
                if self.stopped:
                    raise RuntimeError("CPU 输入槽已关闭")
            if self.stopped:
                raise RuntimeError("CPU 输入槽已关闭")
            # 原始帧后预留索引区：四组 int64 索引的最坏长度小于帧长的 1/8。
            reserved = size + (size + 7) // 8 + 64
            if segment is None or segment.size < reserved:
                if segment is not None:
                    # 本环槽的所有租约已归还；扩容立即回收旧容量和文件句柄。
                    self.retired.append(segment.name)
                    self.references.pop(segment.name, None)
                    self.segments.remove(segment)
                    segment.close()
                    segment.unlink()
                segment = SharedMemory(create=True, size=1 << (reserved - 1).bit_length())
                self.segments.append(segment)
                self.ring[self.index] = segment
            self.current = segment
            self.index = 1 - self.index
            return segment.buf[:size]

    def packing(self, prepared, size):
        """索引随原帧共享同一租约，跨进程仅发送偏移和长度。"""
        offset = (size + 7) & ~7
        descriptors = []
        for values in prepared[:-1]:
            if values is None:
                descriptors.append(None)
                continue
            target = np.ndarray(
                values.shape, dtype=np.int64, buffer=self.current.buf, offset=offset
            )
            target[:] = values
            descriptors.append((offset, len(values)))
            offset += values.nbytes
        return descriptors, prepared[-1]

    async def acquire_async(self, size):
        while True:
            with self.condition:
                segment = self.ring[self.index]
                if self.stopped or segment is None or not self.references.get(segment.name, 0):
                    return self.acquire(size)
            await asyncio.sleep(0.001)

    def retain(self):
        with self.condition:
            self.serial += 1
            name = self.current.name
            self.references[name] = self.references.get(name, 0) + 1
            self.leases[self.serial] = name
            return name, self.serial

    def release(self, lease):
        with self.condition:
            name = self.leases.pop(lease)
            self.references[name] -= 1
            self.condition.notify_all()

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()

    def close(self):
        for segment in self.segments:
            segment.close()
            try:
                segment.unlink()
            except FileNotFoundError:
                pass  # 中断退出时父进程可能已接管并释放名称。
        self.segments.clear()


def _worker(
    executable, slots, commands, events, frame_limit, prepare_inputs, defer_validation, stopping
):
    """spawn 不继承 CUDA 上下文；Linux 合并管道读取，Windows 保留原槽位读线程。"""
    torch.set_num_threads(1)
    clients, buffers, closed, collectors = {}, {}, {}, {}
    slots = set(slots)
    behavior = (None, None)
    read_loop = ReadLoop() if os.name != "nt" else None

    class Publisher:
        def outgoing(self, value):
            identity, message = value
            slot, _ = identity
            started = time.perf_counter()
            collector = collectors.get(slot)
            if collector is not None and isinstance(message, dict):
                if message.get("type") == "example":
                    # 原帧来自可复用共享槽；只有蓄水入选的样本才独立复制。
                    message["input"].shared = True
                    collector.add(message, *behavior)
                    return None
                if message.get("type") == "infer":
                    message["_trajectory_progress"] = collector.progress()
                elif message.get("type") == "done":
                    message["_collected_trajectory"] = pickle.dumps(
                        collector.trajectory, protocol=5
                    )
                    collectors.pop(slot)
            if isinstance(message, dict) and "input" in message:
                message = dict(message)
                inputs = message.pop("input")
                message["_validated"] = inputs.validated
                message["_retired"], buffers[slot].retired = buffers[slot].retired, []
                if message["type"] == "infer" and prepare_inputs:
                    message["_packing"] = buffers[slot].packing(
                        prepare_row(inputs), message["bytes"]
                    )
                message["_buffer"], message["_lease"] = buffers[slot].retain()
            client = clients[slot]
            metrics = (client.read_seconds, client.decode_seconds, client.tensor_bytes)
            outgoing = ("event", identity, message, metrics, time.perf_counter() - started)
            return slot, outgoing

        def put(self, value, timeout):
            prepared = self.outgoing(value)
            if prepared is None:
                return
            slot, outgoing = prepared
            while not closed[slot].is_set():
                try:
                    events.put(outgoing, timeout=0.1)
                    break
                except queue.Full:
                    pass

        async def put_async(self, value):
            prepared = self.outgoing(value)
            if prepared is None:
                return
            slot, outgoing = prepared
            while not closed[slot].is_set():
                try:
                    events.put_nowait(outgoing)
                    return
                except queue.Full:
                    await asyncio.sleep(0.001)

    def close(slot):
        closed[slot].set()
        buffers[slot].stop()
        clients[slot].close()
        collectors.pop(slot, None)
        del clients[slot]
        buffers.pop(slot).close()

    def create(slot, generation):
        slots.add(slot)
        buffers[slot] = FrameBuffer()
        closed[slot] = threading.Event()
        client = Client(
            executable,
            events=Publisher(),
            identity=(slot, generation),
            arrays=True,
            frame_limit=frame_limit,
            buffer_factory=buffers[slot].acquire,
            buffer_factory_async=buffers[slot].acquire_async,
            defer_validation=defer_validation,
            read_loop=read_loop,
            startup_cancel=stopping,
        )
        clients[slot] = client
        events.put(("ready", slot, generation, client.ready, client.process.pid))

    try:
        for slot in sorted(slots):
            create(slot, 0)
        while not stopping.is_set():
            operation, slot, value = commands.recv()
            if operation == "stop":
                break
            if operation == "send":
                if value.get("op") == "sample":
                    collectors[slot] = value.pop("_collector", None)
                clients[slot].send(value)
            elif operation == "behavior":
                behavior = value
            elif operation == "responses":
                send_many([(clients[target], response) for target, response, _ in value])
                for target, response, lease in value:
                    buffers[target].release(lease)
            elif operation == "release":
                buffers[slot].release(value)
            elif operation == "close":
                close(slot)
            elif operation == "create":
                try:
                    create(slot, value)
                except Exception as error:
                    # 扩容失败只拒绝这个新槽，已有原生局继续留在同一输入进程。
                    if slot in clients:
                        close(slot)
                    else:
                        buffer = buffers.pop(slot, None)
                        if buffer is not None:
                            buffer.close()
                        closed.pop(slot, None)
                    if stopping.is_set():
                        break
                    events.put(("create-error", slot, value, str(error)))
    except BaseException as error:
        events.put(("fatal", tuple(slots), f"{type(error).__name__}: {error}"))
    finally:
        for slot in list(clients):
            close(slot)
        if read_loop is not None:
            read_loop.close()
        commands.close()
        # 父进程关闭时不等待尚未消费的诊断消息；大张量从不进入此队列。
        events.cancel_join_thread()


class SharedInput(ArrayInput):
    shared = True

    def __init__(self, values, packing):
        super().__init__(values)
        self.packing = packing


class ProcessClient:
    """Pool 的客户端代理；只有元数据和小型评分跨队列，共享帧在 release 前有效。"""

    def __init__(self, owner, slot):
        self.owner, self.slot = owner, slot
        self.generation = 0
        self.ready_queue = queue.Queue()
        self.errors = b""
        self.read_seconds = self.decode_seconds = self.tensor_bytes = self.prepare_seconds = 0
        self.closed = False
        self.segment_names = set()

    def initialize(self, generation=0, *, create=False):
        self.generation, self.closed = generation, False
        if generation or create:
            self.owner.command(self.slot, "create", generation)
        until = time.monotonic() + 900
        while not self.owner.closed.is_set():
            remaining = until - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("CPU 输入槽启动超过保护时间")
            try:
                value = self.ready_queue.get(timeout=min(0.1, remaining))
                break
            except queue.Empty:
                pass
        else:
            raise InterruptedError("CPU 输入进程已关闭")
        if isinstance(value, Exception):
            raise value
        self.ready, pid = value
        self.process = SimpleNamespace(pid=pid)
        return self

    def send(self, value):
        self.owner.command(self.slot, "send", value)

    def checked(self, value):
        value = Client.checked(self, value)
        for name in value.pop("_retired", ()):
            self.segment_names.discard(name)
            retired = self.owner.segments.pop(name, None)
            if retired is not None:
                retired.close()
        if "_buffer" in value:
            name = value.pop("_buffer")
            self.segment_names.add(name)
            if name not in self.owner.segments:
                self.owner.segments[name] = SharedMemory(name=name)
                self.owner.peak_segments = max(self.owner.peak_segments, len(self.owner.segments))
            segment = self.owner.segments[name]
            values = array_views(segment.buf, value["entities"], value["candidates"], lazy=True)
            values.validated = value.pop("_validated", False)
            packing = value.pop("_packing", None)
            if packing is not None:
                descriptors, length = packing
                packing = (
                    *(
                        None
                        if descriptor is None
                        else np.ndarray(
                            (descriptor[1],),
                            dtype=np.int64,
                            buffer=segment.buf,
                            offset=descriptor[0],
                        )
                        for descriptor in descriptors
                    ),
                    length,
                )
            value["input"] = SharedInput(values, packing)
        return value

    def release(self, message):
        lease = message.pop("_lease", None)
        if lease is not None:
            message.pop("input", None)
            self.owner.command(self.slot, "release", lease)

    def close(self):
        if not self.closed:
            try:
                self.owner.command(self.slot, "close", None)
            except (BrokenPipeError, OSError):
                pass
            self.closed = True
            for name in self.segment_names:
                segment = self.owner.segments.pop(name, None)
                if segment is not None:
                    segment.close()
            self.segment_names.clear()


class InputProcesses:
    def __init__(
        self,
        executable,
        environments,
        workers,
        events,
        frame_limit,
        *,
        prepare_inputs=True,
        defer_validation=False,
    ):
        self.events, self.segments = events, {}
        self.closed = threading.Event()
        self.clients = [ProcessClient(self, slot) for slot in range(environments)]
        self.workers, self.commands, self.locks, self.routes = [], [], [], {}
        self.prepare_inputs = prepare_inputs
        self.defer_validation = defer_validation
        self.peak_segments = 0
        context = mp.get_context("spawn")
        self.stopping = context.Event()
        self.incoming = context.Queue(maxsize=256)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self._start(context, executable, environments, workers, frame_limit)
        except BaseException:
            self.close()
            raise

    def _start(self, context, executable, environments, workers, frame_limit):
        for index in range(workers):
            slots = list(range(index, environments, workers))
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_worker,
                args=(
                    str(executable),
                    slots,
                    receiver,
                    self.incoming,
                    frame_limit,
                    self.prepare_inputs,
                    self.defer_validation,
                    self.stopping,
                ),
                name=f"haojie-input-{index}",
            )
            try:
                process.start()
            except BaseException:
                sender.close()
                raise
            finally:
                receiver.close()
            self.workers.append(process)
            self.commands.append(sender)
            self.locks.append(threading.Lock())
            self.routes.update(dict.fromkeys(slots, index))

    def client(self, slot, generation):
        if slot == len(self.clients):
            self.clients.append(ProcessClient(self, slot))
            self.routes[slot] = slot % len(self.workers)
            return self.clients[slot].initialize(generation, create=True)
        if generation:
            self.clients[slot] = ProcessClient(self, slot)
        return self.clients[slot].initialize(generation)

    def command(self, slot, operation, value):
        index = self.routes[slot]
        with self.locks[index]:
            self.commands[index].send((operation, slot, value))

    def collection_version(self, value):
        for sender, lock in zip(self.commands, self.locks):
            with lock:
                sender.send(("behavior", 0, value))

    def respond(self, responses):
        """同一模型批次按 CPU 进程合并回传与归还，避免每条评分两次 IPC。"""
        groups = {}
        for slot, response, message in responses:
            groups.setdefault(self.routes[slot], []).append((slot, response, message.pop("_lease")))
            message.pop("input", None)
        for index, values in groups.items():
            with self.locks[index]:
                self.commands[index].send(("responses", 0, values))

    def _read(self):
        failed = set()
        while not self.closed.is_set():
            try:
                item = self.incoming.get(timeout=0.1)
            except queue.Empty:
                for index, process in enumerate(self.workers):
                    if process.exitcode is not None and index not in failed:
                        failed.add(index)
                        for slot, route in tuple(self.routes.items()):
                            if route == index:
                                self._failure(slot, f"CPU 输入进程退出：{process.exitcode}")
                continue
            kind, *values = item
            if kind == "ready":
                slot, generation, ready, pid = values
                client = self.clients[slot]
                if generation == client.generation:
                    client.ready_queue.put((ready, pid))
            elif kind == "create-error":
                slot, generation, error = values
                client = self.clients[slot]
                if generation == client.generation:
                    client.ready_queue.put(RuntimeError(error))
            elif kind == "fatal":
                slots, error = values
                for slot in slots:
                    self._failure(slot, error)
            else:
                identity, message, metrics, elapsed = values
                client = self.clients[identity[0]]
                if identity[1] != client.generation:
                    continue  # 重建槽位后，旧读线程的尾消息不能改写新进程统计。
                client.read_seconds, client.decode_seconds, client.tensor_bytes = metrics
                client.prepare_seconds += elapsed
                self._publish((identity, message))

    def _failure(self, slot, error):
        client = self.clients[slot]
        if client.closed:
            return
        value = RuntimeError(error)
        client.ready_queue.put(value)
        self._publish(((slot, client.generation), value))

    def _publish(self, value):
        while not self.closed.is_set():
            try:
                self.events.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def close(self):
        self.closed.set()
        self.stopping.set()
        for sender, lock in zip(self.commands, self.locks):
            try:
                with lock:
                    sender.send(("stop", 0, None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.reader.join(timeout=2)
        for sender in self.commands:
            sender.close()
        self.incoming.close()
        for segment in self.segments.values():
            segment.close()
            try:
                segment.unlink()
            except FileNotFoundError:
                pass  # 正常退出的输入进程已释放；崩溃时由父进程接管。
        self.segments.clear()
