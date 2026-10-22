"""CPU 输入进程分摊协议与索引准备；共享槽在显式归还前不可覆盖，GPU 仍由主进程独占。"""

import math
import multiprocessing as mp
import os
import queue
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace

import torch

from .client import Client, array_views
from .packed import prepare_row


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
            if segment is None or segment.size < size:
                segment = SharedMemory(create=True, size=1 << (size - 1).bit_length())
                self.segments.append(segment)
                self.ring[self.index] = segment
            self.current = segment
            self.index = 1 - self.index
            return segment.buf[:size]

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
            segment.unlink()
        self.segments.clear()


def _worker(executable, slots, commands, events, frame_limit):
    """spawn 不继承 CUDA 上下文；每个引擎仍有独立读线程，阻塞 I/O 不占其他槽。"""
    torch.set_num_threads(1)
    clients, buffers, closed = {}, {}, {}

    class Publisher:
        def put(self, value, timeout):
            identity, message = value
            slot, _ = identity
            started = time.perf_counter()
            if isinstance(message, dict) and "input" in message:
                message = dict(message)
                inputs = message.pop("input")
                if message["type"] == "infer":
                    message["_packing"] = prepare_row(inputs)
                message["_buffer"], message["_lease"] = buffers[slot].retain()
            client = clients[slot]
            metrics = (client.read_seconds, client.decode_seconds, client.tensor_bytes)
            outgoing = ("event", identity, message, metrics, time.perf_counter() - started)
            while not closed[slot].is_set():
                try:
                    events.put(outgoing, timeout=0.1)
                    break
                except queue.Full:
                    pass

    def close(slot):
        closed[slot].set()
        buffers[slot].stop()
        clients[slot].close()
        del clients[slot]
        buffers.pop(slot).close()

    def create(slot, generation):
        buffers[slot] = FrameBuffer()
        closed[slot] = threading.Event()
        client = Client(
            executable,
            events=Publisher(),
            identity=(slot, generation),
            arrays=True,
            frame_limit=frame_limit,
            buffer_factory=buffers[slot].acquire,
        )
        clients[slot] = client
        events.put(("ready", slot, client.ready, client.process.pid))

    try:
        for slot in slots:
            create(slot, 0)
        while True:
            operation, slot, value = commands.recv()
            if operation == "stop":
                break
            if operation == "send":
                clients[slot].send(value)
            elif operation == "responses":
                for target, response, lease in value:
                    clients[target].send(response)
                    buffers[target].release(lease)
            elif operation == "release":
                buffers[slot].release(value)
            elif operation == "close":
                close(slot)
            elif operation == "create":
                create(slot, value)
    except BaseException as error:
        events.put(("fatal", tuple(slots), f"{type(error).__name__}: {error}"))
    finally:
        for slot in list(clients):
            close(slot)
        commands.close()
        # 父进程关闭时不等待尚未消费的诊断消息；大张量从不进入此队列。
        events.cancel_join_thread()


class SharedInput(dict):
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

    def initialize(self, generation=0):
        self.generation, self.closed = generation, False
        if generation:
            self.owner.command(self.slot, "create", generation)
        value = self.ready_queue.get(timeout=900)
        if isinstance(value, Exception):
            raise value
        self.ready, pid = value
        self.process = SimpleNamespace(pid=pid)
        return self

    def send(self, value):
        self.owner.command(self.slot, "send", value)

    def checked(self, value):
        value = Client.checked(self, value)
        if "_buffer" in value:
            name = value.pop("_buffer")
            if name not in self.owner.segments:
                self.owner.segments[name] = SharedMemory(name=name)
            segment = self.owner.segments[name]
            values = array_views(
                segment.buf[: value["bytes"]], value["entities"], value["candidates"]
            )
            value["input"] = SharedInput(values, value.pop("_packing", None))
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


class InputProcesses:
    def __init__(self, executable, environments, workers, events, frame_limit):
        self.events, self.segments = events, {}
        self.closed = threading.Event()
        self.clients = [ProcessClient(self, slot) for slot in range(environments)]
        self.workers, self.commands, self.locks, self.routes = [], [], [], {}
        context = mp.get_context("spawn")
        self.incoming = context.Queue(maxsize=2 * environments)
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
                args=(str(executable), slots, receiver, self.incoming, frame_limit),
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
        if generation:
            self.clients[slot] = ProcessClient(self, slot)
        return self.clients[slot].initialize(generation)

    def command(self, slot, operation, value):
        index = self.routes[slot]
        with self.locks[index]:
            self.commands[index].send((operation, slot, value))

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
                        for slot, route in self.routes.items():
                            if route == index:
                                self._failure(slot, f"CPU 输入进程退出：{process.exitcode}")
                continue
            kind, *values = item
            if kind == "ready":
                slot, ready, pid = values
                self.clients[slot].ready_queue.put((ready, pid))
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
