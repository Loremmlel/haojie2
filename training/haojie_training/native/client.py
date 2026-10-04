"""有界二进制协议。每帧独占接收内存，张量视图持有该帧至最后消费者释放。"""

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..model import ModelConfig


class Client:
    def __init__(self, executable, timeout=900, *, events=None, identity=None):
        self.timeout = timeout
        self.events, self.identity = events, identity
        self._handshake = True
        self.read_seconds = self.decode_seconds = 0.0
        self.tensor_bytes = 0
        self.process = subprocess.Popen(
            [str(Path(executable).resolve()), "--training"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.messages = queue.Queue(maxsize=2)
        self.errors = bytearray()
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._stderr, daemon=True).start()
        try:
            self.ready = self.receive()
            if self.ready.get("protocol") != "haojie-training-binary-v1":
                raise ValueError("原生训练协议不匹配")
        except BaseException:
            self.close()
            raise

    def _stderr(self):
        for line in self.process.stderr:
            self.errors.extend(line)
            del self.errors[:-8192]

    def _read(self):
        try:
            while line := self.process.stdout.readline(16 * 1024 * 1024 + 1):
                if len(line) > 16 * 1024 * 1024 or not line.endswith(b"\n"):
                    raise ValueError("控制消息过长或不完整")
                meta = json.loads(line)
                if "bytes" in meta:
                    size = meta["bytes"]
                    if type(size) is not int or not 0 < size <= 256 * 1024 * 1024:
                        raise ValueError("张量消息长度无效")
                    data = bytearray(size)
                    started = time.perf_counter()
                    view = memoryview(data)
                    offset = 0
                    while offset < size:
                        count = self.process.stdout.readinto(view[offset:])
                        if not count:
                            raise ValueError("张量消息不完整")
                        offset += count
                    self.read_seconds += time.perf_counter() - started
                    self.tensor_bytes += size
                    started = time.perf_counter()
                    meta["input"] = decode(data, meta["entities"], meta["candidates"], _owned=True)
                    self.decode_seconds += time.perf_counter() - started
                self._publish(meta)
            raise RuntimeError("原生进程已退出")
        except Exception as error:
            self._publish(error)

    def _publish(self, message):
        if isinstance(message, dict) and message.get("type") == "infer":
            message["received_at"] = time.perf_counter()
        if self._handshake or self.events is None:
            self._handshake = False
            self.messages.put(message)
        else:
            self.events.put((self.identity, message))

    def send(self, value):
        self.process.stdin.write(
            json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        )
        self.process.stdin.flush()

    def receive(self):
        try:
            value = self.messages.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError("原生请求超过保护时间") from None
        return self.checked(value)

    def checked(self, value):
        if isinstance(value, Exception):
            raise RuntimeError(f"{value}: {self.errors.decode(errors='replace')}")
        if value.get("type") == "error":
            raise ValueError(value["error"])
        return value

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def decode(data, entities, candidates, *, _owned=False):
    if any(type(n) is not int or n < 1 for n in (entities, candidates)):
        raise ValueError("张量维度无效")
    if len(data) != entities * 265 + candidates * 273 + 128:
        raise ValueError("张量字节长度不匹配")
    # 可变调用者缓冲不能借用；读取线程交付独占 bytearray 后不再写入。
    if not _owned:
        data = bytearray(data)
    offset = 0
    inputs = {}
    for key, shape, dtype in (
        ("entities", (entities, 64), "<f4"),
        ("globals", (32,), "<f4"),
        ("candidates", (candidates, 64), "<f4"),
        ("kinds", (entities,), "<i8"),
        ("sources", (candidates,), "<i8"),
        ("targets", (candidates,), "<i8"),
        ("entity_mask", (entities,), "?"),
        ("candidate_mask", (candidates,), "?"),
    ):
        count = int(np.prod(shape))
        array = np.frombuffer(data, dtype=dtype, count=count, offset=offset).reshape(shape)
        inputs[key] = array
        offset += count * np.dtype(dtype).itemsize
    if offset != len(data):
        raise ValueError("张量字节长度不匹配")
    # 帧布局已限定白名单、dtype与形状；直接在零拷贝视图校验，避免读取线程
    # 为几十个小型CPU算子反复进出PyTorch/GIL。与validate_inputs保持同一数值边界。
    if any(not np.isfinite(inputs[k]).all() for k in ("entities", "globals", "candidates")):
        raise ValueError("输入含非有限数值")
    kinds = inputs["kinds"]
    if (kinds < 0).any() or (kinds >= ModelConfig().kind_count).any():
        raise ValueError("实体类别超出词表")
    mask = inputs["candidate_mask"]
    if not mask.any():
        raise ValueError("每个样本至少要有一个可训练候选")
    for key in ("sources", "targets"):
        indices = inputs[key]
        if (indices < -1).any() or (indices >= entities).any():
            raise ValueError(f"{key}实体指针越界")
        selected = indices[mask & (indices >= 0)]
        if not inputs["entity_mask"][selected].all():
            raise ValueError(f"{key}指向填充实体")
    return {key: torch.from_numpy(value) for key, value in inputs.items()}
