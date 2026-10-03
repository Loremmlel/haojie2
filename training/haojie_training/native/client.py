"""有界二进制协议。读到的字节复制为独立张量，不借用下次请求会覆盖的缓冲。"""

import json
import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch

from ..data import validate_inputs
from ..model import ModelConfig


class Client:
    def __init__(self, executable, timeout=900):
        self.timeout = timeout
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
                    data = bytearray()
                    while len(data) < size:
                        chunk = self.process.stdout.read(size - len(data))
                        if not chunk:
                            raise ValueError("张量消息不完整")
                        data.extend(chunk)
                    meta["input"] = decode(data, meta["entities"], meta["candidates"])
                self.messages.put(meta)
            raise RuntimeError("原生进程已退出")
        except Exception as error:
            self.messages.put(error)

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


def decode(data, entities, candidates):
    if any(type(n) is not int or n < 1 for n in (entities, candidates)):
        raise ValueError("张量维度无效")
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
        inputs[key] = torch.from_numpy(array.copy())
        offset += count * np.dtype(dtype).itemsize
    if offset != len(data):
        raise ValueError("张量字节长度不匹配")
    validate_inputs({key: value.unsqueeze(0) for key, value in inputs.items()}, ModelConfig())
    return inputs
