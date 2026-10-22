"""有界二进制协议。每帧独占接收内存，张量视图持有该帧至最后消费者释放。"""

import ctypes
import json
import math
import os
import queue
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from ..model import ModelConfig
from .aten import native_io

LOGITS_CAPABILITY = "logits-f32-v1"
KIND_COUNT = ModelConfig().kind_count


class Client:
    def __init__(
        self,
        executable,
        timeout=900,
        *,
        events=None,
        identity=None,
        frame_limit=256 * 1024 * 1024,
        arrays=False,
        buffer_factory=None,
        defer_validation=False,
    ):
        self.timeout = timeout
        self.frame_limit = frame_limit
        self.events, self.identity = events, identity
        self.arrays = arrays
        self.buffer_factory = buffer_factory
        self.defer_validation = defer_validation
        self._handshake = True
        self._input_refs = False
        self._closed = threading.Event()
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
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.stderr_reader = threading.Thread(target=self._stderr, daemon=True)
        self.reader.start()
        self.stderr_reader.start()
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
        last_input = None
        try:
            while line := self.process.stdout.readline(16 * 1024 * 1024 + 1):
                if len(line) > 16 * 1024 * 1024:
                    raise ValueError("控制消息过长")
                if not line.endswith(b"\n"):
                    raise RuntimeError("原生进程已退出")
                meta = json.loads(line)
                if "inputRef" in meta:
                    identity = (
                        meta["inputRef"],
                        meta.get("model"),
                        meta.get("actor"),
                        meta.get("index"),
                    )
                    if (
                        not self._input_refs
                        or meta.get("type") != "example"
                        or type(meta["inputRef"]) is not int
                        or meta["inputRef"] < 1
                        or any(value is None for value in identity)
                        or any(key in meta for key in ("bytes", "entities", "candidates", "input"))
                        or last_input is None
                        or any(
                            type(value) is not type(previous) or value != previous
                            for value, previous in zip(identity, last_input[0])
                        )
                    ):
                        raise ValueError("推理输入引用未协商、过期或身份不匹配")
                    meta.update(last_input[1])
                elif "bytes" in meta:
                    size = meta["bytes"]
                    entities, candidates = meta.get("entities"), meta.get("candidates")
                    if (
                        any(type(n) is not int or n < 1 for n in (size, entities, candidates))
                        or size != entities * 265 + candidates * 273 + 128
                    ):
                        raise ValueError("张量消息长度无效")
                    if size > self.frame_limit:
                        raise MemoryError(
                            f"合法张量需要{size}字节，超过本槽位{self.frame_limit}字节内存预算"
                        )
                    factory = getattr(self, "buffer_factory", None)
                    data = factory(size) if factory else bytearray(size)
                    started = time.perf_counter()
                    view = memoryview(data)
                    offset = 0
                    while offset < size:
                        count = self.process.stdout.readinto(view[offset:])
                        if not count:
                            raise RuntimeError("原生进程已退出")
                        offset += count
                    self.read_seconds += time.perf_counter() - started
                    self.tensor_bytes += size
                    started = time.perf_counter()
                    meta["input"] = decode(
                        data,
                        meta["entities"],
                        meta["candidates"],
                        _owned=True,
                        _arrays=getattr(self, "arrays", False),
                        _defer=getattr(self, "defer_validation", False)
                        and meta.get("type") == "infer",
                    )
                    self.decode_seconds += time.perf_counter() - started
                    del data, view
                    if meta.get("type") == "infer" and self._input_refs:
                        last_input = (
                            (
                                meta.get("id"),
                                meta.get("model"),
                                meta.get("actor"),
                                meta.get("commandIndex"),
                            ),
                            {
                                key: meta[key]
                                for key in ("bytes", "entities", "candidates", "input")
                            },
                        )
                if meta.get("type") in ("example", "done", "error", "teacher"):
                    last_input = None
                self._publish(meta)
                del meta
            raise RuntimeError("原生进程已退出")
        except Exception as error:
            last_input = None
            self._publish(error)

    def _publish(self, message):
        if isinstance(message, dict) and message.get("type") == "infer":
            message["received_at"] = time.perf_counter()
        if self._handshake or self.events is None:
            self._handshake = False
            destination, value = self.messages, message
        else:
            destination, value = self.events, (self.identity, message)
        # 背压只保留当前帧；关闭时必须能解除阻塞并释放张量。
        while not self._closed.is_set():
            try:
                destination.put(value, timeout=0.1)
                break
            except queue.Full:
                pass

    def send(self, value):
        if "op" in value:
            enabled = value.get("inputRefs") is True
            if enabled and "mc-input-ref-v1" not in self.ready.get("capabilities", []):
                raise ValueError("原生引擎未协商推理输入引用能力")
            self._input_refs = enabled
        payload = b""
        logits = value.get("logits")
        if isinstance(logits, np.ndarray):
            if logits.ndim != 1 or logits.dtype != np.float32:
                raise ValueError("模型评分需要一维 CPU FP32 数组")
            if not np.isfinite(logits).all():
                raise ValueError("模型输出非有限")
            if LOGITS_CAPABILITY in self.ready.get("capabilities", []):
                payload = logits.astype("<f4", copy=False).tobytes()
                value = {**value, "logitsBytes": len(payload)}
                del value["logits"]
            else:
                value = {**value, "logits": logits.tolist()}
        try:
            self.process.stdin.write(
                json.dumps(value, allow_nan=False, separators=(",", ":")).encode() + b"\n" + payload
            )
            self.process.stdin.flush()
        except OSError as error:
            # Windows 退出中的匿名管道也可能报 EINVAL；仅确认进程退出后归为断管。
            try:
                self.process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                raise error
            raise BrokenPipeError("原生进程已退出") from error

    def receive(self):
        try:
            value = self.messages.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError("原生请求超过保护时间") from None
        return self.checked(value)

    def release(self, message):
        """普通客户端的帧独占内存，无需归还共享槽位。"""

    def checked(self, value):
        if isinstance(value, MemoryError):
            raise value
        if isinstance(value, Exception):
            raise RuntimeError(f"{value}: {self.errors.decode(errors='replace')}")
        if value.get("type") == "error":
            raise ValueError(value["error"])
        return value

    def close(self):
        self._closed.set()
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        self.reader.join(timeout=2)
        self.stderr_reader.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except OSError:
                # 进程已经退出；BufferedWriter 关闭时冲刷旧字节可能再次报断管。
                if self.process.poll() is None:
                    raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ArrayInput(Mapping):
    """公开帧及按需数组视图；C++ 拼装只读取地址，不为每条请求创建八个数组。"""

    def __init__(self, values=None, *, data=None, entities=None, candidates=None):
        if isinstance(values, ArrayInput):
            self.__dict__.update(values.__dict__)
            return
        self._values, self._data = values, data
        self.entity_count = len(values["entities"]) if values is not None else entities
        self.candidate_count = len(values["sources"]) if values is not None else candidates
        self.frame_bytes = self.entity_count * 265 + self.candidate_count * 273 + 128
        if data is not None:
            self.address = (
                values["entities"].ctypes.data
                if values is not None
                else ctypes.addressof(ctypes.c_char.from_buffer(data))
            )
        self.validated = getattr(values, "validated", False)

    def __getitem__(self, key):
        if self._values is None:
            self._values = array_views(
                memoryview(self._data)[: self.frame_bytes], self.entity_count, self.candidate_count
            )._values
        return self._values[key]

    def __iter__(self):
        return iter(
            (
                "entities",
                "globals",
                "candidates",
                "kinds",
                "sources",
                "targets",
                "entity_mask",
                "candidate_mask",
            )
        )

    def __len__(self):
        return 8

    def ensure_validated(self):
        if not self.validated:
            validate(self)


def array_views(data, entities, candidates, *, lazy=False):
    """仅重建已在输入进程校验的布局；调用者负责持有帧至消费者归还。"""
    if lazy:
        return ArrayInput(data=data, entities=entities, candidates=candidates)
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
        count = math.prod(shape)
        array = np.ndarray(shape, dtype=dtype, buffer=data, offset=offset)
        inputs[key] = array
        offset += count * np.dtype(dtype).itemsize
    if offset != len(data):
        raise ValueError("张量字节长度不匹配")
    return ArrayInput(inputs, data=data)


def decode(data, entities, candidates, *, _owned=False, _arrays=False, _defer=False):
    if any(type(n) is not int or n < 1 for n in (entities, candidates)):
        raise ValueError("张量维度无效")
    if len(data) != entities * 265 + candidates * 273 + 128:
        raise ValueError("张量字节长度不匹配")
    # 可变调用者缓冲不能借用；读取线程交付独占 bytearray 后不再写入。
    inputs = array_views(
        data if _owned else bytearray(data), entities, candidates, lazy=_defer and _arrays
    )
    if not _defer:
        validate(inputs)
    return inputs if _arrays else {key: torch.from_numpy(value) for key, value in inputs.items()}


def validate(inputs):
    entities, candidates = inputs.entity_count, inputs.candidate_count
    extension = native_io() if hasattr(inputs, "address") else None
    if extension is not None:
        error = extension.haojie_validate(inputs.address, entities, candidates, KIND_COUNT)
        if error:
            raise ValueError(
                {
                    1: "输入含非有限数值",
                    2: "实体类别超出词表",
                    3: "每个样本至少要有一个可训练候选",
                    4: "sources实体指针越界",
                    5: "sources指向填充实体",
                    6: "targets实体指针越界",
                    7: "targets指向填充实体",
                }[error]
            )
        inputs.validated = True
        return
    # 帧布局已限定白名单、dtype与形状；直接在零拷贝视图校验，避免读取线程
    # 为几十个小型CPU算子反复进出PyTorch/GIL。与validate_inputs保持同一数值边界。
    if any(not np.isfinite(inputs[k]).all() for k in ("entities", "globals", "candidates")):
        raise ValueError("输入含非有限数值")
    kinds = inputs["kinds"]
    if (kinds < 0).any() or (kinds >= KIND_COUNT).any():
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
    inputs.validated = True


def send_many(responses):
    """一次释放 GIL 回传一批评分；取消、旧协议及非连续数组沿用单条出口。"""
    extension = native_io()
    if (
        extension is None
        or not responses
        or any(
            LOGITS_CAPABILITY not in getattr(client, "ready", {}).get("capabilities", [])
            or not isinstance(value.get("logits"), np.ndarray)
            or value["logits"].dtype != np.float32
            or value["logits"].ndim != 1
            or not value["logits"].flags.c_contiguous
            for client, value in responses
        )
    ):
        for client, value in responses:
            client.send(value)
        return
    count = len(responses)
    integers = ctypes.c_int64 * count
    headers = [
        json.dumps(
            {
                **{k: v for k, v in value.items() if k != "logits"},
                "logitsBytes": value["logits"].nbytes,
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for _, value in responses
    ]
    error = extension.haojie_send(
        integers(*(client.process.stdin.fileno() for client, _ in responses)),
        integers(*(value["logits"].ctypes.data for _, value in responses)),
        integers(*(len(value["logits"]) for _, value in responses)),
        (ctypes.c_char_p * count)(*headers),
        count,
    )
    if error == -1:
        raise ValueError("模型输出非有限")
    if error:
        raise OSError(error, os.strerror(error))
