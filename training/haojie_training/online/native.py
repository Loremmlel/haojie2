"""线程池的批量 ctypes 边界；热路径只有连续 BF16/整数缓冲。"""

import ctypes
import json
import os
import struct
import time
from pathlib import Path

import torch


class NativeFailure(RuntimeError):
    """保留错误时的原生局面，供控制层写入独立诊断文件。"""

    def __init__(self, payload):
        self.payload = payload
        message = payload.decode("utf-8")
        if message.startswith("{"):
            message = json.loads(message)["error"]
        super().__init__(message)


def library_path():
    root = Path(__file__).resolve().parents[3]
    name = "haojie_native.dll" if os.name == "nt" else "libhaojie_native.so"
    return root / "artifacts" / "native-target" / "release" / name


class Worker:
    def __init__(self, library=None):
        self.dll = ctypes.CDLL(
            str(library or os.environ.get("HAOJIE_NATIVE_LIB") or library_path())
        )
        self.dll.hj_create.restype = ctypes.c_void_p
        self.dll.hj_submit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.dll.hj_poll.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.dll.hj_close.argtypes = [ctypes.c_void_p]
        self.dll.hj_memory_bytes.restype = ctypes.c_size_t
        self.dll.hj_identity.restype = ctypes.c_char_p
        self.identity = json.loads(self.dll.hj_identity())
        self.handle = self.dll.hj_create()
        self.busy = False

    def submit(self, op, payload=b""):
        if self.busy:
            raise RuntimeError("原生 worker 已有在途工作")
        if self.dll.hj_submit(self.handle, op, payload, len(payload)):
            raise RuntimeError("原生 worker 拒绝提交")
        self.busy = True

    def poll(self):
        data, size = ctypes.c_void_p(), ctypes.c_size_t()
        status = self.dll.hj_poll(self.handle, ctypes.byref(data), ctypes.byref(size))
        if status == 0:
            return None
        self.busy = False
        payload = ctypes.string_at(data, size.value)
        if status < 0:
            raise NativeFailure(payload)
        return payload

    def wait(self):
        while (value := self.poll()) is None:
            time.sleep(0.001)
        return value

    def close(self):
        if self.handle:
            self.dll.hj_close(self.handle)
            self.handle = None

    def memory(self):
        return self.dll.hj_memory_bytes()


def packet(admissions=(), replies=()):
    data = bytearray(struct.pack("<I", len(admissions)))
    for game, seed, seat, opponent, shrine in admissions:
        data.extend(struct.pack("<QIIQI", game, seed, seat, opponent, int(shrine)))
    data.extend(struct.pack("<I", len(replies)))
    for game, probabilities in replies:
        if probabilities.device.type != "cpu" or probabilities.dtype != torch.bfloat16:
            raise ValueError("原生回复必须是 CPU BF16 概率")
        data.extend(struct.pack("<QI", game, len(probabilities)))
        data.extend(probabilities.contiguous().view(torch.uint16).numpy().tobytes())
    return bytes(data)


def frames(payload):
    view = memoryview(payload)
    offset = 4
    (count,) = struct.unpack_from("<I", view)
    result = []
    for _ in range(count):
        (size,) = struct.unpack_from("<I", view, offset)
        offset += 4
        end = offset + size
        fields = struct.unpack_from("<7Q3I2id6I", view, offset)
        offset += 108
        (
            game,
            root,
            opponent,
            commands,
            steps,
            failures,
            backtracks,
            actor,
            learner,
            kind,
            winner,
            selected,
            logp,
            has_root,
            entities,
            prefix,
            candidates,
            events,
            event_start,
        ) = fields

        def tensor(n, dtype, shape):
            nonlocal offset
            width = torch.empty((), dtype=dtype).element_size()
            if n == 0:
                return torch.empty(shape, dtype=dtype)
            data = bytearray(view[offset : offset + n * width])
            offset += n * width
            return torch.frombuffer(data, dtype=dtype).reshape(shape)

        root_input = None
        node = None
        if kind == 0:
            if has_root:
                root_input = {
                    "entities": tensor(entities * 64, torch.bfloat16, (entities, 64)),
                    "kinds": tensor(entities, torch.uint32, (entities,)).long(),
                    "globals": tensor(32, torch.bfloat16, (32,)),
                }
            node = {
                "root": (game, root),
                "prefix": tensor(prefix * 64, torch.bfloat16, (prefix, 64)),
                "candidates": tensor(candidates * 64, torch.bfloat16, (candidates, 64)),
                "sources": tensor(candidates, torch.int32, (candidates,)).long(),
                "targets": tensor(candidates, torch.int32, (candidates,)).long(),
                "mask": tensor(candidates, torch.bool, (candidates,)),
                "history": {"events": tensor(events * 68, torch.bfloat16, (events, 68))},
                "history_start": event_start,
                "history_end": event_start + events,
            }
            if node["mask"].sum() < 2 or any(
                bool(((node[k] < -1) | (node[k] >= entities)).any()) for k in ("sources", "targets")
            ):
                raise ValueError("原生候选支持集或实体引用非法")
        if offset != end:
            raise ValueError(f"原生帧长度错误: {offset} != {end}")
        result.append(
            dict(
                game=game,
                root=root,
                opponent=opponent,
                commands=commands,
                microsteps=steps,
                failures=failures,
                backtracks=backtracks,
                actor=actor,
                learner=learner,
                kind=kind,
                winner=winner,
                selected=selected,
                logp=logp,
                root_input=root_input,
                node=node,
            )
        )
    if offset != len(payload):
        raise ValueError("原生批次存在尾随字节")
    return result
