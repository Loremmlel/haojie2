"""Rust 规则库的进程内桥；热路径张量通过连续内存传输。"""

import ctypes
import json
import os
from pathlib import Path

import numpy as np

STATE_WIDTH = 128
OPTION_WIDTH = 704
RELATION_WIDTH = 8
_library = None


def library():
    global _library
    if _library is None:
        root = Path(__file__).resolve().parents[2]
        name = "haojie_native.dll" if os.name == "nt" else "libhaojie_native.so"
        path = Path(
            os.environ.get(
                "HAOJIE_ENGINE", root / "artifacts/native-target/release" / name
            )
        )
        lib = ctypes.CDLL(str(path))
        lib.hc_new.argtypes = [ctypes.c_uint32]
        lib.hc_new.restype = ctypes.c_void_p
        lib.hc_call.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.hc_call.restype = ctypes.c_void_p
        lib.hc_length.argtypes = [ctypes.c_void_p]
        lib.hc_length.restype = ctypes.c_size_t
        lib.hc_tensor.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.hc_tensor.restype = ctypes.c_size_t
        lib.hc_close.argtypes = [ctypes.c_void_p]
        _library = lib
    return _library


class Environment:
    def __init__(self, seed):
        self.lib = library()
        self.handle = self.lib.hc_new(seed)
        if not self.handle:
            raise RuntimeError("Rust 环境初始化失败")

    def call(self, op, **args):
        data = json.dumps(
            {"op": op, **args}, separators=(",", ":"), ensure_ascii=False
        ).encode()
        pointer = self.lib.hc_call(self.handle, data, len(data))
        value = json.loads(ctypes.string_at(pointer, self.lib.hc_length(self.handle)))
        if isinstance(value, dict) and "error" in value:
            raise RuntimeError(value["error"])
        return value

    def tensor(self, count):
        data = np.empty(count, np.float32)
        assert self.lib.hc_tensor(self.handle, data.ctypes.data, data.size) == data.size
        return data

    def encode(self, position=0):
        info = self.call("encode", id=position)
        data = self.tensor(info["floats"])
        n, m = info["entities"], info["records"]
        first = n * STATE_WIDTH
        second = first + m * STATE_WIDTH
        third = second + m
        return info, (
            data[:first].reshape(n, STATE_WIDTH),
            data[first:second].reshape(m, STATE_WIDTH),
            data[second:third],
            data[third:].reshape(n, n, RELATION_WIDTH),
        )

    def decode(self, position, prefix=0):
        front = self.call("decode", id=position, prefix=prefix)
        return front, None if front["leaf"] else self.tensor(front["floats"]).reshape(
            front["count"], OPTION_WIDTH
        )

    def close(self):
        if self.handle:
            self.lib.hc_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
