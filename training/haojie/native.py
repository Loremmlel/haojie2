"""Rust 规则库的进程内桥；热路径张量通过连续内存传输。"""

import ctypes
import json
import os
from pathlib import Path

import numpy as np

STATE_WIDTH = 128
ACTION_WIDTH = 576
_library = None


def library():
    global _library
    if _library is None:
        root = Path(__file__).resolve().parents[2]
        name = "haojie_native.dll" if os.name == "nt" else "libhaojie_native.so"
        path = Path(os.environ.get("HAOJIE_ENGINE", root / "artifacts/native-target/release" / name))
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
        data = json.dumps({"op": op, **args}, separators=(",", ":"), ensure_ascii=False).encode()
        pointer = self.lib.hc_call(self.handle, data, len(data))
        value = json.loads(ctypes.string_at(pointer, self.lib.hc_length(self.handle)))
        if isinstance(value, dict) and "error" in value:
            raise RuntimeError(value["error"])
        return value

    def prepare(self, state, seed, count, include=None):
        args = {"id": state, "seed": seed, "count": count}
        if include is not None:
            args["include"] = include
        info = self.call("prepare", **args)
        data = np.empty(info["floats"], np.float32)
        assert self.lib.hc_tensor(self.handle, data.ctypes.data, data.size) == data.size
        split = info["rows"] * STATE_WIDTH
        return info, data[:split].reshape(-1, STATE_WIDTH), data[split:].reshape(-1, ACTION_WIDTH)

    def close(self):
        if self.handle:
            self.lib.hc_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
