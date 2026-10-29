"""单进程原生线程池桥；神经请求借用 Rust 连续数组，记录转为 Python 所有。"""

import copy
import ctypes as C
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import cached_property
import json
import threading

import numpy as np
import numpy.random._generator as generator_module

from .native import library


@dataclass
class Request:
    op: str
    context: object
    arrays: tuple


def search_values(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def rng_transport(state, to_native):
    state = copy.deepcopy(state)
    for key in ("state", "inc"):
        state["state"][key] = (str if to_native else int)(state["state"][key])
    return state


class ArrayView(C.Structure):
    _fields_ = [
        ("data", C.POINTER(C.c_float)),
        ("length", C.c_size_t),
        ("ndim", C.c_size_t),
        ("shape", C.c_size_t * 3),
    ]

    def array(self, copy_array=False):
        shape = tuple(self.shape[: self.ndim])
        value = (
            np.ctypeslib.as_array(self.data, shape=(self.length,))
            if self.length
            else np.empty(0, np.float32)
        )
        value = value.reshape(shape)
        if copy_array:
            return value.copy()
        # 借用视图只读使用；PyTorch 的 from_numpy 不支持 NumPy 只读标记。
        return value


class RequestView(C.Structure):
    _fields_ = [
        ("id", C.c_uint64),
        ("lane", C.c_uint64),
        ("game_id", C.c_uint64),
        ("step", C.c_uint64),
        ("context", C.c_uint64),
        ("model", C.c_int64),
        ("kind", C.c_uint32),
        ("array_count", C.c_uint32),
        ("key", C.c_void_p),
        ("key_len", C.c_size_t),
        ("arrays", ArrayView * 4),
    ]


class ReplyView(C.Structure):
    _fields_ = [
        ("id", C.c_uint64),
        ("data", C.POINTER(C.c_uint16)),
        ("length", C.c_size_t),
        ("version", C.c_int64),
    ]


class UFuncInfo(C.Structure):
    _fields_ = [
        ("function", C.c_void_p),
        ("context", C.c_void_p),
        ("auxdata", C.c_void_p),
        ("requires_pyapi", C.c_ubyte),
        ("no_fp", C.c_ubyte),
    ]


@dataclass
class PoolRequest(Request):
    id: int
    lane: int
    game_id: int
    step: int
    model: int
    _key_pointer: int
    _key_length: int

    @cached_property
    def key(self):
        return (
            C.string_at(self._key_pointer, self._key_length).decode()
            if self._key_length
            else ""
        )


class RequestBatch:
    def __init__(self, pool, count):
        self.pool = pool
        self.requests = []
        for index in range(count):
            view = RequestView()
            pool.check(pool.lib.hc_pool_request_view(pool.handle, index, C.byref(view)))
            self.requests.append(
                PoolRequest(
                    "encode" if view.kind == 0 else "decode",
                    view.context,
                    tuple(a.array() for a in view.arrays[: view.array_count]),
                    view.id,
                    view.lane,
                    view.game_id,
                    view.step,
                    view.model,
                    view.key,
                    view.key_len,
                )
            )

    def reply(self, outputs, versions):
        if len(outputs) != len(self.requests) or len(versions) != len(outputs):
            raise ValueError("神经回复数量与请求不一致")
        buffers = [
            np.ascontiguousarray(value, dtype=np.uint16).reshape(-1)
            for value in outputs
        ]
        views = (ReplyView * len(buffers))(
            *[
                ReplyView(
                    request.id,
                    value.ctypes.data_as(C.POINTER(C.c_uint16)),
                    value.size,
                    version,
                )
                for request, value, version in zip(self.requests, buffers, versions)
            ]
        )
        self.pool.check(
            self.pool.lib.hc_pool_reply(self.pool.handle, views, len(views))
        )

    def close(self):
        if self.pool is not None:
            self.pool.check(self.pool.lib.hc_pool_release_requests(self.pool.handle))
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NativePool:
    def __init__(self, config, audit=False):
        # ufunc capsule 是版本相关 ABI，升级 NumPy 时须重新完成搜索差分。
        if np.__version__ != "2.5.3":
            raise RuntimeError(
                "原生搜索需要 NumPy 2.5.3，请按 training/requirements.txt 安装"
            )
        self.lib = library()
        self.lock = threading.Lock()
        self.distributions = C.CDLL(generator_module.__file__)
        # 复用原 NumPy exp 内核，保留 f32 舍入；capsule 活到全部 Rust 线程退出。
        _, self.exp_capsule = np.exp._resolve_dtypes_and_context(
            (np.dtype("float32"), np.dtype("float32"))
        )
        np.exp._get_strided_loop(self.exp_capsule, fixed_strides=(4, 4))
        capsule_pointer = C.pythonapi.PyCapsule_GetPointer
        capsule_pointer.argtypes = [C.py_object, C.c_char_p]
        capsule_pointer.restype = C.c_void_p
        exp = UFuncInfo.from_address(
            capsule_pointer(self.exp_capsule, b"numpy_1.24_ufunc_call_info")
        )
        if exp.requires_pyapi:
            raise RuntimeError("NumPy f32 exp 内核需要 GIL，无法用于原生工作线程")
        signatures = {
            "new": ([C.c_char_p, C.c_size_t], C.c_void_p),
            "control": ([C.c_void_p, C.c_char_p, C.c_size_t], C.c_void_p),
            "response_len": ([C.c_void_p], C.c_size_t),
            "take_requests": ([C.c_void_p, C.c_size_t, C.c_uint64], C.c_size_t),
            "request_count": ([C.c_void_p], C.c_size_t),
            "request_view": (
                [C.c_void_p, C.c_size_t, C.POINTER(RequestView)],
                C.c_int32,
            ),
            "reply": ([C.c_void_p, C.POINTER(ReplyView), C.c_size_t], C.c_int32),
            "release_requests": ([C.c_void_p], C.c_int32),
            "take_records": ([C.c_void_p, C.c_size_t, C.c_uint64], C.c_size_t),
            "records_meta": ([C.c_void_p], C.c_void_p),
            "records_len": ([C.c_void_p], C.c_size_t),
            "record_array": ([C.c_void_p, C.c_size_t, C.POINTER(ArrayView)], C.c_int32),
            "ack_records": ([C.c_void_p, C.POINTER(C.c_uint64), C.c_size_t], C.c_int32),
            "release_records": ([C.c_void_p], None),
            "wake": ([C.c_void_p], None),
            "close": ([C.c_void_p], None),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, "hc_pool_" + name)
            function.argtypes, function.restype = args, result
        values = asdict(config) if not isinstance(config, dict) else dict(config)
        values.update(
            audit=audit,
            exp_address=exp.function,
            exp_context=exp.context,
            exp_auxdata=exp.auxdata or 0,
            gamma_address=C.cast(
                self.distributions.random_standard_gamma, C.c_void_p
            ).value,
            bounded_address=C.cast(
                self.distributions.random_bounded_uint64, C.c_void_p
            ).value,
        )
        data = json.dumps(values, separators=(",", ":")).encode()
        self.handle = self.lib.hc_pool_new(data, len(data))
        if not self.handle:
            raise RuntimeError("原生线程池初始化失败")

    def control(self, op, **args):
        data = json.dumps(
            {"op": op, **args}, separators=(",", ":"), ensure_ascii=False
        ).encode()
        with self.lock:
            pointer = self.lib.hc_pool_control(self.handle, data, len(data))
            result = json.loads(
                C.string_at(pointer, self.lib.hc_pool_response_len(self.handle))
            )
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def check(self, result):
        if result != 1:
            status = self.control("status")
            raise RuntimeError(f"原生线程池操作失败: {status}")

    def assign(self, lane, game, restored=None, rng=None):
        # 回放缓冲只属于 Python；不得复制数组或将其送入原生 JSON 控制协议。
        previous = (
            copy.deepcopy({k: v for k, v in restored.items() if k != "buffer"})
            if restored
            else None
        )
        rng = (
            rng
            if rng is not None
            else (previous or {}).get(
                "rng",
                np.random.default_rng(game["seed"] ^ 0xA53113).bit_generator.state,
            )
        )
        rng = rng_transport(rng, True)
        if previous is not None:
            previous["rng"] = rng
        return self.control("assign", lane=lane, game=game, rng=rng, restored=previous)

    def take_requests(self, max_batch=32, wait_ms=2, profile=None):
        with profile.stage("bridge.request_wait") if profile else nullcontext():
            count = self.lib.hc_pool_take_requests(self.handle, max_batch, wait_ms)
        if count == C.c_size_t(-1).value:
            self.check(0)
        if not count:
            return None
        with profile.stage("bridge.request_views") if profile else nullcontext():
            return RequestBatch(self, count)

    def take_records(self, max_records=64, wait_ms=0):
        count = self.lib.hc_pool_take_records(self.handle, max_records, wait_ms)
        if count == C.c_size_t(-1).value:
            self.check(0)
        if not count:
            return []
        try:
            records = json.loads(
                C.string_at(
                    self.lib.hc_pool_records_meta(self.handle),
                    self.lib.hc_pool_records_len(self.handle),
                )
            )

            def array(index):
                view = ArrayView()
                self.check(
                    self.lib.hc_pool_record_array(self.handle, index, C.byref(view))
                )
                return view.array(copy_array=True)

            for record in records:
                sample = record.get("sample")
                if sample:
                    sample["bundle"] = tuple(array(i) for i in sample["bundle"])
                    sample["decisions"] = [
                        (array(d["options"]), array(d["target"]), d["weight"])
                        for d in sample["decisions"]
                    ]
                audit = record.get("audit")
                if audit and audit.get("rng_after"):
                    audit["rng_after"] = rng_transport(audit["rng_after"], False)
            return records
        finally:
            self.lib.hc_pool_release_records(self.handle)

    def ack_records(self, lanes):
        if lanes:
            values = (C.c_uint64 * len(lanes))(*lanes)
            self.check(self.lib.hc_pool_ack_records(self.handle, values, len(lanes)))

    def snapshot(self):
        result = self.control("snapshot")
        lanes = result["lanes"] if isinstance(result, dict) else result
        for lane in lanes:
            lane["rng"] = rng_transport(lane["rng"], False)
        return lanes

    def status(self):
        return self.control("status")

    def active_versions(self):
        return self.control("active_versions")

    def pause(self):
        return self.control("pause")

    def barrier(self):
        return self.control("barrier")

    def resume(self):
        return self.control("resume")

    def wake(self):
        self.lib.hc_pool_wake(self.handle)

    def close(self):
        if self.handle:
            self.lib.hc_pool_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
