"""可选 ATen C++ 发射器；复用 Torch 的构建缓存、CUDA 算子和 AMP，不另存模型。"""

import ctypes
import hashlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch


def library():
    source = Path(__file__).with_suffix(".cpp")
    identity = hashlib.sha256(source.read_bytes() + torch.__version__.encode()).hexdigest()[:16]
    root = Path(os.environ.get("TORCH_EXTENSIONS_DIR", Path.home() / ".cache/haojie/aten"))
    return source, root / identity / ("haojie_aten.pyd" if os.name == "nt" else "haojie_aten.so")


def build():
    """安装阶段编译；正常服务启动只装载与源码及 Torch 版本匹配的库。"""
    from torch.utils.cpp_extension import load

    source, path = library()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "1")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    load(
        name="haojie_aten",
        sources=[str(source)],
        build_directory=str(path.parent),
        extra_cflags=["/O2"] if os.name == "nt" else ["-O2"],
        with_cuda=False,
        is_python_module=False,
    )
    return path


@lru_cache(maxsize=1)
def operator():
    mode = os.environ.get("HAOJIE_ATEN", "auto")
    if mode == "0":
        return None
    _, path = library()
    if not path.is_file():
        if mode != "1":
            return None
        build()
    else:
        torch.ops.load_library(str(path))
    return torch.ops.haojie.packed_encode


@lru_cache(maxsize=1)
def native_io():
    if os.environ.get("HAOJIE_NATIVE_IO") == "0" or operator() is None:
        return None
    _, path = library()
    extension = ctypes.CDLL(str(path))
    extension.haojie_validate.argtypes = [ctypes.c_void_p, *([ctypes.c_int64] * 3)]
    extension.haojie_validate.restype = ctypes.c_int
    integers = ctypes.POINTER(ctypes.c_int64)
    extension.haojie_send.argtypes = [
        integers,
        integers,
        integers,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int64,
    ]
    extension.haojie_send.restype = ctypes.c_int
    return extension


def parameters(model):
    weights = [
        model.entity_projection.weight,
        model.entity_projection.bias,
        model.kind_embedding.weight,
        model.global_projection.weight,
        model.global_projection.bias,
    ]
    epsilon = []
    for block in model.blocks:
        for layer in (
            block.attention_norm,
            block.qkv,
            block.output,
            block.ffn_norm,
            block.ffn[0],
            block.ffn[2],
        ):
            weights.extend((layer.weight, layer.bias))
        epsilon.extend((block.attention_norm.eps, block.ffn_norm.eps))
    weights.extend((model.norm.weight, model.norm.bias))
    epsilon.append(model.norm.eps)
    return weights, epsilon


if __name__ == "__main__":
    print(build())
