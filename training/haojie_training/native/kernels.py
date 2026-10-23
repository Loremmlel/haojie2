"""正式宽度的只读推理融合核；使用 Torch 已带的 NVRTC，编译失败沿用 ATen。"""

import ctypes
from functools import lru_cache

import torch

SOURCE = r"""
__device__ float row_sum(float value) {
  __shared__ float partial[4], total;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  if (!lane) partial[warp] = value;
  __syncthreads();
  if (!warp) {
    value = lane < 4 ? partial[lane] : 0.0f;
    for (int offset = 16; offset; offset >>= 1)
      value += __shfl_down_sync(0xffffffff, value, offset);
    if (!lane) total = value;
  }
  __syncthreads();
  return total;
}

extern "C" __global__ void residual_norm(
    const unsigned short* residual, const unsigned short* added,
    const unsigned short* weight, const unsigned short* bias,
    unsigned short* sum, unsigned short* normalized, float epsilon) {
  const int column = threadIdx.x;
  const long long base = static_cast<long long>(blockIdx.x) * 384 + column;
  float value[3];
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    const auto index = base + i * 128;
    const float added_value = __uint_as_float(unsigned(residual[index]) << 16)
                            + __uint_as_float(unsigned(added[index]) << 16);
    const unsigned bits = __float_as_uint(added_value);
    sum[index] = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
    value[i] = __uint_as_float(unsigned(sum[index]) << 16);
  }
  const float mean = row_sum(value[0] + value[1] + value[2]) / 384.0f;
  float variance = 0.0f;
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    value[i] -= mean;
    variance += value[i] * value[i];
  }
  const float inverse = rsqrtf(row_sum(variance) / 384.0f + epsilon);
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int channel = column + i * 128;
    const float output = value[i] * inverse * __uint_as_float(unsigned(weight[channel]) << 16)
                       + __uint_as_float(unsigned(bias[channel]) << 16);
    const unsigned bits = __float_as_uint(output);
    normalized[base + i * 128] = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
  }
}
"""


@lru_cache(maxsize=8)
def residual_norm(device):
    """每设备持有一个模块至进程退出；编译不依赖 CUDA toolkit 或 nvcc。"""
    from torch.cuda._utils import (
        _cuda_load_module,
        _get_gpu_rtc_library,
        _get_gpu_runtime_library,
    )

    program = ctypes.c_void_p()
    compiler = None
    try:
        with torch.cuda.device(device):
            # Torch 的 _compile_kernel 仍查询本机 toolkit；该内核只需 NVRTC 自带声明。
            compiler = _get_gpu_rtc_library()

            def check(code):
                if code:
                    size = ctypes.c_size_t()
                    compiler.nvrtcGetProgramLogSize(program, ctypes.byref(size))
                    log = ctypes.create_string_buffer(max(1, size.value))
                    compiler.nvrtcGetProgramLog(program, log)
                    raise RuntimeError(f"NVRTC {code}: {log.value.decode(errors='replace')}")

            check(
                compiler.nvrtcCreateProgram(
                    ctypes.byref(program), SOURCE.encode(), b"haojie_norm.cu", 0, None, None
                )
            )
            major, minor = torch.cuda.get_device_capability(device)
            options = (ctypes.c_char_p * 2)(
                f"--gpu-architecture=compute_{major}{minor}".encode(), b"--std=c++17"
            )
            check(compiler.nvrtcCompileProgram(program, len(options), options))
            size = ctypes.c_size_t()
            check(compiler.nvrtcGetPTXSize(program, ctypes.byref(size)))
            binary = ctypes.create_string_buffer(size.value)
            check(compiler.nvrtcGetPTX(program, binary))
            kernel = _cuda_load_module(binary.raw, ["residual_norm"])["residual_norm"]
            driver = _get_gpu_runtime_library()
            launch = ctypes.cast(driver.cuLaunchKernel, ctypes.c_void_p).value
            return (kernel.func.value, launch, kernel, driver), None
    except (OSError, RuntimeError) as error:
        return None, str(error)
    finally:
        if compiler is not None and program.value:
            compiler.nvrtcDestroyProgram(ctypes.byref(program))
