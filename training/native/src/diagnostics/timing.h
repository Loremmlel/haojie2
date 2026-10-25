#pragma once
#include <atomic>
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <vector>
#ifdef HJ_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace hj::diagnostic {
using Clock = std::chrono::steady_clock;
inline bool enabled = false;
inline Clock::time_point origin = Clock::now();
inline double now() {
  return std::chrono::duration<double, std::milli>(Clock::now() - origin).count();
}
struct Row {
  const char *kind;
  double at, id, a, b, c, d, e, f;
};
inline std::mutex mutex;
inline std::vector<Row> rows;
// 仅诊断命令启用。保留原始观测，禁止溢出后静默抽样。
inline void record(const char *kind, double id, double a = 0, double b = 0, double c = 0,
                   double d = 0, double e = 0, double f = 0) {
  if (!enabled)
    return;
  std::lock_guard lock(mutex);
  TORCH_CHECK(rows.size() < 2000000, "diagnostic sample capacity exceeded");
  rows.push_back({kind, now(), id, a, b, c, d, e, f});
}
inline void save(const std::filesystem::path &path) {
  std::lock_guard lock(mutex);
  std::ofstream out(path);
  out << std::setprecision(12) << "kind,at_ms,id,a,b,c,d,e,f\n";
  for (auto &r : rows)
    out << r.kind << ',' << r.at << ',' << r.id << ',' << r.a << ',' << r.b << ',' << r.c << ','
        << r.d << ',' << r.e << ',' << r.f << '\n';
  TORCH_CHECK(bool(out), "diagnostic write failed");
}
inline void memory(const c10::Device &device) {
#ifdef HJ_CUDA
  if (enabled && device.is_cuda()) {
    auto stats =
        c10::cuda::CUDACachingAllocator::getDeviceStats(device.index() < 0 ? 0 : device.index());
    record("cuda_memory", 0, stats.allocated_bytes[0].peak, stats.reserved_bytes[0].peak,
           stats.allocated_bytes[0].current, stats.reserved_bytes[0].current);
  }
#endif
}
// CUDA event 测设备流区间；主机锚点给出入队延迟估计及锚点区间宽度。
// 区间含流内空隙与设备调度等待，不能作为逐内核活动并集。
struct GpuSpan {
#ifdef HJ_CUDA
  cudaEvent_t anchor = nullptr, begin = nullptr, computed = nullptr, returned = nullptr;
  double anchor_ms = 0, uncertainty_ms = 0;
  void initialize() {
    if (!enabled || anchor)
      return;
    TORCH_CHECK(cudaEventCreate(&anchor) == cudaSuccess && cudaEventCreate(&begin) == cudaSuccess &&
                    cudaEventCreate(&computed) == cudaSuccess &&
                    cudaEventCreate(&returned) == cudaSuccess,
                "diagnostic event creation failed");
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto before = now();
    TORCH_CHECK(cudaEventRecord(anchor, stream) == cudaSuccess &&
                    cudaEventSynchronize(anchor) == cudaSuccess,
                "diagnostic anchor failed");
    uncertainty_ms = now() - before;
    anchor_ms = before + uncertainty_ms / 2;
  }
  void start() {
    if (enabled) {
      initialize();
      TORCH_CHECK(cudaEventRecord(begin, c10::cuda::getCurrentCUDAStream()) == cudaSuccess,
                  "diagnostic start failed");
    }
  }
  void compute_end() {
    if (enabled)
      TORCH_CHECK(cudaEventRecord(computed, c10::cuda::getCurrentCUDAStream()) == cudaSuccess,
                  "diagnostic compute event failed");
  }
  void return_end() {
    if (enabled)
      TORCH_CHECK(cudaEventRecord(returned, c10::cuda::getCurrentCUDAStream()) == cudaSuccess,
                  "diagnostic return event failed");
  }
  void report(const char *kind, double id, double submitted, double callback, double count,
              double tokens) {
    if (!enabled)
      return;
    float offset = 0, gpu = 0, copy = 0;
    TORCH_CHECK(cudaEventElapsedTime(&offset, anchor, begin) == cudaSuccess &&
                    cudaEventElapsedTime(&gpu, begin, computed) == cudaSuccess &&
                    cudaEventElapsedTime(&copy, computed, returned) == cudaSuccess,
                "diagnostic elapsed event failed");
    record(kind, id, count, tokens, gpu, copy, anchor_ms + offset - submitted,
           callback - (anchor_ms + offset + gpu + copy));
  }
  ~GpuSpan() {
    if (anchor) {
      cudaEventDestroy(anchor);
      cudaEventDestroy(begin);
      cudaEventDestroy(computed);
      cudaEventDestroy(returned);
    }
  }
#endif
};
} // namespace hj::diagnostic
