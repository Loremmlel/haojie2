#pragma once
#include "runtime.h"
#include <fstream>
#include <iomanip>
#ifdef HJ_CUDA
// CUDA_VERSION 必须先于图对象声明可见，保持发行库的条件成员布局。
// clang-format off
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
// clang-format on
#endif

namespace hj::diagnostic {
// 独立算子诊断：使用检查点第一层权重和固定随机 BF16 输入，不进入采样或学习。
// 图内重复 32 次以摊薄宿主提交，报告设备区间；布局对照不改变权重与输入。
inline void gemm(Network &actor, const Settings &settings) {
#ifdef HJ_CUDA
  TORCH_CHECK(actor->parameters()[0].is_cuda(), "gemm diagnostic requires CUDA");
  torch::NoGradGuard guard;
  auto stream = c10::cuda::getStreamFromPool(true);
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  struct Event {
    cudaEvent_t value;
    Event() {
      TORCH_CHECK(cudaEventCreate(&value) == cudaSuccess, "CUDA timing event creation failed");
    }
    ~Event() {
      cudaEventDestroy(value);
    }
  };
  auto &block = *actor->blocks[0]->as<TransformerBlock>();
  std::vector<std::pair<std::string, torch::nn::Linear>> layers{
      {"qkv", block.qkv}, {"output", block.output}, {"up", block.up}, {"down", block.down}};
  // 方阵仅校准本机稠密 BF16 乘法能力，不计入网络工作量或真实根吞吐。
  for (auto size : {2048, 4096}) {
    torch::nn::Linear layer(size, size);
    layer->to(actor->parameters()[0].device(), torch::kBFloat16);
    layers.emplace_back("calibration_" + std::to_string(size), layer);
  }
  cudaDeviceProp properties;
  TORCH_CHECK(cudaGetDeviceProperties(&properties, actor->parameters()[0].get_device()) ==
              cudaSuccess);
  std::ofstream device(settings.directory / "gemm-device.json");
  device << "{\"name\":\"" << properties.name
         << "\",\"sm_count\":" << properties.multiProcessorCount
         << ",\"major\":" << properties.major << ",\"minor\":" << properties.minor
         << ",\"total_memory\":" << properties.totalGlobalMem << "}\n";
  TORCH_CHECK(bool(device), "GEMM device metadata write failed");
  std::ofstream out(settings.directory / "gemm.csv");
  out << std::setprecision(12)
      << "operation,m,n,k,backend,repeat,gpu_ms,tflops,max_abs_difference,unequal_fraction,"
         "start_unix_ms,end_unix_ms\n";
  auto utc_ms = [] {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  };
  for (auto m : {16, 32, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192}) {
    for (auto &[name, layer] : layers) {
      if (name.starts_with("calibration_") && m != layer->weight.size(0))
        continue;
      auto k = layer->weight.size(1), n = layer->weight.size(0);
      auto x = torch::randn({m, k}, layer->weight.options());
      auto reference = layer(x).clone();
      auto packed_weight = layer->weight.t().contiguous();
      for (auto layout : {"default", "weight_kn", "output_nm"}) {
        auto forward = [&]() {
          if (std::string_view(layout) == "weight_kn")
            return torch::addmm(layer->bias, x, packed_weight);
          if (std::string_view(layout) == "output_nm")
            return torch::addmm(layer->bias.unsqueeze(1), layer->weight, x.t()).t().contiguous();
          return layer(x);
        };
        T result;
        for (int i = 0; i < 4; ++i)
          result = forward();
        stream.synchronize();
        auto difference = (result - reference).abs().max().item<double>();
        auto unequal = result.ne(reference).sum().item<int64_t>() / double(result.numel());
        TORCH_CHECK(torch::allclose(result, reference, 0.016, 0.016),
                    "matrix layout changed BF16 linear beyond tolerance");
        at::cuda::CUDAGraph graph;
        graph.capture_begin();
        for (int i = 0; i < 32; ++i)
          result = forward();
        graph.capture_end();
        graph.replay();
        stream.synchronize();
        for (int repeat = 0; repeat < 3; ++repeat) {
          Event begin, end;
          auto start = utc_ms();
          TORCH_CHECK(cudaEventRecord(begin.value, stream) == cudaSuccess);
          for (int i = 0; i < 8; ++i)
            graph.replay();
          TORCH_CHECK(cudaEventRecord(end.value, stream) == cudaSuccess);
          TORCH_CHECK(cudaEventSynchronize(end.value) == cudaSuccess);
          float duration;
          TORCH_CHECK(cudaEventElapsedTime(&duration, begin.value, end.value) == cudaSuccess);
          double ms = duration / 256;
          out << name << ',' << m << ',' << n << ',' << k << ',' << layout << ',' << repeat << ','
              << ms << ',' << (2.0 * m * n * k / (ms * 1e9)) << ',' << difference << ',' << unequal
              << ',' << start << ',' << utc_ms() << '\n';
        }
      }
    }
  }
  TORCH_CHECK(bool(out), "gemm diagnostic write failed");
#else
  TORCH_CHECK(false, "gemm diagnostic requires CUDA build");
#endif
}
} // namespace hj::diagnostic
