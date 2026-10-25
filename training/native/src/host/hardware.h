#pragma once
#include "runtime.h"
#include <ATen/Parallel.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sched.h>
#include <unistd.h>
#endif
#ifdef HJ_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#endif

namespace hj {
struct Hardware {
  uint64_t logical = 1, cores = 1, ram_available = 0;
  uint64_t gpu_total = 0, gpu_free = 0;
  int sm = 0, major = 0, minor = 0;
  std::string device = "cpu", name = "CPU";
  double cpu_capacity = 0;
};
// cgroup 当前层和父层均可限额；容器内可见核数及 MemAvailable 不是实际配额。
inline void cgroup_limits(Hardware &h, const std::filesystem::path &root,
                          std::filesystem::path current, bool unified) {
  double cpus = h.cpu_capacity > 0 ? h.cpu_capacity : double(h.logical);
  for (;;) {
    std::string quota;
    double period = 0;
    if (unified) {
      std::ifstream(current / "cpu.max") >> quota >> period;
    } else {
      std::ifstream(current / "cpu.cfs_quota_us") >> quota;
      std::ifstream(current / "cpu.cfs_period_us") >> period;
    }
    if (!quota.empty() && quota != "max" && period > 0) {
      auto value = std::stod(quota);
      if (value > 0)
        cpus = std::min(cpus, value / period);
    }
    std::string limit;
    uint64_t used = 0;
    std::ifstream(current / (unified ? "memory.max" : "memory.limit_in_bytes")) >> limit;
    bool has_usage = bool(
        std::ifstream(current / (unified ? "memory.current" : "memory.usage_in_bytes")) >> used);
    if (!limit.empty() && limit != "max" && has_usage) {
      uint64_t maximum = std::stoull(limit);
      uint64_t reclaimable = 0, dirty = 0, writeback = 0, value;
      std::string key;
      std::ifstream statistics(current / "memory.stat");
      while (statistics >> key >> value) {
        if (key == (unified ? "inactive_file" : "total_inactive_file"))
          reclaimable = value;
        if (key == (unified ? "file_dirty" : "total_dirty"))
          dirty = value;
        if (key == (unified ? "file_writeback" : "total_writeback"))
          writeback = value;
      }
      auto clean = reclaimable > dirty + writeback ? reclaimable - dirty - writeback : 0;
      auto occupied = used - std::min(used, clean);
      h.ram_available = std::min<uint64_t>(h.ram_available, maximum - std::min(maximum, occupied));
    }
    if (current == root || current == current.parent_path())
      break;
    current = current.parent_path();
  }
  h.cpu_capacity = cpus;
  auto whole = std::max<uint64_t>(1, uint64_t(cpus));
  h.logical = std::min(h.logical, whole);
  h.cores = std::min(h.cores, whole);
}
inline void container_limits(Hardware &h) {
  std::ifstream groups("/proc/self/cgroup");
  std::string line;
  while (std::getline(groups, line)) {
    auto first = line.find(':'), last = line.find(':', first + 1);
    if (first == std::string::npos || last == std::string::npos)
      continue;
    auto controllers = line.substr(first + 1, last - first - 1);
    auto relative = std::filesystem::path(line.substr(last + 1)).relative_path();
    if (controllers.empty()) {
      auto root = std::filesystem::path("/sys/fs/cgroup");
      auto current = (root / relative).lexically_normal();
      // cgroup namespace 可把容器当前组直接挂载为根，/proc 中的祖先路径可能不可见。
      if (!std::filesystem::exists(current / "cgroup.procs"))
        current = root;
      cgroup_limits(h, root, current, true);
    } else {
      for (auto name : {"cpu", "memory"}) {
        if (("," + controllers + ",").find("," + std::string(name) + ",") == std::string::npos)
          continue;
        auto root = std::filesystem::path("/sys/fs/cgroup") / name;
        if (!std::filesystem::exists(root))
          root = root.parent_path() / controllers;
        auto current = (root / relative).lexically_normal();
        if (!std::filesystem::exists(current / "cgroup.procs"))
          current = root;
        cgroup_limits(h, root, current, false);
      }
    }
  }
}
inline Hardware hardware(const std::string &requested) {
  Hardware h;
  h.logical = std::max(1u, std::thread::hardware_concurrency());
#ifdef _WIN32
  DWORD_PTR affinity = 0, system = 0;
  if (GetProcessAffinityMask(GetCurrentProcess(), &affinity, &system))
    h.logical = std::popcount(uint64_t(affinity));
  DWORD size = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &size);
  std::vector<unsigned char> info(size);
  uint64_t cores = 0;
  if (size && GetLogicalProcessorInformationEx(
                  RelationProcessorCore,
                  reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(info.data()), &size))
    for (size_t at = 0; at < size;) {
      auto p = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(info.data() + at);
      if (!affinity || (p->Processor.GroupMask[0].Mask & affinity))
        ++cores;
      at += p->Size;
    }
  h.cores = cores ? cores : h.logical;
  MEMORYSTATUSEX memory{};
  memory.dwLength = sizeof(memory);
  TORCH_CHECK(GlobalMemoryStatusEx(&memory), "cannot query available host memory");
  h.ram_available = memory.ullAvailPhys;
#else
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  bool bound = sched_getaffinity(0, sizeof(affinity), &affinity) == 0;
  if (bound)
    h.logical = CPU_COUNT(&affinity);
  std::set<std::pair<int, int>> cores;
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (bound && !CPU_ISSET(i, &affinity))
      continue;
    auto base = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/";
    int core, package;
    if (std::ifstream(base + "core_id") >> core &&
        std::ifstream(base + "physical_package_id") >> package)
      cores.emplace(package, core);
  }
  h.cores = cores.empty() ? h.logical : cores.size();
  std::ifstream memory("/proc/meminfo");
  std::string line;
  while (std::getline(memory, line)) {
    std::istringstream in(line);
    std::string key;
    uint64_t kib;
    if (in >> key >> kib && key == "MemAvailable:")
      h.ram_available = kib * 1024;
  }
  container_limits(h);
#endif
  h.cores = std::clamp<uint64_t>(h.cores, 1, h.logical);
  TORCH_CHECK(h.ram_available >= (uint64_t(512) << 20), "insufficient available host memory");
  if (requested == "cpu")
    return h;
#ifdef HJ_CUDA
  int count = 0;
  auto status = cudaGetDeviceCount(&count);
  if (requested == "auto" && (status != cudaSuccess || count == 0)) {
    cudaGetLastError();
    return h;
  }
  TORCH_CHECK(status == cudaSuccess && count,
              "CUDA device unavailable; use --device cpu explicitly");
  torch::Device device(requested == "auto" ? "cuda:0" : requested);
  TORCH_CHECK(device.is_cuda(), "device must be auto, cpu, or cuda[:index]");
  c10::cuda::CUDAGuard guard(device);
  cudaDeviceProp properties{};
  TORCH_CHECK(cudaGetDeviceProperties(&properties, device.index() < 0 ? 0 : device.index()) ==
                  cudaSuccess,
              "cannot query CUDA device");
  TORCH_CHECK(properties.major >= 8,
              "native BF16 training requires Ampere or newer; T4/P100 are unsupported");
  size_t free = 0, total = 0;
  TORCH_CHECK(cudaMemGetInfo(&free, &total) == cudaSuccess, "cannot query available GPU memory");
  h.device = device.str();
  h.name = properties.name;
  h.major = properties.major;
  h.minor = properties.minor;
  h.sm = properties.multiProcessorCount;
  h.gpu_total = properties.totalGlobalMem;
  h.gpu_free = free;
#else
  TORCH_CHECK(requested == "auto", "this build has no CUDA support");
#endif
  return h;
}
// 依据可用资源选择起始配置，不执行试训；显式参数优先，恢复不改学习语义。
inline void select_configuration(Settings &s, const Hardware &h) {
  auto set = [&](const char *option, uint64_t &value, uint64_t selected) {
    if (!s.supplied.count(option))
      value = selected;
  };
  s.device = h.device;
  uint64_t batch = 32, limit = 4, concurrency = h.cores;
  auto workers = std::min<uint64_t>(h.cores, 16);
  set("--workers", s.workers, workers);
  if (h.gpu_total) {
    auto gib = std::min(h.gpu_total, h.gpu_free) / double(uint64_t(1) << 30);
    batch = gib >= 18 ? 512 : gib >= 10 ? 256 : gib >= 6 ? 128 : gib >= 4 ? 64 : 32;
    limit = gib >= 18 ? 64 : gib >= 10 ? 32 : gib >= 6 ? 16 : 8;
    auto compute = std::bit_ceil(uint64_t(std::max(1, h.sm / 2)));
    concurrency = std::min({s.workers * 4, compute, limit});
  }
  // 恢复不自动停放旧局，避免其未决后继阻塞严格滞后排空。
  if (s.resume.empty())
    set("--concurrency", s.concurrency, concurrency);
  set("--inference-batch", s.inference_batch,
      std::max<uint64_t>(1, std::min(limit, s.concurrency)));
  if (s.resume.empty())
    set("--batch", s.batch, batch);
  set("--memory-mb", s.memory_mb,
      std::max<uint64_t>(h.ram_available / (3 * 1024 * 1024), 128));
  set("--threads", s.threads, h.gpu_total ? 1 : std::min<uint64_t>(h.cores, 8));
}
inline std::string configuration_json(const Settings &s, const Hardware &h) {
  std::ostringstream out;
  out << "{\"device\":" << std::quoted(s.device) << ",\"gpu\":" << std::quoted(h.name)
      << ",\"logical_cpus\":" << h.logical << ",\"physical_cores\":" << h.cores
      << ",\"host_available_bytes\":" << h.ram_available << ",\"gpu_total_bytes\":" << h.gpu_total
      << ",\"cpu_capacity\":" << (h.cpu_capacity > 0 ? h.cpu_capacity : double(h.logical))
      << ",\"gpu_free_bytes\":" << h.gpu_free << ",\"sm_count\":" << h.sm
      << ",\"compute_major\":" << h.major << ",\"compute_minor\":" << h.minor
      << ",\"workers\":" << s.workers << ",\"concurrency\":" << s.concurrency
      << ",\"batch\":" << s.batch << ",\"inference_batch\":" << s.inference_batch
      << ",\"threads\":" << s.threads << ",\"memory_mb\":" << s.memory_mb
      << ",\"max_game_steps\":" << s.max_game_steps
      << ",\"training_method\":\"terminal-vtrace\",\"batch_unit\":\"memory-microbatch\""
      << ",\"checkpoint_batch_preserved\":" << (s.resume.empty() ? "false" : "true") << "}\n";
  return out.str();
}
inline void hardware_checks() {
  {
    namespace fs = std::filesystem;
    auto root = fs::temp_directory_path() /
                ("haojie-cgroup-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TORCH_CHECK(!fs::exists(root));
    fs::create_directories(root / "child");
    auto write = [&](const fs::path &path, const char *value) { std::ofstream(path) << value; };
    write(root / "cpu.max", "150000 100000");
    write(root / "child/cpu.max", "250000 100000");
    write(root / "memory.max", "1000");
    write(root / "memory.current", "800");
    write(root / "memory.stat", "inactive_file 300\nfile_dirty 40\nfile_writeback 10\n");
    Hardware limited;
    limited.logical = 12;
    limited.cores = 6;
    limited.ram_available = 10000;
    cgroup_limits(limited, root, root / "child", true);
    TORCH_CHECK(limited.cpu_capacity == 1.5 && limited.cores == 1 && limited.ram_available == 450,
                "parent cgroup quota or clean reclaimable memory ignored");
    write(root / "cpu.cfs_quota_us", "-1");
    write(root / "cpu.cfs_period_us", "100000");
    write(root / "child/cpu.cfs_quota_us", "400000");
    write(root / "child/cpu.cfs_period_us", "100000");
    write(root / "child/memory.limit_in_bytes", "900");
    write(root / "child/memory.usage_in_bytes", "950");
    write(root / "child/memory.stat", "total_inactive_file 100\ntotal_dirty 20\n");
    limited = Hardware{};
    limited.logical = 12;
    limited.cores = 6;
    limited.ram_available = 10000;
    cgroup_limits(limited, root, root / "child", false);
    TORCH_CHECK(limited.cores == 4 && limited.ram_available == 30, "v1 cgroup quota ignored");
    TORCH_CHECK(fs::equivalent(root.parent_path(), fs::temp_directory_path()));
    fs::remove_all(root);
  }
  Hardware laptop{
      16, 8, uint64_t(12) << 30, uint64_t(6) << 30, uint64_t(5) << 30, 30, 8, 6, "cuda:0", "test"};
  Settings s;
  select_configuration(s, laptop);
  TORCH_CHECK(s.batch == 64 && s.concurrency == 8 && s.workers == 8 && s.inference_batch == 8);
  auto server = laptop;
  server.gpu_total = server.gpu_free = uint64_t(24) << 30;
  server.sm = 60;
  select_configuration(s, server);
  TORCH_CHECK(s.batch == 512 && s.concurrency == 32 && s.inference_batch == 32);
  Settings limited;
  limited.workers = 2;
  limited.supplied.insert("--workers");
  select_configuration(limited, server);
  TORCH_CHECK(limited.concurrency == 8 && limited.workers == 2);
  server.gpu_free = uint64_t(5) << 30;
  select_configuration(s, server);
  TORCH_CHECK(s.batch == 64, "configuration ignored occupied GPU memory");
  s.resume = "checkpoint.pt";
  s.batch = 512;
  s.concurrency = 0;
  s.workers = 3;
  s.supplied.insert("--workers");
  select_configuration(s, laptop);
  TORCH_CHECK(s.batch == 512 && s.concurrency == 0 && s.workers == 3,
              "automatic configuration changed resume semantics or explicit override");
  s.concurrency = 17;
  select_configuration(s, laptop);
  TORCH_CHECK(s.concurrency == 17, "automatic resume parked unfinished games");
  Hardware cpu;
  cpu.ram_available = uint64_t(2) << 30;
  Settings single;
  select_configuration(single, cpu);
  TORCH_CHECK(single.batch == 32 && single.workers == 1 && single.concurrency == 1 &&
              single.threads == 1);
  cpu.ram_available = uint64_t(54) << 30;
  select_configuration(single, cpu);
  TORCH_CHECK(single.memory_mb == 18432, "large-memory host retained the old 8 GiB cap");
  single.memory_mb = 24576;
  single.supplied.insert("--memory-mb");
  select_configuration(single, cpu);
  TORCH_CHECK(single.memory_mb == 24576, "explicit memory budget was overwritten");
}
} // namespace hj
