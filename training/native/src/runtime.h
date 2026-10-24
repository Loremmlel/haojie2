#pragma once
#include "model.h"
#include <filesystem>
#include <set>
#include <string>

namespace hj {
struct Settings {
  std::filesystem::path directory, resume;
  std::string device = "cuda";
  uint64_t concurrency = 64, workers = 8, seed = 1, chunk = 16, batch = 512, inference_batch = 256,
           width = 128, publish = 8;
  uint64_t memory_mb = 4096, updates = 0, samples = 0;
  double seconds = 0, save_seconds = 300;
  bool shrine = false;
  std::set<std::string> supplied;
};
void run(const Settings &);
void runtime_checks();
void atomic_text(const std::filesystem::path &, const std::string &);
} // namespace hj
