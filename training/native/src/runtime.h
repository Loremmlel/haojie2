#pragma once
#include "model.h"
#include <filesystem>
#include <functional>
#include <optional>
#include <set>
#include <string>

namespace hj {
struct Settings {
  std::filesystem::path directory, resume, profile, replay;
  std::string diagnostic;
  double warmup = 5;
  uint64_t root_slots = 2;
  bool defer_learning = false;
  std::string device = "cuda";
  uint64_t concurrency = 12, workers = 8, seed = 1, chunk = 16, batch = 512, inference_batch = 64,
           width = DefaultWidth, publish = 8;
  uint64_t memory_mb = 4096, updates = 0, samples = 0;
  uint64_t max_game_steps = 262144;
  uint64_t search_simulations = 0, search_depth = 64;
  uint64_t history_size = 8, history_interval = 64, history_percent = 20;
  uint64_t threads = 1;
  uint64_t workset_games = 0, game_samples = 256;
  double seconds = 0, save_seconds = 300;
  bool shrine = false;
  std::set<std::string> supplied;
  bool terminal_training() const {
    return !workset_games && (diagnostic.empty() || diagnostic == "mixed");
  }
};
// 槽持有全部输入、缓存来源与回送内存，完成通知不销毁张量或调用 CUDA。
struct InferenceFlight {
  std::atomic<bool> complete{false};
  uint64_t version = 0, revision = 0;
  bool graphed = false;
  double seconds = 0;
  double callback_ms = 0;
  std::vector<std::shared_ptr<Node>> nodes;
};
struct InferenceStats {
  uint64_t replays = 0, fallbacks = 0, input_bytes = 0, capacity_bytes = 0;
  uint64_t tokens = 0, token_capacity = 0;
};
struct RootStats {
  uint64_t nodes = 0, batches = 0;
  double seconds = 0;
};
class InferenceEngine {
  struct Impl;
  std::unique_ptr<Impl> impl;

public:
  InferenceEngine(Network &, const torch::Device &, std::function<void()>, size_t slots = 2,
                  uint64_t max_roots = 128);
  ~InferenceEngine();
  std::shared_ptr<InferenceFlight> submit(size_t slot, std::vector<std::shared_ptr<Node>>,
                                          uint64_t version);
  void finish(const std::shared_ptr<InferenceFlight> &);
  void wait_reads();
  InferenceStats stats() const;
};
void run(const Settings &);
void evaluate(const std::filesystem::path &, const std::string &,
              std::optional<uint64_t> search_simulations = {});
void evaluation_checks(const torch::Device &);
std::string describe_configuration(Settings);
void runtime_checks();
void inference_checks(const torch::Device &);
void atomic_text(const std::filesystem::path &, const std::string &);
} // namespace hj
