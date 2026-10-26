#pragma once
#include "runtime.h"
#include <condition_variable>
#include <future>
#include <thread>

namespace hj {
// 冻结完整网络供当前搜索及历史对手共用；没有优化器，整次搜索内版本不变。
struct FrozenPolicy {
  Network model;
  uint64_t version;
  CpuSnapshot cpu;
  StorageBytes storage;
  FrozenPolicy(Network &source, uint64_t version);
};
using Frozen = std::shared_ptr<FrozenPolicy>;
struct SearchPrediction {
  std::vector<uint16_t> mass;
  double value = 0;
};
// 集中合批编码公开根，策略头回到各调用 lane 并行执行。
class SearchService {
  struct Request {
    Frozen policy;
    std::shared_ptr<Node> node;
    std::promise<void> result;
  };
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Request> requests;
  bool closing = false;
  uint64_t revision = 0;
  uint64_t batch_limit;
  Network model{nullptr};
  std::unique_ptr<InferenceEngine> encoder;
  std::atomic<uint64_t> encoded_nodes{0}, encoded_batches{0}, encoding_ns{0}, decoding_ns{0};
  std::atomic<uint64_t> waiting{0}, running{0};
  std::thread thread;

public:
  explicit SearchService(const torch::Device &, uint64_t width = DefaultWidth,
                         uint64_t max_roots = 64);
  ~SearchService();
  SearchPrediction predict(const Frozen &, const std::shared_ptr<Node> &, bool with_value = true);
  uint64_t queued_roots() const {
    return waiting.load();
  }
  uint64_t running_roots() const {
    return running.load();
  }
  std::atomic<uint64_t> queries{0};
  InferenceStats stats() const {
    return encoder->stats();
  }
  RootStats progress() const {
    return {encoded_nodes.load(), encoded_batches.load(), encoding_ns.load() / 1e9};
  }
  double policy_seconds() const {
    return decoding_ns.load() / 1e9;
  }
};
struct SearchResult {
  std::vector<uint16_t> mass, target;
  uint64_t version = 0, simulations = 0, expanded = 0, depth = 0, opponent_nodes = 0;
  bool parameter_limit = false;
  bool empty_branch = false;
};
struct SearchPlan {
  uint64_t root = 0;
  uint32_t actor = 0;
  Frozen policy;
  std::vector<std::pair<std::vector<uint32_t>, uint64_t>> paths;
  StorageBytes storage;
  void clear() {
    root = 0;
    policy.reset();
    decltype(paths){}.swap(paths);
    storage.set(0);
  }
};
SearchResult search_actor(void *actor, uint64_t game, uint32_t perspective,
                          const std::shared_ptr<Node> &, const Frozen &, SearchService &,
                          uint64_t simulations, uint64_t depth,
                          const std::function<bool()> &cancelled, SearchPlan *plan = nullptr);
void search_checks(const torch::Device &);
} // namespace hj
