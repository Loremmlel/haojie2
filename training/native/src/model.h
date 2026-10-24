#pragma once
#include "data.h"
#include <atomic>
#include <mutex>

namespace hj {
struct NetworkImpl : torch::nn::Module {
  int64_t width;
  std::atomic<uint64_t> revision{0};
  uint64_t encoded_roots = 0, cache_hits = 0;
  torch::nn::Linear entity{nullptr}, action{nullptr}, globals{nullptr}, mix{nullptr},
      context{nullptr}, policy{nullptr}, value{nullptr};
  torch::nn::Embedding kinds{nullptr};
  explicit NetworkImpl(int64_t width = 128);
  bool cache_valid(const Root &r) const {
    return r.cached_owner == this && r.cached_revision == revision.load() &&
           r.cached_tokens.defined();
  }
  // 根、节点、候选均为不补零的张量表，同一根/候选在一次前向内只投影一次。
  std::pair<T, T> forward(const std::vector<std::shared_ptr<Node>> &nodes);
};
TORCH_MODULE(Network);
struct Optimizer {
  std::vector<T> state;
  uint64_t steps = 0;
  double lr = 0.0003;
  explicit Optimizer(Network &);
  void step(Network &);
  void save(torch::serialize::OutputArchive &);
  void load(torch::serialize::InputArchive &, Network &);
};
struct TrainResult {
  uint64_t consumed = 0;
  double loss = 0;
};
TrainResult learn(Network &, Optimizer &, const std::vector<Block> &);
std::vector<std::vector<uint16_t>> probabilities(const T &,
                                                 const std::vector<std::shared_ptr<Node>> &);
std::vector<double> canonical_logps(const std::vector<uint16_t> &);
void copy_weights(Network &, Network &);
void model_checks(const torch::Device &);
} // namespace hj
