#pragma once
#include "model_data.h"
#include <atomic>
#include <functional>
#include <mutex>

namespace hj {
constexpr int64_t DefaultWidth = 384, EncoderLayers = 6;
using Forward = std::function<std::pair<T, T>()>;
// 整数位精确对应独立可学习向量；置位向量求和等价于零一矩阵的线性投影。
struct TypedProjectionImpl : torch::nn::Module {
  torch::nn::EmbeddingBag bits{nullptr};
  torch::nn::Linear continuous{nullptr};
  T bias;
  explicit TypedProjectionImpl(int64_t width, bool rows);
  T forward(const T &ids, const T &offsets, const T &values = {});
};
TORCH_MODULE(TypedProjection);
struct TransformerBlockImpl : torch::nn::Module {
  int64_t heads;
  torch::nn::LayerNorm attention_norm{nullptr}, ffn_norm{nullptr};
  torch::nn::Linear qkv{nullptr}, output{nullptr}, up{nullptr}, down{nullptr};
  explicit TransformerBlockImpl(int64_t width);
  T forward(const T &tokens, const T &cumulative, const std::vector<int64_t> &lengths,
            const T &key_lengths = {}, const T &key_cumulative = {});
};
TORCH_MODULE(TransformerBlock);
struct NetworkImpl : torch::nn::Module {
  int64_t width;
  std::atomic<uint64_t> revision{0};
  std::atomic<uint64_t> encoded_roots{0}, cache_hits{0};
  std::atomic<uint64_t> computed_tokens{0}, scored_candidates{0}, projected_candidates{0};
  TypedProjection entity{nullptr}, action{nullptr};
  torch::nn::Linear context{nullptr}, prior{nullptr};
  torch::nn::Sequential query{nullptr}, key{nullptr}, value{nullptr};
  torch::nn::ModuleList blocks;
  torch::nn::LayerNorm norm{nullptr};
  torch::nn::Embedding kinds{nullptr};
  T packed_weights;
  explicit NetworkImpl(int64_t width = DefaultWidth);
  void pack_weights();
  T encode_table(const T &entities, const T &global_values, const T &kind_ids, const T &gather,
                 const T &cumulative, const std::vector<int64_t> &lengths,
                 const T &key_lengths = {});
  void encode_roots(const std::vector<std::shared_ptr<Root>> &);
  T project_root_context(const T &);
  T project_entities(const T &);
  std::pair<T, T> score_table(const packing::ScoreTables &);
  bool cache_valid(const Root &r) const {
    std::lock_guard lock(r.cache_mutex);
    return r.cached_owner == this && r.cached_revision == revision.load() &&
           r.cached_tokens.defined();
  }
  // 根、节点、候选均为不补零的张量表，同一根/候选在一次前向内只投影一次。
  std::pair<T, T> forward(const std::vector<std::shared_ptr<Node>> &nodes,
                          int64_t policy_nodes = -1);
  Forward prepare_forward(const std::vector<std::shared_ptr<Node>> &, int64_t policy_nodes = -1);
};
TORCH_MODULE(Network);
struct CpuWeights {
  uint64_t version = 0, revision = 0;
  const void *owner = nullptr;
  int64_t width = 0;
  T rows, continuous, row_bias, actions, action_bias;
  T context, context_bias, query, query_bias, query_out, query_out_bias;
  T key, key_bias, key_out, key_out_bias, prior, prior_bias;
};
using CpuSnapshot = std::shared_ptr<const CpuWeights>;
class CpuPolicy {
  struct Impl;
  std::unique_ptr<Impl> impl;

public:
  CpuPolicy();
  ~CpuPolicy();
  void clear();
  static CpuSnapshot snapshot(Network &, uint64_t version);
  std::vector<uint16_t> score(const Node &, const CpuSnapshot &);
};
void cpu_policy_checks(const torch::Device &);
struct UpdateResult {
  T values;
  double loss = 0;
  bool states_committed = false;
};
using TargetKernel = std::function<T(const T &, const T &, const T &, const T &)>;
using UpdateKernel = std::function<UpdateResult(const T &, const T &, const std::vector<T> &,
                                                const T &, double, uint64_t, const T &)>;
using CapturedUpdate =
    std::function<T(const T &, const T &, const std::vector<T> &, const T &, const T &, const T &)>;
UpdateKernel make_update_kernel(const torch::Device &, TargetKernel &);
CapturedUpdate make_captured_update(const torch::Device &);
T typed_binary(const T &, bool rows);
T segmented_table(const T &, const T &, const T &, bool mean);
T bf16_norm(const T &, torch::nn::LayerNorm &);
std::vector<T> residual_norm_cuda(const T &, const T &, const T &, const T &, double);
T short_attention_cuda(const T &, const T &, const T &);
struct Optimizer {
  UpdateKernel native_update;
  TargetKernel targets;
  std::vector<T> state;
  T gradients;
  uint64_t steps = 0;
  double lr = 0.0003;
  explicit Optimizer(Network &);
  void prepare_gradients(Network &);
  T collect_gradients(Network &);
  double step(Network &, const T &loss = {});
  void save(torch::serialize::OutputArchive &);
  void load(torch::serialize::InputArchive &, Network &);
};
struct TrainResult {
  uint64_t consumed = 0;
  double loss = 0;
  uint64_t terminal_games = 0;
  double value_mse = 0, policy_entropy = 0, gradient_norm = 0;
};
TrainResult learn(Network &, Optimizer &, const std::vector<Block> &, Forward prepared = {});
TrainResult learn_episodes(Network &, Optimizer &, const std::vector<Block> &, uint64_t batch);
class LearningEngine {
  struct Impl;
  std::unique_ptr<Impl> impl;

public:
  LearningEngine(Network &, Optimizer &, uint64_t batch);
  ~LearningEngine();
  Forward prepare(const std::vector<Block> &);
  TrainResult step(const std::vector<Block> &, Forward prepared = {});
  uint64_t replays() const;
  uint64_t fallbacks() const;
  std::array<uint64_t, 4> workload() const;
};
void learning_checks(const torch::Device &);
T policy_mass(const T &logits, const T &owner, const T &lengths, const T &valid);
std::vector<std::vector<uint16_t>> probabilities(const T &,
                                                 const std::vector<std::shared_ptr<Node>> &);
std::vector<double> canonical_logps(const std::vector<uint16_t> &);
void copy_weights(Network &, Network &);
void model_checks(const torch::Device &);
} // namespace hj
