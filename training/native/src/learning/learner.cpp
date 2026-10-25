#include "diagnostics/timing.h"
#include "prepared.h"
#include <ATen/record_function.h>
#include <iostream>
#include <torch/csrc/autograd/custom_function.h>
#ifdef HJ_CUDA
// CUDA 头必须先于图类，保证成员布局与发行库一致。
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace hj {
using namespace training;
namespace {
class PrefixMean : public torch::autograd::Function<PrefixMean> {
public:
  static T forward(torch::autograd::AutogradContext *ctx, T x, T ids, T offsets, T back_ids,
                   T back_offsets) {
    ctx->save_for_backward({offsets, back_ids, back_offsets});
    return segmented_table(x, ids, offsets, true);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::variable_list gradients) {
    auto t = ctx->get_saved_variables();
    auto lengths = (t[0].narrow(0, 1, t[0].numel() - 1) - t[0].narrow(0, 0, t[0].numel() - 1))
                       .clamp_min(1)
                       .to(torch::kBFloat16)
                       .unsqueeze(1);
    return {segmented_table((gradients[0] / lengths).contiguous(), t[1], t[2], false), T(), T(),
            T(), T()};
  }
};
Block warmup_block() {
  auto node = std::make_shared<Node>();
  node->root = std::make_shared<Root>();
  node->root->game = 1;
  node->root->id = 1;
  InputPool pool;
  node->context = make_context(pool, {}, {}, {});
  Action action{};
  action[63] = 1;
  node->candidates = {pool.actions.get(action)};
  node->mask = {1};
  node->sources = {-1};
  node->targets = {-1};
  return {{{node, 0, 0, 0, true, 1}}, {}};
}
} // namespace
struct LearningEngine::Impl {
  Network &model;
  Optimizer &optimizer;
  std::atomic<uint64_t> replay_count{0}, fallback_count{0};
  std::array<std::atomic<uint64_t>, 4> work{};
  diagnostic::GpuSpan timing;
#ifdef HJ_CUDA
  struct Family {
    Prepared input;
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    T receipt;
    Family(int64_t batch, int64_t tokens, int64_t maximum, const torch::Device &device)
        : input(batch, tokens, maximum, device) {}
  };
  std::vector<std::unique_ptr<Family>> families;
  T host_receipt;
  CapturedUpdate update;
  T compute(Prepared &t) {
    optimizer.prepare_gradients(model);
    auto raw = t.raw_rows.tensor.view({t.rows, 75});
    auto embeddings = torch::matmul(typed_binary(raw, true), model->entity->bits->weight) +
                      model->entity->continuous(raw.narrow(1, 62, 13).view(torch::kBFloat16));
    auto x = t.token_rows(embeddings) +
             model->kinds(t.kind.tensor) * t.kind_mask.tensor.to(torch::kBFloat16).unsqueeze(1);
    auto cuq = t.cuq.tensor.to(torch::kInt32), cuk = t.cuk.tensor.to(torch::kInt32);
    for (auto &block : *model->blocks)
      x = block->as<TransformerBlock>()->forward(x, cuq, {t.maximum}, {}, cuk);
    x = bf16_norm(x, model->norm);
    auto root_context = model->project_root_context(t.globals(x));
    auto pc =
        PrefixMean::apply(torch::gelu(embeddings), t.prefix_ids.tensor, t.prefix_offsets.tensor,
                          t.prefix_back_ids.tensor, t.prefix_back_offsets.tensor);
    auto affine = model->project_entities(
        t.entity_refs(torch::cat({torch::zeros({1, model->width}, x.options()), x}, 0)));
    auto actions =
        torch::gelu(torch::matmul(typed_binary(t.raw_actions.tensor.view({t.actions, 64}), false),
                                  model->action->bits->weight) +
                    model->action->bias);
    auto output = model->score_table({embeddings,
                                      {},
                                      {},
                                      root_context,
                                      affine,
                                      actions,
                                      t.node_roots.bind(),
                                      t.live_owner.bind(),
                                      t.key_actions.bind(),
                                      t.key_source.bind(),
                                      t.key_target.bind(),
                                      t.live_key.bind(),
                                      {},
                                      t.candidates,
                                      pc});
    auto logits = output.first;
    auto mass = policy_mass(logits, t.policy_owner.tensor, t.policy_lengths.tensor,
                            t.policy_mask.tensor.ne(0));
    auto logp =
        mass.clamp_min(1e-38).log() -
        segments(mass, "sum", t.policy_lengths.tensor).log().index_select(0, t.policy_owner.tensor);
    auto entropy = segments(-(mass * logp).masked_fill(t.policy_mask.tensor.eq(0), 0), "sum",
                            t.policy_lengths.tensor);
    auto values = output.second;
    auto targets = optimizer.targets(mass, values, t.records.tensor, t.blocks.tensor);
    auto chosen = logp.index_select(0, t.selected.tensor);
    auto corrected = chosen + (targets[2] - chosen.detach());
    auto policy_logp = torch::where(targets[1].ne(0), corrected, torch::zeros_like(corrected));
    auto valid = t.valid.tensor.to(torch::kBFloat16);
    auto loss = ((-policy_logp * targets[1] +
                  0.5 * (values.narrow(0, 0, t.batch) - targets[0]).square() - 0.01 * entropy) *
                 valid)
                    .sum() /
                valid.sum();
    loss.backward();
    torch::NoGradGuard no_grad;
    optimizer.collect_gradients(model);
    auto norm = optimizer.gradients.square().sum().sqrt();
    return update(optimizer.gradients, model->packed_weights, optimizer.state, norm, loss,
                  t.control.tensor);
  }
#endif
  Impl(Network &model, Optimizer &optimizer, uint64_t batch) : model(model), optimizer(optimizer) {
#ifdef HJ_CUDA
    auto device = model->parameters()[0].device();
    if (!device.is_cuda() || batch > 64)
      return;
    c10::cuda::CUDAGuard device_guard(device);
    update = make_captured_update(device);
    c10::cuda::getCurrentCUDAStream().synchronize();
    auto stream = c10::cuda::getStreamFromPool(false, device.index() < 0 ? 0 : device.index());
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    optimizer.prepare_gradients(model);
    host_receipt = torch::empty(
        {8}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    auto pool = at::cuda::graph_pool_handle();
    auto warmup = std::vector<Block>{warmup_block()};
    for (int64_t tokens : {128, 256, 512, 1024, 2048}) {
      auto family = std::make_unique<Family>(batch, tokens, std::min<int64_t>(tokens, 256), device);
      TORCH_CHECK(family->input.pack(warmup, optimizer.steps + 1, optimizer.lr, false),
                  "training warmup layout overflow");
      family->input.bank.upload();
      compute(family->input);
      stream.synchronize();
      family->graph = std::make_unique<at::cuda::CUDAGraph>();
      family->graph->capture_begin(pool);
      family->receipt = compute(family->input);
      family->graph->capture_end();
      stream.synchronize();
      families.push_back(std::move(family));
    }
#endif
  }
  TrainResult step(const std::vector<Block> &blocks, Forward prepared) {
    auto started = diagnostic::now();
#ifdef HJ_CUDA
    for (auto &family : families) {
      c10::cuda::CUDAGuard device_guard(model->parameters()[0].device());
      auto &input = family->input;
      if (!input.pack(blocks, optimizer.steps + 1, optimizer.lr, true))
        continue;
      RECORD_FUNCTION("hj.training_graph", std::vector<c10::IValue>());
      auto submitted = diagnostic::now();
      timing.start();
      input.bank.upload();
      family->graph->replay();
      timing.compute_end();
      auto stream = c10::cuda::getCurrentCUDAStream();
      TORCH_CHECK(cudaMemcpyAsync(host_receipt.data_ptr(), family->receipt.data_ptr(), 8,
                                  cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                  "training receipt copy failed");
      timing.return_end();
      TORCH_CHECK(cudaStreamSynchronize(stream) == cudaSuccess, "training graph execution failed");
      auto callback = diagnostic::now();
      timing.report("learn_gpu", optimizer.steps, submitted, callback, input.count,
                    input.used_tokens);
      diagnostic::record("learn_wall", optimizer.steps, callback - started, submitted - started,
                         input.count, 1, timing.uncertainty_ms);
      auto bytes = host_receipt.data_ptr<uint8_t>();
      int accepted;
      std::memcpy(&accepted, bytes, 4);
      TORCH_CHECK(accepted,
                  "non-finite BF16 training graph; parameters and optimizer not committed");
      BF loss;
      std::memcpy(&loss.x, bytes + 4, 2);
      ++optimizer.steps;
      ++model->revision;
      ++replay_count;
      model->packed_weights.unsafeGetTensorImpl()->bump_version();
      model->encoded_roots += input.used_roots;
      model->computed_tokens += input.used_tokens;
      model->scored_candidates += input.used_candidates;
      model->projected_candidates += input.used_actions;
      work[0] += input.used_tokens;
      work[1] += input.tokens;
      work[2] += input.used_keys;
      work[3] += input.keys;
      return {input.count, double(float(loss))};
    }
#endif
    ++fallback_count;
    auto tokens_before = model->computed_tokens.load();
#ifdef HJ_CUDA
    if (model->parameters()[0].is_cuda())
      timing.start();
#endif
    auto result = learn(model, optimizer, blocks, std::move(prepared));
#ifdef HJ_CUDA
    if (model->parameters()[0].is_cuda() && diagnostic::enabled) {
      timing.compute_end();
      timing.return_end();
      c10::cuda::getCurrentCUDAStream().synchronize();
      timing.report("learn_gpu", optimizer.steps - 1, started, diagnostic::now(), result.consumed,
                    model->computed_tokens.load() - tokens_before);
    }
#endif
    diagnostic::record("learn_wall", optimizer.steps - 1, diagnostic::now() - started, 0,
                       result.consumed, 0);
    return result;
  }
};
LearningEngine::LearningEngine(Network &model, Optimizer &optimizer, uint64_t batch)
    : impl(std::make_unique<Impl>(model, optimizer, batch)) {}
LearningEngine::~LearningEngine() = default;
Forward LearningEngine::prepare(const std::vector<Block> &blocks) {
#ifdef HJ_CUDA
  if (!impl->families.empty())
    return {};
#endif
  std::vector<std::shared_ptr<Node>> nodes;
  for (auto &b : blocks)
    for (auto &step : b.steps)
      nodes.push_back(step.node);
  auto count = nodes.size();
  for (auto &b : blocks)
    if (b.bootstrap)
      nodes.push_back(b.bootstrap);
  return impl->model->prepare_forward(nodes, count);
}
TrainResult LearningEngine::step(const std::vector<Block> &blocks, Forward prepared) {
  return impl->step(blocks, std::move(prepared));
}
uint64_t LearningEngine::replays() const {
  return impl->replay_count;
}
uint64_t LearningEngine::fallbacks() const {
  return impl->fallback_count;
}
std::array<uint64_t, 4> LearningEngine::workload() const {
  return {impl->work[0].load(), impl->work[1].load(), impl->work[2].load(), impl->work[3].load()};
}
void learning_checks(const torch::Device &device) {
  if (!device.is_cuda())
    return;
  Network model(32);
  model->to(device);
  Optimizer optimizer(model);
  Network reference(32);
  reference->to(device);
  copy_weights(model, reference);
  Optimizer reference_optimizer(reference);
  auto block = warmup_block();
  InputPool pool;
  auto node = block.steps[0].node;
  Row entity;
  entity.ids[0] = 2;
  entity.ids[2] = 4097;
  entity.present = 1;
  entity.integers = 1;
  entity.ids[8] = 7;
  node->root->rows = {pool.rows.get(entity)};
  node->root->kinds = {128};
  node->candidates.clear();
  for (int i = 0; i < 3; ++i) {
    Action a{};
    a[i] = 1;
    node->candidates.push_back(pool.actions.get(a));
  }
  node->mask = {1, 1, 0};
  node->sources = {0, 0, 0};
  node->targets = {-1, 0, -1};
  node->context = make_context(pool, {}, {1, 2, 0}, {{{1}, {0, 3}}});
  auto next = std::make_shared<Node>(*node);
  next->mask = {1, 0, 1};
  next->context = make_context(pool, {}, {2, 0}, {});
  auto bootstrap = std::make_shared<Node>(*next);
  bootstrap->root = std::make_shared<Root>(*node->root);
  bootstrap->root->id = 2;
  block.steps = {{node, 1, std::log(0.5), 0, false, 0}, {next, 2, std::log(0.25), 0, false, 0}};
  block.bootstrap = bootstrap;
  LearningEngine engine(model, optimizer, 4);
  auto original = reference->packed_weights.detach().clone();
  TORCH_CHECK(torch::equal(model->packed_weights, original) && optimizer.steps == 0,
              "training graph capture committed a warmup update");
  auto terminal = block;
  terminal.steps.back().terminal = true;
  terminal.steps.back().reward = 1;
  auto work = std::vector<Block>{block, terminal};
  auto actual = engine.step(work), expected = learn(reference, reference_optimizer, work);
  TORCH_CHECK(actual.consumed == 4 && std::abs(actual.loss - expected.loss) < 0.03,
              "captured loss differs from complete learner");
  TORCH_CHECK(torch::allclose(model->packed_weights, reference->packed_weights, 0.03, 0.004),
              "captured update differs from complete learner");
  for (size_t i = 0; i < 2; ++i) {
    auto a = optimizer.state[i].to(torch::kFloat32),
         b = reference_optimizer.state[i].to(torch::kFloat32);
    // 稠密和稀疏投影的归约顺序不同；按整体范数检查，避免近零抵消项的相对误差失真。
    auto relative =
        ((a - b).square().sum() / b.square().sum().clamp_min(1e-30)).sqrt().item<double>();
    auto maximum = (a - b).abs().max().item<double>(), scale = b.abs().max().item<double>();
    TORCH_CHECK(relative < 0.04 && maximum < 0.04 * scale + 0.00001,
                "captured first update changed optimizer moments: state=", i,
                " relative=", relative, " maximum=", maximum, " scale=", scale);
  }
  int64_t moment_offset = 0;
  for (auto &entry : model->named_parameters()) {
    auto n = entry.value().numel();
    if (entry.key() == "query.0.weight" || entry.key() == "key.0.weight" ||
        entry.key() == "blocks.0.qkv.weight")
      TORCH_CHECK(optimizer.state[0].narrow(0, moment_offset, n).abs().sum().item<double>() > 0,
                  "captured policy gradient lost query, key or Transformer attention");
    moment_offset += n;
  }
  Network restored(32);
  restored->to(device);
  copy_weights(model, restored);
  Optimizer restored_optimizer(restored);
  restored_optimizer.steps = optimizer.steps;
  for (size_t i = 0; i < optimizer.state.size(); ++i)
    restored_optimizer.state[i].copy_(optimizer.state[i]);
  LearningEngine restored_engine(restored, restored_optimizer, 4);
  auto large = block;
  large.steps[0].node = std::make_shared<Node>(*block.steps[0].node);
  large.steps[0].node->root = std::make_shared<Root>(*block.steps[0].node->root);
  large.steps[0].node->root->id = 99;
  for (int i = 0; i < 150; ++i) {
    Row row;
    row.ids[2] = i + 1;
    large.steps[0].node->root->rows.push_back(pool.rows.get(row));
    large.steps[0].node->root->kinds.push_back(128);
  }
  auto oversized = large;
  oversized.steps[0].node = std::make_shared<Node>(*large.steps[0].node);
  oversized.steps[0].node->root = std::make_shared<Root>(*large.steps[0].node->root);
  oversized.steps[0].node->root->id = 101;
  for (int i = 0; i < 150; ++i) {
    oversized.steps[0].node->root->rows.push_back(pool.rows.get(Row{}));
    oversized.steps[0].node->root->kinds.push_back(128);
  }
  auto gradient_address = optimizer.gradients.data_ptr();
  for (auto &b : std::vector<Block>{block, large, block, oversized, block}) {
    engine.step({b});
    restored_engine.step({b});
    TORCH_CHECK(torch::equal(model->packed_weights, restored->packed_weights),
                "A-B-A graph replay or restored update changed parameters");
    for (size_t i = 0; i < optimizer.state.size(); ++i)
      TORCH_CHECK(torch::equal(optimizer.state[i], restored_optimizer.state[i]),
                  "A-B-A graph replay changed BF16 optimizer state");
    TORCH_CHECK(optimizer.gradients.data_ptr() == gradient_address,
                "eager fallback replaced shared gradient storage");
  }
  TORCH_CHECK(engine.replays() == 5 && engine.fallbacks() == 1,
              "full oversized fallback or subsequent graph replay was skipped");
  auto invalid = block;
  invalid.steps[0].node = std::make_shared<Node>(*node);
  invalid.steps[0].node->root = std::make_shared<Root>(*node->root);
  invalid.steps[0].node->root->id = 100;
  Row bad;
  bad.present = 1;
  bad.values[0] = BF(NAN).x;
  invalid.steps[0].node->root->globals = pool.rows.get(bad);
  auto before = model->packed_weights.clone();
  std::vector<T> states;
  for (auto &state : optimizer.state)
    states.push_back(state.clone());
  auto steps = optimizer.steps;
  bool rejected = false;
  try {
    engine.step({invalid});
  } catch (const std::exception &) {
    rejected = true;
  }
  TORCH_CHECK(rejected && optimizer.steps == steps && torch::equal(before, model->packed_weights),
              "non-finite captured update committed parameters");
  for (size_t i = 0; i < states.size(); ++i)
    TORCH_CHECK(torch::equal(states[i], optimizer.state[i]),
                "non-finite captured update committed state");
  engine.step({block});
  restored_engine.step({block});
  TORCH_CHECK(torch::equal(model->packed_weights, restored->packed_weights),
              "rejected graph update changed next valid update");
  LearningEngine eager(model, optimizer, 128);
  // 提前打包下一批后先更新权重；预取不得冻结旧权重或改变下一步更新。
  auto prepared = eager.prepare({block});
  eager.step({block});
  learn(restored, restored_optimizer, {block});
  eager.step({block}, std::move(prepared));
  learn(restored, restored_optimizer, {block});
  TORCH_CHECK(torch::equal(model->packed_weights, restored->packed_weights),
              "prefetched input changed the next committed weights");
  for (size_t i = 0; i < optimizer.state.size(); ++i)
    TORCH_CHECK(torch::equal(optimizer.state[i], restored_optimizer.state[i]),
                "prefetched input changed BF16 optimizer state");
  std::cout << "complete training graph, typed tables, masked support, bootstrap, A-B-A replay, "
               "BF16 atomic update and restored continuation: passed\n";
}
} // namespace hj
