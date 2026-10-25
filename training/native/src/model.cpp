#include "model.h"
#include <ATen/Parallel.h>
#include <ATen/ops/_efficient_attention_backward.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/_flash_attention_backward.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/embedding_bag.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#ifdef HJ_CUDA
#include <ATen/native/transformers/cuda/sdp_utils.h>
#endif
#include <bit>
#include <cmath>
#include <iostream>
#include <map>
#include <span>
#include <sstream>
#include <torch/csrc/autograd/custom_function.h>
#include <unordered_map>

namespace hj {
using namespace packing;
namespace {
// 使用 LibTorch 自带的变长内核；Windows 发行库可能只提供 efficient 后端。
class RaggedAttention : public torch::autograd::Function<RaggedAttention> {
public:
  static T forward(torch::autograd::AutogradContext *ctx, T projected, T cumulative,
                   int64_t maximum, std::optional<T> key_lengths, std::optional<T> key_cumulative) {
    auto q = projected.select(1, 0), k = projected.select(1, 1), v = projected.select(1, 2);
    auto used = key_lengths;
    auto keys = key_cumulative.value_or(cumulative);
    bool record = projected.requires_grad();
    // 小表避免通用注意力的启动开销；大表保留张量核心后端，按真实形状选择算法。
    if (!record && projected.size(0) <= 32 && maximum <= 512 && projected.size(3) <= 64 &&
        !key_cumulative)
      return short_attention_cuda(projected, cumulative, key_lengths.value_or(T()));
    bool flash = false;
#ifdef HJ_CUDA
    flash = sdp::is_flash_attention_available();
#endif
    T out, lse, rng, unused;
    if (flash)
      std::tie(out, lse, rng, unused, std::ignore) = at::_flash_attention_forward(
          q, k, v, cumulative, keys, maximum, maximum, 0, false, false, {}, {}, {}, used);
    else {
      q = q.unsqueeze(0);
      k = k.unsqueeze(0);
      v = v.unsqueeze(0);
      std::tie(out, lse, rng, unused, std::ignore, std::ignore) = at::_efficient_attention_forward(
          q, k, v, {}, cumulative, keys, maximum, maximum, 0, 0, record, {}, used);
    }
    auto result = flash ? out : out.squeeze(0);
    if (key_cumulative || key_lengths) {
      auto padding = torch::arange(projected.size(0), cumulative.options())
                         .ge(cumulative.select(0, -1))
                         .view({-1, 1, 1});
      result = result.masked_fill(padding, 0);
      out = flash ? result : result.unsqueeze(0);
    }
    if (record) {
      ctx->saved_data["lse_type"] = int64_t(lse.scalar_type());
      ctx->saved_data["padded"] = key_cumulative.has_value();
      ctx->save_for_backward(
          {q, k, v, cumulative, out, lse.to(torch::kBFloat16), rng, unused, keys});
      ctx->saved_data["maximum"] = maximum;
      ctx->saved_data["flash"] = flash;
    }
    return result;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::variable_list grad) {
    auto t = ctx->get_saved_variables();
    auto maximum = ctx->saved_data["maximum"].toInt();
    auto lse = t[5].to(torch::ScalarType(ctx->saved_data["lse_type"].toInt()));
    T dq, dk, dv;
    if (ctx->saved_data["flash"].toBool())
      std::tie(dq, dk, dv) =
          at::_flash_attention_backward(grad[0].contiguous(), t[0], t[1], t[2], t[4], lse, t[3],
                                        t[8], maximum, maximum, 0, false, t[6], t[7]);
    else {
      std::tie(dq, dk, dv, std::ignore) = at::_efficient_attention_backward(
          grad[0].unsqueeze(0).contiguous(), t[0], t[1], t[2], {}, t[4], t[3], t[8], maximum,
          maximum, lse, 0, t[6], t[7], 0, false);
      dq = dq.squeeze(0);
      dk = dk.squeeze(0);
      dv = dv.squeeze(0);
    }
    if (ctx->saved_data["padded"].toBool()) {
      auto padding =
          torch::arange(dq.size(0), t[8].options()).ge(t[8].select(0, -1)).view({-1, 1, 1});
      dq = dq.masked_fill(padding, 0);
      dk = dk.masked_fill(padding, 0);
      dv = dv.masked_fill(padding, 0);
    }
    // Q/K/V 属于同一投影表，反向一次拼回；不为三次 select 分别清零整张梯度表。
    return {torch::stack({dq, dk, dv}, 1), T(), T(), T(), T()};
  }
};
// 算子寄存器可累加到 FP32；跨前后向保存的归一化统计同样量化为 BF16。
class BF16Norm : public torch::autograd::Function<BF16Norm> {
public:
  static T forward(torch::autograd::AutogradContext *ctx, T x, T weight, T bias, double eps) {
    auto [out, mean, inverse] = at::native_layer_norm(x, {x.size(-1)}, weight, bias, eps);
    ctx->saved_data["stat_type"] = int64_t(mean.scalar_type());
    ctx->save_for_backward(
        {x, weight, bias, mean.to(torch::kBFloat16), inverse.to(torch::kBFloat16)});
    return out;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::variable_list gradients) {
    auto t = ctx->get_saved_variables();
    auto type = torch::ScalarType(ctx->saved_data["stat_type"].toInt());
    auto [dx, dw, db] = at::native_layer_norm_backward(
        gradients[0].contiguous(), t[0], {t[0].size(-1)}, t[3].to(type), t[4].to(type), t[1], t[2],
        {true, true, true});
    return {dx, dw, db, T()};
  }
};
class ResidualNorm : public torch::autograd::Function<ResidualNorm> {
public:
  static torch::autograd::variable_list forward(torch::autograd::AutogradContext *ctx, T a, T b,
                                                T weight, T bias, double eps) {
    auto result = residual_norm_cuda(a, b, weight, bias, eps);
    ctx->save_for_backward({result[0], weight, bias, result[2], result[3]});
    return {result[0], result[1]};
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::variable_list gradients) {
    auto t = ctx->get_saved_variables();
    auto [dx, dw, db] = at::native_layer_norm_backward(
        gradients[1].contiguous(), t[0], {t[0].size(-1)}, t[3].to(torch::kFloat32),
        t[4].to(torch::kFloat32), t[1], t[2], {true, true, true});
    auto residual = dx + gradients[0];
    return {residual, residual, dw, db, T()};
  }
};
class GroupedGather : public torch::autograd::Function<GroupedGather> {
public:
  static T forward(torch::autograd::AutogradContext *ctx, T x, T ids, T positions, T offsets) {
    ctx->save_for_backward({positions, offsets});
    return x.index_select(0, ids);
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::variable_list gradients) {
    auto saved = ctx->get_saved_variables();
    // 反向按输入行顺序归约，不在热门引用上执行大量 BF16 原子加法。
    auto gradient = segmented_table(gradients[0].contiguous(), saved[0], saved[1], false);
    return {gradient, T(), T(), T()};
  }
};
struct PolicyTable {
  T mass, logp, entropy;
  std::vector<int64_t> lengths;
};
class ValueTanh : public torch::autograd::Function<ValueTanh> {
public:
  static T forward(torch::autograd::AutogradContext *ctx, T x) {
    ctx->save_for_backward({x});
    return x.tanh();
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::variable_list gradients) {
    // BF16 输出在有限输入处就会舍入为 ±1，不能用该舍入结果计算 1-y²。
    auto x = ctx->get_saved_variables()[0].to(torch::kFloat32);
    auto e = (-2 * x.abs()).exp();
    return {(gradients[0].to(torch::kFloat32) * (4 * e / (1 + e).square())).to(torch::kBFloat16)};
  }
};
class PolicyMass : public torch::autograd::Function<PolicyMass> {
public:
  static T forward(torch::autograd::AutogradContext *ctx, T logits, T owner, T lengths, T valid) {
    auto masked = logits.masked_fill(~valid, -INFINITY);
    auto maximum = segments(masked, "max", lengths);
    auto mass = (masked - maximum.index_select(0, owner)).exp();
    mass = mass / segments(mass, "sum", lengths).index_select(0, owner);
    ctx->save_for_backward({mass, owner, lengths});
    return mass;
  }
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::variable_list gradients) {
    auto saved = ctx->get_saved_variables();
    auto mass = saved[0].to(torch::kFloat32), grad = gradients[0].to(torch::kFloat32);
    auto owner = saved[1], lengths = saved[2];
    // 保存的质量与返回梯度仍是 BF16；归约工作区使用算子累加精度。
    // 先移除共同项，使均匀分布的熵梯度严格为零，避免 BF16 舍入驱动评分整体漂移。
    auto baseline = segments(grad.masked_fill(mass.eq(0), -INFINITY), "max", lengths);
    auto centered = grad - baseline.index_select(0, owner);
    auto mean = segments(mass * centered, "sum", lengths) / segments(mass, "sum", lengths);
    auto result = (mass * (centered - mean.index_select(0, owner))).to(torch::kBFloat16);
    return {result, T(), T(), T()};
  }
};
PolicyTable policy_table(const T &logits, std::span<const std::shared_ptr<Node>> nodes,
                         bool statistics = true, Inputs *provided = nullptr) {
  std::vector<int64_t> owner, lengths, mask;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto n = nodes[i]->mask.size();
    TORCH_CHECK(n && std::any_of(nodes[i]->mask.begin(), nodes[i]->mask.end(),
                                 [](auto v) { return v != 0; }),
                "empty policy support");
    lengths.push_back(n);
    owner.insert(owner.end(), n, i);
    mask.insert(mask.end(), nodes[i]->mask.begin(), nodes[i]->mask.end());
  }
  TORCH_CHECK(int64_t(mask.size()) == logits.numel(), "policy segment lengths differ");
  Inputs local;
  auto &input = provided ? *provided : local;
  auto owners = input.add(owner), sizes = input.add(lengths), masks = input.add(mask);
  input.upload(logits.device());
  auto index = input.get(owners), length = input.get(sizes);
  auto valid = input.get(masks).ne(0);
  auto mass = policy_mass(logits, index, length, valid);
  if (!statistics)
    return {mass, {}, {}, lengths};
  auto logp =
      mass.clamp_min(1e-38).log() - segments(mass, "sum", length).log().index_select(0, index);
  auto entropy = segments(-(mass * logp).masked_fill(~valid, 0), "sum", length);
  return {mass, logp, entropy, lengths};
}
std::vector<std::vector<uint16_t>> cpu_masses(const PolicyTable &table) {
  auto cpu = table.mass.to(torch::kCPU).contiguous();
  auto p = cpu.data_ptr<BF>();
  std::vector<std::vector<uint16_t>> result;
  int64_t offset = 0;
  for (auto length : table.lengths) {
    std::vector<uint16_t> v;
    v.reserve(length);
    for (int64_t j = 0; j < length; ++j)
      v.push_back(p[offset++].x);
    result.push_back(std::move(v));
  }
  return result;
}
} // namespace
T bf16_norm(const T &x, torch::nn::LayerNorm &norm) {
  if (!torch::GradMode::is_enabled())
    return norm(x);
  return BF16Norm::apply(x, norm->weight, norm->bias, norm->options.eps());
}
T packing::grouped_gather(const T &x, const T &ids, const T &positions, const T &offsets) {
  return GroupedGather::apply(x, ids, positions, offsets);
}
T policy_mass(const T &logits, const T &owner, const T &lengths, const T &valid) {
  return PolicyMass::apply(logits, owner, lengths, valid);
}
TypedProjectionImpl::TypedProjectionImpl(int64_t w, bool rows) {
  auto features = rows ? RowFeatures : ActionFeatures;
  bits =
      register_module("bits", torch::nn::EmbeddingBag(torch::nn::EmbeddingBagOptions(
                                                          rows ? 60 * 64 + 104 : ActionFeatures, w)
                                                          .mode(torch::kSum)
                                                          .include_last_offset(true)));
  if (rows)
    continuous = register_module("continuous", torch::nn::Linear(52, w));
  else
    bias = register_parameter("bias", torch::empty({w}, bfopts()));
  torch::NoGradGuard no_grad;
  auto bound = 1 / std::sqrt(double(features));
  bits->weight.uniform_(-bound, bound);
  if (rows) {
    continuous->weight.uniform_(-bound, bound);
    continuous->bias.uniform_(-bound, bound);
  } else
    bias.uniform_(-bound, bound);
}
T TypedProjectionImpl::forward(const T &ids, const T &offsets, const T &values) {
  auto x = bits(ids, offsets);
  return continuous ? x + continuous(values) : x + bias;
}
TransformerBlockImpl::TransformerBlockImpl(int64_t w) : heads(std::max<int64_t>(1, w / 64)) {
  TORCH_CHECK(w % heads == 0, "Transformer width must divide attention heads");
  attention_norm = register_module("attention_norm", torch::nn::LayerNorm(std::vector<int64_t>{w}));
  ffn_norm = register_module("ffn_norm", torch::nn::LayerNorm(std::vector<int64_t>{w}));
  qkv = register_module("qkv", torch::nn::Linear(w, w * 3));
  output = register_module("output", torch::nn::Linear(w, w));
  up = register_module("up", torch::nn::Linear(w, w * 4));
  down = register_module("down", torch::nn::Linear(w * 4, w));
}
T TransformerBlockImpl::forward(const T &tokens, const T &cumulative,
                                const std::vector<int64_t> &lengths, const T &key_lengths,
                                const T &key_cumulative) {
  TORCH_CHECK(!key_lengths.defined() || (tokens.is_cuda() && !torch::GradMode::is_enabled()),
              "padded attention is CUDA inference-only");
  auto width = tokens.size(1);
  auto projected = qkv(bf16_norm(tokens, attention_norm)).view({-1, 3, heads, width / heads});
  T attended;
  if (tokens.is_cuda())
    attended = RaggedAttention::apply(
        projected, cumulative, *std::max_element(lengths.begin(), lengths.end()),
        key_lengths.defined() ? std::make_optional(key_lengths) : std::nullopt,
        key_cumulative.defined() ? std::make_optional(key_cumulative) : std::nullopt);
  else {
    auto q = projected.select(1, 0), k = projected.select(1, 1), v = projected.select(1, 2);
    std::vector<T> parts;
    int64_t offset = 0;
    for (auto length : lengths) {
      auto view = [&](const T &t) {
        return t.narrow(0, offset, length).transpose(0, 1).unsqueeze(0);
      };
      auto weights =
          (torch::matmul(view(q), view(k).transpose(-2, -1)) / std::sqrt(double(width / heads)))
              .softmax(-1);
      parts.push_back(torch::matmul(weights, view(v)).squeeze(0).transpose(0, 1));
      offset += length;
    }
    attended = torch::cat(parts, 0);
  }
  auto attention = output(attended.reshape({-1, width}));
  if (tokens.is_cuda()) {
    auto fused = torch::GradMode::is_enabled()
                     ? ResidualNorm::apply(tokens, attention, ffn_norm->weight, ffn_norm->bias,
                                           ffn_norm->options.eps())
                     : residual_norm_cuda(tokens, attention, ffn_norm->weight, ffn_norm->bias,
                                          ffn_norm->options.eps());
    return fused[0] + down(torch::gelu(up(fused[1])));
  }
  auto x = tokens + attention;
  return x + down(torch::gelu(up(bf16_norm(x, ffn_norm))));
}
NetworkImpl::NetworkImpl(int64_t w) : width(w) {
  entity = register_module("entity", TypedProjection(w, true));
  action = register_module("action", TypedProjection(w, false));
  kinds = register_module("kinds", torch::nn::Embedding(256, w));
  blocks = register_module("blocks", torch::nn::ModuleList());
  for (int64_t i = 0; i < EncoderLayers; ++i)
    blocks->push_back(TransformerBlock(w));
  norm = register_module("norm", torch::nn::LayerNorm(std::vector<int64_t>{w}));
  context = register_module("context", torch::nn::Linear(w * 2, w));
  query = register_module("query", torch::nn::Sequential(torch::nn::Linear(w, w), torch::nn::GELU(),
                                                         torch::nn::Linear(w, w)));
  key = register_module("key", torch::nn::Sequential(torch::nn::Linear(w * 3, w), torch::nn::GELU(),
                                                     torch::nn::Linear(w, w)));
  prior = register_module("prior", torch::nn::Linear(w, 1));
  value = register_module(
      "value",
      torch::nn::Sequential(torch::nn::Linear(w, w), torch::nn::GELU(), torch::nn::Linear(w, 1),
                            torch::nn::Functional([](T x) { return ValueTanh::apply(x); })));
  to(torch::kBFloat16);
}
void NetworkImpl::pack_weights() {
  torch::NoGradGuard no_grad;
  auto ps = parameters();
  if (packed_weights.defined() && packed_weights.data_ptr() == ps.front().data_ptr())
    return;
  std::vector<T> flat;
  for (auto &p : ps)
    flat.push_back(p.detach().reshape({-1}));
  packed_weights = torch::cat(flat);
  int64_t offset = 0;
  for (auto &p : ps) {
    p.set_data(packed_weights.narrow(0, offset, p.numel()).view(p.sizes()));
    offset += p.numel();
  }
}
T NetworkImpl::encode_table(const T &entities, const T &global_values, const T &kind_ids,
                            const T &gather, const T &cumulative,
                            const std::vector<int64_t> &lengths, const T &key_lengths) {
  auto e = entities + kinds(kind_ids);
  auto x = torch::cat({global_values, e}, 0).index_select(0, gather);
  for (auto &block : *blocks)
    x = block->as<TransformerBlock>()->forward(x, cumulative, lengths, key_lengths);
  return bf16_norm(x, norm);
}
T NetworkImpl::project_root_context(const T &x) {
  return torch::linear(x, context->weight.narrow(1, 0, width));
}
T NetworkImpl::project_entities(const T &x) {
  auto &first = key->at<torch::nn::LinearImpl>(0);
  return torch::cat({torch::linear(x, first.weight.narrow(1, width, width)),
                     torch::linear(x, first.weight.narrow(1, width * 2, width))},
                    1);
}
std::pair<T, T> NetworkImpl::score_table(const ScoreTables &t) {
  auto pc = t.prefix_context.defined()
                ? t.prefix_context
                : std::get<0>(at::embedding_bag(torch::gelu(t.row_embeddings), t.prefix_ids,
                                                t.prefix_offsets, false, 1, false, {}, true));
  auto ctx = torch::gelu(t.node_root(t.root_context) +
                         torch::linear(pc, context->weight.narrow(1, width, width), context->bias));
  auto &first = key->at<torch::nn::LinearImpl>(0);
  auto candidate = torch::linear(t.actions, first.weight.narrow(1, 0, width), first.bias);
  auto hidden = torch::gelu(t.candidate(candidate) + t.source(t.entity_affine.narrow(1, 0, width)) +
                            t.target(t.entity_affine.narrow(1, width, width)));
  auto keys = key->at<torch::nn::LinearImpl>(2).forward(hidden);
  auto queries = query->forward<T>(ctx);
  auto live = (t.owner(queries) * t.key(keys)).sum(-1) / std::sqrt(double(width)) +
              t.key(prior(keys)).squeeze(-1);
  auto logits =
      t.live_positions.defined()
          ? torch::zeros({t.full_candidates}, live.options()).index_copy(0, t.live_positions, live)
          : live;
  return {logits, value->forward<T>(ctx).squeeze(-1)};
}
void NetworkImpl::encode_roots(const std::vector<std::shared_ptr<Root>> &requested) {
  TORCH_CHECK(!torch::GradMode::is_enabled(), "root cache requires inference mode");
  std::vector<std::shared_ptr<Root>> roots;
  for (auto &root : requested)
    if (!cache_valid(*root))
      roots.push_back(root);
  if (roots.empty())
    return;
  Rows rows;
  Inputs input;
  std::vector<int64_t> row_ids, global_ids, kinds_, gather, cumulative{0}, lengths,
      global_positions;
  for (size_t i = 0; i < roots.size(); ++i) {
    auto &root = *roots[i];
    global_positions.push_back(cumulative.back());
    gather.push_back(i);
    for (size_t j = 0; j < root.rows.size(); ++j) {
      gather.push_back(roots.size() + row_ids.size());
      row_ids.push_back(rows.add(root.rows[j]));
      kinds_.push_back(root.kinds[j]);
    }
    lengths.push_back(root.rows.size() + 1);
    cumulative.push_back(cumulative.back() + lengths.back());
    global_ids.push_back(rows.add(root.globals));
  }
  auto packed = rows.pack(input);
  auto ids = input.add(row_ids), global = input.add(global_ids), kind = input.add(kinds_),
       order = input.add(gather);
  auto cu = input.add(cumulative), positions = input.add(global_positions);
  auto device = parameters()[0].device();
  input.upload(device);
  auto projected = entity(input.get(packed[0]), input.get(packed[1]), input.bf(packed[2], 52));
  auto encoded = encode_table(projected.index_select(0, input.get(ids)),
                              projected.index_select(0, input.get(global)), input.get(kind),
                              input.get(order), input.get(cu).to(torch::kInt32), lengths);
  auto affine = project_entities(encoded);
  auto context_part = project_root_context(encoded.index_select(0, input.get(positions)));
  for (size_t i = 0; i < roots.size(); ++i) {
    std::lock_guard lock(roots[i]->cache_mutex);
    roots[i]->cached_tokens = affine.narrow(0, cumulative[i] + 1, lengths[i] - 1).clone();
    roots[i]->cached_context = context_part[i].clone();
    roots[i]->host_tokens = T();
    roots[i]->host_context = T();
    roots[i]->account_cache();
    roots[i]->cached_owner = this;
    roots[i]->cached_revision = revision.load();
  }
  encoded_roots += roots.size();
}
std::pair<T, T> NetworkImpl::forward(const std::vector<std::shared_ptr<Node>> &nodes,
                                     int64_t policy_nodes) {
  return prepare_forward(nodes, policy_nodes)();
}
Forward NetworkImpl::prepare_forward(const std::vector<std::shared_ptr<Node>> &nodes,
                                     int64_t policy_nodes) {
  RECORD_FUNCTION("hj.forward", std::vector<c10::IValue>());
  if (nodes.empty())
    throw std::runtime_error("empty model batch");
  if (policy_nodes < 0)
    policy_nodes = nodes.size();
  TORCH_CHECK(policy_nodes > 0 && policy_nodes <= int64_t(nodes.size()),
              "invalid policy node count");
  auto device = parameters()[0].device();
  std::map<std::pair<uint64_t, uint64_t>, int64_t> root_ids;
  std::vector<std::shared_ptr<Root>> roots;
  std::vector<int64_t> node_roots, kind_ids, offsets;
  for (auto &n : nodes) {
    auto key = std::make_pair(n->root->game, n->root->id);
    auto it = root_ids.find(key);
    if (it == root_ids.end()) {
      int64_t id = roots.size();
      root_ids[key] = id;
      roots.push_back(n->root);
      node_roots.push_back(id);
    } else
      node_roots.push_back(it->second);
  }
  Rows rows;
  Inputs input;
  std::vector<int64_t> er, global_rows;
  std::vector<int64_t> missing, missing_offsets;
  std::vector<T> root_tokens(roots.size()), root_context(roots.size());
  int64_t offset = 0;
  bool caching = !torch::GradMode::is_enabled();
  std::vector<uint8_t> cold(roots.size(), 0);
  if (caching) {
    std::vector<std::shared_ptr<Root>> to_encode;
    for (size_t i = 0; i < roots.size(); ++i)
      if (!cache_valid(*roots[i])) {
        cold[i] = 1;
        to_encode.push_back(roots[i]);
      }
    encode_roots(to_encode);
  }
  for (size_t i = 0; i < roots.size(); ++i) {
    offsets.push_back(offset);
    offset += roots[i]->rows.size();
    if (caching) {
      std::lock_guard lock(roots[i]->cache_mutex);
      if (roots[i]->cached_owner == this && roots[i]->cached_revision == revision.load() &&
          roots[i]->cached_tokens.defined()) {
        root_tokens[i] = roots[i]->cached_tokens;
        root_context[i] = roots[i]->cached_context;
        cache_hits += !cold[i];
        continue;
      }
    }
    missing_offsets.push_back(kind_ids.size());
    missing.push_back(i);
    for (size_t j = 0; j < roots[i]->rows.size(); ++j) {
      er.push_back(rows.add(roots[i]->rows[j]));
      kind_ids.push_back(roots[i]->kinds[j]);
    }
    global_rows.push_back(rows.add(roots[i]->globals));
  }
  // 参数行先非线性映射再聚合；顺序、所属参数和路径编号保留在整数位特征中。
  std::vector<int64_t> pr, prefix_offsets{0};
  for (size_t i = 0; i < nodes.size(); ++i) {
    append_context(*nodes[i], rows, pr);
    prefix_offsets.push_back(pr.size());
  }
  std::unordered_map<std::array<int64_t, 64>, int64_t, ArrayHash> candidate_ids;
  std::vector<std::array<int64_t, 64>> actions;
  std::vector<int64_t> candidates, owners, source, target, key_ids, live_positions;
  std::map<std::array<int64_t, 4>, int64_t> keys;
  int64_t full_candidates = 0;
  for (size_t i = 0; i < size_t(policy_nodes); ++i) {
    auto &n = *nodes[i];
    auto off = offsets[node_roots[i]];
    for (size_t j = 0; j < n.candidates.size(); ++j) {
      auto position = full_candidates++;
      if (!n.mask[j])
        continue;
      auto &candidate_row = n.candidates[j]->value;
      auto [it, fresh] = candidate_ids.try_emplace(candidate_row, actions.size());
      if (fresh)
        actions.push_back(candidate_row);
      auto from = n.sources[j] < 0 ? 0 : off + n.sources[j] + 1;
      auto to = n.targets[j] < 0 ? 0 : off + n.targets[j] + 1;
      std::array<int64_t, 4> binding{node_roots[i], it->second, from, to};
      auto [bound, new_key] = keys.try_emplace(binding, keys.size());
      if (new_key) {
        candidates.push_back(it->second);
        source.push_back(from);
        target.push_back(to);
      }
      key_ids.push_back(bound->second);
      owners.push_back(i);
      live_positions.push_back(position);
    }
  }
  std::vector<int64_t> action_bits, action_offsets{0};
  for (auto &a : actions) {
    for (size_t j = 0; j < a.size(); ++j)
      set_bits(action_bits, integer_bits(a[j]), j * 64);
    action_offsets.push_back(action_bits.size());
  }
  std::vector<int64_t> gather, cumulative{0}, lengths, encoded_entities, encoded_globals;
  for (size_t j = 0; j < missing.size(); ++j) {
    auto length = roots[missing[j]]->rows.size();
    encoded_globals.push_back(cumulative.back());
    gather.push_back(j);
    for (size_t k = 0; k < length; ++k) {
      gather.push_back(missing.size() + missing_offsets[j] + k);
      encoded_entities.push_back(cumulative.back() + k + 1);
    }
    lengths.push_back(length + 1);
    cumulative.push_back(cumulative.back() + length + 1);
  }
  auto row_input = rows.pack(input);
  auto a_bits = input.add(action_bits), a_offsets = input.add(action_offsets);
  GatherLayout er_ids(input, er, rows.rows.size(), !caching);
  GatherLayout global_ids(input, global_rows, rows.rows.size(), !caching);
  auto kind = input.add(kind_ids), root_gather = input.add(gather);
  auto root_cumulative = input.add(cumulative), prefix_ids = input.add(pr);
  auto prefix_offset = input.add(prefix_offsets);
  GatherLayout node_root(input, node_roots, roots.size(), !caching);
  auto encoded_entity = input.add(encoded_entities), encoded_global = input.add(encoded_globals);
  std::vector<int64_t> references{0};
  if (!caching) {
    std::unordered_map<int64_t, int64_t> lookup{{0, 0}};
    auto remap = [&](std::vector<int64_t> &ids) {
      for (auto &id : ids) {
        auto [it, inserted] = lookup.try_emplace(id, references.size());
        if (inserted)
          references.push_back(id);
        id = it->second;
      }
    };
    remap(source);
    remap(target);
  }
  GatherLayout referenced(input, references, offset + 1, !caching);
  GatherLayout owner(input, owners, nodes.size(), !caching);
  GatherLayout candidate(input, candidates, actions.size(), !caching);
  GatherLayout key_refs(input, key_ids, keys.size(), !caching);
  auto live = input.add(live_positions);
  auto entity_count = caching ? offset + 1 : int64_t(references.size());
  GatherLayout src(input, source, entity_count, !caching),
      dst(input, target, entity_count, !caching);
  // 一份整数表和一份 BF16 表上传，后续算子只使用设备端视图。
  input.upload(device);
  // 返回的闭包只持本批输入；后台打包和上传不读取或更新浮点权重。
  return [=, this, input = std::move(input)]() mutable -> std::pair<T, T> {
    auto row_embeddings =
        entity(input.get(row_input[0]), input.get(row_input[1]), input.bf(row_input[2], 52));
    T mixed, r;
    if (!missing.empty()) {
      auto cu = input.get(root_cumulative).to(torch::kInt32);
      auto x =
          encode_table(er_ids.bind(input)(row_embeddings), global_ids.bind(input)(row_embeddings),
                       input.get(kind), input.get(root_gather), cu, lengths);
      auto context_part = project_root_context(x.index_select(0, input.get(encoded_global)));
      TORCH_CHECK(!caching, "root cache changed during inference");
      // 学习只进行两次整表 gather；逐根取视图会在反向为每根重复分配整表梯度。
      mixed = x.index_select(0, input.get(encoded_entity));
      r = context_part;
      encoded_roots += missing.size();
      computed_tokens += cumulative.back();
    }
    if (!caching) {
      scored_candidates += owners.size();
      projected_candidates += actions.size();
    }
    if (caching) {
      mixed = torch::cat(root_tokens, 0);
      r = torch::stack(root_context, 0);
    }
    auto projected = torch::gelu(action(input.get(a_bits), input.get(a_offsets)));
    auto entity_table = torch::cat({torch::zeros({1, mixed.size(1)}, mixed.options()), mixed}, 0);
    if (!caching)
      entity_table = project_entities(referenced.bind(input)(entity_table));
    return score_table({row_embeddings, input.get(prefix_ids), input.get(prefix_offset), r,
                        entity_table, projected, node_root.bind(input), owner.bind(input),
                        candidate.bind(input), src.bind(input), dst.bind(input),
                        key_refs.bind(input), input.get(live), full_candidates});
  };
}

std::vector<double> canonical_logps(const std::vector<uint16_t> &bits) {
  double sum = 0;
  for (auto b : bits) {
    BF v;
    v.x = b;
    sum += double(float(v));
  }
  if (!(sum > 0) || !std::isfinite(sum))
    throw std::runtime_error("empty categorical distribution");
  std::vector<double> out;
  for (auto b : bits) {
    BF v;
    v.x = b;
    out.push_back(float(v) > 0 ? std::log(double(float(v)) / sum) : -INFINITY);
  }
  return out;
}
std::vector<std::vector<uint16_t>> probabilities(const T &logits,
                                                 const std::vector<std::shared_ptr<Node>> &nodes) {
  // 支持集分段归约，跨角色和批次无需补零或拆批。
  return cpu_masses(policy_table(logits, nodes, false));
}

Optimizer::Optimizer(Network &model) {
  model->pack_weights();
  int64_t count = 0;
  for (auto &p : model->parameters())
    count += p.numel();
  for (int i = 0; i < 5; ++i)
    state.push_back(torch::zeros({count}, model->parameters()[0].options()));
  gradients = torch::zeros({count}, model->parameters()[0].options());
  native_update = make_update_kernel(model->parameters()[0].device(), targets);
}
void Optimizer::prepare_gradients(Network &model) {
  for (auto &p : model->parameters())
    p.mutable_grad() = T();
}
T Optimizer::collect_gradients(Network &model) {
  torch::NoGradGuard no_grad;
  std::vector<T> grads;
  for (auto &p : model->parameters())
    grads.push_back(p.grad().defined() ? p.grad().reshape({-1})
                                       : torch::zeros_like(p).reshape({-1}));
  // 每批完整反向覆盖梯度；整包写入固定存储，不逐参数执行零梯度累加。
  torch::cat_out(gradients, grads);
  return gradients;
}
namespace {
T reference_update(T g, const T &p, const std::vector<T> &state, const T &norm, double lr,
                   uint64_t step) {
  g = g / norm.clamp_min(1);
  auto add = [](const T &x, const T &residual, const T &change) {
    auto r = residual + change;
    auto y = x + r;
    return std::make_pair(y, r + (x - y));
  };
  auto m = add(state[0], state[2], (g - state[0]) * 0.1);
  auto v = add(state[1], state[3], (g.square() - state[1]) * 0.001);
  auto change = -lr * (m.first / (1 - std::pow(0.9, double(step))) /
                           ((v.first / (1 - std::pow(0.999, double(step)))).sqrt() + 1e-8) +
                       0.01 * p);
  auto updated = add(p, state[4], change);
  auto valid = torch::stack({torch::isfinite(norm).all(), torch::isfinite(g).all(),
                             torch::isfinite(updated.first).all(), torch::isfinite(m.first).all(),
                             torch::isfinite(v.first).all(), torch::isfinite(updated.second).all(),
                             torch::isfinite(m.second).all(), torch::isfinite(v.second).all()})
                   .all();
  if (!valid.item<bool>())
    throw std::runtime_error("non-finite BF16 update; parameters and optimizer not committed");
  return torch::stack({updated.first, m.first, v.first, m.second, v.second, updated.second});
}
} // namespace
double Optimizer::step(Network &model, const T &loss) {
  RECORD_FUNCTION("hj.optimizer", std::vector<c10::IValue>());
  torch::NoGradGuard no_grad;
  auto g = collect_gradients(model), p = model->packed_weights;
  auto norm = g.square().sum().sqrt();
  UpdateResult result;
  if (native_update) {
    result = native_update(g, p, state, norm, lr, steps + 1, loss);
    p.unsafeGetTensorImpl()->bump_version();
  } else {
    if (loss.defined() && !torch::isfinite(loss).item<bool>())
      throw std::runtime_error("non-finite learning loss");
    result = {reference_update(g, p, state, norm, lr, steps + 1),
              loss.defined() ? loss.item<double>() : 0};
    p.copy_(result.values[0]);
  }
  for (size_t i = 0; i < state.size(); ++i)
    if (!result.states_committed)
      state[i].copy_(result.values[i + 1]);
  ++steps;
  ++model->revision;
  return result.loss;
}
void Optimizer::save(torch::serialize::OutputArchive &out) {
  out.write("steps", torch::tensor(int64_t(steps), torch::kInt64));
  for (size_t i = 0; i < state.size(); ++i)
    out.write("state" + std::to_string(i),
              state[i].to(state[i].options().device(torch::kCPU), false, true));
}
void Optimizer::load(torch::serialize::InputArchive &in, Network &model) {
  T s;
  in.read("steps", s);
  steps = s.item<int64_t>();
  for (size_t i = 0; i < state.size(); ++i) {
    T v;
    in.read("state" + std::to_string(i), v);
    TORCH_CHECK(v.scalar_type() == torch::kBFloat16 && v.sizes() == state[i].sizes() &&
                    torch::isfinite(v).all().item<bool>(),
                "invalid optimizer state");
    state[i].copy_(v.to(model->parameters()[0].device()));
  }
}
namespace {
struct TraceCarry {
  double value = 0, target = 0, correction = 0;
  std::array<double, 2> advance(double current, double reward, bool terminal, double ratio) {
    double discount = terminal ? 0 : 1;
    correction = ratio * (reward + discount * value - current) + discount * ratio * correction;
    double next_target = current + correction;
    double advantage = ratio * (reward + discount * target - current);
    value = current;
    target = next_target;
    return {target, advantage};
  }
};
// CPU 路径也是设备端目标递推的数值参照；记录表仅用整数保存索引及行为概率的原始位。
T reference_targets(const T &mass, const T &value_tensor, const T &record_tensor,
                    const T &block_tensor) {
  auto count = record_tensor.numel() / 6;
  auto result = torch::empty({3, count}, value_tensor.options());
  auto p = mass.data_ptr<BF>(), values = value_tensor.data_ptr<BF>(), out = result.data_ptr<BF>();
  auto records = record_tensor.data_ptr<int64_t>(), blocks = block_tensor.data_ptr<int64_t>();
  for (int64_t bi = 0; bi < block_tensor.numel() / 3; ++bi) {
    auto begin = blocks[bi * 3], end = blocks[bi * 3 + 1], boot = blocks[bi * 3 + 2];
    double bootstrap = boot < 0 ? 0 : float(values[boot]);
    TraceCarry carry{bootstrap, bootstrap, 0};
    for (auto i = end; i-- > begin;) {
      auto s = records + i * 6;
      double sum = 0;
      for (int64_t j = 0; j < s[1]; ++j)
        sum += float(p[s[0] + j]);
      auto chosen = float(p[s[0] + s[2]]);
      double logp = chosen > 0 ? std::log(double(chosen) / sum) : -INFINITY;
      double ratio = std::min(1.0, std::exp(logp - std::bit_cast<double>(s[5])));
      auto result = carry.advance(float(values[i]), double(s[3]), s[4] != 0, ratio);
      out[i] = BF(float(result[0]));
      out[count + i] = BF(float(result[1]));
      out[count * 2 + i] = BF(float(logp));
    }
  }
  return result;
}
} // namespace
TrainResult learn(Network &model, Optimizer &optimizer, const std::vector<Block> &blocks,
                  Forward prepared) {
  RECORD_FUNCTION("hj.learn", std::vector<c10::IValue>());
  std::vector<std::shared_ptr<Node>> nodes;
  for (auto &b : blocks)
    for (auto &s : b.steps) {
      nodes.push_back(s.node);
    }
  auto count = nodes.size();
  if (!count)
    throw std::runtime_error("empty learning batch");
  std::vector<int64_t> boot;
  for (auto &b : blocks) {
    boot.push_back(b.bootstrap ? int64_t(nodes.size()) : -1);
    if (b.bootstrap)
      nodes.push_back(b.bootstrap);
  }
  std::vector<int64_t> records, trajectories, selected;
  size_t offset = 0;
  int64_t action_offset = 0;
  for (size_t bi = 0; bi < blocks.size(); ++bi) {
    auto &block = blocks[bi];
    trajectories.insert(trajectories.end(),
                        {int64_t(offset), int64_t(offset + block.steps.size()), boot[bi]});
    for (auto &s : block.steps) {
      auto n = int64_t(s.node->mask.size());
      TORCH_CHECK(s.selected >= 0 && s.selected < n && s.node->mask[s.selected],
                  "invalid learning action");
      records.insert(records.end(), {action_offset, n, s.selected, s.reward, s.terminal,
                                     std::bit_cast<int64_t>(s.logp)});
      selected.push_back(action_offset + s.selected);
      action_offset += n;
    }
    offset += block.steps.size();
  }
  Inputs input;
  auto record_slice = input.add(records), block_slice = input.add(trajectories),
       selected_slice = input.add(selected);
  // 全部价值、目标和损失共用这一次前向；bootstrap 节点不执行未使用的候选策略分支。
  optimizer.prepare_gradients(model);
  auto output = prepared ? prepared() : model->forward(nodes, count);
  auto table = policy_table(output.first, std::span(nodes).first(count), true, &input);
  auto target_table = optimizer.targets
                          ? optimizer.targets(table.mass, output.second, input.get(record_slice),
                                              input.get(block_slice))
                          : reference_targets(table.mass, output.second, input.get(record_slice),
                                              input.get(block_slice));
  auto logp = table.logp.index_select(0, input.get(selected_slice));
  auto corrected = logp + (target_table[2] - logp.detach());
  auto targets = target_table[0], advantages = target_table[1];
  // 新策略可将旧行为动作的质量舍入为零；V-trace 权重随之为零，策略项也应严格为零。
  auto policy_logp = torch::where(advantages.ne(0), corrected, torch::zeros_like(corrected));
  auto loss =
      (-policy_logp * advantages + 0.5 * (output.second.narrow(0, 0, count) - targets).square() -
       0.01 * table.entropy.narrow(0, 0, count))
          .mean();
  if (!optimizer.native_update && !torch::isfinite(loss).item<bool>())
    throw std::runtime_error("non-finite learning loss");
  loss.backward();
  return {count, optimizer.step(model, loss)};
}
TrainResult learn_episodes(Network &model, Optimizer &optimizer, const std::vector<Block> &episodes,
                           uint64_t batch) {
  // 完整轨迹 V-trace：先冻结当前参数计算整局目标，再按显存小批反向，整批只提交一次。
  // 分块边界不重置回报递推。参考 DeepNash 补充材料的完整对局/反向分块组织。
  std::vector<const Step *> steps;
  std::vector<size_t> ends;
  for (auto &episode : episodes) {
    TORCH_CHECK(!episode.steps.empty() && !episode.bootstrap && episode.steps.back().terminal,
                "terminal learning requires complete games");
    for (size_t i = 0; i < episode.steps.size(); ++i) {
      auto &s = episode.steps[i];
      TORCH_CHECK(s.terminal == (i + 1 == episode.steps.size()) &&
                      (s.terminal ? s.reward >= -1 && s.reward <= 1 : s.reward == 0),
                  "invalid terminal trajectory");
      TORCH_CHECK(s.selected >= 0 && size_t(s.selected) < s.node->mask.size() &&
                      s.node->mask[s.selected] && std::isfinite(s.logp),
                  "invalid terminal behavior action");
      steps.push_back(&s);
    }
    ends.push_back(steps.size());
  }
  auto count = steps.size();
  TORCH_CHECK(count && batch, "empty terminal learning batch");
  auto device = model->parameters()[0].device();
  std::vector<BF> values(count), targets(count), advantages(count), logps(count), outcomes(count);
  std::vector<double> canonical(count);
  auto nodes_at = [&](size_t begin) {
    std::vector<std::shared_ptr<Node>> nodes;
    for (auto i = begin; i < std::min(count, begin + batch); ++i)
      nodes.push_back(steps[i]->node);
    return nodes;
  };
  for (size_t begin = 0; begin < count; begin += batch) {
    auto nodes = nodes_at(begin);
    // 与反向使用同一前向后端；仅本小批保留计算图，随后立即释放。
    auto output = model->forward(nodes);
    auto masses = probabilities(output.first.detach(), nodes);
    auto v = output.second.detach().to(torch::kCPU).contiguous();
    std::copy_n(v.data_ptr<BF>(), nodes.size(), values.begin() + begin);
    for (size_t j = 0; j < nodes.size(); ++j) {
      canonical[begin + j] = canonical_logps(masses[j])[steps[begin + j]->selected];
      logps[begin + j] = BF(float(canonical[begin + j]));
    }
  }
  size_t begin = 0;
  for (auto end : ends) {
    TraceCarry carry;
    std::fill(outcomes.begin() + begin, outcomes.begin() + end, BF(float(steps[end - 1]->reward)));
    for (auto i = end; i-- > begin;) {
      auto &s = *steps[i];
      auto ratio = std::min(1.0, std::exp(canonical[i] - s.logp));
      auto result = carry.advance(float(values[i]), s.reward, s.terminal, ratio);
      targets[i] = BF(float(result[0]));
      advantages[i] = BF(float(result[1]));
    }
    begin = end;
  }
  auto accumulated = torch::zeros_like(optimizer.gradients);
  auto residual = torch::zeros_like(accumulated);
  // 整局标量目标一次上传；每个反向小批只取设备视图。
  auto device_targets = bf_tensor(targets, 1, device).view({-1});
  auto device_advantages = bf_tensor(advantages, 1, device).view({-1});
  auto device_logps = bf_tensor(logps, 1, device).view({-1});
  auto device_outcomes = bf_tensor(outcomes, 1, device).view({-1});
  std::vector<T> statistics;
  TrainResult result{count, 0, episodes.size()};
  for (size_t at = 0; at < count; at += batch) {
    auto nodes = nodes_at(at);
    auto size = nodes.size();
    optimizer.prepare_gradients(model);
    auto output = model->forward(nodes);
    auto table = policy_table(output.first, nodes);
    std::vector<int64_t> selected;
    int64_t offset = 0;
    for (size_t j = 0; j < size; ++j) {
      selected.push_back(offset + steps[at + j]->selected);
      offset += nodes[j]->mask.size();
    }
    auto part = [&](const T &data) { return data.narrow(0, at, size); };
    auto value_targets = part(device_targets), adv = part(device_advantages);
    auto chosen = table.logp.index_select(0, indices(selected, device));
    auto corrected = chosen + (part(device_logps) - chosen.detach());
    auto policy_logp = torch::where(adv.ne(0), corrected, torch::zeros_like(corrected));
    auto error = (output.second - value_targets).square();
    auto loss = (-policy_logp * adv + 0.5 * error - 0.01 * table.entropy).mean();
    loss.backward();
    torch::NoGradGuard guard;
    double weight = double(size) / count;
    // 保留各小批原来的 BF16 统计值，整局结束后一次回读并按原顺序加权。
    statistics.push_back(loss.detach());
    statistics.push_back((output.second - part(device_outcomes)).square().mean());
    statistics.push_back(table.entropy.mean());
    // BF16 补偿累加避免长局的小批梯度被反复舍入吞掉；优化器状态仍保持原五组。
    compensated_gradient_add(accumulated, residual, optimizer.collect_gradients(model), weight);
  }
  auto host_statistics = torch::stack(statistics).to(torch::kCPU).contiguous();
  auto stats = host_statistics.data_ptr<BF>();
  for (size_t at = 0, j = 0; at < count; at += batch, j += 3) {
    double weight = double(std::min(count - at, size_t(batch))) / count;
    result.loss += float(stats[j]) * weight;
    result.value_mse += float(stats[j + 1]) * weight;
    result.policy_entropy += float(stats[j + 2]) * weight;
  }
  {
    torch::NoGradGuard guard;
    result.gradient_norm = accumulated.to(torch::kFloat32).norm().item<double>();
    int64_t offset = 0;
    for (auto &p : model->parameters()) {
      p.mutable_grad() = accumulated.narrow(0, offset, p.numel()).view(p.sizes());
      offset += p.numel();
    }
  }
  result.loss = optimizer.step(model, torch::tensor(result.loss, bfopts().device(device)));
  return result;
}
void copy_weights(Network &source, Network &destination) {
  torch::NoGradGuard guard;
  source->pack_weights();
  destination->pack_weights();
  destination->packed_weights.copy_(source->packed_weights);
  ++destination->revision;
}
void model_checks(const torch::Device &device) {
  {
    torch::NoGradGuard guard;
    auto sum = torch::randn({4097}, bfopts().device(device));
    auto residual = torch::randn_like(sum) / 256;
    auto expected_sum = sum.clone(), expected_residual = residual.clone();
    for (double weight : {1.0, 0.0037, 0.125, 0.00001}) {
      auto gradient = torch::randn_like(sum);
      auto change = expected_residual + gradient * weight;
      auto updated = expected_sum + change;
      expected_residual = change + (expected_sum - updated);
      expected_sum = updated;
      compensated_gradient_add(sum, residual, gradient, weight);
      TORCH_CHECK(torch::equal(sum, expected_sum) && torch::equal(residual, expected_residual),
                  "fused terminal gradient accumulation changed BF16 rounding");
    }
  }
  {
    auto x = torch::tensor({-5.0, 5.0}, bfopts().device(device)).requires_grad_(true);
    auto y = ValueTanh::apply(x);
    TORCH_CHECK(y.abs().eq(1).all().item<bool>());
    y.sum().backward();
    auto expected = 1 - x.detach().to(torch::kFloat64).tanh().square();
    TORCH_CHECK(torch::allclose(x.grad().to(torch::kFloat64), expected, 0.01, 1e-7),
                "rounded BF16 value saturation erased a finite tanh derivative");
    TraceCarry carry;
    for (int i = 256; i-- > 0;) {
      auto target = carry.advance(0, i == 255 ? 1 : 0, i == 255, 1);
      TORCH_CHECK(target[0] == 1 && target[1] == 1,
                  "terminal reward did not reach the start of the trajectory");
    }
  }
  {
    // 非二次幂候选数会让旧 BF16 分步归一化在最大熵点产生伪梯度。
    auto lengths = indices({59, 125}, device);
    auto owners =
        torch::cat({torch::zeros({59}, lengths.options()), torch::ones({125}, lengths.options())});
    auto valid = torch::ones({184}, lengths.options()).ne(0);
    auto logits = torch::zeros({184}, bfopts().device(device)).requires_grad_(true);
    auto mass = policy_mass(logits, owners, lengths, valid);
    auto logp = mass.log() - segments(mass, "sum", lengths).log().index_select(0, owners);
    (0.01 * (mass * logp).sum()).backward();
    TORCH_CHECK(logits.grad().count_nonzero().item<int64_t>() == 0,
                "maximum-entropy policy has a spurious common-score gradient");
    logits = (torch::arange(184, lengths.options()).remainder(13).to(torch::kBFloat16) / 8)
                 .requires_grad_(true);
    auto upstream =
        torch::sin(torch::arange(184, lengths.options()).to(torch::kFloat32)).to(torch::kBFloat16);
    mass = policy_mass(logits, owners, lengths, valid);
    mass.backward(upstream);
    // 独立参照使用 LibTorch 的双精度 softmax 自动求导，不重复自定义反向公式。
    auto reference = logits.detach().to(torch::kFloat64).requires_grad_(true);
    auto reference_mass = torch::cat(
        {reference.narrow(0, 0, 59).softmax(0), reference.narrow(0, 59, 125).softmax(0)});
    reference_mass.backward(upstream.to(torch::kFloat64));
    TORCH_CHECK(torch::allclose(logits.grad().to(torch::kFloat64), reference.grad(), 0.02, 1e-5),
                "BF16 policy gradient differs from double-precision softmax");
  }
  if (device.is_cuda()) {
    {
      torch::NoGradGuard guard;
      auto qkv = torch::randn({250, 3, 6, 64}, bfopts().device(device));
      auto cu = torch::tensor({0, 3, 58, 250}, torch::kInt32).to(device);
      auto used = torch::tensor({3, 55, 128}, torch::kInt32).to(device);
      auto expected = std::get<0>(at::_efficient_attention_forward(
                                      qkv.select(1, 0).unsqueeze(0), qkv.select(1, 1).unsqueeze(0),
                                      qkv.select(1, 2).unsqueeze(0), {}, cu, cu, 192, 192, 0, 0,
                                      false, {}, used))
                          .squeeze(0);
      auto actual = short_attention_cuda(qkv, cu, used);
      TORCH_CHECK(
          torch::allclose(actual.narrow(0, 0, 186), expected.narrow(0, 0, 186), 0.02, 0.016),
          "short attention changed real variable-length outputs");
      TORCH_CHECK(actual.narrow(0, 186, 64).count_nonzero().item<int64_t>() == 0,
                  "short attention exposed padding");
    }
    for (auto rows : {1, 31, 32, 33, 55, 1760}) {
      auto options = bfopts().device(device);
      auto a = torch::randn({rows, 384}, options).requires_grad_(true);
      auto b = torch::randn_like(a).requires_grad_(true);
      torch::nn::LayerNorm norm(torch::nn::LayerNormOptions({384}));
      norm->to(device, torch::kBFloat16);
      auto fused = ResidualNorm::apply(a, b, norm->weight, norm->bias, norm->options.eps());
      auto x = a + b, y = bf16_norm(x, norm);
      TORCH_CHECK(torch::equal(fused[0], x) && torch::allclose(fused[1], y, 0.008, 0.008),
                  "fused residual rounding or normalization differs");
      auto gx = torch::randn_like(x), gy = torch::randn_like(y);
      auto actual =
          torch::autograd::grad({fused[0], fused[1]}, {a, b, norm->weight, norm->bias}, {gx, gy});
      auto expected = torch::autograd::grad({x, y}, {a, b, norm->weight, norm->bias}, {gx, gy});
      for (size_t i = 0; i < actual.size(); ++i)
        TORCH_CHECK(torch::allclose(actual[i], expected[i], 0.02, 0.016),
                    "fused residual normalization gradient differs: ", i);
    }
  }
  {
    Inputs input;
    GatherLayout gather(input, {2, 0, 2, 2, 3, 0}, 5, true);
    input.upload(device);
    auto x = torch::ones({5, 32}, bfopts().device(device)).requires_grad_(true);
    auto weights = torch::arange(1, 7, x.options()).unsqueeze(1) / 8;
    (gather.bind(input)(x) * weights).sum().backward();
    std::vector<BF> expected(5 * 32, BF(0));
    for (int j = 0; j < 32; ++j) {
      expected[j] = BF(1);
      expected[64 + j] = BF(1);
      expected[96 + j] = BF(0.625f);
    }
    TORCH_CHECK(torch::equal(x.grad(), bf_tensor(expected, 32, device)),
                "CSR gather transpose lost repeated or empty rows");
  }
  {
    // 负整数、高位身份、存在位和连续字段均与显式零一矩阵的线性映射对照。
    TypedProjection projection(32, true);
    projection->to(device);
    Rows rows;
    Row a, b;
    a.ids[2] = 4097;
    a.ids[59] = -7;
    a.present = (uint64_t(1) << 51) | 1;
    a.integers = 1;
    a.values[1] = BF(0.75f).x;
    b = a;
    b.ids[2] = 4096;
    TORCH_CHECK(rows.add(a) == 0 && rows.add(b) == 1 && rows.add(a) == 0);
    rows.add(Row{});
    Inputs input;
    auto packed = rows.pack(input);
    input.upload(device);
    auto sparse = projection(input.get(packed[0]), input.get(packed[1]), input.bf(packed[2], 52));
    std::vector<BF> dense(rows.rows.size() * RowFeatures, BF(0));
    for (size_t i = 0; i < rows.rows.size(); ++i) {
      auto &r = rows.rows[i];
      for (int64_t field = 0; field < 60; ++field)
        for (int64_t bit = 0; bit < 64; ++bit)
          dense[i * RowFeatures + field * 64 + bit] =
              BF(float((integer_bits(r.ids[field]) >> bit) & 1));
      for (int64_t j = 0; j < 52; ++j) {
        dense[i * RowFeatures + 3840 + j] = BF(float((r.present >> j) & 1));
        dense[i * RowFeatures + 3892 + j] = BF(float((r.integers >> j) & 1));
        dense[i * RowFeatures + 3944 + j].x = r.values[j];
      }
    }
    auto weight =
        torch::cat({projection->bits->weight.transpose(0, 1), projection->continuous->weight}, 1);
    auto reference =
        torch::linear(bf_tensor(dense, RowFeatures, device), weight, projection->continuous->bias);
    TORCH_CHECK(torch::allclose(sparse, reference, 0.04, 0.004),
                "typed sparse projection changed the represented linear map");
    sparse.sum().backward();
    TORCH_CHECK(projection->bits->weight.grad().scalar_type() == torch::kBFloat16 &&
                torch::isfinite(projection->bits->weight.grad()).all().item<bool>());
  }
  Network model(32);
  model->to(device);
  if (device.is_cuda()) {
    torch::NoGradGuard no_grad;
    auto options = model->parameters()[0].options();
    auto entities = torch::randn({7, 32}, options), globals = torch::randn({1, 32}, options);
    auto actual = model->encode_table(entities.narrow(0, 0, 2), globals, indices({0, 1}, device),
                                      indices({0, 1, 2}, device),
                                      indices({0, 3}, device).to(torch::kInt32), {3});
    auto padded = model->encode_table(entities, globals, indices({0, 1, 2, 3, 4, 5, 6}, device),
                                      indices({0, 1, 2, 3, 4, 5, 6, 7}, device),
                                      indices({0, 8}, device).to(torch::kInt32), {8},
                                      indices({3}, device).to(torch::kInt32));
    TORCH_CHECK(torch::allclose(actual, padded.narrow(0, 0, 3), 0.02, 0.005),
                "capacity padding changed real root attention");
  }
  Optimizer optimizer(model);
  InputPool inputs;
  auto n = std::make_shared<Node>();
  n->root = std::make_shared<Root>();
  n->root->game = 1;
  n->root->id = 1;
  Row row;
  row.ids[0] = 1;
  row.ids[2] = 4097;
  row.present = 1;
  row.integers = 1;
  row.ids[8] = 7;
  n->root->rows = {inputs.rows.get(row)};
  n->root->kinds = {128};
  n->mask = {1, 1, 0};
  n->sources = {0, -1, 0};
  n->targets = {-1, 0, 0};
  for (size_t i = 0; i < 3; ++i) {
    Action action{};
    action[i] = 1;
    n->candidates.push_back(inputs.actions.get(action));
  }
  n->context = make_context(inputs, {}, {}, {});
  auto outputs = model->forward({n, n});
  TORCH_CHECK(outputs.first.scalar_type() == torch::kBFloat16 &&
              outputs.second.scalar_type() == torch::kBFloat16);
  auto probs = probabilities(outputs.first.detach(), {n, n});
  TORCH_CHECK(probs[0] == probs[1] && probs[0][2] == 0);
  auto logs = canonical_logps(probs[0]);
  TORCH_CHECK(std::abs(std::exp(logs[0]) + std::exp(logs[1]) - 1) < 1e-12);
  TORCH_CHECK(model->blocks->size() == EncoderLayers);
  {
    Network whole(32), split(32);
    whole->to(device);
    split->to(device);
    copy_weights(model, whole);
    copy_weights(model, split);
    Optimizer whole_optimizer(whole), split_optimizer(split);
    Block episode;
    for (int i = 0; i < 7; ++i)
      episode.steps.push_back({n, 0, logs[0], 0, i == 6, int8_t(i == 6 ? 1 : 0)});
    auto before = whole->forward({n}).second.item<double>();
    auto one = learn_episodes(whole, whole_optimizer, {episode}, 7);
    auto many = learn_episodes(split, split_optimizer, {episode}, 2);
    TORCH_CHECK(one.consumed == 7 && many.consumed == 7 && whole_optimizer.steps == 1 &&
                    split_optimizer.steps == 1 && many.terminal_games == 1,
                "memory microbatches changed terminal consumption or update count");
    TORCH_CHECK(whole->forward({n}).second.item<double>() > before &&
                    split->forward({n}).second.item<double>() > before,
                "true winning return did not raise predicted value");
    auto full_gradient = whole_optimizer.gradients.to(torch::kFloat32);
    auto split_gradient = split_optimizer.gradients.to(torch::kFloat32);
    auto relative_error = (full_gradient - split_gradient).norm().item<double>() /
                          full_gradient.norm().item<double>();
    // 不同矩阵批形状的 BF16 舍入会影响近零分量；约束整条梯度的相对误差。
    TORCH_CHECK(
        relative_error < 0.02,
        "microbatch accumulation changed the full-episode gradient; relative L2=", relative_error,
        "; max absolute=", (full_gradient - split_gradient).abs().max().item<double>());
    auto original = split->packed_weights.clone();
    episode.steps.back().terminal = false;
    bool rejected = false;
    try {
      learn_episodes(split, split_optimizer, {episode}, 2);
    } catch (const c10::Error &) {
      rejected = true;
    }
    TORCH_CHECK(rejected && torch::equal(original, split->packed_weights) &&
                    split_optimizer.steps == 1,
                "unfinished episode entered training or changed parameters");
  }
  {
    // 相同实体引用下，候选差分必须反传到上下文和注意力，防止共享偏置被 softmax 消掉。
    auto probe = std::make_shared<Node>(*n);
    probe->sources.assign(3, -1);
    probe->targets.assign(3, -1);
    probe->context = make_context(inputs, {}, {2, 5, 1}, {});
    auto logits = model->forward({probe}).first;
    (logits[0] - logits[1]).backward();
    auto context_gradient = model->context->weight.grad();
    auto attention_gradient = model->blocks[0]->as<TransformerBlock>()->qkv->weight.grad();
    TORCH_CHECK(context_gradient.scalar_type() == torch::kBFloat16 &&
                    context_gradient.abs().sum().item<double>() > 0,
                "candidate relative scores do not depend on context");
    TORCH_CHECK(attention_gradient.scalar_type() == torch::kBFloat16 &&
                    attention_gradient.abs().sum().item<double>() > 0,
                "policy gradient does not reach Transformer attention");
    model->zero_grad();
    auto other = std::make_shared<Node>(*n);
    other->root = std::make_shared<Root>(*n->root);
    other->root->id = 2;
    other->root->rows.push_back(inputs.rows.get(row));
    other->root->kinds.push_back(129);
    auto alone = model->forward({n});
    auto together = model->forward({n, other});
    TORCH_CHECK(torch::allclose(alone.first, together.first.narrow(0, 0, 3), 0.02, 0.002),
                "attention padding or another root changed isolated scores");
  }
  {
    torch::NoGradGuard guard;
    auto cold = model->forward({n});
    auto encoded = model->encoded_roots.load();
    auto warm = model->forward({n});
    TORCH_CHECK(model->encoded_roots == encoded && model->cache_hits > 0 &&
                torch::equal(cold.first, warm.first) && torch::equal(cold.second, warm.second));
    ++model->revision;
    model->forward({n});
    TORCH_CHECK(model->encoded_roots == encoded + 1,
                "published parameters did not invalidate root cache");
  }
  {
    Network skew(32);
    skew->to(device);
    copy_weights(model, skew);
    {
      torch::NoGradGuard guard;
      skew->prior->weight.mul_(1000000);
    }
    auto mass = probabilities(skew->forward({n}).first, {n})[0];
    size_t zero = 0;
    while (zero < mass.size() && (!n->mask[zero] || mass[zero] != 0))
      ++zero;
    TORCH_CHECK(zero < mass.size(), "zero-mass off-policy fixture has no zero action");
    Optimizer skew_optimizer(skew);
    Block old_behavior;
    old_behavior.steps = {{n, int64_t(zero), std::log(0.5), 0, true, 1}};
    auto update = learn(skew, skew_optimizer, {old_behavior});
    TORCH_CHECK(update.consumed == 1 && std::isfinite(update.loss),
                "zero-mass off-policy action produced a non-finite loss");
    for (auto &p : skew->parameters())
      TORCH_CHECK(torch::isfinite(p).all().item<bool>());
  }
  Block block;
  if (optimizer.native_update) {
    torch::NoGradGuard guard;
    auto full = model->forward({n, n}), pruned = model->forward({n, n}, 1);
    TORCH_CHECK(torch::equal(full.first.narrow(0, 0, n->mask.size()), pruned.first) &&
                    torch::equal(full.second, pruned.second),
                "bootstrap pruning changed live policy or value outputs");
    std::vector<BF> masses, values;
    std::vector<int64_t> records, trajectories;
    constexpr int64_t trajectories_count = 100, length = 17, count = trajectories_count * length;
    for (int64_t i = 0; i < count + trajectories_count; ++i)
      values.emplace_back(float(i % 11 - 5) / 8);
    for (int64_t bi = 0; bi < trajectories_count; ++bi) {
      trajectories.insert(trajectories.end(),
                          {bi * length, (bi + 1) * length, bi % 3 ? count + bi : -1});
      for (int64_t j = 0; j < length; ++j) {
        auto i = bi * length + j, actions = 1 + i % 23;
        records.insert(records.end(),
                       {int64_t(masses.size()), actions, i % actions,
                        j == length - 1 && bi % 3 == 0 ? (bi / 3) % 3 - 1 : 0,
                        j == length - 1 && bi % 3 == 0,
                        std::bit_cast<int64_t>(-std::log(double(actions)) + (i % 3 - 1) / 4.0)});
        for (int64_t k = 0; k < actions; ++k)
          masses.emplace_back(k % 7 == 0 && actions > 1 ? 0 : float(k % 5 + 1) / 64);
      }
    }
    auto m = bf_tensor(masses, 1, torch::kCPU).squeeze(1),
         v = bf_tensor(values, 1, torch::kCPU).squeeze(1);
    auto r = indices(records, torch::kCPU), b = indices(trajectories, torch::kCPU);
    auto expected = reference_targets(m, v, r, b);
    auto actual = optimizer.targets(m.to(device), v.to(device), r.to(device), b.to(device));
    TORCH_CHECK(torch::equal(expected.to(device), actual),
                "device V-trace targets changed canonical probability or trajectory recursion");
    auto p = torch::randn({4097}, model->parameters()[0].options());
    std::vector<T> state{torch::randn_like(p) / 8, (torch::rand_like(p) + 0.5) / 8,
                         torch::randn_like(p) / 4096, torch::randn_like(p) / 4096,
                         torch::randn_like(p) / 4096};
    for (uint64_t step : {1, 2, 64, 100000}) {
      auto g = torch::randn_like(p) / (step == 1 ? 128 : 16);
      auto norm = g.square().sum().sqrt();
      auto reference = reference_update(g, p, state, norm, optimizer.lr, step);
      optimizer.native_update(g, p, state, norm, optimizer.lr, step, {});
      auto fused = torch::stack({p, state[0], state[1], state[2], state[3], state[4]});
      TORCH_CHECK(torch::equal(reference, fused) && torch::equal(p, reference[0]),
                  "fused BF16 compensated update differs");
      p = reference[0];
      for (size_t i = 0; i < state.size(); ++i)
        state[i] = reference[i + 1];
    }
  }
  block.steps = {{n, 0, logs[0], 0, false, 0}, {n, 1, logs[1], 0, true, 1}};
  auto result = learn(model, optimizer, {block});
  TORCH_CHECK(result.consumed == 2 && optimizer.steps == 1);
  for (auto &p : model->parameters())
    TORCH_CHECK(p.scalar_type() == torch::kBFloat16 && torch::isfinite(p).all().item<bool>());
  for (auto &p : optimizer.state)
    TORCH_CHECK(p.scalar_type() == torch::kBFloat16);
  auto first = model->parameters()[0].detach().clone();
  auto state = optimizer.state[0].clone();
  auto old_steps = optimizer.steps;
  bool loss_rejected = false;
  try {
    optimizer.step(model, torch::full({}, NAN, first.options()));
  } catch (const std::runtime_error &) {
    loss_rejected = true;
  }
  TORCH_CHECK(loss_rejected && torch::equal(first, model->parameters()[0]) &&
                  torch::equal(state, optimizer.state[0]) && optimizer.steps == old_steps,
              "non-finite loss committed finite-gradient parameters");
  model->parameters()[0].mutable_grad() = torch::full_like(first, NAN);
  bool rejected = false;
  try {
    optimizer.step(model);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  TORCH_CHECK(rejected && torch::equal(first, model->parameters()[0]) &&
              torch::equal(state, optimizer.state[0]) && optimizer.steps == 1);
  model->parameters()[0].mutable_grad() = torch::full_like(first, 1e30);
  rejected = false;
  try {
    optimizer.step(model);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  TORCH_CHECK(rejected && torch::equal(first, model->parameters()[0]) && optimizer.steps == 1,
              "overflowing gradient norm was accepted");
  for (auto &p : model->parameters())
    p.mutable_grad() = T();
  torch::serialize::OutputArchive archive, weights, opt;
  model->save(weights);
  optimizer.save(opt);
  archive.write("model", weights);
  archive.write("optimizer", opt);
  std::vector<T> frozen_states;
  // 后台写盘期间学习可以继续；归档不得共享仍会原地更新的 CPU 状态。
  for (auto &state : optimizer.state) {
    frozen_states.push_back(state.clone());
    state.add_(1);
  }
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  archive.save_to(stream);
  for (size_t i = 0; i < optimizer.state.size(); ++i)
    optimizer.state[i].copy_(frozen_states[i]);
  stream.seekg(0);
  torch::serialize::InputArchive restored_archive, restored_weights, restored_opt;
  restored_archive.load_from(stream, device);
  restored_archive.read("model", restored_weights);
  restored_archive.read("optimizer", restored_opt);
  Network restored(32);
  restored->to(device);
  restored->load(restored_weights);
  Optimizer restored_optimizer(restored);
  restored_optimizer.load(restored_opt, restored);
  auto before = model->parameters(), after = restored->parameters();
  for (size_t i = 0; i < before.size(); ++i)
    TORCH_CHECK(torch::equal(before[i], after[i]));
  for (size_t i = 0; i < optimizer.state.size(); ++i)
    TORCH_CHECK(torch::equal(optimizer.state[i], restored_optimizer.state[i]));
  learn(model, optimizer, {block});
  learn(restored, restored_optimizer, {block});
  before = model->parameters();
  after = restored->parameters();
  for (size_t i = 0; i < before.size(); ++i)
    TORCH_CHECK(torch::equal(before[i], after[i]), "resumed BF16 update differs");
  std::cout << "six-layer Transformer, contextual candidate interaction, BF16 "
               "forward/backward/compensated optimizer, masked behavior probability and "
               "failure atomicity: passed\n";
}
} // namespace hj
