#include "model.h"
#include <ATen/Parallel.h>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>

namespace hj {
namespace {
using BF = c10::BFloat16;
auto bfopts() {
  return torch::TensorOptions().dtype(torch::kBFloat16);
}
T bf_tensor(const std::vector<BF> &v, int64_t columns, const torch::Device &device) {
  return torch::from_blob(const_cast<BF *>(v.data()), {int64_t(v.size() / columns), columns},
                          bfopts())
      .clone()
      .to(device);
}
T indices(const std::vector<int64_t> &v, const torch::Device &d) {
  return torch::tensor(v, torch::kInt64).to(d);
}
T bit_features(const T &integers, int64_t width) {
  auto shifts = torch::arange(width, integers.options());
  return torch::bitwise_and(torch::bitwise_right_shift(integers.unsqueeze(-1), shifts), 1)
      .to(torch::kBFloat16)
      .flatten(1);
}
T row_features(std::vector<Row> &rows, const torch::Device &d) {
  static_assert(sizeof(Row) == 600);
  int64_t count = rows.size();
  auto ids =
      torch::from_blob(rows[0].ids.data(), {count, 60}, {75, 1}, torch::kInt64).clone().to(d);
  auto present =
      torch::from_blob(&rows[0].present, {count, 1}, {75, 1}, torch::kInt64).clone().to(d);
  auto types =
      torch::from_blob(&rows[0].integers, {count, 1}, {75, 1}, torch::kInt64).clone().to(d);
  auto values =
      torch::from_blob(rows[0].values.data(), {count, 52}, {300, 1}, bfopts()).clone().to(d);
  // 原始整数直接搬到设备后展开为精确的 0/1 激活，避免宿主逐位扩容和大量拷贝。
  return torch::cat(
      {bit_features(ids, 64), values, bit_features(present, 52), bit_features(types, 52)}, 1);
}
T mean_table(const T &rows, const std::vector<int64_t> &owner, int64_t count) {
  std::vector<int64_t> lengths(count, 0);
  for (auto i : owner)
    ++lengths[i];
  return at::segment_reduce(rows, "mean", indices(lengths, rows.device()));
}
struct ArrayHash {
  size_t operator()(const std::array<int64_t, 64> &a) const {
    size_t h = 1469598103934665603ull;
    for (auto v : a) {
      h ^= size_t(v);
      h *= 1099511628211ull;
    }
    return h;
  }
};
struct PolicyTable {
  T mass, logp, entropy;
  std::vector<int64_t> lengths;
};
PolicyTable policy_table(const T &logits, const std::vector<std::shared_ptr<Node>> &nodes) {
  std::vector<int64_t> owner, lengths;
  std::vector<uint8_t> mask;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto n = nodes[i]->mask.size();
    lengths.push_back(n);
    owner.insert(owner.end(), n, i);
    mask.insert(mask.end(), nodes[i]->mask.begin(), nodes[i]->mask.end());
  }
  auto d = logits.device();
  auto index = indices(owner, d), length = indices(lengths, d);
  auto valid = torch::from_blob(mask.data(), {int64_t(mask.size())}, torch::kUInt8)
                   .clone()
                   .to(d, torch::kBool);
  auto masked = logits.masked_fill(~valid, -INFINITY);
  auto maximum = at::segment_reduce(masked.detach(), "max", length);
  auto mass = (masked - maximum.index_select(0, index)).exp();
  mass = mass / at::segment_reduce(mass, "sum", length).index_select(0, index);
  auto logp = mass.clamp_min(1e-38).log() -
              at::segment_reduce(mass, "sum", length).log().index_select(0, index);
  auto entropy = at::segment_reduce(-(mass * logp).masked_fill(~valid, 0), "sum", length);
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
NetworkImpl::NetworkImpl(int64_t w) : width(w) {
  entity = register_module("entity", torch::nn::Linear(RowFeatures, w));
  action = register_module("action", torch::nn::Linear(ActionFeatures, w));
  globals = register_module("globals", torch::nn::Linear(32, w));
  kinds = register_module("kinds", torch::nn::Embedding(256, w));
  mix = register_module("mix", torch::nn::Linear(w * 2, w));
  context = register_module("context", torch::nn::Linear(w * 2, w));
  policy = register_module("policy", torch::nn::Linear(w * 4, 1));
  value = register_module("value", torch::nn::Linear(w, 1));
  to(torch::kBFloat16);
}
std::pair<T, T> NetworkImpl::forward(const std::vector<std::shared_ptr<Node>> &nodes) {
  if (nodes.empty())
    throw std::runtime_error("empty model batch");
  auto device = parameters()[0].device();
  std::map<std::pair<uint64_t, uint64_t>, int64_t> root_ids;
  std::vector<std::shared_ptr<Root>> roots;
  std::vector<int64_t> node_roots, entity_owner, kind_ids, offsets;
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
  std::vector<Row> er;
  std::vector<BF> gf;
  std::vector<int64_t> missing, missing_offsets;
  std::vector<T> root_tokens(roots.size()), root_context(roots.size());
  int64_t offset = 0;
  bool caching = !torch::GradMode::is_enabled();
  for (size_t i = 0; i < roots.size(); ++i) {
    offsets.push_back(offset);
    offset += roots[i]->rows.size();
    if (caching && cache_valid(*roots[i])) {
      root_tokens[i] = roots[i]->cached_tokens;
      root_context[i] = roots[i]->cached_context;
      ++cache_hits;
      continue;
    }
    missing_offsets.push_back(kind_ids.size());
    auto missing_id = missing.size();
    missing.push_back(i);
    for (size_t j = 0; j < roots[i]->rows.size(); ++j) {
      er.push_back(roots[i]->rows[j]);
      kind_ids.push_back(roots[i]->kinds[j]);
      entity_owner.push_back(missing_id);
    }
    for (auto b : roots[i]->globals) {
      BF v;
      v.x = b;
      gf.push_back(v);
    }
  }
  if (!missing.empty()) {
    auto e = torch::gelu(entity(row_features(er, device)) + kinds(indices(kind_ids, device)));
    auto r = mean_table(e, entity_owner, missing.size()) + globals(bf_tensor(gf, 32, device));
    auto m =
        torch::gelu(mix(torch::cat({e, r.index_select(0, indices(entity_owner, device))}, -1)));
    r = r + mean_table(m, entity_owner, missing.size());
    for (size_t j = 0; j < missing.size(); ++j) {
      auto i = missing[j];
      root_tokens[i] = m.narrow(0, missing_offsets[j], roots[i]->rows.size());
      root_context[i] = r[j];
      if (caching) {
        roots[i]->cached_tokens = root_tokens[i].clone();
        roots[i]->cached_context = root_context[i].clone();
        roots[i]->cached_owner = this;
        roots[i]->cached_revision = revision.load();
      }
    }
    encoded_roots += missing.size();
  }
  auto mixed = torch::cat(root_tokens, 0), r = torch::stack(root_context, 0);
  // 参数行先非线性映射再聚合；顺序、所属参数和路径编号保留在整数位特征中。
  std::vector<Row> pr;
  std::vector<int64_t> prefix_owner;
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (auto &row : nodes[i]->prefix) {
      pr.push_back(row);
      prefix_owner.push_back(i);
    }
    auto path_row = [&](int64_t group, int64_t position, int64_t index, int64_t length,
                        int64_t type) {
      Row row;
      row.ids[0] = type;
      row.ids[2] = group;
      row.ids[3] = position;
      row.ids[4] = index;
      row.ids[5] = length;
      pr.push_back(row);
      prefix_owner.push_back(i);
    };
    for (size_t j = 0; j < nodes[i]->cursor.size(); ++j)
      path_row(0, j, nodes[i]->cursor[j], nodes[i]->cursor.size(), 25);
    for (size_t k = 0; k < nodes[i]->excluded.size(); ++k) {
      auto &p = nodes[i]->excluded[k];
      for (size_t j = 0; j < p.path.size(); ++j)
        path_row(k + 1, j, p.path[j], p.path.size(), 26);
      for (auto child : p.children)
        path_row(k + 1, p.path.size(), child, p.path.size(), 28);
    }
    // 空路径仍有一个显式节点，值网络不会把它与不存在的请求混合。
    path_row(0, 0, 0, 0, 27);
  }
  auto pc = mean_table(torch::gelu(entity(row_features(pr, device))), prefix_owner, nodes.size());
  auto ctx =
      torch::gelu(context(torch::cat({r.index_select(0, indices(node_roots, device)), pc}, -1)));
  std::unordered_map<std::array<int64_t, 64>, int64_t, ArrayHash> candidate_ids;
  std::vector<std::array<int64_t, 64>> actions;
  std::vector<int64_t> candidates, owners, source, target;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto &n = *nodes[i];
    auto off = offsets[node_roots[i]];
    for (size_t j = 0; j < n.candidates.size(); ++j) {
      auto it = candidate_ids.find(n.candidates[j]);
      if (it == candidate_ids.end()) {
        int64_t index = candidate_ids.size();
        candidate_ids[n.candidates[j]] = index;
        candidates.push_back(index);
        actions.push_back(n.candidates[j]);
      } else
        candidates.push_back(it->second);
      owners.push_back(i);
      source.push_back(n.sources[j] < 0 ? -1 : off + n.sources[j]);
      target.push_back(n.targets[j] < 0 ? -1 : off + n.targets[j]);
    }
  }
  auto raw_actions = torch::from_blob(actions.data(), {int64_t(actions.size()), 64}, torch::kInt64)
                         .clone()
                         .to(device);
  auto projected = torch::gelu(action(bit_features(raw_actions, 64)));
  auto entity_table = torch::cat({torch::zeros({1, width}, mixed.options()), mixed}, 0);
  for (auto &i : source)
    ++i;
  for (auto &i : target)
    ++i;
  auto logits = policy(torch::cat({ctx.index_select(0, indices(owners, device)),
                                   projected.index_select(0, indices(candidates, device)),
                                   entity_table.index_select(0, indices(source, device)),
                                   entity_table.index_select(0, indices(target, device))},
                                  -1))
                    .squeeze(-1);
  return {logits, value(ctx).squeeze(-1)};
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
  return cpu_masses(policy_table(logits, nodes));
}

Optimizer::Optimizer(Network &model) {
  int64_t count = 0;
  for (auto &p : model->parameters())
    count += p.numel();
  for (int i = 0; i < 5; ++i)
    state.push_back(torch::zeros({count}, model->parameters()[0].options()));
}
void Optimizer::step(Network &model) {
  torch::NoGradGuard guard;
  auto ps = model->parameters();
  std::vector<T> grads;
  std::vector<T> params;
  for (auto &p : ps) {
    if (!p.grad().defined())
      p.mutable_grad() = torch::zeros_like(p);
    grads.push_back(p.grad().reshape({-1}));
    params.push_back(p.reshape({-1}));
  }
  auto g = torch::cat(grads), p = torch::cat(params);
  auto norm = g.square().sum().sqrt();
  g = g / (norm.clamp_min(1));
  auto add = [](const T &x, const T &residual, const T &change) {
    auto r = residual + change;
    auto y = x + r;
    return std::make_pair(y, r + (x - y));
  };
  auto m = add(state[0], state[2], (g - state[0]) * 0.1);
  auto v = add(state[1], state[3], (g.square() - state[1]) * 0.001);
  auto change = -lr * (m.first / (1 - std::pow(0.9, double(steps + 1))) /
                           ((v.first / (1 - std::pow(0.999, double(steps + 1)))).sqrt() + 1e-8) +
                       0.01 * p);
  auto updated = add(p, state[4], change);
  auto valid = torch::stack({torch::isfinite(norm).all(), torch::isfinite(g).all(),
                             torch::isfinite(updated.first).all(), torch::isfinite(m.first).all(),
                             torch::isfinite(v.first).all(), torch::isfinite(updated.second).all(),
                             torch::isfinite(m.second).all(), torch::isfinite(v.second).all()})
                   .all();
  if (!valid.item<bool>())
    throw std::runtime_error("non-finite BF16 update; parameters and optimizer not committed");
  int64_t offset = 0;
  for (auto &parameter : ps) {
    parameter.copy_(updated.first.narrow(0, offset, parameter.numel()).view(parameter.sizes()));
    offset += parameter.numel();
  }
  state = {m.first, v.first, m.second, v.second, updated.second};
  ++steps;
  ++model->revision;
  for (auto &parameter : ps)
    parameter.mutable_grad() = T();
}
void Optimizer::save(torch::serialize::OutputArchive &out) {
  out.write("steps", torch::tensor(int64_t(steps), torch::kInt64));
  for (size_t i = 0; i < state.size(); ++i)
    out.write("state" + std::to_string(i), state[i].to(torch::kCPU));
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
    state[i] = v.to(model->parameters()[0].device());
  }
}
TrainResult learn(Network &model, Optimizer &optimizer, const std::vector<Block> &blocks) {
  std::vector<std::shared_ptr<Node>> nodes;
  std::vector<const Step *> steps;
  for (auto &b : blocks)
    for (auto &s : b.steps) {
      nodes.push_back(s.node);
      steps.push_back(&s);
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
  auto output = model->forward(nodes); // 全部目标和损失使用这一次前向，保留同一张计算图。
  auto table = policy_table(output.first, nodes);
  auto masses = cpu_masses(table);
  auto cpu_values = output.second.detach().to(torch::kCPU).contiguous();
  auto values = cpu_values.data_ptr<BF>();
  std::vector<double> logps;
  for (size_t i = 0; i < count; ++i)
    logps.push_back(canonical_logps(masses[i]).at(steps[i]->selected));
  std::vector<BF> target(count), advantage(count), exact(count);
  size_t offset = 0;
  for (size_t bi = 0; bi < blocks.size(); ++bi) {
    auto &block = blocks[bi];
    double correction = 0, following = boot[bi] < 0 ? 0 : float(values[boot[bi]]),
           following_target = following;
    for (size_t j = block.steps.size(); j-- > 0;) {
      size_t i = offset + j;
      auto &s = *steps[i];
      double value = float(values[i]);
      double d = s.terminal ? 0 : 1;
      double ratio = std::min(1.0, std::exp(logps[i] - s.logp));
      correction = ratio * (s.reward + d * following - value) + d * ratio * correction;
      double t = value + correction;
      target[i] = BF(float(t));
      advantage[i] = BF(float(ratio * (s.reward + d * following_target - value)));
      exact[i] = BF(float(logps[i]));
      following = value;
      following_target = t;
    }
    offset += block.steps.size();
  }
  std::vector<int64_t> selected;
  int64_t action_offset = 0;
  for (size_t i = 0; i < count; ++i) {
    auto &n = *nodes[i];
    selected.push_back(action_offset + steps[i]->selected);
    action_offset += n.mask.size();
  }
  auto device = output.first.device();
  auto logp = table.logp.index_select(0, indices(selected, device));
  auto corrected = logp + (bf_tensor(exact, 1, device).squeeze(1) - logp.detach());
  auto targets = bf_tensor(target, 1, device).squeeze(1),
       advantages = bf_tensor(advantage, 1, device).squeeze(1);
  auto loss =
      (-corrected * advantages + 0.5 * (output.second.narrow(0, 0, count) - targets).square() -
       0.01 * table.entropy.narrow(0, 0, count))
          .mean();
  if (!torch::isfinite(loss).item<bool>())
    throw std::runtime_error("non-finite learning loss");
  loss.backward();
  optimizer.step(model);
  return {count, loss.item<double>()};
}
void copy_weights(Network &source, Network &destination) {
  torch::NoGradGuard guard;
  auto a = source->parameters(), b = destination->parameters();
  TORCH_CHECK(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i)
    b[i].copy_(a[i]);
  ++destination->revision;
}
void model_checks(const torch::Device &device) {
  Network model(32);
  model->to(device);
  Optimizer optimizer(model);
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
  n->root->rows = {row};
  n->root->kinds = {128};
  n->mask = {1, 1, 0};
  n->sources = {0, -1, 0};
  n->targets = {-1, 0, 0};
  n->candidates.resize(3);
  n->candidates[0][0] = 1;
  n->candidates[1][1] = 1;
  n->candidates[2][2] = 1;
  auto outputs = model->forward({n, n});
  TORCH_CHECK(outputs.first.scalar_type() == torch::kBFloat16 &&
              outputs.second.scalar_type() == torch::kBFloat16);
  auto probs = probabilities(outputs.first.detach(), {n, n});
  TORCH_CHECK(probs[0] == probs[1] && probs[0][2] == 0);
  auto logs = canonical_logps(probs[0]);
  TORCH_CHECK(std::abs(std::exp(logs[0]) + std::exp(logs[1]) - 1) < 1e-12);
  {
    torch::NoGradGuard guard;
    auto cold = model->forward({n});
    auto encoded = model->encoded_roots;
    auto warm = model->forward({n});
    TORCH_CHECK(model->encoded_roots == encoded && model->cache_hits > 0 &&
                torch::equal(cold.first, warm.first) && torch::equal(cold.second, warm.second));
    ++model->revision;
    model->forward({n});
    TORCH_CHECK(model->encoded_roots == encoded + 1,
                "published parameters did not invalidate root cache");
  }
  Block block;
  block.steps = {{n, 0, logs[0], 0, false, 0}, {n, 1, logs[1], 0, true, 1}};
  auto result = learn(model, optimizer, {block});
  TORCH_CHECK(result.consumed == 2 && optimizer.steps == 1);
  for (auto &p : model->parameters())
    TORCH_CHECK(p.scalar_type() == torch::kBFloat16 && torch::isfinite(p).all().item<bool>());
  for (auto &p : optimizer.state)
    TORCH_CHECK(p.scalar_type() == torch::kBFloat16);
  auto first = model->parameters()[0].detach().clone();
  auto state = optimizer.state[0].clone();
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
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  archive.save_to(stream);
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
  std::cout << "BF16 forward/backward/compensated optimizer, masked behavior probability and "
               "failure atomicity: passed\n";
}
} // namespace hj
