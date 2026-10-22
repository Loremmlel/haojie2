// 只把原紧凑主干的 ATen 发射循环移到 C++；参数、精度与每层算子顺序均由调用方提供。
#include <ATen/Functions.h>
#include <torch/library.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define HAOJIE_EXPORT __declspec(dllexport)
#else
#include <unistd.h>
#define HAOJIE_EXPORT __attribute__((visibility("default")))
#endif

// ctypes 在整个调用期间释放 GIL；帧和评分的所有者由 Python 调用栈持有。
// 返回校验错误编号，输入仍遵守原二进制协议，不跳过公开边界检查。
extern "C" HAOJIE_EXPORT int haojie_validate(
    const void* address, int64_t entities, int64_t candidates, int64_t kind_count) {
  const auto* values = static_cast<const float*>(address);
  const auto floats = entities * 64 + 32 + candidates * 64;
  for (int64_t index = 0; index < floats; ++index)
    if (!std::isfinite(values[index])) return 1;
  const auto* kinds = reinterpret_cast<const int64_t*>(values + floats);
  for (int64_t index = 0; index < entities; ++index)
    if (kinds[index] < 0 || kinds[index] >= kind_count) return 2;
  const auto* sources = kinds + entities;
  const auto* targets = sources + candidates;
  const auto* mask = reinterpret_cast<const uint8_t*>(targets + candidates);
  const auto* choices = mask + entities;
  bool chosen = false;
  for (int64_t index = 0; index < candidates; ++index) chosen |= choices[index] != 0;
  if (!chosen) return 3;
  for (int side = 0; side < 2; ++side) {
    const auto* pointers = side == 0 ? sources : targets;
    for (int64_t index = 0; index < candidates; ++index)
      if (pointers[index] < -1 || pointers[index] >= entities) return 4 + side * 2;
    for (int64_t index = 0; index < candidates; ++index)
      if (choices[index] && pointers[index] >= 0 && !mask[pointers[index]]) return 5 + side * 2;
  }
  return 0;
}

extern "C" HAOJIE_EXPORT int haojie_send(
    const int64_t* descriptors, const int64_t* addresses, const int64_t* counts,
    const char* const* headers, int64_t size) {
  // 所有评分先检查后写管道，避免非法数值成为已执行动作。
  for (int64_t row = 0; row < size; ++row) {
    const auto* values = reinterpret_cast<const float*>(addresses[row]);
    for (int64_t index = 0; index < counts[row]; ++index)
      if (!std::isfinite(values[index])) return -1;
  }
  for (int64_t row = 0; row < size; ++row) {
    std::string packet(headers[row]);
    packet.append(reinterpret_cast<const char*>(addresses[row]), counts[row] * sizeof(float));
    size_t offset = 0;
    while (offset < packet.size()) {
#ifdef _WIN32
      const auto written = _write(static_cast<int>(descriptors[row]), packet.data() + offset,
                                  static_cast<unsigned int>(packet.size() - offset));
#else
      const auto written = ::write(static_cast<int>(descriptors[row]), packet.data() + offset,
                                   packet.size() - offset);
#endif
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) return written < 0 ? errno : EIO;
      offset += written;
    }
  }
  return 0;
}

// 输入地址只来自已验证且持有租约的协议帧；不接收模型提供的地址或权威游戏状态。
struct PackedRow {
  const float *entities, *globals, *candidates;
  const int64_t *kinds, *sources, *targets;
  const bool *mask, *candidate_mask;
  int64_t count, actions;
  std::vector<int64_t> valid, remap, positions, inverse, active;
};

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, std::vector<int64_t>>
pack(at::IntArrayRef addresses, at::IntArrayRef entities, at::IntArrayRef actions,
     at::TensorList cached, bool pinned, int64_t recent_journal) {
  const int64_t size = addresses.size();
  TORCH_CHECK(size > 0 && entities.size() == addresses.size() &&
                  actions.size() == addresses.size() && (cached.empty() || cached.size() == 15),
              "invalid packed frame descriptors");
  std::vector<PackedRow> rows;
  int64_t tokens = 0, reads = 0, active_count = 0, max_k = 0, max_q = 0, max_active = 0;
  int64_t max_actions = 0, total_actions = 0;
  for (int64_t index = 0; index < size; ++index) {
    TORCH_CHECK(addresses[index] > 0 && entities[index] > 0 && actions[index] > 0,
                "invalid packed frame shape");
    TORCH_CHECK(haojie_validate(reinterpret_cast<const void*>(addresses[index]),
                                entities[index], actions[index], 256) == 0,
                "invalid packed frame values");
    PackedRow row;
    row.count = entities[index];
    row.actions = actions[index];
    row.entities = reinterpret_cast<const float*>(addresses[index]);
    row.globals = row.entities + row.count * 64;
    row.candidates = row.globals + 32;
    row.kinds = reinterpret_cast<const int64_t*>(row.candidates + row.actions * 64);
    row.sources = row.kinds + row.count;
    row.targets = row.sources + row.actions;
    row.mask = reinterpret_cast<const bool*>(row.targets + row.actions);
    row.candidate_mask = row.mask + row.count;
    row.remap.resize(row.count + 1, 0);
    for (int64_t entity = 0; entity < row.count; ++entity) {
      if (row.mask[entity]) {
        row.valid.push_back(entity);
        row.remap[entity + 1] = row.valid.size();
      }
    }
    row.positions.push_back(0);
    for (const auto* pointers : {row.sources, row.targets}) {
      for (int64_t action = 0; action < row.actions; ++action)
        row.positions.push_back(row.remap[pointers[action] + 1]);
    }
    std::sort(row.positions.begin(), row.positions.end());
    row.positions.erase(std::unique(row.positions.begin(), row.positions.end()), row.positions.end());
    std::vector<int64_t> position_index(row.valid.size() + 1, 0);
    for (size_t query = 0; query < row.positions.size(); ++query)
      position_index[row.positions[query]] = query;
    row.inverse.push_back(0);
    for (const auto* pointers : {row.sources, row.targets})
      for (int64_t action = 0; action < row.actions; ++action)
        row.inverse.push_back(position_index[row.remap[pointers[action] + 1]]);
    std::vector<bool> live(row.valid.size() + 1, false);
    live[0] = true;
    int64_t journals = 0;
    for (int64_t entity = row.count - 1; entity >= 0; --entity) {
      if (!row.mask[entity]) continue;
      const bool journal = row.kinds[entity] == 250;
      if (journal) ++journals;
      const bool history = journal ? journals > recent_journal
                                   : row.entities[entity * 64] == 5.0f / 32;
      live[row.remap[entity + 1]] = !history;
    }
    for (int64_t action = 0; action < row.actions; ++action) {
      if (!row.candidate_mask[action]) continue;
      live[row.remap[row.sources[action] + 1]] = true;
      live[row.remap[row.targets[action] + 1]] = true;
    }
    for (size_t entity = 0; entity < live.size(); ++entity)
      if (live[entity]) row.active.push_back(entity);
    tokens += row.valid.size() + 1;
    reads += row.positions.size();
    active_count += row.active.size();
    max_k = std::max(max_k, int64_t(row.valid.size() + 1));
    max_q = std::max(max_q, int64_t(row.positions.size()));
    max_active = std::max(max_active, int64_t(row.active.size()));
    max_actions = std::max(max_actions, row.actions);
    total_actions += row.actions;
    rows.push_back(std::move(row));
  }
  TORCH_CHECK(tokens <= std::numeric_limits<int32_t>::max(), "packed token index overflow");
  const std::vector<std::vector<int64_t>> shapes = {
      {tokens,64}, {tokens}, {size,32}, {size}, {size+1}, {size+1}, {size+1},
      {reads}, {active_count}, {size,max_q}, {size,max_actions}, {size,max_actions},
      {size,max_actions,64}, {size,max_actions}, {total_actions}};
  const std::vector<at::ScalarType> types = {
      at::kFloat, at::kLong, at::kFloat, at::kLong, at::kInt, at::kInt, at::kInt,
      at::kLong, at::kLong, at::kLong, at::kLong, at::kLong, at::kFloat, at::kBool, at::kLong};
  std::vector<at::Tensor> storage, batch;
  for (size_t key = 0; key < shapes.size(); ++key) {
    int64_t count = 1;
    for (auto dimension : shapes[key]) count *= dimension;
    auto buffer = cached.empty() ? at::Tensor() : cached[key];
    if (!buffer.defined() || buffer.numel() < count)
      buffer = at::empty({count}, at::TensorOptions().dtype(types[key]).pinned_memory(pinned));
    storage.push_back(buffer);
    batch.push_back(buffer.narrow(0, 0, count).view(shapes[key]));
  }
  batch[10].fill_(-1);
  batch[11].fill_(-1);
  batch[12].zero_();
  batch[13].zero_();
  int64_t start = 0, qstart = 0, astart = 0, candidate_start = 0;
  batch[4].data_ptr<int32_t>()[0] = 0;
  batch[5].data_ptr<int32_t>()[0] = 0;
  batch[6].data_ptr<int32_t>()[0] = 0;
  for (int64_t index = 0; index < size; ++index) {
    const auto& row = rows[index];
    auto* destination = batch[0].data_ptr<float>() + start * 64;
    std::fill_n(destination, 64, 0);
    auto* kinds = batch[1].data_ptr<int64_t>() + start;
    kinds[0] = 0;
    for (size_t entity = 0; entity < row.valid.size(); ++entity) {
      std::memcpy(destination + (entity + 1) * 64, row.entities + row.valid[entity] * 64, 64 * sizeof(float));
      kinds[entity + 1] = row.kinds[row.valid[entity]];
    }
    std::memcpy(batch[2].data_ptr<float>() + index * 32, row.globals, 32 * sizeof(float));
    batch[3].data_ptr<int64_t>()[index] = start;
    for (size_t query = 0; query < row.positions.size(); ++query)
      batch[7].data_ptr<int64_t>()[qstart + query] = row.positions[query] + start;
    for (size_t entity = 0; entity < row.active.size(); ++entity)
      batch[8].data_ptr<int64_t>()[astart + entity] = row.active[entity] + start;
    auto* head = batch[9].data_ptr<int64_t>() + index * max_q;
    std::fill_n(head, max_q, qstart);
    for (size_t query = 0; query < row.positions.size(); ++query) head[query] = qstart + query;
    for (int64_t action = 0; action < row.actions; ++action) {
      batch[14].data_ptr<int64_t>()[candidate_start + action] = index * max_actions + action;
      batch[10].data_ptr<int64_t>()[index * max_actions + action] = row.inverse[1 + action] - 1;
      batch[11].data_ptr<int64_t>()[index * max_actions + action] = row.inverse[1 + row.actions + action] - 1;
    }
    std::memcpy(batch[12].data_ptr<float>() + index * max_actions * 64, row.candidates, row.actions * 64 * sizeof(float));
    std::memcpy(batch[13].data_ptr<bool>() + index * max_actions, row.candidate_mask, row.actions * sizeof(bool));
    start += row.valid.size() + 1;
    qstart += row.positions.size();
    astart += row.active.size();
    candidate_start += row.actions;
    batch[4].data_ptr<int32_t>()[index + 1] = start;
    batch[5].data_ptr<int32_t>()[index + 1] = qstart;
    batch[6].data_ptr<int32_t>()[index + 1] = astart;
  }
  return {storage, batch, {max_k,max_q,max_active,size,tokens-size,total_actions,size*max_actions}};
}

at::Tensor packed_encode(
    at::TensorList input,
    at::TensorList weights,
    int64_t heads,
    int64_t history_layers,
    at::IntArrayRef maxima,
    at::ArrayRef<double> epsilon) {
  TORCH_CHECK(input.size() == 9 && weights.size() >= 19 &&
                  (weights.size() - 7) % 12 == 0,
              "invalid packed encoder arguments");
  const int64_t layers = (weights.size() - 7) / 12;
  const int64_t width = weights[0].size(0);
  TORCH_CHECK(heads > 0 && width % heads == 0 && maxima.size() == 3 &&
                  epsilon.size() == static_cast<size_t>(layers * 2 + 1),
              "invalid packed encoder layout");
  auto x = at::add(at::linear(input[0], weights[0], weights[1]),
                   at::embedding(weights[2], input[1], 0));
  auto global = at::linear(input[2], weights[3], weights[4]).to(x.scalar_type());
  x.index_copy_(0, input[3], global);
  for (int64_t index = 0; index < layers; ++index) {
    const auto base = 5 + index * 12;
    auto normalized = at::layer_norm(
        x, {width}, weights[base], weights[base + 1], epsilon[index * 2]);
    auto qkv = at::linear(normalized, weights[base + 2], weights[base + 3])
                   .view({-1, 3, heads, width / heads})
                   .unbind(1);
    const bool last = index == layers - 1;
    const bool sparse = last || index >= history_layers;
    const auto query = last ? 1 : sparse ? 2 : 0;
    const auto positions = input[last ? 7 : 8];
    auto q = sparse ? qkv[0].index_select(0, positions) : qkv[0];
    auto residual = sparse ? x.index_select(0, positions) : x;
    auto result = at::_efficient_attention_forward(
        q.unsqueeze(0), qkv[1].unsqueeze(0), qkv[2].unsqueeze(0),
        std::nullopt, input[query == 0 ? 4 : query == 1 ? 5 : 6], input[4],
        maxima[query], maxima[0], 0.0, 0, false);
    auto attended = std::get<0>(result).squeeze(0).reshape({-1, width});
    auto updated = at::add(residual, at::linear(attended, weights[base + 4], weights[base + 5]));
    normalized = at::layer_norm(
        updated, {width}, weights[base + 6], weights[base + 7], epsilon[index * 2 + 1]);
    auto hidden = at::gelu(at::linear(normalized, weights[base + 8], weights[base + 9]), "tanh");
    updated = at::add(updated, at::linear(hidden, weights[base + 10], weights[base + 11]));
    x = sparse && !last ? x.index_copy(0, positions, updated) : updated;
  }
  return at::layer_norm(x, {width}, weights[weights.size() - 2],
                        weights[weights.size() - 1], epsilon.back());
}

at::Tensor packed_score(at::Tensor x, at::TensorList input, at::TensorList weights,
                        double epsilon) {
  // 仅省去跨局补齐的候选行；每个原始候选仍经过相同投影、评分与最终遮罩。
  TORCH_CHECK(input.size() == 6 && weights.size() == 10, "invalid packed policy arguments");
  const auto size = input[1].size(0), actions = input[1].size(1), width = x.size(1);
  const auto positions = input[5];
  const auto rows = at::floor_divide(positions, actions);
  const auto starts = at::mul(rows, input[0].size(1));
  const auto heads = input[0].reshape({-1});
  auto context = x.index_select(0, heads.index_select(0, starts));
  auto sources = x.index_select(0, heads.index_select(
      0, at::add(at::add(starts, input[1].reshape({-1}).index_select(0, positions)), 1)));
  auto targets = x.index_select(0, heads.index_select(
      0, at::add(at::add(starts, input[2].reshape({-1}).index_select(0, positions)), 1)));
  auto candidates = input[3].reshape({-1, 64}).index_select(0, positions);
  auto projected = at::gelu(at::linear(candidates, weights[0], weights[1]), "tanh");
  projected = at::layer_norm(projected, {width}, weights[2], weights[3], epsilon);
  auto output = at::cat(std::vector<at::Tensor>{context, sources, targets, projected}, -1);
  output = at::gelu(at::linear(output, weights[4], weights[5]), "tanh");
  output = at::gelu(at::linear(output, weights[6], weights[7]), "tanh");
  output = at::linear(output, weights[8], weights[9]).reshape({-1}).to(at::kFloat);
  auto logits = at::full({size * actions}, -1e9, output.options());
  logits.index_copy_(0, positions, output);
  return logits.view({size, actions}).masked_fill(input[4].logical_not(), -1e9);
}

TORCH_LIBRARY(haojie, library) {
  library.def("packed_score(Tensor encoded, Tensor[] input, Tensor[] weights, float epsilon) -> Tensor");
  library.impl("packed_score", c10::DispatchKey::CompositeImplicitAutograd,
               TORCH_FN(packed_score));
  library.def(
      "pack(int[] addresses, int[] entities, int[] actions, Tensor[] cache, "
      "bool pinned, int recent_journal) -> (Tensor[], Tensor[], int[])");
  library.impl("pack", c10::DispatchKey::CompositeExplicitAutograd, TORCH_FN(pack));
  library.def(
      "packed_encode(Tensor[] input, Tensor[] weights, int heads, int history_layers, "
      "int[] maxima, float[] epsilon) -> Tensor");
  library.impl("packed_encode", c10::DispatchKey::CompositeImplicitAutograd,
               TORCH_FN(packed_encode));
}
