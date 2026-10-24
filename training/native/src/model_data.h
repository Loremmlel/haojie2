#pragma once
#include "data.h"
#include <bit>
#include <numeric>
#include <unordered_map>

namespace hj::packing {
using BF = c10::BFloat16;
inline auto bfopts() {
  return torch::TensorOptions().dtype(torch::kBFloat16);
}
inline T bf_tensor(const std::vector<BF> &v, int64_t columns, const torch::Device &device) {
  return torch::from_blob(const_cast<BF *>(v.data()), {int64_t(v.size() / columns), columns},
                          bfopts())
      .clone()
      .to(device);
}
inline T indices(const std::vector<int64_t> &v, const torch::Device &d) {
  return torch::tensor(v, torch::kInt64).to(d, true);
}
inline T segments(const T &data, const char *reduction, const T &lengths) {
  // 长度已由宿主构造并验证，不重复触发设备标量回读。
  return at::segment_reduce(data, reduction, lengths, {}, {}, 0, true);
}
using Slice = std::array<int64_t, 2>;
struct Inputs {
  std::vector<int64_t> integers;
  std::vector<uint16_t> floats;
  T ints, values;
  Slice add(const std::vector<int64_t> &v) {
    Slice s{int64_t(integers.size()), int64_t(v.size())};
    integers.insert(integers.end(), v.begin(), v.end());
    return s;
  }
  Slice add_values(const uint16_t *v, size_t n) {
    Slice s{int64_t(floats.size()), int64_t(n)};
    floats.insert(floats.end(), v, v + n);
    return s;
  }
  void upload(const torch::Device &d) {
    auto upload = [&](const void *p, int64_t count, torch::ScalarType type, size_t bytes) {
      auto host = torch::empty(
          {count},
          torch::TensorOptions().dtype(type).device(torch::kCPU).pinned_memory(d.is_cuda()));
      if (bytes)
        std::memcpy(host.data_ptr(), p, bytes);
      return d.is_cpu() ? host : host.to(d, true);
    };
    ints = upload(integers.data(), integers.size(), torch::kInt64, integers.size() * 8);
    values = upload(floats.data(), floats.size(), torch::kBFloat16, floats.size() * 2);
  }
  T get(Slice s) const {
    return ints.narrow(0, s[0], s[1]);
  }
  T bf(Slice s, int64_t width) const {
    return values.narrow(0, s[0], s[1]).view({-1, width});
  }
};
inline void set_bits(std::vector<int64_t> &out, uint64_t v, int64_t base) {
  while (v) {
    out.push_back(base + std::countr_zero(v));
    v &= v - 1;
  }
}
// ZigZag 是完整 64 位双射，避免常见的 -1 展开为 64 个重复符号位。
inline uint64_t integer_bits(int64_t value) {
  return (uint64_t(value) << 1) ^ uint64_t(-int64_t(value < 0));
}
using RowHash = ValueHash<Row>;
using RowEqual = ValueEqual<Row>;
struct Rows {
  std::vector<Row> rows;
  std::unordered_map<Row, int64_t, RowHash, RowEqual> unique;
  int64_t add(const Row &r) {
    auto [it, inserted] = unique.try_emplace(r, rows.size());
    if (inserted)
      rows.push_back(r);
    return it->second;
  }
  int64_t add(const RowRef &row) {
    return add(row->value);
  }
  std::array<Slice, 3> pack(Inputs &input) const {
    std::vector<int64_t> bits, offsets{0};
    Slice values{int64_t(input.floats.size()), int64_t(rows.size() * 52)};
    for (auto &r : rows) {
      TORCH_CHECK((r.present >> 52) == 0 && (r.integers >> 52) == 0,
                  "row presence/type bits exceed field count");
      for (size_t j = 0; j < r.ids.size(); ++j)
        set_bits(bits, integer_bits(r.ids[j]), j * 64);
      set_bits(bits, r.present, 60 * 64);
      set_bits(bits, r.integers, 60 * 64 + 52);
      offsets.push_back(bits.size());
      input.add_values(r.values.data(), r.values.size());
    }
    return {input.add(bits), input.add(offsets), values};
  }
};
inline void append_context(const Node &n, Rows &rows, std::vector<int64_t> &indices) {
  for (auto &row : n.context)
    indices.push_back(rows.add(row));
}
using ArrayHash = ValueHash<Action>;
T grouped_gather(const T &, const T &, const T &, const T &);
struct GatherTable {
  T ids, positions, offsets;
  T operator()(const T &x) const {
    return offsets.defined() && torch::GradMode::is_enabled()
               ? grouped_gather(x, ids, positions, offsets)
               : x.index_select(0, ids);
  }
};
struct GatherLayout {
  Slice ids{}, positions{}, offsets{};
  bool gradient = false;
  GatherLayout() = default;
  GatherLayout(Inputs &input, const std::vector<int64_t> &indices, int64_t count, bool grad)
      : ids(input.add(indices)), gradient(grad) {
    if (!grad)
      return;
    std::vector<int64_t> offset(count + 1, 0), position(indices.size());
    for (auto id : indices) {
      TORCH_CHECK(id >= 0 && id < count, "invalid gather reference");
      ++offset[id + 1];
    }
    std::partial_sum(offset.begin(), offset.end(), offset.begin());
    auto cursor = offset;
    for (size_t i = 0; i < indices.size(); ++i)
      position[cursor[indices[i]]++] = i;
    positions = input.add(position);
    offsets = input.add(offset);
  }
  GatherTable bind(const Inputs &input) const {
    return {input.get(ids), gradient ? input.get(positions) : T(),
            gradient ? input.get(offsets) : T()};
  }
};
struct ScoreTables {
  T row_embeddings, prefix_ids, prefix_offsets, root_context, entity_affine, actions;
  GatherTable node_root, owner, candidate, source, target, key;
  T live_positions;
  int64_t full_candidates = 0;
  T prefix_context;
};
} // namespace hj::packing
