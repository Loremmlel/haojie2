#pragma once
#include <array>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>
#include <vector>

namespace hj {
using Bytes = std::vector<uint8_t>;
using T = torch::Tensor;
constexpr int RowFeatures = 60 * 64 + 52 * 3;
constexpr int ActionFeatures = 64 * 64;
struct Row {
  std::array<int64_t, 60> ids{};
  uint64_t present = 0, integers = 0;
  std::array<uint16_t, 52> values{};
};
struct Root {
  uint64_t game = 0, id = 0;
  std::vector<Row> rows;
  std::vector<int64_t> kinds;
  std::array<uint16_t, 32> globals{};
  // 仅当前 actor 参数的可丢弃编码；检查点保存原始类型化输入，恢复后重建。
  T cached_tokens, cached_context;
  uint64_t cached_revision = 0;
  const void *cached_owner = nullptr;
  void clear_cache() {
    cached_tokens = T();
    cached_context = T();
    cached_owner = nullptr;
  }
};
struct Frontier {
  std::vector<uint32_t> path, children;
};
struct Node {
  std::shared_ptr<Root> root;
  std::vector<Row> prefix;
  std::vector<std::array<int64_t, 64>> candidates;
  std::vector<int32_t> sources, targets;
  std::vector<uint8_t> mask;
  std::vector<uint32_t> cursor;
  std::vector<Frontier> excluded;
  size_t bytes() const {
    size_t total = sizeof(Node) + prefix.size() * sizeof(Row) +
                   candidates.size() * sizeof(candidates[0]) + sources.size() * 9 +
                   cursor.size() * 4;
    for (auto &p : excluded)
      total += sizeof(p) + (p.path.size() + p.children.size()) * 4;
    return total;
  }
};
struct Frame {
  uint64_t game = 0, root = 0, opponent = 0, commands = 0, microsteps = 0, failures = 0,
           backtracks = 0;
  uint32_t actor = 0, learner = 0, kind = 0;
  int32_t winner = -1, selected = -1;
  double logp = 0;
  std::shared_ptr<Node> node;
};
struct Step {
  std::shared_ptr<Node> node;
  int64_t selected = 0;
  double logp = 0;
  uint64_t version = 0;
  bool terminal = false;
  int8_t reward = 0;
};
struct Block {
  std::vector<Step> steps;
  std::shared_ptr<Node> bootstrap;
};

struct Reader {
  const uint8_t *p;
  size_t left;
  explicit Reader(const Bytes &b) : p(b.data()), left(b.size()) {}
  template <class V> V get() {
    if (left < sizeof(V))
      throw std::runtime_error("truncated native data");
    V v;
    std::memcpy(&v, p, sizeof(V));
    p += sizeof(V);
    left -= sizeof(V);
    return v;
  }
  void raw(void *to, size_t n) {
    if (!n)
      return;
    if (n > left)
      throw std::runtime_error("truncated native array");
    std::memcpy(to, p, n);
    p += n;
    left -= n;
  }
  template <class V> std::vector<V> array(size_t n) {
    if (n > left / sizeof(V))
      throw std::runtime_error("invalid native array length");
    std::vector<V> v(n);
    raw(v.data(), n * sizeof(V));
    return v;
  }
  void end() {
    if (left)
      throw std::runtime_error("trailing native data");
  }
};
template <class V> void put(Bytes &b, V v) {
  auto p = reinterpret_cast<const uint8_t *>(&v);
  b.insert(b.end(), p, p + sizeof(V));
}
inline void raw(Bytes &b, const void *p, size_t n) {
  if (!n)
    return;
  auto s = static_cast<const uint8_t *>(p);
  b.insert(b.end(), s, s + n);
}

// 控制/检查点的整数和字节与模型张量分开存储，不经浮点转换。
inline void put_bytes(Bytes &out, const Bytes &b) {
  put<uint64_t>(out, b.size());
  raw(out, b.data(), b.size());
}
inline Bytes get_bytes(Reader &in) {
  return in.array<uint8_t>(in.get<uint64_t>());
}
Frame parse(const Bytes &, std::shared_ptr<Root> &);
Bytes encode_node(const Node &);
std::shared_ptr<Node> decode_node(Reader &);
} // namespace hj
