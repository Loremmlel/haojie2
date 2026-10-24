#pragma once
#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace hj {
using Bytes = std::vector<uint8_t>;
using T = torch::Tensor;
constexpr int RowFeatures = 60 * 64 + 52 * 3;
constexpr int ActionFeatures = 64 * 64;
// 输入按所有权存活期计费；共享引用不重复收费，释放轨迹时自动归还预算。
class StorageBytes {
  size_t held = 0;

public:
  inline static std::atomic<uint64_t> total{0};
  StorageBytes() = default;
  StorageBytes(const StorageBytes &other) {
    set(other.held);
  }
  StorageBytes &operator=(const StorageBytes &other) {
    set(other.held);
    return *this;
  }
  ~StorageBytes() {
    set(0);
  }
  void set(size_t bytes) {
    if (bytes >= held)
      total.fetch_add(bytes - held, std::memory_order_relaxed);
    else
      total.fetch_sub(held - bytes, std::memory_order_relaxed);
    held = bytes;
  }
};
struct Row {
  std::array<int64_t, 60> ids{};
  uint64_t present = 0, integers = 0;
  std::array<uint16_t, 52> values{};
};
using Action = std::array<int64_t, 64>;
template <class V> struct ValueHash {
  size_t operator()(const V &value) const {
    static_assert(sizeof(V) % 8 == 0);
    size_t h = 1469598103934665603ull;
    auto p = reinterpret_cast<const uint8_t *>(&value);
    for (size_t i = 0; i < sizeof(V); i += 8) {
      uint64_t word;
      std::memcpy(&word, p + i, 8);
      h = (h ^ word) * 1099511628211ull;
    }
    return h;
  }
};
template <class V> struct ValueEqual {
  bool operator()(const V &a, const V &b) const {
    return std::memcmp(&a, &b, sizeof(V)) == 0;
  }
};
template <class V> struct InputValue {
  V value;
  StorageBytes storage;
  explicit InputValue(V value) : value(std::move(value)) {
    storage.set(sizeof(InputValue) + 32);
  }
};
template <class V> using InputRef = std::shared_ptr<const InputValue<V>>;
template <class V> InputRef<V> input_value(V value) {
  return std::make_shared<InputValue<V>>(std::move(value));
}
template <class V> class InputTable {
  std::unordered_map<V, InputRef<V>, ValueHash<V>, ValueEqual<V>> entries;
  StorageBytes storage;
  void account() {
    storage.set(entries.size() * (sizeof(V) + sizeof(InputRef<V>) + 32) +
                entries.bucket_count() * 8);
  }

public:
  InputRef<V> get(const V &value) {
    auto found = entries.find(value);
    if (found != entries.end())
      return found->second;
    auto ref = input_value(value);
    entries.emplace(value, ref);
    account();
    return ref;
  }
  void trim() {
    std::erase_if(entries, [](auto &entry) { return entry.second.use_count() == 1; });
    account();
  }
};
struct InputPool {
  InputTable<Row> rows;
  InputTable<Action> actions;
  void trim() {
    rows.trim();
    actions.trim();
  }
};
using RowRef = InputRef<Row>;
using ActionRef = InputRef<Action>;
struct Root {
  uint64_t game = 0, id = 0;
  std::vector<RowRef> rows;
  std::vector<int64_t> kinds;
  RowRef globals = input_value(Row{});
  // 当前 actor 的实体仿射项及根上下文投影；检查点只保存原始输入，恢复后重建。
  T cached_tokens, cached_context, host_tokens, host_context;
  mutable std::mutex cache_mutex;
  uint64_t cached_revision = 0;
  const void *cached_owner = nullptr;
  StorageBytes input_storage, cache_storage;
  Root() = default;
  Root(const Root &r) : game(r.game), id(r.id), rows(r.rows), kinds(r.kinds), globals(r.globals) {
    account_input();
  }
  void account_input() {
    input_storage.set(sizeof(Root) + rows.capacity() * sizeof(RowRef) + kinds.capacity() * 8);
  }
  void clear_cache() {
    std::lock_guard lock(cache_mutex);
    cached_tokens = T();
    cached_context = T();
    host_tokens = T();
    host_context = T();
    cached_owner = nullptr;
    cache_storage.set(0);
  }
  void account_cache() {
    size_t bytes = cached_tokens.defined() ? cached_tokens.nbytes() + cached_context.nbytes() : 0;
    if (host_tokens.defined() &&
        (!cached_tokens.defined() || host_tokens.data_ptr() != cached_tokens.data_ptr()))
      bytes += host_tokens.nbytes() + host_context.nbytes();
    cache_storage.set(bytes);
  }
  bool host_valid(const void *owner, uint64_t version) const {
    std::lock_guard lock(cache_mutex);
    return cached_owner == owner && cached_revision == version && host_tokens.defined() &&
           host_context.defined();
  }
};
struct Frontier {
  std::vector<uint32_t> path, children;
};
inline std::vector<RowRef> make_context(InputPool &pool, const std::vector<Row> &prefix,
                                        const std::vector<uint32_t> &cursor,
                                        const std::vector<Frontier> &excluded) {
  std::vector<RowRef> rows;
  for (auto &row : prefix)
    rows.push_back(pool.rows.get(row));
  auto path_row = [&](int64_t group, int64_t position, int64_t index, int64_t length,
                      int64_t type) {
    Row row;
    row.ids[0] = type;
    row.ids[2] = group;
    row.ids[3] = position;
    row.ids[4] = index;
    row.ids[5] = length;
    rows.push_back(pool.rows.get(row));
  };
  for (size_t j = 0; j < cursor.size(); ++j)
    path_row(0, j, cursor[j], cursor.size(), 25);
  for (size_t k = 0; k < excluded.size(); ++k) {
    auto &p = excluded[k];
    for (size_t j = 0; j < p.path.size(); ++j)
      path_row(k + 1, j, p.path[j], p.path.size(), 26);
    for (auto child : p.children)
      path_row(k + 1, p.path.size(), child, p.path.size(), 28);
  }
  // 空路径仍有显式节点，保持与原公开输入的每一行及顺序一致。
  path_row(0, 0, 0, 0, 27);
  return rows;
}
struct Node {
  std::shared_ptr<Root> root;
  std::vector<RowRef> context;
  std::vector<ActionRef> candidates;
  std::vector<int32_t> sources, targets;
  std::vector<uint8_t> mask;
  StorageBytes storage;
  size_t bytes() const {
    size_t total = sizeof(Node) + context.capacity() * sizeof(RowRef) +
                   candidates.capacity() * sizeof(candidates[0]) + sources.capacity() * 4 +
                   targets.capacity() * 4 + mask.capacity();
    return total;
  }
  void account_input() {
    storage.set(bytes());
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
  Reader(const uint8_t *p, size_t size) : p(p), left(size) {}
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
Frame parse(Reader, std::shared_ptr<Root> &, InputPool &);
inline Frame parse(const Bytes &bytes, std::shared_ptr<Root> &root, InputPool &pool) {
  return parse(Reader(bytes), root, pool);
}
} // namespace hj
