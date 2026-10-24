#pragma once
#include "data.h"

namespace hj {
template <class V> void put_array(Bytes &out, const std::vector<V> &values) {
  put<uint64_t>(out, values.size());
  raw(out, values.data(), values.size() * sizeof(V));
}
template <class V> std::vector<V> get_array(Reader &in) {
  return in.array<V>(in.get<uint64_t>());
}
template <class Ptr> struct References {
  std::vector<Ptr> values;
  std::unordered_map<const void *, uint64_t> ids;
  uint64_t add(const Ptr &p) {
    if (!p)
      return 0;
    auto [entry, inserted] = ids.try_emplace(p.get(), values.size() + 1);
    if (inserted)
      values.push_back(p);
    return entry->second;
  }
  uint64_t id(const Ptr &p) const {
    return p ? ids.at(p.get()) : 0;
  }
};

// 表中每个共享值只写一次；后续根、节点、轨迹只写整数引用。
struct GraphWriter {
  References<RowRef> rows;
  References<ActionRef> actions;
  References<std::shared_ptr<Root>> roots;
  References<std::shared_ptr<Node>> nodes;
  uint64_t node(const std::shared_ptr<Node> &n) {
    if (!n)
      return 0;
    auto found = nodes.ids.find(n.get());
    if (found != nodes.ids.end())
      return found->second;
    auto id = nodes.add(n);
    auto before = roots.values.size();
    roots.add(n->root);
    if (before != roots.values.size()) {
      rows.add(n->root->globals);
      for (auto &r : n->root->rows)
        rows.add(r);
    }
    for (auto &r : n->context)
      rows.add(r);
    for (auto &a : n->candidates)
      actions.add(a);
    return id;
  }
  void ref(Bytes &out, const std::shared_ptr<Node> &n) {
    put(out, node(n));
  }
  template <class Ptr>
  static void refs(Bytes &out, const std::vector<Ptr> &values, const References<Ptr> &table) {
    put<uint64_t>(out, values.size());
    for (auto &p : values)
      put(out, table.id(p));
  }
  void write(Bytes &out) const {
    put<uint64_t>(out, rows.values.size());
    for (auto &r : rows.values)
      put(out, r->value);
    put<uint64_t>(out, actions.values.size());
    for (auto &a : actions.values)
      put(out, a->value);
    put<uint64_t>(out, roots.values.size());
    for (auto &root : roots.values) {
      put(out, root->game);
      put(out, root->id);
      refs(out, root->rows, rows);
      put_array(out, root->kinds);
      put(out, rows.id(root->globals));
    }
    put<uint64_t>(out, nodes.values.size());
    for (auto &n : nodes.values) {
      put(out, roots.id(n->root));
      refs(out, n->context, rows);
      refs(out, n->candidates, actions);
      put_array(out, n->sources);
      put_array(out, n->targets);
      put_array(out, n->mask);
    }
  }
};

struct GraphReader {
  std::vector<RowRef> rows;
  std::vector<ActionRef> actions;
  std::vector<std::shared_ptr<Root>> roots;
  std::vector<std::shared_ptr<Node>> nodes;
  static uint64_t count(Reader &r, size_t minimum) {
    auto n = r.get<uint64_t>();
    if (n > r.left / minimum)
      throw std::runtime_error("invalid checkpoint table length");
    return n;
  }
  template <class Ptr> static Ptr at(const std::vector<Ptr> &table, uint64_t id) {
    if (!id || id > table.size())
      throw std::runtime_error("invalid checkpoint reference");
    return table[id - 1];
  }
  template <class Ptr> static std::vector<Ptr> refs(Reader &r, const std::vector<Ptr> &table) {
    auto ids = get_array<uint64_t>(r);
    std::vector<Ptr> out;
    out.reserve(ids.size());
    for (auto id : ids)
      out.push_back(at(table, id));
    return out;
  }
  GraphReader(Reader &r, InputPool &pool) {
    auto n = count(r, sizeof(Row));
    rows.reserve(n);
    for (uint64_t i = 0; i < n; ++i)
      rows.push_back(pool.rows.get(r.get<Row>()));
    n = count(r, sizeof(Action));
    actions.reserve(n);
    for (uint64_t i = 0; i < n; ++i)
      actions.push_back(pool.actions.get(r.get<Action>()));
    n = count(r, 96);
    roots.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
      auto root = std::make_shared<Root>();
      root->game = r.get<uint64_t>();
      root->id = r.get<uint64_t>();
      root->rows = refs(r, rows);
      root->kinds = get_array<int64_t>(r);
      if (root->rows.size() != root->kinds.size())
        throw std::runtime_error("root kind length differs");
      root->globals = at(rows, r.get<uint64_t>());
      root->account_input();
      roots.push_back(std::move(root));
    }
    n = count(r, 48);
    nodes.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
      auto node = std::make_shared<Node>();
      node->root = at(roots, r.get<uint64_t>());
      node->context = refs(r, rows);
      node->candidates = refs(r, actions);
      node->sources = get_array<int32_t>(r);
      node->targets = get_array<int32_t>(r);
      node->mask = get_array<uint8_t>(r);
      if (node->context.empty() || node->mask.size() != node->candidates.size() ||
          node->sources.size() != node->mask.size() || node->targets.size() != node->mask.size())
        throw std::runtime_error("node table length differs");
      node->account_input();
      nodes.push_back(std::move(node));
    }
  }
  std::shared_ptr<Node> ref(Reader &r) const {
    auto id = r.get<uint64_t>();
    return id ? at(nodes, id) : nullptr;
  }
};
} // namespace hj
