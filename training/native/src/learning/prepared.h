#pragma once
#include "model.h"
#include <cmath>
#include <map>

namespace hj::training {
using namespace packing;
// 每个族只有一份原始整数表；所有切片固定地址，重放只替换字节和长度元数据。
struct Bank {
  struct Field {
    int64_t offset, count;
    T tensor;
  };
  std::vector<int64_t> words;
  T host, device;
  Field add(int64_t count) {
    auto offset = int64_t(words.size());
    words.resize(offset + count);
    return {offset, count, {}};
  }
  void bind(std::initializer_list<Field *> list, const torch::Device &where) {
    host = torch::empty(
        {int64_t(words.size())},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true));
    device = torch::empty_like(host, host.options().device(where));
    for (auto *f : list)
      f->tensor = device.narrow(0, f->offset, f->count);
  }
  void put(const Field &f, const std::vector<int64_t> &values) {
    TORCH_CHECK(values.size() <= size_t(f.count), "prepared field overflow");
    std::copy(values.begin(), values.end(), words.begin() + f.offset);
  }
  template <class V> void raw(const Field &f, const std::vector<V> &values) {
    TORCH_CHECK(values.size() * sizeof(V) <= size_t(f.count) * 8, "prepared raw field overflow");
    if (!values.empty())
      std::memcpy(words.data() + f.offset, values.data(), values.size() * sizeof(V));
  }
  void upload() {
    std::memcpy(host.data_ptr(), words.data(), words.size() * 8);
    device.copy_(host, true);
  }
};
struct Gather {
  Bank::Field ids, positions, offsets;
  int64_t inputs, outputs;
  Gather(Bank &bank, int64_t in, int64_t out)
      : ids(bank.add(out)), positions(bank.add(out)), offsets(bank.add(in + 1)), inputs(in),
        outputs(out) {}
  void put(Bank &bank, std::vector<int64_t> values) const {
    TORCH_CHECK(values.size() <= size_t(outputs), "prepared gather overflow");
    auto used = int64_t(values.size());
    values.resize(outputs, 0);
    std::vector<int64_t> offset(inputs + 1), position(outputs);
    for (int64_t j = 0; j < used; ++j) {
      auto id = values[j];
      TORCH_CHECK(id >= 0 && id < inputs, "prepared gather input differs");
      ++offset[id + 1];
    }
    std::partial_sum(offset.begin(), offset.end(), offset.begin());
    auto cursor = offset;
    for (int64_t j = 0; j < used; ++j)
      position[cursor[values[j]]++] = j;
    bank.put(ids, values);
    bank.put(positions, position);
    bank.put(offsets, offset);
  }
  T operator()(const T &input) const {
    return bind()(input);
  }
  GatherTable bind() const {
    return {ids.tensor, positions.tensor, offsets.tensor};
  }
};
struct Prepared {
  int64_t batch, nodes, roots, tokens, rows, actions, keys, candidates, prefixes, references,
      maximum;
  Bank bank;
  Bank::Field raw_rows, raw_actions, kind, kind_mask, cuq, cuk, prefix_ids, prefix_offsets,
      prefix_back_ids, prefix_back_offsets, policy_owner, policy_lengths, policy_mask, records,
      blocks, selected, valid, control;
  Gather token_rows, globals, node_roots, entity_refs, key_actions, key_source, key_target,
      live_owner, live_key;
  uint64_t used_roots = 0, used_tokens = 0, used_candidates = 0, used_actions = 0, used_keys = 0,
           count = 0;
  Prepared(int64_t batch, int64_t token_capacity, int64_t maximum, const torch::Device &device)
      : batch(batch), nodes(batch * 2), roots(nodes), tokens(token_capacity),
        rows(std::max<int64_t>(512, batch * 32)), actions(std::max<int64_t>(64, batch * 8)),
        keys(std::max<int64_t>(256, batch * 32)), candidates(std::max<int64_t>(2048, batch * 256)),
        prefixes(std::max<int64_t>(4096, batch * 512)),
        references(std::max<int64_t>(129, batch * 8 + 1)), maximum(maximum),
        raw_rows(bank.add(rows * 75)), raw_actions(bank.add(actions * 64)), kind(bank.add(tokens)),
        kind_mask(bank.add(tokens)), cuq(bank.add(roots + 1)), cuk(bank.add(roots + 1)),
        prefix_ids(bank.add(prefixes)), prefix_offsets(bank.add(nodes + 1)),
        prefix_back_ids(bank.add(prefixes)), prefix_back_offsets(bank.add(rows + 1)),
        policy_owner(bank.add(candidates)), policy_lengths(bank.add(batch)),
        policy_mask(bank.add(candidates)), records(bank.add(batch * 6)),
        blocks(bank.add(batch * 3)), selected(bank.add(batch)), valid(bank.add(batch)),
        control(bank.add(4)), token_rows(bank, rows, tokens), globals(bank, tokens, roots),
        node_roots(bank, roots, nodes), entity_refs(bank, tokens + 1, references),
        key_actions(bank, actions, keys), key_source(bank, references, keys),
        key_target(bank, references, keys), live_owner(bank, nodes, candidates),
        live_key(bank, keys, candidates) {
    bank.bind({&raw_rows,
               &raw_actions,
               &kind,
               &kind_mask,
               &cuq,
               &cuk,
               &prefix_ids,
               &prefix_offsets,
               &prefix_back_ids,
               &prefix_back_offsets,
               &policy_owner,
               &policy_lengths,
               &policy_mask,
               &records,
               &blocks,
               &selected,
               &valid,
               &control,
               &token_rows.ids,
               &token_rows.positions,
               &token_rows.offsets,
               &globals.ids,
               &globals.positions,
               &globals.offsets,
               &node_roots.ids,
               &node_roots.positions,
               &node_roots.offsets,
               &entity_refs.ids,
               &entity_refs.positions,
               &entity_refs.offsets,
               &key_actions.ids,
               &key_actions.positions,
               &key_actions.offsets,
               &key_source.ids,
               &key_source.positions,
               &key_source.offsets,
               &key_target.ids,
               &key_target.positions,
               &key_target.offsets,
               &live_owner.ids,
               &live_owner.positions,
               &live_owner.offsets,
               &live_key.ids,
               &live_key.positions,
               &live_key.offsets},
              device);
  }
  bool pack(const std::vector<Block> &trajectories, uint64_t step, double lr, bool enabled) {
    std::vector<std::shared_ptr<Node>> ns;
    for (auto &block : trajectories)
      for (auto &s : block.steps)
        ns.push_back(s.node);
    count = ns.size();
    if (!count || count > uint64_t(batch) || trajectories.size() > size_t(batch))
      return false;
    std::vector<int64_t> bootstrap;
    for (auto &b : trajectories) {
      bootstrap.push_back(b.bootstrap ? int64_t(batch + ns.size() - count) : -1);
      if (b.bootstrap)
        ns.push_back(b.bootstrap);
    }
    // 学习槽和 bootstrap 槽分开，实际有效长度通过整数表传给设备。
    std::vector<std::shared_ptr<Node>> ordered(nodes);
    std::copy_n(ns.begin(), count, ordered.begin());
    std::copy(ns.begin() + count, ns.end(), ordered.begin() + batch);
    std::vector<std::shared_ptr<Root>> rs;
    std::map<std::pair<uint64_t, uint64_t>, int64_t> root_map;
    std::vector<int64_t> node_root(nodes);
    for (size_t i = 0; i < ordered.size(); ++i)
      if (ordered[i]) {
        auto &r = ordered[i]->root;
        auto [entry, fresh] = root_map.try_emplace({r->game, r->id}, rs.size());
        if (fresh)
          rs.push_back(r);
        node_root[i] = entry->second;
      }
    if (rs.size() > size_t(roots))
      return false;
    Rows row_table;
    row_table.add(Row{});
    std::vector<int64_t> token_ids, kind_ids, kinds_valid, global_positions, cumulative{0},
        root_starts;
    used_tokens = 0;
    for (auto &r : rs) {
      if (r->rows.size() + 1 > size_t(maximum))
        return false;
      root_starts.push_back(token_ids.size());
      global_positions.push_back(token_ids.size());
      token_ids.push_back(row_table.add(r->globals));
      kind_ids.push_back(0);
      kinds_valid.push_back(0);
      for (size_t j = 0; j < r->rows.size(); ++j) {
        token_ids.push_back(row_table.add(r->rows[j]));
        kind_ids.push_back(r->kinds[j]);
        kinds_valid.push_back(1);
      }
      cumulative.push_back(token_ids.size());
    }
    used_tokens = token_ids.size();
    used_roots = rs.size();
    for (size_t j = rs.size(); j < size_t(roots); ++j) {
      global_positions.push_back(0);
      cumulative.push_back(token_ids.size());
    }
    if (token_ids.size() > size_t(tokens))
      return false;
    auto query_cumulative = cumulative;
    std::vector<int64_t> prefix, offsets{0}, back_ids;
    std::vector<std::vector<int64_t>> row_owners(rows);
    for (size_t i = 0; i < ordered.size(); ++i) {
      if (ordered[i])
        for (auto &r : ordered[i]->context) {
          auto id = row_table.add(r);
          if (id >= rows)
            return false;
          prefix.push_back(id);
          row_owners[id].push_back(i);
        }
      offsets.push_back(prefix.size());
    }
    if (row_table.rows.size() > size_t(rows) || prefix.size() > size_t(prefixes))
      return false;
    std::vector<int64_t> back_offsets{0};
    for (auto &owners : row_owners) {
      back_ids.insert(back_ids.end(), owners.begin(), owners.end());
      back_offsets.push_back(back_ids.size());
    }
    std::unordered_map<Action, int64_t, ArrayHash> action_map;
    std::vector<Action> action_rows;
    std::map<std::array<int64_t, 4>, int64_t> key_map;
    std::vector<int64_t> key_action, key_src, key_dst, owner, key_id, mask, lengths, selection;
    std::vector<int64_t> reference{0};
    std::map<int64_t, int64_t> ref_map{{0, 0}};
    auto ref = [&](int64_t value) {
      auto [entry, fresh] = ref_map.try_emplace(value, reference.size());
      if (fresh)
        reference.push_back(value);
      return entry->second;
    };
    std::vector<int64_t> record_rows(batch * 6), block_rows(batch * 3), valid_rows(batch);
    size_t transition = 0;
    for (size_t bi = 0; bi < trajectories.size(); ++bi) {
      auto &trajectory = trajectories[bi];
      block_rows[bi * 3] = transition;
      block_rows[bi * 3 + 1] = transition + trajectory.steps.size();
      block_rows[bi * 3 + 2] = bootstrap[bi];
      for (auto &s : trajectory.steps) {
        auto n = int64_t(s.node->mask.size());
        TORCH_CHECK(s.selected >= 0 && s.selected < n && s.node->mask[s.selected],
                    "invalid prepared action");
        auto at = record_rows.data() + transition * 6;
        at[0] = owner.size();
        at[1] = n;
        at[2] = s.selected;
        at[3] = s.reward;
        at[4] = s.terminal;
        at[5] = std::bit_cast<int64_t>(s.logp);
        selection.push_back(owner.size() + s.selected);
        valid_rows[transition] = 1;
        auto nr = node_root[transition];
        for (int64_t j = 0; j < n; ++j) {
          owner.push_back(transition);
          mask.push_back(s.node->mask[j]);
          if (!s.node->mask[j]) {
            key_id.push_back(0);
            continue;
          }
          auto &a = s.node->candidates[j]->value;
          auto [ai, fresh] = action_map.try_emplace(a, action_rows.size());
          if (fresh)
            action_rows.push_back(a);
          auto source = s.node->sources[j] < 0 ? 0 : root_starts[nr] + s.node->sources[j] + 2;
          auto target = s.node->targets[j] < 0 ? 0 : root_starts[nr] + s.node->targets[j] + 2;
          auto [ki, new_key] = key_map.try_emplace(
              std::array<int64_t, 4>{nr, ai->second, source, target}, key_map.size());
          if (new_key) {
            key_action.push_back(ai->second);
            key_src.push_back(ref(source));
            key_dst.push_back(ref(target));
          }
          key_id.push_back(ki->second);
        }
        lengths.push_back(n);
        ++transition;
      }
    }
    used_candidates = std::count(mask.begin(), mask.end(), int64_t(1));
    used_actions = action_rows.size();
    used_keys = key_action.size();
    for (size_t i = trajectories.size(); i < size_t(batch); ++i) {
      block_rows[i * 3] = count;
      block_rows[i * 3 + 1] = count;
      block_rows[i * 3 + 2] = -1;
    }
    for (size_t i = count; i < size_t(batch); ++i) {
      selection.push_back(owner.size());
      owner.push_back(i);
      key_id.push_back(0);
      mask.push_back(1);
      lengths.push_back(1);
    }
    if (owner.size() > size_t(candidates) || action_rows.size() > size_t(actions) ||
        key_action.size() > size_t(keys) || reference.size() > size_t(references))
      return false;
    lengths.back() += candidates - owner.size();
    auto live = int64_t(owner.size());
    owner.resize(candidates, batch - 1);
    key_id.resize(candidates, 0);
    mask.resize(candidates, 0);
    std::fill(bank.words.begin(), bank.words.end(), 0);
    bank.raw(raw_rows, row_table.rows);
    bank.raw(raw_actions, action_rows);
    bank.put(kind, kind_ids);
    bank.put(kind_mask, kinds_valid);
    bank.put(cuq, query_cumulative);
    bank.put(cuk, cumulative);
    bank.put(prefix_ids, prefix);
    bank.put(prefix_offsets, offsets);
    bank.put(prefix_back_ids, back_ids);
    bank.put(prefix_back_offsets, back_offsets);
    bank.put(policy_owner, owner);
    bank.put(policy_lengths, lengths);
    bank.put(policy_mask, mask);
    bank.put(records, record_rows);
    bank.put(blocks, block_rows);
    bank.put(selected, selection);
    bank.put(valid, valid_rows);
    bank.put(control,
             {enabled ? 1 : 0, int64_t(std::bit_cast<uint32_t>(float(lr))),
              int64_t(std::bit_cast<uint32_t>(float(1 / (1 - std::pow(0.9, double(step)))))),
              int64_t(std::bit_cast<uint32_t>(float(1 / (1 - std::pow(0.999, double(step))))))});
    token_rows.put(bank, token_ids);
    globals.put(bank, global_positions);
    node_roots.put(bank, node_root);
    entity_refs.put(bank, reference);
    key_actions.put(bank, key_action);
    key_source.put(bank, key_src);
    key_target.put(bank, key_dst);
    owner.resize(live);
    key_id.resize(live);
    live_owner.put(bank, owner);
    live_key.put(bank, key_id);
    return true;
  }
};
} // namespace hj::training
