#pragma once
#include "mcts.h"
#include "search.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <map>

namespace hj {
// 有界历史池；在途对局引用的旧模型继续存活，恢复时保留引用而非重新抽签。
class OpponentHistory {
  mutable std::mutex mutex;
  std::vector<Frozen> pool;

public:
  std::atomic<Frozen> latest;
  uint64_t limit = 8, interval = 64, percent = 20;
  void publish(const Frozen &policy) {
    std::lock_guard lock(mutex);
    if (pool.empty() || policy->version >= pool.back()->version + interval) {
      pool.push_back(policy);
      if (pool.size() > limit)
        pool.erase(pool.begin());
    }
    latest.store(policy);
  }
  Frozen choose(uint64_t game, uint32_t seed) const {
    std::lock_guard lock(mutex);
    if ((game * 37) % 100 >= percent)
      return {};
    std::vector<Frozen> eligible;
    auto current = latest.load();
    for (auto &policy : pool)
      if (policy->version < current->version)
        eligible.push_back(policy);
    if (eligible.empty())
      return {};
    mcts::Random random(uint32_t(game * 2654435761u) ^ seed);
    return eligible[random.next() % eligible.size()];
  }
  std::vector<Frozen> snapshots() const {
    std::lock_guard lock(mutex);
    return pool;
  }
  void restore(std::vector<Frozen> policies) {
    std::lock_guard lock(mutex);
    if (policies.size() > limit)
      policies.erase(policies.begin(), policies.end() - limit);
    pool = std::move(policies);
  }
  void save(torch::serialize::OutputArchive &archive,
            const std::vector<std::pair<uint64_t, Frozen>> &bindings) const {
    torch::NoGradGuard guard;
    auto policies = snapshots();
    std::map<uint64_t, Frozen> all;
    std::vector<int64_t> versions, active, members;
    for (auto &policy : policies) {
      all.emplace(policy->version, policy);
      members.push_back(policy->version);
    }
    for (auto &[game, policy] : bindings) {
      all.emplace(policy->version, policy);
      active.insert(active.end(), {int64_t(game), int64_t(policy->version)});
    }
    for (auto &[version, policy] : all) {
      versions.push_back(version);
      archive.write("history_weights_" + std::to_string(version),
                    policy->model->packed_weights.to(torch::kCPU).clone());
    }
    archive.write("history_versions", torch::tensor(versions, torch::kInt64));
    archive.write("history_members", torch::tensor(members, torch::kInt64));
    archive.write("history_bindings", torch::tensor(active, torch::kInt64));
  }
  std::map<uint64_t, Frozen> load(torch::serialize::InputArchive &archive, Network &source,
                                  uint64_t current_version) {
    torch::NoGradGuard guard;
    T versions, members, active;
    if (!archive.try_read("history_versions", versions))
      return {};
    archive.read("history_members", members);
    archive.read("history_bindings", active);
    for (auto &tensor : {versions, members, active})
      TORCH_CHECK(tensor.scalar_type() == torch::kInt64 && tensor.dim() == 1 &&
                      tensor.ge(0).all().item<bool>(),
                  "invalid history metadata");
    TORCH_CHECK(active.numel() % 2 == 0, "invalid history pool or bindings");
    std::map<uint64_t, Frozen> all, bindings;
    for (int64_t i = 0; i < versions.numel(); ++i) {
      auto version = versions[i].item<int64_t>();
      TORCH_CHECK(uint64_t(version) <= current_version && !all.count(version),
                  "invalid historical model version");
      T weights;
      archive.read("history_weights_" + std::to_string(version), weights);
      TORCH_CHECK(weights.scalar_type() == torch::kBFloat16 &&
                      weights.sizes() == source->packed_weights.sizes() &&
                      torch::isfinite(weights).all().item<bool>(),
                  "invalid historical BF16 weights");
      auto policy = std::make_shared<FrozenPolicy>(source, version);
      policy->model->packed_weights.copy_(weights.to(source->packed_weights.device()));
      if (source->packed_weights.is_cuda()) {
        c10::impl::VirtualGuardImpl implementation(source->packed_weights.device().type());
        implementation.synchronizeStream(implementation.getStream(source->packed_weights.device()));
      }
      policy->cpu = CpuPolicy::snapshot(policy->model, version);
      all.emplace(version, policy);
    }
    std::vector<Frozen> policies;
    uint64_t previous = 0;
    for (int64_t i = 0; i < members.numel(); ++i) {
      auto version = members[i].item<int64_t>();
      TORCH_CHECK(all.count(version) && (i == 0 || uint64_t(version) > previous),
                  "history pool versions are not ordered");
      previous = version;
      policies.push_back(all.at(version));
    }
    for (int64_t i = 0; i < active.numel(); i += 2) {
      auto game = active[i].item<int64_t>(), version = active[i + 1].item<int64_t>();
      TORCH_CHECK(game > 0 && all.count(version) && !bindings.count(game),
                  "invalid historical opponent binding");
      bindings.emplace(game, all.at(version));
    }
    restore(std::move(policies));
    return bindings;
  }
};
} // namespace hj
