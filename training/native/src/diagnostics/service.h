#pragma once
#include "actor/roots.h"
#include <iostream>
#include <sstream>
#include <torch/csrc/autograd/profiler_kineto.h>

namespace hj::diagnostic {
// 只读诊断：真实根按固定顺序循环，完成即补发，去掉规则与 CPU 策略等待。
// 复用正式根服务、图选择和回送；各请求拥有独立缓存，完整计算后立即消费并释放输出。
inline void service(Network &actor, const torch::Device &device, const Settings &settings,
                    const std::vector<std::shared_ptr<Node>> &fixture, CpuSnapshot snapshot,
                    uint64_t version) {
  const auto concurrency = settings.concurrency;
  const auto samples = settings.samples ? settings.samples : 32768;
  TORCH_CHECK(concurrency > 0 && concurrency <= 128 && fixture.size() == 32 &&
                  samples >= concurrency && samples % fixture.size() == 0,
              "service diagnostic requires concurrency 1..128 and samples divisible by 32");
  std::vector<Game> clients(concurrency);
  std::vector<std::vector<std::shared_ptr<Node>>> nodes(concurrency);
  for (uint64_t c = 0; c < concurrency; ++c)
    for (uint64_t i = 0; i < fixture.size(); ++i) {
      auto node = std::make_shared<Node>();
      node->root = std::make_shared<Root>(*fixture[i]->root);
      node->root->game = c + 1;
      node->root->id = i;
      nodes[c].push_back(std::move(node));
    }
  Mailbox mailbox;
  std::atomic<bool> paused{false}, publishing{false};
  std::atomic<CpuSnapshot> weights{snapshot};
  std::shared_mutex actor_weights;
  std::unique_ptr<RootService> roots;
  std::atomic<uint64_t> issued{0}, completed{0};
  std::atomic<uint64_t> quota{0};
  auto issue = [&](uint64_t client) {
    auto index = issued.fetch_add(1);
    if (index >= quota)
      return;
    Job job{client, 1, {}};
    job.node = nodes[client][index % fixture.size()];
    job.state = &clients[client];
    job.state->flow->store(Flow::RootQueued);
    TORCH_CHECK(roots->submit(job), "diagnostic root service stopped");
  };
  roots = std::make_unique<RootService>(
      actor, device, paused, publishing, weights, actor_weights, mailbox, settings.inference_batch,
      [&](Job job) {
        TORCH_CHECK(job.root_reply && job.weights->version == version &&
                        job.node->root->host_valid(actor.ptr().get(), actor->revision.load()),
                    "diagnostic root reply lost its snapshot or computed output");
        job.node->root->clear_cache();
        issue(job.game);
        if (completed.fetch_add(1) + 1 == quota)
          mailbox.wake();
      },
      settings.root_slots);
  auto pass = [&](uint64_t count) {
    quota = count;
    issued = completed = 0;
    for (uint64_t c = 0; c < concurrency; ++c)
      issue(c);
    while (completed != quota) {
      if (auto event = mailbox.pop())
        TORCH_CHECK(event->error.empty(), event->error);
      std::unique_lock lock(mailbox.mutex);
      mailbox.changed.wait_for(lock, std::chrono::milliseconds(100),
                               [&] { return completed == quota || !mailbox.events.empty(); });
    }
    TORCH_CHECK(mailbox.bytes == 0, "diagnostic request accounting differs");
  };
  auto elapsed = [](Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  auto warm = Clock::now();
  do {
    pass(std::max<uint64_t>(32, concurrency * 32));
  } while (elapsed(warm) < settings.warmup);
  if (!settings.profile.empty()) {
    namespace pi = torch::profiler::impl;
    pi::ExperimentalConfig experimental;
    experimental.profile_all_threads = true;
    pi::ProfilerConfig config(pi::ProfilerState::KINETO, false, false, false, false, false,
                              experimental);
    std::set<pi::ActivityType> activities{pi::ActivityType::CPU, pi::ActivityType::CUDA};
    torch::autograd::profiler::prepareProfiler(config, activities);
    torch::autograd::profiler::enableProfiler(config, activities);
  }
  auto before = roots->progress();
  auto stats_before = roots->stats();
  auto started = Clock::now();
  record("window_begin", 0);
  pass(samples);
  auto seconds = elapsed(started);
  record("window_end", 0);
  auto after = roots->progress();
  auto stats_after = roots->stats();
  roots->stop();
  TORCH_CHECK(after.nodes - before.nodes == samples, "diagnostic root count differs");
  if (!settings.profile.empty())
    torch::autograd::profiler::disableProfiler()->save(settings.profile.string());
  memory(device);
  save(settings.directory / "timing.csv");
  std::ostringstream out;
  out << std::setprecision(12) << "{\"seconds\":" << seconds << ",\"root_forwards\":" << samples
      << ",\"root_forwards_per_second\":" << samples / seconds << ",\"concurrency\":" << concurrency
      << ",\"root_slots\":" << settings.root_slots << ",\"batch\":" << settings.inference_batch
      << ",\"updates\":0"
      << ",\"batches\":" << after.batches - before.batches
      << ",\"graph_replays\":" << stats_after.replays - stats_before.replays
      << ",\"graph_fallbacks\":" << stats_after.fallbacks - stats_before.fallbacks
      << ",\"tokens\":" << stats_after.tokens - stats_before.tokens
      << ",\"token_capacity\":" << stats_after.token_capacity - stats_before.token_capacity
      << ",\"fixture_tokens\":[";
  for (size_t i = 0; i < fixture.size(); ++i)
    out << (i ? "," : "") << fixture[i]->root->rows.size() + 1;
  out << "]}\n";
  atomic_text(settings.directory / "window.json", out.str());
  std::cout << out.str();
}
} // namespace hj::diagnostic
