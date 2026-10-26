#pragma once
#include "diagnostics/timing.h"
#include "flow.h"
#include <shared_mutex>
#include <thread>

namespace hj {
// 根请求独立于控制与学习回执；完成后把匹配的头部快照直接交还所属 lane。
class RootService {
  struct Slot {
    std::shared_ptr<InferenceFlight> flight;
    std::vector<Job> jobs;
    CpuSnapshot weights;
  };
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Job> requests;
  std::atomic<bool> closing{false};
  uint64_t revision = 0;
  std::atomic<bool> &paused, &publishing;
  std::atomic<CpuSnapshot> &weights;
  std::shared_mutex &actor_weights;
  Mailbox &output;
  uint64_t batch_limit;
  std::function<void(Job)> send;
  std::unique_ptr<InferenceEngine> engine;
  std::vector<Slot> slots;
  std::atomic<uint64_t> nodes{0}, batches{0};
  std::atomic<double> seconds{0};
  std::thread thread;
  void deliver(Job job, const CpuSnapshot &snapshot) {
    if (snapshot)
      diagnostic::record("root", job.game, job.root_started_ms - job.root_queued_ms,
                         diagnostic::now() - job.root_started_ms,
                         diagnostic::now() - job.root_queued_ms);
    output.bytes -= job.bytes.capacity() + sizeof(Job);
    job.weights = snapshot;
    job.root_reply = bool(snapshot);
    job.state->flow->store(Flow::Cpu);
    send(std::move(job));
  }
  void complete() {
    for (auto &slot : slots) {
      if (!slot.flight || !slot.flight->complete.load(std::memory_order_acquire))
        continue;
      engine->finish(slot.flight);
      nodes += slot.jobs.size();
      ++batches;
      seconds += slot.flight->seconds;
      for (auto &job : slot.jobs)
        deliver(std::move(job), slot.weights);
      slot.jobs.clear();
      slot.weights.reset();
      slot.flight.reset();
    }
  }
  bool pending() {
    std::lock_guard lock(mutex);
    return !requests.empty();
  }
  void loop() {
    for (;;) {
      complete();
      if (closing || paused) {
        std::deque<Job> stopped;
        {
          std::lock_guard lock(mutex);
          stopped.swap(requests);
        }
        for (auto &job : stopped)
          deliver(std::move(job), {});
      } else if (!publishing) {
        for (size_t i = 0; i < slots.size(); ++i) {
          auto &slot = slots[i];
          if (slot.flight || !pending())
            continue;
          // 发布者持写锁等待设备时，服务仍须能处理已完成的结果。
          std::shared_lock lock(actor_weights, std::try_to_lock);
          if (!lock.owns_lock() || publishing)
            break;
          std::vector<Job> jobs;
          std::vector<Job> stopped;
          std::vector<std::shared_ptr<Node>> inputs;
          size_t work = 0;
          {
            std::lock_guard queue_lock(mutex);
            // 两槽都空闲时，较大的闭环工作集分开启动，覆盖另一槽的回送和回调。
            // 已有工作在途或积压超过 16 根时仍取完整批；小表保留合批以摊薄启动开销。
            auto limit = batch_limit;
            auto free =
                std::count_if(slots.begin(), slots.end(), [](auto &s) { return !s.flight; });
            if (free > 1) {
              size_t count = 0, tokens = 0;
              for (auto &job : requests)
                if (!*job.state->pause) {
                  ++count;
                  tokens += job.node->root->rows.size() + 1;
                }
              if (count <= 16 && tokens >= 1024)
                limit = std::min<uint64_t>(limit, (count + free - 1) / free);
            }
            while (!requests.empty()) {
              auto &next = requests.front();
              if (*next.state->pause) {
                stopped.push_back(std::move(next));
                requests.pop_front();
                continue;
              }
              auto cost = next.node->root->rows.size() + 1;
              if (!jobs.empty() && (jobs.size() >= limit || work + cost > 8192))
                break;
              work += cost;
              inputs.push_back(next.node);
              jobs.push_back(std::move(next));
              requests.pop_front();
            }
          }
          for (auto &job : stopped)
            deliver(std::move(job), {});
          if (jobs.empty())
            continue;
          slot.weights = weights.load();
          for (auto &job : jobs) {
            job.root_started_ms = diagnostic::now();
            job.state->flow->store(Flow::RootRunning);
          }
          try {
            slot.flight = engine->submit(i, std::move(inputs), slot.weights->version);
            slot.jobs = std::move(jobs);
          } catch (const std::exception &error) {
            for (auto &job : jobs) {
              output.bytes -= job.bytes.capacity() + sizeof(Job);
              job.state->flow->store(Flow::Cpu);
              output.push({job.game, 1, {}, error.what()});
            }
            slot.weights.reset();
          }
        }
      }
      bool inflight =
          std::any_of(slots.begin(), slots.end(), [](auto &slot) { return bool(slot.flight); });
      if (closing && !inflight && !pending())
        break;
      std::unique_lock lock(mutex);
      auto seen = revision;
      bool available =
          std::any_of(slots.begin(), slots.end(), [](auto &slot) { return !slot.flight; });
      if (!requests.empty() && available && !paused && !publishing && !closing)
        continue;
      changed.wait_for(lock, std::chrono::milliseconds(20), [&] {
        return revision != seen || std::any_of(slots.begin(), slots.end(), [](auto &slot) {
                 return slot.flight && slot.flight->complete.load(std::memory_order_acquire);
               });
      });
    }
  }

public:
  RootService(Network &actor, const torch::Device &device, std::atomic<bool> &paused,
              std::atomic<bool> &publishing, std::atomic<CpuSnapshot> &weights,
              std::shared_mutex &actor_weights, Mailbox &output, uint64_t batch_limit,
              std::function<void(Job)> send, size_t slot_count = 2)
      : paused(paused), publishing(publishing), weights(weights), actor_weights(actor_weights),
        output(output), batch_limit(batch_limit), send(std::move(send)),
        engine(std::make_unique<InferenceEngine>(
            actor, device, [this] { wake(); }, slot_count, batch_limit)),
        slots(slot_count) {
    thread = std::thread([this] { loop(); });
  }
  bool submit(Job &job) {
    {
      std::lock_guard lock(mutex);
      if (closing)
        return false;
      output.bytes += job.bytes.capacity() + sizeof(Job);
      job.weights.reset();
      job.root_queued_ms = diagnostic::now();
      requests.push_back(std::move(job));
      ++revision;
    }
    changed.notify_one();
    return true;
  }
  void wake() {
    {
      std::lock_guard lock(mutex);
      ++revision;
    }
    changed.notify_one();
  }
  void stop() {
    closing = true;
    wake();
    if (thread.joinable())
      thread.join();
  }
  ~RootService() {
    stop();
  }
  void wait_reads() {
    engine->wait_reads();
  }
  InferenceStats stats() const {
    return engine->stats();
  }
  RootStats progress() const {
    return {nodes.load(), batches.load(), seconds.load()};
  }
  bool queued() {
    return pending();
  }
};
} // namespace hj
