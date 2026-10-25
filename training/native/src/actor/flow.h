#pragma once
#include "runtime.h"
#include "workset.h"
#include <condition_variable>
#include <optional>

namespace hj {
struct Counters {
  uint64_t consumed = 0, produced = 0, commands = 0, microsteps = 0, finished = 0, backtracks = 0,
           failures = 0;
};
enum class Flow : uint8_t { Idle, Cpu, RootQueued, RootRunning };
// 容量在行为开始前预留，直到对应真实转换提交后归还；包含未决、部分块和设备在途。
struct Admission {
  uint64_t capacity = 0;
  std::atomic<uint64_t> outstanding{0};
  std::atomic<bool> draining{false};
  bool acquire() {
    auto n = outstanding.load();
    while (!draining && n < capacity)
      if (outstanding.compare_exchange_weak(n, n + 1)) {
        if (!draining)
          return true;
        --outstanding;
        return false;
      }
    return false;
  }
};
struct Game {
  uint64_t id = 0, commands = 0, microsteps = 0, backtracks = 0, failures = 0;
  bool parked = false, parking = false, sent_training = false;
  uint32_t actor = 0, learner = 0;
  std::shared_ptr<Root> root;
  std::shared_ptr<Node> current, sent;
  uint64_t sent_version = 0;
  std::optional<Step> pending;
  std::vector<Step> partial;
  Bytes snapshot;
  std::chrono::steady_clock::time_point last_advance{};
  // 一个原子阶段表示所有权交接，不把两个独立 busy 位拼成易撕裂的快照。
  std::shared_ptr<std::atomic<Flow>> flow = std::make_shared<std::atomic<Flow>>(Flow::Idle);
  std::shared_ptr<std::mutex> progress_mutex = std::make_shared<std::mutex>();
  std::shared_ptr<std::atomic<bool>> pause = std::make_shared<std::atomic<bool>>(false);
  Counters progress;
  std::deque<Block> completed;
  uint64_t scored = 0;
  double policy_seconds = 0;
  bool workset_done = false;
  bool reserved = false;
  bool idle() const {
    return flow->load() == Flow::Idle;
  }
};
struct Job {
  uint64_t game;
  uint32_t op;
  Bytes bytes;
  std::shared_ptr<Node> node;
  CpuSnapshot weights;
  std::shared_ptr<Workset::Record> record;
  Game *state = nullptr;
  uint64_t chunk = 16, quota = 0;
  bool root_reply = false;
  double queued_ms = 0, root_queued_ms = 0, root_started_ms = 0;
};
struct Event {
  uint64_t game;
  uint32_t op;
  Bytes bytes;
  std::string error;
  uint32_t kind = 0;
};
struct Mailbox {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Event> events;
  std::atomic<size_t> bytes{0};
  std::atomic<uint64_t> revision{0};
  void push(Event e) {
    {
      std::lock_guard lock(mutex);
      bytes += e.bytes.capacity() + e.error.capacity();
      events.push_back(std::move(e));
      ++revision;
    }
    changed.notify_one();
  }
  std::optional<Event> pop() {
    std::lock_guard lock(mutex);
    if (events.empty())
      return {};
    Event e = std::move(events.front());
    events.pop_front();
    bytes -= e.bytes.capacity() + e.error.capacity();
    return e;
  }
  void wake() {
    {
      std::lock_guard lock(mutex);
      ++revision;
    }
    changed.notify_one();
  }
};
} // namespace hj
