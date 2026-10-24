#include "runtime.h"
#include "actor/roots.h"
#include "engine.h"
#include "storage/checkpoint.h"
#include "workset.h"
#include <ATen/record_function.h>
#include <atomic>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <thread>
#include <torch/csrc/autograd/profiler_kineto.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace hj {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace {
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
int64_t utc_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
void replace(const fs::path &source, const fs::path &target) {
#ifdef _WIN32
  auto start = Clock::now();
  for (;;) {
    if (MoveFileExW(source.c_str(), target.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
      break;
    auto error = GetLastError();
    // 状态轮询客户端可能短暂未共享删除权限，等待其关闭读取句柄。
    if ((error != ERROR_SHARING_VIOLATION && error != ERROR_ACCESS_DENIED) || elapsed(start) > 0.1)
      throw std::runtime_error("atomic file replacement failed: " + target.string() + " (Windows " +
                               std::to_string(error) + ")");
    Sleep(1);
  }
#else
  fs::rename(source, target);
#endif
}
std::string read_text(const fs::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read " + path.string());
  return {std::istreambuf_iterator<char>(in), {}};
}
void write_step(Bytes &b, const Step &s, GraphWriter &graph) {
  graph.ref(b, s.node);
  put(b, s.selected);
  put(b, s.logp);
  put(b, s.version);
  put<uint8_t>(b, s.terminal);
  put(b, s.reward);
}
Step read_step(Reader &r, const GraphReader &graph) {
  Step s;
  s.node = graph.ref(r);
  s.selected = r.get<int64_t>();
  s.logp = r.get<double>();
  s.version = r.get<uint64_t>();
  s.terminal = r.get<uint8_t>() != 0;
  s.reward = r.get<int8_t>();
  return s;
}
void write_block(Bytes &b, const Block &block, GraphWriter &graph) {
  put<uint64_t>(b, block.steps.size());
  for (auto &s : block.steps)
    write_step(b, s, graph);
  graph.ref(b, block.bootstrap);
}
Block read_block(Reader &r, const GraphReader &graph) {
  Block b;
  auto count = r.get<uint64_t>();
  for (uint64_t i = 0; i < count; ++i)
    b.steps.push_back(read_step(r, graph));
  b.bootstrap = graph.ref(r);
  return b;
}

// 每个模型有固定提交线程和设备流；只在空闲时等待，不为每次更新创建线程。
class Executor {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<std::function<void()>> jobs;
  bool closing = false;
  Mailbox &output;
  std::optional<c10::Stream> stream;
  std::thread thread;
  static std::optional<c10::Stream> new_stream(const torch::Device &device, bool priority) {
    if (!device.is_cuda())
      return {};
    return c10::impl::VirtualGuardImpl(device.type()).getStreamFromGlobalPool(device, priority);
  }

public:
  Executor(Mailbox &out, const torch::Device &device, bool priority)
      : output(out), stream(new_stream(device, priority)), thread([this] {
          c10::OptionalStreamGuard stream_guard(stream);
          while (true) {
            std::function<void()> job;
            {
              std::unique_lock lock(mutex);
              changed.wait(lock, [&] { return closing || !jobs.empty(); });
              if (jobs.empty())
                break;
              job = std::move(jobs.front());
              jobs.pop_front();
            }
            job();
            output.wake();
          }
        }) {}
  template <class F> auto submit(F function) {
    using R = std::invoke_result_t<F>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::move(function));
    auto future = task->get_future();
    {
      std::lock_guard lock(mutex);
      jobs.push_back([task] { (*task)(); });
    }
    changed.notify_one();
    return future;
  }
  ~Executor() {
    {
      std::lock_guard lock(mutex);
      closing = true;
    }
    changed.notify_one();
    thread.join();
  }
};
void synchronize(const torch::Device &device) {
  if (device.is_cuda()) {
    c10::impl::VirtualGuardImpl implementation(device.type());
    implementation.synchronizeStream(implementation.getStream(device));
  }
}
class Lane {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Job> jobs;
  std::atomic<bool> closing{false};
  Mailbox &output;
  std::atomic<bool> &paused;
  std::atomic<CpuSnapshot> &weights;
  std::function<bool(Job &)> request_root;
  std::atomic<uint64_t> &cache_hits;
  std::thread thread;
  static bool host_ready(const Root &root, const CpuSnapshot &weights) {
    return weights && root.host_valid(weights->owner, weights->revision);
  }
  static void successor(Game &g, const std::shared_ptr<Node> &next, const Job &job,
                        std::optional<int8_t> reward = {}) {
    if (g.pending) {
      auto step = std::move(*g.pending);
      g.pending.reset();
      if (reward) {
        step.terminal = true;
        step.reward = *reward;
      }
      g.partial.push_back(std::move(step));
      ++g.progress.produced;
      if (job.record && ++job.record->steps >= job.quota)
        g.workset_done = true;
    }
    if (!g.partial.empty() && (g.partial.size() >= job.chunk || reward || g.workset_done)) {
      g.completed.push_back({std::move(g.partial), next});
      g.partial.clear();
    }
  }

public:
  Lane(Mailbox &out, std::atomic<bool> &paused, std::atomic<CpuSnapshot> &weights,
       std::function<bool(Job &)> request_root, std::atomic<uint64_t> &cache_hits)
      : output(out), paused(paused), weights(weights), request_root(std::move(request_root)),
        cache_hits(cache_hits), thread([this] {
          void *actor = hj_actor_create();
          CpuPolicy policy;
          InputPool inputs;
          auto last_trim = Clock::now();
          uint64_t active_games = 0;
          while (true) {
            Job job;
            {
              std::unique_lock<std::mutex> lock(mutex);
              changed.wait(lock, [&] { return closing || !jobs.empty(); });
              if (jobs.empty())
                break;
              job = std::move(jobs.front());
              jobs.pop_front();
            }
            Event e{job.game, job.op, {}, {}};
            if (job.op == 1 && (this->paused || *job.state->pause || closing)) {
              e.kind = job.node ? 0 : 2;
              output.bytes -= job.bytes.capacity();
              output.push(std::move(e));
              continue;
            }
            if (!actor)
              e.error = "cannot initialize Rust actor lane";
            else
              try {
                Bytes reply;
                const Bytes *input = &job.bytes;
                if (job.node) {
                  if (!job.weights)
                    job.weights = this->weights.load();
                  if (!host_ready(*job.node->root, job.weights)) {
                    auto bytes = job.bytes.capacity();
                    job.state->flow->store(Flow::RootQueued);
                    if (this->request_root(job)) {
                      output.bytes -= bytes;
                      continue;
                    }
                    job.state->flow->store(Flow::Cpu);
                    output.bytes -= bytes;
                    output.push(std::move(e));
                    continue;
                  }
                  {
                    std::lock_guard lock(*job.state->progress_mutex);
                    job.state->sent = job.node;
                    job.state->sent_training = job.state->actor == job.state->learner;
                    job.state->sent_version = job.weights->version;
                  }
                  if (!job.root_reply)
                    ++this->cache_hits;
                  job.root_reply = false;
                  auto start = Clock::now();
                  auto mass = policy.score(*job.node, job.weights);
                  {
                    std::lock_guard lock(*job.state->progress_mutex);
                    job.state->policy_seconds += elapsed(start);
                    ++job.state->scored;
                  }
                  if (job.record)
                    mass = Workset::probabilities(*job.record, std::move(mass), job.node->mask);
                  put(reply, job.game);
                  put<uint32_t>(reply, mass.size());
                  raw(reply, mass.data(), mass.size() * 2);
                  input = &reply;
                }
                const uint8_t *p = nullptr;
                size_t n = 0;
                RECORD_FUNCTION("haojie::rust_actor", std::vector<c10::IValue>());
                int status = hj_actor_call(actor, job.op, input->data(), input->size(), &p, &n);
                if (status)
                  e.error = std::string(reinterpret_cast<const char *>(p), n);
                else {
                  if (job.op == 1) {
                    auto &g = *job.state;
                    bool again = false, completed = false;
                    {
                      std::lock_guard lock(*g.progress_mutex);
                      auto before = g.completed.size();
                      auto f = parse(Reader(p, n), g.root, inputs);
                      if (f.commands > g.commands || f.microsteps > g.microsteps)
                        g.last_advance = Clock::now();
                      g.progress.commands += f.commands - g.commands;
                      g.progress.microsteps += f.microsteps - g.microsteps;
                      g.progress.backtracks += f.backtracks - g.backtracks;
                      g.progress.failures += f.failures - g.failures;
                      g.commands = f.commands;
                      g.microsteps = f.microsteps;
                      g.backtracks = f.backtracks;
                      g.failures = f.failures;
                      g.actor = f.actor;
                      g.learner = f.learner;
                      if (f.selected >= 0) {
                        TORCH_CHECK(g.sent, "selection without behavior node");
                        if (g.sent_training) {
                          TORCH_CHECK(!g.pending, "missing same-seat successor");
                          g.pending = Step{g.sent, f.selected, f.logp, g.sent_version, false, 0};
                        }
                        g.sent.reset();
                      }
                      g.current = f.node;
                      e.kind = f.kind;
                      if (f.kind == 1) {
                        auto reward =
                            f.winner == 0 ? 0 : (uint32_t(f.winner) == g.learner ? 1 : -1);
                        successor(g, {}, job, int8_t(reward));
                        ++g.progress.finished;
                        if (job.record) {
                          job.record->terminal = true;
                          g.workset_done = true;
                        }
                      } else {
                        if (f.kind == 0 && g.actor == g.learner)
                          successor(g, g.current, job);
                        if (!this->paused && !*g.pause && !closing && !g.workset_done) {
                          job.weights = this->weights.load();
                          again = true;
                          if (again && f.kind == 0) {
                            job.node = g.current;
                          } else
                            job.node.reset();
                        }
                      }
                      completed = g.completed.size() > before;
                    }
                    if (completed)
                      output.wake();
                    if (elapsed(last_trim) >= 0.25) {
                      inputs.trim();
                      last_trim = Clock::now();
                    }
                    if (again) {
                      output.bytes -= job.bytes.capacity();
                      job.bytes.clear();
                      put(job.bytes, job.game);
                      put<uint32_t>(job.bytes, 0);
                      submit(std::move(job));
                      continue;
                    }
                  } else if (n)
                    e.bytes.assign(p, p + n);
                  if (job.op == 0 || job.op == 4)
                    ++active_games;
                  if (job.op == 3 || job.op == 5)
                    --active_games;
                  if (!active_games)
                    policy.clear();
                  if (job.op == 3 || job.op == 5)
                    inputs.trim();
                }
              } catch (const std::exception &error) {
                e.error = error.what();
              }
            output.bytes -= job.bytes.capacity();
            output.push(std::move(e));
          }
          hj_actor_close(actor);
        }) {}
  void submit(Job job) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      output.bytes += job.bytes.capacity();
      jobs.push_back(std::move(job));
    }
    changed.notify_one();
  }
  ~Lane() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      closing = true;
    }
    changed.notify_one();
    thread.join();
  }
};
std::pair<Settings, std::shared_ptr<torch::serialize::InputArchive>> resume_settings(Settings s) {
  if (s.resume.empty())
    return {std::move(s), {}};
  auto archive = std::make_shared<torch::serialize::InputArchive>();
  std::ifstream file(s.resume, std::ios::binary);
  archive->load_from(file, torch::kCPU);
  T config;
  archive->read("config", config);
  TORCH_CHECK(config.scalar_type() == torch::kInt64 && config.dim() == 1 && config.numel() == 13,
              "invalid checkpoint configuration");
  auto c = config.data_ptr<int64_t>();
  auto fixed = [&](const std::string &name, uint64_t &field, size_t index) {
    auto value = uint64_t(c[index]);
    if (s.supplied.count(name) && field != value)
      throw std::runtime_error("resume configuration differs: " + name);
    field = value;
  };
  fixed("--width", s.width, 0);
  fixed("--seed", s.seed, 1);
  fixed("--chunk", s.chunk, 2);
  fixed("--batch", s.batch, 3);
  fixed("--publish", s.publish, 7);
  if (!s.supplied.count("--concurrency"))
    s.concurrency = c[4];
  if (!s.supplied.count("--workers"))
    s.workers = c[5];
  if (!s.supplied.count("--memory-mb"))
    s.memory_mb = c[6];
  if (!s.supplied.count("--inference-batch"))
    s.inference_batch = c[8];
  if (s.supplied.count("--shrine") && !c[9])
    throw std::runtime_error("resume rules mode differs");
  s.shrine = c[9] != 0;
  if (!s.supplied.count("--device"))
    s.device = c[10] ? (c[11] < 0 ? "cuda" : "cuda:" + std::to_string(c[11])) : "cpu";
  if (!s.supplied.count("--save-seconds"))
    s.save_seconds = double(c[12]) / 1000000;
  return {std::move(s), std::move(archive)};
}
class Session {
  Settings settings;
  torch::Device device;
  Network learner, actor;
  std::atomic<CpuSnapshot> cpu_weights;
  Optimizer optimizer;
  Mailbox mailbox;
  InputPool inputs;
  std::atomic<bool> sampling_paused{false};
  bool quiescing = false;
  std::vector<std::unique_ptr<Lane>> lanes;
  std::map<uint64_t, Game> games;
  std::deque<Block> ready;
  Counters counters, initial;
  uint64_t next_id = 1, published = 0, initial_updates = 0, committed_updates = 0;
  uint64_t committed_graph_replays = 0, committed_graph_fallbacks = 0;
  std::array<uint64_t, 4> committed_graph_work{};
  std::atomic<uint64_t> actor_version{0};
  std::shared_mutex actor_weights;
  std::atomic<bool> publishing{false};
  std::future<TrainResult> learning;
  struct Commit {
    TrainResult result;
    uint64_t updates;
    double seconds;
  };
  std::optional<Commit> learning_commit;
  std::vector<Block> in_learning;
  std::future<void> writer;
  Clock::time_point start, last_status, last_save, activity_at;
  double active_game_seconds = 0;
  int64_t window_start_unix_ms = 0;
  uint64_t infer_nodes = 0, peak_active = 0, peak_managed = 0;
  double learning_seconds = 0, inference_seconds = 0;
  Clock::time_point learn_start;
  std::unique_ptr<Executor> learning_executor;
  std::unique_ptr<RootService> roots;
  std::unique_ptr<LearningEngine> learning_engine;
  std::unique_ptr<Workset> workset;

public:
  explicit Session(Settings s) : Session(resume_settings(std::move(s))) {}
  explicit Session(std::pair<Settings, std::shared_ptr<torch::serialize::InputArchive>> resumed)
      : settings(std::move(resumed.first)), device(settings.device), learner(settings.width),
        actor(settings.width), optimizer(learner) {
    learner->to(device);
    actor->to(device);
    // 优化器在设备迁移后重建，浮点参数及状态始终为 BF16。
    optimizer = Optimizer(learner);
    copy_weights(learner, actor);
    if (settings.workset_games) {
      if ((settings.workset_games * settings.game_samples + settings.batch - 1) / settings.batch >
          settings.publish)
        throw std::runtime_error("workset must finish before its first actor publication");
      workset =
          std::make_unique<Workset>(settings.workset_games, settings.game_samples, settings.replay);
    }
    for (uint64_t i = 0; i < settings.workers; ++i)
      lanes.push_back(std::make_unique<Lane>(
          mailbox, sampling_paused, cpu_weights, [this](Job &job) { return roots->submit(job); },
          actor->cache_hits));
    if (resumed.second)
      restore(*resumed.second);
    // 主线程的初始化完成后，持久流才可读取模型和优化器。
    synchronize(learner->parameters()[0].device());
    cpu_weights.store(CpuPolicy::snapshot(actor, actor_version.load()));
    learning_engine = std::make_unique<LearningEngine>(learner, optimizer, settings.batch);
    roots = std::make_unique<RootService>(
        actor, device, sampling_paused, publishing, cpu_weights, actor_weights, mailbox,
        settings.inference_batch,
        [this](Job job) { lanes[job.game % lanes.size()]->submit(std::move(job)); });
    learning_executor =
        std::make_unique<Executor>(mailbox, learner->parameters()[0].device(), false);
    initial = counters;
    initial_updates = committed_updates = optimizer.steps;
    start = last_status = last_save = activity_at = Clock::now();
    window_start_unix_ms = utc_ms();
  }
  ~Session() {
    sampling_paused = true;
    learning_executor.reset();
    roots->stop();
    for (auto &lane : lanes)
      lane.reset();
    roots.reset();
  }
  void submit(Game &g, uint32_t op, Bytes b) {
    if (!g.idle())
      throw std::runtime_error("actor already has work");
    g.flow->store(Flow::Cpu);
    Job job{g.id, op, std::move(b)};
    job.state = &g;
    job.chunk = settings.chunk;
    if (workset) {
      job.record = workset->records.at(g.id);
      job.quota = workset->quota;
    }
    lanes[g.id % lanes.size()]->submit(std::move(job));
  }
  void enqueue(Game &g) {
    if (g.idle() && !g.parked && !g.parking && !sampling_paused && !*g.pause && g.current) {
      g.flow->store(Flow::Cpu);
      Job job{g.id, 1, {}};
      job.state = &g;
      job.chunk = settings.chunk;
      job.node = g.current;
      if (workset) {
        job.record = workset->records.at(g.id);
        job.quota = workset->quota;
      }
      lanes[g.id % lanes.size()]->submit(std::move(job));
    }
  }
  static Bytes id_packet(uint64_t id) {
    Bytes b;
    put(b, id);
    return b;
  }
  uint64_t ready_steps() const {
    uint64_t total = 0;
    for (auto &b : ready)
      total += b.steps.size();
    return total;
  }
  uint64_t active() const {
    uint64_t n = 0;
    for (auto &pair : games)
      if (!pair.second.parked)
        ++n;
    return n;
  }
  void accrue_activity() {
    auto now = Clock::now();
    active_game_seconds += active() * std::chrono::duration<double>(now - activity_at).count();
    activity_at = now;
  }
  uint64_t managed() const {
    RECORD_FUNCTION("haojie::managed", std::vector<c10::IValue>());
    uint64_t bytes = hj_memory_bytes() + StorageBytes::total.load(std::memory_order_relaxed) +
                     mailbox.bytes.load();
    if (workset)
      for (auto &p : workset->records)
        bytes += p.second->bytes.load();
    for (auto &pair : games) {
      auto &g = pair.second;
      std::lock_guard lock(*g.progress_mutex);
      bytes += g.snapshot.capacity() + sizeof(Game) + g.partial.capacity() * sizeof(Step);
      for (auto &b : g.completed)
        bytes += sizeof(Block) + b.steps.capacity() * sizeof(Step);
    }
    for (auto &b : ready)
      bytes += sizeof(Block) + b.steps.capacity() * sizeof(Step);
    for (auto &b : in_learning)
      bytes += sizeof(Block) + b.steps.capacity() * sizeof(Step);
    return bytes;
  }
  void progress_locked(Game &g) {
    counters.produced += std::exchange(g.progress.produced, 0);
    counters.commands += std::exchange(g.progress.commands, 0);
    counters.microsteps += std::exchange(g.progress.microsteps, 0);
    counters.backtracks += std::exchange(g.progress.backtracks, 0);
    counters.failures += std::exchange(g.progress.failures, 0);
    counters.finished += std::exchange(g.progress.finished, 0);
    infer_nodes += std::exchange(g.scored, 0);
    inference_seconds += std::exchange(g.policy_seconds, 0);
    while (!g.completed.empty()) {
      ready.push_back(std::move(g.completed.front()));
      g.completed.pop_front();
    }
    if (workset && g.workset_done)
      workset->done.insert(g.id);
  }
  void event(Event e) {
    accrue_activity();
    if (!e.error.empty())
      throw std::runtime_error(e.error);
    auto found = games.find(e.game);
    if (found == games.end())
      throw std::runtime_error("unknown lane event");
    auto &g = found->second;
    {
      std::lock_guard lock(*g.progress_mutex);
      progress_locked(g);
    }
    g.flow->store(Flow::Idle);
    if (e.op == 0 || e.op == 4) {
      if (e.op == 4) {
        g.snapshot.clear();
        g.parked = false;
        *g.pause = false;
      }
      if (!g.current) {
        Bytes b = id_packet(g.id);
        put<uint32_t>(b, 0);
        submit(g, 1, std::move(b));
      } else
        enqueue(g);
    } else if (e.op == 1) {
      if (e.kind == 1) {
        submit(g, 5, id_packet(g.id));
      } else {
        if (g.parking || (workset && workset->done.count(g.id))) {
          g.parking = true;
          submit(g, 3, id_packet(g.id));
        } else if (e.kind == 2 && !sampling_paused) {
          Bytes b = id_packet(g.id);
          put<uint32_t>(b, 0);
          submit(g, 1, std::move(b));
        } else
          enqueue(g);
      }
    } else if (e.op == 2 || e.op == 3) {
      g.snapshot = std::move(e.bytes);
      if (e.op == 3) {
        g.parked = true;
        g.parking = false;
        if (g.current && g.actor == g.learner && !g.partial.empty()) {
          ready.push_back({std::move(g.partial), g.current});
          g.partial.clear();
        }
        g.current.reset();
        if (g.root)
          g.root->clear_cache();
        g.root.reset();
      }
    } else if (e.op == 5)
      games.erase(found);
  }
  void collect() {
    RECORD_FUNCTION("haojie::collect", std::vector<c10::IValue>());
    while (auto e = mailbox.pop())
      event(std::move(*e));
    for (auto &entry : games) {
      std::lock_guard lock(*entry.second.progress_mutex);
      progress_locked(entry.second);
    }
  }
  void settle() {
    for (;;) {
      collect();
      bool busy = false;
      for (auto &p : games)
        busy |= !p.second.idle();
      if (!busy)
        break;
      std::unique_lock<std::mutex> lock(mailbox.mutex);
      mailbox.changed.wait_for(lock, std::chrono::milliseconds(10));
    }
  }
  void collect_learning(bool wait = false) {
    if (learning.valid() &&
        (wait || learning.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
      learning.get();
    }
    std::optional<Commit> commit;
    {
      std::lock_guard lock(mailbox.mutex);
      commit = std::exchange(learning_commit, std::nullopt);
    }
    if (commit) {
      committed_updates = commit->updates;
      counters.consumed += commit->result.consumed;
      committed_graph_replays = learning_engine->replays();
      committed_graph_fallbacks = learning_engine->fallbacks();
      committed_graph_work = learning_engine->workload();
      learning_seconds += commit->seconds;
      in_learning.clear();
    }
  }
  void train() {
    RECORD_FUNCTION("haojie::train", std::vector<c10::IValue>());
    collect_learning();
    if (learning.valid() || ready.empty())
      return;
    uint64_t desired = settings.batch;
    if (settings.samples)
      desired = std::min(desired, settings.samples - (counters.consumed - initial.consumed));
    if (!desired)
      return;
    uint64_t n = 0;
    while (!ready.empty() && n < desired) {
      auto &block = ready.front();
      auto take = std::min<uint64_t>(block.steps.size(), desired - n);
      if (take == block.steps.size()) {
        n += take;
        in_learning.push_back(std::move(block));
        ready.pop_front();
      } else {
        Block head;
        head.steps.assign(std::make_move_iterator(block.steps.begin()),
                          std::make_move_iterator(block.steps.begin() + take));
        head.bootstrap = block.steps[take].node;
        block.steps.erase(block.steps.begin(), block.steps.begin() + take);
        in_learning.push_back(std::move(head));
        n += take;
      }
    }
    if (n < desired && !(workset && workset->complete()) &&
        managed() < settings.memory_mb * 1024 * 1024 / 2) {
      for (auto it = in_learning.rbegin(); it != in_learning.rend(); ++it)
        ready.push_front(std::move(*it));
      in_learning.clear();
      return;
    }
    learn_start = Clock::now();
    learning = learning_executor->submit([this] {
      auto result = learning_engine->step(in_learning);
      {
        std::lock_guard lock(mailbox.mutex);
        learning_commit = Commit{result, optimizer.steps, elapsed(learn_start)};
      }
      mailbox.changed.notify_one();
      if (optimizer.steps - published >= settings.publish) {
        // 前向与反向不持推理锁。只在整批推理结束后发布，并等待拷贝完成再交权。
        publishing = true;
        std::unique_lock lock(actor_weights);
        roots->wait_reads();
        copy_weights(learner, actor);
        synchronize(learner->parameters()[0].device());
        cpu_weights.store(CpuPolicy::snapshot(actor, optimizer.steps));
        actor_version = published = optimizer.steps;
        publishing = false;
        roots->wake();
      }
      return result;
    });
  }
  void resize() {
    RECORD_FUNCTION("haojie::resize", std::vector<c10::IValue>());
    accrue_activity();
    uint64_t running = 0;
    for (auto &entry : games)
      running += !entry.second.parked && !entry.second.parking;
    // 大编号先停放；在途原子规则操作完成后立即停放，进度原样保留。
    for (auto it = games.rbegin(); it != games.rend() && running > settings.concurrency; ++it) {
      auto &g = it->second;
      if (!g.parked && !g.parking) {
        g.parking = true;
        *g.pause = true;
        if (g.idle())
          submit(g, 3, id_packet(g.id));
        --running;
      }
    }
    for (auto &p : games) {
      auto &g = p.second;
      if (running >= settings.concurrency || active() >= settings.concurrency)
        break;
      if (g.parked && g.idle() && !(workset && workset->done.count(g.id))) {
        g.parked = false;
        *g.pause = false;
        submit(g, 4, g.snapshot);
        ++running;
      }
    }
    while (running < settings.concurrency && active() < settings.concurrency &&
           managed() < settings.memory_mb * 1024 * 1024 / 2) {
      if (std::any_of(games.begin(), games.end(), [&](auto &entry) {
            return entry.second.parking && !(workset && workset->done.count(entry.first));
          }))
        break;
      if (workset && next_id > workset->games)
        break;
      Game game;
      game.id = next_id++;
      game.learner = uint32_t(1 + (game.id & 1));
      auto id = game.id;
      games.emplace(id, std::move(game));
      if (workset)
        workset->opened(id);
      Bytes b = id_packet(id);
      put<uint32_t>(b, uint32_t(settings.seed + id * 2654435761u));
      put<uint32_t>(b, games[id].learner);
      put<uint32_t>(b, settings.shrine);
      submit(games[id], 0, std::move(b));
      ++running;
    }
    peak_active = std::max(peak_active, running);
  }
  void sampling_budget() {
    uint64_t budget = settings.memory_mb * 1024 * 1024;
    if (managed() > budget * 3 / 4)
      inputs.trim();
    if (managed() > budget * 3 / 4) {
      sampling_paused = true;
      roots->wake();
      for (auto &p : games) {
        auto &g = p.second;
        std::lock_guard lock(*g.progress_mutex);
        progress_locked(g);
        if (g.idle() && !g.partial.empty() && g.current && g.actor == g.learner) {
          ready.push_back({std::move(g.partial), g.current});
          g.partial.clear();
        }
      }
      if (!ready.empty() || learning.valid())
        return;
    }
    if (sampling_paused && !quiescing) {
      sampling_paused = false;
      roots->wake();
      for (auto &entry : games) {
        auto &g = entry.second;
        if (g.idle() && !g.parked && !g.parking) {
          if (g.current)
            enqueue(g);
          else {
            auto b = id_packet(g.id);
            put<uint32_t>(b, 0);
            submit(g, 1, std::move(b));
          }
        }
      }
    }
  }
  void save_workset() {
    if (!workset)
      return;
    auto path = settings.directory / "workset";
    fs::create_directory(path);
    std::ostringstream index;
    index << "{\"complete\":" << (workset->complete() ? "true" : "false") << ",\"games\":[";
    bool first = true;
    for (auto &[id, record] : workset->records) {
      auto &entry = *record;
      if (workset->complete() && !workset->replay.empty() && entry.offset != entry.reference.size())
        throw std::runtime_error("reference behavior tape has unconsumed decisions");
      if (!first)
        index << ',';
      first = false;
      index << "{\"id\":" << id << ",\"steps\":" << entry.steps
            << ",\"terminal\":" << (entry.terminal ? "true" : "false") << '}';
      auto prefix = path / ("game-" + std::to_string(id));
      atomic_text(prefix.string() + ".bin", std::string(entry.tape.begin(), entry.tape.end()));
      auto game = games.find(id);
      if (game != games.end())
        atomic_text(prefix.string() + ".json",
                    std::string(game->second.snapshot.begin(), game->second.snapshot.end()));
    }
    index << "]}\n";
    atomic_text(path / "index.json", index.str());
  }
  std::string metrics(bool window) {
    accrue_activity();
    auto used = managed();
    std::vector<std::unique_lock<std::mutex>> progress_locks;
    for (auto &entry : games)
      progress_locks.emplace_back(*entry.second.progress_mutex);
    for (auto &entry : games)
      progress_locked(entry.second);
    auto delta = window ? initial : Counters{};
    uint64_t partial = 0, pending = 0, busy = 0, parked = 0, advancing = 0, inference_waiting = 0,
             inference_running = 0;
    for (auto &p : games) {
      partial += p.second.partial.size();
      pending += bool(p.second.pending);
      auto phase = p.second.flow->load();
      busy += phase == Flow::Cpu;
      inference_running += phase == Flow::RootRunning;
      parked += p.second.parked;
      if (!p.second.parked) {
        advancing += elapsed(p.second.last_advance) < 1;
        inference_waiting +=
            phase == Flow::RootQueued || (phase == Flow::Idle && bool(p.second.current));
      }
    }
    auto graph = roots->stats();
    auto encoder = roots->progress();
    peak_managed = std::max(peak_managed, used);
    double wall = elapsed(start);
    std::ostringstream o;
    o << std::setprecision(12) << "{\n  \"format\": \"haojie-native-v3\",\n  \"seconds\": " << wall
      << ",\n  \"window_start_unix_ms\": ";
    if (window)
      o << window_start_unix_ms;
    else
      o << "null";
    o << ",\n  \"active_game_seconds\": " << active_game_seconds
      << ",\n  \"mean_active\": " << active_game_seconds / std::max(wall, 1e-9)
      << ",\n  \"consumed\": " << counters.consumed - delta.consumed
      << ",\n  \"produced\": " << counters.produced - delta.produced
      << ",\n  \"commands\": " << counters.commands - delta.commands
      << ",\n  \"microsteps\": " << counters.microsteps - delta.microsteps
      << ",\n  \"updates\": " << committed_updates - (window ? initial_updates : 0)
      << ",\n  \"consumed_per_second\": ";
    if (window)
      o << (counters.consumed - initial.consumed) / std::max(wall, 1e-9);
    else
      o << "null";
    o << ",\n  \"completed_games\": " << counters.finished - delta.finished
      << ",\n  \"admitted_total\": " << next_id - 1 << ",\n  \"active\": " << active()
      << ",\n  \"parked\": " << parked << ",\n  \"busy\": " << busy
      << ",\n  \"advancing_1s\": " << advancing
      << ",\n  \"inference_waiting\": " << inference_waiting
      << ",\n  \"inference_running\": " << inference_running
      << ",\n  \"ready_steps\": " << ready_steps() << ",\n  \"partial_steps\": " << partial
      << ",\n  \"pending_steps\": " << pending << ",\n  \"learning_steps\": ";
    uint64_t learning_steps = 0;
    for (auto &b : in_learning)
      learning_steps += b.steps.size();
    if (counters.consumed + ready_steps() + partial + learning_steps != counters.produced)
      throw std::runtime_error("trajectory accounting differs");
    o << learning_steps << ",\n  \"inference_nodes\": " << infer_nodes
      << ",\n  \"encoder_nodes\": " << encoder.nodes
      << ",\n  \"encoder_batches\": " << encoder.batches
      << ",\n  \"policy_cpu_seconds\": " << inference_seconds
      << ",\n  \"encoder_seconds\": " << encoder.seconds
      << ",\n  \"graph_replays\": " << graph.replays
      << ",\n  \"graph_fallbacks\": " << graph.fallbacks
      << ",\n  \"input_actual_bytes\": " << graph.input_bytes
      << ",\n  \"input_capacity_bytes\": " << graph.capacity_bytes
      << ",\n  \"encoded_tokens_actual\": " << graph.tokens
      << ",\n  \"encoded_tokens_capacity\": " << graph.token_capacity
      << ",\n  \"peak_active\": " << peak_active
      << ",\n  \"encoded_roots\": " << actor->encoded_roots
      << ",\n  \"root_cache_hits\": " << actor->cache_hits
      << ",\n  \"learner_encoded_roots\": " << learner->encoded_roots
      << ",\n  \"learner_encoded_tokens\": " << learner->computed_tokens
      << ",\n  \"learner_scored_candidates\": " << learner->scored_candidates
      << ",\n  \"learner_projected_candidates\": " << learner->projected_candidates
      << ",\n  \"learning_graph_replays\": " << committed_graph_replays
      << ",\n  \"learning_graph_fallbacks\": " << committed_graph_fallbacks
      << ",\n  \"learning_graph_tokens_actual\": " << committed_graph_work[0]
      << ",\n  \"learning_graph_tokens_capacity\": " << committed_graph_work[1]
      << ",\n  \"learning_graph_keys_actual\": " << committed_graph_work[2]
      << ",\n  \"learning_graph_keys_capacity\": " << committed_graph_work[3]
      << ",\n  \"learning_seconds\": " << learning_seconds << ",\n  \"managed_bytes\": " << used
      << ",\n  \"peak_managed_bytes\": " << peak_managed
      << ",\n  \"backtracks\": " << counters.backtracks - delta.backtracks
      << ",\n  \"failures\": " << counters.failures - delta.failures
      << ",\n  \"actor_version\": " << actor_version << ",\n  \"parameters\": ";
    int64_t parameters = 0;
    for (auto &p : actor->parameters())
      parameters += p.numel();
    o << parameters
      << ",\n  \"architecture\": \"entity-transformer-native\",\n  \"layers\": " << EncoderLayers;
    o << ",\n  \"width\": " << settings.width << ",\n  \"batch\": " << settings.batch
      << ",\n  \"chunk\": " << settings.chunk << "\n}\n";
    return o.str();
  }
  void checkpoint(bool resume = true) {
    quiescing = true;
    sampling_paused = true;
    roots->wake();
    settle();
    collect_learning(true);
    if (writer.valid())
      writer.get();
    for (auto &p : games)
      if (!p.second.parked)
        submit(p.second, 2, id_packet(p.first));
    settle();
    Bytes state;
    put<uint64_t>(state, 0x484a4e4154495633ull);
    std::string identity = hj_identity();
    put_bytes(state, Bytes(identity.begin(), identity.end()));
    for (auto n : {settings.width, settings.seed, settings.chunk, settings.batch, next_id,
                   actor_version.load(), published, optimizer.steps})
      put(state, n);
    for (auto n : {counters.consumed, counters.produced, counters.commands, counters.microsteps,
                   counters.finished, counters.backtracks, counters.failures})
      put(state, n);
    GraphWriter graph;
    Bytes body;
    put<uint64_t>(body, games.size());
    for (auto &p : games) {
      auto &g = p.second;
      put(body, g.id);
      put(body, g.commands);
      put(body, g.microsteps);
      put(body, g.backtracks);
      put(body, g.failures);
      put(body, g.actor);
      put(body, g.learner);
      put<uint8_t>(body, g.parked);
      put_bytes(body, g.snapshot);
      graph.ref(body, g.current);
      put<uint8_t>(body, bool(g.pending));
      if (g.pending)
        write_step(body, *g.pending, graph);
      put<uint64_t>(body, g.partial.size());
      for (auto &s : g.partial)
        write_step(body, s, graph);
      if (!g.parked)
        Bytes{}.swap(g.snapshot);
    }
    put<uint64_t>(body, ready.size());
    for (auto &b : ready)
      write_block(body, b, graph);
    graph.write(state);
    raw(state, body.data(), body.size());
    auto archive = std::make_shared<torch::serialize::OutputArchive>();
    std::vector<int64_t> config = {int64_t(settings.width),
                                   int64_t(settings.seed),
                                   int64_t(settings.chunk),
                                   int64_t(settings.batch),
                                   int64_t(settings.concurrency),
                                   int64_t(settings.workers),
                                   int64_t(settings.memory_mb),
                                   int64_t(settings.publish),
                                   int64_t(settings.inference_batch),
                                   int64_t(settings.shrine),
                                   int64_t(device.is_cuda()),
                                   int64_t(device.index()),
                                   int64_t(std::llround(settings.save_seconds * 1000000))};
    archive->write("config", torch::tensor(config, torch::kInt64));
    auto state_owner = std::make_shared<Bytes>(std::move(state));
    archive->write("state", torch::from_blob(
                                state_owner->data(), {int64_t(state_owner->size())},
                                [state_owner](void *) {}, torch::kUInt8));
    {
      std::unique_lock lock(actor_weights);
      torch::serialize::OutputArchive opt;
      auto save_weights = [&](const char *name, Network &n) {
        archive->write(name, n->packed_weights.detach().to(
                                 n->packed_weights.options().device(torch::kCPU), false, true));
      };
      save_weights("weights", learner);
      save_weights("actor", actor);
      optimizer.save(opt);
      archive->write("optimizer", opt);
    }
    auto path = settings.directory / "checkpoint.pt";
    writer = std::async(std::launch::async, [archive, path] {
      auto tmp = path;
      tmp += ".tmp";
      std::ofstream file(tmp, std::ios::binary);
      archive->save_to(file);
      file.flush();
      if (!file)
        throw std::runtime_error("checkpoint write failed");
      file.close();
      replace(tmp, path);
    });
    last_save = Clock::now();
    quiescing = !resume;
    sampling_paused = !resume;
    roots->wake();
    for (auto &entry : games) {
      auto &g = entry.second;
      if (resume && !g.parked) {
        if (g.current)
          enqueue(g);
        else {
          auto b = id_packet(g.id);
          put<uint32_t>(b, 0);
          submit(g, 1, std::move(b));
        }
      }
    }
  }
  void restore(torch::serialize::InputArchive &archive) {
    T s;
    archive.read("state", s);
    TORCH_CHECK(s.scalar_type() == torch::kUInt8 && s.dim() == 1, "invalid checkpoint metadata");
    Reader r(s.data_ptr<uint8_t>(), s.numel());
    if (r.get<uint64_t>() != 0x484a4e4154495633ull)
      throw std::runtime_error("unsupported checkpoint");
    auto identity = get_bytes(r);
    if (std::string(identity.begin(), identity.end()) != hj_identity())
      throw std::runtime_error("checkpoint rules fingerprint differs");
    auto width = r.get<uint64_t>();
    if (width != settings.width)
      throw std::runtime_error("resume width differs");
    settings.seed = r.get<uint64_t>();
    settings.chunk = r.get<uint64_t>();
    settings.batch = r.get<uint64_t>();
    next_id = r.get<uint64_t>();
    actor_version = r.get<uint64_t>();
    published = r.get<uint64_t>();
    auto steps = r.get<uint64_t>();
    counters.consumed = r.get<uint64_t>();
    counters.produced = r.get<uint64_t>();
    counters.commands = r.get<uint64_t>();
    counters.microsteps = r.get<uint64_t>();
    counters.finished = r.get<uint64_t>();
    counters.backtracks = r.get<uint64_t>();
    counters.failures = r.get<uint64_t>();
    GraphReader graph(r, inputs);
    auto count = r.get<uint64_t>();
    for (uint64_t i = 0; i < count; ++i) {
      Game g;
      g.id = r.get<uint64_t>();
      g.commands = r.get<uint64_t>();
      g.microsteps = r.get<uint64_t>();
      g.backtracks = r.get<uint64_t>();
      g.failures = r.get<uint64_t>();
      g.actor = r.get<uint32_t>();
      g.learner = r.get<uint32_t>();
      g.parked = r.get<uint8_t>() != 0;
      g.snapshot = get_bytes(r);
      g.current = graph.ref(r);
      if (g.current)
        g.root = g.current->root;
      if (r.get<uint8_t>())
        g.pending = read_step(r, graph);
      auto n = r.get<uint64_t>();
      for (uint64_t j = 0; j < n; ++j)
        g.partial.push_back(read_step(r, graph));
      g.parked = true;
      games.emplace(g.id, std::move(g));
    }
    count = r.get<uint64_t>();
    for (uint64_t i = 0; i < count; ++i)
      ready.push_back(read_block(r, graph));
    r.end();
    torch::serialize::InputArchive opt;
    archive.read("optimizer", opt);
    auto load_weights = [&](Network &n, const char *name) {
      torch::NoGradGuard guard;
      T v;
      archive.read(name, v);
      TORCH_CHECK(v.scalar_type() == torch::kBFloat16 && v.sizes() == n->packed_weights.sizes() &&
                      torch::isfinite(v).all().item<bool>(),
                  "invalid BF16 weights");
      n->packed_weights.copy_(v.to(device));
    };
    load_weights(learner, "weights");
    load_weights(actor, "actor");
    optimizer.load(opt, learner);
    if (optimizer.steps != steps)
      throw std::runtime_error("optimizer metadata differs");
  }
  void execute() {
    bool stopping = false;
    bool profiling = false, profiled = false;
    auto last_control = start - std::chrono::seconds(1);
    while (!stopping) {
      auto observed_revision = mailbox.revision.load();
      // 诊断只截取热身后的两秒，使用与当前 LibTorch 一起发布的 CUPTI。
      if (!settings.profile.empty() && !profiled && elapsed(start) >= 8) {
        namespace pi = torch::profiler::impl;
        pi::ExperimentalConfig experimental;
        experimental.profile_all_threads = true;
        pi::ProfilerConfig config(pi::ProfilerState::KINETO, true, true, false, false, false,
                                  experimental);
        std::set<pi::ActivityType> activities{pi::ActivityType::CPU};
        if (device.is_cuda())
          activities.insert(pi::ActivityType::CUDA);
        torch::autograd::profiler::prepareProfiler(config, activities);
        torch::autograd::profiler::enableProfiler(config, activities);
        profiling = profiled = true;
      }
      if (profiling && elapsed(start) >= 10) {
        auto result = torch::autograd::profiler::disableProfiler();
        result->save(settings.profile.string());
        profiling = false;
      }
      collect();
      collect_learning();
      // 控制文件只需毫秒级响应；规则和设备事件直接唤醒，不重复访问文件系统。
      bool control_tick = elapsed(last_control) >= 0.02;
      if (control_tick) {
        last_control = Clock::now();
        auto stop = settings.directory / "stop";
        if (fs::exists(stop)) {
          fs::remove(stop);
          stopping = true;
        }
        auto resize_path = settings.directory / "resize";
        if (fs::exists(resize_path)) {
          auto n = std::stoull(read_text(resize_path));
          settings.concurrency = n;
          fs::remove(resize_path);
        }
        auto budget_path = settings.directory / "budget";
        if (fs::exists(budget_path)) {
          auto n = std::stoull(read_text(budget_path));
          if (n < 128)
            throw std::runtime_error("memory budget too small");
          settings.memory_mb = n;
          fs::remove(budget_path);
        }
      }
      if (settings.seconds > 0 && elapsed(start) >= settings.seconds)
        stopping = true;
      if (settings.updates > 0 && committed_updates - initial_updates >= settings.updates)
        stopping = true;
      if (settings.samples > 0 && counters.consumed - initial.consumed >= settings.samples)
        stopping = true;
      if (workset && workset->complete() && !learning.valid() && ready.empty() &&
          counters.consumed == counters.produced &&
          std::none_of(games.begin(), games.end(), [](auto &p) { return !p.second.idle(); }))
        stopping = true;
      if (stopping)
        break;
      resize();
      train();
      sampling_budget();
      auto save = settings.directory / "save";
      if ((control_tick && fs::exists(save)) ||
          (settings.save_seconds > 0 && elapsed(last_save) >= settings.save_seconds)) {
        if (fs::exists(save))
          fs::remove(save);
        checkpoint();
      }
      if (elapsed(last_status) >= 1) {
        inputs.trim();
        atomic_text(settings.directory / "status.json", metrics(true));
        last_status = Clock::now();
      }
      if (managed() > settings.memory_mb * 1024 * 1024) {
        checkpoint(false);
        if (writer.valid())
          writer.get();
        throw std::runtime_error(
            "managed memory budget exceeded; saved checkpoint, increase --memory-mb");
      }
      std::unique_lock<std::mutex> lock(mailbox.mutex);
      auto completed = [](auto &future) {
        return future.valid() &&
               future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
      };
      if (learning.valid() || ready_steps() < settings.batch)
        mailbox.changed.wait_for(lock, std::chrono::milliseconds(20), [&] {
          return !mailbox.events.empty() || mailbox.revision.load() != observed_revision ||
                 learning_commit.has_value() || completed(learning);
        });
    }
    if (profiling) {
      auto result = torch::autograd::profiler::disableProfiler();
      result->save(settings.profile.string());
    }
    // 测量窗口在退出排空之前冻结，恢复累计计数不会进入本轮速率。
    collect_learning();
    auto window = metrics(true);
    atomic_text(settings.directory / "window.json", window);
    checkpoint(false);
    save_workset();
    if (writer.valid())
      writer.get();
    atomic_text(settings.directory / "saved.json", metrics(false));
    std::cout << window;
  }
};
} // namespace
void atomic_text(const fs::path &path, const std::string &text) {
  auto tmp = path;
  tmp += ".tmp";
  std::ofstream out(tmp, std::ios::binary);
  out << text;
  out.flush();
  if (!out)
    throw std::runtime_error("control write failed");
  out.close();
  replace(tmp, path);
}
Frame parse(Reader r, std::shared_ptr<Root> &cached, InputPool &pool) {
  Frame f;
  f.game = r.get<uint64_t>();
  f.root = r.get<uint64_t>();
  f.opponent = r.get<uint64_t>();
  f.commands = r.get<uint64_t>();
  f.microsteps = r.get<uint64_t>();
  f.failures = r.get<uint64_t>();
  f.backtracks = r.get<uint64_t>();
  f.actor = r.get<uint32_t>();
  f.learner = r.get<uint32_t>();
  f.kind = r.get<uint32_t>();
  f.winner = r.get<int32_t>();
  f.selected = r.get<int32_t>();
  f.logp = r.get<double>();
  if (f.kind == 0) {
    bool has_root = r.get<uint32_t>() != 0;
    auto entities = r.get<uint32_t>(), prefix = r.get<uint32_t>(), candidates = r.get<uint32_t>(),
         cursor = r.get<uint32_t>(), facts = r.get<uint32_t>();
    auto root = std::make_shared<Root>();
    root->game = f.game;
    root->id = f.root;
    if (has_root) {
      root->rows.reserve(entities);
      for (uint32_t i = 0; i < entities; ++i)
        root->rows.push_back(pool.rows.get(r.get<Row>()));
    }
    auto n = std::make_shared<Node>();
    auto prefix_rows = r.array<Row>(prefix);
    root->kinds.reserve(entities);
    if (has_root) {
      for (uint32_t i = 0; i < entities; ++i)
        root->kinds.push_back(r.get<uint32_t>());
      root->globals = pool.rows.get(r.get<Row>());
    }
    if (!has_root && (!cached || cached->id != f.root))
      throw std::runtime_error("missing native root table");
    if (cached && cached->id == f.root)
      root = cached;
    else {
      if (cached)
        cached->clear_cache();
      cached = root;
    }
    n->root = root;
    n->candidates.reserve(candidates);
    for (uint32_t i = 0; i < candidates; ++i)
      n->candidates.push_back(pool.actions.get(r.get<Action>()));
    n->sources = r.array<int32_t>(candidates);
    n->targets = r.array<int32_t>(candidates);
    n->mask = r.array<uint8_t>(candidates);
    auto path = r.array<uint32_t>(cursor);
    std::vector<Frontier> excluded;
    for (uint32_t i = 0; i < facts; ++i) {
      auto path = r.array<uint32_t>(r.get<uint32_t>());
      auto children = r.array<uint32_t>(r.get<uint32_t>());
      excluded.push_back({std::move(path), std::move(children)});
    }
    n->context = make_context(pool, prefix_rows, path, excluded);
    root->account_input();
    n->account_input();
    f.node = n;
  }
  r.end();
  return f;
}
void run(const Settings &settings) {
  if (fs::exists(settings.directory) && !fs::is_empty(settings.directory))
    throw std::runtime_error("run directory must be new or empty");
  fs::create_directories(settings.directory);
  auto full_started = Clock::now();
  try {
    Session session(settings);
    double init = elapsed(full_started);
    session.execute();
    std::ostringstream wall;
    wall << "{\"full_pipeline_seconds\":" << elapsed(full_started)
         << ",\"initialization_seconds\":" << init << "}\n";
    atomic_text(settings.directory / "wall.json", wall.str());
  } catch (const std::exception &e) {
    atomic_text(settings.directory / "failure.txt", e.what());
    throw;
  }
}
void runtime_checks() {
  static_assert(sizeof(Row) == 600);
  auto baseline = StorageBytes::total.load();
  {
    InputPool pool;
    auto root = std::make_shared<Root>();
    auto shared_row = pool.rows.get(Row{});
    root->rows.assign(3, shared_row);
    root->kinds.assign(3, 1);
    root->account_input();
    auto node = std::make_shared<Node>();
    node->root = root;
    node->candidates.assign(5, pool.actions.get(Action{}));
    node->mask.assign(5, 1);
    node->sources.assign(5, -1);
    node->targets.assign(5, -1);
    node->context = make_context(pool, {}, {}, {});
    node->account_input();
    GraphWriter graph;
    Bytes refs, encoded;
    graph.ref(refs, node);
    graph.ref(refs, node);
    graph.write(encoded);
    raw(encoded, refs.data(), refs.size());
    Reader reader(encoded);
    GraphReader restored(reader, pool);
    auto a = restored.ref(reader), b = restored.ref(reader);
    reader.end();
    TORCH_CHECK(a == b && a->root->rows[0] == shared_row && a->candidates[0] == node->candidates[0],
                "checkpoint lost shared identities");
    auto charged = StorageBytes::total.load();
    auto same = node;
    node.reset();
    TORCH_CHECK(charged == StorageBytes::total.load(), "shared trajectory double charged");
    root.reset();
    same.reset();
  }
  TORCH_CHECK(baseline == StorageBytes::total.load(), "released trajectory retained budget");
  void *handle = hj_actor_create();
  if (!handle)
    throw std::runtime_error("actor initialization failed");
  auto call = [&](uint32_t op, const Bytes &b) {
    const uint8_t *data = nullptr;
    size_t size = 0;
    auto status = hj_actor_call(handle, op, b.data(), b.size(), &data, &size);
    Bytes result(data, data + size);
    if (status)
      throw std::runtime_error(std::string(result.begin(), result.end()));
    return result;
  };
  Bytes admission;
  put<uint64_t>(admission, 1);
  put<uint32_t>(admission, 73127);
  put<uint32_t>(admission, 1);
  put<uint32_t>(admission, 0);
  call(0, admission);
  Bytes request;
  put<uint64_t>(request, 1);
  put<uint32_t>(request, 0);
  std::shared_ptr<Root> root;
  InputPool inputs;
  auto encode = [](const std::shared_ptr<Node> &n) {
    GraphWriter graph;
    Bytes body, out;
    graph.ref(body, n);
    graph.write(out);
    raw(out, body.data(), body.size());
    return out;
  };
  auto first = call(1, request);
  auto f = parse(first, root, inputs);
  TORCH_CHECK(f.kind == 0 && f.node && f.node->mask.size() > 1);
  auto encoded = encode(f.node);
  Reader node_reader(encoded);
  GraphReader graph(node_reader, inputs);
  auto decoded = graph.ref(node_reader);
  node_reader.end();
  TORCH_CHECK(encode(decoded) == encoded);
  Bytes id;
  put<uint64_t>(id, 1);
  auto saved = call(3, id);
  call(4, saved);
  auto after = call(1, request);
  TORCH_CHECK(first == after, "park/restore changed request");
  Bytes reply;
  put<uint64_t>(reply, 1);
  put<uint32_t>(reply, f.node->mask.size());
  std::vector<uint16_t> mass;
  for (auto bit : f.node->mask) {
    c10::BFloat16 v{float(bit)};
    put(reply, v.x);
    mass.push_back(v.x);
  }
  auto next = call(1, reply);
  auto next_root = root;
  auto selected = parse(next, next_root, inputs);
  auto logps = canonical_logps(mass);
  TORCH_CHECK(selected.selected >= 0 && std::abs(selected.logp - logps[selected.selected]) < 1e-12);
  call(3, id);
  call(4, saved);
  auto replay = call(1, reply);
  auto replay_root = root;
  auto continued = parse(replay, replay_root, inputs);
  TORCH_CHECK(continued.commands == selected.commands &&
                  continued.microsteps == selected.microsteps &&
                  continued.selected == selected.selected && continued.logp == selected.logp,
              "actor restore changed RNG or command progress");
  TORCH_CHECK(bool(continued.node) == bool(selected.node));
  if (continued.node)
    TORCH_CHECK(encode(continued.node) == encode(selected.node));
  hj_actor_close(handle);
  std::cout << "typed node roundtrip, actor parking, RNG/progress continuation and canonical "
               "behavior probability: passed\n";
}
} // namespace hj
