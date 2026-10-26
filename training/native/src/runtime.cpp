#include "runtime.h"
#include "actor/flow.h"
#include "actor/history.h"
#include "engine.h"
#include "host/hardware.h"
#include "storage/checkpoint.h"
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
#include <sstream>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
#ifdef HJ_CUDA
#include <c10/core/AllocatorConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
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

constexpr uint64_t LagLimit = 64;
enum class Discard { None, Long, Stale, Memory };
Discard unfinished_limit(const Game &game, uint64_t version, uint64_t limit) {
  if (game.partial.size() >= limit)
    return Discard::Long;
  const Step *oldest = !game.partial.empty() ? &game.partial.front()
                       : game.pending        ? &*game.pending
                                             : nullptr;
  if (oldest) {
    if (version >= oldest->version && version - oldest->version > LagLimit)
      return Discard::Stale;
  }
  return Discard::None;
}
bool terminal_queue_full(size_t queued, uint64_t concurrency) {
  return queued >= std::max<uint64_t>(1, concurrency);
}
std::vector<Block> terminal_batch(std::deque<Block> &ready, uint64_t version, uint64_t batch,
                                  uint64_t max_game_steps, Counters &counters) {
  std::vector<Block> result;
  uint64_t count = 0;
  while (!ready.empty() && count < batch) {
    auto &episode = ready.front();
    TORCH_CHECK(!episode.bootstrap && !episode.steps.empty() && episode.steps.back().terminal,
                "terminal queue contains an unfinished game");
    bool stale = false;
    for (auto &step : episode.steps) {
      TORCH_CHECK(step.version <= version, "behavior version exceeds learner version");
      stale |= version - step.version > LagLimit;
    }
    if (episode.steps.size() > max_game_steps) {
      counters.discarded_steps += episode.steps.size();
      ++counters.discarded_games;
      ++counters.discarded_long_games;
    } else if (stale) {
      counters.stale_steps += episode.steps.size();
      ++counters.stale_games;
    } else {
      count += episode.steps.size();
      result.push_back(std::move(episode));
    }
    ready.pop_front();
  }
  return result;
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
  std::function<SearchResult(void *, Job &, const std::function<bool()> &)> search;
  std::thread thread;
  static void successor(Game &g, std::optional<int8_t> reward = {}) {
    if (g.pending) {
      auto step = std::move(*g.pending);
      g.pending.reset();
      if (reward) {
        step.terminal = true;
        step.reward = *reward;
      }
      g.partial.push_back(std::move(step));
      ++g.progress.produced;
    }
    if (!g.partial.empty() && reward) {
      g.completed.push_back({std::move(g.partial), {}});
      g.partial.clear();
    }
  }

public:
  Lane(Mailbox &out, std::atomic<bool> &paused,
       std::function<SearchResult(void *, Job &, const std::function<bool()> &)> search)
      : output(out), paused(paused), search(std::move(search)), thread([this] {
          void *actor = hj_actor_create();
          InputPool inputs;
          auto last_trim = Clock::now();
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
                  std::vector<uint16_t> mass;
                  uint64_t version = 0;
                  if (this->search) {
                    auto cancelled = [&] { return this->paused || *job.state->pause || closing; };
                    try {
                      auto result = this->search(actor, job, cancelled);
                      mass = std::move(result.mass);
                      version = result.version;
                      if (job.state->actor == job.state->learner) {
                        job.node->search_target = std::move(result.target);
                        job.node->account_input();
                      }
                    } catch (const std::exception &) {
                      if (!cancelled())
                        throw;
                    }
                    if (cancelled()) {
                      e.kind = 0;
                      output.bytes -= job.bytes.capacity();
                      output.push(std::move(e));
                      continue;
                    }
                  }
                  {
                    std::lock_guard lock(*job.state->progress_mutex);
                    job.state->sent = job.node;
                    job.state->sent_training = job.state->actor == job.state->learner;
                    job.state->sent_version = version;
                  }
                  {
                    std::lock_guard lock(*job.state->progress_mutex);
                    ++job.state->scored;
                  }
                  put(reply, job.game);
                  put<uint32_t>(reply, mass.size());
                  raw(reply, mass.data(), mass.size() * 2);
                  input = &reply;
                }
                const uint8_t *p = nullptr;
                size_t n = 0;
                RECORD_FUNCTION("haojie::rust_actor", std::vector<c10::IValue>());
                int status = hj_actor_call(actor, job.op, input->data(), input->size(), &p, &n);
                if (status) {
                  e.error = std::string(reinterpret_cast<const char *>(p), n);
                  // 仅隔离已知的单局空根；协议、设备和其他规则错误仍上报宿主。
                  if (e.error == "no legal action at root") {
                    Bytes id;
                    put(id, job.game);
                    if (!hj_actor_call(actor, 3, id.data(), id.size(), &p, &n)) {
                      e.bytes.assign(p, p + n);
                      e.kind = 4;
                      inputs.trim();
                    }
                  }
                } else {
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
                        successor(g, int8_t(reward));
                        ++g.progress.finished;
                      } else {
                        if (f.kind == 0 && g.actor == g.learner)
                          successor(g);
                        if (!this->paused && !*g.pause && !closing) {
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
  TORCH_CHECK(c[2] == 0,
              "this checkpoint contains short-trajectory training; start a new terminal run");
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
    s.device = "auto";
  if (!s.supplied.count("--save-seconds"))
    s.save_seconds = double(c[12]) / 1000000;
  T limits;
  if (archive->try_read("trajectory_limits", limits)) {
    TORCH_CHECK(limits.scalar_type() == torch::kInt64 && limits.dim() == 1 && limits.numel() == 1 &&
                    limits[0].item<int64_t>() > 0,
                "invalid trajectory limit");
    if (!s.supplied.count("--max-game-steps"))
      s.max_game_steps = limits[0].item<int64_t>();
  }
  T selfplay;
  if (archive->try_read("selfplay_config", selfplay)) {
    TORCH_CHECK(selfplay.scalar_type() == torch::kInt64 && selfplay.dim() == 1 &&
                    selfplay.numel() == 5,
                "invalid self-play configuration");
    std::array<uint64_t *, 5> fields{&s.search_simulations, &s.search_depth, &s.history_size,
                                     &s.history_interval, &s.history_percent};
    std::array<const char *, 5> names{"--search-simulations", "--search-depth", "--history-size",
                                      "--history-interval", "--history-percent"};
    for (size_t i = 0; i < fields.size(); ++i) {
      auto value = selfplay[i].item<int64_t>();
      TORCH_CHECK(value >= (i == 0 || i == 4 ? 0 : 1) && (i != 4 || value <= 100),
                  "invalid self-play budget");
      if (!s.supplied.count(names[i]))
        *fields[i] = value;
    }
  }
  return {std::move(s), std::move(archive)};
}
class Session {
  friend void hj::runtime_checks();
  Settings settings;
  torch::Device device;
  Network learner, actor;
  Optimizer optimizer;
  Mailbox mailbox;
  InputPool inputs;
  std::atomic<bool> sampling_paused{false};
  bool ready_paused = false;
  bool memory_paused = false;
  uint64_t consumed_within_lag = 0, consumed_lag_max = 0, peak_ready = 0;
  bool quiescing = false;
  std::vector<std::unique_ptr<Lane>> lanes;
  std::map<uint64_t, Game> games;
  std::deque<Block> ready;
  Counters counters, initial;
  uint64_t next_id = 1, published = 0, initial_updates = 0, committed_updates = 0;
  std::atomic<uint64_t> actor_version{0};
  std::future<TrainResult> learning;
  struct Commit {
    TrainResult result;
    uint64_t updates;
    double seconds;
    uint64_t within_lag, lag_max;
  };
  std::optional<Commit> learning_commit;
  std::vector<Block> in_learning;
  std::future<void> writer;
  Clock::time_point start, last_status, last_save, activity_at;
  double active_game_seconds = 0;
  int64_t window_start_unix_ms = 0;
  uint64_t infer_nodes = 0, peak_active = 0, peak_managed = 0;
  double learning_seconds = 0;
  std::optional<double> latest_loss;
  std::optional<TrainResult> latest_terminal;
  uint64_t terminal_games_consumed = 0;
  Clock::time_point learn_start;
  std::unique_ptr<Executor> learning_executor;
  OpponentHistory history;
  std::unique_ptr<SearchService> search_service;
  std::atomic<uint64_t> search_simulations{0}, search_expanded{0}, search_opponent_nodes{0},
      search_max_depth{0}, search_parameter_fallbacks{0}, search_empty_fallbacks{0};
  uint64_t historical_games = 0;

public:
  explicit Session(std::pair<Settings, std::shared_ptr<torch::serialize::InputArchive>> resumed)
      : settings(std::move(resumed.first)), device(settings.device), learner(settings.width),
        actor(settings.width), optimizer(learner) {
    learner->to(device);
#ifdef HJ_CUDA
    if (device.is_cuda()) {
      // 先建立明确设备预算，分配器才会按阈值回收空闲缓存；留出驱动和外部驻留余量。
      c10::CachingAllocator::setAllocatorSettings("garbage_collection_threshold:0.7");
      c10::cuda::CUDACachingAllocator::setMemoryFraction(0.9,
                                                         device.index() < 0 ? 0 : device.index());
    }
#endif
    actor->to(device);
    // 优化器在设备迁移后重建，浮点参数及状态始终为 BF16。
    optimizer = Optimizer(learner);
    copy_weights(learner, actor);
    for (uint64_t i = 0; i < settings.workers; ++i)
      lanes.push_back(std::make_unique<Lane>(
          mailbox, sampling_paused,
          [this](void *handle, Job &job, const std::function<bool()> &cancelled) {
            auto &game = *job.state;
            auto policy =
                game.actor != game.learner && game.opponent ? game.opponent : history.latest.load();
            auto result = search_actor(handle, game.id, game.actor, job.node, policy,
                                       *search_service, settings.search_simulations,
                                       settings.search_depth, cancelled, &game.search_plan);
            search_simulations += result.simulations;
            search_expanded += result.expanded;
            search_opponent_nodes += result.opponent_nodes;
            search_parameter_fallbacks += result.parameter_limit;
            search_empty_fallbacks += result.empty_branch;
            auto old_depth = search_max_depth.load();
            while (old_depth < result.depth &&
                   !search_max_depth.compare_exchange_weak(old_depth, result.depth)) {
            }
            return result;
          }));
    if (resumed.second)
      restore(*resumed.second);
    for (auto &block : ready)
      TORCH_CHECK(!block.bootstrap && !block.steps.empty() && block.steps.back().terminal,
                  "checkpoint contains unfinished ready trajectories");
    synchronize(learner->parameters()[0].device());
    history.limit = settings.history_size;
    history.interval = settings.history_interval;
    history.percent = settings.history_percent;
    if (resumed.second) {
      auto bindings = history.load(*resumed.second, actor, actor_version.load());
      for (auto &[id, policy] : bindings) {
        TORCH_CHECK(games.count(id), "historical opponent references missing game");
        games.at(id).opponent = policy;
      }
    }
    history.publish(std::make_shared<FrozenPolicy>(actor, actor_version.load()));
    search_service = std::make_unique<SearchService>(
        device, settings.width, std::min(settings.workers, settings.inference_batch));
    learning_executor = std::make_unique<Executor>(mailbox, device, false);
    initial = counters;
    initial_updates = committed_updates = optimizer.steps;
    start = last_status = last_save = activity_at = Clock::now();
    window_start_unix_ms = utc_ms();
  }
  ~Session() {
    sampling_paused = true;
    std::cerr << "Shutdown: joining learner and actor threads.\n";
    learning_executor.reset();
    for (auto &lane : lanes)
      lane.reset();
    std::cerr << "Shutdown: releasing inference service.\n";
    search_service.reset();
    std::cerr << "Shutdown: session workers released.\n";
  }
  void submit(Game &g, uint32_t op, Bytes b) {
    if (!g.idle())
      throw std::runtime_error("actor already has work");
    g.flow->store(Flow::Cpu);
    Job job{g.id, op, std::move(b)};
    job.state = &g;
    lanes[g.id % lanes.size()]->submit(std::move(job));
  }
  void enqueue(Game &g) {
    if (g.idle() && !g.parked && !g.parking && !sampling_paused && !*g.pause && g.current) {
      g.flow->store(Flow::Cpu);
      Job job{g.id, 1, {}};
      job.state = &g;
      job.node = g.current;
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
    while (!g.completed.empty()) {
      ready.push_back(std::move(g.completed.front()));
      g.completed.pop_front();
    }
  }
  void event(Event e) {
    accrue_activity();
    auto found = games.find(e.game);
    if (found == games.end())
      throw std::runtime_error("unknown lane event");
    auto &g = found->second;
    if (e.kind == 4 && e.error == "no legal action at root") {
      auto progress_mutex = g.progress_mutex;
      std::lock_guard lock(*progress_mutex);
      progress_locked(g);
      auto directory = settings.directory / "failed-games";
      fs::create_directories(directory);
      auto path = directory / ("game-" + std::to_string(g.id) + ".json");
      atomic_text(path, std::string(e.bytes.begin(), e.bytes.end()));
      counters.discarded_steps += g.partial.size();
      ++counters.discarded_games;
      ++counters.discarded_error_games;
      std::cerr << "discarded game " << g.id << ": " << e.error << "; saved " << path << '\n';
      // 删除整局的未完成轨迹；不补奖励，也不记为真实终局。
      g.flow->store(Flow::Idle);
      games.erase(found);
      return;
    }
    if (!e.error.empty())
      throw std::runtime_error(e.error);
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
        // 原始下标保持兼容；恢复时由 Rust 刷新当前未采样节点的空前缀掩码。
        // 已记录的行为节点及其历史掩码不改写。
        g.current.reset();
        g.search_plan.clear();
        g.root.reset();
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
        if (g.parking) {
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
        g.current.reset();
        if (g.root)
          g.root->clear_cache();
        g.search_plan.clear();
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
      latest_loss = commit->result.loss;
      counters.consumed += commit->result.consumed;
      consumed_within_lag += commit->within_lag;
      consumed_lag_max = std::max(consumed_lag_max, commit->lag_max);
      if (commit->result.terminal_games) {
        latest_terminal = commit->result;
        terminal_games_consumed += commit->result.terminal_games;
      }
      learning_seconds += commit->seconds;
      in_learning.clear();
    }
  }
  void publish_weights() {
    // 完成整局梯度更新后拷贝并冻结新快照，等待设备拷贝完成再交给策略服务。
    copy_weights(learner, actor);
    synchronize(device);
    history.publish(std::make_shared<FrozenPolicy>(actor, optimizer.steps));
    actor_version = published = optimizer.steps;
  }
  void train() {
    collect_learning();
    if (learning.valid())
      return;
    // 完成局独立取走；长局、停放局与更新次数不参与就绪条件。
    // 批次达到计算小批大小即提交已有完整局，不把整个队列合成一次大更新。
    in_learning =
        terminal_batch(ready, committed_updates, settings.batch, settings.max_game_steps, counters);
    if (in_learning.empty())
      return;
    learn_start = Clock::now();
    learning = learning_executor->submit([this] {
      uint64_t count = 0, maximum = 0;
      for (auto &episode : in_learning)
        for (auto &step : episode.steps) {
          TORCH_CHECK(step.version <= optimizer.steps && optimizer.steps - step.version <= LagLimit,
                      "strict terminal policy lag exceeded");
          maximum = std::max(maximum, optimizer.steps - step.version);
          ++count;
        }
      auto result = learn_episodes(learner, optimizer, in_learning, settings.batch);
      TORCH_CHECK(result.consumed == count, "terminal learning receipt count differs");
      {
        std::lock_guard lock(mailbox.mutex);
        learning_commit = Commit{result, optimizer.steps, elapsed(learn_start), count, maximum};
      }
      mailbox.changed.notify_one();
      publish_weights();
      return result;
    });
  }
  void resume_sampling() {
    sampling_paused = quiescing || memory_paused;
    if (sampling_paused)
      return;
    for (auto &entry : games) {
      auto &g = entry.second;
      if (!g.parked && !g.parking && g.idle()) {
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
  void discard_game(uint64_t id, Discard reason) {
    auto found = games.find(id);
    TORCH_CHECK(found != games.end() && found->second.idle() && sampling_paused,
                "discard requires a settled actor boundary");
    auto &g = found->second;
    TORCH_CHECK(g.completed.empty(), "discard must collect real terminals first");
    counters.discarded_steps += g.partial.size();
    ++counters.discarded_games;
    counters.discarded_long_games += reason == Discard::Long;
    counters.discarded_stale_games += reason == Discard::Stale;
    counters.discarded_memory_games += reason == Discard::Memory;
    // 停放局已从 Rust lane 移出；活动局等待删除回执后再释放跨线程引用。
    if (g.parked)
      games.erase(found);
    else {
      submit(g, 5, id_packet(id));
      settle();
    }
    inputs.trim();
  }
  void limit_trajectories() {
    auto budget = settings.memory_mb * 1024 * 1024;
    auto used = managed();
    bool pressure = used >= budget * 3 / 4;
    bool needed = pressure;
    for (auto &entry : games) {
      std::lock_guard lock(*entry.second.progress_mutex);
      needed |= unfinished_limit(entry.second, committed_updates, settings.max_game_steps) !=
                Discard::None;
    }
    if (!needed) {
      if (memory_paused && used < budget / 2) {
        memory_paused = false;
        resume_sampling();
      }
      return;
    }
    // 先收回全部 lane 和设备槽的所有权，不能释放仍被在途推理引用的轨迹。
    sampling_paused = true;
    settle();
    collect_learning();
    std::vector<std::pair<uint64_t, Discard>> rejected;
    for (auto &entry : games) {
      auto reason = unfinished_limit(entry.second, committed_updates, settings.max_game_steps);
      if (reason != Discard::None)
        rejected.emplace_back(entry.first, reason);
    }
    for (auto [id, reason] : rejected)
      discard_game(id, reason);
    inputs.trim();
    pressure |= managed() >= budget * 3 / 4;
    if (pressure) {
      // 回到半预算后再准入；优先牺牲最长的未结束局，完整终局继续由 learner 消费。
      while (managed() >= budget / 2 && !games.empty()) {
        auto largest = std::max_element(games.begin(), games.end(), [](auto &a, auto &b) {
          auto retained = [](const Game &g) {
            return g.partial.capacity() * sizeof(Step) + g.snapshot.capacity();
          };
          return retained(a.second) < retained(b.second);
        });
        discard_game(largest->first, Discard::Memory);
      }
      // 就绪或在学习的完整对局暂时独占预算时，只暂停采样，不丢弃完整终局。
      memory_paused = managed() >= budget / 2;
    } else if (managed() < budget / 2)
      memory_paused = false;
    resume_sampling();
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
    if (sampling_paused)
      return;
    for (auto &p : games) {
      auto &g = p.second;
      if (running >= settings.concurrency || active() >= settings.concurrency)
        break;
      if (g.parked && g.idle()) {
        g.parked = false;
        *g.pause = false;
        submit(g, 4, g.snapshot);
        ++running;
      }
    }
    if (terminal_queue_full(ready.size(), settings.concurrency))
      return;
    while (running < settings.concurrency && active() < settings.concurrency &&
           managed() < settings.memory_mb * 1024 * 1024 / 2) {
      if (std::any_of(games.begin(), games.end(),
                      [&](auto &entry) { return entry.second.parking; }))
        break;
      Game game;
      game.id = next_id++;
      game.learner = uint32_t(1 + (game.id & 1));
      game.opponent = history.choose(game.id, uint32_t(settings.seed));
      historical_games += bool(game.opponent);
      auto id = game.id;
      games.emplace(id, std::move(game));
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
    peak_ready = std::max(peak_ready, ready_steps());
    ready_paused = terminal_queue_full(ready.size(), settings.concurrency);
    if (managed() > settings.memory_mb * 1024 * 1024 * 3 / 4)
      inputs.trim();
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
    uint64_t partial = 0, pending = 0, busy = 0, parked = 0, advancing = 0;
    for (auto &p : games) {
      partial += p.second.partial.size();
      pending += bool(p.second.pending);
      auto phase = p.second.flow->load();
      busy += phase == Flow::Cpu;
      parked += p.second.parked;
      if (!p.second.parked) {
        advancing += elapsed(p.second.last_advance) < 1;
      }
    }
    auto inference_waiting = search_service->queued_roots();
    auto inference_running = search_service->running_roots();
    auto graph = search_service->stats();
    auto encoder = search_service->progress();
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
      << ",\n  \"target_concurrency\": " << settings.concurrency << ",\n  \"parked\": " << parked
      << ",\n  \"busy\": " << busy << ",\n  \"advancing_1s\": " << advancing
      << ",\n  \"inference_waiting\": " << inference_waiting
      << ",\n  \"inference_running\": " << inference_running
      << ",\n  \"ready_steps\": " << ready_steps() << ",\n  \"partial_steps\": " << partial
      << ",\n  \"pending_steps\": " << pending << ",\n  \"learning_steps\": ";
    uint64_t learning_steps = 0;
    for (auto &b : in_learning)
      learning_steps += b.steps.size();
    if (counters.consumed + counters.stale_steps + counters.discarded_steps + ready_steps() +
            partial + learning_steps !=
        counters.produced)
      throw std::runtime_error("trajectory accounting differs");
    o << learning_steps;
    o << ",\n  \"lag_limit\": " << LagLimit << ",\n  \"strict_lag\": " << "true"
      << ",\n  \"stale_steps\": " << counters.stale_steps - delta.stale_steps
      << ",\n  \"stale_games\": " << counters.stale_games - delta.stale_games
      << ",\n  \"discarded_steps\": " << counters.discarded_steps - delta.discarded_steps
      << ",\n  \"discarded_games\": " << counters.discarded_games - delta.discarded_games
      << ",\n  \"discarded_long_games\": "
      << counters.discarded_long_games - delta.discarded_long_games
      << ",\n  \"discarded_stale_games\": "
      << counters.discarded_stale_games - delta.discarded_stale_games
      << ",\n  \"discarded_memory_games\": "
      << counters.discarded_memory_games - delta.discarded_memory_games
      << ",\n  \"discarded_error_games\": "
      << counters.discarded_error_games - delta.discarded_error_games
      << ",\n  \"outstanding_steps\": "
      << counters.produced - counters.consumed - counters.stale_steps - counters.discarded_steps +
             pending
      << ",\n  \"ready_games\": " << ready.size()
      << ",\n  \"ready_game_limit\": " << std::max<uint64_t>(1, settings.concurrency)
      << ",\n  \"consumed_within_lag\": " << consumed_within_lag
      << ",\n  \"consumed_lag_max\": " << consumed_lag_max
      << ",\n  \"peak_ready_steps\": " << peak_ready
      << ",\n  \"ready_backpressure\": " << (ready_paused ? "true" : "false")
      << ",\n  \"memory_backpressure\": " << (memory_paused ? "true" : "false")
      << ",\n  \"max_game_steps\": " << settings.max_game_steps;
    o << ",\n  \"search_simulations\": " << search_simulations.load()
      << ",\n  \"search_expanded\": " << search_expanded.load()
      << ",\n  \"search_max_depth\": " << search_max_depth.load()
      << ",\n  \"search_opponent_nodes\": " << search_opponent_nodes.load()
      << ",\n  \"search_budget\": " << settings.search_simulations
      << ",\n  \"search_parameter_fallbacks\": " << search_parameter_fallbacks.load()
      << ",\n  \"search_empty_fallbacks\": " << search_empty_fallbacks.load()
      << ",\n  \"search_model_queries\": " << (search_service ? search_service->queries.load() : 0)
      << ",\n  \"history_pool_size\": " << history.snapshots().size()
      << ",\n  \"historical_games_admitted\": " << historical_games;
    o << ",\n  \"latest_loss\": ";
    if (latest_loss)
      o << *latest_loss;
    else
      o << "null";
    o << ",\n  \"loss_update\": " << committed_updates;
    o << ",\n  \"training_method\": \""
      << (settings.search_simulations ? "terminal-vtrace-mcts" : "terminal-vtrace") << "\""
      << ",\n  \"terminal_games_consumed\": " << terminal_games_consumed;
    if (latest_terminal)
      o << ",\n  \"latest_value_mse\": " << latest_terminal->value_mse
        << ",\n  \"latest_policy_entropy\": " << latest_terminal->policy_entropy
        << ",\n  \"latest_gradient_norm\": " << latest_terminal->gradient_norm;
    o << ",\n  \"inference_nodes\": " << infer_nodes << ",\n  \"encoder_nodes\": " << encoder.nodes
      << ",\n  \"encoder_batches\": " << encoder.batches
      << ",\n  \"policy_cpu_seconds\": " << search_service->policy_seconds()
      << ",\n  \"encoder_seconds\": " << encoder.seconds
      << ",\n  \"graph_replays\": " << graph.replays
      << ",\n  \"graph_fallbacks\": " << graph.fallbacks
      << ",\n  \"input_actual_bytes\": " << graph.input_bytes
      << ",\n  \"input_capacity_bytes\": " << graph.capacity_bytes
      << ",\n  \"encoded_tokens_actual\": " << graph.tokens
      << ",\n  \"encoded_tokens_capacity\": " << graph.token_capacity
      << ",\n  \"peak_active\": " << peak_active << ",\n  \"encoded_roots\": " << encoder.nodes
      << ",\n  \"root_cache_hits\": " << (search_service->queries.load() - encoder.nodes)
      << ",\n  \"learner_encoded_roots\": " << learner->encoded_roots
      << ",\n  \"learner_encoded_tokens\": " << learner->computed_tokens
      << ",\n  \"learner_scored_candidates\": " << learner->scored_candidates
      << ",\n  \"learner_projected_candidates\": " << learner->projected_candidates
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
    settle();
    collect_learning(true);
    if (writer.valid())
      writer.get();
    for (auto &p : games)
      if (!p.second.parked)
        submit(p.second, 2, id_packet(p.first));
    settle();
    Bytes state;
    // 冻结点两侧都从相同公开前缀重建搜索；缓存不改变恢复后的策略或正式随机序列。
    for (auto &[id, game] : games)
      game.search_plan.clear();
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
    archive->write(
        "selfplay_config",
        torch::tensor({int64_t(settings.search_simulations), int64_t(settings.search_depth),
                       int64_t(settings.history_size), int64_t(settings.history_interval),
                       int64_t(settings.history_percent)},
                      torch::kInt64));
    {
      std::vector<std::pair<uint64_t, Frozen>> bindings;
      for (auto &[id, game] : games)
        if (game.opponent)
          bindings.emplace_back(id, game.opponent);
      history.save(*archive, bindings);
    }
    auto search_data = graph.search_data();
    archive->write(
        "search_targets",
        torch::from_blob(search_data.data(), {int64_t(search_data.size())}, torch::kUInt8).clone());
    archive->write("stale_counts",
                   torch::tensor({int64_t(counters.stale_steps), int64_t(counters.stale_games)},
                                 torch::kInt64));
    // 新元数据为可选字段，旧 v3 的二进制状态和配置布局保持不变。
    archive->write(
        "discard_counts",
        torch::tensor(
            {int64_t(counters.discarded_steps), int64_t(counters.discarded_games),
             int64_t(counters.discarded_long_games), int64_t(counters.discarded_stale_games),
             int64_t(counters.discarded_memory_games), int64_t(counters.discarded_error_games)},
            torch::kInt64));
    archive->write("trajectory_limits",
                   torch::tensor({int64_t(settings.max_game_steps)}, torch::kInt64));
    auto state_owner = std::make_shared<Bytes>(std::move(state));
    archive->write("state", torch::from_blob(
                                state_owner->data(), {int64_t(state_owner->size())},
                                [state_owner](void *) {}, torch::kUInt8));
    {
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
    auto receipt = metrics(false);
    writer = std::async(std::launch::async, [archive, path, receipt] {
      auto tmp = path;
      tmp += ".tmp";
      std::ofstream file(tmp, std::ios::binary);
      archive->save_to(file);
      file.flush();
      if (!file)
        throw std::runtime_error("checkpoint write failed");
      file.close();
      replace(tmp, path);
      // 回执与已落盘的冻结快照绑定；手动保存和后台测评无需等待训练退出。
      atomic_text(path.parent_path() / "saved.json", receipt);
    });
    last_save = Clock::now();
    quiescing = !resume;
    resume_sampling();
  }
  void restore(torch::serialize::InputArchive &archive) {
    T discard_counts;
    if (archive.try_read("discard_counts", discard_counts)) {
      TORCH_CHECK(discard_counts.scalar_type() == torch::kInt64 && discard_counts.dim() == 1 &&
                      (discard_counts.numel() == 5 || discard_counts.numel() == 6) &&
                      discard_counts.ge(0).all().item<bool>(),
                  "invalid discarded trajectory counters");
      counters.discarded_steps = discard_counts[0].item<int64_t>();
      counters.discarded_games = discard_counts[1].item<int64_t>();
      counters.discarded_long_games = discard_counts[2].item<int64_t>();
      counters.discarded_stale_games = discard_counts[3].item<int64_t>();
      counters.discarded_memory_games = discard_counts[4].item<int64_t>();
      if (discard_counts.numel() == 6)
        counters.discarded_error_games = discard_counts[5].item<int64_t>();
      TORCH_CHECK(counters.discarded_games ==
                      counters.discarded_long_games + counters.discarded_stale_games +
                          counters.discarded_memory_games + counters.discarded_error_games,
                  "discarded trajectory reasons differ");
    }
    T stale_counts;
    if (archive.try_read("stale_counts", stale_counts)) {
      TORCH_CHECK(stale_counts.scalar_type() == torch::kInt64 && stale_counts.dim() == 1 &&
                      stale_counts.numel() == 2 && stale_counts.ge(0).all().item<bool>(),
                  "invalid stale trajectory counters");
      counters.stale_steps = stale_counts[0].item<int64_t>();
      counters.stale_games = stale_counts[1].item<int64_t>();
    }
    T s;
    archive.read("state", s);
    TORCH_CHECK(s.scalar_type() == torch::kUInt8 && s.dim() == 1, "invalid checkpoint metadata");
    Reader r(s.data_ptr<uint8_t>(), s.numel());
    if (r.get<uint64_t>() != 0x484a4e4154495633ull)
      throw std::runtime_error("unsupported checkpoint");
    // 构建身份只记录来源；更新测评或控制代码不应阻止读取已有 v3 检查点。
    get_bytes(r);
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
    T search_data;
    if (archive.try_read("search_targets", search_data))
      graph.search_data(search_data);
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
    auto last_control = start - std::chrono::seconds(1);
    while (!stopping) {
      auto observed_revision = mailbox.revision.load();
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
      if (stopping)
        break;
      if (control_tick)
        limit_trajectories();
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
      auto budget = settings.memory_mb * 1024 * 1024;
      if (managed() >= budget * 3 / 4)
        limit_trajectories();
      if (managed() > budget && (ready.empty() && !learning.valid())) {
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
    // 测量窗口在退出排空之前冻结，恢复累计计数不会进入本轮速率。
    collect_learning();
    auto window = metrics(true);
    atomic_text(settings.directory / "window.json", window);
    checkpoint(false);
    if (writer.valid())
      writer.get();
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
    n->cursor = path;
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
std::string describe_configuration(Settings settings) {
  auto resumed = resume_settings(std::move(settings));
  auto h = hardware(resumed.first.device);
  select_configuration(resumed.first, h);
  return configuration_json(resumed.first, h);
}
void run(const Settings &settings) {
  if (fs::exists(settings.directory) && !fs::is_empty(settings.directory))
    throw std::runtime_error("run directory must be new or empty");
  fs::create_directories(settings.directory);
  auto full_started = Clock::now();
  try {
    auto resumed = resume_settings(settings);
    {
      auto h = hardware(resumed.first.device);
      select_configuration(resumed.first, h);
      at::set_num_threads(int(resumed.first.threads));
      auto description = configuration_json(resumed.first, h);
      std::cout << description;
      atomic_text(settings.directory / "configuration.json", description);
    }
    Session session(std::move(resumed));
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
  {
    Settings settings;
    settings.device = "cpu";
    TORCH_CHECK(settings.search_simulations == 0, "MCTS must be disabled by default");
    settings.width = 32;
    settings.workers = settings.inference_batch = settings.batch = settings.publish = 1;
    settings.concurrency = settings.chunk = 0;
    settings.supplied.insert("--device");
    Session session({settings, {}});
    auto action =
        int64_t(std::find(f.node->mask.begin(), f.node->mask.end(), 1) - f.node->mask.begin());
    auto episode = [&](uint64_t version) {
      return Block{{Step{f.node, action, logps[action], version, true, 1}}, {}};
    };
    {
      auto boundary = episode(0), oversized = episode(0);
      boundary.steps.assign(4, Step{f.node, action, logps[action], 0, false, 0});
      boundary.steps.back().terminal = true;
      oversized.steps = boundary.steps;
      oversized.steps.insert(oversized.steps.begin(), boundary.steps.front());
      std::deque<Block> queue{std::move(boundary), std::move(oversized)};
      Counters rejected;
      auto accepted = terminal_batch(queue, 0, 1, 4, rejected);
      TORCH_CHECK(accepted.size() == 1 && accepted[0].steps.size() == 4 && queue.size() == 1);
      TORCH_CHECK(terminal_batch(queue, 0, 1, 4, rejected).empty() && queue.empty() &&
                      rejected.discarded_long_games == 1 && rejected.discarded_steps == 5,
                  "length limit truncated a complete episode or learned an oversized prefix");
    }
    Game parked;
    parked.id = 1;
    parked.actor = f.actor;
    parked.learner = f.learner;
    parked.parked = true;
    parked.snapshot = saved;
    parked.current = f.node;
    parked.partial = episode(0).steps;
    parked.partial.back().terminal = false;
    parked.partial.back().reward = 0;
    session.games.emplace(1, std::move(parked));
    session.next_id = 2;
    session.counters.produced = 1;
    // 旧版本停放局一直存在，短局仍须跨过 48/64 次边界独立更新。
    for (uint64_t version = 0; version < 65; ++version) {
      session.ready.push_back(episode(version));
      ++session.counters.produced;
      session.train();
      TORCH_CHECK(session.learning.valid(), "parked game blocked terminal learning");
      session.collect_learning(true);
      TORCH_CHECK(session.committed_updates == version + 1);
    }
    session.ready = {episode(0), episode(1), episode(65)};
    session.counters.produced += 3;
    session.train();
    TORCH_CHECK(session.counters.stale_games == 1 && session.counters.stale_steps == 1 &&
                    session.ready.size() == 1 && session.learning.valid(),
                "stale episode blocked fresh data or terminal batch consumed the entire queue");
    // learner 在途仍可恢复原局并补入新局，队列容量只限制后续准入。
    session.settings.concurrency = 2;
    session.resize();
    TORCH_CHECK(session.games.size() == 2, "learning in flight blocked game admission");
    session.sampling_paused = true;
    session.settle();
    session.collect_learning(true);
    TORCH_CHECK(session.consumed_lag_max == 64 && session.counters.consumed == 66 &&
                    session.games.at(1).partial.size() == 1,
                "lag boundary or parked trajectory changed");
    TORCH_CHECK(!terminal_queue_full(1, 2) && terminal_queue_full(2, 2));
    auto directory =
        fs::temp_directory_path() /
        ("haojie-async-check-" + std::to_string(Clock::now().time_since_epoch().count()));
    fs::create_directory(directory);
    session.settings.directory = directory;
    session.settings.search_simulations = 4;
    session.checkpoint(false);
    session.writer.get();
    auto search_settings = settings;
    search_settings.resume = directory / "checkpoint.pt";
    TORCH_CHECK(resume_settings(search_settings).first.search_simulations == 4,
                "resume lost an explicitly enabled search budget");
    search_settings.supplied.insert("--search-simulations");
    TORCH_CHECK(resume_settings(search_settings).first.search_simulations == 0,
                "explicit zero did not disable search from an old checkpoint");
    auto archive = std::make_shared<torch::serialize::InputArchive>();
    std::ifstream file(directory / "checkpoint.pt", std::ios::binary);
    archive->load_from(file, torch::kCPU);
    file.close();
    Session restored({settings, archive});
    TORCH_CHECK(restored.counters.stale_steps == 1 && restored.counters.stale_games == 1 &&
                    restored.counters.consumed == 66 && restored.committed_updates == 66 &&
                    restored.ready.size() == 1 && restored.games.at(1).partial.size() == 1,
                "asynchronous checkpoint lost counters or unfinished games");
    restored.metrics(false);
    TORCH_CHECK(fs::exists(directory / "saved.json"), "checkpoint did not publish its receipt");
    // 构造不含新增字段的旧 v3，验证原始状态、优化器及未结束局仍可读。
    torch::serialize::OutputArchive legacy, legacy_optimizer;
    for (auto name : {"state", "config", "weights", "actor", "stale_counts"}) {
      T tensor;
      archive->read(name, tensor);
      legacy.write(name, tensor);
    }
    session.optimizer.save(legacy_optimizer);
    legacy.write("optimizer", legacy_optimizer);
    legacy.save_to((directory / "legacy.pt").string());
    settings.resume = directory / "legacy.pt";
    Session compatible(resume_settings(settings));
    TORCH_CHECK(compatible.counters.discarded_games == 0 &&
                    compatible.games.at(1).partial.size() == 1 && compatible.optimizer.steps == 66,
                "legacy v3 checkpoint no longer restores");
    {
      auto snapshot = compatible.games.at(1).snapshot;
      auto completed = compatible.counters.finished;
      compatible.settings.directory = directory / "isolated";
      compatible.event({1, 1, snapshot, "no legal action at root", 4});
      TORCH_CHECK(!compatible.games.count(1) && compatible.counters.discarded_error_games == 1 &&
                      compatible.counters.finished == completed &&
                      fs::exists(compatible.settings.directory / "failed-games/game-1.json"),
                  "empty root stopped training, fabricated terminal or lost diagnostic state");
      compatible.checkpoint(false);
      if (compatible.writer.valid())
        compatible.writer.get();
      auto reload = compatible.settings;
      reload.resume = compatible.settings.directory / "checkpoint.pt";
      Session recovered(resume_settings(reload));
      TORCH_CHECK(recovered.counters.discarded_error_games == 1 && !recovered.games.count(1),
                  "empty-root discard did not persist");
    }
    // 正好 K=64 的未结束轨迹保留，K=65 即提前整局释放，不伪造终局。
    restored.quiescing = true;
    auto &old = restored.games.at(1);
    old.partial.front().version = 2;
    restored.limit_trajectories();
    TORCH_CHECK(restored.games.count(1) && restored.counters.discarded_games == 0);
    old.partial.front().version = 1;
    restored.limit_trajectories();
    TORCH_CHECK(!restored.games.count(1) && restored.counters.discarded_stale_games == 1 &&
                    restored.counters.discarded_steps == 1 && restored.counters.finished == 0,
                "stale unfinished game was retained or counted as terminal");
    restored.settings.max_game_steps = 4;
    Game long_game;
    long_game.id = 900;
    long_game.parked = true;
    long_game.partial.assign(4, Step{f.node, action, logps[action], 66, false, 0});
    long_game.pending = long_game.partial.front();
    restored.games.emplace(900, std::move(long_game));
    restored.counters.produced += 4;
    restored.limit_trajectories();
    TORCH_CHECK(!restored.games.count(900) && restored.counters.discarded_long_games == 1 &&
                    restored.counters.discarded_steps == 5 && restored.ready.size() == 1,
                "length rejection leaked pending input or discarded a complete game");
    restored.settings.max_game_steps = 262144;
    restored.settings.memory_mb = std::max<uint64_t>(4, restored.managed() / (1024 * 1024) * 4 + 4);
    auto memory_budget = restored.settings.memory_mb * 1024 * 1024;
    // 多轮制造真实计费内存，释放后必须回到半预算且允许重新准入。
    for (int round = 0; round < 16; ++round) {
      {
        auto heavy = std::make_shared<Node>();
        heavy->root = f.node->root;
        heavy->context.assign(memory_budget / sizeof(RowRef), f.node->root->globals);
        heavy->account_input();
        Game pressure;
        pressure.id = 901 + round;
        pressure.parked = true;
        pressure.partial.assign(4, Step{heavy, action, logps[action], 66, false, 0});
        restored.games.emplace(pressure.id, std::move(pressure));
      }
      restored.counters.produced += 4;
      restored.limit_trajectories();
      TORCH_CHECK(restored.managed() < memory_budget / 2 && !restored.memory_paused &&
                      !restored.games.count(901 + round) &&
                      restored.counters.discarded_memory_games >= uint64_t(round + 1) &&
                      restored.ready.size() == 1,
                  "memory pressure did not reclaim unfinished inputs across repeated rounds");
      restored.metrics(false);
    }
    // 完整终局独占预算时保留它并暂停采样，消费者释放后解除背压。
    Block large_terminal;
    large_terminal.steps.assign(memory_budget / sizeof(Step),
                                Step{f.node, action, logps[action], 66, false, 0});
    large_terminal.steps.back().terminal = true;
    auto retained_steps = large_terminal.steps.size();
    restored.settings.max_game_steps = std::max<uint64_t>(262144, retained_steps + 1);
    restored.ready.push_back(std::move(large_terminal));
    restored.counters.produced += retained_steps;
    restored.limit_trajectories();
    TORCH_CHECK(restored.memory_paused && restored.sampling_paused && restored.ready.size() == 2,
                "memory pressure discarded complete terminal data instead of pausing sampling");
    restored.ready.pop_back();
    restored.counters.produced -= retained_steps;
    restored.limit_trajectories();
    TORCH_CHECK(!restored.memory_paused);
    restored.metrics(false);
    restored.settings.directory = directory;
    restored.settings.max_game_steps = 4096;
    restored.checkpoint(false);
    restored.writer.get();
    settings.resume = directory / "checkpoint.pt";
    Session after_discard(resume_settings(settings));
    TORCH_CHECK(after_discard.settings.max_game_steps == 4096 &&
                    after_discard.counters.discarded_long_games == 1 &&
                    after_discard.counters.discarded_stale_games == 1 &&
                    after_discard.counters.discarded_memory_games >= 16 &&
                    after_discard.optimizer.steps == 66 && after_discard.ready.size() == 1,
                "checkpoint lost discard reasons, limit, optimizer or complete games");
    after_discard.metrics(false);
    // 恢复可显式覆盖长度上限，不改变模型或优化器配置。
    settings.max_game_steps = 8192;
    settings.supplied.insert("--max-game-steps");
    TORCH_CHECK(resume_settings(settings).first.max_game_steps == 8192);
    TORCH_CHECK(fs::equivalent(directory.parent_path(), fs::temp_directory_path()));
    fs::remove_all(directory / "isolated");
    fs::remove(directory / "legacy.pt");
    fs::remove(directory / "checkpoint.pt");
    fs::remove(directory / "saved.json");
    fs::remove(directory);
  }
  {
    Settings settings;
    settings.device = "cpu";
    settings.width = 32;
    settings.workers = settings.inference_batch = settings.batch = settings.publish = 1;
    settings.concurrency = 2;
    settings.supplied.insert("--device");
    settings.chunk = 0;
    settings.max_game_steps = 8;
    settings.seconds = 3;
    settings.search_simulations = settings.search_depth = 1;
    settings.save_seconds = 0;
    settings.directory =
        fs::temp_directory_path() /
        ("haojie-discard-check-" + std::to_string(Clock::now().time_since_epoch().count()));
    fs::create_directory(settings.directory);
    {
      Session live({settings, {}});
      live.execute();
      TORCH_CHECK(live.counters.discarded_long_games > 2 && live.next_id > 5 &&
                      live.managed() < settings.memory_mb * 1024 * 1024,
                  "live actor deletion failed to free inputs and admit replacement games");
      live.metrics(false);
    }
    settings.resume = settings.directory / "checkpoint.pt";
    Session continued(resume_settings(settings));
    TORCH_CHECK(continued.counters.discarded_long_games > 2 &&
                    continued.settings.max_game_steps == 8,
                "live discard checkpoint did not resume");
    continued.metrics(false);
    fs::remove_all(settings.directory);
  }
  std::cout << "bounded unfinished trajectories, live discard/replacement, memory pressure and "
               "legacy v3 recovery: passed\n";
  std::cout << "asynchronous terminal learning, parked long game, stale rejection and busy "
               "admission: passed\n";
  std::cout << "typed node roundtrip, actor parking, RNG/progress continuation and canonical "
               "behavior probability: passed\n";
}
} // namespace hj
