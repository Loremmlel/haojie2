#include "runtime.h"
#include "engine.h"
#include <atomic>
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

namespace hj {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace {
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void replace(const fs::path &source, const fs::path &target) {
#ifdef _WIN32
  if (!MoveFileExW(source.c_str(), target.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    throw std::runtime_error("atomic file replacement failed");
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
void write_node(Bytes &b, const std::shared_ptr<Node> &n) {
  put<uint8_t>(b, bool(n));
  if (n)
    put_bytes(b, encode_node(*n));
}
std::shared_ptr<Node> read_node(Reader &r) {
  if (!r.get<uint8_t>())
    return {};
  auto b = get_bytes(r);
  Reader inner(b);
  auto node = decode_node(inner);
  inner.end();
  return node;
}
void write_step(Bytes &b, const Step &s) {
  write_node(b, s.node);
  put(b, s.selected);
  put(b, s.logp);
  put(b, s.version);
  put<uint8_t>(b, s.terminal);
  put(b, s.reward);
}
Step read_step(Reader &r) {
  Step s;
  s.node = read_node(r);
  s.selected = r.get<int64_t>();
  s.logp = r.get<double>();
  s.version = r.get<uint64_t>();
  s.terminal = r.get<uint8_t>() != 0;
  s.reward = r.get<int8_t>();
  return s;
}
template <class V> void write_array(Bytes &b, const std::vector<V> &v) {
  put<uint64_t>(b, v.size());
  raw(b, v.data(), v.size() * sizeof(V));
}
template <class V> std::vector<V> read_array(Reader &r) {
  return r.array<V>(r.get<uint64_t>());
}
void write_block(Bytes &b, const Block &block) {
  put<uint64_t>(b, block.steps.size());
  for (auto &s : block.steps)
    write_step(b, s);
  write_node(b, block.bootstrap);
}
Block read_block(Reader &r) {
  Block b;
  auto count = r.get<uint64_t>();
  for (uint64_t i = 0; i < count; ++i)
    b.steps.push_back(read_step(r));
  b.bootstrap = read_node(r);
  return b;
}

struct Job {
  uint64_t game;
  uint32_t op;
  Bytes bytes;
};
struct Event {
  uint64_t game;
  uint32_t op;
  Bytes bytes;
  std::string error;
};
struct Mailbox {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Event> events;
  std::atomic<size_t> bytes{0};
  void push(Event e) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      bytes += e.bytes.capacity() + e.error.capacity();
      events.push_back(std::move(e));
    }
    changed.notify_one();
  }
  std::optional<Event> pop() {
    std::lock_guard<std::mutex> lock(mutex);
    if (events.empty())
      return {};
    Event e = std::move(events.front());
    events.pop_front();
    bytes -= e.bytes.capacity() + e.error.capacity();
    return e;
  }
};
class Lane {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Job> jobs;
  bool closing = false;
  Mailbox &output;
  std::thread thread;

public:
  explicit Lane(Mailbox &out)
      : output(out), thread([this] {
          void *actor = hj_actor_create();
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
            if (!actor)
              e.error = "cannot initialize Rust actor lane";
            else {
              const uint8_t *p = nullptr;
              size_t n = 0;
              int status = hj_actor_call(actor, job.op, job.bytes.data(), job.bytes.size(), &p, &n);
              if (n)
                e.bytes.assign(p, p + n);
              if (status)
                e.error = std::string(e.bytes.begin(), e.bytes.end());
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
struct Game {
  uint64_t id = 0, commands = 0, microsteps = 0, backtracks = 0, failures = 0;
  bool busy = false, parked = false, sent_training = false, queued = false;
  uint32_t actor = 0, learner = 0;
  std::shared_ptr<Root> root;
  std::shared_ptr<Node> current, sent;
  uint64_t sent_version = 0;
  std::optional<Step> pending;
  std::vector<Step> partial;
  Bytes snapshot;
  Clock::time_point last_advance{};
};
struct Counters {
  uint64_t consumed = 0, produced = 0, commands = 0, microsteps = 0, finished = 0, backtracks = 0,
           failures = 0;
};
Settings resume_settings(Settings s) {
  if (s.resume.empty())
    return s;
  torch::serialize::InputArchive archive;
  std::ifstream file(s.resume, std::ios::binary);
  archive.load_from(file, torch::kCPU);
  T config;
  archive.read("config", config);
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
  return s;
}
class Session {
  Settings settings;
  torch::Device device;
  Network learner, actor;
  Optimizer optimizer;
  Mailbox mailbox;
  std::vector<std::unique_ptr<Lane>> lanes;
  std::map<uint64_t, Game> games;
  std::deque<Block> ready;
  std::deque<uint64_t> inference_ready;
  Counters counters, initial;
  uint64_t next_id = 1, published = 0, initial_updates = 0, committed_updates = 0;
  std::atomic<uint64_t> actor_version{0};
  std::mutex gpu;
  std::future<TrainResult> learning;
  std::vector<Block> in_learning;
  std::future<void> writer;
  Clock::time_point start, last_status, last_save;
  uint64_t infer_batches = 0, infer_nodes = 0, peak_active = 0, peak_managed = 0;
  double learning_seconds = 0, inference_seconds = 0;
  Clock::time_point learn_start;

public:
  explicit Session(Settings s)
      : settings(resume_settings(std::move(s))), device(settings.device), learner(settings.width),
        actor(settings.width), optimizer(learner) {
    learner->to(device);
    actor->to(device);
    // 优化器在设备迁移后重建，浮点参数及状态始终为 BF16。
    optimizer = Optimizer(learner);
    copy_weights(learner, actor);
    for (uint64_t i = 0; i < settings.workers; ++i)
      lanes.push_back(std::make_unique<Lane>(mailbox));
    if (!settings.resume.empty())
      restore(settings.resume);
    initial = counters;
    initial_updates = committed_updates = optimizer.steps;
    start = last_status = last_save = Clock::now();
  }
  void submit(Game &g, uint32_t op, Bytes b) {
    if (g.busy)
      throw std::runtime_error("actor already has work");
    g.busy = true;
    lanes[g.id % lanes.size()]->submit({g.id, op, std::move(b)});
  }
  void enqueue(Game &g) {
    if (!g.queued && !g.busy && !g.parked && g.current) {
      g.queued = true;
      inference_ready.push_back(g.id);
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
  uint64_t managed() const {
    // 根和节点按地址去重；未完成、待推理、在途回复和学习中的块全部纳入预算。
    std::map<const Node *, size_t> nodes;
    std::map<const Root *, size_t> roots;
    uint64_t bytes = hj_memory_bytes() + mailbox.bytes.load();
    auto node = [&](const std::shared_ptr<Node> &n) {
      if (n) {
        nodes[n.get()] = n->bytes();
        auto &r = *n->root;
        roots[&r] =
            sizeof(Root) + r.rows.size() * sizeof(Row) + r.kinds.size() * 8 +
            (r.cached_tokens.defined() ? r.cached_tokens.nbytes() + r.cached_context.nbytes() : 0);
      }
    };
    auto block = [&](const Block &b) {
      for (auto &s : b.steps)
        node(s.node);
      node(b.bootstrap);
    };
    for (auto &pair : games) {
      auto &g = pair.second;
      bytes += g.snapshot.size() + sizeof(Game);
      node(g.current);
      node(g.sent);
      if (g.pending)
        node(g.pending->node);
      for (auto &s : g.partial)
        node(s.node);
    }
    for (auto &b : ready)
      block(b);
    for (auto &b : in_learning)
      block(b);
    for (auto p : nodes)
      bytes += p.second;
    for (auto p : roots)
      bytes += p.second;
    return bytes;
  }
  void successor(Game &g, const std::shared_ptr<Node> &next, std::optional<int8_t> reward = {}) {
    if (g.pending) {
      auto step = std::move(*g.pending);
      g.pending.reset();
      if (reward) {
        step.terminal = true;
        step.reward = *reward;
      }
      g.partial.push_back(std::move(step));
      ++counters.produced;
    }
    if (!g.partial.empty() && (g.partial.size() >= settings.chunk || reward)) {
      ready.push_back({std::move(g.partial), next});
      g.partial.clear();
    }
  }
  void event(Event e) {
    if (!e.error.empty())
      throw std::runtime_error(e.error);
    auto found = games.find(e.game);
    if (found == games.end())
      throw std::runtime_error("unknown lane event");
    auto &g = found->second;
    g.busy = false;
    if (e.op == 0 || e.op == 4) {
      if (e.op == 4) {
        g.snapshot.clear();
        g.parked = false;
      }
      if (!g.current) {
        Bytes b = id_packet(g.id);
        put<uint32_t>(b, 0);
        submit(g, 1, std::move(b));
      } else
        enqueue(g);
    } else if (e.op == 1) {
      Frame f = parse(e.bytes, g.root);
      if (f.commands > g.commands || f.microsteps > g.microsteps)
        g.last_advance = Clock::now();
      counters.commands += f.commands - g.commands;
      counters.microsteps += f.microsteps - g.microsteps;
      counters.backtracks += f.backtracks - g.backtracks;
      counters.failures += f.failures - g.failures;
      g.commands = f.commands;
      g.microsteps = f.microsteps;
      g.backtracks = f.backtracks;
      g.failures = f.failures;
      g.actor = f.actor;
      g.learner = f.learner;
      if (f.selected >= 0) {
        if (!g.sent)
          throw std::runtime_error("selection without behavior node");
        if (g.sent_training) {
          if (g.pending)
            throw std::runtime_error("missing same-seat successor");
          g.pending = Step{g.sent, f.selected, f.logp, g.sent_version, false, 0};
        }
        g.sent.reset();
      }
      g.current = f.node;
      if (f.kind == 1) {
        int8_t reward = f.winner == 0 ? 0 : (uint32_t(f.winner) == g.learner ? 1 : -1);
        successor(g, {}, reward);
        ++counters.finished;
        submit(g, 5, id_packet(g.id));
      } else if (f.kind == 2) {
        Bytes b = id_packet(g.id);
        put<uint32_t>(b, 0);
        submit(g, 1, std::move(b));
      } else {
        if (g.actor == g.learner)
          successor(g, g.current);
        enqueue(g);
      }
    } else if (e.op == 2 || e.op == 3) {
      g.snapshot = std::move(e.bytes);
      if (e.op == 3) {
        g.parked = true;
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
    while (auto e = mailbox.pop())
      event(std::move(*e));
  }
  void settle() {
    for (;;) {
      collect();
      bool busy = false;
      for (auto &p : games)
        busy |= p.second.busy;
      if (!busy)
        break;
      std::unique_lock<std::mutex> lock(mailbox.mutex);
      mailbox.changed.wait_for(lock, std::chrono::milliseconds(10));
    }
  }
  void collect_learning(bool wait = false) {
    if (learning.valid() &&
        (wait || learning.wait_for(std::chrono::seconds(0)) == std::future_status::ready)) {
      auto result = learning.get();
      committed_updates = optimizer.steps;
      counters.consumed += result.consumed;
      learning_seconds += elapsed(learn_start);
      in_learning.clear();
    }
  }
  void train() {
    collect_learning();
    if (learning.valid() || ready.empty())
      return;
    uint64_t desired = settings.batch;
    if (settings.samples)
      desired = std::min(desired, settings.samples - (counters.consumed - initial.consumed));
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
    if (n < desired && managed() < settings.memory_mb * 1024 * 1024 / 2) {
      for (auto it = in_learning.rbegin(); it != in_learning.rend(); ++it)
        ready.push_front(std::move(*it));
      in_learning.clear();
      return;
    }
    learn_start = Clock::now();
    learning = std::async(std::launch::async, [this] {
      std::lock_guard<std::mutex> lock(gpu);
      auto result = learn(learner, optimizer, in_learning);
      if (optimizer.steps - published >= settings.publish) {
        copy_weights(learner, actor);
        actor_version = published = optimizer.steps;
      }
      return result;
    });
  }
  void resize() {
    uint64_t running = active();
    // 大编号先停放；在途原子规则操作完成后立即停放，进度原样保留。
    for (auto it = games.rbegin(); it != games.rend() && running > settings.concurrency; ++it) {
      auto &g = it->second;
      if (!g.parked && !g.busy) {
        g.parked = true;
        submit(g, 3, id_packet(g.id));
        --running;
      }
    }
    for (auto &p : games) {
      auto &g = p.second;
      if (running >= settings.concurrency)
        break;
      if (g.parked && !g.busy) {
        g.parked = false;
        submit(g, 4, g.snapshot);
        ++running;
      }
    }
    while (running < settings.concurrency && managed() < settings.memory_mb * 1024 * 1024 / 2) {
      Game game;
      game.id = next_id++;
      game.learner = uint32_t(1 + (game.id & 1));
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
  void infer() {
    std::vector<Game *> chosen;
    std::vector<std::shared_ptr<Node>> nodes;
    size_t work = 0;
    uint64_t budget = settings.memory_mb * 1024 * 1024;
    // 所有未完成轨迹也计费；在内存紧张时切出已有短块，不锁住固定长度轨迹。
    if (managed() > budget * 3 / 4) {
      for (auto &p : games) {
        auto &g = p.second;
        if (!g.partial.empty() && g.current && g.actor == g.learner) {
          ready.push_back({std::move(g.partial), g.current});
          g.partial.clear();
        }
      }
      if (!ready.empty() || learning.valid())
        return;
    }
    // 选批与缓存计费使用同一参数版本；学习占用设备时继续处理 CPU 和控制消息。
    std::unique_lock<std::mutex> device_lock(gpu, std::try_to_lock);
    if (!device_lock.owns_lock())
      return;
    while (!inference_ready.empty()) {
      auto found = games.find(inference_ready.front());
      inference_ready.pop_front();
      if (found == games.end())
        continue;
      auto &g = found->second;
      if (!g.queued)
        continue;
      g.queued = false;
      if (g.busy || g.parked || !g.current)
        continue;
      size_t next = (actor->cache_valid(*g.current->root) ? 0 : g.current->root->rows.size()) +
                    g.current->prefix.size() + g.current->candidates.size();
      if (!nodes.empty() && (chosen.size() >= settings.inference_batch || work + next > 8192)) {
        g.queued = true;
        inference_ready.push_front(g.id);
        break;
      }
      chosen.push_back(&g);
      nodes.push_back(g.current);
      work += next;
    }
    if (nodes.empty())
      return;
    std::vector<std::vector<uint16_t>> probs;
    uint64_t version;
    auto t = Clock::now();
    {
      torch::NoGradGuard guard;
      auto output = actor->forward(nodes);
      probs = probabilities(output.first, nodes);
      version = actor_version;
    }
    device_lock.unlock();
    inference_seconds += elapsed(t);
    ++infer_batches;
    infer_nodes += nodes.size();
    // GPU 锁已释放；回送只入 CPU lane 队列，不等待任何学习工作。
    for (size_t i = 0; i < chosen.size(); ++i) {
      auto &g = *chosen[i];
      Bytes b = id_packet(g.id);
      put<uint32_t>(b, probs[i].size());
      raw(b, probs[i].data(), probs[i].size() * 2);
      g.sent = g.current;
      g.sent_training = g.actor == g.learner;
      g.sent_version = version;
      g.current.reset();
      submit(g, 1, std::move(b));
    }
  }
  std::string metrics(bool window) {
    auto delta = window ? initial : Counters{};
    uint64_t partial = 0, pending = 0, busy = 0, parked = 0, advancing = 0, inference_waiting = 0;
    for (auto &p : games) {
      partial += p.second.partial.size();
      pending += bool(p.second.pending);
      busy += p.second.busy;
      parked += p.second.parked;
      if (!p.second.parked) {
        advancing += elapsed(p.second.last_advance) < 1;
        inference_waiting += bool(p.second.current) && !p.second.busy;
      }
    }
    auto used = managed();
    peak_managed = std::max(peak_managed, used);
    double wall = elapsed(start);
    std::ostringstream o;
    o << std::setprecision(12) << "{\n  \"format\": \"haojie-native-v1\",\n  \"seconds\": " << wall
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
      << ",\n  \"ready_steps\": " << ready_steps() << ",\n  \"partial_steps\": " << partial
      << ",\n  \"pending_steps\": " << pending << ",\n  \"learning_steps\": ";
    uint64_t learning_steps = 0;
    for (auto &b : in_learning)
      learning_steps += b.steps.size();
    if (counters.consumed + ready_steps() + partial + learning_steps != counters.produced)
      throw std::runtime_error("trajectory accounting differs");
    o << learning_steps << ",\n  \"inference_nodes\": " << infer_nodes
      << ",\n  \"inference_batches\": " << infer_batches
      << ",\n  \"encoded_roots\": " << actor->encoded_roots
      << ",\n  \"root_cache_hits\": " << actor->cache_hits
      << ",\n  \"inference_seconds\": " << inference_seconds
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
  void checkpoint() {
    settle();
    collect_learning(true);
    if (writer.valid())
      writer.get();
    for (auto &p : games)
      if (!p.second.parked)
        submit(p.second, 2, id_packet(p.first));
    settle();
    Bytes state;
    put<uint64_t>(state, 0x484a4e4154495631ull);
    std::string identity = hj_identity();
    put_bytes(state, Bytes(identity.begin(), identity.end()));
    for (auto n : {settings.width, settings.seed, settings.chunk, settings.batch, next_id,
                   actor_version.load(), published, optimizer.steps})
      put(state, n);
    for (auto n : {counters.consumed, counters.produced, counters.commands, counters.microsteps,
                   counters.finished, counters.backtracks, counters.failures})
      put(state, n);
    put<uint64_t>(state, games.size());
    for (auto &p : games) {
      auto &g = p.second;
      put(state, g.id);
      put(state, g.commands);
      put(state, g.microsteps);
      put(state, g.backtracks);
      put(state, g.failures);
      put(state, g.actor);
      put(state, g.learner);
      put<uint8_t>(state, g.parked);
      put_bytes(state, g.snapshot);
      write_node(state, g.current);
      put<uint8_t>(state, bool(g.pending));
      if (g.pending)
        write_step(state, *g.pending);
      put<uint64_t>(state, g.partial.size());
      for (auto &s : g.partial)
        write_step(state, s);
      if (!g.parked)
        Bytes{}.swap(g.snapshot);
    }
    put<uint64_t>(state, ready.size());
    for (auto &b : ready)
      write_block(state, b);
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
    archive->write("state",
                   torch::from_blob(state.data(), {int64_t(state.size())}, torch::kUInt8).clone());
    {
      std::lock_guard<std::mutex> lock(gpu);
      torch::serialize::OutputArchive weights, behavior, opt;
      auto save_weights = [](Network &n, torch::serialize::OutputArchive &a) {
        size_t i = 0;
        for (auto &p : n->parameters())
          a.write("p" + std::to_string(i++), p.detach().to(torch::kCPU).clone());
      };
      save_weights(learner, weights);
      save_weights(actor, behavior);
      optimizer.save(opt);
      archive->write("weights", weights);
      archive->write("actor", behavior);
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
  }
  void restore(const fs::path &path) {
    torch::serialize::InputArchive archive;
    std::ifstream file(path, std::ios::binary);
    archive.load_from(file, torch::kCPU);
    T s;
    archive.read("state", s);
    TORCH_CHECK(s.scalar_type() == torch::kUInt8 && s.dim() == 1, "invalid checkpoint metadata");
    Bytes bytes(s.data_ptr<uint8_t>(), s.data_ptr<uint8_t>() + s.numel());
    Reader r(bytes);
    if (r.get<uint64_t>() != 0x484a4e4154495631ull)
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
      g.current = read_node(r);
      if (g.current)
        g.root = g.current->root;
      if (r.get<uint8_t>())
        g.pending = read_step(r);
      auto n = r.get<uint64_t>();
      for (uint64_t j = 0; j < n; ++j)
        g.partial.push_back(read_step(r));
      g.parked = true;
      games.emplace(g.id, std::move(g));
    }
    count = r.get<uint64_t>();
    for (uint64_t i = 0; i < count; ++i)
      ready.push_back(read_block(r));
    r.end();
    torch::serialize::InputArchive weights, behavior, opt;
    archive.read("weights", weights);
    archive.read("actor", behavior);
    archive.read("optimizer", opt);
    auto load_weights = [&](Network &n, torch::serialize::InputArchive &a) {
      torch::NoGradGuard guard;
      size_t i = 0;
      for (auto &p : n->parameters()) {
        T v;
        a.read("p" + std::to_string(i++), v);
        TORCH_CHECK(v.scalar_type() == torch::kBFloat16 && v.sizes() == p.sizes() &&
                        torch::isfinite(v).all().item<bool>(),
                    "invalid BF16 weights");
        p.copy_(v.to(device));
      }
    };
    load_weights(learner, weights);
    load_weights(actor, behavior);
    optimizer.load(opt, learner);
    if (optimizer.steps != steps)
      throw std::runtime_error("optimizer metadata differs");
  }
  void execute() {
    bool stopping = false;
    while (!stopping) {
      collect();
      collect_learning();
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
      if (settings.seconds > 0 && elapsed(start) >= settings.seconds)
        stopping = true;
      if (settings.updates > 0 && committed_updates - initial_updates >= settings.updates)
        stopping = true;
      if (settings.samples > 0 && counters.consumed - initial.consumed >= settings.samples)
        stopping = true;
      if (stopping)
        break;
      resize();
      train();
      infer();
      auto save = settings.directory / "save";
      if (fs::exists(save) ||
          (settings.save_seconds > 0 && elapsed(last_save) >= settings.save_seconds)) {
        if (fs::exists(save))
          fs::remove(save);
        checkpoint();
      }
      if (elapsed(last_status) >= 1) {
        atomic_text(settings.directory / "status.json", metrics(true));
        last_status = Clock::now();
      }
      if (managed() > settings.memory_mb * 1024 * 1024) {
        checkpoint();
        if (writer.valid())
          writer.get();
        throw std::runtime_error(
            "managed memory budget exceeded; saved checkpoint, increase --memory-mb");
      }
      std::unique_lock<std::mutex> lock(mailbox.mutex);
      mailbox.changed.wait_for(lock, std::chrono::milliseconds(1));
    }
    // 测量窗口在退出排空之前冻结，恢复累计计数不会进入本轮速率。
    collect_learning();
    auto window = metrics(true);
    atomic_text(settings.directory / "window.json", window);
    checkpoint();
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
Frame parse(const Bytes &bytes, std::shared_ptr<Root> &cached) {
  Reader r(bytes);
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
    if (has_root)
      root->rows = r.array<Row>(entities);
    auto n = std::make_shared<Node>();
    n->prefix = r.array<Row>(prefix);
    root->kinds.reserve(entities);
    if (has_root) {
      for (uint32_t i = 0; i < entities; ++i)
        root->kinds.push_back(r.get<uint32_t>());
      r.raw(root->globals.data(), 64);
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
    n->candidates = r.array<std::array<int64_t, 64>>(candidates);
    n->sources = r.array<int32_t>(candidates);
    n->targets = r.array<int32_t>(candidates);
    n->mask = r.array<uint8_t>(candidates);
    n->cursor = r.array<uint32_t>(cursor);
    for (uint32_t i = 0; i < facts; ++i) {
      auto path = r.array<uint32_t>(r.get<uint32_t>());
      auto children = r.array<uint32_t>(r.get<uint32_t>());
      n->excluded.push_back({std::move(path), std::move(children)});
    }
    f.node = n;
  }
  r.end();
  return f;
}
Bytes encode_node(const Node &n) {
  Bytes b;
  auto &root = *n.root;
  put(b, root.game);
  put(b, root.id);
  write_array(b, root.rows);
  write_array(b, root.kinds);
  raw(b, root.globals.data(), 64);
  write_array(b, n.prefix);
  write_array(b, n.candidates);
  write_array(b, n.sources);
  write_array(b, n.targets);
  write_array(b, n.mask);
  write_array(b, n.cursor);
  put<uint64_t>(b, n.excluded.size());
  for (auto &p : n.excluded) {
    write_array(b, p.path);
    write_array(b, p.children);
  }
  return b;
}
std::shared_ptr<Node> decode_node(Reader &r) {
  auto n = std::make_shared<Node>();
  n->root = std::make_shared<Root>();
  auto &root = *n->root;
  root.game = r.get<uint64_t>();
  root.id = r.get<uint64_t>();
  root.rows = read_array<Row>(r);
  root.kinds = read_array<int64_t>(r);
  r.raw(root.globals.data(), 64);
  n->prefix = read_array<Row>(r);
  n->candidates = read_array<std::array<int64_t, 64>>(r);
  n->sources = read_array<int32_t>(r);
  n->targets = read_array<int32_t>(r);
  n->mask = read_array<uint8_t>(r);
  n->cursor = read_array<uint32_t>(r);
  auto count = r.get<uint64_t>();
  for (uint64_t i = 0; i < count; ++i) {
    auto path = read_array<uint32_t>(r);
    auto children = read_array<uint32_t>(r);
    n->excluded.push_back({std::move(path), std::move(children)});
  }
  return n;
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
  auto first = call(1, request);
  auto f = parse(first, root);
  TORCH_CHECK(f.kind == 0 && f.node && f.node->mask.size() > 1);
  auto encoded = encode_node(*f.node);
  Reader node_reader(encoded);
  auto decoded = decode_node(node_reader);
  node_reader.end();
  TORCH_CHECK(encode_node(*decoded) == encoded);
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
  auto selected = parse(next, next_root);
  auto logps = canonical_logps(mass);
  TORCH_CHECK(selected.selected >= 0 && std::abs(selected.logp - logps[selected.selected]) < 1e-12);
  call(3, id);
  call(4, saved);
  auto replay = call(1, reply);
  auto replay_root = root;
  auto continued = parse(replay, replay_root);
  TORCH_CHECK(continued.commands == selected.commands &&
                  continued.microsteps == selected.microsteps &&
                  continued.selected == selected.selected && continued.logp == selected.logp,
              "actor restore changed RNG or command progress");
  TORCH_CHECK(bool(continued.node) == bool(selected.node));
  if (continued.node)
    TORCH_CHECK(encode_node(*continued.node) == encode_node(*selected.node));
  hj_actor_close(handle);
  std::cout << "typed node roundtrip, actor parking, RNG/progress continuation and canonical "
               "behavior probability: passed\n";
}
} // namespace hj
