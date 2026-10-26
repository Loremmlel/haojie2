#include "search.h"
#include "engine.h"
#include "history.h"
#include "mcts.h"
#include "storage/checkpoint.h"
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <iostream>
#include <sstream>

namespace hj {
FrozenPolicy::FrozenPolicy(Network &source, uint64_t version)
    : model(source->width), version(version) {
  torch::NoGradGuard guard;
  model->to(source->packed_weights.device());
  copy_weights(source, model);
  // 冻结对象释放后地址可能复用；版本也参与根缓存身份，不能把所有快照都记为 revision=1。
  model->revision = version + 1;
  model->eval();
  if (source->packed_weights.is_cuda()) {
    c10::impl::VirtualGuardImpl implementation(source->packed_weights.device().type());
    implementation.synchronizeStream(implementation.getStream(source->packed_weights.device()));
  }
  cpu = CpuPolicy::snapshot(model, version);
  storage.set(model->packed_weights.nbytes() + cpu->rows.storage().nbytes());
}
SearchService::SearchService(const torch::Device &device, uint64_t width, uint64_t max_roots)
    : batch_limit(max_roots), model(width) {
  torch::NoGradGuard guard;
  model->to(device);
  model->pack_weights();
  encoder = std::make_unique<InferenceEngine>(
      model, device, [this] { changed.notify_all(); }, 1, max_roots);
  thread = std::thread([this, device] {
    torch::NoGradGuard guard;
    CpuPolicy decoder;
    Frozen loaded;
    std::optional<c10::Stream> stream;
    if (device.is_cuda())
      stream = c10::impl::VirtualGuardImpl(device.type()).getStreamFromGlobalPool(device, false);
    c10::OptionalStreamGuard stream_guard(stream);
    for (;;) {
      std::vector<Request> batch;
      {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return closing || !requests.empty(); });
        if (requests.empty())
          break;
        auto policy = requests.front().policy;
        size_t tokens = 0;
        for (auto it = requests.begin(); it != requests.end() && batch.size() < batch_limit;) {
          if (it->policy != policy || (tokens && tokens + it->node->root->rows.size() > 8192)) {
            ++it;
            continue;
          }
          tokens += it->node->root->rows.size() + 1;
          batch.push_back(std::move(*it));
          it = requests.erase(it);
        }
      }
      try {
        std::vector<std::shared_ptr<Node>> nodes;
        for (auto &request : batch)
          nodes.push_back(request.node);
        auto policy = batch.front().policy;
        std::vector<std::shared_ptr<Node>> missing;
        for (auto &node : nodes)
          if (!node->root->host_valid(policy->cpu->owner, policy->cpu->revision))
            missing.push_back(node);
        if (!missing.empty()) {
          if (loaded != policy) {
            copy_weights(policy->model, model);
            if (device.is_cuda())
              c10::impl::VirtualGuardImpl(device.type())
                  .synchronizeStream(c10::impl::VirtualGuardImpl(device.type()).getStream(device));
            loaded = policy;
          }
          auto flight = encoder->submit(0, missing, policy->version);
          {
            std::unique_lock lock(mutex);
            while (!flight->complete.load(std::memory_order_acquire))
              changed.wait_for(lock, std::chrono::milliseconds(10));
          }
          encoder->finish(flight);
          encoded_nodes += missing.size();
          ++encoded_batches;
          encoding_ns += uint64_t(std::llround(flight->seconds * 1e9));
          for (auto &node : missing) {
            std::lock_guard lock(node->root->cache_mutex);
            node->root->cached_owner = policy->cpu->owner;
            node->root->cached_revision = policy->cpu->revision;
          }
        }
        std::vector<SearchPrediction> predictions;
        auto decode_start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < batch.size(); ++i) {
          double value;
          auto mass = decoder.score(*nodes[i], policy->cpu, &value);
          predictions.push_back({std::move(mass), value});
        }
        decoding_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now() - decode_start)
                           .count();
        for (size_t i = 0; i < batch.size(); ++i)
          batch[i].result.set_value(std::move(predictions[i]));
      } catch (...) {
        for (auto &request : batch)
          request.result.set_exception(std::current_exception());
      }
    }
  });
}
SearchService::~SearchService() {
  {
    std::lock_guard lock(mutex);
    closing = true;
  }
  changed.notify_all();
  thread.join();
}
SearchPrediction SearchService::predict(const Frozen &policy, const std::shared_ptr<Node> &node) {
  ++queries;
  Request request{policy, node};
  auto result = request.result.get_future();
  {
    std::lock_guard lock(mutex);
    TORCH_CHECK(!closing, "search service closed");
    requests.push_back(std::move(request));
  }
  changed.notify_one();
  return result.get();
}
namespace {
struct PublicState {
  uint64_t snapshot = 0;
  std::function<void(uint64_t)> release;
  std::shared_ptr<Node> query;
  SearchPrediction prediction;
  StorageBytes storage;
  ~PublicState() {
    if (release)
      release(snapshot);
  }
};
using Position = mcts::Position<std::shared_ptr<PublicState>>;
using Path = std::vector<uint32_t>;
uint32_t public_search_seed(const Node &node) {
  uint32_t seed = 2166136261u;
  auto mix = [&](uint32_t value) { seed = (seed ^ value) * 16777619u; };
  auto &row = node.root->globals->value;
  for (auto value : row.ids) {
    auto bits = uint64_t(value);
    mix(uint32_t(bits));
    mix(uint32_t(bits >> 32));
  }
  for (auto bits : {row.present, row.integers}) {
    mix(uint32_t(bits));
    mix(uint32_t(bits >> 32));
  }
  for (auto value : row.values)
    mix(value);
  for (auto index : node.cursor)
    mix(index);
  return seed ? seed : 1;
}
struct SearchLimit : std::runtime_error {
  SearchLimit() : std::runtime_error("search command parameter budget exceeded") {}
};
struct EmptySearchBranch : std::runtime_error {
  EmptySearchBranch() : std::runtime_error("no legal action at root") {}
};
// 临时 actor 只拥有公开模拟局面；析构删除全部搜索状态，原对局保持原样。
class SearchActor {
  void *actor;
  uint64_t id;
  bool opened = false;
  Frozen policy;
  SearchService &service;
  uint32_t perspective;
  InputPool pool;
  std::shared_ptr<Root> root;
  const std::function<bool()> &cancelled;
  std::map<std::string, Position> cache;
  Bytes packet() const {
    Bytes out;
    put(out, id);
    return out;
  }
  Bytes call(uint32_t op, const Bytes &request) {
    const uint8_t *data = nullptr;
    size_t size = 0;
    auto status = hj_actor_call(actor, op, request.data(), request.size(), &data, &size);
    if (status &&
        std::string(reinterpret_cast<const char *>(data), size) == "no legal action at root")
      throw EmptySearchBranch();
    TORCH_CHECK(status == 0, std::string(reinterpret_cast<const char *>(data), size));
    return Bytes(data, data + size);
  }
  Frame query() {
    auto bytes = packet();
    put<uint32_t>(bytes, 0);
    return parse(call(10, bytes), root, pool);
  }
  Frame advance(const std::vector<uint16_t> &mass) {
    auto bytes = packet();
    put<uint32_t>(bytes, mass.size());
    raw(bytes, mass.data(), mass.size() * 2);
    return parse(call(6, bytes), root, pool);
  }
  void reset(const Position &p, uint32_t seed) {
    if (cancelled())
      throw std::runtime_error("search cancelled");
    auto request = packet();
    put(request, p.state->snapshot);
    put(request, seed);
    call(14, request);
    root.reset();
  }

public:
  SearchActor(void *actor, uint64_t game, Frozen policy, SearchService &service,
              uint32_t perspective, const std::function<bool()> &cancelled)
      : actor(actor), id(game | (uint64_t(1) << 63)), policy(std::move(policy)), service(service),
        perspective(perspective), cancelled(cancelled) {}
  ~SearchActor() {
    if (opened) {
      auto bytes = packet();
      const uint8_t *data;
      size_t size;
      hj_actor_call(actor, 5, bytes.data(), bytes.size(), &data, &size);
    }
  }
  bool fork(uint64_t source) {
    Bytes request;
    put(request, source);
    put(request, id);
    auto result = call(9, request);
    opened = !result.empty() && result[0] == 1;
    return opened;
  }
  Position position(uint32_t seed) {
    // 内核先打开公开根；只在首次到达这个公开后继时编码和估值。
    auto request = packet();
    put(request, seed);
    auto data = call(11, request);
    Reader reader(data);
    auto p = std::make_shared<PublicState>();
    p->snapshot = reader.get<uint64_t>();
    auto key = get_bytes(reader);
    reader.end();
    p->release = [actor = actor, id = id](uint64_t snapshot) {
      Bytes request;
      put(request, id);
      put(request, snapshot);
      const uint8_t *data;
      size_t size;
      hj_actor_call(actor, 15, request.data(), request.size(), &data, &size);
    };
    std::string public_key(key.begin(), key.end());
    auto found = cache.find(public_key);
    if (found != cache.end())
      return found->second;
    p->storage.set(key.capacity() * 3 + sizeof(PublicState));
    auto frame = query();
    Position out{p, std::move(public_key), frame.actor};
    out.terminal = frame.kind == 1;
    if (out.terminal) {
      out.value = frame.winner == 0 ? 0 : frame.winner == int32_t(perspective) ? 1 : -1;
    } else {
      p->query = frame.node;
      p->prediction = service.predict(policy, frame.node);
      out.value = p->prediction.value * (frame.actor == perspective ? 1 : -1);
      out.cutoff = p->prediction.mass.empty();
    }
    cache.emplace(out.key, out);
    return out;
  }
  mcts::Choice<Path> sample(const Position &p, uint32_t seed) {
    reset(p, seed);
    auto frame = query();
    double prior = 1;
    bool first = true;
    for (uint64_t nodes = 0;;) {
      if (cancelled())
        throw std::runtime_error("search cancelled");
      if (frame.kind == 3 || frame.kind == 1)
        break;
      if (frame.kind == 2) {
        frame = advance({});
        continue;
      }
      TORCH_CHECK(frame.kind == 0, "unexpected search sampling boundary");
      auto count = std::count(frame.node->mask.begin(), frame.node->mask.end(), 1);
      std::vector<uint16_t> mass;
      if (count > 1) {
        if (nodes >= 4096)
          throw SearchLimit();
        mass = first ? p.state->prediction.mass : service.predict(policy, frame.node).mass;
        ++nodes;
      }
      auto next = advance(mass);
      first = false;
      if (next.selected >= 0 && !mass.empty())
        prior *= std::exp(canonical_logps(mass)[next.selected]);
      frame = std::move(next);
    }
    auto data = call(12, packet());
    Reader reader(data);
    auto path = reader.array<uint32_t>(reader.get<uint32_t>());
    reader.end();
    TORCH_CHECK(!path.empty(), "search completed without command path");
    std::string key;
    for (auto index : path)
      key += std::to_string(index) + "/";
    return {std::move(path), std::move(key), std::max(1e-12, prior)};
  }
  Position step(const Position &p, const Path &path, uint32_t seed) {
    reset(p, seed);
    auto frame = query();
    for (uint64_t nodes = 0;;) {
      if (frame.kind == 3 || frame.kind == 1)
        break;
      if (frame.kind == 2) {
        frame = advance({});
        continue;
      }
      TORCH_CHECK(frame.kind == 0, "unexpected search replay boundary");
      auto count = std::count(frame.node->mask.begin(), frame.node->mask.end(), 1);
      std::vector<uint16_t> mass;
      if (count > 1) {
        if (nodes++ >= 4096)
          throw SearchLimit();
        auto level = frame.node->cursor.size();
        TORCH_CHECK(level < path.size() && path[level] < frame.node->mask.size() &&
                        frame.node->mask[path[level]],
                    "search path differs from public root");
        mass.resize(frame.node->mask.size());
        mass[path[level]] = c10::BFloat16(1.0f).x;
      }
      frame = advance(mass);
    }
    return position(seed ^ 0x9e3779b9u);
  }
};
} // namespace
SearchResult search_actor(void *actor, uint64_t game, uint32_t perspective,
                          const std::shared_ptr<Node> &node, const Frozen &policy,
                          SearchService &service, uint64_t simulations, uint64_t depth,
                          const std::function<bool()> &cancelled, SearchPlan *plan) {
  if (!simulations && plan)
    plan->clear();
  auto selected_policy =
      plan && plan->root == node->root->id && plan->actor == perspective ? plan->policy : policy;
  SearchResult result;
  result.version = selected_policy->version;
  auto base = service.predict(selected_policy, node);
  result.mass = base.mass;
  // 关闭搜索时直接返回冻结策略，不创建模拟 actor，也不生成搜索监督标签。
  if (!simulations)
    return result;
  auto condition = [&](const auto &paths) {
    std::vector<double> counts(base.mass.size(), 0);
    double total = 0;
    for (auto &[path, visits] : paths) {
      auto length = node->cursor.size();
      if (path.size() <= length ||
          !std::equal(node->cursor.begin(), node->cursor.end(), path.begin()))
        continue;
      auto index = path[length];
      if (index < counts.size() && node->mask[index]) {
        counts[index] += visits;
        total += visits;
      }
    }
    if (!total)
      return false;
    auto logps = canonical_logps(base.mass);
    result.target.resize(counts.size());
    for (size_t i = 0; i < counts.size(); ++i) {
      auto target = counts[i] / total;
      result.target[i] = c10::BFloat16(float(target)).x;
      result.mass[i] = c10::BFloat16(float(0.95 * target + 0.05 * std::exp(logps[i]))).x;
    }
    return true;
  };
  // 同一完整命令只搜索一次，再按实际选定的参数前缀条件化访问分布。
  if (plan && plan->root == node->root->id && plan->actor == perspective && condition(plan->paths))
    return result;
  SearchActor environment(actor, game, selected_policy, service, perspective, cancelled);
  if (!environment.fork(game))
    return result;
  auto seed = public_search_seed(*node);
  auto initial = environment.position(seed);
  TORCH_CHECK(initial.actor == perspective, "search decision owner differs");
  TORCH_CHECK(!initial.cutoff, "no legal action at root");
  mcts::Search<std::shared_ptr<PublicState>, Path> search(
      initial, [&](auto &p, uint32_t seed) { return environment.sample(p, seed); },
      [&](auto &p, auto &action, uint32_t seed) { return environment.step(p, action, seed); },
      depth, seed, cancelled);
  try {
    search.run(simulations);
  } catch (const SearchLimit &) {
    // 仅搜索命令展开达到工作上限时退回完整原策略；不截断实局、不制造训练标签。
    result.parameter_limit = true;
    return result;
  } catch (const EmptySearchBranch &) {
    // 假想分支的失败不能判定实局失败；仍由真实规则裁决原策略的合法尝试。
    result.empty_branch = true;
    return result;
  }
  decltype(SearchPlan::paths) paths;
  for (auto &edge : search.root.edges)
    paths.emplace_back(edge.action, edge.visits);
  TORCH_CHECK(condition(paths), "search did not visit a supported command");
  if (plan) {
    plan->clear();
    plan->root = node->root->id;
    plan->actor = perspective;
    plan->policy = selected_policy;
    plan->paths = std::move(paths);
    size_t bytes = plan->paths.capacity() * sizeof(plan->paths[0]);
    for (auto &[path, visits] : plan->paths)
      bytes += path.capacity() * 4;
    plan->storage.set(bytes);
  }
  result.simulations = search.root.visits;
  result.expanded = search.expanded;
  result.depth = search.deepest;
  result.opponent_nodes = search.opponent_nodes;
  return result;
}
void search_checks(const torch::Device &device) {
  {
    Node node;
    node.root = std::make_shared<Root>();
    Row globals;
    globals.ids[3] = 9007199254740993ll;
    globals.present = 4097;
    globals.integers = 7;
    globals.values[0] = 0x3f80;
    node.root->globals = input_value(globals);
    node.cursor = {2, 1};
    TORCH_CHECK(public_search_seed(node) == 32189891u, "public search seed differs from TS");
    node.root->game = 99;
    node.root->id = 123;
    TORCH_CHECK(public_search_seed(node) == 32189891u,
                "search seed depends on private scheduling ids");
  }
  {
    using P = mcts::Position<std::string>;
    auto position = [](std::string state) {
      return P{state, state, state == "reply" ? 2u : 1u,
               state == "loss"    ? -1.0
               : state == "win"   ? 1.0
               : state == "reply" ? 0.9
                                  : 0.0,
               state == "loss" || state == "win" || state == "safe"};
    };
    mcts::Search<std::string, std::string> tree(
        position("root"),
        [](auto &p, uint32_t seed) {
          std::string a = p.state == "root"   ? (seed & 1 ? "trap" : "safe")
                          : p.state == "prep" ? "next"
                          : seed & 1          ? "punish"
                                              : "blunder";
          return mcts::Choice<std::string>{a, a, 0.5};
        },
        [&](auto &p, auto &action, uint32_t) {
          return position(p.state == "root"    ? action == "trap" ? "prep" : "safe"
                          : p.state == "prep"  ? "reply"
                          : action == "punish" ? "loss"
                                               : "win");
        },
        16, 73127, [] { return false; });
    tree.run(512);
    uint64_t safe = 0, trap = 0;
    for (auto &edge : tree.root.edges)
      (edge.action == "safe" ? safe : trap) = edge.visits;
    TORCH_CHECK(safe == 491 && trap == 21 && tree.expanded == 6 && tree.deepest == 3 &&
                    tree.opponent_nodes == 1,
                "MCTS missed opponent reply or flipped a same-seat continuation");
  }
  {
    using P = mcts::Position<std::string>;
    auto position = [](std::string state) {
      return P{state, state, 1,
               state == "win"    ? 1.0
               : state == "loss" ? -1.0
                                 : 0.0,
               state != "root"};
    };
    mcts::Search<std::string, std::string> tree(
        position("root"),
        [](auto &, uint32_t seed) {
          std::string a = seed & 1 ? "risky" : "safe";
          return mcts::Choice<std::string>{a, a, 0.5};
        },
        [&](auto &, auto &a, uint32_t seed) {
          return position(a == "safe" ? "safe" : seed % 8 == 0 ? "win" : "loss");
        },
        8, 123, [] { return false; });
    tree.run(512);
    for (auto &edge : tree.root.edges)
      TORCH_CHECK(edge.visits == (edge.action == "safe" ? 490 : 22),
                  "stochastic MCTS differs from TS visits");
    TORCH_CHECK(tree.expanded == 4 && tree.deepest == 1 && tree.opponent_nodes == 0,
                "stochastic outcomes were merged or mistaken for opponent decisions");
  }
  torch::NoGradGuard guard;
  Network model(32);
  model->to(device);
  model->pack_weights();
  auto policy = std::make_shared<FrozenPolicy>(model, 17);
  {
    OpponentHistory history;
    history.limit = 2;
    history.interval = 1;
    history.publish(policy);
    auto original = policy->model->packed_weights.clone();
    model->packed_weights.add_(0.001);
    auto next = std::make_shared<FrozenPolicy>(model, 18);
    TORCH_CHECK(policy->cpu->revision != next->cpu->revision,
                "frozen model versions alias the same root-cache revision");
    history.publish(next);
    uint64_t historical = 0;
    for (uint64_t game = 1; game <= 100; ++game) {
      auto chosen = history.choose(game, 73127);
      historical += bool(chosen);
      TORCH_CHECK(!chosen || chosen->version == 17, "history sampled the latest model");
    }
    TORCH_CHECK(historical == 20 && torch::equal(original, policy->model->packed_weights),
                "history ratio or frozen weights changed");
    auto latest = std::make_shared<FrozenPolicy>(model, 19);
    history.publish(latest);
    std::stringstream buffer;
    torch::serialize::OutputArchive saved;
    history.save(saved, {{91, policy}});
    saved.save_to(buffer);
    buffer.seekg(0);
    torch::serialize::InputArchive loaded;
    loaded.load_from(buffer, torch::kCPU);
    OpponentHistory restored;
    restored.limit = 2;
    restored.interval = 1;
    auto bindings = restored.load(loaded, model, 19);
    restored.publish(latest);
    TORCH_CHECK(bindings.size() == 1 && bindings.at(91)->version == 17 &&
                    restored.snapshots().size() == 2 &&
                    torch::equal(original, bindings.at(91)->model->packed_weights),
                "checkpoint lost a pinned opponent after history pool eviction");
  }
  SearchService service(device, 32, 2);
  std::unique_ptr<void, decltype(&hj_actor_close)> actor(hj_actor_create(), hj_actor_close);
  auto call = [&](uint32_t op, const Bytes &request) {
    const uint8_t *data;
    size_t size;
    auto status = hj_actor_call(actor.get(), op, request.data(), request.size(), &data, &size);
    TORCH_CHECK(status == 0, std::string(reinterpret_cast<const char *>(data), size));
    return Bytes(data, data + size);
  };
  Bytes id;
  put<uint64_t>(id, 1);
  auto request = id;
  put<uint32_t>(request, 73127);
  put<uint32_t>(request, 1);
  put<uint32_t>(request, 0);
  call(0, request);
  request = id;
  put<uint32_t>(request, 0);
  InputPool inputs;
  std::shared_ptr<Root> root;
  auto frame = parse(call(1, request), root, inputs);
  TORCH_CHECK(frame.kind == 0);
  auto predicted = service.predict(policy, frame.node);
  auto expected = policy->model->forward({frame.node}).second.item<double>();
  TORCH_CHECK(std::abs(predicted.value - expected) < 0.02,
              "CPU search value differs from BF16 model value");
  auto before = call(2, id);
  SearchPlan plan;
  auto result = search_actor(
      actor.get(), 1, frame.actor, frame.node, policy, service, 4, 8, [] { return false; }, &plan);
  auto cached = search_actor(
      actor.get(), 1, frame.actor, frame.node, policy, service, 4, 8, [] { return false; }, &plan);
  TORCH_CHECK(cached.simulations == 0 && cached.mass == result.mass &&
                  cached.target == result.target,
              "parameter continuation did not reuse the complete-command visits");
  auto queries = service.queries.load();
  auto direct = search_actor(
      actor.get(), 1, frame.actor, frame.node, policy, service, 0, 8, [] { return false; }, &plan);
  TORCH_CHECK(direct.mass == predicted.mass && direct.target.empty() && direct.simulations == 0 &&
                  direct.expanded == 0 && direct.depth == 0 && !plan.policy &&
                  service.queries.load() == queries + 1 && call(2, id) == before,
              "disabled MCTS simulated, retained cached visits or changed the raw policy");
  plan.clear();
  auto restarted = search_actor(
      actor.get(), 1, frame.actor, frame.node, policy, service, 4, 8, [] { return false; }, &plan);
  TORCH_CHECK(restarted.mass == result.mass && restarted.target == result.target,
              "cold search after checkpoint changed the behavior distribution");
  TORCH_CHECK(call(2, id) == before, "search changed formal state, frontier or RNG");
  TORCH_CHECK(result.simulations == 4 && result.expanded > 1 && result.depth > 0 &&
                  result.mass.size() == frame.node->mask.size() && result.version == 17,
              "native MCTS did not simulate complete commands");
  auto logps = canonical_logps(result.mass);
  double sum = 0;
  for (size_t i = 0; i < logps.size(); ++i) {
    sum += std::exp(logps[i]);
    TORCH_CHECK(!frame.node->mask[i] || std::isfinite(logps[i]),
                "search removed a supported policy action");
  }
  TORCH_CHECK(std::abs(sum - 1) < 1e-9, "search behavior mass is not normalized");
  frame.node->search_target = result.target;
  GraphWriter saved;
  Bytes graph_bytes, reference;
  saved.ref(reference, frame.node);
  saved.write(graph_bytes);
  raw(graph_bytes, reference.data(), reference.size());
  auto target_bytes = saved.search_data();
  Reader reader(graph_bytes);
  GraphReader restored(reader, inputs);
  restored.search_data(
      torch::from_blob(target_bytes.data(), {int64_t(target_bytes.size())}, torch::kUInt8));
  auto node = restored.ref(reader);
  reader.end();
  TORCH_CHECK(node->search_target == result.target && node->cursor == frame.node->cursor,
              "checkpoint lost MCTS visits or parameter prefix");
  std::cout << "public command MCTS, opponent reply, stochastic backup and formal RNG isolation: "
               "passed\n";
}
} // namespace hj
