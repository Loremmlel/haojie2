#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hj::mcts {
// 与 src/ai/training/mcts.ts 保持同一 PUCT、渐进扩宽及随机后继流程。
struct Random {
  uint32_t state;
  explicit Random(uint32_t seed) : state(seed ? seed : 1) {}
  uint32_t next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  }
};
template <class State> struct Position {
  State state;
  std::string key;
  uint32_t actor = 1;
  double value = 0;
  bool terminal = false;
  bool cutoff = false;
};
template <class Action> struct Choice {
  Action action;
  std::string key;
  double prior = 1;
};
template <class State, class Action> class Search {
public:
  struct Node;
  struct Edge : Choice<Action> {
    uint64_t visits = 0;
    double total = 0;
    std::map<std::string, std::shared_ptr<Node>> outcomes;
  };
  struct Node {
    Position<State> position;
    uint64_t visits = 0;
    std::vector<Edge> edges;
  };
  using Sample = std::function<Choice<Action>(const Position<State> &, uint32_t)>;
  using Step = std::function<Position<State>(const Position<State> &, const Action &, uint32_t)>;
  Node root;
  uint64_t expanded = 1, deepest = 0, opponent_nodes = 0;
  Sample sample;
  Step step;
  Random random;
  std::function<bool()> cancelled;
  uint64_t depth;
  Search(Position<State> p, Sample sample, Step step, uint64_t depth, uint32_t seed,
         std::function<bool()> cancelled)
      : root{std::move(p)}, sample(std::move(sample)), step(std::move(step)), random(seed),
        cancelled(std::move(cancelled)), depth(depth) {}
  double run(Node &node, uint64_t level) {
    deepest = std::max(deepest, level);
    if (cancelled())
      throw std::runtime_error("search cancelled");
    if (node.position.terminal || node.position.cutoff || level == depth)
      return node.position.value;
    if (node.edges.size() < uint64_t(std::sqrt(double(node.visits))) + 1) {
      auto choice = sample(node.position, random.next());
      if (std::none_of(node.edges.begin(), node.edges.end(),
                       [&](auto &e) { return e.key == choice.key; }))
        node.edges.push_back(Edge{std::move(choice)});
    }
    double sum = 0;
    for (auto &edge : node.edges)
      sum += edge.prior;
    double sign = node.position.actor == root.position.actor ? 1 : -1;
    auto score = [&](const Edge &edge) {
      return sign * (edge.visits ? edge.total / edge.visits : node.position.value) +
             1.5 * edge.prior * std::sqrt(double(node.visits + 1)) / (sum * (edge.visits + 1));
    };
    auto selected = node.edges.begin();
    for (auto edge = node.edges.begin(); edge != node.edges.end(); ++edge)
      if (score(*edge) > score(*selected))
        selected = edge;
    auto next = step(node.position, selected->action, random.next());
    auto &child = selected->outcomes[next.key];
    double value;
    if (child)
      value = run(*child, level + 1);
    else {
      child = std::make_shared<Node>(Node{std::move(next)});
      ++expanded;
      opponent_nodes += !child->position.terminal && !child->position.cutoff &&
                        child->position.actor != root.position.actor;
      deepest = std::max(deepest, level + 1);
      value = child->position.value;
    }
    ++selected->visits;
    selected->total += value;
    ++node.visits;
    return value;
  }
  void run(uint64_t simulations) {
    if (!root.position.terminal)
      for (uint64_t i = 0; i < simulations; ++i)
        run(root, 0);
  }
};
} // namespace hj::mcts
