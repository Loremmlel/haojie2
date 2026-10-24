#pragma once
#include "data.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

namespace hj {
// 固定真实工作集诊断：所有并发档位处理同一组对局及同一份实际行为分布。
// 网络照常前向、轨迹照常学习；回放只冻结采样数值，消除批形状舍入造成的路径分叉。
struct Workset {
  struct Record {
    uint64_t steps = 0;
    Bytes tape, reference;
    size_t offset = 0;
    bool terminal = false, replaying = false;
    std::atomic<size_t> bytes{0};
  };
  uint64_t games, quota;
  std::filesystem::path replay;
  std::map<uint64_t, std::shared_ptr<Record>> records;
  std::set<uint64_t> done;
  Workset(uint64_t games, uint64_t quota, std::filesystem::path replay)
      : games(games), quota(quota), replay(std::move(replay)) {}
  bool complete() const {
    return done.size() == games;
  }
  void opened(uint64_t id) {
    auto &entry = records[id];
    entry = std::make_shared<Record>();
    auto &r = *entry;
    r.replaying = !replay.empty();
    if (!replay.empty()) {
      std::ifstream in(replay / ("game-" + std::to_string(id) + ".bin"), std::ios::binary);
      if (!in)
        throw std::runtime_error("cannot read reference behavior tape");
      r.reference.assign(std::istreambuf_iterator<char>(in), {});
    }
    r.bytes = r.reference.capacity();
  }
  void transition(uint64_t id) {
    if (++records.at(id)->steps >= quota)
      done.insert(id);
  }
  void terminal(uint64_t id) {
    records.at(id)->terminal = true;
    done.insert(id);
  }
  static std::vector<uint16_t> probabilities(Record &r, std::vector<uint16_t> generated,
                                             const std::vector<uint8_t> &mask) {
    if (r.replaying) {
      if (r.reference.size() - r.offset < 4)
        throw std::runtime_error("reference behavior tape ended early");
      uint32_t count;
      std::memcpy(&count, r.reference.data() + r.offset, 4);
      r.offset += 4;
      if (count != generated.size() || count > (r.reference.size() - r.offset) / 2)
        throw std::runtime_error("reference behavior support differs");
      std::memcpy(generated.data(), r.reference.data() + r.offset, count * 2);
      r.offset += count * 2;
      for (size_t i = 0; i < mask.size(); ++i)
        if (!mask[i] && generated[i])
          throw std::runtime_error("reference behavior chose an excluded action");
    }
    put<uint32_t>(r.tape, generated.size());
    raw(r.tape, generated.data(), generated.size() * 2);
    r.bytes = r.tape.capacity() + r.reference.capacity();
    return generated;
  }
};
} // namespace hj
