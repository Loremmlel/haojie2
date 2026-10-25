#include "engine.h"
#include "host/hardware.h"
#include "runtime.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

namespace hj {
// 独立测评进程只加载已提交权重，不创建优化器、训练轨迹或发布新版本。
void evaluate(const std::filesystem::path &checkpoint, const std::string &requested) {
  torch::NoGradGuard guard;
  torch::Device device(hardware(requested).device);
  Network model(DefaultWidth);
  model->to(device);
  model->pack_weights();
  uint64_t version;
  {
    torch::serialize::InputArchive archive;
    std::ifstream file(checkpoint, std::ios::binary);
    TORCH_CHECK(file, "evaluation checkpoint is missing");
    archive.load_from(file, torch::kCPU);
    T state, weights;
    archive.read("state", state);
    TORCH_CHECK(state.scalar_type() == torch::kUInt8 && state.dim() == 1,
                "invalid checkpoint metadata");
    Reader r(state.data_ptr<uint8_t>(), state.numel());
    TORCH_CHECK(r.get<uint64_t>() == 0x484a4e4154495633ull, "unsupported checkpoint");
    // 与训练恢复一致，构建身份不作为权重加载条件。
    get_bytes(r);
    TORCH_CHECK(r.get<uint64_t>() == DefaultWidth, "checkpoint model width differs");
    for (int i = 0; i < 6; ++i)
      r.get<uint64_t>();
    version = r.get<uint64_t>();
    archive.read("weights", weights);
    TORCH_CHECK(weights.scalar_type() == torch::kBFloat16 &&
                    weights.sizes() == model->packed_weights.sizes() &&
                    torch::isfinite(weights).all().item<bool>(),
                "invalid BF16 weights");
    model->packed_weights.copy_(weights.to(device));
  }
  model->eval();
  auto weights = CpuPolicy::snapshot(model, version);
  InferenceEngine inference(model, device, [] {}, 1, 1);
  CpuPolicy policy;
  InputPool inputs;
  std::shared_ptr<Root> root;
  std::unique_ptr<void, decltype(&hj_actor_close)> handle(hj_actor_create(), hj_actor_close);
  TORCH_CHECK(handle, "cannot create evaluation actor");
  auto call = [&](uint32_t op, const Bytes &request) {
    const uint8_t *data = nullptr;
    size_t size = 0;
    int status = hj_actor_call(handle.get(), op, request.data(), request.size(), &data, &size);
    TORCH_CHECK(status == 0, std::string(reinterpret_cast<const char *>(data), size));
    return Bytes(data, data + size);
  };
  Bytes id;
  put<uint64_t>(id, 1);
  bool started = false;
  std::cout << "{\"updates\":" << version << ",\"identity\":" << hj_identity() << "}" << std::endl;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "quit")
      break;
    if (line.rfind("start ", 0) == 0) {
      uint32_t seed, side, shrine;
      std::istringstream in(line.substr(6));
      TORCH_CHECK(bool(in >> seed >> side >> shrine), "invalid evaluation start");
      if (started)
        call(5, id);
      root.reset();
      policy.clear();
      auto request = id;
      put(request, seed);
      put(request, side);
      put(request, shrine);
      call(0, request);
      started = true;
    } else if (line == "advance") {
      TORCH_CHECK(started, "evaluation game not started");
      auto request = id;
      put<uint32_t>(request, 0);
      // 每次返回根边界；不会替真实 AI 越过其回合或反应窗口。
      for (int slice = 0; slice < 64; ++slice) {
        auto frame = parse(call(6, request), root, inputs);
        if (frame.kind == 1 || frame.kind == 3)
          break;
        request = id;
        std::vector<uint16_t> mass;
        if (frame.kind == 0) {
          if (!model->cache_valid(*root)) {
            auto flight = inference.submit(0, {frame.node}, version);
            while (!flight->complete.load())
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            inference.finish(flight);
          }
          mass = policy.score(*frame.node, weights);
        }
        put<uint32_t>(request, uint32_t(mass.size()));
        raw(request, mass.data(), mass.size() * sizeof(uint16_t));
      }
    } else if (line.rfind("command ", 0) == 0) {
      TORCH_CHECK(started, "evaluation game not started");
      auto request = id;
      raw(request, line.data() + 8, line.size() - 8);
      call(8, request);
    } else {
      throw std::runtime_error("unknown evaluation request");
    }
    auto state = call(7, id);
    std::cout.write(reinterpret_cast<const char *>(state.data()), state.size());
    std::cout << std::endl;
  }
}
void evaluation_checks(const torch::Device &device) {
  torch::NoGradGuard guard;
  // 只生成初始化权重夹具，检查加载与推理协议，不创建优化器或学习转换。
  auto path = std::filesystem::temp_directory_path() /
              ("haojie-evaluation-check-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".pt");
  auto old_in = std::cin.rdbuf();
  auto old_out = std::cout.rdbuf();
  try {
    Network fixture(DefaultWidth);
    fixture->pack_weights();
    Bytes metadata;
    put<uint64_t>(metadata, 0x484a4e4154495633ull);
    // 模拟旧构建标记，验证加载不再依赖当前源码指纹。
    std::string identity = "legacy-native-build";
    put_bytes(metadata, Bytes(identity.begin(), identity.end()));
    for (uint64_t n : std::array<uint64_t, 8>{DefaultWidth, 1, 16, 512, 1, 40, 40, 42})
      put(metadata, n);
    torch::serialize::OutputArchive archive;
    archive.write("weights", fixture->packed_weights);
    archive.write(
        "state",
        torch::from_blob(metadata.data(), {int64_t(metadata.size())}, torch::kUInt8).clone());
    archive.save_to(path.string());
    std::istringstream input("start 73127 1 0\nadvance\nstart 73127 2 1\nadvance\nquit\n");
    std::ostringstream output;
    std::cin.rdbuf(input.rdbuf());
    std::cout.rdbuf(output.rdbuf());
    evaluate(path, device.str());
    std::cin.rdbuf(old_in);
    std::cout.rdbuf(old_out);
    TORCH_CHECK(output.str().find("\"updates\":42") != std::string::npos &&
                    output.str().find("\"observation\"") != std::string::npos,
                "evaluation checkpoint or public response missing");
    std::filesystem::remove(path);
  } catch (...) {
    std::cin.rdbuf(old_in);
    std::cout.rdbuf(old_out);
    std::filesystem::remove(path);
    throw;
  }
  std::cout << "native frozen evaluation checkpoint, BF16 inference and root boundary: passed\n";
}
} // namespace hj
