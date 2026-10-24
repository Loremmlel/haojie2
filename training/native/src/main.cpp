#include "runtime.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <fstream>
#include <iostream>
#include <map>

int main(int argc, char **argv) {
  try {
    std::string command = argc > 1 ? argv[1] : "help";
    std::map<std::string, std::string> options;
    for (int i = 2; i < argc; ++i) {
      std::string key = argv[i];
      if (key.rfind("--", 0) != 0)
        throw std::runtime_error("expected option");
      if (key == "--shrine") {
        options[key] = "1";
        continue;
      }
      if (++i == argc)
        throw std::runtime_error("missing option value");
      options[key] = argv[i];
    }
    auto text = [&](std::string key, std::string fallback = "") {
      auto it = options.find(key);
      return it == options.end() ? fallback : it->second;
    };
    auto number = [&](std::string key, uint64_t fallback) {
      auto v = text(key);
      if (v.empty())
        return fallback;
      size_t end = 0;
      auto n = std::stoull(v, &end);
      if (end != v.size() || v[0] == '-')
        throw std::runtime_error("invalid integer " + key);
      return uint64_t(n);
    };
    auto directory = std::filesystem::path(text("--directory"));
    if (command == "status") {
      auto status = std::filesystem::exists(directory / "window.json") ? directory / "window.json"
                                                                       : directory / "status.json";
      std::ifstream in(status, std::ios::binary);
      if (!in)
        in.open(directory / "saved.json", std::ios::binary);
      if (!in)
        throw std::runtime_error("no saved status");
      std::cout << in.rdbuf();
      return 0;
    }
    if (command == "resize" || command == "budget" || command == "save" || command == "stop") {
      if (directory.empty() || !std::filesystem::exists(directory))
        throw std::runtime_error("run directory required");
      auto payload = command == "resize"   ? std::to_string(number("--concurrency", 64))
                     : command == "budget" ? std::to_string(number("--memory-mb", 4096))
                                           : "1";
      hj::atomic_text(directory / command, payload);
      return 0;
    }
    if (command == "help") {
      std::cout << "haojie-train run --directory PATH [--resume CHECKPOINT] [--device cuda|cpu] "
                   "[--concurrency 64] [--workers 8] [--batch 512] [--chunk 16] [--updates N] "
                   "[--seconds N]\nhaojie-train status|resize|budget|save|stop --directory "
                   "PATH\nhaojie-train check [--device cpu|cuda]\n";
      return 0;
    }
    at::set_num_threads(int(number("--threads", 1)));
    at::set_num_interop_threads(1);
    c10::set_default_dtype(c10::scalarTypeToTypeMeta(torch::kBFloat16));
    torch::manual_seed(number("--seed", 1));
    if (command == "check") {
      hj::model_checks(torch::Device(text("--device", "cpu")));
      hj::runtime_checks();
      return 0;
    }
    if (command != "run" || directory.empty())
      throw std::runtime_error("use run --directory PATH");
    hj::Settings s;
    for (auto &option : options)
      s.supplied.insert(option.first);
    s.directory = directory;
    s.resume = text("--resume");
    s.device = text("--device", "cuda");
    s.concurrency = number("--concurrency", 64);
    s.workers = number("--workers", 8);
    s.seed = number("--seed", 1);
    s.chunk = number("--chunk", 16);
    s.batch = number("--batch", 512);
    s.inference_batch = number("--inference-batch", 256);
    s.width = number("--width", hj::DefaultWidth);
    s.publish = number("--publish", 8);
    s.memory_mb = number("--memory-mb", 4096);
    s.updates = number("--updates", 0);
    s.seconds = std::stod(text("--seconds", "0"));
    s.save_seconds = std::stod(text("--save-seconds", "300"));
    s.shrine = options.count("--shrine");
    s.samples = number("--samples", 0);
    if (!s.workers || !s.chunk || !s.batch || !s.inference_batch || !s.width || !s.publish ||
        s.memory_mb < 128 || s.seconds < 0 || s.save_seconds < 0)
      throw std::runtime_error("invalid training budget");
    hj::run(s);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
