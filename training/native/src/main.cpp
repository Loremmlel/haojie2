#include "actor/search.h"
#include "host/hardware.h"
#include "runtime.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/util/signal_handler.h>
#include <fstream>
#include <iostream>
#include <map>

namespace {
int command_main(int argc, char **argv) {
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
    const std::set<std::string> supported{"--directory",
                                          "--resume",
                                          "--device",
                                          "--threads",
                                          "--concurrency",
                                          "--workers",
                                          "--seed",
                                          "--batch",
                                          "--inference-batch",
                                          "--width",
                                          "--memory-mb",
                                          "--max-game-steps",
                                          "--search-simulations",
                                          "--search-depth",
                                          "--history-size",
                                          "--history-interval",
                                          "--history-percent",
                                          "--updates",
                                          "--seconds",
                                          "--save-seconds",
                                          "--shrine",
                                          "--samples",
                                          "--checkpoint"};
    for (const auto &[key, value] : options)
      if (!supported.count(key) && !(command == "evaluate" && key == "--temperature"))
        throw std::runtime_error("unknown option " + key);
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
      auto payload = command == "resize"   ? std::to_string(number("--concurrency", 12))
                     : command == "budget" ? std::to_string(number("--memory-mb", 4096))
                                           : "1";
      hj::atomic_text(directory / command, payload);
      return 0;
    }
    if (command == "help") {
      std::cout
          << "haojie-train run --directory PATH [--resume CHECKPOINT] [--device auto|cuda|cpu] "
             "[--concurrency N] [--workers N] [--batch N] [--search-simulations 0] "
             "[--search-depth 64] [--history-percent 20] [--updates N] "
             "[--seconds N] [--max-game-steps N]\nhaojie-train status|resize|budget|save|stop "
             "--directory "
             "PATH\nhaojie-train check [--device cpu|cuda]\n"
             "haojie-train configure [--device auto|cuda|cpu] [--resume CHECKPOINT]\n"
             "haojie-train evaluate --checkpoint PATH [--device cpu|cuda] "
             "[--search-simulations N] [--temperature 0.5]\n";
      return 0;
    }
    at::set_num_threads(int(number("--threads", 1)));
    at::set_num_interop_threads(1);
    at::globalContext().setAllowBF16ReductionCuBLAS(false);
    c10::set_default_dtype(c10::scalarTypeToTypeMeta(torch::kBFloat16));
    torch::manual_seed(number("--seed", 1));
    if (command == "evaluate") {
      auto temperature_text = text("--temperature", "0.5");
      size_t end = 0;
      double temperature = std::stod(temperature_text, &end);
      if (end != temperature_text.size())
        throw std::runtime_error("invalid evaluation temperature");
      hj::evaluate(text("--checkpoint"), text("--device", "cuda"),
                   options.count("--search-simulations")
                       ? std::optional<uint64_t>(number("--search-simulations", 0))
                       : std::nullopt,
                   temperature);
      return 0;
    }
    if (command == "check-search") {
      hj::search_checks(torch::Device(text("--device", "cpu")));
      return 0;
    }
    if (command == "check") {
      hj::hardware_checks();
      hj::model_checks(torch::Device(text("--device", "cpu")));
      hj::inference_checks(torch::Device(text("--device", "cpu")));
      hj::runtime_checks();
      hj::evaluation_checks(torch::Device(text("--device", "cpu")));
      hj::search_checks(torch::Device(text("--device", "cpu")));
      return 0;
    }
    if ((command != "run" && command != "configure") ||
        (command != "configure" && directory.empty()))
      throw std::runtime_error("use run --directory PATH");
    hj::Settings s;
    for (auto &option : options)
      s.supplied.insert(option.first);
    s.directory = directory;
    s.resume = text("--resume");
    s.device = text("--device", "auto");
    s.threads = number("--threads", 1);
    s.concurrency = number("--concurrency", 12);
    s.workers = number("--workers", 8);
    s.seed = number("--seed", 1);
    s.batch = number("--batch", 512);
    s.inference_batch = number("--inference-batch", 64);
    s.width = number("--width", hj::DefaultWidth);
    s.memory_mb = number("--memory-mb", 4096);
    s.max_game_steps = number("--max-game-steps", 262144);
    s.search_simulations = number("--search-simulations", 0);
    s.search_depth = number("--search-depth", 64);
    s.history_size = number("--history-size", 8);
    s.history_interval = number("--history-interval", 64);
    s.history_percent = number("--history-percent", 20);
    s.updates = number("--updates", 0);
    s.seconds = std::stod(text("--seconds", "0"));
    s.save_seconds = std::stod(text("--save-seconds", "300"));
    s.shrine = options.count("--shrine");
    s.samples = number("--samples", 0);
    if (!s.workers || !s.batch || !s.inference_batch || !s.width || !s.publish ||
        !s.max_game_steps || !s.search_depth || !s.history_size || !s.history_interval ||
        s.history_percent > 100 || s.memory_mb < 128 || s.seconds < 0 || s.save_seconds < 0)
      throw std::runtime_error("invalid training budget");
    if (command == "configure")
      std::cout << hj::describe_configuration(s);
    else
      hj::run(s);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
} // namespace

int main(int argc, char **argv) {
  // 原生崩溃不会走 C++ 异常捕获；文件输出即时刷新，Linux 复用 LibTorch 的致命信号栈。
  std::cout << std::unitbuf;
#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
  c10::FatalSignalHandler::getInstance().setPrintStackTracesOnFatalSignal(true);
#endif
  auto code = command_main(argc, argv);
  try {
    hj::release_device_kernels();
    if (argc > 1 && std::string(argv[1]) == "run")
      std::cerr << "Native shutdown complete.\n";
  } catch (const std::exception &error) {
    std::cerr << "Native shutdown failed: " << error.what() << '\n';
    return 1;
  }
  return code;
}
