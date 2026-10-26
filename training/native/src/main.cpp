#include "actor/search.h"
#include "host/hardware.h"
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
             "haojie-train workset --directory PATH [--games 16] [--game-samples 256] "
             "[--concurrency 12] [--replay REFERENCE_DIRECTORY]\n";
      return 0;
    }
    at::set_num_threads(int(number("--threads", 1)));
    at::set_num_interop_threads(1);
    at::globalContext().setAllowBF16ReductionCuBLAS(false);
    c10::set_default_dtype(c10::scalarTypeToTypeMeta(torch::kBFloat16));
    torch::manual_seed(number("--seed", 1));
    if (command == "evaluate") {
      hj::evaluate(text("--checkpoint"), text("--device", "cuda"),
                   options.count("--search-simulations")
                       ? std::optional<uint64_t>(number("--search-simulations", 0))
                       : std::nullopt);
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
      hj::learning_checks(torch::Device(text("--device", "cpu")));
      hj::runtime_checks();
      hj::evaluation_checks(torch::Device(text("--device", "cpu")));
      hj::search_checks(torch::Device(text("--device", "cpu")));
      return 0;
    }
    if ((command != "run" && command != "workset" && command != "diagnose" &&
         command != "configure") ||
        (command != "configure" && directory.empty()))
      throw std::runtime_error("use run --directory PATH");
    hj::Settings s;
    if (command == "diagnose") {
      s.diagnostic = text("--mode", "actor");
      s.warmup = std::stod(text("--warmup", "5"));
      s.root_slots = number("--root-slots", 2);
      s.defer_learning = number("--defer-learning", 0) != 0;
      if (s.diagnostic != "actor" && s.diagnostic != "learner" && s.diagnostic != "mixed" &&
          s.diagnostic != "inference" && s.diagnostic != "service" && s.diagnostic != "gemm" &&
          s.diagnostic != "episodes")
        throw std::runtime_error(
            "diagnose mode must be actor, learner, mixed, inference, service, gemm or episodes");
      if (s.warmup < 0)
        throw std::runtime_error("negative warmup");
      if (s.root_slots < 1 || s.root_slots > 2)
        throw std::runtime_error("diagnostic root slots must be 1 or 2");
    }
    for (auto &option : options)
      s.supplied.insert(option.first);
    s.directory = directory;
    s.resume = text("--resume");
    s.profile = text("--profile");
    if (command == "workset" || (s.diagnostic == "actor" && options.count("--games"))) {
      s.workset_games = number("--games", 16);
      s.game_samples = number("--game-samples", 256);
      s.replay = text("--replay");
      if (!s.resume.empty() || !s.workset_games || !s.game_samples || !text("--samples").empty())
        throw std::runtime_error("workset requires a fresh session and per-game sample quota");
    }
    s.device = text("--device", command == "run" || command == "configure" ? "auto" : "cuda");
    s.threads = number("--threads", 1);
    s.concurrency = number("--concurrency", 12);
    s.workers = number("--workers", 8);
    s.seed = number("--seed", 1);
    s.chunk = number("--chunk", 16);
    s.batch = number("--batch", 512);
    s.inference_batch = number("--inference-batch", 64);
    s.width = number("--width", hj::DefaultWidth);
    s.publish = number("--publish", 8);
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
    if (!s.workers || (!s.chunk && !s.terminal_training()) || !s.batch || !s.inference_batch ||
        !s.width || !s.publish || !s.max_game_steps || !s.search_depth || !s.history_size ||
        !s.history_interval || s.history_percent > 100 || s.memory_mb < 128 || s.seconds < 0 ||
        s.save_seconds < 0)
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
