"""重放有序真实请求；比较 v3、单向参考和含冷启动的 KV 缓存，不训练参数。"""

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MODES = ("v3", "active_reference", "active_cached", "readout_reference", "readout_cached")


def file_hash(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def stream_key(record):
    return tuple(record[key] for key in ("slot", "generation", "task", "model"))


def root_key(record):
    return (*stream_key(record), record["command_index"], record["actor"])


def run_case(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "training"))
    importlib.import_module("haojie_training")
    import torch
    from model import CausalMemoryPolicy

    from haojie_training.benchmark import tensor_hash
    from haojie_training.console.config.resources import hardware
    from haojie_training.model import ModelConfig, PolicyValueNet
    from haojie_training.native.execution import PolicyInference
    from haojie_training.runtime import autocast, resolve_device, synchronize
    from scripts.training.benchmarks.cycle import hardware_identity

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    trace = torch.load(args.trace, map_location="cpu", weights_only=True)
    if trace["format"] != "haojie-causal-memory-trace-v1" or not trace["records"]:
        raise ValueError("需要未抽样、有序的公开请求轨迹")
    torch.manual_seed(trace["seed"])
    model = PolicyValueNet(ModelConfig(**trace["model_config"])).eval()
    initial_hash = tensor_hash(model.state_dict())
    if initial_hash != trace["initial_model_sha256"]:
        raise ValueError("重放初始化权重与采集时不同")
    device = resolve_device(args.device)
    machine = hardware_identity(hardware())
    model.to(device)
    records = trace["records"]
    scope = args.mode.split("_")[0] if args.mode != "v3" else None

    def runners():
        baseline = PolicyInference(model, device, args.precision) if scope is None else None
        reference = CausalMemoryPolicy(model, query_scope=scope) if scope else None
        return baseline, reference, {}

    baseline, reference, caches = runners()

    def invoke(record):
        row = record["input"]
        if baseline is not None:
            return baseline([row])[0]
        if args.mode.endswith("reference"):
            output = reference.forward_reference(row, record["base_entities"], policy_only=True)
        else:
            key = stream_key(record)
            if key not in caches:
                caches[key] = CausalMemoryPolicy(model, query_scope=scope)
            output = caches[key].forward_cached(
                row, record["base_entities"], root_key(record), policy_only=True
            )
        return output[0]

    # 预热算子后重建全部缓存；正式计时包含每一根的首次编码和所有后续追加。
    with torch.inference_mode(), autocast(device, args.precision):
        for record in records[: min(16, len(records))]:
            invoke(record)
        synchronize(device)
        baseline, reference, caches = runners()
        outputs, request_seconds = [], []
        started = time.perf_counter()
        for record in records:
            before = time.perf_counter()
            logits = invoke(record).cpu()
            synchronize(device)
            request_seconds.append(time.perf_counter() - before)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("重放输出非有限")
            outputs.append(logits.clone())
        elapsed = time.perf_counter() - started

    digest = hashlib.sha256()
    for logits in outputs:
        digest.update(logits.float().numpy().tobytes())
    if tensor_hash(model.state_dict()) != initial_hash:
        raise AssertionError("性能测量期间修改了模型权重")
    stats = []
    for key, runner in caches.items():
        value = runner.stats
        stats.append({"stream": key, "stats": dict(value() if callable(value) else value)})
    first, repeat, seen = [], [], set()
    for record, seconds in zip(records, request_seconds):
        key = root_key(record)
        (repeat if key in seen else first).append(seconds)
        seen.add(key)
    result = {
        "mode": args.mode,
        "trace": str(args.trace.resolve()),
        "trace_sha256": file_hash(args.trace),
        "initial_model_sha256": initial_hash,
        "source_sha256": {
            "experiment": file_hash(Path(__file__).with_name("model.py")),
            "baseline": file_hash(ROOT / "training/haojie_training/model.py"),
            "benchmark": file_hash(Path(__file__)),
        },
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "device": str(device),
        "precision": args.precision,
        "threads": args.threads,
        "hardware_identity": machine,
        "requests": len(records),
        "streams": len({stream_key(r) for r in records}),
        "roots": len(seen),
        "seconds": elapsed,
        "requests_per_second": len(records) / elapsed,
        "request_median_ms": statistics.median(request_seconds) * 1000,
        "root_first_seconds": sum(first),
        "root_repeat_seconds": sum(repeat),
        "first_requests": len(first),
        "repeat_requests": len(repeat),
        "outputs_sha256": digest.hexdigest(),
        "cache_stats": stats,
        "weights_unchanged": True,
        "training_updates": 0,
    }
    if args.outputs:
        torch.save(outputs, args.outputs)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("mode", "requests", "roots", "seconds")}), flush=True)


def summarize(cases):
    result = {}
    by_trace = {}
    for case in cases:
        by_trace.setdefault(case["trace_sha256"], []).append(case)
    for identity, rows in by_trace.items():
        for key in ("initial_model_sha256", "requests", "roots", "threads", "device", "precision"):
            if len({row[key] for row in rows}) != 1:
                raise AssertionError(f"配对工作量或配置不一致：{key}")
        for key in ("source_sha256", "hardware_identity"):
            if len({json.dumps(row[key], sort_keys=True) for row in rows}) != 1:
                raise AssertionError(f"配对源码或硬件不一致：{key}")
        modes = {}
        for mode in MODES:
            selected = [row for row in rows if row["mode"] == mode]
            if not selected:
                continue
            seconds = [row["seconds"] for row in selected]
            if len({row["outputs_sha256"] for row in selected}) != 1:
                raise AssertionError(f"相同模式重复执行输出不稳定：{mode}")
            modes[mode] = {"seconds": seconds, "median_seconds": statistics.median(seconds)}
        if "v3" in modes:
            before = modes["v3"]["median_seconds"]
            for value in modes.values():
                value["time_reduction_vs_v3_percent"] = 100 * (1 - value["median_seconds"] / before)
                value["throughput_vs_v3"] = before / value["median_seconds"]
        for scope in ("active", "readout"):
            if f"{scope}_cached" in modes and f"{scope}_reference" in modes:
                modes[f"{scope}_cached"]["cache_speedup"] = (
                    modes[f"{scope}_reference"]["median_seconds"]
                    / modes[f"{scope}_cached"]["median_seconds"]
                )
        result[identity] = {
            "trace": rows[0]["trace"],
            "requests": rows[0]["requests"],
            "roots": rows[0]["roots"],
            "modes": modes,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--outputs", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.trials) < 1:
        parser.error("线程数和配对次数必须为正数")
    if args.mode:
        if len(args.trace) != 1:
            parser.error("单进程测量需要一条轨迹")
        args.trace = args.trace[0]
        run_case(args)
        return
    cases = []
    directory = args.output.with_suffix("")
    directory.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for trace in args.trace:
        for trial in range(args.trials):
            order = args.modes if trial % 2 == 0 else list(reversed(args.modes))
            for mode in order:
                target = directory / f"{trace.stem}-{mode}-{trial + 1}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--trace",
                    str(trace.resolve()),
                    "--output",
                    str(target.resolve()),
                    "--mode",
                    mode,
                    "--threads",
                    str(args.threads),
                    "--device",
                    args.device,
                    "--precision",
                    args.precision,
                ]
                if trial == 0 and mode != "v3":
                    command += ["--outputs", str(target.with_suffix(".pt").resolve())]
                print(f"{trace.stem}: {mode} {trial + 1}/{args.trials}", flush=True)
                subprocess.run(command, check=True, env=environment)
                cases.append(json.loads(target.read_text()))
                write_json(
                    args.output, {"complete": False, "cases": cases, "summary": summarize(cases)}
                )
    # 同一单向结构的无缓存与有缓存输出必须逐请求匹配；v3 是结构基线，单独报告。
    import torch

    equivalence = {}
    for trace in args.trace:
        for scope in ("active", "readout"):
            reference = directory / f"{trace.stem}-{scope}_reference-1.pt"
            cached = directory / f"{trace.stem}-{scope}_cached-1.pt"
            if not reference.exists() or not cached.exists():
                continue
            before = torch.load(reference, weights_only=True)
            after = torch.load(cached, weights_only=True)
            if len(before) != len(after):
                raise AssertionError("缓存与参考消费量不同")
            error = 0.0
            for expected, actual in zip(before, after):
                atol, rtol = (2e-5, 2e-4) if args.precision == "fp32" else (2e-3, 2e-2)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                error = max(error, float((actual - expected).abs().max()))
            equivalence[f"{trace.stem}/{scope}"] = {
                "requests": len(before),
                "max_logit_error": error,
            }
    write_json(
        args.output,
        {"complete": True, "cases": cases, "summary": summarize(cases), "equivalence": equivalence},
    )
    print(json.dumps(summarize(cases), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
