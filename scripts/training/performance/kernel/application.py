"""相同实际模型、起点、预算和并发的交错应用层基准；探针轮与正式计时分开。"""

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from haojie_training.data import load_dataset, sample_indices, select_batch
from haojie_training.evaluate import evaluate
from haojie_training.native import pipeline
from haojie_training.native.client import Client
from haojie_training.runtime import Trainer
from usage import Usage


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global pipeline
    current_pipeline = pipeline
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "candidate", "checkpoint", "starts", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--commands", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--pipeline-reference", type=Path)
    parser.add_argument("--single-pass-candidate", action="store_true")
    args = parser.parse_args()
    if args.pipeline_reference:
        spec = importlib.util.spec_from_file_location("haojie_training.native.strict_reference", args.pipeline_reference)
        pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline)
    if min(args.commands, args.concurrency, args.rounds) < 1:
        raise ValueError("预算、并发和轮数须为正")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    starts = json.loads(args.starts.read_text(encoding="utf-8"))
    engines = {}
    for key in ("baseline", "candidate"):
        engines[key] = args.output / f"{key}.exe"
        shutil.copyfile(getattr(args, key), engines[key])
    shutil.copyfile(__file__, args.output / "application.py")
    shutil.copyfile(Path(__file__).with_name("usage.py"), args.output / "usage.py")
    manifest = {"python": platform.python_version(), "torch": str(torch.__version__),
                "platform": platform.platform(), "threads": 1, "device": "cpu",
                "commands": args.commands, "concurrency": args.concurrency, "starts": len(starts),
                "profile": args.profile, "checkpoint_sha256": digest(args.checkpoint),
                "pipeline_reference_sha256": digest(args.pipeline_reference) if args.pipeline_reference else None,
                "single_pass_candidate": args.single_pass_candidate,
                "starts_sha256": digest(args.starts),
                "engines": {key: digest(path) for key, path in engines.items()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    probes = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    if args.profile:
        # 只用于独立剖析：读取包含等待，不能把它与 Rust 的 inferenceMs 或模型前向相加。
        from haojie_training.native import client as protocol

        def instrument(owner, name, label):
            original = getattr(owner, name)

            def measured(*a, **kw):
                t = time.perf_counter()
                try:
                    return original(*a, **kw)
                finally:
                    probes[label]["calls"] += 1
                    probes[label]["seconds"] += time.perf_counter() - t
            setattr(owner, name, measured)

        instrument(pipeline, "model_from", "model_load")
        instrument(pipeline, "collate_examples", "collation")
        instrument(protocol, "decode", "tensor_decode")
        instrument(Client, "send", "control_send")
        instrument(pipeline.PolicyValueNet, "forward", "model_forward")

    references = None

    def work(label, name, usage):
        nonlocal references
        output = args.output / name
        output.mkdir()
        stages, reports, records = {}, [], []
        begin = time.perf_counter()
        t = time.perf_counter()
        for offset in range(0, len(starts), args.concurrency):
            folder = output / f"sample-{offset}"
            reports.append(pipeline.sample(engines[label], args.checkpoint,
                starts[offset:offset + args.concurrency], folder, args.commands, 60,
                seed=20261003 + offset))
            records.extend(sorted(folder.glob("*.jsonl")))
        stages["sampling"] = time.perf_counter() - t
        t = time.perf_counter()
        single = args.single_pass_candidate and label == "candidate"
        if single:
            audited = [current_pipeline.audit(engines[label], path) for path in records]
        else:
            with Client(engines[label]) as client:
                for path in records:
                    client.send({"op": "audit", "record": str(path.resolve()), "encode": False})
                    client.receive()
        stages["audit_encode" if single else "audit"] = time.perf_counter() - t
        t = time.perf_counter()
        prepared = current_pipeline.prepare(engines[label], records, output / "data", audited=audited) if single else pipeline.prepare(engines[label], records, output / "data")
        stages["prepare"] = time.perf_counter() - t
        t = time.perf_counter()
        with Client(engines[label]) as client:
            model, _ = pipeline.model_from(args.checkpoint, client.ready)
        trainer = Trainer(model, torch.device("cpu"))
        stages["training_model_load"] = time.perf_counter() - t
        t = time.perf_counter()
        data, _ = load_dataset(output / "data/train.pt", model.config)
        validation, _ = load_dataset(output / "data/validation.pt", model.config)
        stages["data_load"] = time.perf_counter() - t
        t = time.perf_counter()
        for step in range(2):
            indices = sample_indices(data, 4, torch.Generator().manual_seed(81 + step))
            trainer.step(select_batch(data, indices))
        stages["optimizer"] = time.perf_counter() - t
        t = time.perf_counter()
        evaluation = evaluate(model, validation, torch.device("cpu"), "fp32", 4)
        stages["evaluation"] = time.perf_counter() - t
        assert trainer.health()["finite_optimizer"]
        elapsed = time.perf_counter() - begin
        resources = usage.stop()
        # 两端已经独立审核全部哈希链；JSON 对象字段书写顺序不属于规范哈希语义。
        # 核对每行的规范链哈希，文件字节哈希另外留档，不误判类型化序列化的字段顺序。
        identities = [[json.loads(line)["sha256"] for line in path.read_text(encoding="utf-8").splitlines()]
                      for path in records]
        if references is None:
            references = identities
        if identities != references:
            raise AssertionError("相同模型/策略随机源的记录发生差异；保留现场")
        metrics = defaultdict(float)
        for report in reports:
            for game in report["games"]:
                for key, value in game.get("metrics", {}).items():
                    if key in ("maxEntities", "maxCandidates"):
                        metrics[key] = max(metrics[key], value)
                    else:
                        metrics[key] += value
        result = {"label": label, "seconds": elapsed, "phases": stages, "resources": resources,
                  "reports": reports, "metrics": dict(metrics), "records": identities,
                  "record_file_sha256": [digest(path) for path in records],
                  "splits": prepared["splits"], "validation_examples": evaluation["examples"],
                  "parameters": sum(p.numel() for p in model.parameters())}
        (output / "measurement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"run": name, "seconds": elapsed, "phases": stages}), flush=True)
        return result

    def run(label, name):
        usage = Usage().start()
        try:
            return work(label, name, usage)
        finally:
            usage.stop()

    pairs = []
    if args.profile:
        run("candidate", "profile")
        (args.output / "profile.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")
    else:
        run("baseline", "warmup-baseline")
        run("candidate", "warmup-candidate")
        for round_number in range(args.rounds):
            labels = ("candidate", "baseline") if round_number % 2 else ("baseline", "candidate")
            pair = {label: run(label, f"round-{round_number}-{label}") for label in labels}
            pairs.append(pair)
            print(json.dumps({"round": round_number, "speedup":
                pair["baseline"]["seconds"] / pair["candidate"]["seconds"]}), flush=True)
        summary = {"pairs": pairs, "speedup_median": statistics.median(
            p["baseline"]["seconds"] / p["candidate"]["seconds"] for p in pairs),
            "note": "同一真实未训练权重；完整候选、前向、记录、两步更新与验证。strict 含独立审核加再次审核编码；single-pass 合并审核编码并消费已审核张量。不是棋力结论。"}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
