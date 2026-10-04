"""复用已审核实局分片；冻结输入、输出及连续更新索引，不重新生成对局。"""

import argparse
import json
import time
from pathlib import Path

import torch

from haojie_training.data import (
    INPUT_KEYS,
    collate_examples,
    load_dataset,
    select_batch,
)
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.runtime import autocast, synchronize


def timed(device, fn, repeats):
    synchronize(device)
    begin = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - begin) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    model = PolicyValueNet(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    if args.freeze:
        dataset, meta = load_dataset(args.data, model.config)
        # 每个实际轨迹等距取四个节点，再按形状排序组成小批，覆盖规则/盘面/动作阶段。
        games = {}
        for i, record in enumerate(meta["records"]):
            games.setdefault(record["game_id"], []).append(i)
        indices = sorted(
            {
                rows[j * (len(rows) - 1) // 3]
                for rows in games.values()
                for j in range(4)
            }
        )
        examples = [
            {k: v[0] for k, v in select_batch(dataset, torch.tensor([i])).items()}
            for i in indices
        ]
        examples.sort(key=lambda e: (len(e["entities"]), len(e["candidates"])))
        torch.save(
            {
                "examples": examples,
                "indices": indices,
                "records": [meta["records"][i] for i in indices],
                "data": str(args.data),
                "metadata": {k: v for k, v in meta.items() if k != "records"},
            },
            args.fixture,
        )
    frozen = torch.load(args.fixture, weights_only=True)
    examples = frozen["examples"]
    model.to(device).eval()
    groups = [examples[i : i + 4] for i in range(0, len(examples), 4)]
    groups = [g for g in groups if len(g) == 4]
    batches = [collate_examples(g, model.config) for g in groups]
    calls = 0
    if args.optimized:
        from haojie_training.native.execution import PolicyInference

        runner = PolicyInference(model, device, args.precision)

    def infer(i):
        nonlocal calls
        calls += 1
        if args.optimized:
            return runner([{k: e[k] for k in INPUT_KEYS} for e in groups[i]])
        batch = collate_examples(groups[i], model.config)
        with torch.inference_mode(), autocast(device, args.precision):
            logits, values = model({k: v.to(device) for k, v in batch.items()})
            logits, values = logits.cpu(), values.cpu()
        if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
            raise ValueError("非有限输出")
        return logits

    for i in range(len(groups)):
        infer(i)
    rows, outputs = [], []
    for i, batch in enumerate(batches):
        seconds = timed(device, lambda: infer(i), 5)
        outputs.append(infer(i))
        rows.append(
            {
                "shape": list(batch["entities"].shape[:2])
                + [batch["candidates"].shape[1]],
                "seconds": seconds,
                "entities": int(batch["entity_mask"].sum()),
                "candidates": int(batch["candidate_mask"].sum()),
            }
        )
    torch.save(outputs, args.output / "logits.pt")
    if args.profile:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(activities=activities) as prof:
            infer(len(groups) // 2)
            synchronize(device)
        (args.output / "profile.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25),
            encoding="utf-8",
        )
        events = [
            {
                "name": e.key,
                "calls": e.count,
                "self_cpu_us": e.self_cpu_time_total,
                "self_device_us": e.self_device_time_total,
                "device_us": e.device_time_total,
            }
            for e in prof.key_averages()
        ]
        (args.output / "profile-events.json").write_text(
            json.dumps(events, indent=2), encoding="utf-8"
        )
    compilation = None
    if args.compile:
        begin = time.perf_counter()
        try:
            compiled = torch.compile(model, fullgraph=True)
            with torch.inference_mode():
                compiled({k: v.to(device) for k, v in batches[0].items()})
            synchronize(device)
            compilation = {"seconds": time.perf_counter() - begin, "ok": True}
        except Exception as error:
            compilation = {"seconds": time.perf_counter() - begin, "error": str(error)}
    report = {
        "torch": str(torch.__version__),
        "device": str(device),
        "threads": args.threads,
        "parameters": sum(p.numel() for p in model.parameters()),
        "precision": args.precision,
        "rows": rows,
        "requests_per_second": sum(r["shape"][0] for r in rows)
        / sum(r["seconds"] for r in rows),
        "calls": calls,
        "compile": compilation,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
