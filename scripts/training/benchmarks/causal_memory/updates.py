"""固定真实输入的 B8 前向/反向成本探针；不创建优化器、不更新权重、不评估棋力。"""

import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

from bench import ROOT, file_hash, write_json

sys.path.insert(0, str(ROOT / "training"))
importlib.import_module("haojie_training")

import torch  # noqa: E402
from model import CausalMemoryPolicy  # noqa: E402

from haojie_training.benchmark import tensor_hash  # noqa: E402
from haojie_training.data import collate_inputs, validate_inputs  # noqa: E402
from haojie_training.model import ModelConfig, PolicyValueNet  # noqa: E402

MODES = ("v3", "active", "readout")
BATCH = 8
SAMPLES = 16


def expanded_batch(wrapper, records, selected):
    """只在计时外展开 CPU 布局；计时内每层仍是一次真正的批矩阵运算。"""
    layouts = [wrapper._layout(record["input"], record["base_entities"]) for record in records]
    lengths = [layout.memory_tokens + 1 + len(layout.dynamic) for layout in layouts]
    size, tokens = len(records), max(lengths)
    config = wrapper.model.config
    batch = {
        "entities": torch.zeros(size, tokens, config.entity_features),
        "kinds": torch.zeros(size, tokens, dtype=torch.long),
        "globals": torch.zeros(size, 2, config.global_features),
        "global_positions": torch.zeros(size, 2, dtype=torch.long),
        "readout": torch.zeros(size, 3, dtype=torch.long),
        "candidates": torch.zeros(size, 1, config.action_features),
        "candidate_mask": torch.ones(size, 1, dtype=torch.bool),
        "sources": torch.zeros(size, 1, dtype=torch.long),
        "targets": torch.ones(size, 1, dtype=torch.long),
    }
    for index, (layout, chosen) in enumerate(zip(layouts, selected.tolist())):
        row, memory = layout.row, layout.memory_tokens
        memory_entities = torch.cat((layout.base, layout.journal))
        for key in ("entities", "kinds"):
            batch[key][index, 1:memory] = row[key][memory_entities]
            batch[key][index, memory + 1 : lengths[index]] = row[key][layout.dynamic]
        batch["globals"][index] = torch.stack((layout.stable_globals, row["globals"]))
        batch["global_positions"][index, 1] = memory
        remap = torch.zeros(len(row["entities"]) + 1, dtype=torch.long)
        remap[layout.dynamic + 1] = torch.arange(1, len(layout.dynamic) + 1)
        batch["readout"][index] = torch.tensor(
            [
                memory,
                memory + remap[row["sources"][chosen] + 1],
                memory + remap[row["targets"][chosen] + 1],
            ]
        )
        batch["candidates"][index, 0] = row["candidates"][chosen]
    base = torch.tensor([layout.base_tokens for layout in layouts])[:, None, None]
    memory = torch.tensor([layout.memory_tokens for layout in layouts])[:, None, None]
    total = torch.tensor(lengths)[:, None, None]
    positions = torch.arange(tokens)
    query, key = positions[None, :, None], positions[None, None, :]
    batch["allowed"] = (
        (
            ((query < base) & (key < base))
            | ((query >= base) & (query < memory) & (key <= query))
            | (query >= memory)
        )
        & (query < total)
        & (key < total)
    )[:, None]
    batch["last_allowed"] = (key < total)[:, None]
    return batch, {
        "tokens": lengths,
        "padded_tokens": tokens,
        "memory_tokens": [layout.memory_tokens for layout in layouts],
        "dynamic_query_tokens": [1 + len(layout.dynamic) for layout in layouts],
        "readout_query_tokens": 3,
    }


def selected_reference(model, batch):
    """同实验结构的批参考；中间动态集合仍取全部候选，末层只取已选候选。"""
    x = model.entity_projection(batch["entities"]) + model.kind_embedding(batch["kinds"])
    globals_ = model.global_projection(batch["globals"])
    x = x.scatter(1, batch["global_positions"][..., None].expand(-1, -1, x.shape[-1]), globals_)
    for index, block in enumerate(model.blocks):
        last = index == len(model.blocks) - 1
        x = block(
            x,
            batch["last_allowed"] if last else batch["allowed"],
            batch["readout"] if last else None,
        )
    x = model.norm(x)
    return model.score_policy(batch, x).squeeze(1), model.value(x[:, 0]).squeeze(-1).float()


def check_reference(wrapper, records, selected):
    """较短与较长 B1 对照现有单请求参考，防止批实现测了另一种结构。"""
    sizes = [len(record["input"]["entities"]) for record in records]
    positions = sorted({sizes.index(min(sizes)), sizes.index(max(sizes))})
    errors = []
    with torch.no_grad():
        for index in positions:
            record, choice = records[index], selected[index : index + 1]
            batch, _ = expanded_batch(wrapper, [record], choice)
            actual = selected_reference(wrapper.model, batch)
            logits, value = wrapper.forward_reference(record["input"], record["base_entities"])
            expected = (logits[choice], value[None])
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)
                errors.append(float((result - reference).abs().max()))
    return {"cases": len(positions), "max_output_error": max(errors)}


def check_batched_reference(wrapper, records, selected, batch):
    """混合长度 B8 对逐条完整参考，检查 padding 遮罩与各行的读取位置。"""
    errors = []
    with torch.no_grad():
        actual = selected_reference(wrapper.model, batch)
        logits, values = [], []
        for record, choice in zip(records, selected):
            policy, value = wrapper.forward_reference(record["input"], record["base_entities"])
            logits.append(policy[choice])
            values.append(value)
        for result, reference in zip(actual, (torch.stack(logits), torch.stack(values))):
            torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)
            errors.append(float((result - reference).abs().max()))
    return {"cases": len(records), "max_output_error": max(errors)}


def measure(model, batch, selected, targets, mode):
    model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    chosen, value = (
        model.selected_value(batch, selected) if mode == "v3" else selected_reference(model, batch)
    )
    # 与 Trainer 的 MC-Q 相同公式；固定替代标签全为已知，value_weight=1。
    loss = (chosen.tanh() - targets).square().mean() + (value - targets).square().mean()
    forward_end = time.perf_counter()
    loss.backward()
    end = time.perf_counter()
    if not torch.isfinite(loss):
        raise FloatingPointError("成本探针出现非有限损失")
    if any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise FloatingPointError("反向梯度不完整或非有限")
    return {
        "forward_seconds": forward_end - start,
        "backward_seconds": end - forward_end,
        "seconds": end - start,
        "loss": float(loss.detach()),
    }


def summarize(cases, traces, trials):
    result = {}
    for trace in traces:
        modes = {}
        for mode in MODES:
            paired = [
                {
                    metric: sum(
                        case[metric]
                        for case in cases
                        if case["trace"] == trace
                        and case["trial"] == trial
                        and case["mode"] == mode
                    )
                    for metric in ("forward_seconds", "backward_seconds", "seconds")
                }
                for trial in range(trials)
            ]
            modes[mode] = {
                "trials": paired,
                **{
                    f"median_{metric}": statistics.median(row[metric] for row in paired)
                    for metric in paired[0]
                },
            }
        for value in modes.values():
            value["cost_ratio_vs_v3"] = value["median_seconds"] / modes["v3"]["median_seconds"]
        result[trace] = modes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.trials) < 1:
        parser.error("线程数与轮数必须为正数")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    traces = [
        (path, torch.load(path, map_location="cpu", weights_only=True)) for path in args.trace
    ]
    first = traces[0][1]
    torch.manual_seed(first["seed"])
    model = PolicyValueNet(ModelConfig(**first["model_config"])).train()
    weights = tensor_hash(model.state_dict())
    if weights != first["initial_model_sha256"]:
        raise AssertionError("初始化参数与真实采集权重不匹配")
    wrappers = {scope: CausalMemoryPolicy(model, query_scope=scope) for scope in MODES[1:]}
    prepared, provenance, checks = [], {}, {}
    for path, trace in traces:
        if (
            trace["format"] != "haojie-causal-memory-trace-v1"
            or len(trace["records"]) < SAMPLES
            or trace["initial_model_sha256"] != weights
            or trace["model_config"] != first["model_config"]
        ):
            raise ValueError("探针需要相同参数与配置、至少16条请求的真实轨迹")
        identity = str(path.resolve())
        indices = [index * (len(trace["records"]) - 1) // (SAMPLES - 1) for index in range(SAMPLES)]
        records = [trace["records"][index] for index in indices]
        selected = torch.tensor(
            [
                int(
                    record["input"]["candidate_mask"]
                    .nonzero()
                    .flatten()[index % int(record["input"]["candidate_mask"].sum())]
                )
                for index, record in enumerate(records)
            ]
        )
        checks[identity] = {
            scope: check_reference(wrapper, records, selected)
            for scope, wrapper in wrappers.items()
        }
        provenance[identity] = {
            "trace_sha256": file_hash(path),
            "record_indices": indices,
            "selected_candidates": selected.tolist(),
            "sequences": [record["sequence"] for record in records],
            "batches": [],
        }
        for offset in range(0, SAMPLES, BATCH):
            rows, choices = records[offset : offset + BATCH], selected[offset : offset + BATCH]
            original = collate_inputs([record["input"] for record in rows], model.config)
            validate_inputs(original, model.config)
            batches, metadata = {"v3": original}, {}
            for scope, wrapper in wrappers.items():
                batches[scope], metadata[scope] = expanded_batch(wrapper, rows, choices)
                if (
                    "batch8" not in checks[identity][scope]
                    and len(set(metadata[scope]["tokens"])) > 1
                ):
                    checks[identity][scope]["batch8"] = {
                        "batch": offset // BATCH,
                        "tokens": metadata[scope]["tokens"],
                        **check_batched_reference(wrapper, rows, choices, batches[scope]),
                    }
            provenance[identity]["batches"].append(
                {
                    "input_sha256": tensor_hash(original),
                    "entity_counts": [int(record["input"]["entity_mask"].sum()) for record in rows],
                    "candidate_counts": [
                        int(record["input"]["candidate_mask"].sum()) for record in rows
                    ],
                    "expanded": metadata,
                }
            )
            prepared.append(
                {
                    "trace": identity,
                    "batch": offset // BATCH,
                    "inputs": batches,
                    "selected": choices,
                    "targets": torch.tensor([-1.0, 1.0] * (BATCH // 2)),
                }
            )
        if any("batch8" not in checks[identity][scope] for scope in wrappers):
            raise ValueError("每条轨迹需要至少一个混合长度 B8，才能验证批遮罩")
    result = {
        "format": "haojie-causal-memory-update-cost-v1",
        "initial_model_sha256": weights,
        "model_config": first["model_config"],
        "source_sha256": {
            "experiment": file_hash(Path(__file__).with_name("model.py")),
            "baseline": file_hash(ROOT / "training/haojie_training/model.py"),
            "probe": file_hash(Path(__file__)),
        },
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "device": "cpu",
        "precision": "fp32",
        "threads": args.threads,
        "batch_size": BATCH,
        "samples_per_trace": SAMPLES,
        "trials": args.trials,
        "sampling": "16 evenly spaced request indices, including both endpoints",
        "labels": "fixed alternating -1/+1 substitutes; all value_mask=True; no strength labels",
        "loss": "mean((tanh(selected_logit)-target)^2) + mean((value-target)^2)",
        "timed": "forward, heads, loss, backward; data prep, zero_grad, optimizer excluded",
        "ratio_scope": (
            "forward+backward only; experiment _layout, selected-action gather and readout "
            "remap CPU preparation excluded; v3.selected_value history_queries and candidate "
            "gathers included; not production Trainer throughput"
        ),
        "gradient_check": "all parameters after every warmup/measured backward, outside timing",
        "optimizer_steps": 0,
        "checks": checks,
        "inputs": provenance,
        "cases": [],
        "complete": False,
    }
    if not args.check_only:
        # 每种形状和模式各预热一次；三轮中循环交换模式顺序，参数始终不变。
        for sample in prepared:
            for mode in MODES:
                measure(model, sample["inputs"][mode], sample["selected"], sample["targets"], mode)
        for trial in range(args.trials):
            order = MODES[trial % len(MODES) :] + MODES[: trial % len(MODES)]
            for sample in prepared:
                for mode in order:
                    row = measure(
                        model, sample["inputs"][mode], sample["selected"], sample["targets"], mode
                    )
                    result["cases"].append(
                        {
                            "trace": sample["trace"],
                            "batch": sample["batch"],
                            "trial": trial,
                            "mode": mode,
                            **row,
                        }
                    )
            write_json(args.output, result)
            print(
                json.dumps({"trial": trial + 1, "trials": args.trials, "optimizer_steps": 0}),
                flush=True,
            )
        result["summary"] = summarize(result["cases"], list(provenance), args.trials)
    result["final_model_sha256"] = tensor_hash(model.state_dict())
    if result["final_model_sha256"] != weights:
        raise AssertionError("成本探针改变了模型权重")
    result["check_only"] = args.check_only
    result["weights_unchanged"] = True
    result["complete"] = True
    write_json(args.output, result)
    print(
        json.dumps({"complete": True, "check_only": args.check_only, "weights_unchanged": True}),
        flush=True,
    )


if __name__ == "__main__":
    main()
