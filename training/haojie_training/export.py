"""导出动态ONNX和少量真实公开输入；CPU数值对照不代替浏览器验收或完整对局。"""

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .data import load_dataset, select_batch, synthetic_batch, token_counts
from .model import NETWORK_VERSION, PolicyValueNet
from .runtime import checkpoint_config

INPUTS = (
    "entities",
    "kinds",
    "entity_mask",
    "globals",
    "candidates",
    "sources",
    "targets",
    "candidate_mask",
)


class ExportNet(nn.Module):
    """保持八个公开输入；混合FP16仅转换主干/价值头，候选评分和外部浮点协议仍FP32。"""

    def __init__(self, model, mixed=False):
        super().__init__()
        self.model = copy.deepcopy(model).eval()
        if mixed:
            self.model.half()
            self.model.action_projection.float()
            self.model.policy.float()

    def forward(self, *inputs):
        batch = dict(zip(INPUTS, inputs, strict=True))
        dtype = self.model.entity_projection.weight.dtype
        for key in ("entities", "globals"):
            batch[key] = batch[key].to(dtype)
        return self.model(batch)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def cases_from_data(dataset):
    """固定选择长短序列、最多候选和混合批量，不根据模型输出挑选样本。"""
    counts = token_counts(dataset, "entity_mask")
    action_counts = token_counts(dataset, "candidate_mask")
    nonforced = (action_counts > 1).nonzero().flatten()
    if not len(nonforced):
        raise ValueError("至少需要一个多候选真实样本")
    ordered = nonforced[torch.argsort(counts[nonforced], stable=True)]
    choices = [
        ("small", [int(ordered[0])]),
        ("medium", [int(ordered[len(ordered) // 2])]),
        ("large", [int(counts.argmax())]),
        ("wide", [int(action_counts.argmax())]),
    ]
    choices.append(("batch4", [indices[0] for _, indices in choices]))
    return [
        (name, indices, select_batch(dataset, torch.tensor(indices))) for name, indices in choices
    ]


def output_error(actual, expected, mask):
    logits, value = [torch.from_numpy(x) for x in actual]
    reference, reference_value = expected
    return {
        "finite": bool(torch.isfinite(logits).all() and torch.isfinite(value).all()),
        "max_logit_abs": float((logits - reference).abs()[mask].max()),
        "max_probability_abs": float((logits.softmax(-1) - reference.softmax(-1)).abs().max()),
        "max_value_abs": float((value - reference_value).abs().max()),
        "top1_equal": int((logits.argmax(-1) == reference.argmax(-1)).sum()),
        "samples": len(value),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在；请使用新目录，保留旧实验")
    torch.set_num_threads(4)
    config = checkpoint_config(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PolicyValueNet(config).eval()
    model.load_state_dict(payload["model"])
    dataset, metadata = load_dataset(args.data, config)
    if metadata["synthetic"]:
        parser.error("浏览器验收必须使用真实编码数据")
    if any(payload["metadata"].get(key) != metadata[key] for key in ("ruleset", "encoding")):
        parser.error("检查点和验收数据的规则/编码版本不一致")
    cases = cases_from_data(dataset)
    args.output.mkdir(parents=True)
    references, encoded = {}, []
    with torch.inference_mode():
        for name, indices, batch in cases:
            references[name] = model(batch)
            encoded.append(
                {
                    "name": name,
                    "indices": indices,
                    "inputs": {
                        key: {
                            "type": str(batch[key].dtype).removeprefix("torch."),
                            "dims": list(batch[key].shape),
                            "data": batch[key].flatten().tolist(),
                        }
                        for key in INPUTS
                    },
                    "reference": {
                        key: value.flatten().tolist()
                        for key, value in zip(("logits", "value"), references[name], strict=True)
                    },
                }
            )
    case_path = args.output / "cases.json"
    case_path.write_text(json.dumps(encoded, separators=(",", ":")), encoding="utf-8")

    # 示例只用于图追踪；真实输入在导出后逐个验证B/N/A动态维度，包含B=1。
    example = synthetic_batch(config, size=2, entities=8, actions=5)
    b, n, a = (torch.export.Dim(name, min=1) for name in ("batch", "entities", "actions"))
    shapes = (
        {0: b, 1: n},
        {0: b, 1: n},
        {0: b, 1: n},
        {0: b},
        {0: b, 1: a},
        {0: b, 1: a},
        {0: b, 1: a},
        {0: b, 1: a},
    )
    manifest = {
        "format": "haojie-browser-benchmark-v1",
        "network": NETWORK_VERSION,
        "config": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_sha256": sha256(args.data),
        "ruleset": metadata["ruleset"],
        "encoding": metadata["encoding"],
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "cases_sha256": sha256(case_path),
        "models": {},
    }
    for precision in ("fp32", "mixed-fp16"):
        path = args.output / f"{precision}.onnx"
        wrapped = ExportNet(model, mixed=precision == "mixed-fp16").eval()
        torch.onnx.export(
            wrapped,
            tuple(example[key] for key in INPUTS),
            str(path),
            input_names=list(INPUTS),
            output_names=["logits", "value"],
            dynamic_shapes=(shapes,),
            dynamo=True,
            opset_version=18,
            external_data=False,
        )
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        results = {}
        for name, _, batch in cases:
            actual = session.run(None, {key: batch[key].numpy() for key in INPUTS})
            if not all(np.isfinite(value).all() for value in actual):
                raise ValueError(f"{precision}/{name}输出非有限数值")
            results[name] = output_error(actual, references[name], batch["candidate_mask"])
            if (
                precision == "fp32"
                and max(results[name]["max_probability_abs"], results[name]["max_value_abs"]) > 1e-3
            ):
                raise ValueError(f"FP32导出数值对照失败：{results[name]}")
        manifest["models"][precision] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "native_cpu_parity": results,
        }
        print(f"{precision}: {path.stat().st_size} bytes; {results}", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
