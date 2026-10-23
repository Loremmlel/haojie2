"""导出 BF16 ONNX 与真实公开输入；保留 Torch 参考值供目标后端验收。"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import onnx
import torch
from torch import nn

from .data import load_dataset, select_batch, synthetic_batch, token_counts
from .model import NETWORK_VERSION, PolicyValueNet
from .runtime import read_checkpoint

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
    """保持八个公开输入及 BF16 参数，不插入精度回退。"""

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, *inputs):
        batch = dict(zip(INPUTS, inputs, strict=True))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("输出目录已存在；请使用新目录，保留旧实验")
    torch.set_num_threads(4)
    config, payload = read_checkpoint(args.checkpoint)
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
        },
        "cases_sha256": sha256(case_path),
        "models": {},
    }
    path = args.output / "bf16.onnx"
    # 导出数学图，避免把某一设备的私有 Flash 算子写入可移植文件。
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        torch.onnx.export(
            ExportNet(model),
            tuple(example[key] for key in INPUTS),
            str(path),
            input_names=list(INPUTS),
            output_names=["logits", "value"],
            dynamic_shapes=(shapes,),
            dynamo=True,
            opset_version=18,
            external_data=False,
        )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    if any(
        t.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE)
        for t in graph.graph.initializer
    ):
        raise ValueError("导出图包含非 BF16 浮点参数")
    if any(
        node.op_type == "Cast"
        and any(
            attribute.name == "to"
            and attribute.i
            in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE)
            for attribute in node.attribute
        )
        for node in graph.graph.node
    ):
        raise ValueError("导出后端插入了非 BF16 计算转换；未发布导出清单")
    manifest["models"]["bf16"] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "verification": "ONNX 结构及参数类型已检查；目标后端须另验 cases.json 数值",
    }
    print(f"BF16: {path.stat().st_size} bytes", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
