"""常驻本地JSONL推理；只接受八项编码张量，不持有游戏、教师、标签或正式随机数。"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .data import FLOAT_KEYS, INDEX_KEYS, INPUT_KEYS, MASK_KEYS, validate_inputs
from .model import NETWORK_VERSION, PolicyValueNet
from .runtime import autocast, checkpoint_config, resolve_device, synchronize


def decode_input(raw, config):
    """JSON类型先校验再转张量，不能将浮点指针截断、数字遮罩或未知字段悄悄接受。"""
    if not isinstance(raw, dict) or set(raw) != INPUT_KEYS:
        raise ValueError("只接受八项公开输入张量")
    batch = {}
    for key, values in raw.items():

        def check(value):
            if isinstance(value, list):
                for item in value:
                    check(item)
            elif key in MASK_KEYS:
                if type(value) is not bool:
                    raise ValueError(f"{key}必须为布尔值")
            elif key in INDEX_KEYS:
                if type(value) is not int:
                    raise ValueError(f"{key}必须为整数")
            elif type(value) not in (float, int):
                raise ValueError(f"{key}必须为数值")

        if not isinstance(values, list):
            raise ValueError(f"{key}必须是数组")
        check(values)
        dtype = (
            torch.float32 if key in FLOAT_KEYS else torch.bool if key in MASK_KEYS else torch.long
        )
        batch[key] = torch.tensor(values, dtype=dtype).unsqueeze(0)
    validate_inputs(batch, config)
    return batch


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "xpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads必须是正整数")
    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = resolve_device(args.device)
    config = checkpoint_config(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if metadata.get("synthetic") is not False:
        parser.error("实局推理需要真实编码数据的检查点")
    model = PolicyValueNet(config).to(device).eval()
    model.load_state_dict(payload["model"])
    with args.checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    synchronize(device)
    emit(
        {
            "type": "ready",
            "format": "haojie-policy-jsonl-v1",
            "network": NETWORK_VERSION,
            "config": asdict(config),
            "ruleset": metadata["ruleset"],
            "encoding": metadata["encoding"],
            "schema": metadata["schema"],
            "checkpoint_sha256": digest,
            "device": str(device),
            "precision": args.precision,
            "threads": args.threads,
            "torch": torch.__version__,
            "load_ms": (time.perf_counter() - start) * 1000,
        }
    )
    # 单请求串行处理；行上限是协议保护，超限报错，不截断实体或悄悄换模型。
    while line := sys.stdin.buffer.readline(4 * 1024 * 1024 + 1):
        request_id = None
        try:
            if len(line) > 4 * 1024 * 1024:
                raise ValueError("请求超过4MiB，请更换传输方案")
            start = time.perf_counter()
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"id", "inputs"}:
                raise ValueError("请求只允许id/inputs")
            request_id = request["id"]
            if type(request_id) is not int:
                raise ValueError("id必须为整数")
            batch = decode_input(request["inputs"], config)
            validated = time.perf_counter()
            with torch.inference_mode(), autocast(device, args.precision):
                batch = {key: tensor.to(device) for key, tensor in batch.items()}
                logits, value = model(batch)
                logits, value = logits.cpu(), value.cpu()
            synchronize(device)
            evaluated = time.perf_counter()
            if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                raise ValueError("推理输出包含NaN/Inf")
            emit(
                {
                    "id": request_id,
                    "ok": True,
                    "logits": logits[0].tolist(),
                    "value": float(value[0]),
                    "timing": {
                        "validation_ms": (validated - start) * 1000,
                        "model_ms": (evaluated - validated) * 1000,
                    },
                }
            )
        except Exception as error:
            emit({"id": request_id, "ok": False, "error": str(error)})
            # 超长行无法安全继续分帧；其余坏请求明确返回失败后仍可接收下一行。
            if len(line) > 4 * 1024 * 1024:
                break


if __name__ == "__main__":
    main()
