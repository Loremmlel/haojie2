"""内存训练共用的原生规则元数据、模型加载与样本转换。"""

import torch

from ..data import bf16_tensors
from ..model import PolicyValueNet
from ..prepare import digest
from ..runtime import read_checkpoint
from .aten import operator
from .client import ArrayInput


def metadata(ready):
    # decision_stages 是原生树协议扩展，模型沿用原 ENCODING_SCHEMA。
    schema = {k: v for k, v in ready["schema"].items() if k != "decision_stages"}
    return {
        "synthetic": False,
        "ruleset": ready["ruleset"],
        "encoding": schema["encoding"],
        "schema": schema,
        "source_sha256": ready["rulesHash"],
        "rules_package_sha256": ready["rulesHash"],
    }


def model_from(path, ready, device="cpu"):
    config, payload = read_checkpoint(path)
    expected = metadata(ready)
    if any(
        payload["metadata"].get(key) != expected[key]
        for key in ("ruleset", "encoding", "schema", "rules_package_sha256")
    ):
        raise ValueError("检查点规则/编码/规则包哈希不匹配")
    model = PolicyValueNet(config)
    model.load_state_dict(payload["model"])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("检查点包含非有限参数")
    return model.to(device).eval(), digest(path)


def example(inputs, selected=0):
    native = isinstance(inputs, ArrayInput) and inputs._data is not None and operator() is not None
    count = inputs.candidate_count if native else len(inputs["candidate_mask"])
    policy = torch.zeros(count, dtype=torch.bfloat16)
    policy[selected] = 1
    if native:
        storage = torch.frombuffer(
            memoryview(inputs._data)[: inputs.frame_bytes], dtype=torch.uint8
        )
        # 共享槽只整帧复制一次；各字段仍独立持有可序列化的 typed Storage。
        if getattr(inputs, "shared", False):
            storage = storage.clone()
        tensors = dict(
            zip(
                inputs,
                torch.ops.haojie.input_views(
                    storage,
                    inputs.entity_count,
                    inputs.candidate_count,
                ),
            )
        )
    else:
        tensors = {
            key: value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
            for key, value in inputs.items()
        }
        # 蓄水抽中的样本独立持有内存；共享输入槽随后会被下一条请求覆盖。
        if getattr(inputs, "shared", False):
            tensors = {key: value.clone() for key, value in tensors.items()}
    return {
        **bf16_tensors(tensors),
        "policy": policy,
        "value": torch.tensor(0.0, dtype=torch.bfloat16),
        "value_mask": torch.tensor(False),
    }
