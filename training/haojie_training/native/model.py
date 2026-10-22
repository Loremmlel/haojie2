"""内存训练共用的原生规则元数据、模型加载与样本转换。"""

import torch

from ..model import PolicyValueNet
from ..prepare import digest
from ..runtime import read_checkpoint


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
    count = len(inputs["candidate_mask"])
    policy = torch.zeros(count)
    policy[selected] = 1
    return {
        **{
            key: value if isinstance(value, torch.Tensor) else torch.from_numpy(value)
            for key, value in inputs.items()
        },
        "policy": policy,
        "value": torch.tensor(0.0),
        "value_mask": torch.tensor(False),
    }
