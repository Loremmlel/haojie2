"""显式研究推理入口；复用正式公开张量协议，实验模型不进入默认加载路径。"""

import hashlib
import json
from pathlib import Path

import torch

from haojie_training.inference import main
from haojie_training.model import ModelConfig, PolicyValueNet

from .spatial import SpatialPolicyNet


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "haojie-recovery-policy-pilot-v1":
        raise ValueError("研究入口只接受本轮独立格式")
    protocol_path = path.parent / "protocol.json"
    if digest(protocol_path) != payload["protocol_sha256"]:
        raise ValueError("检查点与训练协议不一致")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    training = Path(protocol["inputs"]["train"]["path"])
    if digest(training) != protocol["inputs"]["train"]["sha256"]:
        raise ValueError("原训练元数据索引已改变")
    metadata = torch.load(training, map_location="cpu", weights_only=True)["metadata"]
    if metadata["encoding"] != "haojie-entities-factorized-v1" or metadata["synthetic"]:
        raise ValueError("研究推理要求真实v1公开编码")
    if payload["arm"] not in ("baseline", "capacity", "spatial"):
        raise ValueError("未知研究模型")
    config = ModelConfig(**payload["config"])
    cls = SpatialPolicyNet if payload["arm"] == "spatial" else PolicyValueNet
    model = cls(config)
    model.load_state_dict(payload["model"])
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("研究模型参数不是有限数")
    return config, model, metadata, "recovery-policy-pilot-v1"


if __name__ == "__main__":
    main(load_checkpoint)
