"""组合研究模型的显式JSONL推理入口；复用正式边界，不启用游戏或隐藏数据。"""

import torch
from haojie_training.inference import main
from haojie_training.model import ModelConfig

from .data import digest
from .model import ImprovementNet
from .train import FORMAT


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format"] != FORMAT:
        raise ValueError("不是本轮研究模型")
    if digest(path.parent / "protocol.json") != payload["protocol_sha256"]:
        raise ValueError("训练协议已改变")
    model = ImprovementNet(ModelConfig(**payload["config"]), payload["arm"], payload["features"])
    model.load_state_dict(payload["model"])
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("研究参数不是有限值")
    return model.config, model, payload["metadata"], FORMAT


if __name__ == "__main__":
    main(load_checkpoint)
