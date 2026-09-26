"""候选位置的空间上下文对照；只读取v1公开张量，实验模型不能冒充原网络。"""

import torch
from torch import nn

from haojie_training.model import PolicyValueNet


class SpatialPolicyNet(PolicyValueNet):
    """把场上实体放到固定棋盘格，再读取每个候选落点周围的上下文。

    不模拟战斗、不裁剪候选、不接收教师分数；基地外沿也保留。相同格的堆叠实体求和。
    原来源实体、全局状态、非坐标目标保持原评分输入。输入与源张量不变。
    """

    def __init__(self, config):
        super().__init__(config)
        channels = 32
        self.board_projection = nn.Linear(config.width, channels)
        self.board = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, config.width, 3, padding=1),
        )

    def target_context(self, batch, entities, targets):
        raw = batch["entities"]
        roles = (raw[..., 0] * 32).round().long()
        on_board = batch["entity_mask"] & (roles >= 1) & (roles <= 3)
        # v1的基地、单位、地标前两个字段都是log1p坐标；其他角色必须先遮罩。
        encoded = torch.where(on_board[..., None], raw[..., 8:10], 0)
        xy = (encoded.sign() * torch.expm1(encoded.abs() * 8)).round().long()
        index = xy[..., 1].clamp(0, 14) * 11 + xy[..., 0].clamp(0, 10)
        features = self.board_projection(entities[:, 1:]) * on_board[..., None]
        grid = features.new_zeros(features.shape[0], 165, features.shape[-1])
        grid.scatter_add_(1, index[..., None].expand_as(features), features)
        grid = self.board(grid.transpose(1, 2).reshape(-1, features.shape[-1], 15, 11))
        coords = batch["candidates"][..., 36:38] * raw.new_tensor([9, 13])
        coords = coords.round().long()
        point = (coords[..., 0] > 0) & (coords[..., 1] > 0) & (batch["targets"] == -1)
        at = coords[..., 1].clamp(0, 14) * 11 + coords[..., 0].clamp(0, 10)
        local = (
            grid.flatten(2)
            .transpose(1, 2)
            .gather(1, at[..., None].expand(-1, -1, self.config.width))
        )
        return torch.where(point[..., None], local, targets)
