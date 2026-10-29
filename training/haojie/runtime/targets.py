"""以独立完整对局留出集校准截断软目标；估计值始终与真实胜负分开。"""

import torch
from torch import nn
from torch.nn import functional as F


class Calibration:
    def __init__(self, saved=None):
        self.games = []
        self.weights = None
        self.validation = None
        if saved:
            self.games, self.weights, self.validation = (
                saved["games"],
                saved["weights"],
                saved["validation"],
            )

    def snapshot(self):
        return {
            "games": self.games,
            "weights": self.weights,
            "validation": self.validation,
        }

    def observe(self, meta, minimum):
        winner = meta["winner"]
        if winner is None or meta["game"]["opponent"] != "self":
            return
        timeline = meta["timeline"]
        indices = sorted(
            {min(len(timeline) - 1, i * len(timeline) // 16) for i in range(16)}
        )
        self.games.append(
            {
                "id": meta["game"]["id"],
                "features": [timeline[i][1:] for i in indices],
                "target": 0.5 if winner == "draw" else float(winner == 1),
            }
        )
        self.games = self.games[-256:]
        if len(self.games) < minimum:
            return
        train = []
        validation = []
        for i, game in enumerate(self.games):
            destination = validation if i % 5 == 0 else train
            for features in game["features"]:
                destination.extend(
                    [
                        (features, game["target"]),
                        ([-v for v in features], 1 - game["target"]),
                    ]
                )

        def tensors(rows):
            return torch.tensor(
                [r[0] for r in rows], dtype=torch.bfloat16
            ), torch.tensor([r[1] for r in rows], dtype=torch.bfloat16)

        x, y = tensors(train)
        vx, vy = tensors(validation)
        # 无偏置与镜像样本保持换边对称；校准器本身也使用 BF16 参数和梯度。
        model = nn.Linear(4, 1, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(256):
            optimizer.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y).backward()
            optimizer.step()
        with torch.no_grad():
            score = float((model(vx).squeeze(-1).sigmoid() - vy).square().mean())
            baseline = float((vy - 0.5).square().mean())
        self.validation = {
            "games": len(self.games),
            "heldout_games": sum(i % 5 == 0 for i in range(len(self.games))),
            "brier": score,
            "baseline": baseline,
        }
        self.weights = model.weight.detach().clone() if score < baseline else None

    def predict(self, features, actor):
        if self.weights is None:
            return None
        with torch.no_grad():
            value = float(
                (
                    F.linear(
                        torch.tensor(features, dtype=torch.bfloat16), self.weights
                    ).sigmoid()
                    * 2
                    - 1
                ).squeeze()
            )
        return max(-0.98, min(0.98, value)) * (1 if actor == 1 else -1)
