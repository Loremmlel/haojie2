"""可解析短环境的独立初始化前后对照；正式原生 MC 搜索与 Trainer，样本只来自自身采样。"""

import argparse
import json
import random
from pathlib import Path

import torch

from haojie_training.console.memory import SamplePool, Trajectory
from haojie_training.data import collate_examples
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.native.client import Client
from haojie_training.native.execution import PolicyInference
from haojie_training.runtime import Trainer


def episode(client, model, rng, case, learning):
    trajectory = Trajectory(rng.randrange(1, 2**32), capacity=64, context={"rules": "exercise"})
    infer = PolicyInference(model.eval(), "cpu")
    client.send(
        {
            "op": "mc-exercise",
            "case": case,
            "samplerSeed": rng.randrange(1, 2**32),
            "outcomeSeed": rng.randrange(1, 2**32),
        }
    )
    while True:
        message = client.receive()
        if message["type"] == "infer":
            values = infer([message["input"]])[0].tanh() / 0.25
            client.send({"id": message["id"], "model": message["model"], "logits": values.tolist()})
        elif message["type"] == "example":
            trajectory.add(message)
        elif message["type"] == "done":
            return message, list(trajectory.terminal({"1": message["reward"]})) if learning else []


def run(engine, seed, rounds=320):
    torch.manual_seed(seed)
    model = PolicyValueNet(ModelConfig.tiny())
    trainer = Trainer(model, torch.device("cpu"), objective="decomposed-mc-q-v2")
    rng = random.Random(seed)
    pool = SamplePool(16 * 2**20)
    with Client(engine) as client:

        def evaluate():
            evaluation_rng = random.Random(7001)
            rows = [episode(client, model, evaluation_rng, i % 3, False)[0] for i in range(240)]
            return {
                "expected": sum(r["expectation"] for r in rows) / len(rows),
                "by_case": [
                    sum(r["expectation"] for r in rows if r["case"] == c) / 80 for c in range(3)
                ],
                "pass": [sum(r["pass"] for r in rows if r["case"] == c) / 80 for c in range(3)],
            }

        before = evaluate()
        backtracks = 0
        for generation in range(rounds):
            for i in range(12):
                result, rows = episode(client, model, rng, i % 3, True)
                backtracks += result["metrics"]["backtracks"]
                for row in rows:
                    pool.add(row)
            model.train()
            for _ in range(pool.budget(16, 64)):
                if not pool.rows:
                    break
                trainer.step(collate_examples(pool.batch(16, rng), model.config))
            pool.expire(generation + 1)
        after = evaluate()
    return {
        "seed": seed,
        "before": before,
        "after": after,
        "updates": trainer.updates,
        "consumption": pool.snapshot(),
        "backtracks": backtracks,
        "reference_expected": (0.6 + 0.8 + 0.8) / 3,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=320)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for seed in (17, 29, 43):
        result = run(args.engine, seed, args.rounds)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(r["after"]["expected"] > r["before"]["expected"] + 0.2 for r in results)
    assert all(r["after"]["pass"][0] > 0.7 and max(r["after"]["pass"][1:]) < 0.3 for r in results)
