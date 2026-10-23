"""可解析短环境的独立初始化前后对照；正式原生 MC 搜索与 Trainer，样本只来自自身采样。"""

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from haojie_training.console.memory import SamplePool, Trajectory
from haojie_training.data import collate_examples
from haojie_training.model import NETWORK_VERSION, ModelConfig, PolicyValueNet
from haojie_training.native.client import Client
from haojie_training.native.execution import PolicyInference
from haojie_training.runtime import Trainer, resolve_device


def episode(client, infer, rng, case, learning):
    trajectory = Trajectory(rng.randrange(1, 2**32), capacity=64, context={"rules": "exercise"})
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


def run(engine, seed, rounds=320, device="cpu", layers=2):
    began = time.perf_counter()
    torch.manual_seed(seed)
    device = resolve_device(device)
    model = PolicyValueNet(replace(ModelConfig.tiny(), layers=layers)).to(device)
    trainer = Trainer(model, device, objective="decomposed-mc-q-v2")
    infer = PolicyInference(model, device)
    rng = random.Random(seed)
    pool = SamplePool(16 * 2**20)
    with Client(engine) as client:

        def evaluate():
            model.eval()
            evaluation_rng = random.Random(7001)
            rows = [episode(client, infer, evaluation_rng, i % 3, False)[0] for i in range(240)]
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
            model.eval()
            for i in range(12):
                result, rows = episode(client, infer, rng, i % 3, True)
                backtracks += result["metrics"]["backtracks"]
                for row in rows:
                    pool.add(row)
            model.train()
            for _ in range(pool.budget(16, 64)):
                if not pool.rows:
                    break
                batch = collate_examples(pool.batch(16, rng), model.config)
                trainer.step({key: value.to(device) for key, value in batch.items()})
            pool.expire(generation + 1)
        after = evaluate()
    return {
        "seed": seed,
        "device": str(device),
        "precision": "bf16",
        "network": NETWORK_VERSION,
        "config": asdict(model.config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "before": before,
        "after": after,
        "updates": trainer.updates,
        "consumption": pool.snapshot(),
        "backtracks": backtracks,
        "health": trainer.health(),
        "seconds": time.perf_counter() - began,
        "reference_expected": (0.6 + 0.8 + 0.8) / 3,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    args = parser.parse_args()
    if args.layers < 1:
        parser.error("层数须为正整数")
    torch.set_num_threads(1)
    results = []
    for seed in args.seeds:
        result = run(args.engine, seed, args.rounds, args.device, args.layers)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(r["after"]["expected"] > r["before"]["expected"] + 0.2 for r in results)
    assert all(r["after"]["pass"][0] > 0.7 and max(r["after"]["pass"][1:]) < 0.3 for r in results)
