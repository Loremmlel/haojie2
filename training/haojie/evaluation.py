"""冻结权重、配对换边评测；超限与失败不计作平局。"""

import json
from pathlib import Path
import time

import numpy as np
import torch

from .model import Network, collate_states, collate_options, device_for_bf16
from .native import Environment
from .opponents import Teacher
from .search import search
from .runtime.ipc import search_values


def load(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") == "haojie-classic-training-v1":
        raise ValueError("模型格式不兼容，当前需要分层解码 v2")
    weights = (
        saved["model"] if saved.get("format") == "haojie-classic-training-v2" else saved
    )
    network = Network().to(device)
    network.load_state_dict(weights)
    return network.eval()


def infer(network, request, device, contexts):
    with torch.inference_mode():
        if request.op == "encode":
            context, value, aux = network.encode(
                *collate_states([request.arrays], device)
            )
            contexts[request.context] = context
            output = torch.cat((value[:, None], aux), 1)[0]
        else:
            output = network.decode(
                contexts[request.context], *collate_options([request.arrays[0]], device)
            )[0]
        return search_values(output.view(torch.uint16).cpu().numpy()), 0


def choose(environment, network, rng, device, simulations):
    generator = search(environment, rng, simulations=simulations, depth=16, noise=False)
    answer = None
    contexts = {}
    while True:
        try:
            request = next(generator) if answer is None else generator.send(answer)
            answer = infer(network, request, device, contexts)
        except StopIteration as completed:
            root, policy, _ = completed.value
            return root.edges[int(np.argmax(policy))].command


def evaluate(args):
    if args.games <= 0 or args.games % 2:
        raise ValueError("评测局数须为正偶数，保证配对换边")
    torch.set_num_threads(1)
    device = device_for_bf16(args.device)
    model = load(args.checkpoint, device)
    scripted = args.opponent in ("easy", "medium", "hard")
    other = None if scripted else load(args.opponent, device)
    root_rng = np.random.default_rng(args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen = path.with_suffix(".model.pt")
    frozen_other = path.with_suffix(".opponent.pt")
    if (
        path.exists()
        or frozen.exists()
        or (other is not None and frozen_other.exists())
    ):
        raise FileExistsError("评测输出与冻结模型需要新的文件名")
    with frozen.open("xb") as stream:
        torch.save(model.state_dict(), stream)
    if other is not None:
        with frozen_other.open("xb") as stream:
            torch.save(other.state_dict(), stream)
    summary = {"wins": 0, "losses": 0, "draws": 0, "unknown": 0}
    with path.open("x", encoding="utf-8") as output:
        for pair in range(args.games // 2):
            seed = int(root_rng.integers(1, 2**32))
            for learner in (1, 2):
                started = time.perf_counter()
                rng = np.random.default_rng(seed ^ 0x98AA)
                with Environment(seed) as environment:
                    teacher = Teacher(args.opponent, environment) if scripted else None
                    result = environment.call("info")
                    steps = 0
                    for step in range(args.max_commands):
                        if result["winner"] is not None:
                            break
                        if result["actor"] == learner or other is not None:
                            network = model if result["actor"] == learner else other
                            command = choose(
                                environment, network, rng, device, args.simulations
                            )
                        else:
                            command = teacher.next()
                        result = environment.call("apply", command=command)
                        steps += 1
                        if result.get("rejected"):
                            raise RuntimeError(f"评测命令被拒绝: {result}")
                    winner = result["winner"]
                    category = (
                        "unknown"
                        if winner is None
                        else "draws"
                        if winner == "draw"
                        else "wins"
                        if winner == learner
                        else "losses"
                    )
                    summary[category] += 1
                    record = {
                        "pair": pair,
                        "seed": seed,
                        "learner": learner,
                        "winner": winner,
                        "steps": steps,
                        "ply": result["ply"],
                        "seconds": time.perf_counter() - started,
                        "simulations": args.simulations,
                        "checkpoint": str(frozen),
                        "source_checkpoint": args.checkpoint,
                        "opponent": args.opponent if scripted else str(frozen_other),
                        "dtype": "bfloat16",
                    }
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(json.dumps(record), flush=True)
                if teacher:
                    teacher.close()
    print(json.dumps(summary), flush=True)
