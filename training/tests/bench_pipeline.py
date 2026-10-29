"""真实轨迹的中后期窗口；固定 64/16 搜索、反馈和学习样本，比较完整流水线。"""

import argparse
import copy
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import tempfile
import time

from training.haojie.runtime.config import Config
from training.haojie.runtime.ipc import Slot, pack, unpack
from training.haojie.runtime.worker import worker
from training.haojie.search import Request
from training.haojie.native import Environment
from training.haojie.runtime.storage import GameWriter, Replay


def starting_positions(directory):
    positions = []
    paths = sorted(
        (directory / "games").glob("*/game.json"), key=lambda p: int(p.parent.name)
    )
    for path in paths[:4]:
        meta = json.loads(path.read_text(encoding="utf-8"))
        boundaries = {
            int(meta["steps"] * fraction) for fraction in (0.25, 0.5, 0.75, 0.9)
        }
        with Environment(meta["game"]["seed"]) as env:
            step = 0
            for name in meta["command_segments"]:
                for line in (
                    (path.parent / name).read_text(encoding="utf-8").splitlines()
                ):
                    result = env.call("apply", command=json.loads(line))
                    if result.get("rejected"):
                        raise RuntimeError(f"基准源轨迹重放失败: {path} {step}")
                    step += 1
                    if step in boundaries and result["winner"] is None:
                        positions.append(
                            {
                                "source_game": meta["game"]["id"],
                                "source_step": step,
                                "ply": result["ply"],
                                "environment": env.call("snapshot"),
                            }
                        )
    if len(positions) != 16:
        raise ValueError("基准需要四局已有轨迹，每局可提取四个非终局起点")
    return positions, [str(p) for p in paths[:4]]


def run(
    workers, lanes, commands, updates, saved, starts, replay, workset, record=False
):
    import numpy as np
    import torch
    from training.haojie.runtime.trainer import Trainer

    config = Config(
        workers=workers,
        games_per_worker=lanes,
        warmup_games=0,
        teacher_fraction=0,
        history_fraction=0,
        max_commands=commands,
        sample_reuse=0,
        publish_updates=updates + 1,
        save_seconds=0,
    )
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="haojie-pipeline-") as directory:
        trainer = Trainer(directory, config)
        trainer.model.load_state_dict(saved["model"])
        trainer.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        for state in trainer.optimizer.state.values():
            state["step"] = state["step"].to(device="cpu", dtype=torch.int64)
        trainer.inference_models[0] = copy.deepcopy(trainer.model).eval()
        trainer.replay = Replay(games=replay)
        trainer.rng = np.random.default_rng(818031)
        torch.cuda.reset_peak_memory_stats()
        processes, pipes, slots = [], [], []
        done, finished, outcomes = set(), set(), {}
        started = time.perf_counter()
        first_ready = None
        positions, progress_commands = {}, {}
        try:
            for i in range(workers):
                parent, child = ctx.Pipe()
                slot = Slot()
                games = [
                    {
                        "id": i * lanes + j,
                        "seed": 910001 + i * lanes + j,
                        "learner": 1,
                        "opponent": "self",
                        "difficulty": "easy",
                        "ply_limit": 0,
                    }
                    for j in range(lanes)
                ]
                restored = [
                    {
                        "game": game,
                        "environment": starts[game["id"]]["environment"],
                        "metadata": GameWriter(directory, game).metadata,
                        "rng": np.random.default_rng(
                            game["seed"] ^ 0xA53113
                        ).bit_generator.state,
                        "root_version": None,
                        "teacher": None,
                    }
                    for game in games
                ]
                process = ctx.Process(
                    target=worker,
                    args=(i, child, slot.name, directory, config, games, restored),
                )
                process.start()
                child.close()
                processes.append(process)
                pipes.append(parent)
                slots.append(slot)
            while len(finished) < workers:
                jobs, replies, keys = [], [], []
                for pipe in wait(
                    [p for i, p in enumerate(pipes) if i not in finished], 30
                ):
                    index = pipes.index(pipe)
                    message = pipe.recv()
                    if message[0] == "error":
                        raise RuntimeError(message[1])
                    if message[0] == "saved":
                        finished.add(index)
                        continue
                    if message[0] == "grow":
                        slots[index].close()
                        slots[index] = Slot(size=message[1] * 2)
                        pipe.send(slots[index].name)
                        continue
                    first_ready = first_ready or time.perf_counter()
                    _, requests, shapes, events, progress = message
                    progress_commands[index] = progress["commands"]
                    for event in events:
                        if event["type"] == "done":
                            game = event["game"]["id"]
                            done.add(game)
                            # 保留逐局实际工作量与结果，确保并发拓扑未偷偷减少任务。
                            outcomes[game] = {
                                k: event[k] for k in ("steps", "ply", "winner")
                            }
                    if all(
                        i in done for i in range(index * lanes, (index + 1) * lanes)
                    ):
                        pipe.send({"op": "pause", "versions": []})
                        continue
                    arrays = unpack(slots[index], shapes)
                    first = len(jobs)
                    at = 0
                    for j, request in enumerate(requests):
                        end = at + request["arrays"]
                        values = tuple(arrays[at:end])
                        at = end
                        trainer.resolve(index, request)
                        jobs.append(
                            (
                                0,
                                Request(request["op"], request["context"], values),
                                (index, request["lane"]),
                            )
                        )
                        root = request["root"]
                        number = positions.get(root, 0)
                        positions[root] = number + 1
                        key = (root, number)
                        keys.append(key)
                        if record:
                            workset[key] = (
                                request["op"],
                                request["context"],
                                tuple((v.shape, v.dtype.str) for v in values),
                                None,
                            )
                        else:
                            expected = workset[key]
                            if (
                                request["op"] != expected[0]
                                or request["context"] != expected[1]
                                or tuple((v.shape, v.dtype.str) for v in values)
                                != expected[2]
                            ):
                                raise AssertionError(
                                    "并发拓扑改变了公开状态键或请求形状"
                                )
                    replies.append((index, first, len(requests)))
                outputs = trainer.infer(jobs) if jobs else []
                for i, key in enumerate(keys):
                    if record:
                        workset[key] = (*workset[key][:3], outputs[i].copy())
                    else:
                        # 批大小会造成 BF16 舍入差异；控制搜索路径，实际 GPU 工作照常执行。
                        outputs[i] = workset[key][3]
                for index, first, count in replies:
                    pipes[index].send(
                        {
                            "op": "work",
                            "shapes": pack(
                                slots[index], outputs[first : first + count]
                            ),
                            "versions": [0] * count,
                            "assignments": [],
                        }
                    )
                due = min(
                    updates,
                    sum(progress_commands.values())
                    * updates
                    // (len(starts) * commands),
                )
                while trainer.stats["updates"] < due:
                    trainer.credit = config.batch
                    trainer.learn()
            # 自然终局可能提前结束窗口，仍执行约定的同量学习。
            while trainer.stats["updates"] < updates:
                trainer.credit = config.batch
                trainer.learn()
            elapsed = time.perf_counter() - started
            result = {
                "workers": workers,
                "lanes": lanes,
                "seconds": elapsed,
                "startup_seconds": first_ready - started,
                "commands": sum(g["steps"] for g in outcomes.values()),
                "inference_positions": trainer.stats["inference_roots"],
                "decoder_calls": trainer.stats["inference_decodes"],
                "inference_batches": trainer.stats["inference_batches"],
                "inference_seconds": trainer.stats["inference_seconds"],
                "learning_seconds": trainer.stats["learning_seconds"],
                "updates": trainer.stats["updates"],
                "trained_samples": trainer.stats["trained_samples"],
                "value_samples": trainer.stats["value_samples"],
                "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                "games": dict(sorted(outcomes.items())),
            }
            trace = {}
            for game in sorted(outcomes):
                root = Path(directory) / "games" / str(game)
                metadata = json.loads((root / "game.json").read_text())
                trace[game] = [
                    json.loads(line)
                    for name in metadata["command_segments"]
                    for line in (root / name).read_text(encoding="utf-8").splitlines()
                ]
            result["_trace"] = trace
            result["_learning_rng"] = trainer.rng.bit_generator.state
            result["commands_per_second"] = result["commands"] / elapsed
            result["trained_samples_per_second"] = result["trained_samples"] / elapsed
            return result
        finally:
            for process in processes:
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join()
            for pipe in pipes:
                pipe.close()
            for slot in slots:
                slot.close()


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", type=int, default=8)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    starts, replay = starting_positions(checkpoint.parent)
    print(
        json.dumps(
            {
                "event": "workload",
                "simulations": [64, 16],
                "full_fraction": 0.25,
                "commands_per_position": args.commands,
                "learning_updates": args.updates,
                "starts": [
                    {k: v for k, v in s.items() if k != "environment"} for s in starts
                ],
            }
        ),
        flush=True,
    )
    workset = {}
    captured = run(
        1, 16, args.commands, args.updates, saved, starts, replay, workset, record=True
    )

    def signature(result):
        return (
            result.pop("_trace"),
            result.pop("_learning_rng"),
            result["games"],
            result["inference_positions"],
            result["decoder_calls"],
            result["updates"],
            result["trained_samples"],
            result["value_samples"],
        )

    baseline = signature(captured)
    print(json.dumps({"event": "captured", **captured}), flush=True)
    configurations = [(1, 16), (2, 8), (4, 4)]
    for trial in range(args.trials):
        for workers, lanes in configurations[trial % 3 :] + configurations[: trial % 3]:
            result = run(
                workers,
                lanes,
                args.commands,
                args.updates,
                saved,
                starts,
                replay,
                workset,
            )
            if signature(result) != baseline:
                raise AssertionError("拓扑改变了冻结轨迹、搜索或学习工作量")
            print(json.dumps({"trial": trial, **result}), flush=True)


if __name__ == "__main__":
    main()
