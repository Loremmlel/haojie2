"""整体架构验收：冻结同一搜索反馈、命令与学习工作量，最后交错比较旧/新运行时。"""

import argparse
import copy
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from training.haojie.native import Environment
from training.haojie.runtime.config import Config
from training.haojie.runtime.gpu import Executor, Remote, serve
from training.haojie.runtime.storage import GameWriter, Replay
from training.haojie.runtime.trainer import Trainer
from training.haojie.runtime.worker import worker


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


def benchmark_gpu(control, actors, directory, config, initial, workset_path, record):
    # 仅基准替换反馈；GPU 仍执行所有真实推理。正式服务没有录制逻辑。
    workset = {} if record else pickle.loads(Path(workset_path).read_bytes())
    positions = {}
    initialize, infer = Executor.__init__, Executor.infer

    def setup(self, *args):
        initialize(self, *args)
        self.inference_models[0] = copy.deepcopy(self.model).eval()

    def controlled(self, jobs):
        outputs = infer(self, jobs)
        for i, (_, request, owner) in enumerate(jobs):
            root = self.bindings[owner][0]
            number = positions.get(root, 0)
            positions[root] = number + 1
            key = root, number
            descriptor = (
                request.op,
                request.context,
                tuple((v.shape, v.dtype.str) for v in request.arrays),
            )
            if record:
                workset[key] = (*descriptor, outputs[i].copy())
            else:
                if descriptor != workset[key][:3]:
                    raise AssertionError("架构改变了公开状态键或请求形状")
                outputs[i] = workset[key][3]
        return outputs

    Executor.__init__, Executor.infer = setup, controlled
    try:
        serve(control, actors, directory, config, initial)
    finally:
        if record:
            Path(workset_path).write_bytes(pickle.dumps(workset))


def run_async(saved, starts, replay, workset, commands, updates, record=False):
    config = Config(
        workers=4,
        games_per_worker=len(starts) // 4,
        warmup_games=0,
        teacher_fraction=0,
        history_fraction=0,
        max_commands=commands,
        sample_reuse=0,
        save_seconds=0,
        publish_updates=updates + 1,
    )
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="haojie-async-bench-") as directory:
        trainer = Trainer(directory, config)
        trainer.model.load_state_dict(saved["model"])
        trainer.restore_optimizer(copy.deepcopy(saved["optimizer"]))
        trainer.replay = Replay(games=replay)
        trainer.rng = np.random.default_rng(818031)
        controls, processes, outcomes = {}, {}, {}
        parent, child = ctx.Pipe()
        pairs = [ctx.Pipe() for _ in range(4)]
        remote = Remote(parent, lambda: {0})
        payload = {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": None,
            "stats": trainer.stats,
            "published": 0,
            "retained": [0],
        }
        gpu = ctx.Process(
            target=benchmark_gpu,
            args=(
                child,
                [p[0] for p in pairs],
                directory,
                config,
                payload,
                str(workset),
                record,
            ),
        )
        gpu.start()
        child.close()
        if remote.receive()["op"] != "ready":
            raise RuntimeError("GPU 初始化失败")
        # 与旧基准一致：模型准备不计时，actor 启动及之后全部计时。
        started = time.perf_counter()
        first_ready = None
        progress, snapshots = {}, {}
        try:
            for index in range(4):
                local, other = ctx.Pipe()
                games = [
                    {
                        "id": index * config.games_per_worker + lane,
                        "seed": 910001 + index * config.games_per_worker + lane,
                        "learner": 1,
                        "opponent": "self",
                        "difficulty": "easy",
                        "ply_limit": 0,
                    }
                    for lane in range(config.games_per_worker)
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
                    args=(
                        index,
                        other,
                        pairs[index][1],
                        directory,
                        config,
                        games,
                        restored,
                    ),
                )
                process.start()
                other.close()
                controls[index], processes[index] = local, process
                pairs[index][0].close()
                pairs[index][1].close()
            while (
                len(outcomes) < len(starts)
                or remote.learning
                or trainer.stats["updates"] < updates
            ):
                for connection in wait([parent, *controls.values()], 0.01):
                    if connection is parent:
                        remote.receive()
                    else:
                        index = next(i for i, c in controls.items() if c is connection)
                        message = connection.recv()
                        first_ready = first_ready or time.perf_counter()
                        if message[0] == "error":
                            raise RuntimeError(message[1])
                        if message[0] == "events":
                            progress[index] = message[2]
                            for event in message[1]:
                                if event["type"] == "done":
                                    outcomes[event["game"]["id"]] = {
                                        k: event[k] for k in ("steps", "ply", "winner")
                                    }
                while remote.completed:
                    trainer.stats.update(remote.completed.popleft()["stats"])
                due = (
                    updates
                    if len(outcomes) == len(starts)
                    else min(
                        updates,
                        sum(progress.values()) * updates // (len(starts) * commands),
                    )
                )
                if not remote.learning and trainer.stats["updates"] < due:
                    remote.learn(trainer.replay.batch(config.batch, trainer.rng))
            # 计时覆盖与旧入口对应的 actor 静止确认。
            for connection in controls.values():
                connection.send({"op": "save"})
            while len(snapshots) < 4:
                for connection in wait(
                    [c for i, c in controls.items() if i not in snapshots], 10
                ):
                    index = next(i for i, c in controls.items() if c is connection)
                    message = connection.recv()
                    if message[0] == "error":
                        raise RuntimeError(message[1])
                    if message[0] == "saved":
                        snapshots[index] = message[1]
            stats = remote.call({"op": "stats"})["stats"]
            elapsed = time.perf_counter() - started
            trace = {}
            for game in sorted(outcomes):
                root = Path(directory) / "games" / str(game)
                meta = json.loads((root / "game.json").read_text(encoding="utf-8"))
                trace[game] = [
                    json.loads(line)
                    for name in meta["command_segments"]
                    for line in (root / name).read_text(encoding="utf-8").splitlines()
                ]
            return {
                "workers": 4,
                "lanes": config.games_per_worker,
                "seconds": elapsed,
                "startup_seconds": first_ready - started,
                "commands": sum(g["steps"] for g in outcomes.values()),
                "inference_positions": stats["inference_roots"],
                "decoder_calls": stats["inference_decodes"],
                **{
                    k: stats[k]
                    for k in (
                        "inference_batches",
                        "inference_seconds",
                        "learning_seconds",
                        "updates",
                        "trained_samples",
                        "value_samples",
                        "cuda_peak_bytes",
                    )
                },
                "queue_seconds": stats["inference_queue_seconds"],
                "games": dict(sorted(outcomes.items())),
                "_trace": trace,
                "_learning_rng": trainer.rng.bit_generator.state,
                "commands_per_second": sum(g["steps"] for g in outcomes.values())
                / elapsed,
                "trained_samples_per_second": stats["trained_samples"] / elapsed,
            }
        finally:
            if gpu.is_alive():
                parent.send({"op": "stop"})
            gpu.join(10)
            if gpu.is_alive():
                gpu.terminate()
                gpu.join()
            for index, process in processes.items():
                if process.is_alive():
                    controls[index].send({"op": "stop"})
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join()
            for connection in controls.values():
                connection.close()
            remote.close()


def run_baseline(root, checkpoint, workset, commands, updates, count, record):
    # 基线来自已提交代码的独立导出目录；不把旧实现重新放进生产包。
    program = """
import json, pickle, sys, torch
from pathlib import Path
from training.tests.bench_pipeline import starting_positions, run
checkpoint, workset, commands, updates, count, record, output = sys.argv[1:]
commands, updates, count, record = map(int, (commands, updates, count, record))
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
starts, replay = starting_positions(Path(checkpoint).parent)
starts *= count // 16
feedback = {} if record else pickle.loads(Path(workset).read_bytes())
result = run(4, count // 4, commands, updates, saved, starts, replay, feedback, record=bool(record))
if record:
    Path(workset).write_bytes(pickle.dumps(feedback))
Path(output).write_bytes(pickle.dumps(result))
"""
    with tempfile.TemporaryDirectory(prefix="haojie-baseline-result-") as directory:
        path = Path(directory) / "result.pkl"
        subprocess.run(
            [
                sys.executable,
                "-c",
                program,
                str(checkpoint),
                str(workset),
                str(commands),
                str(updates),
                str(count),
                str(int(record)),
                str(path),
            ],
            cwd=root,
            env={
                **os.environ,
                "HAOJIE_ENGINE": str(
                    Path(
                        os.environ.get(
                            "HAOJIE_ENGINE",
                            "artifacts/native-target/release/haojie_native.dll",
                        )
                    ).resolve()
                ),
            },
            check=True,
        )
        return pickle.loads(path.read_bytes())


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--commands", type=int, default=8)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
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
                "commands": args.commands,
                "learning_updates_per_16_games": args.updates,
                "starts": [
                    {k: v for k, v in s.items() if k != "environment"} for s in starts
                ],
            }
        ),
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="haojie-frozen-feedback-") as directory:
        for count in (16, 64):
            path = Path(directory) / f"{count}.pkl"
            updates = args.updates * (count // 16)

            def execute(backend, record=False):
                if backend == "baseline":
                    return run_baseline(
                        Path(args.baseline).resolve(),
                        checkpoint,
                        path,
                        args.commands,
                        updates,
                        count,
                        record,
                    )
                return run_async(
                    saved,
                    starts * (count // 16),
                    replay,
                    path,
                    args.commands,
                    updates,
                    record,
                )

            reference = execute("baseline" if args.baseline else "asynchronous", True)
            expected = signature(reference)
            print(
                json.dumps({"event": "captured", "concurrency": count, **reference}),
                flush=True,
            )
            for trial in range(args.trials):
                backends = (
                    ["baseline", "asynchronous"] if args.baseline else ["asynchronous"]
                )
                if trial % 2:
                    backends.reverse()
                for backend in backends:
                    result = execute(backend)
                    if signature(result) != expected:
                        raise AssertionError("整体改造改变了冻结轨迹、搜索或学习工作量")
                    print(
                        json.dumps(
                            {
                                "trial": trial,
                                "backend": backend,
                                "concurrency": count,
                                **result,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
