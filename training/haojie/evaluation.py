"""冻结权重、配对换边评测；超限与失败不计作平局。"""

import json
import os
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .model import Network, collate_states, collate_options, device_for_bf16
from .native_pool import NativePool
from .runtime.config import Config
from .runtime.storage import atomic_json


def load(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if "format" in saved and saved["format"] != "haojie-classic-training-v3":
        raise ValueError("评测只接受 v3 检查点或历史池权重")
    weights = saved["model"] if "format" in saved else saved
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
        return output.view(torch.uint16).cpu().numpy()


def paired_games(seed, games=4, difficulties=("easy", "medium", "hard")):
    if games <= 0 or games % 2:
        raise ValueError("评测局数须为正偶数，保证配对换边")
    rng = np.random.default_rng(seed)
    for difficulty in difficulties:
        for pair in range(games // 2):
            game_seed = int(rng.integers(1, 2**32))
            for learner in (1, 2):
                yield dict(
                    difficulty=difficulty, pair=pair, seed=game_seed, learner=learner
                )


def paired_summary(results):
    summary = {}
    for difficulty in ("easy", "medium", "hard"):
        rows = [r for r in results if r["difficulty"] == difficulty]
        counts = dict(
            pairs=0,
            wins=0,
            losses=0,
            draws=0,
            unknown=sum(r.get("winner") is None and not r.get("error") for r in rows),
            errors=sum(bool(r.get("error")) for r in rows),
        )
        for pair in {r["pair"] for r in rows}:
            mirror = [r for r in rows if r["pair"] == pair]
            if (
                len(mirror) != 2
                or {r["learner"] for r in mirror} != {1, 2}
                or len({r["seed"] for r in mirror}) != 1
                or any(r.get("winner") is None or r.get("error") for r in mirror)
            ):
                continue
            counts["pairs"] += 1
            for row in mirror:
                counts[
                    (
                        "draws"
                        if row["winner"] == "draw"
                        else "wins" if row["winner"] == row["learner"] else "losses"
                    )
                ] += 1
        summary[difficulty] = counts
    return summary


def native_game(row, number, scripted=True):
    game = dict(
        id=number,
        seed=row["seed"],
        learner=row["learner"],
        opponent="teacher" if scripted else "history",
        difficulty=row["difficulty"] if scripted else "easy",
        evaluation=True,
        ply_limit=0,
    )
    if not scripted:
        game["version"] = 1
    return game


def play_game(model, other, device, game, simulations, max_commands, progress=None):
    started = time.perf_counter()
    last_progress = started
    config = Config(
        workers=1,
        games_per_worker=1,
        warmup_games=0,
        simulations=simulations,
        fast_simulations=simulations,
        temperature=0,
        max_commands=max_commands,
        device=str(device),
        evaluation_seconds=0,
    )
    with NativePool(config) as pool:
        pool.assign(
            0,
            game,
            rng=np.random.default_rng(game["seed"] ^ 0x98AA).bit_generator.state,
        )
        contexts = {}
        current_root = None
        while True:
            batch = pool.take_requests(wait_ms=10)
            if batch is not None:
                with batch:
                    outputs, versions = [], []
                    for request in batch.requests:
                        root = request.game_id, request.step
                        if current_root != root:
                            contexts.clear()
                            current_root = root
                        version = 1 if request.model == 1 else 0
                        outputs.append(
                            infer(
                                other if version == 1 else model,
                                request,
                                device,
                                contexts,
                            )
                        )
                        versions.append(version)
                    batch.reply(outputs, versions)
            records = pool.take_records()
            ended = next((r for r in records if r["done"]), None)
            pool.ack_records([r["lane"] for r in records])
            if ended is not None:
                return dict(
                    winner=ended["info"]["winner"],
                    steps=ended["steps"],
                    ply=ended["info"]["ply"],
                    seconds=time.perf_counter() - started,
                )
            if progress and records and time.perf_counter() - last_progress >= 10:
                progress(records[-1]["steps"])
                last_progress = time.perf_counter()


def evaluate_round(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "round.json").read_text(encoding="utf-8"))
    status = dict(
        **manifest,
        state="running",
        round=directory.name,
        pid=os.getpid(),
        process_created=psutil.Process().create_time(),
        completed=0,
        total=manifest["games"] * 3,
        results=[]
    )

    def report():
        status["summary"] = paired_summary(status["results"])
        atomic_json(directory / "status.json", status)
        atomic_json(directory.parent / "status.json", status)

    try:
        report()
        torch.set_num_threads(1)
        device = device_for_bf16(manifest["device"])
        model = load(directory / "model.pt", device)
        with (directory / "results.jsonl").open("x", encoding="utf-8") as output:
            for number, row in enumerate(
                paired_games(manifest["seed"], manifest["games"])
            ):
                status["current"] = {**row, "steps": 0}
                report()

                def progress(steps):
                    status["current"]["steps"] = steps
                    report()

                try:
                    result = play_game(
                        model,
                        None,
                        device,
                        native_game(row, number),
                        manifest["simulations"],
                        manifest["max_commands"],
                        progress,
                    )
                except Exception as error:
                    result = {"winner": None, "error": str(error)}
                record = {**row, **result, "version": manifest["version"]}
                output.write(json.dumps(record) + "\n")
                output.flush()
                status["results"].append(record)
                status["completed"] += 1
                report()
        status.update(state="completed", current=None, finished_at=time.time())
        report()
    except Exception as error:
        status.update(state="failed", error=str(error), finished_at=time.time())
        report()
        raise


def evaluate(args):
    if args.games <= 0 or args.games % 2:
        raise ValueError("评测局数须为正偶数，保证配对换边")
    torch.set_num_threads(1)
    device = device_for_bf16(args.device)
    model = load(args.checkpoint, device)
    scripted = args.opponent in ("easy", "medium", "hard")
    other = None if scripted else load(args.opponent, device)
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
        for number, row in enumerate(
            paired_games(args.seed, args.games, (args.opponent,))
        ):
            result = play_game(
                model,
                other,
                device,
                native_game(row, number, scripted),
                args.simulations,
                args.max_commands,
            )
            winner = result["winner"]
            category = (
                "unknown"
                if winner is None
                else (
                    "draws"
                    if winner == "draw"
                    else "wins" if winner == row["learner"] else "losses"
                )
            )
            summary[category] += 1
            record = {
                **row,
                **result,
                "simulations": args.simulations,
                "checkpoint": str(frozen),
                "source_checkpoint": args.checkpoint,
                "opponent": args.opponent if scripted else str(frozen_other),
                "dtype": "bfloat16",
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(json.dumps(record), flush=True)
    print(json.dumps(summary), flush=True)
