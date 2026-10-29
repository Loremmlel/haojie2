"""冻结搜索反馈工作集；各拓扑仍执行完整 Rust 搜索和 BF16 推理。"""

import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import tempfile
import time

from training.haojie.runtime.config import Config
from training.haojie.runtime.ipc import Slot, pack, unpack
from training.haojie.runtime.worker import worker


def run(workers, lanes, commands, simulations, checkpoint, workset, record=False):
    import numpy as np
    import torch
    from training.haojie.runtime.trainer import Trainer

    config = Config(workers=workers, games_per_worker=lanes, candidates=32,
                    simulations=simulations, fast_simulations=simulations, full_fraction=1,
                    warmup_games=0, teacher_fraction=0, history_fraction=0,
                    max_commands=commands, sample_reuse=0, save_seconds=0)
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="haojie-pipeline-") as directory:
        trainer = Trainer(directory, config)
        if checkpoint:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            trainer.model.load_state_dict(saved.get("model", saved))
        trainer.inference_models[0] = trainer.model.eval()
        processes, pipes, slots = [], [], []
        done, finished, outcomes = set(), set(), {}
        started = time.perf_counter()
        first_ready = None
        positions = {}
        try:
            for i in range(workers):
                parent, child = ctx.Pipe()
                slot = Slot()
                games = [{"id": i*lanes+j, "seed": 910001+i*lanes+j, "learner": 1,
                          "opponent": "self", "difficulty": "easy"} for j in range(lanes)]
                process = ctx.Process(target=worker, args=(i, child, slot.name, directory, config, games, []))
                process.start()
                child.close()
                processes.append(process)
                pipes.append(parent)
                slots.append(slot)
            while len(finished) < workers:
                jobs, replies, keys = [], [], []
                for pipe in wait([p for i,p in enumerate(pipes) if i not in finished], 30):
                    index = pipes.index(pipe)
                    message = pipe.recv()
                    if message[0] == "error":
                        raise RuntimeError(message[1])
                    if message[0] == "saved":
                        finished.add(index)
                        continue
                    if message[0] == "grow":
                        slots[index].close()
                        slots[index] = Slot(size=message[1]*2)
                        pipe.send(slots[index].name)
                        continue
                    first_ready = first_ready or time.perf_counter()
                    _, requests, shapes, events, _ = message
                    for event in events:
                        if event["type"] == "done":
                            game = event["game"]["id"]
                            done.add(game)
                            # 保留逐局实际工作量与结果，确保并发拓扑未偷偷减少任务。
                            outcomes[game] = {k:event[k] for k in ("steps","ply","winner")}
                    if all(i in done for i in range(index*lanes,(index+1)*lanes)):
                        pipe.send({"op":"pause","versions":[]})
                        continue
                    arrays = unpack(slots[index], shapes)
                    first = len(jobs)
                    jobs.extend((0,arrays[j*2],arrays[j*2+1]) for j in range(len(requests)))
                    for j, request in enumerate(requests):
                        root = request["root"]
                        number = positions.get(root,0)
                        positions[root] = number+1
                        key = (root,number)
                        keys.append(key)
                        if record:
                            workset[key] = (arrays[j*2].copy(),arrays[j*2+1].copy(),None)
                        else:
                            expected = workset[key]
                            if not np.array_equal(arrays[j*2],expected[0]) or not np.array_equal(arrays[j*2+1],expected[1]):
                                raise AssertionError("并发拓扑改变了冻结搜索的输入")
                    replies.append((index, first, len(requests)))
                outputs = trainer.infer(jobs) if jobs else []
                for i,key in enumerate(keys):
                    if record:
                        workset[key] = (*workset[key][:2],outputs[i].copy())
                    else:
                        # 批大小会造成 BF16 舍入差异；控制搜索路径，实际 GPU 工作照常执行。
                        outputs[i] = workset[key][2]
                for index, first, count in replies:
                    pipes[index].send({"op":"work","shapes":pack(slots[index],outputs[first:first+count]),
                                       "versions":[0]*count,"assignments":[]})
            elapsed = time.perf_counter()-started
            result = {"workers":workers,"lanes":lanes,"seconds":elapsed,
                      "startup_seconds":first_ready-started,"commands":sum(g["steps"] for g in outcomes.values()),
                      "inference_positions":trainer.stats["inference_roots"],
                      "inference_batches":trainer.stats["inference_batches"],
                      "games":dict(sorted(outcomes.items()))}
            trace = {}
            for game in sorted(outcomes):
                root = Path(directory) / "games" / str(game)
                metadata = json.loads((root / "game.json").read_text())
                trace[game] = [json.loads(line) for name in metadata["command_segments"]
                               for line in (root / name).read_text(encoding="utf-8").splitlines()]
            result["_trace"] = trace
            result["commands_per_second"] = result["commands"]/elapsed
            return result
        finally:
            for process in processes:
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join()
            for pipe in pipes: pipe.close()
            for slot in slots: slot.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    workset = {}
    captured = run(1,8,args.commands,args.simulations,args.checkpoint,workset,record=True)
    baseline = (captured.pop("_trace"),captured["games"],captured["inference_positions"])
    configurations = [(1,8),(2,4),(4,2)]
    for trial in range(args.trials):
        for workers, lanes in configurations[trial%3:]+configurations[:trial%3]:
            result = run(workers,lanes,args.commands,args.simulations,args.checkpoint,workset)
            signature = (result.pop("_trace"),result["games"],result["inference_positions"])
            if signature != baseline:
                raise AssertionError("拓扑改变了固定种子轨迹或搜索工作量，不能比较吞吐")
            print(json.dumps({"trial":trial,**result}),flush=True)


if __name__ == "__main__": main()
