"""CPU 协调循环；神经请求从 actor 直接流向 GPU，不经过此循环。"""

import copy
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import signal
import time
from dataclasses import asdict

import torch

from .gpu import Remote, serve
from .worker import worker


def run(trainer):
    config = trainer.config
    ctx = mp.get_context("spawn")
    controls, actors, snapshots, active_games = {}, {}, {}, {}
    totals = {}
    gpu_connections, worker_connections = [], []
    parent, child = ctx.Pipe()
    initial = copy.deepcopy(trainer.stats)
    started = last_status = last_save = time.perf_counter()
    mode = None
    gpu = None
    completed_run = False
    previous_handler = signal.signal(
        signal.SIGINT, lambda *_: setattr(trainer, "stop_requested", True)
    )

    def retained():
        return (
            trainer.checkpoint_versions
            | {e["version"] for e in trainer.pool.entries}
            | {game["version"] for game in active_games.values() if "version" in game}
        )

    remote = Remote(parent, retained)

    def complete_learning():
        while remote.completed:
            result = remote.completed.popleft()
            trainer.stats.update(result["stats"])
            trainer.credit = max(0, trainer.credit - result["samples"])
            publish = trainer.stats["updates"] % config.publish_updates == 0
            if trainer.phase == "imitation":
                trainer.imitation_updates += 1
                if trainer.imitation_updates >= config.warmup_updates:
                    trainer.phase = "self"
                    trainer.credit = 0
                    publish = True
            if publish:
                trainer.publish()

    def handle(index, message):
        kind = message[0]
        if kind == "error":
            snapshots[index] = message[2]
            totals[index] = message[3]
            raise RuntimeError(message[1])
        if kind == "saved":
            snapshots[index], totals[index] = message[1:]
        elif kind == "events":
            totals[index] = message[2]
            for event in message[1]:
                trainer.event(event)
                if event["type"] == "done":
                    active_games.pop((index, event["lane"]), None)
        trainer.stats["commands"] = initial["commands"] + sum(totals.values())

    def checkpoint():
        state = remote.call({"op": "snapshot"})["state"]
        complete_learning()
        trainer.model.load_state_dict(state["model"])
        trainer.restore_optimizer(state["optimizer"])
        trainer.remote_torch_rng = state["torch_rng"]
        trainer.remote_cuda_rng = state["cuda_rng"]
        trainer.stats.update(state["stats"])
        trainer.checkpoint(snapshots)

    try:
        for _ in range(config.workers):
            a, b = ctx.Pipe()
            gpu_connections.append(a)
            worker_connections.append(b)
        payload = {
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "torch_rng": getattr(trainer, "remote_torch_rng", torch.get_rng_state()),
            "cuda_rng": getattr(trainer, "remote_cuda_rng", None),
            "stats": trainer.stats,
            "published": trainer.published,
            "retained": list(retained()),
        }
        gpu = ctx.Process(
            target=serve,
            args=(child, gpu_connections, str(trainer.directory), config, payload),
            name="haojie-gpu",
        )
        gpu.start()
        child.close()
        message = remote.receive()
        if message["op"] != "ready":
            raise RuntimeError("GPU 服务未完成初始化")
        trainer.remote = remote
        for index in range(config.workers):
            local, other = ctx.Pipe()
            previous = {
                v["lane"]: v for v in (trainer.saved_workers or {}).get(index, [])
            }
            games = [
                previous[i]["game"] if i in previous else trainer.game()
                for i in range(config.games_per_worker)
            ]
            for i, game in enumerate(games):
                if game is not None:
                    active_games[index, i] = game
            process = ctx.Process(
                target=worker,
                args=(
                    index,
                    other,
                    worker_connections[index],
                    str(trainer.directory),
                    config,
                    games,
                    [previous.get(i) for i in range(config.games_per_worker)],
                ),
                name=f"haojie-cpu-{index}",
            )
            process.start()
            other.close()
            controls[index], actors[index] = local, process
            worker_connections[index].close()
            gpu_connections[index].close()
        print(
            json.dumps(
                {
                    "event": "started",
                    "directory": str(trainer.directory),
                    "config": asdict(config),
                    "runtime": "asynchronous",
                    **trainer.model.description(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        while True:
            now = time.perf_counter()
            stopped = (
                trainer.stop_requested
                or (config.seconds and now - started >= config.seconds)
                or (
                    config.games
                    and trainer.stats["completed"] - initial["completed"]
                    >= config.games
                )
                or (
                    config.updates
                    and trainer.stats["updates"] - initial["updates"] >= config.updates
                )
                or (trainer.directory / "STOP").exists()
            )
            if mode is None and (
                stopped
                or (config.save_seconds and now - last_save >= config.save_seconds)
            ):
                mode = "pause" if stopped else "save"
                for connection in controls.values():
                    connection.send({"op": "save"})
            connections = [c for i, c in controls.items() if i not in snapshots]
            for connection in wait([parent, *connections], timeout=0.01):
                if connection is parent:
                    remote.receive()
                else:
                    index = next(i for i, c in controls.items() if c is connection)
                    handle(index, connection.recv())
            complete_learning()
            for index, process in actors.items():
                if process.exitcode is not None and index not in snapshots:
                    raise RuntimeError(f"actor {index} 意外退出 {process.exitcode}")
            if gpu.exitcode is not None:
                raise RuntimeError(f"GPU 服务意外退出 {gpu.exitcode}")
            if mode:
                if len(snapshots) == len(actors) and not remote.learning:
                    checkpoint()
                    if mode == "pause":
                        break
                    for connection in controls.values():
                        connection.send({"op": "resume"})
                    snapshots.clear()
                    mode = None
                    last_save = time.perf_counter()
            else:
                for index, connection in controls.items():
                    assignments = []
                    for lane in range(config.games_per_worker):
                        if (index, lane) not in active_games:
                            game = trainer.game()
                            if game is not None:
                                active_games[index, lane] = game
                                assignments.append({"lane": lane, "game": game})
                    if assignments:
                        connection.send({"op": "assign", "assignments": assignments})
                allowed = trainer.phase == "imitation" or (
                    trainer.phase == "self" and trainer.credit >= config.batch
                )
                reached = (
                    config.updates
                    and trainer.stats["updates"] - initial["updates"] >= config.updates
                )
                if allowed and not reached and not remote.learning:
                    batch = trainer.replay.batch(config.batch, trainer.rng)
                    if batch:
                        remote.learn(batch)
            if now - last_status >= 5:
                trainer.stats.update(remote.call({"op": "stats"})["stats"])
                complete_learning()
                trainer.status("running", started, initial)
                last_status = now
        trainer.status("saved", started, initial)
        completed_run = True
        return trainer.stats
    except BaseException as error:
        trainer.status("failed", started, initial, error=str(error))
        # 其他 actor 尽量交还正式命令边界；GPU 不可用时保留上一完整检查点。
        for index, connection in controls.items():
            if index not in snapshots and actors[index].is_alive():
                try:
                    connection.send({"op": "save"})
                except OSError:
                    pass
        deadline = time.perf_counter() + 10
        while (
            gpu is not None
            and gpu.is_alive()
            and len(snapshots) < len(actors)
            and time.perf_counter() < deadline
        ):
            for connection in wait(
                [c for i, c in controls.items() if i not in snapshots], 0.1
            ):
                index = next(i for i, c in controls.items() if c is connection)
                try:
                    handle(index, connection.recv())
                except (RuntimeError, EOFError, OSError):
                    pass
        if gpu is not None and gpu.is_alive() and len(snapshots) == len(actors):
            while remote.learning:
                remote.receive()
            complete_learning()
            checkpoint()
        raise
    finally:
        if gpu is not None and gpu.is_alive():
            try:
                parent.send({"op": "stop"})
            except OSError:
                pass
            gpu.join(5)
            if gpu.is_alive():
                gpu.terminate()
                gpu.join()
        for index, process in actors.items():
            if process.is_alive():
                try:
                    controls[index].send({"op": "stop"})
                except OSError:
                    pass
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join()
        for connection in [*controls.values(), *gpu_connections, *worker_connections]:
            connection.close()
        remote.close()
        if hasattr(trainer, "remote"):
            del trainer.remote
        signal.signal(signal.SIGINT, previous_handler)
        if completed_run and any(p.exitcode != 0 for p in [gpu, *actors.values()]):
            raise RuntimeError("检查点已保存，但子进程未正常退出")
