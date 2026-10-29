"""单进程运行时：GPU 主线程、一个协调/I/O 线程、固定归属的 Rust 线程池。"""

import copy
import json
import signal
import threading
import time
from dataclasses import asdict

import torch

from ..native_pool import NativePool
from ..profiling import Profile, TraceWindow
from .gpu import Executor, Local, infer_batch
from .storage import GameWriter, atomic_json


class Coordinator:
    def __init__(self, trainer, pool, started):
        self.trainer, self.pool, self.started = trainer, pool, started
        self.initial = copy.deepcopy(trainer.stats)
        self.active, self.writers = {}, {}
        self.lane_order = [
            worker * trainer.config.games_per_worker + slot
            for slot in range(trainer.config.games_per_worker)
            for worker in range(trainer.config.workers)
        ]
        self.finished = threading.Event()
        self.error = None
        self.thread_id = None
        self.profile = Profile()
        self.local = Local(pool, self.retained)

    def retained(self):
        trainer = self.trainer
        return (
            trainer.checkpoint_versions
            | {e["version"] for e in trainer.pool.entries}
            | {game["version"] for game in self.active.values() if "version" in game}
        )

    def assign(self, lane, game, restored=None):
        with self.profile.stage("coordinator.assign"):
            self.active[lane] = game
            self.writers[lane] = GameWriter(
                self.trainer.directory, game, restored["metadata"] if restored else None
            )
            self.pool.assign(lane, game, restored)

    def complete_learning(self, block=False):
        self.local.collect(block)
        trainer, config = self.trainer, self.trainer.config
        while self.local.completed:
            result = self.local.completed.popleft()
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

    def consume(self, records):
        trainer, config = self.trainer, self.trainer.config
        ended, acknowledged = [], []
        for record in records:
            lane = record["lane"]
            writer = self.writers[lane]
            if (
                writer.game["id"] != record["game"]["id"]
                or record["steps"] != writer.metadata["steps"] + 1
            ):
                raise RuntimeError("原生正式命令顺序与记录不一致")
            sample = record.get("sample")
            if sample:
                writer.sample(
                    sample["bundle"],
                    sample["decisions"],
                    sample["actor"],
                    sample["ply"],
                    sample["version"],
                )
            info, command = record["info"], record["command"]
            writer.transition(info, command)
            trainer.stats["commands"] += 1
            done = record["done"]
            if done and info["winner"] is None:
                writer.metadata["status"] = "truncated"
                writer.metadata["truncation"] = record["truncation"]
                atomic_json(writer.path / "environment.json", record["environment"])
            if done or writer.metadata["steps"] % config.chunk_commands == 0:
                trainer.event({"type": "data", "path": writer.flush()})
            if done:
                trainer.event(
                    {
                        "type": "done",
                        "game": writer.game,
                        "winner": info["winner"],
                        "steps": writer.metadata["steps"],
                        "ply": info["ply"],
                        "path": str(writer.path / "game.json"),
                        "actor": info["actor"],
                        "metrics": info["metrics"],
                        "truncation": record.get("truncation"),
                    }
                )
                ended.append(lane)
                del self.active[lane]
                del self.writers[lane]
            acknowledged.append(lane)
        self.pool.ack_records(acknowledged)
        if ended:
            with self.profile.stage("coordinator.gpu_control_wait"):
                self.local.call("release", lanes=ended)

    def native_stats(self):
        status = self.pool.status()
        if status.get("error"):
            raise RuntimeError(status["error"])
        self.trainer.stats["runtime_threads"] = {
            "gpu": self.gpu_thread_id,
            "coordinator": self.thread_id,
            "workers": status.get("threads", []),
        }
        if "queue_seconds" in status:
            self.trainer.stats["inference_queue_seconds"] = (
                self.initial.get("inference_queue_seconds", 0.0)
                + status["queue_seconds"]
            )
        if self.profile.enabled:
            self.trainer.stats["profile_native"] = status.get("profile")
            self.trainer.stats["profile_coordinator"] = self.profile.report()
        return status

    def save(self):
        self.pool.pause()
        self.complete_learning(block=True)
        while True:
            self.local.call("drain")
            records = self.pool.take_records()
            if records:
                self.consume(records)
            if self.pool.barrier()["ready"]:
                break
        snapshots = {i: [] for i in range(self.trainer.config.workers)}
        for value in self.pool.snapshot():
            lane = value["lane"]
            writer = self.writers[lane]
            writer.flush()
            value["metadata"] = writer.metadata
            value["lane"] = lane % self.trainer.config.games_per_worker
            snapshots[lane // self.trainer.config.games_per_worker].append(value)
        state = self.local.call("snapshot")
        trainer = self.trainer
        trainer.model.load_state_dict(state["model"])
        trainer.restore_optimizer(state["optimizer"])
        trainer.remote_torch_rng = state["torch_rng"]
        trainer.remote_cuda_rng = state["cuda_rng"]
        trainer.stats.update(state["stats"])
        self.native_stats()
        trainer.checkpoint(snapshots)

    def run(self):
        trainer, config = self.trainer, self.trainer.config
        self.thread_id = threading.get_native_id()
        last_status = last_save = time.perf_counter()
        try:
            previous = {
                worker * config.games_per_worker + value["lane"]: value
                for worker, values in (trainer.saved_workers or {}).items()
                for value in values
            }
            for lane in self.lane_order:
                restored = previous.get(lane)
                game = restored["game"] if restored else trainer.game()
                if game is not None:
                    self.assign(lane, game, restored)
            print(
                json.dumps(
                    {
                        "event": "started",
                        "directory": str(trainer.directory),
                        "config": asdict(config),
                        "runtime": "native-thread-pool",
                        **trainer.model.description(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            while True:
                self.complete_learning()
                now = time.perf_counter()
                stopped = (
                    trainer.stop_requested
                    or (config.seconds and now - self.started >= config.seconds)
                    or (
                        config.games
                        and trainer.stats["completed"] - self.initial["completed"]
                        >= config.games
                    )
                    or (
                        config.updates
                        and trainer.stats["updates"] - self.initial["updates"]
                        >= config.updates
                    )
                    or (trainer.directory / "STOP").exists()
                )
                periodic = (
                    config.save_seconds and now - last_save >= config.save_seconds
                )
                if stopped or periodic:
                    with self.profile.stage("coordinator.checkpoint"):
                        self.save()
                    if stopped:
                        break
                    self.pool.resume()
                    last_save = time.perf_counter()
                for lane in self.lane_order:
                    if lane not in self.active:
                        game = trainer.game()
                        if game is not None:
                            self.assign(lane, game)
                if not self.local.learning and (
                    trainer.phase == "imitation"
                    or (trainer.phase == "self" and trainer.credit >= config.batch)
                ):
                    with self.profile.stage("coordinator.replay_batch"):
                        batch = trainer.replay.batch(config.batch, trainer.rng)
                    if batch:
                        self.local.learn(batch)
                with self.profile.stage("coordinator.receive_records"):
                    records = self.pool.take_records(wait_ms=100)
                if records:
                    with self.profile.stage("coordinator.record_io"):
                        self.consume(records)
                if now - last_status >= 5:
                    trainer.stats.update(self.local.call("stats"))
                    self.native_stats()
                    trainer.status("running", self.started, self.initial)
                    last_status = now
            if self.profile.enabled:
                trainer.stats["profile_coordinator"] = self.profile.report()
            trainer.status("saved", self.started, self.initial)
        except BaseException as error:
            self.error = error
            trainer.stop_requested = True
            trainer.status("failed", self.started, self.initial, error=str(error))
            # 故障时不以缺失原生状态的快照覆盖上一次完整检查点。
            atomic_json(
                trainer.directory / "failed-native.json", {"error": repr(error)}
            )
        finally:
            self.finished.set()
            self.pool.wake()


def run(trainer):
    # 诊断只描述本次启动；恢复模型时不沿用上次运行的分段计数。
    for key in ("profile_gpu", "profile_native", "profile_coordinator"):
        trainer.stats.pop(key, None)
    started = time.perf_counter()
    initial = {
        "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "torch_rng": getattr(trainer, "remote_torch_rng", torch.get_rng_state()),
        "cuda_rng": getattr(trainer, "remote_cuda_rng", None),
        "stats": trainer.stats,
        "published": trainer.published,
        "retained": list(
            trainer.checkpoint_versions | {e["version"] for e in trainer.pool.entries}
        ),
    }
    executor = Executor(trainer.directory, trainer.config, initial)
    trace = TraceWindow(executor.profile, executor.device)
    pool = NativePool(trainer.config)
    previous = signal.signal(
        signal.SIGINT, lambda *_: setattr(trainer, "stop_requested", True)
    )
    coordinator = Coordinator(trainer, pool, started)
    coordinator.gpu_thread_id = threading.get_native_id()
    trainer.remote = coordinator.local
    thread = threading.Thread(target=coordinator.run, name="haojie-io")
    thread.start()
    try:
        while not coordinator.finished.is_set():
            trace.step(executor.stats)
            with executor.profile.stage("runtime.loop"):
                batch = pool.take_requests(
                    trainer.config.inference_batch,
                    wait_ms=1000,
                    profile=executor.profile,
                )
                if batch is not None:
                    with executor.profile.stage("runtime.inference"):
                        with batch:
                            infer_batch(executor, batch)
                with executor.profile.stage("runtime.control"):
                    coordinator.local.execute_one(executor)
            trace.complete(executor.stats)
        if coordinator.error is not None:
            raise coordinator.error
        if executor.profile.enabled:
            trainer.stats["profile_gpu"] = executor.profile.report()
        return trainer.stats
    except BaseException as error:
        trainer.stop_requested = True
        coordinator.local.fail(error)
        pool.wake()
        raise
    finally:
        thread.join()
        pool.close()
        trace.close(executor.stats)
        del trainer.remote
        signal.signal(signal.SIGINT, previous)
