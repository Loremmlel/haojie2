"""单 GPU 所有者：有界推理、一步学习、双槽发布和未完成工作快照。"""

import copy
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .learner import Learner
from .metrics import Measurements, resident_bytes
from .model import (
    DTYPE,
    FORMAT,
    Config,
    Network,
    node_batch,
    node_events,
    root_batch,
    validate_weights,
    work_batches,
)
from .native import Worker, frames, packet
from .staging import Staging
from .trajectory import Trajectories


@dataclass
class Settings:
    concurrency: int = 16
    workers: int = 2
    seed: int = 1
    block: int = 64
    logical_batch: int = 128
    microbatch: int = 32
    publish_steps: int = 10
    opponent_steps: int = 200
    memory_bytes: int = 4 * 1024**3
    shrine: bool = False


class MemoryBudgetExceeded(RuntimeError):
    """当前预算已无法容纳下一批工作；所有者在安全边界保存后明确退出。"""

    def __init__(self, used, limit):
        super().__init__(f"实际内存 {used} 字节超过预算 {limit} 字节；请调整预算后从检查点恢复")


def byte_size(value, seen=None):
    """按对象身份去重统计 CPU 张量；Python 控制对象计入保守的容器开销。"""
    import sys

    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() + 128
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            byte_size(k, seen) + byte_size(v, seen) for k, v in value.items()
        )
    if isinstance(value, (list, tuple)) or type(value).__name__ == "deque":
        return sys.getsizeof(value) + sum(byte_size(v, seen) for v in value)
    if isinstance(value, Trajectories):
        return byte_size(vars(value), seen)
    return sys.getsizeof(value)


class Session:
    def __init__(self, settings=None, device="cuda", config=Config(), *, training=True):
        self.training = training
        self.settings = settings or Settings()
        if not 0 <= self.settings.concurrency <= 128 or not 1 <= self.settings.workers <= 16:
            raise ValueError("并发必须在 0–128，worker 必须在 1–16")
        if (
            min(
                self.settings.block,
                self.settings.logical_batch,
                self.settings.microbatch,
                self.settings.publish_steps,
                self.settings.opponent_steps,
                self.settings.memory_bytes,
            )
            <= 0
        ):
            raise ValueError("训练预算必须为正数")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_bf16_supported(
            including_emulation=False
        ):
            raise RuntimeError("训练需要原生 CUDA BF16 支持")
        torch.set_num_threads(1)
        torch.manual_seed(self.settings.seed)
        self.learner = Learner(Network(config).to(self.device))
        self.learner.model.flatten_history()
        self.models = {0: copy.deepcopy(self.learner.model).eval()}
        self.opponents = {0: copy.deepcopy(self.learner.model).eval()}
        self.models[0].flatten_history()
        self.opponents[0].flatten_history()
        self.staging = Staging(self.device)
        self.current = 0
        self.last_opponent = 0
        self.workers = [Worker() for _ in range(self.settings.workers)]
        self.requests = [[] for _ in self.workers]
        self.awaiting = {}
        self.games = {}
        self.next_game = 1
        self.roots = {}
        self.versions = {}
        self.cache = {}
        self.history_cache = {}
        self.histories = {}
        self.traces = Trajectories(self.settings.block)
        self.metrics = dict(
            completed=0,
            wins=0,
            losses=0,
            draws=0,
            commands=0,
            microsteps=0,
            root_encodings=0,
            decode_batches=0,
            root_seconds=0.0,
            decode_seconds=0.0,
            train_seconds=0.0,
            publish_delayed=0,
            failed_attempts=0,
            backtracks=0,
            loss=None,
        )
        self.elapsed = 0.0
        self.started = time.monotonic()
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="haojie-save")
        self.saving = None
        self.admission_paused = False
        self.resident_sample = (0.0, 0)
        self.managed_sample = (0.0, 0)
        self.measurements = Measurements(self.device, self.metrics)
        self.inference_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.submission_cursor = 0
        self.on_result = None

    @staticmethod
    def _freeze(model):
        result = copy.deepcopy(model).eval()
        result.flatten_history()
        return result

    def _release(self, game, root):
        for key in [k for k in self.versions if k[0] == game and k != root]:
            self.versions.pop(key)
            self.cache.pop(key, None)
            self.history_cache.pop(key, None)

    def _accept(self, worker, payload):
        for frame in frames(payload):
            game = frame["game"]
            info = self.games[game]
            self.metrics["commands"] += frame["commands"] - info["commands"]
            self.metrics["microsteps"] += frame["microsteps"] - info["microsteps"]
            self.metrics["failed_attempts"] += frame["failures"] - info["failures"]
            self.metrics["backtracks"] += frame["backtracks"] - info["backtracks"]
            info.update(
                commands=frame["commands"],
                microsteps=frame["microsteps"],
                failures=frame["failures"],
                backtracks=frame["backtracks"],
                waiting=frame["kind"] == 0,
            )
            if frame["selected"] >= 0:
                old = self.awaiting.pop(game)
                if self.training and old["actor"] == old["learner"]:
                    self.traces.selected(
                        game, old["node"], frame["selected"], frame["logp"], old["version"]
                    )
            key = (game, frame["root"])
            self._release(game, key if frame["kind"] != 1 else None)
            if frame["kind"] == 1:
                reward = (
                    0 if frame["winner"] == 0 else 1 if frame["winner"] == info["learner"] else -1
                )
                if self.training:
                    self.traces.successor(game, reward=reward)
                self.metrics["completed"] += 1
                self.metrics[{1: "wins", -1: "losses", 0: "draws"}[reward]] += 1
                if self.on_result is not None:
                    self.on_result(
                        dict(
                            game=game,
                            **info,
                            reward=reward,
                            seconds=self.elapsed + time.monotonic() - self.started,
                        )
                    )
                del self.games[game]
                continue
            if frame["kind"] == 2:
                continue
            if frame["root_input"] is not None:
                self.roots[key] = frame.pop("root_input")
            if key not in self.versions:
                self.versions[key] = (
                    ("learner", self.current)
                    if frame["actor"] == frame["learner"]
                    else ("opponent", frame["opponent"])
                )
            frame["version"] = self.versions[key][1]
            frame["queued_at"] = time.monotonic()
            delta = frame["node"]["history"]["events"]
            history = self.histories.setdefault(key, {"events": delta[:0], "length": 0})
            start = frame["node"].pop("history_start")
            if history["length"] != start:
                raise ValueError("原生增量事件不连续")
            end = start + len(delta)
            if end > len(history["events"]):
                grown = delta.new_zeros((max(end, 16, 2 * len(history["events"])), 68))
                grown[:start] = history["events"][:start]
                history["events"] = grown
            history["events"][start:end] = delta
            history["length"] = end
            frame["node"]["history"] = history
            if self.training and frame["actor"] == frame["learner"]:
                self.traces.successor(game, frame["node"])
            self.requests[worker].append(frame)

    @torch.no_grad()
    def _infer(self, requests, *, defer=False):
        responses = []
        transfers = []
        groups = defaultdict(list)
        for frame in requests:
            groups[self.versions[frame["node"]["root"]]].append(frame)
        for (role, version), group in sorted(groups.items()):
            model = (self.models if role == "learner" else self.opponents)[version]
            # 先服务旧版本；每批上限同时约束候选数和实体数对应的工作量。
            group.sort(key=lambda f: len(self.roots[f["node"]["root"]]["entities"]))
            for chunk in work_batches(
                group,
                self.settings.microbatch,
                lambda f: (
                    (len(self.roots[f["node"]["root"]]["entities"]) + 1) ** 2,
                    len(f["node"]["candidates"]) * model.config.width,
                    f["node"]["history_end"] * model.config.history,
                ),
            ):
                missing = [f["node"]["root"] for f in chunk if f["node"]["root"] not in self.cache]
                if missing:
                    began = time.monotonic()
                    timing = self.measurements.begin("root", len(missing))
                    encoded = model.encode(
                        *root_batch([self.roots[k] for k in missing], self.device, self.staging)
                    )
                    for i, key in enumerate(missing):
                        self.cache[key] = encoded[i, : len(self.roots[key]["entities"]) + 1].clone()
                    self.metrics["root_encodings"] += len(missing)
                    self.metrics["root_seconds"] += time.monotonic() - began
                    self.measurements.end(timing)
                began = time.monotonic()
                timing = self.measurements.begin("decode", len(chunk))
                nodes = [f["node"] for f in chunk]
                max_entities = max(len(self.cache[n["root"]]) for n in nodes)
                encoded = torch.zeros(
                    (len(nodes), max_entities, model.config.width), device=self.device, dtype=DTYPE
                )
                initial = torch.zeros(
                    (len(nodes), model.config.history), device=self.device, dtype=DTYPE
                )
                incremental = []
                for i, node in enumerate(nodes):
                    key = node["root"]
                    encoded[i, : len(self.cache[key])] = self.cache[key]
                    previous, hidden = self.history_cache.get(key, (0, initial[i]))
                    initial[i] = hidden
                    new_events = node_events(node)[previous:]
                    incremental.append(
                        {**node, "history": {"events": new_events}, "history_end": len(new_events)}
                    )
                prefix, pm, events, lengths, candidates, sources, targets, mask = node_batch(
                    incremental, self.device, self.staging
                )
                history = model.histories(events, lengths, initial)
                logits, _ = model.decode(
                    encoded, prefix, pm, history, candidates, sources, targets, mask
                )
                probabilities = logits.softmax(-1)
                if defer:
                    staging = torch.empty(probabilities.shape, dtype=DTYPE, pin_memory=True)
                    staging.copy_(probabilities, non_blocking=True)
                    probabilities = staging
                else:
                    probabilities = probabilities.cpu()
                for i, (frame, node) in enumerate(zip(chunk, nodes)):
                    wait = time.monotonic() - frame["queued_at"]
                    self.metrics["inference_wait_seconds"] = (
                        self.metrics.get("inference_wait_seconds", 0.0) + wait
                    )
                    self.metrics["inference_wait_max"] = max(
                        self.metrics.get("inference_wait_max", 0.0), wait
                    )
                    self.history_cache[node["root"]] = (node["history_end"], history[i].clone())
                    self.awaiting[frame["game"]] = frame
                    responses.append((frame["game"], probabilities[i, : len(node["candidates"])]))
                self.metrics["decode_batches"] += 1
                self.metrics["decode_seconds"] += time.monotonic() - began
                self.measurements.end(timing)
                if defer:
                    completed = torch.cuda.Event()
                    completed.record()
                    transfers.append(completed)
        return (responses, transfers) if defer else responses

    def _publish(self):
        referenced = {version for role, version in self.versions.values() if role == "learner"}
        for version in list(self.models):
            if version != self.current and version not in referenced:
                del self.models[version]
        step = self.learner.steps
        if step - self.current >= self.settings.publish_steps:
            if len(self.models) == 2 and self.current in referenced:
                self.metrics["publish_delayed"] += 1
            else:
                if self.current not in referenced:
                    del self.models[self.current]
                self.models[step] = self._freeze(self.learner.model)
                self.current = step
        if step - self.last_opponent >= self.settings.opponent_steps:
            used = {g["opponent"] for g in self.games.values()}
            removable = [v for v in self.opponents if v != 0 and v not in used]
            if len(self.opponents) >= 3 and removable:
                del self.opponents[min(removable)]
            if len(self.opponents) < 3:
                self.opponents[step] = self._freeze(self.learner.model)
                self.last_opponent = step

    def tick(self, admit=True):
        if self.saving is not None and self.saving.done():
            self.saving.result()
            self.saving = None
        used = self.memory()
        if used > self.settings.memory_bytes:
            raise MemoryBudgetExceeded(used, self.settings.memory_bytes)
        for i, worker in enumerate(self.workers):
            if worker.busy:
                payload = worker.poll()
                if payload is not None:
                    self._accept(i, payload)
        deferred = {}
        self._collect_roots()
        self.admission_paused = self.memory() >= self.settings.memory_bytes * 0.9
        if self.inference_stream is not None:
            self.inference_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.inference_stream):
                # 最多预提交两个工作批；其余请求仍留在有界的每局等待位置。
                for offset in range(len(self.workers)):
                    i = (self.submission_cursor + offset) % len(self.workers)
                    if not self.workers[i].busy and self.requests[i]:
                        count = min(self.settings.microbatch, len(self.requests[i]))
                        deferred[i] = self._infer(self.requests[i][:count], defer=True)
                        del self.requests[i][:count]
                        if len(deferred) == 2:
                            break
            self.submission_cursor = (self.submission_cursor + 1) % len(self.workers)
        queued = sum(len(b["steps"]) for b in self.traces.ready)
        if queued >= self.settings.logical_batch or (
            self.traces.ready and (not self.games or self.admission_paused)
        ):
            blocks, count = [], 0
            for block in self.traces.ready:
                blocks.append(block)
                count += len(block["steps"])
                if count >= self.settings.logical_batch:
                    break
            began = time.monotonic()
            timing = self.measurements.begin("train", count)
            self.metrics["loss"] = self.learner.update(blocks, self.roots, self.settings.microbatch)
            lag = [self.learner.steps - s["version"] for b in blocks for s in b["steps"]]
            self.metrics["behavior_lag_sum"] = self.metrics.get("behavior_lag_sum", 0) + sum(lag)
            self.metrics["behavior_lag_max"] = max(
                self.metrics.get("behavior_lag_max", 0), max(lag)
            )
            self.measurements.end(timing)
            self.traces.commit([b["id"] for b in blocks])
            self.metrics["train_seconds"] += time.monotonic() - began
        if self.training:
            self._publish()
        # 接近实际预算时只停止补局，已有工作继续按正常批量推进。
        self.admission_paused = self.memory() >= self.settings.memory_bytes * 0.9
        for i, worker in enumerate(self.workers):
            if worker.busy:
                continue
            if i in deferred:
                replies, transfers = deferred[i]
                for event in transfers:
                    event.synchronize()
            elif self.inference_stream is not None:
                replies = []
            else:
                replies = self._infer(self.requests[i])
                self.requests[i].clear()
            admissions = []
            # 新局预算独立于已有根的服务额度；旧根和已接纳块始终会继续推进。
            if admit and len(self.games) < self.settings.concurrency and not self.admission_paused:
                quota = (self.settings.concurrency + len(self.workers) - 1) // len(self.workers)
                live = sum(g["worker"] == i for g in self.games.values())
                for _ in range(min(quota - live, self.settings.concurrency - len(self.games))):
                    game = self.next_game
                    self.next_game += 1
                    opponent = list(self.opponents)[game % len(self.opponents)]
                    seat = 1 + game % 2
                    seed = (self.settings.seed + game * 2654435761) & 0xFFFFFFFF
                    admissions.append((game, seed, seat, opponent, self.settings.shrine))
                    self.games[game] = dict(
                        worker=i,
                        learner=seat,
                        opponent=opponent,
                        commands=0,
                        microsteps=0,
                        failures=0,
                        backtracks=0,
                        waiting=False,
                        seed=seed,
                    )
            if (
                replies
                or admissions
                or any(g["worker"] == i and not g["waiting"] for g in self.games.values())
            ):
                worker.submit(0, packet(admissions, replies))
        self._collect_roots()

    def _collect_roots(self):
        keep = self.traces.root_ids() | set(self.versions)
        keep.update(f["node"]["root"] for f in self.awaiting.values())
        self.roots = {k: v for k, v in self.roots.items() if k in keep}
        self.histories = {k: v for k, v in self.histories.items() if k in keep}

    def memory(self):
        now = time.monotonic()
        if now - self.resident_sample[0] >= 0.05:
            self.resident_sample = (now, resident_bytes())
        gpu = torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0
        return self.resident_sample[1] + gpu

    def managed_bytes(self):
        now = time.monotonic()
        if now - self.managed_sample[0] < 0.05:
            return self.managed_sample[1]
        gpu = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        size = (
            byte_size(
                (
                    self.roots,
                    self.histories,
                    self.requests,
                    self.awaiting,
                    self.traces,
                )
            )
            + self.workers[0].memory()
            + self.staging.bytes
            + self.learner.staging.bytes
            + gpu
        )
        self.managed_sample = (now, size)
        return size

    def status(self):
        self.measurements.collect()
        self.metrics["resident_peak_bytes"] = max(
            self.metrics.get("resident_peak_bytes", 0), resident_bytes()
        )
        return {
            **self.metrics,
            "seconds": self.elapsed + time.monotonic() - self.started,
            "target_concurrency": self.settings.concurrency,
            "inflight": len(self.games),
            "waiting_inference": sum(len(r) for r in self.requests),
            "waiting_successor": len(self.traces.pending),
            "waiting_learning": len(self.traces.ready),
            "optimizer_steps": self.learner.steps,
            "consumed_microsteps": self.learner.consumed,
            "published": self.current,
            "publish_lag": self.learner.steps - self.current,
            "accounted_bytes": self.memory(),
            "managed_bytes": self.managed_bytes(),
            "gpu_reserved_bytes": torch.cuda.memory_reserved(self.device)
            if self.device.type == "cuda"
            else 0,
            "native_heap_bytes": self.workers[0].memory(),
            "admission_paused": self.admission_paused,
            "pinned_bytes": self.staging.bytes + self.learner.staging.bytes,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else 0,
        }

    def save(self, path, *, background=False):
        """完成已提交原子操作，冻结独立 CPU 快照；不等待整局终局。"""
        if self.saving is not None:
            self.saving.result()
            self.saving = None
        for i, worker in enumerate(self.workers):
            if worker.busy:
                self._accept(i, worker.wait())
        native = []
        for worker in self.workers:
            worker.submit(1)
        for worker in self.workers:
            native.append(worker.wait())
        self.measurements.collect(wait=True)

        def weights(model):
            return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        def cpu_copy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu_copy(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu_copy(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu_copy(v) for v in value)
            return copy.deepcopy(value)

        checkpoint = dict(
            format=FORMAT,
            training=self.training,
            native_identity=self.workers[0].identity,
            settings=asdict(self.settings),
            config=asdict(self.learner.model.config),
            model=weights(self.learner.model),
            optimizer=cpu_copy(self.learner.optimizer.state_dict()),
            steps=self.learner.steps,
            consumed=self.learner.consumed,
            models={v: weights(m) for v, m in self.models.items()},
            opponents={v: weights(m) for v, m in self.opponents.items()},
            current=self.current,
            last_opponent=self.last_opponent,
            native=native,
            state=copy.deepcopy(
                dict(
                    requests=self.requests,
                    awaiting=self.awaiting,
                    games=self.games,
                    next_game=self.next_game,
                    roots=self.roots,
                    histories=self.histories,
                    versions=self.versions,
                    traces=self.traces,
                    submission_cursor=self.submission_cursor,
                    metrics=self.metrics,
                )
            ),
            elapsed=self.elapsed + time.monotonic() - self.started,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        )
        self.metrics["snapshot_bytes"] = byte_size(checkpoint)

        def write():
            path_ = Path(path)
            path_.parent.mkdir(parents=True, exist_ok=True)
            temporary = path_.with_name(path_.name + ".partial")
            with temporary.open("xb") as stream:
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path_)
            return str(path_)

        self.saving = self.writer.submit(write)
        if not background:
            self.saving.result()
            self.saving = None

    @classmethod
    def restore(cls, path, device="cuda"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["format"] != FORMAT:
            raise ValueError("只接受新连续轨迹检查点")
        for weights in [
            checkpoint["model"],
            *checkpoint["models"].values(),
            *checkpoint["opponents"].values(),
        ]:
            validate_weights(weights)
        self = cls(
            Settings(**checkpoint["settings"]),
            device,
            Config(**checkpoint["config"]),
            training=checkpoint["training"],
        )
        for key in ("ruleset", "rulesHash"):
            if self.workers[0].identity[key] != checkpoint["native_identity"][key]:
                self.close()
                raise ValueError("检查点与当前正式规则不一致")
        self.learner.model.load_state_dict(checkpoint["model"])
        self.learner.optimizer.load_state_dict(checkpoint["optimizer"])
        self.learner.steps, self.learner.consumed = checkpoint["steps"], checkpoint["consumed"]
        for collection in ("models", "opponents"):
            models = {}
            for version, weights in checkpoint[collection].items():
                model = Network(self.learner.model.config).to(self.device).eval()
                model.load_state_dict(weights)
                model.flatten_history()
                models[version] = model
            setattr(self, collection, models)
        for name in ("current", "last_opponent", "elapsed"):
            setattr(self, name, checkpoint[name])
        for name, value in checkpoint["state"].items():
            setattr(self, name, value)
        self.measurements = Measurements(self.device, self.metrics)
        for frames_ in self.requests:
            for frame in frames_:
                frame["queued_at"] = time.monotonic()
        for worker, payload in zip(self.workers, checkpoint["native"]):
            worker.submit(2, payload)
            worker.wait()
        # 根表示及 GRU 缓存用仍被引用的冻结模型重建，不再次抽样。
        torch.set_rng_state(checkpoint["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        self.started = time.monotonic()
        return self

    def close(self):
        try:
            if self.saving is not None:
                self.saving.result()
        finally:
            self.writer.shutdown(wait=True)
            for worker in self.workers:
                worker.close()
