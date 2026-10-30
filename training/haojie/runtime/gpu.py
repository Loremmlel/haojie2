"""单进程 GPU 所有者与本地控制队列；原生线程池直接供给神经请求。"""

import copy
from collections import deque
from concurrent.futures import Future
from queue import Queue, Empty
from pathlib import Path

import torch

from .compute import Compute
from .storage import Weights

GPU_KEYS = (
    "updates",
    "trained_samples",
    "value_samples",
    "soft_value_samples",
    "inference_roots",
    "inference_decodes",
    "inference_batches",
    "inference_seconds",
    "learning_seconds",
    "loss",
)


class Executor(Compute):
    def __init__(self, directory, config, initial):
        self.config = config
        self.models = Path(directory) / "models"
        self.initialize_network(config.device)
        self.model.load_state_dict(initial["model"])
        self.restore_optimizer(initial["optimizer"])
        torch.set_rng_state(initial["torch_rng"])
        if self.device.type == "cuda" and initial["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(initial["cuda_rng"])
        self.stats = {k: initial["stats"].get(k, 0) for k in GPU_KEYS if k != "loss"}
        self.contexts, self.inference_models, self.bindings = {}, {}, {}
        self.published = initial["published"]
        self.pending_versions = initial.get("pending_versions", {})
        self.weights = Weights(self.models, initial["weights"], initial["history"])
        self.queue_seconds = initial["stats"].get("inference_queue_seconds", 0.0)
        self.queued_requests = initial["stats"].get("inference_queue_requests", 0)

    def resolve(self, owner, message):
        previous = self.bindings.get(owner)
        if previous is not None and previous[0] != message["root"]:
            self.contexts = {k: v for k, v in self.contexts.items() if k[0] != owner}
        if message["model"] != "current":
            version = int(message["model"])
        elif previous is not None and previous[0] == message["root"]:
            version = previous[1]
        else:
            version = self.published
        self.bindings[owner] = message["root"], version
        self.pending_versions.pop(message["root"][0], None)
        if previous != self.bindings[owner]:
            self.prune_models()
        return version

    def release(self, owner):
        self.bindings.pop(owner, None)
        self.contexts = {k: v for k, v in self.contexts.items() if k[0] != owner}
        self.prune_models()

    def publish(self, version):
        self.published = version
        self.weights.publish(version, self.model)
        self.inference_models[version] = copy.deepcopy(self.model).eval()
        while len(self.inference_models) > 3:
            self.inference_models.pop(next(iter(self.inference_models)))
        self.prune_models()

    def report(self):
        host_stats = (
            torch.cuda.memory.host_memory_stats()
            if self.device.type == "cuda"
            and hasattr(torch.cuda.memory, "host_memory_stats")
            else {}
        )
        result = {
            **self.stats,
            "inference_queue_seconds": self.queue_seconds,
            "inference_queue_requests": self.queued_requests,
            "cuda_peak_bytes": (
                torch.cuda.max_memory_allocated() if self.device.type == "cuda" else 0
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if self.device.type == "cuda" else 0
            ),
            "pinned_host_bytes": host_stats.get("allocated_bytes.current"),
            "pinned_host_peak_bytes": host_stats.get("allocated_bytes.peak"),
            "cpu_weight_versions": len(self.weights.states),
        }
        if self.profile.enabled:
            result["profile_gpu"] = self.profile.report()
        return result

    def snapshot(self):
        def cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu(v) for v in value]
            return value

        return {
            "model": cpu(self.model.state_dict()),
            "optimizer": cpu(self.optimizer.state_dict()),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
            "stats": self.report(),
            "weights": self.weights.snapshot(self.live_versions()),
        }


def infer_batch(executor, batch):
    with executor.profile.stage("inference.resolve"):
        jobs, versions = [], []
        for request in batch.requests:
            owner = (0, request.lane)
            version = executor.resolve(
                owner,
                {
                    "root": (request.game_id, request.step),
                    "model": "current" if request.model == -1 else request.model,
                },
            )
            jobs.append((version, request, owner))
            versions.append(version)
    outputs = executor.infer(jobs)
    with executor.profile.stage("inference.reply"):
        batch.reply(outputs, versions)
    executor.queued_requests += len(jobs)


class Local:
    """协调线程只提交直接数组引用，GPU 主线程在批次边界执行。"""

    def __init__(self, pool):
        self.pool = pool
        self.commands = Queue(maxsize=4)
        self.pending = None
        self.completed = deque()
        self.error = None

    @property
    def learning(self):
        return self.pending is not None

    def submit(self, op, **payload):
        if self.error is not None:
            raise RuntimeError("GPU 执行已停止") from self.error
        future = Future()
        self.commands.put((op, payload, future))
        self.pool.wake()
        return future

    def call(self, op, **payload):
        return self.submit(op, **payload).result()

    def learn(self, batch):
        if self.pending is not None:
            raise RuntimeError("上一学习批次尚未确认")
        self.pending = self.submit("learn", batch=batch)

    def collect(self, block=False):
        if self.pending is not None and (block or self.pending.done()):
            self.completed.append(self.pending.result())
            self.pending = None

    def publish(self, version):
        self.call("publish", version=version)

    def infer(self, jobs):
        return self.call("infer", jobs=jobs)

    def fail(self, error):
        self.error = error
        while True:
            try:
                _, _, future = self.commands.get_nowait()
            except Empty:
                break
            future.set_exception(error)

    def execute_one(self, executor):
        try:
            op, payload, future = self.commands.get_nowait()
        except Empty:
            return
        try:
            if op == "learn":
                executor.update(payload["batch"])
                result = {"stats": executor.report(), "samples": len(payload["batch"])}
            elif op == "publish":
                executor.publish(payload["version"])
                result = None
            elif op == "archive":
                executor.weights.archive(payload["version"])
                result = None
            elif op == "infer":
                result = executor.infer(payload["jobs"])
                for _, _, owner in payload["jobs"]:
                    executor.release(owner)
            elif op == "release":
                for lane in payload["lanes"]:
                    executor.release((0, lane))
                result = None
            elif op == "drain":
                while (
                    batch := self.pool.take_requests(
                        executor.config.inference_batch, 0, profile=executor.profile
                    )
                ) is not None:
                    with batch:
                        infer_batch(executor, batch)
                result = None
            elif op == "snapshot":
                result = executor.snapshot()
            elif op == "evaluation":
                # 已发布的 CPU 权重不可变；评测持有引用，不暂停原生池或复制回放。
                result = {
                    "version": executor.published,
                    "model": executor.weights.get(executor.published),
                }
            elif op == "stats":
                result = executor.report()
            else:
                raise ValueError(f"未知 GPU 控制: {op}")
            future.set_result(result)
        except BaseException as error:
            future.set_exception(error)
            # 评测取权重失败由调度器报告，不终止训练执行循环。
            if op != "evaluation" or not isinstance(error, Exception):
                raise
        finally:
            self.pool.wake()
