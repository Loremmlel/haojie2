"""独占 CUDA 的执行服务：跨 actor 组批，学习在推理批次边界执行。"""

import copy
import signal
import time
import traceback
from collections import deque
from multiprocessing.connection import wait
from pathlib import Path

import torch

from ..search import Request
from .compute import Compute
from .ipc import Slot, pack, unpack, batch_arrays, restore_batch


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
        self.retained = set(initial["retained"])
        self.queue_seconds = initial["stats"].get("inference_queue_seconds", 0.0)
        self.queued_requests = initial["stats"].get("inference_queue_requests", 0)

    def resolve(self, owner, message):
        previous = self.bindings.get(owner)
        if previous is not None and previous[0] != message["root"]:
            self.release(owner)
        if message["model"] != "current":
            version = int(message["model"])
        elif previous is not None and previous[0] == message["root"]:
            version = previous[1]
        else:
            version = self.published
        self.bindings[owner] = message["root"], version
        return version

    def release(self, owner):
        self.bindings.pop(owner, None)
        self.contexts = {k: v for k, v in self.contexts.items() if k[0] != owner}

    def publish(self, version, retained):
        self.published = version
        self.retained = set(retained)
        path = self.models / f"model-{version:08d}.pt"
        if not path.exists():
            torch.save(
                {k: v.detach().cpu() for k, v in self.model.state_dict().items()}, path
            )
        self.inference_models[version] = copy.deepcopy(self.model).eval()
        while len(self.inference_models) > 3:
            self.inference_models.pop(next(iter(self.inference_models)))
        protected = self.retained | {version} | {v for _, v in self.bindings.values()}
        for candidate in self.models.glob("model-*.pt"):
            if int(candidate.stem.split("-")[1]) not in protected:
                candidate.unlink()

    def report(self):
        return {
            **self.stats,
            "inference_queue_seconds": self.queue_seconds,
            "inference_queue_requests": self.queued_requests,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated()
            if self.device.type == "cuda"
            else 0,
        }

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
            "cuda_rng": torch.cuda.get_rng_state_all()
            if self.device.type == "cuda"
            else None,
            "stats": self.report(),
        }


def serve(control, actors, directory, config, initial):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    slots, pending = {}, deque()
    learning = None
    executor = None

    def arrays(owner, message):
        previous = slots.get(owner)
        if previous is None or previous.name != message["slot"]:
            if previous is not None:
                previous.close()
            slots[owner] = Slot(message["slot"])
        return unpack(slots[owner], message["shapes"])

    try:
        executor = Executor(directory, config, initial)
        control.send({"op": "ready"})
        while True:
            timeout = (
                max(0, 0.002 - (time.perf_counter() - pending[0][0]))
                if pending
                else 0.05
            )
            if learning:
                timeout = 0
            ready = wait([control, *actors], timeout)
            if control in ready:
                message = control.recv()
                op = message["op"]
                if op == "stop":
                    break
                if op == "learn":
                    if learning is not None:
                        raise RuntimeError("学习槽在确认前被重复提交")
                    learning = message
                elif op == "publish":
                    executor.publish(message["version"], message["retained"])
                    control.send({"op": "published"})
                elif op == "snapshot":
                    if pending or learning:
                        raise RuntimeError("必须排空推理和学习后才能保存")
                    control.send({"op": "snapshot", "state": executor.snapshot()})
                elif op == "stats":
                    control.send({"op": "stats", "stats": executor.report()})
                elif op == "infer":
                    values = arrays("boundary", message)
                    job = (
                        message["version"],
                        Request("encode", message["context"], tuple(values)),
                        ("boundary", 0),
                    )
                    output = executor.infer([job])[0]
                    executor.release(("boundary", 0))
                    control.send(
                        {"op": "inferred", "shapes": pack(slots["boundary"], [output])}
                    )
            for index, connection in enumerate(actors):
                if connection not in ready:
                    continue
                while connection.poll():
                    message = connection.recv()
                    owner = index, message["lane"]
                    if message["op"] == "release":
                        executor.release(owner)
                        continue
                    version = executor.resolve(owner, message)
                    values = arrays(owner, message)
                    pending.append(
                        (
                            time.perf_counter(),
                            index,
                            message,
                            (
                                version,
                                Request(
                                    message["kind"], message["context"], tuple(values)
                                ),
                                owner,
                            ),
                        )
                    )
            if pending and (
                len(pending) >= config.inference_batch
                or time.perf_counter() - pending[0][0] >= 0.002
            ):
                selected = [
                    pending.popleft()
                    for _ in range(min(len(pending), config.inference_batch))
                ]
                now = time.perf_counter()
                executor.queue_seconds += sum(now - item[0] for item in selected)
                executor.queued_requests += len(selected)
                outputs = executor.infer([item[3] for item in selected])
                for item, output in zip(selected, outputs):
                    _, index, message, job = item
                    actors[index].send(
                        {
                            "op": "answer",
                            "lane": message["lane"],
                            "version": job[0],
                            "shapes": pack(slots[job[2]], [output]),
                        }
                    )
            if learning:
                batch = restore_batch(arrays("learning", learning), learning["rows"])
                executor.update(batch)
                control.send(
                    {"op": "learned", "stats": executor.report(), "samples": len(batch)}
                )
                learning = None
    except BaseException:
        error = traceback.format_exc()
        control.send({"op": "error", "error": error})
        for connection in actors:
            try:
                connection.send({"op": "error", "error": error})
            except (BrokenPipeError, OSError):
                pass
        raise
    finally:
        for slot in slots.values():
            slot.close()
        for connection in actors:
            connection.close()
        control.close()


class Remote:
    """CPU 协调端；学习只有一个在途批次，其余低频请求使用独立槽。"""

    def __init__(self, connection, retained):
        self.connection, self.retained = connection, retained
        self.learning = False
        self.completed = deque()
        self.slot = Slot(size=16 * 1024 * 1024)
        self.boundary = Slot(size=256 * 1024)

    def receive(self):
        message = self.connection.recv()
        if message["op"] == "error":
            raise RuntimeError(message["error"])
        if message["op"] == "learned":
            self.learning = False
            self.completed.append(message)
        return message

    def call(self, message):
        self.connection.send(message)
        while True:
            result = self.receive()
            if result["op"] != "learned":
                return result

    def prepare(self, attribute, arrays):
        slot = getattr(self, attribute)
        needed = sum(a.nbytes for a in arrays)
        if needed > slot.memory.size:
            slot.close()
            slot = Slot(size=needed * 2)
            setattr(self, attribute, slot)
        return {"slot": slot.name, "shapes": pack(slot, arrays)}

    def learn(self, batch):
        if self.learning or self.completed:
            raise RuntimeError("必须先提交上一批学习结果")
        arrays, rows = batch_arrays(batch)
        self.connection.send(
            {"op": "learn", "rows": rows, **self.prepare("slot", arrays)}
        )
        self.learning = True

    def publish(self, version):
        self.call(
            {"op": "publish", "version": version, "retained": list(self.retained())}
        )

    def infer(self, jobs):
        result = []
        for version, request, _ in jobs:
            answer = self.call(
                {
                    "op": "infer",
                    "version": version,
                    "context": request.context,
                    **self.prepare("boundary", request.arrays),
                }
            )
            result.append(unpack(self.boundary, answer["shapes"], copy=True)[0])
        return result

    def close(self):
        self.slot.close()
        self.boundary.close()
        self.connection.close()
