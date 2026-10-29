"""单 GPU 所有者：跨 worker 合批推理、短学习批次和一致边界恢复。"""

import copy
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import signal
from pathlib import Path
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.nn import functional as F

from ..model import Network, collate, device_for_bf16
from ..opponents import Opponents
from .config import Config
from .ipc import Slot, pack, unpack
from .storage import Replay, atomic_json
from .worker import worker


class Trainer:
    def __init__(self, directory, config, resume=False):
        self.directory = Path(directory).resolve()
        self.config = config
        config.validate()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models = self.directory / "models"
        self.models.mkdir(exist_ok=True)
        torch.set_num_threads(1)
        self.device = device_for_bf16(config.device)
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.model = Network().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=.0001, foreach=False)
        for parameter in self.model.parameters():
            # 整数步数避免 BF16 在第 256 步后停止递增；动量和权重均为 BF16。
            self.optimizer.state[parameter] = {"step": torch.tensor(0, dtype=torch.int64),
                "exp_avg": torch.zeros_like(parameter), "exp_avg_sq": torch.zeros_like(parameter)}
        self.replay = Replay(config.replay_games)
        self.pool = Opponents()
        self.published = 0
        self.inference_models = {}
        self.bindings = {}
        self.seen_samples = {}
        self.credit = 0.
        self.next_game = 0
        self.saved_workers = None
        self.checkpoint_versions = set()
        self.stop_requested = False
        self.stats = {"updates": 0, "completed": 0, "truncated": 0, "demonstrations": 0,
                      "selfplay_terminal": 0, "samples": 0, "trained_samples": 0,
                      "value_samples": 0, "inference_roots": 0, "inference_batches": 0,
                      "inference_seconds": 0., "learning_seconds": 0., "commands": 0,
                      "parameters": self.model.description()["parameters"]}
        checkpoint = self.directory / "checkpoint.pt"
        if resume:
            # 仅加载用户指定目录中本程序写出的完整恢复快照。
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["format"] != "haojie-classic-training-v1":
                raise ValueError("训练检查点格式不匹配")
            old_config = saved["config"]
            for key in ("workers", "games_per_worker", "candidates"):
                if old_config[key] != getattr(config, key):
                    raise ValueError(f"续训保持 {key}={old_config[key]}，请先按原并发恢复")
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            for group in self.optimizer.param_groups:
                group["foreach"] = False
            for state in self.optimizer.state.values():
                state["step"] = state["step"].to(device="cpu", dtype=torch.int64)
            self.rng.bit_generator.state = saved["rng"]
            torch.set_rng_state(saved["torch_rng"])
            if self.device.type == "cuda" and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            self.stats = saved["stats"]
            self.replay = Replay(config.replay_games, saved["replay"])
            self.pool = Opponents(saved["pool"])
            self.published, self.next_game = saved["published"], saved["next_game"]
            self.seen_samples, self.credit = saved["seen_samples"], saved["credit"]
            self.saved_workers = saved["workers"]
            self.checkpoint_versions = {saved["published"]} | {
                lane["root_version"] for lanes in self.saved_workers.values() for lane in lanes
                if lane["root_version"] is not None and lane["root_version"] >= 0}
            # 检查点后的数据文件保留；当前恢复分支只关联已提交的元数据前缀。
            for lanes in self.saved_workers.values():
                for lane in lanes:
                    atomic_json(self.directory / "games" / str(lane["game"]["id"]) / "game.json", lane["metadata"])
        else:
            if checkpoint.exists() or any(self.models.iterdir()):
                raise ValueError("目录已有训练，请使用 --resume 或新目录")
            self.publish()

    def publish(self):
        self.published = self.stats["updates"]
        state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        path = self.models / f"model-{self.published:08d}.pt"
        if not path.exists():
            torch.save(state, path)
        retained = {self.published} | self.checkpoint_versions | {v for _, v in self.bindings.values()} | {
            e["version"] for e in self.pool.entries}
        for candidate in self.models.glob("model-*.pt"):
            if int(candidate.stem.split("-")[1]) not in retained:
                candidate.unlink()

    def game(self):
        number = self.next_game
        self.next_game += 1
        game = {"id": number, "seed": int(self.rng.integers(1, 2**32)), "learner": 1 + number % 2,
                "opponent": "self", "difficulty": "easy"}
        if number < self.config.warmup_games:
            game["opponent"] = "demo"
        else:
            draw = self.rng.random()
            if draw < self.config.teacher_fraction:
                game["opponent"] = "teacher"
            elif draw < self.config.teacher_fraction + self.config.history_fraction and self.pool.entries:
                version = self.pool.choose(self.rng)
                if version != self.published:
                    game.update(opponent="history", version=version)
        return game

    def resolve(self, index, request):
        binding = (index, request["lane"])
        if request["model"] != "current":
            version = int(request["model"])
        elif binding in self.bindings and self.bindings[binding][0] == request["root"]:
            version = self.bindings[binding][1]
        else:
            version = self.published
        self.bindings[binding] = (request["root"], version)
        return version

    def inference_model(self, version):
        if version not in self.inference_models:
            if len(self.inference_models) >= 3:
                self.inference_models.pop(next(iter(self.inference_models)))
            network = Network().to(self.device)
            network.load_state_dict(torch.load(self.models / f"model-{version:08d}.pt",
                                               map_location="cpu", weights_only=True))
            self.inference_models[version] = network.eval()
        network = self.inference_models.pop(version)
        self.inference_models[version] = network
        return network

    def infer(self, jobs):
        output = [None] * len(jobs)
        grouped = {}
        for i, (version, states, actions) in enumerate(jobs):
            bucket = max(1, (len(states) + 31) // 32)
            grouped.setdefault((version, bucket), []).append(i)
        start = time.perf_counter()
        with torch.inference_mode():
            for (version, _), indices in grouped.items():
                model = self.inference_model(version)
                for offset in range(0, len(indices), self.config.inference_batch):
                    selection = indices[offset:offset+self.config.inference_batch]
                    tensors = collate([(jobs[i][1], jobs[i][2]) for i in selection], self.device,
                                      pinned=self.device.type == "cuda")
                    logits, values, aux = model(*tensors)
                    packed = torch.cat((logits, values[:, None], aux), 1).view(torch.uint16).cpu().numpy()
                    for row, i in enumerate(selection):
                        count = len(jobs[i][2])
                        output[i] = np.concatenate((packed[row, :count], packed[row, logits.shape[1]:]))
                    self.stats["inference_batches"] += 1
        self.stats["inference_roots"] += len(jobs)
        self.stats["inference_seconds"] += time.perf_counter() - start
        return output

    def learn(self):
        if self.credit < self.config.batch:
            return
        batch = self.replay.batch(self.config.batch, self.rng)
        if not batch:
            return
        started = time.perf_counter()
        tensors = collate([(b[0], b[1]) for b in batch], self.device, pinned=self.device.type == "cuda")
        target = torch.zeros_like(tensors[3], dtype=torch.bfloat16)
        for i, row in enumerate(batch):
            target[i, :len(row[2])] = torch.from_numpy(row[2]).to(self.device, dtype=torch.bfloat16)
        returns = torch.tensor([b[3] for b in batch], device=self.device, dtype=torch.bfloat16)
        value_mask = torch.tensor([b[4] for b in batch], device=self.device, dtype=torch.bool)
        auxiliary = torch.tensor(np.stack([b[5] for b in batch]), device=self.device, dtype=torch.bfloat16)
        aux_mask = torch.tensor([b[6] for b in batch], device=self.device, dtype=torch.bool)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits, values, predicted_aux = self.model(*tensors)
        logp = F.log_softmax(logits, -1).masked_fill(~tensors[3], 0)
        policy_loss = -(target * logp).sum(-1).mean()
        value_loss = ((values - returns).square() * value_mask).sum() / value_mask.sum().clamp_min(1).to(torch.bfloat16)
        aux_loss = (F.smooth_l1_loss(predicted_aux, auxiliary, reduction="none").mean(-1) * aux_mask).sum() / aux_mask.sum().clamp_min(1).to(torch.bfloat16)
        loss = policy_loss + value_loss + .1 * aux_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("非有限损失")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.credit -= len(batch)
        self.stats["updates"] += 1
        self.stats["trained_samples"] += len(batch)
        self.stats["value_samples"] += int(value_mask.sum())
        self.stats["loss"] = {"policy": float(policy_loss.detach()), "value": float(value_loss.detach()),
                              "auxiliary": float(aux_loss.detach()), "gradient_norm": float(norm)}
        self.stats["learning_seconds"] += time.perf_counter() - started
        if self.stats["updates"] % self.config.publish_updates == 0:
            self.publish()

    def event(self, event):
        if event["type"] == "data":
            path = event["path"]
            self.replay.add(path)
            meta = json.loads(Path(path).read_text(encoding="utf-8"))
            fresh = meta["samples"] - self.seen_samples.get(path, 0)
            self.seen_samples[path] = meta["samples"]
            self.credit += fresh * self.config.sample_reuse
            self.stats["samples"] += fresh
        elif event["type"] == "done":
            game = event["game"]
            if event["winner"] is None:
                self.stats["truncated"] += 1
            else:
                self.stats["completed"] += 1
                if game["opponent"] == "demo":
                    self.stats["demonstrations"] += 1
                if game["opponent"] == "self":
                    self.stats["selfplay_terminal"] += 1
                    self.credit += self.config.batch
                if game["opponent"] == "history":
                    self.pool.record(game["version"], event["winner"] == 3 - game["learner"])
                if self.stats["completed"] % self.config.history_games == 0:
                    self.pool.add(self.published, self.stats["completed"])
            with (self.directory / "results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def checkpoint(self, workers):
        # 学习暂停且所有 worker 已交还槽；未完成搜索在恢复时用原命令 RNG 重算。
        for lanes in workers.values():
            for lane in lanes:
                self.event({"type": "data", "path": str(self.directory / "games" / str(lane["game"]["id"]) / "game.json")})
        payload = {"format": "haojie-classic-training-v1", "config": asdict(self.config),
                   "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
                   "stats": self.stats, "pool": self.pool.entries, "replay": self.replay.games,
                   "published": self.published, "next_game": self.next_game,
                   "seen_samples": self.seen_samples, "credit": self.credit, "workers": workers}
        path = self.directory / "checkpoint.pt"
        torch.save(payload, path.with_suffix(".tmp"))
        os.replace(path.with_suffix(".tmp"), path)
        self.checkpoint_versions = {self.published} | {
            lane["root_version"] for lanes in workers.values() for lane in lanes
            if lane["root_version"] is not None and lane["root_version"] >= 0}

    def run(self):
        ctx = mp.get_context("spawn")
        processes, connections, slots = {}, {}, {}
        pending, snapshots = {}, {}
        awaiting = {}
        start = time.perf_counter()
        initial = copy.deepcopy(self.stats)
        last_status = last_save = start
        mode = None
        previous_handler = signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        failures = []
        totals = {}
        try:
            for index in range(self.config.workers):
                parent, remote = ctx.Pipe()
                slots[index] = Slot()
                restored = self.saved_workers.get(index, []) if self.saved_workers else []
                previous = {lane["lane"]: lane for lane in restored}
                games = [previous[i]["game"] if i in previous else self.game()
                         for i in range(self.config.games_per_worker)]
                resume = [previous.get(i) for i in range(self.config.games_per_worker)]
                process = ctx.Process(target=worker, args=(index, remote, slots[index].name, str(self.directory),
                                                           self.config, games, resume), name=f"haojie-cpu-{index}")
                process.start()
                remote.close()
                processes[index], connections[index] = process, parent
            print(json.dumps({"event": "started", "directory": str(self.directory),
                              "config": asdict(self.config), **self.model.description()}, ensure_ascii=False), flush=True)
            while len(snapshots) < len(processes):
                now = time.perf_counter()
                if mode is None:
                    stopped = (self.stop_requested or (self.config.seconds and now-start >= self.config.seconds)
                               or (self.config.games and self.stats["completed"]-initial["completed"] >= self.config.games)
                               or (self.config.updates and self.stats["updates"]-initial["updates"] >= self.config.updates)
                               or (self.directory / "STOP").exists())
                    mode = "pause" if stopped else ("save" if self.config.save_seconds and now-last_save >= self.config.save_seconds else None)
                active = [c for i, c in connections.items() if i not in snapshots]
                try:
                    ready = wait(active, timeout=.1)
                except KeyboardInterrupt:
                    mode = "pause"
                    continue
                for i, process in processes.items():
                    if i not in snapshots and process.exitcode is not None and connections[i] not in ready:
                        raise RuntimeError(f"worker {i} 意外退出 {process.exitcode}")
                jobs, work = [], []
                for connection in ready:
                    index = next(i for i,c in connections.items() if c is connection)
                    message = connection.recv()
                    if message[0] == "grow":
                        slots[index].close()
                        slots[index] = Slot(size=max(message[1]*2, 16*1024*1024))
                        connection.send(slots[index].name)
                        continue
                    if message[0] == "error":
                        raise RuntimeError(message[1])
                    if message[0] == "saved":
                        snapshots[index] = message[1]
                        continue
                    _, requests, shapes, events, progress = message
                    awaiting[index] = requests
                    totals[index] = progress["commands"]
                    for event in events:
                        self.event(event)
                    arrays = unpack(slots[index], shapes)
                    versions = [self.resolve(index, request) for request in requests]
                    first = len(jobs)
                    jobs.extend((v, arrays[j*2], arrays[j*2+1]) for j,v in enumerate(versions))
                    work.append((index, requests, events, versions, first))
                outputs = self.infer(jobs) if jobs and mode != "pause" else []
                for index, requests, events, versions, first in work:
                    assignments = ([{"lane": event["lane"], "game": self.game()}
                                    for event in events if event["type"] == "done"] if mode is None else [])
                    shapes = pack(slots[index], outputs[first:first+len(requests)]) if mode != "pause" else []
                    response = {"op": mode or "work", "shapes": shapes, "versions": versions, "assignments": assignments}
                    pending[index] = {**response, "op": "work"}
                    connections[index].send(response)
                    awaiting.pop(index, None)
                if mode is None:
                    self.learn()
                self.stats["commands"] = initial["commands"] + sum(totals.values())
                if now-last_status >= 5:
                    self.status("running", start, initial)
                    last_status = now
                if mode and len(snapshots) == len(processes):
                    self.checkpoint(snapshots)
                    if mode == "pause":
                        break
                    for index, connection in connections.items():
                        occupied = {s["lane"] for s in snapshots[index]}
                        pending[index]["assignments"] = [{"lane":i,"game":self.game()}
                            for i in range(self.config.games_per_worker) if i not in occupied]
                        connection.send(pending[index])
                    snapshots.clear()
                    mode = None
                    last_save = time.perf_counter()
            self.status("saved", start, initial)
        except BaseException as error:
            failures.append(str(error))
            self.status("failed", start, initial, error=str(error))
            # 规则异常的 worker 已落盘；其他 worker 在下一次交还共享槽时保存。
            for index in processes:
                failed = self.directory / f"failed-worker-{index}.json"
                if failed.exists():
                    report = json.loads(failed.read_text(encoding="utf-8"))
                    if report.get("pid") == processes[index].pid:
                        snapshots[index] = report["lanes"]
            for index, requests in awaiting.items():
                if index not in snapshots:
                    connections[index].send({"op": "pause", "versions": [self.resolve(index, r) for r in requests]})
            deadline = time.perf_counter() + 30
            while len(snapshots) < len(processes) and time.perf_counter() < deadline:
                active = [c for i,c in connections.items() if i not in snapshots]
                for connection in wait(active, .1):
                    index = next(i for i,c in connections.items() if c is connection)
                    try:
                        message = connection.recv()
                    except EOFError:
                        continue
                    if message[0] == "saved":
                        snapshots[index] = message[1]
                    elif message[0] == "work":
                        requests = message[1]
                        for event in message[3]:
                            self.event(event)
                        connection.send({"op":"pause","versions":[self.resolve(index,r) for r in requests]})
            if len(snapshots) == len(processes):
                self.checkpoint(snapshots)
            raise
        finally:
            for process in processes.values():
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join(10)
            for connection in connections.values():
                connection.close()
            for slot in slots.values():
                slot.close()
            signal.signal(signal.SIGINT, previous_handler)
        return self.stats

    def status(self, status, start, initial, **extra):
        elapsed = time.perf_counter()-start
        value = {"status": status, **self.stats, "elapsed_seconds": elapsed,
                 "commands_per_second": (self.stats["commands"]-initial["commands"])/max(elapsed,.001),
                 "published": self.published, **extra}
        if self.device.type == "cuda":
            value["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
        atomic_json(self.directory / "status.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
