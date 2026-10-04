"""所有受管写入先预留完整字节；临时副本也在同一根目录与配额内。"""

import hashlib
import io
import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

import torch

LIMIT = 20_000_000_000


class QuotaError(RuntimeError):
    pass


class Store:
    def __init__(self, root, limit=LIMIT):
        if Path(root).is_symlink() or Path(root).is_junction():
            raise ValueError("受管根目录不能是联接或符号链接")
        self.root = Path(root).resolve()
        if not 0 < limit <= LIMIT:
            raise ValueError("配额必须在1至20,000,000,000字节内")
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit, self.reserved, self.written = limit, 0, 0
        self.lock = threading.RLock()
        self.writes = deque(maxlen=120)
        self.files = {}
        self.adopted = []
        manifest = self.root / "adopted.json"
        if manifest.exists():
            self.adopted = json.loads(manifest.read_text(encoding="utf-8"))
        self.reconcile()

    def reconcile(self):
        with self.lock:
            files = {}
            for path in self.root.rglob("*"):
                if path.is_symlink() or path.is_junction():
                    raise ValueError("受管目录禁止符号链接或联接")
                if path.is_file():
                    files[path.relative_to(self.root).as_posix()] = path.stat().st_size
            self.files = files
            for i, entry in enumerate(self.adopted):
                # 显式纳管的原目录只计账、保护原件；不拥有删除权。
                total = 0
                for directory, dirs, names in os.walk(entry):
                    dirs[:] = [
                        d
                        for d in dirs
                        if not (Path(directory) / d).is_junction()
                        and not (Path(directory) / d).is_symlink()
                    ]
                    total += sum(
                        (Path(directory) / n).stat().st_size
                        for n in names
                        if not (Path(directory) / n).is_symlink()
                    )
                self.files[f"adopted/{i}"] = total

    def adopt(self, directories):
        previous = self.adopted[:]
        for directory in directories:
            path = Path(directory).resolve(strict=True)
            if (
                not path.is_dir()
                or path == self.root
                or path in self.root.parents
                or self.root in path.parents
            ):
                raise ValueError("纳管目录不能包含受管根目录或被其包含")
            if any(
                path == Path(p) or path in Path(p).parents or Path(p) in path.parents
                for p in self.adopted
            ):
                continue
            self.adopted.append(str(path))
        try:
            self.reconcile()
            self.write("adopted.json", json.dumps(self.adopted).encode())
        except Exception:
            self.adopted = previous
            self.reconcile()
            raise

    def snapshot(self):
        with self.lock:
            categories = {}
            for name, size in self.files.items():
                parts = name.split("/")
                category = parts[2] if len(parts) > 2 and parts[0] == "experiments" else parts[0]
                categories[category] = categories.get(category, 0) + size
            recent = sum(size for at, size in self.writes if time.monotonic() - at < 60)
            return {
                "used": sum(self.files.values()),
                "reserved": self.reserved,
                "limit": self.limit,
                "categories": categories,
                "written": self.written,
                "write_rate": recent / 60,
                "root": str(self.root),
                "adopted": self.adopted[:],
            }

    def remove(self, name):
        # 删除仅接受内部索引的自有文件，不向 HTTP 暴露路径 API。
        with self.lock:
            if name in self.files and not name.startswith("adopted/"):
                (self.root / name).unlink(missing_ok=True)
                self.files.pop(name)

    def names(self, category):
        return sorted(n for n in self.files if n.startswith(category + "/") and n.endswith(".pt"))

    def reclaim(self, needed):
        # 最后一份恢复点受保护；候选保存验证前不轮换最近两份。
        for category, keep in (("recovery", 2), ("history", 8), ("best", 1)):
            for name in self.names(category)[:-keep]:
                self.remove(name)
        for name in list(self.files):
            if name in {"active.json.partial", "adopted.json.partial"}:
                self.remove(name)
                continue
            if name.endswith(".partial") and name.split("/")[-2] in {
                "recovery",
                "history",
                "best",
                "metrics",
                "baseline",
                "export",
            }:
                self.remove(name)
        if sum(self.files.values()) + self.reserved + needed > self.limit:
            raise QuotaError("20GB受管配额不足；保留有效恢复点，已拒绝写入并暂停")

    def write(self, name, data, verify=None):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("非法受管路径")
        with self.lock:
            self.reclaim(len(data))
            self.reserved += len(data)
            target = self.root / name
            temporary = target.with_name(target.name + ".partial")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with temporary.open("xb") as file:
                    file.write(data)
                    file.flush()
                    os.fsync(file.fileno())
                self.written += len(data)
                now = int(time.monotonic())
                if self.writes and self.writes[-1][0] == now:
                    at, size = self.writes.pop()
                    self.writes.append((at, size + len(data)))
                else:
                    self.writes.append((now, len(data)))
                # 回读校验发生在发布/轮换前；最后一个恢复点永不提前删除。
                if hashlib.sha256(temporary.read_bytes()).digest() != hashlib.sha256(data).digest():
                    raise IOError("写后哈希校验失败")
                if verify:
                    verify(temporary)
                temporary.replace(target)
                self.files[name] = len(data)
            finally:
                temporary.unlink(missing_ok=True)
                self.reserved -= len(data)

    def save_tensor(self, category, step, payload, keep):
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        name = f"{category}/{step:016d}.pt"
        self.write(
            name, buffer.getvalue(), lambda p: torch.load(p, weights_only=True, map_location="cpu")
        )
        names = self.names(category)
        if category.split("/")[-1] == "history":
            # 保留首尾，优先移除时间距离最密集的中间权重，避免只剩最近一小段。
            while len(names) > keep:
                steps = [int(Path(n).stem) for n in names]
                index = min(range(1, len(names) - 1), key=lambda i: steps[i + 1] - steps[i - 1])
                self.remove(names.pop(index))
        else:
            for old in names[:-keep]:
                self.remove(old)
        return name

    def metrics(self, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if len(data) > 2_000_000:
            raise QuotaError("指标超过2MB额度")
        self.write("metrics/recent.json", data)

    def read_metrics(self, prefix=""):
        path = self.root / prefix / "metrics/recent.json"
        if not path.exists():
            return {}
        if path.stat().st_size > 2_000_000:
            raise ValueError("指标摘要超过2MB")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("指标摘要必须是对象")
        return payload

    def scope(self, experiment):
        if not re.fullmatch(r"[0-9a-f]{12}", experiment):
            raise ValueError("实验身份无效")
        return ExperimentStore(self, experiment)

    def experiments(self):
        result = {}
        with self.lock:
            for name, size in self.files.items():
                parts = name.split("/")
                if len(parts) >= 4 and parts[0] == "experiments":
                    identity, category = parts[1:3]
                    if not re.fullmatch(r"[0-9a-f]{12}", identity):
                        continue
                elif len(parts) == 2 and parts[0] in {
                    "recovery",
                    "history",
                    "baseline",
                    "best",
                    "export",
                }:
                    identity, category = "legacy", parts[0]
                else:
                    continue
                entry = result.setdefault(identity, {"id": identity, "bytes": 0, "recoveries": 0})
                entry["bytes"] += size
                entry["recoveries"] += int(category == "recovery" and name.endswith(".pt"))
        return sorted(result.values(), key=lambda entry: entry["id"])


class ExperimentStore:
    """实验只有路径命名空间；计账、预留和写入锁始终属于同一个全局 Store。"""

    def __init__(self, store, experiment):
        self.store, self.root = store, store.root
        self.prefix = f"experiments/{experiment}"

    def names(self, category):
        return self.store.names(f"{self.prefix}/{category}")

    def save_tensor(self, category, step, payload, keep):
        return self.store.save_tensor(f"{self.prefix}/{category}", step, payload, keep)

    def metrics(self, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if len(data) > 2_000_000:
            raise QuotaError("指标超过2MB额度")
        self.store.write(f"{self.prefix}/metrics/recent.json", data)

    def snapshot(self):
        return self.store.snapshot()

    def read_metrics(self):
        return self.store.read_metrics(self.prefix)

    def reconcile(self):
        self.store.reconcile()


class Lease:
    """操作系统释放的进程锁；崩溃不会留下必须手删的锁文件。"""

    def __init__(self, store):
        path = store.root / "service.lock"
        if not path.exists():
            store.write("service.lock", b"0")
        self.file = path.open("r+b")
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("此受管根目录已有训练服务；请连接已有页面") from None

    def close(self):
        self.file.close()
