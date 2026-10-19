"""冻结已发布的恢复文件，异步归档；完成标记最后发布，恢复不覆盖原目录。"""

import errno
import hashlib
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..console.config import upgrade
from ..console.storage import FORMAT as RECOVERY_FORMAT
from ..console.storage import LIMIT, Lease, Store

FORMAT = "haojie-cloud-backup-v1"
_MARKER = "haojie-backup-store.json"
_FILES = re.compile(
    r"(?:experiments/[0-9a-f]{12}/)?"
    r"(?:(?:recovery|history|baseline|best|export)/[0-9]+\.pt|metrics/recent\.json)"
)
_ARCHIVE = re.compile(r"haojie-[0-9a-f]{8}-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{12}\.tar\.gz")


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def owned(name):
    return isinstance(name, str) and (name == "active.json" or bool(_FILES.fullmatch(name)))


def _json(path, payload):
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2).encode()
    with path.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def _space(directory, needed):
    if shutil.disk_usage(directory).free < needed + 64 * 2**20:
        raise OSError(f"归档空间不足：需要约 {needed / 2**20:.0f} MiB 和 64 MiB 余量")


def freeze(controller, destination):
    """仅由 Controller.snapshot 在保存边界调用；短暂持锁固定磁盘文件身份。

    Store 替换发布而非原位改写，同盘硬链接在后续轮换后仍有效。
    跨盘时持锁复制；压缩、哈希与远端 I/O 留给备份线程，不读取活跃张量。
    """
    store = controller.root_store
    with store.lock:
        paths = [name for name in store.files if owned(name)]
        if not controller.store.names("recovery"):
            raise ValueError("尚无完整恢复点")
        size = sum(store.files[name] for name in paths)
        _space(destination, size * 2)
        for name in paths:
            source, target = store.root / name, destination / name
            if source.is_symlink() or source.is_junction() or not source.is_file():
                raise ValueError("恢复文件必须是普通文件")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        if "active.json" not in paths:
            _json(destination / "active.json", {"experiment": controller.experiment})
        return {
            "experiment": controller.experiment,
            "updates": controller.saved_step,
            "saved_at": controller.saved_at,
            "recovery": controller.store.names("recovery")[-1],
            "device": str(controller.device),
            "ram_samples": "重新采样；沿用控制台磁盘恢复协议",
        }


def _inventory(root):
    result = {}
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if path.is_symlink() or path.is_junction():
            raise ValueError("导入目录禁止符号链接或联接")
        if path.is_file() and owned(name):
            stat = path.stat()
            result[name] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
    return result


def _check_root(root):
    active = root / "active.json"
    experiment = (
        json.loads(active.read_text(encoding="utf-8"))["experiment"]
        if active.exists() else "legacy"
    )
    store = Store(root)
    scoped = store if experiment == "legacy" else store.scope(experiment)
    for name in reversed(scoped.names("recovery")):
        try:
            payload = torch.load(root / name, weights_only=True, map_location="cpu")
            if payload.get("format") != RECOVERY_FORMAT:
                continue
            if payload["trainer"]["format"] != "haojie-checkpoint-v1":
                continue
            if not all(key in payload for key in ("config", "counts", "random", "saved_at")):
                continue
            upgrade(payload["config"])
            return
        except Exception:
            continue
    raise ValueError("导入内容缺少当前实验的完整恢复点；单独权重请使用 --import-weights")


def _empty(destination):
    if destination.is_symlink() or destination.is_junction():
        raise ValueError("恢复目标不能是链接或联接")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("恢复目标必须为空；已有训练目录保持原样")


def _publish_root(staging, destination):
    destination = Path(destination)
    _empty(destination)
    if destination.exists():
        destination.rmdir()
    (staging / "service.lock").write_bytes(b"0")
    staging.replace(destination)


@contextmanager
def _source_lock(source):
    """本机源采用系统锁；不支持锁的远端只做只读、前后清单与逐文件核对。"""
    if os.name == "nt":
        lease = Lease(Store(source))
        try:
            yield "locked"
        finally:
            lease.close()
        return
    import fcntl

    with (source / "service.lock").open("rb") as file:
        mode = "locked"
        try:
            fcntl.flock(file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in (errno.ENOSYS, errno.EOPNOTSUPP):
                mode = "unsupported; verified stable copy"
            elif error.errno in (errno.EAGAIN, errno.EACCES):
                raise RuntimeError("此受管根目录已有训练服务；请先停止服务") from error
            else:
                raise
        yield mode


def import_root(source, destination):
    """导入停止写入的训练根或单独恢复点；完整校验后发布至空目录。

    完整根复用进程锁并核对复制前后清单；单点包含的优化器与进度原样保留。
    外部纳管目录、锁文件及陌生文件不复制，原目录不写入或清理。
    """
    source, destination = Path(source).resolve(), Path(destination).absolute()
    source_lock = "single published checkpoint"
    _empty(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("恢复源和目标必须是不同且互不包含的目录")
    with tempfile.TemporaryDirectory(prefix="haojie-import-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "root"
        staging.mkdir()
        if source.is_file():
            if source.suffix != ".pt":
                raise ValueError("请选择完成备份清单、训练根或 .pt 恢复点")
            before = source.stat()
            if before.st_size > LIMIT:
                raise ValueError("恢复点超过训练根配额")
            target = staging / "recovery/0000000000000000.pt"
            target.parent.mkdir()
            _space(staging, before.st_size)
            shutil.copyfile(source, target)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino
            ) or digest(source) != digest(target):
                raise ValueError("恢复点在复制过程中变化；请先停止写入")
        else:
            if not (source / "service.lock").is_file():
                raise ValueError("完整训练根应包含 service.lock；请先停止服务并复制整个根目录")
            with _source_lock(source) as source_lock:
                before = _inventory(source)
                size = sum(row[0] for row in before.values())
                if size > LIMIT:
                    raise ValueError("导入文件超过训练根配额")
                _space(staging, size)
                for name in before:
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, target)
                if before != _inventory(source) or any(
                    digest(source / name) != digest(staging / name) for name in before
                ):
                    raise ValueError("训练根在复制过程中变化；请先停止服务并完成文件传输")
        _check_root(staging)
        _publish_root(staging, destination)
    return {"source": str(source), "root": str(destination), "kind": "recovery",
            "source_lock": source_lock}


class Backups:
    """一个显式备份目录内的自有归档；只轮换本目录标识对应的完成版本。"""

    def __init__(self, directory, keep=3, *, create=True):
        if type(keep) is not int or not 2 <= keep <= 100:
            raise ValueError("备份保留份数必须为 2 至 100")
        self.root, self.keep = Path(directory).resolve(), keep
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / _MARKER
        if not marker.exists():
            if not create:
                raise ValueError("所选目录没有备份标识；原目录保持原样")
            try:
                _json(marker, {"format": FORMAT, "owner": uuid.uuid4().hex})
            except FileExistsError:
                pass
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict) or payload.get("format") != FORMAT
            or not isinstance(payload.get("owner"), str)
            or not re.fullmatch("[0-9a-f]{32}", payload["owner"])
        ):
            raise ValueError("备份目录标识无效，原文件保持原样")
        self.owner = payload["owner"]

    def records(self):
        result = []
        for path in self.root.glob("haojie-*.json"):
            try:
                if path.stat().st_size > 2_000_000 or path.is_symlink():
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    continue
                name, created = record.get("archive"), record.get("created_at")
                if (
                    record.get("format") == FORMAT and record.get("owner") == self.owner
                    and isinstance(name, str) and name.startswith(f"haojie-{self.owner[:8]}-")
                    and _ARCHIVE.fullmatch(name)
                    and path.name == name.removesuffix(".tar.gz") + ".json"
                    and type(created) in (int, float) and math.isfinite(created)
                ):
                    result.append((path, record))
            except (OSError, ValueError, TypeError):
                continue
        return sorted(result, key=lambda row: (row[1]["created_at"], row[0].name))

    def publish(self, staging, info, work, progress=lambda _: None):
        if not isinstance(info, dict):
            raise ValueError("备份恢复元数据必须为对象")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        name = f"haojie-{self.owner[:8]}-{stamp}-{uuid.uuid4().hex[:12]}"
        archive = work / (name + ".tar.gz")
        progress("正在本地归档")
        files = {}
        with tarfile.open(archive, "w:gz", compresslevel=1, dereference=True) as packed:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(staging).as_posix()
                    if not owned(relative) or path.is_symlink():
                        raise ValueError("快照包含非受管文件")
                    files[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
                    packed.add(path, arcname=relative, recursive=False)
        record = {
            "format": FORMAT, "owner": self.owner, "created_at": time.time(),
            "archive": archive.name, "bytes": archive.stat().st_size,
            "sha256": digest(archive), "files": files, "checkpoint": info,
        }
        remote = self.root / archive.name
        temporary = remote.with_name(remote.name + ".partial")
        manifest = self.root / (name + ".json")
        manifest_temporary = manifest.with_name(manifest.name + ".partial")
        progress("正在上传并校验备份")
        try:
            _space(self.root, record["bytes"])
            with archive.open("rb") as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, length=4 * 2**20)
                target.flush()
                os.fsync(target.fileno())
            if temporary.stat().st_size != record["bytes"] or digest(temporary) != record["sha256"]:
                raise OSError("远端归档回读校验失败；上一恢复版本保留")
            temporary.replace(remote)
            _json(manifest_temporary, record)
            if json.loads(manifest_temporary.read_text(encoding="utf-8")) != record:
                raise OSError("备份完成清单回读校验失败")
            manifest_temporary.replace(manifest)
        finally:
            temporary.unlink(missing_ok=True)
            manifest_temporary.unlink(missing_ok=True)
        # 新版归档与完成标记均发布后，才轮换本工具自己的旧版。
        warnings = []
        for old_manifest, old in self.records()[:-self.keep]:
            try:
                old_manifest.unlink()
                (self.root / old["archive"]).unlink(missing_ok=True)
            except OSError as error:
                warnings.append(f"旧备份轮换未完成：{error}")
        return {"manifest": str(manifest), **info, "bytes": record["bytes"], "warnings": warnings}

    def restore(self, destination, selected=None, log=lambda _: None):
        """只认完成清单；哈希或内容损坏时尝试前版，不替换非空目标。"""
        destination = Path(destination).absolute()
        _empty(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = self.records()
        if selected is not None:
            rows = [row for row in rows if row[0] == Path(selected).resolve()]
        if not rows:
            if selected is None and any(self.root.glob(f"haojie-{self.owner[:8]}-*.json")):
                raise ValueError("存在备份清单，但没有有效完成版本；未创建新的训练进度")
            return None
        for manifest, record in reversed(rows):
            try:
                return self._restore_one(destination, manifest, record)
            except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
                log(f"备份 {manifest.name} 未通过恢复校验：{error}")
        raise ValueError("没有通过完整性校验的备份；原归档和本地目录均保留")

    def _restore_one(self, destination, manifest, record):
        files, checkpoint = record.get("files"), record.get("checkpoint")
        if not isinstance(checkpoint, dict):
            raise ValueError("备份恢复元数据无效")
        if (
            not isinstance(files, dict) or not files or len(files) > 8192
            or any(not owned(name) for name in files)
            or any(
                not isinstance(row, dict) or type(row.get("bytes")) is not int
                or row["bytes"] < 0 or not isinstance(row.get("sha256"), str)
                or not re.fullmatch("[0-9a-f]{64}", row["sha256"])
                for row in files.values()
            )
        ):
            raise ValueError("备份文件清单无效")
        total = sum(row["bytes"] for row in files.values())
        if total > LIMIT or type(record["bytes"]) is not int or not 0 < record["bytes"] <= LIMIT:
            raise ValueError("备份大小超过训练根配额")
        _space(destination.parent, total + record["bytes"])
        source = self.root / record["archive"]
        if source.is_symlink() or source.stat().st_size != record["bytes"]:
            raise ValueError("归档大小与完成清单不匹配")
        with tempfile.TemporaryDirectory(prefix="haojie-restore-", dir=destination.parent) as folder:
            work = Path(folder)
            archive = work / "backup.tar.gz"
            shutil.copyfile(source, archive)
            if archive.stat().st_size != record["bytes"] or digest(archive) != record["sha256"]:
                raise ValueError("归档大小或 SHA256 不匹配")
            staging = work / "root"
            staging.mkdir()
            seen = set()
            with tarfile.open(archive, "r:gz") as packed:
                for member in packed:
                    if (
                        not member.isfile() or member.name not in files
                        or member.name in seen or member.size != files[member.name]["bytes"]
                    ):
                        raise ValueError("归档包含非法路径、链接或不匹配的文件")
                    target = staging / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with packed.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=4 * 2**20)
                    if digest(target) != files[member.name]["sha256"]:
                        raise ValueError(f"恢复文件 SHA256 不匹配：{member.name}")
                    seen.add(member.name)
            if seen != files.keys():
                raise ValueError("归档缺少清单文件")
            _check_root(staging)
            _publish_root(staging, destination)
        return {**checkpoint, "manifest": str(manifest), "root": str(destination)}


class BackupWorker:
    """最多一个正在上传和一个合并请求；控制器只负责保存与快速冻结。"""

    def __init__(self, controller, backups, *, seconds=600, scratch=None):
        if type(seconds) not in (int, float) or not 10 <= seconds <= 86400:
            raise ValueError("备份间隔必须为 10 至 86400 秒")
        self.controller, self.backups, self.seconds = controller, backups, seconds
        self.scratch = Path(scratch) if scratch else controller.root_store.root.parent
        self.scratch.mkdir(parents=True, exist_ok=True)
        root = controller.root_store.root.resolve()
        if self.scratch.resolve().is_relative_to(root):
            raise ValueError("归档临时目录必须位于活跃训练根之外")
        self.lock, self.wake, self.stop = threading.Lock(), threading.Event(), threading.Event()
        self.pending = None
        self.info = {"state": "等待备份", "error": None, "last": None, "directory": str(backups.root)}
        self.thread = threading.Thread(target=self._loop, name="haojie-backup", daemon=True)
        self.thread.start()

    def status(self):
        with self.lock:
            return {**self.info, "pending": self.pending is not None, "keep": self.backups.keep}

    def request(self):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("备份线程已关闭")
            if self.pending is None:
                self.pending = Future()
            future = self.pending
        self.wake.set()
        return future

    def _progress(self, state):
        with self.lock:
            self.info["state"] = state

    def _loop(self):
        while not self.stop.is_set():
            self.wake.wait(self.seconds)
            self.wake.clear()
            with self.lock:
                if self.stop.is_set():
                    break
                future, self.pending = self.pending or Future(), None
                self.info.update(state="等待控制器保存", error=None)
            if not future.set_running_or_notify_cancel():
                continue
            try:
                with tempfile.TemporaryDirectory(prefix="haojie-backup-", dir=self.scratch) as folder:
                    work = Path(folder)
                    staging = work / "root"
                    staging.mkdir()
                    info = self.controller.snapshot(lambda c: freeze(c, staging)).result()
                    result = self.backups.publish(staging, info, work, self._progress)
                with self.lock:
                    self.info.update(state="备份完成", last=result)
                self.controller.log(f"备份完成：更新 {result['updates']}；{result['manifest']}")
                for warning in result["warnings"]:
                    self.controller.log(warning)
                future.set_result(result)
            except Exception as error:
                with self.lock:
                    self.info.update(state="备份失败，下个周期重试", error=str(error))
                self.controller.log(f"备份失败：{error}")
                future.set_exception(error)

    def close(self):
        self.stop.set()
        self.wake.set()
        with self.lock:
            if self.pending is not None:
                if not self.pending.done():
                    self.pending.set_exception(RuntimeError("服务关闭，备份请求未执行"))
                self.pending = None
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise TimeoutError("备份线程仍在结束写入；此前完成归档保留")
