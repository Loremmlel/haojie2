"""本机长任务的独占、进程回收与可恢复阶段；历史尝试只追加，回执绑定全部产物。"""

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def lock_run(path):
    """持有文件锁至进程退出，崩溃自动释放；锁文件保留，不能靠删除锁抢占任务。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("已有连续训练进程持有独占锁") from None
    return handle


def contain_process_tree():
    """Windows宿主加入退出即回收的Job，强杀调度器也不会遗留GPU子进程。"""
    if os.name != "nt":
        return None
    from ctypes import wintypes as w

    class Basic(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_int64),
            ("job_time", ctypes.c_int64),
            ("flags", w.DWORD),
            ("minimum", ctypes.c_size_t),
            ("maximum", ctypes.c_size_t),
            ("active", w.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", w.DWORD),
            ("scheduling", w.DWORD),
        ]

    class Extended(ctypes.Structure):
        _fields_ = [
            ("basic", Basic),
            ("io", ctypes.c_uint64 * 6),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process", ctypes.c_size_t),
            ("peak_job", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    kernel.SetInformationJobObject.argtypes = [
        w.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        w.DWORD,
    ]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.GetCurrentProcess.restype = w.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    limits = Extended()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (
        not job
        or not kernel.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        )
        or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    # 不主动关闭：本进程也属于该Job，由系统在进程退出时关闭句柄并回收全部后代。
    return job


class Stopped(Exception):
    pass


class Stages:
    """阶段调用者传入固定签名和命令构造器；成功产物只读复用，失败另建尝试目录。"""

    def __init__(
        self,
        root,
        deadline,
        minimum_free_gb=20,
        attempts=2,
        timeout=7200,
        verify=lambda: None,
    ):
        self.root, self.deadline = Path(root), datetime.fromisoformat(deadline)
        self.minimum_free = minimum_free_gb * 1024**3
        self.attempts, self.timeout = attempts, timeout
        self.cancelled = threading.Event()
        self.verify = verify

    def check(self):
        if (
            self.cancelled.is_set()
            or (self.root / "STOP").exists()
            or datetime.now(timezone.utc) >= self.deadline
        ):
            raise Stopped("收到停止请求或到达运行截止时间")
        if shutil.disk_usage(self.root).free < self.minimum_free:
            raise Stopped("可用磁盘低于约定下限，保留全部产物")

    def run(self, name, signature, build):
        self.verify()
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        receipt = folder / "done.json"
        key = identity(signature)
        if receipt.exists():
            done = read(receipt)
            if done["signature"] != key:
                raise ValueError(f"已完成阶段配置变化：{name}")
            output = folder / done["attempt"]
            for file, sha in done["files"].items():
                if digest(output / file) != sha:
                    raise ValueError(f"已完成产物被修改：{name}/{file}")
            return output
        existing = sorted(folder.glob("attempt-*"))
        for attempt in range(len(existing), self.attempts):
            self.check()
            output = folder / f"attempt-{attempt:02d}"
            output.mkdir()
            started = time.monotonic()
            try:
                command, required = build(output)
                write(output / "command.json", command)
                with (output / "process.log").open("w", encoding="utf-8") as log:
                    child = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW
                        if os.name == "nt"
                        else 0,
                        start_new_session=os.name != "nt",
                    )
                    write(
                        folder / "status.json",
                        {
                            "state": "running",
                            "pid": child.pid,
                            "attempt": attempt,
                            "started": utc(),
                        },
                    )
                    try:
                        while child.poll() is None:
                            self.check()
                            if time.monotonic() - started > self.timeout:
                                raise TimeoutError(f"阶段超时：{name}")
                            time.sleep(1)
                        if child.returncode:
                            raise RuntimeError(
                                f"阶段退出码{child.returncode}：{name}，见{output / 'process.log'}"
                            )
                    finally:
                        if child.poll() is None:
                            if os.name == "nt":
                                subprocess.run(
                                    ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                    creationflags=subprocess.CREATE_NO_WINDOW,
                                )
                            else:
                                import signal

                                os.killpg(child.pid, signal.SIGTERM)
                            child.wait()
                if any(not (output / file).is_file() for file in required):
                    raise ValueError(f"阶段缺少完成产物：{name}")
                self.verify()
                files = {
                    str(p.relative_to(output)): digest(p)
                    for p in output.rglob("*")
                    if p.is_file()
                }
                write(
                    receipt,
                    {
                        "signature": key,
                        "attempt": output.name,
                        "files": files,
                        "elapsed_seconds": time.monotonic() - started,
                        "completed": utc(),
                    },
                )
                write(folder / "status.json", {"state": "complete", "completed": utc()})
                return output
            except BaseException as error:
                write(output / "failure.json", {"error": str(error), "time": utc()})
                write(
                    folder / "status.json",
                    {
                        "state": "stopped" if isinstance(error, Stopped) else "failed",
                        "error": str(error),
                        "attempt": attempt,
                    },
                )
                if isinstance(error, (Stopped, KeyboardInterrupt, SystemExit)):
                    raise
        raise RuntimeError(f"阶段已耗尽{self.attempts}次尝试：{name}")
