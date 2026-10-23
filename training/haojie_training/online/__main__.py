"""训练控制命令；控制线程只提交请求，CUDA 和快照边界由所有者线程处理。"""

import argparse
import json
import queue
import secrets
import threading
import time
import traceback
import urllib.request
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .native import NativeFailure
from .session import MemoryBudgetExceeded, Session, Settings


def serve(directory, port, commands, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("请求长度必须在 1–4096 字节")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("控制请求必须是对象")
            except (ValueError, UnicodeError) as error:
                self.send_error(400, "invalid control request", str(error))
                return
            future = Future()
            try:
                commands.put((request, future), timeout=5)
                result = future.result(timeout=120)
            except queue.Full:
                self.send_error(503, "control queue full")
                return
            except TimeoutError:
                self.send_error(504, "owner did not reach control boundary")
                return
            payload = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    (directory / "control.json").write_text(
        json.dumps(dict(port=server.server_port, token=token)), encoding="utf-8"
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    parser = argparse.ArgumentParser(description="浩劫原生连续轨迹训练控制")
    parser.add_argument("command", choices=["run", "status", "resize", "budget", "save", "stop"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--memory-mb", type=int)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--save-seconds", type=float, default=300)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        raise ValueError("测量窗口必须为正数")
    if args.save_seconds < 0:
        raise ValueError("保存间隔不能为负数；0 表示只手动保存")
    directory = args.directory.resolve()
    if args.command != "run":
        control = json.loads((directory / "control.json").read_text(encoding="utf-8"))
        request = urllib.request.Request(
            f"http://127.0.0.1:{control['port']}/",
            data=json.dumps(
                dict(command=args.command, concurrency=args.concurrency, memory_mb=args.memory_mb)
            ).encode(),
            headers={"Authorization": "Bearer " + control["token"]},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            print(response.read().decode())
        return
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("启动和恢复需要新的空运行目录，以保留已有模型与原始测量")
    # 独占文件避免重复启动；崩溃时保留诊断和锁，由操作者明确处理。
    lock = directory / "owner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        import os

        stream.write(str(os.getpid()))
    session = None
    server = None
    outcomes = None
    try:
        session = (
            Session.restore(args.resume, args.device)
            if args.resume
            else Session(
                Settings(
                    concurrency=args.concurrency,
                    workers=args.workers,
                    seed=args.seed,
                    memory_bytes=(4096 if args.memory_mb is None else args.memory_mb) * 1024**2,
                ),
                args.device,
            )
        )
        if args.resume and args.memory_mb is not None:
            session.settings.memory_bytes = args.memory_mb * 1024**2
        commands = queue.Queue(maxsize=16)
        outcomes = (directory / "completed.jsonl").open("a", encoding="utf-8", buffering=1)
        session.on_result = lambda result: outcomes.write(json.dumps(result) + "\n")
        server = serve(directory, args.port, commands, secrets.token_urlsafe(32))
        print(
            json.dumps(dict(started=True, directory=str(directory)), ensure_ascii=False), flush=True
        )
        started = time.monotonic()
        last_saved = started
        stop = False
        while not stop and (args.seconds is None or time.monotonic() - started < args.seconds):
            session.tick()
            while not commands.empty():
                request, future = commands.get_nowait()
                action = request.get("command")
                if action == "resize":
                    target = request.get("concurrency")
                    if type(target) is not int or not 0 <= target <= 128:
                        future.set_result(dict(error="目标并发必须在 0–128"))
                        continue
                    session.settings.concurrency = target
                elif action == "budget":
                    value = request.get("memory_mb")
                    minimum = session.memory()
                    if type(value) is not int or value * 1024**2 < minimum:
                        future.set_result(dict(error=f"当前工作至少需要 {minimum} 字节预算"))
                        continue
                    session.settings.memory_bytes = value * 1024**2
                elif action in ("save", "stop"):
                    stop = action == "stop"
                    if action == "save":
                        session.save(directory / "checkpoint.pt", background=True)
                        last_saved = time.monotonic()
                elif action != "status":
                    future.set_result(dict(error="未知命令"))
                    continue
                future.set_result(session.status())
            if (
                not stop
                and args.save_seconds
                and session.saving is None
                and time.monotonic() - last_saved >= args.save_seconds
            ):
                session.save(directory / "checkpoint.pt", background=True)
                last_saved = time.monotonic()
        window = session.status()
        (directory / "window.json").write_text(
            json.dumps(window, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        session.save(directory / "checkpoint.pt")
        (directory / "saved.json").write_text(
            json.dumps(session.status(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(window, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        if session is not None:
            (directory / "window.json").write_text(
                json.dumps(session.status(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            session.save(directory / "checkpoint.pt")
        raise
    except BaseException as error:
        (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if isinstance(error, NativeFailure):
            (directory / "native-failure.json").write_bytes(error.payload)
        if isinstance(error, (MemoryBudgetExceeded, FloatingPointError)) and session is not None:
            (directory / "window.json").write_text(
                json.dumps(session.status(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            session.save(directory / "checkpoint.pt")
        raise
    finally:
        if server:
            server.shutdown()
        try:
            if session:
                session.close()
        finally:
            if outcomes:
                outcomes.close()
            lock.unlink()


if __name__ == "__main__":
    main()
