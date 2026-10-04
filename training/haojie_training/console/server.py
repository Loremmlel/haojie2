"""仅回环 HTTP 控制面；固定静态路由、来源与令牌校验，无任意命令/删除接口。"""

import argparse
import hmac
import json
import os
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil
import torch

from .controller import Controller
from .opponents import ASSETS
from .storage import Lease, Store


def monitor(controller):
    process = psutil.Process()
    process.cpu_percent()
    psutil.cpu_percent()
    handle = None
    if controller.device.type == "cuda":
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(controller.device.index or 0)
        except Exception:
            pass
    while not controller.shutdown.wait(2):
        try:
            children = process.children(recursive=True)
            controller.telemetry = {
                "rss": process.memory_info().rss + sum(p.memory_info().rss for p in children),
                "cpu_percent": psutil.cpu_percent(),
                "available_ram": psutil.virtual_memory().available,
                "workers": len(children),
                "gpu_allocated": torch.cuda.memory_allocated()
                if torch.cuda.is_available()
                else None,
                "gpu_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else None,
                "gpu_utilization": None,
            }
            if handle is not None:
                try:
                    controller.telemetry["gpu_utilization"] = pynvml.nvmlDeviceGetUtilizationRates(
                        handle
                    ).gpu
                    controller.telemetry["gpu_total"] = pynvml.nvmlDeviceGetMemoryInfo(handle).total
                except Exception:
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if handle is not None:
        pynvml.nvmlShutdown()


def http_server(controller, port=8765):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, data, content_type="application/json; charset=utf-8"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; frame-ancestors 'none'; object-src 'none'",
            )
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.valid_host():
                return self.reply(403, {"error": "仅允许本机来源"})
            if self.path == "/api/status":
                return self.reply(200, controller.status())
            if self.path == "/api/session":
                return self.reply(200, {"token": token})
            routes = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"),
                "/tokens.css": ("tokens.css", "text/css"),
            }
            if self.path not in routes:
                return self.reply(404, {"error": "未找到页面"})
            name, kind = routes[self.path]
            self.reply(200, (ASSETS / name).read_bytes(), kind + "; charset=utf-8")

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (
                not self.valid_host()
                or self.headers.get("Origin") != origin
                or not hmac.compare_digest(self.headers.get("X-Session-Token", ""), token)
            ):
                return self.reply(403, {"error": "来源或会话令牌无效，请刷新页面"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192 or self.path != "/api/command":
                    raise ValueError("无效控制请求")
                self.connection.settimeout(3)
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict) or not isinstance(body.get("action"), str):
                    raise ValueError("控制请求必须包含操作名称")
                controller.command(body["action"], body.get("values"))
                self.reply(202, {"accepted": True})
            except (ValueError, KeyError, TimeoutError) as error:
                self.reply(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def footprint(path, excluded=None):
    """启动时一次盘点旧数据；跳过联接，绝不清理或跟随到其他项目。"""
    total, count = 0, 0
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [
            d
            for d in dirs
            if not (Path(root) / d).is_junction()
            and not (Path(root) / d).is_symlink()
            and (Path(root) / d).resolve() != excluded
        ]
        for name in files:
            candidate = Path(root) / name
            if not candidate.is_symlink():
                total += candidate.stat().st_size
                count += 1
    return {"path": str(path), "bytes": total, "files": count, "managed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=os.environ.get("HAOJIE_NATIVE"))
    parser.add_argument("--root", type=Path, default=Path.home() / ".haojie-training")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--import-weights", type=Path)
    parser.add_argument("--legacy", type=Path, action="append", default=[])
    parser.add_argument(
        "--adopt",
        type=Path,
        action="append",
        default=[],
        help="将指定旧目录计入统一配额，保留原件不自动删除",
    )
    parser.add_argument("--tiny", action="store_true", help="接线验收小模型；默认使用现有正式结构")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    engine = args.engine or Path(
        "artifacts/native-target/release/haojie-engine" + (".exe" if os.name == "nt" else "")
    )
    if not engine.is_file():
        parser.error("原生引擎未安装，请先构建或用 --engine 指定正式 haojie-engine")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (
        args.root.exists()
        and any(args.root.iterdir())
        and not (args.root / "service.lock").exists()
    ):
        parser.error("拒绝将非空用户目录当作工具自有根目录；旧目录请用 --adopt 显式纳管")
    store = Store(args.root)
    lease = Lease(store)
    if args.adopt:
        store.adopt(args.adopt)
    if args.import_weights and store.names("recovery"):
        lease.close()
        parser.error("已有恢复点；导入原件不会覆盖持续学习状态")
    controller = Controller(
        engine, store, device=args.device, tiny=args.tiny, imported=args.import_weights
    )
    # 代码/运行库不是训练产物；已知旧训练位置分别报告，不宣称它们也小于20GB。
    paths = args.legacy or (
        [p for p in Path("artifacts").iterdir() if p.is_dir() and p.name != "native-target"]
        if Path("artifacts").exists()
        else []
    )
    controller.external = [
        footprint(p, store.root)
        for p in paths
        if p.is_dir() and p.resolve() != store.root and str(p.resolve()) not in store.adopted
    ]
    server = http_server(controller, args.port)
    telemetry = threading.Thread(target=monitor, args=(controller,), daemon=True)
    telemetry.start()
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"浩劫训练控制台 {url}；受管目录 {store.root}；关闭页面不停止训练", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()
        server.server_close()
        lease.close()


if __name__ == "__main__":
    main()
