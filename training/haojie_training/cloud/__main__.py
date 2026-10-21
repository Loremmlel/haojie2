"""独立服务器入口；Notebook 通过回环 API 显示状态和提交控制意图。"""

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

import torch

from ..console.config.resources import torch_threads
from ..console.server import http_server
from .session import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=os.environ.get("HAOJIE_NATIVE"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--settings", type=Path, help="现有控制台设置字段的 JSON 对象")
    parser.add_argument("--backup-dir", type=Path, help="Drive 挂载目录或其他持久存储目录")
    parser.add_argument("--backup-seconds", type=float, default=600)
    parser.add_argument("--backup-keep", type=int, default=3)
    parser.add_argument("--scratch", type=Path, help="本地归档临时目录，默认与训练根同盘")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--restore", type=Path, help="备份目录/完成清单/停止写入的训练根/恢复点")
    sources.add_argument("--import-weights", type=Path, help="仅继承兼容权重，建立新优化器")
    parser.add_argument(
        "--paused", action="store_true", help="加载后暂停，先通过 Notebook 查看设置"
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Torch CPU 算子线程数，0 按设备及有效核配额自动选择",
    )
    parser.add_argument("--connection-file", type=Path, help="将本机 API 地址写入指定本地文件")
    parser.add_argument("--tiny", action="store_true", help="功能验收小模型")
    args = parser.parse_args()
    engine = args.engine or Path(
        "artifacts/native-target/release/haojie-engine" + (".exe" if os.name == "nt" else "")
    )
    try:
        torch.set_num_threads(torch_threads(args.device, args.torch_threads))
        torch.set_num_interop_threads(1)
        settings = json.loads(args.settings.read_text(encoding="utf-8")) if args.settings else None
        session = Session(
            engine,
            args.root,
            device=args.device,
            threads=args.torch_threads,
            settings=settings,
            backup_dir=args.backup_dir,
            backup_seconds=args.backup_seconds,
            backup_keep=args.backup_keep,
            restore=args.restore,
            import_weights=args.import_weights,
            scratch=args.scratch,
            tiny=args.tiny,
        )
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        parser.error(str(error))
    server = control_thread = None
    errors = []
    try:
        server = http_server(session, args.port)
        control_thread = threading.Thread(
            target=server.serve_forever, name="haojie-control", daemon=True
        )
        control_thread.start()
        connection = {"url": f"http://127.0.0.1:{server.server_port}", "pid": os.getpid()}
        if args.connection_file:
            args.connection_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.connection_file.with_name(args.connection_file.name + ".partial")
            temporary.write_text(json.dumps(connection), encoding="utf-8")
            temporary.replace(args.connection_file)
        print(json.dumps(connection), flush=True)
        signal.signal(signal.SIGTERM, lambda *_: session.stop.set())
        if not args.paused:
            session.command("start")
        session.stop.wait()
    except KeyboardInterrupt:
        pass
    finally:
        errors = session.close()
        if server:
            if control_thread and control_thread.is_alive():
                server.shutdown()
            server.server_close()
    if errors:
        print(json.dumps({"shutdown_errors": errors}, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
