"""仅监听本机的 JSON 控制接口及单文件页面。"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
from urllib.parse import urlsplit, parse_qs

from .controller import Controller
from .settings import defaults, schema


def server(controller, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, value, content_type="application/json; charset=utf-8"):
            data = (
                value
                if isinstance(value, bytes)
                else json.dumps(value, ensure_ascii=False, allow_nan=False).encode(
                    "utf-8"
                )
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/":
                    self.reply(
                        200,
                        Path(__file__).with_name("index.html").read_bytes(),
                        "text/html; charset=utf-8",
                    )
                elif url.path == "/api/state":
                    since = int(parse_qs(url.query).get("since", [0])[0])
                    self.reply(200, controller.snapshot(since))
                elif url.path == "/api/schema":
                    self.reply(200, dict(defaults=defaults(), fields=schema()))
                else:
                    self.reply(404, {"error": "页面不存在"})
            except (ValueError, OSError) as error:
                self.reply(400, {"error": str(error)})

        def do_POST(self):
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            allowed = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            if (
                host not in allowed
                or origin not in (None, "http://" + host)
                or self.headers.get("X-Haojie-Control") != "1"
            ):
                self.reply(403, {"error": "控制请求须来自本机控制台"})
                return
            try:
                if (
                    self.path != "/api/control"
                    or self.headers.get_content_type() != "application/json"
                ):
                    raise ValueError("无效的控制请求")
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 65536:
                    raise ValueError("请求大小无效")
                data = json.loads(self.rfile.read(size))
                result = controller.control(**data)
                self.reply(200, result)
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.reply(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(root, port):
    controller = Controller(root)
    http = server(controller, port)

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    print(
        f"浩劫训练台：http://127.0.0.1:{http.server_port}  ·  {controller.root}",
        flush=True,
    )
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        http.server_close()
        controller.close()
