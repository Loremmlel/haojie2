"""浏览器验收专用宿主；复用真实控制器、模型、原生引擎及登记晚盘，禁止当正式棋力。"""

import argparse
import gzip
import json
import threading
from pathlib import Path

import torch

from haojie_training.console.controller import Controller
from haojie_training.console.server import http_server, monitor
from haojie_training.console.storage import Lease, Store

parser = argparse.ArgumentParser()
parser.add_argument("--engine", required=True)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--quota", type=int, default=20_000_000_000)
parser.add_argument("--device", default="cpu")
parser.add_argument("--default-model", action="store_true")
args = parser.parse_args()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
repo = Path(__file__).resolve().parents[3]
starts = json.loads(
    gzip.decompress((repo / "tests/fixtures/native/late-starts.json.gz").read_bytes())
)
store = Store(args.root, limit=args.quota)
lease = Lease(store)
c = Controller(
    args.engine, store, device=args.device, tiny=not args.default_model, starts=starts[:2]
)
c.command(
    "settings",
    {
        "environments": 2,
        "games_per_round": 2,
        "updates_per_round": 2,
        "pool_mib": 16,
        "max_commands": 500,
        "max_plies": 60,
    },
)
server = http_server(c, 0)
threading.Thread(target=monitor, args=(c,), daemon=True).start()
print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}"}), flush=True)
try:
    server.serve_forever(poll_interval=0.1)
except KeyboardInterrupt:
    pass
finally:
    c.close()
    server.server_close()
    lease.close()
