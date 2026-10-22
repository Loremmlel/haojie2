"""固定真实公开帧的主机解码、CUDA 组批准备及评分回传对照；不计为 GPU 吞吐。"""

import argparse
import gzip
import hashlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture", type=Path, metavar="ENGINE")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "training"))
    import numpy as np
    import torch

    from haojie_training.model import ModelConfig
    from haojie_training.native import client as protocol
    from haojie_training.native.packed import PackedPolicy

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.capture:
        starts = json.loads(
            gzip.decompress(
                (args.source / "tests/fixtures/native/late-starts.json.gz").read_bytes()
            )
        )
        rows = []
        with protocol.Client(args.capture) as client:
            for start in starts:
                count = 0
                client.send(
                    {
                        "op": "sample",
                        "start": start,
                        "model": "a" * 64,
                        "samplerSeed": 91,
                        "maxCommands": 32,
                        "maxPlies": 1000,
                        "memory": True,
                        "mcContext": True,
                        "inputRefs": True,
                    }
                )
                while True:
                    message = client.receive()
                    if message["type"] == "done":
                        break
                    if message["type"] != "infer":
                        continue
                    count += 1
                    if count <= 32:
                        rows.append(message["input"])
                    response = {"id": message["id"], "model": message["model"]}
                    response.update(
                        {"cancel": True}
                        if count > 32
                        else {"logits": [0.0] * message["candidates"]}
                    )
                    client.send(response)
        args.requests.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "haojie-host-requests-v1", "requests": rows}, args.requests)
        print(json.dumps({"requests": len(rows), "path": str(args.requests)}))
        return
    if not args.output or min(args.repeats, args.batch_size) < 1:
        parser.error("计时需要 --output，重复数及批量必须为正整数")
    payload = torch.load(args.requests, weights_only=True, map_location="cpu")
    if payload["format"] != "haojie-host-requests-v1" or not payload["requests"]:
        raise ValueError("需要固定的真实公开请求")
    rows = payload["requests"]
    keys = (
        "entities",
        "globals",
        "candidates",
        "kinds",
        "sources",
        "targets",
        "entity_mask",
        "candidate_mask",
    )
    frames = [b"".join(row[key].numpy().tobytes() for key in keys) for row in rows]
    groups = [rows[i : i + args.batch_size] for i in range(0, len(rows), args.batch_size)]
    arrays = "_arrays" in inspect.signature(protocol.decode).parameters
    if arrays:
        groups = [
            [{key: value.numpy() for key, value in row.items()} for row in group]
            for group in groups
        ]
    packed = PackedPolicy(SimpleNamespace(config=ModelConfig()), {}, False)
    scores = [np.linspace(-3, 3, len(row["sources"]), dtype=np.float32) for row in rows]

    class Sink:
        def write(self, data):
            return len(data)

        def flush(self):
            pass

    client = protocol.Client.__new__(protocol.Client)
    client.process = SimpleNamespace(stdin=Sink())
    client.ready = {"capabilities": ["logits-f32-v1"]}
    binary = hasattr(protocol, "LOGITS_CAPABILITY")

    def decode():
        for row, frame in zip(rows, frames):
            options = {"_arrays": True} if arrays else {}
            protocol.decode(frame, len(row["entities"]), len(row["sources"]), **options)

    def batch():
        for group in groups:
            packed.batch(group)

    def respond():
        for index, score in enumerate(scores):
            client.send(
                {
                    "id": index + 1,
                    "model": "a" * 64,
                    "logits": score if binary else score.tolist(),
                }
            )

    # 逐字段摘要约束组批数值和布局；不同输出编码另外由原生集成测试核对轨迹。
    digest = hashlib.sha256()
    for group in groups:
        tensors, layout = packed.batch(group)
        digest.update(json.dumps(layout, sort_keys=True).encode())
        for key, tensor in sorted(tensors.items()):
            digest.update(key.encode())
            digest.update(tensor.numpy().tobytes())
    timings = {}
    for name, run in (
        ("decode", decode),
        ("packed_batch", batch),
        ("respond", respond),
    ):
        run()
        began, cpu = time.perf_counter(), time.process_time()
        for _ in range(args.repeats):
            run()
        timings[name] = {
            "seconds": time.perf_counter() - began,
            "cpu_seconds": time.process_time() - cpu,
        }
    report = {
        "scope": "fixed-real-input-host-only",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "requests": len(rows),
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "packed_sha256": digest.hexdigest(),
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "binary_logits": binary,
        "timings": timings,
        "numpy_inputs": arrays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
