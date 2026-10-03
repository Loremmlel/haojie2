"""比较真实记录的 TS 编码与原生二进制编码，并用同一检查点比较前向。"""

import argparse
import json
from pathlib import Path

import torch

from haojie_training.inference import decode_input
from haojie_training.native.client import Client
from haojie_training.native.pipeline import model_from


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = json.loads(args.reference.read_text())["examples"]
    count, error = 0, 0.0
    with Client(args.engine) as client:
        model, _ = model_from(args.checkpoint, client.ready)
        client.send(
            {"op": "audit", "record": str(args.record.resolve()), "encode": True}
        )
        while True:
            row = client.receive()
            if row["type"] == "done":
                break
            if row["type"] != "example":
                continue
            expected = reference[count]
            assert (row["index"], row["step"]) == (expected["index"], expected["step"])
            native = {k: v.unsqueeze(0) for k, v in row["input"].items()}
            ts = decode_input(expected["input"], model.config)
            for key in ts:
                torch.testing.assert_close(ts[key], native[key], rtol=0, atol=1e-6)
            with torch.inference_mode():
                a, b = model(ts), model(native)
            for x, y in zip(a, b):
                error = max(error, float((x - y).abs().max()))
                torch.testing.assert_close(x, y, rtol=0, atol=1e-6)
            count += 1
    assert count == len(reference)
    print(
        json.dumps({"examples": count, "max_forward_error": error, "tolerance": 1e-6})
    )


if __name__ == "__main__":
    main()
