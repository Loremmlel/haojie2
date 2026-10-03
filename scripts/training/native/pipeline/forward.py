"""比较真实记录的 TS 编码与原生二进制编码，并用同一检查点比较前向。"""

import argparse
import json
from pathlib import Path

import torch

from haojie_training.inference import decode_input
from haojie_training.data import collate_examples
from haojie_training.native.client import Client
from haojie_training.native.pipeline import example, model_from


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
    inputs = []
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
            inputs.append(row["input"])
    assert count == len(reference)
    batch_error = 0.0
    # 预声明容差同时约束就绪调度带来的批次形状变化；比较所有有效候选和价值。
    with torch.inference_mode():
        singles = [
            model({k: v.unsqueeze(0) for k, v in item.items()}) for item in inputs
        ]
        for size in (2, 8):
            for offset in range(0, len(inputs), size):
                chunk = inputs[offset : offset + size]
                batch = collate_examples(
                    [example(item) for item in chunk], model.config
                )
                logits, values = model(batch)
                for index, item in enumerate(chunk):
                    reference_logits, reference_value = singles[offset + index]
                    for actual, expected in (
                        (logits[index, : len(item["candidates"])], reference_logits[0]),
                        (values[index], reference_value[0]),
                    ):
                        batch_error = max(
                            batch_error, float((actual - expected).abs().max())
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    print(
        json.dumps(
            {
                "examples": count,
                "max_forward_error": error,
                "max_batch_error": batch_error,
                "tolerance": 1e-6,
            }
        )
    )


if __name__ == "__main__":
    main()
