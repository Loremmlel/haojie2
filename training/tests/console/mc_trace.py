"""给 TS 差分测试返回真实原生决策张量；仅验证，不属于控制台运行依赖。"""

import json
import sys

from haojie_training.native.client import Client

if "--real" in sys.argv:
    start = json.load(sys.stdin)
    rows, last = [], None
    with Client(sys.argv[1]) as client:
        client.send(
            {
                "op": "sample",
                "start": start,
                "memory": True,
                "mcContext": True,
                "model": "a" * 64,
                "samplerSeed": 871,
                "maxCommands": 1,
                "maxPlies": 1,
            }
        )
        while True:
            message = client.receive()
            if message["type"] == "infer":
                last = {k: v.tolist() for k, v in message["input"].items()}
                scores = [100.0 if row[63] == 1 else 0.0 for row in last["candidates"]]
                client.send({"id": message["id"], "model": message["model"], "logits": scores})
            elif message["type"] == "example":
                inputs = {k: v.tolist() for k, v in message["input"].items()}
                if sum(inputs["candidate_mask"]) > 1:
                    assert inputs == last
                rows.append(
                    {
                        "input": inputs,
                        "selected": message["selected"],
                        "depth": message["step"],
                        "stage": message["stage"],
                        "pass": message["pass"],
                    }
                )
            elif message["type"] == "done":
                print(json.dumps({"rows": rows, "done": message}))
                sys.exit(0)

result = []
with Client(sys.argv[1]) as client:
    for seed, scores in ((17, [100, 0, 0, -100]), (29, [0, 0, 100, -100]), (43, [0, 0, 0, 100])):
        client.send({"op": "mc-exercise", "case": 1, "samplerSeed": seed, "outcomeSeed": 77})
        rows, last = [], None
        while True:
            message = client.receive()
            if message["type"] == "infer":
                last = {k: v.tolist() for k, v in message["input"].items()}
                values = scores[: message["candidates"]] if message["candidates"] != 2 else [0, 100]
                client.send({"id": message["id"], "model": message["model"], "logits": values})
            elif message["type"] == "example":
                inputs = {k: v.tolist() for k, v in message["input"].items()}
                if sum(inputs["candidate_mask"]) > 1:
                    assert inputs == last, "样本必须是实际推理输入"
                rows.append(
                    {
                        "input": inputs,
                        "selected": message["selected"],
                        "depth": message["step"],
                        "stage": message["stage"],
                        "pass": message["pass"],
                    }
                )
            elif message["type"] == "done":
                result.append(
                    {
                        "seed": seed,
                        "scores": scores,
                        "rows": rows,
                        "path": message["path"],
                        "pass": message["pass"],
                    }
                )
                break
print(json.dumps(result))
