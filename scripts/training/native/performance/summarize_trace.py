"""汇总原生 Kineto 诊断；CPU 为包含子调用的耗时，设备忙碌按时间区间并集计算。"""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def union(intervals):
    end = float("-inf")
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(path):
    payload = path.read_bytes()
    trace = json.loads(payload)
    events = [
        event
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("dur", 0) > 0
    ]
    categories = {}
    for category in ("cpu_op", "kernel", "cuda_runtime", "gpu_memcpy"):
        aggregate = collections.defaultdict(lambda: [0, 0])
        for event in events:
            if event.get("cat") == category:
                item = aggregate[event["name"]]
                item[0] += 1
                item[1] += event["dur"] / 1000
        categories[category] = [
            {"name": name, "calls": count, "inclusive_ms": duration}
            for name, (count, duration) in sorted(
                aggregate.items(), key=lambda entry: -entry[1][1]
            )
        ]
    gpu = [
        event
        for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    streams = collections.defaultdict(list)
    for event in gpu:
        streams[str(event.get("args", {}).get("stream", "unknown"))].append(
            (event["ts"], event["ts"] + event["dur"])
        )
    profile = [event for event in events if event.get("cat") == "Trace"]
    duration = max((event["dur"] for event in profile), default=0)
    busy = union((event["ts"], event["ts"] + event["dur"]) for event in gpu)
    return {
        "trace": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "profile_ms": duration / 1000,
        "device_activity_union_ms": busy / 1000,
        "device_activity_fraction": busy / duration if duration else None,
        "stream_activity_union_ms": {
            stream: union(intervals) / 1000 for stream, intervals in streams.items()
        },
        "categories": categories,
        "note": "诊断包含 profiler 开销；CPU 包含子调用，设备活动并集不表示 SM 占用率。",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(
        json.dumps({key: value for key, value in result.items() if key != "categories"})
    )
