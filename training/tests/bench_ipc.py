"""同载荷比较 Queue、Pipe 和持久共享槽；启动时间与稳态往返分开计量。"""

import argparse
import json
import multiprocessing as mp
import statistics
import time

import numpy as np

from training.haojie.runtime.ipc import Slot


def child(mode, connection, incoming, outgoing, name, count, rounds):
    slot = Slot(name) if mode == "shared" else None
    connection.send("ready")
    for _ in range(rounds):
        if mode == "queue":
            value = incoming.get()
        elif mode == "pipe":
            value = connection.recv()
        else:
            connection.recv_bytes()
            value = slot.array(count)
        # 确认每次完整读取真实载荷，返回相同大小的结果。
        value += np.float32(1)
        if mode == "queue":
            outgoing.put(value)
        elif mode == "pipe":
            connection.send(value)
        else:
            connection.send_bytes(b"1")
    if slot:
        del value
        slot.close()
    connection.close()


def run(mode, count, rounds):
    ctx = mp.get_context("spawn")
    parent, remote = ctx.Pipe()
    incoming, outgoing = (ctx.Queue(1), ctx.Queue(1)) if mode == "queue" else (None, None)
    slot = Slot(size=count * 4) if mode == "shared" else None
    value = np.zeros(count, np.float32)
    if slot:
        slot.array(count)[:] = value
    process = ctx.Process(target=child, args=(mode, remote, incoming, outgoing,
                                            slot.name if slot else None, count, rounds))
    started = time.perf_counter()
    process.start()
    remote.close()
    assert parent.recv() == "ready"
    startup = time.perf_counter() - started
    timings = []
    for i in range(rounds):
        start = time.perf_counter()
        if mode == "queue":
            incoming.put(value)
            value = outgoing.get(timeout=30)
        elif mode == "pipe":
            parent.send(value)
            value = parent.recv()
        else:
            # 包括正式 worker 填槽和消费输出的复制，不只测通知。
            slot.array(count)[:] = value
            parent.send_bytes(b"1")
            parent.recv_bytes()
            value = slot.array(count).copy()
        timings.append(time.perf_counter() - start)
        assert value[0] == i + 1 and value[-1] == i + 1
    process.join(30)
    assert process.exitcode == 0
    parent.close()
    if slot:
        slot.close()
    for queue in (incoming, outgoing):
        if queue:
            queue.close()
            queue.join_thread()
    return {"startup_s": startup, "median_us": statistics.median(timings[10:]) * 1e6,
            "p95_us": sorted(timings[10:])[int((rounds - 11) * .95)] * 1e6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=300)
    args = parser.parse_args()
    results = {}
    for size in (4096, 65536, 262144, 1048576):
        trials = {mode: [] for mode in ("queue", "pipe", "shared")}
        for repeat in range(3):
            order = list(trials)
            order = order[repeat:] + order[:repeat]
            for mode in order:
                trials[mode].append(run(mode, size // 4, args.rounds))
        results[size] = {mode: {"median_us": statistics.median(t["median_us"] for t in samples),
                                "p95_us": statistics.median(t["p95_us"] for t in samples),
                                "startup_s": statistics.median(t["startup_s"] for t in samples)}
                         for mode, samples in trials.items()}
        print(json.dumps({"bytes": size, **results[size]}), flush=True)


if __name__ == "__main__":
    main()
