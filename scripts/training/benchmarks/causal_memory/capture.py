"""只采样、不更新参数，逐条保存真实 MC 请求及隔离缓存所需的公开身份。"""

import argparse
import gzip
import hashlib
import importlib
import json
import random
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path


def audit(requests, model_config):
    """按槽位核验无遗漏顺序、固定前缀与日志追加；产物不包含局面或随机状态。"""
    import torch

    from haojie_training.data import validate_inputs

    previous, roots = {}, Counter()
    digest = hashlib.sha256()
    result = Counter()
    for sequence, record in enumerate(requests):
        assert record["sequence"] == sequence
        row = record["input"]
        validate_inputs({key: value.unsqueeze(0) for key, value in row.items()}, model_config)
        kinds = row["kinds"]
        # roles 的末三项固定为 prefix / argument / path；MC 日志使用独立词类 250。
        dynamic = (kinds == 149) | (kinds == 150) | (kinds == 151) | (kinds == 250)
        positions = dynamic.nonzero().flatten()
        base = int(positions[0]) if len(positions) else len(kinds)
        assert base > 0 and dynamic[base:].all()
        record["base_entities"] = base
        key = tuple(
            record[name]
            for name in (
                "slot",
                "generation",
                "task",
                "model",
                "command_index",
                "actor",
            )
        )
        record["root_key"] = key
        roots[key] += 1
        journal = row["entities"][kinds == 250]
        journal_mask = row["entity_mask"][kinds == 250]
        stream = (record["slot"], record["generation"])
        old = previous.get(stream)
        if old:
            assert record["slot_sequence"] == old["slot_sequence"] + 1
            assert record["request_id"] > old["request_id"]
            if old["root_key"] == key:
                result["same_root_requests"] += 1
                old_row, old_base = old["input"], old["base_entities"]
                stable = base == old_base and all(
                    torch.equal(row[name][:base], old_row[name][:base])
                    for name in ("entities", "kinds", "entity_mask")
                )
                stable_globals = torch.equal(row["globals"][:31], old_row["globals"][:31])
                old_journal = old_row["entities"][old_row["kinds"] == 250]
                old_mask = old_row["entity_mask"][old_row["kinds"] == 250]
                prefix = (
                    len(journal) >= len(old_journal)
                    and torch.equal(journal[: len(old_journal)], old_journal)
                    and torch.equal(journal_mask[: len(old_mask)], old_mask)
                )
                result["base_changes_within_root"] += not stable
                result["global_changes_within_root"] += not stable_globals
                result["journal_prefix_mismatches"] += not prefix
                if stable and stable_globals and prefix:
                    result["reusable_requests"] += 1
                    result["reused_base_tokens"] += base + 1
                    result["reused_journal_tokens"] += len(old_journal)
        else:
            assert record["slot_sequence"] == 0
        previous[stream] = record
        result["full_memory_tokens"] += base + 1 + len(journal)
        for name in sorted(row):
            value = row[name].contiguous()
            result["tensor_bytes"] += value.numel() * value.element_size()
            digest.update(f"{name}:{value.dtype}:{list(value.shape)}".encode())
            digest.update(value.numpy().tobytes())
        digest.update(
            json.dumps({k: v for k, v in record.items() if k != "input"}, sort_keys=True).encode()
        )
        result["max_entities"] = max(result["max_entities"], len(kinds))
        result["max_candidates"] = max(result["max_candidates"], len(row["candidates"]))
        result["max_journal"] = max(result["max_journal"], len(journal))
    return {
        **result,
        "requests": len(requests),
        "roots": len(roots),
        "max_requests_per_root": max(roots.values(), default=0),
        "reusable_request_fraction": result["reusable_requests"] / max(1, len(requests)),
        "requests_sha256": digest.hexdigest(),
    }


def capture(args, mode, provenance):
    import torch

    from haojie_training.benchmark import tensor_hash
    from haojie_training.console.config import validate
    from haojie_training.console.controller import Controller, identity
    from haojie_training.console.sampling import MemoryJobs
    from haojie_training.console.storage import Store
    from haojie_training.data import INPUT_KEYS
    from haojie_training.model import NETWORK_VERSION
    from haojie_training.native.client import Client
    from haojie_training.runtime import synchronize

    class CaptureController(Controller):
        def _loop(self):
            pass

    starts = None
    if mode == "late":
        starts = json.loads(gzip.decompress(args.starts.read_bytes()))
        starts = [starts[i] for i in (0, 2)]
    requests, tasks, slots = [], [], Counter()
    received = 0
    name = (
        mode if mode == "late" or args.natural_commands == 8 else f"natural-{args.natural_commands}"
    )
    report = {**provenance, "mode": mode, "complete": False, "network": NETWORK_VERSION}
    with tempfile.TemporaryDirectory(prefix="haojie-causal-capture-") as folder:
        c = CaptureController(args.engine, Store(Path(folder)), device="cpu", starts=starts)
        c.thread.join()
        c.config.update(
            environments=2,
            games_per_round=2,
            history_percent=0,
            seed=args.seed,
            pool_mib=256,
            inference_batch_size=32,
            batch_wait_ms=0,
            max_commands=20000 if mode == "late" else args.natural_commands,
        )
        validate(c.config)
        c.rng, c.want_run = random.Random(args.seed), True
        original_checked, original_infer, original_finish = (
            Client.checked,
            MemoryJobs.infer,
            MemoryJobs.finish,
        )
        original_progress = c._progress
        began = progress_at = time.perf_counter()

        def checked(client, value):
            nonlocal received
            message = original_checked(client, value)
            if message.get("type") == "infer":
                slot, generation = client.identity
                task = c.pool.slots[slot]
                stream = (slot, generation)
                message["_capture"] = {
                    "sequence": received,
                    "slot": slot,
                    "slot_sequence": slots[stream],
                    "generation": generation,
                    "task": task["id"],
                    "rules": task["context"]["rules"],
                    "request_id": message["id"],
                    "model": message["model"],
                    "actor": message["actor"],
                    "command_index": message["commandIndex"],
                    "ply": message["ply"],
                }
                received += 1
                slots[stream] += 1
            return message

        def infer(jobs, messages, **kwargs):
            output = original_infer(jobs, messages, **kwargs)
            for message in messages:
                requests.append(
                    {
                        **message["_capture"],
                        "input": {key: message["input"][key].clone() for key in sorted(INPUT_KEYS)},
                    }
                )
            return output

        def finish(jobs, task, message):
            tasks.append(
                {
                    "task": task["id"],
                    "rules": task["context"]["rules"],
                    "outcome": message["outcome"],
                    "final_hash": message["finalHash"],
                    "requests": task["requests"],
                    "ply": message["ply"],
                    "error": message.get("error"),
                }
            )
            return original_finish(jobs, task, message)

        def progress(metrics):
            nonlocal progress_at
            original_progress(metrics)
            now = time.perf_counter()
            if now - progress_at >= 30:
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "requests": received,
                            "finished_tasks": len(tasks),
                            "elapsed_seconds": round(now - began, 1),
                        }
                    ),
                    flush=True,
                )
                progress_at = now
            if now - began > args.deadline:
                raise TimeoutError("采样超过保护时间，保留不完整轨迹供诊断")

        Client.checked, MemoryJobs.infer, MemoryJobs.finish = checked, infer, finish
        c._progress = progress
        try:
            c._initialize()
            model_hash = identity(c.trainer.model)
            report.update(
                model_identity=model_hash,
                initial_model_sha256=tensor_hash(c.trainer.model.state_dict()),
                model_config=asdict(c.trainer.model.config),
                config=dict(c.config),
                seed=c.config["seed"],
            )
            with c._phase("sampling"):
                c._sample()
            synchronize(c.device)
            report["setup_and_sampling_seconds_including_capture"] = time.perf_counter() - began
            metrics = c.pool.metrics()
            assert received == len(requests) == metrics["requests"] == metrics["inference_requests"]
            assert len(tasks) == 2 and all(task["error"] is None for task in tasks)
            assert {task["rules"] for task in tasks} == {"classic", "shrine"}
            if mode == "late":
                assert all(task["outcome"]["terminated"] for task in tasks)
            else:
                assert all(
                    task["outcome"]["commands"] == args.natural_commands
                    and task["outcome"]["reason"] == "commands"
                    for task in tasks
                )
            assert c.trainer.updates == c.samples.consumed == 0
            assert identity(c.trainer.model) == model_hash
            assert tensor_hash(c.trainer.model.state_dict()) == report["initial_model_sha256"]
            report.update(
                updates=0,
                samples_consumed=0,
                parameters_unchanged=True,
                sampling=metrics,
            )
            requests.sort(key=lambda row: row["sequence"])
            report["audit"] = audit(requests, c.trainer.model.config)
            report["complete"] = True
        except BaseException as error:
            report["error"] = repr(error)
            raise
        finally:
            Client.checked, MemoryJobs.infer, MemoryJobs.finish = (
                original_checked,
                original_infer,
                original_finish,
            )
            c._close_pool()
            c.close()
            report["tasks"] = sorted(tasks, key=lambda task: task["task"])
            report["wall_seconds"] = time.perf_counter() - began
            output = args.output_dir / f"{name}.pt"
            torch.save({**report, "records": requests}, output)
            with output.open("rb") as stream:
                report["file_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            output.with_suffix(".json").write_text(
                json.dumps(report, ensure_ascii=True, indent=2) + "\n"
            )
    print(
        json.dumps(
            {
                "mode": mode,
                "complete": report["complete"],
                "audit": report.get("audit"),
                "tasks": report["tasks"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--starts", type=Path)
    parser.add_argument("--mode", choices=("late", "natural", "both"), default="both")
    parser.add_argument("--natural-commands", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--deadline", type=float, default=180)
    args = parser.parse_args()
    if args.threads < 1 or args.deadline <= 0 or not 1 <= args.natural_commands <= 20000:
        parser.error("线程数和保护时间必须为正数，开局命令预算为1至20000")
    args.source, args.engine = args.source.resolve(), args.engine.resolve()
    args.starts = args.starts or args.source / "tests/fixtures/native/late-starts.json.gz"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ("late", "natural") if args.mode == "both" else (args.mode,)
    names = [
        mode if mode == "late" or args.natural_commands == 8 else f"natural-{args.natural_commands}"
        for mode in modes
    ]
    if any(
        (args.output_dir / f"{name}.{ext}").exists() for name in names for ext in ("pt", "json")
    ):
        parser.error("输出已存在，请使用新的采集目录，保留原始证据")
    sys.path.insert(0, str(args.source / "training"))
    importlib.import_module("haojie_training")
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    digest = hashlib.sha256()
    for path in sorted((args.source / "training/haojie_training").rglob("*.py")):
        digest.update(path.relative_to(args.source).as_posix().encode())
        digest.update(path.read_bytes())
    provenance = {
        "format": "haojie-causal-memory-trace-v1",
        "measurement_eligible": False,
        "source_sha256": digest.hexdigest(),
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "starts_sha256": hashlib.sha256(args.starts.read_bytes()).hexdigest(),
        "seed": args.seed,
        "device": "cpu",
        "precision": "fp32",
        "threads": args.threads,
        "torch": str(torch.__version__),
        "root_fields": [
            "slot",
            "generation",
            "task",
            "model",
            "command_index",
            "actor",
        ],
    }
    for mode in modes:
        capture(args, mode, provenance)


if __name__ == "__main__":
    main()
