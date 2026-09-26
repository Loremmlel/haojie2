"""本轮训练/最终留出使用显式用途；共享重放、张量校验和二进制分片，软纠错始终禁用价值。"""

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from haojie_training.data import collate_examples
from haojie_training.model import ModelConfig
from haojie_training.prepare import ROOT, load_file, tensor_example

from scripts.training.search.continuous.dataset import digest, save_split


def experimental_file(path, config, value_only=False):
    """Node校验引用及实局；软纠错无胜负，价值专用流只使用真实执行后的终局。"""
    command = [
        "node",
        "--import",
        "tsx",
        "scripts/training/improvement/value/encoding.ts"
        if value_only
        else "scripts/training/improvement/labels.ts",
        "--encode",
        str(path.resolve()),
    ]
    header, games, current = None, {}, None
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            encoding="utf-8",
        )
        try:
            for line in process.stdout:
                row = json.loads(line)
                if row["type"] == "encoding":
                    if header is not None:
                        raise ValueError("编码头重复")
                    header = row
                elif row["type"] == "game":
                    if current is not None or row["game_id"] in games:
                        raise ValueError("重复或未闭合纠错对局")
                    current = {
                        "metadata": row,
                        "records": [],
                        "examples": [],
                        "outcome": None,
                    }
                    games[row["game_id"]] = current
                elif row["type"] == "example":
                    current["examples"].append(tensor_example(row, config))
                    current["records"].append(
                        {
                            k: row[k]
                            for k in (
                                "index",
                                "step",
                                "actor",
                                "command",
                                "stage",
                                "phase",
                            )
                            if k in row
                        }
                    )
                elif row["type"] == "outcome":
                    if not value_only and (
                        row["terminated"] or row["returns"] is not None or row["winner"] is not None
                    ):
                        raise ValueError("纠错没有反事实终局，不能生成价值标签")
                    if value_only and row["terminated"]:
                        for example, record in zip(current["examples"], current["records"]):
                            if record["step"] != 0:
                                raise ValueError("价值专用流只能包含真实决策根")
                            example["value"].fill_(row["returns"][str(record["actor"])])
                            example["value_mask"].fill_(True)
                    current["outcome"] = row
                    current = None
                else:
                    raise ValueError("未知软编码行")
            if process.wait() != 0:
                errors.seek(0)
                raise ValueError(errors.read())
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
    if header is None or current is not None or not games:
        raise ValueError("软编码流不完整")
    return header, games, {"path": str(path.resolve()), "sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--purpose", choices=["training", "calibration", "final-holdout"], required=True
    )
    parser.add_argument("--soft", action="store_true")
    parser.add_argument("--value-records", action="store_true")
    args = parser.parse_args()
    if args.soft and args.value_records:
        parser.error("软反事实策略与实局价值须分开编码")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    config = ModelConfig.tiny()
    with ThreadPoolExecutor(max_workers=2) as workers:
        files = list(
            workers.map(
                lambda p: (
                    experimental_file(p, config, args.value_records)
                    if args.soft or args.value_records
                    else load_file(p, "node", config, "hard")
                ),
                args.inputs,
            )
        )
    reference = files[0][0]
    records, examples, games, identities = [], [], [], set()
    for header, items, _ in files:
        if header != reference:
            raise ValueError("编码或规则来源不同")
        for identity, game in items.items():
            if identity in identities or game["outcome"] is None or game["outcome"]["interrupted"]:
                raise ValueError("重复或未完成的来源")
            identities.add(identity)
            group = game["metadata"]["group"]
            records.extend({**r, "game_id": identity, "group": group} for r in game["records"])
            examples.extend(game["examples"])
            games.append({"game_id": identity, "group": group, **game["outcome"]})
    metadata = {
        "synthetic": False,
        "ruleset": reference["ruleset"],
        "encoding": reference["schema"]["encoding"],
        "schema": reference["schema"],
        "source_sha256": reference["source_sha256"],
        "sources": [f[2] for f in files],
        "purpose": args.purpose,
        "policy_source": "value-only-behavior"
        if args.value_records
        else "ranked-corrections"
        if args.soft
        else "hard-teacher",
    }
    summary = save_split(
        args.output,
        "data",
        (collate_examples(examples[i : i + 128], config) for i in range(0, len(examples), 128)),
        metadata,
        records,
    )
    report = {
        "summary": summary,
        "games": games,
        "sources": metadata["sources"],
        "purpose": args.purpose,
    }
    (args.output / "encoding.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "path": str(args.output / "data.pt"),
                "sha256": digest(args.output / "data.pt"),
                **summary,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
