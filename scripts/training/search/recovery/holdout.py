"""新种子教师复核：完整重放后经二进制分片编码，两个配对模型只做前向。"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from spatial import SpatialPolicyNet

from haojie_training.data import collate_examples, load_dataset, select_batch
from haojie_training.evaluate import evaluate
from haojie_training.model import ModelConfig, PolicyValueNet
from haojie_training.prepare import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "continuous"))
from dataset import digest, save_split  # noqa: E402


def encode(paths, output):
    """不按终局是否成功筛掉策略局；缺失结尾、中断或来源不一致必须失败。"""
    config = ModelConfig.tiny()
    with ThreadPoolExecutor(max_workers=2) as workers:
        files = list(workers.map(lambda p: load_file(p, "node", config, "hard"), paths))
    reference = files[0][0]
    records, examples, games = [], [], []
    identities = set()
    for header, items, _ in files:
        if header != reference:
            raise ValueError("新复核文件的编码与来源不一致")
        for game_id, game in items.items():
            if game_id in identities or not game["outcome"] or game["outcome"]["interrupted"]:
                raise ValueError("重复或未完成的教师记录")
            identities.add(game_id)
            group = game["metadata"]["group"]
            records.extend({**r, "game_id": game_id, "group": group} for r in game["records"])
            examples.extend(game["examples"])
            games.append({"game_id": game_id, "group": group, **game["outcome"]})
    metadata = {
        "synthetic": False,
        "ruleset": reference["ruleset"],
        "encoding": reference["schema"]["encoding"],
        "schema": reference["schema"],
        "source_sha256": reference["source_sha256"],
        "sources": [f[2] for f in files],
        "purpose": "fresh teacher holdout; never optimized",
    }
    summary = save_split(
        output,
        "holdout",
        (collate_examples(examples[i : i + 128], config) for i in range(0, len(examples), 128)),
        metadata,
        records,
    )
    result = {"summary": summary, "games": games, "sources": metadata["sources"]}
    (output / "encoding.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output / "holdout.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.inputs) == bool(args.data):
        parser.error("提供新记录inputs或已编码data，不能同时提供")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    path = encode(args.inputs, args.output) if args.inputs else args.data
    data, metadata = load_dataset(path, ModelConfig.tiny())
    protocol = json.loads((args.pilot / "protocol.json").read_text(encoding="utf-8"))
    training_path = Path(protocol["inputs"]["train"]["path"])
    if digest(training_path) != protocol["inputs"]["train"]["sha256"]:
        raise ValueError("原配对训练数据索引已改变")
    original = torch.load(training_path, map_location="cpu", weights_only=True)["metadata"]
    if metadata["encoding"] != "haojie-entities-factorized-v1" or any(
        metadata[key] != original[key] for key in ("ruleset", "encoding", "schema", "source_sha256")
    ):
        raise ValueError("新复核数据与训练数据的规则/编码不同")
    results = []
    for run in json.loads((args.pilot / "results.json").read_text(encoding="utf-8")):
        checkpoint = args.pilot / f"{run['seed']}-{run['arm']}.pt"
        if digest(checkpoint) != run["checkpoint_sha256"]:
            raise ValueError("配对试验检查点已改变")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["format"] != "haojie-recovery-policy-pilot-v1":
            raise ValueError("检查点不是本次隔离实验")
        if set(payload["training_groups"]) & set(metadata["groups"]):
            raise ValueError("新复核集包含已经训练的种子族")
        cls = SpatialPolicyNet if payload["arm"] == "spatial" else PolicyValueNet
        model = cls(ModelConfig(**payload["config"])).to("cuda").eval()
        model.load_state_dict(payload["model"])
        metrics = evaluate(model, data, torch.device("cuda"), "fp32", 32, metadata["records"])
        row = {
            "seed": run["seed"],
            "arm": run["arm"],
            "checkpoint_sha256": digest(checkpoint),
            "data_sha256": digest(path),
            "metrics": metrics,
            "per_group": {},
        }
        # 每族按实际命令统计，不把大量相邻位置冒充独立实验。
        for group in metadata["groups"]:
            indices = [i for i, r in enumerate(metadata["records"]) if r["group"] == group]
            row["per_group"][group] = evaluate(
                model,
                select_batch(data, torch.tensor(indices)),
                torch.device("cuda"),
                "fp32",
                32,
                [metadata["records"][i] for i in indices],
            )
        results.append(row)
        print(
            json.dumps(
                {
                    "seed": row["seed"],
                    "arm": row["arm"],
                    "nll": metrics["policy_loss"],
                    "path": metrics["teacher_path_accuracy"],
                    "deployment": metrics["by_command_stage"].get("deploy:point"),
                }
            ),
            flush=True,
        )
        del model
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
