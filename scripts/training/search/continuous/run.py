"""固定配置的教师辅助自对弈长跑：阶段回执、历史池、候选筛选和截止时间均可审计。"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from runtime import (
    Stages,
    Stopped,
    contain_process_tree,
    digest,
    identity,
    lock_run,
    read,
    utc,
    write,
)

ROOT = Path(__file__).resolve().parents[4]
PYTHON = str(ROOT / "training/.venv/Scripts/python.exe")
SCRIPTS = ROOT / "scripts/training/search"


def source_files():
    return sorted(
        p
        for folder in (
            ROOT / "src",
            ROOT / "scripts/training",
            ROOT / "training/haojie_training",
        )
        for p in folder.rglob("*")
        if p.suffix in {".ts", ".mjs", ".py"} and "__pycache__" not in p.parts
    )


def choose(results, pairs):
    """所有对局终局才做保守开发筛选；同族相关、样本小，不宣称统计显著或作者胜率。"""

    def score(items):
        if not items or any(not r["terminated"] or r["truncated"] for r in items):
            return None
        return sum(
            0.5 if r["winner"] == "draw" else int(r["winner"] == r["primary"])
            for r in items
        )

    parent = results[: pairs * 2]
    candidate_teacher = results[pairs * 2 : pairs * 2 + 2]
    parent_teacher = results[pairs * 2 + 2 :]
    complete = len(results) == pairs * 2 + 4 and all(
        r["terminated"] and not r["truncated"] for r in results
    )
    accepted = (
        complete
        and score(parent) >= pairs * 1.5
        and score(candidate_teacher) >= score(parent_teacher)
    )
    return {
        "accepted": accepted,
        "all_terminal": complete,
        "candidate_vs_parent_score": score(parent),
        "candidate_vs_parent_games": len(parent),
        "candidate_vs_teacher_score": score(candidate_teacher),
        "parent_vs_teacher_score": score(parent_teacher),
        "reason": "开发筛选通过"
        if accepted
        else "结果不充分或未达到预定门槛，保留父模型",
        "statistical_strength_claim": False,
    }


class Experiment:
    """按冻结协议编排可恢复实验，不修改规则、历史轨迹或发行模型。

    输入配置先校验范围，再冻结源码及基础输入；子进程产物经回放、哈希和数值
    检查后才能进入下一阶段。只复用已完成回执，未完成尝试保留并限次重跑；
    截止、停止或异常会终止所属子进程，模型筛选只影响本实验下一轮父代。
    """

    def __init__(self, root, config, pause_after_round=0):
        self.root, self.config = root, config
        self.pause_after_round = pause_after_round
        self.state = {
            "state": "running",
            "pid": os.getpid(),
            "started": utc(),
            "rounds": [],
        }
        proto = root / "protocol.json"
        if proto.exists():
            self.protocol = read(proto)
            if self.protocol["config"] != config:
                raise ValueError("恢复时不能改变原实验配置，另建实验目录")
        else:
            files = {str(p.relative_to(ROOT)): digest(p) for p in source_files()}
            inputs = {
                str(Path(p).resolve()): digest(p)
                for p in [
                    config["checkpoint"],
                    *config["base_train"],
                    *config["validation"],
                ]
            }
            self.protocol = {
                "format": "haojie-continuous-selfplay-v1",
                "config": config,
                "source_hashes": files,
                "input_hashes": inputs,
                "created": utc(),
                "purpose": "教师辅助开发实验，不升级发行模型",
            }
            write(proto, self.protocol)
            for name in files:
                target = root / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
        self.verify()
        self.stages = Stages(
            root,
            config["deadline"],
            config["minimum_free_gb"],
            config["attempts"],
            config["stage_timeout_seconds"],
            self.verify,
        )

    def verify(self):
        if set(self.protocol["source_hashes"]) != {
            str(p.relative_to(ROOT)) for p in source_files()
        }:
            raise ValueError("实验运行期间源码文件集合改变")
        for name, sha in {
            **self.protocol["source_hashes"],
            **self.protocol["input_hashes"],
        }.items():
            if digest(ROOT / name) != sha:
                raise ValueError(f"冻结的源码或输入改变：{name}")

    def status(self, **fields):
        self.state.update(fields, updated=utc())
        write(self.root / "status.json", self.state)

    def command(self, name, parameters, build):
        return self.stages.run(
            name, {"parameters": parameters, "protocol": identity(self.protocol)}, build
        )

    def parallel(self, workers, function, items):
        futures = {workers.submit(function, item): i for i, item in enumerate(items)}
        results = [None] * len(futures)
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        except BaseException:
            self.stages.cancelled.set()
            raise
        return results

    def game(self, name, job):
        def build(output):
            write(output / "job.json", {**job, "game": 0, "output": str(output)})
            return [
                "node",
                "--import",
                "tsx",
                str(SCRIPTS / "bootstrap/games.ts"),
                "--job",
                str(output / "job.json"),
            ], ["result.json", "game-0.jsonl.gz"]

        folder = self.command(name, job, build)
        audit = self.command(
            name + "-audit",
            {"result": digest(folder / "result.json")},
            lambda out: (
                [
                    "node",
                    "--import",
                    "tsx",
                    str(SCRIPTS / "continuous/audit.ts"),
                    str(folder / "result.json"),
                    str(out / "audit.json"),
                ],
                ["audit.json"],
            ),
        )
        assert read(audit / "audit.json")["passed"]
        return folder

    def gate(self, name, checkpoint, required=True):
        folder = self.command(
            name,
            {"checkpoint": digest(checkpoint), "required": required},
            lambda out: (
                [
                    "node",
                    "--import",
                    "tsx",
                    str(SCRIPTS / "bootstrap/gate.ts"),
                    str(out / "gate"),
                    str(checkpoint),
                    *([] if required else ["--report-only"]),
                ],
                ["gate/summary.json"],
            ),
        )
        gate = read(folder / "gate/summary.json")
        passed = (
            gate["fixtureOptimal"] == 60
            and gate["supportedNaturalWins"] == 10
            and gate["networkCalls"] > 0
        )
        if required and not passed:
            raise ValueError("候选未通过原战术门槛")
        return passed

    def data(self, name, mode, value):
        def build(out):
            if mode == "encode":
                path = value
            else:
                path = out / "input.json"
                write(path, value)
            return [
                PYTHON,
                "-X",
                "utf8",
                str(SCRIPTS / "continuous/dataset.py"),
                mode,
                str(path),
                str(out / "data"),
            ], ["data/manifest.json"]

        signature = {
            "mode": mode,
            "value": value,
            "sha256": digest(value) if mode == "encode" else None,
        }
        return self.command(name, signature, build) / "data"

    def run(self):
        c = self.config
        checkpoint = Path(c["checkpoint"]).resolve()
        pool = []
        self.status(phase="initial-gate", champion=str(checkpoint))
        self.gate("initial-gate", checkpoint)
        limits = {"maxCommands": c["max_commands"], "maxPlies": c["max_plies"]}
        with ThreadPoolExecutor(max_workers=c["workers"]) as workers:
            for number in range(c["rounds"]):
                prefix = f"round-{number:03d}"
                seed = c["seed"] + number * 100
                self.status(round=number, phase="selfplay", champion=str(checkpoint))
                jobs = [
                    {
                        "seed": seed + i,
                        "primary": 1,
                        "kind": "selfplay",
                        "checkpoint": str(checkpoint),
                        **limits,
                    }
                    for i in range(c["games_per_round"])
                ]
                games = self.parallel(
                    workers,
                    lambda item: self.game(f"{prefix}/selfplay-{item[0]:02d}", item[1]),
                    enumerate(jobs),
                )
                self.status(phase="encode")
                encoded = self.parallel(
                    workers,
                    lambda item: self.data(
                        f"{prefix}/encode-{item[0]:02d}",
                        "encode",
                        str(item[1] / "game-0.jsonl.gz"),
                    ),
                    enumerate(games),
                )
                pool.append([str(path / "roots.pt") for path in encoded])
                pool = pool[-c["pool_rounds"] :]
                self.status(phase="dataset")
                data = self.data(
                    f"{prefix}/dataset",
                    "combine",
                    {
                        "train": [
                            *c["base_train"],
                            *(p for batch in pool for p in batch),
                        ],
                        "validation": c["validation"],
                    },
                )
                self.status(phase="train")

                def train(out):
                    return [
                        PYTHON,
                        "-X",
                        "utf8",
                        "-m",
                        "haojie_training.train",
                        "--data",
                        str(data / "train.pt"),
                        "--validation",
                        str(data / "validation.pt"),
                        "--initialize-from",
                        str(checkpoint),
                        "--checkpoint",
                        str(out / "model.pt"),
                        "--report",
                        str(out / "report.json"),
                        "--device",
                        "cuda",
                        "--precision",
                        "bf16",
                        "--threads",
                        "2",
                        "--steps",
                        str(c["steps"]),
                        "--batch-size",
                        "16",
                        "--learning-rate",
                        str(c["learning_rate"]),
                        "--value-weight",
                        "1",
                        "--length-bucket-size",
                        "128",
                        "--seed",
                        str(seed + 90),
                    ], ["model.pt", "report.json"]

                trained = self.command(
                    f"{prefix}/train",
                    {
                        "data": digest(data / "train.pt"),
                        "validation": digest(data / "validation.pt"),
                        "parent": digest(checkpoint),
                    },
                    train,
                )
                candidate = trained / "model.pt"
                self.status(phase="health")
                health = self.data(
                    f"{prefix}/health",
                    "check",
                    {
                        "checkpoint": str(candidate),
                        "report": str(trained / "report.json"),
                        "validation": str(data / "validation.pt"),
                        "steps": c["steps"],
                    },
                )
                assert read(health / "manifest.json")["passed"]
                self.status(phase="candidate-gate")
                gate_passed = self.gate(f"{prefix}/gate", candidate, required=False)
                self.status(phase="evaluation")
                evaluations = []
                for pair in range(c["evaluation_pairs"]):
                    for seat in (1, 2):
                        evaluations.append(
                            {
                                "seed": seed + 20 + pair,
                                "primary": seat,
                                "kind": "evaluation",
                                "checkpoint": str(candidate),
                                "opponentCheckpoint": str(checkpoint),
                                **limits,
                            }
                        )
                for model in (candidate, checkpoint):
                    for seat in (1, 2):
                        evaluations.append(
                            {
                                "seed": seed + 40,
                                "primary": seat,
                                "kind": "evaluation",
                                "checkpoint": str(model),
                                **limits,
                            }
                        )
                evaluated = (
                    self.parallel(
                        workers,
                        lambda item: self.game(
                            f"{prefix}/evaluation-{item[0]:02d}", item[1]
                        ),
                        enumerate(evaluations),
                    )
                    if gate_passed
                    else []
                )
                results = [read(path / "result.json") for path in evaluated]
                selection = choose(results, c["evaluation_pairs"])
                selection["tactical_gate_passed"] = gate_passed
                if not gate_passed:
                    selection["reason"] = "战术门槛未通过，跳过对战并保留父模型"
                report = {
                    "round": number,
                    "parent": str(checkpoint),
                    "parent_sha256": digest(checkpoint),
                    "candidate": str(candidate),
                    "candidate_sha256": digest(candidate),
                    "selection": selection,
                    "training": read(trained / "report.json")["after"],
                    "actual_updates": c["steps"],
                    "dataset": read(data / "manifest.json"),
                    "selfplay": [
                        {
                            k: read(p / "result.json")[k]
                            for k in (
                                "seed",
                                "commands",
                                "ply",
                                "terminated",
                                "truncated",
                                "winner",
                            )
                        }
                        for p in games
                    ],
                    "evaluation": [
                        {
                            k: r[k]
                            for k in (
                                "seed",
                                "primary",
                                "commands",
                                "terminated",
                                "truncated",
                                "winner",
                            )
                        }
                        for r in results
                    ],
                }
                destination = self.root / prefix / "report.json"
                if destination.exists() and read(destination) != report:
                    raise ValueError("已完成轮次的结果发生变化")
                if not destination.exists():
                    write(destination, report)
                if selection["accepted"]:
                    checkpoint = candidate
                self.state["rounds"].append(
                    {
                        "round": number,
                        "accepted": selection["accepted"],
                        "updates": c["steps"],
                        "report": str(destination),
                    }
                )
                self.status(phase="round-complete", champion=str(checkpoint))
                if self.pause_after_round and number + 1 == self.pause_after_round:
                    raise Stopped("已在指定轮次边界暂停，可从回执恢复")
        self.status(state="complete", phase="finished")


def validate(config):
    for key, low, high in (
        ("workers", 1, 4),
        ("rounds", 1, 100),
        ("games_per_round", 2, 8),
        ("evaluation_pairs", 1, 8),
        ("pool_rounds", 1, 10),
        ("steps", 1, 2000),
        ("attempts", 1, 3),
        ("max_commands", 10, 1800),
        ("max_plies", 2, 120),
    ):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"配置范围无效：{key}")
    if not datetime.fromisoformat(config["deadline"]).tzinfo:
        raise ValueError("截止时间必须含时区")
    if (
        not 0 < config["learning_rate"] <= 0.001
        or config["minimum_free_gb"] < 5
        or not 30 <= config["stage_timeout_seconds"] <= 10800
    ):
        raise ValueError("学习率、磁盘或阶段超时配置无效")
    if not 0 < config["seed"] < config["seed"] + config["rounds"] * 100 < 0xFFFFFFFF:
        raise ValueError("种子范围无效")
    if config["purpose"] not in ("trial", "overnight"):
        raise ValueError("必须标记验收或正式长跑用途")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pause-after-round", type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = read(args.config)
    validate(config)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = lock_run(ROOT / "artifacts/training/continuous.lock")
    job = contain_process_tree()
    experiment = None
    try:
        experiment = Experiment(root, config, args.pause_after_round)
        experiment.run()
    except Stopped as error:
        if experiment:
            experiment.status(state="stopped", reason=str(error))
        print(str(error), flush=True)
    except BaseException as error:
        if experiment:
            experiment.stages.cancelled.set()
            experiment.status(state="failed", reason=str(error))
        raise
    finally:
        # Job句柄由进程退出回收；保留Python引用到最后，锁随后自然释放。
        assert lock and (job or os.name != "nt")


if __name__ == "__main__":
    main()
