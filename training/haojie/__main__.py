"""普通局训练命令行。"""

import argparse
from dataclasses import fields
from pathlib import Path

from .runtime.config import Config, cpu_limit


def main():
    parser = argparse.ArgumentParser(description="浩劫普通局搜索自对弈")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--directory", required=True)
    run.add_argument("--resume", action="store_true")
    defaults = Config()
    for field in fields(Config):
        run.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(getattr(defaults, field.name)),
        )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--opponent", default="easy")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--games", type=int, default=20)
    evaluate.add_argument("--simulations", type=int, default=64)
    evaluate.add_argument("--max-commands", type=int, default=20000)
    evaluate.add_argument("--seed", type=int, default=709901)
    evaluate.add_argument("--device", default="cuda")
    evaluation_round = commands.add_parser(
        "evaluate-round", help="运行一轮三档配对评测"
    )
    evaluation_round.add_argument("--directory", required=True)
    web = commands.add_parser("web", help="本地训练控制台")
    web.add_argument("--root", default="artifacts/training")
    web.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "run":
        from .runtime.trainer import Trainer

        if args.resume:
            import torch

            saved = torch.load(
                Path(args.directory) / "checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )
            if saved.get("format") != "haojie-classic-training-v3":
                raise ValueError("续训只接受 haojie-classic-training-v3 检查点")
            config = Config(**saved["config"])
            del saved
            config.seconds = config.games = config.updates = 0
        else:
            config = Config(workers=min(defaults.workers, cpu_limit()))
        for field in fields(Config):
            value = getattr(args, field.name)
            if value is not None:
                setattr(config, field.name, value)
        Trainer(args.directory, config, args.resume).run()
    elif args.command == "web":
        from .web.server import serve

        serve(args.root, args.port)
    elif args.command == "evaluate-round":
        from .evaluation import evaluate_round

        evaluate_round(args.directory)
    else:
        from .evaluation import evaluate

        evaluate(args)


if __name__ == "__main__":
    main()
