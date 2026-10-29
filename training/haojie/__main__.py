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
        run.add_argument("--" + field.name.replace("_", "-"), type=type(getattr(defaults, field.name)))
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--opponent", default="easy")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--games", type=int, default=20)
    evaluate.add_argument("--simulations", type=int, default=64)
    evaluate.add_argument("--candidates", type=int, default=32)
    evaluate.add_argument("--max-commands", type=int, default=20000)
    evaluate.add_argument("--seed", type=int, default=709901)
    evaluate.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "run":
        from .runtime.trainer import Trainer
        if args.resume:
            import torch
            saved = torch.load(Path(args.directory) / "checkpoint.pt", map_location="cpu", weights_only=False)
            config = Config(**saved["config"])
            config.seconds = config.games = config.updates = 0
        else:
            config = Config(workers=min(4, cpu_limit()))
        for field in fields(Config):
            value = getattr(args, field.name)
            if value is not None:
                setattr(config, field.name, value)
        Trainer(args.directory, config, args.resume).run()
    else:
        from .evaluation import evaluate
        evaluate(args)


if __name__ == "__main__":
    main()
