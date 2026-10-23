"""在独立安装环境中验收交付包，子进程仅使用安装后的包与显式引擎。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import haojie_training

    source = args.source.resolve(strict=True)
    engine = args.engine.resolve(strict=True)
    if source in Path(haojie_training.__file__).resolve().parents:
        raise RuntimeError("验收必须使用独立安装的包，不能从源码或 PYTHONPATH 回退")
    env = {**os.environ, "HAOJIE_NATIVE_LIB": str(args.library.resolve(strict=True))}
    env.pop("PYTHONPATH", None)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, argv in (
        (
            "identity",
            [source / "scripts/training/native/ci/verify.py", "--engine", engine],
        ),
        (
            "maintained-tests",
            [
                "-m",
                "unittest",
                "discover",
                "-s",
                source / "training/tests/online",
                "-p",
                "test_*.py",
                "-v",
            ],
        ),
    ):
        # 源码根只提供随包测试与验收脚本；训练包仍从独立环境的 site-packages 导入。
        with (args.output / f"{name}.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, *map(str, argv)],
                cwd=source,
                env=env,
                check=True,
                timeout=1200,
                stdout=log,
                stderr=subprocess.STDOUT,
            )


if __name__ == "__main__":
    main()
