"""在独立安装环境中验收交付包，子进程仅使用安装后的包与显式引擎。"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import haojie_training

    source = args.source.resolve(strict=True)
    if source in Path(haojie_training.__file__).resolve().parents:
        raise RuntimeError("验收必须使用独立安装的包，不能从源码或 PYTHONPATH 回退")
    env = {**os.environ, "HAOJIE_NATIVE": str(args.engine.resolve(strict=True))}
    env.pop("PYTHONPATH", None)
    for argv in (
        [source / "scripts/training/native/ci/verify.py", "--engine", args.engine],
        ["-m", "unittest", "discover", "-s", source / "training/tests", "-p", "test_native*.py", "-v"],
        ["-m", "unittest", "discover", "-s", source / "training/tests", "-p", "test_resident.py", "-v"],
        [source / "scripts/training/native/pipeline/accept.py", "--engine", args.engine,
         "--starts", source / "fixtures/late-starts.json", "--output", args.output],
    ):
        subprocess.run([sys.executable, *map(str, argv)], env=env, check=True, timeout=1200)


if __name__ == "__main__":
    main()
