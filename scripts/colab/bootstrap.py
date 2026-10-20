"""Colab / Linux 安装：独立 Python 3.12、仓库固定依赖和锁定的 Rust release 引擎。"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path("/content/haojie-venv"))
    parser.add_argument("--target", type=Path, default=Path("/content/haojie-native"))
    parser.add_argument(
        "--backend", default="auto", choices=("auto", "cpu", "cu126", "cu128", "cu130")
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if sys.platform != "linux":
        parser.error("该安装入口面向 Colab/Linux；其他平台参见 training/README.md")
    for path in (repo, args.venv, args.target):
        if Path("/content/drive") in path.resolve().parents:
            parser.error("源码、解释器和构建输出应位于本地磁盘，而不是 Drive 挂载目录")
    python = args.venv.resolve() / "bin/python"
    if args.venv.exists() and any(args.venv.iterdir()) and not python.is_file():
        parser.error("虚拟环境目标是非空用户目录，请指定新的目录")
    # uv 只装入临时引导目录，不更换 Notebook 内核的 Torch/Python。
    with tempfile.TemporaryDirectory(prefix="haojie-bootstrap-") as temporary:
        run([sys.executable, "-m", "pip", "install", "--quiet", "--target", temporary, "uv"])
        uv = [sys.executable, "-m", "uv"]
        env = {**os.environ, "PYTHONPATH": temporary}
        if not python.exists():
            run([*uv, "venv", "--python", "3.12", "--seed", args.venv], env=env)
        run([python, "-c", "import sys; assert sys.version_info[:2] == (3, 12)"])
        backend = "cpu" if args.device == "cpu" and args.backend == "auto" else args.backend
        # uv 官方的自动后端选择依据驱动；版本仍受仓库精确 pin 约束。
        run(
            [
                *uv,
                "pip",
                "install",
                "--python",
                python,
                "--torch-backend",
                backend,
                "--reinstall-package",
                "torch",
                "-e",
                repo / "training",
            ],
            env=env,
        )
        run([*uv, "pip", "check", "--python", python], env=env)
    probe = """import json, torch
from haojie_training.console.config.resources import hardware, server_defaults
print(json.dumps({"torch": torch.__version__, "hardware": hardware(),
                  "initial_settings": server_defaults(DEVICE)}, indent=2, ensure_ascii=False))
if DEVICE == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 未就绪：请选择 GPU 运行时和兼容后端，不会退回 CPU 继续训练")
    x = torch.ones((64, 64), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    assert y[0, 0].item() == 64
"""
    run([python, "-c", probe.replace("DEVICE", repr(args.device))])
    cargo_bin = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")) / "bin"
    if not (cargo_bin / "rustup").exists():
        with tempfile.TemporaryDirectory(prefix="haojie-rustup-") as temporary:
            installer = Path(temporary) / "rustup.sh"
            urllib.request.urlretrieve("https://sh.rustup.rs", installer)
            run(
                [
                    "sh",
                    installer,
                    "-y",
                    "--no-modify-path",
                    "--profile",
                    "minimal",
                    "--default-toolchain",
                    "none",
                ]
            )
    env = {**os.environ, "PATH": f"{cargo_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
    run([cargo_bin / "rustup", "toolchain", "install", "1.98.1", "--profile", "minimal"], env=env)
    run(
        [
            cargo_bin / "cargo",
            "+1.98.1",
            "build",
            "--locked",
            "--release",
            "--manifest-path",
            repo / "native/engine/Cargo.toml",
            "--target-dir",
            args.target.resolve(),
        ],
        env=env,
    )
    print(
        json.dumps(
            {"python": str(python), "engine": str(args.target.resolve() / "release/haojie-engine")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
