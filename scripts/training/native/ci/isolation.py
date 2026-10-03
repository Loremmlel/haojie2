"""隔离根内核验运行时与实际 execve；绝对路径、失败尝试和包装脚本也不能绕过。"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

FORBIDDEN = {"node", "nodejs", "npm", "npx", "bun", "bunx", "tsx", "ts-node", "deno", "qjs", "d8"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--installation", action="store_true")
    args = parser.parse_args()
    if args.trace:
        calls = re.findall(r'execve\("([^"\n]+)"', args.trace.read_text())
        if not calls:
            raise RuntimeError("没有实际子进程证据")
        allowed = {"python", "python3", "python3.12", "haojie-engine"}
        if args.installation:
            # pip 查询发行版/平台信息；这些系统查询不允许 JS 或回退脚本执行。
            allowed.update({"uname", "lsb_release"})
        unexpected = [p for p in calls if Path(p).name not in allowed]
        if unexpected:
            raise RuntimeError(f"非 Python/Rust 执行或回退尝试: {unexpected}")
        print(json.dumps({"execve": calls, "installation": args.installation, "allowed_executables": sorted(allowed)}, indent=2))
        return
    if Path("/init").exists():
        raise RuntimeError("隔离根不应可见宿主互操作入口")
    found = [name for name in FORBIDDEN if shutil.which(name)]
    for root, dirs, files in os.walk("/", followlinks=False):
        dirs[:] = [d for d in dirs if str(Path(root) / d) not in {"/proc", "/sys", "/dev"}]
        found.extend(str(Path(root) / f) for f in files if f.lower() in FORBIDDEN and os.access(Path(root) / f, os.X_OK))
    if found:
        raise RuntimeError(f"隔离根存在 JS 执行入口: {found}")
    print(json.dumps({"filesystem_scan": "no JavaScript runtime", "root": "/"}))


if __name__ == "__main__":
    main()
