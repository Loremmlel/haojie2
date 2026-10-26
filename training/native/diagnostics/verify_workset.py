"""逐字节核对固定真实工作集，并保存可重查的行为和正式状态摘要。"""

import argparse
import hashlib
import json
from pathlib import Path


def hashes(path):
    return {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.iterdir())
        if file.is_file()
    }


def verify(reference, sessions):
    expected = hashes(reference)
    cases = []
    for session in sessions:
        actual = hashes(session / "workset")
        different = sorted(
            name
            for name in expected.keys() | actual.keys()
            if expected.get(name) != actual.get(name)
        )
        window = json.loads((session / "window.json").read_text(encoding="utf-8"))
        cases.append(
            {
                "session": str(session),
                "identical": not different,
                "different_files": different,
                "work": {
                    key: window[key]
                    for key in (
                        "consumed",
                        "produced",
                        "commands",
                        "microsteps",
                        "inference_nodes",
                        "backtracks",
                    )
                },
            }
        )
    return {"reference": str(reference), "sha256": expected, "cases": cases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.reference, args.sessions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    if not all(case["identical"] for case in result["cases"]):
        raise RuntimeError("工作集行为或正式状态不同，见输出中的 different_files")
    print(f"{len(result['cases'])} 个工作集与参考的所有文件完全相同。")
