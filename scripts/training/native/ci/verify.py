"""交付程序的发布入口检查；显式程序、身份及默认无探针都是硬要求。"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    args = parser.parse_args()
    engine = args.engine.resolve(strict=True)
    identity = json.loads(subprocess.check_output([engine, "--version"], timeout=30))
    assert identity["name"] == "haojie-engine"
    assert identity["build"]["profile"] == "release"
    assert identity["build"]["kernelProfile"] is False
    assert len(identity["build"]["sourceSha256"]) == 64
    assert identity["record"] == "haojie-native-record-v1"
    assert identity["encoding"] == "haojie-entities-factorized-v1"
    assert "record-audit" in identity["capabilities"]
    response = subprocess.run(
        [engine, "--training"], input=b'{"op":"close"}\n',
        capture_output=True, check=True, timeout=30,
    )
    ready = json.loads(response.stdout)
    assert ready["engine"] == identity
    assert ready["protocol"] == identity["protocols"]["training"]
    assert ready["rulesHash"] == identity["rulesHash"]
    print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
