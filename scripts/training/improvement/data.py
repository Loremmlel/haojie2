"""研究检查点与训练协议共用内容指纹。"""

import hashlib
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
