"""One-use recovery of the interrupted 3.0 upload. Every source delta is hash-checked."""
from pathlib import Path
import base64
import hashlib
import json
import lzma

root = Path.cwd()

def digest(data):
    return hashlib.sha256(data).hexdigest()

def apply(entries):
    for name, (before, after, edits) in entries.items():
        path = root / name
        assert path.resolve().is_relative_to(root), name
        assert not name.startswith('.git/'), name
        original = path.read_bytes() if path.exists() else b''
        if digest(original) == after:
            continue
        assert (before is None and not path.exists()) or digest(original) == before, ('base changed', name)
        lines = original.decode('utf-8').splitlines(keepends=True)
        for start, end, replacement in reversed(edits):
            assert 0 <= start <= end <= len(lines), name
            lines[start:end] = [replacement]
        final = ''.join(lines).encode('utf-8')
        assert digest(final) == after, ('delta checksum', name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(final)

parts = [(root / f'.github/shrine-delta.part{i:02d}').read_text().strip() for i in range(1, 4)]
assert list(map(len, parts)) == [18000, 17999, 18000]
prefix = (parts[0] + parts[1] + '0' + parts[2]).encode()
assert digest(prefix) == 'ebc482cb556d3ad3ea1dda17ad03ac10083603c5f773798d9bc8c9d53ff79250'
partial = lzma.LZMADecompressor().decompress(base64.b64decode(prefix)).decode('utf-8', errors='replace')
decoder = json.JSONDecoder()
entries = {}
pos = 1
while pos < len(partial):
    try:
        key, end = decoder.raw_decode(partial, pos)
        value, end = decoder.raw_decode(partial, end + 1)
    except ValueError:
        break
    entries[key] = value
    pos = end + 1
assert len(entries) == 36, len(entries)
apply(entries)

encoded = ''.join((root / f'.github/shrine-finish.part{i:02d}').read_text().strip() for i in range(1, 5)).encode()
assert digest(encoded) == 'b8820eb7d22d30ba9097160304904520f680b73b3071a3af4579874e86e507fe', 'finishing upload checksum'
data = lzma.decompress(base64.b64decode(encoded))
assert digest(data) == '3084933ba3285a584cdb2533b08b445f977541a9a62d0beec1b0a9c0f49d5823'
finished = json.loads(data)
apply(finished['delta'])
for name, expected in finished['manifest'].items():
    assert digest((root / name).read_bytes()) == expected, ('final source checksum', name)
for pattern in ('shrine-delta.part*', 'shrine-finish.part*', 'apply-shrine.py'):
    for path in (root / '.github').glob(pattern):
        path.unlink()
print(f"Recovered and verified {len(finished['manifest'])} final source files.")
