from pathlib import Path
import hashlib
p = Path('/tmp/feedback.patch.gz.b64')
s = p.read_text()
assert len(s) == 12369 and s[2146] == 'T'
s = s[:2146] + s[2147:]
assert hashlib.sha256(s.encode()).hexdigest() == '54bdf3b621299d9bcebc4cb4f440add0ef3f401b885a8e01967a6090a21b6d41'
p.write_text(s)
