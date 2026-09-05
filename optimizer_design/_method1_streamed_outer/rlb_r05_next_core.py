"""Load the hash-gated execution-optimized R05-next source in this overlay."""

import hashlib
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "rlb_r05_next_core.py"
_EXPECTED = "9ef1f7430a954ff3a4c90079951ccdc6a10335d4c4d85792a899e154fc6223a4"
_BYTES = _SOURCE.read_bytes()
if hashlib.sha256(_BYTES).hexdigest() != _EXPECTED:
    raise RuntimeError("execution-optimized R05-next source hash changed")
exec(compile(_BYTES, str(_SOURCE), "exec"), globals(), globals())
