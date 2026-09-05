"""Load the hash-gated execution-optimized frame source in this overlay."""

import hashlib
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[1] / "rlb_r07_frame_core.py"
_EXPECTED = "63795eeba58edc81a4e83ad7e75d5ec42e459adaba650cb9280f2f7614ff26ef"
_BYTES = _SOURCE.read_bytes()
if hashlib.sha256(_BYTES).hexdigest() != _EXPECTED:
    raise RuntimeError("execution-optimized R07-frame source hash changed")
exec(compile(_BYTES, str(_SOURCE), "exec"), globals(), globals())
