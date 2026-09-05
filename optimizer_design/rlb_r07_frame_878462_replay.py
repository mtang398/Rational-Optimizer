"""Hash-gated access to the exact optimizer runtime from job 878462_0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARCHIVE_MANIFEST_SHA256 = "23a941b4136f6930a4a6acd199c2bff6861fde7def475dc9725fe0b744a2c7ed"
_ARCHIVE = Path(__file__).with_name("_archive_r07_frame_878462")
_MAPPING = _ARCHIVE / "ARCHIVE_MAPPING.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_r07_frame_878462_archive() -> dict:
    if not _MAPPING.is_file() or _MAPPING.is_symlink():
        raise RuntimeError("R07 frame 878462 archive mapping is absent or indirect")
    if _sha256(_MAPPING) != ARCHIVE_MANIFEST_SHA256:
        raise RuntimeError("R07 frame 878462 archive mapping digest changed")
    record = json.loads(_MAPPING.read_text())
    if record.get("schema") != "r07_frame_878462_import_isolated_archive_v1":
        raise RuntimeError("R07 frame 878462 archive schema changed")
    if record.get("historical_job_id") != "878462_0":
        raise RuntimeError("R07 frame 878462 historical identity changed")
    for name, item in record.get("files", {}).items():
        path = _ARCHIVE / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"R07 frame archive file is absent or indirect: {name}")
        data = path.read_bytes()
        if len(data) != int(item["bytes"]):
            raise RuntimeError(f"R07 frame archive byte count changed: {name}")
        if data.count(b"\n") != int(item["lines"]):
            raise RuntimeError(f"R07 frame archive line count changed: {name}")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError(f"R07 frame archive digest changed: {name}")
    return record


ARCHIVE_CERTIFICATE = verify_r07_frame_878462_archive()

from ._archive_r07_frame_878462.rlb_r01_core import R01Core  # noqa: E402
from ._archive_r07_frame_878462.rlb_r01 import (  # noqa: E402
    R01Optimizer as R01Frame878462Optimizer,
)
from ._archive_r07_frame_878462.rlb_r02 import (  # noqa: E402
    R02AttentionOptimizer as R02Frame878462AttentionOptimizer,
)
from ._archive_r07_frame_878462.rlb_r07 import (  # noqa: E402
    R07AttentionOptimizer as R07Frame878462AttentionOptimizer,
)
from ._archive_r07_frame_878462.rlb_r07 import (  # noqa: E402
    R07Optimizer as R07Frame878462Optimizer,
)
from ._archive_r07_frame_878462.rlb_r07_frame_core import (  # noqa: E402
    R07FrameCore as R07Frame878462Core,
)

__all__ = (
    "ARCHIVE_CERTIFICATE",
    "ARCHIVE_MANIFEST_SHA256",
    "R01Core",
    "R01Frame878462Optimizer",
    "R02Frame878462AttentionOptimizer",
    "R07Frame878462Core",
    "R07Frame878462Optimizer",
    "R07Frame878462AttentionOptimizer",
    "verify_r07_frame_878462_archive",
)
