"""Hash-gated replay access to exact Method2 from completed job 881693_0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .rlb_r07_frame_878462_replay import verify_r07_frame_878462_archive


HISTORICAL_JOB_ID = "881693_0"
ARCHIVE_MANIFEST_SHA256 = (
    "341e78e1871d256ee5498366d8f32635265aed73861258120043ff78be3b5ea2"
)
HISTORICAL_CORE_SHA256 = (
    "a8f66c2d690a88064f5a7c5c4064dd0363353e4fc759bc8b26905baa3e03db48"
)
HISTORICAL_WRAPPER_SHA256 = (
    "01d70935393fd0b66a653f19072618efa5ac44a911ab651fe2313b17637c7840"
)
HISTORICAL_ENTRYPOINT_SHA256 = (
    "556bd2fa9b906574d61f5dc0f9e650c5b4903fbd569e58ee8da165e55db27b6a"
)

_ARCHIVE = Path(__file__).with_name("_archive_r07_paired_postpolar_881693")
_MAPPING = _ARCHIVE / "ARCHIVE_MAPPING.json"
_PARENT_DIRECT_HASHES = {
    "rlb_r07_frame_core.py": (
        "f3972bc060e26959ae33170992bd2b1fd2b0635a36d64e29e7c47ff7c6d06bef"
    ),
    "rlb_r07.py": (
        "736ccd319392f05eb6881c1aea3ecf22786d1163d828874b5deae0fbc1f83827"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_r07_paired_postpolar_881693_archive() -> dict:
    """Reject a missing, indirect, incomplete, or modified Method2 closure."""
    parent = verify_r07_frame_878462_archive()
    if not _MAPPING.is_file() or _MAPPING.is_symlink():
        raise RuntimeError("Method2 881693 archive mapping is absent or indirect")
    if _sha256(_MAPPING) != ARCHIVE_MANIFEST_SHA256:
        raise RuntimeError("Method2 881693 archive mapping digest changed")
    record = json.loads(_MAPPING.read_text())
    if record.get("schema") != "r07_paired_postpolar_881693_import_isolated_archive_v1":
        raise RuntimeError("Method2 881693 archive schema changed")
    if record.get("historical_job_id") != HISTORICAL_JOB_ID:
        raise RuntimeError("Method2 881693 historical identity changed")
    for name, item in record.get("files", {}).items():
        path = _ARCHIVE / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Method2 archive file is absent or indirect: {name}")
        data = path.read_bytes()
        if len(data) != int(item["bytes"]):
            raise RuntimeError(f"Method2 archive byte count changed: {name}")
        if data.count(b"\n") != int(item["lines"]):
            raise RuntimeError(f"Method2 archive line count changed: {name}")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Method2 archive digest changed: {name}")
    parent_files = parent.get("files", {})
    for name, expected in _PARENT_DIRECT_HASHES.items():
        if parent_files.get(name, {}).get("sha256") != expected:
            raise RuntimeError(f"Method2 direct parent identity changed: {name}")
    if (
        record.get("parent_archive", {}).get("mapping_sha256")
        != "23a941b4136f6930a4a6acd199c2bff6861fde7def475dc9725fe0b744a2c7ed"
    ):
        raise RuntimeError("Method2 parent archive declaration changed")
    return record


ARCHIVE_CERTIFICATE = verify_r07_paired_postpolar_881693_archive()

# Import only after the modified Method2 files and every inherited Method1
# dependency have passed independent byte gates.
from ._archive_r07_paired_postpolar_881693.rlb_r07 import (  # noqa: E402
    R07AttentionOptimizer as R07PairedPostpolar881693AttentionOptimizer,
)
from ._archive_r07_paired_postpolar_881693.rlb_r07 import (  # noqa: E402
    R07Optimizer as R07PairedPostpolar881693Optimizer,
)
from ._archive_r07_paired_postpolar_881693.rlb_r07_frame_core import (  # noqa: E402
    R07FrameCore as R07FrameParent881693Core,
)
from ._archive_r07_paired_postpolar_881693.rlb_r07_frame_core import (  # noqa: E402
    R07PairedAdaptiveFrameCore as R07PairedPostpolar881693Core,
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "ARCHIVE_MANIFEST_SHA256",
    "HISTORICAL_CORE_SHA256",
    "HISTORICAL_ENTRYPOINT_SHA256",
    "HISTORICAL_JOB_ID",
    "HISTORICAL_WRAPPER_SHA256",
    "R07FrameParent881693Core",
    "R07PairedPostpolar881693AttentionOptimizer",
    "R07PairedPostpolar881693Core",
    "R07PairedPostpolar881693Optimizer",
    "verify_r07_paired_postpolar_881693_archive",
)
