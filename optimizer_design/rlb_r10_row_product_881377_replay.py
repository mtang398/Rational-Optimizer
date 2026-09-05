"""Hash-gated replay access to exact Method3 from completed job 881377_0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .rlb_r07_frame_878462_replay import (
    verify_r07_frame_878462_archive,
)


HISTORICAL_JOB_ID = "881377_0"
ARCHIVE_MANIFEST_SHA256 = (
    "3fdca8a43318824d650df6dbd3cb96df10ddbd262aca22904a2ddcf1a9cbef17"
)
HISTORICAL_CORE_SHA256 = (
    "fa1bee6c4d5fa0a879d2542a7f135520bde6877e5a85dc34e2bb77aa715585fd"
)
HISTORICAL_WRAPPER_SHA256 = (
    "db3edafa0ccd9b8e6824f739cd68829dec13db5e4b8689bd622af1d4a128f076"
)
HISTORICAL_ENTRYPOINT_SHA256 = (
    "f298767aa13fe3c1b33459fbc323594cb69dd537ab6241b455e58d035fa3f754"
)

_ARCHIVE = Path(__file__).with_name("_archive_r10_row_product_881377")
_MAPPING = _ARCHIVE / "ARCHIVE_MAPPING.json"
_PARENT_DIRECT_HASHES = {
    "rlb_r03.py": (
        "bf8111944954bc8d80a4e0befcf94c364bbe9544badb4d4d0514b55cc2b08566"
    ),
    "rlb_r03_core.py": (
        "31fda9c367806b27710858510c98138e05f1680aa7fea3a9712c3259355b1a35"
    ),
    "rlb_r02.py": (
        "85914e6146c55d36e6be85a3c7e8e5a578aba3cac433531887cf81fb53ec24be"
    ),
    "rlb_r02_core.py": (
        "740cbd164a98b37ef7deeb01f1a49f3e932c8489f3cad66dec3b352b7b99a9c5"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_r10_row_product_881377_archive() -> dict:
    """Reject a missing, indirect, incomplete, or modified Method3 closure."""
    parent = verify_r07_frame_878462_archive()
    if not _MAPPING.is_file() or _MAPPING.is_symlink():
        raise RuntimeError("Method3 881377 archive mapping is absent or indirect")
    if _sha256(_MAPPING) != ARCHIVE_MANIFEST_SHA256:
        raise RuntimeError("Method3 881377 archive mapping digest changed")
    record = json.loads(_MAPPING.read_text())
    if record.get("schema") != "r10_row_product_881377_import_isolated_archive_v1":
        raise RuntimeError("Method3 881377 archive schema changed")
    if record.get("historical_job_id") != HISTORICAL_JOB_ID:
        raise RuntimeError("Method3 881377 historical identity changed")
    for name, item in record.get("files", {}).items():
        path = _ARCHIVE / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Method3 archive file is absent or indirect: {name}")
        data = path.read_bytes()
        if len(data) != int(item["bytes"]):
            raise RuntimeError(f"Method3 archive byte count changed: {name}")
        if data.count(b"\n") != int(item["lines"]):
            raise RuntimeError(f"Method3 archive line count changed: {name}")
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Method3 archive digest changed: {name}")
    parent_files = parent.get("files", {})
    for name, expected in _PARENT_DIRECT_HASHES.items():
        if parent_files.get(name, {}).get("sha256") != expected:
            raise RuntimeError(f"Method3 direct parent identity changed: {name}")
    if (
        record.get("parent_archive", {}).get("mapping_sha256")
        != "23a941b4136f6930a4a6acd199c2bff6861fde7def475dc9725fe0b744a2c7ed"
    ):
        raise RuntimeError("Method3 parent archive declaration changed")
    return record


ARCHIVE_CERTIFICATE = verify_r10_row_product_881377_archive()

# Import only after both the local Method3 sources and the complete inherited
# optimizer closure have passed independent byte gates.
from ._archive_r07_frame_878462.rlb_r03 import (  # noqa: E402
    R03Optimizer as R03Method3Router881377,
)
from ._archive_r10_row_product_881377.rlb_r10 import (  # noqa: E402
    R10AttentionOptimizer as R10RowProduct881377AttentionOptimizer,
)
from ._archive_r10_row_product_881377.rlb_r10_core import (  # noqa: E402
    R10AttentionCore as R10RowProduct881377Core,
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "ARCHIVE_MANIFEST_SHA256",
    "HISTORICAL_CORE_SHA256",
    "HISTORICAL_ENTRYPOINT_SHA256",
    "HISTORICAL_JOB_ID",
    "HISTORICAL_WRAPPER_SHA256",
    "R03Method3Router881377",
    "R10RowProduct881377AttentionOptimizer",
    "R10RowProduct881377Core",
    "verify_r10_row_product_881377_archive",
)
