"""Hash-gated access to the exact completed R01:K1 optimizer runtime.

Both the 4,000-step DCLM selection run (job 825162_0) and the 9,150-step
FineWeb-Edu run (array row 835798_2, numeric job 840009) imported the same
fifteen optimizer source files.  Fourteen are reused byte-for-byte from the
content-addressed job-878462 backing archive; the R01-era ``rlb_r05_core`` is
stored in the dedicated namespace package.  The namespace gives every module
an R01-specific import identity, and this module verifies the full closure
before importing any optimizer class.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


DCLM_SOURCE_FREEZE_SHA256 = (
    "0d9263fab3c6e99425445b183f92ec9089537a1ad7b1c6bb73fc50d5bdda159c"
)
FINEWEB_9150_SOURCE_FREEZE_SHA256 = (
    "2c547ce048ff4e29b24d29a727c420800a831aa80cbfb4d7c1a01e8d43bc6c83"
)

_ROOT = Path(__file__).resolve().parent
_DEDICATED = _ROOT / "_archive_r01_9150"
_SHARED = _ROOT / "_archive_r07_frame_878462"
_NAMESPACE_INIT = _DEDICATED / "__init__.py"
_NAMESPACE_INIT_SHA256 = (
    "787bfdc39e7245c25bba537c660f5a8f5739272425061b09b9b2a77f198eb651"
)

# (backing store, bytes, newline count, sha256).  These are exactly the
# optimizer-closure rows common to both completed source-freeze manifests.
_EXPECTED = {
    "rlb_group_muon_core.py": (
        _SHARED, 7953, 196,
        "cec37fee0a4ae52d1e5d296dc0e4cad1db5ade8e5f644cc247df1226f760da17",
    ),
    "rlb_response_capture_core.py": (
        _SHARED, 15394, 357,
        "cc084d17663831f971f419d29b8f199e2c0a96d7fac7b8dc63a3d046aea45968",
    ),
    "rlb_r08_core.py": (
        _SHARED, 15233, 390,
        "89031fc7dbd7056461fbe2b951916e24ed93d3176062c0bf83f270c2f999136d",
    ),
    "rlb_r05_core.py": (
        _DEDICATED, 76542, 1735,
        "849ed8b02332dfaed00059942d12395cbc441943d7e18fd982dd1b10cb09a15b",
    ),
    "rlb_r04_core.py": (
        _SHARED, 18731, 449,
        "461e4dba14d205aaf1dd3b9f961da1c4162a192306e4aad9c3eaf6360d47f884",
    ),
    "rlb_r07_core.py": (
        _SHARED, 7363, 178,
        "0d16158d9e489b88f1d7d3df08f3c57a7197f805c65ac4c542780edb82b15255",
    ),
    "rlb_r04_revision_core.py": (
        _SHARED, 17577, 399,
        "a4bebb96b5d50150ebbcdd5472885f887710cc6dace33bbd401e44953a0cae03",
    ),
    "rlb_r05_revision_core.py": (
        _SHARED, 9634, 209,
        "49ad4f2106c30b7608722003113288a662e3252f58c85ba4ca4dbfc10958eac1",
    ),
    "rlb_r06_revision_core.py": (
        _SHARED, 8241, 195,
        "79dafe29a88d778a880115b50e0eab6471f463e29b190fbc2b4f388725dcbe07",
    ),
    "rlb_r02_core.py": (
        _SHARED, 37119, 879,
        "740cbd164a98b37ef7deeb01f1a49f3e932c8489f3cad66dec3b352b7b99a9c5",
    ),
    "rlb_r02.py": (
        _SHARED, 216, 11,
        "85914e6146c55d36e6be85a3c7e8e5a578aba3cac433531887cf81fb53ec24be",
    ),
    "rlb_r09_core.py": (
        _SHARED, 27258, 657,
        "248507f032ba4f672cf56f1da6487e956b2a2020e06018c34fbbb78783cbd9e9",
    ),
    "rlb_r09.py": (
        _SHARED, 256, 10,
        "439c5a7c01f5d2148f7256a1c8caf598cf505a35a1657785f70c054566fcfc4e",
    ),
    "rlb_r01_core.py": (
        _SHARED, 18331, 427,
        "c49245355e438cc70868973ba38411d4d627fdb812777c68870b6a3b5388df0d",
    ),
    "rlb_r01.py": (
        _SHARED, 182, 7,
        "285c15a7e4e0acf5721f902fb25855e3715f4f9e5b8803412af74247f0c56f62",
    ),
}


def verify_r01_9150_archive() -> dict[str, dict[str, object]]:
    """Fail closed unless every historical runtime source is byte-exact."""
    if not _NAMESPACE_INIT.is_file() or _NAMESPACE_INIT.is_symlink():
        raise RuntimeError("exact R01 archive namespace initializer is absent")
    namespace_data = _NAMESPACE_INIT.read_bytes()
    if hashlib.sha256(namespace_data).hexdigest() != _NAMESPACE_INIT_SHA256:
        raise RuntimeError("exact R01 archive namespace initializer changed")
    certificate: dict[str, dict[str, object]] = {}
    for name, (directory, expected_bytes, expected_lines, expected_sha256) in (
        _EXPECTED.items()
    ):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"exact R01 archive source is absent or indirect: {name}")
        data = path.read_bytes()
        observed = hashlib.sha256(data).hexdigest()
        if len(data) != expected_bytes:
            raise RuntimeError(f"exact R01 archive byte count changed: {name}")
        if data.count(b"\n") != expected_lines:
            raise RuntimeError(f"exact R01 archive line count changed: {name}")
        if observed != expected_sha256:
            raise RuntimeError(f"exact R01 archive digest changed: {name}")
        certificate[name] = {
            "path": str(path),
            "bytes": expected_bytes,
            "lines": expected_lines,
            "sha256": observed,
        }
    return certificate


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()

# Import only after the complete source closure has passed its byte gate.
from ._archive_r01_9150.rlb_r01_core import R01Core  # noqa: E402
from ._archive_r01_9150.rlb_r01 import R01Optimizer  # noqa: E402
from ._archive_r01_9150.rlb_r02 import R02AttentionOptimizer  # noqa: E402


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "DCLM_SOURCE_FREEZE_SHA256",
    "FINEWEB_9150_SOURCE_FREEZE_SHA256",
    "R01Core",
    "R01Optimizer",
    "R02AttentionOptimizer",
    "verify_r01_9150_archive",
)
