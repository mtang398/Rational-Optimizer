"""Import-isolated namespace for the exact completed R01:K1 runtime.

The completed R01 closure shares fourteen byte-identical modules with the
content-addressed job-878462 archive.  Keeping those immutable files in one
backing store avoids a second, drifting copy.  This package supplies the one
R01-era dependency that differs and resolves it first.  Public consumers must
import through :mod:`optimizer_design.rlb_r01_9150_archive`, whose hash gate
runs before any historical optimizer module is imported.
"""

from pathlib import Path


_SHARED_EXACT_SOURCE = (
    Path(__file__).resolve().parent.parent / "_archive_r07_frame_878462"
)
if not _SHARED_EXACT_SOURCE.is_dir():
    raise RuntimeError("shared exact R01 source backing store is absent")

# Preserve this package directory first so its exact R01-era rlb_r05_core.py
# shadows the later job-878462 implementation with the same equation.
__path__.append(str(_SHARED_EXACT_SOURCE))

