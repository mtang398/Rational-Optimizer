"""Import overlay for the job-878462 Method1 runtime approximation.

Only ``rlb_r01_core`` is replaced locally.  Every other module is loaded
from the hash-gated immutable job-878462 archive under this package identity,
so the complete R03 and cross-role-frame equations remain the historical
ones while their R01 ancestor uses the separately quality-gated execution
approximation.
"""

from pathlib import Path


_HERE = Path(__file__).resolve().parent
_ARCHIVE = _HERE.parent / "_archive_r07_frame_878462"
if not _ARCHIVE.is_dir():
    raise RuntimeError("job-878462 Method1 archive is absent")

# Python searches the overlay first and the immutable archive second.
__path__ = [str(_HERE), str(_ARCHIVE)]

