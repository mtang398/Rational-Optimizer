"""Hash-gated execution overlay for streamed qualified Method1 outer math."""

from pathlib import Path


_HERE = Path(__file__).resolve().parent
_ARCHIVE = _HERE.parent / "_archive_r07_frame_878462"
if not _ARCHIVE.is_dir():
    raise RuntimeError("job-878462 Method1 archive is absent")

# Local files replace only the qualified R01 ancestor and the two execution-
# optimized outer modules. Every other scientific module resolves to the
# immutable job-878462 archive under this package identity.
__path__ = [str(_HERE), str(_ARCHIVE)]
