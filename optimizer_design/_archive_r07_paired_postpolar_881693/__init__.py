"""Import-isolated recovery of Method2 from completed job 881693_0."""

from pathlib import Path


_SHARED_EXACT_SOURCE = (
    Path(__file__).resolve().parent.parent / "_archive_r07_frame_878462"
)
if not _SHARED_EXACT_SOURCE.is_dir():
    raise RuntimeError("shared exact Method1 source backing store is absent")

# The two local historical files shadow their Method1 versions. Every
# unchanged relative import resolves from the independently hash-gated shared
# archive without duplicating or modifying it.
__path__.append(str(_SHARED_EXACT_SOURCE))
