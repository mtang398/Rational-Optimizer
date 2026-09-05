"""Import overlay for Method3 with the metric-2 R01 ancestor."""

from pathlib import Path


_HERE = Path(__file__).resolve().parent
_METHOD3 = _HERE.parent / "_archive_r10_row_product_881377"
_PARENT = _HERE.parent / "_archive_r07_frame_878462"
if not _METHOD3.is_dir() or not _PARENT.is_dir():
    raise RuntimeError("Method3 or parent source archive is absent")
__path__ = [str(_HERE), str(_METHOD3), str(_PARENT)]

