"""Import overlay for Method2 with the metric-2 R01 ancestor."""

from pathlib import Path


_HERE = Path(__file__).resolve().parent
_METHOD2 = _HERE.parent / "_archive_r07_paired_postpolar_881693"
_METHOD1 = _HERE.parent / "_archive_r07_frame_878462"
if not _METHOD2.is_dir() or not _METHOD1.is_dir():
    raise RuntimeError("Method2 or Method1 source archive is absent")
__path__ = [str(_HERE), str(_METHOD2), str(_METHOD1)]

