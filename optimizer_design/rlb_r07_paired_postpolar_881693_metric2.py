"""Exact Method2 outer equation with the metric-2 R01 approximation."""

from __future__ import annotations

from .rlb_r07_paired_postpolar_881693_replay import (
    verify_r07_paired_postpolar_881693_archive,
)
from ._periodic_outer_metric2 import PeriodicOuterMetric2Mixin


ARCHIVE_CERTIFICATE = verify_r07_paired_postpolar_881693_archive()

from ._method2_metric2_approx.rlb_r07 import (  # noqa: E402
    R07AttentionOptimizer as _AttentionOptimizer,
)
from ._method2_metric2_approx.rlb_r07_frame_core import (  # noqa: E402
    R07PairedAdaptiveFrameCore as _Method2Core,
)


FAMILY_ID = "method2_881693_with_metric2_r01_approximation_v1"


class Method2Metric2Core(_Method2Core):
    checkpoint_schema = FAMILY_ID


class Method2Metric2Optimizer(Method2Metric2Core):
    pass


class Method2Metric2Outer4Optimizer(
    PeriodicOuterMetric2Mixin, Method2Metric2Core
):
    outer_refresh_interval = 4
    periodic_outer_label = "method2"
    periodic_outer_metadata_names = (
        "_r05_next_metadata",
        "_r08_inverse_sqrt",
        "_r08_role_direction",
        "_r08_response_metadata",
        "_r03_persistent_metadata",
        "_r07_frame_metadata",
        "_r07_pair_adaptive_metadata",
    )
    checkpoint_schema = FAMILY_ID + "_outer4"


Method2Metric2AttentionOptimizer = _AttentionOptimizer


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "FAMILY_ID",
    "Method2Metric2AttentionOptimizer",
    "Method2Metric2Core",
    "Method2Metric2Optimizer",
    "Method2Metric2Outer4Optimizer",
)
