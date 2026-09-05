"""Method3 row-product attention with a metric-2 approximate R03 router."""

from __future__ import annotations

from .rlb_r10_row_product_881377_replay import (
    verify_r10_row_product_881377_archive,
)
from ._periodic_outer_metric2 import PeriodicOuterMetric2Mixin


ARCHIVE_CERTIFICATE = verify_r10_row_product_881377_archive()

from ._method3_metric2_approx.rlb_r03_core import R03Core as _R03Core  # noqa: E402
from ._method3_metric2_approx.rlb_r10 import (  # noqa: E402
    R10AttentionOptimizer as _RowProductAttentionOptimizer,
)


FAMILY_ID = "method3_881377_with_metric2_r01_approximation_v1"


class Method3Metric2Router(_R03Core):
    checkpoint_schema = FAMILY_ID


class Method3Metric2Outer4Router(
    PeriodicOuterMetric2Mixin, Method3Metric2Router
):
    outer_refresh_interval = 4
    periodic_outer_label = "method3"
    periodic_outer_metadata_names = (
        "_r05_next_metadata",
        "_r08_inverse_sqrt",
        "_r08_role_direction",
        "_r08_response_metadata",
        "_r03_persistent_metadata",
    )
    checkpoint_schema = FAMILY_ID + "_outer4"


Method3Metric2AttentionOptimizer = _RowProductAttentionOptimizer


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "FAMILY_ID",
    "Method3Metric2AttentionOptimizer",
    "Method3Metric2Outer4Router",
    "Method3Metric2Router",
)
