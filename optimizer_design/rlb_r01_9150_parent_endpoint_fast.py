"""Execution-only fast path for the U6-parent R01 quality candidate.

The U6-parent candidate never consumes R02's U5 response-congruence branch.
The archived R02 implementation nevertheless constructs that second blend and
stores it for the subsequently deleted geodesic endpoint.  This mixin omits
only that dead result.  The selected U6 direction, all optimizer state, and
the installed parameter update are unchanged.
"""

from __future__ import annotations

from ._archive_r01_9150.rlb_r05_core import R05Core
from .rlb_r01_9150_archive import verify_r01_9150_archive
from .rlb_r01_9150_parent_endpoint import (
    R01StaleMetricAllocation8BF16InverseParentEndpointRowOptimizer,
    R02StaleMetricAllocation8BF16InverseParentEndpointRowAttentionOptimizer,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
PARENT_ENDPOINT_FAST_FAMILY_ID = (
    "r01_stale8_bf16_inverse_parent_endpoint_dead_u5_elided_v1"
)


class _DeadU5BlendElisionMixin:
    """Return the historical U6 blend without constructing unused U5 data."""

    def _blend_equalized(self, ordinary, adaptive_equal, alignment):
        role = len(self._r02_blend_records)
        if role not in (0, 1) or self._r02_congruences is None:
            raise RuntimeError("R02 shared branch construction order changed")
        # This is byte-for-byte the call that produces archived R02's returned
        # U6 branch.  The omitted second call only formed ``u5_parent``, which
        # the U6-parent endpoint override deliberately never reads.
        u6_result = R05Core._blend_equalized(
            ordinary, adaptive_equal, alignment
        )
        self._r02_blend_records.append((ordinary, None))
        return u6_result

    def parent_endpoint_fast_runtime_report(self):
        return {
            "family_id": PARENT_ENDPOINT_FAST_FAMILY_ID,
            "trajectory_relative_to_u6_parent": "intended_bitwise",
            "dead_u5_response_congruence_blend_elided": True,
            "selected_u6_blend_unchanged": True,
            "lr_or_wd_changed": False,
        }


class R01StaleMetricAllocation8BF16InverseParentEndpointFastRowOptimizer(
    _DeadU5BlendElisionMixin,
    R01StaleMetricAllocation8BF16InverseParentEndpointRowOptimizer,
):
    pass


R02StaleMetricAllocation8BF16InverseParentEndpointFastRowAttentionOptimizer = (
    R02StaleMetricAllocation8BF16InverseParentEndpointRowAttentionOptimizer
)


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "PARENT_ENDPOINT_FAST_FAMILY_ID",
    "R01StaleMetricAllocation8BF16InverseParentEndpointFastRowOptimizer",
    "R02StaleMetricAllocation8BF16InverseParentEndpointFastRowAttentionOptimizer",
)
