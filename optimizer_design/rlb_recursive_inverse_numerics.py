"""Recursive tensor-core inverse coordinates for qualified RLB methods.

The qualified metric-2 execution forms a complete Cholesky factor ``L`` and
uses the full map ``L^{-1}``; it does not use a diagonal or block-diagonal
metric approximation.  The historical implementation forms that inverse by
one large triangular solve.  This numerical branch instead applies the exact
block identity

    [[A, 0], [C, D]]^{-1}
      = [[A^{-1}, 0], [-D^{-1} C A^{-1}, D^{-1}]]

recursively.  Only the small diagonal leaves use triangular solves.  The
large off-diagonal work is expressed as batched matrix multiplication, which
can use the GPU tensor cores.  Coordinate and adjoint actions then use the
same complete inverse with BF16 GEMMs on every non-telemetry transition.

The represented Cholesky coordinate, metric construction and refresh cadence,
all NS5 maps, optimizer state recurrences, LR, and WD are unchanged.  GEMM
rounding differs from the literal solve, so this remains a numerical execution
branch and requires a fresh complete 4,000-step quality trajectory if faster.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .rlb_method1_grouped_collectives import Method1GroupedCollectiveOptimizer
from .rlb_other_grouped_collectives import Method2GroupedQualifiedRouter
from .rlb_r01_9150_inverse_coordinate import _BF16InverseCoordinateMixin


FAMILY_ID = "rlb_recursive_block_inverse64_full_coordinate_v1"
LEAF_WIDTH = 64


@contextmanager
def _tf32_enabled():
    previous = bool(torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _recursive_lower_inverse(
    lower: torch.Tensor, leaf_width: int = LEAF_WIDTH
) -> torch.Tensor:
    """Invert a batch of lower-triangular matrices by recursive blocks."""

    if lower.ndim < 2 or lower.shape[-1] != lower.shape[-2]:
        raise RuntimeError("recursive inverse requires square matrix batches")
    dimension = int(lower.shape[-1])
    leaf = int(leaf_width)
    if leaf < 1:
        raise RuntimeError("recursive inverse leaf must be positive")
    if dimension <= leaf:
        identity = torch.eye(
            dimension, device=lower.device, dtype=lower.dtype
        ).expand_as(lower)
        return torch.linalg.solve_triangular(lower, identity, upper=False)
    if dimension % leaf != 0:
        raise RuntimeError("recursive inverse leaf does not divide the matrix")
    if dimension % 2:
        raise RuntimeError("recursive inverse requires even internal blocks")

    split = dimension // 2
    leading = lower[..., :split, :split]
    trailing = lower[..., split:, split:]
    coupling = lower[..., split:, :split]

    # Put both diagonal children in one larger batch so every recursion level
    # launches one leaf solve rather than a Python tree of small kernels.
    children = torch.stack((leading, trailing), dim=-3)
    child_shape = children.shape
    flat_children = children.reshape(-1, split, split)
    flat_inverses = _recursive_lower_inverse(flat_children, leaf)
    inverses = flat_inverses.view(child_shape)
    leading_inverse = inverses[..., 0, :, :]
    trailing_inverse = inverses[..., 1, :, :]

    with _tf32_enabled():
        off_diagonal = -torch.matmul(
            torch.matmul(trailing_inverse, coupling), leading_inverse
        )
    zeros = torch.zeros_like(leading_inverse)
    top = torch.cat((leading_inverse, zeros), dim=-1)
    bottom = torch.cat((off_diagonal, trailing_inverse), dim=-1)
    return torch.cat((top, bottom), dim=-2)


class _RecursiveAllStepInverseMixin:
    """Form recursive full inverses and use them for every ordinary action."""

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        group = self.param_groups[0]
        group["rlb_recursive_inverse_family_id"] = FAMILY_ID
        group["rlb_recursive_inverse_leaf_width"] = LEAF_WIDTH
        group["rlb_all_step_full_inverse_coordinate"] = True

    def _unit_volume_cholesky(self, metric, *, capture_spectrum=False):
        call = self._r02_metric_factor_call

        # Bypass only the historical large inverse-to-identity solve.  The
        # periodic metric owner below it still constructs/caches the literal
        # Cholesky factor and volume on the exact original cadence.
        result = super(
            _BF16InverseCoordinateMixin, self
        )._unit_volume_cholesky(metric, capture_spectrum=capture_spectrum)
        if (
            call in (1, 2)
            and bool(self._capture_full_metric_this_step)
            and not capture_spectrum
        ):
            lower = result[0]
            inverse = _recursive_lower_inverse(lower, LEAF_WIDTH)
            torch._assert_async(torch.isfinite(inverse).all())
            self._cached_bf16_metric_inverses[int(call)] = (
                lower,
                inverse.to(dtype=torch.bfloat16),
            )
        return result

    def _literal_left_coordinate(self, lower, volume, value):
        return super(
            _BF16InverseCoordinateMixin, self
        )._left_coordinate(lower, volume, value)

    def _literal_left_adjoint(self, lower, volume, value):
        return super(
            _BF16InverseCoordinateMixin, self
        )._left_adjoint(lower, volume, value)

    def _literal_right_coordinate(self, lower, volume, value):
        return super(
            _BF16InverseCoordinateMixin, self
        )._right_coordinate(lower, volume, value)

    def _literal_right_adjoint(self, lower, volume, value):
        return super(
            _BF16InverseCoordinateMixin, self
        )._right_adjoint(lower, volume, value)

    def _left_coordinate(self, lower, volume, value):
        if bool(self._capture_telemetry_next_step):
            return self._literal_left_coordinate(lower, volume, value)
        if lower is self._r02_identity_lower:
            return value
        result = self._bf16_matmul(self._inverse_for(lower), value)
        return result * volume[..., None, None]

    def _left_adjoint(self, lower, volume, value):
        if bool(self._capture_telemetry_next_step):
            return self._literal_left_adjoint(lower, volume, value)
        if lower is self._r02_identity_lower:
            return value
        inverse_transpose = self._inverse_for(lower).transpose(-2, -1)
        result = self._bf16_matmul(inverse_transpose, value)
        return result * volume[..., None, None]

    def _right_coordinate(self, lower, volume, value):
        if bool(self._capture_telemetry_next_step):
            return self._literal_right_coordinate(lower, volume, value)
        inverse_transpose = self._inverse_for(lower).transpose(-2, -1)
        result = self._bf16_matmul(value, inverse_transpose)
        return result * volume[..., None, None]

    def _right_adjoint(self, lower, volume, value):
        if bool(self._capture_telemetry_next_step):
            return self._literal_right_adjoint(lower, volume, value)
        result = self._bf16_matmul(value, self._inverse_for(lower))
        return result * volume[..., None, None]

    def recursive_inverse_runtime_report(self):
        return {
            "family_id": FAMILY_ID,
            "leaf_width": LEAF_WIDTH,
            "large_inverse_formation": "recursive_block_gemm",
            "leaf_inverse_formation": "fp32_triangular_solve",
            "coordinate_application": "complete_bf16_inverse_gemm",
            "cross_channel_metric_coupling_preserved": True,
            "metric_or_refresh_changed": False,
            "ns5_changed": False,
            "lr_or_wd_changed": False,
            "floating_point_execution_changed": True,
            "fresh_quality_trajectory_required": True,
        }


class Method1RecursiveInverseRouter(
    _RecursiveAllStepInverseMixin, Method1GroupedCollectiveOptimizer
):
    checkpoint_schema = FAMILY_ID + "_method1"


class Method2RecursiveInverseRouter(
    _RecursiveAllStepInverseMixin, Method2GroupedQualifiedRouter
):
    checkpoint_schema = FAMILY_ID + "_method2"


__all__ = (
    "FAMILY_ID",
    "LEAF_WIDTH",
    "Method1RecursiveInverseRouter",
    "Method2RecursiveInverseRouter",
    "_RecursiveAllStepInverseMixin",
    "_recursive_lower_inverse",
)
