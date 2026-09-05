"""Owner-free temporal response sketch with structured group-polar Muon.

The positive response-transaction endpoints show that current Global-RLB
loss images are useful, while the successful Method-3 lineage shows that a
matched-beta2 *persistent* score geometry is needed to keep that advantage
late in training.  This candidate retains the exact response-parent loss
image and replaces row-aligned EMA with a rotation-invariant discounted
Frequent-Directions sketch.  Selection is predictive: the previous sketch is
used before the current loss image is incorporated.

The sketch has the campaign's already fixed 32 rows.  Its state is therefore
``32 * (L G + 1)`` rather than ``O((L G)^2)``.  A fixed 64-by-64 row Gram is
the largest temporal algebra.  Only scalar score/coefficient summaries are
communicated; selected updates remain on their native parameter shards.

To remove the dominant full-matrix NS5 cost, MLP matrices are polarized in
native rational-group blocks and attention matrices in native four-head
blocks.  The literal BF16 quintic Newton--Schulz polynomial and all five
iterations are unchanged.  A nominal-rank factor preserves each full
matrix's Muon Frobenius calibration.  These block maps change the optimizer
geometry, so this method has no inherited endpoint claim.
"""

from __future__ import annotations

import math
import threading

import torch

from . import rlb_compact_four_role_response_homotopy_muon as _compact
from . import rlb_lagged_predictive_response_transaction_muon as _predictive
from .rlb_fixed32_functional_row_muon import FIXED_GLOBAL_PROBE_COUNT
from .rlb_group_muon_core import _batched_zero_power as _literal_zero_power
from .rlb_lagged_predictive_response_transaction_muon import (
    LaggedPredictiveResponseTransactionAttentionOptimizer,
    LaggedPredictiveResponseTransactionRouter,
    MatchedBeta2PredictiveRows,
    lagged_predictive_response_transaction_scaling_formula,
)


FAMILY_ID = "temporal_response_group_polar_muon_v1"
ATTENTION_HEADS = 16
ATTENTION_HEAD_GROUP_SIZE = 4
_PATCH_LOCK = threading.RLock()
_EXPECTED_PREDICTIVE_ROWS = _predictive.matched_beta2_predictive_rows
_EXPECTED_TRANSACTION = _predictive._transaction_from_replicated_global_rows
_EXPECTED_ROUTER_POLAR = _predictive._batched_zero_power
_EXPECTED_ATTENTION_POLAR = _compact._batched_zero_power


def _validate_predictive_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_decay_action: torch.Tensor | None,
    *,
    beta2: float,
) -> None:
    rows = FIXED_GLOBAL_PROBE_COUNT
    if (
        current_scores.ndim != 2
        or current_scores.shape[0] != rows
        or current_decay_action.shape != (rows,)
        or current_decay_action.dtype != current_scores.dtype
        or current_decay_action.device != current_scores.device
        or not current_scores.is_floating_point()
        or float(beta2) != 0.95
        or not bool(torch.isfinite(current_scores).all())
        or not bool(torch.isfinite(current_decay_action).all())
    ):
        raise RuntimeError("temporal response sketch inventory changed")
    if (previous_scores is None) != (previous_decay_action is None):
        raise RuntimeError("temporal response sketch histories must coinitialize")
    if previous_scores is not None and (
        previous_scores.shape != current_scores.shape
        or previous_decay_action is None
        or previous_decay_action.shape != current_decay_action.shape
        or previous_scores.dtype != current_scores.dtype
        or previous_decay_action.dtype != current_scores.dtype
        or previous_scores.device != current_scores.device
        or previous_decay_action.device != current_scores.device
        or not bool(torch.isfinite(previous_scores).all())
        or not bool(torch.isfinite(previous_decay_action).all())
    ):
        raise RuntimeError("temporal response sketch checkpoint changed")


def temporal_response_predictive_rows(
    current_scores: torch.Tensor,
    current_decay_action: torch.Tensor,
    previous_scores: torch.Tensor | None,
    previous_decay_action: torch.Tensor | None,
    *,
    beta2: float,
) -> MatchedBeta2PredictiveRows:
    """Predict with the old covariance factor, then update it by FD.

    If ``B`` is the previous factor and ``S`` the current fixed loss image,
    the augmented factor is ``[sqrt(beta2) B; sqrt(1-beta2) S]``.  The top 32
    Frequent-Directions rows and the identically transformed decay action are
    retained.  Consequently row rotations or permutations do not define the
    temporal correspondence, unlike an elementwise row EMA.
    """

    _validate_predictive_rows(
        current_scores,
        current_decay_action,
        previous_scores,
        previous_decay_action,
        beta2=beta2,
    )
    if previous_scores is None:
        zero = current_scores.new_zeros(())
        return MatchedBeta2PredictiveRows(
            selection_scores=current_scores,
            selection_decay_action=current_decay_action,
            updated_scores=current_scores.detach().clone(),
            updated_decay_action=current_decay_action.detach().clone(),
            history_used=False,
            relative_innovation=zero,
        )

    assert previous_decay_action is not None
    rows = FIXED_GLOBAL_PROBE_COUNT
    beta = float(beta2)
    augmented = torch.cat((
        previous_scores * math.sqrt(beta),
        current_scores * math.sqrt(1.0 - beta),
    ), dim=0)
    augmented_decay = torch.cat((
        previous_decay_action * math.sqrt(beta),
        current_decay_action * math.sqrt(1.0 - beta),
    ), dim=0)
    row_gram = augmented @ augmented.T
    row_gram = 0.5 * (row_gram + row_gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(row_gram)
    eigenvalues = eigenvalues.flip(0).clamp_min(0.0)
    eigenvectors = eigenvectors.flip(1)
    kept = eigenvalues[:rows]
    shrinkage = eigenvalues[rows]
    tiny = torch.finfo(current_scores.dtype).tiny
    ratios = torch.sqrt(
        (kept - shrinkage).clamp_min(0.0) / kept.clamp_min(tiny)
    )
    transform = ratios[:, None] * eigenvectors[:, :rows].T
    updated_scores = transform @ augmented
    updated_decay = transform @ augmented_decay

    # Rotation-invariant covariance innovation.  The cross term identity
    # ||S^T S - B^T B||_F^2 uses only fixed 32-by-32 row products.
    current_gram = current_scores @ current_scores.T
    previous_gram = previous_scores @ previous_scores.T
    cross_gram = current_scores @ previous_scores.T
    innovation2 = (
        current_gram.square().sum()
        + previous_gram.square().sum()
        - 2.0 * cross_gram.square().sum()
    ).clamp_min(0.0)
    relative_innovation = torch.sqrt(
        innovation2 / current_gram.square().sum().clamp_min(tiny)
    )
    torch._assert_async(
        torch.isfinite(updated_scores).all()
        & torch.isfinite(updated_decay).all()
        & torch.isfinite(relative_innovation)
    )
    return MatchedBeta2PredictiveRows(
        selection_scores=previous_scores,
        selection_decay_action=previous_decay_action,
        updated_scores=updated_scores,
        updated_decay_action=updated_decay,
        history_used=True,
        relative_innovation=relative_innovation,
    )


def rational_group_zero_power(
    source: torch.Tensor,
    steps: int,
    *,
    groups: int,
    width: int,
) -> torch.Tensor:
    """Apply literal NS5 to rational-group row or column blocks."""

    if source.ndim != 3 or int(steps) != 5:
        raise RuntimeError("rational-group polar requires [layers,rows,cols] NS5")
    layers, rows, columns = map(int, source.shape)
    if int(groups) <= 0 or int(width) <= 0 or int(groups) * int(width) <= 0:
        raise ValueError("rational-group polar dimensions must be positive")
    hidden = int(groups) * int(width)
    if rows == hidden:
        blocks = source.reshape(
            layers, int(groups), int(width), columns
        ).reshape(layers * int(groups), int(width), columns)
        polar = _literal_zero_power(blocks, steps).reshape(
            layers, int(groups), int(width), columns
        ).reshape_as(source)
        block_rank = min(int(width), columns)
    elif columns == hidden:
        blocks = source.reshape(
            layers, rows, int(groups), int(width)
        ).permute(0, 2, 1, 3).reshape(
            layers * int(groups), rows, int(width)
        )
        polar = _literal_zero_power(blocks, steps).reshape(
            layers, int(groups), rows, int(width)
        ).permute(0, 2, 1, 3).reshape_as(source)
        block_rank = min(rows, int(width))
    else:
        raise RuntimeError("rational-group polar cannot identify grouped axis")
    global_rank = min(rows, columns)
    nominal_block_rank = int(groups) * int(block_rank)
    if global_rank <= 0 or nominal_block_rank <= 0:
        raise RuntimeError("rational-group polar rank is empty")
    return polar.mul(math.sqrt(float(global_rank) / float(nominal_block_rank)))


def attention_head_group_zero_power(
    source: torch.Tensor,
    steps: int,
    *,
    heads: int = ATTENTION_HEADS,
    head_group_size: int = ATTENTION_HEAD_GROUP_SIZE,
) -> torch.Tensor:
    """Apply literal NS5 to native Q/K/V or output head groups."""

    if source.ndim != 3 or int(steps) != 5:
        raise RuntimeError("attention head-group polar requires batched NS5")
    layers, rows, columns = map(int, source.shape)
    if columns % int(heads) or int(heads) % int(head_group_size):
        raise RuntimeError("attention head-group inventory is not divisible")
    head_width = columns // int(heads)
    group_count = int(heads) // int(head_group_size)
    block_width = int(head_group_size) * head_width
    if rows == 3 * columns:
        blocks = source.reshape(
            layers, 3, group_count, block_width, columns
        ).reshape(layers * 3 * group_count, block_width, columns)
        polar = _literal_zero_power(blocks, steps).reshape(
            layers, 3, group_count, block_width, columns
        ).reshape_as(source)
        return polar.mul(1.0 / math.sqrt(3.0))
    if rows == columns:
        blocks = source.reshape(
            layers, rows, group_count, block_width
        ).permute(0, 2, 1, 3).reshape(
            layers * group_count, rows, block_width
        )
        return _literal_zero_power(blocks, steps).reshape(
            layers, group_count, rows, block_width
        ).permute(0, 2, 1, 3).reshape_as(source)
    raise RuntimeError("attention head-group polar supports only QKV/output roles")


def temporal_response_group_polar_scaling_formula(
    *,
    total_positions: int,
    total_layers: int,
    total_groups: int,
    intermediate_width: int,
    model_width: int,
) -> dict[str, int]:
    result = lagged_predictive_response_transaction_scaling_formula(
        total_positions=total_positions,
        total_layers=total_layers,
        total_groups=total_groups,
        intermediate_width=intermediate_width,
        model_width=model_width,
    )
    result = dict(result)
    result["temporal_sketch_rows"] = FIXED_GLOBAL_PROBE_COUNT
    result["largest_temporal_dense_dimension"] = 2 * FIXED_GLOBAL_PROBE_COUNT
    result["dense_coordinate_metric_elements"] = 0
    result["owner_count"] = 0
    result["selected_update_elements_published"] = 0
    return result


class TemporalResponseGroupPolarRouter(
    LaggedPredictiveResponseTransactionRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = "temporal_response_group_polar_"
    fairness_component = "temporal_response_group_polar_lr_scale"
    # A class-level hook keeps the temporal factor update and its use as a
    # predictive versus posterior metric explicit.  Subclasses may change
    # this mathematical choice without copying the parameter-update path.
    predictive_rows_fn = staticmethod(temporal_response_predictive_rows)
    transaction_fn = staticmethod(_EXPECTED_TRANSACTION)

    def __init__(self, pairs, **kwargs):
        super().__init__(pairs, **kwargs)
        self.param_groups[0]["temporal_response_group_polar_family_id"] = (
            FAMILY_ID
        )

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report.update({
            "rotation_invariant_temporal_response_sketch_lr_scale": 1.0,
            "rational_group_polar_lr_scale": 1.0,
            "nominal_polar_rank_calibration_lr_scale": 1.0,
        })
        return report

    @torch.no_grad()
    def step(self, closure=None):
        with _PATCH_LOCK:
            if (
                _predictive.matched_beta2_predictive_rows
                is not _EXPECTED_PREDICTIVE_ROWS
                or _predictive._transaction_from_replicated_global_rows
                is not _EXPECTED_TRANSACTION
                or _predictive._batched_zero_power is not _EXPECTED_ROUTER_POLAR
            ):
                raise RuntimeError("temporal response router binding changed")

            def grouped(source, steps):
                return rational_group_zero_power(
                    source, steps, groups=self.groups, width=self.width
                )

            row_update = self.predictive_rows_fn
            transaction = self.transaction_fn
            if not callable(row_update) or not callable(transaction):
                raise RuntimeError("temporal response transaction hook changed")
            _predictive.matched_beta2_predictive_rows = row_update
            _predictive._transaction_from_replicated_global_rows = transaction
            _predictive._batched_zero_power = grouped
            try:
                loss = super().step(closure)
            finally:
                _predictive.matched_beta2_predictive_rows = (
                    _EXPECTED_PREDICTIVE_ROWS
                )
                _predictive._transaction_from_replicated_global_rows = (
                    _EXPECTED_TRANSACTION
                )
                _predictive._batched_zero_power = _EXPECTED_ROUTER_POLAR
        if self._last_telemetry:
            renamed = {}
            old = "lagged_predictive_response_transaction_"
            new = "temporal_response_group_polar_"
            for key, value in self._last_telemetry.items():
                renamed[key.replace(old, new, 1)] = (
                    FAMILY_ID if value == _predictive.FAMILY_ID else value
                )
            self._last_telemetry = renamed
            self._last_telemetry.update({
                new + "family_id": FAMILY_ID,
                new + "temporal_sketch_rows": FIXED_GLOBAL_PROBE_COUNT,
                new + "largest_temporal_dense_dimension": (
                    2 * FIXED_GLOBAL_PROBE_COUNT
                ),
                new + "rational_group_polar": 1,
                new + "owner_count": 0,
                new + "dense_lg_metric_elements": 0,
                new + "selected_update_elements_published": 0,
            })
        return loss


class TemporalResponseHeadGroupPolarAttentionOptimizer(
    LaggedPredictiveResponseTransactionAttentionOptimizer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0]["temporal_response_group_polar_family_id"] = (
            FAMILY_ID
        )

    def lr_wd_fairness_audit(self):
        report = dict(super().lr_wd_fairness_audit())
        report["attention_head_group_polar_lr_scale"] = 1.0
        report["attention_nominal_polar_rank_calibration_lr_scale"] = 1.0
        return report

    @torch.no_grad()
    def step(self, closure=None):
        with _PATCH_LOCK:
            if _compact._batched_zero_power is not _EXPECTED_ATTENTION_POLAR:
                raise RuntimeError("temporal response attention binding changed")
            _compact._batched_zero_power = attention_head_group_zero_power
            try:
                loss = super().step(closure)
            finally:
                _compact._batched_zero_power = _EXPECTED_ATTENTION_POLAR
        if self._last_telemetry:
            for key, value in tuple(self._last_telemetry.items()):
                if value in {
                    _predictive.FAMILY_ID,
                    _predictive.CURRENT_IMPLEMENTATION_FAMILY_ID,
                }:
                    self._last_telemetry[key] = FAMILY_ID
            self._last_telemetry.update({
                "temporal_response_group_polar_attention_family_id": FAMILY_ID,
                "temporal_response_group_polar_attention_head_count": (
                    ATTENTION_HEADS
                ),
                "temporal_response_group_polar_attention_head_group_size": (
                    ATTENTION_HEAD_GROUP_SIZE
                ),
                "temporal_response_group_polar_attention_ns_steps": 5,
                "temporal_response_group_polar_attention_owner_count": 0,
                "temporal_response_group_polar_attention_selected_update_elements_published": 0,
            })
        return loss


__all__ = (
    "ATTENTION_HEADS",
    "ATTENTION_HEAD_GROUP_SIZE",
    "FAMILY_ID",
    "TemporalResponseGroupPolarRouter",
    "TemporalResponseHeadGroupPolarAttentionOptimizer",
    "attention_head_group_zero_power",
    "rational_group_zero_power",
    "temporal_response_group_polar_scaling_formula",
    "temporal_response_predictive_rows",
)
