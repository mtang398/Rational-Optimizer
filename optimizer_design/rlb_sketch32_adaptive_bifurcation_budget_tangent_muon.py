"""Budget-tangent Global-RLB with a sketched adaptive/sign bifurcation.

The strongest positive Method-3 trajectory kept two structured directions
substantially separated throughout training: a factorized adaptive source and
a sign-family source.  The compact owner-free descendants retained the
loss-aware coefficient transaction but collapsed both sources into
``span(polar(momentum), sign(momentum))``.  Their early lead consequently
decayed as the selected direction returned toward the parent.

This method restores the missing source diversity without a second polar map.
For every logical layer/group and matrix role, 32 deterministic row buckets
and 32 deterministic column buckets maintain matched-beta2 gradient-square
statistics.  Their separable inverse-root reconstruction supplies an adaptive
post-polar branch; the other branch is the equal-budget Nesterov sign family.
Global-RLB participation controls both inner branches, frozen-response
congruence closes their outer chord, and the parent-quotient fixed-32 response
transaction still selects fresh signed coefficients across all layer/groups.

The bucket state is O(32 L (G+1)), independent of activation positions N,
intermediate width H, and model width d.  Arbitrary shards need communicate
only bucket statistics and scalar transaction summaries.  No complete-layer
owner, dense (LG)x(LG) metric, dense projector, or selected update publication
is used.
"""

from __future__ import annotations

import threading

import torch

from . import rlb_compact_four_role_response_homotopy_muon as _compact
from . import rlb_lagged_predictive_response_transaction_muon as _predictive
from . import rlb_rank64_cadence8_response_group_polar_muon as _cadence
from .rlb_budget_tangent_window32_cadence8_group_polar_muon import (
    FAMILY_ID as BUDGET_PARENT_FAMILY_ID,
    PREFIX as BUDGET_PARENT_PREFIX,
    BudgetTangentWindow32Cadence8GroupPolarRouter,
    BudgetTangentWindow32Cadence8HeadPolarAttentionOptimizer,
    budget_tangent_window32_scaling_formula,
)


FAMILY_ID = "sketch32_adaptive_bifurcation_budget_tangent_muon_v1"
PREFIX = "sketch32_adaptive_bifurcation_budget_tangent_"
SKETCH_BINS = 32
MATCHED_BETA2 = 0.95
LOCKED_EPS = 1.0e-8

_PATCH_LOCK = threading.RLock()
_EXPECTED_REFRESH_DIRECTION = (
    _predictive.compact_postpolar_group_response_homotopy
)
_EXPECTED_ORDINARY_DIRECTION = (
    _cadence.compact_postpolar_group_response_homotopy
)
_EXPECTED_ATTENTION_DIRECTION = (
    _compact.compact_postpolar_group_response_homotopy
)


def _group_view(
    value: torch.Tensor,
    *,
    groups: int,
    width: int | None,
    grouped_axis: str,
) -> tuple[torch.Tensor, object]:
    """Return [layers, groups, rows, columns] and a restoring closure."""

    if value.ndim != 3:
        raise RuntimeError("sketch32 bifurcation requires batched matrices")
    layers, rows, columns = map(int, value.shape)
    if grouped_axis == "rows":
        if width is None or rows != int(groups) * int(width):
            raise RuntimeError("sketch32 row inventory changed")
        return value.float().view(
            layers, int(groups), int(width), columns
        ), lambda selected: selected.reshape_as(value)
    if grouped_axis == "columns":
        if width is None or columns != int(groups) * int(width):
            raise RuntimeError("sketch32 column inventory changed")
        grouped = value.float().transpose(-2, -1).contiguous().view(
            layers, int(groups), int(width), rows
        )
        return grouped, lambda selected: selected.reshape(
            layers, columns, rows
        ).transpose(-2, -1).contiguous()
    if grouped_axis == "matrix":
        if int(groups) != 1 or width is not None:
            raise RuntimeError("sketch32 matrix inventory changed")
        return value.float()[:, None], lambda selected: selected[:, 0]
    raise ValueError(f"unknown sketch32 grouped axis: {grouped_axis}")


def _bucket_ids(length: int, bins: int, device: torch.device) -> torch.Tensor:
    if int(length) <= 0 or int(bins) <= 0 or int(bins) > int(length):
        raise ValueError("sketch32 bucket dimensions are invalid")
    logical = torch.arange(int(length), device=device, dtype=torch.int64)
    return torch.div(logical * int(bins), int(length), rounding_mode="floor")


def _bucket_means(value: torch.Tensor, bins: int) -> torch.Tensor:
    """Average the final logical axis into deterministic contiguous buckets."""

    length = int(value.shape[-1])
    bins = int(bins)
    if bins <= 0 or bins > length:
        raise ValueError("sketch32 realized bucket count is invalid")
    if length % bins == 0:
        return value.reshape(*value.shape[:-1], bins, length // bins).mean(-1)
    ids = _bucket_ids(length, bins, value.device)
    result = value.new_zeros(*value.shape[:-1], bins)
    result.scatter_add_(
        -1,
        ids.view(*((1,) * (value.ndim - 1)), length).expand_as(value),
        value,
    )
    counts = torch.bincount(ids, minlength=bins).to(value.dtype)
    return result / counts.view(*((1,) * (value.ndim - 1)), bins)


def _expand_buckets(value: torch.Tensor, length: int) -> torch.Tensor:
    bins = int(value.shape[-1])
    ids = _bucket_ids(int(length), bins, value.device)
    return value.index_select(-1, ids)


def sketch32_adaptive_bifurcation_direction(
    parent: torch.Tensor,
    momentum: torch.Tensor,
    gradient: torch.Tensor,
    row_second_moment: torch.Tensor,
    column_second_moment: torch.Tensor,
    participation: torch.Tensor,
    congruence: torch.Tensor,
    *,
    groups: int,
    width: int | None,
    grouped_axis: str,
    beta2: float,
    step: int,
    eps: float = LOCKED_EPS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Form one-polar adaptive/sign branches from fixed bucket statistics."""

    if not (
        parent.shape == momentum.shape == gradient.shape and parent.ndim == 3
    ):
        raise RuntimeError("sketch32 bifurcation tensor inventory changed")
    if (
        float(beta2) != MATCHED_BETA2
        or int(step) < 1
        or float(eps) != LOCKED_EPS
    ):
        raise ValueError("sketch32 bifurcation locked numerics changed")
    p, restore = _group_view(
        parent, groups=groups, width=width, grouped_axis=grouped_axis
    )
    m, _ = _group_view(
        momentum, groups=groups, width=width, grouped_axis=grouped_axis
    )
    g, _ = _group_view(
        gradient, groups=groups, width=width, grouped_axis=grouped_axis
    )
    expected = p.shape[:2]
    row_bins = min(SKETCH_BINS, int(p.shape[-2]))
    column_bins = min(SKETCH_BINS, int(p.shape[-1]))
    if (
        participation.shape != expected
        or congruence.shape != expected
        or row_second_moment.shape != expected + (row_bins,)
        or column_second_moment.shape != expected + (column_bins,)
    ):
        raise RuntimeError("sketch32 bifurcation state inventory changed")

    squared = g.square()
    row_observation = _bucket_means(squared.mean(dim=-1), row_bins)
    column_observation = _bucket_means(squared.mean(dim=-2), column_bins)
    row_second_moment.mul_(beta2).add_(
        row_observation, alpha=1.0 - beta2
    )
    column_second_moment.mul_(beta2).add_(
        column_observation, alpha=1.0 - beta2
    )
    correction = 1.0 - float(beta2) ** int(step)
    row = row_second_moment / correction
    column = column_second_moment / correction
    row_inverse = torch.rsqrt(row + float(eps))
    column_inverse = torch.rsqrt(column + float(eps))
    factor_scale = torch.sqrt(row.mean(dim=-1, keepdim=True) + float(eps))
    adaptive = (
        m
        * _expand_buckets(row_inverse, int(p.shape[-2]))[..., :, None]
        * _expand_buckets(column_inverse, int(p.shape[-1]))[..., None, :]
        * factor_scale[..., None]
    )

    dims = (-2, -1)
    tiny = torch.finfo(p.dtype).tiny
    parent_norm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
    adaptive_norm = torch.linalg.vector_norm(adaptive, dim=dims, keepdim=True)
    sign = torch.sign(m)
    sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
    valid = (
        torch.isfinite(p).all(dim=dims, keepdim=True)
        & torch.isfinite(adaptive).all(dim=dims, keepdim=True)
        & torch.isfinite(sign).all(dim=dims, keepdim=True)
        & (parent_norm > 0.0)
        & (adaptive_norm > 0.0)
        & (sign_norm > 0.0)
    )
    adaptive_equal = adaptive * (
        parent_norm / adaptive_norm.clamp_min(tiny)
    )
    sign_equal = sign * (parent_norm / sign_norm.clamp_min(tiny))

    c = participation.float().clamp(0.0, 1.0)[..., None, None]
    root_c = torch.sqrt(c)
    root_residual = torch.sqrt((1.0 - c).clamp_min(0.0))
    u6_source = root_c * p + root_residual * adaptive_equal
    u5_source = root_c * p + root_residual * sign_equal
    u6 = u6_source * (
        parent_norm
        / torch.linalg.vector_norm(u6_source, dim=dims, keepdim=True).clamp_min(
            tiny
        )
    )
    u5 = u5_source * (
        parent_norm
        / torch.linalg.vector_norm(u5_source, dim=dims, keepdim=True).clamp_min(
            tiny
        )
    )
    u6 = torch.where(c == 1.0, p, u6)
    u5 = torch.where(c == 1.0, p, u5)

    a = congruence.float().clamp(0.0, 1.0)[..., None, None]
    delta = torch.sqrt((1.0 - a.square()).clamp_min(0.0))
    candidate_source = a * u6 + delta * u5
    candidate = candidate_source * (
        parent_norm
        / torch.linalg.vector_norm(
            candidate_source, dim=dims, keepdim=True
        ).clamp_min(tiny)
    )
    candidate = torch.where(a == 1.0, u6, candidate)
    candidate = torch.where(a == 0.0, u5, candidate)
    candidate = torch.where(valid, candidate, p)

    descents = torch.stack(
        tuple((g * value).sum(dim=dims) for value in (p, u6, u5, candidate)),
        dim=-1,
    )
    safe = (
        valid[..., 0, 0]
        & torch.isfinite(descents).all(dim=-1)
        & (descents[..., 0] > 0.0)
        & (descents[..., 3] > 0.0)
    )
    selected = torch.where(safe[..., None, None], candidate, p)
    selected_norm = torch.linalg.vector_norm(selected, dim=dims)
    parent_norm_flat = parent_norm[..., 0, 0]
    parent_cosine = (
        (p * selected).sum(dim=dims)
        / (parent_norm_flat * selected_norm).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    branch_cosine = (
        (u6 * u5).sum(dim=dims)
        / (
            torch.linalg.vector_norm(u6, dim=dims)
            * torch.linalg.vector_norm(u5, dim=dims)
        ).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    adaptive_parent_cosine = (
        (p * adaptive_equal).sum(dim=dims)
        / (
            parent_norm_flat
            * torch.linalg.vector_norm(adaptive_equal, dim=dims)
        ).clamp_min(float(eps))
    ).clamp(-1.0, 1.0)
    budget = (
        (selected_norm - parent_norm_flat).abs()
        / parent_norm_flat.clamp_min(1.0)
    )
    torch._assert_async(torch.isfinite(selected).all())
    return restore(selected), {
        "active": valid[..., 0, 0],
        "safe": safe,
        "parent_cosine": parent_cosine,
        "branch_cosine": branch_cosine,
        "branch_disagreement": 1.0 - branch_cosine,
        "adaptive_parent_cosine": adaptive_parent_cosine,
        "budget_residual": budget,
        "branch_descents": descents,
        "row_bins": torch.tensor(row_bins, device=p.device),
        "column_bins": torch.tensor(column_bins, device=p.device),
    }


def sketch32_adaptive_bifurcation_scaling_formula(
    *,
    total_positions: int,
    total_layers: int,
    total_groups: int,
    intermediate_width: int,
    model_width: int,
) -> dict[str, int | float]:
    result = dict(budget_tangent_window32_scaling_formula(
        total_positions=total_positions,
        total_layers=total_layers,
        total_groups=total_groups,
        intermediate_width=intermediate_width,
        model_width=model_width,
    ))
    layers = int(total_layers)
    groups = int(total_groups)
    factor_state = 4 * SKETCH_BINS * layers * (groups + 1)
    arbitrary_shard_direction_summary = (
        2 * layers * (groups + 1) * (4 * SKETCH_BINS + 10)
    )
    result["sketch_bins_per_axis"] = SKETCH_BINS
    result["factorized_bucket_state_elements"] = factor_state
    result["persistent_state_elements"] += factor_state
    result["communicated_summary_elements"] += (
        arbitrary_shard_direction_summary
    )
    result["ordinary_communicated_summary_elements"] += (
        arbitrary_shard_direction_summary
    )
    result["arbitrary_shard_direction_summary_elements"] = (
        arbitrary_shard_direction_summary
    )
    result["additional_persistent_state_elements"] = factor_state
    result["state_depends_on_total_activation_positions"] = 0
    result["state_depends_on_intermediate_or_model_width"] = 0
    result["additional_dense_solve_dimension"] = 0
    return result


def _state_for_direction(
    optimizer,
    *,
    parent: torch.Tensor,
    groups: int,
    width: int | None,
    grouped_axis: str,
    key_prefix: str,
):
    grouped, _ = _group_view(
        parent, groups=groups, width=width, grouped_axis=grouped_axis
    )
    row_shape = grouped.shape[:2] + (min(SKETCH_BINS, grouped.shape[-2]),)
    column_shape = grouped.shape[:2] + (
        min(SKETCH_BINS, grouped.shape[-1]),
    )
    anchor = optimizer.state[
        optimizer.pairs[0]["in_weight"]
        if hasattr(optimizer, "pairs")
        else optimizer.role_parameters["qkv"][0]
    ]
    row_key = key_prefix + "_row_second_moment"
    column_key = key_prefix + "_column_second_moment"
    row = anchor.get(row_key)
    column = anchor.get(column_key)
    if row is None:
        row = torch.zeros(row_shape, device=grouped.device, dtype=torch.float32)
        anchor[row_key] = row
    if column is None:
        column = torch.zeros(
            column_shape, device=grouped.device, dtype=torch.float32
        )
        anchor[column_key] = column
    if row.shape != row_shape or column.shape != column_shape:
        raise RuntimeError("sketch32 checkpoint bucket inventory changed")
    return anchor, row, column


class Sketch32AdaptiveBifurcationBudgetTangentRouter(
    BudgetTangentWindow32Cadence8GroupPolarRouter
):
    family_id = FAMILY_ID
    telemetry_prefix = PREFIX
    fairness_component = (
        "sketch32_adaptive_bifurcation_budget_tangent_lr_scale"
    )

    def __init__(self, pairs, **kwargs):
        if float(kwargs.get("beta2", -1.0)) != MATCHED_BETA2:
            raise ValueError("sketch32 bifurcation requires locked beta2=.95")
        if float(kwargs.get("eps", -1.0)) != LOCKED_EPS:
            raise ValueError("sketch32 bifurcation requires locked eps=1e-8")
        super().__init__(pairs, **kwargs)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "sketch32_factorized_adaptive_branch_lr_scale": 1.0,
            "adaptive_sign_bifurcation_lr_scale": 1.0,
            "signed_global_transaction_lr_scale": 1.0,
        })
        return result

    @torch.no_grad()
    def step(self, closure=None):
        records = []

        def direction(
            parent,
            momentum,
            gradient,
            participation,
            congruence,
            *,
            groups,
            width,
            grouped_axis,
            eps=LOCKED_EPS,
        ):
            anchor, rows, columns = _state_for_direction(
                self,
                parent=parent,
                groups=groups,
                width=width,
                grouped_axis=grouped_axis,
                key_prefix="sketch32_bifurcation_" + grouped_axis,
            )
            step = int(
                anchor.get("predictive_response_transaction_updates", 0)
            ) + 1
            selected, metadata = sketch32_adaptive_bifurcation_direction(
                parent,
                momentum,
                gradient,
                rows,
                columns,
                participation,
                congruence,
                groups=groups,
                width=width,
                grouped_axis=grouped_axis,
                beta2=MATCHED_BETA2,
                step=step,
                eps=eps,
            )
            records.append(metadata)
            return selected, metadata

        with _PATCH_LOCK:
            if (
                _predictive.compact_postpolar_group_response_homotopy
                is not _EXPECTED_REFRESH_DIRECTION
                or _cadence.compact_postpolar_group_response_homotopy
                is not _EXPECTED_ORDINARY_DIRECTION
            ):
                raise RuntimeError("sketch32 router direction binding changed")
            _predictive.compact_postpolar_group_response_homotopy = direction
            _cadence.compact_postpolar_group_response_homotopy = direction
            try:
                loss = super().step(closure)
            finally:
                _predictive.compact_postpolar_group_response_homotopy = (
                    _EXPECTED_REFRESH_DIRECTION
                )
                _cadence.compact_postpolar_group_response_homotopy = (
                    _EXPECTED_ORDINARY_DIRECTION
                )

        self._last_telemetry = {
            key.replace(BUDGET_PARENT_PREFIX, PREFIX, 1): (
                FAMILY_ID if value == BUDGET_PARENT_FAMILY_ID else value
            )
            for key, value in self._last_telemetry.items()
        }
        if self._last_telemetry and records:
            safe = torch.cat([record["safe"].reshape(-1) for record in records])
            disagreement = torch.cat([
                record["branch_disagreement"].reshape(-1)
                for record in records
            ])
            adaptive = torch.cat([
                record["adaptive_parent_cosine"].reshape(-1)
                for record in records
            ])
            budget = torch.cat([
                record["budget_residual"].reshape(-1) for record in records
            ])
            scaling = sketch32_adaptive_bifurcation_scaling_formula(
                total_positions=1,
                total_layers=len(self.pairs),
                total_groups=self.groups,
                intermediate_width=self.hidden,
                model_width=self.external,
            )
            self._last_telemetry.update({
                PREFIX + "family_id": FAMILY_ID,
                PREFIX + "adaptive_branch_safe_fraction": float(
                    safe.float().mean().item()
                ),
                PREFIX + "branch_disagreement_median": float(
                    disagreement.median().item()
                ),
                PREFIX + "adaptive_parent_cosine_median": float(
                    adaptive.median().item()
                ),
                PREFIX + "adaptive_budget_residual_max": float(
                    budget.amax().item()
                ),
                PREFIX + "sketch_bins_per_axis": SKETCH_BINS,
                PREFIX + "factorized_bucket_state_elements": scaling[
                    "factorized_bucket_state_elements"
                ],
                PREFIX + "state_depends_on_total_activation_positions": 0,
                PREFIX + "state_depends_on_intermediate_or_model_width": 0,
                PREFIX + "additional_dense_solve_dimension": 0,
            })
        return loss


class Sketch32AdaptiveBifurcationBudgetTangentAttentionOptimizer(
    BudgetTangentWindow32Cadence8HeadPolarAttentionOptimizer
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_groups[0][PREFIX + "family_id"] = FAMILY_ID

    def lr_wd_fairness_audit(self):
        result = dict(super().lr_wd_fairness_audit())
        result.update({
            "sketch32_attention_adaptive_branch_lr_scale": 1.0,
            "attention_adaptive_sign_bifurcation_lr_scale": 1.0,
        })
        return result

    @torch.no_grad()
    def step(self, closure=None):
        records = []
        calls = 0
        anchor = self.state[self.role_parameters["qkv"][0]]
        adaptive_step = int(
            anchor.get("sketch32_attention_bifurcation_updates", 0)
        ) + 1

        def direction(
            parent,
            momentum,
            gradient,
            participation,
            congruence,
            *,
            groups,
            width,
            grouped_axis,
            eps=LOCKED_EPS,
        ):
            nonlocal calls
            role = "qkv" if calls == 0 else "attention_output"
            if calls >= 2:
                raise RuntimeError("sketch32 attention role count changed")
            calls += 1
            _anchor, rows, columns = _state_for_direction(
                self,
                parent=parent,
                groups=groups,
                width=width,
                grouped_axis=grouped_axis,
                key_prefix="sketch32_bifurcation_" + role,
            )
            selected, metadata = sketch32_adaptive_bifurcation_direction(
                parent,
                momentum,
                gradient,
                rows,
                columns,
                participation,
                congruence,
                groups=groups,
                width=width,
                grouped_axis=grouped_axis,
                beta2=MATCHED_BETA2,
                step=adaptive_step,
                eps=eps,
            )
            records.append(metadata)
            return selected, metadata

        with _PATCH_LOCK:
            if (
                _compact.compact_postpolar_group_response_homotopy
                is not _EXPECTED_ATTENTION_DIRECTION
            ):
                raise RuntimeError("sketch32 attention direction binding changed")
            _compact.compact_postpolar_group_response_homotopy = direction
            try:
                loss = super().step(closure)
            finally:
                _compact.compact_postpolar_group_response_homotopy = (
                    _EXPECTED_ATTENTION_DIRECTION
                )
        if calls != 2:
            raise RuntimeError("sketch32 attention omitted a matrix role")
        anchor["sketch32_attention_bifurcation_updates"] = adaptive_step
        self._last_telemetry = {
            key.replace(BUDGET_PARENT_PREFIX, PREFIX, 1): (
                FAMILY_ID if value == BUDGET_PARENT_FAMILY_ID else value
            )
            for key, value in self._last_telemetry.items()
        }
        if self._last_telemetry:
            disagreement = torch.cat([
                record["branch_disagreement"].reshape(-1)
                for record in records
            ])
            self._last_telemetry.update({
                PREFIX + "attention_family_id": FAMILY_ID,
                PREFIX + "attention_branch_disagreement_median": float(
                    disagreement.median().item()
                ),
                PREFIX + "attention_sketch_bins_per_axis": SKETCH_BINS,
                PREFIX + "attention_owner_count": 0,
                PREFIX + "attention_selected_update_elements_published": 0,
            })
        return loss


__all__ = (
    "FAMILY_ID",
    "LOCKED_EPS",
    "MATCHED_BETA2",
    "PREFIX",
    "SKETCH_BINS",
    "Sketch32AdaptiveBifurcationBudgetTangentAttentionOptimizer",
    "Sketch32AdaptiveBifurcationBudgetTangentRouter",
    "sketch32_adaptive_bifurcation_direction",
    "sketch32_adaptive_bifurcation_scaling_formula",
)
