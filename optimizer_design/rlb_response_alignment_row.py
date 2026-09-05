"""Response-alignment homotopy for owner-free Global-RLB directions.

The learned P5/Q4 response supplies a parameter-free angle between the live
and initial rational functions.  A lagged exact-loss comparison decides
whether a cheap row-normalized alternative is allowed to enter the ordinary
Muon direction.  All persistent objects are indexed by ``(layer, group)``;
none depends on the number of activation positions.
"""

from __future__ import annotations

import torch


def _evaluate_response(
    unit: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a grouped P5/Q4 rational function and its derivative."""

    if unit.ndim != 3:
        raise RuntimeError("response-alignment unit inventory changed")
    groups = unit.shape[1]
    if numerator.shape != (groups, 6) or denominator.shape != (groups, 4):
        raise RuntimeError("response-alignment coefficient inventory changed")
    square = unit.square()
    cube = square * unit
    fourth = square.square()
    fifth = fourth * unit
    absolute = unit.abs()
    num = numerator.float()[None, :, None, :]
    den = denominator.float().abs()[None, :, None, :]
    polynomial = (
        num[..., 0]
        + num[..., 1] * unit
        + num[..., 2] * square
        + num[..., 3] * cube
        + num[..., 4] * fourth
        + num[..., 5] * fifth
    )
    polynomial_derivative = (
        num[..., 1]
        + 2.0 * num[..., 2] * unit
        + 3.0 * num[..., 3] * square
        + 4.0 * num[..., 4] * cube
        + 5.0 * num[..., 5] * fourth
    )
    divisor = (
        1.0
        + den[..., 0] * absolute
        + den[..., 1] * square
        + den[..., 2] * absolute * square
        + den[..., 3] * fourth
    )
    divisor_derivative = (
        den[..., 0] * torch.sign(unit)
        + 2.0 * den[..., 1] * unit
        + 3.0 * den[..., 2] * unit * absolute
        + 4.0 * den[..., 3] * cube
    )
    function = polynomial / divisor
    derivative = (
        polynomial_derivative * divisor
        - polynomial * divisor_derivative
    ) / divisor.square()
    return function, derivative


def _jacobian_kernel_inner(
    unit: torch.Tensor,
    function_a: torch.Tensor,
    derivative_a: torch.Tensor,
    function_b: torch.Tensor,
    derivative_b: torch.Tensor,
) -> torch.Tensor:
    """Exact O(width) inner product of two normalized-RLB Jacobian kernels."""

    width = float(unit.shape[-1])
    radial_a = (function_a - unit * derivative_a) / width
    radial_b = (function_b - unit * derivative_b) / width
    product_a = derivative_a * unit
    product_b = derivative_b * unit
    square_a = derivative_a.square()
    square_b = derivative_b.square()
    unit_energy = unit.square().sum(dim=-1)

    diagonal = (square_a * square_b).sum(dim=-1)
    diagonal = diagonal + 2.0 * (
        square_a * product_b * radial_b
    ).sum(dim=-1)
    diagonal = diagonal + unit_energy * (
        square_a * radial_b.square()
    ).sum(dim=-1)
    diagonal = diagonal + 2.0 * (
        square_b * product_a * radial_a
    ).sum(dim=-1)
    diagonal = diagonal + unit_energy * (
        square_b * radial_a.square()
    ).sum(dim=-1)

    basis_a = torch.stack((product_a, radial_a), dim=-1)
    basis_b = torch.stack((product_b, radial_b), dim=-1)
    gram = basis_a.transpose(-2, -1) @ basis_b
    coupling = torch.zeros_like(gram)
    coupling[..., 0, 1] = 1.0
    coupling[..., 1, 0] = 1.0
    coupling[..., 1, 1] = unit_energy
    rank_two = ((coupling @ gram @ coupling) * gram).sum(dim=(-2, -1))
    return diagonal + rank_two


def grouped_response_statistics(
    preactivation: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    frozen_numerator: torch.Tensor,
    frozen_denominator: torch.Tensor,
    *,
    groups: int,
    width: int,
    eps: float,
) -> torch.Tensor:
    """Return additive ``[group, role, cross/live/frozen]`` statistics."""

    if preactivation.ndim != 2 or preactivation.shape[1] != groups * width:
        raise RuntimeError("response-alignment preactivation inventory changed")
    if float(eps) <= 0.0:
        raise ValueError("response alignment requires the activation's positive epsilon")
    value = preactivation.float().view(-1, int(groups), int(width))
    rms = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    unit = value / rms
    live_function, live_derivative = _evaluate_response(
        unit, numerator, denominator
    )
    frozen_function, frozen_derivative = _evaluate_response(
        unit, frozen_numerator, frozen_denominator
    )

    incoming_cross = _jacobian_kernel_inner(
        unit,
        live_function,
        live_derivative,
        frozen_function,
        frozen_derivative,
    ).sum(dim=0)
    incoming_live = _jacobian_kernel_inner(
        unit,
        live_function,
        live_derivative,
        live_function,
        live_derivative,
    ).sum(dim=0)
    incoming_frozen = _jacobian_kernel_inner(
        unit,
        frozen_function,
        frozen_derivative,
        frozen_function,
        frozen_derivative,
    ).sum(dim=0)

    live_feature = rms * live_function
    frozen_feature = rms * frozen_function
    outgoing_cross = (
        (live_feature * frozen_feature).sum(dim=-1).square().sum(dim=0)
    )
    outgoing_live = live_feature.square().sum(dim=-1).square().sum(dim=0)
    outgoing_frozen = (
        frozen_feature.square().sum(dim=-1).square().sum(dim=0)
    )
    statistics = torch.stack((
        torch.stack((incoming_cross, incoming_live, incoming_frozen), dim=-1),
        torch.stack((outgoing_cross, outgoing_live, outgoing_frozen), dim=-1),
    ), dim=1)
    torch._assert_async(torch.isfinite(statistics).all())
    return statistics


def response_alignment_from_statistics(
    statistics: torch.Tensor,
    exact_initializer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert additive kernel statistics to squared CKA alignments."""

    if statistics.ndim < 3 or statistics.shape[-2:] != (2, 3):
        raise RuntimeError("response-alignment statistic inventory changed")
    denominator = torch.sqrt(statistics[..., 1] * statistics[..., 2])
    if bool((denominator <= 0.0).any()) or not bool(torch.isfinite(statistics).all()):
        raise RuntimeError("response alignment lost finite positive scale")
    alignment = (statistics[..., 0] / denominator).clamp(0.0, 1.0)
    if exact_initializer is not None:
        if exact_initializer.shape != statistics.shape[:-2]:
            raise RuntimeError("response-alignment initializer mask changed")
        alignment = torch.where(
            exact_initializer.unsqueeze(-1), torch.ones_like(alignment), alignment
        )
    return alignment


def row_normalized_equal_budget(
    parent: torch.Tensor,
    source: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Build a row-normalized alternative on every parent group sphere."""

    if parent.shape != source.shape or parent.ndim != 3:
        raise RuntimeError("response-alignment row direction inventory changed")
    if float(eps) != 1.0e-8:
        raise ValueError("response-alignment row direction uses locked epsilon")
    parent64 = parent.float()
    source64 = source.float()
    source_norm = torch.linalg.vector_norm(
        source64, dim=-1, keepdim=True
    )
    parent_row_norm = torch.linalg.vector_norm(
        parent64, dim=-1, keepdim=True
    )
    source_unit = source64 / source_norm.clamp_min(float(eps))
    parent_unit = parent64 / parent_row_norm.clamp_min(float(eps))
    row_nonzero = source_norm > 0.0
    alternative = torch.where(row_nonzero, source_unit, parent_unit)
    parent_norm = torch.linalg.vector_norm(
        parent64, dim=(-2, -1), keepdim=True
    )
    alternative_norm = torch.linalg.vector_norm(
        alternative, dim=(-2, -1), keepdim=True
    )
    alternative = alternative * (
        parent_norm / alternative_norm.clamp_min(float(eps))
    )
    valid = (
        torch.isfinite(alternative).all(dim=(-2, -1), keepdim=True)
        & (parent_norm > 0.0)
        & (alternative_norm > 0.0)
    )
    return torch.where(valid, alternative, parent64)


def gated_response_homotopy(
    parent: torch.Tensor,
    alternative: torch.Tensor,
    alignment: torch.Tensor,
    active: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Blend equal-budget directions using the canonical response angle."""

    if parent.shape != alternative.shape or parent.ndim != 3:
        raise RuntimeError("response homotopy direction inventory changed")
    if alignment.shape != (parent.shape[0],) or active.shape != alignment.shape:
        raise RuntimeError("response homotopy coordinate inventory changed")
    if active.dtype != torch.bool or float(eps) != 1.0e-8:
        raise RuntimeError("response homotopy gate or epsilon changed")
    parent64 = parent.float()
    alternative64 = alternative.float()
    c = alignment.float().clamp(0.0, 1.0)[:, None, None]
    enabled = active[:, None, None]
    parent_amplitude = torch.sqrt(c)
    alternative_amplitude = torch.sqrt((1.0 - c).clamp_min(0.0))
    mixed = parent_amplitude * parent64 + alternative_amplitude * alternative64
    parent_norm = torch.linalg.vector_norm(
        parent64, dim=(-2, -1), keepdim=True
    )
    mixed_norm = torch.linalg.vector_norm(mixed, dim=(-2, -1), keepdim=True)
    mixed = mixed * (parent_norm / mixed_norm.clamp_min(float(eps)))
    use_parent = (~enabled) | (c == 1.0) | (~torch.isfinite(mixed))
    return torch.where(use_parent, parent64, mixed)


def lagged_loss_gate(
    current_scores: torch.Tensor,
    previous_scores: torch.Tensor | None,
    *,
    decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use only the previous EMA to admit a loss-favored alternative."""

    if current_scores.ndim != 3 or current_scores.shape[-1] != 2:
        raise RuntimeError("response-alignment loss score inventory changed")
    if not 0.0 < float(decay) < 1.0:
        raise ValueError("response-alignment decay must lie in (0,1)")
    torch._assert_async(torch.isfinite(current_scores).all())
    if previous_scores is None:
        return (
            torch.zeros(
                current_scores.shape[:-1],
                dtype=torch.bool,
                device=current_scores.device,
            ),
            current_scores.detach().clone(),
        )
    if previous_scores.shape != current_scores.shape:
        raise RuntimeError("response-alignment persistent score inventory changed")
    active = previous_scores[..., 1] > previous_scores[..., 0]
    updated = previous_scores.detach().clone().mul_(float(decay)).add_(
        current_scores, alpha=1.0 - float(decay)
    )
    return active, updated


def response_alignment_state_elements(*, layers: int, groups: int) -> int:
    """Persistent method state excluding the locked Muon momentum."""

    if int(layers) <= 0 or int(groups) <= 0:
        raise ValueError("response-alignment dimensions must be positive")
    # Ten frozen P5/Q4 coefficients plus two loss scores per coordinate.
    return 12 * int(layers) * int(groups) + 1


__all__ = (
    "gated_response_homotopy",
    "grouped_response_statistics",
    "lagged_loss_gate",
    "response_alignment_from_statistics",
    "response_alignment_state_elements",
    "row_normalized_equal_budget",
)
