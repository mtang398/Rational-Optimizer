"""Diagonal Global-RLB response-Fisher factors for matrix-sign updates.

The helpers in this module deliberately operate on fixed activation-position
summaries.  Their storage is linear in the logical matrix dimensions and is
independent of the total number of activation positions seen by training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResponseFisherDiagonalSums:
    """Unnormalised fixed-probe diagonal curvature summaries."""

    incoming: torch.Tensor
    outgoing: torch.Tensor
    probe_count: int


def response_fisher_diagonal_sums(
    inputs: torch.Tensor,
    features: torch.Tensor,
    cotangents: torch.Tensor,
    response_adjoint: torch.Tensor,
) -> ResponseFisherDiagonalSums:
    """Form groupwise empirical-Fisher diagonals without dense covariance.

    ``response_adjoint`` is the exact Global-RLB pullback through the
    normalised rational function.  It weights input-coordinate energy for the
    incoming matrix.  Output-cotangent energy weights rational-feature energy
    for the outgoing matrix.
    """

    if inputs.ndim != 2 or cotangents.shape != inputs.shape:
        raise RuntimeError("response-Fisher input/cotangent inventory changed")
    if response_adjoint.ndim != 3:
        raise RuntimeError("response-Fisher adjoint rank changed")
    probes, groups, width = response_adjoint.shape
    if (
        inputs.shape[0] != probes
        or features.shape != (probes, groups * width)
    ):
        raise RuntimeError("response-Fisher probe inventory changed")
    if probes == 0:
        return ResponseFisherDiagonalSums(
            incoming=torch.zeros(
                groups, inputs.shape[1], dtype=inputs.dtype, device=inputs.device
            ),
            outgoing=torch.zeros(
                groups, width, dtype=features.dtype, device=features.device
            ),
            probe_count=0,
        )

    incoming_loss_power = response_adjoint.square().mean(dim=-1)
    incoming = incoming_loss_power.transpose(0, 1) @ inputs.square()
    output_loss_power = cotangents.square().mean(dim=-1)
    outgoing = (
        features.view(probes, groups, width).square()
        * output_loss_power[:, None, None]
    ).sum(dim=0)
    torch._assert_async(torch.isfinite(incoming).all())
    torch._assert_async(torch.isfinite(outgoing).all())
    return ResponseFisherDiagonalSums(
        incoming=incoming,
        outgoing=outgoing,
        probe_count=int(probes),
    )


def lagged_exponential_diagonal(
    current: torch.Tensor,
    previous: torch.Tensor | None,
    *,
    decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the lagged factor used now and the updated persistent factor.

    On the first step the current fixed-probe statistic initializes both
    values.  Afterwards the direction uses only the preceding estimate; the
    current batch enters the state for the next step.  This prevents a
    same-batch curvature estimate from selecting its own update direction.
    """

    if current.ndim != 2:
        raise RuntimeError("response-Fisher current diagonal changed")
    torch._assert_async(torch.isfinite(current).all())
    if not 0.0 < float(decay) < 1.0:
        raise ValueError("response-Fisher decay must lie strictly inside (0,1)")
    if previous is None:
        initialized = current.detach().clone()
        return initialized, initialized
    if previous.shape != current.shape or previous.device != current.device:
        raise RuntimeError("response-Fisher persistent diagonal changed")
    lagged = previous
    updated = previous.detach().clone().mul_(float(decay)).add_(
        current, alpha=1.0 - float(decay)
    )
    return lagged, updated


def inverse_root_diagonal_scale(
    diagonal: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Parameter-free, mean-one-reference inverse-root scaling.

    The scale is invariant to a common rescaling of the loss cotangent.  The
    only numerical stabilizer is the already locked optimizer epsilon.
    """

    if diagonal.ndim != 2:
        raise RuntimeError("response-Fisher scale diagonal changed")
    torch._assert_async(torch.isfinite(diagonal).all())
    if float(eps) != 1.0e-8:
        raise ValueError("response-Fisher uses the locked optimizer epsilon")
    reference = diagonal.mean()
    safe_reference = reference.clamp_min(torch.finfo(diagonal.dtype).tiny)
    normalized = diagonal / safe_reference
    scale = torch.rsqrt(normalized + float(eps))
    return torch.where(reference > 0.0, scale, torch.ones_like(scale))


def method_state_elements(
    *,
    layers: int,
    groups: int,
    width: int,
    model_width: int,
) -> int:
    """Closed-form persistent method state, excluding locked Muon momentum."""

    values = (int(layers), int(groups), int(width), int(model_width))
    if any(value <= 0 for value in values):
        raise ValueError("response-Fisher dimensions must be positive")
    return int(layers) * int(groups) * (int(model_width) + int(width)) + 1


__all__ = (
    "ResponseFisherDiagonalSums",
    "inverse_root_diagonal_scale",
    "lagged_exponential_diagonal",
    "method_state_elements",
    "response_fisher_diagonal_sums",
)
