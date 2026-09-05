"""Opt-in direct-score execution for the exact archived R01 optimizer.

The scientific equation is unchanged.  The archived implementation first
materializes every residual-width tangent image and then contracts it with the
loss cotangent.  This implementation applies the adjoint first and returns the
same scalar group scores directly.  The changed GEMM/reduction association
makes this a numerical-equivalence path, never a bitwise claim.
"""

from __future__ import annotations

import torch

from .rlb_r01_9150_archive import (
    R01Core as _ExactR01Core,
    R01Optimizer as _ExactR01Optimizer,
    R02AttentionOptimizer,
    verify_r01_9150_archive,
)


ARCHIVE_CERTIFICATE = verify_r01_9150_archive()
DIRECT_SCORE_ID = "r01_9150_direct_loss_score_contraction_v1"


def direct_group_tangent_scores(
    inputs: torch.Tensor,
    features: torch.Tensor,
    cotangents: torch.Tensor,
    incoming_direction: torch.Tensor,
    outgoing_direction_transpose: torch.Tensor,
    outgoing_weights: torch.Tensor | None,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    groups: int,
    width: int,
    cached_response_adjoint: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[layer, sample, group]`` scores without output JVPs."""

    if (
        inputs.ndim != 3
        or features.ndim != 3
        or cotangents.ndim != 3
        or incoming_direction.ndim != 3
        or outgoing_direction_transpose.shape != incoming_direction.shape
    ):
        raise RuntimeError("R01 direct-score tensor rank changed")
    layers, samples, residual = inputs.shape
    hidden = int(groups) * int(width)
    if (
        features.shape != (layers, samples, hidden)
        or cotangents.shape != (layers, samples, residual)
        or incoming_direction.shape != (layers, hidden, residual)
    ):
        raise RuntimeError("R01 direct-score inventory changed")
    u, derivative, radial = factors
    expected_factors = (layers, samples, int(groups), int(width))
    if any(value.shape != expected_factors for value in (u, derivative, radial)):
        raise RuntimeError("R01 direct-score factor inventory changed")

    response_adjoint = cached_response_adjoint
    if response_adjoint is None:
        if (
            outgoing_weights is None
            or outgoing_weights.ndim != 4
            or outgoing_weights.shape
            != (layers, int(groups), int(width), residual)
        ):
            raise RuntimeError("R01 direct-score outgoing inventory changed")
        response_cotangent = torch.einsum(
            "lnd,lgwd->lngw", cotangents, outgoing_weights
        )
        response_adjoint = (
            derivative * response_cotangent
            + u * (radial * response_cotangent).mean(dim=-1, keepdim=True)
        )
    elif response_adjoint.shape != expected_factors:
        raise RuntimeError("R01 cached response adjoint changed")

    perturbation = torch.bmm(
        inputs, incoming_direction.transpose(-2, -1)
    ).view(expected_factors)
    incoming_score = (perturbation * response_adjoint).sum(dim=-1)
    outgoing_projection = torch.bmm(
        cotangents, outgoing_direction_transpose.transpose(-2, -1)
    ).view(expected_factors)
    outgoing_score = (
        features.view(expected_factors) * outgoing_projection
    ).sum(dim=-1)
    scores = incoming_score + outgoing_score
    torch._assert_async(torch.isfinite(scores).all())
    return scores, response_adjoint


class R01DirectScoreOptimizer(_ExactR01Optimizer):
    """Exact R01 state/equations with one numerical direct-score contraction."""

    execution_variant = DIRECT_SCORE_ID

    def __init__(self, *args, **kwargs):
        self._direct_score_response_adjoint = None
        self._direct_score_calls = 0
        self.direct_score_last_call_count = 0
        self.direct_score_last_materialized_image_bytes_avoided = 0
        super().__init__(*args, **kwargs)

    def _group_tangent_images(
        self,
        inputs,
        preactivations,
        features,
        incoming_direction,
        outgoing_direction_transpose,
        *,
        factors,
    ):
        del preactivations
        cotangents = self._r09_loss_cotangents
        if cotangents is None:
            raise RuntimeError("R01 direct-score path lacks loss cotangents")
        layers = len(self.pairs)
        outgoing_weights = None
        if self._direct_score_response_adjoint is None:
            # Only the first call needs W_out to form J^T W_out^T e.  The
            # endpoint and decay calls share the same current parameters, loss
            # cotangent, samples, and response factors, so rebuilding the full
            # stacked W_out on the second call would be a dead model-sized copy.
            outgoing_weights = torch.stack(self.outgoing).float().view(
                layers, self.external_width, self.groups, self.width
            ).permute(0, 2, 3, 1)
        scores, response_adjoint = direct_group_tangent_scores(
            inputs,
            features,
            cotangents,
            incoming_direction,
            outgoing_direction_transpose,
            outgoing_weights,
            factors,
            groups=self.groups,
            width=self.width,
            cached_response_adjoint=self._direct_score_response_adjoint,
        )
        self._direct_score_response_adjoint = response_adjoint
        self._direct_score_calls += 1
        # The archived image has one residual-width value per score.
        self.direct_score_last_materialized_image_bytes_avoided += (
            int(scores.numel()) * int(self.external_width) * scores.element_size()
        )
        return scores[..., None]

    @staticmethod
    def _reduce_global_loss_metric(images, cotangents, group_decay_images):
        del cotangents
        if images.shape[-1] != 1 or group_decay_images.shape != images.shape:
            raise RuntimeError("R01 direct-score reducer received full images")
        metric_cotangents = torch.ones_like(images[:, :, 0, :])
        return _ExactR01Core._reduce_global_loss_metric(
            images, metric_cotangents, group_decay_images
        )

    @torch.no_grad()
    def step(self, closure=None):
        self._direct_score_response_adjoint = None
        self._direct_score_calls = 0
        self.direct_score_last_materialized_image_bytes_avoided = 0
        try:
            result = super().step(closure)
            if self._direct_score_calls != 2:
                raise RuntimeError("R01 direct-score call inventory changed")
            self.direct_score_last_call_count = self._direct_score_calls
            return result
        finally:
            self._direct_score_response_adjoint = None


__all__ = (
    "ARCHIVE_CERTIFICATE",
    "DIRECT_SCORE_ID",
    "R01DirectScoreOptimizer",
    "R02AttentionOptimizer",
    "direct_group_tangent_scores",
)
