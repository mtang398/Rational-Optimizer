"""Full-Muon integration of the Global-RLB basis-cotangent trust method."""

from __future__ import annotations

import math

import torch

from .rlb_basis_cotangent_trust import (
    FAMILY_ID,
    RATIONAL_CHANNELS,
    basis_cotangent_trust_allocate,
)


_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_MUON_EPS = 1.0e-7


def _zeropower_via_newton_schulz(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Literal torch.optim.Muon NS path for the pinned campaign runtime."""

    if matrix.ndim != 2 or steps != 5:
        raise RuntimeError("basis-cotangent Muon requires a 2D NS5 input")
    a, b, c = _NS_COEFFICIENTS
    value = matrix.bfloat16()
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.T
    value.div_(value.norm().clamp(min=_MUON_EPS))
    for _ in range(steps):
        gram = value @ value.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        value = torch.addmm(value, gram_update, value, beta=a)
    return value.T if transposed else value


def _match_rms_adamw_adjustment(shape: torch.Size) -> float:
    if len(shape) != 2:
        raise RuntimeError("basis-cotangent Muon adjustment requires a matrix")
    return 0.2 * math.sqrt(float(max(shape)))


class BasisCotangentTrustMuonOptimizer(torch.optim.Optimizer):
    """Exact full-matrix Muon directions plus a rank-ten global allocation.

    The optimizer owns only the Global-RLB incoming and outgoing matrices.
    Numerator and denominator parameters remain under the ordinary matched
    AdamW child; their clipped gradients are read as the loss-aware sketch.
    """

    def __init__(
        self,
        pairs,
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        ns_steps: int,
        beta2: float,
        eps: float,
    ):
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("basis-cotangent Muon requires Global-RLB layers")
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        if self.momentum != 0.95 or self.ns_steps != 5:
            raise ValueError("basis-cotangent Muon requires matched momentum .95 and NS5")
        if float(beta2) != 0.95 or float(eps) != 1.0e-8:
            raise ValueError("basis-cotangent Muon requires matched Adam beta2/eps")

        first = self.pairs[0]
        self.groups = int(first["groups"])
        self.hidden = int(first["hidden_dim"])
        if self.hidden % self.groups:
            raise ValueError("Global-RLB hidden width is not group divisible")
        self.group_width = self.hidden // self.groups
        self.external_width = int(first["in_weight"].shape[1])
        parameters = []
        seen = set()
        for pair in self.pairs:
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            numerator = pair["numerator"]
            denominator = pair["denominator"]
            if (
                int(pair["groups"]) != self.groups
                or int(pair["hidden_dim"]) != self.hidden
                or incoming.shape != (self.hidden, self.external_width)
                or outgoing.shape != (self.external_width, self.hidden)
                or numerator.shape != (self.groups, 6)
                or denominator.shape != (self.groups, 4)
            ):
                raise ValueError("basis-cotangent Global-RLB layer inventory changed")
            for parameter in (incoming, outgoing):
                if id(parameter) in seen:
                    raise ValueError("basis-cotangent matrix ownership overlaps")
                seen.add(id(parameter))
                parameters.append(parameter)

        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "lr_scale": 1.0,
            "basis_cotangent_family_id": FAMILY_ID,
            "muon_momentum": self.momentum,
            "muon_ns_steps": self.ns_steps,
            "muon_adjust_lr_fn": "match_rms_adamw",
            "muon_eps": _MUON_EPS,
        }
        super().__init__([{"params": parameters}], defaults)
        self._clip_factor = None
        self._capture_telemetry_next_step = False
        self._last_telemetry = {}

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "matrix_role_lr_scale": 1.0,
            "basis_cotangent_sketch_lr_scale": 1.0,
            "equality_budget_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def record_realized_clipping(self, preclip_norm, max_norm):
        if self._clip_factor is not None:
            raise RuntimeError("basis-cotangent Muon observed multiple clipping calls")
        value = float(preclip_norm)
        maximum = float(max_norm)
        if not math.isfinite(value) or value < 0.0 or maximum != 1.0:
            raise RuntimeError("basis-cotangent Muon received invalid clipping")
        self._clip_factor = min(1.0, maximum / (value + 1.0e-6))

    def set_telemetry_capture(self, enabled=True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

    def _nesterov(self, parameter: torch.Tensor) -> torch.Tensor:
        if parameter.grad is None:
            raise RuntimeError("basis-cotangent Muon matrix gradient is missing")
        state = self.state[parameter]
        buffer = state.get("momentum_buffer")
        if buffer is None:
            buffer = torch.zeros_like(
                parameter.grad, memory_format=torch.preserve_format
            )
            state["momentum_buffer"] = buffer
        buffer.lerp_(parameter.grad, 1.0 - self.momentum)
        return parameter.grad.lerp(buffer, self.momentum)

    @torch.no_grad()
    def step(self, closure=None):
        if self._clip_factor is None:
            raise RuntimeError("basis-cotangent Muon did not receive realized clipping")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("basis-cotangent Muon refuses nonunit LR scale")

        incoming_directions = []
        outgoing_directions = []
        incoming_adjustments = []
        outgoing_adjustments = []
        exact_incoming = []
        exact_outgoing = []
        momentum_incoming = []
        momentum_outgoing = []
        budget_weights = []
        coefficient_cotangents = []
        for pair in self.pairs:
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            numerator = pair["numerator"]
            denominator = pair["denominator"]
            if (
                incoming.grad is None
                or outgoing.grad is None
                or numerator.grad is None
                or denominator.grad is None
            ):
                raise RuntimeError("basis-cotangent gradient inventory is incomplete")

            incoming_momentum = self._nesterov(incoming)
            outgoing_momentum = self._nesterov(outgoing)
            incoming_direction = _zeropower_via_newton_schulz(
                incoming_momentum, self.ns_steps
            )
            outgoing_direction = _zeropower_via_newton_schulz(
                outgoing_momentum, self.ns_steps
            )
            incoming_adjustment = _match_rms_adamw_adjustment(incoming.shape)
            outgoing_adjustment = _match_rms_adamw_adjustment(outgoing.shape)

            incoming_view = incoming_direction.view(
                self.groups, self.group_width, self.external_width
            )
            outgoing_view = (
                outgoing_direction.view(
                    self.external_width, self.groups, self.group_width
                )
                .permute(1, 2, 0)
            )
            incoming_gradient = incoming.grad.detach().view_as(incoming_view)
            outgoing_gradient = (
                outgoing.grad.detach()
                .view(self.external_width, self.groups, self.group_width)
                .permute(1, 2, 0)
            )
            incoming_momentum_view = incoming_momentum.view_as(incoming_view)
            outgoing_momentum_view = (
                outgoing_momentum.view(
                    self.external_width, self.groups, self.group_width
                )
                .permute(1, 2, 0)
            )
            # PyTorch Muon applies the RMS adjustment through the FP32 alpha
            # argument of ``add_``.  Keep the stored BF16 orthogonal factor
            # unscaled so an all-one fallback is bitwise identical.
            incoming_direction_float = incoming_view.float() * incoming_adjustment
            outgoing_direction_float = outgoing_view.float() * outgoing_adjustment
            exact_incoming.append(
                (incoming_gradient.float() * incoming_direction_float).sum(
                    dim=(-2, -1)
                )
            )
            exact_outgoing.append(
                (outgoing_gradient.float() * outgoing_direction_float).sum(
                    dim=(-2, -1)
                )
            )
            momentum_incoming.append(
                (incoming_momentum_view.float() * incoming_direction_float).sum(
                    dim=(-2, -1)
                )
            )
            momentum_outgoing.append(
                (outgoing_momentum_view.float() * outgoing_direction_float).sum(
                    dim=(-2, -1)
                )
            )
            budget_weights.append(
                incoming_direction_float.square().sum(dim=(-2, -1))
                + outgoing_direction_float.square().sum(dim=(-2, -1))
            )
            coefficient_cotangents.append(
                torch.cat(
                    (numerator.grad.detach().float(), denominator.grad.detach().float()),
                    dim=-1,
                )
            )
            incoming_directions.append(incoming_direction)
            outgoing_directions.append(outgoing_direction)
            incoming_adjustments.append(incoming_adjustment)
            outgoing_adjustments.append(outgoing_adjustment)

        exact_by_role = torch.stack(
            (torch.stack(exact_incoming), torch.stack(exact_outgoing))
        )
        momentum_by_role = torch.stack(
            (torch.stack(momentum_incoming), torch.stack(momentum_outgoing))
        )
        weights = torch.stack(budget_weights)
        cotangents = torch.stack(coefficient_cotangents)
        if cotangents.shape[-1] != RATIONAL_CHANNELS:
            raise RuntimeError("basis-cotangent sketch width changed")
        selection = basis_cotangent_trust_allocate(
            exact_by_role,
            momentum_by_role,
            weights,
            cotangents,
            eta=lr,
        )
        coefficients = selection.coefficients

        for index, pair in enumerate(self.pairs):
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            incoming.mul_(1.0 - lr * weight_decay)
            outgoing.mul_(1.0 - lr * weight_decay)
            incoming_direction = incoming_directions[index]
            outgoing_direction = outgoing_directions[index]
            incoming_direction.view(
                self.groups, self.group_width, self.external_width
            ).mul_(coefficients[index, :, None, None].to(incoming_direction.dtype))
            (
                outgoing_direction.view(
                    self.external_width, self.groups, self.group_width
                )
                .permute(1, 2, 0)
                .mul_(coefficients[index, :, None, None].to(outgoing_direction.dtype))
            )
            incoming.add_(
                incoming_direction.to(incoming.dtype),
                alpha=-lr * incoming_adjustments[index],
            )
            outgoing.add_(
                outgoing_direction.to(outgoing.dtype),
                alpha=-lr * outgoing_adjustments[index],
            )

        if self._capture_telemetry_next_step:
            flat = coefficients.reshape(-1)
            self._last_telemetry = {
                "basis_cotangent_family_id": FAMILY_ID,
                "basis_cotangent_owner_count": 0,
                "basis_cotangent_global_rows": RATIONAL_CHANNELS,
                "basis_cotangent_coordinate_count": len(self.pairs) * self.groups,
                "basis_cotangent_state_depends_on_total_tokens": 0,
                "basis_cotangent_dense_lg_metric_elements": 0,
                "basis_cotangent_selected_update_elements_published": 0,
                "basis_cotangent_transaction_accepted": int(
                    selection.accepted.item()
                ),
                "basis_cotangent_rank": int(selection.rank.item()),
                "basis_cotangent_budget_residual": float(
                    selection.budget_residual.item()
                ),
                "basis_cotangent_parent_score": float(selection.parent_score.item()),
                "basis_cotangent_candidate_score": float(
                    selection.candidate_score.item()
                ),
                "basis_cotangent_exact_descent": float(
                    selection.exact_descent.item()
                ),
                "basis_cotangent_momentum_descent": float(
                    selection.momentum_descent.item()
                ),
                "basis_cotangent_coefficient_min": float(flat.amin().item()),
                "basis_cotangent_coefficient_median": float(flat.median().item()),
                "basis_cotangent_coefficient_max": float(flat.amax().item()),
                "basis_cotangent_sketch_row_norm_min": float(
                    selection.sketch_row_norm_min.item()
                ),
                "basis_cotangent_sketch_row_norm_max": float(
                    selection.sketch_row_norm_max.item()
                ),
                "basis_cotangent_realized_clip_factor": float(self._clip_factor),
            }
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss


__all__ = ("BasisCotangentTrustMuonOptimizer",)
