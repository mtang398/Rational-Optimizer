"""Fixed-ten downstream-loss image sketch for full Global-RLB Muon.

The P5/Q4 architecture fixes ten globally distributed activation-position
probes.  Their exact downstream-loss actions on the current paired NS5 Muon
group directions form a ``10 x (L G)`` score lattice.  The selected update is
an equality-energy row-space transaction; no activation-position-dependent
state, coordinate Fisher, complete-layer owner, or selected matrix update is
communicated.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from .rlb_basis_cotangent_trust_muon import (
    _match_rms_adamw_adjustment,
    _zeropower_via_newton_schulz,
)
from .rlb_fixed_global_probe_layout import (
    evenly_spaced_indices,
    fixed_global_probe_layout,
)
from .rlb_fixed_probe_transaction import replicated_fixed_probe_transaction


FAMILY_ID = "ten_probe_loss_image_muon_v1"
GLOBAL_PROBE_COUNT = 10
EXPECTED_MICROBATCHES = 4


def _version_a_factors(
    preactivation: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    groups: int,
    width: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact normalized P5/Q4 derivative factors for one layer's probes."""

    if preactivation.ndim != 2 or preactivation.shape[1] != groups * width:
        raise RuntimeError("ten-probe preactivation inventory changed")
    if numerator.shape != (groups, 6) or denominator.shape != (groups, 4):
        raise RuntimeError("ten-probe coefficient inventory changed")
    value = preactivation.float().view(-1, groups, width)
    rms = torch.sqrt(value.square().mean(dim=-1, keepdim=True) + float(eps))
    unit = value / rms
    unit2 = unit.square()
    unit3 = unit2 * unit
    unit4 = unit2.square()
    unit5 = unit4 * unit
    absolute = unit.abs()
    num = numerator.detach().float()[None, :, None, :]
    den = denominator.detach().float().abs()[None, :, None, :]
    polynomial = (
        num[..., 0]
        + num[..., 1] * unit
        + num[..., 2] * unit2
        + num[..., 3] * unit3
        + num[..., 4] * unit4
        + num[..., 5] * unit5
    )
    polynomial_derivative = (
        num[..., 1]
        + 2.0 * num[..., 2] * unit
        + 3.0 * num[..., 3] * unit2
        + 4.0 * num[..., 4] * unit3
        + 5.0 * num[..., 5] * unit4
    )
    quotient = (
        1.0
        + den[..., 0] * absolute
        + den[..., 1] * unit2
        + den[..., 2] * absolute * unit2
        + den[..., 3] * unit4
    )
    quotient_derivative = (
        den[..., 0] * torch.sign(unit)
        + 2.0 * den[..., 1] * unit
        + 3.0 * den[..., 2] * unit * absolute
        + 4.0 * den[..., 3] * unit3
    )
    function = polynomial / quotient
    derivative = (
        polynomial_derivative * quotient - polynomial * quotient_derivative
    ) / quotient.square()
    radial = function - unit * derivative
    return unit, derivative, radial


def _one_layer_group_scores(
    inputs: torch.Tensor,
    preactivations: torch.Tensor,
    features: torch.Tensor,
    cotangents: torch.Tensor,
    incoming_direction: torch.Tensor,
    outgoing_direction_transpose: torch.Tensor,
    outgoing_weight: torch.Tensor,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    groups: int,
    width: int,
    cached_response_adjoint: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contract exact tangent images without materializing residual images."""

    probes, residual = inputs.shape
    hidden = int(groups) * int(width)
    if (
        preactivations.shape != (probes, hidden)
        or features.shape != (probes, hidden)
        or cotangents.shape != (probes, residual)
        or incoming_direction.shape != (hidden, residual)
        or outgoing_direction_transpose.shape != (hidden, residual)
        or outgoing_weight.shape != (residual, hidden)
    ):
        raise RuntimeError("ten-probe direct-score inventory changed")
    unit, derivative, radial = factors
    expected = (probes, int(groups), int(width))
    if any(value.shape != expected for value in (unit, derivative, radial)):
        raise RuntimeError("ten-probe factor inventory changed")

    response_adjoint = cached_response_adjoint
    if response_adjoint is None:
        response_cotangent = (cotangents.float() @ outgoing_weight.float()).view(
            expected
        )
        response_adjoint = (
            derivative * response_cotangent
            + unit * (radial * response_cotangent).mean(dim=-1, keepdim=True)
        )
    elif response_adjoint.shape != expected:
        raise RuntimeError("ten-probe cached adjoint inventory changed")

    perturbation = (inputs.float() @ incoming_direction.float().T).view(expected)
    incoming_score = (perturbation * response_adjoint).sum(dim=-1)
    outgoing_projection = (
        cotangents.float() @ outgoing_direction_transpose.float().T
    ).view(expected)
    outgoing_score = (
        features.float().view(expected) * outgoing_projection
    ).sum(dim=-1)
    score = incoming_score + outgoing_score
    if not bool(torch.isfinite(score).all()):
        raise RuntimeError("ten-probe loss score is nonfinite")
    return score, response_adjoint


class TenProbeLossImageMuonOptimizer(torch.optim.Optimizer):
    """Full NS5 Muon with a globally fixed ten-row loss-image transaction."""

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
        loss_probe_group=None,
    ):
        self.pairs = list(pairs)
        if not self.pairs:
            raise ValueError("ten-probe Muon requires Global-RLB layers")
        self.momentum = float(momentum)
        self.ns_steps = int(ns_steps)
        if self.momentum != 0.95 or self.ns_steps != 5:
            raise ValueError("ten-probe Muon requires matched momentum .95 and NS5")
        if float(beta2) != 0.95 or float(eps) != 1.0e-8:
            raise ValueError("ten-probe Muon requires matched Adam beta2/eps")
        self.loss_probe_group = loss_probe_group
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank(group=loss_probe_group)
            world = dist.get_world_size(group=loss_probe_group)
        else:
            rank, world = 0, 1
        self.probe_layout = fixed_global_probe_layout(
            GLOBAL_PROBE_COUNT, rank, world
        )
        self.capture_rows = (
            self.probe_layout.local_probe_count + EXPECTED_MICROBATCHES - 1
        ) // EXPECTED_MICROBATCHES

        first = self.pairs[0]
        self.groups = int(first["groups"])
        self.hidden = int(first["hidden_dim"])
        self.external = int(first["in_weight"].shape[1])
        self.rlb_eps = float(first["eps"])
        if self.hidden % self.groups:
            raise ValueError("ten-probe hidden width is not group divisible")
        self.width = self.hidden // self.groups
        self._pending_inputs = [None for _ in self.pairs]
        self._functional_records = [[] for _ in self.pairs]
        self._cotangent_records = [[] for _ in self.pairs]
        self._hook_handles = []

        parameters = []
        seen = set()
        for layer, pair in enumerate(self.pairs):
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            if (
                int(pair["groups"]) != self.groups
                or int(pair["hidden_dim"]) != self.hidden
                or float(pair["eps"]) != self.rlb_eps
                or incoming.shape != (self.hidden, self.external)
                or outgoing.shape != (self.external, self.hidden)
                or pair["numerator"].shape != (self.groups, 6)
                or pair["denominator"].shape != (self.groups, 4)
            ):
                raise ValueError("ten-probe Global-RLB inventory changed")
            for parameter in (incoming, outgoing):
                if id(parameter) in seen:
                    raise ValueError("ten-probe matrix ownership overlaps")
                seen.add(id(parameter))
                parameters.append(parameter)
            self._hook_handles.extend((
                pair["mlp"].register_forward_pre_hook(
                    self._make_input_hook(layer)
                ),
                pair["module"].register_forward_hook(
                    self._make_feature_hook(layer)
                ),
                pair["mlp"].register_forward_hook(
                    self._make_cotangent_hook(layer)
                ),
            ))

        defaults = {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "lr_scale": 1.0,
            "ten_probe_family_id": FAMILY_ID,
            "muon_momentum": self.momentum,
            "muon_ns_steps": self.ns_steps,
            "muon_adjust_lr_fn": "match_rms_adamw",
        }
        super().__init__([{"params": parameters}], defaults)
        self._clip_factor = None
        self._capture_telemetry_next_step = False
        self._last_telemetry = {}

    def _sample_indices(self, rows: int, device: torch.device) -> torch.Tensor:
        return evenly_spaced_indices(
            int(rows), self.capture_rows, device=device
        )

    def _make_input_hook(self, layer: int):
        @torch.no_grad()
        def capture(module, inputs):
            if not module.training:
                return
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError("ten-probe MLP input hook changed")
            if self._pending_inputs[layer] is not None:
                raise RuntimeError("ten-probe input remained pending")
            flat = inputs[0].detach().reshape(-1, self.external)
            indices = self._sample_indices(flat.shape[0], flat.device)
            self._pending_inputs[layer] = (
                int(flat.shape[0]), indices, flat.index_select(0, indices).clone()
            )

        return capture

    def _make_feature_hook(self, layer: int):
        @torch.no_grad()
        def capture(module, inputs, output):
            if not module.training:
                return
            pending = self._pending_inputs[layer]
            if pending is None:
                raise RuntimeError("ten-probe feature lacks aligned input")
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
                raise RuntimeError("ten-probe activation hook changed")
            rows, indices, sampled_input = pending
            preactivation = inputs[0].detach().reshape(-1, self.hidden)
            feature = output.detach().reshape(-1, self.hidden)
            if preactivation.shape[0] != rows or feature.shape[0] != rows:
                raise RuntimeError("ten-probe x/z/h rows differ")
            self._functional_records[layer].append((
                sampled_input,
                preactivation.index_select(0, indices).clone(),
                feature.index_select(0, indices).clone(),
            ))
            self._pending_inputs[layer] = None

        return capture

    def _make_cotangent_hook(self, layer: int):
        def capture(module, _inputs, output):
            if not module.training:
                return
            if not torch.is_tensor(output) or not output.requires_grad:
                raise RuntimeError("ten-probe MLP output hook changed")
            flat = output.reshape(-1, self.external)
            rows = int(flat.shape[0])
            indices = self._sample_indices(rows, flat.device)

            def capture_gradient(cotangent):
                with torch.no_grad():
                    value = cotangent.detach().reshape(-1, self.external)
                    if value.shape[0] != rows:
                        raise RuntimeError("ten-probe output/cotangent rows differ")
                    self._cotangent_records[layer].append((
                        value.index_select(0, indices).clone(), rows
                    ))
                return cotangent

            output.register_hook(capture_gradient)

        return capture

    def _consume_probes(self):
        if self._clip_factor is None:
            raise RuntimeError("ten-probe Muon lacks realized clipping")
        packets = []
        for layer in range(len(self.pairs)):
            if self._pending_inputs[layer] is not None:
                raise RuntimeError("ten-probe input remained unmatched")
            records = self._functional_records[layer]
            cotangents = self._cotangent_records[layer]
            self._functional_records[layer] = []
            self._cotangent_records[layer] = []
            if (
                len(records) != EXPECTED_MICROBATCHES
                or len(cotangents) != EXPECTED_MICROBATCHES
            ):
                raise RuntimeError("ten-probe microbatch inventory changed")
            inputs = torch.cat([record[0] for record in records], dim=0)
            preactivations = torch.cat([record[1] for record in records], dim=0)
            features = torch.cat([record[2] for record in records], dim=0)
            scaled_cotangents = torch.cat([
                value.float() * float(rows * EXPECTED_MICROBATCHES)
                for value, rows in cotangents
            ], dim=0) * float(self._clip_factor)
            captured = self.capture_rows * EXPECTED_MICROBATCHES
            if (
                inputs.shape != (captured, self.external)
                or preactivations.shape != (captured, self.hidden)
                or features.shape != (captured, self.hidden)
                or scaled_cotangents.shape != (captured, self.external)
            ):
                raise RuntimeError("ten-probe captured packet changed")
            selected = evenly_spaced_indices(
                captured,
                self.probe_layout.local_probe_count,
                device=inputs.device,
            )
            packets.append((
                inputs.index_select(0, selected).float(),
                preactivations.index_select(0, selected).float(),
                features.index_select(0, selected).float(),
                scaled_cotangents.index_select(0, selected).float(),
            ))
        return packets

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "matrix_role_lr_scale": 1.0,
            "fixed_loss_probe_lr_scale": 1.0,
            "loss_image_transaction_lr_scale": 1.0,
            "equality_budget_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    def record_realized_clipping(self, preclip_norm, max_norm):
        if self._clip_factor is not None:
            raise RuntimeError("ten-probe Muon observed multiple clipping calls")
        value = float(preclip_norm)
        maximum = float(max_norm)
        if not math.isfinite(value) or value < 0.0 or maximum != 1.0:
            raise RuntimeError("ten-probe Muon received invalid clipping")
        self._clip_factor = min(1.0, maximum / (value + 1.0e-6))

    def set_telemetry_capture(self, enabled=True):
        self._capture_telemetry_next_step = bool(enabled)

    def telemetry(self):
        return dict(self._last_telemetry)

    def _nesterov(self, parameter: torch.Tensor) -> torch.Tensor:
        if parameter.grad is None:
            raise RuntimeError("ten-probe Muon matrix gradient is missing")
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
            raise RuntimeError("ten-probe Muon did not receive realized clipping")
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("ten-probe Muon refuses nonunit LR scale")
        packets = self._consume_probes()

        incoming_directions = []
        outgoing_directions = []
        incoming_adjustments = []
        outgoing_adjustments = []
        exact_incoming = []
        exact_outgoing = []
        momentum_incoming = []
        momentum_outgoing = []
        budget_weights = []
        for pair in self.pairs:
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
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
                self.groups, self.width, self.external
            )
            outgoing_view = outgoing_direction.view(
                self.external, self.groups, self.width
            ).permute(1, 2, 0)
            incoming_gradient = incoming.grad.detach().view_as(incoming_view)
            outgoing_gradient = outgoing.grad.detach().view(
                self.external, self.groups, self.width
            ).permute(1, 2, 0)
            incoming_momentum_view = incoming_momentum.view_as(incoming_view)
            outgoing_momentum_view = outgoing_momentum.view(
                self.external, self.groups, self.width
            ).permute(1, 2, 0)
            incoming_float = incoming_view.float() * incoming_adjustment
            outgoing_float = outgoing_view.float() * outgoing_adjustment
            exact_incoming.append(
                (incoming_gradient.float() * incoming_float).sum(dim=(-2, -1))
            )
            exact_outgoing.append(
                (outgoing_gradient.float() * outgoing_float).sum(dim=(-2, -1))
            )
            momentum_incoming.append(
                (incoming_momentum_view.float() * incoming_float).sum(dim=(-2, -1))
            )
            momentum_outgoing.append(
                (outgoing_momentum_view.float() * outgoing_float).sum(dim=(-2, -1))
            )
            budget_weights.append(
                incoming_float.square().sum(dim=(-2, -1))
                + outgoing_float.square().sum(dim=(-2, -1))
            )
            incoming_directions.append(incoming_direction)
            outgoing_directions.append(outgoing_direction)
            incoming_adjustments.append(incoming_adjustment)
            outgoing_adjustments.append(outgoing_adjustment)

        local_scores = []
        local_decay = None
        for layer, (pair, packet) in enumerate(zip(self.pairs, packets)):
            inputs, preactivations, features, cotangents = packet
            factors = _version_a_factors(
                preactivations,
                pair["numerator"],
                pair["denominator"],
                groups=self.groups,
                width=self.width,
                eps=self.rlb_eps,
            )
            incoming_direction = (
                incoming_directions[layer].float() * incoming_adjustments[layer]
            )
            outgoing_direction_transpose = (
                outgoing_directions[layer]
                .view(self.external, self.groups, self.width)
                .permute(1, 2, 0)
                .reshape(self.hidden, self.external)
                .float()
                * outgoing_adjustments[layer]
            )
            score, response_adjoint = _one_layer_group_scores(
                inputs,
                preactivations,
                features,
                cotangents,
                incoming_direction,
                outgoing_direction_transpose,
                pair["out_weight"],
                factors,
                groups=self.groups,
                width=self.width,
            )
            decay_score, _ = _one_layer_group_scores(
                inputs,
                preactivations,
                features,
                cotangents,
                pair["in_weight"].float() * weight_decay,
                pair["out_weight"].T.float() * weight_decay,
                pair["out_weight"],
                factors,
                groups=self.groups,
                width=self.width,
                cached_response_adjoint=response_adjoint,
            )
            local_scores.append(score)
            layer_decay = decay_score.sum(dim=-1)
            local_decay = (
                layer_decay if local_decay is None else local_decay + layer_decay
            )

        score_lattice = torch.stack(local_scores, dim=1).reshape(
            self.probe_layout.local_probe_count, len(self.pairs) * self.groups
        )
        if local_decay is None:
            raise RuntimeError("ten-probe decay action was not formed")
        exact_by_role = torch.stack((
            torch.stack(exact_incoming).reshape(-1),
            torch.stack(exact_outgoing).reshape(-1),
        ))
        momentum_by_role = torch.stack((
            torch.stack(momentum_incoming).reshape(-1),
            torch.stack(momentum_outgoing).reshape(-1),
        ))
        weights = torch.stack(budget_weights).reshape(-1)
        layer_ids = torch.arange(
            len(self.pairs), device=weights.device, dtype=torch.int64
        ).repeat_interleave(self.groups)
        selection = replicated_fixed_probe_transaction(
            score_lattice,
            local_decay,
            exact_by_role,
            momentum_by_role,
            weights,
            layer_ids,
            global_probe_count=GLOBAL_PROBE_COUNT,
            total_layers=len(self.pairs),
            eta=lr,
            rounds=64,
            group=self.loss_probe_group,
        )
        coefficients = selection.coefficients.view(len(self.pairs), self.groups)

        for layer, pair in enumerate(self.pairs):
            incoming = pair["in_weight"]
            outgoing = pair["out_weight"]
            incoming.mul_(1.0 - lr * weight_decay)
            outgoing.mul_(1.0 - lr * weight_decay)
            incoming_direction = incoming_directions[layer]
            outgoing_direction = outgoing_directions[layer]
            incoming_direction.view(
                self.groups, self.width, self.external
            ).mul_(coefficients[layer, :, None, None].to(incoming_direction.dtype))
            outgoing_direction.view(
                self.external, self.groups, self.width
            ).permute(1, 2, 0).mul_(
                coefficients[layer, :, None, None].to(outgoing_direction.dtype)
            )
            incoming.add_(
                incoming_direction.to(incoming.dtype),
                alpha=-lr * incoming_adjustments[layer],
            )
            outgoing.add_(
                outgoing_direction.to(outgoing.dtype),
                alpha=-lr * outgoing_adjustments[layer],
            )

        if self._capture_telemetry_next_step:
            flat = coefficients.reshape(-1)
            transaction = selection.sharded_result
            self._last_telemetry = {
                "ten_probe_family_id": FAMILY_ID,
                "ten_probe_owner_count": transaction.owner_count,
                "ten_probe_global_rows": GLOBAL_PROBE_COUNT,
                "ten_probe_local_rows": self.probe_layout.local_probe_count,
                "ten_probe_coordinate_count": len(self.pairs) * self.groups,
                "ten_probe_state_depends_on_total_tokens": 0,
                "ten_probe_dense_lg_metric_elements": transaction.dense_LG_by_LG_metric_elements,
                "ten_probe_selected_update_elements_published": transaction.selected_update_elements_published,
                "ten_probe_transaction_accepted": int(transaction.accepted.item()),
                "ten_probe_rank": int(transaction.rank.item()),
                "ten_probe_budget_residual": float(transaction.budget_residual.item()),
                "ten_probe_parent_score": float(transaction.parent_score.item()),
                "ten_probe_candidate_score": float(transaction.candidate_score.item()),
                "ten_probe_cross_layer_coupling_ratio": float(selection.cross_layer_coupling_ratio.item()),
                "ten_probe_coefficient_min": float(flat.amin().item()),
                "ten_probe_coefficient_median": float(flat.median().item()),
                "ten_probe_coefficient_max": float(flat.amax().item()),
                "ten_probe_realized_clip_factor": float(self._clip_factor),
            }
        self._capture_telemetry_next_step = False
        self._clip_factor = None
        return loss


__all__ = (
    "EXPECTED_MICROBATCHES",
    "FAMILY_ID",
    "GLOBAL_PROBE_COUNT",
    "TenProbeLossImageMuonOptimizer",
    "_one_layer_group_scores",
    "_version_a_factors",
)
