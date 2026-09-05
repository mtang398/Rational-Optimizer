"""Opaque R04 optimizer-family routing from current Global-RLB morphology.

The two RLB-adjacent matrices retain the complete positive R05-generation-one
parent.  For every learned rational group, exact current Jacobian and response
participation select continuously between that parent and the equal-budget
Frobenius steepest direction of the same unmodified Nesterov momentum.  Their
geometric mean makes the same family decision for QKV and attention output.

Every branch has the parent's Frobenius budget and positive momentum descent.
The scheduled LR, WD, momentum, NS5 recurrence, clipping, and shape calibration
are unchanged; no statistic is used as an LR or WD multiplier.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .rlb_group_muon_core import _batched_zero_power, _match_rms_adamw_scale
from .rlb_r04_core import R04AttentionCore
from .rlb_r07_core import R07RLBRouterCore


class R04RevisionRouterCore(R07RLBRouterCore):
    """R05 pair parent plus groupwise learned-RLB optimizer-family routing."""

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
        pairs = list(pairs)
        self._r04r_local_role_participation = [None for _ in pairs]
        self._r04r_role_participation = None
        self._r04r_attention_participation = None
        self._r04r_endpoint_role = 0
        self._r04r_endpoint_metadata = []
        super().__init__(
            pairs,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            ns_steps=ns_steps,
            beta2=beta2,
            eps=eps,
        )

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "groupwise_intrinsic_family_router_lr_scale": 1.0,
            "equal_budget_frobenius_family_lr_scale": 1.0,
            "attention_family_transport_lr_scale": 1.0,
        })
        return report

    def _consume_probe(self, layer_index):
        probe = super()._consume_probe(layer_index)
        z = probe.float().view(self.probe_count, self.groups, self.width)
        rms = torch.sqrt(z.square().mean(dim=-1, keepdim=True) + self.rlb_eps)
        u = z / rms
        pair = self.pairs[layer_index]
        function, derivative = self._evaluate_response(
            u, pair["numerator"], pair["denominator"]
        )
        incoming = self._jacobian_participation(u, function, derivative)
        energy = function.square().sum(dim=-1)
        fourth = function.pow(4).sum(dim=-1)
        tiny = torch.finfo(function.dtype).tiny
        outgoing = energy.square() / (
            float(self.width) * fourth.clamp_min(tiny)
        )
        valid = (
            torch.isfinite(incoming)
            & torch.isfinite(outgoing)
            & (incoming >= 0.0)
            & (incoming <= 1.0)
            & (outgoing >= 0.0)
            & (outgoing <= 1.0)
            & (energy > 0.0)
            & (fourth > 0.0)
        )
        torch._assert_async(valid.all())
        count = torch.full(
            (self.groups,),
            float(self.probe_count),
            device=u.device,
            dtype=u.dtype,
        )
        self._r04r_local_role_participation[layer_index] = torch.stack((
            incoming.sum(dim=0), outgoing.sum(dim=0), count,
        ))
        return probe

    def _consume_router_alignments(self):
        # This call forms the complete positive R05 parent and the independent
        # aggregate R07 certificate.  R04 returns its literal relative
        # alignments so intrinsic morphology cannot alter the inherited parent.
        relative = super()._consume_router_alignments()
        if any(value is None for value in self._r04r_local_role_participation):
            raise RuntimeError("R04 did not form every groupwise intrinsic sensor")
        statistics = torch.stack(self._r04r_local_role_participation)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        role = statistics[:, :2] / statistics[:, 2:3].clamp_min(1.0)
        valid = torch.isfinite(role) & (role >= 0.0) & (role <= 1.0)
        torch._assert_async(valid.all())
        incoming = role[:, 0].mean(dim=-1)
        outgoing = role[:, 1].mean(dim=-1)
        attention = torch.sqrt((incoming * outgoing).clamp_min(0.0))
        self._r04r_role_participation = role
        self._r04r_attention_participation = attention[:, None].expand(-1, 2).clone()
        self._r04r_local_role_participation = [None for _ in self.pairs]
        return relative

    @staticmethod
    def _family_route(parent, momentum, participation, *, groups, width):
        """Move from the spectral parent toward the Frobenius LMO at equal budget."""
        if parent.shape != momentum.shape or parent.ndim != 3:
            raise RuntimeError("R04 optimizer-family tensor inventory changed")
        layers, hidden, external = parent.shape
        if hidden != groups * width or participation.shape != (layers, groups):
            raise RuntimeError("R04 optimizer-family group inventory changed")
        shape = (layers, groups, width, external)
        p = parent.view(shape)
        m = momentum.view(shape)
        dims = (-2, -1)
        tiny = torch.finfo(parent.dtype).tiny
        machine = torch.finfo(parent.dtype).eps
        p_norm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
        m_norm = torch.linalg.vector_norm(m, dim=dims, keepdim=True)
        valid_norm = (p_norm > 0.0) & (m_norm > 0.0)
        raw = m * (p_norm / m_norm.clamp_min(tiny))
        c = participation[:, :, None, None]
        valid_c = torch.isfinite(c) & (c >= 0.0) & (c <= 1.0)
        torch._assert_async(valid_c.all())
        source = torch.sqrt(c) * p
        source.add_(torch.sqrt((1.0 - c).clamp_min(0.0)) * raw)
        source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
        provisional = source * (p_norm / source_norm.clamp_min(tiny))
        provisional = torch.where(c == 1.0, p, provisional)
        parent_descent = (m * p).sum(dim=dims, keepdim=True)
        raw_descent = (m * raw).sum(dim=dims, keepdim=True)
        provisional_descent = (m * provisional).sum(dim=dims, keepdim=True)
        tolerance = 256.0 * machine * parent_descent.abs().clamp_min(1.0)
        accepted = (
            valid_norm
            & torch.isfinite(provisional).all(dim=dims, keepdim=True)
            & torch.isfinite(provisional_descent)
            & (provisional_descent + tolerance >= parent_descent)
        )
        direction = torch.where(accepted, provisional, p)
        direction_norm = torch.linalg.vector_norm(direction, dim=dims, keepdim=True)
        direction_descent = (m * direction).sum(dim=dims, keepdim=True)
        direction_cosine = (
            (p * direction).sum(dim=dims, keepdim=True)
            / (p_norm * direction_norm).clamp_min(tiny)
        ).clamp(-1.0, 1.0)
        direction_cosine = torch.where(
            valid_norm, direction_cosine, torch.ones_like(direction_cosine)
        )
        half_angle = torch.sqrt(
            ((1.0 - direction_cosine) / (1.0 + direction_cosine).clamp_min(tiny))
            .clamp_min(0.0)
        )
        budget_residual = (
            (direction_norm - p_norm).abs() / p_norm.clamp_min(1.0)
        )
        descent_margin = direction_descent - parent_descent
        torch._assert_async(torch.isfinite(direction).all())
        torch._assert_async((budget_residual <= 512.0 * machine).all())
        torch._assert_async((descent_margin + tolerance >= 0.0).all())
        torch._assert_async((~valid_norm | (raw_descent > 0.0)).all())
        return direction.reshape_as(parent), {
            "participation": participation.flatten(),
            "parent_descent": parent_descent.flatten(),
            "raw_descent": raw_descent.flatten(),
            "direction_descent": direction_descent.flatten(),
            # R05 consumes this compatibility inventory while forming its
            # private telemetry.  These are measurements of the route, never
            # operands of the update; R04 publishes its own family telemetry.
            "endpoint_descent": direction_descent.flatten(),
            "half_angle": half_angle.flatten(),
            "response_cap": half_angle.flatten(),
            "branch_cap": half_angle.flatten(),
            "descent_cap": half_angle.flatten(),
            "gamma": direction_cosine.flatten(),
            "descent_margin": descent_margin.flatten(),
            "budget_residual": budget_residual.flatten(),
            "accepted": accepted.flatten(),
        }

    def _descent_safe_endpoint(self, parent, adaptive, momentum, alignment):
        del adaptive, alignment
        role = self._r04r_endpoint_role
        self._r04r_endpoint_role += 1
        if self._r04r_role_participation is None or role not in (0, 1):
            raise RuntimeError("R04 endpoint did not receive a current role sensor")
        endpoint, metadata = self._family_route(
            parent,
            momentum,
            self._r04r_role_participation[:, role],
            groups=self.groups,
            width=self.width,
        )
        self._r04r_endpoint_metadata.append(metadata)
        return endpoint, metadata

    def _select_functional_corner(
        self,
        functional_inputs,
        functional_preactivations,
        functional_features,
        incoming_parent,
        incoming_endpoint,
        outgoing_parent_transpose,
        outgoing_endpoint_transpose,
        incoming_parent_descent,
        incoming_endpoint_descent,
        outgoing_parent_descent,
        outgoing_endpoint_descent,
        lr,
        *,
        force_parent,
    ):
        """Consume both morphology-routed endpoints without a loss oracle."""
        del (
            functional_inputs,
            functional_preactivations,
            functional_features,
            incoming_parent,
            outgoing_parent_transpose,
            incoming_parent_descent,
            incoming_endpoint_descent,
            outgoing_parent_descent,
            outgoing_endpoint_descent,
            lr,
            force_parent,
        )
        layers = incoming_endpoint.shape[0]
        device = incoming_endpoint.device
        dtype = incoming_endpoint.dtype
        choices = torch.full(
            (layers,), 3, device=device, dtype=torch.int64
        )
        zeros = torch.zeros(layers, device=device, dtype=dtype)
        return incoming_endpoint, outgoing_endpoint_transpose, {
            "choices": choices,
            "scores": torch.zeros((layers, 4), device=device, dtype=dtype),
            "score_margin": zeros,
            "energies": torch.zeros((layers, 4), device=device, dtype=dtype),
            "global_count": torch.zeros((), device=device, dtype=dtype),
        }

    def current_attention_alignments(self):
        if self._r04r_attention_participation is None:
            raise RuntimeError("R04 attention requested a stale family sensor")
        return self._r04r_attention_participation

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        self._r04r_role_participation = None
        self._r04r_attention_participation = None
        self._r04r_endpoint_role = 0
        self._r04r_endpoint_metadata = []
        loss = super().step(closure)
        if self._r04r_endpoint_role != 2 or len(self._r04r_endpoint_metadata) != 2:
            raise RuntimeError("R04 did not execute both optimizer-family decisions")
        if self._r04r_attention_participation is None:
            raise RuntimeError("R04 did not publish its attention family sensor")
        if publish:
            renamed = {
                key.replace("rlb_r07_", "rlb_r04_", 1): value
                for key, value in self._last_telemetry.items()
                if key.startswith("rlb_r07_")
            }
            role = self._r04r_role_participation
            incoming, outgoing = self._r04r_endpoint_metadata
            metadata = (incoming, outgoing)
            renamed.update({
                "rlb_r04_incoming_participation_min": float(role[:, 0].amin().item()),
                "rlb_r04_incoming_participation_median": float(role[:, 0].median().item()),
                "rlb_r04_incoming_participation_max": float(role[:, 0].amax().item()),
                "rlb_r04_outgoing_participation_min": float(role[:, 1].amin().item()),
                "rlb_r04_outgoing_participation_median": float(role[:, 1].median().item()),
                "rlb_r04_outgoing_participation_max": float(role[:, 1].amax().item()),
                "rlb_r04_family_activity_max": float(
                    torch.sqrt((1.0 - role).clamp_min(0.0)).amax().item()
                ),
                "rlb_r04_family_raw_descent_min": float(min(
                    item["raw_descent"].amin().item() for item in metadata
                )),
                "rlb_r04_family_descent_margin_min": float(min(
                    item["descent_margin"].amin().item() for item in metadata
                )),
                "rlb_r04_family_budget_residual_max": float(max(
                    item["budget_residual"].amax().item() for item in metadata
                )),
                "rlb_r04_family_accepted_count": int(sum(
                    item["accepted"].sum().item() for item in metadata
                )),
                "rlb_r04_structural_matrix_elements": 245_366_784,
            })
            self._last_telemetry = renamed
        return loss


class R04RevisionAttentionCore(R04AttentionCore):
    """RLB-morphology-selected spectral/Frobenius optimizer for attention."""

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "qkv_lr_scale": 1.0,
            "attention_output_lr_scale": 1.0,
            "intrinsic_rlb_family_router_lr_scale": 1.0,
            "equal_frobenius_budget_lr_scale": 1.0,
            "spectral_parent_lr_scale": 1.0,
            "frobenius_family_lr_scale": 1.0,
            "polar_lr_scale": 1.0,
            "phase_lr_scale": 1.0,
            "weight_decay_scale": 1.0,
        }

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None
        group = self.param_groups[0]
        if float(group.get("lr_scale", 1.0)) != 1.0:
            raise RuntimeError("R04 refuses a nonunit attention LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        alignments = self.router.current_attention_alignments()
        participation = alignments[:, 0]
        router_step = int(self.router._r05_step)
        anchor_state = self.state[self.role_parameters["qkv"][0]]
        previous_step = anchor_state.get("r04_revision_attention_step", 0)
        if type(previous_step) is not int or router_step != previous_step + 1:
            raise RuntimeError("R04 attention did not consume one current RLB sensor")
        anchor_state["r04_revision_attention_step"] = router_step

        records = {}
        for role in self._ROLES:
            parameters = self.role_parameters[role]
            momenta = torch.stack([
                self._nesterov(parameter) for parameter in parameters
            ]).float()
            parent = _batched_zero_power(momenta, self.ns_steps).float()
            parent.mul_(
                _match_rms_adamw_scale(momenta.shape[-2], momenta.shape[-1])
            )
            direction, metadata = R04RevisionRouterCore._family_route(
                parent,
                momenta,
                participation[:, None],
                groups=1,
                width=parent.shape[1],
            )
            for index, parameter in enumerate(parameters):
                parameter.mul_(1.0 - lr * weight_decay)
                parameter.add_(direction[index].to(parameter.dtype), alpha=-lr)
            records[role] = metadata

        if self._capture_telemetry_next_step:
            metadata = tuple(records.values())
            self._last_telemetry = {
                "rlb_r04_attention_step": router_step,
                "rlb_r04_attention_participation_min": float(participation.amin().item()),
                "rlb_r04_attention_participation_median": float(participation.median().item()),
                "rlb_r04_attention_participation_max": float(participation.amax().item()),
                "rlb_r04_attention_family_activity_max": float(
                    torch.sqrt((1.0 - participation).clamp_min(0.0)).amax().item()
                ),
                "rlb_r04_attention_raw_descent_min": float(min(
                    item["raw_descent"].amin().item() for item in metadata
                )),
                "rlb_r04_attention_descent_margin_min": float(min(
                    item["descent_margin"].amin().item() for item in metadata
                )),
                "rlb_r04_attention_budget_residual_max": float(max(
                    item["budget_residual"].amax().item() for item in metadata
                )),
                "rlb_r04_attention_family_accepted_count": int(sum(
                    item["accepted"].sum().item() for item in metadata
                )),
            }
        self._capture_telemetry_next_step = False
        return loss
