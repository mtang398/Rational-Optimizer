"""Opaque R05 optimizer-family routing from current Global-RLB morphology.

The complete positive R05-generation-one pair parent is retained.  Exact
current P5/Q4 Jacobian and response participation route each learned rational
group between that spectral parent and the equal-Frobenius-budget coordinate
sign direction of the same Nesterov momentum.  Their geometric mean makes the
same decision for QKV and attention output.

The family choice changes update geometry, while scheduled LR, WD, beta1,
clipping, NS5, shape calibration, and every internal LR/WD multiplier remain
identical to the control.
"""

from __future__ import annotations

import torch

from .rlb_group_muon_core import _batched_zero_power, _match_rms_adamw_scale
from .rlb_r04_core import R04AttentionCore
from .rlb_r04_revision_core import R04RevisionRouterCore


class R05RevisionRouterCore(R04RevisionRouterCore):
    """RLB-morphology-selected spectral/coordinate-sign family router."""

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.pop("equal_budget_frobenius_family_lr_scale", None)
        report.update({
            "equal_budget_coordinate_sign_family_lr_scale": 1.0,
            "rlb_intrinsic_family_selection_lr_scale": 1.0,
        })
        return report

    @staticmethod
    def _family_route(parent, momentum, participation, *, groups, width):
        """Route to coordinate-sign geometry at the parent's exact budget."""
        if parent.shape != momentum.shape or parent.ndim != 3:
            raise RuntimeError("R05 optimizer-family tensor inventory changed")
        layers, hidden, external = parent.shape
        if hidden != groups * width or participation.shape != (layers, groups):
            raise RuntimeError("R05 optimizer-family group inventory changed")
        shape = (layers, groups, width, external)
        p = parent.view(shape)
        m = momentum.view(shape)
        dims = (-2, -1)
        tiny = torch.finfo(parent.dtype).tiny
        machine = torch.finfo(parent.dtype).eps
        p_norm = torch.linalg.vector_norm(p, dim=dims, keepdim=True)
        sign = torch.sign(m)
        sign_norm = torch.linalg.vector_norm(sign, dim=dims, keepdim=True)
        valid_norm = (p_norm > 0.0) & (sign_norm > 0.0)
        coordinate = sign * (p_norm / sign_norm.clamp_min(tiny))
        c = participation[:, :, None, None]
        valid_c = torch.isfinite(c) & (c >= 0.0) & (c <= 1.0)
        torch._assert_async(valid_c.all())
        source = torch.sqrt(c) * p
        source.add_(torch.sqrt((1.0 - c).clamp_min(0.0)) * coordinate)
        source_norm = torch.linalg.vector_norm(source, dim=dims, keepdim=True)
        provisional = source * (p_norm / source_norm.clamp_min(tiny))
        provisional = torch.where(c == 1.0, p, provisional)
        finite = valid_norm & torch.isfinite(provisional).all(
            dim=dims, keepdim=True
        )
        direction = torch.where(finite, provisional, p)
        direction_norm = torch.linalg.vector_norm(direction, dim=dims, keepdim=True)
        parent_descent = (m * p).sum(dim=dims, keepdim=True)
        sign_descent = (m * coordinate).sum(dim=dims, keepdim=True)
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
        tolerance = 512.0 * machine * direction_descent.abs().clamp_min(1.0)
        torch._assert_async(torch.isfinite(direction).all())
        torch._assert_async((budget_residual <= 512.0 * machine).all())
        torch._assert_async((~valid_norm | (sign_descent > 0.0)).all())
        torch._assert_async((~valid_norm | (direction_descent + tolerance > 0.0)).all())
        return direction.reshape_as(parent), {
            "participation": participation.flatten(),
            "parent_descent": parent_descent.flatten(),
            "sign_descent": sign_descent.flatten(),
            # Compatibility aliases consumed only by the inherited private
            # telemetry builder.  R05 publishes sign-family terminology.
            "raw_descent": sign_descent.flatten(),
            "direction_descent": direction_descent.flatten(),
            "endpoint_descent": direction_descent.flatten(),
            "half_angle": half_angle.flatten(),
            "response_cap": half_angle.flatten(),
            "branch_cap": half_angle.flatten(),
            "descent_cap": half_angle.flatten(),
            "gamma": direction_cosine.flatten(),
            "descent_margin": descent_margin.flatten(),
            "budget_residual": budget_residual.flatten(),
            "accepted": finite.flatten(),
        }

    @torch.no_grad()
    def step(self, closure=None):
        publish = bool(self._capture_telemetry_next_step)
        loss = super().step(closure)
        if publish:
            renamed = {}
            for key, value in self._last_telemetry.items():
                if not key.startswith("rlb_r04_"):
                    continue
                name = key.replace("rlb_r04_", "rlb_r05_", 1)
                name = name.replace("family_raw_descent", "family_sign_descent")
                renamed[name] = value
            self._last_telemetry = renamed
        return loss


class R05RevisionAttentionCore(R04AttentionCore):
    """RLB-morphology-selected spectral/coordinate-sign attention family."""

    def lr_wd_fairness_audit(self):
        return {
            "global_lr_scale": 1.0,
            "qkv_lr_scale": 1.0,
            "attention_output_lr_scale": 1.0,
            "intrinsic_rlb_family_router_lr_scale": 1.0,
            "equal_coordinate_sign_budget_lr_scale": 1.0,
            "spectral_parent_lr_scale": 1.0,
            "coordinate_sign_family_lr_scale": 1.0,
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
            raise RuntimeError("R05 refuses a nonunit attention LR scale")
        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        alignments = self.router.current_attention_alignments()
        participation = alignments[:, 0]
        router_step = int(self.router._r05_step)
        anchor_state = self.state[self.role_parameters["qkv"][0]]
        previous_step = anchor_state.get("r05_revision_attention_step", 0)
        if type(previous_step) is not int or router_step != previous_step + 1:
            raise RuntimeError("R05 attention did not consume one current RLB sensor")
        anchor_state["r05_revision_attention_step"] = router_step

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
            direction, metadata = R05RevisionRouterCore._family_route(
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
                "rlb_r05_attention_step": router_step,
                "rlb_r05_attention_participation_min": float(participation.amin().item()),
                "rlb_r05_attention_participation_median": float(participation.median().item()),
                "rlb_r05_attention_participation_max": float(participation.amax().item()),
                "rlb_r05_attention_family_activity_max": float(
                    torch.sqrt((1.0 - participation).clamp_min(0.0)).amax().item()
                ),
                "rlb_r05_attention_sign_descent_min": float(min(
                    item["sign_descent"].amin().item() for item in metadata
                )),
                "rlb_r05_attention_direction_descent_min": float(min(
                    item["direction_descent"].amin().item() for item in metadata
                )),
                "rlb_r05_attention_budget_residual_max": float(max(
                    item["budget_residual"].amax().item() for item in metadata
                )),
                "rlb_r05_attention_family_accepted_count": int(sum(
                    item["accepted"].sum().item() for item in metadata
                )),
            }
        self._capture_telemetry_next_step = False
        return loss
