import os

import torch
from torch import nn

try:
    from . import _C
except ImportError as exc:
    _C = None
    _C_IMPORT_ERROR = exc
else:
    _C_IMPORT_ERROR = None


_GRAIN_NUMERATOR = (
    2.4915714620520875e-05,
    0.5000668663485823,
    0.2499441908995771,
    0.0526200910151016,
    0.005525765640115002,
    0.00024199321178747251,
)
_GRAIN_DENOMINATOR = (
    0.00022554017910995938,
    0.10511420350691994,
    2.8656992394109812e-05,
    0.0004816962110562025,
)


def _use_torch_fallback() -> bool:
    return os.environ.get("RATIONAL_OPT_TORCH_FALLBACK", "0") in {
        "1",
        "true",
        "True",
        "yes",
    }


def _rational_local_basis_torch(
    x,
    numerator,
    denominator,
    coeff_logits,
    centers,
    beta,
    coeff_limit,
    eps,
    hidden_dim,
    groups,
):
    shape = x.shape
    groups = int(groups)
    hidden_dim = int(hidden_dim)
    width = hidden_dim // groups
    grouped = x.view(*shape[:-1], groups, width)
    rms = torch.sqrt(
        grouped.square().mean(dim=-1, keepdim=True) + float(eps)
    )
    t = grouped / rms

    prefix = (1,) * (grouped.dim() - 2)
    coeff_shape = (*prefix, groups, 1)
    a = numerator.to(device=x.device, dtype=x.dtype)
    b = denominator.abs().to(device=x.device, dtype=x.dtype)
    a0 = a[:, 0].view(coeff_shape)
    a1 = a[:, 1].view(coeff_shape)
    a2 = a[:, 2].view(coeff_shape)
    a3 = a[:, 3].view(coeff_shape)
    a4 = a[:, 4].view(coeff_shape)
    a5 = a[:, 5].view(coeff_shape)
    b0 = b[:, 0].view(coeff_shape)
    b1 = b[:, 1].view(coeff_shape)
    b2 = b[:, 2].view(coeff_shape)
    b3 = b[:, 3].view(coeff_shape)
    t2 = t.square()
    t4 = t2.square()
    base_num = (
        ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t + a0
    )
    base_den = (
        1.0
        + b0 * t.abs()
        + b1 * t2
        + b2 * t.abs() * t2
        + b3 * t4
    )
    base = base_num / base_den.clamp_min(torch.finfo(x.dtype).tiny)

    centers = centers.to(device=x.device, dtype=x.dtype)
    beta = beta.to(device=x.device, dtype=x.dtype)
    coeff = float(coeff_limit) * torch.tanh(coeff_logits).to(
        device=x.device, dtype=x.dtype
    )
    atom_shape = (*prefix, groups, 1)
    delta = torch.zeros_like(t)
    for basis_idx in range(centers.shape[1]):
        center = centers[:, basis_idx].view(atom_shape)
        beta_i = beta[:, basis_idx].view(atom_shape)
        u = t - center
        den = 1.0 + beta_i * u.square()
        odd = u / den
        zero_level = 1.0 / (1.0 + beta_i * center.square())
        bump = 1.0 / den - zero_level
        coeff_i = coeff[:, basis_idx].view(*atom_shape, 2)
        delta = delta + coeff_i[..., 0] * odd + coeff_i[..., 1] * bump
    return ((base + delta) * rms).reshape(shape)


class _RationalLocalBasisFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        numerator,
        denominator,
        coeff_logits,
        centers,
        beta,
        coeff_limit,
        eps,
        hidden_dim,
        groups,
    ):
        x_work = x.contiguous()
        numerator_work = numerator.contiguous()
        denominator_work = denominator.contiguous()
        coeff_work = coeff_logits.contiguous()
        centers_work = centers.contiguous()
        beta_work = beta.contiguous()
        y = _C.local_basis_forward(
            x_work,
            numerator_work,
            denominator_work,
            coeff_work,
            centers_work,
            beta_work,
            float(coeff_limit),
            float(eps),
            int(hidden_dim),
            int(groups),
        )
        ctx.save_for_backward(
            x_work,
            numerator_work,
            denominator_work,
            coeff_work,
            centers_work,
            beta_work,
        )
        ctx.coeff_limit = float(coeff_limit)
        ctx.eps = float(eps)
        ctx.hidden_dim = int(hidden_dim)
        ctx.groups = int(groups)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (
            x,
            numerator,
            denominator,
            coeff_logits,
            centers,
            beta,
        ) = ctx.saved_tensors
        grad_work = grad_output.contiguous()
        grad_x, grad_num, grad_den, grad_coeff = _C.local_basis_backward(
            grad_work,
            x,
            numerator,
            denominator,
            coeff_logits,
            centers,
            beta,
            ctx.coeff_limit,
            ctx.eps,
            ctx.hidden_dim,
            ctx.groups,
        )
        return (
            grad_x,
            grad_num,
            grad_den,
            grad_coeff,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rational_local_basis(
    x,
    numerator,
    denominator,
    coeff_logits,
    centers,
    beta,
    coeff_limit,
    eps,
    hidden_dim,
    groups,
):
    """Apply the fused local-basis operator used internally by GRAIN."""

    if _use_torch_fallback():
        return _rational_local_basis_torch(
            x,
            numerator,
            denominator,
            coeff_logits,
            centers,
            beta,
            coeff_limit,
            eps,
            hidden_dim,
            groups,
        )
    if _C is None:
        raise RuntimeError(
            "the fused GRAIN extension is unavailable; build it with "
            "`python setup.py build_ext --inplace` or set "
            "RATIONAL_OPT_TORCH_FALLBACK=1 for the reference PyTorch path"
        ) from _C_IMPORT_ERROR
    return _RationalLocalBasisFunction.apply(
        x,
        numerator,
        denominator,
        coeff_logits,
        centers,
        beta,
        coeff_limit,
        eps,
        hidden_dim,
        groups,
    )


class GRAIN(nn.Module):
    """Groupwise Rational Activation with Internal Normalization."""

    def __init__(
        self,
        hidden_dim,
        groups,
        init="silu",
        fit_range=5.0,
        eps=1e-6,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        groups = int(groups)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if groups <= 0:
            raise ValueError("groups must be positive")
        if hidden_dim % groups != 0:
            raise ValueError("hidden_dim must be divisible by groups")
        if hidden_dim // groups > 256:
            raise ValueError("GRAIN supports group width <= 256")
        if init != "silu" or float(fit_range) != 5.0:
            raise ValueError("GRAIN requires the published silu@5 initializer")

        numerator_tensor = torch.tensor(
            _GRAIN_NUMERATOR, dtype=torch.float32
        ).repeat(groups, 1)
        denominator_tensor = torch.tensor(
            _GRAIN_DENOMINATOR, dtype=torch.float32
        ).repeat(groups, 1)

        self.hidden_dim = hidden_dim
        self.groups = groups
        self.init_name = init
        self.fit_range = float(fit_range)
        self.eps = float(eps)
        self.numerator = nn.Parameter(numerator_tensor)
        self.denominator = nn.Parameter(denominator_tensor)
        self.register_buffer(
            "_empty_coeff", torch.empty(groups, 0, 2, dtype=torch.float32)
        )
        self.register_buffer(
            "_empty_centers", torch.empty(groups, 0, dtype=torch.float32)
        )
        self.register_buffer(
            "_empty_beta", torch.empty(groups, 0, dtype=torch.float32)
        )

    @torch.no_grad()
    def _update_optimizer_stats(self, x):
        if not bool(getattr(self, "_rlb_optimizer_track_stats", False)):
            return
        if bool(
            getattr(self, "_rlb_optimizer_stats_training_only", False)
        ) and not self.training:
            return
        stat_every = int(getattr(self, "_rlb_optimizer_stat_every", 1))
        counter = int(getattr(self, "_rlb_optimizer_stat_counter", 0)) + 1
        self._rlb_optimizer_stat_counter = counter
        if (
            stat_every > 1
            and counter % stat_every != 0
            and hasattr(self, "_rlb_optimizer_stats")
        ):
            return

        flat = x.detach().reshape(-1, self.hidden_dim)
        max_samples = int(getattr(self, "_rlb_optimizer_stat_samples", 512))
        if max_samples > 0 and flat.size(0) > max_samples:
            index = torch.linspace(
                0,
                flat.size(0) - 1,
                max_samples,
                device=flat.device,
            ).long()
            flat = flat.index_select(0, index)
        grouped = flat.float().view(
            -1, self.groups, self.hidden_dim // self.groups
        )
        rms = torch.sqrt(
            grouped.square().mean(dim=-1, keepdim=True) + self.eps
        )
        t = grouped / rms
        abs_t = t.abs()

        moments = [
            torch.ones(self.groups, device=t.device, dtype=torch.float32)
        ]
        signed_moments = [
            torch.ones(self.groups, device=t.device, dtype=torch.float32)
        ]
        abs_power = torch.ones_like(abs_t)
        signed_power = torch.ones_like(t)
        for _ in range(1, 11):
            abs_power = abs_power * abs_t
            signed_power = signed_power * t
            moments.append(abs_power.mean(dim=(0, 2)))
            signed_moments.append(signed_power.mean(dim=(0, 2)))
        abs_moments = torch.stack(moments, dim=1)
        raw_moments = torch.stack(signed_moments, dim=1)

        t2 = t.square()
        t3 = t2 * t
        t4 = t2.square()
        t5 = t4 * t
        ax3 = abs_t * t2
        numerator = self.numerator.detach().float().view(
            1, self.groups, 1, 6
        )
        denominator = self.denominator.detach().float()
        denominator_abs = denominator.abs().view(1, self.groups, 1, 4)
        denominator_sign = torch.where(
            denominator >= 0.0, 1.0, -1.0
        ).view(1, self.groups, 1, 4)
        powers = torch.stack(
            (torch.ones_like(t), t, t2, t3, t4, t5), dim=-1
        )
        den_powers = torch.stack((abs_t, t2, ax3, t4), dim=-1)
        q = (
            1.0
            + denominator_abs[..., 0] * abs_t
            + denominator_abs[..., 1] * t2
            + denominator_abs[..., 2] * ax3
            + denominator_abs[..., 3] * t4
        )
        poly = (numerator * powers).sum(dim=-1)
        dpoly = (
            numerator[..., 1]
            + 2.0 * numerator[..., 2] * t
            + 3.0 * numerator[..., 3] * t2
            + 4.0 * numerator[..., 4] * t3
            + 5.0 * numerator[..., 5] * t4
        )
        dq = (
            denominator_abs[..., 0] * torch.sign(t)
            + 2.0 * denominator_abs[..., 1] * t
            + 3.0 * denominator_abs[..., 2] * t * abs_t
            + 4.0 * denominator_abs[..., 3] * t3
        )
        output = poly / q.clamp_min(self.eps)
        derivative = (dpoly * q - poly * dq) / q.square().clamp_min(
            self.eps
        )
        num_features = powers / q.unsqueeze(-1).clamp_min(self.eps)
        den_features = (
            -poly.unsqueeze(-1)
            * denominator_sign
            * den_powers
            / q.square().unsqueeze(-1).clamp_min(self.eps)
        )

        num_flat = num_features.permute(1, 0, 2, 3).reshape(
            self.groups, -1, 6
        )
        den_flat = den_features.permute(1, 0, 2, 3).reshape(
            self.groups, -1, 4
        )
        num_gram = torch.einsum(
            "gni,gnj->gij", num_flat, num_flat
        ) / max(1, num_flat.size(1))
        den_gram = torch.einsum(
            "gni,gnj->gij", den_flat, den_flat
        ) / max(1, den_flat.size(1))

        output_rms = torch.sqrt(
            output.square().mean(dim=(0, 2)) + self.eps
        )
        derivative_rms = torch.sqrt(
            derivative.square().mean(dim=(0, 2)) + self.eps
        )
        sample_count = torch.full(
            (self.groups,),
            float(output.size(0) * output.size(2)),
            device=output.device,
            dtype=torch.float32,
        )
        stat_version = (
            int(getattr(self, "_rlb_optimizer_stat_version", 0)) + 1
        )
        self._rlb_optimizer_stat_version = stat_version

        self._rlb_optimizer_stats = {
            "abs_moments": abs_moments.detach(),
            "raw_moments": raw_moments.detach(),
            "num_gram": num_gram.detach(),
            "den_gram": den_gram.detach(),
            "output_rms": output_rms.detach(),
            "derivative_rms": derivative_rms.detach(),
            "output_sq_sum": output.square().sum(dim=(0, 2)).detach(),
            "derivative_sq_sum": derivative.square()
            .sum(dim=(0, 2))
            .detach(),
            "sample_count": sample_count.detach(),
            "stat_version": stat_version,
        }

    def forward(self, x):
        if x.size(-1) != self.hidden_dim:
            raise ValueError(
                f"expected last dimension {self.hidden_dim}, got {x.size(-1)}"
            )
        self._update_optimizer_stats(x)
        return rational_local_basis(
            x,
            self.numerator,
            self.denominator,
            self._empty_coeff,
            self._empty_centers,
            self._empty_beta,
            0.0,
            self.eps,
            self.hidden_dim,
            self.groups,
        )

    def extra_repr(self):
        return (
            f"hidden_dim={self.hidden_dim}, groups={self.groups}, "
            f"init={self.init_name!r}, fit_range={self.fit_range:g}"
        )


__all__ = ["GRAIN", "rational_local_basis"]
