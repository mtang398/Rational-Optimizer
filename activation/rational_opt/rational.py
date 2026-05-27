import math

import torch
from torch import nn

from . import _C


_INIT_TABLE = {
    ("relu", 3.0): (
        [
            0.033901097187046862,
            0.499999999978465,
            1.6699463819733569,
            1.9896443701539706,
            0.9409837717351536,
            0.1508438988454924,
        ],
        [
            -2.0847636051209747e-10,
            3.9792887406007273,
            -1.5184617191916304e-10,
            0.30168779771678483,
        ],
    ),
    ("relu", 5.0): (
        [
            0.05650182920786639,
            0.49999999999711053,
            1.0019678214855787,
            0.7162719637360658,
            0.20325249117614147,
            0.019549368894503073,
        ],
        [
            -1.5444869040830349e-11,
            1.432543927483962,
            -3.3501021559631508e-12,
            0.039098737789319785,
        ],
    ),
    ("relu", 7.0): (
        [
            0.07910256072584829,
            0.5000000003352276,
            0.7156913029247412,
            0.3654448809037376,
            0.07407160793559167,
            0.0050888611526869075,
        ],
        [
            1.3944705623247637e-09,
            0.7308897609663136,
            1.8728335972005116e-10,
            0.010177722291718682,
        ],
    ),
    ("gelu", 3.0): (
        [
            -0.0004223506711992556,
            0.4999999999999481,
            0.4026822132227292,
            0.07366295251774879,
            -0.012950949163890231,
            -0.0037401813470032217,
        ],
        [
            7.928890161927364e-12,
            0.14732590502888199,
            1.8041810903819121e-12,
            -0.007480362694164852,
        ],
    ),
    ("gelu", 5.0): (
        [
            -0.002323546278301879,
            0.5010930262146732,
            0.508862265943374,
            0.19217533573000617,
            0.03200295355303943,
            0.0019824528641465746,
        ],
        [
            0.19015085308074264,
            0.23125739259275593,
            0.0406782234638605,
            0.00041446716044879055,
        ],
    ),
    ("gelu", 7.0): (
        [
            -0.004217354146563875,
            0.5000000001934006,
            0.42241671565655503,
            0.1289555377166242,
            0.01693570003937073,
            0.000810248450647167,
        ],
        [
            2.4344917944701426e-09,
            0.2579110738481794,
            3.430952534630306e-10,
            0.0016204968773733701,
        ],
    ),
}


_VERSION_A_INIT_TABLE = {
    ("identity", 3.0): (
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1e-08, 1e-08, 1e-08, 1e-08],
    ),
    ("identity", 5.0): (
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1e-08, 1e-08, 1e-08, 1e-08],
    ),
    ("identity", 7.0): (
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1e-08, 1e-08, 1e-08, 1e-08],
    ),
    ("relu", 3.0): (
        [
            0.03407140096537871,
            0.500000000785916,
            1.6686247095521778,
            1.989644378791154,
            0.9412665688538142,
            0.15084389978512813,
        ],
        [
            1e-08,
            3.9792887406007273,
            1e-08,
            0.30168779771678483,
        ],
    ),
    ("relu", 5.0): (
        [
            0.05678566882939273,
            0.499999998601522,
            1.001174819993277,
            0.7162719746045968,
            0.20331357925790883,
            0.019549369530291263,
        ],
        [
            1e-08,
            1.432543927483962,
            1e-08,
            0.039098737789319785,
        ],
    ),
    ("relu", 7.0): (
        [
            0.07949993622820213,
            0.49999999235465464,
            0.7151248739593938,
            0.36544489442231154,
            0.07409387344196872,
            0.005088861615544671,
        ],
        [
            1e-08,
            0.7308897609663136,
            1e-08,
            0.010177722291718682,
        ],
    ),
    ("gelu", 3.0): (
        [
            0.011264773384878041,
            0.5000000000462278,
            0.35608641035561195,
            0.07366296157761802,
            0.013302515590133956,
            0.0037401822045816844,
        ],
        [
            1e-08,
            0.147325905028882,
            1e-08,
            0.007480362694164852,
        ],
    ),
    ("gelu", 5.0): (
        [
            -0.01708892945210021,
            0.5385611231879953,
            0.5024356148637584,
            0.18681350949076436,
            0.03243657130432953,
            0.0021540413477319988,
        ],
        [
            0.19015085308074264,
            0.23125739259275593,
            0.0406782234638605,
            0.00041446716044879055,
        ],
    ),
    ("gelu", 7.0): (
        [
            -0.0042528740219461075,
            0.4999999864484506,
            0.4224393137730823,
            0.12895555195244504,
            0.01693498923215231,
            0.0008102488817252929,
        ],
        [
            1e-08,
            0.2579110738481794,
            1e-08,
            0.00162049687737337,
        ],
    ),
    ("silu", 3.0): (
        [
            4.254201060600725e-07,
            0.5000003872470142,
            0.24999710309055798,
            0.053255907123573906,
            0.005798273066098062,
            0.0002745689707158847,
        ],
        [
            1.2870900772250533e-06,
            0.10651127591950058,
            1.1778714531718678e-19,
            0.0005491605551471916,
        ],
    ),
    ("silu", 5.0): (
        [
            2.4915714620520875e-05,
            0.5000668663485823,
            0.2499441908995771,
            0.0526200910151016,
            0.005525765640115002,
            0.00024199321178747251,
        ],
        [
            0.00022554017910995938,
            0.10511420350691994,
            2.8656992394109812e-05,
            0.0004816962110562025,
        ],
    ),
    ("silu", 7.0): (
        [
            0.0002476663375808408,
            0.5007083110046961,
            0.2497313152135398,
            0.05148239082053322,
            0.005127787279384726,
            0.00020428720649868503,
        ],
        [
            0.0018511692913042153,
            0.1021761709465005,
            0.0001351822732411503,
            0.00040050758014702404,
        ],
    ),
}


def _load_init(table, init, fit_range):
    key = (init, float(fit_range))
    if key not in table:
        choices = ", ".join(f"{name}@{fit_range:g}" for name, fit_range in sorted(table))
        raise ValueError(f"unknown rational initializer {init}@{float(fit_range):g}; expected one of: {choices}")
    return table[key]


def _repeat_init(numerator, denominator, groups):
    numerator_tensor = torch.tensor(numerator, dtype=torch.float32).repeat(groups, 1)
    denominator_tensor = torch.tensor(denominator, dtype=torch.float32).repeat(groups, 1)
    return numerator_tensor, denominator_tensor


def _balanced_chunk_sizes(hidden_dim, groups):
    base_size, remainder = divmod(hidden_dim, groups)
    return tuple(base_size + (idx < remainder) for idx in range(groups))


def _width_scaled_groups(hidden_dim, group_size, max_groups):
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if max_groups <= 0:
        raise ValueError("max_groups must be positive")
    return min(hidden_dim, max(1, min(max_groups, math.ceil(hidden_dim / group_size))))


class _RationalA5_4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, numerator, denominator):
        x_work = x.contiguous()
        y = _C.forward(x_work, numerator, denominator)
        ctx.save_for_backward(x_work, numerator, denominator)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, numerator, denominator = ctx.saved_tensors
        grad_work = grad_output.contiguous()
        return tuple(_C.backward(grad_work, x, numerator, denominator))


def rational_a5_4(x, numerator, denominator):
    return _RationalA5_4Function.apply(x, numerator, denominator)


class _RationalVersionA5_4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, numerator, denominator):
        x_work = x.contiguous()
        y = _C.version_a_forward(x_work, numerator, denominator)
        ctx.save_for_backward(x_work, numerator, denominator)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, numerator, denominator = ctx.saved_tensors
        grad_work = grad_output.contiguous()
        return tuple(_C.version_a_backward(grad_work, x, numerator, denominator))


def rational_version_a5_4(x, numerator, denominator):
    return _RationalVersionA5_4Function.apply(x, numerator, denominator)



class _RationalLocalBasisFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, numerator, denominator, coeff_logits, centers, beta, coeff_limit, eps, hidden_dim, groups):
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
        ctx.save_for_backward(x_work, numerator_work, denominator_work, coeff_work, centers_work, beta_work)
        ctx.coeff_limit = float(coeff_limit)
        ctx.eps = float(eps)
        ctx.hidden_dim = int(hidden_dim)
        ctx.groups = int(groups)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, numerator, denominator, coeff_logits, centers, beta = ctx.saved_tensors
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
        return grad_x, grad_num, grad_den, grad_coeff, None, None, None, None, None, None


def rational_local_basis(x, numerator, denominator, coeff_logits, centers, beta, coeff_limit, eps, hidden_dim, groups):
    return _RationalLocalBasisFunction.apply(
        x, numerator, denominator, coeff_logits, centers, beta, coeff_limit, eps, hidden_dim, groups
    )


class RationalFusedLocalBasisA5_4(nn.Module):
    """Fused grouped Version A rational activation plus fixed local rational atoms.

    This is a rational-only single-branch activation. It uses group RMS only to
    keep the rational input domain controlled; the scalar nonlinearity itself is
    the trainable Version A rational plus local rational atoms.
    """

    def __init__(
        self,
        hidden_dim,
        groups,
        init="silu",
        fit_range=5.0,
        centers=(-0.75, 0.75),
        coeff_limit=0.60,
        odd_init=0.06,
        bump_init=0.04,
        beta=0.75,
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
            raise ValueError("fused local-basis rational supports group width <= 256")
        centers = tuple(float(c) for c in centers)
        if len(centers) < 1 or len(centers) > 4:
            raise ValueError("fused local-basis rational supports 1 to 4 centers")

        numerator, denominator = _load_init(_VERSION_A_INIT_TABLE, init, fit_range)
        numerator_tensor, denominator_tensor = _repeat_init(numerator, denominator, groups)
        center_tensor = torch.tensor(centers, dtype=torch.float32).view(1, len(centers)).repeat(groups, 1)
        beta_tensor = torch.full((groups, len(centers)), float(beta), dtype=torch.float32)

        coeff = torch.zeros(groups, len(centers), 2, dtype=torch.float32)
        signs = torch.sign(center_tensor)
        signs[signs == 0] = 1.0
        coeff[..., 0] = float(odd_init) * signs
        coeff[..., 1] = float(bump_init)
        coeff = torch.clamp(coeff / float(coeff_limit), -0.999, 0.999)

        self.hidden_dim = hidden_dim
        self.groups = groups
        self.init_name = init
        self.fit_range = float(fit_range)
        self.coeff_limit = float(coeff_limit)
        self.eps = float(eps)
        self.numerator = nn.Parameter(numerator_tensor)
        self.denominator = nn.Parameter(denominator_tensor)
        self.coeff_logits = nn.Parameter(torch.atanh(coeff))
        self.register_buffer("centers", center_tensor)
        self.register_buffer("beta", beta_tensor)

    @torch.no_grad()
    def _update_optimizer_stats(self, x):
        if not bool(getattr(self, "_rlb_optimizer_track_stats", False)):
            return
        stat_every = int(getattr(self, "_rlb_optimizer_stat_every", 1))
        counter = int(getattr(self, "_rlb_optimizer_stat_counter", 0)) + 1
        self._rlb_optimizer_stat_counter = counter
        if stat_every > 1 and counter % stat_every != 0 and hasattr(self, "_rlb_optimizer_stats"):
            return

        flat = x.detach().reshape(-1, self.hidden_dim)
        max_samples = int(getattr(self, "_rlb_optimizer_stat_samples", 512))
        if max_samples > 0 and flat.size(0) > max_samples:
            index = torch.linspace(0, flat.size(0) - 1, max_samples, device=flat.device).long()
            flat = flat.index_select(0, index)
        grouped = flat.float().view(-1, self.groups, self.hidden_dim // self.groups)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = grouped / rms
        abs_t = t.abs()

        moments = [torch.ones(self.groups, device=t.device, dtype=torch.float32)]
        signed_moments = [torch.ones(self.groups, device=t.device, dtype=torch.float32)]
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
        numerator = self.numerator.detach().float().view(1, self.groups, 1, 6)
        denominator = self.denominator.detach().float()
        denominator_abs = denominator.abs().view(1, self.groups, 1, 4)
        denominator_sign = torch.where(denominator >= 0.0, 1.0, -1.0).view(1, self.groups, 1, 4)
        powers = torch.stack((torch.ones_like(t), t, t2, t3, t4, t5), dim=-1)
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
        base_output = poly / q.clamp_min(self.eps)
        base_derivative = (dpoly * q - poly * dq) / q.square().clamp_min(self.eps)
        num_features = powers / q.unsqueeze(-1).clamp_min(self.eps)
        den_features = -poly.unsqueeze(-1) * denominator_sign * den_powers / q.square().unsqueeze(-1).clamp_min(self.eps)

        num_flat = num_features.permute(1, 0, 2, 3).reshape(self.groups, -1, 6)
        den_flat = den_features.permute(1, 0, 2, 3).reshape(self.groups, -1, 4)
        num_gram = torch.einsum("gni,gnj->gij", num_flat, num_flat) / max(1, num_flat.size(1))
        den_gram = torch.einsum("gni,gnj->gij", den_flat, den_flat) / max(1, den_flat.size(1))

        centers = self.centers.float().view(1, self.groups, 1, -1)
        beta = self.beta.float().view(1, self.groups, 1, -1)
        u = t.unsqueeze(-1) - centers
        den = 1.0 + beta * u.square()
        odd = u / den
        zero = (1.0 + beta * centers.square()).reciprocal()
        bump = den.reciprocal() - zero
        den2 = den.square().clamp_min(self.eps)
        odd_dt = (1.0 - beta * u.square()) / den2
        bump_dt = -2.0 * beta * u / den2
        coeff = self.coeff_limit * torch.tanh(self.coeff_logits.detach().float()).view(1, self.groups, 1, -1, 2)
        output = base_output + (coeff[..., 0] * odd + coeff[..., 1] * bump).sum(dim=-1)
        derivative = base_derivative + (coeff[..., 0] * odd_dt + coeff[..., 1] * bump_dt).sum(dim=-1)
        output_rms = torch.sqrt(output.square().mean(dim=(0, 2)) + self.eps)
        derivative_rms = torch.sqrt(derivative.square().mean(dim=(0, 2)) + self.eps)
        atom_basis = torch.stack((odd, bump), dim=-1)
        atom_rms = torch.sqrt(atom_basis.square().mean(dim=(0, 2)) + self.eps)
        atom_flat = atom_basis.permute(1, 0, 2, 3, 4).reshape(self.groups, -1, atom_basis.size(-2) * atom_basis.size(-1))
        atom_gram = torch.einsum("gni,gnj->gij", atom_flat, atom_flat) / max(1, atom_flat.size(1))
        coeff_gain = self.coeff_limit * (1.0 - torch.tanh(self.coeff_logits.detach().float()).square())
        coeff_gain = coeff_gain.reshape(self.groups, -1)
        atom_gram = atom_gram * coeff_gain.unsqueeze(-1) * coeff_gain.unsqueeze(-2)
        atom_rms = atom_rms * coeff_gain.view(self.groups, atom_rms.size(1), atom_rms.size(2)).clamp_min(self.eps)

        self._rlb_optimizer_stats = {
            "abs_moments": abs_moments.detach(),
            "raw_moments": raw_moments.detach(),
            "num_gram": num_gram.detach(),
            "den_gram": den_gram.detach(),
            "atom_gram": atom_gram.detach(),
            "atom_rms": atom_rms.detach(),
            "output_rms": output_rms.detach(),
            "derivative_rms": derivative_rms.detach(),
        }

    def forward(self, x):
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {x.size(-1)}")
        self._update_optimizer_stats(x)
        return rational_local_basis(
            x,
            self.numerator,
            self.denominator,
            self.coeff_logits,
            self.centers,
            self.beta,
            self.coeff_limit,
            self.eps,
            self.hidden_dim,
            self.groups,
        )

    def extra_repr(self):
        return (
            f"hidden_dim={self.hidden_dim}, groups={self.groups}, centers={tuple(float(x) for x in self.centers[0])}, "
            f"init={self.init_name!r}, fit_range={self.fit_range:g}"
        )


class RationalA5_4(nn.Module):
    """Trainable plain P5/(1+Q4) rational activation backed by the CUDA extension."""

    def __init__(self, init="relu", fit_range=5.0):
        super().__init__()
        numerator, denominator = _load_init(_INIT_TABLE, init, fit_range)
        self.init_name = init
        self.fit_range = float(fit_range)
        self.numerator = nn.Parameter(
            torch.tensor(numerator, dtype=torch.float32),
        )
        self.denominator = nn.Parameter(
            torch.tensor(denominator, dtype=torch.float32),
        )

    def forward(self, x):
        return rational_a5_4(x, self.numerator, self.denominator)


class RationalVersionA5_4(nn.Module):
    """Trainable Version A P5/(1+sum |b_i x^i|) rational activation."""

    def __init__(self, init="relu", fit_range=5.0):
        super().__init__()
        numerator, denominator = _load_init(_VERSION_A_INIT_TABLE, init, fit_range)
        self.init_name = init
        self.fit_range = float(fit_range)
        self.numerator = nn.Parameter(
            torch.tensor(numerator, dtype=torch.float32),
        )
        self.denominator = nn.Parameter(
            torch.tensor(denominator, dtype=torch.float32),
        )

    def forward(self, x):
        return rational_version_a5_4(x, self.numerator, self.denominator)


class RationalGroupedVersionA5_4(nn.Module):
    """Width-scaled Version A rational activation with one coefficient set per channel group."""

    def __init__(
        self,
        hidden_dim,
        init="gelu",
        fit_range=5.0,
        group_size=256,
        max_groups=32,
        groups=None,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        if groups is None:
            groups = _width_scaled_groups(hidden_dim, int(group_size), int(max_groups))
        else:
            groups = int(groups)
            if groups <= 0:
                raise ValueError("groups must be positive")
            if groups > hidden_dim:
                raise ValueError("groups cannot exceed hidden_dim")

        numerator, denominator = _load_init(_VERSION_A_INIT_TABLE, init, fit_range)
        numerator_tensor, denominator_tensor = _repeat_init(numerator, denominator, groups)

        self.hidden_dim = hidden_dim
        self.init_name = init
        self.fit_range = float(fit_range)
        self.group_size = int(group_size)
        self.max_groups = int(max_groups)
        self.groups = groups
        self.chunk_sizes = _balanced_chunk_sizes(hidden_dim, groups)
        self.numerator = nn.Parameter(numerator_tensor)
        self.denominator = nn.Parameter(denominator_tensor)

    def forward(self, x):
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {x.size(-1)}")
        if self.groups == 1:
            return rational_version_a5_4(x, self.numerator[0], self.denominator[0])

        chunks = torch.split(x, self.chunk_sizes, dim=-1)
        outputs = [
            rational_version_a5_4(chunk, self.numerator[idx], self.denominator[idx])
            for idx, chunk in enumerate(chunks)
        ]
        return torch.cat(outputs, dim=-1)

    def extra_repr(self):
        return (
            f"hidden_dim={self.hidden_dim}, groups={self.groups}, "
            f"group_size={self.group_size}, init={self.init_name!r}, fit_range={self.fit_range:g}"
        )


RationalDefaultA5_4 = RationalVersionA5_4
RationalWidthScaledA5_4 = RationalGroupedVersionA5_4
rational_default_a5_4 = rational_version_a5_4
