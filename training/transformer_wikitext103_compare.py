import argparse
import json
import math
import os
import re
import time
from array import array
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def precompute_rope(seq_len, head_dim, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, seq_len):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        cos, sin = precompute_rope(seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        cos = self.rope_cos[:, :, :seq_len, :].to(dtype=q.dtype)
        sin = self.rope_sin[:, :, :seq_len, :].to(dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out(y)


RATIONAL_ACTIVATIONS = {
    "rational_a",
    "rational_grouped",
    "rational_up",
    "rational_up_grouped",
    "rational_both",
    "rational_both_grouped",
    "rational_product",
    "rational_product_grouped",
    "rational_swiglu_post",
    "rational_swiglu_post_grouped",
}


CLASSIC_OPTIMIZERS = {
    "adamw",
    "factored_adamw",
    "muon",
    "lion",
    "ademamix",
    "schedule_free_adamw",
    "adafactor_came",
    "soap_adamw",
}
RATIONAL_SPECIFIC_OPTIMIZERS = {
    "rational_onpolicy_balance",
    "rational_quotient_onpolicy",
    "rational_jacobian_onpolicy",
    "rational_jacobian_factored_onpolicy",
    "rational_layerwise_switch_onpolicy",
    "rational_layerwise_factored_switch_onpolicy",
    "rational_quotient_jacobian_onpolicy",
    "rational_adaptive_metric_onpolicy",
    "rational_transport_onpolicy",
    "rational_functional_trust_onpolicy",
    "rational_matrix_policy_onpolicy",
}
ACTIVE_OPTIMIZERS = sorted(CLASSIC_OPTIMIZERS | RATIONAL_SPECIFIC_OPTIMIZERS)


RATIONAL_BASIS_ACTIVATIONS = {
    "plain_rational_ffn",
    "rational_basis_k2",
    "rational_basis_k3_equal",
    "rational_basis_k2_per_channel",
}


RQM_ACTIVATIONS = {
    "rqm_ffn",
    "rqm_ffn_shared128",
    "rqm_ffn_beta005",
    "rqm_ffn_kappa05",
    "rqm_ffn_wide_lowrank",
    "rqm_ffn_narrow_highrank",
}


RKM_ACTIVATIONS = {
    "rkm_ffn",
    "rkm_ffn_more_regions",
    "rkm_ffn_fewer_regions",
}


RAPM_ACTIVATIONS = {
    "rapm_ffn",
    "rapm_ffn_beta005",
    "rapm_ffn_kappa05",
    "rapm_ffn_per_channel",
}


RPF_ACTIVATIONS = {
    "rpf_ffn",
    "rpf_ffn_curv05",
}


RKDM_ACTIVATIONS = {
    "rkdm_ffn",
    "rkdm_ffn_more_regions",
    "rkdm_ffn_highrank",
}


RPB_ACTIVATIONS = {
    "rpb_ffn",
    "rpb_ffn_diff",
    "rpb_ffn_norm",
}


RWF_ACTIVATIONS = {
    "rwf_ffn",
    "rwf_ffn_gelu",
}


RMB_ACTIVATIONS = {
    "rmb_k2_ffn",
    "rmb_k3_ffn",
}


RMA_ACTIVATIONS = {
    "rma_ffn",
    "rma_ffn_curv",
    "rma_ffn_strong",
    "rma_raw_ffn_curv",
    "rma_gelu_curv_ffn",
    "rma_gelu_strong_ffn",
    "rma_relu_curv_ffn",
    "rma_identity_curv_ffn",
    "rma_shift_ffn",
    "rma_gelu_shift_ffn",
    "rma_identity_shift_ffn",
    "rma_silu_shift_strong_ffn",
    "rma_silu_momaffine_ffn",
    "rma_gelu_momaffine_ffn",
    "rma_identity_momaffine_ffn",
    "rma_silu_denwide_shift_ffn",
    "rma_silu_dennarrow_shift_ffn",
    "rma_skip_ffn",
    "rma_raw_skip_ffn",
    "rma_radial_ffn",
    "rma_center_ffn",
    "rma_center_strong_ffn",
    "rma_hermite_ffn",
    "rma_center_hermite_ffn",
    "rma_divnorm25_ffn",
    "rma_divnorm50_ffn",
    "rma_divnorm75_ffn",
    "rma_divnorm_strong_ffn",
    "rma_pair_ffn",
    "rma_pair_ffn_strong",
}


RDA_ACTIVATIONS = {
    "rda_silu_ffn",
    "rda_silu_shift_ffn",
    "rda_silu_momaffine_ffn",
    "rda_gelu_momaffine_ffn",
    "rda_identity_momaffine_ffn",
}


RLBX_ACTIVATIONS = {
    "rlbx_k2_ffn",
    "rlbx_k2_shift_ffn",
    "rlbx_k2_strong_ffn",
    "rlbx_k2_identity_ffn",
}


RLB_ACTIVATIONS = {
    "rlb_shift_ffn",
    "rlb_strong_ffn",
    "rlb_wide_ffn",
    "rlb_identity_ffn",
    "rlb_fixed_strong_ffn",
    "rlb_fast_ffn",
    "rlb_fast_train_ffn",
    "rlb_fast_scaled_ffn",
    "rlb_fused_fast_ffn",
    "rlb_fused_fast_h2816_ffn",
    "rlb_fused_fast_h2640_ffn",
    "rlb_fused_fast_h2560_ffn",
    "rlb_fused_fixed_strong_ffn",
    "rlb_fused_fixed_strong_h2880_ffn",
    "rlb_fused_fixed_strong_h2816_ffn",
    "rlb_fused_fixed_strong_h2640_ffn",
    "rlb_fused_fixed_strong_h2560_ffn",
    "rlb_fused_boost_h2560_ffn",
    "rlb_fused_boost_h2400_ffn",
    "rlb_fused_quantile4_ffn",
    "rlb_fused_core4_ffn",
    "rlb_centered_strong_ffn",
    "rlb_centered_scaled_ffn",
    "rlb_fixed_centered_ffn",
}


RCQ_ACTIVATIONS = {
    "rcq_ffn",
    "rcq_shift_ffn",
    "rcq_strong_ffn",
    "rcq_identity_ffn",
}


RGC_ACTIVATIONS = {
    "rgc_ffn",
    "rgc_shift_ffn",
    "rgc_strong_ffn",
    "rgc_moment_ffn",
    "rgc_identity_ffn",
}


RSM_ACTIVATIONS = {
    "rsm_ffn",
    "rsm_ffn_basis",
    "rsm_ffn_strong",
}


RHG_ACTIVATIONS = {
    "crv_rhg",
    "rhg_ffn",
    "rhg_ffn_balanced",
    "rhg_ffn_basisgate",
    "rhg_ffn_basisgate_resvalue_wide",
    "rhg_ffn_basisgate_wide",
    "rhg_ffn_gateact",
    "rhg_ffn_fullact",
    "rhg_ffn_highgate",
    "rhg_ffn_resboth",
    "rhg_ffn_resgate",
    "rhg_ffn_resvalue",
    "rhg_ffn_resvalue_dual",
    "rhg_ffn_resvalue_gated_dual",
    "rhg_ffn_resvalue_gated",
    "rhg_ffn_resvalue_gated_channel",
    "rhg_ffn_resvalue_gated_channel_strong",
    "rhg_ffn_resvalue_gated_crossmix64",
    "rhg_ffn_resvalue_gated_crossmix128",
    "rhg_ffn_resvalue_gated_beta075",
    "rhg_ffn_resvalue_gated_beta10",
    "rhg_ffn_resvalue_gated_beta10_depthup",
    "rhg_ffn_resvalue_gated_beta10_groupdepth",
    "rhg_ffn_resvalue_gated_beta10_groupscale",
    "rhg_ffn_resvalue_gated_beta10_groupscale_moment",
    "rhg_ffn_resvalue_gated_beta10_groupscale_safegate",
    "rhg_ffn_resvalue_gated_beta10_groupscale_safegate_low",
    "rhg_ffn_resvalue_gated_beta125",
    "rhg_ffn_resvalue_gated_beta20",
    "rhg_ffn_resvalue_gated_highgate",
    "rhg_ffn_resvalue_gated_norm",
    "rhg_ffn_resvalue_gated_valuewide",
    "rhg_ffn_resvalue_norm",
    "rhg_ffn_valueact",
    "rhg_ffn_valuewide",
}


def resolve_group_count(hidden_dim, group_size, max_groups):
    target = min(int(max_groups), max(1, math.ceil(int(hidden_dim) / int(group_size))))
    target = min(target, int(hidden_dim))
    for groups in range(target, 0, -1):
        if hidden_dim % groups == 0:
            return groups
    return 1


def _birational_glu_compute(gate, value, numerator, denominator, alpha, groups, eps):
    hidden_dim = gate.shape[-1]
    if hidden_dim % groups != 0:
        raise ValueError("BiRationalGLU requires hidden_dim divisible by groups")
    width_per_group = hidden_dim // groups

    base = F.silu(gate) * value
    shape = gate.shape
    s_scale = torch.rsqrt(gate.pow(2).mean(dim=-1, keepdim=True) + eps)
    t_scale = torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)
    s = (gate * s_scale).view(*shape[:-1], groups, width_per_group)
    t = (value * t_scale).view(*shape[:-1], groups, width_per_group)

    coeff_shape = (1,) * (s.dim() - 2) + (groups, 1)
    a0 = numerator[:, 0].view(coeff_shape)
    a1 = numerator[:, 1].view(coeff_shape)
    a2 = numerator[:, 2].view(coeff_shape)
    a3 = numerator[:, 3].view(coeff_shape)
    a4 = numerator[:, 4].view(coeff_shape)
    b0 = denominator[:, 0].view(coeff_shape)
    b1 = denominator[:, 1].view(coeff_shape)
    b2 = denominator[:, 2].view(coeff_shape)
    b3 = denominator[:, 3].view(coeff_shape)
    b4 = denominator[:, 4].view(coeff_shape)

    st = s * t
    s2 = s.square()
    t2 = t.square()
    rational_num = a0 * s + a1 * t + a2 * st + a3 * s2 + a4 * t2
    rational_den = 1.0 + (b0 * s).abs() + (b1 * t).abs() + (b2 * st).abs() + (b3 * s2).abs() + (b4 * t2).abs()
    psi = rational_num / rational_den
    alpha_shape = (1,) * (psi.dim() - 2) + (groups, 1)
    residual = alpha.view(alpha_shape) * psi
    return base + residual.reshape(shape)


class _BiRationalGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, value, numerator, denominator, alpha, groups, eps):
        ctx.groups = int(groups)
        ctx.eps = float(eps)
        ctx.save_for_backward(gate, value, numerator, denominator, alpha)
        with torch.no_grad():
            return _birational_glu_compute(gate, value, numerator, denominator, alpha, ctx.groups, ctx.eps)

    @staticmethod
    def backward(ctx, grad_output):
        gate, value, numerator, denominator, alpha = ctx.saved_tensors
        with torch.enable_grad():
            gate_replay = gate.detach().requires_grad_(True)
            value_replay = value.detach().requires_grad_(True)
            numerator_replay = numerator.detach().requires_grad_(True)
            denominator_replay = denominator.detach().requires_grad_(True)
            alpha_replay = alpha.detach().requires_grad_(True)
            replay = _birational_glu_compute(
                gate_replay,
                value_replay,
                numerator_replay,
                denominator_replay,
                alpha_replay,
                ctx.groups,
                ctx.eps,
            )
            grads = torch.autograd.grad(
                replay,
                (gate_replay, value_replay, numerator_replay, denominator_replay, alpha_replay),
                grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        return grads[0], grads[1], grads[2], grads[3], grads[4], None, None


class BiRationalGLUInteraction(nn.Module):
    def __init__(
        self,
        hidden_dim,
        group_size=256,
        max_groups=32,
        alpha_init=1e-3,
        denominator_init=1e-3,
        eps=1e-6,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.group_size = int(group_size)
        self.max_groups = int(max_groups)
        self.groups = min(self.max_groups, max(1, math.ceil(self.hidden_dim / self.group_size)))
        if self.hidden_dim % self.groups != 0:
            raise ValueError("BiRationalGLUInteraction currently requires hidden_dim divisible by groups")

        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.numerator = nn.Parameter(torch.zeros(self.groups, 5))
        self.denominator = nn.Parameter(torch.full((self.groups, 5), float(denominator_init)))
        self.alpha = nn.Parameter(torch.full((self.groups,), float(alpha_init)))

    def forward(self, gate, value):
        return _BiRationalGLUFunction.apply(
            gate,
            value,
            self.numerator,
            self.denominator,
            self.alpha,
            self.groups,
            self.eps,
        )


def rational_basis_settings(activation, ffn_dim, group_size, max_groups):
    if activation == "plain_rational_ffn":
        hidden_dim = int(ffn_dim)
        groups = resolve_group_count(hidden_dim, group_size, max_groups)
        return {
            "basis_count": 1,
            "hidden_dim": hidden_dim,
            "groups": groups,
            "init": "identity",
        }
    if activation == "rational_basis_k2":
        hidden_dim = int(ffn_dim)
        groups = resolve_group_count(hidden_dim, group_size, max_groups)
        return {
            "basis_count": 2,
            "hidden_dim": hidden_dim,
            "groups": groups,
            "init": "identity_curvature",
        }
    if activation == "rational_basis_k3_equal":
        hidden_dim = (3 * int(ffn_dim)) // 4
        groups = resolve_group_count(hidden_dim, group_size, max_groups)
        return {
            "basis_count": 3,
            "hidden_dim": hidden_dim,
            "groups": groups,
            "init": "identity_curvature_odd_saturation",
        }
    if activation == "rational_basis_k2_per_channel":
        hidden_dim = int(ffn_dim)
        return {
            "basis_count": 2,
            "hidden_dim": hidden_dim,
            "groups": hidden_dim,
            "init": "identity_curvature",
        }
    return None


def _rational_basis_compute(v, numerator, denominator, groups, eps):
    hidden_dim = v.shape[-1]
    if hidden_dim % groups != 0:
        raise ValueError("Rational basis hidden_dim must be divisible by groups")

    basis_count = numerator.shape[0]
    width_per_group = hidden_dim // groups
    shape = v.shape
    grouped = v.view(*shape[:-1], groups, width_per_group)
    scale = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + eps)
    t = (grouped * scale).unsqueeze(-3)

    coeff_shape = (1,) * (grouped.dim() - 2) + (basis_count, groups, 1)
    a0 = numerator[:, :, 0].view(coeff_shape)
    a1 = numerator[:, :, 1].view(coeff_shape)
    a2 = numerator[:, :, 2].view(coeff_shape)
    a3 = numerator[:, :, 3].view(coeff_shape)
    a4 = numerator[:, :, 4].view(coeff_shape)
    a5 = numerator[:, :, 5].view(coeff_shape)
    b1 = denominator[:, :, 0].view(coeff_shape)
    b2 = denominator[:, :, 1].view(coeff_shape)
    b3 = denominator[:, :, 2].view(coeff_shape)
    b4 = denominator[:, :, 3].view(coeff_shape)

    t2 = t.square()
    t3 = t2 * t
    t4 = t2.square()
    t5 = t4 * t
    rational_num = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
    rational_den = 1.0 + (b1 * t).abs() + (b2 * t2).abs() + (b3 * t3).abs() + (b4 * t4).abs()
    basis = rational_num / rational_den
    return basis.reshape(*shape[:-1], basis_count * hidden_dim)


class _RationalBasisFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, numerator, denominator, groups, eps):
        ctx.groups = int(groups)
        ctx.eps = float(eps)
        ctx.save_for_backward(v, numerator, denominator)
        with torch.no_grad():
            return _rational_basis_compute(v, numerator, denominator, ctx.groups, ctx.eps)

    @staticmethod
    def backward(ctx, grad_output):
        v, numerator, denominator = ctx.saved_tensors
        with torch.enable_grad():
            v_replay = v.detach().requires_grad_(True)
            numerator_replay = numerator.detach().requires_grad_(True)
            denominator_replay = denominator.detach().requires_grad_(True)
            replay = _rational_basis_compute(
                v_replay,
                numerator_replay,
                denominator_replay,
                ctx.groups,
                ctx.eps,
            )
            grad_v, grad_num, grad_den = torch.autograd.grad(
                replay,
                (v_replay, numerator_replay, denominator_replay),
                grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        return grad_v, grad_num, grad_den, None, None


class RationalBasisExpansion(nn.Module):
    def __init__(self, hidden_dim, basis_count, groups, init, eps=1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.basis_count = int(basis_count)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RationalBasisExpansion requires hidden_dim divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.numerator = nn.Parameter(torch.zeros(self.basis_count, self.groups, 6))
        self.denominator = nn.Parameter(torch.zeros(self.basis_count, self.groups, 4))
        self.reset_parameters(init)

    def reset_parameters(self, init):
        with torch.no_grad():
            self.numerator.zero_()
            self.denominator.zero_()
            self.numerator[0, :, 1] = 1.0
            if self.basis_count >= 2:
                self.numerator[1, :, 0] = -1.0
                self.numerator[1, :, 2] = 1.0
                self.denominator[1, :, 1] = 1.0
            if self.basis_count >= 3:
                self.numerator[2, :, 1] = 1.0
                self.denominator[2, :, 1] = 1.0
            if init == "identity" and self.basis_count != 1:
                raise ValueError("identity init requires one basis")

    def forward(self, v):
        return _RationalBasisFunction.apply(v, self.numerator, self.denominator, self.groups, self.eps)


class RationalBasisMLP(nn.Module):
    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, rational_basis_eps):
        super().__init__()
        settings = rational_basis_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown rational basis activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.basis_count = settings["basis_count"]
        self.groups = settings["groups"]
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.rational_basis = RationalBasisExpansion(
            self.hidden_dim,
            self.basis_count,
            self.groups,
            settings["init"],
            eps=rational_basis_eps,
        )
        self.out_proj = nn.Linear(self.basis_count * self.hidden_dim, dim, bias=False)

    def forward(self, x):
        v = self.in_proj(x)
        z = self.rational_basis(v)
        return self.out_proj(z)


def rqm_settings(activation, d_model, ffn_dim, group_size, max_groups):
    if activation not in RQM_ACTIVATIONS:
        return None

    if activation == "rqm_ffn_wide_lowrank":
        hidden_dim = int(ffn_dim)
        rank = max(1, int(d_model) // 3)
    elif activation == "rqm_ffn_narrow_highrank":
        hidden_dim = int(ffn_dim) // 2
        rank = int(d_model)
    else:
        hidden_dim = (3 * int(ffn_dim)) // 4
        rank = (2 * int(d_model)) // 3

    if activation == "rqm_ffn_shared128":
        coeff_groups = resolve_group_count(rank, 128, rank)
    else:
        coeff_groups = rank

    return {
        "hidden_dim": hidden_dim,
        "rank": rank,
        "latent_groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "coeff_groups": coeff_groups,
        "kappa": 0.50 if activation == "rqm_ffn_kappa05" else 0.25,
        "beta": 0.05 if activation == "rqm_ffn_beta005" else 0.10,
    }


def _rqm_interaction_compute(s, t, numerator, denominator, coeff_groups):
    rank = s.shape[-1]
    if rank % coeff_groups != 0:
        raise ValueError("RQM rank must be divisible by coefficient groups")
    width_per_group = rank // coeff_groups
    shape = s.shape
    s = s.view(*shape[:-1], coeff_groups, width_per_group)
    t = t.view(*shape[:-1], coeff_groups, width_per_group)

    coeff_shape = (1,) * (s.dim() - 2) + (coeff_groups, 1)
    a10 = numerator[:, 0].view(coeff_shape)
    a01 = numerator[:, 1].view(coeff_shape)
    a11 = numerator[:, 2].view(coeff_shape)
    a20 = numerator[:, 3].view(coeff_shape)
    a02 = numerator[:, 4].view(coeff_shape)
    a30 = numerator[:, 5].view(coeff_shape)
    a03 = numerator[:, 6].view(coeff_shape)
    b10 = denominator[:, 0].view(coeff_shape)
    b01 = denominator[:, 1].view(coeff_shape)
    b11 = denominator[:, 2].view(coeff_shape)
    b20 = denominator[:, 3].view(coeff_shape)
    b02 = denominator[:, 4].view(coeff_shape)

    st = s * t
    s2 = s.square()
    t2 = t.square()
    s3 = s2 * s
    t3 = t2 * t
    rational_num = a10 * s + a01 * t + a11 * st + a20 * s2 + a02 * t2 + a30 * s3 + a03 * t3
    rational_den = 1.0 + (b10 * s).abs() + (b01 * t).abs() + (b11 * st).abs() + (b20 * s2).abs() + (b02 * t2).abs()
    return (rational_num / rational_den).reshape(shape)


class _RQMInteractionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, t, numerator, denominator, coeff_groups):
        ctx.coeff_groups = int(coeff_groups)
        ctx.save_for_backward(s, t, numerator, denominator)
        with torch.no_grad():
            return _rqm_interaction_compute(s, t, numerator, denominator, ctx.coeff_groups)

    @staticmethod
    def backward(ctx, grad_output):
        s, t, numerator, denominator = ctx.saved_tensors
        with torch.enable_grad():
            s_replay = s.detach().requires_grad_(True)
            t_replay = t.detach().requires_grad_(True)
            numerator_replay = numerator.detach().requires_grad_(True)
            denominator_replay = denominator.detach().requires_grad_(True)
            replay = _rqm_interaction_compute(
                s_replay,
                t_replay,
                numerator_replay,
                denominator_replay,
                ctx.coeff_groups,
            )
            grads = torch.autograd.grad(
                replay,
                (s_replay, t_replay, numerator_replay, denominator_replay),
                grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        return grads[0], grads[1], grads[2], grads[3], None


class RQMInteraction(nn.Module):
    def __init__(self, rank, coeff_groups, kappa=0.25, beta=0.10):
        super().__init__()
        self.rank = int(rank)
        self.coeff_groups = int(coeff_groups)
        if self.rank % self.coeff_groups != 0:
            raise ValueError("RQM rank must be divisible by coefficient groups")
        self.numerator = nn.Parameter(torch.zeros(self.coeff_groups, 7))
        self.denominator = nn.Parameter(torch.zeros(self.coeff_groups, 5))
        self.reset_parameters(kappa, beta)

    def reset_parameters(self, kappa, beta):
        with torch.no_grad():
            self.numerator.zero_()
            self.denominator.zero_()
            linear = 1.0 / math.sqrt(2.0)
            self.numerator[:, 0] = linear
            self.numerator[:, 1] = linear
            self.numerator[:, 2] = float(kappa)
            self.denominator[:, 3] = float(beta)
            self.denominator[:, 4] = float(beta)

    def forward(self, s, t):
        return _RQMInteractionFunction.apply(s, t, self.numerator, self.denominator, self.coeff_groups)


class RationalQuotientMixerMLP(nn.Module):
    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps):
        super().__init__()
        settings = rqm_settings(activation, dim, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RQM activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.rank = settings["rank"]
        self.latent_groups = settings["latent_groups"]
        if self.hidden_dim % self.latent_groups != 0:
            raise ValueError("RQM hidden_dim must be divisible by latent groups")
        self.width_per_group = self.hidden_dim // self.latent_groups
        self.eps = float(eps)

        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.mix_a = nn.Linear(self.hidden_dim, self.rank, bias=False)
        self.mix_b = nn.Linear(self.hidden_dim, self.rank, bias=False)
        self.rqm_interaction = RQMInteraction(
            self.rank,
            settings["coeff_groups"],
            kappa=settings["kappa"],
            beta=settings["beta"],
        )
        self.mix_c = nn.Linear(self.rank, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.layer_scale = nn.Parameter(torch.tensor(self.rank ** -0.5, dtype=torch.float32))

    def _group_rms_unit(self, v):
        shape = v.shape
        grouped = v.view(*shape[:-1], self.latent_groups, self.width_per_group)
        scale = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        return (grouped * scale).reshape(shape)

    def forward(self, x):
        v = self.in_proj(x)
        v = self._group_rms_unit(v)
        s = self.mix_a(v)
        t = self.mix_b(v)
        eta = self.rqm_interaction(s, t)
        u = self.mix_c(eta)
        z = v + self.layer_scale.to(dtype=u.dtype) * u
        return self.out_proj(z)


def rkm_settings(activation, d_model, ffn_dim, group_size, max_groups):
    if activation not in RKM_ACTIVATIONS:
        return None

    if activation == "rkm_ffn":
        query_rank = 128
        experts = 16
        expert_rank = 22
    elif activation == "rkm_ffn_more_regions":
        query_rank = 96
        experts = 32
        expert_rank = 11
    elif activation == "rkm_ffn_fewer_regions":
        query_rank = 192
        experts = 8
        expert_rank = 43
    else:
        raise ValueError(f"unknown RKM activation {activation}")

    hidden_dim = int(ffn_dim)
    return {
        "hidden_dim": hidden_dim,
        "query_rank": int(query_rank),
        "experts": int(experts),
        "expert_rank": int(expert_rank),
        "latent_groups": resolve_group_count(hidden_dim, group_size, max_groups),
    }


class RationalKernelMixtureMLP(nn.Module):
    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps):
        super().__init__()
        settings = rkm_settings(activation, dim, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RKM activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.query_rank = settings["query_rank"]
        self.experts = settings["experts"]
        self.expert_rank = settings["expert_rank"]
        self.latent_groups = settings["latent_groups"]
        if self.hidden_dim % self.latent_groups != 0:
            raise ValueError("RKM hidden_dim must be divisible by latent groups")
        self.width_per_group = self.hidden_dim // self.latent_groups
        self.eps = float(eps)
        self.routing_eps = 1e-6

        self.value_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.query_proj = nn.Linear(dim, self.query_rank, bias=False)
        self.expert_a = nn.Parameter(torch.empty(self.experts, self.hidden_dim, self.expert_rank))
        self.expert_b = nn.Parameter(torch.empty(self.experts, self.expert_rank, self.hidden_dim))
        self.rkm_centers = nn.Parameter(torch.empty(self.experts, self.query_rank))
        self.rkm_gamma_sqrt = nn.Parameter(torch.ones(self.experts, self.query_rank))
        self.rkm_tau_sqrt = nn.Parameter(torch.ones(self.experts))
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.layer_scale = nn.Parameter(torch.tensor(self.expert_rank ** -0.5, dtype=torch.float32))
        self.reset_rkm_parameters()

    def reset_rkm_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.expert_a, mean=0.0, std=0.02)
            nn.init.normal_(self.expert_b, mean=0.0, std=0.02)
            self.rkm_centers.normal_()
            center_scale = torch.rsqrt(self.rkm_centers.square().mean(dim=-1, keepdim=True) + self.eps)
            self.rkm_centers.mul_(center_scale)
            self.rkm_gamma_sqrt.fill_(1.0)
            self.rkm_tau_sqrt.fill_(1.0)

    def _group_rms_unit(self, v):
        shape = v.shape
        grouped = v.view(*shape[:-1], self.latent_groups, self.width_per_group)
        scale = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        return (grouped * scale).reshape(shape)

    def _query_rms_unit(self, q):
        return q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + self.eps)

    def _routing_weights(self, q):
        gamma = self.rkm_gamma_sqrt.square() + self.routing_eps
        tau = self.rkm_tau_sqrt.square() + self.routing_eps
        diff = q.unsqueeze(-2) - self.rkm_centers.view(*((1,) * (q.dim() - 1)), self.experts, self.query_rank)
        weighted_dist = (gamma.view(*((1,) * (q.dim() - 1)), self.experts, self.query_rank) * diff.square()).sum(dim=-1)
        kappa = tau.view(*((1,) * (q.dim() - 1)), self.experts) / (self.routing_eps + weighted_dist)
        return kappa / (kappa.sum(dim=-1, keepdim=True) + self.routing_eps)

    def forward(self, x):
        v = self._group_rms_unit(self.value_proj(x))
        q = self._query_rms_unit(self.query_proj(x))
        pi = self._routing_weights(q)
        hidden_rank = torch.einsum("...m,emr->...er", v, self.expert_a)
        mixed_rank = hidden_rank * pi.unsqueeze(-1)
        u = torch.einsum("...er,erm->...m", mixed_rank, self.expert_b)
        z = v + self.layer_scale.to(dtype=u.dtype) * u
        return self.out_proj(z)


def rapm_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RAPM_ACTIVATIONS:
        return None

    hidden_dim = int(ffn_dim)
    if activation == "rapm_ffn_per_channel":
        groups = hidden_dim
    else:
        groups = resolve_group_count(hidden_dim, group_size, max_groups)

    return {
        "hidden_dim": hidden_dim,
        "groups": groups,
        "kappa": 0.50 if activation == "rapm_ffn_kappa05" else 0.25,
        "beta": 0.05 if activation == "rapm_ffn_beta005" else 0.10,
    }


def _rapm_compute(left, right, numerator, denominator, groups, eps):
    hidden_dim = left.shape[-1]
    if hidden_dim % groups != 0:
        raise ValueError("RAPM hidden_dim must be divisible by groups")
    width_per_group = hidden_dim // groups
    shape = left.shape
    left_grouped = left.view(*shape[:-1], groups, width_per_group)
    right_grouped = right.view(*shape[:-1], groups, width_per_group)

    rms = torch.sqrt(0.5 * (left_grouped.square().mean(dim=-1, keepdim=True) + right_grouped.square().mean(dim=-1, keepdim=True)) + eps)
    s = left_grouped / rms
    t = right_grouped / rms

    coeff_shape = (1,) * (s.dim() - 2) + (groups, 1)
    a10 = numerator[:, 0].view(coeff_shape)
    a01 = numerator[:, 1].view(coeff_shape)
    a11 = numerator[:, 2].view(coeff_shape)
    a20 = numerator[:, 3].view(coeff_shape)
    a02 = numerator[:, 4].view(coeff_shape)
    a30 = numerator[:, 5].view(coeff_shape)
    a03 = numerator[:, 6].view(coeff_shape)
    b10 = denominator[:, 0].view(coeff_shape)
    b01 = denominator[:, 1].view(coeff_shape)
    b11 = denominator[:, 2].view(coeff_shape)
    b20 = denominator[:, 3].view(coeff_shape)
    b02 = denominator[:, 4].view(coeff_shape)

    st = s * t
    s2 = s.square()
    t2 = t.square()
    s3 = s2 * s
    t3 = t2 * t
    rational_num = a10 * s + a01 * t + a11 * st + a20 * s2 + a02 * t2 + a30 * s3 + a03 * t3
    rational_den = 1.0 + (b10 * s).abs() + (b01 * t).abs() + (b11 * st).abs() + (b20 * s2).abs() + (b02 * t2).abs()
    return (rms * rational_num / rational_den).reshape(shape)


class _RAPMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, numerator, denominator, groups, eps):
        ctx.groups = int(groups)
        ctx.eps = float(eps)
        ctx.save_for_backward(left, right, numerator, denominator)
        with torch.no_grad():
            return _rapm_compute(left, right, numerator, denominator, ctx.groups, ctx.eps)

    @staticmethod
    def backward(ctx, grad_output):
        left, right, numerator, denominator = ctx.saved_tensors
        with torch.enable_grad():
            left_replay = left.detach().requires_grad_(True)
            right_replay = right.detach().requires_grad_(True)
            numerator_replay = numerator.detach().requires_grad_(True)
            denominator_replay = denominator.detach().requires_grad_(True)
            replay = _rapm_compute(
                left_replay,
                right_replay,
                numerator_replay,
                denominator_replay,
                ctx.groups,
                ctx.eps,
            )
            grads = torch.autograd.grad(
                replay,
                (left_replay, right_replay, numerator_replay, denominator_replay),
                grad_output,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        return grads[0], grads[1], grads[2], grads[3], None, None


class RationalAmplitudePairActivation(nn.Module):
    def __init__(self, groups, kappa=0.25, beta=0.10, eps=1e-6):
        super().__init__()
        self.groups = int(groups)
        self.eps = float(eps)
        self.numerator = nn.Parameter(torch.zeros(self.groups, 7))
        self.denominator = nn.Parameter(torch.zeros(self.groups, 5))
        self.reset_parameters(kappa, beta)

    def reset_parameters(self, kappa, beta):
        with torch.no_grad():
            self.numerator.zero_()
            self.denominator.zero_()
            linear = 1.0 / math.sqrt(2.0)
            self.numerator[:, 0] = linear
            self.numerator[:, 1] = linear
            self.numerator[:, 2] = float(kappa)
            self.denominator[:, 3] = float(beta)
            self.denominator[:, 4] = float(beta)

    def forward(self, left, right):
        return _RAPMFunction.apply(left, right, self.numerator, self.denominator, self.groups, self.eps)


class RationalAmplitudePairMixerMLP(nn.Module):
    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps):
        super().__init__()
        settings = rapm_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RAPM activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.left = nn.Linear(dim, self.hidden_dim, bias=False)
        self.right = nn.Linear(dim, self.hidden_dim, bias=False)
        self.rapm_activation = RationalAmplitudePairActivation(
            self.groups,
            kappa=settings["kappa"],
            beta=settings["beta"],
            eps=eps,
        )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        hidden = self.rapm_activation(left, right)
        return self.out_proj(hidden)


def rpf_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RPF_ACTIVATIONS:
        return None
    hidden_dim = (3 * int(ffn_dim)) // 4
    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "curvature": 0.50 if activation == "rpf_ffn_curv05" else 0.25,
    }


class RationalPolarizationActivation(nn.Module):
    def __init__(self, hidden_dim, groups, curvature=0.25):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RPF hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.numerator = nn.Parameter(torch.zeros(self.groups, 6))
        self.denominator = nn.Parameter(torch.zeros(self.groups, 4))
        self.reset_parameters(curvature)

    def reset_parameters(self, curvature):
        # R(t) = t + c * (t^2 - 1) / (1 + t^2)
        #      = (-c + t + c t^2 + t^3) / (1 + t^2).
        with torch.no_grad():
            self.numerator.zero_()
            self.denominator.zero_()
            self.numerator[:, 0] = -float(curvature)
            self.numerator[:, 1] = 1.0
            self.numerator[:, 2] = float(curvature)
            self.numerator[:, 3] = 1.0
            self.denominator[:, 1] = 1.0

    def forward(self, x):
        from rational_opt import rational_version_a5_4

        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {x.size(-1)}")
        if self.groups == 1:
            return rational_version_a5_4(x, self.numerator[0], self.denominator[0])
        chunks = torch.split(x, self.width_per_group, dim=-1)
        outputs = [
            rational_version_a5_4(chunk, self.numerator[idx], self.denominator[idx])
            for idx, chunk in enumerate(chunks)
        ]
        return torch.cat(outputs, dim=-1)


class RationalPolarizationMLP(nn.Module):
    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups):
        super().__init__()
        settings = rpf_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RPF activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.left = nn.Linear(dim, self.hidden_dim, bias=False)
        self.right = nn.Linear(dim, self.hidden_dim, bias=False)
        self.rpf_plus = RationalPolarizationActivation(
            self.hidden_dim,
            self.groups,
            curvature=settings["curvature"],
        )
        self.rpf_minus = RationalPolarizationActivation(
            self.hidden_dim,
            self.groups,
            curvature=settings["curvature"],
        )
        self.out_proj = nn.Linear(2 * self.hidden_dim, dim, bias=False)

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        scale = 1.0 / math.sqrt(2.0)
        plus = (left + right) * scale
        minus = (left - right) * scale
        hidden = torch.cat((self.rpf_plus(plus), self.rpf_minus(minus)), dim=-1)
        return self.out_proj(hidden)


def rpb_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RPB_ACTIVATIONS:
        return None

    if activation == "rpb_ffn_diff":
        hidden_dim = (3 * int(ffn_dim)) // 5
        basis_count = 3
        mode = "diff"
        normalize = False
    elif activation == "rpb_ffn_norm":
        hidden_dim = int(ffn_dim) // 2
        basis_count = 4
        mode = "separate"
        normalize = True
    else:
        hidden_dim = int(ffn_dim) // 2
        basis_count = 4
        mode = "separate"
        normalize = False

    return {
        "hidden_dim": hidden_dim,
        "basis_count": basis_count,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "mode": mode,
        "normalize": normalize,
    }


class RationalPolarizedBasisActivation(nn.Module):
    def __init__(self, hidden_dim, groups, init):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RPB hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.numerator = nn.Parameter(torch.zeros(self.groups, 6))
        self.denominator = nn.Parameter(torch.zeros(self.groups, 4))
        self.reset_parameters(init)

    def reset_parameters(self, init):
        with torch.no_grad():
            self.numerator.zero_()
            self.denominator.zero_()
            if init == "identity":
                self.numerator[:, 1] = 1.0
            elif init == "bounded_even":
                # R(t) = (t^2 - 1) / (1 + t^2). Paired as R(p)-R(n),
                # this gives a bounded rational route to the polarization term.
                self.numerator[:, 0] = -1.0
                self.numerator[:, 2] = 1.0
                self.denominator[:, 1] = 1.0
            elif init == "bounded_odd":
                # R(t) = t / (1 + t^2). This adds a sign-sensitive bounded
                # correction without replacing the value stream.
                self.numerator[:, 1] = 1.0
                self.denominator[:, 1] = 1.0
            else:
                raise ValueError(f"unknown RPB init {init}")

    def forward(self, x):
        from rational_opt import rational_version_a5_4

        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {x.size(-1)}")
        if self.groups == 1:
            return rational_version_a5_4(x, self.numerator[0], self.denominator[0])
        chunks = torch.split(x, self.width_per_group, dim=-1)
        outputs = [
            rational_version_a5_4(chunk, self.numerator[idx], self.denominator[idx])
            for idx, chunk in enumerate(chunks)
        ]
        return torch.cat(outputs, dim=-1)


class NormalizedRationalValueResidual(nn.Module):
    def __init__(self, hidden_dim, groups, init, eps=1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("normalized value residual hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.rational = RationalPolarizedBasisActivation(hidden_dim, groups, init)

    def forward(self, value):
        if value.size(-1) != self.hidden_dim:
            raise ValueError(f"expected value dimension {self.hidden_dim}, got {value.size(-1)}")
        shape = value.shape
        grouped = value.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (grouped / rms).reshape(shape)
        residual = self.rational(normalized)
        return (residual.view_as(grouped) * rms).reshape(shape)


class ConditionalRationalValueResidual(nn.Module):
    def __init__(self, hidden_dim, groups, basis_count=3, eps=1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        self.basis_count = int(basis_count)
        if self.basis_count != 3:
            raise ValueError("ConditionalRationalValueResidual currently expects three bases")
        if self.hidden_dim % self.groups != 0:
            raise ValueError("conditional value residual hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.numerator = nn.Parameter(torch.zeros(self.basis_count, self.groups, 6))
        self.denominator = nn.Parameter(torch.zeros(self.basis_count, self.groups, 4))
        self.base_mix = nn.Parameter(torch.zeros(self.groups, self.basis_count))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.numerator.zero_()
            self.denominator.zero_()
            self.base_mix.zero_()

            # R_1(t) = (t^2 - 1) / (1 + t^2).
            self.numerator[0, :, 0] = -1.0
            self.numerator[0, :, 2] = 1.0
            self.denominator[0, :, 1] = 1.0

            # R_2(t) = t / (1 + t^2).
            self.numerator[1, :, 1] = 1.0
            self.denominator[1, :, 1] = 1.0

            # R_3(t) = t^3 / (1 + t^2).
            self.numerator[2, :, 3] = 1.0
            self.denominator[2, :, 1] = 1.0

            # Start from a stable bounded-even value residual, then let the
            # token-conditioned projection move the mixture away from it.
            self.base_mix[:, 0] = 1.0

    def forward(self, value, condition):
        if value.size(-1) != self.hidden_dim:
            raise ValueError(f"expected value dimension {self.hidden_dim}, got {value.size(-1)}")
        if condition.size(-1) != self.groups * self.basis_count:
            raise ValueError(
                f"expected condition dimension {self.groups * self.basis_count}, got {condition.size(-1)}"
            )

        shape = value.shape
        grouped = value.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = (grouped / rms).unsqueeze(-3)

        coeff_shape = (1,) * (grouped.dim() - 2) + (self.basis_count, self.groups, 1)
        a0 = self.numerator[:, :, 0].view(coeff_shape)
        a1 = self.numerator[:, :, 1].view(coeff_shape)
        a2 = self.numerator[:, :, 2].view(coeff_shape)
        a3 = self.numerator[:, :, 3].view(coeff_shape)
        a4 = self.numerator[:, :, 4].view(coeff_shape)
        a5 = self.numerator[:, :, 5].view(coeff_shape)
        b1 = self.denominator[:, :, 0].view(coeff_shape)
        b2 = self.denominator[:, :, 1].view(coeff_shape)
        b3 = self.denominator[:, :, 2].view(coeff_shape)
        b4 = self.denominator[:, :, 3].view(coeff_shape)

        t2 = t.square()
        t3 = t2 * t
        t4 = t2.square()
        t5 = t4 * t
        rational_num = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
        rational_den = 1.0 + (b1 * t).abs() + (b2 * t2).abs() + (b3 * t3).abs() + (b4 * t4).abs()
        basis = rational_num / rational_den

        dynamic_mix = condition.view(*shape[:-1], self.groups, self.basis_count)
        dynamic_mix = torch.tanh(dynamic_mix).transpose(-1, -2).unsqueeze(-1)
        base_mix = self.base_mix.transpose(0, 1).view(
            *((1,) * (value.dim() - 1)),
            self.basis_count,
            self.groups,
            1,
        )
        residual = (basis * (base_mix.to(dtype=value.dtype) + dynamic_mix)).sum(dim=-3)
        return (rms * residual).reshape(shape)


class RationalPolarizedBasisMLP(nn.Module):
    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps):
        super().__init__()
        settings = rpb_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RPB activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.basis_count = settings["basis_count"]
        self.mode = settings["mode"]
        self.normalize = settings["normalize"]
        self.eps = float(eps)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RPB hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups

        self.left = nn.Linear(dim, self.hidden_dim, bias=False)
        self.right = nn.Linear(dim, self.hidden_dim, bias=False)
        self.rpb_odd = RationalPolarizedBasisActivation(self.hidden_dim, self.groups, "identity")
        self.rpb_even = RationalPolarizedBasisActivation(self.hidden_dim, self.groups, "bounded_even")
        self.out_proj = nn.Linear(self.basis_count * self.hidden_dim, dim, bias=False)

    def _group_rms_unit(self, v):
        shape = v.shape
        grouped = v.view(*shape[:-1], self.groups, self.width_per_group)
        scale = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        return (grouped * scale).reshape(shape)

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        scale = 1.0 / math.sqrt(2.0)
        plus = (left + right) * scale
        minus = (left - right) * scale
        if self.normalize:
            plus = self._group_rms_unit(plus)
            minus = self._group_rms_unit(minus)

        odd_plus = self.rpb_odd(plus)
        odd_minus = self.rpb_odd(minus)
        even_plus = self.rpb_even(plus)
        even_minus = self.rpb_even(minus)
        if self.mode == "diff":
            hidden = torch.cat((odd_plus, odd_minus, even_plus - even_minus), dim=-1)
        else:
            hidden = torch.cat((odd_plus, odd_minus, even_plus, even_minus), dim=-1)
        return self.out_proj(hidden)


def rwf_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RWF_ACTIVATIONS:
        return None
    hidden_dim = (3 * int(ffn_dim)) // 2
    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "init": "gelu" if activation == "rwf_ffn_gelu" else "silu",
    }


class RationalWideFFN(nn.Module):
    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups):
        super().__init__()
        settings = rwf_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RWF activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        from rational_opt import RationalGroupedVersionA5_4

        self.rwf_act = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init=settings["init"],
            fit_range=5.0,
            group_size=rational_group_size,
            max_groups=rational_max_groups,
            groups=self.groups,
        )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out_proj(self.rwf_act(self.in_proj(x)))


def rmb_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RMB_ACTIVATIONS:
        return None
    if activation == "rmb_k2_ffn":
        basis_count = 2
        hidden_dim = int(ffn_dim)
        mix_init = ((1.0, 0.0, 0.4),)
    elif activation == "rmb_k3_ffn":
        basis_count = 3
        hidden_dim = (3 * int(ffn_dim)) // 4
        mix_init = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    else:
        raise ValueError(f"unknown RMB activation {activation}")
    return {
        "hidden_dim": hidden_dim,
        "basis_count": basis_count,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "mix_init": mix_init,
    }


class RationalMomentBasisFFN(nn.Module):
    """No-GLU rational basis FFN with matched SwiGLU matrix budget.

    One projection creates v. The activation normalizes v by channel group and
    emits rational basis features directly. There is no gate projection, no up
    branch, and no elementwise product between branches.
    """

    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps=1e-6):
        super().__init__()
        settings = rmb_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RMB activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.basis_count = settings["basis_count"]
        self.groups = settings["groups"]
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RMB hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        from rational_opt import RationalGroupedVersionA5_4

        self.base = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init="silu",
            fit_range=5.0,
            group_size=rational_group_size,
            max_groups=rational_max_groups,
            groups=self.groups,
        )
        mix = torch.tensor(settings["mix_init"], dtype=torch.float32)
        if mix.shape != (self.basis_count - 1, 3):
            raise ValueError("RMB mix_init shape does not match basis_count")
        self.basis_mix = nn.Parameter(mix.unsqueeze(1).repeat(1, self.groups, 1))
        self.out_proj = nn.Linear(self.basis_count * self.hidden_dim, dim, bias=False)

    def forward(self, x):
        value = self.in_proj(x)
        shape = value.shape
        grouped = value.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = grouped / rms
        flat_t = t.reshape(shape)

        base = self.base(flat_t).view_as(grouped)
        t2 = t.square()
        denom = 1.0 + t2
        even = (t2 - 1.0) / denom
        odd = t / denom
        cubic = t * t2 / denom
        primitive = torch.stack((even, odd, cubic), dim=-2)
        mix = self.basis_mix.view(
            *((1,) * (grouped.dim() - 2)),
            self.basis_count - 1,
            self.groups,
            3,
        ).to(dtype=value.dtype)
        extra = torch.einsum("...bgk,...gkw->...bgw", mix, primitive)
        features = torch.cat((base.unsqueeze(-3), extra), dim=-3)
        features = (features * rms.unsqueeze(-3)).transpose(-3, -2)
        return self.out_proj(features.reshape(*shape[:-1], self.basis_count * self.hidden_dim))


def rma_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RMA_ACTIVATIONS:
        return None
    hidden_dim = (3 * int(ffn_dim)) // 2
    pair_init = None
    pair_beta = 0.25
    normalize = True
    linear_skip_init = 0.0
    radial_init = 0.0
    basis_center = False
    basis_mode = "standard"
    output_norm_init = None
    base_init = "silu"
    input_affine = False
    moment_affine_init = 0.0
    basis_den_scale_init = 1.0
    if activation == "rma_ffn":
        basis_init = (0.0, 0.0, 0.0)
        coeff_limit = 0.50
    elif activation == "rma_ffn_curv":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
    elif activation == "rma_ffn_strong":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
    elif activation == "rma_raw_ffn_curv":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        normalize = False
    elif activation == "rma_gelu_curv_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        base_init = "gelu"
    elif activation == "rma_gelu_strong_ffn":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        base_init = "gelu"
    elif activation == "rma_relu_curv_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        base_init = "relu"
    elif activation == "rma_identity_curv_ffn":
        basis_init = (0.15, 0.00, 0.05)
        coeff_limit = 0.75
        base_init = "identity"
    elif activation == "rma_shift_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        input_affine = True
    elif activation == "rma_gelu_shift_ffn":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        base_init = "gelu"
        input_affine = True
    elif activation == "rma_identity_shift_ffn":
        basis_init = (0.15, 0.00, 0.05)
        coeff_limit = 0.75
        base_init = "identity"
        input_affine = True
    elif activation == "rma_silu_shift_strong_ffn":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        input_affine = True
    elif activation == "rma_silu_momaffine_ffn":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        input_affine = True
        moment_affine_init = 0.20
    elif activation == "rma_gelu_momaffine_ffn":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        base_init = "gelu"
        input_affine = True
        moment_affine_init = 0.20
    elif activation == "rma_identity_momaffine_ffn":
        basis_init = (0.15, 0.00, 0.05)
        coeff_limit = 0.75
        base_init = "identity"
        input_affine = True
        moment_affine_init = 0.20
    elif activation == "rma_silu_denwide_shift_ffn":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        input_affine = True
        basis_den_scale_init = 0.50
    elif activation == "rma_silu_dennarrow_shift_ffn":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        input_affine = True
        basis_den_scale_init = 2.00
    elif activation == "rma_skip_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        linear_skip_init = 0.25
    elif activation == "rma_raw_skip_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        normalize = False
        linear_skip_init = 0.25
    elif activation == "rma_radial_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        radial_init = 0.08
    elif activation == "rma_center_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        basis_center = True
    elif activation == "rma_center_strong_ffn":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        basis_center = True
    elif activation == "rma_hermite_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        basis_mode = "hermite"
    elif activation == "rma_center_hermite_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.60
        basis_center = True
        basis_mode = "hermite"
    elif activation == "rma_divnorm25_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        output_norm_init = 0.25
    elif activation == "rma_divnorm50_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        output_norm_init = 0.50
    elif activation == "rma_divnorm75_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.50
        output_norm_init = 0.75
    elif activation == "rma_divnorm_strong_ffn":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        output_norm_init = 0.25
    elif activation == "rma_pair_ffn":
        basis_init = (0.05, 0.00, 0.02)
        coeff_limit = 0.60
        pair_init = ((0.04, 0.02), (-0.04, 0.02))
    elif activation == "rma_pair_ffn_strong":
        basis_init = (0.10, 0.02, 0.05)
        coeff_limit = 0.75
        pair_init = ((0.08, 0.04), (-0.08, 0.04))
    else:
        raise ValueError(f"unknown RMA activation {activation}")
    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "basis_init": basis_init,
        "coeff_limit": coeff_limit,
        "pair_init": pair_init,
        "pair_beta": pair_beta,
        "normalize": normalize,
        "linear_skip_init": linear_skip_init,
        "radial_init": radial_init,
        "basis_center": basis_center,
        "basis_mode": basis_mode,
        "output_norm_init": output_norm_init,
        "base_init": base_init,
        "input_affine": input_affine,
        "moment_affine_init": moment_affine_init,
        "basis_den_scale_init": basis_den_scale_init,
    }


class RationalMomentAdaptiveActivation(nn.Module):
    """Single-input rational activation with token/group-conditioned coefficients.

    This module is intentionally not a GLU: it receives only the single expanded
    hidden vector v and returns Phi(v). The adaptive coefficients are functions
    of moments of v's own channel group, so there is no gate branch and no
    elementwise product between two learned projections.
    """

    def __init__(
        self,
        hidden_dim,
        groups,
        basis_init,
        coeff_limit,
        group_size,
        max_groups,
        eps=1e-6,
        pair_init=None,
        pair_beta=0.25,
        normalize=True,
        linear_skip_init=0.0,
        radial_init=0.0,
        basis_center=False,
        basis_mode="standard",
        output_norm_init=None,
        base_init="silu",
        input_affine=False,
        moment_affine_init=0.0,
        basis_den_scale_init=1.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RMA hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.coeff_limit = float(coeff_limit)
        self.pair_beta = float(pair_beta)
        self.normalize = bool(normalize)
        self.basis_center = bool(basis_center)
        self.basis_mode = str(basis_mode)
        self.input_affine = bool(input_affine)
        self.input_shift = None
        self.input_log_scale = None
        if self.input_affine:
            self.input_shift = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))
            self.input_log_scale = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))
        self.input_moment_affine = None
        if float(moment_affine_init) != 0.0:
            self.input_moment_affine = nn.Parameter(torch.zeros(2, self.groups, 3, dtype=torch.float32))
            with torch.no_grad():
                self.input_moment_affine[0, :, 0] = -float(moment_affine_init)
                self.input_moment_affine[1, :, 1] = -0.5 * float(moment_affine_init)
        self.basis_den_log_scale = None
        if float(basis_den_scale_init) <= 0.0:
            raise ValueError("basis_den_scale_init must be positive")
        if abs(float(basis_den_scale_init) - 1.0) > 1e-12:
            self.basis_den_log_scale = nn.Parameter(
                torch.full((self.groups,), math.log(float(basis_den_scale_init)), dtype=torch.float32)
            )
        if self.basis_mode not in {"standard", "hermite"}:
            raise ValueError(f"unknown RMA basis mode {self.basis_mode}")
        self.output_norm_logit = None
        if output_norm_init is not None:
            probability = min(max(float(output_norm_init), 1e-4), 1.0 - 1e-4)
            self.output_norm_logit = nn.Parameter(
                torch.full((self.groups,), math.log(probability / (1.0 - probability)), dtype=torch.float32)
            )
        self.pair_logits = None
        from rational_opt import RationalGroupedVersionA5_4

        self.base = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init=base_init,
            fit_range=5.0,
            group_size=group_size,
            max_groups=max_groups,
            groups=self.groups,
        )
        init = torch.tensor(basis_init, dtype=torch.float32).view(3, 1).repeat(1, self.groups)
        init = torch.clamp(init / self.coeff_limit, -0.999, 0.999)
        self.basis_logits = nn.Parameter(torch.atanh(init))
        self.moment_mix = nn.Parameter(torch.zeros(3, self.groups, 3, dtype=torch.float32))
        self.linear_skip = nn.Parameter(torch.full((self.groups,), float(linear_skip_init), dtype=torch.float32))
        self.radial_scale = nn.Parameter(torch.full((self.groups,), float(radial_init), dtype=torch.float32))
        if pair_init is not None:
            if self.width_per_group % 2 != 0:
                raise ValueError("RMA pair activation requires an even group width")
            pair = torch.tensor(pair_init, dtype=torch.float32)
            if pair.shape != (2, 2):
                raise ValueError("pair_init must have shape (2 output channels, 2 pair bases)")
            pair = torch.clamp(pair / self.coeff_limit, -0.999, 0.999)
            self.pair_logits = nn.Parameter(torch.atanh(pair).repeat(self.groups, 1, 1))

    def forward(self, value):
        if value.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {value.size(-1)}")
        shape = value.shape
        grouped = value.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized_grouped = grouped / rms
        normalized = normalized_grouped.reshape(shape)

        norm_t2 = normalized_grouped.square()
        mean = normalized_grouped.mean(dim=-1)
        log_rms = torch.log(rms.squeeze(-1) + self.eps)
        skew = (normalized_grouped * norm_t2).mean(dim=-1)
        stats = torch.stack(
            (torch.tanh(mean), torch.tanh(log_rms), torch.tanh(0.5 * skew)),
            dim=-1,
        )

        activation_grouped = normalized_grouped if self.normalize else grouped
        if self.input_affine:
            affine_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
            input_scale = torch.exp(self.input_log_scale).view(affine_shape).to(dtype=value.dtype)
            input_shift = self.input_shift.view(affine_shape).to(dtype=value.dtype)
            activation_grouped = input_scale * activation_grouped + input_shift
        if self.input_moment_affine is not None:
            moment_affine = torch.einsum(
                "...gs,ogs->...go",
                stats,
                self.input_moment_affine.to(dtype=value.dtype),
            )
            dynamic_shift = moment_affine[..., 0].unsqueeze(-1)
            dynamic_log_scale = 0.25 * torch.tanh(moment_affine[..., 1].unsqueeze(-1))
            activation_grouped = torch.exp(dynamic_log_scale) * activation_grouped + dynamic_shift
        activation_flat = activation_grouped.reshape(shape)
        base = self.base(activation_flat).view_as(grouped)
        t2 = activation_grouped.square()
        if self.basis_den_log_scale is None:
            denom = 1.0 + t2
        else:
            den_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
            den_scale = torch.exp(self.basis_den_log_scale).view(den_shape).to(dtype=value.dtype)
            denom = 1.0 + den_scale * t2
        even = (t2 - 1.0) / denom
        odd = activation_grouped / denom
        if self.basis_mode == "hermite":
            cubic = (activation_grouped * t2 - 3.0 * activation_grouped) / denom
        else:
            cubic = activation_grouped * t2 / denom
        basis = torch.stack((even, odd, cubic), dim=-2)
        if self.basis_center:
            basis = basis - basis.mean(dim=-1, keepdim=True)

        dynamic = torch.einsum("...gs,bgs->...gb", stats, self.moment_mix.to(dtype=value.dtype))
        logits = self.basis_logits.transpose(0, 1).view(
            *((1,) * (grouped.dim() - 2)),
            self.groups,
            3,
        )
        coeff = self.coeff_limit * torch.tanh(logits.to(dtype=value.dtype) + dynamic)
        activated = base + (basis * coeff.unsqueeze(-1)).sum(dim=-2)
        linear_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
        activated = activated + self.linear_skip.view(linear_shape).to(dtype=value.dtype) * activation_grouped
        radial = self.radial_scale.view(linear_shape).to(dtype=value.dtype) * torch.tanh(log_rms).unsqueeze(-1)
        activated = activated * (1.0 + radial)

        if self.pair_logits is not None:
            pair_input = normalized_grouped.view(
                *shape[:-1],
                self.groups,
                self.width_per_group // 2,
                2,
            )
            left = pair_input[..., 0]
            right = pair_input[..., 1]
            pair_den = 1.0 + self.pair_beta * (left.square() + right.square())
            product = (left * right) / pair_den
            contrast = (left.square() - right.square()) / pair_den
            pair_basis = torch.stack((product, contrast), dim=-1)
            pair_coeff = self.coeff_limit * torch.tanh(self.pair_logits.to(dtype=value.dtype))
            pair_delta = torch.einsum("...gpb,gob->...gpo", pair_basis, pair_coeff)
            activated = activated.view(
                *shape[:-1],
                self.groups,
                self.width_per_group // 2,
                2,
            )
            activated = (activated + pair_delta).reshape(*shape[:-1], self.groups, self.width_per_group)

        if self.output_norm_logit is not None:
            activated_rms = torch.sqrt(activated.square().mean(dim=-1, keepdim=True) + self.eps)
            normed_activated = activated / activated_rms
            blend = torch.sigmoid(self.output_norm_logit).view(linear_shape).to(dtype=value.dtype)
            activated = activated * (1.0 - blend) + normed_activated * blend

        if self.normalize:
            activated = activated * rms
        return activated.reshape(shape)


class RationalMomentAdaptiveFFN(nn.Module):
    """Matched-budget no-GLU FFN using only a moment-adaptive rational activation."""

    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps=1e-6):
        super().__init__()
        settings = rma_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RMA activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.activation = RationalMomentAdaptiveActivation(
            self.hidden_dim,
            self.groups,
            settings["basis_init"],
            settings["coeff_limit"],
            rational_group_size,
            rational_max_groups,
            eps=eps,
            pair_init=settings["pair_init"],
            pair_beta=settings["pair_beta"],
            normalize=settings["normalize"],
            linear_skip_init=settings["linear_skip_init"],
            radial_init=settings["radial_init"],
            basis_center=settings["basis_center"],
            basis_mode=settings["basis_mode"],
            output_norm_init=settings["output_norm_init"],
            base_init=settings["base_init"],
            input_affine=settings["input_affine"],
            moment_affine_init=settings["moment_affine_init"],
            basis_den_scale_init=settings["basis_den_scale_init"],
        )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out_proj(self.activation(self.in_proj(x)))


def rda_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RDA_ACTIVATIONS:
        return None

    hidden_dim = (3 * int(ffn_dim)) // 2
    base_init = "silu"
    coeff_limit = 0.18
    denominator_limit = 0.12
    dynamic_init = 0.04
    input_affine = False
    moment_affine_init = 0.0
    if activation == "rda_silu_ffn":
        pass
    elif activation == "rda_silu_shift_ffn":
        input_affine = True
    elif activation == "rda_silu_momaffine_ffn":
        input_affine = True
        moment_affine_init = 0.20
    elif activation == "rda_gelu_momaffine_ffn":
        base_init = "gelu"
        input_affine = True
        moment_affine_init = 0.20
    elif activation == "rda_identity_momaffine_ffn":
        base_init = "identity"
        coeff_limit = 0.30
        denominator_limit = 0.18
        dynamic_init = 0.08
        input_affine = True
        moment_affine_init = 0.20
    else:
        raise ValueError(f"unknown RDA activation {activation}")

    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "base_init": base_init,
        "coeff_limit": coeff_limit,
        "denominator_limit": denominator_limit,
        "dynamic_init": dynamic_init,
        "input_affine": input_affine,
        "moment_affine_init": moment_affine_init,
    }


class RationalDynamicA5_4Activation(nn.Module):
    """Single-branch rational activation with token/group-conditioned A5/4 coefficients.

    The activation receives only one expanded hidden vector. Its coefficients are
    small functions of the same branch's group moments, so it is adaptive without
    a gate projection, up projection, or elementwise product between branches.
    """

    def __init__(
        self,
        hidden_dim,
        groups,
        base_init,
        coeff_limit,
        denominator_limit,
        dynamic_init,
        group_size,
        max_groups,
        eps=1e-6,
        input_affine=False,
        moment_affine_init=0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RDA hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.coeff_limit = float(coeff_limit)
        self.denominator_limit = float(denominator_limit)
        self.input_affine = bool(input_affine)
        self.input_shift = None
        self.input_log_scale = None
        if self.input_affine:
            self.input_shift = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))
            self.input_log_scale = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))
        self.input_moment_affine = None
        if float(moment_affine_init) != 0.0:
            self.input_moment_affine = nn.Parameter(torch.zeros(2, self.groups, 3, dtype=torch.float32))
            with torch.no_grad():
                self.input_moment_affine[0, :, 0] = -float(moment_affine_init)
                self.input_moment_affine[1, :, 1] = -0.5 * float(moment_affine_init)

        from rational_opt import RationalGroupedVersionA5_4

        self.base = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init=base_init,
            fit_range=5.0,
            group_size=group_size,
            max_groups=max_groups,
            groups=self.groups,
        )
        self.num_moment_mix = nn.Parameter(torch.zeros(self.groups, 6, 3, dtype=torch.float32))
        self.den_moment_mix = nn.Parameter(torch.zeros(self.groups, 4, 3, dtype=torch.float32))
        self.reset_dynamic_parameters(dynamic_init)

    def reset_dynamic_parameters(self, dynamic_init):
        with torch.no_grad():
            c = float(dynamic_init)
            self.num_moment_mix.zero_()
            self.den_moment_mix.zero_()
            # Bias, quadratic curvature, and cubic shoulder respond to mean/RMS/skew.
            self.num_moment_mix[:, 0, 0] = -c
            self.num_moment_mix[:, 2, 1] = c
            self.num_moment_mix[:, 3, 2] = 0.5 * c
            self.den_moment_mix[:, 1, 1] = 0.5 * c

    def forward(self, value):
        if value.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {value.size(-1)}")
        shape = value.shape
        grouped = value.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = grouped / rms

        t2_norm = t.square()
        mean = t.mean(dim=-1)
        log_rms = torch.log(rms.squeeze(-1) + self.eps)
        skew = (t * t2_norm).mean(dim=-1)
        stats = torch.stack(
            (torch.tanh(mean), torch.tanh(log_rms), torch.tanh(0.5 * skew)),
            dim=-1,
        )

        if self.input_affine:
            affine_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
            scale = torch.exp(self.input_log_scale).view(affine_shape).to(dtype=value.dtype)
            shift = self.input_shift.view(affine_shape).to(dtype=value.dtype)
            t = scale * t + shift
        if self.input_moment_affine is not None:
            moment_affine = torch.einsum(
                "...gs,ogs->...go",
                stats,
                self.input_moment_affine.to(dtype=value.dtype),
            )
            shift = moment_affine[..., 0].unsqueeze(-1)
            log_scale = 0.25 * torch.tanh(moment_affine[..., 1].unsqueeze(-1))
            t = torch.exp(log_scale) * t + shift

        delta_num = self.coeff_limit * torch.tanh(
            torch.einsum("...gs,gcs->...gc", stats, self.num_moment_mix.to(dtype=value.dtype))
        )
        delta_den = self.denominator_limit * torch.tanh(
            torch.einsum("...gs,gcs->...gc", stats, self.den_moment_mix.to(dtype=value.dtype))
        )
        coeff_shape = *((1,) * (grouped.dim() - 2)), self.groups, 6
        den_shape = *((1,) * (grouped.dim() - 2)), self.groups, 4
        numerator = self.base.numerator.view(coeff_shape).to(dtype=value.dtype) + delta_num
        denominator = self.base.denominator.view(den_shape).to(dtype=value.dtype) + delta_den

        t2 = t.square()
        t3 = t2 * t
        t4 = t2.square()
        t5 = t4 * t
        n = (
            numerator[..., 0].unsqueeze(-1)
            + numerator[..., 1].unsqueeze(-1) * t
            + numerator[..., 2].unsqueeze(-1) * t2
            + numerator[..., 3].unsqueeze(-1) * t3
            + numerator[..., 4].unsqueeze(-1) * t4
            + numerator[..., 5].unsqueeze(-1) * t5
        )
        d = (
            1.0
            + (denominator[..., 0].unsqueeze(-1) * t).abs()
            + (denominator[..., 1].unsqueeze(-1) * t2).abs()
            + (denominator[..., 2].unsqueeze(-1) * t3).abs()
            + (denominator[..., 3].unsqueeze(-1) * t4).abs()
        )
        return ((n / d) * rms).reshape(shape)


class RationalDynamicA5_4FFN(nn.Module):
    """Matched-budget no-GLU FFN using dynamic rational A5/4 coefficients."""

    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps=1e-6):
        super().__init__()
        settings = rda_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RDA activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.rda_activation = RationalDynamicA5_4Activation(
            self.hidden_dim,
            self.groups,
            settings["base_init"],
            settings["coeff_limit"],
            settings["denominator_limit"],
            settings["dynamic_init"],
            rational_group_size,
            rational_max_groups,
            eps=eps,
            input_affine=settings["input_affine"],
            moment_affine_init=settings["moment_affine_init"],
        )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out_proj(self.rda_activation(self.in_proj(x)))


def rlbx_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RLBX_ACTIVATIONS:
        return None

    hidden_dim = int(ffn_dim)
    base_init = "silu"
    centers = (-1.5, -0.5, 0.5, 1.5)
    coeff_limit = 0.50
    odd_init = 0.04
    bump_init = 0.03
    beta = 0.85
    input_affine = False
    if activation == "rlbx_k2_ffn":
        pass
    elif activation == "rlbx_k2_shift_ffn":
        input_affine = True
    elif activation == "rlbx_k2_strong_ffn":
        coeff_limit = 0.75
        odd_init = 0.08
        bump_init = 0.05
        beta = 0.75
        input_affine = True
    elif activation == "rlbx_k2_identity_ffn":
        base_init = "identity"
        coeff_limit = 1.00
        odd_init = 0.12
        bump_init = 0.08
        beta = 1.0
        input_affine = True
    else:
        raise ValueError(f"unknown RLBX activation {activation}")

    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "base_init": base_init,
        "centers": centers,
        "coeff_limit": coeff_limit,
        "odd_init": odd_init,
        "bump_init": bump_init,
        "beta": beta,
        "input_affine": input_affine,
    }


class RationalLocalBasisExpansionActivation(nn.Module):
    """Two-stream local rational basis feature generator with no GLU branch.

    The module maps one projected vector v to [base(v), local(v)]. With
    hidden_dim=d_ff, the output projection has width 2*d_ff, so the matrix count
    d*d_ff + 2*d_ff*d equals the standard SwiGLU 3*d*d_ff budget.
    """

    def __init__(
        self,
        hidden_dim,
        groups,
        base_init,
        centers,
        coeff_limit,
        odd_init,
        bump_init,
        beta,
        group_size,
        max_groups,
        eps=1e-6,
        input_affine=False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RLBX hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.coeff_limit = float(coeff_limit)
        self.basis_count = len(tuple(centers))
        self.input_affine = bool(input_affine)
        self.input_shift = None
        self.input_log_scale = None
        if self.input_affine:
            self.input_shift = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))
            self.input_log_scale = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))

        from rational_opt import RationalGroupedVersionA5_4

        self.base = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init=base_init,
            fit_range=5.0,
            group_size=group_size,
            max_groups=max_groups,
            groups=self.groups,
        )
        center_tensor = torch.tensor(centers, dtype=torch.float32).view(1, self.basis_count).repeat(self.groups, 1)
        self.centers = nn.Parameter(center_tensor)
        self.log_beta = nn.Parameter(torch.full((self.groups, self.basis_count), math.log(float(beta)), dtype=torch.float32))
        coeff = torch.zeros(self.groups, self.basis_count, 2, dtype=torch.float32)
        signs = torch.sign(center_tensor)
        signs[signs == 0] = 1.0
        coeff[..., 0] = float(odd_init) * signs
        coeff[..., 1] = float(bump_init)
        coeff = torch.clamp(coeff / self.coeff_limit, -0.999, 0.999)
        self.coeff_logits = nn.Parameter(torch.atanh(coeff))

    def forward(self, value):
        if value.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {value.size(-1)}")
        shape = value.shape
        grouped = value.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = grouped / rms
        if self.input_affine:
            affine_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
            scale = torch.exp(self.input_log_scale).view(affine_shape).to(dtype=value.dtype)
            shift = self.input_shift.view(affine_shape).to(dtype=value.dtype)
            t = scale * t + shift

        base = self.base(t.reshape(shape)).view_as(grouped)
        atom_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1, self.basis_count
        centers = self.centers.view(atom_shape).to(dtype=value.dtype)
        beta = torch.exp(self.log_beta).view(atom_shape).to(dtype=value.dtype)
        u = t.unsqueeze(-1) - centers
        den = 1.0 + beta * u.square()
        odd = u / den
        zero_level = 1.0 / (1.0 + beta * centers.square())
        bump = 1.0 / den - zero_level
        coeff = self.coeff_limit * torch.tanh(self.coeff_logits).view(atom_shape[:-1] + (self.basis_count, 2)).to(dtype=value.dtype)
        local = (coeff[..., 0] * odd + coeff[..., 1] * bump).sum(dim=-1)
        base = (base * rms).reshape(shape)
        local = (local * rms).reshape(shape)
        return torch.cat((base, local), dim=-1)


class RationalLocalBasisExpansionFFN(nn.Module):
    """Matched-budget no-GLU FFN using separate base and local rational features."""

    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps=1e-6):
        super().__init__()
        settings = rlbx_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RLBX activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.rlbx_activation = RationalLocalBasisExpansionActivation(
            self.hidden_dim,
            self.groups,
            settings["base_init"],
            settings["centers"],
            settings["coeff_limit"],
            settings["odd_init"],
            settings["bump_init"],
            settings["beta"],
            rational_group_size,
            rational_max_groups,
            eps=eps,
            input_affine=settings["input_affine"],
        )
        self.out_proj = nn.Linear(2 * self.hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out_proj(self.rlbx_activation(self.in_proj(x)))


def rlb_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RLB_ACTIVATIONS:
        return None

    hidden_dim = (3 * int(ffn_dim)) // 2
    base_init = "silu"
    centers = (-1.5, -0.5, 0.5, 1.5)
    coeff_limit = 0.35
    odd_init = 0.03
    bump_init = 0.02
    beta = 1.0
    input_affine = True
    center_odd = False
    train_centers = True
    atom_scale_init = None
    atom_scale_limit = 1.0
    fused = False
    if activation == "rlb_shift_ffn":
        pass
    elif activation == "rlb_strong_ffn":
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
    elif activation == "rlb_fixed_strong_ffn":
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        train_centers = False
    elif activation == "rlb_fused_fixed_strong_ffn":
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_fixed_strong_h2880_ffn":
        hidden_dim = 2880
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_fixed_strong_h2816_ffn":
        hidden_dim = 2816
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_fixed_strong_h2640_ffn":
        hidden_dim = 2640
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_fixed_strong_h2560_ffn":
        hidden_dim = 2560
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_boost_h2560_ffn":
        hidden_dim = 2560
        coeff_limit = 0.85
        odd_init = 0.10
        bump_init = 0.065
        beta = 0.65
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_boost_h2400_ffn":
        hidden_dim = 2400
        coeff_limit = 0.85
        odd_init = 0.10
        bump_init = 0.065
        beta = 0.65
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_quantile4_ffn":
        centers = (-1.35, -0.45, 0.45, 1.35)
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_core4_ffn":
        centers = (-1.10, -0.35, 0.35, 1.10)
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fast_ffn":
        centers = (-0.75, 0.75)
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        train_centers = False
    elif activation == "rlb_fused_fast_ffn":
        centers = (-0.75, 0.75)
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_fast_h2816_ffn":
        hidden_dim = 2816
        centers = (-0.75, 0.75)
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_fast_h2640_ffn":
        hidden_dim = 2640
        centers = (-0.75, 0.75)
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fused_fast_h2560_ffn":
        hidden_dim = 2560
        centers = (-0.75, 0.75)
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        input_affine = False
        train_centers = False
        fused = True
    elif activation == "rlb_fast_train_ffn":
        centers = (-0.75, 0.75)
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
    elif activation == "rlb_fast_scaled_ffn":
        centers = (-0.75, 0.75)
        coeff_limit = 0.70
        odd_init = 0.075
        bump_init = 0.05
        beta = 0.75
        train_centers = False
        atom_scale_init = 0.80
        atom_scale_limit = 1.25
    elif activation == "rlb_wide_ffn":
        centers = (-2.0, -1.0, 0.0, 1.0, 2.0)
        coeff_limit = 0.50
        odd_init = 0.04
        bump_init = 0.03
        beta = 0.80
    elif activation == "rlb_identity_ffn":
        base_init = "identity"
        centers = (-2.0, -1.0, 0.0, 1.0, 2.0)
        coeff_limit = 0.80
        odd_init = 0.10
        bump_init = 0.08
        beta = 1.0
    elif activation == "rlb_centered_strong_ffn":
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        center_odd = True
    elif activation == "rlb_centered_scaled_ffn":
        coeff_limit = 0.70
        odd_init = 0.075
        bump_init = 0.05
        beta = 0.75
        center_odd = True
        atom_scale_init = 0.60
        atom_scale_limit = 1.25
    elif activation == "rlb_fixed_centered_ffn":
        coeff_limit = 0.60
        odd_init = 0.06
        bump_init = 0.04
        beta = 0.75
        center_odd = True
        train_centers = False
    else:
        raise ValueError(f"unknown RLB activation {activation}")

    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "base_init": base_init,
        "centers": centers,
        "coeff_limit": coeff_limit,
        "odd_init": odd_init,
        "bump_init": bump_init,
        "beta": beta,
        "input_affine": input_affine,
        "center_odd": center_odd,
        "train_centers": train_centers,
        "atom_scale_init": atom_scale_init,
        "atom_scale_limit": atom_scale_limit,
        "fused": fused,
    }


class RationalLocalBasisActivation(nn.Module):
    """Single-branch rational activation with trainable local Cauchy atoms.

    For normalized scalar t, each group learns local rational atoms centered at
    c_k:

        odd_k(t)  = (t - c_k) / (1 + beta_k (t - c_k)^2)
        bump_k(t) = 1 / (1 + beta_k (t - c_k)^2) - 1 / (1 + beta_k c_k^2)

    The second term is centered so the atom is initially neutral around zero.
    This gives rational activations local shape control, which global polynomial
    residuals cannot express cheaply, while still using one projection only.
    """

    def __init__(
        self,
        hidden_dim,
        groups,
        base_init,
        centers,
        coeff_limit,
        odd_init,
        bump_init,
        beta,
        group_size,
        max_groups,
        eps=1e-6,
        input_affine=True,
        center_odd=False,
        train_centers=True,
        atom_scale_init=None,
        atom_scale_limit=1.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RLB hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.coeff_limit = float(coeff_limit)
        self.basis_count = len(tuple(centers))
        self.input_affine = bool(input_affine)
        self.center_odd = bool(center_odd)
        self.input_shift = None
        self.input_log_scale = None
        self.atom_scale_limit = float(atom_scale_limit)
        self.atom_scale_logit = None
        if self.input_affine:
            self.input_shift = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))
            self.input_log_scale = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))
        if atom_scale_init is not None:
            init = min(max(float(atom_scale_init) / self.atom_scale_limit, 1e-4), 1.0 - 1e-4)
            self.atom_scale_logit = nn.Parameter(torch.full((self.groups,), math.log(init / (1.0 - init)), dtype=torch.float32))

        from rational_opt import RationalGroupedVersionA5_4

        self.base = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init=base_init,
            fit_range=5.0,
            group_size=group_size,
            max_groups=max_groups,
            groups=self.groups,
        )
        center_tensor = torch.tensor(centers, dtype=torch.float32).view(1, self.basis_count).repeat(self.groups, 1)
        if train_centers:
            self.centers = nn.Parameter(center_tensor)
        else:
            self.register_buffer("centers", center_tensor)
        self.log_beta = nn.Parameter(torch.full((self.groups, self.basis_count), math.log(float(beta)), dtype=torch.float32))
        coeff = torch.zeros(self.groups, self.basis_count, 2, dtype=torch.float32)
        signs = torch.sign(center_tensor)
        signs[signs == 0] = 1.0
        coeff[..., 0] = float(odd_init) * signs
        coeff[..., 1] = float(bump_init)
        coeff = torch.clamp(coeff / self.coeff_limit, -0.999, 0.999)
        self.coeff_logits = nn.Parameter(torch.atanh(coeff))

    def forward(self, value):
        if value.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {value.size(-1)}")
        shape = value.shape
        grouped = value.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = grouped / rms
        if self.input_affine:
            affine_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
            scale = torch.exp(self.input_log_scale).view(affine_shape).to(dtype=value.dtype)
            shift = self.input_shift.view(affine_shape).to(dtype=value.dtype)
            t = scale * t + shift

        base = self.base(t.reshape(shape)).view_as(grouped)
        atom_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
        centers = self.centers.to(dtype=value.dtype)
        beta = torch.exp(self.log_beta).to(dtype=value.dtype)
        coeff = self.coeff_limit * torch.tanh(self.coeff_logits).to(dtype=value.dtype)
        delta = torch.zeros_like(t)
        for basis_idx in range(self.basis_count):
            center = centers[:, basis_idx].view(atom_shape)
            beta_i = beta[:, basis_idx].view(atom_shape)
            u = t - center
            den = 1.0 + beta_i * u.square()
            odd = u / den
            zero_level = 1.0 / (1.0 + beta_i * center.square())
            if self.center_odd:
                odd = odd + center * zero_level
            bump = 1.0 / den - zero_level
            coeff_i = coeff[:, basis_idx].view(*atom_shape, 2)
            delta = delta + coeff_i[..., 0] * odd + coeff_i[..., 1] * bump
        if self.atom_scale_logit is not None:
            scale_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
            atom_scale = self.atom_scale_limit * torch.sigmoid(self.atom_scale_logit).view(scale_shape).to(dtype=value.dtype)
            delta = atom_scale * delta
        return ((base + delta) * rms).reshape(shape)


class RationalLocalBasisFFN(nn.Module):
    """Matched-budget no-GLU FFN using local rational basis activations."""

    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps=1e-6):
        super().__init__()
        settings = rlb_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RLB activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        if settings["fused"]:
            from rational_opt import RationalFusedLocalBasisA5_4

            self.rlb_activation = RationalFusedLocalBasisA5_4(
                self.hidden_dim,
                self.groups,
                init=settings["base_init"],
                fit_range=5.0,
                centers=settings["centers"],
                coeff_limit=settings["coeff_limit"],
                odd_init=settings["odd_init"],
                bump_init=settings["bump_init"],
                beta=settings["beta"],
                eps=eps,
            )
        else:
            self.rlb_activation = RationalLocalBasisActivation(
                self.hidden_dim,
                self.groups,
                settings["base_init"],
                settings["centers"],
                settings["coeff_limit"],
                settings["odd_init"],
                settings["bump_init"],
                settings["beta"],
                rational_group_size,
                rational_max_groups,
                eps=eps,
                input_affine=settings["input_affine"],
                center_odd=settings["center_odd"],
                train_centers=settings["train_centers"],
                atom_scale_init=settings["atom_scale_init"],
                atom_scale_limit=settings["atom_scale_limit"],
            )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out_proj(self.rlb_activation(self.in_proj(x)))


def apply_rlb_positive_gauge(model: nn.Module, log_scale: float, seed: int) -> int:
    """Apply a function-preserving positive gauge to RLB FFN matrices.

    Scaling one group of W_in by a > 0 and the matching W_out columns by 1/a
    leaves the represented RLB block function unchanged at initialization, but it
    changes matrix conditioning. This is useful for optimizer stress tests.
    """

    if log_scale <= 0.0:
        return 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    group_count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, RationalLocalBasisFFN):
                continue
            groups = int(module.groups)
            width = int(module.hidden_dim // module.groups)
            logs = torch.empty(groups, dtype=torch.float32).uniform_(
                -float(log_scale), float(log_scale), generator=generator
            )
            scales = torch.exp(logs).to(device=module.in_proj.weight.device, dtype=module.in_proj.weight.dtype)
            module.in_proj.weight.view(groups, width, -1).mul_(scales.view(groups, 1, 1))
            module.out_proj.weight.view(module.out_proj.weight.shape[0], groups, width).mul_(
                scales.reciprocal().view(1, groups, 1)
            )
            group_count += groups
    return group_count


def rcq_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RCQ_ACTIVATIONS:
        return None

    hidden_dim = (3 * int(ffn_dim)) // 2
    base_init = "silu"
    coeff_limit = 0.75
    init = (0.08, 0.02)
    beta = 1.0
    input_affine = False
    if activation == "rcq_ffn":
        pass
    elif activation == "rcq_shift_ffn":
        input_affine = True
    elif activation == "rcq_strong_ffn":
        coeff_limit = 1.00
        init = (0.18, 0.05)
        beta = 0.75
        input_affine = True
    elif activation == "rcq_identity_ffn":
        base_init = "identity"
        coeff_limit = 1.00
        init = (0.35, 0.10)
        beta = 1.25
        input_affine = True
    else:
        raise ValueError(f"unknown RCQ activation {activation}")

    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "base_init": base_init,
        "coeff_limit": coeff_limit,
        "init": init,
        "beta": beta,
        "input_affine": input_affine,
    }


class RationalComplexQuadraticActivation(nn.Module):
    """Single-branch pairwise rational activation with complex quadratic coupling.

    Adjacent normalized channels are treated as z = s + i t. The activation adds
    a bounded rational quadratic term

        (a + i b) z^2 / (1 + beta |z|^2)

    to a grouped rational base. This gives explicit pairwise second-order
    features without a separate gate/up branch or a GLU product.
    """

    def __init__(
        self,
        hidden_dim,
        groups,
        base_init,
        coeff_limit,
        init,
        beta,
        group_size,
        max_groups,
        eps=1e-6,
        input_affine=False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RCQ hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        if self.width_per_group % 2 != 0:
            raise ValueError("RCQ width_per_group must be even")
        self.eps = float(eps)
        self.coeff_limit = float(coeff_limit)
        if float(beta) <= 0.0:
            raise ValueError("RCQ beta must be positive")
        self.beta_log = nn.Parameter(torch.full((self.groups,), math.log(float(beta)), dtype=torch.float32))
        self.input_affine = bool(input_affine)
        self.input_shift = None
        self.input_log_scale = None
        if self.input_affine:
            self.input_shift = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))
            self.input_log_scale = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))

        from rational_opt import RationalGroupedVersionA5_4

        self.base = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init=base_init,
            fit_range=5.0,
            group_size=group_size,
            max_groups=max_groups,
            groups=self.groups,
        )
        init_tensor = torch.tensor(init, dtype=torch.float32).view(2, 1).repeat(1, self.groups)
        init_tensor = torch.clamp(init_tensor / self.coeff_limit, -0.999, 0.999)
        self.coeff_logits = nn.Parameter(torch.atanh(init_tensor))

    def forward(self, value):
        if value.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {value.size(-1)}")
        shape = value.shape
        grouped = value.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = grouped / rms
        if self.input_affine:
            affine_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
            scale = torch.exp(self.input_log_scale).view(affine_shape).to(dtype=value.dtype)
            shift = self.input_shift.view(affine_shape).to(dtype=value.dtype)
            t = scale * t + shift

        base = self.base(t.reshape(shape)).view_as(grouped)
        pair = t.view(*shape[:-1], self.groups, self.width_per_group // 2, 2)
        base_pair = base.view_as(pair)
        real = pair[..., 0]
        imag = pair[..., 1]
        radius2 = real.square() + imag.square()
        beta = torch.exp(self.beta_log).view(*((1,) * (pair.dim() - 3)), self.groups, 1).to(dtype=value.dtype)
        den = 1.0 + beta * radius2
        quad_real = (real.square() - imag.square()) / den
        quad_imag = (2.0 * real * imag) / den
        coeff = self.coeff_limit * torch.tanh(self.coeff_logits).transpose(0, 1).to(dtype=value.dtype)
        coeff = coeff.view(*((1,) * (pair.dim() - 3)), self.groups, 1, 2)
        a = coeff[..., 0]
        b = coeff[..., 1]
        delta_real = a * quad_real - b * quad_imag
        delta_imag = a * quad_imag + b * quad_real
        out_pair = torch.stack((base_pair[..., 0] + delta_real, base_pair[..., 1] + delta_imag), dim=-1)
        out = out_pair.reshape_as(grouped)
        return (out * rms).reshape(shape)


class RationalComplexQuadraticFFN(nn.Module):
    """Matched-budget no-GLU FFN using pairwise complex rational quadratic activation."""

    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps=1e-6):
        super().__init__()
        settings = rcq_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RCQ activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.rcq_activation = RationalComplexQuadraticActivation(
            self.hidden_dim,
            self.groups,
            settings["base_init"],
            settings["coeff_limit"],
            settings["init"],
            settings["beta"],
            rational_group_size,
            rational_max_groups,
            eps=eps,
            input_affine=settings["input_affine"],
        )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out_proj(self.rcq_activation(self.in_proj(x)))


def rgc_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RGC_ACTIVATIONS:
        return None

    hidden_dim = (3 * int(ffn_dim)) // 2
    base_init = "silu"
    coeff_limit = 0.75
    init = (0.18, 0.04, 0.02)
    input_affine = False
    beta = 1.0
    moment_init = 0.0
    if activation == "rgc_ffn":
        pass
    elif activation == "rgc_shift_ffn":
        input_affine = True
    elif activation == "rgc_strong_ffn":
        init = (0.30, 0.08, 0.04)
        input_affine = True
    elif activation == "rgc_moment_ffn":
        init = (0.22, 0.06, 0.03)
        input_affine = True
        moment_init = 0.08
    elif activation == "rgc_identity_ffn":
        base_init = "identity"
        coeff_limit = 1.00
        init = (0.45, 0.15, 0.10)
        input_affine = True
        beta = 1.25
    else:
        raise ValueError(f"unknown RGC activation {activation}")

    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "base_init": base_init,
        "coeff_limit": coeff_limit,
        "init": init,
        "input_affine": input_affine,
        "beta": beta,
        "moment_init": moment_init,
    }


class RationalGroupCompetitiveActivation(nn.Module):
    """Single-branch rational activation with within-group energy competition.

    This is not a GLU: it has one expansion projection and one output projection.
    The activation couples channels only through same-branch group moments. The
    rational contrast term creates multiplicative second-order behavior inside
    the activation itself:

        c_j = (t_j^2 - mean_q t_q^2) / (1 + beta (t_j^2 + mean_q t_q^2)).

    Then the rational base feature is modulated by c_j, and bounded rational
    signed/even residuals are added. This makes the nonlinearity depend on which
    channels are large relative to their group, without a separate gate stream.
    """

    def __init__(
        self,
        hidden_dim,
        groups,
        base_init,
        coeff_limit,
        init,
        beta,
        group_size,
        max_groups,
        eps=1e-6,
        input_affine=False,
        moment_init=0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.groups = int(groups)
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RGC hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.coeff_limit = float(coeff_limit)
        if float(beta) <= 0.0:
            raise ValueError("RGC beta must be positive")
        self.beta_log = nn.Parameter(torch.full((self.groups,), math.log(float(beta)), dtype=torch.float32))
        self.input_affine = bool(input_affine)
        self.input_shift = None
        self.input_log_scale = None
        if self.input_affine:
            self.input_shift = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))
            self.input_log_scale = nn.Parameter(torch.zeros(self.groups, dtype=torch.float32))

        from rational_opt import RationalGroupedVersionA5_4

        self.base = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init=base_init,
            fit_range=5.0,
            group_size=group_size,
            max_groups=max_groups,
            groups=self.groups,
        )
        init_tensor = torch.tensor(init, dtype=torch.float32).view(3, 1).repeat(1, self.groups)
        init_tensor = torch.clamp(init_tensor / self.coeff_limit, -0.999, 0.999)
        self.coeff_logits = nn.Parameter(torch.atanh(init_tensor))
        self.moment_mix = nn.Parameter(torch.zeros(3, self.groups, 3, dtype=torch.float32))
        if float(moment_init) != 0.0:
            with torch.no_grad():
                c = float(moment_init)
                self.moment_mix[0, :, 1] = c
                self.moment_mix[1, :, 2] = 0.5 * c
                self.moment_mix[2, :, 0] = -0.5 * c

    def forward(self, value):
        if value.size(-1) != self.hidden_dim:
            raise ValueError(f"expected last dimension {self.hidden_dim}, got {value.size(-1)}")
        shape = value.shape
        grouped = value.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        t = grouped / rms
        mean = t.mean(dim=-1)
        log_rms = torch.log(rms.squeeze(-1) + self.eps)
        skew = (t * t.square()).mean(dim=-1)
        stats = torch.stack((torch.tanh(mean), torch.tanh(log_rms), torch.tanh(0.5 * skew)), dim=-1)

        if self.input_affine:
            affine_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
            scale = torch.exp(self.input_log_scale).view(affine_shape).to(dtype=value.dtype)
            shift = self.input_shift.view(affine_shape).to(dtype=value.dtype)
            t = scale * t + shift

        base = self.base(t.reshape(shape)).view_as(grouped)
        energy = t.square()
        mean_energy = energy.mean(dim=-1, keepdim=True)
        beta_shape = *((1,) * (grouped.dim() - 2)), self.groups, 1
        beta = torch.exp(self.beta_log).view(beta_shape).to(dtype=value.dtype)
        contrast = (energy - mean_energy) / (1.0 + beta * (energy + mean_energy))
        signed = t * contrast
        even = (energy - 1.0) / (1.0 + beta * energy)

        dynamic = torch.einsum("...gs,bgs->...gb", stats, self.moment_mix.to(dtype=value.dtype))
        logits = self.coeff_logits.transpose(0, 1).view(*((1,) * (grouped.dim() - 2)), self.groups, 3)
        coeff = self.coeff_limit * torch.tanh(logits.to(dtype=value.dtype) + dynamic)
        comp = coeff[..., 0].unsqueeze(-1)
        signed_scale = coeff[..., 1].unsqueeze(-1)
        even_scale = coeff[..., 2].unsqueeze(-1)
        activated = base * (1.0 + comp * contrast) + signed_scale * signed + even_scale * even
        return (activated * rms).reshape(shape)


class RationalGroupCompetitiveFFN(nn.Module):
    """Matched-budget no-GLU FFN using rational group competition."""

    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps=1e-6):
        super().__init__()
        settings = rgc_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RGC activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.rgc_activation = RationalGroupCompetitiveActivation(
            self.hidden_dim,
            self.groups,
            settings["base_init"],
            settings["coeff_limit"],
            settings["init"],
            settings["beta"],
            rational_group_size,
            rational_max_groups,
            eps=eps,
            input_affine=settings["input_affine"],
            moment_init=settings["moment_init"],
        )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out_proj(self.rgc_activation(self.in_proj(x)))


def rsm_settings(activation, ffn_dim, group_size, max_groups):
    if activation not in RSM_ACTIVATIONS:
        return None

    hidden_dim = (3 * int(ffn_dim)) // 2
    if activation == "rsm_ffn":
        odd_scale = 0.50
        even_gate_scale = 0.10
        even_basis_scale = 0.0
    elif activation == "rsm_ffn_basis":
        odd_scale = 0.50
        even_gate_scale = 0.10
        even_basis_scale = 0.05
    elif activation == "rsm_ffn_strong":
        odd_scale = 0.75
        even_gate_scale = 0.15
        even_basis_scale = 0.05
    else:
        raise ValueError(f"unknown RSM activation {activation}")

    return {
        "hidden_dim": hidden_dim,
        "groups": resolve_group_count(hidden_dim, group_size, max_groups),
        "odd_scale": odd_scale,
        "even_gate_scale": even_gate_scale,
        "even_basis_scale": even_basis_scale,
    }


class RationalSelfModulatedFFN(nn.Module):
    """Single-branch rational FFN with no GLU gate/up split.

    The matrix budget is matched to SwiGLU by using hidden_dim = 1.5 * d_ff:
    W_in and W_out then contain 2 * d_model * hidden_dim = 3 * d_model * d_ff
    parameters. The rational functions do not replace the value branch; they
    token-condition the value coordinates generated by the single expansion.
    """

    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps=1e-6):
        super().__init__()
        settings = rsm_settings(activation, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RSM activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.groups = settings["groups"]
        if self.hidden_dim % self.groups != 0:
            raise ValueError("RSM hidden_dim must be divisible by groups")
        self.width_per_group = self.hidden_dim // self.groups
        self.eps = float(eps)
        self.in_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        from rational_opt import RationalGroupedVersionA5_4

        self.odd_gate = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init="identity",
            fit_range=5.0,
            group_size=rational_group_size,
            max_groups=rational_max_groups,
            groups=self.groups,
        )
        self.even_gate = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init="identity",
            fit_range=5.0,
            group_size=rational_group_size,
            max_groups=rational_max_groups,
            groups=self.groups,
        )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.odd_scale = nn.Parameter(torch.full((self.groups,), settings["odd_scale"], dtype=torch.float32))
        self.even_gate_scale = nn.Parameter(
            torch.full((self.groups,), settings["even_gate_scale"], dtype=torch.float32)
        )
        self.even_basis_scale = nn.Parameter(
            torch.full((self.groups,), settings["even_basis_scale"], dtype=torch.float32)
        )
        self.reset_rsm_rationals()

    def reset_rsm_rationals(self):
        with torch.no_grad():
            # Odd bounded rational: R_o(t) = t / (1 + |t|).
            self.odd_gate.numerator.zero_()
            self.odd_gate.denominator.zero_()
            self.odd_gate.numerator[:, 1] = 1.0
            self.odd_gate.denominator[:, 0] = 1.0

            # Even bounded curvature: R_e(t) = (t^2 - 1) / (1 + t^2).
            self.even_gate.numerator.zero_()
            self.even_gate.denominator.zero_()
            self.even_gate.numerator[:, 0] = -1.0
            self.even_gate.numerator[:, 2] = 1.0
            self.even_gate.denominator[:, 1] = 1.0

    def _group_rms_unit(self, v):
        shape = v.shape
        grouped = v.view(*shape[:-1], self.groups, self.width_per_group)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        return (grouped / rms).reshape(shape), rms

    def _apply_group_scale(self, y, scale):
        shape = y.shape
        grouped = y.view(*shape[:-1], self.groups, self.width_per_group)
        scale_view = scale.view(*((1,) * (grouped.dim() - 2)), self.groups, 1)
        return (grouped * scale_view.to(dtype=y.dtype)).reshape(shape)

    def forward(self, x):
        value = self.in_proj(x)
        normalized, rms = self._group_rms_unit(value)
        odd = self.odd_gate(normalized)
        even = self.even_gate(normalized)
        gate = 1.0 + self._apply_group_scale(odd, self.odd_scale)
        gate = gate + self._apply_group_scale(even, self.even_gate_scale)
        hidden = value * gate
        even_grouped = even.view(*value.shape[:-1], self.groups, self.width_per_group)
        even_basis = (rms * even_grouped).reshape_as(value)
        hidden = hidden + self._apply_group_scale(even_basis, self.even_basis_scale)
        return self.out_proj(hidden)


def rhg_settings(activation, ffn_dim, group_size, max_groups, layer_idx=None, layer_count=None):
    if activation not in RHG_ACTIVATIONS:
        return None
    if activation == "crv_rhg":
        hidden_dim = (7 * int(ffn_dim)) // 8 - 7
        gate_rank = int(ffn_dim) * 3 // 8
    elif activation == "rhg_ffn_resvalue_gated_highgate":
        hidden_dim = (3 * int(ffn_dim)) // 4
        gate_rank = int(ffn_dim) // 2
    elif activation == "rhg_ffn_highgate":
        hidden_dim = (3 * int(ffn_dim)) // 4
        gate_rank = int(ffn_dim) // 2
    elif activation == "rhg_ffn_resvalue_gated_valuewide":
        hidden_dim = (15 * int(ffn_dim)) // 16
        gate_rank = 656
    elif activation == "rhg_ffn_valuewide":
        hidden_dim = (15 * int(ffn_dim)) // 16
        gate_rank = 656
    elif activation == "rhg_ffn_basisgate":
        hidden_dim = (105 * int(ffn_dim)) // 128
        gate_rank = int(ffn_dim) // 4
    elif activation in {"rhg_ffn_basisgate_wide", "rhg_ffn_basisgate_resvalue_wide"}:
        hidden_dim = (15 * int(ffn_dim)) // 16
        gate_rank = (3 * int(ffn_dim)) // 16
    elif activation == "rhg_ffn_resvalue_gated_crossmix64":
        hidden_dim = (53 * int(ffn_dim)) // 64
        gate_rank = int(ffn_dim) * 3 // 8
    elif activation == "rhg_ffn_resvalue_gated_crossmix128":
        hidden_dim = (25 * int(ffn_dim)) // 32
        gate_rank = int(ffn_dim) * 3 // 8
    elif activation in {
        "rhg_ffn_balanced",
        "rhg_ffn_gateact",
        "rhg_ffn_valueact",
        "rhg_ffn_fullact",
        "rhg_ffn_resgate",
        "rhg_ffn_resvalue",
        "rhg_ffn_resvalue_dual",
        "rhg_ffn_resvalue_gated_dual",
        "rhg_ffn_resvalue_gated",
        "rhg_ffn_resvalue_gated_channel",
        "rhg_ffn_resvalue_gated_channel_strong",
        "rhg_ffn_resvalue_gated_beta075",
        "rhg_ffn_resvalue_gated_beta10",
        "rhg_ffn_resvalue_gated_beta10_depthup",
        "rhg_ffn_resvalue_gated_beta10_groupdepth",
        "rhg_ffn_resvalue_gated_beta10_groupscale",
        "rhg_ffn_resvalue_gated_beta10_groupscale_moment",
        "rhg_ffn_resvalue_gated_beta10_groupscale_safegate",
        "rhg_ffn_resvalue_gated_beta10_groupscale_safegate_low",
        "rhg_ffn_resvalue_gated_beta125",
        "rhg_ffn_resvalue_gated_beta20",
        "rhg_ffn_resvalue_gated_norm",
        "rhg_ffn_resvalue_norm",
        "rhg_ffn_resboth",
    }:
        hidden_dim = (7 * int(ffn_dim)) // 8
        gate_rank = int(ffn_dim) * 3 // 8
    else:
        hidden_dim = int(ffn_dim)
        gate_rank = 512
    gate_basis_count = (
        2
        if activation
        in {
            "rhg_ffn_basisgate",
            "rhg_ffn_basisgate_wide",
            "rhg_ffn_basisgate_resvalue_wide",
        }
        else 1
    )
    diag_groups = resolve_group_count(int(hidden_dim), group_size, max_groups)
    conditional_value_basis_count = 3 if activation == "crv_rhg" else 0
    scale_schedule = (
        "depth_up"
        if activation
        in {
            "rhg_ffn_resvalue_gated_beta10_depthup",
            "rhg_ffn_resvalue_gated_beta10_groupdepth",
        }
        else "flat"
    )
    scale_mode = (
        "group"
        if activation
        in {
            "rhg_ffn_resvalue_gated_beta10_groupdepth",
            "rhg_ffn_resvalue_gated_beta10_groupscale",
            "rhg_ffn_resvalue_gated_beta10_groupscale_moment",
            "rhg_ffn_resvalue_gated_beta10_groupscale_safegate",
            "rhg_ffn_resvalue_gated_beta10_groupscale_safegate_low",
        }
        else "scalar"
    )
    if activation == "rhg_ffn_resvalue_gated_beta20":
        value_residual_scale_init = 0.20
    elif activation == "rhg_ffn_resvalue_gated_beta125":
        value_residual_scale_init = 0.125
    elif activation in {
        "rhg_ffn_resvalue_gated_beta10",
        "rhg_ffn_resvalue_gated_beta10_groupscale",
        "rhg_ffn_resvalue_gated_beta10_groupscale_moment",
        "rhg_ffn_resvalue_gated_beta10_groupscale_safegate",
        "rhg_ffn_resvalue_gated_beta10_groupscale_safegate_low",
    }:
        value_residual_scale_init = 0.10
    elif activation in {
        "rhg_ffn_resvalue_gated_beta10_depthup",
        "rhg_ffn_resvalue_gated_beta10_groupdepth",
    }:
        if layer_idx is None or layer_count is None:
            value_residual_scale_init = 0.10
        else:
            depth = float(layer_idx) / float(max(1, int(layer_count) - 1))
            value_residual_scale_init = 0.05 + 0.10 * depth
    elif activation == "rhg_ffn_resvalue_gated_beta075":
        value_residual_scale_init = 0.075
    else:
        value_residual_scale_init = 0.05
    return {
        "hidden_dim": int(hidden_dim),
        "gate_rank": int(gate_rank),
        "gate_groups": resolve_group_count(int(gate_rank), group_size, max_groups),
        "diag_groups": diag_groups,
        "diag_activation": activation in {"rhg_ffn_gateact", "rhg_ffn_fullact"},
        "gate_basis_count": gate_basis_count,
        "gate_residual": activation in {"rhg_ffn_resgate", "rhg_ffn_resboth"},
        "value_activation": activation in {"rhg_ffn_valueact", "rhg_ffn_fullact"},
        "value_residual": activation
        in {
            "rhg_ffn_resvalue",
            "rhg_ffn_resvalue_dual",
            "rhg_ffn_resvalue_gated_dual",
            "rhg_ffn_resvalue_gated",
            "rhg_ffn_resvalue_gated_channel",
            "rhg_ffn_resvalue_gated_channel_strong",
            "rhg_ffn_resvalue_gated_crossmix64",
            "rhg_ffn_resvalue_gated_crossmix128",
            "rhg_ffn_resvalue_gated_beta075",
            "rhg_ffn_resvalue_gated_beta10",
            "rhg_ffn_resvalue_gated_beta10_depthup",
            "rhg_ffn_resvalue_gated_beta10_groupdepth",
            "rhg_ffn_resvalue_gated_beta10_groupscale",
            "rhg_ffn_resvalue_gated_beta10_groupscale_moment",
            "rhg_ffn_resvalue_gated_beta10_groupscale_safegate",
            "rhg_ffn_resvalue_gated_beta10_groupscale_safegate_low",
            "rhg_ffn_resvalue_gated_beta125",
            "rhg_ffn_resvalue_gated_beta20",
            "rhg_ffn_resvalue_gated_highgate",
            "rhg_ffn_resvalue_gated_norm",
            "rhg_ffn_resvalue_gated_valuewide",
            "rhg_ffn_resvalue_norm",
            "rhg_ffn_resboth",
            "rhg_ffn_basisgate_resvalue_wide",
        },
        "value_residual_condition": activation
        in {
            "rhg_ffn_resvalue_gated",
            "rhg_ffn_resvalue_gated_dual",
            "rhg_ffn_resvalue_gated_channel",
            "rhg_ffn_resvalue_gated_channel_strong",
            "rhg_ffn_resvalue_gated_crossmix64",
            "rhg_ffn_resvalue_gated_crossmix128",
            "rhg_ffn_resvalue_gated_beta075",
            "rhg_ffn_resvalue_gated_beta10",
            "rhg_ffn_resvalue_gated_beta10_depthup",
            "rhg_ffn_resvalue_gated_beta10_groupdepth",
            "rhg_ffn_resvalue_gated_beta10_groupscale",
            "rhg_ffn_resvalue_gated_beta10_groupscale_moment",
            "rhg_ffn_resvalue_gated_beta10_groupscale_safegate",
            "rhg_ffn_resvalue_gated_beta10_groupscale_safegate_low",
            "rhg_ffn_resvalue_gated_beta125",
            "rhg_ffn_resvalue_gated_beta20",
            "rhg_ffn_resvalue_gated_highgate",
            "rhg_ffn_resvalue_gated_norm",
            "rhg_ffn_resvalue_gated_valuewide",
            "rhg_ffn_basisgate_resvalue_wide",
        },
        "value_residual_condition_mode": (
            "channel"
            if activation in {"rhg_ffn_resvalue_gated_channel", "rhg_ffn_resvalue_gated_channel_strong"}
            else "group"
        ),
        "value_residual_condition_init": (
            0.25 if activation == "rhg_ffn_resvalue_gated_channel_strong" else 0.10
        ),
        "value_residual_normalized": activation
        in {"rhg_ffn_resvalue_norm", "rhg_ffn_resvalue_gated_norm"},
        "value_residual_odd": activation
        in {"rhg_ffn_resvalue_dual", "rhg_ffn_resvalue_gated_dual"},
        "value_residual_scale_init": value_residual_scale_init,
        "value_residual_scale_mode": scale_mode,
        "value_residual_scale_schedule": scale_schedule,
        "value_residual_moment_condition": activation == "rhg_ffn_resvalue_gated_beta10_groupscale_moment",
        "value_residual_safe_gate_multiplier_init": (
            1.0
            if activation == "rhg_ffn_resvalue_gated_beta10_groupscale_safegate"
            else 0.5
            if activation == "rhg_ffn_resvalue_gated_beta10_groupscale_safegate_low"
            else None
        ),
        "value_residual_cross_rank": (
            64
            if activation == "rhg_ffn_resvalue_gated_crossmix64"
            else 128
            if activation == "rhg_ffn_resvalue_gated_crossmix128"
            else 0
        ),
        "conditional_value_basis_count": conditional_value_basis_count,
        "conditional_value_dim": diag_groups * conditional_value_basis_count,
    }


class RationalHyperGateFFN(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        activation,
        rational_group_size,
        rational_max_groups,
        layer_idx=None,
        layer_count=None,
    ):
        super().__init__()
        settings = rhg_settings(
            activation,
            ffn_dim,
            rational_group_size,
            rational_max_groups,
            layer_idx=layer_idx,
            layer_count=layer_count,
        )
        if settings is None:
            raise ValueError(f"unknown RHG activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.gate_rank = settings["gate_rank"]
        self.gate_groups = settings["gate_groups"]
        self.diag_groups = settings["diag_groups"]
        self.diag_activation = settings["diag_activation"]
        self.gate_basis_count = settings["gate_basis_count"]
        self.gate_residual = settings["gate_residual"]
        self.value_activation = settings["value_activation"]
        self.value_residual = settings["value_residual"]
        self.value_residual_condition = settings["value_residual_condition"]
        self.value_residual_condition_mode = settings["value_residual_condition_mode"]
        self.value_residual_normalized = settings["value_residual_normalized"]
        self.value_residual_odd = settings["value_residual_odd"]
        self.value_residual_scale_mode = settings["value_residual_scale_mode"]
        self.value_residual_moment_condition = settings["value_residual_moment_condition"]
        self.value_residual_safe_gate_multiplier_init = settings["value_residual_safe_gate_multiplier_init"]
        self.value_residual_safe_gate = None
        self.value_residual_cross_rank = settings["value_residual_cross_rank"]
        self.conditional_value_basis_count = settings["conditional_value_basis_count"]
        self.conditional_value_dim = settings["conditional_value_dim"]
        self.value_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.query_proj = nn.Linear(dim, self.gate_rank, bias=False)
        from rational_opt import RationalGroupedVersionA5_4

        self.rhg_value_act = None
        if self.value_activation:
            self.rhg_value_act = RationalGroupedVersionA5_4(
                self.hidden_dim,
                init="silu",
                fit_range=5.0,
                group_size=rational_group_size,
                max_groups=rational_max_groups,
                groups=self.diag_groups,
            )
        self.rhg_value_residual = None
        self.rhg_value_residual_odd = None
        self.value_cross_down = None
        self.value_cross_up = None
        if self.value_residual:
            residual_cls = NormalizedRationalValueResidual if self.value_residual_normalized else RationalPolarizedBasisActivation
            self.rhg_value_residual = residual_cls(self.hidden_dim, self.diag_groups, "bounded_even")
            if self.value_residual_odd:
                self.rhg_value_residual_odd = RationalPolarizedBasisActivation(
                    self.hidden_dim,
                    self.diag_groups,
                    "bounded_odd",
                )
                self.value_residual_odd_scale = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))
            if self.value_residual_scale_mode == "group":
                self.value_residual_scale = nn.Parameter(
                    torch.full(
                        (self.diag_groups,),
                        settings["value_residual_scale_init"],
                        dtype=torch.float32,
                    )
                )
            else:
                self.value_residual_scale = nn.Parameter(
                    torch.tensor(settings["value_residual_scale_init"], dtype=torch.float32)
                )
            if self.value_residual_safe_gate_multiplier_init is not None:
                gate_probability = float(self.value_residual_safe_gate_multiplier_init) / 2.0
                gate_probability = min(max(gate_probability, 1e-4), 1.0 - 1e-4)
                init_logit = math.log(gate_probability / (1.0 - gate_probability))
                self.value_residual_safe_gate = nn.Parameter(
                    torch.full((self.diag_groups,), init_logit, dtype=torch.float32)
                )
            if self.value_residual_condition:
                self.value_residual_condition_scale = nn.Parameter(
                    torch.full(
                        (self.diag_groups,),
                        settings["value_residual_condition_init"],
                        dtype=torch.float32,
                    )
                )
                if self.value_residual_moment_condition:
                    self.value_residual_rms_condition_scale = nn.Parameter(
                        torch.zeros(self.diag_groups, dtype=torch.float32)
                    )
            if self.value_residual_cross_rank > 0:
                self.value_cross_down = nn.Linear(self.hidden_dim, self.value_residual_cross_rank, bias=False)
                self.value_cross_up = nn.Linear(self.value_residual_cross_rank, self.hidden_dim, bias=False)
                self.value_cross_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.rhg_cond_value_residual = None
        self.value_condition_proj = None
        if self.conditional_value_basis_count > 0:
            self.rhg_cond_value_residual = ConditionalRationalValueResidual(
                self.hidden_dim,
                self.diag_groups,
                basis_count=self.conditional_value_basis_count,
                eps=1e-6,
            )
            self.value_condition_proj = nn.Linear(
                self.gate_basis_count * self.gate_rank,
                self.conditional_value_dim,
                bias=False,
            )
            self.value_condition_proj.zero_init = True
            self.value_residual_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.rhg_gate_act = None
        self.rhg_gate_basis = None
        if self.gate_basis_count == 1:
            self.rhg_gate_act = RationalGroupedVersionA5_4(
                self.gate_rank,
                init="silu",
                fit_range=5.0,
                group_size=rational_group_size,
                max_groups=rational_max_groups,
                groups=self.gate_groups,
            )
        else:
            self.rhg_gate_basis = RationalBasisExpansion(
                self.gate_rank,
                self.gate_basis_count,
                self.gate_groups,
                "identity_curvature",
                eps=1e-6,
            )
        self.rhg_gate_residual = None
        if self.gate_residual:
            self.rhg_gate_residual = RationalPolarizedBasisActivation(
                self.gate_rank,
                self.gate_groups,
                "bounded_even",
            )
            self.gate_residual_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.gate_proj = nn.Linear(self.gate_basis_count * self.gate_rank, self.hidden_dim, bias=False)
        self.rhg_diag_act = None
        if self.diag_activation:
            self.rhg_diag_act = RationalGroupedVersionA5_4(
                self.hidden_dim,
                init="identity",
                fit_range=5.0,
                group_size=rational_group_size,
                max_groups=rational_max_groups,
                groups=self.diag_groups,
            )
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.gate_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def _apply_diag_group_scale(self, value, scale):
        shape = value.shape
        grouped = value.view(*shape[:-1], self.diag_groups, self.hidden_dim // self.diag_groups)
        scale_view = scale.view(*((1,) * (grouped.dim() - 2)), self.diag_groups, 1)
        return (grouped * scale_view.to(dtype=value.dtype)).reshape(shape)

    def forward(self, x):
        value = self.value_proj(x)
        gate_source = self.query_proj(x)
        if self.rhg_gate_basis is None:
            gate_features = self.rhg_gate_act(gate_source)
        else:
            gate_features = self.rhg_gate_basis(gate_source)
        if self.rhg_gate_residual is not None:
            gate_features = gate_features + self.gate_residual_scale.to(dtype=gate_features.dtype) * self.rhg_gate_residual(gate_source)
        gate = self.gate_proj(gate_features)
        if self.rhg_diag_act is not None:
            gate = self.rhg_diag_act(gate)
        if self.rhg_cond_value_residual is not None:
            condition = self.value_condition_proj(gate_features)
            value = value + self.value_residual_scale.to(dtype=value.dtype) * self.rhg_cond_value_residual(value, condition)
        if self.rhg_value_residual is not None:
            value_residual = self.rhg_value_residual(value)
            if self.rhg_value_residual_odd is not None:
                value_residual = value_residual + self.value_residual_odd_scale.to(
                    dtype=value.dtype
                ) * self.rhg_value_residual_odd(value)
            if self.value_residual_condition:
                gate_shape = gate.shape
                grouped_gate = gate.view(*gate_shape[:-1], self.diag_groups, self.hidden_dim // self.diag_groups)
                condition_scale = self.value_residual_condition_scale.view(
                    *((1,) * (grouped_gate.dim() - 2)),
                    self.diag_groups,
                    1,
                )
                value_residual = value_residual.view_as(grouped_gate)
                if self.value_residual_condition_mode == "channel":
                    modulation = condition_scale.to(dtype=value.dtype) * torch.tanh(grouped_gate)
                else:
                    gate_mean = grouped_gate.mean(dim=-1, keepdim=True)
                    modulation = condition_scale.to(dtype=value.dtype) * torch.tanh(gate_mean)
                    if self.value_residual_moment_condition:
                        gate_rms = torch.sqrt(grouped_gate.square().mean(dim=-1, keepdim=True) + 1e-6)
                        rms_scale = self.value_residual_rms_condition_scale.view(
                            *((1,) * (grouped_gate.dim() - 2)),
                            self.diag_groups,
                            1,
                        )
                        modulation = modulation + rms_scale.to(dtype=value.dtype) * torch.tanh(torch.log(gate_rms))
                value_residual = value_residual * (1.0 + modulation)
                value_residual = value_residual.reshape(gate_shape)
            if self.value_cross_down is not None:
                cross_residual = self.value_cross_up(self.value_cross_down(value_residual))
                value_residual = value_residual + self.value_cross_scale.to(dtype=value.dtype) * cross_residual
            residual_scale = self.value_residual_scale
            if self.value_residual_safe_gate is not None:
                safe_gate = 2.0 * torch.sigmoid(self.value_residual_safe_gate)
                residual_scale = residual_scale * safe_gate
            if residual_scale.dim() == 1:
                value = value + self._apply_diag_group_scale(value_residual, residual_scale)
            else:
                value = value + residual_scale.to(dtype=value.dtype) * value_residual
        if self.rhg_value_act is not None:
            value = self.rhg_value_act(value)
        hidden = value * (1.0 + self.gate_scale.to(dtype=value.dtype) * gate)
        return self.out_proj(hidden)


def rkdm_settings(activation, d_model, ffn_dim, group_size, max_groups):
    if activation not in RKDM_ACTIVATIONS:
        return None

    if activation == "rkdm_ffn":
        hidden_dim = int(ffn_dim)
        query_rank = 128
        experts = 8
        expert_rank = 44
    elif activation == "rkdm_ffn_more_regions":
        hidden_dim = int(ffn_dim)
        query_rank = 128
        experts = 16
        expert_rank = 21
    elif activation == "rkdm_ffn_highrank":
        hidden_dim = (3 * int(ffn_dim)) // 4
        query_rank = 128
        experts = 4
        expert_rank = 183
    else:
        raise ValueError(f"unknown RKDM activation {activation}")

    return {
        "hidden_dim": hidden_dim,
        "query_rank": int(query_rank),
        "experts": int(experts),
        "expert_rank": int(expert_rank),
        "latent_groups": resolve_group_count(hidden_dim, group_size, max_groups),
    }


class RationalKernelDiagonalMixerMLP(nn.Module):
    def __init__(self, dim, ffn_dim, activation, rational_group_size, rational_max_groups, eps):
        super().__init__()
        settings = rkdm_settings(activation, dim, ffn_dim, rational_group_size, rational_max_groups)
        if settings is None:
            raise ValueError(f"unknown RKDM activation {activation}")
        self.activation_name = activation
        self.hidden_dim = settings["hidden_dim"]
        self.query_rank = settings["query_rank"]
        self.experts = settings["experts"]
        self.expert_rank = settings["expert_rank"]
        self.latent_groups = settings["latent_groups"]
        self.eps = float(eps)
        self.routing_eps = 1e-6

        self.value_proj = nn.Linear(dim, self.hidden_dim, bias=False)
        self.query_proj = nn.Linear(dim, self.query_rank, bias=False)
        from rational_opt import RationalGroupedVersionA5_4

        self.rkdm_value_act = RationalGroupedVersionA5_4(
            self.hidden_dim,
            init="silu",
            fit_range=5.0,
            group_size=rational_group_size,
            max_groups=rational_max_groups,
            groups=self.latent_groups,
        )
        self.expert_a = nn.Parameter(torch.empty(self.experts, self.hidden_dim, self.expert_rank))
        self.expert_b = nn.Parameter(torch.empty(self.experts, self.expert_rank, self.hidden_dim))
        self.expert_diag = nn.Parameter(torch.empty(self.experts, self.hidden_dim))
        self.rkdm_centers = nn.Parameter(torch.empty(self.experts, self.query_rank))
        self.rkdm_gamma_sqrt = nn.Parameter(torch.ones(self.experts, self.query_rank))
        self.rkdm_tau_sqrt = nn.Parameter(torch.ones(self.experts))
        self.out_proj = nn.Linear(self.hidden_dim, dim, bias=False)
        self.diag_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.lowrank_scale = nn.Parameter(torch.tensor(self.expert_rank ** -0.5, dtype=torch.float32))
        self.reset_rkdm_parameters()

    def reset_rkdm_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.expert_a, mean=0.0, std=0.02)
            nn.init.normal_(self.expert_b, mean=0.0, std=0.02)
            nn.init.normal_(self.expert_diag, mean=0.0, std=0.02)
            self.rkdm_centers.normal_()
            center_scale = torch.rsqrt(self.rkdm_centers.square().mean(dim=-1, keepdim=True) + self.eps)
            self.rkdm_centers.mul_(center_scale)
            self.rkdm_gamma_sqrt.fill_(1.0)
            self.rkdm_tau_sqrt.fill_(1.0)

    def _query_rms_unit(self, q):
        return q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + self.eps)

    def _routing_weights(self, q):
        gamma = self.rkdm_gamma_sqrt.square() + self.routing_eps
        tau = self.rkdm_tau_sqrt.square() + self.routing_eps
        diff = q.unsqueeze(-2) - self.rkdm_centers.view(*((1,) * (q.dim() - 1)), self.experts, self.query_rank)
        weighted_dist = (gamma.view(*((1,) * (q.dim() - 1)), self.experts, self.query_rank) * diff.square()).sum(dim=-1)
        kappa = tau.view(*((1,) * (q.dim() - 1)), self.experts) / (self.routing_eps + weighted_dist)
        return kappa / (kappa.sum(dim=-1, keepdim=True) + self.routing_eps)

    def forward(self, x):
        value = self.rkdm_value_act(self.value_proj(x))
        query = self._query_rms_unit(self.query_proj(x))
        weights = self._routing_weights(query)
        diag = torch.einsum("...e,em->...m", weights, self.expert_diag)
        hidden_rank = torch.einsum("...m,emr->...er", value, self.expert_a)
        mixed_rank = hidden_rank * weights.unsqueeze(-1)
        lowrank = torch.einsum("...er,erm->...m", mixed_rank, self.expert_b)
        z = value * (1.0 + self.diag_scale.to(dtype=value.dtype) * diag)
        z = z + self.lowrank_scale.to(dtype=value.dtype) * lowrank
        return self.out_proj(z)


def make_rational_activation(kind, ffn_dim, init, group_size, max_groups):
    if kind.endswith("_grouped"):
        from rational_opt import RationalGroupedVersionA5_4

        return RationalGroupedVersionA5_4(
            ffn_dim,
            init=init,
            fit_range=5.0,
            group_size=group_size,
            max_groups=max_groups,
        )

    from rational_opt import RationalVersionA5_4

    return RationalVersionA5_4(init=init, fit_range=5.0)


class GatedMLP(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        activation,
        rational_init,
        post_rational_init,
        rational_group_size,
        rational_max_groups,
        birational_alpha_init,
        birational_denominator_init,
        birational_eps,
    ):
        super().__init__()
        self.activation_name = activation
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.value = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)
        self.gate_act = None
        self.value_act = None
        self.post_act = None
        self.bi_interaction = None

        if activation == "silu":
            return
        if activation == "birational_glu":
            self.bi_interaction = BiRationalGLUInteraction(
                ffn_dim,
                group_size=rational_group_size,
                max_groups=rational_max_groups,
                alpha_init=birational_alpha_init,
                denominator_init=birational_denominator_init,
                eps=birational_eps,
            )
            return
        if activation not in RATIONAL_ACTIVATIONS:
            raise ValueError(f"unknown activation {activation}")

        is_grouped = activation.endswith("_grouped")
        rational_kind = "rational_grouped" if is_grouped else "rational"
        if activation in {"rational_a", "rational_grouped"}:
            self.gate_act = make_rational_activation(
                rational_kind,
                ffn_dim,
                rational_init,
                rational_group_size,
                rational_max_groups,
            )
        elif activation in {"rational_up", "rational_up_grouped"}:
            self.value_act = make_rational_activation(
                rational_kind,
                ffn_dim,
                rational_init,
                rational_group_size,
                rational_max_groups,
            )
        elif activation in {"rational_both", "rational_both_grouped"}:
            self.gate_act = make_rational_activation(
                rational_kind,
                ffn_dim,
                rational_init,
                rational_group_size,
                rational_max_groups,
            )
            self.value_act = make_rational_activation(
                rational_kind,
                ffn_dim,
                rational_init,
                rational_group_size,
                rational_max_groups,
            )
        elif activation in {"rational_product", "rational_product_grouped"}:
            self.post_act = make_rational_activation(
                rational_kind,
                ffn_dim,
                post_rational_init,
                rational_group_size,
                rational_max_groups,
            )
        elif activation in {"rational_swiglu_post", "rational_swiglu_post_grouped"}:
            self.post_act = make_rational_activation(
                rational_kind,
                ffn_dim,
                post_rational_init,
                rational_group_size,
                rational_max_groups,
            )

    def forward(self, x):
        gate = self.gate(x)
        value = self.value(x)
        if self.activation_name == "silu":
            hidden = F.silu(gate) * value
        elif self.activation_name == "birational_glu":
            hidden = self.bi_interaction(gate, value)
        elif self.activation_name in {"rational_a", "rational_grouped"}:
            hidden = self.gate_act(gate) * value
        elif self.activation_name in {"rational_up", "rational_up_grouped"}:
            hidden = F.silu(gate) * self.value_act(value)
        elif self.activation_name in {"rational_both", "rational_both_grouped"}:
            hidden = self.gate_act(gate) * self.value_act(value)
        elif self.activation_name in {"rational_product", "rational_product_grouped"}:
            hidden = self.post_act(gate * value)
        elif self.activation_name in {"rational_swiglu_post", "rational_swiglu_post_grouped"}:
            hidden = self.post_act(F.silu(gate) * value)
        else:
            raise ValueError(f"unknown activation {self.activation_name}")
        return self.down(hidden)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ffn_dim,
        seq_len,
        activation,
        rational_init,
        post_rational_init,
        rational_group_size,
        rational_max_groups,
        birational_alpha_init,
        birational_denominator_init,
        birational_eps,
        rational_basis_eps,
        layer_idx=0,
        layer_count=1,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, seq_len)
        if activation in RATIONAL_BASIS_ACTIVATIONS:
            self.mlp = RationalBasisMLP(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RQM_ACTIVATIONS:
            self.mlp = RationalQuotientMixerMLP(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RKM_ACTIVATIONS:
            self.mlp = RationalKernelMixtureMLP(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RAPM_ACTIVATIONS:
            self.mlp = RationalAmplitudePairMixerMLP(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RPF_ACTIVATIONS:
            self.mlp = RationalPolarizationMLP(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
            )
        elif activation in RPB_ACTIVATIONS:
            self.mlp = RationalPolarizedBasisMLP(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RWF_ACTIVATIONS:
            self.mlp = RationalWideFFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
            )
        elif activation in RMB_ACTIVATIONS:
            self.mlp = RationalMomentBasisFFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RMA_ACTIVATIONS:
            self.mlp = RationalMomentAdaptiveFFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RDA_ACTIVATIONS:
            self.mlp = RationalDynamicA5_4FFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RLBX_ACTIVATIONS:
            self.mlp = RationalLocalBasisExpansionFFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RLB_ACTIVATIONS:
            self.mlp = RationalLocalBasisFFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RCQ_ACTIVATIONS:
            self.mlp = RationalComplexQuadraticFFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RGC_ACTIVATIONS:
            self.mlp = RationalGroupCompetitiveFFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RSM_ACTIVATIONS:
            self.mlp = RationalSelfModulatedFFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        elif activation in RHG_ACTIVATIONS:
            self.mlp = RationalHyperGateFFN(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                layer_idx=layer_idx,
                layer_count=layer_count,
            )
        elif activation in RKDM_ACTIVATIONS:
            self.mlp = RationalKernelDiagonalMixerMLP(
                dim,
                ffn_dim,
                activation,
                rational_group_size,
                rational_max_groups,
                rational_basis_eps,
            )
        else:
            self.mlp = GatedMLP(
                dim,
                ffn_dim,
                activation,
                rational_init,
                post_rational_init,
                rational_group_size,
                rational_max_groups,
                birational_alpha_init,
                birational_denominator_init,
                birational_eps,
            )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, args.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    args.d_model,
                    args.heads,
                    args.ffn_dim,
                    args.seq_len,
                    args.activation,
                    args.rational_init,
                    args.post_rational_init,
                    args.rational_group_size,
                    args.rational_max_groups,
                    args.birational_alpha_init,
                    args.birational_denominator_init,
                    args.birational_eps,
                    args.rational_basis_eps,
                    layer_idx=layer_idx,
                    layer_count=args.layers,
                )
                for layer_idx in range(args.layers)
            ]
        )
        self.norm = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(lambda module: init_weights(module, args.init_std))

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def init_weights(module, init_std):
    if isinstance(module, nn.Linear):
        if getattr(module, "zero_init", False):
            nn.init.zeros_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=init_std)


def setup_distributed():
    if "RANK" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 0, 1, device

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size, torch.device("cuda", local_rank)


def cleanup_distributed(is_distributed):
    if is_distributed:
        dist.destroy_process_group()


def rank0_print(rank, message):
    if rank == 0:
        print(message, flush=True)


def sanitize_name(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def normalize_dataset_config(config):
    if config is None:
        return None
    if str(config).lower() in {"", "none", "null"}:
        return None
    return config


DOLMA_MANIFESTS = {
    "v1_5-sample": "urls/v1_5-sample.txt",
    "v1_6-sample": "urls/v1_6-sample.txt",
    "v1_6": "urls/v1_6.txt",
}


def actual_dataset_split(args, split):
    if split == "train":
        return args.train_split
    if split == "validation":
        return args.validation_split
    return split


def dataset_skip_documents(args, split):
    if split == "train":
        return max(0, int(args.train_skip_documents))
    if split == "validation":
        return max(0, int(args.validation_skip_documents))
    return 0


def dataset_skip_tokens(args, split):
    if split == "train":
        return max(0, int(args.train_skip_tokens))
    if split == "validation":
        return max(0, int(args.validation_skip_tokens))
    return 0


def append_tokenized_ids(tokens, ids, eos_id, max_tokens, skip_tokens):
    if not ids:
        return skip_tokens, False
    doc = list(ids)
    doc.append(eos_id)
    if skip_tokens > 0:
        if skip_tokens >= len(doc):
            return skip_tokens - len(doc), False
        doc = doc[skip_tokens:]
        skip_tokens = 0
    tokens.extend(doc)
    return skip_tokens, max_tokens is not None and len(tokens) >= max_tokens


def token_cache_path(args, split, max_tokens):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_part = "all" if max_tokens is None else str(max_tokens)
    config_name = args.dataset_config if args.dataset_config else "none"
    actual_split = actual_dataset_split(args, split)
    skip_docs = dataset_skip_documents(args, split)
    skip_tokens = dataset_skip_tokens(args, split)
    legacy_cache_name = (
        not args.dataset_streaming
        and args.dataset_text_column == "text"
        and actual_split == split
        and skip_docs == 0
        and skip_tokens == 0
        and args.train_split == "train"
        and args.validation_split == "validation"
    )
    if legacy_cache_name:
        name = (
            f"{sanitize_name(args.dataset_name)}_{sanitize_name(config_name)}_"
            f"{sanitize_name(args.tokenizer)}_{split}_{max_part}.pt"
        )
    else:
        stream_part = "stream" if args.dataset_streaming else "map"
        name = (
            f"{sanitize_name(args.dataset_name)}_{sanitize_name(config_name)}_"
            f"{sanitize_name(args.tokenizer)}_{sanitize_name(actual_split)}_{split}_"
            f"{stream_part}_{sanitize_name(args.dataset_text_column)}_skipdocs{skip_docs}_"
            f"skiptoks{skip_tokens}_{max_part}.pt"
        )
    return cache_dir / name


def extract_text_from_record(record, text_column):
    if text_column == "auto":
        for candidate in ("text", "content", "document", "raw_content"):
            value = record.get(candidate)
            if value:
                return value
        return None
    value = record
    for part in text_column.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def extract_texts_from_batch(batch, text_column):
    if text_column != "auto" and text_column in batch:
        values = batch[text_column]
    else:
        values = None
        for candidate in ("text", "content", "document", "raw_content"):
            if candidate in batch:
                values = batch[candidate]
                break
    if values is None:
        available = ", ".join(sorted(batch.keys()))
        raise KeyError(f"could not find text column {text_column!r}; available columns: {available}")
    return [text for text in values if isinstance(text, str) and text.strip()]


def load_dolma_manifest_dataset(args, dataset_config, actual_split):
    if actual_split != "train":
        raise ValueError("allenai/dolma manifest loading supports train split only; use train plus skip offsets for validation")
    manifest = DOLMA_MANIFESTS.get(dataset_config)
    if manifest is None:
        known = ", ".join(sorted(DOLMA_MANIFESTS))
        raise ValueError(f"unsupported allenai/dolma config {dataset_config!r}; expected one of: {known}")
    manifest_path = hf_hub_download(
        repo_id="allenai/dolma",
        filename=manifest,
        repo_type="dataset",
        cache_dir=args.hf_cache,
    )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        urls = [line.strip() for line in handle if line.strip()]
    if not urls:
        raise ValueError(f"empty allenai/dolma URL manifest: {manifest}")
    return load_dataset(
        "json",
        data_files={"train": urls},
        split="train",
        cache_dir=args.hf_cache,
        streaming=args.dataset_streaming,
    )


def load_hf_dataset(args, split):
    dataset_config = normalize_dataset_config(args.dataset_config)
    actual_split = actual_dataset_split(args, split)
    if args.dataset_name == "allenai/dolma" and dataset_config in DOLMA_MANIFESTS:
        return load_dolma_manifest_dataset(args, dataset_config, actual_split)
    kwargs = {
        "split": actual_split,
        "cache_dir": args.hf_cache,
        "streaming": args.dataset_streaming,
    }
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if dataset_config is None:
        return load_dataset(args.dataset_name, **kwargs)
    return load_dataset(args.dataset_name, dataset_config, **kwargs)


def synthetic_arithmetic_text(index: int) -> str:
    a = (index * 48271 + 17) % 10000
    b = (index * 69621 + 23) % 10000
    c = a + b
    lo = min(a, b)
    hi = max(a, b)
    step = index % 17 + 2
    first = (index * 37 + 11) % 2000
    seq = [first + step * offset for offset in range(5)]
    mode = index % 4
    if mode == 0:
        return f"Task add: {a} plus {b} equals {c}. Check: {c} minus {b} equals {a}."
    if mode == 1:
        return f"Task compare: between {a} and {b}, smaller is {lo}, larger is {hi}. Difference is {hi - lo}."
    if mode == 2:
        return f"Task sequence: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}; next is {seq[4]}. Step size is {step}."
    product = (a % 100) * (b % 100)
    return f"Task multiply-small: {a % 100} times {b % 100} equals {product}. Inputs came from {a} and {b}."


def synthetic_code_text(index: int) -> str:
    base = (index * 1103515245 + 12345) & 0x7FFFFFFF
    x = base % 97
    y = (base // 97) % 89
    z = (3 * x + 2 * y + index) % 211
    name = f"v{index % 17}"
    mode = index % 5
    if mode == 0:
        return f"def f({name}, y):\n    total = {name} + {x}\n    total = total * {y % 7 + 2}\n    return total\n# f({y}, {z % 11}) -> {(y + x) * (y % 7 + 2)}"
    if mode == 1:
        values = [(x + k * (y % 5 + 1)) % 100 for k in range(5)]
        return f"items = {values}\nacc = 0\nfor item in items:\n    acc += item\nprint(acc)  # {sum(values)}"
    if mode == 2:
        flag = (x + y) % 2 == 0
        chosen = x if flag else y
        return f"left = {x}\nright = {y}\nanswer = left if (left + right) % 2 == 0 else right\nanswer == {chosen}"
    if mode == 3:
        keys = [f"k{(index + j) % 9}" for j in range(3)]
        vals = [(x + y + j * 13) % 50 for j in range(3)]
        return f"data = {{{keys[0]!r}: {vals[0]}, {keys[1]!r}: {vals[1]}, {keys[2]!r}: {vals[2]}}}\nlookup = {keys[1]!r}\nvalue = data[lookup]  # {vals[1]}"
    final = x + 6 if x % 2 == (x + 6) % 2 else x + 7
    return f"while {name} < {x + 6}:\n    {name} += 2\n# start {x}, stop {x + 6}, final {final}"


def synthetic_symbolic_text(index: int) -> str:
    a = chr(ord("A") + index % 8)
    b = chr(ord("A") + (index * 3 + 1) % 8)
    c = chr(ord("A") + (index * 5 + 2) % 8)
    n = index % 7 + 3
    mode = index % 5
    if mode == 0:
        return f"Rewrite rule: ({a} -> {b}) and ({b} -> {c}). Query {a}. Result {c}."
    if mode == 1:
        seq = [((index + 2) * (k + 1) + n) % 19 for k in range(6)]
        return f"Map every token by +{n}: input {seq[:4]} output {[value + n for value in seq[:4]]}."
    if mode == 2:
        bits = [(index >> k) & 1 for k in range(6)]
        parity = sum(bits) % 2
        return f"Boolean trace: bits {bits}; xor parity is {parity}; not parity is {1 - parity}."
    if mode == 3:
        opens = index % 4 + 1
        token = "(" * opens + a + ")" * opens
        return f"Bracket task: source {token}; depth {opens}; core token {a}."
    left = [a, b, c, a]
    right = list(reversed(left))
    return f"Reverse-copy task: source {' '.join(left)}; target {' '.join(right)}."


def synthetic_reasoning_text(index: int) -> str:
    mode = index % 3
    if mode == 0:
        return synthetic_arithmetic_text(index)
    if mode == 1:
        return synthetic_code_text(index)
    return synthetic_symbolic_text(index)


def synthetic_text_for_dataset(dataset_name: str, index: int) -> str:
    if dataset_name == "synthetic/arithmetic":
        return synthetic_arithmetic_text(index)
    if dataset_name == "synthetic/code":
        return synthetic_code_text(index)
    if dataset_name == "synthetic/symbolic":
        return synthetic_symbolic_text(index)
    if dataset_name == "synthetic/reasoning_mix":
        return synthetic_reasoning_text(index)
    allowed = "synthetic/arithmetic, synthetic/code, synthetic/symbolic, synthetic/reasoning_mix"
    raise ValueError(f"unknown synthetic dataset {dataset_name!r}; expected one of: {allowed}")


def tokenize_synthetic_text(args, split, max_tokens, tokenizer, eos_id):
    if max_tokens is None:
        raise ValueError(f"{args.dataset_name} requires max_tokens")
    tokens = array("I")
    split_offset = {"train": 0, "validation": 50_000_000, "test": 75_000_000}.get(split, 90_000_000)
    index = 0
    while len(tokens) < max_tokens:
        texts = [
            synthetic_text_for_dataset(args.dataset_name, split_offset + index + item)
            for item in range(args.tokenize_batch_size)
        ]
        index += args.tokenize_batch_size
        batch = tokenizer(texts, add_special_tokens=False)
        for ids in batch["input_ids"]:
            if ids:
                tokens.extend(ids)
                tokens.append(eos_id)
            if len(tokens) >= max_tokens:
                break
    return tokens


def load_or_tokenize(args, split, max_tokens):
    cache_file = token_cache_path(args, split, max_tokens)
    if cache_file.exists() and not args.refresh_cache:
        payload = torch.load(cache_file, map_location="cpu")
        return payload["tokens"] if isinstance(payload, dict) else payload

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.hf_cache)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id or pad_token_id")

    if args.dataset_name.startswith("synthetic/"):
        tokens = tokenize_synthetic_text(args, split, max_tokens, tokenizer, eos_id)
    else:
        dataset = load_hf_dataset(args, split)
        skip_docs = dataset_skip_documents(args, split)
        skip_tokens = dataset_skip_tokens(args, split)
        tokens = array("I")
        if args.dataset_streaming:
            if skip_docs > 0 and hasattr(dataset, "skip"):
                dataset = dataset.skip(skip_docs)
                skipped = skip_docs
            else:
                skipped = 0
            texts = []
            for record in dataset:
                if skipped < skip_docs:
                    skipped += 1
                    continue
                text = extract_text_from_record(record, args.dataset_text_column)
                if isinstance(text, str) and text.strip():
                    texts.append(text)
                if len(texts) < args.tokenize_batch_size:
                    continue
                batch = tokenizer(texts, add_special_tokens=False)
                texts.clear()
                for ids in batch["input_ids"]:
                    skip_tokens, done = append_tokenized_ids(tokens, ids, eos_id, max_tokens, skip_tokens)
                    if done:
                        break
                if max_tokens is not None and len(tokens) >= max_tokens:
                    break
            if texts and (max_tokens is None or len(tokens) < max_tokens):
                batch = tokenizer(texts, add_special_tokens=False)
                for ids in batch["input_ids"]:
                    skip_tokens, done = append_tokenized_ids(tokens, ids, eos_id, max_tokens, skip_tokens)
                    if done:
                        break
        else:
            for start in range(skip_docs, len(dataset), args.tokenize_batch_size):
                end = min(start + args.tokenize_batch_size, len(dataset))
                texts = extract_texts_from_batch(dataset[start:end], args.dataset_text_column)
                if not texts:
                    continue
                batch = tokenizer(texts, add_special_tokens=False)
                for ids in batch["input_ids"]:
                    skip_tokens, done = append_tokenized_ids(tokens, ids, eos_id, max_tokens, skip_tokens)
                    if done:
                        break
                if max_tokens is not None and len(tokens) >= max_tokens:
                    break

    np_tokens = np.frombuffer(tokens, dtype=np.uint32)
    if max_tokens is not None:
        np_tokens = np_tokens[:max_tokens]
    if np_tokens.size < args.seq_len + 2:
        raise ValueError(f"{split} split produced too few tokens")
    if int(np_tokens.max()) > np.iinfo(np.int32).max:
        raise ValueError("token ids exceed int32 range")
    tensor = torch.from_numpy(np_tokens.astype(np.int32, copy=True))
    torch.save(
        {
            "tokens": tensor,
            "dataset": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": split,
            "actual_split": actual_dataset_split(args, split),
            "dataset_streaming": args.dataset_streaming,
            "dataset_text_column": args.dataset_text_column,
            "skip_documents": dataset_skip_documents(args, split),
            "skip_tokens": dataset_skip_tokens(args, split),
            "tokenizer": args.tokenizer,
        },
        cache_file,
    )
    return tensor


def sample_batch(tokens, batch_size, seq_len, offsets, generator, device):
    max_start = tokens.numel() - seq_len - 1
    starts = torch.randint(max_start, (batch_size,), generator=generator)
    block = tokens[starts[:, None] + offsets[None, :]]
    x = block[:, :-1].to(device=device, dtype=torch.long, non_blocking=True)
    y = block[:, 1:].to(device=device, dtype=torch.long, non_blocking=True)
    return x, y


def reduce_mean(value, device, is_distributed):
    tensor = torch.tensor(float(value), device=device)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def unwrap_model(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def _finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _tensor_mean_std(tensor):
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None, None
    values = tensor.detach().float().reshape(-1)
    return _finite_float(values.mean().item()), _finite_float(values.std(unbiased=False).item() if values.numel() > 1 else 0.0)


def _tensor_quantiles(tensor):
    if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
        return None, None, None
    values = tensor.detach().float().reshape(-1).cpu()
    return (
        _finite_float(values.min().item()),
        _finite_float(torch.quantile(values, 0.01).item()),
        _finite_float(values.median().item()),
    )


def grad_global_norm(model):
    total = None
    for param in model.parameters():
        if param.grad is None:
            continue
        value = param.grad.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(torch.sqrt(total).item())


def clip_or_measure_gradients(model, grad_clip, capture_norm):
    if grad_clip > 0:
        norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        norm_value = float(norm.item() if torch.is_tensor(norm) else norm)
        return norm_value, bool(norm_value > float(grad_clip))
    if capture_norm:
        return grad_global_norm(model), False
    return None, False


def iter_optimizer_tree(optimizer):
    yield optimizer
    for child in getattr(optimizer, "optimizers", []):
        yield from iter_optimizer_tree(child)
    for attr in ("adam", "muon"):
        child = getattr(optimizer, attr, None)
        if child is not None:
            yield child


def set_optimizer_telemetry_capture(optimizer, enabled):
    for item in iter_optimizer_tree(optimizer):
        setter = getattr(item, "set_telemetry_capture", None)
        if setter is not None:
            setter(enabled)


def collect_optimizer_telemetry(optimizer):
    record = {}
    for item in iter_optimizer_tree(optimizer):
        getter = getattr(item, "telemetry", None)
        if getter is None:
            continue
        telemetry = getter()
        if telemetry:
            record.update(telemetry)
    return record


def enable_rlb_training_telemetry(model, args):
    if args.activation not in RLB_ACTIVATIONS:
        return
    for module in model.modules():
        if not all(hasattr(module, attr) for attr in ("groups", "hidden_dim", "numerator", "denominator", "coeff_logits")):
            continue
        setattr(module, "_rlb_optimizer_track_stats", True)
        setattr(module, "_rlb_optimizer_stat_every", max(1, int(args.telemetry_rlb_stat_every)))
        setattr(module, "_rlb_optimizer_stat_samples", max(1, int(args.telemetry_rlb_stat_samples)))


def _rlb_denominator_probe(group, points=129, probe_range=5.0):
    module = group.get("module")
    if module is None:
        return None
    denominator = getattr(module, "denominator", None)
    centers = getattr(module, "centers", None)
    beta = getattr(module, "beta", None)
    if denominator is None:
        return None
    device = denominator.device
    dtype = torch.float32
    t = torch.linspace(-float(probe_range), float(probe_range), int(points), device=device, dtype=dtype)
    abs_t = t.abs()
    t2 = t.square()
    t3_abs = abs_t * t2
    t4 = t2.square()
    den = denominator.detach().float().abs()
    if den.dim() != 2 or den.size(-1) < 4:
        return None
    base_q = 1.0 + den[:, 0:1] * abs_t + den[:, 1:2] * t2 + den[:, 2:3] * t3_abs + den[:, 3:4] * t4
    values = [base_q.reshape(-1)]
    if centers is not None and beta is not None:
        center = centers.detach().float().unsqueeze(-1)
        beta_v = beta.detach().float().unsqueeze(-1).clamp_min(0.0)
        local_q = 1.0 + beta_v * (t.view(1, 1, -1) - center).square()
        values.append(local_q.reshape(-1))
    return torch.cat(values)


def collect_rlb_telemetry(model, args):
    if args.activation not in RLB_ACTIVATIONS:
        return {}
    groups = collect_rlb_optimizer_groups(unwrap_model(model), args)
    if not groups:
        return {}

    result = {}
    output_means, output_stds = [], []
    derivative_means, derivative_stds = [], []
    atom_means, atom_stds = [], []
    abs_moment_means, abs_moment_stds = [], []
    denom_mins, denom_p01s, denom_medians = [], [], []
    w_in_means, w_in_stds = [], []
    w_out_means, w_out_stds = [], []
    log_ratio_means, log_product_means = [], []

    for group in groups:
        module = group.get("module")
        stats = getattr(module, "_rlb_optimizer_stats", None) if module is not None else None
        for key, means, stds in (
            ("output_rms", output_means, output_stds),
            ("derivative_rms", derivative_means, derivative_stds),
            ("atom_rms", atom_means, atom_stds),
            ("abs_moments", abs_moment_means, abs_moment_stds),
        ):
            mean, std = _tensor_mean_std(None if not stats else stats.get(key))
            means.append(mean)
            stds.append(std)

        den_values = _rlb_denominator_probe(group, args.telemetry_denominator_probe_points)
        den_min, den_p01, den_median = _tensor_quantiles(den_values)
        denom_mins.append(den_min)
        denom_p01s.append(den_p01)
        denom_medians.append(den_median)

        in_weight = group["in_weight"].detach().float()
        out_weight = group["out_weight"].detach().float()
        groups_count = int(group["groups"])
        hidden_dim = int(group["hidden_dim"])
        width = hidden_dim // groups_count
        in_view = in_weight.view(groups_count, width, -1)
        out_view = out_weight.view(out_weight.shape[0], groups_count, width).permute(1, 2, 0)
        in_rms = torch.sqrt(in_view.square().mean(dim=(1, 2)) + 1e-12)
        out_rms = torch.sqrt(out_view.square().mean(dim=(1, 2)) + 1e-12)
        in_mean, in_std = _tensor_mean_std(in_rms)
        out_mean, out_std = _tensor_mean_std(out_rms)
        w_in_means.append(in_mean)
        w_in_stds.append(in_std)
        w_out_means.append(out_mean)
        w_out_stds.append(out_std)
        log_ratio_means.append(_finite_float((torch.log(in_rms) - torch.log(out_rms)).mean().item()))
        log_product_means.append(_finite_float((torch.log(in_rms) + torch.log(out_rms)).mean().item()))

    result.update(
        {
            "rlb_output_rms_mean_by_layer": output_means,
            "rlb_output_rms_std_by_layer": output_stds,
            "rlb_derivative_rms_mean_by_layer": derivative_means,
            "rlb_derivative_rms_std_by_layer": derivative_stds,
            "rlb_atom_rms_mean_by_layer": atom_means,
            "rlb_atom_rms_std_by_layer": atom_stds,
            "rlb_abs_moment_mean_by_layer": abs_moment_means,
            "rlb_abs_moment_std_by_layer": abs_moment_stds,
            "denominator_abs_min_by_layer": denom_mins,
            "denominator_abs_p01_by_layer": denom_p01s,
            "denominator_abs_median_by_layer": denom_medians,
            "w_in_rms_mean_by_layer": w_in_means,
            "w_in_rms_std_by_layer": w_in_stds,
            "w_out_rms_mean_by_layer": w_out_means,
            "w_out_rms_std_by_layer": w_out_stds,
            "log_w_in_over_w_out_by_layer": log_ratio_means,
            "log_norm_product_by_layer": log_product_means,
        }
    )
    return result


def svd_entropy(matrix, max_dim):
    weight = matrix.detach().float()
    if weight.dim() != 2 or min(weight.shape) <= 1:
        return None
    rows, cols = weight.shape
    if max_dim > 0 and rows > max_dim:
        row_index = torch.linspace(0, rows - 1, max_dim, device=weight.device).long()
        weight = weight.index_select(0, row_index)
    if max_dim > 0 and cols > max_dim:
        col_index = torch.linspace(0, cols - 1, max_dim, device=weight.device).long()
        weight = weight.index_select(1, col_index)
    try:
        singular = torch.linalg.svdvals(weight)
    except RuntimeError:
        return None
    singular = singular.clamp_min(0.0)
    total = singular.sum()
    if not torch.isfinite(total) or total <= 0:
        return None
    probs = singular / total
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    norm = math.log(max(2, probs.numel()))
    return _finite_float((entropy / norm).item())


def collect_matrix_spectrum_telemetry(model, args):
    if args.matrix_spectrum_interval <= 0:
        return {}
    raw_model = unwrap_model(model)
    values = {
        "svd_entropy_attn_q": [],
        "svd_entropy_attn_k": [],
        "svd_entropy_attn_v": [],
        "svd_entropy_attn_o": [],
        "svd_entropy_rlb_in": [],
        "svd_entropy_rlb_out": [],
    }
    for layer in getattr(raw_model, "layers", []):
        attn = getattr(layer, "attn", None)
        if attn is not None and hasattr(attn, "qkv"):
            q_weight, k_weight, v_weight = attn.qkv.weight.chunk(3, dim=0)
            for key, weight in (("svd_entropy_attn_q", q_weight), ("svd_entropy_attn_k", k_weight), ("svd_entropy_attn_v", v_weight)):
                value = svd_entropy(weight, args.matrix_spectrum_max_dim)
                if value is not None:
                    values[key].append(value)
        if attn is not None and hasattr(attn, "out"):
            value = svd_entropy(attn.out.weight, args.matrix_spectrum_max_dim)
            if value is not None:
                values["svd_entropy_attn_o"].append(value)
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, RationalLocalBasisFFN):
            value = svd_entropy(mlp.in_proj.weight, args.matrix_spectrum_max_dim)
            if value is not None:
                values["svd_entropy_rlb_in"].append(value)
            value = svd_entropy(mlp.out_proj.weight, args.matrix_spectrum_max_dim)
            if value is not None:
                values["svd_entropy_rlb_out"].append(value)
    return {key: _finite_float(sum(items) / len(items)) for key, items in values.items() if items}


def prepare_probe_batch(tokens, args, offsets, rank, device, out_path):
    if args.probe_batch_size <= 0:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 2_000_003 + rank)
    batch_size = min(int(args.probe_batch_size), int(args.batch_size))
    x, y = sample_batch(tokens, batch_size, args.seq_len, offsets, generator, device)
    probe_path = out_path.parent / f"{sanitize_name(args.activation)}_probe_rank{rank}.pt"
    if rank == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    torch.save(
        {
            "probe_x": x.detach().cpu(),
            "probe_y": y.detach().cpu(),
            "dataset": args.dataset_name,
            "dataset_config": args.dataset_config,
            "validation_skip_tokens": args.validation_skip_tokens,
            "seed": args.seed,
            "rank": rank,
        },
        probe_path,
    )
    return {"x": x, "prev_logits": None, "first_logits": None}


@torch.no_grad()
def evaluate_probe(model, probe_state, device, is_distributed):
    if probe_state is None:
        return {}
    model.eval()
    logits = model(probe_state["x"]).detach().float()
    logit_rms_local = torch.sqrt(logits.square().mean() + 1e-12)
    metrics = {"probe_logit_rms": reduce_mean(float(logit_rms_local.item()), device, is_distributed)}

    for label, reference in (("since_prev_eval", probe_state.get("prev_logits")), ("since_step1", probe_state.get("first_logits"))):
        if reference is None:
            metrics[f"probe_logit_delta_rms_{label}"] = 0.0
            metrics[f"probe_kl_{label}"] = 0.0
            continue
        ref = reference.to(device=device, dtype=torch.float32)
        delta_rms = torch.sqrt((logits - ref).square().mean() + 1e-12)
        log_probs = F.log_softmax(logits, dim=-1)
        ref_log_probs = F.log_softmax(ref, dim=-1)
        kl = (ref_log_probs.exp() * (ref_log_probs - log_probs)).sum(dim=-1).mean()
        metrics[f"probe_logit_delta_rms_{label}"] = reduce_mean(float(delta_rms.item()), device, is_distributed)
        metrics[f"probe_kl_{label}"] = reduce_mean(float(kl.item()), device, is_distributed)

    stored = logits.detach().to(device="cpu", dtype=torch.float16)
    if probe_state.get("first_logits") is None:
        probe_state["first_logits"] = stored.clone()
    probe_state["prev_logits"] = stored
    return metrics


@torch.no_grad()
def evaluate(model, tokens, args, offsets, rank, world_size, device, is_distributed):
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1_000_003 + rank)
    total_loss = 0.0
    for _ in range(args.eval_batches):
        x, y = sample_batch(tokens, args.batch_size, args.seq_len, offsets, generator, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        total_loss += float(loss.item())
    local_mean = total_loss / max(1, args.eval_batches)
    return reduce_mean(local_mean, device, is_distributed)


class CompositeOptimizer:
    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dict):
        for optimizer, optimizer_state in zip(self.optimizers, state_dict):
            optimizer.load_state_dict(optimizer_state)


def is_no_decay_parameter(name, param):
    return (
        param.dim() < 2
        or ".act." in name
        or ".activation." in name
        or ".rda_activation." in name
        or ".rlbx_activation." in name
        or ".rlb_activation." in name
        or ".rgc_activation." in name
        or ".rcq_activation." in name
        or ".bi_interaction." in name
        or ".rational_basis." in name
        or ".rqm_interaction." in name
        or ".rkm_centers" in name
        or ".rkm_gamma_sqrt" in name
        or ".rapm_activation." in name
        or ".rpf_plus." in name
        or ".rpf_minus." in name
        or ".rpb_odd." in name
        or ".rpb_even." in name
        or ".rwf_act." in name
        or ".rhg_cond_value_residual." in name
        or ".rhg_value_act." in name
        or ".rhg_value_residual." in name
        or ".rhg_gate_act." in name
        or ".rhg_gate_basis." in name
        or ".rhg_gate_residual." in name
        or ".rhg_diag_act." in name
        or ".rkdm_value_act." in name
        or ".rkdm_centers" in name
        or ".rkdm_gamma_sqrt" in name
    )


def split_decay_parameters(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_no_decay_parameter(name, param):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def is_rational_optimizer_parameter_name(name):
    rational_module_markers = (
        ".act.",
        ".activation.",
        ".rda_activation.",
        ".rlbx_activation.",
        ".rlb_activation.",
        ".rgc_activation.",
        ".rcq_activation.",
        ".bi_interaction.",
        ".rational_basis.",
        ".rqm_interaction.",
        ".rapm_activation.",
        ".rpf_plus.",
        ".rpf_minus.",
        ".rpb_odd.",
        ".rpb_even.",
        ".rwf_act.",
        ".rhg_cond_value_residual.",
        ".rhg_value_act.",
        ".rhg_value_residual.",
        ".rhg_gate_act.",
        ".rhg_gate_basis.",
        ".rhg_gate_residual.",
        ".rhg_diag_act.",
        ".rkdm_value_act.",
    )
    rational_parameter_names = (
        "numerator",
        "denominator",
        "coeff_logits",
        "center_logits",
        "atom_scale",
        "centers",
        "rkm_centers",
        "rkm_gamma_sqrt",
        "rkdm_centers",
        "rkdm_gamma_sqrt",
    )
    return any(marker in name for marker in rational_module_markers) or any(
        part in name for part in rational_parameter_names
    )


def rational_optimizer_role(name):
    if "numerator" in name:
        return "numerator"
    if "denominator" in name:
        return "denominator"
    if "coeff_logits" in name or "atom_scale" in name:
        return "atom"
    if "center" in name:
        return "center"
    return "other"


def rational_optimizer_layer_index(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return int(match.group(1)) if match else -1


def dense_layerwise_lr_scale(name, param, args):
    layer_index = rational_optimizer_layer_index(name)
    if layer_index < 0 or args.layers <= 1:
        return 1.0
    depth = float(layer_index) / float(args.layers - 1)
    scale = 1.0 + float(args.rational_dense_depth_gain) * (depth - 0.5)
    if is_no_decay_parameter(name, param):
        scale *= float(args.rational_dense_no_decay_lr_scale)
    return min(
        float(args.rational_dense_max_lr_scale),
        max(float(args.rational_dense_min_lr_scale), scale),
    )


def append_dense_layerwise_group(group_map, name, param, args):
    weight_decay = 0.0 if is_no_decay_parameter(name, param) else float(args.weight_decay)
    lr_scale = round(float(dense_layerwise_lr_scale(name, param, args)), 6)
    key = (weight_decay, lr_scale)
    group = group_map.get(key)
    if group is None:
        group = {"params": [], "weight_decay": weight_decay, "lr_scale": lr_scale}
        group_map[key] = group
    group["params"].append(param)


def count_rational_optimizer_parameters(model):
    return sum(
        param.numel()
        for name, param in model.named_parameters()
        if param.requires_grad and is_rational_optimizer_parameter_name(name)
    )


def collect_rlb_optimizer_groups(model, args):
    groups = []
    for module_name, module in model.named_modules():
        activation = getattr(module, "rlb_activation", None)
        in_proj = getattr(module, "in_proj", None)
        out_proj = getattr(module, "out_proj", None)
        if activation is None or in_proj is None or out_proj is None:
            continue
        required = ("groups", "hidden_dim", "numerator", "denominator", "coeff_logits", "centers", "beta", "coeff_limit")
        if not isinstance(in_proj, nn.Linear) or not isinstance(out_proj, nn.Linear):
            continue
        if not all(hasattr(activation, attr) for attr in required):
            continue
        if not isinstance(activation.numerator, nn.Parameter):
            continue
        if not isinstance(activation.denominator, nn.Parameter):
            continue
        if not isinstance(activation.coeff_logits, nn.Parameter):
            continue
        hidden_dim = int(activation.hidden_dim)
        groups_count = int(activation.groups)
        if in_proj.weight.shape[0] != hidden_dim or out_proj.weight.shape[1] != hidden_dim:
            continue
        groups.append(
            {
                "module": activation,
                "in_weight": in_proj.weight,
                "out_weight": out_proj.weight,
                "numerator": activation.numerator,
                "denominator": activation.denominator,
                "coeff_logits": activation.coeff_logits,
                "centers": activation.centers,
                "beta": activation.beta,
                "coeff_limit": float(activation.coeff_limit),
                "groups": groups_count,
                "hidden_dim": hidden_dim,
                "layer_index": rational_optimizer_layer_index(module_name + "."),
                "num_layers": args.layers,
            }
        )
    return groups


def resolve_ademamix_warmup(value, steps):
    if value is None:
        return None
    if int(value) < 0:
        return max(1, int(round(0.15 * int(steps))))
    if int(value) == 0:
        return None
    return int(value)


def configure_optimizer(model, args):
    if args.optimizer not in ACTIVE_OPTIMIZERS:
        allowed = ", ".join(ACTIVE_OPTIMIZERS)
        raise ValueError(f"Accepted optimizer choices: {allowed}")
    decay, no_decay = split_decay_parameters(model)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
    if args.optimizer == "factored_adamw":
        from optimizer_design import FactoredAdamW

        return FactoredAdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            factored_min_dim=args.factored_min_dim,
            clip_threshold=args.factored_clip_threshold,
        )
    if args.optimizer == "lion":
        from optimizer_design import Lion

        return Lion(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "ademamix":
        from optimizer_design import AdEMAMix

        beta3_warmup = resolve_ademamix_warmup(args.ademamix_beta3_warmup_steps, args.steps)
        alpha_warmup = resolve_ademamix_warmup(args.ademamix_alpha_warmup_steps, args.steps)
        return AdEMAMix(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.ademamix_beta3),
            eps=args.eps,
            weight_decay=args.weight_decay,
            alpha=args.ademamix_alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
        )
    if args.optimizer == "schedule_free_adamw":
        from optimizer_design import ScheduleFreeAdamW

        return ScheduleFreeAdamW(
            groups,
            lr=args.lr,
            beta1=args.schedule_free_beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
            warmup_steps=args.schedule_free_warmup_steps,
        )
    if args.optimizer == "adafactor_came":
        from optimizer_design import CAMEStyleAdamW

        return CAMEStyleAdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.came_beta3),
            eps=args.eps,
            weight_decay=args.weight_decay,
            factored_min_dim=args.factored_min_dim,
            clip_threshold=args.factored_clip_threshold,
            confidence_scale=args.came_confidence_scale,
            confidence_min=args.came_confidence_min,
            confidence_max=args.came_confidence_max,
        )
    if args.optimizer == "soap_adamw":
        from optimizer_design import SOAPStyleAdamW

        return SOAPStyleAdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.soap_precondition_frequency,
            large_side_identity_threshold=args.soap_large_side_identity_threshold,
            one_sided=args.soap_one_sided,
        )
    if args.optimizer == "muon":
        muon_named = []
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            no_decay_param = is_no_decay_parameter(name, param)
            tied_embedding = name in {"token_embedding.weight", "lm_head.weight"}
            if param.dim() == 2 and not no_decay_param and not tied_embedding:
                muon_named.append((name, param))
            elif no_decay_param or tied_embedding:
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)

        optimizers = []
        if muon_named:
            optimizers.append(
                torch.optim.Muon(
                    muon_named,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=args.muon_momentum,
                    ns_steps=args.muon_ns_steps,
                    adjust_lr_fn=args.muon_adjust_lr_fn,
                )
            )
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            )
        return CompositeOptimizer(optimizers)
    if args.optimizer == "rational_onpolicy_balance":
        from optimizer_design import FunctionSpaceRationalOptimizer, RationalOnPolicyBalanceOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_onpolicy_balance are RLB activations")

        rational_groups = []
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rational_optimizer_parameter_name(name):
                rational_groups.append({"params": [param], "role": rational_optimizer_role(name)})
            elif is_no_decay_parameter(name, param):
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
        if not rational_groups:
            raise ValueError("rational_onpolicy_balance requires trainable rational activation parameters")

        optimizers = []
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            )
        optimizers.append(
            FunctionSpaceRationalOptimizer(
                rational_groups,
                lr=args.lr,
                numerator_lr_scale=args.rational_coeff_num_lr_scale,
                denominator_lr_scale=args.rational_coeff_den_lr_scale,
                atom_lr_scale=args.rational_coeff_atom_lr_scale,
                center_lr_scale=args.rational_coeff_center_lr_scale,
                other_lr_scale=args.rational_coeff_other_lr_scale,
                trust=args.rational_coeff_trust,
                probe_range=args.rational_coeff_probe_range,
                probe_points=args.rational_coeff_probe_points,
                curve_decay=args.rational_coeff_curve_decay,
                update_gain=args.rational_coeff_update_gain,
                metric=args.rational_coeff_metric,
                metric_damping=args.rational_coeff_metric_damping,
                eps=args.rational_coeff_eps,
            )
        )
        return RationalOnPolicyBalanceOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            eps=args.rlb_gauge_eps,
        )

    if args.optimizer == "rational_quotient_onpolicy":
        from optimizer_design import FunctionSpaceRationalOptimizer, RationalQuotientOnPolicyOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_quotient_onpolicy are RLB activations")

        rational_groups = []
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rational_optimizer_parameter_name(name):
                rational_groups.append({"params": [param], "role": rational_optimizer_role(name)})
            elif is_no_decay_parameter(name, param):
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
        if not rational_groups:
            raise ValueError("rational_quotient_onpolicy requires trainable rational activation parameters")

        optimizers = []
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            )
        optimizers.append(
            FunctionSpaceRationalOptimizer(
                rational_groups,
                lr=args.lr,
                numerator_lr_scale=args.rational_coeff_num_lr_scale,
                denominator_lr_scale=args.rational_coeff_den_lr_scale,
                atom_lr_scale=args.rational_coeff_atom_lr_scale,
                center_lr_scale=args.rational_coeff_center_lr_scale,
                other_lr_scale=args.rational_coeff_other_lr_scale,
                trust=args.rational_coeff_trust,
                probe_range=args.rational_coeff_probe_range,
                probe_points=args.rational_coeff_probe_points,
                curve_decay=args.rational_coeff_curve_decay,
                update_gain=args.rational_coeff_update_gain,
                metric=args.rational_coeff_metric,
                metric_damping=args.rational_coeff_metric_damping,
                eps=args.rational_coeff_eps,
            )
        )
        return RationalQuotientOnPolicyOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            quotient_strength=args.rational_quotient_strength,
            eps=args.rlb_gauge_eps,
        )

    if args.optimizer == "rational_jacobian_onpolicy":
        from optimizer_design import FunctionSpaceRationalOptimizer, RationalJacobianOnPolicyOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_jacobian_onpolicy are RLB activations")

        rational_groups = []
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rational_optimizer_parameter_name(name):
                rational_groups.append({"params": [param], "role": rational_optimizer_role(name)})
            elif is_no_decay_parameter(name, param):
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
        if not rational_groups:
            raise ValueError("rational_jacobian_onpolicy requires trainable rational activation parameters")

        optimizers = []
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            )
        optimizers.append(
            FunctionSpaceRationalOptimizer(
                rational_groups,
                lr=args.lr,
                numerator_lr_scale=args.rational_coeff_num_lr_scale,
                denominator_lr_scale=args.rational_coeff_den_lr_scale,
                atom_lr_scale=args.rational_coeff_atom_lr_scale,
                center_lr_scale=args.rational_coeff_center_lr_scale,
                other_lr_scale=args.rational_coeff_other_lr_scale,
                trust=args.rational_coeff_trust,
                probe_range=args.rational_coeff_probe_range,
                probe_points=args.rational_coeff_probe_points,
                curve_decay=args.rational_coeff_curve_decay,
                update_gain=args.rational_coeff_update_gain,
                metric=args.rational_coeff_metric,
                metric_damping=args.rational_coeff_metric_damping,
                eps=args.rational_coeff_eps,
            )
        )
        return RationalJacobianOnPolicyOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            matrix_strength=args.rational_jacobian_matrix_strength,
            matrix_min_scale=args.rational_jacobian_min_scale,
            matrix_max_scale=args.rational_jacobian_max_scale,
            matrix_every=args.rational_jacobian_every,
            eps=args.rlb_gauge_eps,
        )


    if args.optimizer == "rational_jacobian_factored_onpolicy":
        from optimizer_design import FactoredAdamW, FunctionSpaceRationalOptimizer, RationalJacobianOnPolicyOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_jacobian_factored_onpolicy are RLB activations")

        rational_groups = []
        factored_decay = []
        factored_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rational_optimizer_parameter_name(name):
                rational_groups.append({"params": [param], "role": rational_optimizer_role(name)})
            elif is_no_decay_parameter(name, param):
                factored_no_decay.append(param)
            else:
                factored_decay.append(param)
        if not rational_groups:
            raise ValueError("rational_jacobian_factored_onpolicy requires trainable rational activation parameters")

        optimizers = []
        factored_groups = []
        if factored_decay:
            factored_groups.append({"params": factored_decay, "weight_decay": args.weight_decay})
        if factored_no_decay:
            factored_groups.append({"params": factored_no_decay, "weight_decay": 0.0})
        if factored_groups:
            optimizers.append(
                FactoredAdamW(
                    factored_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                    weight_decay=args.weight_decay,
                    factored_min_dim=args.factored_min_dim,
                    clip_threshold=args.factored_clip_threshold,
                )
            )
        optimizers.append(
            FunctionSpaceRationalOptimizer(
                rational_groups,
                lr=args.lr,
                numerator_lr_scale=args.rational_coeff_num_lr_scale,
                denominator_lr_scale=args.rational_coeff_den_lr_scale,
                atom_lr_scale=args.rational_coeff_atom_lr_scale,
                center_lr_scale=args.rational_coeff_center_lr_scale,
                other_lr_scale=args.rational_coeff_other_lr_scale,
                trust=args.rational_coeff_trust,
                probe_range=args.rational_coeff_probe_range,
                probe_points=args.rational_coeff_probe_points,
                curve_decay=args.rational_coeff_curve_decay,
                update_gain=args.rational_coeff_update_gain,
                metric=args.rational_coeff_metric,
                metric_damping=args.rational_coeff_metric_damping,
                eps=args.rational_coeff_eps,
            )
        )
        return RationalJacobianOnPolicyOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            matrix_strength=args.rational_jacobian_matrix_strength,
            matrix_min_scale=args.rational_jacobian_min_scale,
            matrix_max_scale=args.rational_jacobian_max_scale,
            matrix_every=args.rational_jacobian_every,
            eps=args.rlb_gauge_eps,
        )



    if args.optimizer in {"rational_layerwise_switch_onpolicy", "rational_layerwise_factored_switch_onpolicy"}:
        from optimizer_design import FactoredAdamW, RationalJacobianOnPolicyOptimizer, SwitchingRationalOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_layerwise_switch_onpolicy are RLB activations")

        curve_group_indices = {int(group.get("layer_index", -1)): index for index, group in enumerate(curve_groups)}
        rational_groups = []
        dense_group_map = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rational_optimizer_parameter_name(name):
                layer_index = rational_optimizer_layer_index(name)
                rational_groups.append(
                    {
                        "params": [param],
                        "role": rational_optimizer_role(name),
                        "weight_decay": 0.0,
                        "layer_index": layer_index,
                        "num_layers": args.layers,
                        "selector_index": curve_group_indices.get(layer_index, -1),
                    }
                )
            else:
                append_dense_layerwise_group(dense_group_map, name, param, args)
        if not rational_groups:
            raise ValueError("rational_layerwise_switch_onpolicy requires trainable rational activation parameters")

        optimizers = []
        dense_groups = list(dense_group_map.values())
        if dense_groups:
            if args.optimizer == "rational_layerwise_factored_switch_onpolicy":
                optimizers.append(
                    FactoredAdamW(
                        dense_groups,
                        lr=args.lr,
                        betas=(args.beta1, args.beta2),
                        eps=args.eps,
                        weight_decay=args.weight_decay,
                        factored_min_dim=args.factored_min_dim,
                        clip_threshold=args.factored_clip_threshold,
                    )
                )
            else:
                optimizers.append(
                    torch.optim.AdamW(
                        dense_groups,
                        lr=args.lr,
                        betas=(args.beta1, args.beta2),
                        eps=args.eps,
                    )
                )
        optimizers.append(
            SwitchingRationalOptimizer(
                rational_groups,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                weight_decay=0.0,
                total_steps=args.steps,
                switch_start=args.rational_switch_start,
                switch_end=args.rational_switch_end,
                switch_depth_shift=args.rational_switch_depth_shift,
                adam_lr_scale=args.rational_switch_adam_lr_scale,
                function_lr_scale=args.rational_switch_function_lr_scale,
                select_strength=args.rational_switch_select_strength,
                select_start=args.rational_switch_select_start,
                select_end=args.rational_switch_select_end,
                select_activity_threshold=args.rational_switch_select_activity_threshold,
                select_activity_width=args.rational_switch_select_activity_width,
                select_pressure_weight=args.rational_switch_select_pressure_weight,
                selector_groups=curve_groups,
                numerator_lr_scale=args.rational_coeff_num_lr_scale,
                denominator_lr_scale=args.rational_coeff_den_lr_scale,
                atom_lr_scale=args.rational_coeff_atom_lr_scale,
                center_lr_scale=args.rational_coeff_center_lr_scale,
                other_lr_scale=args.rational_coeff_other_lr_scale,
                trust=args.rational_coeff_trust,
                probe_range=args.rational_coeff_probe_range,
                probe_points=args.rational_coeff_probe_points,
                curve_decay=args.rational_coeff_curve_decay,
                update_gain=args.rational_coeff_update_gain,
                metric=args.rational_coeff_metric,
                metric_damping=args.rational_coeff_metric_damping,
                function_eps=args.rational_coeff_eps,
            )
        )
        return RationalJacobianOnPolicyOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            matrix_strength=args.rational_jacobian_matrix_strength,
            matrix_min_scale=args.rational_jacobian_min_scale,
            matrix_max_scale=args.rational_jacobian_max_scale,
            matrix_every=args.rational_jacobian_every,
            eps=args.rlb_gauge_eps,
        )

    if args.optimizer == "rational_quotient_jacobian_onpolicy":
        from optimizer_design import FunctionSpaceRationalOptimizer, RationalQuotientJacobianOnPolicyOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_quotient_jacobian_onpolicy are RLB activations")

        rational_groups = []
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rational_optimizer_parameter_name(name):
                rational_groups.append({"params": [param], "role": rational_optimizer_role(name)})
            elif is_no_decay_parameter(name, param):
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
        if not rational_groups:
            raise ValueError("rational_quotient_jacobian_onpolicy requires trainable rational activation parameters")

        optimizers = []
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            )
        optimizers.append(
            FunctionSpaceRationalOptimizer(
                rational_groups,
                lr=args.lr,
                numerator_lr_scale=args.rational_coeff_num_lr_scale,
                denominator_lr_scale=args.rational_coeff_den_lr_scale,
                atom_lr_scale=args.rational_coeff_atom_lr_scale,
                center_lr_scale=args.rational_coeff_center_lr_scale,
                other_lr_scale=args.rational_coeff_other_lr_scale,
                trust=args.rational_coeff_trust,
                probe_range=args.rational_coeff_probe_range,
                probe_points=args.rational_coeff_probe_points,
                curve_decay=args.rational_coeff_curve_decay,
                update_gain=args.rational_coeff_update_gain,
                metric=args.rational_coeff_metric,
                metric_damping=args.rational_coeff_metric_damping,
                eps=args.rational_coeff_eps,
            )
        )
        return RationalQuotientJacobianOnPolicyOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            matrix_strength=args.rational_jacobian_matrix_strength,
            matrix_min_scale=args.rational_jacobian_min_scale,
            matrix_max_scale=args.rational_jacobian_max_scale,
            matrix_every=args.rational_jacobian_every,
            quotient_strength=args.rational_qjacobian_quotient_strength,
            quotient_start=args.rational_qjacobian_quotient_start,
            quotient_end=args.rational_qjacobian_quotient_end,
            quotient_depth_gain=args.rational_qjacobian_quotient_depth_gain,
            eps=args.rlb_gauge_eps,
        )


    if args.optimizer == "rational_adaptive_metric_onpolicy":
        from optimizer_design import FunctionSpaceRationalOptimizer, RationalAdaptiveMetricOnPolicyOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_adaptive_metric_onpolicy are RLB activations")

        rational_groups = []
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rational_optimizer_parameter_name(name):
                rational_groups.append({"params": [param], "role": rational_optimizer_role(name)})
            elif is_no_decay_parameter(name, param):
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
        if not rational_groups:
            raise ValueError("rational_adaptive_metric_onpolicy requires trainable rational activation parameters")

        optimizers = []
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            )
        optimizers.append(
            FunctionSpaceRationalOptimizer(
                rational_groups,
                lr=args.lr,
                numerator_lr_scale=args.rational_coeff_num_lr_scale,
                denominator_lr_scale=args.rational_coeff_den_lr_scale,
                atom_lr_scale=args.rational_coeff_atom_lr_scale,
                center_lr_scale=args.rational_coeff_center_lr_scale,
                other_lr_scale=args.rational_coeff_other_lr_scale,
                trust=args.rational_coeff_trust,
                probe_range=args.rational_coeff_probe_range,
                probe_points=args.rational_coeff_probe_points,
                curve_decay=args.rational_coeff_curve_decay,
                update_gain=args.rational_coeff_update_gain,
                metric=args.rational_coeff_metric,
                metric_damping=args.rational_coeff_metric_damping,
                eps=args.rational_coeff_eps,
            )
        )
        return RationalAdaptiveMetricOnPolicyOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            matrix_strength=args.rational_adaptive_matrix_strength,
            matrix_min_scale=args.rational_adaptive_min_scale,
            matrix_max_scale=args.rational_adaptive_max_scale,
            matrix_every=args.rational_adaptive_every,
            stat_every=args.rational_adaptive_stat_every,
            stat_samples=args.rational_adaptive_stat_samples,
            coeff_strength=args.rational_adaptive_coeff_strength,
            coeff_start=args.rational_adaptive_coeff_start,
            coeff_end=args.rational_adaptive_coeff_end,
            coeff_late_decay=args.rational_adaptive_coeff_late_decay,
            coeff_metric_damping=args.rational_adaptive_coeff_metric_damping,
            coeff_norm_clip=args.rational_adaptive_coeff_norm_clip,
            coeff_max_blend=args.rational_adaptive_coeff_max_blend,
            coeff_depth_gain=args.rational_adaptive_coeff_depth_gain,
            matrix_time_gain=args.rational_adaptive_matrix_time_gain,
            matrix_depth_gain=args.rational_adaptive_matrix_depth_gain,
            quotient_strength=args.rational_adaptive_quotient_strength,
            eps=args.rlb_gauge_eps,
        )

    if args.optimizer == "rational_functional_trust_onpolicy":
        from optimizer_design import FunctionSpaceRationalOptimizer, RationalFunctionalTrustOnPolicyOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_functional_trust_onpolicy are RLB activations")

        rational_groups = []
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rational_optimizer_parameter_name(name):
                rational_groups.append({"params": [param], "role": rational_optimizer_role(name)})
            elif is_no_decay_parameter(name, param):
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
        if not rational_groups:
            raise ValueError("rational_functional_trust_onpolicy requires trainable rational activation parameters")

        optimizers = []
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            )
        optimizers.append(
            FunctionSpaceRationalOptimizer(
                rational_groups,
                lr=args.lr,
                numerator_lr_scale=args.rational_coeff_num_lr_scale,
                denominator_lr_scale=args.rational_coeff_den_lr_scale,
                atom_lr_scale=args.rational_coeff_atom_lr_scale,
                center_lr_scale=args.rational_coeff_center_lr_scale,
                other_lr_scale=args.rational_coeff_other_lr_scale,
                trust=args.rational_coeff_trust,
                probe_range=args.rational_coeff_probe_range,
                probe_points=args.rational_coeff_probe_points,
                curve_decay=args.rational_coeff_curve_decay,
                update_gain=args.rational_coeff_update_gain,
                metric=args.rational_coeff_metric,
                metric_damping=args.rational_coeff_metric_damping,
                eps=args.rational_coeff_eps,
            )
        )
        return RationalFunctionalTrustOnPolicyOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            matrix_strength=args.rational_adaptive_matrix_strength,
            matrix_min_scale=args.rational_adaptive_min_scale,
            matrix_max_scale=args.rational_adaptive_max_scale,
            matrix_every=args.rational_adaptive_every,
            stat_every=args.rational_adaptive_stat_every,
            stat_samples=args.rational_adaptive_stat_samples,
            coeff_metric_damping=args.rational_adaptive_coeff_metric_damping,
            coeff_norm_clip=args.rational_adaptive_coeff_norm_clip,
            matrix_time_gain=args.rational_adaptive_matrix_time_gain,
            matrix_depth_gain=args.rational_adaptive_matrix_depth_gain,
            quotient_strength=args.rational_adaptive_quotient_strength,
            trust_coeff_strength=args.rational_trust_coeff_strength,
            trust_radius=args.rational_trust_radius,
            trust_min_scale=args.rational_trust_min_scale,
            trust_max_scale=args.rational_trust_max_scale,
            trust_activity_target=args.rational_trust_activity_target,
            trust_activity_width=args.rational_trust_activity_width,
            trust_pressure_weight=args.rational_trust_pressure_weight,
            trust_agreement_decay=args.rational_trust_agreement_decay,
            trust_agreement_floor=args.rational_trust_agreement_floor,
            trust_metric_blend=args.rational_trust_metric_blend,
            trust_denominator_risk=args.rational_trust_denominator_risk,
            trust_atom_risk=args.rational_trust_atom_risk,
            trust_numerator_risk=args.rational_trust_numerator_risk,
            trust_depth_gain=args.rational_trust_depth_gain,
            eps=args.rlb_gauge_eps,
        )


    if args.optimizer == "rational_matrix_policy_onpolicy":
        from optimizer_design import FunctionSpaceRationalOptimizer, RationalMatrixPolicyOptimizer, RationalTransportOnPolicyOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_matrix_policy_onpolicy are RLB activations")

        curve_group_indices = {int(group.get("layer_index", -1)): index for index, group in enumerate(curve_groups)}
        matrix_param_ids = set()
        rational_param_ids = set()
        rational_groups = []
        matrix_groups = []
        for group in curve_groups:
            layer_index = int(group.get("layer_index", -1))
            selector_index = curve_group_indices.get(layer_index, -1)
            matrix_param_ids.add(id(group["in_weight"]))
            matrix_param_ids.add(id(group["out_weight"]))
            matrix_weight_decay = args.weight_decay * args.rational_matrix_policy_weight_decay_scale
            matrix_groups.append(
                {
                    "params": [group["in_weight"]],
                    "weight_decay": matrix_weight_decay,
                    "layer_index": layer_index,
                    "num_layers": args.layers,
                    "selector_index": selector_index,
                    "matrix_role": "in",
                }
            )
            matrix_groups.append(
                {
                    "params": [group["out_weight"]],
                    "weight_decay": matrix_weight_decay,
                    "layer_index": layer_index,
                    "num_layers": args.layers,
                    "selector_index": selector_index,
                    "matrix_role": "out",
                }
            )

        if args.rational_matrix_policy_function_coeff:
            for name, param in model.named_parameters():
                if not param.requires_grad or id(param) in matrix_param_ids:
                    continue
                if not is_rational_optimizer_parameter_name(name):
                    continue
                layer_index = rational_optimizer_layer_index(name)
                rational_param_ids.add(id(param))
                rational_groups.append(
                    {
                        "params": [param],
                        "role": rational_optimizer_role(name),
                        "layer_index": layer_index,
                        "num_layers": args.layers,
                        "selector_index": curve_group_indices.get(layer_index, -1),
                    }
                )

        backbone_muon_named = []
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in matrix_param_ids or id(param) in rational_param_ids:
                continue
            no_decay_param = is_no_decay_parameter(name, param)
            tied_embedding = name in {"token_embedding.weight", "lm_head.weight"}
            if (
                args.rational_matrix_policy_backbone_optimizer == "muon"
                and param.dim() == 2
                and not no_decay_param
                and not tied_embedding
            ):
                backbone_muon_named.append((name, param))
            elif no_decay_param or tied_embedding:
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)

        optimizers = []
        if backbone_muon_named:
            optimizers.append(
                torch.optim.Muon(
                    backbone_muon_named,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=args.muon_momentum,
                    ns_steps=args.muon_ns_steps,
                    adjust_lr_fn=args.muon_adjust_lr_fn,
                )
            )
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(
                        args.beta1,
                        args.beta2
                        if args.rational_matrix_policy_backbone_beta2 is None
                        else args.rational_matrix_policy_backbone_beta2,
                    ),
                    eps=args.eps,
                )
            )
        if rational_groups:
            optimizers.append(
                FunctionSpaceRationalOptimizer(
                    rational_groups,
                    lr=args.lr,
                    numerator_lr_scale=args.rational_coeff_num_lr_scale,
                    denominator_lr_scale=args.rational_transport_coeff_den_lr_scale,
                    denominator_lr_scale_final=args.rational_transport_coeff_den_lr_scale_final,
                    atom_lr_scale=args.rational_transport_coeff_atom_lr_scale,
                    atom_lr_scale_final=args.rational_transport_coeff_atom_lr_scale_final,
                    center_lr_scale=args.rational_coeff_center_lr_scale,
                    other_lr_scale=args.rational_coeff_other_lr_scale,
                    trust=args.rational_transport_coeff_trust,
                    trust_final=args.rational_transport_coeff_trust_final,
                    probe_range=args.rational_coeff_probe_range,
                    probe_points=args.rational_coeff_probe_points,
                    curve_decay=args.rational_coeff_curve_decay,
                    update_gain=args.rational_transport_coeff_update_gain,
                    update_gain_final=args.rational_transport_coeff_update_gain_final,
                    update_gain_decay_start=args.rational_transport_coeff_decay_start,
                    update_gain_decay_end=args.rational_transport_coeff_decay_end,
                    update_depth_gain=args.rational_transport_coeff_depth_gain,
                    update_switch_depth_shift=args.rational_transport_coeff_switch_depth_shift,
                    reset_on_switch=args.rational_transport_coeff_reset_on_switch,
                    selector_groups=curve_groups,
                    select_strength=args.rational_transport_coeff_select_strength,
                    select_start=args.rational_transport_coeff_select_start,
                    select_end=args.rational_transport_coeff_select_end,
                    select_activity_threshold=args.rational_transport_coeff_select_activity_threshold,
                    select_activity_width=args.rational_transport_coeff_select_activity_width,
                    select_pressure_weight=args.rational_transport_coeff_select_pressure_weight,
                    denominator_decay=args.rational_transport_coeff_den_decay,
                    atom_decay=args.rational_transport_coeff_atom_decay,
                    total_steps=args.steps,
                    metric=args.rational_coeff_metric,
                    metric_damping=args.rational_coeff_metric_damping,
                    eps=args.rational_coeff_eps,
                )
            )
        optimizers.append(
            RationalMatrixPolicyOptimizer(
                matrix_groups,
                lr=args.lr,
                betas=(
                    args.beta1,
                    args.beta2 if args.rational_matrix_policy_beta2 is None else args.rational_matrix_policy_beta2,
                ),
                eps=args.eps,
                weight_decay=args.weight_decay,
                total_steps=args.steps,
                selector_groups=curve_groups,
                muon_strength=args.rational_matrix_policy_muon_strength,
                muon_lr_scale=args.rational_matrix_policy_muon_lr_scale,
                adam_lr_scale=args.rational_matrix_policy_adam_lr_scale,
                adam_lr_scale_final=args.rational_matrix_policy_adam_lr_scale_final,
                adam_decay_start=args.rational_matrix_policy_adam_decay_start,
                adam_decay_end=args.rational_matrix_policy_adam_decay_end,
                adam_decay_depth_shift=args.rational_matrix_policy_adam_decay_depth_shift,
                adam_beta2_final=args.rational_matrix_policy_adam_beta2_final,
                adam_beta2_input_final=args.rational_matrix_policy_adam_beta2_input_final,
                adam_beta2_output_final=args.rational_matrix_policy_adam_beta2_output_final,
                adam_beta2_decay_start=args.rational_matrix_policy_adam_beta2_decay_start,
                adam_beta2_decay_end=args.rational_matrix_policy_adam_beta2_decay_end,
                adam_beta2_decay_depth_shift=args.rational_matrix_policy_adam_beta2_decay_depth_shift,
                adam_role_strength=args.rational_matrix_policy_adam_role_strength,
                adam_stat_strength=args.rational_matrix_policy_adam_stat_strength,
                adam_pressure_balance=args.rational_matrix_policy_adam_pressure_balance,
                adam_stat_start=args.rational_matrix_policy_adam_stat_start,
                adam_stat_end=args.rational_matrix_policy_adam_stat_end,
                adam_min_lr_scale=args.rational_matrix_policy_adam_min_lr_scale,
                adam_max_lr_scale=args.rational_matrix_policy_adam_max_lr_scale,
                adam_reset_on_switch=args.rational_matrix_policy_adam_reset_on_switch,
                start=args.rational_matrix_policy_start,
                end=args.rational_matrix_policy_end,
                decay_start=args.rational_matrix_policy_decay_start,
                decay_end=args.rational_matrix_policy_decay_end,
                muon_decay_depth_shift=args.rational_matrix_policy_muon_decay_depth_shift,
                muon_input_decay_shift=args.rational_matrix_policy_muon_input_decay_shift,
                muon_output_decay_shift=args.rational_matrix_policy_muon_output_decay_shift,
                muon_reset_adam_state=args.rational_matrix_policy_muon_reset_adam_state,
                final_muon=args.rational_matrix_policy_final_muon,
                min_muon=args.rational_matrix_policy_min_muon,
                max_muon=args.rational_matrix_policy_max_muon,
                input_depth_gain=args.rational_matrix_policy_input_depth_gain,
                output_depth_gain=args.rational_matrix_policy_output_depth_gain,
                pressure_weight=args.rational_matrix_policy_pressure_weight,
                activity_weight=args.rational_matrix_policy_activity_weight,
                activity_target=args.rational_matrix_policy_activity_target,
                activity_width=args.rational_matrix_policy_activity_width,
                pressure_clip=args.rational_matrix_policy_pressure_clip,
                group_gain_strength=args.rational_matrix_policy_group_gain_strength,
                group_pressure_strength=args.rational_matrix_policy_group_pressure_strength,
                group_activity_damping=args.rational_matrix_policy_group_activity_damping,
                group_activity_target=args.rational_matrix_policy_group_activity_target,
                group_activity_width=args.rational_matrix_policy_group_activity_width,
                group_start=args.rational_matrix_policy_group_start,
                group_end=args.rational_matrix_policy_group_end,
                group_min_scale=args.rational_matrix_policy_group_min_scale,
                group_max_scale=args.rational_matrix_policy_group_max_scale,
                muon_momentum=args.muon_momentum,
                muon_ns_steps=args.muon_ns_steps,
                muon_adjust_lr_fn=args.muon_adjust_lr_fn,
            )
        )
        return RationalTransportOnPolicyOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            matrix_strength=args.rational_transport_matrix_strength,
            matrix_min_scale=args.rational_adaptive_min_scale,
            matrix_max_scale=args.rational_adaptive_max_scale,
            matrix_every=args.rational_adaptive_every,
            stat_every=args.rational_transport_stat_every,
            stat_samples=args.rational_adaptive_stat_samples,
            coeff_strength=args.rational_adaptive_coeff_strength,
            coeff_start=args.rational_adaptive_coeff_start,
            coeff_end=args.rational_adaptive_coeff_end,
            coeff_late_decay=args.rational_adaptive_coeff_late_decay,
            coeff_metric_damping=args.rational_adaptive_coeff_metric_damping,
            coeff_norm_clip=args.rational_adaptive_coeff_norm_clip,
            coeff_max_blend=args.rational_adaptive_coeff_max_blend,
            coeff_depth_gain=args.rational_adaptive_coeff_depth_gain,
            matrix_time_gain=args.rational_transport_matrix_time_gain,
            matrix_depth_gain=args.rational_transport_matrix_depth_gain,
            quotient_strength=args.rational_transport_quotient_strength,
            transport_strength=args.rational_transport_strength,
            transport_final_strength=args.rational_transport_final_strength,
            transport_start=args.rational_transport_start,
            transport_end=args.rational_transport_end,
            transport_decay_start=args.rational_transport_decay_start,
            transport_decay_end=args.rational_transport_decay_end,
            transport_every=args.rational_transport_every,
            transport_max_log_step=args.rational_transport_max_log_step,
            transport_derivative_weight=args.rational_transport_derivative_weight,
            transport_headroom=args.rational_transport_headroom,
            transport_depth_gain=args.rational_transport_depth_gain,
            transport_derivative_depth_gain=args.rational_transport_derivative_depth_gain,
            matrix_input_depth_gain=args.rational_transport_matrix_input_depth_gain,
            matrix_output_depth_gain=args.rational_transport_matrix_output_depth_gain,
            matrix_live_stats=args.rational_transport_live_matrix_stats,
            pressure_precond_strength=args.rational_transport_pressure_strength,
            pressure_precond_depth_gain=args.rational_transport_pressure_depth_gain,
            pressure_precond_min_scale=args.rational_transport_pressure_min_scale,
            pressure_precond_max_scale=args.rational_transport_pressure_max_scale,
            eps=args.rlb_gauge_eps,
        )

    if args.optimizer == "rational_transport_onpolicy":
        from optimizer_design import FunctionSpaceRationalOptimizer, RationalTransportOnPolicyOptimizer

        curve_groups = collect_rlb_optimizer_groups(model, args)
        if not curve_groups:
            raise ValueError("Accepted activations for rational_transport_onpolicy are RLB activations")

        rational_groups = []
        curve_group_indices = {int(group.get("layer_index", -1)): index for index, group in enumerate(curve_groups)}
        adam_decay = []
        adam_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rational_optimizer_parameter_name(name):
                layer_index = rational_optimizer_layer_index(name)
                rational_groups.append(
                    {
                        "params": [param],
                        "role": rational_optimizer_role(name),
                        "layer_index": layer_index,
                        "num_layers": args.layers,
                        "selector_index": curve_group_indices.get(layer_index, -1),
                    }
                )
            elif is_no_decay_parameter(name, param):
                adam_no_decay.append(param)
            else:
                adam_decay.append(param)
        if not rational_groups:
            raise ValueError("rational_transport_onpolicy requires trainable rational activation parameters")

        optimizers = []
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": args.weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(
                    adam_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            )
        optimizers.append(
            FunctionSpaceRationalOptimizer(
                rational_groups,
                lr=args.lr,
                numerator_lr_scale=args.rational_coeff_num_lr_scale,
                denominator_lr_scale=args.rational_transport_coeff_den_lr_scale,
                denominator_lr_scale_final=args.rational_transport_coeff_den_lr_scale_final,
                atom_lr_scale=args.rational_transport_coeff_atom_lr_scale,
                atom_lr_scale_final=args.rational_transport_coeff_atom_lr_scale_final,
                center_lr_scale=args.rational_coeff_center_lr_scale,
                other_lr_scale=args.rational_coeff_other_lr_scale,
                trust=args.rational_transport_coeff_trust,
                trust_final=args.rational_transport_coeff_trust_final,
                probe_range=args.rational_coeff_probe_range,
                probe_points=args.rational_coeff_probe_points,
                curve_decay=args.rational_coeff_curve_decay,
                update_gain=args.rational_transport_coeff_update_gain,
                update_gain_final=args.rational_transport_coeff_update_gain_final,
                update_gain_decay_start=args.rational_transport_coeff_decay_start,
                update_gain_decay_end=args.rational_transport_coeff_decay_end,
                update_depth_gain=args.rational_transport_coeff_depth_gain,
                update_switch_depth_shift=args.rational_transport_coeff_switch_depth_shift,
                reset_on_switch=args.rational_transport_coeff_reset_on_switch,
                selector_groups=curve_groups,
                select_strength=args.rational_transport_coeff_select_strength,
                select_start=args.rational_transport_coeff_select_start,
                select_end=args.rational_transport_coeff_select_end,
                select_activity_threshold=args.rational_transport_coeff_select_activity_threshold,
                select_activity_width=args.rational_transport_coeff_select_activity_width,
                select_pressure_weight=args.rational_transport_coeff_select_pressure_weight,
                denominator_decay=args.rational_transport_coeff_den_decay,
                atom_decay=args.rational_transport_coeff_atom_decay,
                total_steps=args.steps,
                metric=args.rational_coeff_metric,
                metric_damping=args.rational_coeff_metric_damping,
                eps=args.rational_coeff_eps,
            )
        )
        return RationalTransportOnPolicyOptimizer(
            optimizers,
            curve_groups,
            total_steps=args.steps,
            target_weight=args.rlb_gauge_target_weight,
            metric_every=args.rlb_gauge_metric_every,
            probe_range=args.rlb_gauge_probe_range,
            probe_points=args.rlb_gauge_probe_points,
            strength=args.rlb_gauge_strength,
            max_log_step=args.rlb_gauge_max_log_step,
            start=args.rlb_gauge_start,
            end=args.rlb_gauge_end,
            depth_gain=args.rlb_gauge_depth_gain,
            every=args.rlb_gauge_every,
            stat_decay=args.rational_onpolicy_stat_decay,
            pressure_weight=args.rational_onpolicy_pressure_weight,
            pressure_clip=args.rational_onpolicy_pressure_clip,
            rational_activity_weight=args.rational_onpolicy_rational_activity_weight,
            activity_gain_min=args.rational_onpolicy_activity_gain_min,
            activity_gain_max=args.rational_onpolicy_activity_gain_max,
            covariant_state=True,
            matrix_strength=args.rational_transport_matrix_strength,
            matrix_min_scale=args.rational_adaptive_min_scale,
            matrix_max_scale=args.rational_adaptive_max_scale,
            matrix_every=args.rational_adaptive_every,
            stat_every=args.rational_transport_stat_every,
            stat_samples=args.rational_adaptive_stat_samples,
            coeff_strength=args.rational_adaptive_coeff_strength,
            coeff_start=args.rational_adaptive_coeff_start,
            coeff_end=args.rational_adaptive_coeff_end,
            coeff_late_decay=args.rational_adaptive_coeff_late_decay,
            coeff_metric_damping=args.rational_adaptive_coeff_metric_damping,
            coeff_norm_clip=args.rational_adaptive_coeff_norm_clip,
            coeff_max_blend=args.rational_adaptive_coeff_max_blend,
            coeff_depth_gain=args.rational_adaptive_coeff_depth_gain,
            matrix_time_gain=args.rational_transport_matrix_time_gain,
            matrix_depth_gain=args.rational_transport_matrix_depth_gain,
            quotient_strength=args.rational_transport_quotient_strength,
            transport_strength=args.rational_transport_strength,
            transport_final_strength=args.rational_transport_final_strength,
            transport_start=args.rational_transport_start,
            transport_end=args.rational_transport_end,
            transport_decay_start=args.rational_transport_decay_start,
            transport_decay_end=args.rational_transport_decay_end,
            transport_every=args.rational_transport_every,
            transport_max_log_step=args.rational_transport_max_log_step,
            transport_derivative_weight=args.rational_transport_derivative_weight,
            transport_headroom=args.rational_transport_headroom,
            transport_depth_gain=args.rational_transport_depth_gain,
            transport_derivative_depth_gain=args.rational_transport_derivative_depth_gain,
            matrix_input_depth_gain=args.rational_transport_matrix_input_depth_gain,
            matrix_output_depth_gain=args.rational_transport_matrix_output_depth_gain,
            matrix_live_stats=args.rational_transport_live_matrix_stats,
            pressure_precond_strength=args.rational_transport_pressure_strength,
            pressure_precond_depth_gain=args.rational_transport_pressure_depth_gain,
            pressure_precond_min_scale=args.rational_transport_pressure_min_scale,
            pressure_precond_max_scale=args.rational_transport_pressure_max_scale,
            eps=args.rlb_gauge_eps,
        )


    allowed = ", ".join(ACTIVE_OPTIMIZERS)
    raise ValueError(f"Accepted optimizer choices: {allowed}")

def learning_rate(step, args):
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return args.lr * float(step + 1) / float(args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + cosine * (args.lr - args.min_lr)


def sam_rho(step, args):
    rho = float(args.sam_rho)
    if rho <= 0.0:
        return 0.0
    progress = float(step + 1) / max(1, int(args.steps))
    if progress < float(args.sam_start) or progress > float(args.sam_end):
        return 0.0
    if args.sam_warmup > 0.0:
        warm_end = min(float(args.sam_end), float(args.sam_start) + float(args.sam_warmup))
        if progress < warm_end:
            denom = max(1e-8, warm_end - float(args.sam_start))
            rho *= min(1.0, max(0.0, (progress - float(args.sam_start)) / denom))
    return rho


def sam_parameter_scale(name, param, args):
    scale = 1.0
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if match and args.layers > 1:
        depth = float(int(match.group(1))) / float(args.layers - 1)
        scale *= min(2.0, max(0.25, 1.0 + float(args.sam_depth_gain) * (depth - 0.5)))
    if is_rational_optimizer_parameter_name(name):
        scale *= float(args.sam_rational_scale)
    if param.dim() < 2:
        scale *= float(args.sam_no_decay_scale)
    return max(0.0, scale)


@torch.no_grad()
def sam_first_step(model, rho, args, device, is_distributed):
    perturbations = []
    if rho <= 0.0:
        return perturbations
    local_sq = torch.zeros((), device=device)
    entries = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        scale = sam_parameter_scale(name, param, args)
        if scale <= 0.0:
            continue
        grad = param.grad.detach()
        if bool(args.sam_adaptive):
            adaptive = param.detach().abs().clamp_min(float(args.sam_adaptive_eps))
            norm_term = grad * adaptive * scale
        else:
            adaptive = None
            norm_term = grad * scale
        local_sq.add_(norm_term.float().square().sum())
        entries.append((param, grad, adaptive, scale))
    if is_distributed:
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    grad_norm = torch.sqrt(local_sq).clamp_min(float(args.sam_eps))
    rho_over_norm = float(rho) / float(grad_norm.item())
    for param, grad, adaptive, scale in entries:
        if adaptive is None:
            direction = grad * scale
        else:
            direction = grad * adaptive.square() * scale
        perturb = direction * rho_over_norm
        param.add_(perturb)
        perturbations.append((param, perturb))
    return perturbations


@torch.no_grad()
def sam_restore(perturbations):
    for param, perturb in perturbations:
        param.sub_(perturb)


def write_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activation",
        choices=[
            "birational_glu",
            "crv_rhg",
            "plain_rational_ffn",
            "rational_basis_k2",
            "rational_basis_k3_equal",
            "rational_basis_k2_per_channel",
            "rqm_ffn",
            "rqm_ffn_shared128",
            "rqm_ffn_beta005",
            "rqm_ffn_kappa05",
            "rqm_ffn_wide_lowrank",
            "rqm_ffn_narrow_highrank",
            "rkm_ffn",
            "rkm_ffn_more_regions",
            "rkm_ffn_fewer_regions",
            "rapm_ffn",
            "rapm_ffn_beta005",
            "rapm_ffn_kappa05",
            "rapm_ffn_per_channel",
            "rpf_ffn",
            "rpf_ffn_curv05",
            "rpb_ffn",
            "rpb_ffn_diff",
            "rpb_ffn_norm",
            "rwf_ffn",
            "rwf_ffn_gelu",
            "rmb_k2_ffn",
            "rmb_k3_ffn",
            "rma_ffn",
            "rma_ffn_curv",
            "rma_ffn_strong",
            "rma_raw_ffn_curv",
            "rma_gelu_curv_ffn",
            "rma_gelu_strong_ffn",
            "rma_relu_curv_ffn",
            "rma_identity_curv_ffn",
            "rma_shift_ffn",
            "rma_gelu_shift_ffn",
            "rma_identity_shift_ffn",
            "rma_silu_shift_strong_ffn",
            "rma_silu_momaffine_ffn",
            "rma_gelu_momaffine_ffn",
            "rma_identity_momaffine_ffn",
            "rma_silu_denwide_shift_ffn",
            "rma_silu_dennarrow_shift_ffn",
            "rma_skip_ffn",
            "rma_raw_skip_ffn",
            "rma_radial_ffn",
            "rma_center_ffn",
            "rma_center_strong_ffn",
            "rma_hermite_ffn",
            "rma_center_hermite_ffn",
            "rma_divnorm25_ffn",
            "rma_divnorm50_ffn",
            "rma_divnorm75_ffn",
            "rma_divnorm_strong_ffn",
            "rma_pair_ffn",
            "rma_pair_ffn_strong",
            "rda_silu_ffn",
            "rda_silu_shift_ffn",
            "rda_silu_momaffine_ffn",
            "rda_gelu_momaffine_ffn",
            "rda_identity_momaffine_ffn",
            "rlbx_k2_ffn",
            "rlbx_k2_shift_ffn",
            "rlbx_k2_strong_ffn",
            "rlbx_k2_identity_ffn",
            "rlb_shift_ffn",
            "rlb_strong_ffn",
            "rlb_wide_ffn",
            "rlb_identity_ffn",
            "rlb_fixed_strong_ffn",
            "rlb_fast_ffn",
            "rlb_fast_train_ffn",
            "rlb_fast_scaled_ffn",
            "rlb_fused_fast_ffn",
            "rlb_fused_fast_h2816_ffn",
            "rlb_fused_fast_h2640_ffn",
            "rlb_fused_fast_h2560_ffn",
            "rlb_fused_fixed_strong_ffn",
            "rlb_fused_fixed_strong_h2880_ffn",
            "rlb_fused_fixed_strong_h2816_ffn",
            "rlb_fused_fixed_strong_h2640_ffn",
            "rlb_fused_fixed_strong_h2560_ffn",
            "rlb_fused_boost_h2560_ffn",
            "rlb_fused_boost_h2400_ffn",
            "rlb_fused_quantile4_ffn",
            "rlb_fused_core4_ffn",
            "rlb_centered_strong_ffn",
            "rlb_centered_scaled_ffn",
            "rlb_fixed_centered_ffn",
            "rcq_ffn",
            "rcq_shift_ffn",
            "rcq_strong_ffn",
            "rcq_identity_ffn",
            "rgc_ffn",
            "rgc_shift_ffn",
            "rgc_strong_ffn",
            "rgc_moment_ffn",
            "rgc_identity_ffn",
            "rsm_ffn",
            "rsm_ffn_basis",
            "rsm_ffn_strong",
            "rhg_ffn",
            "rhg_ffn_balanced",
            "rhg_ffn_basisgate",
            "rhg_ffn_basisgate_resvalue_wide",
            "rhg_ffn_basisgate_wide",
            "rhg_ffn_gateact",
            "rhg_ffn_fullact",
            "rhg_ffn_highgate",
            "rhg_ffn_resboth",
            "rhg_ffn_resgate",
            "rhg_ffn_resvalue",
            "rhg_ffn_resvalue_dual",
            "rhg_ffn_resvalue_gated_dual",
            "rhg_ffn_resvalue_gated",
            "rhg_ffn_resvalue_gated_channel",
            "rhg_ffn_resvalue_gated_channel_strong",
            "rhg_ffn_resvalue_gated_crossmix64",
            "rhg_ffn_resvalue_gated_crossmix128",
            "rhg_ffn_resvalue_gated_beta075",
            "rhg_ffn_resvalue_gated_beta10",
            "rhg_ffn_resvalue_gated_beta10_depthup",
            "rhg_ffn_resvalue_gated_beta10_groupdepth",
            "rhg_ffn_resvalue_gated_beta10_groupscale",
            "rhg_ffn_resvalue_gated_beta10_groupscale_moment",
            "rhg_ffn_resvalue_gated_beta10_groupscale_safegate",
            "rhg_ffn_resvalue_gated_beta10_groupscale_safegate_low",
            "rhg_ffn_resvalue_gated_beta125",
            "rhg_ffn_resvalue_gated_beta20",
            "rhg_ffn_resvalue_gated_highgate",
            "rhg_ffn_resvalue_gated_norm",
            "rhg_ffn_resvalue_gated_valuewide",
            "rhg_ffn_resvalue_norm",
            "rhg_ffn_valueact",
            "rhg_ffn_valuewide",
            "rkdm_ffn",
            "rkdm_ffn_more_regions",
            "rkdm_ffn_highrank",
            "silu",
            "rational_a",
            "rational_grouped",
            "rational_up",
            "rational_up_grouped",
            "rational_both",
            "rational_both_grouped",
            "rational_product",
            "rational_product_grouped",
            "rational_swiglu_post",
            "rational_swiglu_post_grouped",
        ],
        default="silu",
    )
    parser.add_argument("--run-name", default="wikitext103")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-streaming", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataset-text-column", default="text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-skip-documents", type=int, default=0)
    parser.add_argument("--validation-skip-documents", type=int, default=0)
    parser.add_argument("--train-skip-tokens", type=int, default=0)
    parser.add_argument("--validation-skip-tokens", type=int, default=0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--hf-cache", default="experiments/cache/huggingface")
    parser.add_argument("--cache-dir", default="experiments/cache/tokens")
    parser.add_argument("--output-dir", default="experiments/runs/wikitext103")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-train-tokens", type=int, default=100_000_000)
    parser.add_argument("--max-val-tokens", type=int, default=2_000_000)
    parser.add_argument("--tokenize-batch-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--init-std", type=float, default=0.02)
    parser.add_argument("--rational-init", default="silu")
    parser.add_argument("--post-rational-init", default="identity")
    parser.add_argument("--rational-group-size", type=int, default=256)
    parser.add_argument("--rational-max-groups", type=int, default=32)
    parser.add_argument("--birational-alpha-init", type=float, default=1e-3)
    parser.add_argument("--birational-denominator-init", type=float, default=1e-3)
    parser.add_argument("--birational-eps", type=float, default=1e-6)
    parser.add_argument("--rational-basis-eps", type=float, default=1e-6)
    parser.add_argument("--optimizer", choices=ACTIVE_OPTIMIZERS, default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--factored-min-dim", type=int, default=128)
    parser.add_argument("--factored-clip-threshold", type=float, default=1.0)
    parser.add_argument("--ademamix-beta3", type=float, default=0.9999)
    parser.add_argument("--ademamix-alpha", type=float, default=5.0)
    parser.add_argument("--ademamix-beta3-warmup-steps", type=int, default=-1)
    parser.add_argument("--ademamix-alpha-warmup-steps", type=int, default=-1)
    parser.add_argument("--schedule-free-beta1", type=float, default=0.9)
    parser.add_argument("--schedule-free-warmup-steps", type=int, default=0)
    parser.add_argument("--came-beta3", type=float, default=0.999)
    parser.add_argument("--came-confidence-scale", type=float, default=1.0)
    parser.add_argument("--came-confidence-min", type=float, default=0.25)
    parser.add_argument("--came-confidence-max", type=float, default=4.0)
    parser.add_argument("--soap-precondition-frequency", type=int, default=50)
    parser.add_argument("--soap-large-side-identity-threshold", type=int, default=2048)
    parser.add_argument("--soap-one-sided", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rational-dense-depth-gain", type=float, default=0.15)
    parser.add_argument("--rational-dense-no-decay-lr-scale", type=float, default=0.75)
    parser.add_argument("--rational-dense-min-lr-scale", type=float, default=0.70)
    parser.add_argument("--rational-dense-max-lr-scale", type=float, default=1.20)
    parser.add_argument("--rational-switch-start", type=float, default=0.36)
    parser.add_argument("--rational-switch-end", type=float, default=0.58)
    parser.add_argument("--rational-switch-depth-shift", type=float, default=-0.16)
    parser.add_argument("--rational-switch-adam-lr-scale", type=float, default=1.0)
    parser.add_argument("--rational-switch-function-lr-scale", type=float, default=1.0)
    parser.add_argument("--rational-switch-select-strength", type=float, default=0.35)
    parser.add_argument("--rational-switch-select-start", type=float, default=0.20)
    parser.add_argument("--rational-switch-select-end", type=float, default=0.55)
    parser.add_argument("--rational-switch-select-activity-threshold", type=float, default=0.08)
    parser.add_argument("--rational-switch-select-activity-width", type=float, default=0.32)
    parser.add_argument("--rational-switch-select-pressure-weight", type=float, default=0.25)
    parser.add_argument("--sam-rho", type=float, default=0.0)
    parser.add_argument("--sam-start", type=float, default=0.05)
    parser.add_argument("--sam-end", type=float, default=1.0)
    parser.add_argument("--sam-warmup", type=float, default=0.05)
    parser.add_argument("--sam-adaptive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sam-depth-gain", type=float, default=0.0)
    parser.add_argument("--sam-rational-scale", type=float, default=0.35)
    parser.add_argument("--sam-no-decay-scale", type=float, default=0.35)
    parser.add_argument("--sam-adaptive-eps", type=float, default=1e-3)
    parser.add_argument("--sam-eps", type=float, default=1e-12)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adjust-lr-fn", choices=["original", "match_rms_adamw"], default="match_rms_adamw")
    parser.add_argument("--rational-coeff-num-lr-scale", type=float, default=1.0)
    parser.add_argument("--rational-coeff-den-lr-scale", type=float, default=1.125)
    parser.add_argument("--rational-coeff-atom-lr-scale", type=float, default=2.25)
    parser.add_argument("--rational-coeff-center-lr-scale", type=float, default=0.10)
    parser.add_argument("--rational-coeff-other-lr-scale", type=float, default=0.25)
    parser.add_argument("--rational-coeff-trust", type=float, default=0.01)
    parser.add_argument("--rational-coeff-probe-range", type=float, default=5.0)
    parser.add_argument("--rational-coeff-probe-points", type=int, default=257)
    parser.add_argument("--rational-coeff-curve-decay", type=float, default=0.95)
    parser.add_argument("--rational-coeff-update-gain", type=float, default=4.5)
    parser.add_argument("--rational-coeff-metric", choices=["diag", "gram"], default="diag")
    parser.add_argument("--rational-coeff-metric-damping", type=float, default=1e-3)
    parser.add_argument("--rational-coeff-eps", type=float, default=1e-8)
    parser.add_argument("--rlb-gauge-strength", type=float, default=0.50)
    parser.add_argument("--rlb-gauge-max-log-step", type=float, default=0.030)
    parser.add_argument("--rlb-gauge-start", type=float, default=0.03)
    parser.add_argument("--rlb-gauge-end", type=float, default=0.35)
    parser.add_argument("--rlb-gauge-depth-gain", type=float, default=0.15)
    parser.add_argument("--rlb-gauge-every", type=int, default=5)
    parser.add_argument("--rlb-gauge-eps", type=float, default=1e-8)
    parser.add_argument("--rlb-gauge-metric-every", type=int, default=10)
    parser.add_argument("--rlb-gauge-probe-range", type=float, default=5.0)
    parser.add_argument("--rlb-gauge-probe-points", type=int, default=129)
    parser.add_argument("--rlb-gauge-target-weight", type=float, default=1.0)
    parser.add_argument("--rational-onpolicy-stat-decay", type=float, default=0.95)
    parser.add_argument("--rational-onpolicy-pressure-weight", type=float, default=0.25)
    parser.add_argument("--rational-onpolicy-pressure-clip", type=float, default=1.25)
    parser.add_argument("--rational-onpolicy-rational-activity-weight", type=float, default=0.10)
    parser.add_argument("--rational-onpolicy-activity-gain-min", type=float, default=0.75)
    parser.add_argument("--rational-onpolicy-activity-gain-max", type=float, default=1.25)
    parser.add_argument("--rational-quotient-strength", type=float, default=1.0)
    parser.add_argument("--rational-jacobian-matrix-strength", type=float, default=0.5)
    parser.add_argument("--rational-jacobian-min-scale", type=float, default=0.5)
    parser.add_argument("--rational-jacobian-max-scale", type=float, default=2.0)
    parser.add_argument("--rational-jacobian-every", type=int, default=5)
    parser.add_argument("--rational-qjacobian-quotient-strength", type=float, default=1.0)
    parser.add_argument("--rational-qjacobian-quotient-start", type=float, default=0.02)
    parser.add_argument("--rational-qjacobian-quotient-end", type=float, default=0.30)
    parser.add_argument("--rational-qjacobian-quotient-depth-gain", type=float, default=0.10)
    parser.add_argument("--rational-adaptive-stat-every", type=int, default=4)
    parser.add_argument("--rational-adaptive-stat-samples", type=int, default=512)
    parser.add_argument("--rational-adaptive-coeff-strength", type=float, default=0.0)
    parser.add_argument("--rational-adaptive-coeff-start", type=float, default=0.02)
    parser.add_argument("--rational-adaptive-coeff-end", type=float, default=0.65)
    parser.add_argument("--rational-adaptive-coeff-late-decay", type=float, default=0.35)
    parser.add_argument("--rational-adaptive-coeff-metric-damping", type=float, default=0.03)
    parser.add_argument("--rational-adaptive-coeff-norm-clip", type=float, default=3.0)
    parser.add_argument("--rational-adaptive-coeff-max-blend", type=float, default=0.85)
    parser.add_argument("--rational-adaptive-coeff-depth-gain", type=float, default=0.20)
    parser.add_argument("--rational-adaptive-matrix-strength", type=float, default=0.55)
    parser.add_argument("--rational-adaptive-min-scale", type=float, default=0.5)
    parser.add_argument("--rational-adaptive-max-scale", type=float, default=2.0)
    parser.add_argument("--rational-adaptive-every", type=int, default=5)
    parser.add_argument("--rational-adaptive-matrix-time-gain", type=float, default=0.15)
    parser.add_argument("--rational-adaptive-matrix-depth-gain", type=float, default=0.10)
    parser.add_argument("--rational-adaptive-quotient-strength", type=float, default=0.0)
    parser.add_argument("--rational-trust-coeff-strength", type=float, default=1.0)
    parser.add_argument("--rational-trust-radius", type=float, default=0.018)
    parser.add_argument("--rational-trust-min-scale", type=float, default=0.05)
    parser.add_argument("--rational-trust-max-scale", type=float, default=1.15)
    parser.add_argument("--rational-trust-activity-target", type=float, default=0.85)
    parser.add_argument("--rational-trust-activity-width", type=float, default=0.55)
    parser.add_argument("--rational-trust-pressure-weight", type=float, default=0.25)
    parser.add_argument("--rational-trust-agreement-decay", type=float, default=0.90)
    parser.add_argument("--rational-trust-agreement-floor", type=float, default=0.15)
    parser.add_argument("--rational-trust-metric-blend", type=float, default=0.45)
    parser.add_argument("--rational-trust-denominator-risk", type=float, default=1.75)
    parser.add_argument("--rational-trust-atom-risk", type=float, default=1.00)
    parser.add_argument("--rational-trust-numerator-risk", type=float, default=1.00)
    parser.add_argument("--rational-trust-depth-gain", type=float, default=0.10)
    parser.add_argument("--rational-matrix-policy-backbone-optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--rational-matrix-policy-backbone-beta2", type=float, default=0.999)
    parser.add_argument("--rational-matrix-policy-beta2", type=float, default=0.999)
    parser.add_argument("--rational-matrix-policy-muon-strength", type=float, default=0.75)
    parser.add_argument("--rational-matrix-policy-muon-lr-scale", type=float, default=1.0)
    parser.add_argument("--rational-matrix-policy-adam-lr-scale", type=float, default=3.0)
    parser.add_argument("--rational-matrix-policy-adam-lr-scale-final", type=float, default=None)
    parser.add_argument("--rational-matrix-policy-adam-decay-start", type=float, default=1.1)
    parser.add_argument("--rational-matrix-policy-adam-decay-end", type=float, default=1.1)
    parser.add_argument("--rational-matrix-policy-adam-decay-depth-shift", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-adam-beta2-final", type=float, default=None)
    parser.add_argument("--rational-matrix-policy-adam-beta2-input-final", type=float, default=None)
    parser.add_argument("--rational-matrix-policy-adam-beta2-output-final", type=float, default=None)
    parser.add_argument("--rational-matrix-policy-adam-beta2-decay-start", type=float, default=1.1)
    parser.add_argument("--rational-matrix-policy-adam-beta2-decay-end", type=float, default=1.1)
    parser.add_argument("--rational-matrix-policy-adam-beta2-decay-depth-shift", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-adam-role-strength", type=float, default=1.20)
    parser.add_argument("--rational-matrix-policy-adam-stat-strength", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-adam-pressure-balance", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-adam-stat-start", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-adam-stat-end", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-adam-min-lr-scale", type=float, default=0.40)
    parser.add_argument("--rational-matrix-policy-adam-max-lr-scale", type=float, default=4.0)
    parser.add_argument("--rational-matrix-policy-adam-reset-on-switch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rational-matrix-policy-weight-decay-scale", type=float, default=1.0)
    parser.add_argument("--rational-matrix-policy-function-coeff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rational-matrix-policy-start", type=float, default=0.02)
    parser.add_argument("--rational-matrix-policy-end", type=float, default=0.12)
    parser.add_argument("--rational-matrix-policy-decay-start", type=float, default=0.20)
    parser.add_argument("--rational-matrix-policy-decay-end", type=float, default=0.36)
    parser.add_argument("--rational-matrix-policy-muon-decay-depth-shift", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-muon-input-decay-shift", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-muon-output-decay-shift", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-muon-reset-adam-state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rational-matrix-policy-final-muon", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-min-muon", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-max-muon", type=float, default=0.75)
    parser.add_argument("--rational-matrix-policy-input-depth-gain", type=float, default=-0.50)
    parser.add_argument("--rational-matrix-policy-output-depth-gain", type=float, default=1.00)
    parser.add_argument("--rational-matrix-policy-pressure-weight", type=float, default=0.30)
    parser.add_argument("--rational-matrix-policy-activity-weight", type=float, default=0.65)
    parser.add_argument("--rational-matrix-policy-activity-target", type=float, default=0.05)
    parser.add_argument("--rational-matrix-policy-activity-width", type=float, default=0.45)
    parser.add_argument("--rational-matrix-policy-pressure-clip", type=float, default=1.50)
    parser.add_argument("--rational-matrix-policy-group-gain-strength", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-group-pressure-strength", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-group-activity-damping", type=float, default=0.0)
    parser.add_argument("--rational-matrix-policy-group-activity-target", type=float, default=0.05)
    parser.add_argument("--rational-matrix-policy-group-activity-width", type=float, default=0.45)
    parser.add_argument("--rational-matrix-policy-group-start", type=float, default=0.02)
    parser.add_argument("--rational-matrix-policy-group-end", type=float, default=0.35)
    parser.add_argument("--rational-matrix-policy-group-min-scale", type=float, default=0.65)
    parser.add_argument("--rational-matrix-policy-group-max-scale", type=float, default=1.55)
    parser.add_argument("--rational-transport-quotient-strength", type=float, default=0.0)
    parser.add_argument("--rational-transport-strength", type=float, default=0.0)
    parser.add_argument("--rational-transport-final-strength", type=float, default=None)
    parser.add_argument("--rational-transport-start", type=float, default=0.04)
    parser.add_argument("--rational-transport-end", type=float, default=0.70)
    parser.add_argument("--rational-transport-decay-start", type=float, default=1.1)
    parser.add_argument("--rational-transport-decay-end", type=float, default=1.1)
    parser.add_argument("--rational-transport-every", type=int, default=5)
    parser.add_argument("--rational-transport-max-log-step", type=float, default=0.025)
    parser.add_argument("--rational-transport-derivative-weight", type=float, default=0.50)
    parser.add_argument("--rational-transport-headroom", type=float, default=0.92)
    parser.add_argument("--rational-transport-depth-gain", type=float, default=0.30)
    parser.add_argument("--rational-transport-derivative-depth-gain", type=float, default=0.35)
    parser.add_argument("--rational-transport-matrix-strength", type=float, default=0.0)
    parser.add_argument("--rational-transport-matrix-depth-gain", type=float, default=0.0)
    parser.add_argument("--rational-transport-matrix-time-gain", type=float, default=0.0)
    parser.add_argument("--rational-transport-stat-every", type=int, default=8)
    parser.add_argument("--rational-transport-matrix-input-depth-gain", type=float, default=0.0)
    parser.add_argument("--rational-transport-matrix-output-depth-gain", type=float, default=0.0)
    parser.add_argument("--rational-transport-live-matrix-stats", action="store_true")
    parser.add_argument("--rational-transport-coeff-update-gain", type=float, default=4.5)
    parser.add_argument("--rational-transport-coeff-update-gain-final", type=float, default=4.5)
    parser.add_argument("--rational-transport-coeff-decay-start", type=float, default=1.1)
    parser.add_argument("--rational-transport-coeff-decay-end", type=float, default=1.1)
    parser.add_argument("--rational-transport-coeff-depth-gain", type=float, default=0.0)
    parser.add_argument("--rational-transport-coeff-switch-depth-shift", type=float, default=0.0)
    parser.add_argument("--rational-transport-coeff-reset-on-switch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rational-transport-coeff-select-strength", type=float, default=0.0)
    parser.add_argument("--rational-transport-coeff-select-start", type=float, default=0.25)
    parser.add_argument("--rational-transport-coeff-select-end", type=float, default=0.55)
    parser.add_argument("--rational-transport-coeff-select-activity-threshold", type=float, default=0.10)
    parser.add_argument("--rational-transport-coeff-select-activity-width", type=float, default=0.40)
    parser.add_argument("--rational-transport-coeff-select-pressure-weight", type=float, default=0.25)
    parser.add_argument("--rational-transport-coeff-atom-decay", type=float, default=0.0)
    parser.add_argument("--rational-transport-coeff-den-decay", type=float, default=0.0)
    parser.add_argument("--rational-transport-coeff-atom-lr-scale", type=float, default=2.25)
    parser.add_argument("--rational-transport-coeff-atom-lr-scale-final", type=float, default=2.25)
    parser.add_argument("--rational-transport-coeff-den-lr-scale", type=float, default=1.125)
    parser.add_argument("--rational-transport-coeff-den-lr-scale-final", type=float, default=1.125)
    parser.add_argument("--rational-transport-coeff-trust", type=float, default=0.01)
    parser.add_argument("--rational-transport-coeff-trust-final", type=float, default=0.01)
    parser.add_argument("--rational-transport-pressure-strength", type=float, default=0.0)
    parser.add_argument("--rational-transport-pressure-depth-gain", type=float, default=0.25)
    parser.add_argument("--rational-transport-pressure-min-scale", type=float, default=0.70)
    parser.add_argument("--rational-transport-pressure-max-scale", type=float, default=1.40)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--early-stop-min-step", type=int, default=0)
    parser.add_argument("--early-stop-max-val-loss", type=float, default=0.0)
    parser.add_argument("--early-stop-loss-increase", type=float, default=0.0)
    parser.add_argument("--probe-batch-size", type=int, default=2)
    parser.add_argument("--telemetry-rlb-stat-every", type=int, default=4)
    parser.add_argument("--telemetry-rlb-stat-samples", type=int, default=512)
    parser.add_argument("--telemetry-denominator-probe-points", type=int, default=129)
    parser.add_argument("--matrix-spectrum-interval", type=int, default=500)
    parser.add_argument("--matrix-spectrum-max-dim", type=int, default=512)
    parser.add_argument("--rlb-init-gauge-log-scale", type=float, default=0.0)
    parser.add_argument("--rlb-init-gauge-seed", type=int, default=424242)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    return parser.parse_args()


def validate_optimizer_protocol(args):
    if args.optimizer not in ACTIVE_OPTIMIZERS:
        allowed = ", ".join(ACTIVE_OPTIMIZERS)
        raise ValueError(f"Accepted optimizer choices: {allowed}")
    if args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS and args.activation not in RLB_ACTIVATIONS:
        allowed = ", ".join(sorted(RLB_ACTIVATIONS))
        raise ValueError(f"Accepted RLB activations for {args.optimizer}: {allowed}")


def main():
    args = parse_args()
    validate_optimizer_protocol(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.prepare_only:
        train_tokens = load_or_tokenize(args, "train", args.max_train_tokens)
        val_tokens = load_or_tokenize(args, "validation", args.max_val_tokens)
        print(
            json.dumps({"event": "prepared", "train_tokens": train_tokens.numel(), "val_tokens": val_tokens.numel()}),
            flush=True,
        )
        os._exit(0)

    is_distributed, rank, local_rank, world_size, device = setup_distributed()
    if device.type != "cuda":
        raise RuntimeError("this benchmark is intended to run on CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_tokens = load_or_tokenize(args, "train", args.max_train_tokens)
    val_tokens = load_or_tokenize(args, "validation", args.max_val_tokens)
    train_tokens = train_tokens.pin_memory()
    val_tokens = val_tokens.pin_memory()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.hf_cache)
    vocab_size = len(tokenizer)
    global_tokens = args.batch_size * args.grad_accum * world_size * args.seq_len
    if args.steps <= 0:
        args.steps = max(1, train_tokens.numel() // global_tokens)
    if args.warmup_steps >= args.steps:
        args.warmup_steps = max(1, args.steps // 10)

    model = CausalTransformer(args, vocab_size).to(device)
    rlb_init_gauge_groups = apply_rlb_positive_gauge(
        model, args.rlb_init_gauge_log_scale, args.rlb_init_gauge_seed
    )
    enable_rlb_training_telemetry(model, args)
    param_count = sum(param.numel() for param in model.parameters())
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    rational_optimizer_parameter_count = count_rational_optimizer_parameters(model)
    optimizer = configure_optimizer(model, args)
    offsets = torch.arange(args.seq_len + 1)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(args.seed + 997 * rank)
    out_path = Path(args.output_dir) / args.run_name / f"{args.activation}.jsonl"
    probe_state = prepare_probe_batch(val_tokens, args, offsets, rank, device, out_path)
    rb_settings = rational_basis_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rq_settings = rqm_settings(
        args.activation,
        args.d_model,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rk_settings = rkm_settings(
        args.activation,
        args.d_model,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rp_settings = rapm_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rpf_cfg = rpf_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rpb_cfg = rpb_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rwf_cfg = rwf_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rmb_cfg = rmb_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rma_cfg = rma_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rda_cfg = rda_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rlbx_cfg = rlbx_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rlb_cfg = rlb_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rcq_cfg = rcq_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rgc_cfg = rgc_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rsm_cfg = rsm_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rhg_cfg = rhg_settings(
        args.activation,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )
    rkdm_cfg = rkdm_settings(
        args.activation,
        args.d_model,
        args.ffn_dim,
        args.rational_group_size,
        args.rational_max_groups,
    )

    config_record = {
        "event": "config",
        "activation": args.activation,
        "batch_size_per_gpu": args.batch_size,
        "birational_alpha_init": args.birational_alpha_init,
        "birational_denominator_init": args.birational_denominator_init,
        "birational_eps": args.birational_eps,
        "d_model": args.d_model,
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_streaming": args.dataset_streaming,
        "dataset_text_column": args.dataset_text_column,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "train_skip_documents": args.train_skip_documents,
        "validation_skip_documents": args.validation_skip_documents,
        "train_skip_tokens": args.train_skip_tokens,
        "validation_skip_tokens": args.validation_skip_tokens,
        "ffn_dim": args.ffn_dim,
        "global_tokens_per_step": global_tokens,
        "grad_accum": args.grad_accum,
        "log_interval": args.log_interval,
        "eval_interval": args.eval_interval,
        "eval_batches": args.eval_batches,
        "probe_batch_size": args.probe_batch_size,
        "telemetry_rlb_stat_every": args.telemetry_rlb_stat_every,
        "telemetry_rlb_stat_samples": args.telemetry_rlb_stat_samples,
        "telemetry_denominator_probe_points": args.telemetry_denominator_probe_points,
        "matrix_spectrum_interval": args.matrix_spectrum_interval,
        "matrix_spectrum_max_dim": args.matrix_spectrum_max_dim,
        "rlb_init_gauge_log_scale": args.rlb_init_gauge_log_scale,
        "rlb_init_gauge_seed": args.rlb_init_gauge_seed,
        "rlb_init_gauge_groups": rlb_init_gauge_groups,
        "optimizer": args.optimizer,
        "optimizer_lr": args.lr,
        "optimizer_min_lr": args.min_lr,
        "optimizer_weight_decay": args.weight_decay,
        "optimizer_beta1": args.beta1,
        "optimizer_beta2": args.beta2,
        "early_stop_min_step": args.early_stop_min_step,
        "early_stop_max_val_loss": args.early_stop_max_val_loss,
        "early_stop_loss_increase": args.early_stop_loss_increase,
        "factored_min_dim": args.factored_min_dim if args.optimizer in {"factored_adamw", "adafactor_came"} else None,
        "factored_clip_threshold": args.factored_clip_threshold if args.optimizer in {"factored_adamw", "adafactor_came"} else None,
        "ademamix_beta3": args.ademamix_beta3 if args.optimizer == "ademamix" else None,
        "ademamix_alpha": args.ademamix_alpha if args.optimizer == "ademamix" else None,
        "ademamix_beta3_warmup_steps": resolve_ademamix_warmup(args.ademamix_beta3_warmup_steps, args.steps) if args.optimizer == "ademamix" else None,
        "ademamix_alpha_warmup_steps": resolve_ademamix_warmup(args.ademamix_alpha_warmup_steps, args.steps) if args.optimizer == "ademamix" else None,
        "schedule_free_beta1": args.schedule_free_beta1 if args.optimizer == "schedule_free_adamw" else None,
        "schedule_free_warmup_steps": args.schedule_free_warmup_steps if args.optimizer == "schedule_free_adamw" else None,
        "came_beta3": args.came_beta3 if args.optimizer == "adafactor_came" else None,
        "came_confidence_scale": args.came_confidence_scale if args.optimizer == "adafactor_came" else None,
        "came_confidence_min": args.came_confidence_min if args.optimizer == "adafactor_came" else None,
        "came_confidence_max": args.came_confidence_max if args.optimizer == "adafactor_came" else None,
        "soap_precondition_frequency": args.soap_precondition_frequency if args.optimizer == "soap_adamw" else None,
        "soap_large_side_identity_threshold": args.soap_large_side_identity_threshold if args.optimizer == "soap_adamw" else None,
        "soap_one_sided": args.soap_one_sided if args.optimizer == "soap_adamw" else None,
        "rational_dense_depth_gain": args.rational_dense_depth_gain if "layerwise" in args.optimizer else None,
        "rational_dense_no_decay_lr_scale": args.rational_dense_no_decay_lr_scale if "layerwise" in args.optimizer else None,
        "rational_dense_min_lr_scale": args.rational_dense_min_lr_scale if "layerwise" in args.optimizer else None,
        "rational_dense_max_lr_scale": args.rational_dense_max_lr_scale if "layerwise" in args.optimizer else None,
        "rational_switch_start": args.rational_switch_start if "switch" in args.optimizer else None,
        "rational_switch_end": args.rational_switch_end if "switch" in args.optimizer else None,
        "rational_switch_depth_shift": args.rational_switch_depth_shift if "switch" in args.optimizer else None,
        "rational_switch_adam_lr_scale": args.rational_switch_adam_lr_scale if "switch" in args.optimizer else None,
        "rational_switch_function_lr_scale": args.rational_switch_function_lr_scale if "switch" in args.optimizer else None,
        "rational_switch_select_strength": args.rational_switch_select_strength if "switch" in args.optimizer else None,
        "rational_switch_select_start": args.rational_switch_select_start if "switch" in args.optimizer else None,
        "rational_switch_select_end": args.rational_switch_select_end if "switch" in args.optimizer else None,
        "rational_switch_select_activity_threshold": args.rational_switch_select_activity_threshold if "switch" in args.optimizer else None,
        "rational_switch_select_activity_width": args.rational_switch_select_activity_width if "switch" in args.optimizer else None,
        "rational_switch_select_pressure_weight": args.rational_switch_select_pressure_weight if "switch" in args.optimizer else None,
        "sam_rho": args.sam_rho,
        "sam_start": args.sam_start,
        "sam_end": args.sam_end,
        "sam_warmup": args.sam_warmup,
        "sam_adaptive": args.sam_adaptive,
        "sam_depth_gain": args.sam_depth_gain,
        "sam_rational_scale": args.sam_rational_scale,
        "sam_no_decay_scale": args.sam_no_decay_scale,
        "rational_coeff_parameters": rational_optimizer_parameter_count if args.optimizer.startswith("rational_") else None,
        "rlb_gauge_parameters": rational_optimizer_parameter_count if args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS else None,
        "rlb_gauge_strength": args.rlb_gauge_strength if args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS else None,
        "rlb_gauge_max_log_step": args.rlb_gauge_max_log_step if args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS else None,
        "rlb_gauge_start": args.rlb_gauge_start if args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS else None,
        "rlb_gauge_end": args.rlb_gauge_end if args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS else None,
        "rlb_gauge_depth_gain": args.rlb_gauge_depth_gain if args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS else None,
        "rlb_gauge_every": args.rlb_gauge_every if args.optimizer in RATIONAL_SPECIFIC_OPTIMIZERS else None,
        "rational_quotient_strength": args.rational_quotient_strength if args.optimizer == "rational_quotient_onpolicy" else None,
        "rational_jacobian_matrix_strength": args.rational_jacobian_matrix_strength if args.optimizer in {"rational_jacobian_onpolicy", "rational_jacobian_factored_onpolicy", "rational_layerwise_switch_onpolicy", "rational_layerwise_factored_switch_onpolicy", "rational_quotient_jacobian_onpolicy"} else None,
        "rational_jacobian_min_scale": args.rational_jacobian_min_scale if args.optimizer in {"rational_jacobian_onpolicy", "rational_jacobian_factored_onpolicy", "rational_layerwise_switch_onpolicy", "rational_layerwise_factored_switch_onpolicy", "rational_quotient_jacobian_onpolicy"} else None,
        "rational_jacobian_max_scale": args.rational_jacobian_max_scale if args.optimizer in {"rational_jacobian_onpolicy", "rational_jacobian_factored_onpolicy", "rational_layerwise_switch_onpolicy", "rational_layerwise_factored_switch_onpolicy", "rational_quotient_jacobian_onpolicy"} else None,
        "rational_jacobian_every": args.rational_jacobian_every if args.optimizer in {"rational_jacobian_onpolicy", "rational_jacobian_factored_onpolicy", "rational_layerwise_switch_onpolicy", "rational_layerwise_factored_switch_onpolicy", "rational_quotient_jacobian_onpolicy"} else None,
        "rational_qjacobian_quotient_strength": args.rational_qjacobian_quotient_strength if args.optimizer == "rational_quotient_jacobian_onpolicy" else None,
        "rational_qjacobian_quotient_start": args.rational_qjacobian_quotient_start if args.optimizer == "rational_quotient_jacobian_onpolicy" else None,
        "rational_qjacobian_quotient_end": args.rational_qjacobian_quotient_end if args.optimizer == "rational_quotient_jacobian_onpolicy" else None,
        "rational_qjacobian_quotient_depth_gain": args.rational_qjacobian_quotient_depth_gain if args.optimizer == "rational_quotient_jacobian_onpolicy" else None,
        "rational_adaptive_stat_every": args.rational_adaptive_stat_every if args.optimizer in {"rational_adaptive_metric_onpolicy", "rational_transport_onpolicy", "rational_functional_trust_onpolicy"} else None,
        "rational_adaptive_matrix_strength": args.rational_adaptive_matrix_strength if args.optimizer in {"rational_adaptive_metric_onpolicy", "rational_transport_onpolicy", "rational_functional_trust_onpolicy"} else None,
        "rational_adaptive_matrix_time_gain": args.rational_adaptive_matrix_time_gain if args.optimizer in {"rational_adaptive_metric_onpolicy", "rational_transport_onpolicy", "rational_functional_trust_onpolicy"} else None,
        "rational_adaptive_matrix_depth_gain": args.rational_adaptive_matrix_depth_gain if args.optimizer in {"rational_adaptive_metric_onpolicy", "rational_transport_onpolicy", "rational_functional_trust_onpolicy"} else None,
        "rational_adaptive_quotient_strength": args.rational_adaptive_quotient_strength if args.optimizer in {"rational_adaptive_metric_onpolicy", "rational_functional_trust_onpolicy"} else None,
        "rational_trust_coeff_strength": args.rational_trust_coeff_strength if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_radius": args.rational_trust_radius if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_min_scale": args.rational_trust_min_scale if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_max_scale": args.rational_trust_max_scale if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_activity_target": args.rational_trust_activity_target if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_activity_width": args.rational_trust_activity_width if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_pressure_weight": args.rational_trust_pressure_weight if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_agreement_decay": args.rational_trust_agreement_decay if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_metric_blend": args.rational_trust_metric_blend if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_denominator_risk": args.rational_trust_denominator_risk if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_atom_risk": args.rational_trust_atom_risk if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_numerator_risk": args.rational_trust_numerator_risk if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_trust_depth_gain": args.rational_trust_depth_gain if args.optimizer == "rational_functional_trust_onpolicy" else None,
        "rational_matrix_policy_backbone_optimizer": args.rational_matrix_policy_backbone_optimizer if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_backbone_beta2": args.rational_matrix_policy_backbone_beta2 if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_beta2": args.rational_matrix_policy_beta2 if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_muon_strength": args.rational_matrix_policy_muon_strength if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_muon_lr_scale": args.rational_matrix_policy_muon_lr_scale if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_lr_scale": args.rational_matrix_policy_adam_lr_scale if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_lr_scale_final": args.rational_matrix_policy_adam_lr_scale_final if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_decay_start": args.rational_matrix_policy_adam_decay_start if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_decay_end": args.rational_matrix_policy_adam_decay_end if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_decay_depth_shift": args.rational_matrix_policy_adam_decay_depth_shift if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_beta2_final": args.rational_matrix_policy_adam_beta2_final if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_beta2_input_final": args.rational_matrix_policy_adam_beta2_input_final if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_beta2_output_final": args.rational_matrix_policy_adam_beta2_output_final if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_beta2_decay_start": args.rational_matrix_policy_adam_beta2_decay_start if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_beta2_decay_end": args.rational_matrix_policy_adam_beta2_decay_end if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_beta2_decay_depth_shift": args.rational_matrix_policy_adam_beta2_decay_depth_shift if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_role_strength": args.rational_matrix_policy_adam_role_strength if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_stat_strength": args.rational_matrix_policy_adam_stat_strength if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_pressure_balance": args.rational_matrix_policy_adam_pressure_balance if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_stat_start": args.rational_matrix_policy_adam_stat_start if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_stat_end": args.rational_matrix_policy_adam_stat_end if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_min_lr_scale": args.rational_matrix_policy_adam_min_lr_scale if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_max_lr_scale": args.rational_matrix_policy_adam_max_lr_scale if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_adam_reset_on_switch": args.rational_matrix_policy_adam_reset_on_switch if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_weight_decay_scale": args.rational_matrix_policy_weight_decay_scale if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_function_coeff": args.rational_matrix_policy_function_coeff if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_start": args.rational_matrix_policy_start if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_end": args.rational_matrix_policy_end if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_decay_start": args.rational_matrix_policy_decay_start if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_decay_end": args.rational_matrix_policy_decay_end if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_muon_decay_depth_shift": args.rational_matrix_policy_muon_decay_depth_shift if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_muon_input_decay_shift": args.rational_matrix_policy_muon_input_decay_shift if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_muon_output_decay_shift": args.rational_matrix_policy_muon_output_decay_shift if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_muon_reset_adam_state": args.rational_matrix_policy_muon_reset_adam_state if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_final_muon": args.rational_matrix_policy_final_muon if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_min_muon": args.rational_matrix_policy_min_muon if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_max_muon": args.rational_matrix_policy_max_muon if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_input_depth_gain": args.rational_matrix_policy_input_depth_gain if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_output_depth_gain": args.rational_matrix_policy_output_depth_gain if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_pressure_weight": args.rational_matrix_policy_pressure_weight if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_activity_weight": args.rational_matrix_policy_activity_weight if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_activity_target": args.rational_matrix_policy_activity_target if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_activity_width": args.rational_matrix_policy_activity_width if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_pressure_clip": args.rational_matrix_policy_pressure_clip if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_group_gain_strength": args.rational_matrix_policy_group_gain_strength if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_group_pressure_strength": args.rational_matrix_policy_group_pressure_strength if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_group_activity_damping": args.rational_matrix_policy_group_activity_damping if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_group_activity_target": args.rational_matrix_policy_group_activity_target if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_group_activity_width": args.rational_matrix_policy_group_activity_width if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_group_start": args.rational_matrix_policy_group_start if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_group_end": args.rational_matrix_policy_group_end if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_group_min_scale": args.rational_matrix_policy_group_min_scale if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_matrix_policy_group_max_scale": args.rational_matrix_policy_group_max_scale if args.optimizer == "rational_matrix_policy_onpolicy" else None,
        "rational_transport_quotient_strength": args.rational_transport_quotient_strength if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_strength": args.rational_transport_strength if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_final_strength": args.rational_transport_final_strength if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_start": args.rational_transport_start if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_end": args.rational_transport_end if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_decay_start": args.rational_transport_decay_start if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_decay_end": args.rational_transport_decay_end if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_every": args.rational_transport_every if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_max_log_step": args.rational_transport_max_log_step if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_derivative_weight": args.rational_transport_derivative_weight if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_headroom": args.rational_transport_headroom if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_depth_gain": args.rational_transport_depth_gain if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_derivative_depth_gain": args.rational_transport_derivative_depth_gain if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_matrix_strength": args.rational_transport_matrix_strength if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_matrix_depth_gain": args.rational_transport_matrix_depth_gain if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_matrix_time_gain": args.rational_transport_matrix_time_gain if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_stat_every": args.rational_transport_stat_every if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_matrix_input_depth_gain": args.rational_transport_matrix_input_depth_gain if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_matrix_output_depth_gain": args.rational_transport_matrix_output_depth_gain if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_live_matrix_stats": args.rational_transport_live_matrix_stats if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_update_gain": args.rational_transport_coeff_update_gain if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_update_gain_final": args.rational_transport_coeff_update_gain_final if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_decay_start": args.rational_transport_coeff_decay_start if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_decay_end": args.rational_transport_coeff_decay_end if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_depth_gain": args.rational_transport_coeff_depth_gain if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_switch_depth_shift": args.rational_transport_coeff_switch_depth_shift if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_reset_on_switch": args.rational_transport_coeff_reset_on_switch if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_select_strength": args.rational_transport_coeff_select_strength if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_select_start": args.rational_transport_coeff_select_start if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_select_end": args.rational_transport_coeff_select_end if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_select_activity_threshold": args.rational_transport_coeff_select_activity_threshold if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_select_activity_width": args.rational_transport_coeff_select_activity_width if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_select_pressure_weight": args.rational_transport_coeff_select_pressure_weight if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_atom_decay": args.rational_transport_coeff_atom_decay if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_den_decay": args.rational_transport_coeff_den_decay if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_atom_lr_scale": args.rational_transport_coeff_atom_lr_scale if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_atom_lr_scale_final": args.rational_transport_coeff_atom_lr_scale_final if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_den_lr_scale": args.rational_transport_coeff_den_lr_scale if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_den_lr_scale_final": args.rational_transport_coeff_den_lr_scale_final if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_trust": args.rational_transport_coeff_trust if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_coeff_trust_final": args.rational_transport_coeff_trust_final if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_pressure_strength": args.rational_transport_pressure_strength if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_pressure_depth_gain": args.rational_transport_pressure_depth_gain if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_pressure_min_scale": args.rational_transport_pressure_min_scale if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "rational_transport_pressure_max_scale": args.rational_transport_pressure_max_scale if args.optimizer in {"rational_transport_onpolicy", "rational_matrix_policy_onpolicy"} else None,
        "muon_adjust_lr_fn": args.muon_adjust_lr_fn if args.optimizer in {"muon", "rational_matrix_policy_onpolicy"} else None,
        "muon_momentum": args.muon_momentum if args.optimizer in {"muon", "rational_matrix_policy_onpolicy"} else None,
        "muon_ns_steps": args.muon_ns_steps if args.optimizer in {"muon", "rational_matrix_policy_onpolicy"} else None,
        "heads": args.heads,
        "layers": args.layers,
        "params": param_count,
        "rational_group_size": args.rational_group_size,
        "rational_init": args.rational_init,
        "rational_basis_count": None if rb_settings is None else rb_settings["basis_count"],
        "rational_basis_eps": args.rational_basis_eps,
        "rational_basis_groups": None if rb_settings is None else rb_settings["groups"],
        "rational_basis_hidden_dim": None if rb_settings is None else rb_settings["hidden_dim"],
        "rqm_beta": None if rq_settings is None else rq_settings["beta"],
        "rqm_coeff_groups": None if rq_settings is None else rq_settings["coeff_groups"],
        "rqm_hidden_dim": None if rq_settings is None else rq_settings["hidden_dim"],
        "rqm_kappa": None if rq_settings is None else rq_settings["kappa"],
        "rqm_latent_groups": None if rq_settings is None else rq_settings["latent_groups"],
        "rqm_rank": None if rq_settings is None else rq_settings["rank"],
        "rkm_expert_rank": None if rk_settings is None else rk_settings["expert_rank"],
        "rkm_experts": None if rk_settings is None else rk_settings["experts"],
        "rkm_hidden_dim": None if rk_settings is None else rk_settings["hidden_dim"],
        "rkm_latent_groups": None if rk_settings is None else rk_settings["latent_groups"],
        "rkm_query_rank": None if rk_settings is None else rk_settings["query_rank"],
        "rapm_beta": None if rp_settings is None else rp_settings["beta"],
        "rapm_groups": None if rp_settings is None else rp_settings["groups"],
        "rapm_hidden_dim": None if rp_settings is None else rp_settings["hidden_dim"],
        "rapm_kappa": None if rp_settings is None else rp_settings["kappa"],
        "rpf_curvature": None if rpf_cfg is None else rpf_cfg["curvature"],
        "rpf_groups": None if rpf_cfg is None else rpf_cfg["groups"],
        "rpf_hidden_dim": None if rpf_cfg is None else rpf_cfg["hidden_dim"],
        "rpb_basis_count": None if rpb_cfg is None else rpb_cfg["basis_count"],
        "rpb_groups": None if rpb_cfg is None else rpb_cfg["groups"],
        "rpb_hidden_dim": None if rpb_cfg is None else rpb_cfg["hidden_dim"],
        "rpb_mode": None if rpb_cfg is None else rpb_cfg["mode"],
        "rpb_normalize": None if rpb_cfg is None else rpb_cfg["normalize"],
        "rwf_groups": None if rwf_cfg is None else rwf_cfg["groups"],
        "rwf_hidden_dim": None if rwf_cfg is None else rwf_cfg["hidden_dim"],
        "rwf_init": None if rwf_cfg is None else rwf_cfg["init"],
        "rmb_basis_count": None if rmb_cfg is None else rmb_cfg["basis_count"],
        "rmb_groups": None if rmb_cfg is None else rmb_cfg["groups"],
        "rmb_hidden_dim": None if rmb_cfg is None else rmb_cfg["hidden_dim"],
        "rmb_mix_init": None if rmb_cfg is None else rmb_cfg["mix_init"],
        "rma_basis_init": None if rma_cfg is None else rma_cfg["basis_init"],
        "rma_coeff_limit": None if rma_cfg is None else rma_cfg["coeff_limit"],
        "rma_groups": None if rma_cfg is None else rma_cfg["groups"],
        "rma_hidden_dim": None if rma_cfg is None else rma_cfg["hidden_dim"],
        "rma_pair_beta": None if rma_cfg is None else rma_cfg["pair_beta"],
        "rma_pair_init": None if rma_cfg is None else rma_cfg["pair_init"],
        "rma_normalize": None if rma_cfg is None else rma_cfg["normalize"],
        "rma_linear_skip_init": None if rma_cfg is None else rma_cfg["linear_skip_init"],
        "rma_radial_init": None if rma_cfg is None else rma_cfg["radial_init"],
        "rma_basis_center": None if rma_cfg is None else rma_cfg["basis_center"],
        "rma_basis_mode": None if rma_cfg is None else rma_cfg["basis_mode"],
        "rma_output_norm_init": None if rma_cfg is None else rma_cfg["output_norm_init"],
        "rma_base_init": None if rma_cfg is None else rma_cfg["base_init"],
        "rma_input_affine": None if rma_cfg is None else rma_cfg["input_affine"],
        "rma_moment_affine_init": None if rma_cfg is None else rma_cfg["moment_affine_init"],
        "rma_basis_den_scale_init": None if rma_cfg is None else rma_cfg["basis_den_scale_init"],
        "rda_base_init": None if rda_cfg is None else rda_cfg["base_init"],
        "rda_coeff_limit": None if rda_cfg is None else rda_cfg["coeff_limit"],
        "rda_denominator_limit": None if rda_cfg is None else rda_cfg["denominator_limit"],
        "rda_dynamic_init": None if rda_cfg is None else rda_cfg["dynamic_init"],
        "rda_groups": None if rda_cfg is None else rda_cfg["groups"],
        "rda_hidden_dim": None if rda_cfg is None else rda_cfg["hidden_dim"],
        "rda_input_affine": None if rda_cfg is None else rda_cfg["input_affine"],
        "rda_moment_affine_init": None if rda_cfg is None else rda_cfg["moment_affine_init"],
        "rlbx_base_init": None if rlbx_cfg is None else rlbx_cfg["base_init"],
        "rlbx_beta": None if rlbx_cfg is None else rlbx_cfg["beta"],
        "rlbx_centers": None if rlbx_cfg is None else rlbx_cfg["centers"],
        "rlbx_coeff_limit": None if rlbx_cfg is None else rlbx_cfg["coeff_limit"],
        "rlbx_groups": None if rlbx_cfg is None else rlbx_cfg["groups"],
        "rlbx_hidden_dim": None if rlbx_cfg is None else rlbx_cfg["hidden_dim"],
        "rlbx_input_affine": None if rlbx_cfg is None else rlbx_cfg["input_affine"],
        "rlb_base_init": None if rlb_cfg is None else rlb_cfg["base_init"],
        "rlb_beta": None if rlb_cfg is None else rlb_cfg["beta"],
        "rlb_centers": None if rlb_cfg is None else rlb_cfg["centers"],
        "rlb_coeff_limit": None if rlb_cfg is None else rlb_cfg["coeff_limit"],
        "rlb_groups": None if rlb_cfg is None else rlb_cfg["groups"],
        "rlb_hidden_dim": None if rlb_cfg is None else rlb_cfg["hidden_dim"],
        "rlb_input_affine": None if rlb_cfg is None else rlb_cfg["input_affine"],
        "rlb_fused": None if rlb_cfg is None else rlb_cfg["fused"],
        "rlb_center_odd": None if rlb_cfg is None else rlb_cfg["center_odd"],
        "rlb_train_centers": None if rlb_cfg is None else rlb_cfg["train_centers"],
        "rlb_atom_scale_init": None if rlb_cfg is None else rlb_cfg["atom_scale_init"],
        "rlb_atom_scale_limit": None if rlb_cfg is None else rlb_cfg["atom_scale_limit"],
        "rcq_base_init": None if rcq_cfg is None else rcq_cfg["base_init"],
        "rcq_beta": None if rcq_cfg is None else rcq_cfg["beta"],
        "rcq_coeff_limit": None if rcq_cfg is None else rcq_cfg["coeff_limit"],
        "rcq_groups": None if rcq_cfg is None else rcq_cfg["groups"],
        "rcq_hidden_dim": None if rcq_cfg is None else rcq_cfg["hidden_dim"],
        "rcq_init": None if rcq_cfg is None else rcq_cfg["init"],
        "rcq_input_affine": None if rcq_cfg is None else rcq_cfg["input_affine"],
        "rgc_base_init": None if rgc_cfg is None else rgc_cfg["base_init"],
        "rgc_beta": None if rgc_cfg is None else rgc_cfg["beta"],
        "rgc_coeff_limit": None if rgc_cfg is None else rgc_cfg["coeff_limit"],
        "rgc_groups": None if rgc_cfg is None else rgc_cfg["groups"],
        "rgc_hidden_dim": None if rgc_cfg is None else rgc_cfg["hidden_dim"],
        "rgc_init": None if rgc_cfg is None else rgc_cfg["init"],
        "rgc_input_affine": None if rgc_cfg is None else rgc_cfg["input_affine"],
        "rgc_moment_init": None if rgc_cfg is None else rgc_cfg["moment_init"],
        "rsm_even_basis_scale": None if rsm_cfg is None else rsm_cfg["even_basis_scale"],
        "rsm_even_gate_scale": None if rsm_cfg is None else rsm_cfg["even_gate_scale"],
        "rsm_groups": None if rsm_cfg is None else rsm_cfg["groups"],
        "rsm_hidden_dim": None if rsm_cfg is None else rsm_cfg["hidden_dim"],
        "rsm_odd_scale": None if rsm_cfg is None else rsm_cfg["odd_scale"],
        "rhg_gate_groups": None if rhg_cfg is None else rhg_cfg["gate_groups"],
        "rhg_gate_rank": None if rhg_cfg is None else rhg_cfg["gate_rank"],
        "rhg_gate_basis_count": None if rhg_cfg is None else rhg_cfg["gate_basis_count"],
        "rhg_gate_residual": None if rhg_cfg is None else rhg_cfg["gate_residual"],
        "rhg_conditional_value_basis_count": None if rhg_cfg is None else rhg_cfg["conditional_value_basis_count"],
        "rhg_conditional_value_dim": None if rhg_cfg is None else rhg_cfg["conditional_value_dim"],
        "rhg_value_activation": None if rhg_cfg is None else rhg_cfg["value_activation"],
        "rhg_value_residual": None if rhg_cfg is None else rhg_cfg["value_residual"],
        "rhg_value_residual_condition": None if rhg_cfg is None else rhg_cfg["value_residual_condition"],
        "rhg_value_residual_condition_init": None if rhg_cfg is None else rhg_cfg["value_residual_condition_init"],
        "rhg_value_residual_condition_mode": None if rhg_cfg is None else rhg_cfg["value_residual_condition_mode"],
        "rhg_value_residual_cross_rank": None if rhg_cfg is None else rhg_cfg["value_residual_cross_rank"],
        "rhg_value_residual_normalized": None if rhg_cfg is None else rhg_cfg["value_residual_normalized"],
        "rhg_value_residual_odd": None if rhg_cfg is None else rhg_cfg["value_residual_odd"],
        "rhg_value_residual_scale_init": None if rhg_cfg is None else rhg_cfg["value_residual_scale_init"],
        "rhg_value_residual_scale_mode": None if rhg_cfg is None else rhg_cfg["value_residual_scale_mode"],
        "rhg_value_residual_scale_schedule": None if rhg_cfg is None else rhg_cfg["value_residual_scale_schedule"],
        "rhg_value_residual_moment_condition": None if rhg_cfg is None else rhg_cfg["value_residual_moment_condition"],
        "rhg_value_residual_safe_gate_multiplier_init": None if rhg_cfg is None else rhg_cfg["value_residual_safe_gate_multiplier_init"],
        "rhg_diag_activation": None if rhg_cfg is None else rhg_cfg["diag_activation"],
        "rhg_diag_groups": None if rhg_cfg is None else rhg_cfg["diag_groups"],
        "rhg_hidden_dim": None if rhg_cfg is None else rhg_cfg["hidden_dim"],
        "rkdm_expert_rank": None if rkdm_cfg is None else rkdm_cfg["expert_rank"],
        "rkdm_experts": None if rkdm_cfg is None else rkdm_cfg["experts"],
        "rkdm_hidden_dim": None if rkdm_cfg is None else rkdm_cfg["hidden_dim"],
        "rkdm_latent_groups": None if rkdm_cfg is None else rkdm_cfg["latent_groups"],
        "rkdm_query_rank": None if rkdm_cfg is None else rkdm_cfg["query_rank"],
        "post_rational_init": args.post_rational_init,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "tokenizer": args.tokenizer,
        "train_tokens": train_tokens.numel(),
        "val_tokens": val_tokens.numel(),
        "world_size": world_size,
    }
    rank0_print(rank, json.dumps(config_record, sort_keys=True))
    if rank == 0:
        write_jsonl(out_path, config_record)

    step_times = []
    loss_since_log = 0.0
    steps_since_log = 0
    best_val_loss = math.inf
    stop_reason = None
    stop_step = None
    start_time = time.perf_counter()
    for step in range(args.steps):
        model.train()
        step_start = time.perf_counter()
        will_log = (step + 1) % args.log_interval == 0 or step == 0 or step + 1 == args.steps
        will_eval = args.eval_interval > 0 and (
            step == 0 or (step + 1) % args.eval_interval == 0 or step + 1 == args.steps
        )
        capture_step_telemetry = will_log and rank == 0
        if capture_step_telemetry and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        set_optimizer_telemetry_capture(optimizer, capture_step_telemetry)
        forward_backward_start = time.perf_counter()
        forward_backward_seconds = None
        optimizer_step_seconds = None
        grad_global_norm_before_clip = None
        grad_clip_triggered = False
        sam_first_grad_global_norm_before_clip = None
        sam_first_grad_clip_triggered = False

        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        optimizer.zero_grad(set_to_none=True)

        local_loss = 0.0
        rho = sam_rho(step, args)
        if rho > 0.0:
            batches = []
            for _ in range(args.grad_accum):
                x, y = sample_batch(train_tokens, args.batch_size, args.seq_len, offsets, train_generator, device)
                batches.append((x, y))
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
                (loss / args.grad_accum).backward()
                local_loss += float(loss.item())
            sam_first_grad_global_norm_before_clip, sam_first_grad_clip_triggered = clip_or_measure_gradients(
                model,
                args.grad_clip,
                capture_step_telemetry,
            )
            perturbations = sam_first_step(model, rho, args, device, is_distributed)
            if perturbations:
                optimizer.zero_grad(set_to_none=True)
                for x, y in batches:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
                    (loss / args.grad_accum).backward()
                sam_restore(perturbations)
                if capture_step_telemetry and device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_backward_seconds = time.perf_counter() - forward_backward_start
                grad_global_norm_before_clip, grad_clip_triggered = clip_or_measure_gradients(
                    model,
                    args.grad_clip,
                    capture_step_telemetry,
                )
            else:
                if capture_step_telemetry and device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_backward_seconds = time.perf_counter() - forward_backward_start
                grad_global_norm_before_clip = sam_first_grad_global_norm_before_clip
                grad_clip_triggered = sam_first_grad_clip_triggered
            optimizer_step_start = time.perf_counter()
            optimizer.step()
        else:
            for _ in range(args.grad_accum):
                x, y = sample_batch(train_tokens, args.batch_size, args.seq_len, offsets, train_generator, device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
                (loss / args.grad_accum).backward()
                local_loss += float(loss.item())
            if capture_step_telemetry and device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_backward_seconds = time.perf_counter() - forward_backward_start
            grad_global_norm_before_clip, grad_clip_triggered = clip_or_measure_gradients(
                model,
                args.grad_clip,
                capture_step_telemetry,
            )
            optimizer_step_start = time.perf_counter()
            optimizer.step()
        if capture_step_telemetry and device.type == "cuda":
            torch.cuda.synchronize(device)
        optimizer_step_seconds = time.perf_counter() - optimizer_step_start
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        mean_loss = reduce_mean(local_loss / args.grad_accum, device, is_distributed)
        loss_since_log += mean_loss
        steps_since_log += 1

        if will_log:
            recent = step_times[-args.log_interval :]
            mean_recent_step = sum(recent) / len(recent)
            tokens_per_second = global_tokens / mean_recent_step
            record = {
                "event": "train",
                "activation": args.activation,
                "step": step + 1,
                "loss": loss_since_log / max(1, steps_since_log),
                "lr": lr,
                "sam_rho": rho,
                "tokens_per_second": tokens_per_second,
                "seconds_per_step": mean_recent_step,
                "grad_global_norm_before_clip": _finite_float(grad_global_norm_before_clip),
                "grad_clip_triggered": bool(grad_clip_triggered),
                "grad_clip_threshold": args.grad_clip,
                "forward_backward_seconds": _finite_float(forward_backward_seconds),
                "optimizer_step_seconds": _finite_float(optimizer_step_seconds),
            }
            if sam_first_grad_global_norm_before_clip is not None:
                record["sam_first_grad_global_norm_before_clip"] = _finite_float(sam_first_grad_global_norm_before_clip)
                record["sam_first_grad_clip_triggered"] = bool(sam_first_grad_clip_triggered)
            if device.type == "cuda":
                record["cuda_max_memory_allocated"] = int(torch.cuda.max_memory_allocated(device))
                record["cuda_max_memory_reserved"] = int(torch.cuda.max_memory_reserved(device))
            if rank == 0:
                record.update(collect_optimizer_telemetry(optimizer))
                record.update(collect_rlb_telemetry(model, args))
            loss_since_log = 0.0
            steps_since_log = 0
            rank0_print(rank, json.dumps(record, sort_keys=True))
            if rank == 0:
                write_jsonl(out_path, record)

        if will_eval:
            val_loss = evaluate(model, val_tokens, args, offsets, rank, world_size, device, is_distributed)
            record = {
                "event": "eval",
                "activation": args.activation,
                "step": step + 1,
                "val_loss": val_loss,
                "val_ppl": math.exp(min(20.0, val_loss)),
            }
            record.update(evaluate_probe(model, probe_state, device, is_distributed))
            if rank == 0 and args.matrix_spectrum_interval > 0 and (
                step == 0 or (step + 1) % args.matrix_spectrum_interval == 0 or step + 1 == args.steps
            ):
                record.update(collect_matrix_spectrum_telemetry(model, args))
            rank0_print(rank, json.dumps(record, sort_keys=True))
            if rank == 0:
                write_jsonl(out_path, record)

            current_step = step + 1
            previous_best = best_val_loss
            if math.isfinite(val_loss):
                best_val_loss = min(best_val_loss, val_loss)
            min_step_met = current_step >= max(1, int(args.early_stop_min_step))
            max_loss = float(args.early_stop_max_val_loss)
            loss_increase = float(args.early_stop_loss_increase)
            too_large = max_loss > 0.0 and (not math.isfinite(val_loss) or val_loss > max_loss)
            worsened = (
                loss_increase > 0.0
                and math.isfinite(previous_best)
                and math.isfinite(val_loss)
                and val_loss > previous_best + loss_increase
            )
            if min_step_met and (too_large or worsened):
                stop_reason = "val_loss_above_threshold" if too_large else "val_loss_regressed_from_best"
                stop_step = current_step
                stop_record = {
                    "event": "stopped_early",
                    "activation": args.activation,
                    "step": stop_step,
                    "reason": stop_reason,
                    "val_loss": val_loss,
                    "best_val_loss": None if not math.isfinite(best_val_loss) else best_val_loss,
                    "early_stop_max_val_loss": max_loss,
                    "early_stop_loss_increase": loss_increase,
                }
                rank0_print(rank, json.dumps(stop_record, sort_keys=True))
                if rank == 0:
                    write_jsonl(out_path, stop_record)
                break

    total_time = time.perf_counter() - start_time
    warmup_drop = min(5, max(0, len(step_times) - 1))
    timed_steps = step_times[warmup_drop:]
    mean_step = sum(timed_steps) / max(1, len(timed_steps))
    completed_steps = stop_step if stop_step is not None else args.steps
    summary = {
        "event": "summary",
        "activation": args.activation,
        "mean_seconds_per_step": mean_step,
        "tokens_per_second": global_tokens / mean_step,
        "total_seconds": total_time,
        "steps": args.steps,
        "completed_steps": completed_steps,
        "stopped_early": stop_reason is not None,
        "stop_reason": stop_reason,
    }
    rank0_print(rank, json.dumps(summary, sort_keys=True))
    if rank == 0:
        write_jsonl(out_path, summary)
    if args.save_checkpoint and rank == 0:
        checkpoint_dir = (
            Path(args.checkpoint_dir)
            if args.checkpoint_dir is not None
            else Path(args.output_dir) / args.run_name / "checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        checkpoint_path = checkpoint_dir / f"{sanitize_name(args.activation)}_seed{args.seed}.pt"
        torch.save(
            {
                "model": raw_model.state_dict(),
                "args": vars(args),
                "config": config_record,
                "summary": summary,
                "param_count": param_count,
            },
            checkpoint_path,
        )
        checkpoint_record = {
            "event": "checkpoint",
            "activation": args.activation,
            "path": str(checkpoint_path),
            "seed": args.seed,
        }
        rank0_print(rank, json.dumps(checkpoint_record, sort_keys=True))
        write_jsonl(out_path, checkpoint_record)
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
