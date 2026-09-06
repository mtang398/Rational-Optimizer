#!/usr/bin/env python3
"""Exercise every CUDA operation required by the frozen activation binary."""

from __future__ import annotations

import json

import torch

from rational_opt import _C


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    required = (
        "local_basis_forward",
        "local_basis_backward",
        "local_basis_affine_statistics",
        "local_basis_restricted_backward",
    )
    missing = [name for name in required if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(f"CUDA extension operations missing: {missing}")

    device = torch.device("cuda", 0)
    rows, hidden_dim, groups, basis_count = 3, 8, 2, 2
    generator = torch.Generator(device=device).manual_seed(20260906)
    x = torch.randn(rows, hidden_dim, device=device, generator=generator)
    numerator = torch.randn(groups, 6, device=device, generator=generator)
    denominator = torch.randn(groups, 4, device=device, generator=generator)
    coeff_logits = torch.randn(
        groups, basis_count, 2, device=device, generator=generator
    )
    centers = torch.randn(groups, basis_count, device=device, generator=generator)
    beta = torch.rand(groups, basis_count, device=device, generator=generator)
    grad_output = torch.randn(x.shape, device=device, generator=generator)
    affine_alpha = torch.randn(groups, device=device, generator=generator)
    affine_beta = torch.randn(groups, device=device, generator=generator)
    coeff_limit, eps = 0.25, 1.0e-6

    feature = _C.local_basis_forward(
        x, numerator, denominator, coeff_logits, centers, beta,
        coeff_limit, eps, hidden_dim, groups,
    )
    statistics, rho = _C.local_basis_affine_statistics(
        x, feature, eps, hidden_dim, groups
    )
    backward = _C.local_basis_backward(
        grad_output, x, numerator, denominator, coeff_logits, centers, beta,
        coeff_limit, eps, hidden_dim, groups,
    )
    restricted = _C.local_basis_restricted_backward(
        grad_output, x, numerator, denominator, coeff_logits, centers, beta,
        affine_alpha, affine_beta, coeff_limit, eps, hidden_dim, groups,
    )
    torch.cuda.synchronize(device)

    outputs = [feature, statistics, rho, *backward, *restricted]
    if not all(torch.isfinite(value).all().item() for value in outputs):
        raise RuntimeError("CUDA extension smoke test produced a non-finite value")
    if feature.shape != x.shape or statistics.shape != (groups, 5):
        raise RuntimeError("CUDA extension smoke test returned an invalid shape")
    if rho.shape != (rows, groups, 1):
        raise RuntimeError("CUDA extension affine RMS shape is invalid")
    if restricted[0].shape != x.shape or restricted[1].shape != x.shape:
        raise RuntimeError("CUDA extension restricted-backward shape is invalid")
    print(json.dumps({
        "device": torch.cuda.get_device_name(device),
        "extension": _C.__file__,
        "operations": list(required),
        "passed": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
