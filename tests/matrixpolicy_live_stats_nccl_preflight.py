#!/usr/bin/env python3
"""Four-rank NCCL preflight for MatrixPolicy live-stat synchronization."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "activation"))

from activation.rational_opt.rational import RationalFusedGlobalA5_4
from optimizer_design.matrix_policy_optimizer import RationalMatrixPolicyOptimizer
from optimizer_design.transport_onpolicy_optimizer import (
    RationalTransportOnPolicyOptimizer,
    _RLBAdaptiveMetricBase,
)


def require_equal_across_ranks(value: torch.Tensor, name: str) -> None:
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    for peer_rank, peer in enumerate(gathered[1:], start=1):
        torch.testing.assert_close(
            peer,
            gathered[0],
            rtol=0.0,
            atol=0.0,
            msg=lambda message: f"{name} differs on rank {peer_rank}: {message}",
        )


def make_production_wrapper(device: torch.device):
    module = RationalFusedGlobalA5_4(hidden_dim=8, groups=2).to(device)
    in_weight = torch.nn.Parameter(torch.eye(8, device=device))
    out_weight = torch.nn.Parameter(torch.eye(8, device=device))
    group = {
        "module": module,
        "groups": module.groups,
        "hidden_dim": module.hidden_dim,
        "in_weight": in_weight,
        "out_weight": out_weight,
        "layer_index": 0,
        "num_layers": 1,
    }
    policy = RationalMatrixPolicyOptimizer(
        [
            {
                "params": [in_weight],
                "weight_decay": 0.0,
                "layer_index": 0,
                "num_layers": 1,
                "selector_index": 0,
                "matrix_role": "in",
            },
            {
                "params": [out_weight],
                "weight_decay": 0.0,
                "layer_index": 0,
                "num_layers": 1,
                "selector_index": 0,
                "matrix_role": "out",
            },
        ],
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        total_steps=16,
        selector_groups=[group],
        muon_strength=0.0,
        muon_lr_scale=0.0,
        max_muon=0.0,
        adam_lr_scale=1.0,
        adam_role_strength=0.0,
        group_gain_strength=0.2,
        group_pressure_strength=0.0,
        group_activity_damping=0.0,
        group_start=0.0,
        group_end=0.0,
    )
    wrapper = RationalTransportOnPolicyOptimizer(
        [policy],
        [group],
        total_steps=16,
        strength=0.0,
        stat_every=8,
        stat_samples=512,
        coeff_strength=0.0,
        matrix_strength=0.0,
        transport_strength=0.0,
    )
    return module, wrapper, group


def main() -> None:
    if not torch.cuda.is_available() or not dist.is_nccl_available():
        raise SystemExit("four-GPU NCCL preflight requires CUDA and NCCL")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise SystemExit(f"expected four ranks, found {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    try:
        torch.manual_seed(1701)
        module = RationalFusedGlobalA5_4(hidden_dim=8, groups=2).to(device)
        in_weight = torch.nn.Parameter(torch.eye(8, device=device))
        out_weight = torch.nn.Parameter(torch.eye(8, device=device))
        child = torch.optim.SGD([in_weight, out_weight], lr=0.1)
        child.group_gain_strength = 0.2
        group = {
            "module": module,
            "groups": module.groups,
            "hidden_dim": module.hidden_dim,
            "in_weight": in_weight,
            "out_weight": out_weight,
        }
        wrapper = _RLBAdaptiveMetricBase(
            [child],
            [group],
            total_steps=8,
            stat_every=1,
            stat_samples=512,
        )
        if not module._rlb_optimizer_stats_training_only:
            raise AssertionError("MatrixPolicy did not enable training-only live statistics")

        module.train()
        generator = torch.Generator(device=device).manual_seed(9000 + rank)
        positions = 2 + 2 * rank
        x = torch.randn(positions, 8, generator=generator, device=device)
        x[:, :4] *= 0.25 + rank
        x[:, 4:] *= 2.0 - 0.25 * rank
        module._update_optimizer_stats(x)

        local = torch.cat(
            [
                module._rlb_optimizer_stats["output_sq_sum"],
                module._rlb_optimizer_stats["derivative_sq_sum"],
                module._rlb_optimizer_stats["sample_count"],
            ]
        )
        expected = local.clone()
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)
        wrapper._synchronize_live_optimizer_stats()
        stats = module._rlb_optimizer_stats
        groups = module.groups
        expected_output = torch.sqrt(
            expected[:groups] / expected[2 * groups : 3 * groups] + module.eps
        )
        expected_derivative = torch.sqrt(
            expected[groups : 2 * groups]
            / expected[2 * groups : 3 * groups]
            + module.eps
        )
        torch.testing.assert_close(stats["output_rms"], expected_output)
        torch.testing.assert_close(stats["derivative_rms"], expected_derivative)
        if stats.get("sync_world_size") != world_size:
            raise AssertionError("live statistics do not record the NCCL world size")
        require_equal_across_ranks(stats["output_rms"], "output_rms")
        require_equal_across_ranks(stats["derivative_rms"], "derivative_rms")

        counter = module._rlb_optimizer_stat_counter
        version = module._rlb_optimizer_stat_version
        output_before_eval = stats["output_rms"].clone()
        derivative_before_eval = stats["derivative_rms"].clone()
        module.eval()
        with torch.no_grad():
            module._update_optimizer_stats(torch.randn(5 + rank, 8, device=device))
        if module._rlb_optimizer_stat_counter != counter:
            raise AssertionError("evaluation advanced the live-stat counter")
        if module._rlb_optimizer_stat_version != version:
            raise AssertionError("evaluation advanced the live-stat version")
        torch.testing.assert_close(stats["output_rms"], output_before_eval, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            stats["derivative_rms"], derivative_before_eval, rtol=0.0, atol=0.0
        )

        policy = RationalMatrixPolicyOptimizer(
            [
                {
                    "params": [in_weight],
                    "weight_decay": 0.0,
                    "layer_index": 0,
                    "num_layers": 1,
                    "selector_index": 0,
                    "matrix_role": "in",
                },
                {
                    "params": [out_weight],
                    "weight_decay": 0.0,
                    "layer_index": 0,
                    "num_layers": 1,
                    "selector_index": 0,
                    "matrix_role": "out",
                },
            ],
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            total_steps=8,
            selector_groups=[group],
            muon_strength=0.0,
            muon_lr_scale=0.0,
            max_muon=0.0,
            adam_lr_scale=1.0,
            adam_role_strength=0.0,
            group_gain_strength=0.2,
            group_pressure_strength=0.0,
            group_activity_damping=0.0,
            group_start=0.0,
            group_end=0.0,
        )
        in_weight.grad = torch.ones_like(in_weight)
        out_weight.grad = torch.ones_like(out_weight)
        policy.step()
        updated = torch.cat([in_weight.detach().flatten(), out_weight.detach().flatten()])
        require_equal_across_ranks(updated, "updated RLB matrix parameters")

        torch.manual_seed(777)
        production_module, production_wrapper, production_group = (
            make_production_wrapper(device)
        )

        def assign_synced_gradients(step_index: int) -> None:
            for parameter_index, parameter in enumerate(
                (production_group["in_weight"], production_group["out_weight"])
            ):
                values = torch.arange(
                    1,
                    parameter.numel() + 1,
                    device=device,
                    dtype=parameter.dtype,
                ).reshape_as(parameter)
                local = torch.sin(
                    values * (0.07 + 0.03 * step_index + 0.01 * parameter_index)
                    + 0.13 * rank
                )
                dist.all_reduce(local, op=dist.ReduceOp.SUM)
                parameter.grad = local / world_size

        production_module.train()
        first_input = torch.randn(
            3 + rank,
            8,
            generator=torch.Generator(device=device).manual_seed(12000 + rank),
            device=device,
        )
        production_module(first_input * (0.5 + rank))
        assign_synced_gradients(0)
        production_wrapper.step()

        counter_before_eval = production_module._rlb_optimizer_stat_counter
        version_before_eval = production_module._rlb_optimizer_stat_version
        stats_before_eval = production_module._rlb_optimizer_stats
        production_module.eval()
        with torch.no_grad():
            production_module(
                torch.full((5 + rank, 8), 1000.0 * (rank + 1), device=device)
            )
        if production_module._rlb_optimizer_stat_counter != counter_before_eval:
            raise AssertionError("production evaluation advanced the live-stat counter")
        if production_module._rlb_optimizer_stat_version != version_before_eval:
            raise AssertionError("production evaluation advanced the live-stat version")
        if production_module._rlb_optimizer_stats is not stats_before_eval:
            raise AssertionError("production evaluation replaced the live-stat cache")

        production_module.train()
        refresh_generator = torch.Generator(device=device).manual_seed(14000 + rank)
        for refresh_index in range(7):
            values = torch.randn(
                4 + rank,
                8,
                generator=refresh_generator,
                device=device,
            )
            values[:, :4] *= 0.20 + rank + refresh_index * 0.05
            values[:, 4:] *= 2.50 - 0.25 * rank
            production_module(values)
        if production_module._rlb_optimizer_stat_version <= version_before_eval:
            raise AssertionError("production training did not refresh the live-stat cache")
        assign_synced_gradients(1)
        production_wrapper.step()
        production_updated = torch.cat(
            [
                production_group["in_weight"].detach().flatten(),
                production_group["out_weight"].detach().flatten(),
            ]
        )
        require_equal_across_ranks(
            production_updated,
            "production outer-wrapper RLB matrix parameters",
        )
        if torch.equal(
            production_group["in_weight"].detach(),
            torch.eye(8, device=device),
        ):
            raise AssertionError("production MatrixPolicy did not update the input matrix")

        if rank == 0:
            module.train()
            module._update_optimizer_stats(torch.randn(4, 8, device=device))
        mismatch_detected = False
        try:
            wrapper._synchronize_live_optimizer_stats()
        except RuntimeError as error:
            if "cache versions differ across ranks" not in str(error):
                raise
            mismatch_detected = True
        mismatch_flag = torch.tensor(
            int(mismatch_detected),
            device=device,
            dtype=torch.int32,
        )
        dist.all_reduce(mismatch_flag, op=dist.ReduceOp.MIN)
        if int(mismatch_flag.item()) != 1:
            raise AssertionError("not every rank rejected a mismatched live-stat refresh")

        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "matrixpolicy_live_stats_nccl_preflight",
                        "status": "pass",
                        "world_size": world_size,
                        "weighted_sample_count": stats["sample_count"].tolist(),
                        "eval_cache_unchanged": True,
                        "parameters_identical": True,
                        "production_outer_wrapper_exercised": True,
                        "rank_mismatch_rejected": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
