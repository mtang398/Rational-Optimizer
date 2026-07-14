from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "activation"))

from activation.rational_opt.rational import RationalFusedGlobalA5_4
from optimizer_design.matrix_policy_optimizer import RationalMatrixPolicyOptimizer
from optimizer_design.transport_onpolicy_optimizer import (
    RationalTransportOnPolicyOptimizer,
    _RLBAdaptiveMetricBase,
)


def _make_wrapper(module: RationalFusedGlobalA5_4):
    in_weight = torch.nn.Parameter(torch.eye(module.hidden_dim))
    out_weight = torch.nn.Parameter(torch.eye(module.hidden_dim))
    optimizer = torch.optim.SGD([in_weight, out_weight], lr=0.1)
    optimizer.group_gain_strength = 0.2
    group = {
        "module": module,
        "groups": module.groups,
        "hidden_dim": module.hidden_dim,
        "in_weight": in_weight,
        "out_weight": out_weight,
    }
    wrapper = _RLBAdaptiveMetricBase(
        [optimizer],
        [group],
        total_steps=10,
        stat_every=1,
        stat_samples=512,
    )
    return wrapper


def _make_multilayer_wrapper(modules: list[RationalFusedGlobalA5_4]):
    parameters = []
    groups = []
    for layer_index, module in enumerate(modules):
        in_weight = torch.nn.Parameter(torch.eye(module.hidden_dim))
        out_weight = torch.nn.Parameter(torch.eye(module.hidden_dim))
        parameters.extend((in_weight, out_weight))
        groups.append(
            {
                "module": module,
                "groups": module.groups,
                "hidden_dim": module.hidden_dim,
                "in_weight": in_weight,
                "out_weight": out_weight,
                "layer_index": layer_index,
                "num_layers": len(modules),
            }
        )
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    optimizer.group_gain_strength = 0.2
    wrapper = _RLBAdaptiveMetricBase(
        [optimizer],
        groups,
        total_steps=10,
        stat_every=1,
        stat_samples=512,
    )
    return wrapper


def _make_production_wrapper(module: RationalFusedGlobalA5_4):
    in_weight = torch.nn.Parameter(torch.eye(module.hidden_dim))
    out_weight = torch.nn.Parameter(torch.eye(module.hidden_dim))
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
    return wrapper, policy, group


def _distributed_worker(rank: int, world_size: int, init_file: str):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(100 + rank)
        modules = [
            RationalFusedGlobalA5_4(hidden_dim=8, groups=2),
            RationalFusedGlobalA5_4(hidden_dim=12, groups=3),
        ]
        wrapper = _make_multilayer_wrapper(modules)
        expected_parts = []
        for layer_index, module in enumerate(modules):
            module.train()
            positions = 3 + 2 * rank + layer_index
            x = torch.randn(positions, module.hidden_dim)
            split = module.hidden_dim // 2
            x[:, :split] *= 0.5 + rank + 0.25 * layer_index
            x[:, split:] *= 2.0 - 0.5 * rank + 0.10 * layer_index
            module._update_optimizer_stats(x)
            expected_parts.extend(
                (
                    module._rlb_optimizer_stats["output_sq_sum"].clone(),
                    module._rlb_optimizer_stats["derivative_sq_sum"].clone(),
                    module._rlb_optimizer_stats["sample_count"].clone(),
                )
            )
        expected = torch.cat(expected_parts)
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)

        all_reduce_calls = 0
        original_all_reduce = dist.all_reduce

        def counted_all_reduce(*args, **kwargs):
            nonlocal all_reduce_calls
            all_reduce_calls += 1
            return original_all_reduce(*args, **kwargs)

        with mock.patch.object(dist, "all_reduce", side_effect=counted_all_reduce):
            wrapper._synchronize_live_optimizer_stats()
            assert all_reduce_calls == 1
            wrapper._synchronize_live_optimizer_stats()
            assert all_reduce_calls == 2

        offset = 0
        for module in modules:
            stats = module._rlb_optimizer_stats
            groups = module.groups
            expected_output_sum = expected[offset : offset + groups]
            offset += groups
            expected_derivative_sum = expected[offset : offset + groups]
            offset += groups
            expected_count = expected[offset : offset + groups]
            offset += groups
            expected_output = torch.sqrt(
                expected_output_sum / expected_count + module.eps
            )
            expected_derivative = torch.sqrt(
                expected_derivative_sum / expected_count + module.eps
            )
            torch.testing.assert_close(stats["output_rms"], expected_output)
            torch.testing.assert_close(stats["derivative_rms"], expected_derivative)
            assert stats["sync_world_size"] == world_size

            gathered = [torch.empty_like(stats["output_rms"]) for _ in range(world_size)]
            dist.all_gather(gathered, stats["output_rms"])
            for peer in gathered[1:]:
                torch.testing.assert_close(peer, gathered[0], rtol=0.0, atol=0.0)

        group = wrapper.balance_groups[0]
        policy = RationalMatrixPolicyOptimizer(
            [
                {
                    "params": [group["in_weight"]],
                    "weight_decay": 0.0,
                    "layer_index": 0,
                    "num_layers": 1,
                    "selector_index": 0,
                    "matrix_role": "in",
                },
                {
                    "params": [group["out_weight"]],
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
            total_steps=10,
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
        group["in_weight"].grad = torch.ones_like(group["in_weight"])
        group["out_weight"].grad = torch.ones_like(group["out_weight"])
        policy.step()
        updated = torch.cat(
            [group["in_weight"].detach().reshape(-1), group["out_weight"].detach().reshape(-1)]
        )
        updated_by_rank = [torch.empty_like(updated) for _ in range(world_size)]
        dist.all_gather(updated_by_rank, updated)
        for peer in updated_by_rank[1:]:
            torch.testing.assert_close(peer, updated_by_rank[0], rtol=0.0, atol=0.0)

        module = modules[0]
        stats = module._rlb_optimizer_stats
        synced_version = module._rlb_optimizer_stats_synced_version
        before = stats["output_rms"].clone()
        wrapper._synchronize_live_optimizer_stats()
        assert module._rlb_optimizer_stats_synced_version == synced_version
        torch.testing.assert_close(stats["output_rms"], before, rtol=0.0, atol=0.0)

        torch.manual_seed(777)
        production_module = RationalFusedGlobalA5_4(hidden_dim=8, groups=2)
        production_wrapper, _, production_group = _make_production_wrapper(
            production_module
        )

        def assign_synced_gradients(step_index: int):
            for parameter_index, parameter in enumerate(
                (production_group["in_weight"], production_group["out_weight"])
            ):
                values = torch.arange(
                    1,
                    parameter.numel() + 1,
                    dtype=parameter.dtype,
                ).reshape_as(parameter)
                local = torch.sin(
                    values * (0.07 + 0.03 * step_index + 0.01 * parameter_index)
                    + 0.13 * rank
                )
                dist.all_reduce(local, op=dist.ReduceOp.SUM)
                parameter.grad = local / world_size

        production_module.train()
        production_module(torch.randn(3 + rank, 8) * (0.5 + rank))
        assign_synced_gradients(0)
        production_wrapper.step()

        counter_before_eval = production_module._rlb_optimizer_stat_counter
        version_before_eval = production_module._rlb_optimizer_stat_version
        stats_before_eval = production_module._rlb_optimizer_stats
        production_module.eval()
        production_module(torch.full((5 + rank, 8), 1000.0 * (rank + 1)))
        assert production_module._rlb_optimizer_stat_counter == counter_before_eval
        assert production_module._rlb_optimizer_stat_version == version_before_eval
        assert production_module._rlb_optimizer_stats is stats_before_eval

        production_module.train()
        for refresh_index in range(7):
            values = torch.randn(4 + rank, 8)
            values[:, :4] *= 0.20 + rank + refresh_index * 0.05
            values[:, 4:] *= 2.50 - 0.25 * rank
            production_module(values)
        assert production_module._rlb_optimizer_stat_version > version_before_eval
        assign_synced_gradients(1)
        production_wrapper.step()
        production_updated = torch.cat(
            [
                production_group["in_weight"].detach().reshape(-1),
                production_group["out_weight"].detach().reshape(-1),
            ]
        )
        production_by_rank = [
            torch.empty_like(production_updated) for _ in range(world_size)
        ]
        dist.all_gather(production_by_rank, production_updated)
        for peer in production_by_rank[1:]:
            torch.testing.assert_close(
                peer,
                production_by_rank[0],
                rtol=0.0,
                atol=0.0,
            )
        assert not torch.equal(
            production_group["in_weight"].detach(),
            torch.eye(production_module.hidden_dim),
        )

        if rank == 0:
            module.train()
            module._update_optimizer_stats(torch.randn(4, module.hidden_dim))
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError,
            "cache versions differ across ranks",
        ):
            wrapper._synchronize_live_optimizer_stats()
    finally:
        dist.destroy_process_group()


class MatrixPolicyLiveStatsTest(unittest.TestCase):
    def test_telemetry_only_rlb_keeps_existing_eval_behavior(self):
        module = RationalFusedGlobalA5_4(hidden_dim=8, groups=2)
        module._rlb_optimizer_track_stats = True
        module._rlb_optimizer_stat_every = 1
        module._rlb_optimizer_stat_samples = 512
        module.eval()
        module._update_optimizer_stats(torch.randn(4, 8))
        self.assertEqual(module._rlb_optimizer_stat_counter, 1)
        self.assertEqual(module._rlb_optimizer_stat_version, 1)
        self.assertFalse(
            bool(getattr(module, "_rlb_optimizer_sync_stats", False))
        )

    def test_refresh_version_changes_only_when_cache_refreshes(self):
        module = RationalFusedGlobalA5_4(hidden_dim=8, groups=2)
        module._rlb_optimizer_track_stats = True
        module._rlb_optimizer_stats_training_only = True
        module._rlb_optimizer_stat_every = 8
        module.train()
        for _ in range(7):
            module._update_optimizer_stats(torch.randn(4, 8))
        self.assertEqual(module._rlb_optimizer_stat_counter, 7)
        self.assertEqual(module._rlb_optimizer_stat_version, 1)
        module._update_optimizer_stats(torch.randn(4, 8))
        self.assertEqual(module._rlb_optimizer_stat_counter, 8)
        self.assertEqual(module._rlb_optimizer_stat_version, 2)

    def test_training_only_cache_does_not_advance_during_eval(self):
        module = RationalFusedGlobalA5_4(hidden_dim=8, groups=2)
        wrapper = _make_wrapper(module)
        self.assertTrue(module._rlb_optimizer_stats_training_only)

        module.train()
        module._update_optimizer_stats(torch.randn(4, 8))
        counter = module._rlb_optimizer_stat_counter
        stats = module._rlb_optimizer_stats

        module.eval()
        module._update_optimizer_stats(torch.randn(4, 8))
        self.assertEqual(module._rlb_optimizer_stat_counter, counter)
        self.assertIs(module._rlb_optimizer_stats, stats)

        module._rlb_optimizer_stats_training_only = False
        module._update_optimizer_stats(torch.randn(4, 8))
        self.assertEqual(module._rlb_optimizer_stat_counter, counter + 1)

        del wrapper

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "Gloo distributed backend is required",
    )
    def test_weighted_live_stats_are_identical_across_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "gloo_init")
            mp.spawn(
                _distributed_worker,
                args=(2, init_file),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
