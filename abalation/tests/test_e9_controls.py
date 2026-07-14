from __future__ import annotations

import copy
import math
import unittest
from unittest.mock import patch

import torch

from optimizer_design.matrix_policy_optimizer import RationalMatrixPolicyOptimizer
from optimizer_design.transport_onpolicy_optimizer import RationalTransportOnPolicyOptimizer


class StatTrackingIdentityRational(torch.nn.Module):
    """CPU-only test double for the RLB module's observable state contract."""

    def __init__(self, hidden_dim: int, groups: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.groups = groups
        self.eps = 1.0e-6
        numerator = torch.zeros(groups, 6)
        numerator[:, 1] = 1.0
        self.numerator = torch.nn.Parameter(numerator)
        self.denominator = torch.nn.Parameter(torch.zeros(groups, 4))
        self._rlb_optimizer_track_stats = True
        self._rlb_optimizer_stat_counter = 0
        self._rlb_optimizer_stats = {"sentinel": torch.tensor([3.0])}

    def forward(self, value):
        if self._rlb_optimizer_track_stats:
            self._rlb_optimizer_stat_counter += 1
            self._rlb_optimizer_stats = {"sentinel": value.detach().mean().reshape(1)}
        return value


class E9ControlTests(unittest.TestCase):
    def assert_suppressed_muon_matches_adam_oracle(self, *, role_depth: bool):
        base_lr = 1.0e-2
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        oracle_parameter = torch.nn.Parameter(parameter.detach().clone())
        weight_decay = 0.1
        optimizer = RationalMatrixPolicyOptimizer(
            [
                {
                    "params": [parameter],
                    "matrix_role": "in",
                    "layer_index": 0,
                    "num_layers": 3,
                    "selector_index": -1,
                    "weight_decay": weight_decay,
                }
            ],
            lr=base_lr,
            weight_decay=weight_decay,
            total_steps=4,
            apply_muon_update=False,
            muon_strength=0.4,
            start=0.0,
            end=0.5,
            decay_start=1.0,
            decay_end=1.0,
            max_muon=0.75,
            adam_lr_scale=3.0,
            adam_role_strength=1.2 if role_depth else 0.0,
            input_depth_gain=-0.5 if role_depth else 0.0,
            output_depth_gain=1.0 if role_depth else 0.0,
        )
        oracle = torch.optim.AdamW(
            [oracle_parameter],
            lr=base_lr,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=weight_decay,
        )
        self.assertIsNotNone(optimizer.muon)
        muon_state_before = copy.deepcopy(optimizer.muon.state_dict())
        observed_adam_lrs = []
        adam_step = optimizer.adam.step

        def adam_spy(*args, **kwargs):
            observed_adam_lrs.append(float(optimizer.adam.param_groups[0]["lr"]))
            return adam_step(*args, **kwargs)

        optimizer.adam.step = adam_spy
        with patch.object(optimizer.muon, "step", wraps=optimizer.muon.step) as muon_step:
            fractions = (0.25, 0.50, 0.50) if role_depth else (0.20, 0.40, 0.40)
            nominal_adam_scale = 3.9 if role_depth else 3.0
            for step, fraction in zip((1, 2, 3), fractions):
                gradient = torch.full_like(parameter, float(step))
                parameter.grad = gradient.clone()
                oracle_parameter.grad = gradient.clone()
                optimizer.set_telemetry_capture(True)
                optimizer.step()
                oracle.param_groups[0]["lr"] = base_lr * nominal_adam_scale * (1.0 - fraction)
                oracle.step()
                torch.testing.assert_close(parameter, oracle_parameter, rtol=0.0, atol=0.0)
            muon_step.assert_not_called()

        expected = [base_lr * nominal_adam_scale * (1.0 - fraction) for fraction in fractions]
        for observed, wanted in zip(observed_adam_lrs, expected):
            self.assertTrue(math.isclose(observed, wanted, rel_tol=0.0, abs_tol=1.0e-15))
        self.assertEqual(optimizer.muon.state, {})
        self.assertEqual(optimizer.muon.state_dict(), muon_state_before)
        self.assertEqual(int(optimizer.adam.state[parameter]["step"].item()), 3)
        self.assertEqual(optimizer.adam.param_groups[0]["lr"], base_lr)
        telemetry = optimizer.telemetry()
        if role_depth:
            self.assertEqual(telemetry["matrix_policy_muon_mix_mean_by_role"], {"in": 0.5})
        else:
            self.assertEqual(telemetry["matrix_policy_muon_mix_mean_by_role"], {"in": 0.4})
        self.assertEqual(telemetry["matrix_policy_applied_muon_mix_mean_by_role"], {"in": 0.0})

    def test_a6_suppressed_muon_matches_attenuated_adamw(self):
        self.assert_suppressed_muon_matches_adam_oracle(role_depth=True)

    def test_a7_suppressed_muon_without_role_depth_matches_attenuated_adamw(self):
        self.assert_suppressed_muon_matches_adam_oracle(role_depth=False)

    @staticmethod
    def make_pair_optimizer(strength: float):
        torch.manual_seed(7)
        activation = StatTrackingIdentityRational(hidden_dim=4, groups=2)
        in_weight = torch.nn.Parameter(8.0 * torch.randn(4, 3))
        out_weight = torch.nn.Parameter(0.125 * torch.randn(2, 4))
        child = torch.optim.SGD([in_weight, out_weight], lr=0.0)
        group = {
            "module": activation,
            "in_weight": in_weight,
            "out_weight": out_weight,
            "numerator": activation.numerator,
            "denominator": activation.denominator,
            "coeff_logits": None,
            "centers": None,
            "beta": None,
            "coeff_limit": 0.0,
            "groups": 2,
            "hidden_dim": 4,
            "layer_index": 0,
            "num_layers": 1,
        }
        optimizer = RationalTransportOnPolicyOptimizer(
            [child],
            [group],
            total_steps=10,
            strength=strength,
            start=0.0,
            end=0.0,
            every=1,
            max_log_step=0.03,
            target_weight=0.0,
            pressure_weight=0.0,
            rational_activity_weight=0.0,
            matrix_strength=0.0,
            coeff_strength=0.0,
            quotient_strength=0.0,
            transport_strength=0.0,
        )
        activation._rlb_optimizer_stats = {"sentinel": torch.tensor([3.0])}
        activation._rlb_optimizer_stat_counter = 17
        in_weight.grad = torch.zeros_like(in_weight)
        out_weight.grad = torch.zeros_like(out_weight)
        return optimizer, activation, in_weight, out_weight

    def test_pair_probe_is_observational_and_rescale_is_nearly_invariant(self):
        optimizer, activation, in_weight, out_weight = self.make_pair_optimizer(strength=1.0)
        in_before = in_weight.detach().clone()
        out_before = out_weight.detach().clone()
        in_product_before = in_before.view(2, 2, -1).square().mean((1, 2)).sqrt()
        out_product_before = out_before.view(2, 2, 2).permute(1, 2, 0).square().mean((1, 2)).sqrt()
        norm_products_before = in_product_before * out_product_before
        stats_ref = activation._rlb_optimizer_stats
        stats_before = {key: value.clone() for key, value in stats_ref.items()}
        optimizer.set_telemetry_capture(True)
        optimizer.step()
        telemetry = optimizer.telemetry()

        self.assertTrue(activation._rlb_optimizer_track_stats)
        self.assertEqual(activation._rlb_optimizer_stat_counter, 17)
        self.assertIs(activation._rlb_optimizer_stats, stats_ref)
        for key, expected in stats_before.items():
            torch.testing.assert_close(activation._rlb_optimizer_stats[key], expected, rtol=0.0, atol=0.0)
        self.assertTrue(telemetry["matrix_policy_pair_rescale_scheduled"])
        self.assertTrue(telemetry["matrix_policy_pair_rescale_applied"])
        self.assertEqual(telemetry["matrix_policy_pair_rescale_attempted_count"], 2)
        self.assertGreater(telemetry["matrix_policy_pair_log_move_abs_mean"], 0.0)
        self.assertLess(telemetry["matrix_policy_pair_local_probe_relative_delta_max"], 1.0e-5)
        self.assertFalse(torch.equal(in_weight, in_before))
        self.assertFalse(torch.equal(out_weight, out_before))
        in_product_after = in_weight.view(2, 2, -1).square().mean((1, 2)).sqrt()
        out_product_after = out_weight.view(2, 2, 2).permute(1, 2, 0).square().mean((1, 2)).sqrt()
        torch.testing.assert_close(
            in_product_after * out_product_after,
            norm_products_before,
            rtol=2.0e-6,
            atol=1.0e-7,
        )
        self.assertTrue(torch.isfinite(in_weight).all())
        self.assertTrue(torch.isfinite(out_weight).all())

    def test_disabled_pair_rescale_is_bitwise_identity(self):
        optimizer, _, in_weight, out_weight = self.make_pair_optimizer(strength=0.0)
        in_before = in_weight.detach().clone()
        out_before = out_weight.detach().clone()
        optimizer.set_telemetry_capture(True)
        optimizer.step()
        telemetry = optimizer.telemetry()
        torch.testing.assert_close(in_weight, in_before, rtol=0.0, atol=0.0)
        torch.testing.assert_close(out_weight, out_before, rtol=0.0, atol=0.0)
        self.assertFalse(telemetry["matrix_policy_pair_rescale_enabled"])
        self.assertFalse(telemetry["matrix_policy_pair_rescale_scheduled"])
        self.assertFalse(telemetry["matrix_policy_pair_rescale_applied"])
        self.assertEqual(telemetry["matrix_policy_pair_local_probe_relative_delta_max"], 0.0)
        self.assertEqual(telemetry["matrix_policy_pair_rescale_attempted_count"], 0)
        self.assertEqual(telemetry["matrix_policy_pair_rescale_diagnosed_group_count"], 2)

    def test_pair_telemetry_capture_does_not_change_rank_outcome(self):
        capture, capture_activation, capture_in, capture_out = self.make_pair_optimizer(strength=1.0)
        plain, plain_activation, plain_in, plain_out = self.make_pair_optimizer(strength=1.0)
        capture.set_telemetry_capture(True)
        capture.step()
        plain.step()
        torch.testing.assert_close(capture_in, plain_in, rtol=0.0, atol=0.0)
        torch.testing.assert_close(capture_out, plain_out, rtol=0.0, atol=0.0)
        self.assertEqual(capture_activation._rlb_optimizer_stat_counter, plain_activation._rlb_optimizer_stat_counter)
        for key in capture_activation._rlb_optimizer_stats:
            torch.testing.assert_close(
                capture_activation._rlb_optimizer_stats[key],
                plain_activation._rlb_optimizer_stats[key],
                rtol=0.0,
                atol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
