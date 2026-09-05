"""Lifecycle and scale gates for the factorized every-step RFD ledger."""

from __future__ import annotations

import torch

from optimizer_design.rlb_factorized_every_step_rfd_gradient_ledger_muon import (
    FAMILY_ID,
    PREFIX,
    FactorizedEveryStepRFDGradientLedgerAttentionOptimizer,
    FactorizedEveryStepRFDGradientLedgerRouter,
    factorized_every_step_rfd_gradient_ledger_scaling_formula,
)


class _TinyActivation(torch.nn.Module):
    def __init__(self, hidden: int, groups: int):
        super().__init__()
        self.hidden_dim = hidden
        self.groups = groups
        self.eps = 1.0e-6
        self.numerator = torch.nn.Parameter(torch.randn(groups, 6) * 0.1)
        self.denominator = torch.nn.Parameter(0.4 + torch.rand(groups, 4))

    def forward(self, value):
        shape = value.shape
        width = self.hidden_dim // self.groups
        grouped = value.view(*shape[:-1], self.groups, width)
        rms = torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
        unit = grouped / rms
        powers = torch.stack(tuple(unit.pow(index) for index in range(6)), dim=-1)
        den_powers = torch.stack(
            tuple(unit.abs().pow(index) for index in range(1, 5)), dim=-1
        )
        polynomial = (powers * self.numerator.view(1, self.groups, 1, 6)).sum(-1)
        quotient = 1.0 + (
            den_powers * self.denominator.abs().view(1, self.groups, 1, 4)
        ).sum(-1)
        return (rms * polynomial / quotient).reshape(shape)


class _TinyMLP(torch.nn.Module):
    def __init__(self, residual: int, hidden: int, groups: int):
        super().__init__()
        self.in_proj = torch.nn.Linear(residual, hidden, bias=False)
        self.rlb_activation = _TinyActivation(hidden, groups)
        self.out_proj = torch.nn.Linear(hidden, residual, bias=False)

    def forward(self, value):
        return self.out_proj(self.rlb_activation(self.in_proj(value)))


def _make_pair():
    mlp = _TinyMLP(16, 18, 2)
    qkv = torch.nn.Parameter(torch.randn(48, 16) * 0.02)
    output = torch.nn.Parameter(torch.randn(16, 16) * 0.02)
    activation = mlp.rlb_activation
    pair = {
        "layer_index": 0,
        "mlp": mlp,
        "module": activation,
        "in_weight": mlp.in_proj.weight,
        "out_weight": mlp.out_proj.weight,
        "numerator": activation.numerator,
        "denominator": activation.denominator,
        "groups": activation.groups,
        "hidden_dim": activation.hidden_dim,
        "group_width": activation.hidden_dim // activation.groups,
        "eps": activation.eps,
        "qkv_weight": qkv,
        "attn_out_weight": output,
    }
    return mlp, qkv, output, pair


def _transition(mlp, qkv, output, router, attention, generator):
    mlp.zero_grad(set_to_none=True)
    for _ in range(4):
        value = torch.randn(2, 4, 16, generator=generator)
        (mlp(value).square().mean() / 4.0).backward()
    qkv.grad = torch.randn(qkv.shape, generator=generator)
    output.grad = torch.randn(output.shape, generator=generator)
    preclip = torch.nn.utils.clip_grad_norm_(
        list(mlp.parameters()) + [qkv, output], 1.0
    )
    router.record_realized_clipping(float(preclip), 1.0)
    router.set_telemetry_capture()
    attention.set_telemetry_capture()
    router.step()
    attention.step()


def _optimizers(pair):
    router = FactorizedEveryStepRFDGradientLedgerRouter(
        [pair], lr=3e-4, weight_decay=.1, momentum=.95,
        ns_steps=5, beta2=.95, eps=1e-8,
    )
    attention = FactorizedEveryStepRFDGradientLedgerAttentionOptimizer(
        [pair], router, lr=3e-4, weight_decay=.1, momentum=.95,
        ns_steps=5, beta2=.95, eps=1e-8,
        adjust_lr_fn="match_rms_adamw",
    )
    return router, attention


def test_factorized_rfd_nine_step_lifecycle_fairness_and_checkpoint():
    torch.manual_seed(20261025)
    mlp, qkv, output, pair = _make_pair()
    router, attention = _optimizers(pair)
    generator = torch.Generator().manual_seed(20261026)
    refresh = []
    surrogate = []
    selection_rows = []
    persistent_rows = []
    for expected_step in range(1, 10):
        _transition(mlp, qkv, output, router, attention, generator)
        report = router.telemetry()
        refresh.append(report[PREFIX + "functional_score_refresh"])
        surrogate.append(report[PREFIX + "trace_matched_gradient_surrogate"])
        selection_rows.append(report[PREFIX + "selection_factor_rows"])
        persistent_rows.append(report[PREFIX + "persistent_factor_rows"])
        assert report[PREFIX + "family_id"] == FAMILY_ID
        assert report[PREFIX + "factorized_parameter_direction_unchanged"] == 1
        assert report[PREFIX + "every_step_gradient_score_ledger"] == 1
        assert report[PREFIX + "matched_beta2_every_optimizer_step"] == 1
        assert report[PREFIX + "robust_fd_midpoint_tail"] == 1
        assert report[PREFIX + "signed_coefficients_allowed"] == 1
        assert report[PREFIX + "ledger_step"] == expected_step
        assert report[PREFIX + "owner_count"] == 0
        assert report[PREFIX + "dense_lg_metric_elements"] == 0
        assert report[PREFIX + "selected_update_elements_published"] == 0
        assert report[PREFIX + "largest_dense_solve_dimension"] == 96
        assert report[PREFIX + "largest_transaction_dense_dimension"] == 32
        assert 0 <= report[PREFIX + "cross_layer_coupling_ratio"] <= 1
        assert attention.telemetry()[PREFIX + "attention_family_id"] == FAMILY_ID
    assert refresh == [1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert surrogate == [0, 1, 1, 1, 1, 1, 1, 1, 0]
    assert selection_rows == [32, 64, 96, 96, 96, 96, 96, 96, 96]
    assert persistent_rows == [32, 64, 64, 64, 64, 64, 64, 64, 64]
    assert set(router.lr_wd_fairness_audit().values()) == {1.0}
    assert set(attention.lr_wd_fairness_audit().values()) == {1.0}
    state = router.state_dict()
    router.load_state_dict(state)
    anchor = router.state[pair["in_weight"]]
    assert anchor["cadence8_transition"] == 9
    assert anchor["factorized_rfd_step"] == 9
    assert anchor["factorized_rfd_persistent_scores"].shape == (64, 2)


def test_factorized_rfd_scaling_is_n_invariant_subquadratic_and_owner_free():
    dimensions = dict(
        total_layers=18, total_groups=18,
        intermediate_width=4608, model_width=1024,
    )
    small = factorized_every_step_rfd_gradient_ledger_scaling_formula(
        total_positions=1, **dimensions
    )
    large = factorized_every_step_rfd_gradient_ledger_scaling_formula(
        total_positions=1_050_000, **dimensions
    )
    assert small["persistent_state_elements"] == large["persistent_state_elements"]
    assert large["coordinate_count"] == 324
    assert large["gradient_ledger_mathematical_state_elements"] == 66 * 324 + 4
    assert large["gradient_ledger_mathematical_state_elements"] < 324**2
    assert large["maximum_live_factor_elements"] == 96 * 324
    assert large["state_depends_on_total_activation_positions"] == 0
    assert large["owner_count"] == 0
    assert large["complete_layer_owners"] == 0
    assert large["complete_coordinate_owners"] == 0
    assert large["owner_local_mathematics"] == 0
    assert large["dense_lg_by_lg_metric_elements"] == 0
    assert large["selected_update_elements_published"] == 0
    assert large["largest_dense_solve_dimension"] == 96
    assert large["largest_transaction_dense_dimension"] == 32
