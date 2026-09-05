"""Owner-free 32-row Global-RLB functional trust transaction.

Eight phase rows and ten activation rows underresolve the downstream-loss
curvature and keep choosing extreme group coefficients after their early
benefit has expired.  This method fixes a topology-independent measure of 32
exact functional rows.  It solves the equality-energy quadratic only in that
32-dimensional row space and reconstructs scalar coefficients on native
logical shards.  The fixed row count is independent of activation positions,
model width, machine count, and topology.

This is a new finite-measure approximation, not inherited endpoint evidence
from the dense R01 transaction.  Its endpoint must be measured afresh.
"""

from __future__ import annotations

from .rlb_owner_free_fixed_probe_r01 import OwnerFreeFixedProbeR01Optimizer
from .rlb_owner_free_strict_local import OwnerFreeStrictLocalR01Optimizer


FIXED_GLOBAL_PROBE_COUNT = 32
METHOD_ID = "fixed32_functional_row_muon_v1"


class Fixed32FunctionalRowMuonOptimizer(OwnerFreeFixedProbeR01Optimizer):
    """Exact 32-row loss image with an owner-free equality-budget solve."""

    component_code = 132
    checkpoint_schema = METHOD_ID
    inherited_parent = "matched_global_rlb_matrix_and_attention_muon"
    new_scientific_components = (
        "fixed_32_row_downstream_loss_measure",
        "owner_free_column_sharded_row_space_transaction",
    )
    execution_variant = METHOD_ID

    def __init__(
        self,
        *args,
        global_probe_count=FIXED_GLOBAL_PROBE_COUNT,
        loss_probe_group=None,
        **kwargs,
    ):
        if int(global_probe_count) != FIXED_GLOBAL_PROBE_COUNT:
            raise ValueError("the functional row measure must contain 32 rows")
        self.fixed_global_probe_count = FIXED_GLOBAL_PROBE_COUNT
        self.loss_probe_group = loss_probe_group
        self._loss_probe_layout = None
        self._loss_probe_capture_count = None
        self._fixed_probe_metadata = None
        OwnerFreeStrictLocalR01Optimizer.__init__(self, *args, **kwargs)

    def lr_wd_fairness_audit(self):
        report = super().lr_wd_fairness_audit()
        report.update({
            "fixed32_functional_measure_lr_scale": 1.0,
            "fixed32_row_space_lr_scale": 1.0,
            "fixed32_native_shard_coefficient_lr_scale": 1.0,
        })
        return report


__all__ = (
    "FIXED_GLOBAL_PROBE_COUNT",
    "Fixed32FunctionalRowMuonOptimizer",
    "METHOD_ID",
)
