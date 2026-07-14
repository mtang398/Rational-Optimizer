# MatrixPolicy ablation queue

This folder keeps the new MatrixPolicy ablations separate from the main ICLR result tree.

The original ablation source manifest was `experiments/manifests/iclr26_global_rational_matrixpolicy_manifest.csv`. Results that consume MatrixPolicy live RLB gains are superseded by the live-statistic correction; corrected E9 reruns are tracked under `experiments/corrections/matrixpolicy_live_stats_20260712/` and must pass their stage validator before paper use.

Output root: `abalation/runs/matrixpolicy_ablation_e1_e2`.

Log root: `abalation/logs`.

Variants:

- `rlb_matrixpolicy_no_role_depth`: keeps the current RLB global-rational activation and MatrixPolicy optimizer, but neutralizes the role/depth matrix-step specialization and the early Muon matrix branch. The RLB matrices remain trainable through the common MatrixPolicy AdamW matrix path, and the other MatrixPolicy controls remain enabled.
- `rlb_matrixpolicy_bypass_muon`: keeps the current RLB global-rational activation and MatrixPolicy matrix branch, but changes the AdamW-only bypass path to the existing Muon backbone option.
- `rlb_matrixpolicy_role_depth_v2`: keeps the current early Muon role/depth prior and group controls, but tapers only the persistent Adam role/depth multiplier from `1.20` to `0.40` over training progress `0.24 -> 0.42`. This tests whether the early role/depth routing benefit can be kept while reducing the later static depth bias seen in the 300M-token setting.
- `rlb_matrixpolicy_role_depth_v3`: keeps the Adam role/depth prior early, removes the Muon matrix branch, and tapers the Adam role/depth multiplier from `1.20` to `0.00` over progress `0.20 -> 0.36`. This tests whether the low-cost Adam role/depth signal alone can keep the early token advantage while matching the no-role/depth wall time.
- `rlb_matrixpolicy_role_depth_v4`: keeps a short, weaker Muon matrix pulse (`muon_strength=0.50`, `max_muon=0.50`, decay `0.16 -> 0.28`) and tapers the Adam role/depth multiplier from `1.20` to `0.10` over progress `0.20 -> 0.36`. This tests whether a small early orthogonalized matrix update recovers final-loss benefits without the original mid-phase drag.
- `rlb_matrixpolicy_role_depth_v5`: keeps the V2 residual Adam role/depth floor (`1.20 -> 0.40`) but shortens and weakens the Muon pulse moderately (`muon_strength=0.65`, `max_muon=0.65`, decay `0.18 -> 0.32`). This tests whether V2's final-loss advantage can be kept while removing some of the original mid-phase Muon cost.
- `rlb_matrixpolicy_role_depth_v6`: keeps the V2 residual Adam role/depth floor, keeps V4's shorter Muon pulse (`0.50`, decay `0.16 -> 0.28`), and adds a small scalar Adam pressure/activity guard from detached summaries (`adam_stat_strength=0.10`, `adam_pressure_balance=0.05`, active `0.16 -> 0.34`). This tests whether V4's 300M token-to-target behavior can be kept while recovering the final-loss quality lost by the `0.10` role floor.

Scope:

- 100M-token setting: 5 datasets x 3 seeds x 2 ablations = 30 runs.
- 300M-token setting: 5 datasets x 3 seeds x 2 ablations = 30 runs.
- Total: 60 independent Slurm array tasks, one manifest row per task.
- Role/depth V2: 5 datasets x 3 seeds x 2 token budgets = 30 additional independent Slurm array tasks, written to `abalation/runs/role_depth_v2_e1_e2`.
- Role/depth V3/V4: 2 variants x 5 datasets x 3 seeds x 2 token budgets = 60 additional independent Slurm array tasks, written to `abalation/runs/role_depth_v3_v4_e1_e2`.
- Role/depth V5/V6: 2 variants x 5 datasets x 3 seeds x 2 token budgets = 60 additional independent Slurm array tasks, written to `abalation/runs/role_depth_v5_v6_e1_e2`.
