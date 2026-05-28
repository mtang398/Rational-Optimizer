# Read First

This repo is about designing an RLB-specific optimizer. Do not claim wins from a different global LR schedule.

## GPU Rule

Use A6000 GPUs only. A normal job requests 4 GPUs, and the maximum concurrent request is 8 A6000s total.

```text
--gres=gpu:nvidia_rtx_a6000:4
--nproc_per_node=4
```

Check active jobs before launching:

```bash
squeue -u mt872
```

## Current Optimizer

```text
optimizer                                           rational_matrix_policy_onpolicy
activation                                          rlb_fused_fixed_strong_ffn
optimizer_lr                                        3e-4
optimizer_min_lr                                    3e-5
warmup_steps                                        200
lr_schedule                                         cosine
rational_matrix_policy_backbone_optimizer           adamw
rational_matrix_policy_backbone_beta2               0.999
rational_matrix_policy_beta2                        0.999
rational_matrix_policy_adam_lr_scale                3.00
rational_matrix_policy_adam_lr_scale_final          null
rational_matrix_policy_adam_decay_start             1.10
rational_matrix_policy_adam_decay_end               1.10
rational_matrix_policy_adam_decay_depth_shift       0.00
rational_matrix_policy_adam_beta2_final             null
rational_matrix_policy_adam_beta2_decay_start       1.10
rational_matrix_policy_adam_beta2_decay_end         1.10
rational_matrix_policy_adam_beta2_decay_depth_shift 0.00
rational_matrix_policy_adam_role_strength           1.20
rational_matrix_policy_adam_stat_strength           0.00
rational_matrix_policy_adam_pressure_balance        0.00
rational_matrix_policy_adam_stat_start              0.00
rational_matrix_policy_adam_stat_end                0.00
rational_matrix_policy_adam_min_lr_scale            0.40
rational_matrix_policy_adam_max_lr_scale            4.00
rational_matrix_policy_input_depth_gain            -0.50
rational_matrix_policy_output_depth_gain            1.00
rational_matrix_policy_muon_strength                0.75
rational_matrix_policy_muon_lr_scale                1.00
rational_matrix_policy_max_muon                     0.75
rational_matrix_policy_min_muon                     0.00
rational_matrix_policy_final_muon                   0.00
rational_matrix_policy_start                        0.02
rational_matrix_policy_end                          0.12
rational_matrix_policy_decay_start                  0.20
rational_matrix_policy_decay_end                    0.36
rational_matrix_policy_muon_decay_depth_shift       0.00
rational_matrix_policy_muon_input_decay_shift       0.00
rational_matrix_policy_muon_output_decay_shift      0.00
rational_matrix_policy_muon_reset_adam_state        false
rational_matrix_policy_pressure_weight              0.30
rational_matrix_policy_activity_weight              0.65
rational_matrix_policy_weight_decay_scale           1.00
rational_matrix_policy_function_coeff               false
rational_matrix_policy_group_gain_strength          0.00
rational_matrix_policy_group_pressure_strength      0.00
rational_matrix_policy_group_activity_damping       0.00
rational_transport_strength                         0.00
rational_transport_matrix_strength                  0.00
rational_transport_quotient_strength                0.00
rational_adaptive_coeff_strength                    0.00
rlb_gauge_strength                                  0.50
rlb_gauge_start                                     0.03
rlb_gauge_end                                       0.35
rlb_gauge_every                                     5
```

The optimizer is `rational_matrix_policy_onpolicy` on `rlb_fused_fixed_strong_ffn`. It uses AdamW for the non-RLB backbone and rational coefficients, and `RationalMatrixPolicyOptimizer` for RLB `W_in/W_out` matrices. Only those RLB matrices receive the short early Muon switch.

## Current Result

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU+AdamW | 3.621982 | 37.41 |
| SiLU+Muon | 3.644921 | 38.28 |
| RLB+Muon | 3.657877 | 38.78 |

The result is real but not yet the requested large gap: `0.0731` loss / `2.45` PPL versus `SiLU+AdamW beta2=0.999`.

## Latest Probe Readout

| probe | last step | loss | readout |
| --- | ---: | ---: | --- |
| A6000 matched default | 1250 | 4.052293 | matched fallback screen |
| beta2 tail 0.995 | 1250 | 4.049556 | tiny +0.002738 vs matched default, not close to old best short curve |
| group policy 0.30 | 1000 | 4.141706 | neutral/worse vs matched default at 1000 |
| late Muon 0.05 | 500 | 4.673611 | worse than matched default at 500 |
| layer statgate | 250 | 5.369072 | tied with matched default |
| statgate+group 0.18 | 750 | 4.331103 | tiny +0.000628 vs matched default, noise-level |

Plain Muon is worse than AdamW controls, and RLB+Muon is worse than SiLU+Muon. Synthetic arithmetic transfers early speed but not final loss.

## Artifact Policy

Commit compact summaries and plots under:

```text
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/
```

Do not commit raw local run folders or Slurm logs under `experiments/runs/`.
