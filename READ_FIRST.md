# Read First

This repo is for designing an RLB-specific optimizer. Do not claim wins from a different global LR schedule.

## GPU Limit

Use at most 4 GPUs total.

```text
--gres=gpu:nvidia_rtx_6000_ada_generation:4
--nproc_per_node=4
```

Check active jobs before launching:

```bash
squeue -u mt872
```

## Exact Current Optimizer

The names below match the JSONL config keys where possible. `optimizer_lr` and `optimizer_min_lr` are the values passed by `--lr` and `--min-lr`.

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
rational_matrix_policy_adam_role_strength           1.20
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

The optimizer is `rational_matrix_policy_onpolicy` on `rlb_fused_fixed_strong_ffn`. It uses ordinary AdamW for the non-RLB backbone and rational coefficients, and `RationalMatrixPolicyOptimizer` for RLB `W_in/W_out` matrices. Only those RLB matrices receive the early Muon switch.

## Current Result

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU+AdamW | 3.621982 | 37.41 |

The current best clears 2-3 PPL versus beta2-tuned controls, but does not yet clear the requested 0.2-0.3 final loss gap. Do not run or report LR ablations as the main story until the same-LR loss gap is much larger.

## Operational Notes

```text
1. backward pass computes gradients
2. outer on-policy wrapper updates live RLB statistics
3. transport, coefficient-function optimizer, and group-gradient policy are off by default
4. ordinary AdamW steps the non-RLB backbone, no-decay group, and rational coefficients
5. MatrixPolicy handles RLB W_in/W_out matrices:
   - compute role/depth AdamW scale
   - compute early Muon fraction
   - step AdamW with lr = global_lr * adam_scale * (1 - muon_fraction)
   - step Muon with lr = global_lr * muon_lr_scale * muon_fraction
   - restore scheduler-provided base group lrs
6. outer wrapper applies exact RLB gauge balance
```

## Artifact Policy

Commit compact current summaries and plots under:

```text
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/
```

Do not commit raw local probe folders under:

```text
experiments/runs/wikitext103/
```
