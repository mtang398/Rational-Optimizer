# RationalOPT

RationalOPT is an optimizer-design repo for the no-GLU Rational Local Basis FFN (RLB). The current work is about an RLB-specific optimizer, not changing the global learning-rate schedule.

The hard controls are always:

```text
SiLU+AdamW
RLB+AdamW
SiLU+AdamW beta2=0.999
RLB+AdamW beta2=0.999
```

Jacobian, transport, MatrixPolicy, Muon switches, and other rational optimizers are candidate methods. They are not baselines. LR ablations stay out of the main claim until the same-LR optimizer has a much larger final gap.

## Exact Optimizer

The headline optimizer is exactly:

```text
rational_matrix_policy_onpolicy
```

Use it with:

```text
activation: rlb_fused_fixed_strong_ffn
```

It is not a new activation and it is not an LR schedule. It is a parameter-group optimizer for RLB FFN matrices inside the same Transformer training loop.

## Current Best

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

Result: `3.476232` validation loss and `32.34` PPL at step 3051, seed 1337.

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU+AdamW | 3.621982 | 37.41 |

The final PPL gap clears 2-3 PPL versus beta2-tuned AdamW controls, but the final loss gap is still below the requested 0.2-0.3. This is progress, not the finished research target.

## What The Optimizer Touches

| parameter set | optimizer behavior |
| --- | --- |
| Non-RLB backbone weights | AdamW, same global LR schedule, beta2=0.999 in this branch |
| Norms, biases, tied embeddings | AdamW no-decay group |
| RLB `W_in` and `W_out` matrices | `RationalMatrixPolicyOptimizer`, with AdamW plus early Muon on the same matrix tensors |
| Rational coefficients | ordinary AdamW by default; function-space coefficient optimizer is off |

Muon is not used on the backbone. It is only blended into the RLB `W_in/W_out` matrix update during the early switch window.

## How It Works

RLB factorizes the FFN into domain-forming matrices, per-group rational curves, and output composition:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

`W_in` controls the rational input domain and derivatives. `W_out` composes rational features back into the model stream. MatrixPolicy uses this split by scaling input-side and output-side matrix updates differently by layer depth.

For an RLB matrix group at layer depth `d in [0, 1]`:

```text
role_factor = clamp(1 + gain(role) * (d - 0.5), 0.55, 1.40)
gain(in)    = -0.50
gain(out)   =  1.00

adam_scale = clamp(3.00 * (1 + 1.20 * (role_factor - 1)), 0.40, 4.00)
adam_lr    = global_lr * adam_scale * (1 - muon_fraction)
```

The early Muon fraction is:

```text
on_phase      = smoothstep(0.02, 0.12, progress)
off_phase     = smoothstep(0.20, 0.36, progress)
muon_fraction = clamp(
  0.75 * on_phase * (1 - off_phase)
  * role_factor
  * on_policy_stat_factor,
  0.0,
  0.75
)
muon_lr = global_lr * 1.00 * muon_fraction
```

For the 3051-step run, Muon ramps in around steps 61-366 and fades out around steps 610-1098. After that, RLB matrices are updated by MatrixPolicy AdamW only.

## Step Order

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

The exact gauge balance is function-preserving. It changes the internal representative of each RLB group, not the represented FFN function.

## Plots

Validation begins at the first eval point, step 250. Training loss begins at step 1.

![Same-LR validation loss](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

![Same-LR validation PPL](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

![Same-LR training loss from step 1](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

## What Got Worse

| probe | result |
| --- | --- |
| Smooth-MatrixPolicy without Muon | strong, but final loss was 0.016978 worse |
| Muon strength 0.55 | weaker full final, 3.482822 |
| Muon lr-scale 1.10 | close but worse full final, 3.477492 |
| Muon lr-scale 1.25 | best short screen, worse full final, 3.479543 |
| Post-Muon Adam reset | worse than keeping moments at the winning strength |
| Global earlier Muon shutoff | lost the useful 750-step gain |
| Stronger/extra on-policy Muon damping | worsened the 1000-1250 region |
| Layer/role Muon timing shifts | killed the useful input-side Muon contribution |
| Covariant Adam state under gauge | worse than matched short control |
| RLB matrix weight-decay reduction | worse than matched short control |

## Reproduce Current Best

No extra MatrixPolicy args are needed because the defaults now match the current best.

```bash
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
  RUN_NAME=rlb_matrix_policy_muon_switch \
  STEPS=3051 SEEDS=1337 \
  OPTIMIZERS=rational_matrix_policy_onpolicy \
  ACTIVATIONS=rlb_fused_fixed_strong_ffn \
  EVAL_INTERVAL=250 EVAL_BATCHES=20 LOG_INTERVAL=100 \
  sbatch --time=02:00:00 --gres=gpu:nvidia_rtx_6000_ada_generation:4 \
  training/run_wikitext103_optimizer_sweep.sbatch
```

## Layout

```text
activation/         rational activation package and CUDA extension
training/           WikiText-103 training, sweep, and aggregation scripts
optimizer_design/   RLB-specific optimizer components
experiments/        compact current result artifacts and ignored raw runs
```
