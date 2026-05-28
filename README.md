# RationalOPT

RationalOPT is an optimizer-design repo for the no-GLU Rational Local Basis FFN (RLB). The goal is an RLB-specific optimizer, not a different global learning-rate schedule. The hard comparisons are always:

```text
SiLU+AdamW
RLB+AdamW
SiLU+AdamW beta2=0.999
RLB+AdamW beta2=0.999
```

Jacobian, transport, MatrixPolicy, Muon switches, and other rational optimizers are candidate methods. They are not the baseline. LR ablations stay out of the main claim until the same-LR optimizer has a much larger final gap.

## Current Best

The current best same-LR optimizer is:

```text
optimizer:   rational_matrix_policy_onpolicy
activation:  rlb_fused_fixed_strong_ffn
mechanism:   MatrixPolicy-Muon switch
global LR:   lr=3e-4, min_lr=3e-5, warmup=200, cosine, unchanged
seed:        1337
steps:       3051
```

`rational_matrix_policy_onpolicy` now defaults to the best policy: RLB `W_in/W_out` matrices get a short early Muon phase, then switch back to the RLB MatrixPolicy AdamW update. The rest of the model keeps AdamW. The global LR schedule is unchanged.

## Exact Optimizer

The optimizer used in the headline row is exactly:

```text
rational_matrix_policy_onpolicy
```

Use it with:

```text
activation: rlb_fused_fixed_strong_ffn
```

It is not a standalone new activation and it is not an LR schedule. It is a parameter-group policy for the RLB FFN inside the same Transformer training loop.

### Parameter Groups

The training script splits parameters this way:

| parameter set | optimizer behavior |
| --- | --- |
| Non-RLB backbone weights | AdamW, same global LR schedule, beta2=0.999 in this branch |
| Norms, biases, tied embeddings | AdamW no-decay group |
| RLB `W_in` and `W_out` matrices | `RationalMatrixPolicyOptimizer`, with both AdamW and early Muon on the same matrix tensors |
| Rational coefficients | ordinary AdamW by default; function-space coefficient optimizer is off |

The Muon part is not used on the backbone. It is only used on the RLB `W_in/W_out` matrices during the early switch window.

### MatrixPolicy Formula

For an RLB matrix group at layer depth `d in [0, 1]`, MatrixPolicy computes a role/depth factor:

```text
role_factor = clamp(1 + gain(role) * (d - 0.5), 0.55, 1.40)
gain(in)    = -0.50
gain(out)   =  1.00
```

The AdamW side then uses:

```text
adam_scale = clamp(3.00 * (1 + 1.20 * (role_factor - 1)), 0.40, 4.00)
adam_lr    = global_lr * adam_scale * (1 - muon_fraction)
```

This means shallow input matrices, deep output matrices, and middle-layer matrices do not all receive the same effective matrix update. The policy is tied to the RLB decomposition: `W_in` forms the rational input domain, while `W_out` composes rational features back into the model stream.

### Muon Switch

Muon is blended only for RLB matrices and only early in training:

```text
muon_strength  = 0.75
muon_lr_scale  = 1.00
max_muon       = 0.75
start/end      = 0.02 -> 0.12 of total steps
decay/off      = 0.20 -> 0.36 of total steps
final_muon     = 0.00
```

For the 3051-step run, Muon ramps in during roughly steps 61-366, then fades out during roughly steps 610-1098. After that, RLB matrices are updated by MatrixPolicy AdamW only.

The actual per-group fraction is:

```text
on_phase      = smoothstep(0.02, 0.12, progress)
off_phase     = smoothstep(0.20, 0.36, progress)
muon_fraction = clamp(
  muon_strength * on_phase * (1 - off_phase)
  * role_factor
  * on_policy_stat_factor,
  0.0,
  0.75
)
```

The winning default keeps Adam moments through the switch:

```text
muon_reset_adam_state = false
```

Resetting Adam state after Muon was tested and was worse.

### Step Order

Each training step does this:

```text
1. regular backward pass computes gradients
2. outer on-policy wrapper updates live RLB stats
3. default transport, coefficient, and group-gradient policies are skipped
4. ordinary AdamW child optimizer steps the non-RLB backbone and rational coefficients
5. MatrixPolicy child optimizer handles RLB W_in/W_out matrices:
   - compute MatrixPolicy AdamW scale
   - compute early Muon fraction
   - set AdamW lr to global_lr * matrix_scale * (1 - muon_fraction)
   - set Muon lr to global_lr * muon_lr_scale * muon_fraction
   - step MatrixPolicy AdamW
   - step Muon while its fraction is nonzero
   - restore the original scheduler-provided group lrs
6. outer wrapper applies exact RLB gauge balance
```

The exact gauge balance is function-preserving. It changes the internal representative of each RLB group, not the represented FFN function.

| row | final loss | final PPL | loss gap vs best | PPL gap vs best |
| --- | ---: | ---: | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 | 0.000000 | 0.00 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 | 0.016978 | 0.55 |
| SiLU+AdamW beta2=0.999 | 3.549346 | 34.79 | 0.073114 | 2.45 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 | 0.073786 | 2.48 |
| RLB+AdamW | 3.617501 | 37.24 | 0.141269 | 4.91 |
| SiLU+AdamW | 3.621982 | 37.41 | 0.145750 | 5.07 |

The final PPL gap now clears 2-3 PPL versus beta2-tuned AdamW controls, but the final loss gap is still below the requested 0.2-0.3. The current result is progress, not a finished research target.

## Plots

Validation begins at the first eval point, step 250. Training loss begins at step 1.

![Same-LR validation loss](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

![Same-LR validation PPL](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

![Same-LR training loss from step 1](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

## Why This Helped

RLB factorizes the FFN into domain-forming matrices, per-group rational curves, and output composition:

```text
v = x W_in
u_g = v_g / rms(v_g)
h_g = rms(v_g) R_g(u_g)
y = h W_out
```

The useful optimizer behavior came from using that structure directly:

| mechanism | effect |
| --- | --- |
| Separate RLB `W_in/W_out` groups | lets rational matrices use a different update than the backbone |
| Layer/side MatrixPolicy | gives input/domain and output/composition matrices different scaling |
| beta2=0.999 in the MatrixPolicy branch | stabilizes the larger RLB matrix policy |
| Early matrix-only Muon phase | improves the trajectory before the late-penalty pattern appears |
| Switch back to MatrixPolicy AdamW | avoids making Muon the late optimizer |
| Exact RLB gauge balance | preserves represented function while conditioning matrix representatives |

Current promoted MatrixPolicy settings:

```text
adam_lr_scale                         3.00
adam_role_strength                    1.20
input_depth_gain                     -0.50
output_depth_gain                     1.00
rational_matrix_policy_beta2          0.999
rational_matrix_policy_backbone_beta2 0.999
muon_strength                         0.75
muon_lr_scale                         1.00
muon window                           start 0.02, end 0.12, decay 0.20-0.36
muon_reset_adam_state                 false
transport_strength                    0.00
```

## What Got Worse

| probe | result |
| --- | --- |
| Smooth-MatrixPolicy without Muon | strong, but final loss was 0.016978 worse |
| Muon strength 0.55 | weaker full final, 3.482822 |
| Muon lr-scale 1.10 | close but worse full final, 3.477492 |
| Muon lr-scale 1.25 | best short screen, worse full final, 3.479543 |
| Post-Muon Adam reset | worse than keeping moments at the winning strength |
| Global earlier Muon shutoff | lost the useful 750-step gain |
| On-policy Muon damping | worsened the 1000-1250 region |
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
