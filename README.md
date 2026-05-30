# RationalOPT

RationalOPT studies whether a no-GLU Rational Local Basis FFN can train better than SiLU/SwiGLU when the optimizer uses rational-specific geometry. The comparison is deliberately same-protocol: same model size, token budget, seed, batch shape, base LR schedule, weight decay, dataset slice, and evaluation cadence.

The current best optimizer is `rational_matrix_policy_onpolicy`, run as `RLB+MatrixPolicy (group-stat)` on the real-corpus screen. It is not an LR trick. It keeps the same base LR schedule as the controls and changes only the local update rule for RLB matrix roles plus an exact RLB gauge rebalance.

## Current Evidence

The strongest current evidence is the May 30 real-corpus screen. It uses a 123.6M-parameter GPT-style model, GPT-2 tokenizer, 100M training tokens, 4M heldout tokens after a 110M-token stream offset, 32,768 tokens/step, seed `1337`, and the same base LR schedule across rows.

### FineWeb

| method | last finite validation loss | last finite PPL | val loss AUC <= 1000 | val loss AUC <= 2000 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| SiLU+AdamW | 4.504617 | 90.43 | 5.993426 | 5.401559 | complete |
| RLB+AdamW | 4.493013 | 89.39 | 5.954484 | 5.373016 | complete |
| SiLU+Muon | 4.535766 | 93.29 | 6.664512 | 5.786310 | complete |
| RLB+Muon | 4.548868 | 94.53 | 6.585091 | 5.752002 | complete |
| RLB+MatrixPolicy (group-stat) | 4.344150 | 77.03 | 5.850945 | 5.262783 | complete |

Main gap: `RLB+MatrixPolicy (group-stat)` beats `SiLU+AdamW` by `0.160467` validation loss and `13.40` PPL. It beats `RLB+AdamW` by `0.148863` validation loss and `12.36` PPL.

### FineWeb-Edu

| method | last finite validation loss | last finite PPL | val loss AUC <= 1000 | val loss AUC <= 2000 | note |
| --- | ---: | ---: | ---: | ---: | --- |
| SiLU+AdamW | 4.225019 | 68.38 | 5.835354 | 5.186270 | complete |
| RLB+AdamW | 8.411884 | 4500.23 | 9.684973 | 9.684973 | train nonfinite at step 80; validation nonfinite at step 100 |
| SiLU+Muon | 4.252612 | 70.29 | 6.505154 | 5.563970 | complete |
| RLB+Muon | 4.271556 | 71.63 | 6.425483 | 5.529865 | complete |
| RLB+MatrixPolicy (group-stat) | 4.072055 | 58.68 | 5.670071 | 5.041694 | complete |

Main gap: `RLB+MatrixPolicy (group-stat)` beats `SiLU+AdamW` by `0.152964` validation loss and `9.70` PPL. The plain `RLB+AdamW` row is unstable on this task, so the result is not just an activation win.

### WikiText-103 Anchor

WikiText-103 remains useful as an older real-LM anchor, but it is not the main current claim because the gap is smaller.

| method | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU/SwiGLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU/SwiGLU+AdamW | 3.621982 | 37.41 |
| SiLU/SwiGLU+Muon | 3.644921 | 38.28 |
| RLB+Muon | 3.657877 | 38.78 |

The WikiText gap versus the strongest `SiLU/SwiGLU+AdamW` row is `0.073114` loss and `2.45` PPL. The real-corpus FineWeb and FineWeb-Edu gaps are larger.

## RLB Layer

RLB replaces the standard FFN nonlinearity with grouped normalized rational functions. For hidden groups `g = 1..G` with group width `m`:

```text
z = W_in x
z_g = group_g(z)
r_g = sqrt(mean(z_g^2) + eps)
u_g = z_g / r_g
h_g = r_g R_g(u_g)
y = W_out concat_g(h_g)
```

`R_g` is learned from rational coefficients and local basis terms. There is no GLU gate, no hidden SiLU branch, and no value/gate split.

The useful symmetry is a positive group gauge. For any `a_g > 0`:

```text
W_in[g]  <- a_g W_in[g]
W_out[g] <- W_out[g] / a_g
```

The represented function is unchanged because `u_g` is unchanged, `r_g` and `h_g` scale by `a_g`, and `W_out[g]` cancels the scale. Generic optimizers still see different matrix norms and conditioning. MatrixPolicy tries to update the represented function while controlling this gauge degree of freedom.

## MatrixPolicy Optimizer

The exact real-corpus optimizer row is:

```text
activation:  rlb_fused_fixed_strong_ffn
optimizer:   rational_matrix_policy_onpolicy
variant:     MatrixPolicy with group-stat scaling
backbone:    AdamW
base LR:     optimizer_lr=3e-4, optimizer_min_lr=3e-5
beta2:       MatrixPolicy beta2=0.999, backbone beta2=0.999
```

Parameter partition:

```text
theta_backbone -> AdamW
rational coefficients -> AdamW-style coefficient updates in the wrapper
A_l = W_in,l -> RationalMatrixPolicyOptimizer, role=in
B_l = W_out,l -> RationalMatrixPolicyOptimizer, role=out
RLB group gauge -> exact post-step rebalance
```

For layer depth `d_l = l/(L-1)`, MatrixPolicy uses different role-depth factors for input selectors and output recombiners:

```text
rho_in(l)  = clip(1 - 0.50 (d_l - 0.5), 0.55, 1.40)
rho_out(l) = clip(1 + 1.00 (d_l - 0.5), 0.55, 1.40)
```

The RLB matrix step is a local AdamW/Muon mixture, not a global scheduler change:

```text
M_{l,r} <- M_{l,r}
          + Delta_AdamW(M_{l,r}; eta_t a_mat(l,r,t) [1 - mu(l,r,t)])
          + Delta_Muon (M_{l,r}; eta_t a_muon mu(l,r,t))
```

with current real-corpus defaults:

```text
a_mat base scale = 3.0, clipped to [0.40, 4.0]
a_muon = 1.0
muon strength peak = 0.75
muon active window = start 0.02, full by 0.12, decay 0.20 to 0.36
```

The group-stat variant multiplies RLB matrix gradients by centered, clipped per-group scales from derivative/output activity and pressure:

```text
group_gain_strength = 0.20
group_pressure_strength = 0.10
group_activity_damping = 0.20
group window = progress 0.02 to 0.30
group scale clip = [0.75, 1.35]
```

After child optimizer steps, the wrapper applies the exact gauge rebalance:

```text
A_{l,g} <- s_g A_{l,g}
B_{l,g} <- B_{l,g} / s_g
```

This is why the optimizer is RLB-specific: it acts on the explicit `W_in -> rational groups -> W_out` factorization and the gauge symmetry of the rational block.

## Figures

FineWeb validation loss:

![FineWeb validation loss](experiments/results/real_lm_screen_2026_05_30/fineweb_validation_loss.png)

FineWeb validation PPL:

![FineWeb validation PPL](experiments/results/real_lm_screen_2026_05_30/fineweb_validation_ppl.png)

FineWeb training loss:

![FineWeb training loss](experiments/results/real_lm_screen_2026_05_30/fineweb_training_loss.png)

FineWeb-Edu validation loss:

![FineWeb-Edu validation loss](experiments/results/real_lm_screen_2026_05_30/fineweb_edu_validation_loss.png)

FineWeb-Edu validation PPL:

![FineWeb-Edu validation PPL](experiments/results/real_lm_screen_2026_05_30/fineweb_edu_validation_ppl.png)

FineWeb-Edu training loss:

![FineWeb-Edu training loss](experiments/results/real_lm_screen_2026_05_30/fineweb_edu_training_loss.png)

WikiText-103 validation loss:

![WikiText validation loss](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

WikiText-103 validation PPL:

![WikiText validation PPL](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

WikiText-103 training loss from step 1:

![WikiText training loss from step 1](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

## Interpretation

What is working now:

- The optimizer matters more than the activation alone. FineWeb-Edu plain `RLB+AdamW` diverges, while `RLB+MatrixPolicy (group-stat)` is the best completed row.
- Generic Muon is a poor fit here. Both `SiLU+Muon` and `RLB+Muon` are worse than AdamW controls on the real-corpus screen.
- The role-aware RLB matrix policy appears to preserve the useful rational advantage into heldout loss on two real web corpora, not only on toy saturated tasks.
- The current gaps, around `0.153-0.160` validation loss versus `SiLU+AdamW`, are larger than the WikiText anchor but still slightly below the earlier target of `0.2-0.3` loss.

What is not claimed:

- The removed saturated synthetic tests are not part of the current public evidence.
- This is not yet a multi-seed claim.
- This is not yet a proof of gauge-invariant optimization; it is a strong empirical optimizer result that motivates direct function-space and gauge-drift diagnostics.

## Repository Map

```text
activation/         RLB activation implementation and math
optimizer_design/   MatrixPolicy optimizer definition
training/           LM harness, dataset streaming, optimizer wiring
experiments/        launchers, summarizers, committed result artifacts
```

Current compact result packages:

```text
experiments/results/real_lm_screen_2026_05_30/
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/
```

Raw run directories under `experiments/runs/` are local artifacts. The earlier saturated synthetic result bundles were removed from the tracked public evidence; WikiText and the real-corpus screen are kept.
