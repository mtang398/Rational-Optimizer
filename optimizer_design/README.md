# Optimizer Design

This directory contains the rational-specific optimizer implementations plus self-contained broad optimizer baselines for the language-model harness. The current research optimizer is `RationalMatrixPolicyOptimizer`, exposed in training as `rational_matrix_policy_onpolicy`.

The rejected V2-V12 branches have been removed from the live optimizer surface and raw run tree. Their only retained state is the compact failure log in `proposals/matrixpolicy_variant_failures.md`. The current paper anchor remains original `rational_matrix_policy_onpolicy`; no live Vx alias is active.

## Current Result Anchor

The current paper-facing optimizer evidence is the completed E1 matched main suite plus the completed E2 DCLM M0/300M cell. In E2 DCLM, `rational_matrix_policy_onpolicy` reaches final validation loss `3.957627 +/- 0.030713`, ahead of the next aggregate methods around `3.9934`, and the token-to-target savings table is tracked in `../experiments/results/iclr26_e2_dclm_2026_06_10/README.md`.

## Problem Setup

For RLB layer `l`, write the Transformer MLP sublayer as:

```text
x -> A_l x -> grouped rational functions -> B_l h
```

where:

```text
A_l = W_in,l
B_l = W_out,l
z = A_l x
z_g = group_g(z)
r_g = sqrt(mean(z_g^2) + eps)
u_g = z_g / r_g
h_g = r_g R_{l,g}(u_g)
y = B_l concat_g(h_g)
```

The roles are not interchangeable:

```text
A_l       selects the input domains for rational groups
R_{l,g}  changes the nonlinear rational curve
B_l       recombines rational features into the residual stream
```

MatrixPolicy exists because a generic optimizer treats these roles too uniformly.

## Gauge Symmetry

For any positive group scale `a_g > 0`, define a block diagonal matrix `D_l(a)` with block `a_g I_m`. Then:

```text
A_l' = D_l(a) A_l
B_l' = B_l D_l(a)^(-1)
```

In the homogeneous radius, the represented RLB function is unchanged. The normalized coordinate `u_g` is unchanged, `r_g` and `h_g` scale by `a_g`, and `B_l` cancels that scale. With the stabilizing RMS floor, this is the same local scale structure up to the floor-induced discrepancy measured by the matched run behavior.

The optimizer goal is therefore not merely to move parameters. It should move the represented function while controlling arbitrary scale drift in the RLB matrix representative.

## Parameter Partition

`rational_matrix_policy_onpolicy` partitions parameters as:

```text
theta_backbone = embeddings, attention, norms, and ordinary Transformer weights
theta_coeff    = rational numerator, denominator, and local-basis coefficients
M_l,in         = A_l
M_l,out        = B_l
```

Current real-corpus wiring:

```text
theta_backbone -> AdamW
theta_coeff    -> AdamW-style coefficient updates inside the wrapper
M_l,in/out     -> RationalMatrixPolicyOptimizer
RLB gauge      -> exact post-step rebalance
```

## Role And Depth Policy

Let normalized depth be:

```text
d_l = (l - 1) / (L - 1)
```

MatrixPolicy assigns different depth factors to input selectors and output recombiners:

```text
rho_in(l)  = clip(1 - 0.50 (d_l - 0.5), 0.55, 1.40)
rho_out(l) = clip(1 + 1.00 (d_l - 0.5), 0.55, 1.40)
```

The AdamW role multiplier is:

```text
a_role(l,r) = max(0.10, 1 + alpha_role (rho(l,r) - 1))
alpha_role = 1.20
```

This biases earlier layers toward input-domain selection and later layers toward output recombination. It is a local RLB matrix policy, not a global LR schedule.

## Matrix Update Rule

For RLB matrix role `r in {in,out}`, MatrixPolicy computes a Muon mixture fraction:

```text
mu(l,r,t) = clip(mu_base(t) rho(l,r) s_stat(l,t), 0.0, 0.75)
```

`mu_base(t)` is an early window:

```text
start = 0.02
full strength by = 0.12
decay start = 0.20
decay end = 0.36
peak strength = 0.75
```

The effective matrix step is:

```text
M_{l,r} <- M_{l,r}
          + Delta_AdamW(M_{l,r}; eta_t a_mat(l,r,t) [1 - mu(l,r,t)])
          + Delta_Muon (M_{l,r}; eta_t a_muon mu(l,r,t))
```

where:

```text
a_mat(l,r,t) = clip(3.0 a_role(l,r) a_stat(l,r,t), 0.40, 4.0)
a_muon = 1.0
eta_t = same base LR schedule used by controls
```

The Muon component is deliberately early. The method is not "use Muon everywhere"; generic Muon underperforms in the current control set.

## On-Policy Statistics And Group Scaling

Before stepping, the wrapper records per-layer, per-group pressures:

```text
p_in,g  = rms(grad A_{l,g}) / rms(A_{l,g})
p_out,g = rms(grad B_{l,g}) / rms(B_{l,g})
p_rat,g = rms(rational coefficient gradients for group g)
```

The group-stat variant applies a centered, clipped per-group multiplier before matrix updates:

```text
c_g = c_gain,g c_pressure,g c_activity,g
c_g <- c_g / geomean(c)
c_g <- clip(c_g, 0.75, 1.35)
```

Current best real-corpus settings:

```text
group_gain_strength = 0.20
group_pressure_strength = 0.10
group_activity_damping = 0.20
group window = progress 0.02 to 0.30
group scale clip = [0.75, 1.35]
```

## Gauge Rebalance

After child optimizer steps, the wrapper applies a function-preserving gauge correction. Let:

```text
n_in,g  = rms(A_{l,g})
n_out,g = rms(B_{l,g})
current_g = log n_in,g - log n_out,g
```

The applied correction has the form:

```text
ell_g = 0.5 (target_g - current_g)
ell_g <- clip(schedule(t,l) activity_gain_g ell_g, -ell_max, ell_max)
s_g = exp(ell_g)

A_{l,g} <- s_g A_{l,g}
B_{l,g} <- B_{l,g} / s_g
```

With the homogeneous RLB radius and inactive floors, this changes only the parameterization.
With the stabilized radius and clipped bounded moves used in training, the same operation is
treated as a bounded move along the positive scale gauge, with any floor-induced discrepancy
handled empirically by the matched run curves.

## Generic Baseline Optimizers

`baseline_optimizers.py` adds the broad optimizer-family controls needed for matched paper comparisons:

```text
lion                  -> Lion with decoupled weight decay
ademamix              -> paper-style AdEMAMix: fast AdamW EMA plus uncorrected slow EMA, alpha warmup, beta3 half-life warmup
schedule_free_adamw   -> schedule-free-style AdamW/Polyak interpolation baseline
adafactor_came        -> factored adaptive AdamW with CAME-style confidence correction
soap_adamw            -> SOAP/Shampoo-style eigenbasis AdamW for eligible 2D tensors
```

AdEMAMix now follows the paper/reference implementation details that matter for stability: the slow EMA is not bias-corrected, alpha warms linearly, and beta3 warms by interpolating EMA half-life. The SOAP/Shampoo and CAME rows are intentionally labeled as style baselines until they are matched line-by-line to a reference implementation. They are suitable for matched experiment rows and stability checks, not for claiming exact reproduction of those papers.

## Full Step Order

```text
1. record live RLB pressure/activity statistics
2. optionally precondition RLB matrix gradients with group-stat scaling
3. step backbone AdamW parameters
4. step rational coefficient parameters
5. step RLB matrices with the role/depth AdamW-Muon matrix policy
6. apply bounded W_in/W_out gauge rebalance
```

Retained source files are limited to the current training surface: `matrix_policy_optimizer.py` for RLB matrices, `transport_onpolicy_optimizer.py` for private MatrixPolicy wrapper mechanics, `function_space_rational_optimizer.py` for optional coefficient updates, and `baseline_optimizers.py` for matched broad optimizer controls. The on-policy balance, matrix-metric, and adaptive-stat code is private support inside the wrapper, not a separate optimizer surface.

## Empirical Readout

Current E1 M0/100M manifest-suite readout, with the MatrixPolicy row replaced by the completed safe-speed rerun:

| Dataset | MatrixPolicy final val loss | next best current method | gap |
| --- | ---: | ---: | ---: |
| DCLM | 4.256989 +/- 0.004197 | rlb_lion 4.305728 +/- 0.005836 | 0.048739 |
| FineWeb-Edu | 4.088287 +/- 0.009169 | rlb_lion 4.142669 +/- 0.006812 | 0.054382 |
| FineWeb | 4.319472 +/- 0.012370 | rlb_lion 4.367062 +/- 0.007532 | 0.047590 |
| Dolma-sample | 4.323933 +/- 0.005168 | rlb_lion 4.369254 +/- 0.005561 | 0.045321 |
| C4 | 4.286446 +/- 0.019324 | rlb_lion 4.335663 +/- 0.020917 | 0.049217 |

Full mean +/- std tables and curves are in `../experiments/ICLR_RUN_STATUS.md` and `../experiments/results/iclr26_e1_figures/`.

The paper story should be: RLB creates optimizer-visible geometry, and MatrixPolicy uses it. It should not be sold as an activation-only result or a generic Muon result.

## Rejected Variant Log

Rejected V2-V12 method attempts are consolidated in `proposals/matrixpolicy_variant_failures.md`. Their source hooks, standalone manifests, proposal files, and raw run directories are no longer part of the active repo state.

## Next Proposal Direction

The current implementation includes a method-preserving speed fix: once every MatrixPolicy Muon group has passed its decay end and `final_muon=min_muon=0`, `RationalMatrixPolicyOptimizer` skips the otherwise zero-LR Muon step. P0 validated this as quality-neutral with lower optimizer-step overhead. The full `E1_matrixpolicy_safe_speed_100m` rerun completed all 15 rows on 2026-06-23: final losses match the original MatrixPolicy E1 table within seed/dataset noise, and the clean harness runtime aggregate improved to `27.3` min, `0.5102` s/step, and `67,078.3` tokens/s over 15 rows. This is the paper-facing implementation of the original MatrixPolicy, not a new Vx method.

The next method candidate should be a MatrixPolicy rule that is explainable from existing optimizer/RLB quantities and does not add expensive hooks, extra forward/backward passes, or matrix snapshots. It must pass a fast paired pilot before any E1 expansion.

## Telemetry Status

`RationalMatrixPolicyOptimizer` now exposes log-step telemetry for:

```text
per-role Muon mixture
per-role Adam LR scale
per-role update RMS, weight RMS, and update/weight RMS
group policy scale mean/std/min/max
on-policy pressure and activity mean/std
```

The training harness also logs RLB gauge/rational/denominator metrics and fixed-probe function movement. These fields are implemented for paper diagnostics; CUDA/DDP validation and paper result figures are still pending.

## Evaluation Standard

A new optimizer variant should be judged by:

```text
final heldout loss and PPL
validation loss AUC at early, mid, and full horizons
step-matched train and validation curves from step 1
comparison to SiLU+AdamW, RLB+AdamW, SiLU+Muon, and RLB+Muon
comparison to tuned SOAP/Shampoo-style, Lion, AdEMAMix, Schedule-Free AdamW, and Adafactor/CAME baselines where stable
wall-clock/tokens-to-target, because MatrixPolicy has per-step overhead
function movement per parameter movement
gauge drift and W_in/W_out norm-product diagnostics
denominator margin and rational coefficient activity
```
