# Optimizer Design

This folder contains rational-specific optimizer implementations. The current research optimizer is `RationalMatrixPolicyOptimizer`, used through the `rational_matrix_policy_onpolicy` training option.

## MatrixPolicy In Plain Terms

`RationalMatrixPolicyOptimizer` is the optimizer for the two matrices around each RLB activation. It is not a separate optimizer for the whole Transformer. For RLB layer `l`, the layer is

```text
x -> A_l x -> grouped rational functions -> B_l h
```

where `A_l = W_in,l` chooses what each rational group sees and `B_l = W_out,l` mixes the rational features back into the residual stream. MatrixPolicy updates these two roles differently because they have different mathematical jobs.

In the default verified setup:

```text
ordinary Transformer weights -> AdamW
rational coefficients        -> AdamW/function-space coefficient optimizer when enabled
RLB A_l and B_l matrices     -> MatrixPolicy
RLB group scales             -> exact gauge rebalance after the step
```

The word `policy` means a deterministic rule that selects the local matrix update from metadata and live measurements:

```text
matrix role:     input matrix A_l or output matrix B_l
layer depth:     early, middle, or late Transformer layer
training time:   early Muon component on/off/decay window
RLB statistics:  group pressure, output activity, coefficient activity
```

The optimizer does not change the global LR schedule to win. It uses the same base LR `eta_t` as the controls, then applies RLB-local matrix multipliers and an RLB-local AdamW/Muon mixture:

```text
A_l <- AdamW-style step for an input selector
     + short early Muon-style matrix step

B_l <- AdamW-style step for an output recombiner
     + short early Muon-style matrix step
```

After these child updates, the wrapper applies a function-preserving gauge rebalance:

```text
A_{l,g} <- s_g A_{l,g}
B_{l,g} <- B_{l,g} / s_g
```

This last transform changes the parameterization but not the represented RLB function. That is the main reason MatrixPolicy is rational-specific: it can use the explicit `A_l -> rational groups -> B_l` factorization and its gauge symmetry, while a standard SiLU/SwiGLU FFN optimizer cannot use this exact structure.

## Notation

For RLB layer `l`, let the hidden channels be split into `G` groups of width `m`. Write

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

The learned rational function `R_{l,g}` contains the base rational polynomial and local basis coefficients. The optimizer sees three qualitatively different parameter roles:

```text
A_l: domain selector for rational group inputs
R_{l,g}: nonlinear function shape
B_l: feature recombination back into the residual stream
```

## Exact Gauge Symmetry

For any `a_g > 0`, define block diagonal `D_l(a)` with block `a_g I_m`. Then

```text
A_l' = D_l(a) A_l
B_l' = B_l D_l(a)^(-1)
```

preserves the represented RLB layer. Because `a_g > 0`, `u_g` is unchanged, `r_g` scales by `a_g`, `h_g` scales by `a_g`, and `B_l` cancels that scale. This symmetry is absent from a standard SiLU/SwiGLU FFN in the same explicit grouped rational form.

The optimizer objective is therefore not merely to reduce parameter loss. It is to choose updates that move the represented function while avoiding unnecessary motion along the gauge direction.

## Parameter Partition

`rational_matrix_policy_onpolicy` partitions trainable parameters into:

```text
theta_backbone = ordinary Transformer weights, norms, embeddings
theta_coeff    = rational numerator/denominator/local-basis parameters
M_l,in         = A_l
M_l,out        = B_l
```

The default MatrixPolicy run updates these with different child optimizers:

```text
theta_backbone -> AdamW, or optional Muon for eligible 2D backbone matrices
theta_coeff    -> AdamW/function-space coefficient optimizer when enabled
M_l,in/out     -> RationalMatrixPolicyOptimizer
```

The child optimizers are wrapped by `RationalTransportOnPolicyOptimizer`, which measures live RLB pressure before the step and applies gauge rebalance after the step.

## Role And Depth Policy

Let training progress be `tau = t / T` and normalized layer depth be

```text
d_l = l / (L - 1)
```

for `L > 1`. MatrixPolicy assigns a role-depth factor

```text
rho(l, in)  = clip(1 + gamma_in  (d_l - 0.5), 0.55, 1.40)
rho(l, out) = clip(1 + gamma_out (d_l - 0.5), 0.55, 1.40)
```

with current defaults

```text
gamma_in  = -0.50
gamma_out =  1.00
```

Thus earlier layers receive relatively more input-matrix movement and later layers receive relatively more output-matrix movement. The Adam role multiplier is

```text
a_role(l,r) = max(0.10, 1 + alpha_role (rho(l,r) - 1))
```

where `r in {in,out}` and the default `alpha_role = 1.20`.

## AdamW-Muon Matrix Update

For an RLB matrix group `M_{l,r}`, MatrixPolicy computes a Muon fraction

```text
mu(l,r,t) = clip(mu_base(t) rho(l,r) s_stat(l,t), mu_min, mu_max)
```

where `mu_base(t)` is a smooth early window: it turns on from progress `0.02` to `0.12`, decays from `0.20` to `0.36`, and has default peak strength `0.75`. The statistic term `s_stat` penalizes large input/output pressure imbalance and excessive rational-coefficient activity.

The effective matrix step is the sum of an AdamW step and a Muon step on the same matrix:

```text
M_{l,r} <- M_{l,r}
          + Delta_AdamW(M_{l,r}; eta_t a_mat(l,r,t) [1 - mu(l,r,t)])
          + Delta_Muon (M_{l,r}; eta_t a_muon mu(l,r,t))
```

with

```text
a_mat(l,r,t) = clip(a0(t) a_role(l,r) a_stat(l,r,t), a_min, a_max)
```

Default values are:

```text
a0 = 3.0
a_min = 0.40
a_max = 4.0
a_muon = 1.0
mu_min = 0.0
mu_max = 0.75
```

The base LR `eta_t` is the same scheduler used by the controls. MatrixPolicy changes only local RLB matrix multipliers and the matrix update rule mixture.

## Empirical Behavior So Far

The May 29 dense synthetic and gauge-stress runs clarify what MatrixPolicy is currently doing. It is best understood as an early/mid training accelerator for RLB matrices, not yet as a solved final-loss optimizer.

Observed pattern:

```text
RLB+AdamW                 faster than SiLU+AdamW early on synthetic tasks
RLB MatrixPolicy          faster than RLB+AdamW early/mid
RLB MatrixPolicy group-stat  similar early speed, sometimes better late retention
RLB+Muon                  weak early curve on these RLB synthetic runs
```

The strongest signal is mean validation loss AUC through step 200. On the dense synthetic run, MatrixPolicy improves over generic `RLB+AdamW` on Code (`2.1462` vs `2.4336`), Symbolic (`1.6594` vs `2.0576`), and Reasoning mix (`2.7143` vs `3.1170`). On Reasoning mix, the group-stat variant also gives the best final row (`0.142429` loss / `1.1531` PPL), but Code and Symbolic are too saturated for final loss to carry much meaning.

The gauge-stress run should be read carefully. Gauge log scale `2.0` often improves early AUC for every optimizer, so it is not a clean degradation proof. MatrixPolicy remains the fastest early curve under both gauge settings, but a stronger mechanism claim needs multiple gauge seeds/scales plus direct measurement of gauge drift and function-space movement.

Design implication: the useful part of MatrixPolicy is the role-aware matrix update. The weak part is late retention. Future optimizer work should focus on preserving the early function-space gain rather than increasing global LR or adding another scheduler.

## On-Policy Statistics

Before the child optimizers step, the wrapper records per-layer, per-group relative gradient pressures:

```text
p_in,g  = rms(grad A_{l,g}) / rms(A_{l,g})
p_out,g = rms(grad B_{l,g}) / rms(B_{l,g})
p_rat,g = rms(rational coefficient gradients for group g)
```

Their EMAs are stored as

```text
q_in,g, q_out,g, q_rat,g
```

The matrix policy uses two derived quantities:

```text
pressure_g = log q_in,g - log q_out,g
activity_g = log q_rat,g - 0.5 (log q_in,g + log q_out,g)
```

`pressure_g` describes whether the input or output side of a rational group is receiving more relative update pressure. `activity_g` describes whether rational shape parameters are moving more than the surrounding matrices.

## Group-Stat Gradient Scaling

The group-stat variant applies a per-group gradient multiplier before AdamW/Muon. For each role `r`, construct an unnormalized scale

```text
c_g = c_gain,g c_pressure,g c_activity,g
```

where the gain term uses live RLB statistics:

```text
c_gain,g = (geomean(k) / k_g)^alpha
k_g = derivative_rms_g for W_in
k_g = output_rms_g     for W_out
```

The pressure and activity terms are

```text
c_pressure,g = exp(beta pressure_direction_g)
pressure_direction_g = -pressure_g for W_in, +pressure_g for W_out

c_activity,g = exp(-lambda relu((activity_g - activity_target) / activity_width))
```

Finally the scale is centered and clipped:

```text
c_g <- c_g / geomean(c)
c_g <- clip(c_g, c_min, c_max)
```

`W_in` gradients are multiplied group-row-wise by `c_g`; `W_out` gradients are multiplied group-column-wise by `c_g`. This is a local matrix preconditioner, not a global scheduler.

## Gauge Rebalance

After child optimizers step, the on-policy wrapper applies a function-preserving gauge correction every `k` steps. Let

```text
n_in,g  = rms(A_{l,g})
n_out,g = rms(B_{l,g})
current_g = log n_in,g - log n_out,g
```

A target log-ratio combines rational curve metrics and live gradient pressure:

```text
target_g = 0.5 (log derivative_gain_g - log output_gain_g)
           + beta_pressure (log q_in,g - log q_out,g)
```

The gauge step is

```text
ell_g = 0.5 (target_g - current_g)
ell_g <- clip(schedule(t,l) activity_gain_g ell_g, -ell_max, ell_max)
s_g = exp(ell_g)

A_{l,g} <- s_g A_{l,g}
B_{l,g} <- B_{l,g} / s_g
```

This preserves the represented layer function exactly up to floating-point error while choosing a better-conditioned representative in the RLB gauge class.

## Full Step Order

For `rational_matrix_policy_onpolicy`, one optimizer step is:

```text
1. update live on-policy pressure EMAs q_in, q_out, q_rat
2. optionally project matrix gradients away from pure gauge directions
3. optionally precondition rational coefficient gradients
4. optionally precondition matrix gradients from rational output/derivative gains
5. step the child optimizers: backbone optimizer, coefficient optimizer if enabled, and MatrixPolicy for A_l/B_l
6. apply function-preserving W_in/W_out gauge rebalance
7. optionally apply rational-curve amplitude transport when enabled
```

For the current MatrixPolicy-Muon result, the important active part is the role/depth AdamW-Muon matrix policy plus the on-policy gauge rebalance. More aggressive coefficient and transport options are ablations unless explicitly enabled in the run config.

## Evaluation Standard

A new optimizer design should be judged by:

```text
train/validation AUC
time to loss threshold
final loss and PPL
positive-gauge stress degradation
function movement per parameter movement
group activity, derivative pressure, and denominator margin
```

The decisive mechanism test is positive-gauge stress. If an optimizer does not degrade less than generic `RLB+AdamW` and `RLB+Muon` under an equivalent-function gauge reparameterization, then it is not yet exploiting the central RLB symmetry.
