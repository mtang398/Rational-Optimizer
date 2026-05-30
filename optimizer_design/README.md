# Optimizer Design

This folder contains rational-specific optimizer implementations. The current research optimizer is `RationalMatrixPolicyOptimizer`, used through the training option `rational_matrix_policy_onpolicy`.

## Problem Setup

For RLB layer `l`, write the FFN block as:

```text
x -> A_l x -> grouped rational functions -> B_l h
```

with:

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

The three optimizer roles are different:

```text
A_l       selects the input domain seen by rational groups
R_{l,g}  changes the nonlinear rational curve
B_l       recombines rational features into the residual stream
```

MatrixPolicy exists because treating `A_l`, rational coefficients, and `B_l` as ordinary interchangeable parameters wastes structure.

## Exact Gauge Symmetry

For any positive group scale `a_g > 0`, define block diagonal `D_l(a)` with block `a_g I_m`. Then:

```text
A_l' = D_l(a) A_l
B_l' = B_l D_l(a)^(-1)
```

The represented RLB function is unchanged. The normalized coordinate `u_g` is unchanged, `r_g` and `h_g` scale by `a_g`, and `B_l` cancels that scale.

The optimizer should therefore avoid spending updates on arbitrary gauge drift. It should move the represented function and choose a well-conditioned representative of the same gauge class.

## Parameter Partition

`rational_matrix_policy_onpolicy` partitions parameters as:

```text
theta_backbone = embeddings, attention, norms, ordinary Transformer weights
theta_coeff    = rational numerator, denominator, and local-basis coefficients
M_l,in         = A_l
M_l,out        = B_l
```

The current real-corpus run uses:

```text
theta_backbone -> AdamW
theta_coeff    -> AdamW-style coefficient updates inside the wrapper
M_l,in/out     -> RationalMatrixPolicyOptimizer
RLB gauge      -> exact post-step gauge rebalance
```

## Role And Depth Policy

Let normalized depth be:

```text
d_l = l / (L - 1)
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

The design reason is mathematical: early orthogonalized matrix movement can quickly choose useful rational domains/features, but the mixture decays so late training is not dominated by generic Muon pressure.

## On-Policy Statistics

Before stepping, the wrapper records per-layer, per-group pressures:

```text
p_in,g  = rms(grad A_{l,g}) / rms(A_{l,g})
p_out,g = rms(grad B_{l,g}) / rms(B_{l,g})
p_rat,g = rms(rational coefficient gradients for group g)
```

The EMAs are `q_in,g`, `q_out,g`, and `q_rat,g`. MatrixPolicy derives:

```text
pressure_g = log q_in,g - log q_out,g
activity_g = log q_rat,g - 0.5 (log q_in,g + log q_out,g)
```

`pressure_g` measures which side of a rational group is under more update pressure. `activity_g` measures whether the rational curve itself is moving more than the surrounding matrices.

## Group-Stat Matrix Scaling

The real-corpus best row uses the group-stat variant. It applies a centered, clipped per-group multiplier before AdamW/Muon updates:

```text
c_g = c_gain,g c_pressure,g c_activity,g
c_g <- c_g / geomean(c)
c_g <- clip(c_g, 0.75, 1.35)
```

The terms are:

```text
c_gain,g = (geomean(k) / k_g)^0.20
k_g = derivative_rms_g for W_in
k_g = output_rms_g     for W_out

c_pressure,g = exp(0.10 pressure_direction_g)
pressure_direction_g = -pressure_g for W_in, +pressure_g for W_out

c_activity,g = exp(-0.20 relu((activity_g - 0.05) / 0.45))
```

The group-stat window is active from training progress `0.02` to `0.30`. `W_in` gradients are scaled group-row-wise; `W_out` gradients are scaled group-column-wise.

## Gauge Rebalance

After child optimizer steps, the wrapper applies a function-preserving gauge correction every `k` steps. Let:

```text
n_in,g  = rms(A_{l,g})
n_out,g = rms(B_{l,g})
current_g = log n_in,g - log n_out,g
```

The target combines rational curve statistics and gradient pressure:

```text
target_g = 0.5 (log derivative_gain_g - log output_gain_g)
           + beta_pressure (log q_in,g - log q_out,g)
```

The applied step is:

```text
ell_g = 0.5 (target_g - current_g)
ell_g <- clip(schedule(t,l) activity_gain_g ell_g, -ell_max, ell_max)
s_g = exp(ell_g)

A_{l,g} <- s_g A_{l,g}
B_{l,g} <- B_{l,g} / s_g
```

This changes the parameterization but preserves the represented RLB function up to floating-point error.

## Full Step Order

One `rational_matrix_policy_onpolicy` step is:

```text
1. record live RLB pressure/activity statistics
2. optionally precondition RLB matrix gradients with group-stat scaling
3. step backbone AdamW parameters
4. step rational coefficient parameters
5. step RLB matrices with the role/depth AdamW-Muon matrix policy
6. apply exact W_in/W_out gauge rebalance
```

Jacobian, quotient, coefficient-only, transport, and LR-scheduler variants are ablations. They are not the baseline optimizer claim.

## Empirical Readout

Current real-corpus results:

| task | SiLU+AdamW loss/PPL | RLB+AdamW loss/PPL | RLB+MatrixPolicy group-stat loss/PPL | MatrixPolicy gap vs SiLU+AdamW |
| --- | ---: | ---: | ---: | ---: |
| FineWeb | 4.504617 / 90.43 | 4.493013 / 89.39 | 4.344150 / 77.03 | 0.160467 loss / 13.40 PPL |
| FineWeb-Edu | 4.225019 / 68.38 | diverged | 4.072055 / 58.68 | 0.152964 loss / 9.70 PPL |

This says the useful part is not generic Muon and not RLB alone. Generic Muon is worse on both real-corpus tasks, and plain `RLB+AdamW` is unstable on FineWeb-Edu. The positive result comes from the RLB-specific matrix policy plus group-stat scaling and gauge rebalance.

## Evaluation Standard

A new optimizer design should be judged by:

```text
training and validation curves from step 1
validation loss AUC at early and mid horizons
final heldout loss and PPL
comparison to SiLU+AdamW, RLB+AdamW, SiLU+Muon, and RLB+Muon
function movement per parameter movement
gauge drift and W_in/W_out norm-product diagnostics
denominator margin and rational coefficient activity
```

The next mechanism step is direct function-space auditing: show that MatrixPolicy gets more useful function change per update and less harmful gauge drift than generic optimizers.
