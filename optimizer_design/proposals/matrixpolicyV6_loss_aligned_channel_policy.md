# matrixpolicyV6 Proposal: Loss-Aligned Channel MatrixPolicy

Status: proposal only. This is not wired into `training/transformer_lm_compare.py`, not accepted by the Slurm allowlist, and not queued. The only live MatrixPolicy optimizer remains `rational_matrix_policy_onpolicy`.

## Why V6 Is Not V5 With Tweaks

The failed V4/V5 sequence says something specific:

- V4 measured a function-balance magnitude proxy, but the signal clipped to a role-wise constant and was then mostly normalized away.
- V5 fixed the role-wise centering problem and produced real A/B role scaling, roughly `in=0.817-0.819`, `out=1.221-1.224`, but it still only improved FineWeb-Edu and was neutral/slightly worse on the other four E1 datasets.
- Therefore unsigned function sensitivity is not enough. It tells us how much a matrix channel can move the function, not whether that move points down the loss.

V6 should allocate RLB matrix update budget by signed first-order loss decrease in function space.

## Local Model

For one RLB layer and group:

```text
z_g = A_g x
h_g = phi_g(z_g)
y = sum_g B_g h_g
```

For candidate matrix updates `u_A,g` and `u_B,g`, the first-order function movement is:

```text
delta y_A,g ~= B_g J_g (u_A,g x)
delta y_B,g ~= u_B,g h_g
```

where `J_g` is the local Jacobian of the RLB group with respect to its input block. If `g_y = dL/dy`, the first-order loss change is:

```text
delta L_g,r ~= <g_y, delta y_g,r>
```

The useful score is not `||delta y||`. The useful score is signed descent per unit function movement:

```text
align_g,r = - <g_y, delta y_g,r> / (||g_y|| ||delta y_g,r|| + eps)
```

High positive `align` means the channel is producing function movement aligned with loss decrease. Negative `align` means the proposed channel movement is locally harmful even if its sensitivity is large.

## Proposed Rule

Keep the original V1 MatrixPolicy as the base update. Before applying the matrix update, estimate `align_g,in` and `align_g,out` on the current training batch at a low frequency.

Use a bounded, centered multiplier over the joint set of all RLB matrix channels in the layer:

```text
raw_g,r = beta * clip(align_g,r - mean_{g,r}(align_g,r), -c, c)
scale_g,r = exp(raw_g,r)
scale_g,r <- scale_g,r / geomean_{g,r}(scale_g,r)
scale_g,r <- clip(scale_g,r, s_min, s_max)
```

Apply `scale_g,in` to `A_g` gradients and `scale_g,out` to `B_g` gradients before the existing V1 AdamW/Muon matrix policy. Do not introduce a new optimizer family, global LR schedule, fusion change, or activation change.

Initial conservative settings for a pilot:

```text
align_every = 8 optimizer steps
beta = 0.30
c = 0.50
s_min = 0.80
s_max = 1.25
start = 0.04 training progress
end = 0.45 training progress
ema_decay = 0.90
```

The multipliers must be centered jointly over input and output roles, not separately by role. Role-wise centering would recreate the V4 failure mode.

## Why This Is A Priori Reasonable

For a fixed small function-space movement budget, steepest descent chooses the direction with largest negative inner product with `g_y`. V6 approximates that decision for the two matrix channels that RLB exposes: input-domain selection through `A_g`, and feature recombination through `B_g`.

This is also the right invariance target. The score is computed in represented function space, so a pure positive scale gauge change of `A_g` and `B_g` should not by itself look like progress.

## Why This Is A Posteriori Motivated

V1 already wins E1/E2 because it encodes RLB role/depth structure. V4 and V5 show that a magnitude-only functional metric is too weak. V6 keeps the successful V1 structure and asks the missing question: which RLB matrix channel is currently producing loss-aligned function movement?

FineWeb-Edu being the only V5 improvement is a useful clue: inverse sensitivity sometimes helps, but it is not robust. V6 should only reallocate when the signed loss geometry says the same thing.

## Required Telemetry

A V6 implementation is not credible unless it logs:

```text
matrix_policy_v6_alignment_mean_by_role
matrix_policy_v6_alignment_std_by_role
matrix_policy_v6_scale_mean_by_role
matrix_policy_v6_scale_std_by_role
matrix_policy_v6_scale_clip_frac
matrix_policy_v6_negative_alignment_frac_by_role
```

Immediate rejection conditions:

```text
scale_clip_frac > 0.25 for most logged steps
alignment is near constant by role, as in V4
candidate is slower than V1 by more than 5% without clear early AUC gain
candidate loses paired V1 validation AUC on both pilot datasets
```

## Implementation Constraint

No extra forward/backward pass should be part of the claimed optimizer method. The alignment estimate should use low-frequency hooks on the normal training batch. If exact hook implementation is too expensive, V6 should not be promoted; it should be replaced by V7-style secant trust.
