# Global-RLB: the three leading optimizer methods

This document records the exact mathematics and matched results for the three
leading optimizers designed for the learned Global-RLB activation. The fixed
comparison cell is a 296.9M-parameter causal Transformer trained on 300M DCLM
tokens for 4,000 updates with seed 1337 on four RTX A6000 GPUs. Peak/minimum
learning rates are `3e-4`/`3e-5`, warmup is 200 updates, weight decay is
`0.10`, Adam betas are `(0.9, 0.95)`, epsilon is `1e-8`, gradient clipping is
`1.0`, Muon momentum is `0.95`, and Muon uses five Newton--Schulz iterations.
All internal learning-rate and weight-decay multipliers equal one.

The matched SwiGLU+Muon control ends at validation loss `4.2282772064` and
perplexity `68.59894849`. Every result below uses the same data,
initialization, schedule, batch shape, precision, and distributed setup.

## Shared Global-RLB geometry

For learned group `g`, let `A_g` and `B_g` be the incoming and outgoing
matrices. With group width `w`, the response is

```text
z_g   = A_g x
rho_g = sqrt(mean(z_g^2) + epsilon)
u_g   = z_g / rho_g
h_g   = rho_g P5_g(u_g) / Q4_g(|u_g|)
y_g   = B_g h_g.
```

`P5_g/Q4_g` is learned independently in each layer and group. The optimizer
captures aligned inputs, preactivations, learned features, and downstream
loss cotangents. For an incoming/outgoing direction `(D_A, D_B)`, its exact
first-order functional image is

```text
Y = D_B h + B J_h D_A x,
```

where `J_h` is the Jacobian of the current RMS-rescaled P5/Q4 response. If
`e` is the downstream cotangent, coordinate `j` has functional score

```text
s[n,j] = <e[n], Y[n,j]>,       F = E[s s^T].
```

The registered quadratic transaction uses the all-rank score Fisher, its
weight-decay cross term, exact gradient and Nesterov descent certificates,
and an exact paired Frobenius update budget. The scheduled learning rate and
decoupled weight decay are applied once after direction selection.

The shared structural parent also supplies a paired radial atlas. For each
incoming row/outgoing column around one hidden coordinate, it forms matched
Adam radial signals, applies the Moore--Penrose inverse square root of the
current 2x2 learned-response Gram, removes the component parallel to the
parent direction, and restores the parent's exact group budget. Parent and
radial directions define orthogonal half-energy sum/difference axes. Their
exact P5/Q4 functional scores form a 648-coordinate cross-layer metric.

With the fixed `beta2 = 0.95`, that metric is accumulated as

```text
M_t    = beta2 M_(t-1) + (1-beta2) F_t
C_t    = beta2 C_(t-1) + (1-beta2) c_t
Fbar_t = M_t / (1-beta2^t)
cbar_t = C_t / (1-beta2^t),
```

where `c_t` is the weight-decay score cross term. Bias correction makes the
first update equal to the current-batch transaction; subsequent updates
retain cross-batch rank and cross-layer support.

## Method 1: cross-role frame polar with functional allocation

For every layer/group, transpose outgoing momentum so the incoming and
outgoing sources share shape: `M_A, M_C in R^(w x d)`, with `C = B^T`.
Method 1 then:

1. Stacks both roles and applies the unchanged Muon NS5 map:

   ```text
   [U_A; U_C] = NS5([M_A; M_C]).
   ```

2. Treats `(U_A, U_C)` as one paired direction, removes its projection onto
   the complete persistent-P5/Q4 parent `P`, and restores the same per-layer
   paired Frobenius budget:

   ```text
   R = U - <U,P>/<P,P> P
   H = sqrt(<P,P>/<R,R>) R.
   ```

3. Forms the orthogonal half-energy axes `(P+H)/2` and `(P-H)/2`, computes
   their exact functional images and loss-cotangent scores, and solves the
   all-rank equal-budget quadratic transaction.

4. Accepts the selected direction only if finiteness, budget, exact-gradient
   descent, and Nesterov-descent certificates pass; otherwise it uses the
   feasible parent. Common learning rate and weight decay are applied once.

Attention matrices use the complete response-conditioned attention
transaction: current P5/Q4 response congruence and participation choose an
equal-budget chord of two NS5-derived attention directions.

## Method 2: paired post-polar adaptive frame

Method 2 uses the same structural and attention transaction, adding one
normalization immediately after the joint frame polar and before parent
residualization. For paired incoming/outgoing row `i`, it computes

```text
q_t[i] = 0.5 (mean(U_A[i]^2) + mean(U_C[i]^2))
v_t[i] = beta2 v_(t-1)[i] + (1-beta2) q_t[i]
r_t[i] = 1 / (sqrt(v_t[i] / (1-beta2^t)) + epsilon).
```

Both members of the pair receive the same scale:

```text
U_A[i] <- r_t[i] U_A[i],       U_C[i] <- r_t[i] U_C[i].
```

The method then performs the same projection removal, exact budget closure,
functional scoring, constrained selection, and single learning-rate/weight-
decay application. It uses the cell's fixed `beta2=0.95` and `epsilon=1e-8`
and introduces no learning-rate or weight-decay multiplier.

## Method 3: persistent structural metric with row-product attention

Method 3 uses the shared persistent 648-coordinate P5/Q4 structural
transaction. Its new geometry is applied to every QKV and attention-output
matrix row. Write an effective attention row as

```text
W_i = m_i d_i / ||d_i||,       m_i = ||W_i||.
```

For matrix gradient `G_i`, the exact pullback is

```text
u_i = W_i / m_i
g_m = <G_i, u_i>
G_d = (m_i / ||d_i||) (G_i - u_i g_m).
```

Method 3 then:

1. updates `d_i` with the complete response-conditioned NS5 attention
   transaction using tangent gradient `G_d`;
2. updates magnitude `m_i` with matched Adam (`beta1=0.9`, `beta2=0.95`,
   `epsilon=1e-8`) using `g_m`;
3. reconstructs `W_i = m_i d_i/||d_i||`; and
4. applies effective-weight decoupled weight decay exactly once.

This optimizes spectral/tangent motion and exact row-magnitude motion in
their product coordinates without changing the external learning rate,
weight decay, clipping, NS5 count, or update schedule.

## Why they lead SwiGLU+Muon

SwiGLU supplies a fixed activation map. Its Muon update can use matrix
momentum and spectral geometry, but it has no learned per-layer/per-group
P5/Q4 response, no current response Jacobian, and no P5/Q4 functional tangent
scores against the downstream loss cotangent. These methods use those
Global-RLB-specific signals to allocate the same update budget toward
directions that act favorably in function space. Method 1 adds a joint
analysis/synthesis frame constraint, Method 2 also equalizes persistent
paired post-polar row energy, and Method 3 separates attention-row magnitude
from tangent direction. This is a mechanism interpretation of the controlled
result, not a claim that the endpoint alone proves a unique causal cause.

## Exact 4,000-step results and time

| Method | Validation loss | PPL | Lead over SwiGLU+Muon | Median step | Time ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Method 1 | 4.14794636 | 63.3039 | +0.08033085 | 2.4131 s | 1.841x |
| Method 2 | 4.14811611 | 63.3146 | +0.08016109 | 2.4467 s | 1.849x |
| Method 3 | 4.14817286 | 63.3182 | +0.08010435 | 2.3186 s | 1.734x |

Timing uses same-node A/candidate/B brackets against SwiGLU+Muon. These rows
are the exact scientific methods, not the final runtime target.

## Runtime-optimized variants

NS5 remains unchanged. A cadence, stale-factor, or numerical approximation
is reported only after a fresh complete 4,000-step quality run.

The verified fast Method 1 refreshes its expensive outer-frame transaction
every four steps while keeping current gradients, momentum, budgets,
learning rate, weight decay, and exact inherited attention active every
step. It ends at loss `4.1568298340`, retains a `+0.07144737` lead, and has
median step time `1.7172 s`, or `1.157x` SwiGLU+Muon. Approximations retaining
about `+0.07` with a material speed gain remain eligible bases for further
optimization; timing-only variants are not promoted without the full quality
run.
