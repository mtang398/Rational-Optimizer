# Top three RLB-specific optimizer methods

This note records the three best **completed** RLB-specific optimizer runs in
the exact 300M-token, 4,000-step M1 campaign as of 5 August 2026. The ranking
is by endpoint validation loss. It deliberately excludes unfinished runs and
ablation-only variants.

The most important implementation identity is:

- **First place, R06 K1**, is the production wrapper `rlb_r06.py` backed by
  `rlb_r06_revision_core.py`.
- The older file named `rlb_r06_core.py` implements the **third-place
  group-resolved product-sphere predecessor**. It is not the winning R06 K1
  method.
- **Second place, R05 K1**, is the frozen generation-one R05 method. A later
  same-slot R05 revision is a different method and is not described as R05 K1
  here.

## Experimental cell and evidence status

All three runs used the same frozen comparison cell:

- model: 18 layers, model width 1,024, 16 heads, FFN setting 3,072;
- RLB hidden width: 4,608, split into 18 learned rational groups of width 256;
- trainable parameters: 296,871,080;
- data: the same 300,000,000-token DCLM training cache and disjoint
  8,000,000-token validation cache;
- seed: 1337;
- training: 4,000 optimizer updates, four A6000 ranks, batch 8 per rank,
  gradient accumulation 4, and sequence length 256;
- LR: cosine schedule from `3e-4` to `3e-5`, with 200 warmup updates and a
  4,000-update horizon;
- weight decay: `0.10` on decayed parameters;
- Adam moments: `(0.9, 0.95)`, epsilon `1e-8`, and gradient clipping `1.0`;
- Muon: momentum `0.95`, five Newton--Schulz iterations (NS5), and
  `match_rms_adamw` shape calibration;
- every external and internal LR or WD multiplier: exactly `1.0`.

The stronger frozen control is SwiGLU+Muon (U-S), not RLB+Muon (U-R):

| Run | Endpoint loss | Endpoint PPL | Train pAUC, steps 1,000--4,000 | Validation pAUC, steps 1,000--4,000 |
|---|---:|---:|---:|---:|
| SwiGLU+Muon (U-S) | 4.228466988 | 68.61196851 | 4.199301681 | 4.422074735 |
| RLB+Muon (U-R) | 4.241679192 | 69.52449883 | 4.205564916 | 4.432158089 |

Lower is better for every column. The three completed candidates rank as
follows:

| Rank | Method | Endpoint loss | Loss lead over U-S | Endpoint PPL | Train pAUC | Validation pAUC | Runtime / U-S |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | R06 K1: intrinsic role-participation router | **4.199958801** | **0.028508186** | **66.68358371** | **4.161953121** | **4.395792866** | 1.967x |
| 2 | R05 K1: learned-response amplitude router | 4.207155704 | 0.021311283 | 67.16523011 | 4.172808398 | 4.404217637 | 1.790x |
| 3 | R06 predecessor: group-resolved product sphere | 4.208533764 | 0.019933224 | 67.25785159 | 4.172926977 | 4.404469943 | 1.807x |

These are single-seed discovery results. All three pass the registered LR/WD
fairness audit and improve on U-S in endpoint loss, endpoint PPL, and both
late pAUCs. None reaches the required `0.20` loss lead, none is promoted, and
no complete method-level ablation package is released for any of the three.
Their discovery runtimes also do not meet the eventual `1.02x` runtime gate.

## Shared RLB notation and retained coordinate parent

For one width-`m` rational group, Global-RLB computes

```text
rho = sqrt(mean(z^2) + eps),   u = z / rho,
h   = rho f(u),                f = P5 / Q4.
```

Here `P5` is the learned fifth-order numerator and `Q4` is the positive
fourth-order denominator. The exact Jacobian of the normalized response is

```text
J = diag(f') + ((f - u * f') / m) u^T.
```

All three methods inherit the recursively pruned B+C coordinate geometry for
the two matrices adjacent to the RLB. B is the residual-input covariance
`C_x = E[x x^T]`, used as the right coordinate of the incoming matrix. C is
the groupwise rational-feature covariance `K_g = E[f_g f_g^T]`, used as the
hidden coordinate of the outgoing matrix. Their Cholesky coordinate maps are
normalized to unit determinant and are paired with their exact adjoints.
The previously tested incoming-response metric A is absent.

For either RLB-adjacent matrix role `r`, write the resulting B+C coordinate
map as `C_r`, its exact adjoint as `C_r*`, and the clipped-gradient Nesterov
tensor as `M_r`. The retained coordinate-polar direction is

```text
U_r = C_r* NS5(C_r M_r).
```

The usual Muon shape calibration is applied after the coordinate closure.
The scheduled LR and decoupled WD are each applied once.

## First place: R06 K1

### Current-versus-initial response geometry

R06 K1 retains R05 K1's comparison between the current rational curve `f_t`
and its frozen initializer `f_0`. It forms two uncentred, block-direct-sum
kernel alignments from the current normalized preactivations:

```text
a_in  = <J_t J_t^T, J_0 J_0^T>
        / (||J_t J_t^T||F ||J_0 J_0^T||F),

a_out = <h_t h_t^T, h_0 h_0^T>
        / (||h_t h_t^T||F ||h_0 h_0^T||F).
```

The sufficient statistics are additive across sampled tokens, rational
groups, and data-parallel ranks. Exact initializer equality returns alignment
one exactly.

### Intrinsic current-curve participation

The defining R06 addition is to measure how broadly the **current learned
rational curve** distributes local sensitivity and response energy. For the
singular values `s_i` of the exact normalized-response Jacobian,

```text
c_in = (sum_i s_i^2)^2 / (m * sum_i s_i^4).
```

For the current response coordinates `f_i = f(u_i)`,

```text
c_out = (sum_i f_i^2)^2 / (m * sum_i f_i^4).
```

Both quantities lie in `[0,1]`, are invariant to multiplying the entire
response by a scalar, and vary with the learned P5/Q4 shape. R06 K1 combines
relative morphology and current intrinsic morphology role by role:

```text
r_in  = a_in  * c_in,
r_out = a_out * c_out.
```

At the installed SiLU-equivalent P5/Q4 initializer, an a-priori Gaussian
probe measured mean `c_in` about 0.504, mean `c_out` about 0.146, and mean
`sqrt(c_in*c_out)` about 0.270, so the router is active from the first update.

### RLB-adjacent matrix update

Let `G_r = C_r M_r` be the B+C-coordinate Nesterov tensor. R06 maintains a
bias-corrected coordinatewise second moment of the literal clipped gradient,
using the fixed campaign `beta2=0.95` and epsilon `1e-8`. Its positive inverse
square root is `D_r`. The ordinary and adaptive directions are

```text
U_r = C_r* NS5(C_r M_r),
A_r = C_r* D_r NS5(D_r C_r M_r).
```

The second `D_r` is required by the exact adjoint. `A_r` is matched to
`||U_r||F`; no LR multiplier is used. With `r_r` equal to `r_in` or `r_out`,
the executed RLB-adjacent direction is

```text
Z_r = sqrt(r_r) U_r + sqrt(1-r_r) A_r,
D_r06 = ||U_r||F * Z_r / ||Z_r||F.
```

An explicit exact-limit branch returns the parent direction when the route is
one. Power-of-two projective stabilization and finite-precision adjoint
compensation in the implementation preserve this mathematical map; they are
numerical realizations, not extra optimizer components or update scales.

### RLB-conditioned attention update

R06 K1 also transports the two RLB role statistics to the QKV and attention
output matrices through the parameter-free geometric mean

```text
r_attn = sqrt(c_in * c_out).
```

For each attention role, let `M` be its unchanged Nesterov source. A
row/column-factorized, bias-corrected second moment of the literal gradients
forms an adaptive source `A`, which is normalized to `||M||F`. R06 computes

```text
S = sqrt(r_attn) M + sqrt(1-r_attn) A,
D_attn = match_rms_adamw(NS5(S)).
```

The scheduled LR and WD are then applied once. Thus R06 K1 structurally owns
169,869,312 RLB-adjacent matrix elements plus 75,497,472 attention matrix
elements, for 245,366,784 routed matrix elements. Rational coefficients and
the remaining parameters keep matched AdamW behavior. There is no learned or
tuned gain, threshold, exponent, cadence, candidate schedule, loss feedback,
or internal LR/WD multiplier.

### Frozen identity

The complete R06 K1 run is identified by:

```text
component:                  R06:K1
revision core SHA-256:      79dafe29a88d778a880115b50e0eab6471f463e29b190fbc2b4f388725dcbe07
wrapper SHA-256:            9bc23f30c16debe9d5d2a403baa671f3884471606cf076214bc5012f200210a7
source-freeze SHA-256:      c5f345888cfcea2b9086a88bfeb9f51f612e202eef82843cdce98920b188e496
candidate-report SHA-256:   10dd8004fe997ded5939d2c13cf806f8f475cee69f1a3249bcefdedd96690305
trajectory SHA-256:         c8331ae8e0b92af2c798d4246fa959809fab80bbec5e14cb8c3ca78fefbd636c
```

## Second place: R05 K1

R05 K1 is the generation-one learned-response amplitude router inherited by
R06. It uses `a_in` and `a_out` directly, without multiplying them by the
intrinsic participation statistics and without extending the route to
attention matrices.

For each RLB-adjacent role, R05 K1 forms the same equal-budget ordinary and
enclosed adaptive directions:

```text
U_r = C_r* NS5(C_r M_r),
A_r = C_r* D_r NS5(D_r C_r M_r),
||A_r||F <- ||U_r||F.
```

Because the response congruence is a squared alignment, its canonical
amplitudes are `sqrt(a_r)` and `sqrt(1-a_r)`. The final direction is

```text
Z_r = sqrt(a_r) U_r + sqrt(1-a_r) A_r,
D_r05 = ||U_r||F * Z_r / ||Z_r||F.
```

Both branch amplitudes are nonnegative, both branches are certified descent
directions, and the blend is returned at the parent's Frobenius budget. At
the exact initializer it returns the literal B+C parent while still advancing
the second-moment state. Other matrices use stock Muon; rational coefficients
and remaining non-matrix parameters use matched AdamW. LR, WD, clipping,
schedule, momentum, NS5 count, B/C volume, and update budget are unchanged.

Direct 4,000-step deletion of either B or C from the inherited B+C parent
worsened endpoint loss, endpoint PPL, train late pAUC, and validation late
pAUC. C's loss contribution was `0.009883404`; it was positive under literal
leave-one-out interpretation but missed a separately frozen `0.01`
loss-magnitude field by `0.000116596`. The later R05 K1 response-router
additions were not recursively ablated because the full method did not cross
the campaign's `0.20` promotion gate.

The frozen R05 K1 identity is:

```text
component:                   R05:K1
generation-one core SHA-256: 57e79a23be0bb786039481446ccbab917b9b1c43ec9b16ddfefa11b7dcaccdd8
source-freeze SHA-256:       40037759752e53bee2e33a392d5e91a2e2b490771f841e5a4c981ee273100a29
candidate-report SHA-256:    0abf6e99220ed8bf925a7dd21181cbffddf97595e67e282fa878bf5acf661edd
trajectory SHA-256:          2817ce9688e83ed64a3a59eb1a343b81f8183785d36f55fcf2766d9843da7919
```

## Third place: group-resolved product-sphere R06 predecessor

This historical R06 generation retains the literal R05 K1 parent and its two
equal-budget RLB-adjacent branches, but changes the routing resolution. R05 K1
reduces each incoming/outgoing current-versus-initial response congruence to
one value per layer and role. The product-sphere predecessor also retains the
pre-reduction congruence `a_{g,r}` for each of the 18 independently learned
rational groups.

For group `g` and role `r`, let `P_{g,r}` be the corresponding block of the
literal layerwise R05 K1 direction, and let `U_{g,r}` and `A_{g,r}` be the
ordinary and equal-budget adaptive blocks that formed it. The group-specific
canonical target is

```text
T_{g,r} = normalize_to_||P_{g,r}||(
              sqrt(a_{g,r}) U_{g,r}
            + sqrt(1-a_{g,r}) A_{g,r}).
```

The method considers the shortest fixed-radius arc from `P_{g,r}` to
`T_{g,r}` and chooses

```text
D_{g,r} = argmax_D <M_{g,r}, D>
          subject to D lying on arc(P_{g,r}, T_{g,r}).
```

The exact candidates are the parent, the target, and the unique interior
stationary point when it lies on the arc. The parent is candidate zero and
wins exact ties. If the group congruence equals its layer congruence, the
method returns the literal R05 parent block. Every accepted block preserves
the parent's group Frobenius norm and has no smaller first-order Nesterov
descent than that parent block.

This predecessor acts only on the 169,869,312 RLB-adjacent matrix elements.
Other matrices retain stock Muon, while rational coefficients and remaining
non-matrix parameters retain matched AdamW. It contains no R06 K1 intrinsic
Jacobian/response participation route and no RLB-conditioned attention route.
Its later functional selector was explicitly bypassed; the product-sphere
decision above is the complete scientific mechanism.

The frozen predecessor identity is:

```text
generation:                 group_resolved_product_sphere
source-freeze SHA-256:      c824cd699ed22677524e83e20826c2924efcee902af787fd321c34de57f36200
candidate-report SHA-256:   03afd1cce7042996a6a145fac50e9f9b241a715108cf4c49497c765e01d9b9bd
trajectory SHA-256:         ea9d0109a8e7ca0750599f297c48d0bc2261bc0fa324938eca4d86b427706cb2
```

## Interpretation boundary

The results establish three checksum-valid, LR/WD-matched single-seed
trajectories with modest improvements over SwiGLU+Muon in this exact cell.
They do not yet establish the requested large lead, multiseed robustness,
component closure for the complete methods, or deployment-speed parity. The
method descriptions above therefore document reproducible discovery
mechanisms and evidence, not final promoted optimizers.
