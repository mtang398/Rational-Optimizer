# Exact RLB optimizer specification: R06 K1 and two runner-up methods

## What this document describes

This document explains three optimizer methods that were tested on the same
Global-RLB transformer:

1. **R06 K1**, the best method in the requested three-method snapshot;
2. **R05 K1**, the first runner-up and the parent from which R06 K1 was built;
3. **the group-resolved product-sphere method**, the second runner-up and an
   earlier R06 generation.

All three are hybrid optimizers. They keep AdamW for scalar, bias,
normalization, embedding, output-head, and rational-coefficient parameters.
They keep the ordinary Muon update for matrix parameters outside their stated
scope. Their new behavior is confined to selected matrices whose update can be
informed by the current trainable Global-RLB response.

The activation function, model architecture, data, learning-rate schedule,
weight decay, gradient clipping, Adam moments, Muon momentum, and number of
Newton--Schulz iterations are identical between the three candidate runs. The
activation is never replaced or modified by an optimizer method.

The names R05 and R06 are opaque experiment identifiers. The algorithms are
defined by the equations below, not by the names.

## Method overview

For each Global-RLB block, the methods use the current rational response to
choose update directions for `W_in`, `W_out`, or both; R06 K1 also routes the
attention-matrix update. The scheduled learning rate, weight decay, and update
norm budget remain fixed.

| Method | RLB information used | Matrices changed relative to stock Muon | Central decision |
|---|---|---|---|
| **R06 K1** | Current-versus-initial response alignment, current Jacobian participation, and current response participation | Both matrices around every RLB activation and all QKV/attention-output matrices | Route continuously between an RLB-coordinate polar direction and an equal-budget adaptive direction; use RLB participation to route attention too |
| **R05 K1** | Current-versus-initial response alignment | Only the two matrices around every RLB activation | Route continuously between the same two equal-budget RLB-coordinate directions |
| **Group-resolved product sphere** | The R05 alignment separately for each RLB group | Only the two matrices around every RLB activation | Start from the R05 direction, then choose the best first-order point on a fixed-norm arc toward each group's own target |

The three methods are related as follows:

```text
shared B+C RLB coordinate geometry
              |
              v
      R05 K1 layerwise router
          /                 \
         v                   v
group-resolved          R06 K1 intrinsic
product sphere          participation router
                            +
                   RLB-routed attention
```

In the requested snapshot, R06 K1 had validation loss `4.199958801`, a
`0.028508186` improvement over the stronger SwiGLU+Muon control at
`4.228466988`. This is a real but incomplete result: it is far short of the
required `0.20` improvement, its full recursive ablation is not released, and
its discovery implementation is slower than the final runtime requirement.

## Fixed model boundary

All three methods use the existing Global-RLB model without changing its
activation, forward map, initialization, parameterization, or coefficient
gradients. Only the optimizer applied after backward differs. Global-RLB's
rational coefficients use the same AdamW configuration in every candidate.

## Optimizer definition

### Parameter routing

One training update starts with the same forward pass, cross-entropy loss,
backward pass, all-rank gradient synchronization, and global gradient clipping
for every comparison. Parameters are then assigned as follows:

| Parameter class | R05 K1 | Group-resolved product sphere | R06 K1 |
|---|---|---|---|
| RLB `W_in` and `W_out` matrices | R05 RLB-specific update | Group-resolved RLB-specific update | R06 RLB-specific update |
| QKV and attention-output matrices | Stock Muon | Stock Muon | R06 RLB-conditioned attention update |
| Other eligible 2-D matrices | Stock Muon | Stock Muon | Stock Muon |
| RLB coefficients | Matched AdamW | Matched AdamW | Matched AdamW |
| Biases, normalization weights, embeddings, output head, and other non-Muon parameters | Matched AdamW | Matched AdamW | Matched AdamW |

R06 K1 therefore controls 169,869,312 elements in RLB-adjacent matrices and
75,497,472 elements in attention matrices, for 245,366,784 structurally routed
matrix elements.

### The unchanged Muon source

For an eligible matrix with clipped gradient `G_t`, the momentum buffer and
Nesterov source are the exact recurrence used by the control Muon
implementation:

\[
B_t = 0.95B_{t-1}+0.05G_t,
\qquad
M_t = 0.05G_t+0.95B_t.
\]

Muon applies five Newton--Schulz iterations, written `NS5`, to approximate the
matrix polar or zero-power direction. It then applies the same
`match_rms_adamw` shape calibration used by the control.

The new methods do not change `G_t`, the `0.95` momentum, the Nesterov
recurrence, the five Newton--Schulz iterations, or the final scheduled learning
rate. They change the coordinate system or the equal-norm direction supplied
to the polar map.

## Geometry shared by all three methods

The three methods inherit the same B+C coordinate parent for the two matrices
around each RLB activation. The letters B and C are historical component
labels:

- **B: residual-input covariance.** For `W_in`, sample current residual inputs
  and form `C_x = E[xx^T]`. This supplies the right-hand coordinate of the
  incoming matrix.
- **C: rational-feature covariance.** For each rational group, sample current
  activation outputs and form `K_g = E[h_g h_g^T]`. This supplies the grouped
  hidden coordinate of the outgoing matrix.

Each positive covariance is converted to a Cholesky coordinate and normalized
to unit determinant. Unit-determinant normalization removes a global scale, so
the coordinate changes geometry without becoming an internal learning-rate
multiplier.

Concretely, the incoming map applies B on the right of `W_in`; the outgoing
map applies the groupwise C coordinate to the hidden dimension of `W_out`
(implemented after transposition). The previously tested incoming
rational-response metric A is absent and its coordinate is the identity.

For one matrix role `r`, let:

- `M_r` be its unchanged Nesterov source;
- `C_r(.)` be the complete B+C coordinate map;
- `C_r*(.)` be the exact adjoint that maps a coordinate direction back to
  parameter space.

The ordinary RLB-coordinate polar direction is

\[
U_r = C_r^*\!\left(\operatorname{NS5}(C_r(M_r))\right).
\]

Thus NS5 is evaluated after pushing the Nesterov tensor into the current RLB
coordinate, and the resulting polar direction is returned through the exact
adjoint.

All three methods also construct an adaptive alternative. From the literal
clipped gradient in the same coordinate system, maintain the bias-corrected
second moment

\[
V_{r,t}=0.95V_{r,t-1}+0.05\,C_r(G_t)\odot C_r(G_t),
\qquad
D_{r,t}=\left(
\sqrt{\frac{V_{r,t}}{1-0.95^t}}+10^{-8}
\right)^{-1},
\]

The adaptive direction is

\[
A_r
=C_r^*\!\left[
  D_{r,t}\,
  \operatorname{NS5}\!\left(D_{r,t}C_r(M_r)\right)
\right].
\]

The outer `D_r` is part of the exact adjoint construction. Before routing,
`A_r` is rescaled to the Frobenius norm of `U_r`:

\[
\bar A_r=A_r\frac{\|U_r\|_F}{\|A_r\|_F}.
\]

Consequently, choosing more of the adaptive branch does not grant a larger
step. `U_r` and `A_bar_r` have exactly the same update budget; only their
directions differ.

Power-of-two scale stabilization and finite-precision adjoint compensation in
the implementation preserve these equations. They are numerical safeguards,
not scientific components, schedules, or hidden update multipliers.

## Method 1: R06 K1

### Core idea

R06 K1 combines two complementary properties of each learned RLB response:
current-versus-initial response alignment and current intrinsic participation.
Their product gives a bounded route for `W_in` and `W_out`; the intrinsic
participation also routes attention. The routes therefore change with the
learned RLB response while LR and WD remain fixed.

### Step 1: current-versus-initial response alignment

For the same normalized samples `u`, evaluate both the current learned curve
`f_t` and the frozen initialization curve `f_0`.

For the incoming matrix, compare their exact normalized-response Jacobian
kernels, where `J_t` and `J_0` denote the current and initialized RLB
Jacobians:

\[
a_{\mathrm{in}}
=
\frac{
\langle J_tJ_t^\top,J_0J_0^\top\rangle_F
}{
\|J_tJ_t^\top\|_F\,\|J_0J_0^\top\|_F
}.
\]

For the outgoing matrix, compare their activation-output kernels, with
`h_t=rho f_t(u)` and `h_0=rho f_0(u)`:

\[
a_{\mathrm{out}}
=
\frac{
\langle h_th_t^\top,h_0h_0^\top\rangle_F
}{
\|h_th_t^\top\|_F\,\|h_0h_0^\top\|_F
}.
\]

Both values are in `[0,1]`. A value of one means the current and initialized
response kernels point in the same direction; a lower value means the learned
rational morphology has changed. Statistics are summed across sampled tokens,
rational groups, and all four data-parallel ranks before the alignment is
formed. Exact coefficient equality returns one exactly.

### Step 2: current intrinsic participation

Alignment alone only measures change relative to initialization. R06 K1 also
measures the shape of the current curve itself.

Let `s_i` be the singular values of the exact group Jacobian `J_t`. The
incoming participation is

\[
c_{\mathrm{in}}
=
\frac{(\sum_i s_i^2)^2}{256\sum_i s_i^4}.
\]

Let `f_i` be the 256 current rational responses in the group. The outgoing
participation is

\[
c_{\mathrm{out}}
=
\frac{(\sum_i f_i^2)^2}{256\sum_i f_i^4}.
\]

These are normalized participation ratios in `[0,1]`:

- a value near one means sensitivity or response energy is broadly spread
  across channels;
- a smaller value means it is concentrated in fewer directions;
- multiplying the entire rational response by a positive scalar leaves the
  ratios unchanged.

The implementation computes the exact Jacobian expression without storing a
dense `256 × 256` Jacobian. It averages participation across sampled tokens,
groups, and ranks to produce one incoming and one outgoing intrinsic value per
layer.

### Step 3: route the two RLB-adjacent matrices

R06 K1 forms role-specific routes

\[
r_{\mathrm{in}}=a_{\mathrm{in}}c_{\mathrm{in}},
\qquad
r_{\mathrm{out}}=a_{\mathrm{out}}c_{\mathrm{out}}.
\]

For either role, with `r` equal to its corresponding route, it blends the
equal-budget ordinary and adaptive directions:

\[
Z_r=\sqrt r\,U_r+\sqrt{1-r}\,\bar A_r,
\qquad
P_r=Z_r\frac{\|U_r\|_F}{\|Z_r\|_F}.
\]

As `r` moves from zero to one, the direction moves from `A_bar_r` toward
`U_r`; the final normalization restores the exact `U_r` Frobenius budget.

The square-root amplitudes are the canonical amplitudes for a squared kernel
alignment. At the exact route-one limit, the implementation returns `U_r`
bitwise.

### Step 4: route attention from RLB morphology

R06 K1 uses the geometric mean of the two intrinsic participation values:

\[
r_{\mathrm{attn}}=\sqrt{c_{\mathrm{in}}c_{\mathrm{out}}}.
\]

For each QKV or attention-output matrix, `M` is the unchanged Nesterov source.
A row/column-factorized, bias-corrected second moment of the literal gradient
forms an adaptive source `A_attn`, which is normalized to `||M||_F`. Then

\[
S
=\sqrt{r_{\mathrm{attn}}}\,M
+\sqrt{1-r_{\mathrm{attn}}}\,\bar A_{\mathrm{attn}},
\]

\[
P_{\mathrm{attn}}
=\operatorname{match\_rms\_adamw}(\operatorname{NS5}(S)).
\]

Thus the current trainable RLB curve influences the geometry used for
attention matrices in the same transformer layer. It does not change their
LR, WD, momentum, or update norm calibration.

At the fixed model initialization, a preregistered Gaussian probe measured
approximately

```text
c_in                  = 0.504
c_out                 = 0.146
sqrt(c_in * c_out)    = 0.270
```

The intrinsic router is therefore active at initialization without a tuned
gain, threshold, exponent, schedule, cadence, or loss-feedback signal.

### Complete R06 K1 update in pseudocode

```text
1. Run the unchanged Global-RLB forward and backward passes.
2. Synchronize and clip gradients exactly as in the controls.
3. Update the unchanged Muon Nesterov buffers.
4. From current RLB samples, form B+C coordinate maps for W_in and W_out.
5. Form U_in, U_out: coordinate NS5 directions.
6. Update matched-beta2 coordinate second moments from literal gradients.
7. Form equal-budget A_in, A_out: adaptive coordinate NS5 directions.
8. Compare current and initial RLB responses: a_in, a_out.
9. Measure current RLB participation: c_in, c_out.
10. Route W_in with a_in*c_in and W_out with a_out*c_out.
11. Route QKV and attention-output matrices with sqrt(c_in*c_out).
12. Use stock Muon for remaining eligible matrices.
13. Use matched AdamW for rational coefficients and remaining parameters.
14. Apply the common scheduled LR once and common decoupled WD once.
```

## Method 2: R05 K1

### Core idea

R05 K1 is the direct parent of R06 K1. It asks only how far the current
rational response has moved from its initialization. It uses the same B+C
coordinate maps and the same equal-budget `U_r` and `A_bar_r` directions, but
it does not multiply the alignment by current intrinsic participation.

For the incoming and outgoing matrices, respectively:

\[
Z_{\mathrm{in}}
=\sqrt{a_{\mathrm{in}}}\,U_{\mathrm{in}}
+\sqrt{1-a_{\mathrm{in}}}\,\bar A_{\mathrm{in}},
\]

\[
Z_{\mathrm{out}}
=\sqrt{a_{\mathrm{out}}}\,U_{\mathrm{out}}
+\sqrt{1-a_{\mathrm{out}}}\,\bar A_{\mathrm{out}}.
\]

Each result is normalized back to the Frobenius norm of its corresponding
`U` direction. At exact initialization, both alignments are one and R05 K1
returns the literal B+C parent direction while still updating the adaptive
second-moment state.

R05 K1 specializes only `W_in` and `W_out`. Attention and all other eligible
matrices use stock Muon. Rational coefficients and non-Muon parameters use
matched AdamW.

### Complete R05 K1 update in pseudocode

```text
1. Run the unchanged Global-RLB forward and backward passes.
2. Synchronize and clip gradients exactly as in the controls.
3. Update the unchanged Muon Nesterov buffers.
4. From current RLB samples, form B+C coordinate maps for W_in and W_out.
5. Form U_in, U_out: coordinate NS5 directions.
6. Update matched-beta2 coordinate second moments from literal gradients.
7. Form equal-budget A_in, A_out: adaptive coordinate NS5 directions.
8. Compare current and initial RLB responses: a_in, a_out.
9. Route W_in with a_in and W_out with a_out.
10. Use stock Muon for attention and remaining eligible matrices.
11. Use matched AdamW for rational coefficients and remaining parameters.
12. Apply the common scheduled LR once and common decoupled WD once.
```

## Method 3: group-resolved product sphere

### Core idea

R05 K1 reduces the response statistics of all 18 rational groups to one
incoming alignment and one outgoing alignment per layer. The group-resolved
method retains those individual group alignments and lets each group refine
the R05 direction while preserving that group's update norm.

This method begins by computing the complete R05 K1 direction. It does not
replace the R05 parent. It then asks whether a group-specific route supplies a
better first-order direction for the same Nesterov source.

### Group-specific target

For group `g` and matrix role `r`, define:

- `P_g,r`: the corresponding block of the complete layerwise R05 K1 direction;
- `U_g,r`: the ordinary B+C coordinate-polar block;
- `A_bar_g,r`: its equal-budget adaptive block;
- `a_g,r`: the current-versus-initial response alignment for this group;
- `M_g,r`: the group's unchanged Nesterov source.

The group's own canonical target is

\[
T_{g,r}
=
\operatorname{normalize}_{\|P_{g,r}\|_F}
\left(
\sqrt{a_{g,r}}\,U_{g,r}
+\sqrt{1-a_{g,r}}\,\bar A_{g,r}
\right).
\]

Both `P_g,r` and `T_g,r` have the same Frobenius norm. They therefore lie on
the same fixed-radius sphere.

### Exact arc decision

Consider the shortest spherical arc from the R05 parent block `P_g,r` to the
group-specific target `T_g,r`. The method selects the point on that arc with
the greatest inner product with the Nesterov source:

\[
D_{g,r}
=
\arg\max_{D\in\operatorname{arc}(P_{g,r},T_{g,r})}
\langle M_{g,r},D\rangle_F.
\]

The exact solver checks at most three possibilities:

1. the R05 parent endpoint;
2. the group-specific target endpoint;
3. the unique interior stationary point, when it lies on the arc.

The R05 parent is always feasible and wins exact ties. Therefore every chosen
group block:

- has exactly the same Frobenius norm as its R05 parent block; and
- has first-order Nesterov descent no worse than that parent block.

If a group's alignment equals the aggregate layer alignment, its target
collapses to the literal parent block and the method returns that block
exactly.

The product-sphere method specializes only the RLB `W_in` and `W_out`
matrices. It has no R06 K1 intrinsic participation statistic and no
RLB-conditioned attention route.

### Complete group-resolved update in pseudocode

```text
1. Perform steps 1-8 of R05 K1, retaining both layer and group alignments.
2. Form the complete layerwise R05 parent direction P.
3. For every layer, role, and rational group, form its equal-norm target T.
4. Solve the exact three-candidate fixed-radius arc problem from P to T.
5. Concatenate the selected group blocks into W_in and W_out updates.
6. Use stock Muon for attention and remaining eligible matrices.
7. Use matched AdamW for rational coefficients and remaining parameters.
8. Apply the common scheduled LR once and common decoupled WD once.
```

## Exact differences among the three methods

| Design decision | R05 K1 | Group-resolved product sphere | R06 K1 |
|---|---|---|---|
| B residual-input coordinate | Yes | Yes | Yes |
| C rational-feature coordinate | Yes | Yes | Yes |
| Equal-budget coordinate second-moment branch | Yes | Yes | Yes |
| Current-versus-initial Jacobian/output alignment | Layer and role | Layer, role, and group | Layer and role |
| Current intrinsic Jacobian participation | No | No | Yes |
| Current intrinsic output participation | No | No | Yes |
| RLB pair decision | Alignment-weighted chord | Exact per-group arc maximizer around the R05 parent | Alignment × participation chord |
| Attention decision | Stock Muon | Stock Muon | RLB-participation-routed factorized adaptive Muon |
| Rational-coefficient optimizer | Matched AdamW | Matched AdamW | Matched AdamW |
| Tuned internal LR/WD multiplier | None; all equal 1 | None; all equal 1 | None; all equal 1 |

## Exact experiment cell

All reported candidates used:

| Item | Fixed value |
|---|---|
| Model | 18 layers, width 1,024, 16 attention heads |
| Nominal FFN setting | 3,072 |
| Global-RLB hidden width | 4,608 = 18 groups × 256 channels |
| Global-RLB parameters | 296,871,080 |
| SwiGLU control parameters | 296,867,840 |
| Training data | Same cached 300,000,000-token DCLM stream |
| Validation data | Same disjoint 8,000,000-token cache |
| Seed | 1337 |
| Updates | 4,000 |
| Hardware | Four A6000 GPUs per job |
| Sequence length | 256 |
| Per-rank batch | 8 |
| Gradient accumulation | 4 |
| Peak LR | `3e-4` |
| Minimum LR | `3e-5` |
| Warmup | 200 updates |
| LR schedule | Cosine with a 4,000-update horizon |
| Weight decay | `0.10` on the same decayed parameter classes |
| AdamW betas and epsilon | `(0.9, 0.95)`, `1e-8` |
| Gradient clipping | `1.0` |
| Muon momentum | `0.95` |
| Muon polar map | Five Newton--Schulz iterations |
| Muon output calibration | `match_rms_adamw` |
| Every internal LR multiplier | `1.0` |
| Every internal WD multiplier | `1.0` |

The stronger control is exact SwiGLU+Muon, called U-S. Exact Global-RLB+Muon,
called U-R, is included to separate the activation effect from the optimizer
effect.

## Results for the requested three-method snapshot

Lower is better in every metric.

| Run | Endpoint validation loss | Loss lead over U-S | Endpoint PPL | Train pAUC, steps 1,000--4,000 | Validation pAUC, steps 1,000--4,000 | Runtime / U-S |
|---|---:|---:|---:|---:|---:|---:|
| SwiGLU+Muon (U-S) | 4.228466988 | 0 | 68.61196851 | 4.199301681 | 4.422074735 | 1.000x |
| Global-RLB+Muon (U-R) | 4.241679192 | -0.013212204 | 69.52449883 | 4.205564916 | 4.432158089 | 1.116x |
| **R06 K1** | **4.199958801** | **0.028508186** | **66.68358371** | **4.161953121** | **4.395792866** | 1.967x |
| **R05 K1** | 4.207155704 | 0.021311283 | 67.16523011 | 4.172808398 | 4.404217637 | 1.790x |
| **Group-resolved product sphere** | 4.208533764 | 0.019933224 | 67.25785159 | 4.172926977 | 4.404469943 | 1.807x |

These are single-seed discovery results. All three candidates beat U-S in
endpoint loss, endpoint PPL, train late pAUC, and validation late pAUC under
the exact shared LR/WD cell.

They remain near-misses rather than final methods:

- none reaches the required `0.20` endpoint-loss lead;
- none has a released full recursive leave-one-component-out ablation;
- none meets the eventual `1.02x` runtime gate in its discovery
  implementation.

After this three-method snapshot was requested, R08 K5 completed at loss
`4.199784756`, only `0.000174046` below R06 K1. R08 K5 is outside the requested
three methods and is mentioned only so this document does not imply that R06
K1 remains the live loss leader.

## Ablation evidence and its exact interpretation

The inherited B+C parent has direct 4,000-step leave-one-out evidence:

- removing B worsened endpoint loss, endpoint PPL, train late pAUC, and
  validation late pAUC;
- removing C also worsened all four metrics;
- C's stable conditional endpoint-loss contribution was `0.009883404`.

Both B and C pass the current ablation definition. There is no fixed `0.01`
minimum contribution gate. A component passes when its direct deletion from
the same full method produces a stable conditional deterioration in endpoint
loss and PPL, with the late validation trajectory supporting the same
direction.

This evidence establishes the retained B and C components only. It does not
establish every later R05 or R06 component. Under the active campaign rule,
full 4,000-step recursive ablation begins only after a complete candidate
achieves the required `0.20` lead together with PPL and late-validation-pAUC
improvements. None of these three methods crossed that promotion gate, so no
claim is made that the complete methods have passed ablation.

## Fairness boundary

The optimizer methods are allowed to use RLB structure to choose a direction.
They are not allowed to obtain a larger effective step through a hidden scale.
For every reported candidate:

- the model activation and forward pass are identical Global-RLB;
- peak LR, minimum LR, warmup, cosine horizon, and per-step schedule are
  identical to the controls;
- decoupled WD and parameter-class assignment are identical;
- all external and internal LR and WD scales equal one;
- Muon momentum, NS5 count, clipping, and shape calibration are identical;
- adaptive alternatives are normalized to the parent Frobenius budget;
- coordinate covariances are unit-volume normalized;
- scheduled LR and WD are applied exactly once.

The improvement is therefore attributed to optimizer direction geometry, not
to the earlier rationalOPT mistake of assigning a larger internal LR to RLB
matrices.

## Frozen implementation identities

The file currently named `optimizer_design/rlb_r06.py` is the R06 K1 wrapper
and imports `rlb_r06_revision_core.py`. The older file
`optimizer_design/rlb_r06_core.py` implements the historical group-resolved
product-sphere method. They must not be confused.

### R06 K1

```text
component                     R06:K1
revision core SHA-256         79dafe29a88d778a880115b50e0eab6471f463e29b190fbc2b4f388725dcbe07
wrapper SHA-256               9bc23f30c16debe9d5d2a403baa671f3884471606cf076214bc5012f200210a7
source-freeze SHA-256         c5f345888cfcea2b9086a88bfeb9f51f612e202eef82843cdce98920b188e496
candidate-report SHA-256      10dd8004fe997ded5939d2c13cf806f8f475cee69f1a3249bcefdedd96690305
trajectory SHA-256            c8331ae8e0b92af2c798d4246fa959809fab80bbec5e14cb8c3ca78fefbd636c
```

### R05 K1 generation one

The currently reused R05 slot contains a later revision. The R05 K1 result in
this document refers specifically to the frozen generation-one source below.

```text
component                     R05:K1
generation-one core SHA-256   57e79a23be0bb786039481446ccbab917b9b1c43ec9b16ddfefa11b7dcaccdd8
source-freeze SHA-256         40037759752e53bee2e33a392d5e91a2e2b490771f841e5a4c981ee273100a29
candidate-report SHA-256      0abf6e99220ed8bf925a7dd21181cbffddf97595e67e282fa878bf5acf661edd
trajectory SHA-256            2817ce9688e83ed64a3a59eb1a343b81f8183785d36f55fcf2766d9843da7919
```

### Historical group-resolved product sphere

```text
generation                    group_resolved_product_sphere
source-freeze SHA-256         c824cd699ed22677524e83e20826c2924efcee902af787fd321c34de57f36200
candidate-report SHA-256      03afd1cce7042996a6a145fac50e9f9b241a715108cf4c49497c765e01d9b9bd
trajectory SHA-256            ea9d0109a8e7ca0750599f297c48d0bc2261bc0fa324938eca4d86b427706cb2
```

## Claim boundary

This document specifies reproducible discovery methods and their exact
single-seed evidence. It establishes that all three completed, checksum-valid,
LR/WD-matched trajectories modestly improved over SwiGLU+Muon in the exact
M1 experiment. It does not claim the requested `0.20` lead, multiseed
robustness, complete component closure, or deployment-speed parity.
