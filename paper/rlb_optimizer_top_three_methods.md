# Top three completed Global-RLB optimizer runs

## Evidence status

This file records the three lowest endpoint validation losses among completed,
checksum-valid runs in the exact M1 discovery cell as of 2026-08-11. Opaque
slot names were reused during discovery, so every method below is identified
by its mechanism, scheduler job, and content hashes rather than by a live
filename alone.

These are **not final methods**. All three beat SwiGLU+Muon in endpoint loss,
endpoint perplexity, and late validation partial AUC, but all miss the active
`0.15` endpoint-loss gate. None has recursive full-4,000-step ablation closure
or final `1.02x` runtime closure. The second-ranked run also adds a component
that is slightly harmful relative to its literal parent; that component is
permanently retired.

## Exact ranking

Lower is better. Lead is
`loss(SwiGLU+Muon) - loss(method)`. Partial AUC is the normalized trapezoidal
average over validation evaluations from steps 1,000 through 4,000.

| Rank | Frozen method | Job | Endpoint loss | Lead | Endpoint PPL | Train pAUC | Validation pAUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| — | SwiGLU+Muon control (U-S) | frozen control | 4.228466988 | 0 | 68.61196851 | 4.199301681 | 4.422074735 |
| — | Global-RLB+Muon control (U-R) | frozen control | 4.241679192 | −0.013212204 | 69.52449883 | 4.205564916 | 4.432158089 |
| **1** | **Complete R03 + cross-role RLB frame transaction** | **878462_0** | **4.147946358** | **0.080520630** | **63.30386323** | 4.120433284 | 4.350114052 |
| **2** | **Rank 1 + paired post-polar second moment** | **881693_0** | **4.148116112** | **0.080350876** | **63.31461022** | **4.119669438** | **4.349907168** |
| **3** | **Complete R03 + RLB-conditioned attention row product** | **881377_0** | **4.148172855** | **0.080294132** | **63.31820303** | 4.120061822 | 4.350342087 |

The differences among the three endpoints are small but exact. Rank 2 is
`0.000169754` worse than its literal rank-1 parent. Rank 3 is `0.000226498`
worse than the record, while improving over literal R03 by `0.001781464`.

## Shared model and fairness cell

The primary control is the repository's ordinary SwiGLU feed-forward block
trained with Muon for eligible matrices and AdamW for the remaining
parameters. The candidate model replaces SwiGLU with Global-RLB. In each
feed-forward layer, the 4,608 hidden coordinates are partitioned into 18
groups of width 256. For group `g`,

\[
\rho_g=\sqrt{\operatorname{mean}(z_g^2)+10^{-6}},\qquad
u_g=z_g/\rho_g,\qquad
h_g=\rho_g f_{g,t}(u_g),
\]

where `f_{g,t}` is that group's trainable degree-5/degree-4 rational response.
The optimizer changes do not alter this activation, its initialization, or
the coefficient parameterization.

All rows use the same signed experiment cell:

| Item | Fixed value |
|---|---|
| Training source | checksum-locked 300,000,000-token DCLM cache |
| Validation source | disjoint checksum-locked 8,000,000-token cache |
| Updates / seed | 4,000 / 1,337 |
| Global-RLB parameters | 296,871,080 |
| SwiGLU parameters | 296,867,840 |
| Hardware | four generic A6000 GPUs per job; no node or NVLink pin |
| Sequence / per-rank batch / accumulation | 256 / 8 / 4 |
| Peak LR / minimum LR | `3e-4` / `3e-5` |
| Schedule | warmup 200, cosine horizon 4,000 |
| Weight decay | `0.10` on the matched decayed classes |
| AdamW betas / epsilon | `(0.9, 0.95)` / `1e-8` |
| Gradient clipping | `1.0` |
| Muon momentum / polar map | `0.95` / five Newton--Schulz steps |
| Muon calibration | `match_rms_adamw` |
| Internal LR and WD multipliers | every value exactly `1.0` |

Data order, batch shape, initialization, precision, DDP ownership, realized
LR trace, and weight-decay ownership are identical within the signed cell.
Rational coefficients stay on the same AdamW path in every Global-RLB run.

## Shared complete R03 parent

All three methods use complete R03 for the RLB input/output matrices. R03
inherits a 648-direction functional atlas: two paired radial directions for
each of 18 rational groups in each of 18 layers. Each direction is evaluated
through the current learned P5/Q4 response and group-RMS Jacobian on aligned
training samples.

If `S_t` is the sample-by-atlas score matrix and `c_t` is the corresponding
weight-decay cross vector, R03 replaces the current-batch functional metric
with bias-corrected, fixed-`beta2` sufficient statistics:

\[
M_t=0.95M_{t-1}+0.05S_t^\top S_t/N,
\qquad
C_t=0.95C_{t-1}+0.05c_t,
\]

\[
\bar F_t=M_t/(1-0.95^t),
\qquad
\bar c_t=C_t/(1-0.95^t).
\]

The inherited same-budget spectral solve then uses `(bar F_t, bar c_t)` with
the existing exact-gradient and Nesterov-descent certificates. This is
RLB-specific because its coordinates are the current learned rational
function tangents paired with the two structural matrices around each group.
The recurrence introduces no additional LR, WD, damping, or update budget.

Literal complete R03 ends at loss `4.149954319`, a `0.078512669` lead over
SwiGLU+Muon.

## Method 1: complete R03 plus cross-role RLB frame

For one layer and rational group, orient the incoming momentum block and the
transposed outgoing momentum block as

\[
M^{\mathrm{in}}_{l,g},M^{\mathrm{out}}_{l,g}
\in\mathbb{R}^{256\times1024}.
\]

The method stacks the two roles along the row axis and applies one fixed NS5
polar map:

\[
\begin{bmatrix}F^{\mathrm{in}}_{l,g}\\F^{\mathrm{out}}_{l,g}\end{bmatrix}
=\operatorname{NS5}\!\left(
\begin{bmatrix}M^{\mathrm{in}}_{l,g}\\M^{\mathrm{out}}_{l,g}\end{bmatrix}
\right)
\in\mathbb{R}^{512\times1024}.
\]

Thus incoming and outgoing rows surrounding the same learned rational
function compete for one spectral frame instead of receiving unrelated
polars. Across each layer, the resulting paired direction `F` is
orthogonalized against the complete R03 direction `P` and restored to exactly
the same paired Frobenius budget:

\[
E=F-\frac{\langle F,P\rangle}{\lVert P\rVert_F^2}P,
\qquad
\widehat E=E\frac{\lVert P\rVert_F}{\lVert E\rVert_F}.
\]

The two half-energy axes

\[
C_+=(P+\widehat E)/2,
\qquad
C_-=(P-\widehat E)/2
\]

sum back to literal R03. Their exact current P5/Q4/group-RMS functional JVPs
form a 36-coordinate layerwise loss model. The registered functional
transaction chooses coefficients under the unchanged total budget and
separate positive exact-gradient and Nesterov-descent checks for both matrix
roles. Scheduled LR and decoupled WD are applied once.

The three registered additions are:

1. cross-role frame coupling;
2. fixed NS5 polarization of that frame;
3. exact P5/Q4 functional allocation between the two half-energy axes.

The complete method achieved the current record, but because its lead is only
`0.080520630`, none of these three additions has recursive 4,000-step ablation
credit.

## Method 2: method 1 plus paired post-polar second moment

This run keeps method 1 literally and inserts one additional component before
its residualization and functional allocation. For each paired incoming row
and outgoing column around the same RLB hidden channel, it accumulates a
shared Euclidean row-energy second moment with the fixed `beta2=0.95`. The
bias-corrected inverse-root row scale is applied to both roles of the
cross-role polar, after which the unchanged method-1 orthogonalization,
budget closure, half-energy axes, and functional allocator execute.

This is an update-coordinate transformation, not an LR change: the downstream
closure restores the exact parent budget and every LR/WD scalar remains one.

The added component does **not** survive even the direct parent comparison:
its endpoint is `0.000169754` worse than method 1. It was therefore
permanently retired without further ablation. Method 2 appears in this file
only because its checksum-valid endpoint is numerically the second lowest
completed endpoint, not because it is a viable final optimizer.

## Method 3: complete R03 plus RLB-conditioned attention row product

This method leaves complete R03 on all RLB feed-forward matrices and changes
the complementary attention matrices. Each attention-matrix row is represented
exactly as

\[
W_i=g_i\frac{V_i}{\lVert V_i\rVert_2},
\qquad
g_i=\lVert W_i\rVert_2,\quad V_i=W_i
\quad\text{at initialization}.
\]

For the clipped effective-weight gradient `G`, with
`U_i=V_i/||V_i||_2`, the exact pullback is

\[
\nabla_{g_i}=\langle G_i,U_i\rangle,
\]

\[
\nabla_{V_i}=\frac{g_i}{\lVert V_i\rVert_2}
\left(G_i-U_i\langle G_i,U_i\rangle\right).
\]

The complete RLB-conditioned attention transaction updates tangent direction
`V` using the unchanged momentum, NS5, response-derived allocation, and matrix
budget. Row magnitude `g` uses the cell's literal Adam recurrence with betas
`(0.9,0.95)`. The effective attention weight is reconstructed and decoupled
WD is applied exactly once. There is no magnitude LR, angular LR, role gain,
threshold, or extra update.

This composition improved over literal R03 by `0.001781464` endpoint loss and
`0.008279490` validation pAUC, but its total lead remained only `0.080294132`.
It therefore received no recursive component ablation and is not a finalist.

## Ablation and runtime status

| Method | Direct component evidence | Observed full-run runtime / U-S | Status |
|---|---|---:|---|
| Method 1 | No recursive 4,000-step LOO; lead below gate | 2.611x | Near miss, not promoted |
| Method 2 | Added post-polar component is −0.000169754 vs parent | 2.374x | Added component retired |
| Method 3 | Positive vs separate R03 run, but no recursive LOO closure | 4.331x raw, node-confounded | Near miss, not promoted |

The runtime values are observed discovery-run ratios, not final optimized
certificates. The optimizer-only timing once reported for method 3 is not a
complete-training runtime certificate. No scientific equations may be changed
during any future implementation-only speed work.

## Frozen identities

| Rank | Report SHA-256 | Trajectory SHA-256 | Source-freeze SHA-256 | Preregistration SHA-256 |
|---:|---|---|---|---|
| 1 | `039d0634153857e3a7622c922c9ce52149ac252daea85446b79b2e89d0a892f4` | `0efb3b0dc24c6104c43030533bd4811c2e5770a60921db9172793478f6aecd8c` | `9e84fbcb882bbedb83abffcbcd04ae3e23d5f573d010c3c1ae5b83c6719a0b4f` | `6f1b773b81263d57aa810025daa9e7ac4ecf8a20edadaaaa06a36a10a2e1a81d` |
| 2 | `e216f9768dd5c2d38dec4ca1341dbb2af91fdc9ab86453e0d94b85fd512e1633` | `759a0537e6834c7217f7baaa8ec4561b651e84b8aa62e6b53eb190d569092b72` | `8d7e37e94a2dd669256adc091653e5f3cb81845f71beb59dff77ecf97763570f` | `3092948a3f12a552f5bb571ed6056da92d731c622a8b7015765af63f2ee80141` |
| 3 | `87997575b5f0190406a72c2d8281e4060b72e6f475ae5b1ae2912ac6e1e1b928` | `adccf06d4606c93dce0b8bf9c44a88f0598cf50be9a4dcb7f68d70f87e5ddbd9` | `77c3af57490d4d34c877a36572c661805e11481871bd7f8ab8a81a5b6caa3988` | `3a19b9de81c117f31c0747f6701e19893232395e11656d5084c8dc122fdcfe9a` |

## Claim boundary

The strongest verified evidence is a single-seed, single-dataset endpoint
lead of `0.080520630` over SwiGLU+Muon under the exact matched cell. This is a
real improvement, but it is not the requested final result. The `0.15` quality
gate, recursive component necessity, additional-seed/dataset generalization,
and final runtime closure remain open.
