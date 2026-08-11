# Top three completed Global-RLB optimizer runs

## What this document claims

This document records the three lowest endpoint validation losses among the
completed, checksum-valid runs in the exact M1 discovery cell as of
2026-08-11. It defines the executed optimizer mathematics, lists the steps in
execution order, and explains why the complete Global-RLB systems can beat
the conventional SwiGLU+Muon system.

The explanation is deliberately about **Global-RLB versus SwiGLU+Muon**. RLB
optimizer ancestry is described only where it is necessary to specify what
was executed. No performance argument is based on defeating another RLB
optimizer.

These are strong discovery results, not final methods. All three beat
SwiGLU+Muon in endpoint loss, endpoint perplexity, and late validation partial
AUC, but all miss the active `0.15` endpoint-loss gate. None therefore has the
required recursive 4,000-step leave-one-component-out ablation closure. The
mechanistic explanations below are a-priori explanations of the complete
algorithms; they are not causal claims that an unrun ablation has established.

## Verified results against SwiGLU+Muon

Lower is better. The loss lead is

\[
\Delta L=L_{\text{SwiGLU+Muon}}-L_{\text{method}}.
\]

Validation partial AUC is the normalized trapezoidal average over validation
evaluations from steps 1,000 through 4,000.

| Rank | Frozen method | Job | Endpoint loss | Loss lead | Endpoint PPL | PPL lead | Validation pAUC |
|---:|---|---:|---:|---:|---:|---:|---:|
| — | SwiGLU+Muon control | frozen control | 4.228466988 | 0 | 68.61196851 | 0 | 4.422074735 |
| **1** | **Complete R03 + cross-role RLB frame transaction** | **878462_0** | **4.147946358** | **0.080520630** | **63.30386323** | **5.30810528** | **4.350114052** |
| **2** | **Method 1 + paired post-polar second moment** | **881693_0** | **4.148116112** | **0.080350876** | **63.31461022** | **5.29735829** | **4.349907168** |
| **3** | **Complete R03 + RLB-conditioned attention row product** | **881377_0** | **4.148172855** | **0.080294132** | **63.31820303** | **5.29376548** | **4.350342087** |

The largest verified lead is therefore `0.080520630` loss and `5.30810528`
perplexity, achieved by Method 1.

## Exact model and fairness cell

### SwiGLU+Muon control

The conventional feed-forward block is

\[
y_{\mathrm{SwiGLU}}
=W_{\mathrm{down}}
\left(\operatorname{SiLU}(W_{\mathrm{gate}}x)
\odot W_{\mathrm{up}}x\right).
\]

Eligible matrices use ordinary Muon; remaining parameters use AdamW.

### Global-RLB candidate activation

Global-RLB uses one incoming matrix `A` and one outgoing matrix `B`. Its 4,608
hidden coordinates are partitioned into 18 independently learned groups of
width `m=256`. For layer `l` and group `g`,

\[
z_{l,g}=A_{l,g}x,\qquad
\rho_{l,g}=\sqrt{\frac{1}{m}\lVert z_{l,g}\rVert_2^2+10^{-6}},\qquad
u_{l,g}=z_{l,g}/\rho_{l,g}.
\]

The installed trainable Version-A rational function is exactly

\[
f_{l,g}(u)=
\frac{\sum_{k=0}^{5}a_{l,g,k}u^k}
{1+|b_{l,g,1}u|+|b_{l,g,2}u^2|+|b_{l,g,3}u^3|+|b_{l,g,4}u^4|}.
\]

It is initialized from a SiLU fit, but its ten coefficients per group are
trainable. The group output and layer output are

\[
h_{l,g}=\rho_{l,g}f_{l,g}(u_{l,g}),\qquad
y_{\mathrm{RLB}}=B_l\operatorname{concat}_{g=1}^{18}h_{l,g}.
\]

The exact Jacobian used by the optimizer is

\[
J_{l,g}
=\operatorname{diag}(f'_{l,g}(u_{l,g}))
+\frac{\left(f_{l,g}(u_{l,g})
-u_{l,g}\odot f'_{l,g}(u_{l,g})\right)u_{l,g}^{\mathsf T}}{m}.
\]

Thus, for a preactivation perturbation `delta z`, the exact first-order
feature perturbation is `delta h = J delta z`. This includes both the local
rational derivative and the derivative of group RMS normalization.

The activation, coefficient parameterization, and coefficient AdamW path are
identical in all three candidate runs. The optimizer changes only parameter
updates.

### Fixed comparison settings

| Item | Fixed value |
|---|---|
| Training source | checksum-locked 300,000,000-token DCLM cache |
| Validation source | disjoint checksum-locked 8,000,000-token cache |
| Updates / seed | 4,000 / 1,337 |
| Global-RLB / SwiGLU parameters | 296,871,080 / 296,867,840 |
| Model depth / residual width | 18 / 1,024 |
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
LR trace, and weight-decay ownership are matched. No candidate receives an
internal LR, WD, role, group, or phase multiplier.

## Shared mathematical primitives

### Nesterov source and NS5

After global gradient clipping, each Muon-owned matrix has gradient `G_t`,
momentum buffer

\[
M_t=0.95M_{t-1}+0.05G_t,
\]

and Nesterov source

\[
N_t=0.05G_t+0.95M_t.
\]

For a batch of wide matrices, NS5 first sets

\[
X_0=N_t/\max(\lVert N_t\rVert_F,10^{-7}),
\]

in bfloat16. It then performs exactly five iterations

\[
A_k=X_kX_k^{\mathsf T},\qquad
X_{k+1}=3.4445X_k+
\left(-4.7750A_k+2.0315A_k^2\right)X_k.
\]

This is the same fixed quintic Newton--Schulz map used by the Muon control.

### Exact RLB functional score

For a paired perturbation of the incoming matrix `delta A` and outgoing
matrix `delta B`, the exact group tangent image is

\[
Y_{l,g}
=B_{l,g}J_{l,g}(\delta A_{l,g}x)
+\delta B_{l,g}h_{l,g}.
\]

If `e_l` is the downstream cotangent at the layer output, the sampled loss
score of optimizer direction `j` is

\[
s_{n,j}=\langle e_{n,l},Y_{n,j}\rangle.
\]

These are loss-action coordinates of the current learned RLB function, not
generic matrix norms.

### Complete R03 subroutine

All three methods execute complete R03 on the RLB feed-forward matrices. Its
ordered mathematical steps are:

1. For every hidden coordinate, form radial gradients for its incoming row
   norm and paired outgoing column norm, then apply matched Adam moments with
   betas `(0.9,0.95)`.
2. For the two radial variables, measure the exact current RLB response
   signals

   \[
   r_A=z_i\left[f'_i(u)+
   \frac{(f_i(u)-u_i f'_i(u))u_i}{m}\right],
   \qquad r_B=\rho f_i(u).
   \]

   Form their empirical `2 x 2` Gram matrix and apply its Moore--Penrose
   inverse square root to the two-role radial Adam direction. There is no
   damping or learned threshold.
3. Lift that two-role radial direction back to the incoming and outgoing
   matrices, remove its component along the inherited feasible direction,
   and restore the exact paired Frobenius budget.
4. Form orthogonal sum/difference axes. Across 18 layers, 18 groups, and two
   axes, this gives `2 x 18 x 18 = 648` exact RLB functional directions.
5. Compute every axis image with the current `P5/Q4` function and the exact
   group-RMS Jacobian above. Reduce aligned per-token scores across the four
   ranks and form

   \[
   F_t=S_t^{\mathsf T}S_t/N,
   \qquad
   c_t=S_t^{\mathsf T}s_{\mathrm{WD},t}/N,
   \]

   where `s_WD,t` is the loss score of the single scheduled weight-decay
   image.
6. Update the persistent functional statistics using the already fixed
   `beta2=0.95`:

   \[
   \begin{aligned}
   M_t^{F}&=0.95M_{t-1}^{F}+0.05F_t,\\
   M_t^{c}&=0.95M_{t-1}^{c}+0.05c_t,\\
   \bar F_t&=M_t^{F}/(1-0.95^t),\\
   \bar c_t&=M_t^{c}/(1-0.95^t).
   \end{aligned}
   \]

7. Solve the inherited equality-constrained quadratic transaction in the 648
   coordinates. The all-ones coefficient vector is the first feasible point;
   the total paired squared Frobenius budget is unchanged. Accept a candidate
   only if it is finite, improves the registered functional surrogate,
   satisfies the exact budget, and has positive exact-gradient and Nesterov
   descent separately for both matrix roles. Otherwise use the feasible
   parent direction.
8. Apply `match_rms_adamw`, the scheduled LR, and decoupled WD exactly once.
   The RLB rational coefficients remain on their unchanged AdamW path.

In the complete training optimizer stack, the R03 router is paired with one
RLB-conditioned attention transaction. That transaction forms two same-budget
attention directions from the current Nesterov source: an
RLB-response-routed adaptive spectral direction `U_6` and an
RLB-response-routed coordinate-sign spectral direction `U_5`. If
`alpha_l in [0,1]` is the current RLB response congruence for layer `l`, the
attention transaction uses

\[
D_l=\operatorname{Norm}_{\lVert P_l\rVert_F}
\left(\alpha_l U_{6,l}+\sqrt{1-\alpha_l^2}\,U_{5,l}\right),
\]

with exact endpoint handling, equal budget, and positive Nesterov-descent
checks. Attention WD and LR are then applied once.

## Method 1: complete R03 plus cross-role RLB frame

Method 1 is the best completed method. Let `C=B^T`, so the incoming and
transposed-outgoing Nesterov blocks surrounding the same rational group have
the common shape

\[
N^A_{l,g},N^C_{l,g}\in\mathbb{R}^{256\times1024}.
\]

### Executed steps

1. Execute complete R03 and obtain its paired feed-forward direction
   `P=(P^A,P^C)` for all 18 layers.
2. For each layer and learned rational group, stack the two Nesterov roles and
   apply one NS5 map:

   \[
   \begin{bmatrix}F^A_{l,g}\\F^C_{l,g}\end{bmatrix}
   =\operatorname{NS5}\!\left(
   \begin{bmatrix}N^A_{l,g}\\N^C_{l,g}\end{bmatrix}
   \right)
   \in\mathbb{R}^{512\times1024}.
   \]

   The same spectral frame therefore contains both matrices surrounding the
   same learned function.
3. For each whole layer, use the paired inner product

   \[
   \langle X,Y\rangle_{A,C}
   =\langle X^A,Y^A\rangle_F+\langle X^C,Y^C\rangle_F
   \]

   and remove the R03 component:

   \[
   E=F-\frac{\langle F,P\rangle_{A,C}}
   {\lVert P\rVert_{A,C}^2}P.
   \]

4. Restore exactly the R03 layer budget:

   \[
   \widehat E=E\frac{\lVert P\rVert_{A,C}}{\lVert E\rVert_{A,C}}.
   \]

   Hence `P` and `E_hat` are orthogonal and have equal norm.
5. Form two half-energy axes

   \[
   C_+=(P+\widehat E)/2,\qquad
   C_-=(P-\widehat E)/2.
   \]

   They satisfy `C_+ + C_- = P` and
   `||C_+||^2 = ||C_-||^2 = ||P||^2/2`.
6. Compute the exact current-rational/group-RMS tangent image of each axis.
   There are two axes for each of 18 layers, hence 36 loss coordinates.
7. Solve one current-batch functional transaction for the 36 coefficients.
   Preserve the exact total paired budget; require positive exact-gradient and
   Nesterov descent separately for incoming and outgoing roles; use `P` on a
   failed geometry or solve certificate.
8. Apply the selected feed-forward direction, the complete RLB-conditioned
   attention direction, coefficient AdamW, scheduled LR, and decoupled WD,
   each exactly once.

The three registered additions are the cross-role frame, its fixed NS5
polarization, and the exact 36-coordinate P5/Q4 functional allocation.

### Why the complete method can beat SwiGLU+Muon

SwiGLU+Muon sees fixed SiLU gating and applies a generic spectral map to each
eligible matrix. It does not know that one incoming row and one outgoing
column surround the same learned nonlinear function, and it does not evaluate
candidate matrix directions through that function's current downstream-loss
Jacobian.

Method 1 has three sources of information that the conventional system lacks:

1. the activation itself learns 18 independent rational response shapes in
   every layer while retaining an almost identical parameter budget;
2. complete R03 scores same-budget directions through the exact current
   rational and group-RMS Jacobian and retains their functional score geometry
   across batches; and
3. the added frame makes incoming and outgoing directions around the same
   learned function compete in one spectral row space, then lets the exact
   functional loss model decide how much of that frame to use.

This is a direct mechanism for choosing directions predicted to change the
actual learned RLB feature usefully, rather than choosing from separate matrix
spectra alone. It does so without increasing the scheduled LR or WD and while
enforcing its fixed registered update budget. The complete system achieved
loss `4.147946358` versus `4.228466988` for SwiGLU+Muon. The run proves the
full-system lead; it does not yet prove that each of the three new frame
components is individually necessary.

## Method 2: Method 1 plus paired post-polar second moment

### Executed steps

1. Execute Method 1 through construction of the cross-role post-polar frame.
2. For each paired incoming row and outgoing column belonging to the same RLB
   hidden channel, update one shared Euclidean row-energy second moment using
   the fixed `beta2=0.95`.
3. Apply the same bias-corrected inverse-root row scale to both roles.
4. Execute Method 1's unchanged whole-layer residualization, exact budget
   closure, half-energy axes, 36-coordinate P5/Q4 functional allocation,
   rolewise descent checks, LR, and WD.

The shared scaling is an update-coordinate transformation, not an LR change:
the later norm closure restores the exact parent budget and every LR/WD
multiplier remains one.

### Why it leads over SwiGLU+Muon, and what the result actually supports

Its full-system reason for beating SwiGLU+Muon is the same RLB learned-function
geometry and paired-role functional optimization described for Method 1. The
extra second moment was intended to suppress persistent channel-energy
imbalance before functional allocation. However, it made endpoint loss
`0.000169754` worse than Method 1. It is therefore permanently retired and
must not be credited for the lead. Method 2 remains in this table only because
its complete endpoint is numerically the second lowest completed endpoint.

## Method 3: complete R03 plus RLB-conditioned attention row product

Method 3 executes complete R03 on the feed-forward matrices. It changes the
coordinates in which the complete RLB-conditioned attention transaction acts.

For each effective attention row `W_i`, store a positive magnitude `g_i` and
an unconstrained direction variable `V_i`:

\[
W_i=g_i\frac{V_i}{\lVert V_i\rVert_2},\qquad
g_i=\lVert W_i\rVert_2,\qquad V_i=W_i
\]

at initialization. With `U_i=V_i/||V_i||_2` and clipped effective-weight
gradient `G_i`, the exact pullback is

\[
\nabla_{g_i}L=\langle G_i,U_i\rangle,
\]

\[
\nabla_{V_i}L=
\frac{g_i}{\lVert V_i\rVert_2}
\left(G_i-U_i\langle G_i,U_i\rangle\right).
\]

The second expression is tangent to the direction sphere because
`<nabla_V L,V>=0`.

### Executed steps

1. Execute complete R03 on all RLB feed-forward matrices and produce the
   current learned-response statistics used by its attention route.
2. Before the attention optimizer step, reconstruct `(g_i,V_i)` from the
   stored state and replace each effective attention gradient by the exact
   tangent gradient above.
3. Update `V` with the complete RLB-conditioned same-budget attention
   transaction

   \[
   D_l=\operatorname{Norm}_{\lVert P_l\rVert_F}
   \left(\alpha_l U_{6,l}+\sqrt{1-\alpha_l^2}\,U_{5,l}\right).
   \]

4. Update each scalar `g_i` with literal bias-corrected Adam using betas
   `(0.9,0.95)`, epsilon `1e-8`, and the same scheduled LR.
5. Reconstruct

   \[
   W_i^{\mathrm{preWD}}=g_i^{\mathrm{new}}
   \frac{V_i^{\mathrm{new}}}{\lVert V_i^{\mathrm{new}}\rVert_2}.
   \]

6. Apply the single effective-weight decoupled decay term

   \[
   W_i^{\mathrm{new}}
   =W_i^{\mathrm{preWD}}-\eta_t\lambda W_i^{\mathrm{old}},
   \]

   then synchronize stored magnitude with the norm of the resulting effective
   row. No decay is applied to latent `V` inside the direction transaction.
7. Restore the original effective gradient for bookkeeping. All non-attention,
   non-RLB parameters follow their unchanged AdamW paths.

### Why the complete method can beat SwiGLU+Muon

Ordinary SwiGLU+Muon applies one matrix update geometry to an attention row,
although row norm and row orientation play different roles. Method 3 assigns
the scalar norm to Adam and the tangent orientation to a spectral transaction
whose mixing coefficient comes from the current learned RLB response. The
feed-forward update is simultaneously selected through the exact learned
rational-function score geometry. Thus the attention direction and the
feed-forward direction are conditioned on the same current learned nonlinear
feature system, something fixed-SiLU ordinary Muon does not expose.

The complete system achieved loss `4.148172855`, a `0.080294132` lead over
SwiGLU+Muon. This establishes the composite result only; the row-product
component has not received 4,000-step leave-one-out credit.

## Ablation and runtime status

| Method | Direct component evidence | Observed full-run runtime / SwiGLU+Muon | Status |
|---|---|---:|---|
| Method 1 | No recursive 4,000-step LOO; lead below `0.15` gate | 2.611x | Best completed near miss; not promoted |
| Method 2 | Added post-polar component is `0.000169754` worse than Method 1 | 2.374x | Added component permanently retired |
| Method 3 | No recursive 4,000-step LOO; lead below `0.15` gate | 4.331x raw, node-confounded | Near miss; not promoted |

The runtime values are discovery-run totals, not final optimized certificates.
Method 3's separate optimizer-only timing is not a complete-training runtime
certificate. Future implementation-only speed work may not change any of the
scientific equations above.

## Frozen identities

| Rank | Report SHA-256 | Trajectory SHA-256 | Source-freeze SHA-256 | Preregistration SHA-256 |
|---:|---|---|---|---|
| 1 | `039d0634153857e3a7622c922c9ce52149ac252daea85446b79b2e89d0a892f4` | `0efb3b0dc24c6104c43030533bd4811c2e5770a60921db9172793478f6aecd8c` | `9e84fbcb882bbedb83abffcbcd04ae3e23d5f573d010c3c1ae5b83c6719a0b4f` | `6f1b773b81263d57aa810025daa9e7ac4ecf8a20edadaaaa06a36a10a2e1a81d` |
| 2 | `e216f9768dd5c2d38dec4ca1341dbb2af91fdc9ab86453e0d94b85fd512e1633` | `759a0537e6834c7217f7baaa8ec4561b651e84b8aa62e6b53eb190d569092b72` | `8d7e37e94a2dd669256adc091653e5f3cb81845f71beb59dff77ecf97763570f` | `3092948a3f12a552f5bb571ed6056da92d731c622a8b7015765af63f2ee80141` |
| 3 | `87997575b5f0190406a72c2d8281e4060b72e6f475ae5b1ae2912ac6e1e1b928` | `adccf06d4606c93dce0b8bf9c44a88f0598cf50be9a4dcb7f68d70f87e5ddbd9` | `77c3af57490d4d34c877a36572c661805e11481871bd7f8ab8a81a5b6caa3988` | `3a19b9de81c117f31c0747f6701e19893232395e11656d5084c8dc122fdcfe9a` |

## Evidence boundary

The strongest verified evidence is one dataset and one seed in the exact
4,000-step M1 cell. Method 1 leads SwiGLU+Muon by `0.080520630` endpoint loss,
but the `0.15` gate, recursive component necessity, additional-seed/dataset
generalization, and final runtime closure remain open.
