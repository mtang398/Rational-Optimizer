# Four quality-supported optimized Global-RLB methods

Updated 2026-08-27. The filename is retained so existing links do not break.

## What this document specifies

This document gives a reproducible, mathematics-first specification of the
four optimized methods currently tracked: **Method 1, Method 2, Method 3, and
R01**. Each method is described in the following order:

1. the design problem and mathematical intuition;
2. the exact objects being constructed and the meaning of every symbol;
3. the complete update transaction;
4. the runtime optimization, after the scientific construction is clear;
5. the completed quality and timing evidence.

An optimization is explicitly labeled either **strict** or **quality-tested**.
A strict optimization preserves the selected direction, decoded update,
optimizer state, learning rate, and weight decay, and qualifies by bitwise or
equivalent distributed gates. A quality-tested optimization deliberately
changes a finite-precision realization, a metric approximation, or an update
schedule and therefore receives a fresh endpoint run. No unfinished timing or
quality run is reported as a result.

The four methods occupy two evaluation cells and their raw losses must not be
mixed:

- Method 1--3: DCLM, seed 1,337, 300M training tokens, 4,000 updates.
- R01: FineWeb-Edu, seed 2,027, 300M training tokens, 9,150 updates.

Within an evaluation cell, the loss lead over the matched conventional
control is

\[
\Delta L=L_{\mathrm{SwiGLU+Muon}}-L_{\mathrm{method}}.
\]

Lead retention is always measured against the **original method's completed
endpoint lead in the same cell**:

\[
R_{\mathrm{lead}}=100
\frac{\Delta L_{\mathrm{optimized}}}
{\Delta L_{\mathrm{original}}}.
\]

It is never measured against a direct runtime parent, an incomplete
same-step trajectory, or a different method.

## Completed quality results

Lower loss, perplexity, and validation partial AUC are better. Validation
partial AUC is the normalized trapezoidal mean over checkpoints from step
1,000 through the endpoint.

### DCLM / seed 1,337 / 4,000 updates

| System | Endpoint loss | Lead vs SwiGLU+Muon | Original lead retained | Endpoint PPL | Validation pAUC | Original pAUC lead retained |
|---|---:|---:|---:|---:|---:|---:|
| SwiGLU+Muon | 4.228277206 | 0 | -- | 68.59894849 | 4.422161321 | -- |
| **Method 1 optimized** | **4.150364876** | **+0.077912331** | **96.9893%** | **63.45715005** | **4.361084549** | **84.7732%** |
| **Method 2 optimized** | **4.156803131** | **+0.071474075** | **89.1630%** | **63.86702140** | **4.360444558** | **85.4162%** |
| **Method 3 optimized quality frontier** | **4.144484997** | **+0.083792210** | **104.6038%** | **63.08512449** | **4.338377273** | **116.6596%** |

The original endpoint-lead anchors are `+0.080330849` for Method 1,
`+0.080161095` for Method 2, and `+0.080104351` for Method 3.

### FineWeb-Edu / seed 2,027 / 9,150 updates

| System | Endpoint loss | Lead vs SwiGLU+Muon | Original lead retained | Endpoint PPL | Validation pAUC | Original pAUC lead retained |
|---|---:|---:|---:|---:|---:|---:|
| SwiGLU+Muon | 3.690944910 | 0 | -- | 40.08270362 | 3.928479798 | -- |
| **R01 optimized** | **3.635784388** | **+0.055160522** | **99.0822%** | **37.93159430** | **3.878307914** | **102.0808%** |

The original R01 endpoint-lead anchor is `+0.055671453`.

## Qualified runtime frontier

Every number below is a direct ratio to SwiGLU+Muon from a signed matched
runtime report. The **exact total-time ratio is primary**. A ratio of
`1.022753x` means 2.2753% more total time than SwiGLU+Muon, not 2.2753% faster.

| Method | Exact total-time ratio | Total overhead | Median ratio | p95 ratio | Runtime evidence shape |
|---|---:|---:|---:|---:|---|
| **Method 1** | **1.132811x** | **+13.2811%** | 1.110206x | 1.137827x | Total/median: qualified phase4-to2 Graph64 compiled-score executor; p95: separately qualified SplitGraph64 executor |
| **Method 2** | **1.022753x** | **+2.2753%** | 0.997846x | 1.075310x | Total/median: padded compiled INT8 owner transport; p95: separately qualified paired-attention async-padded executor |
| **Method 3** | **1.048122x** | **+4.8122%** | 1.038522x | 1.075682x | One quality-frontier executor: head-group polar plus strict endpoint/blend elisions |
| **R01** | **1.036097x** | **+3.6097%** | 1.028826x | 1.023006x | Total/p95: qualified lean router/attention executor; median: separately qualified compiled-transport executor |

This is a **per-metric qualified portfolio**, not a claim that the independently
best total, median, and p95 values always came from one executable. Method 1's
p95 executor retained 96.1957% of its original endpoint lead
(`+0.077274799`). Method 2's and R01's alternate entries are strict execution
descendants with signed qualification. No ratio is created by multiplying
measurements from different jobs.

## Shared system construction

### Model inventory and notation

The evaluated transformer has 18 layers, model width `d=1024`, 16 attention
heads, sequence length 256, and 32,768 global tokens per update. The Global-RLB
model has 296,871,080 parameters. Each feed-forward block replaces SwiGLU with
one learned rational analysis/response/synthesis map:

| Symbol | Meaning | Shape in one layer |
|---|---|---:|
| \(x_n\) | input token representation | \(1024\) |
| \(A_l\) | incoming or analysis matrix | \(4608\times1024\) |
| \(B_l\) | outgoing or synthesis matrix | \(1024\times4608\) |
| \(g\) | rational group index | \(1,\ldots,18\) |
| \(m\) | coordinates per rational group | \(256\) |
| \(A_{l,g}\) | rows of \(A_l\) belonging to group \(g\) | \(256\times1024\) |
| \(B_{l,g}\) | columns of \(B_l\) belonging to group \(g\) | \(1024\times256\) |
| \(C_{l,g}=B_{l,g}^{\mathsf T}\) | outgoing block in the same row orientation as \(A_{l,g}\) | \(256\times1024\) |
| \(e_{n,l}\) | downstream cotangent of the block output | \(1024\) |
| \(D=(D^A,D^B)\) | a paired candidate parameter direction | shapes of \(A,B\) |
| \(Y_n(D)\) | first-order output change caused by \(D\) | \(1024\) |
| \(s_n(D)\) | scalar first-order loss score \(\langle e_n,Y_n(D)\rangle\) | scalar |

The token index \(n\) ranges over batch and sequence positions. The layer
index is \(l\), the group index is \(g\), and the hidden-coordinate index
inside a group is \(i\).

### Conventional comparison system

The control feed-forward block is

\[
y_{\mathrm{SwiGLU}}=W_{\mathrm{down}}
\left(\operatorname{SiLU}(W_{\mathrm{gate}}x)
\odot W_{\mathrm{up}}x\right).
\]

Eligible matrices use ordinary Muon; remaining parameters use AdamW. This
control matters because the claimed lead is specifically a full-system lead
over **SwiGLU+Muon**, under the same data order, token budget, learning-rate
schedule, weight decay, clipping, and hardware cell.

### Learned P5/Q4 response

For token \(n\), layer \(l\), and group \(g\), define

\[
z_{n,l,g}=A_{l,g}x_n,
\qquad
\rho_{n,l,g}=\sqrt{\frac{1}{m}\lVert z_{n,l,g}\rVert_2^2+10^{-6}},
\qquad
u_{n,l,g}=z_{n,l,g}/\rho_{n,l,g}.
\]

The scalar P5/Q4 response for that group is applied elementwise:

\[
f_{l,g}(u)=
\frac{\sum_{k=0}^{5}a_{l,g,k}u^k}
{1+\sum_{k=1}^{4}|b_{l,g,k}|\,|u|^k}.
\]

The group feature and block output are

\[
h_{n,l,g}=\rho_{n,l,g}f_{l,g}(u_{n,l,g}),
\qquad
y_{n,l}=\sum_{g=1}^{18}B_{l,g}h_{n,l,g}.
\]

The RMS factor \(\rho\) makes the learned response operate on a normalized
coordinate \(u\) while restoring the feature scale afterward. The rational
coefficients are trained parameters. Thus the response shape itself can
change with training; it is not a fixed activation chosen before training.

### Exact tangent through group RMS normalization

Let \(f=f_{l,g}\), suppress token/layer/group subscripts, and let
\(f'(u)\) denote the elementwise derivative. The Jacobian of
\(h(z)=\rho f(z/\rho)\) is

\[
J=
\operatorname{diag}(f'(u))+
\frac{\left(f(u)-u\odot f'(u)\right)u^{\mathsf T}}{m}.
\]

The diagonal term is the direct coordinate response. The rank-one term is
the coupling introduced by group RMS normalization: perturbing one coordinate
changes \(\rho\), which changes every normalized coordinate in the group.
Dropping this term would no longer be the tangent of the installed forward
map.

For a paired direction \(D=(D^A,D^B)\), the group contribution to the
first-order output change is

\[
Y_{n,l,g}(D)=
B_{l,g}J_{n,l,g}(D^A_{l,g}x_n)
+D^B_{l,g}h_{n,l,g}.
\]

The first term is the change caused by moving the analysis matrix through the
current rational response. The second is the change caused by moving the
synthesis matrix. The complete layer image is the sum over groups.

The scalar downstream-loss score is

\[
s_{n,j}=s_n(D_j)
=\left\langle e_{n,l},Y_{n,l}(D_j)\right\rangle.
\]

This is the central conversion in Global-RLB: a parameter-space direction is
mapped through the **current learned function** and evaluated by its
first-order action on the token loss.

### Gradient source, clipping, momentum, and NS5

The global gradient-norm clip factor is applied consistently to the matrix
gradients and to the per-token cotangents used to construct the functional
scores. This makes the score metric describe the same clipped transaction
that will actually be applied.

For a Muon-owned matrix, with clipped gradient \(G_t\), the stored momentum
and current Nesterov source are

\[
M_t=0.95M_{t-1}+0.05G_t,
\qquad
N_t=0.05G_t+0.95M_t.
\]

For each wide matrix batch, normalize

\[
X_0=N_t/\max(\lVert N_t\rVert_F,10^{-7})
\]

in bfloat16 and execute exactly five Newton--Schulz polynomial steps:

\[
A_k=X_kX_k^{\mathsf T},
\qquad
X_{k+1}=3.4445X_k+
\left(-4.7750A_k+2.0315A_k^2\right)X_k.
\]

All four optimized methods keep this polynomial and all five iterations. The
runtime work did not reduce or replace NS5.

### Functional metric and same-budget allocation

Suppose a method has constructed \(p\) candidate paired directions
\(D_1,\ldots,D_p\). Form the score matrix

\[
S_{n,j}=s_n(D_j),
\qquad
F=\frac{1}{N}S^{\mathsf T}S.
\]

Here \(F\) is an empirical loss-space Fisher/Gram matrix. Its diagonal says
how strongly one direction acts on token losses. Its off-diagonal entries say
whether two directions act together, oppose one another, or are redundant in
the current batch.

Let \(s_{\mathrm{WD}}\) be the score of the scheduled decoupled weight-decay
image and define

\[
c=\frac{1}{N}S^{\mathsf T}s_{\mathrm{WD}}.
\]

Let \(b_j\) be the exact clipped-gradient linear gain of direction \(D_j\),
and let

\[
w_j=\lVert D_j^A\rVert_F^2+\lVert D_j^B\rVert_F^2,
\qquad
B_0=\sum_{j=1}^{p}w_j.
\]

The coefficients \(\alpha\) solve

\[
\min_{\alpha}
-\eta b^{\mathsf T}\alpha+
\frac{\eta^2}{2}
\left(\alpha^{\mathsf T}F\alpha+2c^{\mathsf T}\alpha\right)
\quad\text{subject to}\quad
\sum_{j=1}^{p}w_j\alpha_j^2=B_0.
\]

The linear term rewards immediate descent. The Fisher term penalizes a
combination predicted to move token losses too strongly or redundantly. The
cross term accounts for interaction with scheduled weight decay. The equality
constraint fixes the weighted Frobenius energy to the registered parent
budget, so the solve cannot manufacture a gain merely by increasing the
effective learning rate.

### Exact secular solve and safety certificates

Let \(W=\operatorname{diag}(w)\), and transform to
\(y=W^{1/2}\alpha\). Define

\[
H=W^{-1/2}FW^{-1/2},
\qquad
r=W^{-1/2}(b/\eta-c).
\]

The constrained stationary equation is

\[
(H+\lambda I)y=r,
\qquad
\lVert y\rVert_2^2=B_0.
\]

The implementation symmetrizes and eigendecomposes \(H\), then finds the
secular multiplier \(\lambda\) with a fixed 64-round bracketed solve. It
handles the singular hard case in the eigenspace rather than silently adding
an unregistered learning-rate multiplier. Finally,
\(\alpha=W^{-1/2}y\).

The all-ones allocation is the literal parent point. It is evaluated first,
wins exact ties, and is returned whenever any registered condition fails:

- a direction, score, metric, or solution is non-finite;
- the candidate does not strictly improve the registered surrogate;
- exact clipped-gradient descent is not positive;
- current Nesterov descent is not positive;
- the weighted budget residual exceeds its floating-point tolerance;
- a method-specific orthogonality or rolewise descent certificate fails.

The scheduled learning rate and decoupled weight decay are applied exactly
once after selection.

### Parent coordinates used by the methods

The names below describe construction stages, not additional comparison
methods in the result table.

**Current R02 direction.** For every layer and rational group, current R02
constructs one paired incoming/outgoing same-budget direction from the
clipped Nesterov source, response geometry, and an optimizer-family router.
Across 18 layers and 18 groups, this gives 324 parent coordinates.

One router branch interpolates between a spectral parent \(P\) and a
same-budget coordinate-sign direction \(Q\). If \(p\in[0,1]\) is group
participation, the unnormalized source is

\[
\sqrt{p}\,P+\sqrt{1-p}\,Q,
\]

then it is normalized back to \(\lVert P\rVert_F\). A later response chord
mixes two already equal-budget endpoints \(U_6,U_5\). With response
congruence \(a\in[0,1]\),

\[
U_{\mathrm{R02}}=
\operatorname{Norm}_{\lVert P\rVert_F}
\left(aU_6+\sqrt{1-a^2}\,U_5\right).
\]

The two formulas are intentionally distinct: participation uses squared-energy
amplitudes \(\sqrt{p},\sqrt{1-p}\); the endpoint chord uses
\(a,\sqrt{1-a^2}\). Exact limits return the corresponding endpoint.

**R01 loss-space allocation.** R01 keeps the 324 current-R02 group
directions, computes all of their downstream scores, and solves the global
same-budget transaction above. Its full metric includes cross-layer blocks.

**Paired radial response atlas.** An RLB hidden coordinate has two radial
roles: scaling a row of \(A\), and scaling the matching column of \(B\). Their
exact self-response signals are

\[
r_A=z_i\left[f'_i(u)+
\frac{(f_i(u)-u_if'_i(u))u_i}{m}\right],
\qquad
r_B=\rho f_i(u).
\]

For each channel, the all-rank empirical \(2\times2\) Gram matrix of
\((r_A,r_B)\) is eigendecomposed analytically. Its Moore--Penrose inverse
square root whitens a matched-beta Adam radial direction. The two-role radial
direction is then made orthogonal to the complete R01 parent and rescaled to
the exact same group budget. If that orthogonal direction is \(E\) and the
parent is \(P\), the axes

\[
C_+=(P+E)/2,
\qquad
C_-=(P-E)/2
\]

are orthogonal equal-energy coordinates, and coefficients
\((\alpha_+,\alpha_-)=(1,1)\) reconstruct the parent exactly.

**Persistent R03 parent.** R03 retains the same paired-radial atlas but
replaces only its current-batch Fisher and weight-decay cross term with
bias-corrected exponential states:

\[
\overline F_t=0.95\overline F_{t-1}+0.05F_t,
\qquad
\widehat F_t=\frac{\overline F_t}{1-0.95^t},
\]

with the same construction for \(c\). The first step is the literal current
metric. Persistence gives the atlas a metric whose support is not limited to
one batch, while the parent remains the first feasible allocation and all
budgets, LR, WD, clipping, and NS5 remain registered.

Method 1 and Method 2 construct their novelty around this complete R03 parent.
Method 3 uses it for its periodic outer refresh. R01 uses the 324-coordinate
current-R02 loss-space allocation directly.

## Method 1: joint cross-role spectral frame

### Design intuition

The incoming block \(A_{l,g}\) and outgoing block
\(B_{l,g}\) sit on opposite sides of the **same** learned nonlinear group.
After transposing \(B_{l,g}\), both roles have shape
\(256\times1024\). Treating them as unrelated matrices would allow each role
to choose its spectral frame without seeing redundancy or complementarity in
the other role.

Method 1 therefore asks one geometric question: *what spectral frame is
obtained when the analysis row and matching synthesis column compete in a
single joint polar problem?* It then does not apply that frame blindly. The
new component is residualized against the complete function-aware R03 parent,
closed to the same budget, and exposed to the downstream-loss allocation as a
second orthogonal choice.

The design separates two responsibilities:

- joint NS5 supplies a cross-role matrix geometry;
- the exact RLB tangent and loss-space metric decide whether and how much of
  that geometry should be used on the current batch.

### Exact construction

Let

\[
N^A_{l,g},N^C_{l,g}\in\mathbb R^{256\times1024},
\qquad C_{l,g}=B_{l,g}^{\mathsf T},
\]

be the clipped current Nesterov sources for the analysis and transposed
synthesis roles. Stack them vertically and run one unchanged NS5 map:

\[
\begin{bmatrix}F^A_{l,g}\\F^C_{l,g}\end{bmatrix}
=\operatorname{NS5}\!\left(
\begin{bmatrix}N^A_{l,g}\\N^C_{l,g}\end{bmatrix}
\right).
\]

The stack has shape \(512\times1024\). A single NS5 operation means the two
roles share the same row-orthogonalization problem. Split its output back into
the two 256-row blocks. This gives one paired frame block
\(F_{l,g}=(F^A_{l,g},F^C_{l,g})\) for each group.

The 18 group blocks are then reassembled into the full layer direction
\(F_l=(F^A_l,F^C_l)\). Let \(P_l=(P^A_l,P^C_l)\) be the complete R03 parent
for that layer. The layer-level paired Frobenius inner product is

\[
\langle X_l,Y_l\rangle_{A,C}
=\sum_{g=1}^{18}\left(
\langle X^A_{l,g},Y^A_{l,g}\rangle_F+
\langle X^C_{l,g},Y^C_{l,g}\rangle_F\right).
\]

Remove the parent component:

\[
E_l=F_l-
\frac{\langle F_l,P_l\rangle_{A,C}}
{\lVert P_l\rVert_{A,C}^2}P_l.
\]

If \(E_l\) is finite and has non-negligible energy, close it to the exact
paired **layer** budget:

\[
\widehat E_l=E_l
\frac{\lVert P_l\rVert_{A,C}}
{\lVert E_l\rVert_{A,C}}.
\]

The implementation verifies, to its registered floating-point tolerances,

\[
\langle P_l,\widehat E_l\rangle_{A,C}=0,
\qquad
\lVert\widehat E_l\rVert_{A,C}^2=\lVert P_l\rVert_{A,C}^2.
\]

It then creates

\[
C_{l,+}=(P_l+\widehat E_l)/2,
\qquad
C_{l,-}=(P_l-\widehat E_l)/2.
\]

Because \(P_l\perp\widehat E_l\) and their squared norms match, each layer
axis has exactly half that layer parent's squared norm. The literal parent
remains feasible: \(C_{l,+}+C_{l,-}=P_l\). Across 18 layers this produces 36
functional axes. Thus NS5 couples the two roles within every rational group,
while residualization, budget closure, and coefficient selection act at the
paired layer level.

### Complete update transaction

For one optimizer transition, Method 1 performs the following transaction:

1. Backpropagate the token loss, compute the global clip factor, and form the
   clipped gradients and per-token cotangents.
2. Update Muon momentum and construct \(N^A,N^C\).
3. Construct the complete R03 parent, including paired radial response
   whitening, its persistent metric, exact parent budget, and fallback point.
4. Jointly polarize the stacked analysis/synthesis sources with NS5.
5. Residualize the joint frame against R03 and form \(C_+,C_-\).
6. Push both axes through the exact P5/Q4 Jacobian and downstream cotangents,
   producing 36 score coordinates.
7. Solve the 64-round equality-constrained functional allocation and require
   finite, budget, exact-gradient, Nesterov, and rolewise descent certificates.
8. Reconstruct the selected \(A\) and \(C=B^{\mathsf T}\) directions and apply
   them once with the scheduled LR and WD.
9. Execute the RLB-conditioned attention transaction and AdamW updates for
   the remaining parameters.

### How Method 1 was optimized

The expensive work is the complete R03 hierarchy, its response reductions,
the 36-axis score construction, repeated 64-round secular kernels, and
publication of large owner updates. The optimized quality executor addresses
those costs as follows.

**Quality-tested phase4-to2 schedule.** A complete Method 1/R03 refresh occurs
every four transitions through step 1,500 and every two transitions after
that. This yields 1,625 complete refreshes over 4,000 updates. Every other
transition still computes a current-gradient exact global R01 allocation; it
does not replay a cached parameter update. Replacing a complete outer refresh
with current R01 changes the update schedule, so the full 4,000-step quality
run is the evidence.

**Exact Graph64 component and quality-tested SplitGraph64 alternative.** The
fixed 64-round secular iteration is captured as a CUDA graph and replayed
without changing the rounds or equations. Graph64 itself is an exact execution
component, not a shorter solve. Splitting that graph changes floating-point
association; the p95 SplitGraph64 alternative therefore has its own completed
4,000-step quality result (`+0.077274799`, or `96.1957%` retained) rather than
inheriting Method 1's primary endpoint.

**Quality-tested compiled score programs.** The already specified
response-adjoint, paired-radial, and frame-score calculations are compiled as
fixed programs. Compilation changes floating-point association even when the
symbolic map is unchanged, so the compiled-score executor is covered by the
fresh endpoint run rather than assumed to inherit quality.

**Quality-tested owner publication.** Complete layers are assigned to four
owners with counts `(5,5,4,4)`. Owner deltas use block-256 symmetric INT8 with
error feedback. For an unquantized delta \(\delta_t\) and residual \(r_t\),

\[
q_t=Q(\delta_t+r_t),
\qquad
r_{t+1}=\delta_t+r_t-q_t.
\]

All ranks consume the same decoded \(q_t\). Error feedback keeps the
unpublished quantization residual in the next owner transaction; it does not
alter LR or WD. Because the decoded update is numerical rather than bitwise
equal to an FP32 publication, it belongs to the fresh quality-tested executor.

**Strict certificate/telemetry elision.** Reductions whose values are used
only for ordinary-step telemetry are omitted only when they cannot affect the
selected direction, state, or fallback. Complete paths remain on registered
telemetry transitions.

### Why this can lead over SwiGLU+Muon

SwiGLU+Muon uses a fixed activation and polarizes eligible matrices without an
exact learned-function tangent or a joint analysis/synthesis frame. Method 1
learns the P5/Q4 response, constructs a cross-role spectral alternative around
that response, and accepts its allocation only through a same-budget
downstream-loss transaction with descent and parent-fallback certificates.
The optimized system completed with endpoint lead `+0.077912331`, retaining
`96.9893%` of the original Method 1 endpoint lead.

## Method 2: shared post-polar channel memory

### Design intuition

Method 1 couples the analysis and synthesis roles in one instantaneous joint
polar frame. Method 2 adds a second question: *which paired hidden channels
remain persistently energetic after that joint polarization?*

The key object is a hidden coordinate, not an isolated matrix row. Incoming
row \(A_{l,g,i,:}\) and transposed outgoing row
\(C_{l,g,i,:}=B_{l,g,:,i}^{\mathsf T}\) implement the same learned channel.
Giving them unrelated adaptive scales would break that identity. Method 2
therefore measures their post-polar energy together, keeps one shared
second-moment state, and multiplies both roles by the same inverse-root scale.

The measurement is made **after** joint NS5 because the desired state is the
energy in the already selected cross-role frame, not raw-gradient magnitude.
It is made **before** residualization and budget closure because the scale is
intended to change the candidate geometry, while the later closure prevents it
from acting as an LR multiplier.

### Exact construction

Begin with Method 1's paired post-polar blocks
\(F^A_{l,g},F^C_{l,g}\). For channel \(i\), define the shared observation

\[
q_{t,l,g,i}=\frac{1}{2}\left(
\operatorname{mean}_{k}(F^A_{t,l,g,i,k})^2+
\operatorname{mean}_{k}(F^C_{t,l,g,i,k})^2
\right).
\]

Maintain the exponential second moment

\[
v_{t,l,g,i}=0.95v_{t-1,l,g,i}+0.05q_{t,l,g,i},
\qquad
\widehat v_{t,l,g,i}=\frac{v_{t,l,g,i}}{1-0.95^t}.
\]

The shared inverse-root channel scale is

\[
r_{t,l,g,i}=\frac{1}{\sqrt{\widehat v_{t,l,g,i}}+10^{-8}}.
\]

Apply the same scalar to both roles:

\[
\widetilde F^A_{l,g,i,:}=r_{t,l,g,i}F^A_{l,g,i,:},
\qquad
\widetilde F^C_{l,g,i,:}=r_{t,l,g,i}F^C_{l,g,i,:}.
\]

Reassemble the 18 scaled group blocks into the full paired layer direction
\(\widetilde F_l\), then replace \(F_l\) by \(\widetilde F_l\) in Method 1's
layer-level residualization:

\[
E_l=\widetilde F_l-
\frac{\langle\widetilde F_l,P_l\rangle_{A,C}}
{\lVert P_l\rVert_{A,C}^2}P_l,
\qquad
\widehat E_l=E_l
\frac{\lVert P_l\rVert_{A,C}}{\lVert E_l\rVert_{A,C}}.
\]

The same layer-level orthogonality, budget, and rolewise descent checks apply.
The axes \((P_l\pm\widehat E_l)/2\) again make the parent exactly feasible.
Therefore \(r_t\) changes the **orientation** of the alternative direction but
not the registered update energy.

### Complete update transaction

Method 2 follows Method 1 through the joint cross-role NS5 frame, then:

1. compute one paired post-polar energy \(q_{t,l,g,i}\) for every channel;
2. update and bias-correct its shared second moment;
3. scale the analysis and synthesis rows by the same \(r_{t,l,g,i}\);
4. residualize the scaled frame against the complete R03 parent;
5. restore the exact parent budget and form the two axes;
6. compute exact P5/Q4 tangent scores for all axes;
7. solve the certified same-budget allocation;
8. apply the selected feed-forward, attention, coefficient, LR, and WD
   transactions exactly once.

### How Method 2 was optimized

**Quality-tested outer4 schedule.** One transition in each four-step cycle
executes the complete paired-postpolar/R03 refresh. The other three execute a
current-gradient exact global R01 allocation using all-rank observations. A
reuse transition is still a newly computed update, not a cached matrix step.
This schedule changes which scientific construction is used on those three
transitions and is covered by the fresh 4,000-step quality run.

**Distributed complete-layer ownership.** On a refresh, each owner constructs
all statistics and states for its complete layers. Keeping a layer intact
avoids splitting its response metric across owners. On R01 transitions,
all-rank observations still produce the current global allocation required by
that path.

**Fixed compiled span solve.** The literal 64 secular rounds are captured in a
fixed-shape compiled program. The rounds, bracket logic, hard-case behavior,
budget, and fallback are unchanged.

**Strict compiled INT8 transport descendant.** The qualified total/median
runtime executor compiles INT8 packing, padded collective transport, remote
decode, and local application. Its distributed gates verify the same INT8
codes, FP32 scales, decoded updates, optimizer state, LR, WD, and NS5 result as
its quality parent. Quality inheritance is therefore allowed for this strict
descendant; it is not a new approximation layered onto the endpoint number.

**Strict p95 descendant.** The paired-attention async-padded executor changes
communication scheduling for tail latency while preserving the registered
update transaction and qualifying independently.

### Why this can lead over SwiGLU+Muon

SwiGLU+Muon neither learns the rational response nor represents an incoming
row and matching outgoing column as one adaptive hidden coordinate. Method 2
uses the exact learned-function tangent and a shared post-polar memory to
control persistent channel imbalance before the same-budget functional
decision. The optimized system completed with endpoint lead `+0.071474075`,
retaining `89.1630%` of the original Method 2 endpoint lead.

## Method 3: attention row-product geometry and head-group polar structure

### Design intuition

An attention row has two geometrically different degrees of freedom: its
length controls response scale, while its normalized direction controls which
feature combination the row reads or writes. A single Euclidean matrix update
mixes these radial and angular roles.

Method 3 represents every effective attention row as a product of a positive
magnitude and a point on a sphere. The magnitude receives a scalar adaptive
update; the direction receives a tangent update conditioned by the current RLB
response. The quality-frontier executor adds a second structural decision:
NS5 is applied in four-head blocks so the polar geometry follows native
attention-head organization instead of treating all Q/K/V rows as one
undifferentiated matrix.

### Exact row-product construction

For attention row \(i\), store a magnitude \(g_i>0\) and an unconstrained
latent direction \(V_i\ne0\):

\[
W_i=g_iU_i,
\qquad
U_i=\frac{V_i}{\lVert V_i\rVert_2}.
\]

Let \(G_i=\nabla_{W_i}L\) be the clipped effective gradient. The exact
pullback through the product map is

\[
\nabla_{g_i}L=\langle G_i,U_i\rangle,
\]

\[
\nabla_{V_i}L=
\frac{g_i}{\lVert V_i\rVert_2}
\left(G_i-U_i\langle G_i,U_i\rangle\right).
\]

The projected term is orthogonal to \(V_i\):
\(\langle\nabla_{V_i}L,V_i\rangle=0\). Thus, to first order, the direction
update moves along the sphere rather than redundantly changing row length.

The magnitude gradient uses bias-corrected Adam with
\((\beta_1,\beta_2)=(0.9,0.95)\) and \(\epsilon=10^{-8}\). The tangent
direction passes through the response-conditioned, same-budget attention
router described below.

### Exact response-conditioned attention router

The router has two separate normalized mixes.

First, within an optimizer-family branch, let \(P\) be the spectral direction,
\(Q\) the same-budget coordinate-sign direction, and \(p\in[0,1]\) the
response participation. It forms

\[
U_5=\operatorname{Norm}_{\lVert P\rVert_F}
\left(\sqrt p\,P+\sqrt{1-p}\,Q\right).
\]

Second, let \(U_6\) be the current response-coordinate adaptive/spectral
direction and let \(a\in[0,1]\) be response congruence. The endpoint chord is

\[
D=\operatorname{Norm}_{\lVert P\rVert_F}
\left(aU_6+\sqrt{1-a^2}\,U_5\right).
\]

Both endpoints are normalized to the literal parent's budget before the
chord, and the result is normalized back to that budget. The exact limits
\(a=1\) and \(a=0\) return \(U_6\) and \(U_5\), respectively. Descent and
budget certificates retain the literal parent if the routed endpoint is not
valid.

### Weight decay is applied once in effective-row space

The direction transaction is first evaluated with parent attention WD set to
zero. After updating \(V_i\) and the scalar magnitude state, reconstruct the
effective row \(W_i^{\mathrm{new}}\), then apply decoupled weight decay once:

\[
W_i^{\mathrm{eff}}=
W_i^{\mathrm{new}}-\eta\lambda_{\mathrm{WD}}W_i^{\mathrm{old}}.
\]

Finally reset the product state so it represents that same effective row. If
\(r^{V,\mathrm{new}}_i=\lVert V_i^{\mathrm{new}}\rVert_2\) is the direction
norm produced by the tangent transaction, then

\[
g_i\leftarrow\lVert W_i^{\mathrm{eff}}\rVert_2,
\qquad
V_i\leftarrow
r^{V,\mathrm{new}}_i
\frac{W_i^{\mathrm{eff}}}{\lVert W_i^{\mathrm{eff}}\rVert_2}.
\]

This ordering prevents WD from being applied independently to both product
factors and therefore counted twice.

### Four-head NS5 construction

Here \(d=1024\), there are 16 heads, head width \(d_h=64\), and each polar
group contains four heads, so its row or column width is
\(4d_h=256\).

For a QKV source \(X\in\mathbb R^{3d\times d}\), split Q, K, and V, then split
each into four 256-row blocks \(X_{t,j}\), where
\(t\in\{Q,K,V\}\) and \(j=1,\ldots,4\). Apply the unchanged five-step map to
each block:

\[
P_H(X)=\frac{1}{\sqrt3}
\operatorname{concat}_{t,j}\operatorname{NS5}(X_{t,j}).
\]

The twelve blocks have a total nominal polar rank of \(3d\), whereas the
original global tall-matrix map has nominal rank \(d\). Multiplying by
\(1/\sqrt3\) restores the squared-Frobenius polar-rank calibration; it is not
an LR tuning factor.

For an output source \(Y\in\mathbb R^{d\times d}\), split the input columns
into four blocks \(Y_j\in\mathbb R^{d\times256}\):

\[
P_H(Y)=\operatorname{concat}_{j=1}^{4}\operatorname{NS5}(Y_j).
\]

Their nominal ranks already sum to \(d\), so no additional scale is needed.
Both response-conditioned attention branches use the same head-group map.

### Complete update transaction

1. Materialize the effective attention rows \(W_i=g_iV_i/\lVert V_i\rVert\)
   and backpropagate through the complete model.
2. Apply the global clip factor and compute the exact magnitude and tangent
   pullbacks.
3. Update magnitude Adam states and construct the tangent Nesterov sources.
4. Construct the response participation, response congruence, family branch,
   and exact same-budget endpoint chord.
5. Apply four-head NS5 blocks to QKV and output sources and require the
   registered attention descent/budget fallback.
6. Update \(V\), reconstruct the effective rows, apply WD once in effective
   row space, and synchronize \(g,V\) with the result.
7. Execute the feed-forward RLB transaction: a complete R03 refresh on its
   registered outer transition or a current R01 transaction otherwise.
8. Apply coefficient AdamW and all remaining parameter updates once.

### How Method 3 was optimized

**Quality-tested outer8 schedule.** One transition in each eight-step cycle
executes the complete R03 refresh. The other seven execute current-gradient
compiled R01 score allocation. No cached matrix update is replayed. The
persistent R03 state advances with elapsed-time correction \(0.95^8\) across
an eight-step gap, so its decay is measured in optimizer transitions rather
than refresh calls. The schedule is covered by complete endpoint quality.

**Quality-tested head-group polar.** Four-head blocking changes the numerical
and structural polar geometry, even though each block retains NS5. It therefore
received a fresh 4,000-step run. This quality-frontier run is the source of the
reported `+0.083792210` endpoint lead.

**Compiled R01 scores and ragged INT8 publication.** Repeated score programs
are compiled. Complete owner-layer deltas are published with exact ragged owner
counts rather than forcing every rank to publish the maximum count. These
finite-precision/distributed choices are included in the quality-tested
frontier.

**Strict ordinary-endpoint elision.** On a non-refresh, non-telemetry R01
transition, the complete family route and chord are still constructed. Only
endpoint descent, budget, and angle diagnostics that R01 subsequently discards
are omitted. Complete paths remain on R03 refresh and telemetry transitions.

**Strict blend-norm reuse.** The executor reuses the already computed literal
parent norm for the ordinary normalized blend and omits only cosines read by
telemetry. Gloo and four-rank NCCL gates verified bitwise parameters and state
over complete transition windows, so the optimized executor inherits the
quality-frontier endpoint.

### Why this can lead over SwiGLU+Muon

Ordinary SwiGLU+Muon does not learn the P5/Q4 response, does not split
attention-row scale from spherical orientation, and does not condition the
orientation on response participation and congruence. Method 3 combines those
coordinates with an attention-head-aware NS5 geometry under fixed budgets and
fallbacks. The optimized quality frontier completed with endpoint lead
`+0.083792210`, or `104.6038%` of the original Method 3 endpoint lead.

## R01: downstream-loss allocation with owner-local execution

### Design intuition

Muon constructs matrix directions from parameter-space geometry. R01 adds a
different measurement: *if each rational group moved along its current paired
direction, how would those moves combine at the token loss?*

Two parameter directions can be nearly orthogonal in Frobenius space yet
produce redundant output changes. Conversely, directions in different layers
can cooperate or oppose one another on the same examples. R01 therefore maps
each current-R02 group direction through the exact learned P5/Q4 tangent and
builds its allocation metric from downstream-loss scores. The metric measures
functional interaction, while the equality constraint retains the parent's
parameter-update budget.

### Original global R01 construction

There is one paired current-R02 direction \(D_{l,g}\) for each of 18 groups in
each of 18 layers, hence

\[
p=18\times18=324.
\]

For every token \(n\), compute

\[
s_{n,(l,g)}=\left\langle e_{n,l},
B_{l,g}J_{n,l,g}(D^A_{l,g}x_n)
+D^B_{l,g}h_{n,l,g}\right\rangle.
\]

Stacking those scores gives \(S\in\mathbb R^{N\times324}\) and

\[
F=\frac1N S^{\mathsf T}S\in\mathbb R^{324\times324}.
\]

The off-diagonal block between two different layers is retained in original
R01. With exact clipped-gradient linear gains \(b\), WD cross term \(c\), and
paired weights \(w\), the global 324-coordinate secular solve selects the
same-budget allocation. The all-ones vector reconstructs the complete
current-R02 parent and is the fail-safe allocation.

### Quality-tested owner-local metric

The optimized 9,150-step version assigns complete layers to rank \(r\) by

\[
\mathcal I_r=\{l:l\bmod4=r\}.
\]

The owner counts are `(5,5,4,4)`. Rank \(r\) forms a score matrix

\[
S_r\in\mathbb R^{N\times18|\mathcal I_r|},
\]

with 90 columns on ranks 0 and 1 and 72 on ranks 2 and 3. It solves

\[
F_r=N^{-1}S_r^{\mathsf T}S_r
\]

and the same equality-constrained transaction over its complete owned
coordinates.

This partition preserves every within-owner cross-layer block and every
within-layer cross-group interaction. It deliberately omits cross-owner
Fisher blocks. That is a mathematical approximation, not a strict systems
rewrite, so its validity comes from the fresh completed 9,150-step run.

### Response-coordinate construction

For each owned layer, deterministic probes evaluate both live and frozen
P5/Q4 response factors, including the diagonal response and the rank-one RMS
coupling. These factors define the response coordinate in which the current
R02 directions and their tangent images are constructed.

If an FP32 lower-triangular factor \(L\) defines one response-coordinate map,
the direct implementation repeatedly applies triangular solves. The optimized
quality executor instead forms

\[
K=L^{-1}=\operatorname{solve\_triangular}(L,I)
\]

once for each of three registered factors, then reuses matrix products
\(Kv\) and \(K^{\mathsf T}v\) wherever the same linear map is needed. In exact
arithmetic this is the same map; in finite precision it has different rounding
and therefore remains covered by fresh quality evidence.

### Complete update transaction

1. Backpropagate, compute the global clip factor, and reconstruct the clipped
   per-token cotangents.
2. Construct one paired current-R02 direction for every group of each owned
   complete layer.
3. Evaluate deterministic live/frozen P5/Q4 response probes and build the
   exact tangent images.
4. Form \(S_r,F_r,b_r,c_r,w_r\) without mixing statistics between independent
   owned layers before the intended owner metric.
5. Execute the fixed 64-round same-budget solve, with parent tie-breaking and
   all finiteness/descent/budget fallbacks.
6. Reconstruct every selected owned-layer delta and publish it so all data
   parallel ranks apply the same decoded FP32 tensor.
7. Execute attention, coefficient AdamW, unchanged NS5, scheduled LR, and
   decoupled WD exactly once.

### How R01 was optimized

**Quality-tested complete-layer ownership.** The 324-coordinate global metric
is partitioned into owner-local 90/72-coordinate metrics as specified above.
This removes cross-owner metric construction and parallelizes complete layer
transactions. The approximation is validated by the 9,150-step quality run.

**Quality-tested batched response programs.** The four or five independent
owned-layer response reductions are placed in one compiled program. Batching
does not intentionally mix their statistics, but compilation and reduction
association can change rounding, so the endpoint run covers the executor.

**Quality-tested inverse reuse.** Three explicit FP32 inverses replace
repeated applications of the corresponding triangular solves. This reduces
kernel and solve overhead while retaining the intended linear maps; fresh
quality covers the changed numerical path.

**Strict compiled span.** Three fixed 64-round secular solves are captured in
compiled CUDA programs without changing their iteration count, bracket,
hard-case rule, budget, or fallback.

**Quality-tested block-256 signed INT4 publication.** For each 256-element
block of an owner delta \(\delta_b\), define

\[
s_b=\max_i|\delta_{b,i}|/7,
\qquad
q_{b,i}=\operatorname{clip}_{[-7,7]}
\left(\operatorname{round}(\delta_{b,i}/s_b)\right),
\qquad
\widehat\delta_{b,i}=s_bq_{b,i}.
\]

The two signed codes are shifted into nibbles and packed two per byte; FP32
block scales travel with them. Every rank, including the owner, decodes and
applies the same \(\widehat\delta\), avoiding owner/non-owner parameter drift.

The quality executor uses **ragged owner row counts `(5,5,4,4)`**. Tensor
elements inside one layer row may be padded to a block-256 boundary, but ranks
with four layers do not publish a fictitious fifth owner row. This distinction
is important: block alignment is not padded owner inventory.

**Strict runtime descendants.** The qualified total/p95 executor removes only
ordinary router/attention diagnostics that do not affect the selected update;
the qualified median executor compiles pack/decode and transport while
retaining the same INT4 codes, FP32 scales, and decoded updates. Their signed
distributed execution contracts keep optimizer equations, NS5, LR, WD, and
floating-point parameter updates unchanged, so they inherit the completed R01
quality result.

### Why this can lead over SwiGLU+Muon

SwiGLU+Muon has neither a trainable rational response nor a loss-space metric
that measures how group directions interact across tokens and layers. R01
constructs that metric from the exact current P5/Q4 tangent, reallocates only a
fixed parent budget, and retains explicit descent and parent fallback. The
optimized 9,150-step system completed with endpoint lead `+0.055160522`,
retaining `99.0822%` of the original R01 endpoint lead.

## Optimization classification summary

| Method | Quality-tested changes included in the endpoint | Strict descendants used for timing |
|---|---|---|
| Method 1 | phase4-to2 refresh schedule; compiled score association; complete-layer ownership; error-feedback INT8 publication; separate SplitGraph64 p95 realization | exact Graph64 secular replay component; observer-only certificate elision |
| Method 2 | outer4 refresh schedule; owner-local complete refresh; compiled finite-precision realization in the quality parent | compiled padded INT8 transport with identical codes/scales/decoded update; async-padded p95 path |
| Method 3 | outer8 refresh schedule; compiled R01 scores; four-head polar geometry; ragged INT8 publication | ordinary R01 endpoint diagnostic elision; parent-norm reuse and unobserved-cosine elision |
| R01 | owner-local loss metric; batched compiled response reductions; FP32 inverse reuse; ragged block-256 INT4 publication | fixed compiled secular spans; lean router/attention; compiled transport with identical codes/scales/decoded update |

No optimization changes the five-step Newton--Schulz polynomial, scheduled
learning rate, global clip threshold, or decoupled weight-decay coefficient.

## Fairness and evidence boundary

Both evaluation cells use four generic A6000 GPUs per job, global clipping
`1.0`, peak LR `3e-4`, minimum LR `3e-5`, 200 warmup updates, cosine decay
through the endpoint, weight decay `0.10`, AdamW betas `(0.9,0.95)`, epsilon
`1e-8`, Muon momentum `0.95`, NS5, and `match_rms_adamw`. Every internal LR/WD
multiplier is exactly `1.0`.

The completed results establish a full-system lead over SwiGLU+Muon in one
dataset/seed cell per row. They do not alone establish universal
generalization or isolate the causal contribution of every component. Pending
Method 1 or Method 2 optimizations are excluded until their required endpoint
or strict-equivalence evidence is complete and signed.

Primary signed quality evidence:

- Method 1: `experiments/rlb_300m_4000_design_20260731/runs/METHOD1-PHASE4-TO2-GRAPH64-COMPILED-SCORES-INT8-QUALITY4000/ORIGINAL_PARENT_QUALITY_REPORT.json`, SHA-256 `40b548672eca346f12222b2322d2812f3503d50db8b40d79d0f99636db81de6f`.
- Method 2: `experiments/rlb_300m_4000_design_20260731/runs/METHOD2-GLOBAL-STATISTICS-OWNER-COMPILED-SPAN-QUALITY4000/ORIGINAL_PARENT_QUALITY_REPORT.json`, SHA-256 `6e19a2bc4123e2a218c8ca580e2ec96521b289f84435e97bc991d59397122601`.
- Method 3: `experiments/rlb_300m_4000_design_20260731/runs/METHOD3-OUTER8-HEAD-GROUP-POLAR-QUALITY4000/ORIGINAL_PARENT_QUALITY_REPORT.json`, SHA-256 `920b7639f5d26192d147848c33d536a159a31c4214a19216fa924a897aaa84b7`.
- R01: `experiments/rlb_m1_fineweb_edu_9150_20260806/runs/R01-9150-BATCHED-RESPONSE-INVERSE-COMPILED-SPAN-INT4-FW9150/ORIGINAL_R01_QUALITY_REPORT.json`, SHA-256 `9134a64180e4fa2ea09db78dd857efc42224b8348badb5198a3b1f3f4da7b809`.

Exact runtime source paths and hashes are registered in
`experiments/rlb_300m_4000_design_20260731/TOP4_RUNTIME_INCUMBENTS.json`.
Original endpoint denominators are registered separately in
`experiments/rlb_300m_4000_design_20260731/TOP4_ORIGINAL_PARENT_ANCHORS.json`.
