# Four quality-supported optimized Global-RLB methods

Updated 2026-08-27. The filename is retained so existing links do not break.

## Scope and comparison rule

This document specifies the four optimized methods currently tracked: **Method
1, Method 2, Method 3, and R01**. It records only completed, checksum-valid
quality results and qualified matched runtime measurements. An unfinished
timing or quality run is not a result here.

The methods occupy two different evaluation cells and must not be ranked by
mixing their raw losses:

- Method 1--3: DCLM, seed 1,337, 300M training tokens, 4,000 updates.
- R01: FineWeb-Edu, seed 2,027, 300M training tokens, 9,150 updates.

Within each cell, the loss lead over the matched conventional control is

\[
\Delta L=L_{\mathrm{SwiGLU+Muon}}-L_{\mathrm{method}}.
\]

Lead retention is always measured against the **original method's completed
endpoint lead in the same cell**:

\[
R_{\mathrm{lead}}=100\,
\frac{\Delta L_{\mathrm{optimized}}}{\Delta L_{\mathrm{original}}}.
\]

It is never computed against a direct runtime parent or an incomplete
same-step trajectory.

## Completed quality results

Lower loss, perplexity, and validation partial AUC are better. Partial AUC is
the normalized trapezoidal mean over validation checkpoints from step 1,000
through the endpoint.

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
runtime report. The **total-time ratio is the primary number**. A ratio of
`1.022753x`, for example, means 2.2753% more total time than SwiGLU+Muon; it
does not mean 2.2753% faster.

| Method | Exact total-time ratio | Total overhead | Median ratio | p95 ratio | Runtime evidence shape |
|---|---:|---:|---:|---:|---|
| **Method 1** | **1.132811x** | **+13.2811%** | 1.110206x | 1.137827x | Total/median: qualified phase4-to2 Graph64 compiled-score executor; p95: separately qualified SplitGraph64 executor |
| **Method 2** | **1.022753x** | **+2.2753%** | 0.997846x | 1.075310x | Total/median: padded compiled INT8 owner transport; p95: separately qualified paired-attention async-padded executor |
| **Method 3** | **1.048122x** | **+4.8122%** | 1.038522x | 1.075682x | One quality-frontier executor: head-group polar plus strict endpoint/blend elisions |
| **R01** | **1.036097x** | **+3.6097%** | 1.028826x | 1.023006x | Total/p95: qualified lean router/attention executor; median: separately qualified compiled-transport executor |

This is intentionally a **per-metric qualified portfolio**, not a claim that
the independently best total, median, and p95 values always came from one
binary. Method 1's p95 executor retained 96.1957% of its original endpoint
lead (`+0.077274799`). Method 2's and R01's alternate per-metric entries are
strict execution descendants with their own signed qualification. No runtime
ratio is obtained by multiplying measurements from different jobs.

## Shared model and optimizer mathematics

### Conventional control

The control feed-forward block is

\[
y_{\mathrm{SwiGLU}}=W_{\mathrm{down}}
\left(\operatorname{SiLU}(W_{\mathrm{gate}}x)
\odot W_{\mathrm{up}}x\right).
\]

Eligible matrices use ordinary Muon and the remaining parameters use AdamW.

### Global-RLB activation and exact tangent

Each layer uses one incoming matrix `A` and one outgoing matrix `B`. Its 4,608
hidden coordinates are split into 18 learned groups of width `m=256`. For
layer `l` and group `g`,

\[
z_{l,g}=A_{l,g}x,\qquad
\rho_{l,g}=\sqrt{m^{-1}\lVert z_{l,g}\rVert_2^2+10^{-6}},\qquad
u_{l,g}=z_{l,g}/\rho_{l,g}.
\]

The installed trainable P5/Q4 response is

\[
f_{l,g}(u)=
\frac{\sum_{k=0}^{5}a_{l,g,k}u^k}
{1+|b_{l,g,1}u|+|b_{l,g,2}u^2|+|b_{l,g,3}u^3|+|b_{l,g,4}u^4|},
\]

and

\[
h_{l,g}=\rho_{l,g}f_{l,g}(u_{l,g}),\qquad
y_{\mathrm{RLB}}=B_l\operatorname{concat}_{g=1}^{18}h_{l,g}.
\]

Its exact group Jacobian is

\[
J_{l,g}=\operatorname{diag}(f'_{l,g}(u_{l,g}))+
\frac{\left(f_{l,g}(u_{l,g})-
u_{l,g}\odot f'_{l,g}(u_{l,g})\right)u_{l,g}^{\mathsf T}}{m}.
\]

For paired incoming/outgoing perturbations
`D_j=(D_j^A,D_j^B)`, where `j=(l,g)`, the exact tangent image and downstream
loss score are

\[
Y_{n,j}=B_{l,g}J_{l,g}(D^A_jx_n)+D^B_jh_{n,l,g},\qquad
s_{n,j}=\langle e_{n,l},Y_{n,j}\rangle.
\]

Thus the optimizer evaluates candidate directions through the current learned
rational function and group-RMS normalization, rather than only through a
matrix norm.

### Nesterov source and unchanged NS5

After global clipping, every Muon-owned matrix uses

\[
M_t=0.95M_{t-1}+0.05G_t,\qquad
N_t=0.05G_t+0.95M_t.
\]

For each wide matrix batch, set

\[
X_0=N_t/\max(\lVert N_t\rVert_F,10^{-7})
\]

in bfloat16 and execute exactly five iterations

\[
A_k=X_kX_k^{\mathsf T},\qquad
X_{k+1}=3.4445X_k+
\left(-4.7750A_k+2.0315A_k^2\right)X_k.
\]

All four optimized methods retain this polynomial and all five iterations.

### Functional allocation transaction

For a set of paired directions, define

\[
F=\frac{1}{N}S^{\mathsf T}S,\qquad
c=\frac{1}{N}S^{\mathsf T}s_{\mathrm{WD}},
\]

where `S[n,j]=s[n,j]` and `s_WD` is the score of the scheduled decoupled
weight-decay image. With

\[
w_j=\lVert D_j^A\rVert_F^2+\lVert D_j^B\rVert_F^2,
\]

the R01-form allocation solves

\[
\min_{\alpha}
-\eta b^{\mathsf T}\alpha+
\frac{\eta^2}{2}
\left(\alpha^{\mathsf T}F\alpha+2c^{\mathsf T}\alpha\right)
\quad\text{subject to}\quad
\sum_jw_j\alpha_j^2=\sum_jw_j.
\]

The all-ones parent allocation is feasible, evaluated first, wins exact ties,
and is used if any finiteness, budget, surrogate, exact-gradient-descent, or
Nesterov-descent certificate fails. Scheduled LR and decoupled WD are then
applied exactly once.

## Method 1 optimized: cross-role frame with phase4-to2 Graph64 execution

Let `C=B^T`. For one rational group, the incoming and transposed outgoing
Nesterov blocks share the shape

\[
N^A_{l,g},N^C_{l,g}\in\mathbb{R}^{256\times1024}.
\]

The scientific update is executed in this order:

1. Build the complete paired RLB parent direction `P=(P^A,P^C)`, including
   radial two-role response whitening, exact functional scores, fixed paired
   Frobenius budget, and rolewise descent certificates. For hidden coordinate
   `i`, the exact self-coordinate responses used for the two-role radial Gram
   matrix are

   \[
   r_A=z_i\left[f'_i(u)+
   \frac{(f_i(u)-u_i f'_i(u))u_i}{m}\right],
   \qquad r_B=\rho f_i(u).
   \]

   The paired radial Adam direction is multiplied by the Moore--Penrose
   inverse square root of the empirical `2\times2` response Gram matrix.
2. Stack both roles around the same learned function and apply one NS5 map:

   \[
   \begin{bmatrix}F^A_{l,g}\\F^C_{l,g}\end{bmatrix}
   =\operatorname{NS5}\!\left(
   \begin{bmatrix}N^A_{l,g}\\N^C_{l,g}\end{bmatrix}\right).
   \]

3. Using
   `\langle X,Y\rangle_{A,C}=\langle X^A,Y^A\rangle_F+
   \langle X^C,Y^C\rangle_F`, remove the parent component and restore its
   exact layer budget:

   \[
   E=F-\frac{\langle F,P\rangle_{A,C}}{\lVert P\rVert_{A,C}^2}P,
   \qquad
   \widehat E=E\frac{\lVert P\rVert_{A,C}}{\lVert E\rVert_{A,C}}.
   \]

4. Form the two equal-energy axes

   \[
   C_+=(P+\widehat E)/2,\qquad C_-=(P-\widehat E)/2,
   \]

   compute their exact P5/Q4 tangent scores, and solve the certified
   same-budget functional allocation over 36 axes (two per layer).
5. Execute the RLB-conditioned attention transaction, coefficient AdamW,
   scheduled LR, and decoupled WD once.

The optimized schedule performs a full hierarchy refresh every four
transitions through step 1,500 and every two transitions afterward: 1,625
complete refreshes in 4,000 updates. Each of the other 2,375 transitions
still executes a current-gradient exact global R01 transaction; no cached
matrix update is replayed.

The optimized executor preserves the equations while replaying the unchanged
64-round secular solve as Graph64, compiling the already-factorized R01
response-adjoint/radial/frame score programs, removing ordinary
telemetry-only work, assigning complete layers to four owners, and publishing
error-feedback block-256 INT8 deltas. The changed finite-precision reduction
association received a fresh 4,000-step quality run.

**Why it can lead over SwiGLU+Muon.** SwiGLU+Muon neither learns the P5/Q4
response nor represents the incoming and outgoing matrices around one learned
function in a shared spectral frame. Method 1 scores same-budget directions
through the exact current RLB Jacobian and downstream cotangent, then chooses
the certified allocation in that function-aware space. The completed optimized
system achieved `+0.077912331` endpoint loss lead without an LR or WD
multiplier.

## Method 2 optimized: paired post-polar second moment

Method 2 performs Method 1's joint cross-role polarization, then applies one
shared channel scale before residualization and exact budget closure. If
`F^A_{l,g,i,:}` and `F^C_{l,g,i,:}` are the paired post-polar rows, define

\[
q_{t,l,g,i}=\frac12\left(
\operatorname{mean}_k\!\left[(F^A_{l,g,i,k})^2\right]+
\operatorname{mean}_k\!\left[(F^C_{l,g,i,k})^2\right]\right),
\]

\[
v_t=0.95v_{t-1}+0.05q_t,\qquad
\widehat v_t=\frac{v_t}{1-0.95^t},\qquad
r_t=\frac{1}{\sqrt{\widehat v_t}+10^{-8}}.
\]

Both roles are scaled by the same `r_t`:

\[
\widetilde F^A_{l,g,i,:}=r_{t,l,g,i}F^A_{l,g,i,:},\qquad
\widetilde F^C_{l,g,i,:}=r_{t,l,g,i}F^C_{l,g,i,:}.
\]

The later exact norm closure restores the registered parent budget, so this
is a coordinate preconditioner rather than an LR increase.

The optimized execution order is:

1. Use owner-local complete layers for the full paired-post-polar refresh once
   every four transitions.
2. On each of the other three transitions, use the current-gradient exact
   18-layer global R01 allocation from all-rank observations.
3. Execute the same attention, NS5, rolewise descent, LR, and WD equations.
4. Capture the literal 64-round secular solve in one fixed-shape compiled
   program; compile the block-256 INT8 pack and remote decode; publish the
   exact same INT8 codes and FP32 scales with padded collectives.

The owner boundary and compiled finite-precision realization were validated
by a fresh complete 4,000-step run.

**Why it can lead over SwiGLU+Muon.** In addition to the learned rational
function and downstream-loss allocation, Method 2 ties the adaptive energy of
the incoming row and outgoing column that implement the same hidden channel.
It can suppress persistent channel imbalance before a certified same-budget
functional decision. The completed optimized system achieved
`+0.071474075` endpoint loss lead over SwiGLU+Muon.

## Method 3 optimized: attention row product and four-head polar blocks

For each effective attention row, Method 3 stores a positive magnitude `g_i`
and a direction variable `V_i`:

\[
W_i=g_i\frac{V_i}{\lVert V_i\rVert_2}.
\]

With `U_i=V_i/\lVert V_i\rVert_2` and clipped effective gradient `G_i`, the
exact pullback is

\[
\nabla_{g_i}L=\langle G_i,U_i\rangle,\qquad
\nabla_{V_i}L=\frac{g_i}{\lVert V_i\rVert_2}
\left(G_i-U_i\langle G_i,U_i\rangle\right).
\]

The magnitude uses bias-corrected Adam with `(0.9,0.95)` and `1e-8`; the
tangent direction uses the RLB-conditioned same-budget attention transaction.
If `c_l\in[0,1]` is the registered response congruence, its two same-budget
branches are mixed as

\[
D_l=\operatorname{Norm}_{\lVert P_l\rVert_F}
\left(\sqrt{c_l}\,U_{\mathrm{parent},l}+
\sqrt{1-c_l}\,U_{\mathrm{adaptive},l}\right).
\]

Equivalently, with `\alpha_l=\sqrt{c_l}`, the amplitudes are
`\alpha_l` and `\sqrt{1-\alpha_l^2}`.

The quality-frontier optimization changes only how the unchanged NS5
polynomial is partitioned for attention. Let `d=1024`, 16 heads, head width
`d_h=64`, and group size `g=4`. For QKV source
`X\in\mathbb{R}^{3d\times d}`, split Q, K, and V into twelve four-head row
blocks `X_{t,j}\in\mathbb{R}^{gd_h\times d}` and set

\[
P_H(X)=\frac{1}{\sqrt3}
\operatorname{concat}_{t\in\{Q,K,V\},\,j=1}^{4}P_5(X_{t,j}).
\]

The factor `1/\sqrt3` preserves the nominal squared-Frobenius polar-rank
calibration. For the output source `Y\in\mathbb{R}^{d\times d}`, split its
input columns into four blocks `Y_j\in\mathbb{R}^{d\times gd_h}` and set

\[
P_H(Y)=\operatorname{concat}_{j=1}^{4}P_5(Y_j).
\]

Those four nominal ranks already sum to `d`, so no scale is needed. Both
attention branches use the same block map and every block still performs the
same five NS iterations.

Execution is: one complete outer R03 refresh every eight transitions;
current-gradient compiled R01 score allocation on the other seven; the
head-group attention transaction on every transition; exact ragged INT8 owner
publication; then strict elision of only unobserved endpoint/blend diagnostics
while reusing the identical ordinary blend norm. The update, state, LR, WD,
and telemetry transitions remain unchanged. Because head grouping changes the
numerical polar geometry, it received a fresh complete 4,000-step quality run.

**Why it can lead over SwiGLU+Muon.** The row product separates attention-row
magnitude from tangent orientation, conditions the orientation on the current
learned RLB response, and respects native attention-head structure during the
polar map. Ordinary SwiGLU+Muon exposes none of those learned-function or
head-group coordinates. The completed optimized system achieved
`+0.083792210` endpoint loss lead.

## R01 optimized: owner-local downstream-loss metric with compiled INT4 execution

The optimized 9,150-step R01 retains the functional transaction above but
partitions complete layers across four owners. For rank `r`,

\[
\mathcal I_r=\{l: l\bmod4=r\},
\]

giving owner counts `(5,5,4,4)`. It forms aligned scores only for its complete
owned layers, so `S_r` has `18|\mathcal I_r|` columns (90 or 72), and solves

\[
F_r=N^{-1}S_r^{\mathsf T}S_r,
\qquad
\min_{\alpha_r}
-\eta b_r^{\mathsf T}\alpha_r+
\frac{\eta^2}{2}
(\alpha_r^{\mathsf T}F_r\alpha_r+2c_r^{\mathsf T}\alpha_r)
\]

subject to the exact paired Frobenius budget over the owned coordinates.
This preserves all within-owner cross-layer blocks but intentionally omits
cross-owner blocks; the fresh 9,150-step run is the evidence for that
approximation.

The optimized execution order is:

1. Capture deterministic RLB probes and exact analytic P5/Q4 response
   statistics for each owned layer.
2. Batch the four or five independent owned-layer response reductions into
   one compiled program; no layer statistics are mixed.
3. Reuse one explicit FP32 inverse per response factor and apply coordinates
   by matrix multiplication.
4. Execute the same certified local functional allocation, attention update,
   and unchanged NS5 maps.
5. Capture the three fixed 64-round span solves in one compiled CUDA program.
6. Quantize every block-256 owner delta with

   \[
   s_b=\max_i|\delta_{b,i}|/7,\qquad
   q_{b,i}=\operatorname{clip}_{[-7,7]}
   (\operatorname{round}(\delta_{b,i}/s_b)),\qquad
   \widehat\delta_{b,i}=s_bq_{b,i},
   \]

   pack two signed INT4 codes per byte, publish exact padded owner packets,
   and make every rank, including the owner, apply the same decoded FP32
   update.
7. Compile pack/decode and remove only unobserved ordinary router/attention
   diagnostics; LR, WD, optimizer state, and all five NS steps are unchanged.

**Why it can lead over SwiGLU+Muon.** R01 estimates how directions from
different rational groups and residual blocks jointly act on the same token
loss, then reallocates a fixed update budget under explicit descent and
parent-fallback certificates. A fixed SwiGLU block with ordinary Muon has no
learned rational tangent basis or cross-layer downstream-loss allocation. The
completed optimized system achieved `+0.055160522` endpoint loss lead in the
9,150-step FineWeb-Edu cell.

## Fairness and evidence boundary

Both cells use four generic A6000 GPUs per job, global clipping `1.0`, peak LR
`3e-4`, minimum LR `3e-5`, 200 warmup updates, cosine decay through the cell
endpoint, weight decay `0.10`, AdamW betas `(0.9,0.95)`, epsilon `1e-8`, Muon
momentum `0.95`, NS5, and `match_rms_adamw`. Every internal LR/WD multiplier
is exactly `1.0`.

The results establish full-system leads in one completed dataset/seed cell per
row; they do not by themselves establish universal generalization or isolate
the causal contribution of every component. Pending Method 1 or Method 2
optimizations are deliberately excluded until a complete fresh endpoint is
signed.

Primary signed quality evidence:

- Method 1: `METHOD1-PHASE4-TO2-GRAPH64-COMPILED-SCORES-INT8-QUALITY4000/ORIGINAL_PARENT_QUALITY_REPORT.json`, SHA-256 `40b548672eca346f12222b2322d2812f3503d50db8b40d79d0f99636db81de6f`.
- Method 2: `METHOD2-GLOBAL-STATISTICS-OWNER-COMPILED-SPAN-QUALITY4000/ORIGINAL_PARENT_QUALITY_REPORT.json`, SHA-256 `6e19a2bc4123e2a218c8ca580e2ec96521b289f84435e97bc991d59397122601`.
- Method 3: `METHOD3-OUTER8-HEAD-GROUP-POLAR-QUALITY4000/ORIGINAL_PARENT_QUALITY_REPORT.json`, SHA-256 `920b7639f5d26192d147848c33d536a159a31c4214a19216fa924a897aaa84b7`.
- R01: `R01-9150-BATCHED-RESPONSE-INVERSE-COMPILED-SPAN-INT4-FW9150/ORIGINAL_R01_QUALITY_REPORT.json`, SHA-256 `9134a64180e4fa2ea09db78dd857efc42224b8348badb5198a3b1f3f4da7b809`.

The exact runtime source paths and hashes are registered in
`TOP4_RUNTIME_INCUMBENTS.json`; original endpoint denominators are registered
separately in `TOP4_ORIGINAL_PARENT_ANCHORS.json`.
