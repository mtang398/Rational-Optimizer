# Exact specification of the current top three completed RLB optimizer runs

## Status

This document records the three best completed exact-M1 discovery runs by
endpoint validation loss:

1. **current R05 K1**;
2. **current R08 K5**;
3. **current R06 K1**.

The ranking is not a declaration that these are final methods. All three
completed 4,000 updates and improved endpoint loss, endpoint perplexity, and
late validation partial AUC over the primary SwiGLU+Muon control. None reached
the required \(0.20\) endpoint-loss lead, none has valid recursive ablation
closure for its current frozen implementation, and none meets the final
runtime requirement.

R05, R08, and R06 are short display names whose slots have been reused during
discovery. The content-addressed report, trajectory, and source-freeze hashes
at the end of this document identify the completed runs authoritatively.

## Completed results

Lower is better for every reported metric. “Lead” means
\(\text{loss}_{U-S}-\text{loss}_{\text{method}}\).

| Rank | Run | Endpoint validation loss | Lead over SwiGLU+Muon | Endpoint PPL | Train pAUC, steps 1,000–4,000 | Validation pAUC, steps 1,000–4,000 | Runtime / U-S | Runtime / U-R |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| — | SwiGLU+Muon (U-S) | 4.228466988 | 0 | 68.61196851 | 4.199301681 | 4.422074735 | 1.000x | 0.896x |
| — | Global-RLB+Muon (U-R) | 4.241679192 | −0.013212204 | 69.52449883 | 4.205564916 | 4.432158089 | 1.116x | 1.000x |
| **1** | **current R05 K1** | **4.183662415** | **0.044804573** | **65.60568899** | 4.169669949 | **4.395769207** | 2.251x | 2.017x |
| **2** | **current R08 K5** | 4.199784756 | 0.028682232 | 66.67197873 | 4.162024616 | 4.395831645 | 1.922x | 1.722x |
| **3** | **current R06 K1** | 4.199958801 | 0.028508186 | 66.68358371 | **4.161953121** | 4.395792866 | 1.967x | 1.762x |

The pAUC values are normalized trapezoidal averages over logged points from
updates 1,000 through 4,000. They are not dominated by the initial training
transient.

Current R05 also improves endpoint loss over the activation-matched
Global-RLB+Muon control by \(0.058016777\). R08 improves it by
\(0.041894436\), and R06 by \(0.041720390\). This activation-matched comparison
isolates the optimizer contribution while holding Global-RLB fixed. The
primary performance target remains SwiGLU+Muon because that is the conventional
system the combined RLB method must beat.

## Models and activation boundary

### Primary control: SwiGLU+Muon

The U-S control uses the ordinary gated feed-forward map

\[
\operatorname{SwiGLU}(x)
=W_{\mathrm{down}}\!
\left[\operatorname{SiLU}(W_{\mathrm{gate}}x)
\odot(W_{\mathrm{value}}x)\right].
\]

Its trajectory uses the label `silu`, but the implemented block is SwiGLU:
SiLU is the gate activation and is multiplied by a separate value branch.

### Global-RLB model used by all three candidate optimizers

Global-RLB uses a matched-matrix-budget, single-branch feed-forward map. For
one layer,

\[
z=W_{\mathrm{in}}x,\qquad
y=W_{\mathrm{out}}h.
\]

The 4,608 hidden coordinates are partitioned into 18 groups of 256. For group
\(g\),

\[
\rho_g=\sqrt{\operatorname{mean}(z_g^2)+\epsilon},\qquad
u_g=z_g/\rho_g,
\]

\[
h_g=\rho_g f_{\theta_{g,t}}(u_g).
\]

Here \(f_{\theta_{g,t}}\) is that group's current trainable rational response,
with a degree-five numerator and degree-four denominator. Each of the 18
groups in every layer has its own coefficients. The rational responses are
initialized to approximate SiLU and then learned normally. The three methods
do not change the activation parameterization, forward map, initialization, or
coefficient update rule: the coefficients evolve under the same matched AdamW
path in every RLB run. Forward hooks capture current residual inputs,
preactivations, and rational features during training; after backward, the
matrix optimizer consumes those records and evaluates the current response and
exact normalized-response Jacobian.

The Global-RLB model has 296,871,080 trainable parameters. The matched SwiGLU
control has 296,867,840; the 3,240-parameter difference is exactly the learned
rational-coefficient inventory.

## Why Global-RLB exposes useful optimizer structure

Stock Muon sees a matrix and its momentum, then applies the same spectral
update rule regardless of which learned nonlinear function lies between two
matrices. Global-RLB exposes additional functional state:

- 18 independently learned rational functions per layer;
- a fixed pairing between each rational group, 256 rows of
  \(W_{\mathrm{in}}\), and 256 columns of \(W_{\mathrm{out}}\);
- the current rational response \(f_{\theta_{g,t}}\);
- its exact normalized-response Jacobian \(J_{g,t}\);
- current residual-input and rational-feature distributions.

The three methods use this state to select among equal-budget update
geometries. This is the direct reason the designs are RLB-specific. A fixed
SwiGLU block has neither trainable P5/Q4 response state nor the same
single-function pairing of incoming and outgoing matrix blocks.

## Shared optimizer foundation

### Unchanged gradients, momentum, polar map, LR, and WD

All comparisons use the same training protocol, loss, backward pass, DDP
gradient synchronization, and global clipping. U-R and the three candidates
share the identical Global-RLB forward map; U-S uses the stated SwiGLU forward
map. For a clipped matrix gradient \(G_t\), the Muon momentum recurrence is
unchanged:

\[
B_t=0.95B_{t-1}+0.05G_t,\qquad
M_t=0.05G_t+0.95B_t.
\]

The ordinary matrix direction uses five Newton–Schulz iterations
\(\operatorname{NS5}\) and the control's `match_rms_adamw` shape
calibration. Scheduled LR and decoupled WD are applied exactly once.

### Shared Global-RLB pair branch construction

All three methods inherit the same branch-building ingredients for
\(W_{\mathrm{in}}\) and \(W_{\mathrm{out}}\):

- **B, residual-input geometry:** \(C_x=\mathbb{E}[xx^\top]\), used as the
  incoming matrix's external-coordinate factor;
- **C, rational-feature geometry:** \(K_g=\mathbb{E}[h_gh_g^\top]\), used as
  the outgoing matrix's hidden-coordinate factor;
- an ordinary coordinate-polar branch;
- a bias-corrected, matched-\(\beta_2\) coordinate-adaptive branch;
- current-versus-initial response congruence for the incoming and outgoing
  roles.

The covariance factors use unit-volume Cholesky coordinates. Removing their
global determinant scale prevents a coordinate transform from becoming an
internal learning-rate multiplier.

For role \(r\), write the ordinary coordinate-polar direction as \(U_r\) and
the adaptive coordinate direction, rescaled to the same Frobenius budget, as
\(\bar A_r\):

\[
\|\bar A_r\|_F=\|U_r\|_F.
\]

Let \(a_{\mathrm{in}}\) compare the current and initialized
normalized-Jacobian kernels, and let \(a_{\mathrm{out}}\) compare the current
and initialized response kernels. Both are scale-free values in \([0,1]\).
The generation-one direction routes the equal-budget branches through these
congruences. Current R05 retains that literal generation-one direction before
its sign-family completion. R06 instead executes routes that multiply each
congruence by current intrinsic participation, and R08 retains the resulting
R06 direction before its groupwise completion.

### Whole-layer polar operations

The grouped Cholesky factors are applied to 256-channel blocks, but the blocks
are then reassembled before every base polar operation. NS5 acts once on each
full \(4608\times1024\) layer matrix for each RLB role. It does not perform 18
independent \(256\times1024\) polars.

Current R05's groupwise sign route and current R08's groupwise sphere
completion occur after these whole-layer ordinary/adaptive polar directions
have been formed. Attention NS5 operations are likewise whole-matrix
operations.

## RLB morphology statistics

For one current normalized group of width \(m=256\), let \(s_i\) be the
singular values of its exact Jacobian \(J_{g,t}\). Define incoming
participation

\[
c_{\mathrm{in},g}
=
\frac{(\sum_i s_i^2)^2}
     {m\sum_i s_i^4}.
\]

For current response coordinates \(f_i=f_{\theta_{g,t}}(u_i)\), define
outgoing participation

\[
c_{\mathrm{out},g}
=
\frac{(\sum_i f_i^2)^2}
     {m\sum_i f_i^4}.
\]

These values lie in \([0,1]\), are invariant to a common response scaling, and
measure whether current sensitivity or response energy is broadly distributed
or concentrated. Their sufficient statistics are reduced across all four DDP
ranks.

## Method 1: current R05 K1

### RLB pair update

Current R05 starts from the complete generation-one direction
\(P^{(0)}_r\). After that whole-layer direction has been formed, it partitions
\(P^{(0)}_r\) and the unchanged momentum \(M_r\) into the actual 18
rational-group blocks.

For group \(g\) and role \(r\), construct an equal-budget coordinate-sign
direction

\[
Q_{g,r}
=
\operatorname{sign}(M_{g,r})
\frac{\|P^{(0)}_{g,r}\|_F}
     {\|\operatorname{sign}(M_{g,r})\|_F}.
\]

Here \(\operatorname{sign}\) is the elementwise sign in the matrix's native
parameter coordinates, applied after the whole-layer B+C parent has been
formed; it is not a sign operation inside the B+C coordinate transform.

Use \(c_{g,r}=c_{\mathrm{in},g}\) for the incoming role and
\(c_{g,r}=c_{\mathrm{out},g}\) for the outgoing role. The executed block is

\[
D_{g,r}
=
\operatorname{normalize}_{\|P^{(0)}_{g,r}\|_F}
\left(
\sqrt{c_{g,r}}\,P^{(0)}_{g,r}
+
\sqrt{1-c_{g,r}}\,Q_{g,r}
\right).
\]

Thus broad current participation keeps more of the spectral/coordinate
parent, while concentrated participation moves toward coordinate sign. Every
finite branch preserves the parent block's exact Frobenius budget and has
positive inner product with the unchanged momentum. A zero parent/sign norm or
an invalid provisional blend falls back to a finite parent; any nonfinite
executed direction fails closed.

### Attention update

For each layer, R05 forms

\[
c_{\mathrm{attn}}
=
\sqrt{
\operatorname{mean}_g(c_{\mathrm{in},g})
\operatorname{mean}_g(c_{\mathrm{out},g})
}.
\]

For both QKV and attention-output matrices, \(P\) is the whole-matrix stock
Muon polar direction and \(Q\) is the equal-\(\|P\|_F\) sign direction of the
same momentum. R05 applies the same square-root chord and norm closure using
\(c_{\mathrm{attn}}\).

Current R05 therefore structurally routes all RLB input/output matrices and
all QKV/attention-output matrices: 245,366,784 parameters. It is not the
archived R05 generation-one method, and its attention matrices are not left on
stock Muon.

### Executed update summary

```text
1. Form the generation-one whole-layer RLB pair direction.
2. Measure current groupwise Jacobian and response participation.
3. Route each post-polar RLB block between parent and equal-budget sign.
4. Transport the layer's two-role participation to QKV and attention output.
5. Route each whole attention matrix between stock polar and equal-budget sign.
6. Apply the shared scheduled LR and WD once.
```

## Method 2: current R06 K1

### Layerwise RLB pair route

R06 averages the groupwise participation statistics separately for the two
roles, producing \(c_{\mathrm{in}}\) and \(c_{\mathrm{out}}\) per layer. It
combines them with the shared current-versus-initial congruences:

\[
\gamma_{\mathrm{in}}=a_{\mathrm{in}}c_{\mathrm{in}},\qquad
\gamma_{\mathrm{out}}=a_{\mathrm{out}}c_{\mathrm{out}}.
\]

For either role, let \(\gamma_r\) denote its corresponding incoming or
outgoing route. Then

\[
P^{(06)}_r
=
\operatorname{normalize}_{\|U_r\|_F}
\left(
\sqrt{\gamma_r}\,U_r
+
\sqrt{1-\gamma_r}\,\bar A_r
\right).
\]

This is a layerwise equal-budget route. R06 uses neither the sign family of
current R05 nor the group-sphere completion of current R08.

### RLB-conditioned attention

R06 transports the two current role statistics through

\[
\gamma_{\mathrm{attn}}=\sqrt{c_{\mathrm{in}}c_{\mathrm{out}}}.
\]

For each QKV and attention-output matrix, the unchanged Nesterov momentum and
a row/column-factorized, bias-corrected-\(\beta_2\) adaptive source are first
matched to the same Frobenius budget. R06 then blends these equal-budget
endpoints:

\[
S_{\mathrm{attn}}
=
\sqrt{\gamma_{\mathrm{attn}}}\,M
+
\sqrt{1-\gamma_{\mathrm{attn}}}\,\bar A_{\mathrm{attn}}.
\]

The chord itself is not Frobenius-renormalized before NS5; NS5's input
normalization removes its common scale. The whole source matrix is then passed
through unchanged NS5 and the stock
`match_rms_adamw` calibration. The rational statistic changes update
geometry, not LR, WD, momentum, NS count, or calibration.

## Method 3: current R08 K5

Current R08 is a direct extension of current R06. It retains the literal
complete R06 RLB-pair direction \(P^{(06)}_r\) and the literal R06 attention
path.

For role \(r\), R06 has the layerwise route

\[
\gamma_{\mathrm{layer},r}
=a_{\mathrm{layer},r}c_{\mathrm{layer},r}.
\]

R08 resolves only the participation term by rational group:

\[
\gamma_{g,r}
=a_{\mathrm{layer},r}c_{g,r}.
\]

There is no group-specific current-versus-initial alignment. The same
layerwise \(a_{\mathrm{layer},r}\) is used for all 18 groups.

Let:

- \(P^{(06)}_{g,r}\) be the corresponding block of the literal complete R06
  direction;
- \(U_{g,r}\) and \(\bar A_{g,r}\) be blocks of the same whole-layer ordinary
  and equal-budget adaptive branches used to form that direction;
- \(M_{g,r}\) be the unchanged momentum block.

The group target is

\[
T_{g,r}
=
\operatorname{normalize}_{\|P^{(06)}_{g,r}\|_F}
\left(
\sqrt{\gamma_{g,r}}\,U_{g,r}
+
\sqrt{1-\gamma_{g,r}}\,\bar A_{g,r}
\right).
\]

R08 then solves the exact linear maximization problem on the shortest
fixed-radius spherical arc from \(P^{(06)}_{g,r}\) to \(T_{g,r}\):

\[
D_{g,r}
=
\arg\max_{D\in\operatorname{arc}(P^{(06)}_{g,r},T_{g,r})}
\langle M_{g,r},D\rangle_F.
\]

The analytic solver considers the parent endpoint, target endpoint, and the
unique interior stationary point when it lies on the arc. The parent is
always feasible and wins exact ties. Consequently, each selected group block
preserves its R06 parent budget and has first-order momentum descent no worse
than that parent.

R08 adds no groupwise polar operation. It only completes the already formed
whole-layer R06 directions on their per-group fixed-radius arcs. Its
attention update is exactly the R06 attention update.

## Exact differences

| Decision | Current R05 K1 | Current R08 K5 | Current R06 K1 |
|---|---|---|---|
| Shared B+C whole-layer branch construction | Yes | Through literal R06 | Yes |
| Current-vs-initial response congruence | In retained pair parent | Layerwise, through R06 | Layerwise |
| Intrinsic rational statistic | Per group and role | Per group and role, plus retained layer statistic | Per layer and role |
| RLB-pair completion | Equal-budget parent/sign chord | Exact per-group arc LMO around R06 parent | Equal-budget ordinary/adaptive chord |
| Attention update | RLB-routed polar/sign chord | Literal R06 attention | RLB-routed factorized-adaptive source before NS5 |
| Custom structural parameters | 245,366,784 | 245,366,784 | 245,366,784 |
| Rational coefficients | Matched AdamW | Matched AdamW | Matched AdamW |
| Internal LR or WD multiplier | Every value is 1.0 | Every value is 1.0 | Every value is 1.0 |

Current R05 and R06 are sibling extensions of the shared generation-one pair
parent. Current R05 is not R06's parent. Current R08 is the direct
group-resolved completion of R06.

## Exact experiment and fairness cell

| Item | Fixed value |
|---|---|
| Model scale | Exact historical M1 |
| Layers / width / heads | 18 / 1,024 / 16 |
| Nominal FFN setting | 3,072 |
| Global-RLB hidden width | 4,608 = 18 groups × 256 |
| Global-RLB trainable parameters | 296,871,080 |
| SwiGLU trainable parameters | 296,867,840 |
| Dataset | Same frozen DCLM token cache and order |
| Cached training source | 300,000,000 tokens |
| Tokens actually consumed | 131,072,000 = 4,000 × 32,768 |
| Validation cache | Same disjoint 8,000,000 tokens |
| Seed | 1337 |
| Updates | 4,000 |
| Hardware | Four generic A6000 GPUs per job; no node or NVLink pin |
| Sequence length | 256 |
| Per-rank batch / accumulation / ranks | 8 / 4 / 4 |
| Global tokens per update | 32,768 |
| Peak / minimum LR | \(3\times10^{-4}\) / \(3\times10^{-5}\) |
| LR schedule | 200-update warmup, cosine horizon 4,000 |
| Decayed-class WD | 0.10 |
| AdamW betas / epsilon | (0.9, 0.95) / \(10^{-8}\) |
| Global gradient clipping | 1.0 |
| Muon momentum / polar map | 0.95 / NS5 |
| Muon calibration | `match_rms_adamw` |
| Every internal LR scale | 1.0 |
| Every internal WD scale | 1.0 |

All three candidate reports cover the same 296,871,080 parameters with the
same three routed inventories:

- 169,869,312 RLB input/output matrix parameters, WD 0.10;
- 75,497,472 attention matrix parameters, WD 0.10;
- 51,504,296 matched AdamW/no-decay parameters, including the rational
  coefficients, WD 0.

Every group uses the same scheduled base LR. There is no coefficient-specific,
matrix-role-specific, group-specific, or hidden learning-rate multiplier.
Data fingerprints and the realized LR-trace hash are identical across the
controls and candidates. Within the RLB runs, the initial-state fingerprint is
also identical.

## Ablation status

No current top-three method has passed recursive leave-one-component-out
ablation. The current reports all have `promoted: false` because their
endpoint-loss leads are below 0.20, so the preregistered rule correctly
withheld expensive full-4,000-step ablations.

The preregistered provisional direct LOO units are:

- **current R05:** retained generation-one pair parent; incoming
  participation; outgoing participation; pair spectral/sign routing; transport
  to both attention roles;
- **current R08:** equal-budget coordinate geometry; current-versus-initial
  response geometry; role-specific current participation; RLB-to-attention
  transport; group-resolved product-sphere completion;
- **current R06:** inherited B coordinate geometry; inherited C response
  geometry; paired incoming/outgoing participation route; factorized attention
  geometry selected by the two-role geometric mean.

These lists define initial direct removals; they are neither proven exhaustive
decompositions nor claims that the components have survived ablation. In
particular, R05's retained generation-one parent contains inherited
subcomponents that recursive closure may need to unpack. If a method is
promoted, every component must be removed directly from the same full method
at 4,000 updates. Any useless component is deleted, the pruned full method is
rerun, and LOO repeats until every retained component is demonstrably useful.

Older `R08-LOO-*` artifacts belong to a different archived R08 parent. They
may describe ancestor behavior but cannot establish ablation closure for
current R08 K5.

## Runtime status

The reported runtimes are discovery implementations. Against the
activation-matched U-R control, R05 is 2.017x, R08 is 1.722x, and R06 is
1.762x; their U-S ratios are 2.251x, 1.922x, and 1.967x. These values fail the
eventual 1.02x median/p95 runtime requirement against U-R. The scientific
equations must remain fixed if a promoted implementation is optimized for
speed.

## Frozen run identities

| Run | Component / job | Report SHA-256 | Trajectory SHA-256 | Candidate-freeze-file SHA-256 |
|---|---|---|---|---|
| Current R05 | `R05:K1` / 797516 | `2c387d5e4436e8bd96fa67d6869614b2ca271959d6a3d6b3e8c3c399cde8e6c3` | `683b9d6ef4e8e9dc0e0eeafbd487f0357550d31081280c4bcb02cefe447c19c9` | `98cb0921e9a4148aba441d5631e56214b7dddfdff215733e65c7c741a1b59e7d` |
| Current R08 | `R08:K5` / 797175 | `800bc0219fa0028a89be3340ea4955d50239ccdadb6377c046cdd02dddf355f5` | `de74b32078f650d5ffd06ccbf99444208ee817d788afa7c86f207c27f87580bd` | `2466aa6334542e9745b3416a95cc3cffadc017cb7b602923ad91b7dd9b68126c` |
| Current R06 | `R06:K1` / 792491 | `10dd8004fe997ded5939d2c13cf806f8f475cee69f1a3249bcefdedd96690305` | `c8331ae8e0b92af2c798d4246fa959809fab80bbec5e14cb8c3ca78fefbd636c` | `c5f345888cfcea2b9086a88bfeb9f51f612e202eef82843cdce98920b188e496` |

For R08 K5, `rlb_r08_current.py` and `rlb_r08_current_core.py` identify the
K5-specific wrapper and completion. `rlb_r08.py` is an older wrapper, while
`rlb_r08_core.py` remains an inherited B+C dependency and does not by itself
define K5. The content-addressed freeze hashes, rather than a live filename or
reused opaque label, define each completed run.

## Claim boundary

The strongest completed evidence is current R05's single-seed endpoint-loss
lead of 0.044804573 over SwiGLU+Muon, with lower PPL and lower late validation
pAUC under the exact shared LR/WD cell. This is a meaningful positive
direction, not the requested final result. The required 0.20 lead, recursive
ablation closure, and final runtime closure remain open. Generalization across
additional seeds, datasets, and model scales is also untested for these frozen
methods.
