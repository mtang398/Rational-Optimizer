# Quotient Natural MatrixPolicy Proposal

This is a proposal for the next RLB-specific optimizer after the current
`rlb_matrixpolicy_original` result. The central claim is:

```text
RLB is not only an activation variant. It exposes a quotient geometry and a
local function-space metric. The next MatrixPolicy should approximate a
block-natural-gradient optimizer on that quotient, with conservative transport
steps used only as coordinate conditioning.
```

The plan below is deliberately not "MatrixPolicy plus Muon" or "turn on every
existing transport flag." The E1/E2 evidence says the durable advantage comes
from RLB role structure, per-group pressure/activity, and gauge control. The
next optimizer should make those pieces mathematically cleaner and faster.

## Evidence Constraints From E1 And E2

### What The Completed Runs Say

E1 is M0 at about 100M train tokens over five datasets and three seeds. The
main ordering is:

1. `rlb_matrixpolicy_original`
2. `rlb_lion`
3. `silu_lion`
4. `rlb_adamw`
5. `silu_adamw`
6. SOAP variants
7. Muon variants
8. ScheduleFree/CAME
9. ADeMaMix divergence

E2 is the same M0 shape at about 300M train tokens on the completed larger
cells. The pattern is:

- MatrixPolicy remains best.
- AdamW is no longer the strongest non-MatrixPolicy frontier.
- Lion/Muon/SOAP improve with more steps, especially on RLB.
- MatrixPolicy's gap shrinks but does not vanish.
- MatrixPolicy telemetry shows the explicit Muon fraction is zero late in the
  run, while pressure/activity and group scaling remain active.

So the design constraint is:

```text
Do not explain MatrixPolicy as generic Muon. Explain and improve the part that
keeps mattering after generic matrix transport is gone.
```

### What The Current Code Says

The current optimizer surface has three separable parts:

- `RationalMatrixPolicyOptimizer` owns the matrix AdamW/Muon child optimizers,
  role/depth LR scaling, group gain/pressure/activity gradient scaling, and
  telemetry.
- `RationalTransportOnPolicyOptimizer` wraps child optimizers with pressure
  EMAs, optional quotient projection, optional matrix/coeff metrics, bounded
  `W_in/W_out` gauge rebalance, and optional rational-curve amplitude transport.
- `FunctionSpaceRationalOptimizer` contains an approximate function-space
  update for rational coefficients.

The current best paper-facing row uses only part of this machinery. Important
paths exist but are not active in the result row:

- quotient gradient projection;
- coefficient function-space metrics;
- empirical coefficient Gram metrics;
- derivative/output matrix metrics;
- pressure gradient preconditioning;
- rational curve amplitude transport;
- live matrix stats for matrix gain metrics.

The proposal should not simply enable all of these. The right question is what
mathematical object those mechanisms are trying to approximate.

## Mathematical Object

For one RLB layer and group, write:

```text
z_g = A_g x
r_g = sqrt(mean(z_g^2) + eps)
u_g = z_g / r_g
h_g = r_g R_g(u_g)
y_g = B_g h_g
```

The group parameters are:

```text
theta_g = (A_g, R_g, B_g)
```

Their roles are different:

- `A_g` selects and rotates the input domain seen by the rational curve.
- `R_g` changes the nonlinear scalar function applied to normalized features.
- `B_g` recombines the resulting rational features into the residual stream.

A generic optimizer sees raw tensors. The RLB optimizer should see the
represented map:

```text
F_g(theta_g): x -> B_g [r_g R_g(u_g)]
```

The local first-order function movement is:

```text
dF_g =
    J_A,g dA_g
  + J_R,g dR_g
  + J_B,g dB_g
```

where the Jacobian blocks are induced by the current activation distribution.
Ignoring the RMS floor and using a groupwise approximation:

```text
J_A,g depends on B_g, R'_g(u_g), and x covariance
J_R,g depends on B_g and the rational basis values on u_g
J_B,g depends on h_g covariance
```

The natural trust-region step would solve:

```text
min_dtheta  <grad L, dtheta> + 0.5 lambda ||dF_g||_D^2
```

with `D` the empirical token distribution. Exact `J^T J` is too expensive, but
RLB gives useful low-cost approximations:

- derivative RMS approximates sensitivity of `A_g`;
- output RMS approximates sensitivity of `B_g`;
- rational basis Gram matrices approximate sensitivity of `R_g`;
- gradient pressure approximates which block is currently bottlenecked;
- gauge drift reveals movement in non-identifiable coordinates.

The next MatrixPolicy should be described as a cheap block-diagonal,
quotient-aware approximation to this trust-region problem.

## Quotient Symmetry

In the homogeneous part of RLB, each group has a positive scale symmetry:

```text
A_g -> s A_g
B_g -> B_g / s
R_g unchanged
```

With the stabilized RMS floor this is not perfectly exact at tiny radii, but it
is the scale structure observed by the current matched runs. The vertical gauge
tangent is:

```text
v_g = (A_g, -B_g)
```

Raw optimizer movement along `v_g` mostly changes the representative, not the
function. Post-step gauge rebalance fixes the representative after the fact,
but it does not stop AdamW moments from remembering gauge-direction gradients.

That is the first major weakness of the current design.

For raw matrix gradients `(G_A, G_B)`, the Euclidean horizontal projection is:

```text
c = (<G_A, A_g> - <G_B, B_g>) /
    (||A_g||^2 + ||B_g||^2 + eps)

G_A,h = G_A - c A_g
G_B,h = G_B + c B_g
```

The stronger version is metric-aware:

```text
c = (<G_A, A_g>_{M_A} - <G_B, B_g>_{M_B}) /
    (||A_g||^2_{M_A} + ||B_g||^2_{M_B} + eps)
```

where `M_A` and `M_B` are the same diagonal/group metrics used in the
functional trust region. This is the mathematically clean target. The first
implementation can use the Euclidean projection, then move to the metric-aware
version after the state owner exists.

## Curve-Amplitude Symmetry

There is a second useful coordinate transport:

```text
R_g -> a R_g
B_g -> B_g / a
```

In the current implementation this scales numerator and atom coefficients while
leaving denominator, centers, and beta unchanged. That scales the rational
output and derivative, then compensates in `B_g`. This is function-preserving
up to numerical clipping and atom-logit headroom.

This should not be treated as "learning." It is a coordinate-conditioning
move. It is useful when curve output/derivative gains become uneven across
groups and force `B_g` to carry an arbitrary inverse scale.

## Proposed Optimizer

Working name:

```text
RLB Quotient Natural MatrixPolicy
```

Short name:

```text
QN-MatrixPolicy
```

The optimizer has six core parts:

1. matrix-state ownership instead of child optimizer composition;
2. quotient-native moments for `A/B`;
3. a block functional metric over `A/R/B`;
4. constrained role-budget allocation;
5. coordinate transport for gauge and curve amplitude;
6. fused implementation to reduce overhead.

## 1. Matrix-State Ownership

Current `RationalMatrixPolicyOptimizer` creates AdamW and optional Muon child
optimizers over the same matrix groups. The wrapper then edits gradients,
steps child optimizers, and applies gauge rebalance.

That is convenient, but it is mathematically misaligned:

- the state lives in raw coordinates;
- quotient projection is only a gradient edit, not a state invariant;
- rebalance has to transform external optimizer state;
- telemetry and gradient scaling require repeated passes over the same groups.

QN-MatrixPolicy should own the matrix state directly:

```text
for each group g:
    read A_g, B_g, grad_A, grad_B
    update pressure/activity EMAs
    project gradients to horizontal space
    apply functional metric scaling
    update first/second moments in horizontal coordinates
    compute role/depth/group step
    apply bounded parameter update
    apply bounded gauge rebalance
    transport moments covariantly
```

This is not just an implementation cleanup. It is required if the optimizer is
claiming to optimize on the quotient.

## 2. Quotient-Native Moments

The moment state should remain horizontal:

```text
m_A, m_B = projected EMA of projected gradients
v_A, v_B = EMA of projected-gradient squares or groupwise second moments
```

After every gauge rebalance:

```text
A_g <- s A_g
B_g <- B_g / s
```

the moment state must transform covariantly:

```text
m_A <- s m_A
v_A <- s^2 v_A
m_B <- m_B / s
v_B <- v_B / s^2
```

Then it should be reprojected to remove numerical horizontal drift.

This gives a concrete ablation:

```text
current MatrixPolicy
current + quotient gradient projection
new owned-state MatrixPolicy with quotient moments
new owned-state MatrixPolicy with quotient moments and covariant rebalance
```

The expected benefit is not a magical early win. The expected benefit is lower
long-horizon drift and better late-step stability, which is exactly where E2
shows generic optimizers catching up.

## 3. Block Functional Metric

The ideal metric is:

```text
G_g = E_x [J_g(x)^T J_g(x)]
```

over the group map `F_g`. Full `G_g` is too expensive. Use a structured
approximation:

```text
G_g approx diag(G_A,g, G_R,g, G_B,g)
```

with:

```text
G_A,g from derivative_rms_g, B_g scale, and optional input RMS
G_R,g from rational basis RMS/Gram on live u_g or probe grid
G_B,g from output_rms_g or h_g RMS
```

Concrete first version:

```text
A scale = centered_inverse(derivative_rms_g)^alpha_A
B scale = centered_inverse(output_rms_g)^alpha_B
R metric = existing basis/Gram metric with denominator damping
all scales centered by geometric mean and clipped
```

Important: this is not SOAP/Shampoo. SOAP estimates generic tensor curvature.
This metric is derived from RLB's represented function. The paper distinction
should be explicit:

```text
SOAP: generic parameter tensor metric
QN-MatrixPolicy: RLB pullback metric over A/R/B functional blocks
```

## 4. Role-Budget Allocation

The previous draft used role logits. That is useful engineering, but the math
should be a constrained trust-region allocation.

Define a per-group movement budget:

```text
||dF_A,g||^2 + ||dF_R,g||^2 + ||dF_B,g||^2 <= tau_g^2
```

The optimizer cannot compute exact block movements every step, so it estimates
block pressures:

```text
p_A = ||grad_A|| / ||A||
p_B = ||grad_B|| / ||B||
p_R = ||grad_R|| in coefficient-function metric

a_R = log(p_R) - 0.5 [log(p_A) + log(p_B)]
q_AB = log(p_A) - log(p_B)
```

Then it allocates movement:

```text
w_A + w_R + w_B = 1
```

with bounds:

```text
w_A in [0.20, 0.65]
w_B in [0.20, 0.65]
w_R in [0.03, 0.30] initially
```

A reasonable controller is:

```text
z_A = k_p center(log p_A) + k_d center(log derivative_rms) + phase_A
z_B = k_p center(log p_B) + k_o center(log output_rms)     + phase_B
z_R = k_r center(a_R) - safety_penalty_R                  + phase_R
w   = clipped_softmax(z_A, z_R, z_B)
```

But the document and code should describe this as an approximation to the
trust-region allocation, not as arbitrary optimizer mixing.

Rules:

- Give `A` budget when input-domain pressure is high and derivative gain is
  not already extreme.
- Give `B` budget when output pressure/recombination pressure is high.
- Give `R` budget only when rational activity is high and denominator/atom
  safety is good.
- Never let `R` absorb all movement, because matrices define the domains and
  recombination that make RLB useful.
- Never let `A/B` ignore persistent rational activity, because then the matrix
  policy is solving curve-shape problems with domain/recombination movement.

## 5. Function-Space Coefficient Movement

Rational coefficients are not ordinary coordinates. For numerator and
denominator coefficients, basis powers have very different function-space
sizes. For atom coefficients, `tanh` headroom makes raw logit movement even
less meaningful.

The existing `FunctionSpaceRationalOptimizer` already approximates:

```text
grad_coeff -> basis/Gram preconditioned update
update -> per-curve RMS normalized update
update -> trust clipped
```

QN-MatrixPolicy should reuse that idea, but integrate it into the same
`A/R/B` budget:

- numerator: highest coefficient trust;
- denominator: lower trust, stronger damping, safety gate;
- atoms: function-space metric with headroom gate;
- centers/beta: frozen or extremely low trust until separately validated.

The coefficient update should answer:

```text
How much represented curve movement is allowed for this group right now?
```

not:

```text
What LR should these raw coefficient tensors get?
```

## 6. Coordinate Transport

There are two transport operations.

### Matrix Gauge Rebalance

This already exists and should remain conservative:

```text
target log ratio = function-gain target + pressure target
current log ratio = log ||A_g|| - log ||B_g||
log s = 0.5 (target - current)
A_g <- s A_g
B_g <- B_g / s
```

Improvements:

- use the same telemetry used by the role-budget controller;
- apply after the owned-state matrix update;
- covariantly transform moments;
- reproject moments after transform;
- log correction magnitude by layer/group.

### Rational Curve Amplitude Transport

Use only as conditioning:

```text
R_g <- a R_g
B_g <- B_g / a
```

Target:

```text
log curve_gain = (1 - beta) log output_rms + beta log derivative_rms
log a = centered target - log curve_gain
```

Safety:

- atom-logit headroom must allow scaling;
- denominator margin must be healthy;
- curve transport disabled during large gauge rebalance;
- max log step small, e.g. `0.005` to `0.015` initially;
- moment covariance active for `B_g`;
- cached matrix metrics invalidated or analytically rescaled after transport.

## 7. Phase Control

The current policy relies heavily on progress windows. The E1/E2 difference
shows that fixed windows are too crude.

Use telemetry-derived phase variables:

```text
phase_domain
phase_curve
phase_refine
```

Inputs:

- pressure mean/std;
- activity mean/std;
- update-to-weight by role;
- gauge correction magnitude;
- derivative/output gain dispersion;
- denominator low percentile;
- atom headroom;
- optional fixed-probe function movement.

Suggested behavior:

```text
domain:
    high input/output pressure, early progress prior
    higher A/B budget, low R budget, no curve transport

curve:
    rational activity rises, safety healthy
    allow R budget and coefficient function-space metric

refine:
    update-to-weight falls, gain imbalance persists
    quotient moments, gain metric, small transports dominate
```

Keep a weak progress prior. Telemetry should steer phases, not make them jump
from noisy early statistics.

## 8. Fusing And Runtime Plan

The speed problem is real. The current overhead is structurally plausible
because the implementation performs many separate passes:

- update on-policy stats;
- optionally project gauge gradients;
- optionally precondition coefficients;
- optionally precondition matrices;
- apply group policy gradient scales;
- child AdamW step;
- optional child Muon step;
- gauge rebalance;
- optional curve transport;
- telemetry passes.

QN-MatrixPolicy should reduce this to a small number of group-local passes.

### Safe Same-Method Optimizations

These preserve current behavior and can be done before changing the optimizer:

1. Cache typed group metadata.
   Do not repeatedly parse `layer_index`, `selector_index`, `groups`,
   `hidden_dim`, role strings, and width inside hot loops.

2. Cache per-step group scales.
   `_group_policy_scale` is currently recomputed for telemetry and gradient
   application. Compute once per group/role/step and reuse.

3. Avoid `.item()` on GPU tensors except telemetry steps.
   `_stat_factor` and telemetry helpers can trigger synchronization. Keep
   tensor math on device and convert to Python only when logging.

4. Fuse group norm and gradient norm collection.
   The same `A/B` views can produce pressure EMAs, gauge norms, gauge dot
   products, and update RMS.

5. Use live activation stats before probe-grid recomputation.
   Probe-grid curve gains are useful but expensive. Prefer module stats when
   available, refresh probe metrics at staggered low frequency.

6. Stagger expensive metrics by layer.
   Do not refresh every layer's coefficient Gram or curve gain on the same
   step.

7. Make telemetry lazy.
   Full update snapshots and `.cpu()` collections should happen only on log
   steps. Training steps should not pay for diagnostic summaries.

### New Optimizer Speed Design

Owning the state enables stronger fusing:

```text
for each balance group:
    create A_view, B_view, grad_A_view, grad_B_view once
    compute norms, pressures, gauge projection coefficient
    compute metric/group scales
    update moments
    apply A/B update
    apply gauge correction
    transform moments
```

This removes:

- duplicate AdamW/Muon param-group traversal;
- repeated group-shape checks;
- repeated group-scale construction;
- wrapper state transforms over foreign optimizer states;
- many CPU scalar conversions.

For the coefficient path:

- keep Gram/stat refresh low frequency;
- batch same-shape coefficient metric solves when possible;
- use diagonal metric first for broad ablations;
- use Gram metric only after diagonal improves or ties.

The first implementation target should be:

```text
same final loss as current MatrixPolicy within noise
lower per-step wall-clock overhead
same or lower memory
no change to represented method except mechanical fusing
```

Then add quotient-native behavior.

## Implementation Stages

### Stage 0: Offline Geometry Audit

No training behavior changes.

Add an analysis report over existing JSONL telemetry:

- pressure mean/std by token;
- activity mean/std by token;
- group scale min/max/std;
- update-to-weight by role;
- Muon fraction by role;
- denominator and atom safety metrics;
- derivative/output gain dispersion when available;
- gauge correction magnitude if logged.

Question:

```text
Does MatrixPolicy's late advantage correlate more with pressure/activity/gain
structure than with the early Muon window?
```

Expected output:

- one markdown report;
- one CSV per completed cell;
- plots over tokens.

### Stage 1: Same-Method Fused MatrixPolicy

Build a new folder/code path for acceleration only. It must reproduce current
MatrixPolicy math before changing optimizer behavior.

Acceptance:

- same parameter updates as current path on a tiny CPU deterministic test;
- same gauge-rebalance result after one step;
- no GPU needed for the unit test;
- lower Python overhead in a CPU timing microbenchmark.

This stage prevents the previous mistake of editing the active activation or
changing the method while trying to speed it up.

### Stage 2: Quotient Projection Ablation

Use the existing projection hook first:

```text
current
current + quotient_strength 0.25
current + quotient_strength 0.50
current + quotient_strength 1.00
```

This tests whether removing raw gauge-gradient movement helps before changing
state ownership.

Failure mode:

```text
projection may remove useful scale adaptation because the RMS floor makes the
gauge only approximately exact.
```

If this fails softly but does not destabilize, proceed to metric-aware
projection. If it fails hard, diagnose floor/radius dependence before moving
on.

### Stage 3: Owned-State Quotient Moments

Implement the new matrix-state owner:

```text
horizontalize gradients
update horizontal moments
apply AdamW-like matrix step with role/depth/group scale
rebalance gauge
transport state covariantly
reproject moments
```

Do not add coefficient movement or curve transport yet.

Acceptance:

- better or equal DCLM E1/E2 loss against current MatrixPolicy within matched
  protocol;
- lower gauge drift;
- no increase in denominator or atom safety events;
- runtime overhead lower than current MatrixPolicy.

### Stage 4: Functional Metric For Matrices

Add derivative/output-gain matrix scaling inside the owned-state update.

Start conservative:

```text
alpha_A <= 0.20
alpha_B <= 0.20
scale clip [0.80, 1.25]
live stats preferred
probe refresh staggered
```

Acceptance:

- improves late AUC or final loss without hurting early token efficiency;
- scale distributions are not saturating clips every step.

### Stage 5: A/R/B Role Budget

Add the constrained role budget, initially using `w_R` only as a gate for
coefficient updates that remain off. This lets us test the `A/B` allocator
without introducing coefficient risk.

Then enable conservative `R` movement:

```text
w_R max 0.10 to 0.15
denominator trust low
atom headroom gate strict
centers/beta frozen
```

Acceptance:

- RLB+QN improves over RLB+MatrixPolicy;
- coefficient movement correlates with rational activity rather than raw time;
- no denominator margin regression.

### Stage 6: Curve-Amplitude Transport

Enable small transport after quotient moments and functional metric are stable:

```text
transport_every >= 10
transport_max_log_step <= 0.01
transport phase starts only after curve phase is active
disable transport during large gauge corrections
```

Acceptance:

- output/derivative gain dispersion decreases;
- `B_g` norm compensation does not create gauge oscillations;
- final or late-AUC loss improves.

### Stage 7: Full QN-MatrixPolicy

Promote to a paper-facing candidate only if it beats current MatrixPolicy under
matched protocol:

- DCLM first;
- FineWeb-Edu second;
- then all E1 datasets;
- then E2 completed datasets;
- then larger model/longer horizon only after small cells are clean.

## Self-Review And Rejected Shortcuts

### Rejected: "Just Add Muon"

Muon improves with more steps in E2, but current MatrixPolicy remains best
after its explicit Muon fraction goes to zero. A Muon-centered proposal would
not explain the late MatrixPolicy advantage.

### Rejected: "Turn On All Transport Flags"

The inactive hooks are not automatically a coherent optimizer. Projection,
coefficient metrics, pressure preconditioning, gauge balance, and curve
transport all manipulate related coordinates. Enabling them together could
work accidentally, but it would be hard to defend as an ICLR contribution.

### Rejected: "More Schedule Tuning"

E1/E2 already show horizon dependence. More fixed windows will likely overfit
dataset and token budget. Phase variables should come from telemetry plus a
weak progress prior.

### Rejected: "Only Code Fusing"

Fusing is necessary because MatrixPolicy overhead matters, but speed alone is
not a new optimizer idea. The paper-worthy idea is quotient functional
optimization; fusing makes it practical.

### Remaining Weak Points

- The gauge is approximate because of the RMS floor. Projection strength must
  be validated rather than assumed.
- The block metric ignores cross terms between `A`, `R`, and `B`. This is a
  deliberate cost tradeoff, but ablations should mention it.
- Coefficient function-space metrics can destabilize denominators if safety is
  too weak.
- Curve amplitude transport is function-preserving only within coefficient
  clipping/headroom limits.
- Runtime improvements must be measured separately from loss improvements.

## What Would Make This A Real ICLR Optimizer Story

The story should be:

```text
RLB creates identifiable optimizer geometry: domain selection, rational curve
shape, recombination, and a positive scale quotient. QN-MatrixPolicy is a
cheap quotient natural optimizer for that geometry. It improves token
efficiency and/or final loss while reducing overhead enough to be practical.
```

Minimum evidence:

- current MatrixPolicy vs QN-MatrixPolicy on matched DCLM and FineWeb-Edu;
- SiLU+AdamW, RLB+AdamW, SiLU+Lion, RLB+Lion, SiLU+Muon, RLB+Muon controls;
- token-to-target savings;
- wall-clock/token throughput;
- quotient diagnostics: gauge drift and correction magnitude;
- metric diagnostics: gain dispersion and role budgets;
- safety diagnostics: denominator margin and atom headroom;
- ablations for quotient projection, owned-state moments, functional metric,
  role budget, coefficient movement, and curve transport.

The key paper claim should not be "we found a better activation." It should be:

```text
Structured activations can expose optimizer-visible geometry, and exploiting
that geometry produces better language-model training dynamics than generic
parameter-space optimizers.
```
