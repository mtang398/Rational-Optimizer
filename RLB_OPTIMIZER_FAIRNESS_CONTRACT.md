# RLB optimizer fairness and execution contract

This is the binding fairness contract for the current optimizer-design cycle.
`optimizer_design/RLB_300M_4000_DESIGN_CONTRACT.md` fixes the exact scientific
cell. Launchers and validators must enforce both files and fail closed.

## Fixed scientific cell

- Exact historical M1 model configuration from row 465 of
  `experiments/manifests/iclr26_main_manifest.csv`: 18 layers, width 1024,
  16 heads, FFN width 3072, and sequence length 256.
- The M1 SwiGLU inventory is exactly `296,867,840` unique parameters. The M1
  Global-RLB inventory is exactly `296,871,080`: the difference is exactly
  3,240 RLB coefficients (18 layers x 18 groups x 10 coefficients).
- Exact DCLM training tensor containing 300,000,000 tokens.
- Exactly 4,000 optimizer updates, 32,768 global tokens per update, and
  131,072,000 sampled positions.
- Exact 8,000,000-token validation tensor beginning after token 610,000,000.
- Seed 1337 for design; additional seeds and datasets follow only after
  ablation closure.
- Batch 8/GPU, gradient accumulation 4, world size 4, and four generic RTX
  A6000 GPUs per row, with at most two rows active concurrently.

The cache paths and SHA-256 values in the design contract are mandatory.
Launchers run offline and terminate on any missing file, hash mismatch, tensor
length mismatch, or runtime-config mismatch. Smaller caches, regenerated
caches, streaming data, proxy models, and proxy update horizons are forbidden.

## Exact scalar equality

Every control, candidate, deletion ablation, pruned parent, and replication
uses these literal values:

- peak LR `0.0003`;
- minimum LR `0.00003`;
- weight decay `0.10` on the same decay-eligible partition;
- warmup `200` and cosine horizon `4000`;
- betas `(0.9, 0.95)`, epsilon `1e-8`, and gradient clip `1.0`.

All internal LR and WD scales equal `1.0`. This includes RLB `A`/`B`,
numerator/denominator, coefficient, layer, depth, semantic group, parameter
role, matrix, phase, and mechanism groups. No alternate scalar, schedule,
decay timing, clipping rule, or group multiplier may be introduced. Validators
compare the full realized LR trace, optimizer-group scalars, data-order hash,
initial-state hash, batch shape, precision, and distributed settings.

This rule is executable, not merely documentary. Every campaign launch passes
`--fairness-contract rlb-m1-300mtok-4000-unit-lr-wd-v2`. The trainer refuses a
nonunit outer group scale, a wrong decay partition, a missing/duplicate
parameter, or a nonstandard optimizer that does not enumerate all hidden LR/WD
multipliers through `lr_wd_fairness_audit()`. It rechecks every realized group
LR before and after every optimizer step. The signed run validator requires the
resulting passing inventory in the JSONL configuration record.

The trainer also aborts before the first update unless the architecture is
exactly M1, the parameter inventory equals the activation-specific count above,
the batch geometry is 8 x 4 accumulation x 4 GPUs, the horizon is 4,000, and
the loaded token tensors contain exactly 300,000,000 and 8,000,000 tokens. M0
(`12 x 768`) is a fatal contract violation.

SwiGLU and RLB intentionally differ only in FFN activation architecture.
Their data, seed, schedule, optimizer scalars, batch shape, model depth/width,
evaluation grid, and systems settings remain identical. Each decay-eligible
RLB matrix receives the same literal base WD as each decay-eligible SwiGLU
matrix; ordinary no-decay parameters remain no-decay in both.

## Controls and success gate

Two fresh 4,000-step controls are required:

1. SwiGLU + Muon.
2. Global-RLB + Muon.

The control with lower final validation loss is the primary hurdle. Existing
longer-run checkpoints cannot serve as controls because their cosine horizon
differs. Each of the three final methods must satisfy every rule below:

- final validation loss at least `0.20` below the stronger Muon control;
- strictly lower final validation PPL;
- strictly lower normalized train-loss and validation-loss partial AUC on the
  preregistered step-1000-through-step-4000 interval;
- complete structural certificates and finite execution;
- closed component ablations;
- final matched-runtime median and p95 ratios no larger than `1.02`.

Whole-run AUC remains descriptive. The late partial-AUC interval prevents the
initial transient from dominating convergence evidence.

## Structural design rules

- Every method derives from the coupled RLB rational parameterization and has
  an a-priori or a-posteriori mathematical explanation.
- Optimizer benefit must come from a structural direction, metric, transport,
  state, or constraint induced by RLB. Hyperparameter tuning earns no credit.
- The design pool contains at most ten carefully preregistered opaque
  candidates. Minor code repairs overwrite the same candidate files.
- A mathematical candidate freezes before launch and runs to 4,000 updates.
  Partial trajectories cannot promote or reject it.
- Full-endpoint evidence may reject a proposal during pre-launch reasoning.
- Ready rows enter dependency queues together so two four-GPU lanes remain
  occupied.

## Direct component ablation and recursive closure

For a full method `M = (c1, ..., ck)`, component `ci` is tested only by a
fresh 4,000-step run of `M \ ci` from the same frozen parent. Exactly `ci` is
removed; every other equation, component, scalar, seed, initialization, data
order, evaluation point, and systems setting stays fixed.

Candidate comparisons, stagewise construction history, grouped deletions,
short runs, diagnostics, certificates, and microbenchmarks cannot substitute
for a direct leave-one-out row. Components are mathematical actions in the
optimizer algorithm, determined from the full source code; implementation
helpers, logging, numerical checks, and fused kernels are not scientific
components.

A retained component must give its full parent at least `0.01` lower final
validation loss than the deletion, strictly lower final PPL, and strictly
lower late validation partial AUC. Train partial AUC is also reported for each
deletion. A component missing the contribution gate is removed. The pruned
method is rerun for 4,000 steps, then every remaining component is ablated
again from that exact pruned parent. Prune, rerun, and re-ablate until every
remaining component passes. All three finalists require this recursive
closure.

## Runtime and systems rules

- Each training row requests exactly four RTX A6000 GPUs; no more than two
  rows run at once.
- Launchers request no named node, exclusion list, topology, feature, or
  NVLink capability. `NCCL_P2P_DISABLE=1` is identical for every row.
- Node identity and end-to-end timing are recorded. Quality comparisons never
  use node speed as evidence.
- A sustained end-to-end slowdown of `3.0x` or more is an implementation
  defect; stop and repair the implementation without changing equations.
- Exploratory methods above `1.02x` may finish. Final qualification requires
  order-balanced, same-allocation timing with median and p95 at most `1.02x`.
- Speed optimization may change layout, fusion, allocation, caching, and
  scheduling only when parameter and state trajectories remain equivalent.
  Equations, operation order affecting values, recurrences, and scalars stay
  fixed.
- Campaign monitoring continues through completion and for at least five
  hours after the first scientific submission, with dependency repair and
  lane utilization checks.

## Candidate lifecycle and deletion

A scientific candidate is immutable after submission except for trajectory-
preserving implementation repair. A rejected mathematical method is terminal:
cancel descendants, delete its implementation, tests, launcher, run package,
logs, reports, and design document, and retain only its opaque identifier,
source hash, mechanism fingerprint, terminal metric vector, and rejection
reason in the compact registry. The rejected mechanism cannot be renamed,
repaired mathematically, composed into another candidate, or reused later.

Only the three methods satisfying every success, ablation, fairness, and
runtime gate remain as final method implementations.
