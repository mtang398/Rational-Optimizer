# ICLR Experiment Plan

This plan treats the claim as optimizer-specific: RLB exposes useful optimizer-visible geometry, and MatrixPolicy uses that geometry better than generic AdamW or Muon under the same base training protocol.

## Hard Resource Rules

- At most 4 A6000 GPUs per Slurm job.
- At most 8 A6000 GPUs active at one time.
- Keep the repository below 200G. The real-LM launcher currently refuses to run above 190 GiB.

## Primary Claim Table

Run the exact five-row comparison on real web corpora:

1. `SiLU+AdamW`
2. `RLB+AdamW`
3. `SiLU+Muon`
4. `RLB+Muon`
5. `RLB+MatrixPolicy (group-stat)`

Use the same model size, GPT-2 tokenizer, sequence length, global tokens per step, base LR schedule, weight decay, validation slice, evaluation cadence, and token budget across all rows.

Minimum paper threshold:

- FineWeb and FineWeb-Edu each have at least 3 seeds.
- MatrixPolicy has positive mean validation-loss gap against `SiLU+AdamW`.
- MatrixPolicy has positive mean validation-loss gap against the best non-MatrixPolicy control per seed.
- AUC gaps agree with final-loss gaps.
- Divergence, especially `RLB+AdamW` on FineWeb-Edu, is reported rather than hidden.

## Current Run Batch

Seed `1337` already exists from the May 30 screen. The May 31 batch added seed `2027`:

- Job `968205`: FineWeb, seed `2027`, 4 A6000.
- Job `968204`: FineWeb-Edu, seed `2027`, 4 A6000.

No additional independent GPU jobs should be submitted while both are running; dependent jobs are allowed only if they cannot raise active use above two 4-GPU jobs.

Seed `3407` is queued with dependencies, so it can only start after the corresponding seed-`2027` job succeeds:

- Job `968492`: FineWeb, seed `3407`, after `968205`.
- Job `968491`: FineWeb-Edu, seed `3407`, after `968204`.

This gives 3 seeds total without exceeding the 8-GPU active cap.

## Completed Replication, 2026-06-01

All four seed-`2027`/seed-`3407` Slurm jobs completed with exit code `0:0`.

Resource use stayed within the hard limits:

- 4 A6000 GPUs per job.
- At most two jobs active at once, so 8 A6000 GPUs total.
- Repository size after summary generation: 8.7G.

Summary table from `experiments/results/real_lm_multiseed_2026_05_31/summary.md`:

| task | method | seeds | mean val loss | mean PPL | mean gap vs SiLU+AdamW | mean gap vs best control |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FineWeb | SiLU+AdamW | 3 | 4.528963 | 92.69 | 0.000000 | -0.006960 |
| FineWeb | RLB+AdamW | 3 | 4.522311 | 92.08 | 0.006653 | -0.000308 |
| FineWeb | SiLU+Muon | 3 | 4.566661 | 96.28 | -0.037698 | -0.044658 |
| FineWeb | RLB+Muon | 3 | 4.571341 | 96.70 | -0.042377 | -0.049337 |
| FineWeb | RLB+MatrixPolicy (group-stat) | 3 | 4.369701 | 79.04 | 0.159263 | 0.152302 |
| FineWeb-Edu | SiLU+AdamW | 3 | 4.223572 | 68.28 | 0.000000 | -0.000748 |
| FineWeb-Edu | RLB+AdamW | 3 | 5.618928 | 1545.54 | -1.395356 | -1.396103 |
| FineWeb-Edu | SiLU+Muon | 3 | 4.258871 | 70.74 | -0.035300 | -0.036047 |
| FineWeb-Edu | RLB+Muon | 3 | 4.263744 | 71.08 | -0.040173 | -0.040920 |
| FineWeb-Edu | RLB+MatrixPolicy (group-stat) | 3 | 4.069422 | 58.52 | 0.154149 | 0.153402 |

Interpretation:

- The main result replicated across 3 seeds on both real web corpora.
- MatrixPolicy beats `SiLU+AdamW` and the best non-MatrixPolicy control by about 0.15 validation loss on both tasks.
- Plain `RLB+AdamW` is not the story: it is unstable or only marginal, including one FineWeb-Edu divergence. The paper claim should be that RLB creates optimizer-visible geometry and MatrixPolicy uses it.

## External Venue Guidance Checked, 2026-06-01

ICLR guidance and recent optimizer literature imply the next bar:

- ICLR explicitly encourages a reproducibility statement, source code/supplemental material, and detailed data processing/protocol descriptions.
- ICLR reviewing guidance says new experiments during rebuttal should only validate existing claims, so the original submission must already include the critical baselines and ablations.
- ICLR accepts optimization as a subject area, but the contribution must convincingly demonstrate new, relevant, impactful knowledge rather than only chasing SOTA.
- Recent optimizer benchmarking work argues that fair optimizer papers need tuned baselines, final-checkpoint comparisons, multiple model scales, multiple data-to-model ratios, and caution that intermediate rankings can flip during LR decay.

Operational consequence: the paper should not rely on one short 100M-token budget and untuned controls. The next phase must test whether the gap survives better baselines, larger budgets/scales, and method ablations.

## ICLR-Grade Next Plan

Phase 1: lock the current claim and make it reproducible.

- Add a one-command reproduction entry point for the 3-seed table.
- Save exact commands, Slurm job IDs, dataset slices, token budgets, validation skip tokens, seed list, and commit hash in the result directory.
- Add plots for validation loss, validation PPL, and train loss across all 30 completed rows.
- Report final loss and AUC-to-1000/2000/full, because early curves alone are not enough.
- Add bootstrap confidence intervals over seeds for MatrixPolicy gaps.

Phase 2: isolate the algorithm.

- Full-run ablations on FineWeb-Edu first: no group stats, no early Muon phase, no exact gauge rebalance, group gain only, group pressure only, group activity damping only.
- Promote ablations to FineWeb only if FineWeb-Edu does not collapse.
- Add diagnostics showing gauge imbalance, group norm products, derivative/output RMS, denominator-risk proxy, and function-space movement per update.
- The target mechanism claim is: MatrixPolicy improves function-changing movement while suppressing gauge drift or unstable rational coefficient movement.

Phase 3: make the baseline comparison harder.

- Tune `SiLU+AdamW`, `RLB+AdamW`, and MatrixPolicy LR/weight decay on a small grid under the same token budget.
- Add at least one modern competitive optimizer baseline if feasible in the repo: SOAP/Shampoo-style matrix preconditioning, Sophia, Adam-mini, or a stronger Muon implementation.
- Compare wall-clock and tokens-to-target, not only final loss. MatrixPolicy is slower than SiLU+AdamW per step, so the paper must report whether lower loss compensates for optimizer overhead.

Phase 4: scale and data-budget tests.

- Repeat the five-row table at one larger model size that still fits 4 A6000 GPUs per job.
- Run at least two token budgets, for example 100M and 300M/500M tokens, to check whether the gap survives LR decay and longer training.
- Add a third corpus or evaluation slice, preferably DCLM or Dolma sample, so the story is not only FineWeb/FineWeb-Edu.
- Run a small transfer/perplexity sanity check on held-out datasets that were not used for validation-slice selection.

Phase 5: paper structure.

- Main paper: one clean claim, one optimizer algorithm box, one primary 3-seed table, one ablation table, one mechanism figure, one compute/throughput table.
- Appendix: exact configs, all curves, failed/diverged rows, hyperparameter search ranges, hardware, environment, and reproducibility statement.
- Do not hide the `RLB+AdamW` divergence. Use it to motivate why rational geometry needs optimizer-specific handling.

## Live Monitoring Notes, 2026-05-31

- FineWeb job `968205` requeued once. The launcher has been patched so future requeues archive only incomplete activation JSONL files and skip already complete activation rows.
- The completed archived FineWeb seed-`2027` `SiLU+AdamW` row remains counted by the multi-seed summarizer.
- FineWeb-Edu seed-`2027` `SiLU+AdamW` completed at `4.223898` validation loss, matching seed `1337` (`4.225019`).
- FineWeb-Edu seed-`2027` `RLB+AdamW` did not reproduce the seed-`1337` early nonfinite failure. It is finite past step `1150`, so plain RLB+AdamW failure should be described as seed-sensitive instability, not deterministic divergence.

## Diagnostics Needed For Mechanism

Add diagnostics only after the queued replication jobs finish, because these jobs launch rows sequentially from the working tree.

Per RLB layer and global summary:

- `W_in` group norm, `W_out` group norm, and norm product.
- Gauge imbalance before and after rebalance.
- Derivative RMS and output RMS from `_rlb_optimizer_stats`.
- Group pressure from on-policy `in_rel_ema`, `out_rel_ema`, `rat_rel_ema`.
- Coefficient norm and denominator-risk/pole-margin proxy.
- Function-space probe delta on a fixed grid for rational curves.

Paper-level mechanism criterion:

- MatrixPolicy should show better validation loss/AUC while spending less update magnitude on pure gauge drift, or more update magnitude on measured function change per parameter change.

## Ablations

Run only after the seed-2027 replication is healthy:

1. MatrixPolicy without group-stat scaling.
2. MatrixPolicy without early Muon phase.
3. MatrixPolicy without exact gauge rebalance.
4. Group gain only.
5. Group pressure only.
6. Group activity damping only.

Each ablation can start as a 1000-step FineWeb-Edu probe with the same seed and protocol. Promote only the strongest non-degenerate ablation to the full 100M-token run.

## Optimizer Improvement Direction

Do not count global LR schedule changes as optimizer progress. Candidate v2 changes must use RLB structure:

- Use measured gauge imbalance to modulate rebalance strength or matrix role scale.
- Use denominator risk to reduce coefficient movement when rational curves approach unstable regions.
- Use derivative/output RMS to revive low-activity groups without amplifying already saturated groups.
- Use gradient agreement to avoid switching groups based on noisy one-step pressure.

Do not change the method while the seed-2027 or queued seed-3407 replication jobs can still launch rows from the working tree.
