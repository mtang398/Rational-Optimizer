# Read First

This repo is an optimizer research repo. The README should read like a compact result report: Method, Result, Graphs, Status. Exact flag dumps belong in launchers and JSONL `config` records.

## Claim Boundary

```text
verified current best: RLB MatrixPolicy-Muon on WikiText-103
verified gap:         +0.0731 loss / +2.45 PPL over SiLU/SwiGLU+AdamW beta2=0.999
requested target:     0.2-0.3 loss gap, not reached yet
synthetic status:     Code, Symbolic, and Reasoning mix complete
```

Do not present Jacobian or quotient optimizers as baselines. The important controls are generic AdamW and Muon on both SiLU/SwiGLU and RLB.

## Method

MatrixPolicy is an RLB-matrix optimizer, not a global LR scheduler. It leaves the base warmup/cosine schedule shared with the controls. The optimizer-specific move is local: treat `W_in` and `W_out` differently because they have different rational jobs.

`W_in` chooses the input domain seen by each rational group. `W_out` recombines the resulting rational features. The positive scale gauge means the same represented function can have bad or good matrix conditioning. MatrixPolicy tries to spend optimizer effort on useful function change instead of useless scale drift.

```text
for each optimizer step:
  update ordinary Transformer weights with AdamW
  update rational coefficients with AdamW
  for each RLB layer:
    read the matrix role: W_in or W_out
    read normalized layer depth
    assign a role/depth-specific MatrixPolicy AdamW scale
    during the early window, blend in Muon only for W_in/W_out
    after the early window, return those matrices to MatrixPolicy AdamW
  apply exact positive-gauge rebalance to each rational group
```

## Results

| row | final loss | final PPL | readout |
| --- | ---: | ---: | --- |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 | best verified row |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 | older smooth policy |
| SiLU/SwiGLU+AdamW beta2=0.999 | 3.549346 | 34.79 | strongest AdamW control |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 | generic AdamW on RLB |
| RLB+AdamW | 3.617501 | 37.24 | untuned generic AdamW |
| SiLU/SwiGLU+AdamW | 3.621982 | 37.41 | original AdamW control |
| SiLU/SwiGLU+Muon | 3.644921 | 38.28 | generic Muon control |
| RLB+Muon | 3.657877 | 38.78 | generic Muon on RLB |

Synthetic curve status:

| task | curve signal | strongest early gap vs `SiLU/SwiGLU+AdamW` |
| --- | --- | --- |
| Code | RLB drops much faster, then the task saturates. | step 250 MatrixPolicy group-stat: `0.1661` loss / `1.1807` PPL vs `0.4895` / `1.6314`. |
| Symbolic | MatrixPolicy is fastest early, but all methods are near solved. | step 250 MatrixPolicy: `0.0487` / `1.0499` vs `0.0609` / `1.0628`. |
| Reasoning mix | MatrixPolicy/group-stat lead early and mid curve. | step 250 MatrixPolicy: `0.3450` / `1.4120` vs `0.4127` / `1.5109`. |

The completed synthetic tasks are near saturation, so final-loss/PPL differences are not the main readout. The sparse synthetic run suggests faster rational drops, but it is not sampled densely enough for a final curve claim. The open optimizer problem is preserving the early rational speed on harder tasks and later training, without changing the shared global LR schedule.

## Plots

The top-level README links the verified WikiText and arithmetic plots under:

```text
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/
```

Synthetic Code/Symbolic/Reasoning mix plots are in:

```text
experiments/results/synthetic_fair_full_2026_05_29/
```

That artifact includes final loss/PPL bar charts, validation loss/PPL curves, training loss curves, curve diagnostics, and CSV/Markdown summaries. Regenerate it with:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py
```

## Next Task Standard

Use short tasks only if they are not saturated. The target control loss should stay around `0.25-1.0` at 1250 steps. If a task falls below loss `0.1`, it is a smoke test, not evidence of optimizer superiority.

Next candidates: `synthetic/rule_chain_hard`, `synthetic/key_value_recall`, `synthetic/carry_arithmetic`, `synthetic/stack_brackets`, and `synthetic/noisy_copy_transform`.

## Running Jobs

Use A6000 only and keep total active allocation at or below 8 A6000s.

```bash
squeue -u mt872
```

Latest completed continuation `951127` finished successfully with `Requeue=1`. Dense curve rerun `952433` is pending/running depending on scheduler state; it uses 4x A6000, `LOG_INTERVAL=10`, `EVAL_INTERVAL=25`, and `EVAL_BATCHES=10`.

The launcher is restart-safe at row granularity: completed rows are skipped, incomplete run directories are archived before rerun, and the job asks Slurm to requeue on the pre-timeout signal.
