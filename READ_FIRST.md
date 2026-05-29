# Read First

This repo is an optimizer research repo. The README should read like a compact result report: Method, Result, Graphs, Status. Exact flag dumps belong in launchers and JSONL `config` records.

## Claim Boundary

```text
verified current best: RLB MatrixPolicy-Muon on WikiText-103
verified gap:         +0.0731 loss / +2.45 PPL over SiLU/SwiGLU+AdamW beta2=0.999
requested target:     0.2-0.3 loss gap, not reached yet
synthetic status:     Code and Symbolic complete, Reasoning mix running as job 951127
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

Synthetic status:

| task | best finished row so far | result | interpretation |
| --- | --- | --- | --- |
| Code | SiLU/SwiGLU+AdamW | 0.088975 loss, 1.0931 PPL | saturated task; MatrixPolicy is worse at final loss. |
| Symbolic | SiLU/SwiGLU+Muon | 0.038782 loss, 1.0395 PPL | deltas are too small to claim a real win from one seed. |
| Reasoning mix | pending rerun | job 951127 | SiLU+AdamW complete, remaining rows still running. |

The completed synthetic tasks are near saturation, so tiny final-loss/PPL differences are not strong evidence. At loss `0.04-0.09`, a `0.001` loss difference barely moves PPL and can be seed/order noise. Treat Symbolic as diagnostic, not a win. Code is more useful as a negative diagnostic because MatrixPolicy is consistently behind there, but even that should not be overclaimed from one seed. The meaningful target remains a much larger same-LR gap, or a harder task where final loss is not already near zero.

## Plots

The top-level README links the verified WikiText and arithmetic plots under:

```text
experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/
```

Synthetic Code/Symbolic/Reasoning mix plots should be generated only after Reasoning mix finishes, so the graph set is not mixing complete and partial rows.

## Running Jobs

Use A6000 only and keep total active allocation at or below 8 A6000s.

```bash
squeue -u mt872
```

Current continuation:

```text
job:      951127
name:     synth-reason
GPUs:     4x nvidia_rtx_a6000
requeue:  enabled
```

The launcher is restart-safe at row granularity: completed rows are skipped, incomplete run directories are archived before rerun, and the job asks Slurm to requeue on the pre-timeout signal.
