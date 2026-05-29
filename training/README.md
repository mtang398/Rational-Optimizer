# Training

This folder contains the LLM benchmark harness and launchers. The harness is used to compare optimizer behavior, not to tune unrelated schedules.

## What Is Being Compared

Every serious run should include these controls:

```text
SiLU/SwiGLU+AdamW
RLB+AdamW
SiLU/SwiGLU+Muon
RLB+Muon
RLB MatrixPolicy
```

The global LR schedule, token budget, seed, batch shape, eval cadence, and model size should match across rows. If the global schedule changes, it must change for controls too.

## Tasks

| task | purpose |
| --- | --- |
| WikiText-103 | main language-modeling benchmark. |
| synthetic/code | structured local program patterns; current MatrixPolicy loses final loss here. |
| synthetic/symbolic | rewrite/parity/bracket/copy patterns; RLB variants are slightly favorable but tiny. |
| synthetic/reasoning_mix | mixed arithmetic/code/symbolic transfer check; still running. |

All synthetic tasks use the same 123M-ish decoder-only model setup as the WikiText harness and generate local token streams.

## Step-1 Curves

Fresh runs log validation at step 1, then at the eval interval, then at the final step. Plots should start at step 1; treating step 250 or 1000 as the beginning is misleading.

## Active Fair Run

The full fair synthetic run was split by preemption, not by design:

```text
937608: completed Code and Symbolic, preempted during Reasoning mix
951127: reruns Reasoning mix from scratch with Requeue=1
```

The launcher is restart-safe at row granularity. On restart it skips complete rows and archives incomplete row directories before rerunning them.

```bash
sbatch --job-name=synth-reason \
  --export=ALL,SYNTHETIC_TASKS=synthetic/reasoning_mix,RUN_SUFFIX=20260529_reasoning_rerun,OUTPUT_ROOT=experiments/runs/synthetic_fair_reasoning_mix_20260529 \
  experiments/scripts/run_synthetic_fair_full_20260529.sh
```

## Current Readout

| task | best finished row so far | result | interpretation |
| --- | --- | --- | --- |
| Code | SiLU/SwiGLU+AdamW | 0.088975 loss, 1.0931 PPL | RLB and MatrixPolicy lose final loss on this task. |
| Symbolic | SiLU/SwiGLU+Muon | 0.038782 loss, 1.0395 PPL | RLB generic optimizers and group-stat are close, but gains are tiny. |
| Reasoning mix | pending rerun | job 951127 | No claim until all six rows finish from scratch. |

Do not overread the synthetic results. Code is negative for MatrixPolicy, Symbolic is tiny-positive for some RLB rows, and Reasoning mix is pending.

## A6000 Rule

Use A6000 GPUs only and do not exceed 8 active A6000s total.

```text
--gres=gpu:nvidia_rtx_a6000:4
RATIONAL_OPT_TORCH_FALLBACK=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_DISABLE=1
```

The fallback path keeps the same RLB math in PyTorch when the compiled extension is not usable on A6000.

## Summarize

For the active Reasoning mix rerun:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py \
  --run-root experiments/runs/synthetic_fair_reasoning_mix_20260529
```

For final documentation, merge the complete Code/Symbolic rows with the complete Reasoning mix rows into a compact result artifact under `experiments/results/`. Keep raw JSONL and Slurm logs local.
