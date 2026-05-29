# Training

This folder contains the LLM harness and synthetic task generators used to compare optimizers fairly.

## Comparisons

Every serious run should include:

```text
SiLU/SwiGLU+AdamW
RLB+AdamW
SiLU/SwiGLU+Muon
RLB+Muon
RLB MatrixPolicy
```

The global LR schedule, token budget, seed, batch shape, eval cadence, and model size should match across rows.

## Tasks

| task | purpose | current meaning |
| --- | --- | --- |
| WikiText-103 | main language-modeling benchmark | MatrixPolicy-Muon has the verified lead. |
| synthetic/code | structured program-like patterns | RLB drops much faster early; final loss is saturated. |
| synthetic/symbolic | rewrite/parity/bracket/copy patterns | MatrixPolicy is fastest early; final deltas are tiny. |
| synthetic/reasoning_mix | mixed arithmetic/code/symbolic transfer | MatrixPolicy/group-stat lead early and mid curve; final deltas are tiny. |

## Low-Loss Warning

The completed synthetic tasks are near saturation, so final-loss/PPL differences are not the main readout. The important signal is the curve: Code step 250 has MatrixPolicy group-stat at `0.1661` loss / `1.1807` PPL versus `SiLU/SwiGLU+AdamW` at `0.4895` / `1.6314`; Reasoning mix step 250 has MatrixPolicy at `0.3450` / `1.4120` versus `0.4127` / `1.5109`.

For these saturated synthetic tasks, graphs and early curves matter more than tiny final PPL differences. A future task should be harder if we want to test whether the early rational speed can become a decisive final-loss gap.

## Proposed Short Tasks

The current Code and Symbolic tasks are too close to saturation. The next synthetic tasks should be harder and should target final control loss around `0.25-1.0` at 1250 steps.

| task name | generator idea | primary signal |
| --- | --- | --- |
| `synthetic/rule_chain_hard` | sample rewrite rules, include distractors, ask for a 3-6 hop result. | compositional symbolic updates. |
| `synthetic/key_value_recall` | random in-context key-value table, delayed query, variable distractor density. | binding and retrieval under context noise. |
| `synthetic/carry_arithmetic` | 3-6 digit arithmetic with carries, signs, and irrelevant nearby numbers. | sharp algorithmic boundaries. |
| `synthetic/stack_brackets` | typed brackets, deeper nesting, decoy brackets, ask for stack state/core token. | state tracking and nonlinear transitions. |
| `synthetic/noisy_copy_transform` | copy/reverse/map a random span with noise and variable span length. | sequence transform robustness. |

Acceptance rule: if the baseline control reaches loss `<0.1` by 1250 steps, the task is too easy for optimizer claims. Keep it only as a smoke test.

## Step-1 Curves

Fresh runs log validation at step 1, then at the eval interval, then at the final step. Plots should start at step 1.

## Active Run

```text
937608: completed Code and Symbolic, preempted during Reasoning mix
951127: reran Reasoning mix from scratch with Requeue=1 and completed
```

```bash
sbatch --job-name=synth-reason \
  --export=ALL,SYNTHETIC_TASKS=synthetic/reasoning_mix,RUN_SUFFIX=20260529_reasoning_rerun,OUTPUT_ROOT=experiments/runs/synthetic_fair_reasoning_mix_20260529 \
  experiments/scripts/run_synthetic_fair_full_20260529.sh
```

Completed artifact: `experiments/results/synthetic_fair_full_2026_05_29/`. Regenerate the combined artifact with:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py
```

The default summary combines Code/Symbolic from `experiments/runs/synthetic_fair_full_20260529/` with the clean Reasoning mix rerun from `experiments/runs/synthetic_fair_reasoning_mix_20260529/`.

## A6000 Rule

Use A6000 GPUs only and do not exceed 8 active A6000s total.

```text
--gres=gpu:nvidia_rtx_a6000:4
RATIONAL_OPT_TORCH_FALLBACK=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_DISABLE=1
```
