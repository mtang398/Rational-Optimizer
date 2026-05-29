# RationalOPT

RationalOPT is about one question: can a rational FFN win because the optimizer understands rational structure, not because the run got a different global LR schedule?

The baseline is not a Jacobian optimizer. The baseline is the normal Transformer FFN stack: `SiLU/SwiGLU+AdamW`, `RLB+AdamW`, `SiLU/SwiGLU+Muon`, and `RLB+Muon`, all under the same training budget and base LR schedule.

## Current Claim

The best verified WikiText-103 result is `RLB MatrixPolicy-Muon`. It beats the strongest tuned AdamW control by `0.0731` validation loss and `2.45` PPL. That is a real same-LR win, but it is not yet the target-sized `0.2-0.3` loss gap.

The synthetic transfer story is still incomplete. Code and Symbolic finished; Reasoning mix is rerunning from scratch as Slurm job `951127` with requeue enabled. Do not claim a final synthetic result until that job finishes and the compact summary is regenerated.

## Core Intuition

1. RLB creates explicit rational feature groups instead of a GLU gate.
2. `W_in` controls which input range each rational basis sees; `W_out` controls how those basis features are recombined.
3. Those two matrices have different jobs, so the optimizer should not treat them like ordinary dense FFN matrices.
4. MatrixPolicy uses a short early Muon phase only on RLB matrices, then returns those matrices to role/depth-aware AdamW.
5. Exact gauge balance keeps per-group basis scale from drifting while preserving the represented function as much as possible.

The key idea is that RLB exposes optimizer handles that SiLU/SwiGLU does not have: rational input-domain formation, basis recombination, per-group scale gauge, and live group statistics. The current optimizer only partially uses those handles. The research direction is to use them more effectively without changing the global LR schedule.

## Current Method

```text
activation:      rlb_fused_fixed_strong_ffn
optimizer:       rational_matrix_policy_onpolicy
base schedule:   same 3e-4 -> 3e-5 warmup/cosine schedule as controls
backbone:        AdamW
RLB matrices:    MatrixPolicy AdamW plus early matrix-only Muon
coefficients:    AdamW by default
gauge balance:   enabled
```

Exact flags live in `experiments/scripts/run_synthetic_fair_full_20260529.sh` and each run's JSONL `config` event.

The method is deliberately narrower than a global optimizer replacement. Ordinary non-RLB weights still use AdamW. MatrixPolicy only intervenes on RLB `W_in` and `W_out`, where the rational structure makes role-specific updates meaningful.

## Verified WikiText Result

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

Interpretation: MatrixPolicy-Muon is the only current method with a meaningful WikiText lead over the strongest controls. Plain Muon is not a stronger baseline here; both `SiLU/SwiGLU+Muon` and `RLB+Muon` are worse than AdamW controls.

## Synthetic Fair Rerun

| task | best finished row so far | result | interpretation |
| --- | --- | --- | --- |
| Code | SiLU/SwiGLU+AdamW | 0.088975 loss, 1.0931 PPL | RLB and MatrixPolicy lose final loss on this task. |
| Symbolic | SiLU/SwiGLU+Muon | 0.038782 loss, 1.0395 PPL | RLB generic optimizers and group-stat are close, but gains are tiny. |
| Reasoning mix | pending rerun | job 951127 | No claim until all six rows finish from scratch. |

The completed synthetic rows are useful mainly as diagnostics. Code says the current MatrixPolicy can over-specialize and lose final loss after early progress. Symbolic says rational structure can help, but the margin is tiny. Reasoning mix is the important transfer check because it mixes arithmetic, code, and symbolic patterns.

## What Seems To Help

| design choice | why it helps |
| --- | --- |
| Early matrix-only Muon | gives RLB matrices a fast orthogonalized start without leaving Muon on late. |
| Role/depth asymmetry | `W_in` and `W_out` do different jobs, and late layers should not be updated identically to early layers. |
| AdamW after the early switch | keeps late training stable after the useful Muon window. |
| Gauge balance | prevents rational groups from wasting update budget on scale drift. |

## What Has Not Helped Enough

| design choice | readout |
| --- | --- |
| Full-model Muon | worse than AdamW controls on WikiText. |
| Late Muon tails | worsened early probes. |
| Beta2 tail schedules | only noise-level changes. |
| Function-space coefficient variants | worse first signals. |
| Stronger/weaker role-depth scaling | worse than current default in probes. |
| Group-stat MatrixPolicy | pending as a fair transfer row; not yet a claimed improvement. |

## Active Job

```text
job:        951127
name:       synth-reason
purpose:    rerun synthetic/reasoning_mix from scratch
GPUs:       4x nvidia_rtx_a6000
requeue:    enabled
run root:   experiments/runs/synthetic_fair_reasoning_mix_20260529/
```

After it finishes:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py \
  --run-root experiments/runs/synthetic_fair_reasoning_mix_20260529
```

Raw JSONL and Slurm logs stay local under `experiments/runs/`. Commit only compact summaries, plots, scripts, and README updates.

## Repo Layout

```text
activation/         RLB activation implementation and CUDA fallback path
training/           LLM benchmark harness and synthetic task generators
optimizer_design/   RLB-specific optimizer implementations
experiments/        Slurm launchers and compact result artifacts
```
