# RationalOPT

RationalOPT asks a research question: can a rational FFN win because the optimizer understands rational structure, not because it got a different global LR schedule?

The real controls are `SiLU/SwiGLU+AdamW`, `RLB+AdamW`, `SiLU/SwiGLU+Muon`, and `RLB+Muon` under the same model, token budget, seed, eval cadence, and base LR schedule. Jacobian, quotient, transport, and coefficient variants are ablations, not baselines.

## Method

RLB is not a GLU. It creates grouped rational features:

```text
v = x W_in
s_g = sqrt(mean(v_g^2) + eps)
u_g = v_g / s_g
h_g = s_g R_g(u_g)
y = h W_out
```

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

Current method in one line:

```text
RLB MatrixPolicy-Muon = AdamW backbone + AdamW rational coefficients + role/depth-aware RLB matrix AdamW + early RLB-matrix-only Muon + gauge rebalance
```

Exact scalar flags live in the Slurm launchers and JSONL `config` records. They are not the research story.

## Main Result

The verified WikiText-103 result is a real but modest same-LR win. It is not yet the desired `0.2-0.3` loss gap.

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

Best verified gap versus `SiLU/SwiGLU+AdamW beta2=0.999`: `0.0731` loss and `2.45` PPL.

## Graphs

![WikiText validation loss](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_loss.png)

![WikiText validation PPL](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_validation_ppl.png)

![WikiText training loss from step 1](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/same_lr_training_loss_from_step1.png)

![Synthetic arithmetic validation loss](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/synthetic_arithmetic_validation_loss.png)

![Synthetic arithmetic validation PPL](experiments/results/rlb_matrix_policy_muon_switch_2026_05_28/synthetic_arithmetic_validation_ppl.png)

The completed synthetic fair plots are now committed:

![Synthetic fair final loss](experiments/results/synthetic_fair_full_2026_05_29/final_loss_by_task.png)

![Synthetic fair final PPL](experiments/results/synthetic_fair_full_2026_05_29/final_ppl_by_task.png)

![Reasoning mix validation loss](experiments/results/synthetic_fair_full_2026_05_29/synthetic_reasoning_mix_validation_loss.png)

![Reasoning mix validation PPL](experiments/results/synthetic_fair_full_2026_05_29/synthetic_reasoning_mix_validation_ppl.png)

## Synthetic Transfer Status

| task | best row | result | interpretation |
| --- | --- | --- | --- |
| Code | SiLU/SwiGLU+AdamW | 0.088975 loss, 1.0931 PPL | saturated task; MatrixPolicy is worse at final loss. |
| Symbolic | SiLU/SwiGLU+Muon | 0.038782 loss, 1.0395 PPL | deltas are too small to claim a real win from one seed. |
| Reasoning mix | RLB+Muon | 0.144238 loss, 1.1552 PPL | tiny generic-RLB win; MatrixPolicy does not win meaningfully. |

The completed synthetic tasks are near saturation, so tiny final-loss/PPL differences are not strong evidence. At loss `0.04-0.15`, a `0.001-0.002` loss difference barely moves PPL and can be seed/order noise. Treat Symbolic and Reasoning mix as diagnostics, not wins. Code is more useful as a negative diagnostic because MatrixPolicy is consistently behind there, but even that should not be overclaimed from one seed. The meaningful target remains a much larger same-LR gap, or a harder task where final loss is not already near zero.

## Next Short Tests

The next tests should be short enough to run on 4x A6000, but hard enough that final loss is not already near zero. A useful screening target is final control loss around `0.25-1.0`; below `0.1`, PPL differences are usually too compressed to support an optimizer claim.

| proposed task | what it tests | why it is better than current saturated tasks |
| --- | --- | --- |
| `synthetic/rule_chain_hard` | 3-6 step symbolic rule composition with distractor rules and held-out symbols. | Tests rational piecewise composition while keeping loss away from the floor. |
| `synthetic/key_value_recall` | Random key-value tables followed by delayed queries inside the same context. | Tests whether RLB matrix policy helps binding/retrieval rather than template memorization. |
| `synthetic/carry_arithmetic` | Multi-digit addition/subtraction with carries, signs, and distractor numbers. | Harder than current arithmetic; carries create sharp local decision boundaries. |
| `synthetic/stack_brackets` | Pushdown-style bracket/type tracking with deeper nesting and decoy brackets. | Tests state-like nonlinear transitions instead of one-step bracket depth templates. |
| `synthetic/noisy_copy_transform` | Copy/reverse/map spans with random noise tokens and variable span length. | Keeps sequence loss meaningful and exposes whether early matrix conditioning helps sequence transforms. |

A proposed task should be rejected as a benchmark if `SiLU/SwiGLU+AdamW` reaches loss `<0.1` by 1250 steps. At that point it can still be a smoke test, but not a meaningful optimizer-discrimination task.

## Completed Synthetic Run

```text
937608: Code and Symbolic complete, preempted during Reasoning mix
951127: Reasoning mix rerun complete, 02:40:56 elapsed, Requeue=1
artifact: experiments/results/synthetic_fair_full_2026_05_29/
```

The synthetic artifact contains CSV summaries and loss/PPL plots for Code, Symbolic, and Reasoning mix. Regenerate it with:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py
```

Raw JSONL and Slurm logs stay local under `experiments/runs/`.

## Layout

```text
activation/         RLB activation implementation
training/           LLM benchmark harness and synthetic task generators
optimizer_design/   RLB-specific optimizer implementation
experiments/        Slurm launchers and compact result artifacts
```
