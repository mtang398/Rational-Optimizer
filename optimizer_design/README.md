# Optimizer Design

This folder contains the RLB-specific optimizer implementation. The current research method is `RationalMatrixPolicyOptimizer`, wired through `rational_matrix_policy_onpolicy`.

## Method

RLB exposes optimizer structure that a normal SiLU/SwiGLU FFN does not:

| RLB object | optimizer handle |
| --- | --- |
| `W_in` | chooses rational input domains and derivative exposure. |
| rational basis | supplies learnable nonlinear shape inside each domain. |
| `W_out` | recombines rational features into the residual stream. |
| positive gauge | permits scale rebalance between `W_in` and `W_out`. |
| live stats | can reveal group activity, pressure, and saturation. |

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

## Verified Result

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

The current method is promising because it wins on WikiText under the same LR schedule, but the margin is still too small for the final goal.

## Benchmark Targets

The next short benchmarks should test rational optimizer behavior without saturating. Good candidates are `synthetic/rule_chain_hard`, `synthetic/key_value_recall`, `synthetic/carry_arithmetic`, `synthetic/stack_brackets`, and `synthetic/noisy_copy_transform`.

The acceptance rule is simple: if a generic control reaches loss `<0.1` at the target budget, the task is too easy to support an optimizer claim. MatrixPolicy needs tests where the final loss scale leaves room for a real `0.2-0.3` gap, or at least enough headroom that PPL differences are not compressed to noise.

## Current Research Problem

The sparse synthetic curves suggest a useful signal and a real problem, but they are under-sampled. The useful signal is speed at the observed checkpoints; the problem is that this early rational advantage often compresses near the final loss floor or is not preserved late. The dense rerun logs training every 10 steps and validation every 25 steps, which is the minimum needed before making a stronger curve claim. The next optimizer should preserve the early curve win on harder tasks and later training, likely by using live rational group information more selectively rather than adding another global schedule.
