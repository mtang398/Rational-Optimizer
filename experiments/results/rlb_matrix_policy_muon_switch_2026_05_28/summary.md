# RLB MatrixPolicy-Muon Same-LR Result

This artifact replaces the older stacked result folders. It reports the current best same-global-LR optimizer story only.

## Current Best

```text
optimizer:   rational_matrix_policy_onpolicy
activation:  rlb_fused_fixed_strong_ffn
mechanism:   early RLB-matrix Muon switch, then MatrixPolicy AdamW
global LR:   lr=3e-4, min_lr=3e-5, warmup=200, cosine, unchanged
seed:        1337
steps:       3051
```

Best final: `3.476232` validation loss, `32.34` PPL.

| row | final loss | final PPL | loss gap vs best | PPL gap vs best |
| --- | ---: | ---: | ---: | ---: |
| RLB MatrixPolicy-Muon (best) | 3.476232 | 32.34 | 0.000000 | 0.00 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 | 0.016978 | 0.55 |
| SiLU+AdamW beta2=0.999 | 3.549346 | 34.79 | 0.073114 | 2.45 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 | 0.073786 | 2.48 |
| RLB+AdamW | 3.617501 | 37.24 | 0.141269 | 4.91 |
| SiLU+AdamW | 3.621982 | 37.41 | 0.145750 | 5.07 |

## Gap Readout

- Versus SiLU+AdamW: `0.145750 loss / 5.07 PPL`.
- Versus RLB+AdamW: `0.141269 loss / 4.91 PPL`.
- Versus SiLU+AdamW beta2=0.999: `0.073114 loss / 2.45 PPL`.
- Versus RLB+AdamW beta2=0.999: `0.073786 loss / 2.48 PPL`.
- Versus previous Smooth-MatrixPolicy: `0.016978 loss / 0.55 PPL`.

The final PPL gap clears the 2-3 PPL target versus the beta2-tuned controls, but the final loss gap is still below the requested 0.2-0.3. LR ablations should remain out of the main claim until the same-LR loss gap is much larger.

## Plots

Validation starts at the first eval point, step 250. Training loss starts at step 1.

![Same-LR validation loss](same_lr_validation_loss.png)

![Same-LR validation PPL](same_lr_validation_ppl.png)

![Same-LR training loss from step 1](same_lr_training_loss_from_step1.png)

## What Improved

- The durable base is still the RLB matrix policy: separate `W_in/W_out`, role/depth scaling, beta2=0.999, exact gauge balance.
- A short early Muon phase on only the RLB matrices gives a better trajectory than pure MatrixPolicy AdamW.
- The winning switch is aggressive enough to help early (`muon_lr_scale=1.0`) but turns off early enough to avoid a late penalty.
- Keeping Adam moments across the switch beat resetting them at this strength, so the default reset flag stays off.

## What Got Worse

| probe | result |
| --- | --- |
| lower Muon strength 0.55 | full final `3.482822`, worse than best |
| Muon lr-scale 1.25 | stronger short result, worse full final `3.479543` |
| Muon lr-scale 1.10 | full final `3.477492`, close but worse than best |
| post-Muon Adam reset | full final `3.478842`, worse than keeping moments |
| global earlier Muon shutoff | lost the 750-step gain, cancelled |
| on-policy damping of Muon | worsened the 1000-1250 region, cancelled |
| layer/role timing shifts | killed the useful 750-step gain, cancelled |
| covariant Adam state under gauge | short final `4.069776`, worse than matched control |
| RLB matrix weight-decay reduction | short final `4.075825`, worse than matched control |

## Reproduce

No extra MatrixPolicy args are needed after this commit because the defaults match the current best.

```bash
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
  RUN_NAME=rlb_matrix_policy_muon_switch \
  STEPS=3051 SEEDS=1337 \
  OPTIMIZERS=rational_matrix_policy_onpolicy \
  ACTIVATIONS=rlb_fused_fixed_strong_ffn \
  EVAL_INTERVAL=250 EVAL_BATCHES=20 LOG_INTERVAL=100 \
  sbatch --time=02:00:00 --gres=gpu:nvidia_rtx_6000_ada_generation:4 \
  training/run_wikitext103_optimizer_sweep.sbatch
```
