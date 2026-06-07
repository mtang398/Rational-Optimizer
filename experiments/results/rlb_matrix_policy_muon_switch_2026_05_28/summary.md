# WikiText-103 Same-LR MatrixPolicy Result

This artifact keeps the useful WikiText-103 anchor from May 28. The tracked evidence in this folder is the same-LR real-LM comparison: summary tables, dense validation curves, perplexity curves, and training-loss curves.

## Current Best Row

```text
optimizer:   rational_matrix_policy_onpolicy
activation:  rlb_fused_fixed_strong_ffn
mechanism:   early RLB-matrix Muon switch, then MatrixPolicy AdamW
global LR:   lr=3e-4, min_lr=3e-5, unchanged across controls
seed:        1337
steps:       3051
```

| row | final loss | final PPL |
| --- | ---: | ---: |
| RLB MatrixPolicy-Muon | 3.476232 | 32.34 |
| RLB Smooth-MatrixPolicy | 3.493210 | 32.89 |
| SiLU+AdamW beta2=0.999 | 3.549346 | 34.79 |
| RLB+AdamW beta2=0.999 | 3.550018 | 34.81 |
| RLB+AdamW | 3.617501 | 37.24 |
| SiLU+AdamW | 3.621982 | 37.41 |
| SiLU+Muon | 3.644921 | 38.28 |
| RLB+Muon | 3.657877 | 38.78 |

## Gap Readout

```text
vs SiLU+AdamW beta2=0.999: 0.073114 loss / 2.45 PPL
vs RLB+AdamW beta2=0.999:  0.073786 loss / 2.48 PPL
vs SiLU+AdamW:             0.145750 loss / 5.07 PPL
vs RLB+AdamW:              0.141269 loss / 4.91 PPL
vs SiLU+Muon:              0.168689 loss / 5.94 PPL
vs RLB+Muon:               0.181645 loss / 6.44 PPL
```

The result is a same-LR optimizer win, but the strongest current evidence is now the May 30 FineWeb/FineWeb-Edu screen because those gaps are larger on modern real-corpus data.

## Plots

Validation starts at the first eval point. Training loss starts at step 1.

![Same-LR validation loss](same_lr_validation_loss.png)

![Same-LR validation PPL](same_lr_validation_ppl.png)

![Same-LR training loss from step 1](same_lr_training_loss_from_step1.png)

## Reproduce Current Best

```bash
env RATIONAL_OPT_TORCH_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
  RUN_NAME=rlb_matrix_policy_muon_switch \
  STEPS=3051 SEEDS=1337 \
  OPTIMIZERS=rational_matrix_policy_onpolicy \
  ACTIVATIONS=rlb_fused_fixed_strong_ffn \
  EVAL_INTERVAL=250 EVAL_BATCHES=20 LOG_INTERVAL=100 \
  EXTRA_ARGS="--dataset-name Salesforce/wikitext --dataset-config wikitext-103-raw-v1 --batch-size 16 --grad-accum 2" \
  sbatch --gres=gpu:nvidia_rtx_a6000:4 training/run_lm_optimizer_sweep.sbatch
```
