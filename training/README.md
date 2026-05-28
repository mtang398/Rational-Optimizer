# Training

This folder contains the benchmark entrypoints and Slurm launchers.

## Files

```text
transformer_wikitext103_compare.py      model, activations, optimizers, WikiText and synthetic tasks
run_wikitext103_optimizer_sweep.sbatch  common 4x A6000 sweep launcher
aggregate_wikitext103_multiseed.py      JSONL aggregation helper
```

## Main Benchmark

```text
dataset:      Salesforce/wikitext, wikitext-103-raw-v1
task:         causal language modeling
tokenizer:    GPT-2 tokenizer
model:        LLaMA-style decoder-only Transformer
size:         about 123M parameters
layers:       12
width:        d_model 768
heads:        12
sequence:     256 tokens
```

The synthetic transfer task is selected with `--dataset-name synthetic/arithmetic --dataset-config v1` and uses the same 100M-token, 123M-parameter training setup.

## GPU Rule

Use A6000 GPUs only. Each sweep job requests 4 GPUs, so run at most two concurrent jobs.

```bash
squeue -u mt872
```

The A6000 nodes need the PyTorch fallback because the checked-in CUDA extension was not built with a usable A6000 kernel image:

```text
RATIONAL_OPT_TORCH_FALLBACK=1
```

The A6000 fallback runs used `--batch-size 16 --grad-accum 2`, preserving `32768` global tokens per step.

## Current Best Optimizer

```text
optimizer                                           rational_matrix_policy_onpolicy
activation                                          rlb_fused_fixed_strong_ffn
optimizer_lr                                        3e-4
optimizer_min_lr                                    3e-5
warmup_steps                                        200
lr_schedule                                         cosine
rational_matrix_policy_backbone_optimizer           adamw
rational_matrix_policy_backbone_beta2               0.999
rational_matrix_policy_beta2                        0.999
rational_matrix_policy_adam_lr_scale                3.00
rational_matrix_policy_adam_lr_scale_final          null
rational_matrix_policy_adam_decay_start             1.10
rational_matrix_policy_adam_decay_end               1.10
rational_matrix_policy_adam_decay_depth_shift       0.00
rational_matrix_policy_adam_beta2_final             null
rational_matrix_policy_adam_beta2_decay_start       1.10
rational_matrix_policy_adam_beta2_decay_end         1.10
rational_matrix_policy_adam_beta2_decay_depth_shift 0.00
rational_matrix_policy_adam_role_strength           1.20
rational_matrix_policy_adam_stat_strength           0.00
rational_matrix_policy_adam_pressure_balance        0.00
rational_matrix_policy_adam_stat_start              0.00
rational_matrix_policy_adam_stat_end                0.00
rational_matrix_policy_adam_min_lr_scale            0.40
rational_matrix_policy_adam_max_lr_scale            4.00
rational_matrix_policy_input_depth_gain            -0.50
rational_matrix_policy_output_depth_gain            1.00
rational_matrix_policy_muon_strength                0.75
rational_matrix_policy_muon_lr_scale                1.00
rational_matrix_policy_max_muon                     0.75
rational_matrix_policy_min_muon                     0.00
rational_matrix_policy_final_muon                   0.00
rational_matrix_policy_start                        0.02
rational_matrix_policy_end                          0.12
rational_matrix_policy_decay_start                  0.20
rational_matrix_policy_decay_end                    0.36
rational_matrix_policy_muon_decay_depth_shift       0.00
rational_matrix_policy_muon_input_decay_shift       0.00
rational_matrix_policy_muon_output_decay_shift      0.00
rational_matrix_policy_muon_reset_adam_state        false
rational_matrix_policy_pressure_weight              0.30
rational_matrix_policy_activity_weight              0.65
rational_matrix_policy_weight_decay_scale           1.00
rational_matrix_policy_function_coeff               false
rational_matrix_policy_group_gain_strength          0.00
rational_matrix_policy_group_pressure_strength      0.00
rational_matrix_policy_group_activity_damping       0.00
rational_transport_strength                         0.00
rational_transport_matrix_strength                  0.00
rational_transport_quotient_strength                0.00
rational_adaptive_coeff_strength                    0.00
rlb_gauge_strength                                  0.50
rlb_gauge_start                                     0.03
rlb_gauge_end                                       0.35
rlb_gauge_every                                     5
```

## Optimizer Wiring

```text
1. backward pass computes gradients
2. outer on-policy wrapper updates live RLB statistics
3. transport, coefficient-function optimizer, and group-gradient policy are off by default
4. ordinary AdamW steps the non-RLB backbone, no-decay group, and rational coefficients
5. MatrixPolicy handles RLB W_in/W_out matrices:
   - compute role/depth AdamW scale
   - compute early Muon fraction
   - step AdamW with lr = global_lr * adam_scale * (1 - muon_fraction)
   - step Muon with lr = global_lr * muon_lr_scale * muon_fraction
   - restore scheduler-provided base group lrs
6. outer wrapper applies exact RLB gauge balance
```

## Run Current Best

```bash
env RATIONAL_OPT_TORCH_FALLBACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
  RUN_NAME=rlb_matrix_policy_muon_switch \
  STEPS=3051 SEEDS=1337 \
  OPTIMIZERS=rational_matrix_policy_onpolicy \
  ACTIVATIONS=rlb_fused_fixed_strong_ffn \
  EVAL_INTERVAL=250 EVAL_BATCHES=20 LOG_INTERVAL=100 \
  EXTRA_ARGS="--batch-size 16 --grad-accum 2" \
  sbatch --gres=gpu:nvidia_rtx_a6000:4 training/run_wikitext103_optimizer_sweep.sbatch
```

## Current Result

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

Do not headline LR ablations. The optimizer must win under the same global LR schedule against SiLU+AdamW, RLB+AdamW, and Muon controls.
