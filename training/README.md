# Training

This folder contains the benchmark entrypoints and Slurm launchers.

## Files

```text
transformer_wikitext103_compare.py      model, activations, optimizers, WikiText and synthetic tasks
run_wikitext103_optimizer_sweep.sbatch  common 4x A6000 sweep launcher
aggregate_wikitext103_multiseed.py      JSONL aggregation helper
```

## Model And Data

```text
model:      LLaMA-style decoder-only Transformer
size:       about 123M parameters
layers:     12
width:      d_model 768
heads:      12
sequence:   256 tokens
batch:      16 per rank with grad_accum 2 on 4 A6000s
tokens:     32768 global tokens per optimizer step
```

WikiText uses `Salesforce/wikitext`, `wikitext-103-raw-v1`, with the GPT-2 tokenizer. Synthetic tasks use the same model and tokenizer path and generate 100M-token local corpora.

## Synthetic Tasks

```text
synthetic/arithmetic     arithmetic, comparison, short sequence, small multiplication
synthetic/code           Python-like snippets, loops, dict lookup, branch choice
synthetic/symbolic       rewrite rules, parity, bracket depth, reverse-copy
synthetic/reasoning_mix  round-robin mix of arithmetic, code, and symbolic
```

The current fair rerun uses `synthetic/code`, `synthetic/symbolic`, and `synthetic/reasoning_mix`. It intentionally reruns all rows from scratch instead of mixing partial outputs.

## Step-1 Curves

Validation now logs at step 1, then at `--eval-interval`, then at the final step. Training loss already logged at step 1. Future PPL/loss plots should therefore start at step 1 when produced from fresh runs.

## GPU Rule

Use A6000 GPUs only. A normal sweep requests 4 GPUs, and total concurrent allocation must stay at or below 8 A6000s.

```text
--gres=gpu:nvidia_rtx_a6000:4
--nproc_per_node=4
RATIONAL_OPT_TORCH_FALLBACK=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_P2P_DISABLE=1
```

The PyTorch fallback keeps the same RLB math when the checked-in CUDA extension does not have a usable A6000 image.

## Fair Synthetic Rerun

```bash
sbatch experiments/scripts/run_synthetic_fair_full_20260529.sh
```

The script launches one 24h, 4x A6000 job and runs:

| task | compared rows |
| --- | --- |
| `synthetic/code` | SiLU/SwiGLU+AdamW, RLB+AdamW, SiLU/SwiGLU+Muon, RLB+Muon, RLB MatrixPolicy, RLB MatrixPolicy group-stat |
| `synthetic/symbolic` | same rows |
| `synthetic/reasoning_mix` | same rows |

After completion:

```bash
.venv-cu128/bin/python experiments/scripts/summarize_synthetic_fair_full_20260529.py
```

## Current Optimizer

```text
optimizer                                           rational_matrix_policy_onpolicy
activation                                          rlb_fused_fixed_strong_ffn
base_lr                                             3e-4
min_lr                                              3e-5
warmup_steps                                        200
global_schedule                                     shared warmup/cosine for every compared row
backbone_optimizer                                  AdamW
backbone_beta2                                      0.999
matrix_policy_beta2                                 0.999
matrix_policy_adam_lr_scale                         3.00
matrix_policy_adam_lr_scale_final                   null
matrix_policy_adam_role_strength                    1.20
matrix_policy_adam_min_lr_scale                     0.40
matrix_policy_adam_max_lr_scale                     4.00
matrix_policy_input_depth_gain                     -0.50
matrix_policy_output_depth_gain                     1.00
matrix_policy_muon_strength                         0.75
matrix_policy_muon_lr_scale                         1.00
matrix_policy_start                                 0.02
matrix_policy_end                                   0.12
matrix_policy_decay_start                           0.20
matrix_policy_decay_end                             0.36
matrix_policy_final_muon                            0.00
matrix_policy_muon_reset_adam_state                 false
matrix_policy_function_coeff                        false
matrix_policy_group_gain_strength                   0.00
matrix_policy_group_pressure_strength               0.00
matrix_policy_group_activity_damping                0.00
role_specific_beta2_finals                          null by default
transport_strength                                  0.00
adaptive_coeff_strength                             0.00
rlb_gauge_strength                                  0.50
rlb_gauge_start                                     0.03
rlb_gauge_end                                       0.35
rlb_gauge_every                                     5
```

The global LR schedule is identical for the controls and MatrixPolicy rows. The optimizer-specific part is the update policy for RLB matrices.
