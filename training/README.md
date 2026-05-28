# Training

This folder contains the WikiText-103 benchmark entrypoints.

## Files

```text
transformer_wikitext103_compare.py      model, activations, optimizer wiring, train/eval loop
run_wikitext103_optimizer_sweep.sbatch  accepted 4-GPU Slurm sweep launcher
aggregate_wikitext103_multiseed.py      JSONL aggregation into CSV/JSON/README summaries
```

## Benchmark

```text
dataset:      Salesforce/wikitext, wikitext-103-raw-v1
task:         causal language modeling
tokenizer:    GPT-2 tokenizer
model:        LLaMA-style decoder-only Transformer
size:         about 123M parameters
depth:        12 layers
width:        d_model 768
heads:        12
sequence:     256 tokens
```

## Current Best Optimizer

The names below match the JSONL config keys where possible. `optimizer_lr` and `optimizer_min_lr` are the values passed by `--lr` and `--min-lr`.

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
rational_matrix_policy_adam_role_strength           1.20
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

Use `rational_matrix_policy_onpolicy` only with RLB activations. The training defaults now match the best run, so no extra MatrixPolicy args are needed.

## Optimizer Wiring

`transformer_wikitext103_compare.py` builds the optimizer as child optimizers wrapped by the RLB on-policy/gauge optimizer:

| parameter set | optimizer behavior |
| --- | --- |
| Non-RLB backbone weights | AdamW, beta2=0.999 in this branch |
| Norms, biases, tied embeddings | AdamW no-decay group |
| RLB `W_in/W_out` matrices | `RationalMatrixPolicyOptimizer` with MatrixPolicy AdamW plus early Muon |
| Rational coefficients | AdamW by default |

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
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
  RUN_NAME=rlb_matrix_policy_muon_switch \
  STEPS=3051 SEEDS=1337 \
  OPTIMIZERS=rational_matrix_policy_onpolicy \
  ACTIVATIONS=rlb_fused_fixed_strong_ffn \
  EVAL_INTERVAL=250 EVAL_BATCHES=20 LOG_INTERVAL=100 \
  sbatch --time=02:00:00 --gres=gpu:nvidia_rtx_6000_ada_generation:4 \
  training/run_wikitext103_optimizer_sweep.sbatch
```

## GPU Rule

Use at most 4 GPUs total. Check active jobs before launching:

```bash
squeue -u mt872
```

## Rule For Claims

Do not use high-LR ablations as the headline result. The optimizer must win under the same LR schedule against SiLU+AdamW, RLB+AdamW, and the beta2-tuned AdamW controls.
