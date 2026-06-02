# ICLR Optimizer Validation Summary

runs checked: 13

All validation runs passed required telemetry checks.

## Runs

| optimizer | activation | steps | final loss | seconds/step | source |
| --- | --- | ---: | ---: | ---: | --- |
| adafactor_came | rlb_fused_fixed_strong_ffn | 6 | 10.480968 | 0.0497 | `experiments/runs/iclr_optimizer_validation_20260602/validation_adafactor_came_20260602_tiny_cuda_ddp/rlb_fused_fixed_strong_ffn.jsonl` |
| adafactor_came | silu | 6 | 10.607305 | 0.0298 | `experiments/runs/iclr_optimizer_validation_20260602/validation_adafactor_came_20260602_tiny_cuda_ddp/silu.jsonl` |
| adamw | rlb_fused_fixed_strong_ffn | 6 | 10.402638 | 0.0429 | `experiments/runs/iclr_optimizer_validation_20260602/validation_adamw_20260602_tiny_cuda_ddp/rlb_fused_fixed_strong_ffn.jsonl` |
| adamw | silu | 6 | 10.519886 | 0.0231 | `experiments/runs/iclr_optimizer_validation_20260602/validation_adamw_20260602_tiny_cuda_ddp/silu.jsonl` |
| ademamix | rlb_fused_fixed_strong_ffn | 6 | 9.500518 | 0.0414 | `experiments/runs/iclr_optimizer_validation_20260602/validation_ademamix_20260602_tiny_cuda_ddp/rlb_fused_fixed_strong_ffn.jsonl` |
| ademamix | silu | 6 | 9.534185 | 0.0248 | `experiments/runs/iclr_optimizer_validation_20260602/validation_ademamix_20260602_tiny_cuda_ddp/silu.jsonl` |
| lion | rlb_fused_fixed_strong_ffn | 6 | 10.350974 | 0.0365 | `experiments/runs/iclr_optimizer_validation_20260602/validation_lion_20260602_tiny_cuda_ddp/rlb_fused_fixed_strong_ffn.jsonl` |
| lion | silu | 6 | 10.472309 | 0.0217 | `experiments/runs/iclr_optimizer_validation_20260602/validation_lion_20260602_tiny_cuda_ddp/silu.jsonl` |
| rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 6 | 10.240106 | 0.0465 | `experiments/runs/iclr_optimizer_validation_20260602/validation_rational_matrix_policy_onpolicy_20260602_tiny_cuda_ddp/rlb_fused_fixed_strong_ffn.jsonl` |
| schedule_free_adamw | rlb_fused_fixed_strong_ffn | 6 | 10.530769 | 0.0405 | `experiments/runs/iclr_optimizer_validation_20260602/validation_schedule_free_adamw_20260602_tiny_cuda_ddp/rlb_fused_fixed_strong_ffn.jsonl` |
| schedule_free_adamw | silu | 6 | 10.632385 | 0.0211 | `experiments/runs/iclr_optimizer_validation_20260602/validation_schedule_free_adamw_20260602_tiny_cuda_ddp/silu.jsonl` |
| soap_adamw | rlb_fused_fixed_strong_ffn | 6 | 10.656059 | 0.0399 | `experiments/runs/iclr_optimizer_validation_20260602/validation_soap_adamw_20260602_tiny_cuda_ddp/rlb_fused_fixed_strong_ffn.jsonl` |
| soap_adamw | silu | 6 | 10.672012 | 0.0235 | `experiments/runs/iclr_optimizer_validation_20260602/validation_soap_adamw_20260602_tiny_cuda_ddp/silu.jsonl` |
