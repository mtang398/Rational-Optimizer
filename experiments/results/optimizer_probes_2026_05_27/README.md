# Optimizer Probes 2026-05-27

These are seed-1337 probes for new RLB-specific optimizer ideas. They were used to decide whether to launch a larger multi-seed sweep.

```text
optimizer                              activation                         final / checkpoint
rational_adaptive_metric_onpolicy       rlb_fused_fixed_strong_ffn         final 3.615887, PPL 37.184
rational_adaptive_metric_onpolicy       rlb_fused_fixed_strong_h2880_ffn   final 3.615114, PPL 37.156
rational_quotient_jacobian_onpolicy     rlb_fused_fixed_strong_ffn         final 3.615571, PPL 37.173
rational_quotient_jacobian_onpolicy     rlb_fused_fixed_strong_h2880_ffn   stopped after step 750 at 4.440911
rational_jacobian_onpolicy + coeff gram rlb_fused_fixed_strong_ffn         stopped after step 250 at 5.527315
```

Seed-1337 incumbents from the full result are `3.614862` for `rational_jacobian_onpolicy + rlb_fused_fixed_strong_ffn` and `3.614475` for `rational_quotient_onpolicy + rlb_fused_fixed_strong_h2880_ffn`. These probes therefore did not justify replacing the verified three-seed recommendation.

Relevant Slurm jobs: `825719`, `826667`, `828122`, `828678`.
