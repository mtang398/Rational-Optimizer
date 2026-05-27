# WikiText-103 Benchmark Summary

Generated: 2026-05-27 05:25:10 UTC

Raw run directory:

```text
experiments/runs/wikitext103/rlb_optimizer_empirical_ngram_full
```

Slurm job:

```text
763059+813929+821187
```

Slurm log:

```text
experiments/runs/logs/ract-wt103-opt-821187.out
```

This result folder is an organized summary only. The raw JSONL logs remain in
the run directory above.

## Per-Seed Results

| Optimizer | Seed | Activation | Params | Val loss | Gap vs external | Gap vs classic | PPL | Sec/step | Time vs external | Time vs classic |
|---|---|---|---|---|---|---|---|---|---|---|
| adamw | 1337 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.617501 | -0.004480 | 0.000000 | 37.244 | 0.206287 | 1.090x | 1.000x |
| adamw | 1337 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.615913 | -0.006069 | 0.000000 | 37.185 | 0.202209 | 1.069x | 1.000x |
| adamw | 1337 | silu | 123,551,232 | 3.621982 | 0.000000 | 0.000000 | 37.412 | 0.189237 | 1.000x | 1.000x |
| adamw | 2024 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.615454 | 0.002502 | 0.000000 | 37.168 | 0.204879 | 1.083x | 1.000x |
| adamw | 2024 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.618035 | 0.005082 | 0.000000 | 37.264 | 0.200058 | 1.057x | 1.000x |
| adamw | 2024 | silu | 123,551,232 | 3.612952 | 0.000000 | 0.000000 | 37.075 | 0.189200 | 1.000x | 1.000x |
| adamw | 31415 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.586932 | -0.008521 | 0.000000 | 36.123 | 0.204638 | 1.085x | 1.000x |
| adamw | 31415 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.596571 | 0.001117 | 0.000000 | 36.473 | 0.201713 | 1.070x | 1.000x |
| adamw | 31415 | silu | 123,551,232 | 3.595454 | 0.000000 | 0.000000 | 36.432 | 0.188554 | 1.000x | 1.000x |
| muon | 1337 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.652891 | 0.030909 | 0.035390 | 38.586 | 0.223007 | 1.178x | 1.081x |
| muon | 1337 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.656654 | 0.034672 | 0.040741 | 38.732 | 0.218521 | 1.155x | 1.081x |
| muon | 1337 | silu | 123,551,232 | 3.639733 | 0.017751 | 0.017751 | 38.082 | 0.208511 | 1.102x | 1.102x |
| muon | 2024 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.652237 | 0.039284 | 0.036782 | 38.561 | 0.222866 | 1.178x | 1.088x |
| muon | 2024 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.650362 | 0.037410 | 0.032328 | 38.489 | 0.217013 | 1.147x | 1.085x |
| muon | 2024 | silu | 123,551,232 | 3.630373 | 0.017421 | 0.017421 | 37.727 | 0.207406 | 1.096x | 1.096x |
| muon | 31415 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.629169 | 0.033715 | 0.042236 | 37.681 | 0.222951 | 1.182x | 1.089x |
| muon | 31415 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.630429 | 0.034976 | 0.033858 | 37.729 | 0.216774 | 1.150x | 1.075x |
| muon | 31415 | silu | 123,551,232 | 3.609430 | 0.013977 | 0.013977 | 36.945 | 0.207870 | 1.102x | 1.102x |
| rational_empirical_onpolicy | 1337 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.652580 | 0.030599 | 0.035079 | 38.574 | 0.226503 | 1.197x | 1.098x |
| rational_empirical_onpolicy | 1337 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.647603 | 0.025621 | 0.031690 | 38.383 | 0.222046 | 1.173x | 1.098x |
| rational_empirical_onpolicy | 2024 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.651808 | 0.038856 | 0.036354 | 38.544 | 0.225981 | 1.194x | 1.103x |
| rational_empirical_onpolicy | 2024 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.650583 | 0.037631 | 0.032549 | 38.497 | 0.219786 | 1.162x | 1.099x |
| rational_empirical_onpolicy | 31415 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.618569 | 0.023115 | 0.031637 | 37.284 | 0.225922 | 1.198x | 1.104x |
| rational_empirical_onpolicy | 31415 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.631124 | 0.035670 | 0.034553 | 37.755 | 0.220343 | 1.169x | 1.092x |
| rational_jacobian_onpolicy | 1337 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.614862 | -0.007120 | -0.002639 | 37.146 | 0.204910 | 1.083x | 0.993x |
| rational_jacobian_onpolicy | 1337 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.614954 | -0.007028 | -0.000959 | 37.150 | 0.202224 | 1.069x | 1.000x |
| rational_jacobian_onpolicy | 2024 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.615949 | 0.002996 | 0.000494 | 37.187 | 0.204876 | 1.083x | 1.000x |
| rational_jacobian_onpolicy | 2024 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.616202 | 0.003249 | -0.001833 | 37.196 | 0.201467 | 1.065x | 1.007x |
| rational_jacobian_onpolicy | 31415 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.585370 | -0.010083 | -0.001562 | 36.067 | 0.204870 | 1.087x | 1.001x |
| rational_jacobian_onpolicy | 31415 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.596663 | 0.001210 | 0.000093 | 36.476 | 0.200950 | 1.066x | 0.996x |
| rational_onpolicy_balance | 1337 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.617283 | -0.004699 | -0.000218 | 37.236 | 0.209546 | 1.107x | 1.016x |
| rational_onpolicy_balance | 1337 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.614519 | -0.007463 | -0.001394 | 37.133 | 0.205929 | 1.088x | 1.018x |
| rational_onpolicy_balance | 2024 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.616247 | 0.003295 | 0.000793 | 37.198 | 0.208967 | 1.104x | 1.020x |
| rational_onpolicy_balance | 2024 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.617239 | 0.004286 | -0.000796 | 37.235 | 0.205703 | 1.087x | 1.028x |
| rational_onpolicy_balance | 31415 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.585148 | -0.010305 | -0.001784 | 36.059 | 0.208567 | 1.106x | 1.019x |
| rational_onpolicy_balance | 31415 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.598014 | 0.002560 | 0.001443 | 36.526 | 0.204776 | 1.086x | 1.015x |
| rational_quotient_onpolicy | 1337 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.617213 | -0.004768 | -0.000288 | 37.234 | 0.204777 | 1.082x | 0.993x |
| rational_quotient_onpolicy | 1337 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.614475 | -0.007506 | -0.001437 | 37.132 | 0.201641 | 1.066x | 0.997x |
| rational_quotient_onpolicy | 2024 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.615999 | 0.003047 | 0.000545 | 37.188 | 0.205578 | 1.087x | 1.003x |
| rational_quotient_onpolicy | 2024 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.616855 | 0.003903 | -0.001179 | 37.220 | 0.200604 | 1.060x | 1.003x |
| rational_quotient_onpolicy | 31415 | rlb_fused_fixed_strong_ffn | 123,553,824 | 3.586781 | -0.008673 | -0.000151 | 36.118 | 0.205172 | 1.088x | 1.003x |
| rational_quotient_onpolicy | 31415 | rlb_fused_fixed_strong_h2880_ffn | 120,014,880 | 3.596863 | 0.001409 | 0.000292 | 36.484 | 0.200987 | 1.066x | 0.996x |

## Aggregate Results

| Optimizer | Activation | Seeds | Params | Mean loss | Std loss | Mean gap vs external | Std gap vs external | Mean gap vs classic | Std gap vs classic | Mean PPL | Mean sec/step | Mean time vs external | Mean time vs classic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adamw | rlb_fused_fixed_strong_ffn | 3 | 123,553,824 | 3.606629 | 0.017089 | -0.003500 | 0.005577 | 0.000000 | 0.000000 | 36.845 | 0.205268 | 1.086x | 1.000x |
| adamw | rlb_fused_fixed_strong_h2880_ffn | 3 | 120,014,880 | 3.610173 | 0.011827 | 0.000044 | 0.005653 | 0.000000 | 0.000000 | 36.974 | 0.201326 | 1.065x | 1.000x |
| adamw | silu | 3 | 123,551,232 | 3.610129 | 0.013488 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 36.973 | 0.188997 | 1.000x | 1.000x |
| muon | rlb_fused_fixed_strong_ffn | 3 | 123,553,824 | 3.644765 | 0.013511 | 0.034636 | 0.004263 | 0.038136 | 0.003619 | 38.276 | 0.222941 | 1.180x | 1.086x |
| muon | rlb_fused_fixed_strong_h2880_ffn | 3 | 120,014,880 | 3.645815 | 0.013691 | 0.035686 | 0.001501 | 0.035642 | 0.004481 | 38.316 | 0.217436 | 1.150x | 1.080x |
| muon | silu | 3 | 123,551,232 | 3.626512 | 0.015516 | 0.016383 | 0.002090 | 0.016383 | 0.002090 | 37.585 | 0.207929 | 1.100x | 1.100x |
| rational_empirical_onpolicy | rlb_fused_fixed_strong_ffn | 3 | 123,553,824 | 3.640986 | 0.019417 | 0.030857 | 0.007873 | 0.034357 | 0.002440 | 38.134 | 0.226135 | 1.197x | 1.102x |
| rational_empirical_onpolicy | rlb_fused_fixed_strong_h2880_ffn | 3 | 120,014,880 | 3.643103 | 0.010481 | 0.032974 | 0.006443 | 0.032931 | 0.001469 | 38.212 | 0.220725 | 1.168x | 1.096x |
| rational_jacobian_onpolicy | rlb_fused_fixed_strong_ffn | 3 | 123,553,824 | 3.605394 | 0.017349 | -0.004736 | 0.006858 | -0.001236 | 0.001592 | 36.800 | 0.204885 | 1.084x | 0.998x |
| rational_jacobian_onpolicy | rlb_fused_fixed_strong_h2880_ffn | 3 | 120,014,880 | 3.609273 | 0.010938 | -0.000856 | 0.005441 | -0.000900 | 0.000964 | 36.941 | 0.201547 | 1.066x | 1.001x |
| rational_onpolicy_balance | rlb_fused_fixed_strong_ffn | 3 | 123,553,824 | 3.606226 | 0.018262 | -0.003903 | 0.006835 | -0.000403 | 0.001298 | 36.831 | 0.209027 | 1.106x | 1.018x |
| rational_onpolicy_balance | rlb_fused_fixed_strong_h2880_ffn | 3 | 120,014,880 | 3.609924 | 0.010404 | -0.000205 | 0.006344 | -0.000249 | 0.001495 | 36.965 | 0.205470 | 1.087x | 1.021x |
| rational_quotient_onpolicy | rlb_fused_fixed_strong_ffn | 3 | 123,553,824 | 3.606664 | 0.017230 | -0.003465 | 0.005967 | 0.000035 | 0.000447 | 36.847 | 0.205176 | 1.086x | 1.000x |
| rational_quotient_onpolicy | rlb_fused_fixed_strong_h2880_ffn | 3 | 120,014,880 | 3.609398 | 0.010921 | -0.000732 | 0.005998 | -0.000775 | 0.000933 | 36.945 | 0.201077 | 1.064x | 0.999x |
