# Phase 1 Protocol-Lock Summary

Curve evidence is primary. Use the dense eval/train curve CSVs and plots before interpreting final validation loss. Lower AUC and lower validation loss are better.

Generated curve artifacts:

```text
eval_curves.csv
train_curves.csv
dclm_validation_loss_curves.png
dclm_validation_ppl_curves.png
dclm_training_loss_curves.png
dclm_validation_loss_curves_zoom_step250.png
fineweb_edu_validation_loss_curves.png
fineweb_edu_validation_ppl_curves.png
fineweb_edu_training_loss_curves.png
```

## DCLM

| rank | optimizer | activation | lr | wd | n | running | div | dense | eval pts | final loss | best loss | auc full | auc 500 | sec/step | key knobs |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | adamw | rlb_fused_fixed_strong_ffn | 0.0005 | 0.2 | 1 | 0 | 0 | 1 | 32 | 4.451585 | 4.451585 | 5.152803 | 6.135179 | 1.5446 |  |
| 2 | adamw | rlb_fused_fixed_strong_ffn | 0.0005 | 0.1 | 1 | 0 | 0 | 1 | 32 | 4.456295 | 4.456295 | 5.153664 | 6.133756 | 3.2117 |  |
| 3 | adamw | rlb_fused_fixed_strong_ffn | 0.0005 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.461227 | 4.461227 | 5.155411 | 6.133693 | 3.3006 |  |
| 4 | adamw | silu | 0.0005 | 0.1 | 1 | 0 | 0 | 1 | 32 | 4.472005 | 4.472005 | 5.181259 | 6.169214 | 3.2965 |  |
| 5 | adamw | silu | 0.0005 | 0.2 | 1 | 0 | 0 | 1 | 32 | 4.469075 | 4.469075 | 5.182120 | 6.172786 | 1.3975 |  |
| 6 | adamw | silu | 0.0005 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.476556 | 4.476556 | 5.184424 | 6.171181 | 3.2079 |  |
| 7 | adamw | rlb_fused_fixed_strong_ffn | 0.0003 | 0.2 | 1 | 0 | 0 | 1 | 32 | 4.639377 | 4.639377 | 5.311473 | 6.279032 | 2.8829 |  |
| 8 | adamw | rlb_fused_fixed_strong_ffn | 0.0003 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.644189 | 4.644189 | 5.312965 | 6.279328 | 0.8892 |  |
| 9 | adamw | rlb_fused_fixed_strong_ffn | 0.0003 | 0.1 | 1 | 0 | 0 | 1 | 32 | 4.644200 | 4.644200 | 5.314085 | 6.280586 | 3.2956 |  |
| 10 | adamw | silu | 0.0003 | 0.1 | 1 | 0 | 0 | 1 | 32 | 4.640119 | 4.640119 | 5.326651 | 6.314027 | 2.9929 |  |
| 11 | adamw | silu | 0.0003 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.641523 | 4.641523 | 5.326962 | 6.313283 | 0.7324 |  |
| 12 | adamw | silu | 0.0003 | 0.2 | 1 | 0 | 0 | 1 | 32 | 4.639202 | 4.639202 | 5.327194 | 6.314882 | 3.0927 |  |
| 13 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.1 | 1 | 0 | 0 | 1 | 32 | 4.640576 | 4.640576 | 5.350475 | 6.337530 | 4.4830 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=3.0, rational_matrix_policy_group_gain_strength=0.2, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 14 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.1 | 1 | 0 | 0 | 1 | 32 | 4.640565 | 4.640565 | 5.350665 | 6.337855 | 3.4233 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=3.0, rational_matrix_policy_group_gain_strength=0.35, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 15 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.642742 | 4.642742 | 5.350808 | 6.337743 | 3.0494 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=3.0, rational_matrix_policy_group_gain_strength=0.2, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 16 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.641755 | 4.641755 | 5.350868 | 6.337591 | 3.0960 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=3.0, rational_matrix_policy_group_gain_strength=0.35, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 17 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.638540 | 4.638540 | 5.351815 | 6.339069 | 4.9507 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=4.0, rational_matrix_policy_group_gain_strength=0.35, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 18 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.638531 | 4.638531 | 5.351850 | 6.339145 | 4.8683 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=4.0, rational_matrix_policy_group_gain_strength=0.2, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 19 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.1 | 1 | 0 | 0 | 1 | 32 | 4.644586 | 4.644586 | 5.353611 | 6.340465 | 4.3149 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=2.0, rational_matrix_policy_group_gain_strength=0.35, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 20 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.1 | 1 | 0 | 0 | 1 | 32 | 4.644353 | 4.644353 | 5.353718 | 6.340625 | 4.9638 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=2.0, rational_matrix_policy_group_gain_strength=0.2, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 21 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.647695 | 4.647695 | 5.354155 | 6.340449 | 2.9263 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=2.0, rational_matrix_policy_group_gain_strength=0.35, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 22 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.03 | 1 | 0 | 0 | 1 | 32 | 4.647768 | 4.647768 | 5.354206 | 6.340510 | 3.2424 | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=2.0, rational_matrix_policy_group_gain_strength=0.2, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |
| 23 | adamw | rlb_fused_fixed_strong_ffn | 0.0001 | 0.1 | 1 | 0 | 0 | 1 | 32 | 5.077939 | 5.077939 | 5.753618 | 6.741528 | 0.8859 |  |
| 24 | adamw | rlb_fused_fixed_strong_ffn | 0.0001 | 0.03 | 1 | 0 | 0 | 1 | 32 | 5.078360 | 5.078360 | 5.754447 | 6.743295 | 0.8919 |  |
| 25 | adamw | rlb_fused_fixed_strong_ffn | 0.0001 | 0.2 | 1 | 0 | 0 | 1 | 32 | 5.078014 | 5.078014 | 5.754648 | 6.744362 | 0.8948 |  |
| 26 | adamw | silu | 0.0001 | 0.1 | 1 | 0 | 0 | 1 | 32 | 5.076860 | 5.076860 | 5.765555 | 6.784475 | 0.7316 |  |
| 27 | adamw | silu | 0.0001 | 0.03 | 1 | 0 | 0 | 1 | 32 | 5.077363 | 5.077363 | 5.766011 | 6.785434 | 0.7286 |  |
| 28 | adamw | silu | 0.0001 | 0.2 | 1 | 0 | 0 | 1 | 32 | 5.076833 | 5.076833 | 5.766079 | 6.785757 | 0.7333 |  |
| 29 | rational_matrix_policy_onpolicy | rlb_fused_fixed_strong_ffn | 0.0002 | 0.1 | 1 | 1 | 0 | 1 | 8 | 5.546152 | 5.546152 | 6.737960 | 6.737960 |  | muon_momentum=0.95, muon_ns_steps=5, rational_matrix_policy_adam_lr_scale=4.0, rational_matrix_policy_group_gain_strength=0.2, rational_matrix_policy_group_pressure_strength=0.1, rational_matrix_policy_group_activity_damping=0.2 |

## FineWeb-Edu

| rank | optimizer | activation | lr | wd | n | running | div | dense | eval pts | final loss | best loss | auc full | auc 500 | sec/step | key knobs |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | adamw | silu | 0.0001 | 0.03 | 1 | 1 | 0 | 1 | 1 | 11.026684 | 11.026684 |  |  |  |  |
