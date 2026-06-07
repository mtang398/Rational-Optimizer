# ICLR Run Status

Updated: 2026-06-07 17:10:24 EDT  
Manifest: `experiments/manifests/iclr26_main_manifest.csv`

## Scheduler State

| Job | Phase rows | State | Exit | GPUs | Elapsed | Node |
| --- | --- | --- | --- | --- | --- | --- |
| `151609` | E0 rows 0-8 | completed | `0:0` | 4 A6000 | 00:38:17 | `ma-compute-02` |
| `151610` | E0 rows 9-14 | completed | `0:0` | 4 A6000 | 00:17:14 | `bala-compute-02` |

E0 is complete. E1 has started from whole matched 15-row cells.

## E0 Preflight

E0 is a 15-row smoke pass: 5 corpora times 3 matched methods. Each dataset cell uses the same outer training config across `silu_adamw`, `rlb_adamw`, and `rlb_matrixpolicy_original`: `lr=0.0003`, `min_lr=0.00003`, `weight_decay=0.10`, seed `1337`, `steps=80`, `eval_interval=40`, 4 A6000.

All rows have three eval points: step 1, step 40, and step 80.

## E0 Final Loss Summary

| Dataset | SiLU AdamW | RLB AdamW | MatrixPolicy original | MP gap vs SiLU | MP gap vs RLB AdamW |
| --- | ---: | ---: | ---: | ---: | ---: |
| dclm | 7.346231 | 7.238834 | 7.121294 | 0.224937 | 0.117540 |
| fineweb_edu | 7.426810 | 7.211745 | 7.035759 | 0.391051 | 0.175986 |
| fineweb | 7.527949 | 7.257327 | 7.101671 | 0.426279 | 0.155656 |
| dolma_sample | 8.201650 | 8.212225 | 8.189895 | 0.011755 | 0.022330 |
| c4_en | 7.579340 | 7.278966 | 7.207716 | 0.371624 | 0.071249 |

## E0 Row Table With Runtime, Params, And Eval Curves

`Run time` is `summary.total_seconds` from the JSONL record for that row. The Slurm job wall times above include dataset preparation and sequential row execution.

| Row | Dataset | Method | Params | Run time s | Mean s/step | Final val loss | Tokens/s | Eval curve `step:val_loss` |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | dclm | silu_adamw | 123,551,232 | 53.20 | 0.6213 | 7.346231 | 52741.71 | 1:10.798578, 40:7.553685, 80:7.346231 |
| 1 | dclm | rlb_adamw | 123,553,824 | 60.56 | 0.6929 | 7.238834 | 47290.45 | 1:10.533360, 40:7.471305, 80:7.238834 |
| 2 | dclm | rlb_matrixpolicy_original | 123,553,824 | 60.86 | 0.6975 | 7.121294 | 46977.61 | 1:10.301031, 40:7.391489, 80:7.121294 |
| 3 | fineweb_edu | silu_adamw | 123,551,232 | 53.14 | 0.6177 | 7.426810 | 53050.87 | 1:10.775492, 40:7.637898, 80:7.426810 |
| 4 | fineweb_edu | rlb_adamw | 123,553,824 | 60.63 | 0.6932 | 7.211745 | 47273.08 | 1:10.528216, 40:7.452968, 80:7.211745 |
| 5 | fineweb_edu | rlb_matrixpolicy_original | 123,553,824 | 61.41 | 0.7033 | 7.035759 | 46593.49 | 1:10.283565, 40:7.335752, 80:7.035759 |
| 6 | fineweb | silu_adamw | 123,551,232 | 52.99 | 0.6195 | 7.527949 | 52895.84 | 1:10.824485, 40:7.717688, 80:7.527949 |
| 7 | fineweb | rlb_adamw | 123,553,824 | 60.51 | 0.6893 | 7.257327 | 47540.86 | 1:10.569622, 40:7.519713, 80:7.257327 |
| 8 | fineweb | rlb_matrixpolicy_original | 123,553,824 | 61.12 | 0.7009 | 7.101671 | 46753.24 | 1:10.334897, 40:7.368423, 80:7.101671 |
| 9 | dolma_sample | silu_adamw | 123,551,232 | 53.77 | 0.6234 | 8.201650 | 52563.79 | 1:10.863197, 40:8.279759, 80:8.201650 |
| 10 | dolma_sample | rlb_adamw | 123,553,824 | 61.00 | 0.6954 | 8.212225 | 47123.59 | 1:10.679456, 40:8.226797, 80:8.212225 |
| 11 | dolma_sample | rlb_matrixpolicy_original | 123,553,824 | 61.00 | 0.7008 | 8.189895 | 46758.65 | 1:10.488925, 40:8.187285, 80:8.189895 |
| 12 | c4_en | silu_adamw | 123,551,232 | 53.37 | 0.6232 | 7.579340 | 52581.93 | 1:10.825615, 40:7.731586, 80:7.579340 |
| 13 | c4_en | rlb_adamw | 123,553,824 | 60.63 | 0.6931 | 7.278966 | 47277.57 | 1:10.567333, 40:7.533151, 80:7.278966 |
| 14 | c4_en | rlb_matrixpolicy_original | 123,553,824 | 60.86 | 0.6994 | 7.207716 | 46848.39 | 1:10.340908, 40:7.446258, 80:7.207716 |

## E1 Scheduler State

E1 uses whole matched 15-row cells. Each job uses 4 A6000. The queue is dependency-chained in pairs, except the final single C4 cell.

| Job | Rows | Cell | State at update | GPUs | Elapsed | Node |
| --- | --- | --- | --- | --- | --- | --- |
| `155411` | 15-29 | dclm seed 1337 | completed | 4 A6000 | 09:23:24 | `ma-compute-02` |
| `155412` | 30-44 | dclm seed 2027 | completed | 4 A6000 | 09:15:36 | `bala-compute-02` |
| `158114` | 45-59 | dclm seed 3407 | completed | 4 A6000 | 09:28:58 | `ma-compute-02` |
| `158115` | 60-74 | fineweb_edu seed 1337 | completed | 4 A6000 | 09:11:18 | `bala-compute-02` |
| `158117` | 75-89 | fineweb_edu seed 2027 | completed, `Restarts=6` | 4 A6000 | 16:04:13 | `monakhova-compute-01` |
| `158118` | 90-104 | fineweb_edu seed 3407 | completed | 4 A6000 | 09:02:26 | `bala-compute-02` |
| `158155` | 105-119 | fineweb seed 1337 | completed | 4 A6000 | 05:52:37 | `elor-compute-01` |
| `158156` | 120-134 | fineweb seed 2027 | completed | 4 A6000 | 08:02:33 | `lil-compute-04` |
| `158163` | 135-149 | fineweb seed 3407 | completed | 4 A6000 | 07:53:18 | `ellis-compute-02` |
| `158164` | 150-164 | dolma_sample seed 1337 | completed | 4 A6000 | 08:12:23 | `ellis-compute-02` |
| `158166` | 165-179 | dolma_sample seed 2027 | completed | 4 A6000 | 07:44:29 | `ellis-compute-02` |
| `158165` | 180-194 | dolma_sample seed 3407 | completed | 4 A6000 | 06:15:00 | `damle-compute-01` |
| `158168` | 195-209 | c4_en seed 1337 | completed | 4 A6000 | 07:50:14 | `ellis-compute-02` |
| `158167` | 210-224 | c4_en seed 2027 | completed | 4 A6000 | 06:16:59 | `damle-compute-01` |
| `158169` | 225-239 | c4_en seed 3407 | running | 4 A6000 | 00:24:02 | `damle-compute-01` |

Active allocation at update: 4 A6000 total.

## E1 Continuation Queue

E1 cells are queued in whole 15-row matched blocks. Each job uses 4 A6000. Dependencies advance in pairs until the final single C4 seed.

| Wave | Dependency | Job | Rows | Cell | State at update |
| ---: | --- | --- | --- | --- | --- |
| 0 | none | `155411` | 15-29 | dclm seed 1337 | completed |
| 0 | none | `155412` | 30-44 | dclm seed 2027 | completed |
| 1 | afterok:`155411`:`155412` | `158114` | 45-59 | dclm seed 3407 | completed |
| 1 | afterok:`155411`:`155412` | `158115` | 60-74 | fineweb_edu seed 1337 | completed |
| 2 | afterok:`158114`:`158115` | `158117` | 75-89 | fineweb_edu seed 2027 | completed, `Restarts=6` |
| 2 | afterok:`158114`:`158115` | `158118` | 90-104 | fineweb_edu seed 3407 | completed |
| 3 | afterok:`158117`:`158118` | `158155` | 105-119 | fineweb seed 1337 | completed |
| 3 | afterok:`158117`:`158118` | `158156` | 120-134 | fineweb seed 2027 | completed |
| 4 | afterok:`158155`:`158156` | `158163` | 135-149 | fineweb seed 3407 | completed |
| 4 | afterok:`158155`:`158156` | `158164` | 150-164 | dolma_sample seed 1337 | completed |
| 5 | afterok:`158163`:`158164` | `158166` | 165-179 | dolma_sample seed 2027 | completed |
| 5 | afterok:`158163`:`158164` | `158165` | 180-194 | dolma_sample seed 3407 | completed |
| 6 | afterok:`158166`:`158165` | `158168` | 195-209 | c4_en seed 1337 | completed |
| 6 | afterok:`158166`:`158165` | `158167` | 210-224 | c4_en seed 2027 | completed |
| 7 | afterok:`158168`:`158167` | `158169` | 225-239 | c4_en seed 3407 | running |

## E1 Live Timing

Current check: 2026-06-07 17:10:24 EDT. Running job(s): `158169`.

| Job | Current row | Dataset/seed | Method | Elapsed | Latest progress | Latest val loss | State |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| `158169` | 226 | c4_en / 3407 | rlb_adamw | 00:24:02 | train 270, eval 250 / 3050 | 6.147571 | running |

Completed E1 rows: 211. Running rows: 1. Remaining rows including the running row: 14. Active allocation at check: 4 A6000 total.

At this check, C4 seeds 1337 and 2027 are complete, and C4 seed 3407 is running row 226 of 239 after completing row 225. If the current node speed holds and there is no preemption, E1 should finish in roughly 5-7 hours after this update.

## E1 Dense Curve Figures

The SVG figures below use completed E1 runs at their native logging cadence. Validation curves use every 50-step eval from step 500 through 3050; training-loss curves use every 10-step train log over the same range. The curves remain densely sampled; only the x-axis tick labels are sparse. Each plot has its legend inside the figure, and the shaded region is mean +/- 1 sample std over the three seeds. Two versions are shown: the all-method view keeps MatrixPolicy, RLB/SiLU AdamW, RLB/SiLU Lion, RLB/SiLU SOAP, RLB/SiLU Muon, RLB/SiLU ScheduleFree, and RLB/SiLU CAME; the clean comparison view uses the same style but omits SOAP from the figures. ADeMaMix remains in the final-score tables; the figures focus on the interpretable loss range.

### DCLM

All-method view:

![DCLM E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/dclm_core_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/dclm_core_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/dclm_core_training_loss_mean_std.svg)

Clean comparison view:

![DCLM E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/dclm_clean_validation_loss_mean_std.svg)

![DCLM E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/dclm_clean_validation_ppl_mean_std.svg)

![DCLM E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/dclm_clean_training_loss_mean_std.svg)

DCLM validation-loss checkpoint table, mean +/- sample std:

| Method | 500 | 1000 | 1500 | 2000 | 2500 | 3050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 5.2560 +/- 0.0048 | 4.8276 +/- 0.0062 | 4.5517 +/- 0.0016 | 4.3970 +/- 0.0043 | 4.3061 +/- 0.0039 | 4.2562 +/- 0.0050 |
| RLB+AdamW | 5.3672 +/- 0.0076 | 4.9292 +/- 0.0045 | 4.6748 +/- 0.0011 | 4.5270 +/- 0.0048 | 4.4464 +/- 0.0037 | 4.4047 +/- 0.0046 |
| SiLU+AdamW | 5.3838 +/- 0.0115 | 4.9398 +/- 0.0091 | 4.6788 +/- 0.0083 | 4.5306 +/- 0.0101 | 4.4489 +/- 0.0086 | 4.4056 +/- 0.0099 |
| RLB+Lion | 5.4021 +/- 0.0175 | 4.8664 +/- 0.0063 | 4.5860 +/- 0.0085 | 4.4348 +/- 0.0049 | 4.3511 +/- 0.0078 | 4.3057 +/- 0.0058 |
| SiLU+Lion | 5.4593 +/- 0.0049 | 4.8989 +/- 0.0070 | 4.6081 +/- 0.0038 | 4.4516 +/- 0.0063 | 4.3649 +/- 0.0049 | 4.3183 +/- 0.0069 |
| RLB+SOAP | 5.4086 +/- 0.0216 | 4.9523 +/- 0.0162 | 4.7242 +/- 0.0457 | 4.5516 +/- 0.0158 | 4.4597 +/- 0.0155 | 4.4351 +/- 0.0217 |
| SiLU+SOAP | 5.7233 +/- 0.1468 | 5.0899 +/- 0.0598 | 4.8541 +/- 0.1178 | 4.5941 +/- 0.0321 | 4.4658 +/- 0.0017 | 4.4160 +/- 0.0038 |
| RLB+Muon | 5.7691 +/- 0.0133 | 5.0865 +/- 0.0118 | 4.7917 +/- 0.0057 | 4.6209 +/- 0.0014 | 4.5246 +/- 0.0042 | 4.4742 +/- 0.0041 |
| SiLU+Muon | 5.8027 +/- 0.0123 | 5.0927 +/- 0.0102 | 4.7831 +/- 0.0090 | 4.6036 +/- 0.0127 | 4.5065 +/- 0.0120 | 4.4572 +/- 0.0126 |
| RLB+ScheduleFree | 5.8580 +/- 0.0045 | 5.3702 +/- 0.0099 | 5.1395 +/- 0.0085 | 5.0064 +/- 0.0073 | 4.9281 +/- 0.0061 | 4.8781 +/- 0.0055 |
| SiLU+ScheduleFree | 5.8883 +/- 0.0112 | 5.3887 +/- 0.0131 | 5.1590 +/- 0.0136 | 5.0276 +/- 0.0134 | 4.9514 +/- 0.0116 | 4.9023 +/- 0.0111 |
| RLB+CAME | 5.8999 +/- 0.0064 | 5.4569 +/- 0.0106 | 5.2388 +/- 0.0114 | 5.1140 +/- 0.0101 | 5.0445 +/- 0.0084 | 5.0074 +/- 0.0082 |
| SiLU+CAME | 5.9441 +/- 0.0091 | 5.4677 +/- 0.0067 | 5.2453 +/- 0.0124 | 5.1177 +/- 0.0147 | 5.0477 +/- 0.0139 | 5.0107 +/- 0.0143 |
| RLB+ADeMaMix | 16.1630 +/- 5.4229 | 370.9679 +/- 490.7603 | 43330230.6667 +/- 59088752.2823 | 175540490.6667 +/- 74840716.6504 | 3782729864.0000 +/- 5062818831.9630 (n=2) | 246105152.0000 +/- 0.0000 (n=1) |
| SiLU+ADeMaMix | 6.9823 +/- 0.6147 | 247.6634 +/- 51.2342 | 78.1737 +/- 22.2689 | 62.7083 +/- 16.6845 | 53.5849 +/- 14.3896 | 48.6455 +/- 13.7255 |

### FineWeb-Edu

All-method view:

![FineWeb-Edu E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_edu_core_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/fineweb_edu_core_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_edu_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb-Edu E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_edu_clean_validation_loss_mean_std.svg)

![FineWeb-Edu E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_edu_clean_validation_ppl_mean_std.svg)

![FineWeb-Edu E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_edu_clean_training_loss_mean_std.svg)

FineWeb-Edu validation-loss checkpoint table, mean +/- sample std:

| Method | 500 | 1000 | 1500 | 2000 | 2500 | 3050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 5.2423 +/- 0.0158 | 4.7105 +/- 0.0118 | 4.3976 +/- 0.0107 | 4.2359 +/- 0.0109 | 4.1387 +/- 0.0090 | 4.0882 +/- 0.0094 |
| RLB+AdamW | 5.3781 +/- 0.0151 | 4.8164 +/- 0.0076 | 4.5191 +/- 0.0025 | 4.3651 +/- 0.0057 | 4.2795 +/- 0.0055 | 4.2380 +/- 0.0061 |
| SiLU+AdamW | 5.4084 +/- 0.0172 | 4.8348 +/- 0.0049 | 4.5281 +/- 0.0072 | 4.3683 +/- 0.0072 | 4.2811 +/- 0.0074 | 4.2375 +/- 0.0086 |
| RLB+Lion | 5.4528 +/- 0.0239 | 4.7506 +/- 0.0088 | 4.4331 +/- 0.0041 | 4.2754 +/- 0.0061 | 4.1875 +/- 0.0057 | 4.1427 +/- 0.0068 |
| SiLU+Lion | 5.5121 +/- 0.0186 | 4.7814 +/- 0.0167 | 4.4520 +/- 0.0124 | 4.2870 +/- 0.0101 | 4.1964 +/- 0.0086 | 4.1494 +/- 0.0092 |
| RLB+SOAP | 5.9371 +/- 0.8724 | 4.9025 +/- 0.0578 | 4.5635 +/- 0.0082 | 4.4069 +/- 0.0287 | 4.3073 +/- 0.0147 | 4.2623 +/- 0.0131 |
| SiLU+SOAP | 5.5541 +/- 0.0178 | 5.0548 +/- 0.1513 | 4.6308 +/- 0.0214 | 4.4105 +/- 0.0136 | 4.3051 +/- 0.0112 | 4.2630 +/- 0.0220 |
| RLB+Muon | 5.8189 +/- 0.0126 | 4.9679 +/- 0.0124 | 4.6289 +/- 0.0139 | 4.4407 +/- 0.0180 | 4.3387 +/- 0.0192 | 4.2877 +/- 0.0199 |
| SiLU+Muon | 5.8660 +/- 0.0098 | 4.9736 +/- 0.0094 | 4.6220 +/- 0.0117 | 4.4295 +/- 0.0202 | 4.3292 +/- 0.0230 | 4.2787 +/- 0.0243 |
| RLB+ScheduleFree | 5.9870 +/- 0.0127 | 5.4052 +/- 0.0161 | 5.1139 +/- 0.0151 | 4.9433 +/- 0.0138 | 4.8431 +/- 0.0123 | 4.7797 +/- 0.0113 |
| SiLU+ScheduleFree | 6.0279 +/- 0.0274 | 5.4487 +/- 0.0075 | 5.1579 +/- 0.0075 | 4.9894 +/- 0.0073 | 4.8896 +/- 0.0073 | 4.8258 +/- 0.0072 |
| RLB+CAME | 6.0101 +/- 0.0223 | 5.4786 +/- 0.0122 | 5.1973 +/- 0.0082 | 5.0381 +/- 0.0064 | 4.9489 +/- 0.0052 | 4.9043 +/- 0.0046 |
| SiLU+CAME | 6.0899 +/- 0.0329 | 5.5003 +/- 0.0147 | 5.2179 +/- 0.0104 | 5.0573 +/- 0.0103 | 4.9669 +/- 0.0110 | 4.9207 +/- 0.0123 |
| RLB+ADeMaMix | 16.3601 +/- 9.0794 | 772.2738 +/- 845.8176 | 2575.6854 +/- 3736.6125 | 2426.1191 +/- 2936.8239 | 5742.2397 +/- 8431.4467 | 7880.5115 +/- 12045.6569 |
| SiLU+ADeMaMix | 6.5917 +/- 0.0381 | 323.2475 +/- 232.2217 | 132.7501 +/- 91.4211 | 293.8233 +/- 355.2971 | 262.2238 +/- 285.4760 | 242.8537 +/- 242.0122 |

### FineWeb

All-method view:

![FineWeb E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_core_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/fineweb_core_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/fineweb_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_clean_validation_loss_mean_std.svg)

![FineWeb E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_clean_validation_ppl_mean_std.svg)

![FineWeb E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/fineweb_clean_training_loss_mean_std.svg)

FineWeb validation-loss checkpoint table, mean +/- sample std:

| Method | 500 | 1000 | 1500 | 2000 | 2500 | 3050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 5.4031 +/- 0.0146 | 4.9239 +/- 0.0140 | 4.6275 +/- 0.0145 | 4.4674 +/- 0.0114 | 4.3718 +/- 0.0105 | 4.3186 +/- 0.0109 |
| RLB+AdamW | 5.5178 +/- 0.0196 | 5.0226 +/- 0.0213 | 4.7466 +/- 0.0175 | 4.5966 +/- 0.0132 | 4.5147 +/- 0.0129 | 4.4705 +/- 0.0133 |
| SiLU+AdamW | 5.5394 +/- 0.0181 | 5.0366 +/- 0.0139 | 4.7579 +/- 0.0118 | 4.6047 +/- 0.0092 | 4.5202 +/- 0.0087 | 4.4758 +/- 0.0097 |
| RLB+Lion | 5.5608 +/- 0.0036 | 4.9461 +/- 0.0228 | 4.6521 +/- 0.0108 | 4.4989 +/- 0.0078 | 4.4130 +/- 0.0078 | 4.3671 +/- 0.0075 |
| SiLU+Lion | 5.6186 +/- 0.0104 | 4.9885 +/- 0.0180 | 4.6808 +/- 0.0122 | 4.5218 +/- 0.0101 | 4.4307 +/- 0.0075 | 4.3825 +/- 0.0083 |
| RLB+SOAP | 5.5532 +/- 0.0487 | 5.0552 +/- 0.0394 | 4.8945 +/- 0.2541 | 4.6072 +/- 0.0207 | 4.5216 +/- 0.0175 | 4.4850 +/- 0.0255 |
| SiLU+SOAP | 5.7363 +/- 0.0492 | 5.2006 +/- 0.1139 | 4.8417 +/- 0.0132 | 4.6430 +/- 0.0128 | 4.5362 +/- 0.0096 | 4.4840 +/- 0.0112 |
| RLB+Muon | 5.9544 +/- 0.0286 | 5.1847 +/- 0.0223 | 4.8656 +/- 0.0210 | 4.6768 +/- 0.0130 | 4.5746 +/- 0.0120 | 4.5216 +/- 0.0120 |
| SiLU+Muon | 5.9974 +/- 0.0298 | 5.1931 +/- 0.0229 | 4.8588 +/- 0.0247 | 4.6661 +/- 0.0269 | 4.5666 +/- 0.0259 | 4.5163 +/- 0.0264 |
| RLB+ScheduleFree | 6.0415 +/- 0.0245 | 5.5256 +/- 0.0232 | 5.2750 +/- 0.0196 | 5.1298 +/- 0.0199 | 5.0436 +/- 0.0196 | 4.9877 +/- 0.0196 |
| SiLU+ScheduleFree | 6.0727 +/- 0.0207 | 5.5504 +/- 0.0249 | 5.3021 +/- 0.0209 | 5.1573 +/- 0.0198 | 5.0707 +/- 0.0193 | 5.0142 +/- 0.0188 |
| RLB+CAME | 6.0994 +/- 0.0162 | 5.6161 +/- 0.0098 | 5.3788 +/- 0.0155 | 5.2417 +/- 0.0161 | 5.1651 +/- 0.0167 | 5.1251 +/- 0.0173 |
| SiLU+CAME | 6.1551 +/- 0.0249 | 5.6317 +/- 0.0207 | 5.3872 +/- 0.0165 | 5.2496 +/- 0.0148 | 5.1725 +/- 0.0124 | 5.1325 +/- 0.0126 |
| RLB+ADeMaMix | 36.3461 +/- 22.7023 | 41680578.9202 +/- 72192221.5413 | 44753414.1145 +/- 63289962.8122 (n=2) | 107762.9375 +/- 0.0000 (n=1) | 8023604.0000 +/- 0.0000 (n=1) | 3022914304.0000 +/- 0.0000 (n=1) |
| SiLU+ADeMaMix | 6.5763 +/- 0.2655 | 258.3239 +/- 44.9409 | 83.0640 +/- 6.9598 | 69.4284 +/- 15.1007 | 57.6825 +/- 14.4696 | 51.9965 +/- 15.1322 |

### Dolma-sample

All-method view:

![Dolma-sample E1 validation loss mean +/- std, all methods](results/iclr26_e1_figures/dolma_sample_core_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, all methods](results/iclr26_e1_figures/dolma_sample_core_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, all methods](results/iclr26_e1_figures/dolma_sample_core_training_loss_mean_std.svg)

Clean comparison view:

![Dolma-sample E1 validation loss mean +/- std, clean comparison](results/iclr26_e1_figures/dolma_sample_clean_validation_loss_mean_std.svg)

![Dolma-sample E1 validation PPL mean +/- std, clean comparison](results/iclr26_e1_figures/dolma_sample_clean_validation_ppl_mean_std.svg)

![Dolma-sample E1 training loss mean +/- std, clean comparison](results/iclr26_e1_figures/dolma_sample_clean_training_loss_mean_std.svg)

Dolma-sample validation-loss checkpoint table, mean +/- sample std:

| Method | 500 | 1000 | 1500 | 2000 | 2500 | 3050 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 5.4561 +/- 0.0192 | 4.9542 +/- 0.0159 | 4.6426 +/- 0.0020 | 4.4758 +/- 0.0039 | 4.3776 +/- 0.0037 | 4.3239 +/- 0.0046 |
| RLB+AdamW | 5.5707 +/- 0.0183 | 5.0703 +/- 0.0112 | 4.7756 +/- 0.0054 | 4.6177 +/- 0.0004 | 4.5315 +/- 0.0014 | 4.4881 +/- 0.0009 |
| SiLU+AdamW | 5.5934 +/- 0.0195 | 5.0809 +/- 0.0222 | 4.7821 +/- 0.0118 | 4.6197 +/- 0.0050 | 4.5310 +/- 0.0031 | 4.4862 +/- 0.0012 |
| RLB+Lion | 5.5929 +/- 0.0339 | 4.9824 +/- 0.0271 | 4.6652 +/- 0.0129 | 4.5053 +/- 0.0076 | 4.4155 +/- 0.0075 | 4.3693 +/- 0.0056 |
| SiLU+Lion | 5.6628 +/- 0.0202 | 5.0312 +/- 0.0103 | 4.6966 +/- 0.0021 | 4.5289 +/- 0.0051 | 4.4359 +/- 0.0042 | 4.3878 +/- 0.0046 |
| RLB+SOAP | 5.6096 +/- 0.0145 | 5.3383 +/- 0.2027 | 4.8130 +/- 0.0153 | 4.6354 +/- 0.0064 | 4.5549 +/- 0.0233 | 4.5029 +/- 0.0157 |
| SiLU+SOAP | 6.0964 +/- 0.2025 | 5.2723 +/- 0.0950 | 4.8553 +/- 0.0074 | 4.6570 +/- 0.0040 | 4.5539 +/- 0.0057 | 4.4987 +/- 0.0035 |
| RLB+Muon | 6.0065 +/- 0.0095 | 5.2374 +/- 0.0163 | 4.9153 +/- 0.0067 | 4.7237 +/- 0.0071 | 4.6130 +/- 0.0088 | 4.5586 +/- 0.0093 |
| SiLU+Muon | 6.0530 +/- 0.0100 | 5.2462 +/- 0.0160 | 4.9072 +/- 0.0101 | 4.7048 +/- 0.0142 | 4.5973 +/- 0.0117 | 4.5443 +/- 0.0108 |
| RLB+ScheduleFree | 6.0972 +/- 0.0252 | 5.5799 +/- 0.0211 | 5.3207 +/- 0.0142 | 5.1714 +/- 0.0137 | 5.0815 +/- 0.0141 | 5.0241 +/- 0.0136 |
| SiLU+ScheduleFree | 6.1296 +/- 0.0239 | 5.6030 +/- 0.0177 | 5.3454 +/- 0.0127 | 5.1974 +/- 0.0125 | 5.1075 +/- 0.0140 | 5.0494 +/- 0.0149 |
| RLB+CAME | 6.1255 +/- 0.0218 | 5.6432 +/- 0.0166 | 5.3986 +/- 0.0175 | 5.2610 +/- 0.0162 | 5.1826 +/- 0.0171 | 5.1425 +/- 0.0164 |
| SiLU+CAME | 6.1808 +/- 0.0170 | 5.6664 +/- 0.0188 | 5.4216 +/- 0.0168 | 5.2824 +/- 0.0187 | 5.2049 +/- 0.0189 | 5.1650 +/- 0.0189 |
| RLB+ADeMaMix | 12.7395 +/- 3.3039 | 84.8182 +/- 36.9919 | 27552.1354 +/- 46889.3594 | 10779257.3654 +/- 18662299.4102 | 570914604.1487 +/- 624651995.0860 | 1775543041.7179 +/- 2510996321.2383 (n=2) |
| SiLU+ADeMaMix | 6.4467 +/- 0.1480 | 168.0197 +/- 131.3544 | 5706.2408 +/- 5051.5183 | 10605.0472 +/- 10104.8045 | 30618.4067 +/- 44187.0559 | 28427.1854 +/- 40987.7167 |

## E1 Results Snapshot

Final validation-loss overview across all optimizers. Lower is better. Completed E1 dataset cells use mean +/- sample std over three seeds. C4 is still in progress, so the C4 column uses completed rows only and shows the current completed-seed count in parentheses.

| Method | DCLM final | FineWeb-Edu final | FineWeb final | Dolma-sample final | C4 partial final |
| --- | ---: | ---: | ---: | ---: | ---: |
| rlb_matrixpolicy_original | 4.256224 +/- 0.004972 | 4.088240 +/- 0.009434 | 4.318581 +/- 0.010914 | 4.323851 +/- 0.004565 | 4.281546 +/- 0.027902 (n=2) |
| rlb_lion | 4.305728 +/- 0.005836 | 4.142669 +/- 0.006812 | 4.367062 +/- 0.007532 | 4.369254 +/- 0.005561 | 4.334202 +/- 0.029364 (n=2) |
| silu_lion | 4.318333 +/- 0.006893 | 4.149366 +/- 0.009180 | 4.382518 +/- 0.008308 | 4.387783 +/- 0.004605 | 4.352276 +/- 0.021835 (n=2) |
| rlb_adamw | 4.404748 +/- 0.004551 | 4.237991 +/- 0.006110 | 4.470531 +/- 0.013305 | 4.488137 +/- 0.000894 | 4.436585 +/- 0.023792 (n=2) |
| silu_adamw | 4.405574 +/- 0.009903 | 4.237481 +/- 0.008644 | 4.475841 +/- 0.009656 | 4.486162 +/- 0.001204 | 4.442621 +/- 0.020000 (n=2) |
| silu_soap | 4.415980 +/- 0.003818 | 4.263003 +/- 0.022048 | 4.484025 +/- 0.011241 | 4.498726 +/- 0.003535 | 4.452748 +/- 0.015767 (n=2) |
| rlb_soap | 4.435091 +/- 0.021706 | 4.262287 +/- 0.013054 | 4.484953 +/- 0.025544 | 4.502910 +/- 0.015749 | 4.451351 +/- 0.002696 (n=2) |
| silu_muon | 4.457165 +/- 0.012562 | 4.278738 +/- 0.024267 | 4.516342 +/- 0.026358 | 4.544263 +/- 0.010821 | 4.471201 +/- 0.018488 (n=2) |
| rlb_muon | 4.474236 +/- 0.004136 | 4.287684 +/- 0.019926 | 4.521560 +/- 0.011976 | 4.558568 +/- 0.009268 | 4.482270 +/- 0.008022 (n=2) |
| rlb_schedulefree | 4.878139 +/- 0.005538 | 4.779696 +/- 0.011289 | 4.987660 +/- 0.019623 | 5.024134 +/- 0.013642 | 4.968041 +/- 0.020520 (n=2) |
| silu_schedulefree | 4.902321 +/- 0.011067 | 4.825844 +/- 0.007242 | 5.014212 +/- 0.018820 | 5.049396 +/- 0.014883 | 5.002830 +/- 0.020899 (n=2) |
| rlb_came | 5.007375 +/- 0.008213 | 4.904335 +/- 0.004631 | 5.125061 +/- 0.017257 | 5.142537 +/- 0.016354 | 5.098529 +/- 0.021289 (n=2) |
| silu_came | 5.010657 +/- 0.014306 | 4.920688 +/- 0.012317 | 5.132548 +/- 0.012598 | 5.164954 +/- 0.018870 | 5.121805 +/- 0.025034 (n=2) |
| silu_ademamix | 48.645454 +/- 13.725481 | 242.853696 +/- 242.012155 | 51.996538 +/- 15.132243 | 28427.185402 +/- 40987.716727 | 466.353416 +/- 435.129243 (n=2) |
| rlb_ademamix | 246105152.000000 +/- 0.000000 (n=1) | 7880.511495 +/- 12045.656859 | 3022914304.000000 +/- 0.000000 (n=1) | 1775543041.717896 +/- 2510996321.238340 (n=2) | 17684.628784 +/- 20105.625797 (n=2) |

DCLM is complete for all three E1 seeds. Final validation loss, mean and sample std over seeds:

| Method | Complete seeds | Mean | Std | Seed values |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3 | 4.256224 | 0.004972 | 4.251434, 4.261359, 4.255877 |
| rlb_lion | 3 | 4.305728 | 0.005836 | 4.307827, 4.310225, 4.299133 |
| silu_lion | 3 | 4.318333 | 0.006893 | 4.310379, 4.322079, 4.322542 |
| rlb_adamw | 3 | 4.404748 | 0.004551 | 4.401357, 4.409920, 4.402967 |
| silu_adamw | 3 | 4.405574 | 0.009903 | 4.394192, 4.412221, 4.410308 |
| silu_soap | 3 | 4.415980 | 0.003818 | 4.412983, 4.420279, 4.414679 |
| rlb_soap | 3 | 4.435091 | 0.021706 | 4.458359, 4.431526, 4.415388 |
| silu_muon | 3 | 4.457165 | 0.012562 | 4.442778, 4.465955, 4.462763 |
| rlb_muon | 3 | 4.474236 | 0.004136 | 4.477408, 4.475743, 4.469558 |
| rlb_schedulefree | 3 | 4.878139 | 0.005538 | 4.872795, 4.877769, 4.883852 |
| silu_schedulefree | 3 | 4.902321 | 0.011067 | 4.891297, 4.902236, 4.913431 |
| rlb_came | 3 | 5.007375 | 0.008213 | 5.005087, 5.000548, 5.016489 |
| silu_came | 3 | 5.010657 | 0.014306 | 5.000495, 5.004458, 5.027017 |
| silu_ademamix | 3 | 48.645454 | 13.725481 | 34.271767, 61.614742, 50.049854 |
| rlb_ademamix | 1 | 246105152.000000 | 0.000000 | 246105152.000000, non-finite, non-finite |

FineWeb-Edu is complete for all three E1 seeds. Final validation loss, mean and sample std over seeds:

| Method | Complete seeds | Mean | Std | Seed values |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3 | 4.088240 | 0.009434 | 4.092051, 4.077497, 4.095173 |
| rlb_lion | 3 | 4.142669 | 0.006812 | 4.144132, 4.135244, 4.148631 |
| silu_lion | 3 | 4.149366 | 0.009180 | 4.154374, 4.138771, 4.154952 |
| silu_adamw | 3 | 4.237481 | 0.008644 | 4.242263, 4.227503, 4.242677 |
| rlb_adamw | 3 | 4.237991 | 0.006110 | 4.240171, 4.231090, 4.242713 |
| rlb_soap | 3 | 4.262287 | 0.013054 | 4.260798, 4.276021, 4.250041 |
| silu_soap | 3 | 4.263003 | 0.022048 | 4.262650, 4.241133, 4.285225 |
| silu_muon | 3 | 4.278738 | 0.024267 | 4.280360, 4.253700, 4.302153 |
| rlb_muon | 3 | 4.287684 | 0.019926 | 4.306664, 4.266931, 4.289457 |
| rlb_schedulefree | 3 | 4.779696 | 0.011289 | 4.792676, 4.774242, 4.772171 |
| silu_schedulefree | 3 | 4.825844 | 0.007242 | 4.834194, 4.822071, 4.821268 |
| rlb_came | 3 | 4.904335 | 0.004631 | 4.909057, 4.899800, 4.904150 |
| silu_came | 3 | 4.920688 | 0.012317 | 4.929064, 4.906546, 4.926455 |
| silu_ademamix | 3 | 242.853696 | 242.012155 | 104.331055, 522.301819, 101.928215 |
| rlb_ademamix | 3 | 7880.511495 | 12045.656859 | 1178.819824, 676.105286, 21786.609375 |

FineWeb is complete for all three E1 seeds. Final validation loss, mean and sample std over seeds:

| Method | Complete seeds | Mean | Std | Seed values |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3 | 4.318581 | 0.010914 | 4.306077, 4.323467, 4.326198 |
| rlb_lion | 3 | 4.367062 | 0.007532 | 4.358393, 4.370788, 4.372005 |
| silu_lion | 3 | 4.382518 | 0.008308 | 4.373947, 4.390535, 4.383072 |
| rlb_adamw | 3 | 4.470531 | 0.013305 | 4.455188, 4.478891, 4.477512 |
| silu_adamw | 3 | 4.475841 | 0.009656 | 4.464763, 4.480283, 4.482476 |
| silu_soap | 3 | 4.484025 | 0.011241 | 4.472078, 4.485604, 4.494392 |
| rlb_soap | 3 | 4.484953 | 0.025544 | 4.462595, 4.512793, 4.479470 |
| silu_muon | 3 | 4.516342 | 0.026358 | 4.490554, 4.515236, 4.543236 |
| rlb_muon | 3 | 4.521560 | 0.011976 | 4.508226, 4.525053, 4.531402 |
| rlb_schedulefree | 3 | 4.987660 | 0.019623 | 4.965375, 5.002351, 4.995253 |
| silu_schedulefree | 3 | 5.014212 | 0.018820 | 4.993041, 5.029044, 5.020551 |
| rlb_came | 3 | 5.125061 | 0.017257 | 5.105318, 5.137263, 5.132603 |
| silu_came | 3 | 5.132548 | 0.012598 | 5.118973, 5.143862, 5.134809 |
| silu_ademamix | 3 | 51.996538 | 15.132243 | 51.005760, 67.599823, 37.384029 |
| rlb_ademamix | 1 | 3022914304.000000 | 0.000000 | 3022914304.000000, non-finite, non-finite |

Dolma-sample is complete for all three E1 seeds. Final validation loss, mean and sample std over seeds:

| Method | Complete seeds | Mean | Std | Seed values |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 3 | 4.323851 | 0.004565 | 4.319835, 4.328816, 4.322902 |
| rlb_lion | 3 | 4.369254 | 0.005561 | 4.374695, 4.369488, 4.363580 |
| silu_lion | 3 | 4.387783 | 0.004605 | 4.382976, 4.392156, 4.388218 |
| silu_adamw | 3 | 4.486162 | 0.001204 | 4.487082, 4.486603, 4.484799 |
| rlb_adamw | 3 | 4.488137 | 0.000894 | 4.487473, 4.487784, 4.489154 |
| silu_soap | 3 | 4.498726 | 0.003535 | 4.498732, 4.502258, 4.495188 |
| rlb_soap | 3 | 4.502910 | 0.015749 | 4.489188, 4.499438, 4.520106 |
| silu_muon | 3 | 4.544263 | 0.010821 | 4.536290, 4.539918, 4.556581 |
| rlb_muon | 3 | 4.558568 | 0.009268 | 4.549397, 4.567930, 4.558376 |
| rlb_schedulefree | 3 | 5.024134 | 0.013642 | 5.039133, 5.020803, 5.012465 |
| silu_schedulefree | 3 | 5.049396 | 0.014883 | 5.066480, 5.039243, 5.042466 |
| rlb_came | 3 | 5.142537 | 0.016354 | 5.159671, 5.140844, 5.127095 |
| silu_came | 3 | 5.164954 | 0.018870 | 5.184652, 5.163172, 5.147038 |
| silu_ademamix | 3 | 28427.185402 | 40987.716727 | 9786.633789, 75422.257812, 72.664604 |
| rlb_ademamix | 2 | 1775543041.717896 | 2510996321.238340 | 515.435791, 3551085568.000000, non-finite |

C4 status at check:

| Seed | Complete rows | Running row | Notes |
| ---: | ---: | --- | --- |
| 1337 | 15 / 15 | none | MatrixPolicy final val loss 4.261817 |
| 2027 | 15 / 15 | none | MatrixPolicy final val loss 4.301276 |
| 3407 | 0 / 15 | row 225 silu_adamw | latest train 930, eval 900, val 5.126665 |

C4 partial aggregate from completed rows only:

| Method | Complete seeds | Mean | Std | Seed values |
| --- | ---: | ---: | ---: | --- |
| rlb_matrixpolicy_original | 2 | 4.281546 | 0.027902 | 4.261817, 4.301276 |
| rlb_lion | 2 | 4.334202 | 0.029364 | 4.313438, 4.354966 |
| silu_lion | 2 | 4.352276 | 0.021835 | 4.336836, 4.367715 |
| rlb_adamw | 2 | 4.436585 | 0.023792 | 4.419761, 4.453408 |
| silu_adamw | 2 | 4.442621 | 0.020000 | 4.428479, 4.456763 |
| rlb_soap | 2 | 4.451351 | 0.002696 | 4.449445, 4.453258 |
| silu_soap | 2 | 4.452748 | 0.015767 | 4.441599, 4.463897 |
| silu_muon | 2 | 4.471201 | 0.018488 | 4.458128, 4.484274 |
| rlb_muon | 2 | 4.482270 | 0.008022 | 4.476597, 4.487942 |
| rlb_schedulefree | 2 | 4.968041 | 0.020520 | 4.953531, 4.982551 |
| silu_schedulefree | 2 | 5.002830 | 0.020899 | 4.988052, 5.017608 |
| rlb_came | 2 | 5.098529 | 0.021289 | 5.083476, 5.113583 |
| silu_came | 2 | 5.121805 | 0.025034 | 5.104103, 5.139506 |
| silu_ademamix | 2 | 466.353416 | 435.129243 | 158.670578, 774.036255 |
| rlb_ademamix | 2 | 17684.628784 | 20105.625797 | 31901.453125, 3467.804443 |

## Parameter Counts

M0 parameter counts from the run config records:

| Row family | Activation family | Parameter count |
| --- | --- | ---: |
| `silu_*` | SiLU FFN | 123,551,232 |
| `rlb_*` | RLB FFN | 123,553,824 |
| `rlb_matrixpolicy_original` | RLB FFN with MatrixPolicy optimizer | 123,553,824 |
