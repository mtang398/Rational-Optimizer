# ICLR Run Status

Updated: 2026-06-04 16:47:50 EDT  
Commit: `3a3b415`  
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

## E0 Row Table With Eval Curves

| Row | Dataset | Method | Status | Final val loss | Tokens/s | Stopped early | Eval curve `step:val_loss` |
| ---: | --- | --- | --- | ---: | ---: | --- | --- |
| 0 | dclm | silu_adamw | complete | 7.346231 | 52741.71 | false | 1:10.798578, 40:7.553685, 80:7.346231 |
| 1 | dclm | rlb_adamw | complete | 7.238834 | 47290.45 | false | 1:10.533360, 40:7.471305, 80:7.238834 |
| 2 | dclm | rlb_matrixpolicy_original | complete | 7.121294 | 46977.61 | false | 1:10.301031, 40:7.391489, 80:7.121294 |
| 3 | fineweb_edu | silu_adamw | complete | 7.426810 | 53050.87 | false | 1:10.775492, 40:7.637898, 80:7.426810 |
| 4 | fineweb_edu | rlb_adamw | complete | 7.211745 | 47273.08 | false | 1:10.528216, 40:7.452968, 80:7.211745 |
| 5 | fineweb_edu | rlb_matrixpolicy_original | complete | 7.035759 | 46593.49 | false | 1:10.283565, 40:7.335752, 80:7.035759 |
| 6 | fineweb | silu_adamw | complete | 7.527949 | 52895.84 | false | 1:10.824485, 40:7.717688, 80:7.527949 |
| 7 | fineweb | rlb_adamw | complete | 7.257327 | 47540.86 | false | 1:10.569622, 40:7.519713, 80:7.257327 |
| 8 | fineweb | rlb_matrixpolicy_original | complete | 7.101671 | 46753.24 | false | 1:10.334897, 40:7.368423, 80:7.101671 |
| 9 | dolma_sample | silu_adamw | complete | 8.201650 | 52563.79 | false | 1:10.863197, 40:8.279759, 80:8.201650 |
| 10 | dolma_sample | rlb_adamw | complete | 8.212225 | 47123.59 | false | 1:10.679456, 40:8.226797, 80:8.212225 |
| 11 | dolma_sample | rlb_matrixpolicy_original | complete | 8.189895 | 46758.65 | false | 1:10.488925, 40:8.187285, 80:8.189895 |
| 12 | c4_en | silu_adamw | complete | 7.579340 | 52581.93 | false | 1:10.825615, 40:7.731586, 80:7.579340 |
| 13 | c4_en | rlb_adamw | complete | 7.278966 | 47277.57 | false | 1:10.567333, 40:7.533151, 80:7.278966 |
| 14 | c4_en | rlb_matrixpolicy_original | complete | 7.207716 | 46848.39 | false | 1:10.340908, 40:7.446258, 80:7.207716 |

## E1 Launch Status

E1 uses whole matched 15-row cells. The first two cells are running:

| Job | Row start | Row limit | Cell | State at update | GPUs | Node |
| --- | ---: | ---: | --- | --- | --- | --- |
| `155411` | 15 | 15 | E1 dclm seed 1337, all 15 methods | running | 4 A6000 | `ma-compute-02` |
| `155412` | 30 | 15 | E1 dclm seed 2027, all 15 methods | running | 4 A6000 | `bala-compute-02` |

Active allocation at update: 8 A6000 total.
