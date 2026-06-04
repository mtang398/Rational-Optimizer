# ICLR Run Status

Updated: 2026-06-04 17:22:11 EDT  
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

## E1 Launch Status

E1 uses whole matched 15-row cells. The first two cells are running:

| Job | Row start | Row limit | Cell | State at update | GPUs | Node |
| --- | ---: | ---: | --- | --- | --- | --- |
| `155411` | 15 | 15 | E1 dclm seed 1337, all 15 methods | running | 4 A6000 | `ma-compute-02` |
| `155412` | 30 | 15 | E1 dclm seed 2027, all 15 methods | running | 4 A6000 | `bala-compute-02` |

Active allocation at update: 8 A6000 total.

## E1 Continuation Queue

The remaining E1 cells are queued in whole 15-row matched blocks. Each job uses 4 A6000. Dependencies are chained in pairs so the queue advances at most two jobs at a time.

| Wave | Dependency | Job | Rows | Cell |
| ---: | --- | --- | --- | --- |
| 0 | running now | `155411` | 15-29 | dclm seed 1337 |
| 0 | running now | `155412` | 30-44 | dclm seed 2027 |
| 1 | afterok:`155411`:`155412` | `158114` | 45-59 | dclm seed 3407 |
| 1 | afterok:`155411`:`155412` | `158115` | 60-74 | fineweb_edu seed 1337 |
| 2 | afterok:`158114`:`158115` | `158117` | 75-89 | fineweb_edu seed 2027 |
| 2 | afterok:`158114`:`158115` | `158118` | 90-104 | fineweb_edu seed 3407 |
| 3 | afterok:`158117`:`158118` | `158155` | 105-119 | fineweb seed 1337 |
| 3 | afterok:`158117`:`158118` | `158156` | 120-134 | fineweb seed 2027 |
| 4 | afterok:`158155`:`158156` | `158163` | 135-149 | fineweb seed 3407 |
| 4 | afterok:`158155`:`158156` | `158164` | 150-164 | dolma_sample seed 1337 |
| 5 | afterok:`158163`:`158164` | `158166` | 165-179 | dolma_sample seed 2027 |
| 5 | afterok:`158163`:`158164` | `158165` | 180-194 | dolma_sample seed 3407 |
| 6 | afterok:`158166`:`158165` | `158168` | 195-209 | c4_en seed 1337 |
| 6 | afterok:`158166`:`158165` | `158167` | 210-224 | c4_en seed 2027 |
| 7 | afterok:`158168`:`158167` | `158169` | 225-239 | c4_en seed 3407 |

## E1 Live Timing

Current check: 2026-06-04 17:22:11 EDT.

| Job | Current row | Method | Elapsed | Latest train step | Latest eval | Latest val loss | Recent step time | Recent tokens/s |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `155411` | 15 | silu_adamw | 00:35:17 | 2170 / 3050 | 2150 | 4.488902 | 0.6244 s | 52482.49 |
| `155412` | 30 | silu_adamw | 00:35:00 | 2140 / 3050 | 2100 | 4.514518 | 0.6216 s | 52718.29 |

Finish estimate for the two running E1 cells: 2026-06-05 03:00-07:00 EDT. The first row is running at about 52k tokens/s, but the 15-row cells include optimizer families with different per-step costs.

## Parameter Counts

M0 parameter counts from the run config records:

| Row family | Activation family | Parameter count |
| --- | --- | ---: |
| `silu_*` | SiLU FFN | 123,551,232 |
| `rlb_*` | RLB FFN | 123,553,824 |
| `rlb_matrixpolicy_original` | RLB FFN with MatrixPolicy optimizer | 123,553,824 |
