# ICLR Run Status

Updated: 2026-06-04 16:22:16 EDT  
Commit: `7dba156`  
Manifest: `experiments/manifests/iclr26_main_manifest.csv`

## Current Scheduler State

| Job | Phase rows | State | GPUs | Elapsed at last check | Node |
| --- | --- | --- | --- | --- | --- |
| `151609` | E0 rows 0-8 | running | 4 A6000 | 00:19:01 | `ma-compute-02` |
| `151610` | E0 rows 9-14 | running | 4 A6000 | 00:09:36 | `bala-compute-02` |

Active allocation is 8 A6000 total. E1 has not been launched yet.

## E0 Preflight Gate

E0 is a 15-row smoke pass: 5 corpora times 3 matched methods. Each dataset cell uses the same outer training config across `silu_adamw`, `rlb_adamw`, and `rlb_matrixpolicy_original`: `lr=0.0003`, `min_lr=0.00003`, `weight_decay=0.10`, seed `1337`, `steps=80`, `eval_interval=40`, 4 A6000.

E1 starts only after all 15 E0 rows have summary records.

## E0 Parsed Rows

| Row | Dataset | Method | Status | Last eval step | Last val loss | Tokens/s | Stopped early |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | dclm | silu_adamw | complete | 80 | 7.346231 | 52741.71 | false |
| 1 | dclm | rlb_adamw | complete | 80 | 7.238834 | 47290.45 | false |
| 2 | dclm | rlb_matrixpolicy_original | complete | 80 | 7.121294 | 46977.61 | false |
| 3 | fineweb_edu | silu_adamw | waiting |  |  |  |  |
| 4 | fineweb_edu | rlb_adamw | waiting |  |  |  |  |
| 5 | fineweb_edu | rlb_matrixpolicy_original | waiting |  |  |  |  |
| 6 | fineweb | silu_adamw | waiting |  |  |  |  |
| 7 | fineweb | rlb_adamw | waiting |  |  |  |  |
| 8 | fineweb | rlb_matrixpolicy_original | waiting |  |  |  |  |
| 9 | dolma_sample | silu_adamw | complete | 80 | 8.201650 | 52563.79 | false |
| 10 | dolma_sample | rlb_adamw | running | 1 | 10.679456 |  |  |
| 11 | dolma_sample | rlb_matrixpolicy_original | waiting |  |  |  |  |
| 12 | c4_en | silu_adamw | waiting |  |  |  |  |
| 13 | c4_en | rlb_adamw | waiting |  |  |  |  |
| 14 | c4_en | rlb_matrixpolicy_original | waiting |  |  |  |  |

## Current Observations

- DCLM E0 completed all three matched rows. The last validation losses are `7.346231` for `silu_adamw`, `7.238834` for `rlb_adamw`, and `7.121294` for `rlb_matrixpolicy_original`.
- Dolma row 9 completed and row 10 has started.
- FineWeb-Edu row 3 is still before its prepared record in the parsed JSONL output.

## Next Gate Action

When all E0 rows have summary records, write the final E0 table here, then start E1 using whole matched 15-row cells only. The first planned E1 submissions are:

| Row start | Row limit | Cell |
| ---: | ---: | --- |
| 15 | 15 | E1 dclm seed 1337, all 15 methods |
| 30 | 15 | E1 dclm seed 2027, all 15 methods |

