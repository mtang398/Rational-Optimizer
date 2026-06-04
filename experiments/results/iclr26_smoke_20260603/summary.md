# ICLR 2026 Smoke Summary - 2026-06-03

These are feasibility smokes for the 2026 ICLR experiment plan. They are not paper evidence, but they verify loaders, compiled RLB execution, model geometry, and early optimizer stability before protocol-locked runs.

## Phase 0A/0B M0 Loader/Optimizer Smokes

Final validation loss, lower is better:

| dataset | SiLU+AdamW | RLB+AdamW | SiLU+Muon | RLB+Muon | SiLU+SOAP | RLB+SOAP | RLB+MatrixPolicy group-stat | best smoke row |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `dclm` | 7.2601 | 7.1610 | 8.8333 | 8.3440 | 7.1443 | 7.1319 | 7.0418 | MatrixPolicy |
| `fineweb_edu` | 7.4095 | 7.2029 | 8.8111 | 8.4145 | 7.1700 | 7.1295 | 7.0805 | MatrixPolicy |
| `dolma_sample` | 6.2900 | 6.2468 | 8.1279 | 7.5320 | 6.0741 | 6.1706 | 6.0709 | MatrixPolicy, near SOAP/SiLU |
| `c4_en` | 7.5214 | 7.3066 | 8.8898 | 8.4984 | 7.2832 | 7.2602 | 7.2037 | MatrixPolicy |

Slurm jobs:

| job | tasks | state | elapsed |
| --- | --- | --- | --- |
| `62426` | `dclm fineweb_edu` | completed, exit 0 | 00:31:38 |
| `62425` | `dolma_sample c4_en` | completed, exit 0 | 00:36:49 |

## Phase 0C M1 DCLM Smoke

Job `65084` completed with exit `0:0` in `00:11:12`.

| row | final step | final val loss | tokens/s |
| --- | ---: | ---: | ---: |
| `dclm_adamw_controls_iclr26_smoke_m1_dclm/silu` | 120 | 6.8513 | 30607.3 |
| `dclm_adamw_controls_iclr26_smoke_m1_dclm/rlb_fused_fixed_strong_ffn` | 120 | 6.7335 | 24243.2 |
| `dclm_matrix_policy_groupstat_iclr26_smoke_m1_dclm/rlb_fused_fixed_strong_ffn` | 120 | 6.7349 | 25005.4 |

At this short M1 smoke scale, both RLB+AdamW and RLB+MatrixPolicy beat SiLU+AdamW. MatrixPolicy is essentially tied with RLB+AdamW in final validation loss and is faster than RLB+AdamW in tokens/s for this run.

## Active Continuation At Handoff

Latest observed at 2026-06-04T14:19:04-04:00:

| job | purpose | elapsed | latest observed row | latest train | latest eval | GPU use |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `71047` | Phase 1 protocol-lock DCLM MatrixPolicy shard, configs 8-11 | 03:49:24+ | `matrix_policy lr=0.0002 wd=0.10 adam_scale=4.0 group_gain=0.20` | step 330 loss 5.5522 | eval step 300 loss 5.6774 | 4 A6000 |
| `71048` | Phase 1 protocol-lock FineWeb-Edu AdamW control shard, configs 0-3 | 00:01:39 after restart | `adamw lr=0.0001 wd=0.03 / silu` | step 1 loss 11.0197 | eval step 1 loss 11.0267 | 4 A6000 |

No immediate-start GPU work should be submitted while both jobs are active because the 8 A6000 cap is fully used; dependency-held continuation is already queued.

Future summaries must prioritize curves, AUC, and trajectory behavior over final-number-only tables.

Queued continuation: `71049` is dependency-held on `afterok:71047`; `143550` and `143584` continue FineWeb-Edu AdamW after restarted `71048`; `143591` and `143611` continue FineWeb-Edu MatrixPolicy after `71049`.