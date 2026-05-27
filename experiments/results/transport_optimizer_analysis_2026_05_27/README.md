# Transport optimizer analysis, 2026-05-27

These figures summarize the on-policy rational optimizer probes against the seed-1337 h3072 RLB incumbent. Historical h2880 and SiLU entries are retained in the CSV for context, but the figures and focus table compare the same h3072 RLB setting. Raw transport run directories were cleaned after this compact ledger was generated.

![Validation loss and PPL curves](loss_ppl_curves.png)

![Final validation bars](final_loss_ppl_bars.png)

## h3072 RLB complete runs

| rank | label | final loss | final PPL | delta vs Jacobian h3072 | category |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | Jacobian h3072 RLB | 3.614862 | 37.146 | +0.000000 | best incumbent |
| 2 | Transport matrix 0.65 | 3.615149 | 37.157 | +0.000287 | matrix only |
| 3 | Transport matrix 0.70 | 3.615180 | 37.158 | +0.000318 | matrix only |
| 4 | Transport matrix-only | 3.615939 | 37.186 | +0.001077 | matrix only |
| 5 | Matrix 0.60 + time | 3.616660 | 37.213 | +0.001798 | matrix/time ramp |
| 6 | Adaptive metric h3072 RLB | 3.617174 | 37.232 | +0.002312 | matrix/metric |
| 7 | Matrix 0.35 | 3.617309 | 37.237 | +0.002447 | matrix only |
| 8 | AdamW h3072 RLB | 3.617501 | 37.244 | +0.002639 | baseline |
| 9 | Layer stagger wide pre-fix | 3.619816 | 37.331 | +0.004954 | layer schedule |
| 10 | Selector + pullback | 3.619819 | 37.331 | +0.004957 | selector/pullback |
| 11 | Switch no-stagger 0.43 | 3.620000 | 37.338 | +0.005138 | switching |
| 12 | Late depth smooth | 3.620323 | 37.350 | +0.005461 | layer schedule |
| 13 | Switch no-stagger 0.39 | 3.620544 | 37.358 | +0.005682 | switching |
| 14 | Moderate pullback | 3.620605 | 37.360 | +0.005743 | pullback |
| 15 | Low gain after switch | 3.620783 | 37.367 | +0.005921 | switching |
| 16 | Reset on switch | 3.621302 | 37.386 | +0.006440 | switching |
| 17 | Freeze after switch | 3.621391 | 37.390 | +0.006529 | switching |
| 18 | Transport scheduled | 3.621399 | 37.390 | +0.006537 | scheduled coeff |

See `transport_probe_summary.csv` for all retained probe metrics, including h2880 comparison rows and incomplete/preempted probes.
