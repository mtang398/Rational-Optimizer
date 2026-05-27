# Transport optimizer analysis, 2026-05-27

This analysis includes `AdamW + SiLU/SwiGLU` as the external baseline and `AdamW + RLB h3072` as the activation-controlled baseline. Jacobian is not the baseline; it is the current best rational-specific row. Raw transport run directories were cleaned after this compact ledger was generated.

![Optimizer comparison loss and PPL curves](loss_ppl_curves.png)

![Final validation bars](final_loss_ppl_bars.png)

## Seed-1337 Primary Comparisons

Negative deltas are better than the named baseline.

| row | final loss | final PPL | delta vs SiLU AdamW | delta vs RLB AdamW | category |
| --- | ---: | ---: | ---: | ---: | --- |
| AdamW + SiLU/SwiGLU | 3.621982 | 37.412 | +0.000000 | +0.004480 | primary baseline |
| AdamW + RLB h3072 | 3.617501 | 37.244 | -0.004480 | +0.000000 | activation baseline |
| RLB h3072 + rational_jacobian | 3.614862 | 37.146 | -0.007120 | -0.002639 | current best rational optimizer |
| Transport matrix 0.65 | 3.615149 | 37.157 | -0.006833 | -0.002352 | matrix only |
| Transport matrix 0.70 | 3.615180 | 37.158 | -0.006802 | -0.002321 | matrix only |
| Matrix 0.60 + time | 3.616660 | 37.213 | -0.005322 | -0.000841 | matrix/time ramp |
| Selector + pullback | 3.619819 | 37.331 | -0.002162 | +0.002318 | selector/pullback |
| Layer switch wide | 3.621418 | 37.391 | -0.000564 | +0.003917 | layer schedule |
| Transport xfast | 3.625419 | 37.540 | +0.003438 | +0.007918 | aggressive coeff |

The three-seed headline comparison is still against `AdamW + SiLU/SwiGLU`: the best full row, `RLB h3072 + rational_jacobian_onpolicy`, improves mean loss by `-0.004736` versus that baseline and by `-0.001236` versus `AdamW + RLB h3072`. The transport probes did not increase that headline gap; they mostly show that matrix geometry is useful and aggressive coefficient motion is harmful.

See `transport_probe_summary.csv` for all retained probe metrics, including h2880 comparison rows and incomplete/preempted probes.
