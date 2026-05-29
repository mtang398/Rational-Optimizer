# Synthetic Curve Diagnostics

Final synthetic loss is close to the floor, so the main synthetic signal is curve speed.
The table below reports the best validation-loss row at early eval checkpoints against `SiLU/SwiGLU+AdamW`.

| task | step | best curve row | loss | PPL | delta loss vs SiLU+AdamW | delta PPL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Code | 250 | RLB MatrixPolicy group-stat | 0.166084 | 1.1807 | -0.323368 | -0.4507 |
| Code | 500 | RLB+Muon | 0.128043 | 1.1366 | -0.010649 | -0.0122 |
| Code | 750 | RLB+AdamW | 0.102544 | 1.1080 | -0.000955 | -0.0011 |
| Symbolic | 250 | RLB MatrixPolicy | 0.048723 | 1.0499 | -0.012205 | -0.0129 |
| Symbolic | 500 | RLB+AdamW | 0.042278 | 1.0432 | -0.002179 | -0.0023 |
| Symbolic | 750 | RLB MatrixPolicy group-stat | 0.039988 | 1.0408 | -0.003208 | -0.0033 |
| Reasoning mix | 250 | RLB MatrixPolicy | 0.344989 | 1.4120 | -0.067693 | -0.0989 |
| Reasoning mix | 500 | RLB MatrixPolicy group-stat | 0.210741 | 1.2346 | -0.011161 | -0.0139 |
| Reasoning mix | 750 | RLB MatrixPolicy group-stat | 0.168578 | 1.1836 | -0.008341 | -0.0099 |

The AUC metric integrates validation loss from step 250 through 1250, excluding the random-init step 1 point.

| task | best AUC250 row | AUC loss 250-1250 |
| --- | --- | ---: |
| Code | RLB+AdamW | 114.97 |
| Symbolic | RLB MatrixPolicy | 42.66 |
| Reasoning mix | RLB MatrixPolicy group-stat | 194.62 |
