# Synthetic Curve Diagnostics

The sparse historical run is useful only as a provisional curve smoke test.
Dense reruns should use frequent training logs and validation evals before making optimizer claims.

## Validation Curve

| task | step | best curve row | loss | PPL | delta loss vs SiLU+AdamW | delta PPL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Code | 250 | RLB MatrixPolicy group-stat | 0.166084 | 1.1807 | -0.323368 | -0.4507 |
| Code | 500 | RLB+Muon | 0.128043 | 1.1366 | -0.010649 | -0.0122 |
| Code | 750 | RLB+AdamW | 0.102544 | 1.1080 | -0.000955 | -0.0011 |
| Code | 1000 | SiLU/SwiGLU+AdamW | 0.092413 | 1.0968 | +0.000000 | +0.0000 |
| Code | 1250 | SiLU/SwiGLU+AdamW | 0.088975 | 1.0931 | +0.000000 | +0.0000 |
| Symbolic | 250 | RLB MatrixPolicy | 0.048723 | 1.0499 | -0.012205 | -0.0129 |
| Symbolic | 500 | RLB+AdamW | 0.042278 | 1.0432 | -0.002179 | -0.0023 |
| Symbolic | 750 | RLB MatrixPolicy group-stat | 0.039988 | 1.0408 | -0.003208 | -0.0033 |
| Symbolic | 1000 | SiLU/SwiGLU+Muon | 0.039191 | 1.0400 | -0.001873 | -0.0019 |
| Symbolic | 1250 | SiLU/SwiGLU+Muon | 0.038782 | 1.0395 | -0.001507 | -0.0016 |
| Reasoning mix | 250 | RLB MatrixPolicy | 0.344989 | 1.4120 | -0.067693 | -0.0989 |
| Reasoning mix | 500 | RLB MatrixPolicy group-stat | 0.210741 | 1.2346 | -0.011161 | -0.0139 |
| Reasoning mix | 750 | RLB MatrixPolicy group-stat | 0.168578 | 1.1836 | -0.008341 | -0.0099 |
| Reasoning mix | 1000 | SiLU/SwiGLU+AdamW | 0.152818 | 1.1651 | +0.000000 | +0.0000 |
| Reasoning mix | 1250 | RLB+Muon | 0.144238 | 1.1552 | -0.002233 | -0.0026 |

| task | best AUC row | step range | AUC loss |
| --- | --- | ---: | ---: |
| Code | RLB+AdamW | 250-1250 | 114.97 |
| Symbolic | RLB MatrixPolicy | 250-1250 | 42.66 |
| Reasoning mix | RLB MatrixPolicy group-stat | 250-1250 | 194.62 |

## Training Curve

| task | step | best curve row | loss | PPL | delta loss vs SiLU+AdamW | delta PPL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Code | 100 | RLB MatrixPolicy group-stat | 3.795936 | 44.5199 | -1.065268 | -84.6597 |
| Code | 200 | RLB MatrixPolicy group-stat | 0.432170 | 1.5406 | -0.132588 | -0.2184 |
| Code | 500 | RLB MatrixPolicy | 0.130015 | 1.1388 | -0.013960 | -0.0160 |
| Code | 1000 | SiLU/SwiGLU+AdamW | 0.094018 | 1.0986 | +0.000000 | +0.0000 |
| Code | 1250 | SiLU/SwiGLU+AdamW | 0.088896 | 1.0930 | +0.000000 | +0.0000 |
| Symbolic | 100 | RLB MatrixPolicy group-stat | 3.087313 | 21.9181 | -1.518596 | -78.1558 |
| Symbolic | 200 | RLB MatrixPolicy | 0.178548 | 1.1955 | -0.118522 | -0.1504 |
| Symbolic | 500 | RLB+Muon | 0.044007 | 1.0450 | -0.003110 | -0.0033 |
| Symbolic | 1000 | SiLU/SwiGLU+Muon | 0.039575 | 1.0404 | -0.002144 | -0.0022 |
| Symbolic | 1250 | SiLU/SwiGLU+Muon | 0.038951 | 1.0397 | -0.001483 | -0.0015 |
| Reasoning mix | 100 | RLB MatrixPolicy | 4.643096 | 103.8654 | -1.457139 | -342.0969 |
| Reasoning mix | 200 | RLB MatrixPolicy | 0.724954 | 2.0646 | -0.136498 | -0.3020 |
| Reasoning mix | 500 | RLB+AdamW | 0.231858 | 1.2609 | -0.011258 | -0.0143 |
| Reasoning mix | 1000 | RLB MatrixPolicy group-stat | 0.155724 | 1.1685 | -0.000289 | -0.0003 |
| Reasoning mix | 1250 | RLB+AdamW | 0.145284 | 1.1564 | -0.001490 | -0.0017 |

| task | best AUC row | step range | AUC loss |
| --- | --- | ---: | ---: |
| Code | RLB MatrixPolicy group-stat | 100-1250 | 364.74 |
| Symbolic | RLB MatrixPolicy group-stat | 100-1250 | 216.22 |
| Reasoning mix | RLB MatrixPolicy | 100-1250 | 511.19 |
