# Synthetic Curve Diagnostics

The sparse historical run is useful only as a provisional curve smoke test.
Dense reruns should use frequent training logs and validation evals before making optimizer claims.

## Validation Curve

| task | step | best curve row | loss | PPL | delta loss vs SiLU+AdamW | delta PPL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Code | 25 | RLB MatrixPolicy | 6.300732 | 544.9708 | -1.585426 | -2115.2330 |
| Code | 50 | RLB MatrixPolicy group-stat | 2.523935 | 12.4776 | -2.198752 | -99.9924 |
| Code | 100 | RLB MatrixPolicy group-stat | 0.723051 | 2.0607 | -0.044173 | -0.0931 |
| Code | 200 | RLB MatrixPolicy | 0.190162 | 1.2094 | -0.157474 | -0.2063 |
| Code | 250 | RLB MatrixPolicy | 0.166492 | 1.1812 | -0.040961 | -0.0494 |
| Code | 500 | RLB MatrixPolicy | 0.125350 | 1.1335 | -0.007964 | -0.0091 |
| Code | 750 | RLB+AdamW | 0.101703 | 1.1071 | -0.001762 | -0.0020 |
| Code | 1000 | SiLU/SwiGLU+AdamW | 0.091311 | 1.0956 | +0.000000 | +0.0000 |
| Code | 1250 | SiLU/SwiGLU+AdamW | 0.088662 | 1.0927 | +0.000000 | +0.0000 |
| Symbolic | 25 | RLB MatrixPolicy | 4.700200 | 109.9691 | -3.485178 | -3478.1308 |
| Symbolic | 50 | RLB MatrixPolicy | 1.718203 | 5.5745 | -1.907369 | -31.9717 |
| Symbolic | 100 | RLB MatrixPolicy | 0.328362 | 1.3887 | -0.137013 | -0.2039 |
| Symbolic | 200 | RLB MatrixPolicy group-stat | 0.061641 | 1.0636 | -0.085782 | -0.0953 |
| Symbolic | 250 | RLB MatrixPolicy | 0.058394 | 1.0601 | -0.001648 | -0.0017 |
| Symbolic | 500 | SiLU/SwiGLU+Muon | 0.042311 | 1.0432 | -0.007303 | -0.0076 |
| Symbolic | 750 | SiLU/SwiGLU+Muon | 0.039784 | 1.0406 | -0.003328 | -0.0035 |
| Symbolic | 1000 | SiLU/SwiGLU+Muon | 0.039041 | 1.0398 | -0.001856 | -0.0019 |
| Symbolic | 1250 | SiLU/SwiGLU+AdamW | 0.038529 | 1.0393 | +0.000000 | +0.0000 |
| Reasoning mix | 25 | RLB MatrixPolicy | 7.372604 | 1591.7732 | -1.364088 | -4635.4854 |
| Reasoning mix | 50 | RLB MatrixPolicy group-stat | 3.815826 | 45.4143 | -2.614003 | -574.6537 |
| Reasoning mix | 100 | RLB MatrixPolicy group-stat | 1.092136 | 2.9806 | -0.305052 | -1.0632 |
| Reasoning mix | 200 | RLB MatrixPolicy | 0.449614 | 1.5677 | -0.142467 | -0.2400 |
| Reasoning mix | 250 | RLB MatrixPolicy | 0.348053 | 1.4163 | -0.071602 | -0.1051 |
| Reasoning mix | 500 | RLB MatrixPolicy group-stat | 0.204325 | 1.2267 | -0.013934 | -0.0172 |
| Reasoning mix | 750 | RLB MatrixPolicy group-stat | 0.165798 | 1.1803 | -0.008517 | -0.0101 |
| Reasoning mix | 1000 | RLB MatrixPolicy group-stat | 0.149868 | 1.1617 | -0.002521 | -0.0029 |
| Reasoning mix | 1250 | RLB MatrixPolicy group-stat | 0.142429 | 1.1531 | -0.002138 | -0.0025 |

| task | best AUC row | step range | AUC loss |
| --- | --- | ---: | ---: |
| Code | RLB MatrixPolicy | 25-1250 | 361.23 |
| Symbolic | RLB MatrixPolicy group-stat | 25-1250 | 188.76 |
| Reasoning mix | RLB MatrixPolicy group-stat | 25-1250 | 527.91 |

## Training Curve

| task | step | best curve row | loss | PPL | delta loss vs SiLU+AdamW | delta PPL |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Code | 10 | RLB MatrixPolicy | 9.453035 | 12746.7990 | -0.813512 | -16007.6455 |
| Code | 50 | RLB MatrixPolicy group-stat | 3.236659 | 25.4486 | -2.485530 | -280.1245 |
| Code | 100 | RLB MatrixPolicy group-stat | 0.765602 | 2.1503 | -0.056353 | -0.1247 |
| Code | 200 | RLB MatrixPolicy | 0.201386 | 1.2231 | -0.167149 | -0.2225 |
| Code | 250 | RLB MatrixPolicy | 0.168004 | 1.1829 | -0.044725 | -0.0541 |
| Code | 500 | RLB MatrixPolicy | 0.125526 | 1.1337 | -0.008072 | -0.0092 |
| Code | 750 | RLB+AdamW | 0.101254 | 1.1066 | -0.002833 | -0.0031 |
| Code | 1000 | SiLU/SwiGLU+AdamW | 0.091790 | 1.0961 | +0.000000 | +0.0000 |
| Code | 1250 | SiLU/SwiGLU+AdamW | 0.089119 | 1.0932 | +0.000000 | +0.0000 |
| Symbolic | 10 | RLB MatrixPolicy | 9.767559 | 17458.1045 | -0.693433 | -17468.0975 |
| Symbolic | 50 | RLB MatrixPolicy | 2.076882 | 7.9795 | -2.695557 | -110.2276 |
| Symbolic | 100 | RLB MatrixPolicy | 0.378153 | 1.4596 | -0.201371 | -0.3256 |
| Symbolic | 200 | RLB MatrixPolicy | 0.061609 | 1.0635 | -0.084445 | -0.0937 |
| Symbolic | 250 | RLB MatrixPolicy | 0.059238 | 1.0610 | -0.002591 | -0.0028 |
| Symbolic | 500 | SiLU/SwiGLU+Muon | 0.042938 | 1.0439 | -0.006884 | -0.0072 |
| Symbolic | 750 | SiLU/SwiGLU+Muon | 0.040377 | 1.0412 | -0.003168 | -0.0033 |
| Symbolic | 1000 | SiLU/SwiGLU+Muon | 0.039212 | 1.0400 | -0.001837 | -0.0019 |
| Symbolic | 1250 | SiLU/SwiGLU+AdamW | 0.038732 | 1.0395 | +0.000000 | +0.0000 |
| Reasoning mix | 10 | RLB MatrixPolicy | 10.012796 | 22310.1270 | -0.625566 | -19394.2868 |
| Reasoning mix | 50 | RLB MatrixPolicy | 4.471103 | 87.4531 | -2.659277 | -1161.8983 |
| Reasoning mix | 100 | RLB MatrixPolicy group-stat | 1.193834 | 3.2997 | -0.399709 | -1.6214 |
| Reasoning mix | 200 | RLB MatrixPolicy | 0.463175 | 1.5891 | -0.149746 | -0.2567 |
| Reasoning mix | 250 | RLB MatrixPolicy | 0.347277 | 1.4152 | -0.081000 | -0.1194 |
| Reasoning mix | 500 | RLB MatrixPolicy group-stat | 0.206193 | 1.2290 | -0.012852 | -0.0159 |
| Reasoning mix | 750 | RLB MatrixPolicy group-stat | 0.167439 | 1.1823 | -0.007802 | -0.0093 |
| Reasoning mix | 1000 | RLB MatrixPolicy group-stat | 0.152561 | 1.1648 | -0.001151 | -0.0013 |
| Reasoning mix | 1250 | RLB MatrixPolicy group-stat | 0.145403 | 1.1565 | -0.000689 | -0.0008 |

| task | best AUC row | step range | AUC loss |
| --- | --- | ---: | ---: |
| Code | RLB MatrixPolicy | 10-1250 | 512.22 |
| Symbolic | RLB MatrixPolicy group-stat | 10-1250 | 335.54 |
| Reasoning mix | RLB MatrixPolicy group-stat | 10-1250 | 698.13 |
