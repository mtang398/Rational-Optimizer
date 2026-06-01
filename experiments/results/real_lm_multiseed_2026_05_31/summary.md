# Real-LM Multi-Seed Summary

Positive gaps mean the method has lower validation loss than the comparison row.

## FineWeb

| method | n | div | mean loss | std | mean PPL | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SiLU+AdamW | 3 | 0 | 4.528963 | 0.029611 | 92.69 | 0.000000 | -0.006960 |
| RLB+AdamW | 3 | 0 | 4.522311 | 0.029832 | 92.08 | 0.006653 | -0.000308 |
| SiLU+Muon | 3 | 0 | 4.566661 | 0.041469 | 96.28 | -0.037698 | -0.044658 |
| RLB+Muon | 3 | 0 | 4.571341 | 0.027720 | 96.70 | -0.042377 | -0.049337 |
| RLB+MatrixPolicy (group-stat) | 3 | 0 | 4.369701 | 0.026358 | 79.04 | 0.159263 | 0.152302 |

| seed | method | status | final step | val loss | PPL | gap vs SiLU+AdamW |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1337 | SiLU+AdamW | complete | 3050 | 4.504617 | 90.43 | 0.000000 |
| 1337 | RLB+AdamW | complete | 3050 | 4.493013 | 89.39 | 0.011604 |
| 1337 | SiLU+Muon | complete | 3050 | 4.535766 | 93.29 | -0.031149 |
| 1337 | RLB+Muon | complete | 3050 | 4.548868 | 94.53 | -0.044251 |
| 1337 | RLB+MatrixPolicy (group-stat) | complete | 3050 | 4.344150 | 77.03 | 0.160467 |
| 2027 | SiLU+AdamW | complete | 3050 | 4.520346 | 91.87 | 0.000000 |
| 2027 | RLB+AdamW | complete | 3050 | 4.521269 | 91.95 | -0.000923 |
| 2027 | SiLU+Muon | complete | 3050 | 4.550426 | 94.67 | -0.030079 |
| 2027 | RLB+Muon | complete | 3050 | 4.562838 | 95.86 | -0.042491 |
| 2027 | RLB+MatrixPolicy (group-stat) | complete | 3050 | 4.368154 | 78.90 | 0.152192 |
| 3407 | SiLU+AdamW | complete | 3050 | 4.561927 | 95.77 | 0.000000 |
| 3407 | RLB+AdamW | complete | 3050 | 4.552650 | 94.88 | 0.009277 |
| 3407 | SiLU+Muon | complete | 3050 | 4.613791 | 100.87 | -0.051864 |
| 3407 | RLB+Muon | complete | 3050 | 4.602316 | 99.71 | -0.040389 |
| 3407 | RLB+MatrixPolicy (group-stat) | complete | 3050 | 4.396799 | 81.19 | 0.165129 |

## FineWeb-Edu

| method | n | div | mean loss | std | mean PPL | gap vs SiLU+AdamW | gap vs best control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SiLU+AdamW | 3 | 0 | 4.223572 | 0.001635 | 68.28 | 0.000000 | -0.000748 |
| RLB+AdamW | 3 | 1 | 5.618928 | 2.418773 | 1545.54 | -1.395356 | -1.396103 |
| SiLU+Muon | 3 | 0 | 4.258871 | 0.014706 | 70.74 | -0.035300 | -0.036047 |
| RLB+Muon | 3 | 0 | 4.263744 | 0.008026 | 71.08 | -0.040173 | -0.040920 |
| RLB+MatrixPolicy (group-stat) | 3 | 0 | 4.069422 | 0.002281 | 58.52 | 0.154149 | 0.153402 |

| seed | method | status | final step | val loss | PPL | gap vs SiLU+AdamW |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1337 | SiLU+AdamW | complete | 3050 | 4.225019 | 68.38 | 0.000000 |
| 1337 | RLB+AdamW | diverged | 50 | 8.411884 | 4500.23 | -4.186865 |
| 1337 | SiLU+Muon | complete | 3050 | 4.252612 | 70.29 | -0.027593 |
| 1337 | RLB+Muon | complete | 3050 | 4.271556 | 71.63 | -0.046537 |
| 1337 | RLB+MatrixPolicy (group-stat) | complete | 3050 | 4.072055 | 58.68 | 0.152964 |
| 2027 | SiLU+AdamW | complete | 3050 | 4.223898 | 68.30 | 0.000000 |
| 2027 | RLB+AdamW | complete | 3050 | 4.225344 | 68.40 | -0.001445 |
| 2027 | SiLU+Muon | complete | 3050 | 4.248330 | 69.99 | -0.024432 |
| 2027 | RLB+Muon | complete | 3050 | 4.255519 | 70.49 | -0.031621 |
| 2027 | RLB+MatrixPolicy (group-stat) | complete | 3050 | 4.068038 | 58.44 | 0.155860 |
| 3407 | SiLU+AdamW | complete | 3050 | 4.221798 | 68.16 | 0.000000 |
| 3407 | RLB+AdamW | complete | 3050 | 4.219555 | 68.00 | 0.002243 |
| 3407 | SiLU+Muon | complete | 3050 | 4.275672 | 71.93 | -0.053874 |
| 3407 | RLB+Muon | complete | 3050 | 4.264158 | 71.11 | -0.042360 |
| 3407 | RLB+MatrixPolicy (group-stat) | complete | 3050 | 4.068174 | 58.45 | 0.153624 |
