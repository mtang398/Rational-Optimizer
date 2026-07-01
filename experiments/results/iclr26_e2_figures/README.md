# ICLR26 E2 Dense Curve Figures

Completed E2 M0/300M datasets: DCLM, FineWeb-Edu, FineWeb, Dolma-sample, and C4. Figures use every native JSONL log point from step 500 through 9150. Validation curves use every 50-step eval; training-loss curves use every 10-step train log. Shaded bands are mean +/- 1 sample std over three seeds.

MatrixPolicy curves use the replacement JSONL rows passed with `--matrixpolicy-manifest`. Non-MatrixPolicy RLB optimizer controls use the `rlb_fused_global_rational` replacement rows passed with `--replacement-manifest`; SiLU controls use the main E2 rows.

Final validation-loss overview across completed E2 datasets. Lower is better; cells are mean +/- sample std over three seeds.

| Method | DCLM final | FineWeb-Edu final | FineWeb final | Dolma-sample final | C4 final |
| --- | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 3.9518 +/- 0.0282 | 3.7015 +/- 0.0212 | 3.9623 +/- 0.0081 | 3.8062 +/- 0.0073 | 3.8777 +/- 0.0144 |
| RLB+AdamW | 4.0496 +/- 0.0298 | 3.8024 +/- 0.0206 | 4.0601 +/- 0.0110 | 3.9045 +/- 0.0082 | 3.9787 +/- 0.0098 |
| SiLU+AdamW | 4.0493 +/- 0.0275 | 3.8035 +/- 0.0182 | 4.0612 +/- 0.0101 | 3.9037 +/- 0.0091 | 3.9811 +/- 0.0128 |
| RLB+Lion | 3.9887 +/- 0.0295 | 3.7416 +/- 0.0214 | 3.9960 +/- 0.0105 | 3.8412 +/- 0.0085 | 3.9132 +/- 0.0139 |
| SiLU+Lion | 3.9934 +/- 0.0230 | 3.7440 +/- 0.0208 | 4.0015 +/- 0.0085 | 3.8475 +/- 0.0094 | 3.9213 +/- 0.0105 |
| RLB+SOAP | 4.0608 +/- 0.0331 | 3.8240 +/- 0.0128 | 4.1081 +/- 0.0320 | 3.9269 +/- 0.0125 | 4.0024 +/- 0.0194 |
| SiLU+SOAP | 4.0964 +/- 0.0300 | 3.8629 +/- 0.0201 | 4.1139 +/- 0.0101 | 3.9568 +/- 0.0093 | 4.0349 +/- 0.0108 |
| RLB+Muon | 3.9913 +/- 0.0260 | 3.7373 +/- 0.0187 | 3.9991 +/- 0.0110 | 3.8478 +/- 0.0047 | 3.9189 +/- 0.0149 |
| SiLU+Muon | 3.9973 +/- 0.0305 | 3.7454 +/- 0.0170 | 4.0066 +/- 0.0128 | 3.8581 +/- 0.0101 | 3.9251 +/- 0.0134 |
| RLB+ScheduleFree | 4.3607 +/- 0.0344 | 4.1421 +/- 0.0217 | 4.3864 +/- 0.0081 | 4.2121 +/- 0.0112 | 4.3084 +/- 0.0071 |
| SiLU+ScheduleFree | 4.3657 +/- 0.0298 | 4.1559 +/- 0.0238 | 4.3979 +/- 0.0106 | 4.2151 +/- 0.0053 | 4.3163 +/- 0.0107 |
| RLB+CAME | 4.4503 +/- 0.0426 | 4.2255 +/- 0.0358 | 4.4804 +/- 0.0064 | 4.2665 +/- 0.0409 | 4.3651 +/- 0.0582 |
| SiLU+CAME | 4.3682 +/- 0.0226 | 4.1503 +/- 0.0211 | 4.4060 +/- 0.0189 | 4.2492 +/- 0.0270 | 4.3298 +/- 0.0148 |
| RLB+ADeMaMix | -- | -- | -- | -- | -- |
| SiLU+ADeMaMix | -- | -- | 1361.4141 +/- 0.0000 (n=1) | -- | -- |

## DCLM

All-method view:

![DCLM E2 validation loss mean +/- std, all methods](dclm_core_validation_loss_mean_std.svg)

![DCLM E2 validation PPL mean +/- std, all methods](dclm_core_validation_ppl_mean_std.svg)

![DCLM E2 training loss mean +/- std, all methods](dclm_core_training_loss_mean_std.svg)

Clean comparison view:

![DCLM E2 validation loss mean +/- std, clean comparison](dclm_clean_validation_loss_mean_std.svg)

![DCLM E2 validation PPL mean +/- std, clean comparison](dclm_clean_validation_ppl_mean_std.svg)

![DCLM E2 training loss mean +/- std, clean comparison](dclm_clean_training_loss_mean_std.svg)

DCLM E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.8531 +/- 0.0471 | 4.4417 +/- 0.0315 | 4.1981 +/- 0.0263 | 4.0556 +/- 0.0289 | 3.9734 +/- 0.0270 | 3.9518 +/- 0.0282 |
| RLB+AdamW | 4.9860 +/- 0.0322 | 4.5477 +/- 0.0292 | 4.2697 +/- 0.0283 | 4.1394 +/- 0.0315 | 4.0676 +/- 0.0299 | 4.0496 +/- 0.0298 |
| SiLU+AdamW | 4.9903 +/- 0.0383 | 4.5538 +/- 0.0272 | 4.2691 +/- 0.0242 | 4.1383 +/- 0.0272 | 4.0667 +/- 0.0268 | 4.0493 +/- 0.0275 |
| RLB+Lion | 4.9103 +/- 0.0427 | 4.4743 +/- 0.0318 | 4.2123 +/- 0.0265 | 4.0805 +/- 0.0294 | 4.0068 +/- 0.0295 | 3.9887 +/- 0.0295 |
| SiLU+Lion | 4.9565 +/- 0.0352 | 4.5005 +/- 0.0231 | 4.2217 +/- 0.0219 | 4.0876 +/- 0.0241 | 4.0126 +/- 0.0228 | 3.9934 +/- 0.0230 |
| RLB+SOAP | 4.9971 +/- 0.0416 | 4.5865 +/- 0.0608 | 4.2984 +/- 0.0078 | 4.1464 +/- 0.0341 | 4.0754 +/- 0.0281 | 4.0608 +/- 0.0331 |
| SiLU+SOAP | 5.1360 +/- 0.0184 | 4.6909 +/- 0.0570 | 4.3574 +/- 0.0355 | 4.1959 +/- 0.0296 | 4.1163 +/- 0.0290 | 4.0964 +/- 0.0300 |
| RLB+Muon | 5.1313 +/- 0.0302 | 4.5676 +/- 0.0317 | 4.2230 +/- 0.0267 | 4.0829 +/- 0.0269 | 4.0088 +/- 0.0266 | 3.9913 +/- 0.0260 |
| SiLU+Muon | 5.1356 +/- 0.0339 | 4.5702 +/- 0.0325 | 4.2298 +/- 0.0306 | 4.0879 +/- 0.0310 | 4.0158 +/- 0.0305 | 3.9973 +/- 0.0305 |
| RLB+ScheduleFree | 5.4438 +/- 0.0343 | 5.0255 +/- 0.0361 | 4.6505 +/- 0.0395 | 4.4681 +/- 0.0363 | 4.3863 +/- 0.0341 | 4.3607 +/- 0.0344 |
| SiLU+ScheduleFree | 5.4545 +/- 0.0291 | 5.0363 +/- 0.0325 | 4.6521 +/- 0.0334 | 4.4730 +/- 0.0313 | 4.3908 +/- 0.0298 | 4.3657 +/- 0.0298 |
| RLB+CAME | 5.5213 +/- 0.0384 | 5.1190 +/- 0.0288 | 4.7592 +/- 0.0273 | 4.5640 +/- 0.0381 | 4.4736 +/- 0.0416 | 4.4503 +/- 0.0426 |
| SiLU+CAME | 5.5228 +/- 0.0322 | 5.1088 +/- 0.0252 | 4.6850 +/- 0.0205 | 4.4791 +/- 0.0221 | 4.3907 +/- 0.0223 | 4.3682 +/- 0.0226 |
| RLB+ADeMaMix | 7522.7668 +/- 12918.9198 | -- | -- | -- | -- | -- |
| SiLU+ADeMaMix | 1709.6780 +/- 585.9580 | 2931159.8867 +/- 4767520.6794 | 11241.5732 +/- 0.0000 (n=1) | 196344.0312 +/- 0.0000 (n=1) | 34853138432.0000 +/- 0.0000 (n=1) | -- |

## FineWeb-Edu

All-method view:

![FineWeb-Edu E2 validation loss mean +/- std, all methods](fineweb_edu_core_validation_loss_mean_std.svg)

![FineWeb-Edu E2 validation PPL mean +/- std, all methods](fineweb_edu_core_validation_ppl_mean_std.svg)

![FineWeb-Edu E2 training loss mean +/- std, all methods](fineweb_edu_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb-Edu E2 validation loss mean +/- std, clean comparison](fineweb_edu_clean_validation_loss_mean_std.svg)

![FineWeb-Edu E2 validation PPL mean +/- std, clean comparison](fineweb_edu_clean_validation_ppl_mean_std.svg)

![FineWeb-Edu E2 training loss mean +/- std, clean comparison](fineweb_edu_clean_training_loss_mean_std.svg)

FineWeb-Edu E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.7042 +/- 0.0163 | 4.2474 +/- 0.0255 | 3.9828 +/- 0.0204 | 3.8213 +/- 0.0208 | 3.7279 +/- 0.0226 | 3.7015 +/- 0.0212 |
| RLB+AdamW | 4.8515 +/- 0.0270 | 4.3616 +/- 0.0263 | 4.0497 +/- 0.0226 | 3.9023 +/- 0.0199 | 3.8238 +/- 0.0209 | 3.8024 +/- 0.0206 |
| SiLU+AdamW | 4.8639 +/- 0.0168 | 4.3679 +/- 0.0200 | 4.0543 +/- 0.0197 | 3.9046 +/- 0.0184 | 3.8255 +/- 0.0186 | 3.8035 +/- 0.0182 |
| RLB+Lion | 4.7597 +/- 0.0237 | 4.2879 +/- 0.0256 | 3.9920 +/- 0.0233 | 3.8432 +/- 0.0209 | 3.7636 +/- 0.0227 | 3.7416 +/- 0.0214 |
| SiLU+Lion | 4.8099 +/- 0.0075 | 4.3046 +/- 0.0220 | 3.9982 +/- 0.0231 | 3.8478 +/- 0.0196 | 3.7659 +/- 0.0215 | 3.7440 +/- 0.0208 |
| RLB+SOAP | 4.8982 +/- 0.0420 | 4.4264 +/- 0.0336 | 4.0819 +/- 0.0121 | 3.9314 +/- 0.0240 | 3.8468 +/- 0.0192 | 3.8240 +/- 0.0128 |
| SiLU+SOAP | 5.0606 +/- 0.0646 | 4.5418 +/- 0.0216 | 4.1519 +/- 0.0262 | 3.9789 +/- 0.0195 | 3.8866 +/- 0.0212 | 3.8629 +/- 0.0201 |
| RLB+Muon | 4.9903 +/- 0.0154 | 4.3571 +/- 0.0185 | 3.9970 +/- 0.0185 | 3.8406 +/- 0.0175 | 3.7586 +/- 0.0184 | 3.7373 +/- 0.0187 |
| SiLU+Muon | 4.9953 +/- 0.0177 | 4.3557 +/- 0.0228 | 4.0009 +/- 0.0183 | 3.8475 +/- 0.0161 | 3.7663 +/- 0.0175 | 3.7454 +/- 0.0170 |
| RLB+ScheduleFree | 5.4615 +/- 0.0119 | 4.9220 +/- 0.0147 | 4.4509 +/- 0.0261 | 4.2553 +/- 0.0232 | 4.1689 +/- 0.0220 | 4.1421 +/- 0.0217 |
| SiLU+ScheduleFree | 5.4788 +/- 0.0185 | 4.9453 +/- 0.0231 | 4.4643 +/- 0.0258 | 4.2687 +/- 0.0247 | 4.1826 +/- 0.0246 | 4.1559 +/- 0.0238 |
| RLB+CAME | 5.5313 +/- 0.0176 | 5.0050 +/- 0.0209 | 4.5633 +/- 0.0268 | 4.3561 +/- 0.0283 | 4.2529 +/- 0.0348 | 4.2255 +/- 0.0358 |
| SiLU+CAME | 5.5271 +/- 0.0210 | 5.0004 +/- 0.0184 | 4.5080 +/- 0.0339 | 4.2755 +/- 0.0247 | 4.1761 +/- 0.0218 | 4.1503 +/- 0.0211 |
| RLB+ADeMaMix | 27698.5969 +/- 47596.7828 | 4962161216.0000 +/- 6009468964.3189 (n=2) | -- | -- | -- | -- |
| SiLU+ADeMaMix | 1388.9467 +/- 438.1649 | 929392.9245 +/- 1048745.5316 | 1370518.8750 +/- 0.0000 (n=1) | -- | -- | -- |

## FineWeb

All-method view:

![FineWeb E2 validation loss mean +/- std, all methods](fineweb_core_validation_loss_mean_std.svg)

![FineWeb E2 validation PPL mean +/- std, all methods](fineweb_core_validation_ppl_mean_std.svg)

![FineWeb E2 training loss mean +/- std, all methods](fineweb_core_training_loss_mean_std.svg)

Clean comparison view:

![FineWeb E2 validation loss mean +/- std, clean comparison](fineweb_clean_validation_loss_mean_std.svg)

![FineWeb E2 validation PPL mean +/- std, clean comparison](fineweb_clean_validation_ppl_mean_std.svg)

![FineWeb E2 training loss mean +/- std, clean comparison](fineweb_clean_training_loss_mean_std.svg)

FineWeb E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.9177 +/- 0.0197 | 4.4861 +/- 0.0084 | 4.2306 +/- 0.0113 | 4.0759 +/- 0.0071 | 3.9864 +/- 0.0088 | 3.9623 +/- 0.0081 |
| RLB+AdamW | 5.0480 +/- 0.0115 | 4.5930 +/- 0.0068 | 4.2973 +/- 0.0115 | 4.1554 +/- 0.0105 | 4.0795 +/- 0.0102 | 4.0601 +/- 0.0110 |
| SiLU+AdamW | 5.0527 +/- 0.0192 | 4.6013 +/- 0.0095 | 4.2990 +/- 0.0101 | 4.1566 +/- 0.0097 | 4.0805 +/- 0.0098 | 4.0612 +/- 0.0101 |
| RLB+Lion | 4.9539 +/- 0.0114 | 4.5179 +/- 0.0112 | 4.2366 +/- 0.0122 | 4.0943 +/- 0.0107 | 4.0161 +/- 0.0103 | 3.9960 +/- 0.0105 |
| SiLU+Lion | 5.0084 +/- 0.0139 | 4.5399 +/- 0.0078 | 4.2458 +/- 0.0089 | 4.1009 +/- 0.0102 | 4.0221 +/- 0.0085 | 4.0015 +/- 0.0085 |
| RLB+SOAP | 5.1606 +/- 0.0137 | 4.6923 +/- 0.0454 | 4.3622 +/- 0.0324 | 4.1973 +/- 0.0154 | 4.1192 +/- 0.0103 | 4.1081 +/- 0.0320 |
| SiLU+SOAP | 5.2775 +/- 0.0437 | 4.7942 +/- 0.1152 | 4.4121 +/- 0.0185 | 4.2201 +/- 0.0089 | 4.1358 +/- 0.0106 | 4.1139 +/- 0.0101 |
| RLB+Muon | 5.1855 +/- 0.0153 | 4.5923 +/- 0.0183 | 4.2471 +/- 0.0138 | 4.0966 +/- 0.0107 | 4.0200 +/- 0.0102 | 3.9991 +/- 0.0110 |
| SiLU+Muon | 5.1938 +/- 0.0130 | 4.5962 +/- 0.0095 | 4.2524 +/- 0.0119 | 4.1027 +/- 0.0121 | 4.0266 +/- 0.0117 | 4.0066 +/- 0.0128 |
| RLB+ScheduleFree | 5.5567 +/- 0.0163 | 5.1058 +/- 0.0172 | 4.6764 +/- 0.0084 | 4.4933 +/- 0.0078 | 4.4117 +/- 0.0080 | 4.3864 +/- 0.0081 |
| SiLU+ScheduleFree | 5.5684 +/- 0.0117 | 5.1193 +/- 0.0137 | 4.6901 +/- 0.0124 | 4.5060 +/- 0.0105 | 4.4236 +/- 0.0103 | 4.3979 +/- 0.0106 |
| RLB+CAME | 5.6293 +/- 0.0197 | 5.1998 +/- 0.0126 | 4.8045 +/- 0.0092 | 4.6015 +/- 0.0077 | 4.5053 +/- 0.0061 | 4.4804 +/- 0.0064 |
| SiLU+CAME | 5.6395 +/- 0.0158 | 5.1935 +/- 0.0228 | 4.7432 +/- 0.0503 | 4.5220 +/- 0.0254 | 4.4301 +/- 0.0193 | 4.4060 +/- 0.0189 |
| RLB+ADeMaMix | 257.9755 +/- 250.1753 | 1633877367.1458 +/- 2533289655.0567 | -- | -- | -- | -- |
| SiLU+ADeMaMix | 1046.3267 +/- 385.1340 | 386.1834 +/- 120.1352 | 18508.4055 +/- 25038.3317 (n=2) | 1100.8079 +/- 0.0000 (n=1) | 2696.9573 +/- 0.0000 (n=1) | 1361.4141 +/- 0.0000 (n=1) |

## Dolma-sample

All-method view:

![Dolma-sample E2 validation loss mean +/- std, all methods](dolma_sample_core_validation_loss_mean_std.svg)

![Dolma-sample E2 validation PPL mean +/- std, all methods](dolma_sample_core_validation_ppl_mean_std.svg)

![Dolma-sample E2 training loss mean +/- std, all methods](dolma_sample_core_training_loss_mean_std.svg)

Clean comparison view:

![Dolma-sample E2 validation loss mean +/- std, clean comparison](dolma_sample_clean_validation_loss_mean_std.svg)

![Dolma-sample E2 validation PPL mean +/- std, clean comparison](dolma_sample_clean_validation_ppl_mean_std.svg)

![Dolma-sample E2 training loss mean +/- std, clean comparison](dolma_sample_clean_training_loss_mean_std.svg)

Dolma-sample E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.7148 +/- 0.0094 | 4.3132 +/- 0.0063 | 4.0684 +/- 0.0081 | 3.9171 +/- 0.0089 | 3.8292 +/- 0.0070 | 3.8062 +/- 0.0073 |
| RLB+AdamW | 4.8475 +/- 0.0112 | 4.4228 +/- 0.0070 | 4.1366 +/- 0.0099 | 3.9964 +/- 0.0087 | 3.9231 +/- 0.0078 | 3.9045 +/- 0.0082 |
| SiLU+AdamW | 4.8563 +/- 0.0014 | 4.4240 +/- 0.0099 | 4.1350 +/- 0.0106 | 3.9963 +/- 0.0097 | 3.9227 +/- 0.0091 | 3.9037 +/- 0.0091 |
| RLB+Lion | 4.7624 +/- 0.0186 | 4.3478 +/- 0.0144 | 4.0769 +/- 0.0121 | 3.9361 +/- 0.0106 | 3.8607 +/- 0.0089 | 3.8412 +/- 0.0085 |
| SiLU+Lion | 4.8041 +/- 0.0041 | 4.3710 +/- 0.0090 | 4.0847 +/- 0.0104 | 3.9425 +/- 0.0101 | 3.8670 +/- 0.0094 | 3.8475 +/- 0.0094 |
| RLB+SOAP | 4.8764 +/- 0.0334 | 4.4635 +/- 0.0252 | 4.2149 +/- 0.0260 | 4.0424 +/- 0.0406 | 3.9457 +/- 0.0099 | 3.9269 +/- 0.0125 |
| SiLU+SOAP | 5.0463 +/- 0.0622 | 4.5589 +/- 0.0265 | 4.2314 +/- 0.0103 | 4.0647 +/- 0.0121 | 3.9780 +/- 0.0081 | 3.9568 +/- 0.0093 |
| RLB+Muon | 4.9914 +/- 0.0078 | 4.4326 +/- 0.0103 | 4.0917 +/- 0.0076 | 3.9430 +/- 0.0047 | 3.8677 +/- 0.0046 | 3.8478 +/- 0.0047 |
| SiLU+Muon | 4.9932 +/- 0.0054 | 4.4333 +/- 0.0060 | 4.1013 +/- 0.0120 | 3.9535 +/- 0.0103 | 3.8776 +/- 0.0095 | 3.8581 +/- 0.0101 |
| RLB+ScheduleFree | 5.3440 +/- 0.0087 | 4.8938 +/- 0.0061 | 4.4897 +/- 0.0102 | 4.3132 +/- 0.0109 | 4.2360 +/- 0.0112 | 4.2121 +/- 0.0112 |
| SiLU+ScheduleFree | 5.3512 +/- 0.0087 | 4.8985 +/- 0.0056 | 4.4895 +/- 0.0060 | 4.3161 +/- 0.0042 | 4.2391 +/- 0.0055 | 4.2151 +/- 0.0053 |
| RLB+CAME | 5.4015 +/- 0.0110 | 4.9762 +/- 0.0145 | 4.5928 +/- 0.0262 | 4.3822 +/- 0.0414 | 4.2907 +/- 0.0422 | 4.2665 +/- 0.0409 |
| SiLU+CAME | 5.4110 +/- 0.0084 | 4.9728 +/- 0.0149 | 4.5654 +/- 0.0444 | 4.3689 +/- 0.0414 | 4.2734 +/- 0.0294 | 4.2492 +/- 0.0270 |
| RLB+ADeMaMix | 28371.7812 +/- 2941.2476 | 2482927104.0000 +/- 2466098821.8411 (n=2) | -- | -- | -- | -- |
| SiLU+ADeMaMix | 1127.4188 +/- 977.4469 | 1608323.0966 +/- 2563013.6932 | 70147874816.0000 +/- 0.0000 (n=1) | -- | -- | -- |

## C4

All-method view:

![C4 E2 validation loss mean +/- std, all methods](c4_en_core_validation_loss_mean_std.svg)

![C4 E2 validation PPL mean +/- std, all methods](c4_en_core_validation_ppl_mean_std.svg)

![C4 E2 training loss mean +/- std, all methods](c4_en_core_training_loss_mean_std.svg)

Clean comparison view:

![C4 E2 validation loss mean +/- std, clean comparison](c4_en_clean_validation_loss_mean_std.svg)

![C4 E2 validation PPL mean +/- std, clean comparison](c4_en_clean_validation_ppl_mean_std.svg)

![C4 E2 training loss mean +/- std, clean comparison](c4_en_clean_training_loss_mean_std.svg)

C4 E2 validation-loss checkpoint table, mean +/- sample std:

| Method | 1000 | 2000 | 4000 | 6000 | 8000 | 9150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MatrixPolicy | 4.8588 +/- 0.0133 | 4.4135 +/- 0.0141 | 4.1521 +/- 0.0120 | 3.9952 +/- 0.0160 | 3.9027 +/- 0.0136 | 3.8777 +/- 0.0144 |
| RLB+AdamW | 5.0046 +/- 0.0293 | 4.5251 +/- 0.0107 | 4.2188 +/- 0.0097 | 4.0755 +/- 0.0115 | 3.9980 +/- 0.0101 | 3.9787 +/- 0.0098 |
| SiLU+AdamW | 5.0066 +/- 0.0246 | 4.5330 +/- 0.0144 | 4.2225 +/- 0.0137 | 4.0777 +/- 0.0133 | 4.0005 +/- 0.0114 | 3.9811 +/- 0.0128 |
| RLB+Lion | 4.8989 +/- 0.0205 | 4.4407 +/- 0.0113 | 4.1554 +/- 0.0124 | 4.0137 +/- 0.0148 | 3.9327 +/- 0.0132 | 3.9132 +/- 0.0139 |
| SiLU+Lion | 4.9530 +/- 0.0226 | 4.4680 +/- 0.0107 | 4.1678 +/- 0.0108 | 4.0211 +/- 0.0103 | 3.9407 +/- 0.0113 | 3.9213 +/- 0.0105 |
| RLB+SOAP | 5.1604 +/- 0.2205 | 4.5515 +/- 0.0133 | 4.2526 +/- 0.0263 | 4.1011 +/- 0.0219 | 4.0193 +/- 0.0176 | 4.0024 +/- 0.0194 |
| SiLU+SOAP | 5.1948 +/- 0.0604 | 4.6889 +/- 0.0405 | 4.3065 +/- 0.0112 | 4.1452 +/- 0.0118 | 4.0716 +/- 0.0224 | 4.0349 +/- 0.0108 |
| RLB+Muon | 5.1513 +/- 0.0273 | 4.5259 +/- 0.0109 | 4.1730 +/- 0.0110 | 4.0181 +/- 0.0146 | 3.9381 +/- 0.0140 | 3.9189 +/- 0.0149 |
| SiLU+Muon | 5.1609 +/- 0.0341 | 4.5248 +/- 0.0117 | 4.1777 +/- 0.0099 | 4.0254 +/- 0.0140 | 3.9448 +/- 0.0125 | 3.9251 +/- 0.0134 |
| RLB+ScheduleFree | 5.5390 +/- 0.0303 | 5.0615 +/- 0.0202 | 4.6063 +/- 0.0090 | 4.4169 +/- 0.0077 | 4.3340 +/- 0.0075 | 4.3084 +/- 0.0071 |
| SiLU+ScheduleFree | 5.5598 +/- 0.0344 | 5.0728 +/- 0.0242 | 4.6171 +/- 0.0131 | 4.4261 +/- 0.0100 | 4.3422 +/- 0.0104 | 4.3163 +/- 0.0107 |
| RLB+CAME | 5.6076 +/- 0.0244 | 5.1431 +/- 0.0247 | 4.7251 +/- 0.0416 | 4.4954 +/- 0.0600 | 4.3902 +/- 0.0582 | 4.3651 +/- 0.0582 |
| SiLU+CAME | 5.6174 +/- 0.0280 | 5.1482 +/- 0.0327 | 4.6857 +/- 0.0411 | 4.4522 +/- 0.0187 | 4.3543 +/- 0.0153 | 4.3298 +/- 0.0148 |
| RLB+ADeMaMix | 414.2971 +/- 77.7608 | 553095678.5990 +/- 943240014.0699 | 1687480172544.0000 +/- 0.0000 (n=1) | -- | -- | -- |
| SiLU+ADeMaMix | 1104.9656 +/- 565.1435 | 180791.1562 +/- 0.0000 (n=1) | -- | -- | -- | -- |
