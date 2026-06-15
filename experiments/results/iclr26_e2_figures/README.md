# ICLR26 E2 Dense Curve Figures

Completed E2 M0/300M datasets: DCLM, FineWeb-Edu, and FineWeb. Figures use every native JSONL log point from step 500 through 9150. Validation curves use every 50-step eval; training-loss curves use every 10-step train log. Shaded bands are mean +/- 1 sample std over three seeds.

Final validation-loss overview across completed E2 datasets. Lower is better; cells are mean +/- sample std over three seeds.

| Method | DCLM final | FineWeb-Edu final | FineWeb final |
| --- | ---: | ---: | ---: |
| MatrixPolicy | 3.9576 +/- 0.0307 | 3.7065 +/- 0.0203 | 3.9656 +/- 0.0085 |
| RLB+AdamW | 4.0529 +/- 0.0282 | 3.8069 +/- 0.0176 | 4.0629 +/- 0.0098 |
| SiLU+AdamW | 4.0493 +/- 0.0275 | 3.8035 +/- 0.0182 | 4.0612 +/- 0.0101 |
| RLB+Lion | 3.9943 +/- 0.0301 | 3.7451 +/- 0.0214 | 4.0014 +/- 0.0128 |
| SiLU+Lion | 3.9934 +/- 0.0230 | 3.7440 +/- 0.0208 | 4.0015 +/- 0.0085 |
| RLB+SOAP | 4.0768 +/- 0.0403 | 3.8301 +/- 0.0199 | 4.0841 +/- 0.0079 |
| SiLU+SOAP | 4.0964 +/- 0.0300 | 3.8629 +/- 0.0201 | 4.1139 +/- 0.0101 |
| RLB+Muon | 3.9935 +/- 0.0296 | 3.7382 +/- 0.0210 | 4.0012 +/- 0.0114 |
| SiLU+Muon | 3.9973 +/- 0.0305 | 3.7454 +/- 0.0170 | 4.0066 +/- 0.0128 |
| RLB+ScheduleFree | 4.3563 +/- 0.0332 | 4.1365 +/- 0.0217 | 4.3814 +/- 0.0093 |
| SiLU+ScheduleFree | 4.3657 +/- 0.0298 | 4.1559 +/- 0.0238 | 4.3979 +/- 0.0106 |
| RLB+CAME | 4.4503 +/- 0.0340 | 4.2203 +/- 0.0360 | 4.4732 +/- 0.0011 |
| SiLU+CAME | 4.3682 +/- 0.0226 | 4.1503 +/- 0.0211 | 4.4060 +/- 0.0189 |
| RLB+ADeMaMix | -- | -- | -- |
| SiLU+ADeMaMix | -- | -- | 1361.4141 +/- 0.0000 (n=1) |

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
| MatrixPolicy | 4.8513 +/- 0.0460 | 4.4392 +/- 0.0320 | 4.2030 +/- 0.0281 | 4.0608 +/- 0.0309 | 3.9788 +/- 0.0298 | 3.9576 +/- 0.0307 |
| RLB+AdamW | 4.9827 +/- 0.0317 | 4.5493 +/- 0.0274 | 4.2708 +/- 0.0276 | 4.1410 +/- 0.0304 | 4.0705 +/- 0.0278 | 4.0529 +/- 0.0282 |
| SiLU+AdamW | 4.9903 +/- 0.0383 | 4.5538 +/- 0.0272 | 4.2691 +/- 0.0242 | 4.1383 +/- 0.0272 | 4.0667 +/- 0.0268 | 4.0493 +/- 0.0275 |
| RLB+Lion | 4.9162 +/- 0.0369 | 4.4816 +/- 0.0324 | 4.2186 +/- 0.0282 | 4.0858 +/- 0.0307 | 4.0130 +/- 0.0306 | 3.9943 +/- 0.0301 |
| SiLU+Lion | 4.9565 +/- 0.0352 | 4.5005 +/- 0.0231 | 4.2217 +/- 0.0219 | 4.0876 +/- 0.0241 | 4.0126 +/- 0.0228 | 3.9934 +/- 0.0230 |
| RLB+SOAP | 5.0873 +/- 0.1169 | 4.6116 +/- 0.0673 | 4.3194 +/- 0.0553 | 4.1703 +/- 0.0414 | 4.0963 +/- 0.0407 | 4.0768 +/- 0.0403 |
| SiLU+SOAP | 5.1360 +/- 0.0184 | 4.6909 +/- 0.0570 | 4.3574 +/- 0.0355 | 4.1959 +/- 0.0296 | 4.1163 +/- 0.0290 | 4.0964 +/- 0.0300 |
| RLB+Muon | 5.1264 +/- 0.0284 | 4.5681 +/- 0.0266 | 4.2253 +/- 0.0291 | 4.0836 +/- 0.0301 | 4.0104 +/- 0.0297 | 3.9935 +/- 0.0296 |
| SiLU+Muon | 5.1356 +/- 0.0339 | 4.5702 +/- 0.0325 | 4.2298 +/- 0.0306 | 4.0879 +/- 0.0310 | 4.0158 +/- 0.0305 | 3.9973 +/- 0.0305 |
| RLB+ScheduleFree | 5.4308 +/- 0.0313 | 5.0115 +/- 0.0331 | 4.6380 +/- 0.0371 | 4.4610 +/- 0.0349 | 4.3814 +/- 0.0327 | 4.3563 +/- 0.0332 |
| SiLU+ScheduleFree | 5.4545 +/- 0.0291 | 5.0363 +/- 0.0325 | 4.6521 +/- 0.0334 | 4.4730 +/- 0.0313 | 4.3908 +/- 0.0298 | 4.3657 +/- 0.0298 |
| RLB+CAME | 5.5176 +/- 0.0411 | 5.1128 +/- 0.0298 | 4.7533 +/- 0.0260 | 4.5664 +/- 0.0297 | 4.4742 +/- 0.0332 | 4.4503 +/- 0.0340 |
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
| MatrixPolicy | 4.6982 +/- 0.0128 | 4.2435 +/- 0.0229 | 3.9881 +/- 0.0220 | 3.8257 +/- 0.0198 | 3.7333 +/- 0.0201 | 3.7065 +/- 0.0203 |
| RLB+AdamW | 4.8421 +/- 0.0247 | 4.3623 +/- 0.0256 | 4.0533 +/- 0.0217 | 3.9064 +/- 0.0178 | 3.8278 +/- 0.0189 | 3.8069 +/- 0.0176 |
| SiLU+AdamW | 4.8639 +/- 0.0168 | 4.3679 +/- 0.0200 | 4.0543 +/- 0.0197 | 3.9046 +/- 0.0184 | 3.8255 +/- 0.0186 | 3.8035 +/- 0.0182 |
| RLB+Lion | 4.7857 +/- 0.0153 | 4.2970 +/- 0.0278 | 3.9981 +/- 0.0250 | 3.8475 +/- 0.0214 | 3.7664 +/- 0.0217 | 3.7451 +/- 0.0214 |
| SiLU+Lion | 4.8099 +/- 0.0075 | 4.3046 +/- 0.0220 | 3.9982 +/- 0.0231 | 3.8478 +/- 0.0196 | 3.7659 +/- 0.0215 | 3.7440 +/- 0.0208 |
| RLB+SOAP | 4.8854 +/- 0.0289 | 5.1550 +/- 1.2456 | 4.0786 +/- 0.0068 | 5.7475 +/- 3.1597 | 3.8498 +/- 0.0276 | 3.8301 +/- 0.0199 |
| SiLU+SOAP | 5.0606 +/- 0.0646 | 4.5418 +/- 0.0216 | 4.1519 +/- 0.0262 | 3.9789 +/- 0.0195 | 3.8866 +/- 0.0212 | 3.8629 +/- 0.0201 |
| RLB+Muon | 4.9853 +/- 0.0142 | 4.3609 +/- 0.0198 | 3.9965 +/- 0.0210 | 3.8409 +/- 0.0206 | 3.7600 +/- 0.0216 | 3.7382 +/- 0.0210 |
| SiLU+Muon | 4.9953 +/- 0.0177 | 4.3557 +/- 0.0228 | 4.0009 +/- 0.0183 | 3.8475 +/- 0.0161 | 3.7663 +/- 0.0175 | 3.7454 +/- 0.0170 |
| RLB+ScheduleFree | 5.4400 +/- 0.0107 | 4.9005 +/- 0.0164 | 4.4358 +/- 0.0273 | 4.2460 +/- 0.0229 | 4.1624 +/- 0.0223 | 4.1365 +/- 0.0217 |
| SiLU+ScheduleFree | 5.4788 +/- 0.0185 | 4.9453 +/- 0.0231 | 4.4643 +/- 0.0258 | 4.2687 +/- 0.0247 | 4.1826 +/- 0.0246 | 4.1559 +/- 0.0238 |
| RLB+CAME | 5.5138 +/- 0.0158 | 4.9893 +/- 0.0190 | 4.5507 +/- 0.0308 | 4.3457 +/- 0.0328 | 4.2470 +/- 0.0364 | 4.2203 +/- 0.0360 |
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
| MatrixPolicy | 4.9061 +/- 0.0174 | 4.4828 +/- 0.0067 | 4.2348 +/- 0.0096 | 4.0799 +/- 0.0072 | 3.9891 +/- 0.0084 | 3.9656 +/- 0.0085 |
| RLB+AdamW | 5.0393 +/- 0.0164 | 4.5920 +/- 0.0064 | 4.2998 +/- 0.0082 | 4.1575 +/- 0.0082 | 4.0815 +/- 0.0090 | 4.0629 +/- 0.0098 |
| SiLU+AdamW | 5.0527 +/- 0.0192 | 4.6013 +/- 0.0095 | 4.2990 +/- 0.0101 | 4.1566 +/- 0.0097 | 4.0805 +/- 0.0098 | 4.0612 +/- 0.0101 |
| RLB+Lion | 4.9617 +/- 0.0095 | 4.5227 +/- 0.0115 | 4.2419 +/- 0.0147 | 4.0992 +/- 0.0131 | 4.0210 +/- 0.0131 | 4.0014 +/- 0.0128 |
| SiLU+Lion | 5.0084 +/- 0.0139 | 4.5399 +/- 0.0078 | 4.2458 +/- 0.0089 | 4.1009 +/- 0.0102 | 4.0221 +/- 0.0085 | 4.0015 +/- 0.0085 |
| RLB+SOAP | 5.1011 +/- 0.0489 | 4.6501 +/- 0.0149 | 4.3392 +/- 0.0139 | 4.1790 +/- 0.0052 | 4.1141 +/- 0.0255 | 4.0841 +/- 0.0079 |
| SiLU+SOAP | 5.2775 +/- 0.0437 | 4.7942 +/- 0.1152 | 4.4121 +/- 0.0185 | 4.2201 +/- 0.0089 | 4.1358 +/- 0.0106 | 4.1139 +/- 0.0101 |
| RLB+Muon | 5.1846 +/- 0.0121 | 4.5976 +/- 0.0155 | 4.2490 +/- 0.0133 | 4.0981 +/- 0.0112 | 4.0219 +/- 0.0106 | 4.0012 +/- 0.0114 |
| SiLU+Muon | 5.1938 +/- 0.0130 | 4.5962 +/- 0.0095 | 4.2524 +/- 0.0119 | 4.1027 +/- 0.0121 | 4.0266 +/- 0.0117 | 4.0066 +/- 0.0128 |
| RLB+ScheduleFree | 5.5429 +/- 0.0168 | 5.0898 +/- 0.0162 | 4.6658 +/- 0.0107 | 4.4862 +/- 0.0090 | 4.4064 +/- 0.0091 | 4.3814 +/- 0.0093 |
| SiLU+ScheduleFree | 5.5684 +/- 0.0117 | 5.1193 +/- 0.0137 | 4.6901 +/- 0.0124 | 4.5060 +/- 0.0105 | 4.4236 +/- 0.0103 | 4.3979 +/- 0.0106 |
| RLB+CAME | 5.6199 +/- 0.0164 | 5.1868 +/- 0.0100 | 4.7914 +/- 0.0037 | 4.5916 +/- 0.0004 | 4.4979 +/- 0.0010 | 4.4732 +/- 0.0011 |
| SiLU+CAME | 5.6395 +/- 0.0158 | 5.1935 +/- 0.0228 | 4.7432 +/- 0.0503 | 4.5220 +/- 0.0254 | 4.4301 +/- 0.0193 | 4.4060 +/- 0.0189 |
| RLB+ADeMaMix | 257.9755 +/- 250.1753 | 1633877367.1458 +/- 2533289655.0567 | -- | -- | -- | -- |
| SiLU+ADeMaMix | 1046.3267 +/- 385.1340 | 386.1834 +/- 120.1352 | 18508.4055 +/- 25038.3317 (n=2) | 1100.8079 +/- 0.0000 (n=1) | 2696.9573 +/- 0.0000 (n=1) | 1361.4141 +/- 0.0000 (n=1) |
