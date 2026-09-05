# Current historical-M1 / 300M-token RLB optimizer candidate registry

This is the only candidate namespace for the exact historical-M1,
300M-token, 4,000-step design cycle.
Identifiers are deliberately opaque so their algorithms must be read from
their frozen source and derivation rather than inferred from names.

## Controls

| ID | Architecture | Optimizer | State |
| --- | --- | --- | --- |
| `U-S` | exact M1 SwiGLU (296,867,840 params) | Muon | complete; stronger hurdle, loss `4.2284669876` |
| `U-R` | exact M1 Global-RLB (296,871,080 params) | Muon | complete; loss `4.2416791916` |

Both controls use the exact settings and cache hashes in
`optimizer_design/RLB_300M_4000_DESIGN_CONTRACT.md`. The stronger endpoint is
the hurdle for every candidate.

## Ten-slot design pool

| Slot | State | Frozen source SHA-256 | Mechanism fingerprint | 4,000-step job | Decision |
| --- | --- | --- | --- | ---: | --- |
| `R01` | rejected; artifacts deleted | wrapper `15585388…7d20` | tombstone `e992c1a9…b7d8` | `684706_0` | loss `4.2457833290`; rejected |
| `R02` | rejected; artifacts deleted | wrapper `2cafe909…11f3` | tombstone `b0cbc6d0…57e3` | `684706_1` | loss `4.2511992455`; rejected |
| `R03` | rejected; artifacts deleted | `fafbecf5…3262` | tombstone `6d6821b3…42ec` | `686652_0` | loss `4.2371015549`; rejected |
| `R04` | empty after permanent deletion | `c853fbec…4863` | tombstone `724b37ff…50ef` | `686652_1` | old mechanism permanently retired; no replacement assigned |
| `R05` | empty after permanent deletion | `ea6063d0…4bf4` | tombstone `f3351be5…cc75` | `688369_0` | loss `4.2382307053`; rejected |
| `R06` | empty after permanent deletion | `94094e06…9db7` | tombstone `355881e5…d9b` | `688369_1` | loss `4.2405834198`; rejected |
| `R07` | running | `d37cf5ce…e60f` | groupwise homogeneous-limit matrix-pair quotient lift | `693878_0` | full endpoint pending |
| `R08` | running | `39a66c8c…1a03` | nested local/global polar on RLB groups | `699193_1` | full endpoint pending |
| `R09` | queued | `52c2674a…774b` | coordinate-trust RLB matrix quotient | `699196_0` after R07/R08 analysis | full endpoint pending |
| `R10` | queued | `82d1f75e…558e` | coordinate trust on the complete two-gauge RLB quotient | `699196_1` after R07/R08 analysis | full endpoint pending |

The live pool never exceeds these ten opaque slots. Candidate source,
derivation, launcher row, tests, and result directory use the same slot. A
minor implementation repair overwrites the current generation and preserves
its equations. A promising complete near-miss may be revised in place after
literature-grounded diagnosis; the revision receives a new source hash,
preregistration, audit, and full 4,000-step run. A materially bad generation is
deleted and its slot may be reassigned to a genuinely different mechanism.
Every displaced generation remains represented by a compact hash tombstone,
and no old metric or ablation evidence transfers to the replacement.

## State transitions

`reserved -> preregistered -> audited -> queued -> running -> complete`

A complete candidate moves to one of these evidence states:

- `promoted`: clears endpoint loss, PPL, late partial-AUC, structural, and
  fairness gates and enters recursive full-4,000-step ablation;
- `component-qualified`: beats the stronger control by at least `0.01` in
  endpoint loss and also wins endpoint PPL and late partial-AUC, so it may
  enter the same recursive component testing even if it is not yet a final
  method;
- `rejected`: misses any required gate; all method artifacts are deleted and
  only the compact tombstone below remains.

Early checkpoints never cause a scientific state transition. Crashes and
systems defects are repaired in place with unchanged mathematics.

## Terminal rejection tombstones

| Slot | Source SHA-256 | Mechanism fingerprint | Loss / PPL / train-pAUC / val-pAUC | Reason |
| --- | --- | --- | --- | --- |
| `R01` | `15585388b88a9b9c37758a928100da7d72ad1b27df1906bf0553f087aeb87d20` | `e992c1a99cfe1d3af95bb8a56ab52c1c1d061b833dd70100282794dc5cc9b7d8` | `4.2457833290 / 69.8104232612 / 4.2240088899 / 4.4454157432` | Worse than `U-S` by `0.0173163414` loss; PPL and late val-pAUC also worse. |
| `R02` | `2cafe909382f803438cb54ef8d817a4bbdf28642174efc4948e08feb018911f3` | `b0cbc6d0ff08e845e26f0e47c5f3f8d62f5341afaf1d7ce827d7f49e40a357e3` | `4.2511992455 / 70.1895363763 / 4.2370392184 / 4.4555290182` | Worse than `U-S` by `0.0227322578` loss; PPL and late val-pAUC also worse. |
| `R03` | `fafbecf50098b951c440392d0f29818be73f4f7aa3dc114b5c5b72b64c453262` | `6d6821b3272225c68374dcc9da714fb1f70c2424f2eb56c337bc0927c48342ec` | `4.2371015549 / 69.2069682543 / 4.2018370397 / 4.4293163260` | Worse than `U-S` by `0.0086345673` loss; PPL and late val-pAUC also worse. |
| `R04` | `c853fbecfdc7be279dfb358f35cf7d6ed4a24ff896f6b93982b2ad785c364863` | `724b37ffb393ca451f39a20e0b205040ed87c7d286770e126582eeaca2a450ef` | `4.2855243683 / 72.6406271760 / 4.2566789908 / 4.4771397670` | Worse than `U-S` by `0.0570573807` loss; PPL and late val-pAUC also worse. |
| `R05` | `ea6063d02376c2bfd96efefdc28ad66bfcde4d5ba6b927cd8bb10cc57aee4bf4` | `f3351be54f5eee49a425dd11a09b2c4da7b9ae2a5ff62620cacc668f178fcc75` | `4.2382307053 / 69.2851574649 / 4.2034461000 / 4.4302928686` | Worse than `U-S` by `0.0097637177` loss; PPL and late val-pAUC also worse. |
| `R06` | `94094e06af2c10b17dde3f2d3ded2aa1eb8ab764debcc606bc4335fa38859db7` | `355881e5f506a8d91ecfb289b2cefad28c0be652ee5f4c1b15d527698d107d9b` | `4.2405834198 / 69.4483575686 / 4.2057170724 / 4.4316871087` | Worse than `U-S` by `0.0121164322` loss; PPL and late val-pAUC also worse. |

The mechanism fingerprint is a compact hash of the mathematical component
list. It prevents a rejected method from returning under another identifier or
a later generation of the same slot without retaining its source or artifacts.

## Finalists

Exactly three rows may be filled. Each requires a `>=0.20` final-loss lead and
strict PPL and late partial-AUC wins over the stronger fresh Muon control,
recursive leave-one-component-out closure, and final runtime qualification.

| Final method | Parent slot | Closed source SHA-256 | Components | Evidence |
| --- | --- | --- | --- | --- |
| — | — | — | — | — |
| — | — | — | — | — |
| — | — | — | — | — |
