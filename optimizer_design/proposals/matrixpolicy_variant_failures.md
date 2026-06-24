# MatrixPolicy Variant Failure Log

Status: retained negative-result state only. All V2-V12 proposal files, standalone manifests, live optimizer aliases, and raw V-variant run directories were pruned from the active repo surface on 2026-06-23. The paper-facing MatrixPolicy optimizer is still the original `rational_matrix_policy_onpolicy` method with the accepted safe Muon-off implementation-speed fix.

## Summary

| Variant | What was tried | Outcome | Failure reason |
| --- | --- | --- | --- |
| V2 | Early replacement MatrixPolicy branch. | Rejected and removed. | It did not become the paper anchor and no live method evidence is retained; original MatrixPolicy remained the supported design. |
| V3 | Original MatrixPolicy plus partial horizontal gauge projection and a small confidence-gated Muon tail. | Full E1 failed. | No dataset mean improved over original MatrixPolicy; extra late conditioning was not justified. |
| V4 | Functional-balance proxy intended to reallocate input-selector versus output-recombiner step budget from local RLB linearization. | Full E1 near-tied but failed. | The proxy clipped to a constant signal and was mostly centered away, so it did not produce a useful mechanism. |
| V5 | Joint function-space sensitivity metric over `(A_g, B_g)` with preserved role-level scaling. | Full E1 failed. | The A/B reallocation was real but near-constant; it improved only FineWeb-Edu and did not beat original MatrixPolicy robustly. |
| V6 | Proposal-only next candidate. | Deleted before pilot. | It did not clear the design bar for a runnable method. |
| V7 | Secant-trust multiplier from previous accepted matrix updates and current gradients. | P0 failed. | Tiny DCLM pilot loss improvement was not worth the measured runtime penalty. |
| V8 | Fast-pulse MatrixPolicy schedule using the existing optimizer path. | P0 failed. | Shortening the Muon conditioning window worsened loss/AUC and gave no reliable same-node speed win. |
| V9 | Lower Newton-Schulz accuracy inside the original Muon window. | P0 failed. | Reduced logged optimizer-step time slightly, but worsened loss/AUC; the full-quality early matrix direction appears necessary. |
| V10 | Switch-clean reset of matrix Adam state when Muon permanently turns off. | P0 failed. | Hard reset discarded useful diagonal-memory state and caused a large loss/AUC regression. |
| V11 | Late global matrix Adam beta2 decay after the Muon window. | P0 failed. | Faster second-moment adaptation worsened quality and did not give a reliable speed win. |
| V12 | Late beta2 decay only for input-selector matrices. | P0 failed. | Role-selective beta2 change worsened final loss/AUC against fresh controls. |

## Related Activation Ablation

| Ablation | What was tried | Outcome | Failure reason |
| --- | --- | --- | --- |
| Rational-only RLB flawed alias | `rlb_fused_rational_only`: attempted global-only control by setting `centers=()`, but still constructed a zero-sized `coeff_logits` `Parameter`. | Rejected and excluded from ablation aggregates. | This was not the requested no-atom ablation because an atom/coefficient parameter group still existed structurally; rows that completed under this alias showed early nonfinite behavior and the remaining queue was cancelled. |
| Rational-only RLB corrected alias | `rlb_fused_global_rational`: keep the RLB single-branch wrapper and grouped SiLU-fitted global P5/Q4 rational scalar, with no R-local atom path and no atom/coefficient `Parameter`. | Full E1/E2 ablation queued on 2026-06-24. | Pending; this is the corrected ablation definition. |

## Decision

Do not revive V2-V12 or queue E1/E2 work for them. Future MatrixPolicy candidates must start as short paired pilots against the current original MatrixPolicy implementation, pass both quality and runtime gates, and be pruned immediately if they fail. Engineering-only speed work should remain separate from optimizer-method claims.
