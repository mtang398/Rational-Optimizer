# RLB optimizer design contract: historical M1, 300M-token pool, 4,000 steps

This file is the fail-closed protocol for the current optimizer-design cycle.
It supersedes candidate-selection evidence obtained from smaller source pools.
Those candidate artifacts have been removed and cannot be used for admission,
promotion, composition, ablation, or final claims.

## Primary design cell

- Dataset: DCLM (`mlfoundations/dclm-baseline-1.0`)
- Seed: `1337`
- Training-token tensor: exactly `300,000,000` tokens
- Training steps: exactly `4,000`
- Global tokens per step: exactly `32,768`
- Sampled training positions: exactly `131,072,000`
- Validation tensor: exactly `8,000,000` tokens, starting after token
  `610,000,000`
- Model: exact historical M1 from row 465 of
  `experiments/manifests/iclr26_main_manifest.csv`: 18 layers, width 1024,
  16 heads, FFN width 3072, sequence length 256, batch 8/GPU, accumulation 4,
  world size 4
- Exact unique parameter inventories: SwiGLU `296,867,840`; Global-RLB
  `296,871,080`, including exactly 3,240 RLB coefficients
- LR / minimum LR / weight decay: `3e-4` / `3e-5` / `0.10`
- Warmup / betas / epsilon / clipping: 200 / `(0.9, 0.95)` / `1e-8` / `1.0`
- Evaluation: 10 batches every 50 steps and at step 4,000

The exact token-cache artifacts are:

- Train:
  `experiments/cache/tokens_iclr26_main/dclm/mlfoundations_dclm_baseline_1_0_none_gpt2_train_train_stream_text_skipdocs0_skiptoks0_300000000.pt`
  with SHA-256
  `0ce35903d2e9a54bf1cb4d88cbc77bc1ea6f1385a1f24edff1cd1a89217a8708`.
- Validation:
  `experiments/cache/tokens_iclr26_main/dclm/mlfoundations_dclm_baseline_1_0_none_gpt2_train_validation_stream_text_skipdocs0_skiptoks610000000_8000000.pt`
  with SHA-256
  `2679740c7fa155e3420421cc43df60d08ef29bf929efce9a42f43dee0a1ae8cd`.

Every launcher must verify both file hashes before training and must verify the
runtime config reports `train_tokens == 300000000`, `val_tokens == 8000000`,
`steps == 4000`, `model_scale == M1`, and the activation-specific exact
parameter count above. A missing or mismatched artifact is a hard failure. It may
not fall back to the 100M, 16.384M, 2.62144M, streaming, or regenerated cache.
The M0 `12 x 768` architecture is likewise forbidden; the 4,000-step design
horizon and M1 identity are inseparable in this campaign.

## Muon hurdle

The primary hurdle is the stronger endpoint of two fresh controls:

1. SwiGLU + Muon.
2. Global-RLB + Muon.

Both controls must be newly run for exactly 4,000 steps with the complete
shared configuration above. Step 4,000 from an existing 9,150-step run is not
a matched control because the cosine schedule depends on total steps. The
candidate and controls must have identical realized LR values at every logged
step, not merely the same peak LR.

Each final method requires at least a `0.20` validation-loss lead at step 4,000
over the stronger Muon control, with strictly lower PPL. It must also have
strictly lower normalized train-loss and validation-loss partial AUC over the
inclusive step-1,000-through-step-4,000 interval, using trapezoidal integration
on the common logged grid and dividing by the interval width. Whole-run AUC is
reported only as a diagnostic. A smaller endpoint difference is reported
honestly but does not satisfy the design objective.

## Candidate rules

1. Every candidate is derived from an explicit RLB structural property and has
   an a-priori or a-posteriori mathematical explanation. Merely applying a
   generic optimizer, generic normalization, or generic Muon modification to
   RLB parameters is insufficient. The update must exploit a representation,
   symmetry, low-dimensional exact computation, numerator/denominator
   interaction, or function-space quantity exposed specifically by the RLB
   construction, and the preregistration must state why the same mechanism is
   unavailable or materially less appropriate for ordinary SwiGLU.
2. Peak LR, minimum LR, WD, warmup, betas, epsilon, clipping, batch shape, data
   order, initialization, and schedule are identical to the controls.
3. Every internal LR and WD scale is exactly one. No parameter group receives
   an implicit multiplier, including RLB coefficient or factor parameters.
   Candidate optimizers must expose a complete `lr_wd_fairness_audit()` map;
   the campaign trainer rejects missing or nonunit declarations and audits the
   realized parameter-group LR before and after every optimizer step.
4. A candidate may not be promoted or rejected from an early checkpoint after
   launch. Candidate quality is judged at the full 4,000-step endpoint.
5. Reasoning may reject a proposal before launch when existing full-endpoint
   evidence demonstrates that the same mathematical mechanism is unsuitable.
6. Run at most two jobs concurrently, exactly four A6000 GPUs per job. Request
   no node, topology, NVLink, or feature constraint; disable NCCL P2P.
7. Queue ready candidates together so both four-GPU slots remain occupied.
8. Runtime optimization may change implementation only. It may not change an
   equation, operation order affecting the trajectory, state recurrence, or
   optimizer scalar.
9. The live design pool contains exactly ten reusable opaque slots,
   `R01`--`R10`; candidate-number proliferation is forbidden. A systems-only
   repair overwrites the same generation and must preserve its equations and
   trajectory. A mathematical revision is a new generation in the same slot,
   with a new source hash, preregistration, audit, and complete 4,000-step run.
   Evidence from an older generation never transfers to its replacement.
10. Failure magnitude controls the next action, using only complete 4,000-step
    evidence. A method with a real endpoint-loss lead plus PPL and late-pAUC
    wins, but less than the required `0.20`, is a promising near-miss: diagnose
    it against relevant literature and revise it in place under the same slot.
    A method that is materially worse across endpoint loss, PPL, and the late
    trajectory is a structural failure: delete its implementation and run
    artifacts, retain only a compact generation tombstone, and free the slot
    for a genuinely different mechanism. Borderline non-leading results must
    receive an explicit trajectory/telemetry diagnosis before choosing either
    path; they are never promoted or ablated merely for being close.
11. A rejected mathematical mechanism cannot be renamed, composed, or
    reintroduced. Reusing its opaque slot for a distinct method is allowed and
    does not erase the rejected generation's hash tombstone. A mechanism that
    first passes the component-qualification rule below is not rejected merely
    because its standalone parent misses the final `0.20` hurdle.

## Promotion and ablation

A full parent enters component testing only after all 4,000 updates. It is
eligible in either of two ways:

- **finalist promotion:** it clears the complete `0.20` endpoint, PPL, and
  late-validation-pAUC hurdle;
- **component qualification:** it is at least `0.01` lower in endpoint
  validation loss than the stronger control, with strictly lower endpoint PPL
  and late validation pAUC. This is the minimum "good result" that permits
  ablation or composition; early or merely less-bad trajectories do not
  qualify.

An eligible parent then follows the same ablation protocol:

1. Enumerate its actual mathematical components from code.
2. Run a direct leave-one-component-out experiment for every component from
   the same full method under this exact 300M-pool/4,000-step protocol.
3. Remove any useless component and rerun the pruned full method.
4. Repeat leave-one-out ablations after pruning.
5. A retained component must give the full parent at least `0.01` lower final
   validation loss than its deletion, strictly lower final PPL, and strictly
   lower late validation partial AUC. Train partial AUC is also reported.
6. Retain only a method whose every remaining component passes those gates
   after recursive pruning and re-ablation from the final parent.

Only components that pass this direct deletion test may be composed. Screening
identities remain capped at `R01`--`R10`; post-screening compositions use the
three opaque finalist identities rather than creating more candidate numbers.
A composite must be rerun for 4,000 steps and recursively ablated from its own
full parent. Exactly three methods may enter final validation, and all three
must pass the complete `0.20` endpoint, PPL, partial-AUC, fairness, ablation,
explanation, and runtime rules.

Only after a method clears the exact 300M-model/4,000-step Muon discovery gate
and recursive ablation closure should it enter broader validation. That stage
must include the established exact 100M-model protocol against fresh matched
SwiGLU+AdamW and Global-RLB+AdamW controls, with the same LR/WD and all other
fairness fields, in addition to the required multi-seed/dataset tests. The
method must therefore demonstrate that its gain comes from exploiting RLB at
both scales: against AdamW at 100M and against the stronger Muon hurdle at
300M.

## Evidence ledger required for every run

Each signed report must include token-cache hashes, runtime tensor sizes,
first-batch/data-order hash, initial-state hash, realized LR trace, literal
LR/min-LR/WD/betas/epsilon/clipping, all internal scale telemetry, structural
certificates, endpoint loss/PPL, total runtime, node chosen by Slurm, and the
candidate/control source hashes. Partial or cancelled JSONL files are never
endpoint evidence.
