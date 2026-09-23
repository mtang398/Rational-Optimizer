# Publication verification — September 23, 2026

**21 CPU tests passed:** 16 original fixed-SwiGLU optimizer contracts and five
publication/portable-trainer contracts. These were run in the pinned Python
3.12.11 / PyTorch 2.11.0+cu128 environment on CPU. No GPU allocation was started
for packaging. Review was a distinct self-review, not independent verification.

The original contracts exercise actual ledger/decay and factorized-moment
recurrences, auxiliary AdamW, probe cadence and gradient-surrogate steps,
controlled probe identities across microbatch decompositions, actual masked token
counts, full state restoration across a refresh, original MHA attention and
four-rank CPU/Gloo execution. Import/update also passes with GRAIN imports blocked.

The five package checks cover:

1. Byte identity of the vendored public numerical core, and exact identity of
   the fixed-SwiGLU optimizer after reversing its two import-only substitutions.
2. Actual meta-device model inventories: 2,056,136,704 and 1,092,250,368 unique
   parameters, tied embeddings, and the unchanged full schedule.
3. Every reported step/token identity, complete 100-step validation coverage,
   gap arithmetic and exp(loss) perplexity.
4. Real single-process CLI training for both Muon and TILLER: save at update 3,
   restart, resume through update 10, validation and refresh at updates 1 and 9.
5. The same real CLI workflow through torchrun with four CPU/Gloo ranks for
   both conditions, including checkpoint reload and exact data/counter position.

Additional review established that RMSNorm, attention, SwiGLU and the decoder
initialization/forward classes are AST-identical to the frozen production
source. The unused GRAIN construction branch was removed from the export.
Checkpoint/data helpers derive from the existing portable package; the new
trainer wraps DDP before constructing the fixed-SwiGLU optimizer, performs
whole-update mask preparation and avoids duplicate loss scaling or clipping.
Both shell launchers pass bash syntax validation. The generated plot was viewed.

The publication copies only this new folder. It does not modify a previous
experiment, frozen release, source binding, running allocation or submission
controller. The root repository's current optimizer implementation is not an
implicit numerical dependency: this folder vendors the exact core used by the
reported fixed-SwiGLU experiment.

## Results checks

Both original completion reports state 11,445 updates, passed=true and the same
2,056,136,704-parameter model. Data/environment/seed/accumulation/schedule and
initial shared-tensor hashes match. All 11,445 training updates have matching
token counts and LRs. Periodic validation records cover 100 through 11,400;
the final validation at 11,445 comes from the completion report.

The report field `validation_loss_before` is historically named for the
measurement **after training, before optional checkpoint-reload checking**.
The frozen engine's control flow was inspected to establish this. A null
`validation_loss_after` is not a missing final measurement in these runs.

Both 1B/3B milestone benchmark reports match their checkpoint manifest hashes.
Evaluation protocols, model configuration and evaluation-manifest identities
match between conditions. Tensor payloads were not rehashed for this publication;
the checkpoint-manifest hashes were. Raw benchmark files have source hashes in
their public copies; cluster paths were rewritten to logical provenance paths.

## Limits

The portable launchers have CPU execution evidence, not a new full-size GPU run.
The reported GPU results belong to the frozen campaign trainer. No claim of
bitwise replay across hardware/library changes, automatic reconstruction of the
original curated corpus, or multi-seed statistical significance is made.
Cold distributed numerical replay must not be described as universally bitwise.
The benchmark evaluator itself is not ported here; result JSONs preserve its
protocol and task definitions. The sealed final LM test remains unused.
