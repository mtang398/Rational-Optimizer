# Historical method specification

Verbatim September 18 method text except package imports. Historical statements
about pending GPU evidence/admission refer to that date. See README.md and
results/README.md for September 23 results and portable usage.

# TILLER with fixed SwiGLU, without GRAIN

Implementation identity: `tiller_swiglu_fixed_v1`, in `tiller_swiglu.py`.
This is a separate experimental specialization/extension, not an unchanged
implementation of the published GRAIN + TILLER method. It is not registered as
a replacement for any submitted training package. The later authorized 2B
production integration is documented in SWIGLU_2B_SEQUENTIAL.md.

The adapter uses the existing Muon **SwiGLU model** unchanged: no rational
activation, GRAIN coefficients, GRAIN normalization, or rational CUDA extension.
It currently supports the existing decoder's bias-free, sixteen-head MHA
parameterization with square output, FP32 master weights and optional BF16
autocast. It invokes the **unmodified public `TILLERAttentionOptimizer`**;
the imported QKV packing utility is only a workspace, not the GQA optimizer.

## What carries over, and what does not

The public TILLER response drift compares the current activation's Jacobian and
features against the initial activation coefficients, **at the same current
preactivation**. GRAIN's coefficients learn. Fixed SwiGLU has no corresponding
learnable coefficients. Consequently the current/reference activation maps are
identical and their response congruence is exactly `chi = 1`.

The reference adaptive chord uses departure energy `rho * (1 - chi**2)`.
It is therefore **zero**, in both the MLP and the MLP-guided attention route.
This version executes the full reference factorized/attention code and updates
its histories; it does not pretend the drift rotation is active or invent a
temporal-weight, activation, or gate-drift signal. Making that rotation nonzero
would require another, explicitly justified method extension.

The loss-response ledger and coordinated MLP group reweighting remain active.
Those can produce updates different from Muon despite zero drift rotation.
This is not equivalent to native `torch.optim.Muon`: the reference uses grouped
MLP/head-block polar maps and a coordinated equal-budget transaction. Whether
this specialization is useful is unmeasured; no training advantage is claimed.

## SwiGLU response and coordination

For one token, with native weights `Wg`, `Wv`, `Wd`:

```text
g = Wg x
v = Wv x
h = SiLU(g) * v
y = Wd h
a = SiLU'(g) * v
b = SiLU(g)
q = Wd.T * loss_cotangent
```

For a group of matching hidden channels and proposed directions `Ug,Uv,Ud`,
the functional score is the group sum of

```text
(Ug x) * a * q + (Uv x) * b * q + h * (Ud.T loss_cotangent).
```

All three terms are necessary. Tests compare each group's score with an
independent autograd JVP and a central finite difference. The decay measurement
uses these same derivatives with directions `weight_decay * W`.

Gate and value are two independent native matrix polar maps and momentum
buffers. Their matching channel groups together form **one incoming logical
role**, with down as the outgoing role. One coefficient per layer/group acts on
all three matrices. Incoming gradient/momentum contractions are the sum of gate
and value contractions. This retains the public solver's incoming/outgoing
per-layer descent guards; it does **not** add separate gate/value positivity
guards. The budget sums all three scaled direction norms, with the original
native-matrix `match_rms_adamw` adjustments and group-rank calibration. There is
no extra MLP `1/sqrt(2)` or `1/sqrt(3)` factor. Attention's original stacked-QKV
`1/sqrt(3)`, four-head grouping, and square-output treatment are unchanged.

For the incoming activation `J = [diag(a), diag(b)]`, the original participation
definition extends to its rectangular Jacobian as
`tr(J J.T)**2 / (group_width * ||J J.T||_F**2)`.
The denominator uses its maximum output rank, `group_width`. Loss weighting uses
the joint pullback energy. Outgoing participation uses the feature fourth-moment
definition, as in the reference. Sufficient statistics are globally summed
before ratios are formed. The attention scalar combines incoming/outgoing group
means using the reference geometric mean; congruence remains exactly one.

The explicit new inventory convention is `group_width=256`, grouping matching
gate/value/down channels. Thus SwiGLU widths 3072, 4608, and 6144 have 12, 18,
and 24 groups, respectively. This differs from GRAIN's larger two-projection
hidden width and is part of this named extension, not a historical configuration
change. Small test models use a smaller divisible width.

## Preserved optimizer rules

- Beta2 `.95` every optimizer update for the Robust-FD loss ledger, decay cross
  term, and each gate/value/down/QKV/output factorized history. Factorized bias
  correction is `1 - .95**step`, restored using the saved update count.
- Functional refresh at updates `1, 9, 17, ...`, global `K=32`, persistent sketch
  rank 64. On other updates the **actual public trace-matched gradient surrogate**
  enters the ledger; the ledger advances every step.
- Parent momentum `.95`, Nesterov, NS5, original matrix scaling, structural weight
  decay `.1`, one synchronized global gradient clip at norm 1.
- Tied embeddings and normalization parameters use auxiliary AdamW with
  `betas=(.9,.95)`, `eps=1e-8`, no weight decay. Unknown auxiliary matrices fail
  routing rather than being silently assigned. Each native parameter has one
  owner; packed QKV is only a nontrainable optimizer workspace.
- Every child uses the same caller-supplied LR. The adapter does not construct,
  reset, or shorten a schedule. All child groups/states and their counters are
  persisted and checked on restore. Missing histories fail instead of restarting.

## Loss/probe contract and use

```python
from contextlib import nullcontext
from torch.nn import functional as F
from experiments.swiglu_tiller.tiller_swiglu import SwiGLUTILLER

# model is the existing grain=False MHA decoder, optionally wrapped in DDP.
optimizer = SwiGLUTILLER(model, group_width=256)

# One optimizer update. All masks correspond to the actual unreduced losses.
# Prepare the full local update's microbatches before its first forward.
optimizer.zero_grad()
optimizer.begin_step([batch.loss_mask for batch in microbatches])
for group in optimizer.param_groups:
    group['lr'] = scheduled_lr  # same full-horizon schedule as the caller
for i, batch in enumerate(microbatches):
    sync_context = model.no_sync() if hasattr(model, 'no_sync') and i+1 < len(microbatches) else nullcontext()
    with sync_context:
        logits = model(batch.input_ids)
        losses = F.cross_entropy(
            logits.flatten(0, 1), batch.targets.flatten(), reduction='none'
        ).reshape_as(batch.targets)
        optimizer.backward(losses)
metrics = optimizer.step()  # performs clipping once, then all three children
```

`begin_step` globally sums actual loss-mask counts. `backward` scales each local
loss sum by `world_size / global_loss_tokens`, accounting for DDP averaging.
Do not additionally divide by accumulation or call a gradient scaler. Run
`optimizer.step()` outside the forward autocast context. Probe
cotangents undo this normalization and apply the realized clipping factor.
Fixed probe identities are selected across the whole local update and assigned
to microbatches, so changing only microbatch decomposition preserves identities.
Explicit `probe_indices` can be supplied for controlled diagnostics. This
whole-update sampling convention is explicit and differs from the legacy
per-microbatch capture/selection when the decomposition changes. Rank count is
checkpoint-bound; exact topology-independent probing is not claimed.

Call `model.eval()` for evaluation and `model.train()` before training.
The optimizer records no evaluation probes. `close()` removes its capture hooks.
Do not create two live optimizers on the same model. V1 does not support
activation checkpointing/reentrant forwards, FSDP, parameter sharding, GradScaler,
GQA, changed activation functions, or arbitrary HF model wrappers. Unsupported
projection/activation inventories fail. It supports the existing MHA decoder
used for the Muon backbone; the adapter does not change its forward function.

Checkpoint `model.state_dict()` and `optimizer.state_dict()` only after a completed
update. The latter contains child states/LRs, total actual loss-bearing tokens,
ledger step, refresh phase through that step, and cached response statistics.
The **trainer** must additionally save RNG, scheduler, data/sampler position and
atomic checkpoint manifests. On restore, reconstruct the same model/optimizer,
load both states, and restore those trainer states. This module is not integrated
into campaign admission or checkpoint publication; old GRAIN or Muon optimizer
checkpoints must not be loaded into it. A failed restore requires discarding that
optimizer instance; no guarantee of transactional rollback on invalid input.

## Verification and scope

Run on an existing CPU compute allocation, from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PROJECT_ROOT/envs/qwen06b/bin/python" -m unittest \
  experiments.swiglu_tiller.test_tiller_swiglu \
  experiments.swiglu_tiller.test_optim \
  experiments.swiglu_tiller.test_mha -v
```

Tests exercise independent derivative/participation references, literal attention
polar/packing/scaling, actual momentum/history/auxiliary recurrences, complete
save/reload and next-update equality, controlled-probe microbatch invariance,
four CPU ranks with DDP and resume across refresh, unsupported-state rejection,
and import/update with GRAIN modules explicitly blocked. Existing optimizer/MHA
tests cover regression against the unchanged reference and switch behavior.

This adapter has CPU evidence and no GPU execution/performance or training-result
evidence yet. Its separate 2B production authorization and gate are documented in
SWIGLU_2B_SEQUENTIAL.md; the earlier development review did not authorize a job.
All existing campaign releases, configurations, READY files and the submission
bridge are preserved. Do not use this code to relabel prior GRAIN + TILLER results.
