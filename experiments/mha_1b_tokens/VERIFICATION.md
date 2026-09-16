# Verification and provenance

Prepared September 16, 2026. This is a portable starter package; no new GPU job
was submitted for its development, and no CUDA performance result is claimed.

The optimizer/activation dependencies are byte-identical to public repository
commit `b3a6c5e613545cc20f8d133436289ac34878fdf1`. In particular:

- `optimizer_design/tiller.py`: SHA256 `87ff1a16ce37c7b2ff668fedf97307c2bde01ca3c76c3a5e3be0081c8c94274f`
- `optimizer_design/_tiller/core.py`: SHA256 `333e7b57579f395709efbd820b553b41288d0c5a73a20fc885d651d1919f5256`

No optimizer equations or reference optimizer files were changed. The portable
composition handles native equal-QKV packing, ownership, clipping, learning-rate
assignment, state serialization and the explicitly requested 100-step switch.

## Executed checks

- 12 CPU unit/reference tests passed, then passed again in a clean clone with no
  private campaign module present. Includes MHA upstream forward/loss/gradients,
  independent GRAIN algebra and gradients, exact parameter inventories, original
  attention polar normalization, auxiliary rules, exact switch prefix and
  single-process resume at 99/100/101, checkpoint corruption rejection, cache
  round trips, and preservation of abandoned log records.
- Four CPU/Gloo ranks exercised every condition through CLI training, atomic
  save, fresh process startup, reload, and next-batch continuation. Baseline/full
  TILLER cases used 12 tiny updates and stopped at 8; switch used 103 tiny updates
  and stopped at 99. With the documented CPU-only rank-order reduction control,
  final model and optimizer tensors matched exactly (maximum difference 0), as
  did RNG, counters, and next-batch hashes. Optimizer replicas agreed exactly.
- Actual pinned Qwen tokenizer processed synthetic documents into 4097 train
  tokens and 65 validation tokens; exact duplicate removal, hash split selection,
  uint32 publication and subsequent manifest import passed. These are test
  fixtures, not reported scientific data or losses.
- Shell syntax and all four launcher help paths checked. CUDA setup is supplied
  for the target host; it was not executed against another host's CUDA toolkit.

## Numerical diagnostic, not a hidden method change

Ordinary CPU/Gloo reductions can change summation order when DDP is rebuilt at
resume. The uncontrolled switch check first differed at step 101, despite equal
logged losses and gradient norms through step 100. By step 103 its largest model
weight difference was approximately 3.1e-6 and training loss differed by 4.8e-7.
A Robust-FD score factor also changed row signs while its represented Gram matrix
was exactly unchanged. Counters, data positions, and the decay cross history were
checked separately. No training/optimizer modification was made to suppress these
observations. The deterministic reduction helper lives only in `test_worker.py`;
production launchers never import it. Do not interpret these CPU results as
bitwise distributed replay or GPU validation on arbitrary hardware.

## Distinct critical self-review

This was self-review, not independent review. Review checked parameter coverage,
complete attention routing, momentum/auxiliary switch state, initialization,
packing, offline operation, environment/source binding and portable paths.
It found and fixed two starter issues: importing a cache without an explicit
pinned tokenizer identity, and retaining non-durable log rows in canonical metrics
after rollback. Regression tests exercise both fixes. No private campaign state,
credentials, corpus text, checkpoints, scheduler script or account is published.

Remaining limits: default-scale CUDA runtime/throughput must be established on
the destination hardware. The optional starter corpus preparation does exact
rather than near-duplicate filtering and screens evaluation overlap only when
exclusion texts are provided. Six-task benchmarks are not automatically staged
or executed. README documents importing the exact campaign cache for comparison.
