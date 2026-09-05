# RLB optimizer design surface

Only the exact-300M/4,000-step design cycle is active. Candidate identities,
states, hashes, and tombstones live in
[CANDIDATE_REGISTRY.md](../CANDIDATE_REGISTRY.md); the binding mathematical
and experimental rules live in
[RLB_300M_4000_DESIGN_CONTRACT.md](RLB_300M_4000_DESIGN_CONTRACT.md).

Opaque candidate files use `rlb_r01.py` through at most `rlb_r10.py`. Shared
numerical code does not allocate another candidate identity. Minor repairs
overwrite the same slot. A rejected candidate's implementation, results, logs,
tests, and derivation are deleted; only its registry tombstone remains.

The closed screening pool contains frozen implementations `R01` through
`R10`; no further candidate identity can be created in this cycle. All use the
matched scalar settings, LR schedule, and WD partition. None exposes an
internal LR or WD multiplier. The exact component lists and mathematical
derivations are frozen beside their manifests in the active experiment
package.
