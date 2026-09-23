ScalingOPT review for fixed-SwiGLU TILLER — September 23, 2026

The most practical combination to investigate is stateless conditioning of TILLER's parent direction, starting with MuonEq. The most relevant separate hypothesis about late training is spectral shaping within each group, represented by HTMuon. Neither is an established explanation of our results. The pending parent/coordination ablation should determine which component we build on.

This was a literature and source review. No training condition, production source, configuration, scheduler state, or submission bridge was changed.

**Coverage and evidence**

I screened the ScalingOPT catalog's 89 optimizer entries and its 146-entry paper index, inspected the accessible public discussions and linked training repository, and followed selected entries into primary method sections and author implementations. This is not a claim to have read all 146 papers in full or independently reproduced the results. The screening CSV explicitly distinguishes catalog triage from primary-source and implementation review. The catalog is a discovery resource, not a common-protocol benchmark. Sources: [community](https://tianshijing.github.io/ScalingOpt/), [catalog repository](https://github.com/tianshijing/ScalingOpt), [discussions](https://github.com/tianshijing/ScalingOpt/discussions), [linked training code](https://github.com/OpenEnvision/ScalingOPT).

Author code from six repositories was downloaded without execution, pinned through each repository's commits/HEAD API, and hashed. The files and revisions are in author-source-manifest.json. Six repositories: MuonEq, NorMuon, Muon+, SOAP–Muon, MARS, and Aurora. Relevant routines were read, not exhaustively audited for every supported training environment. Source inspection exposed differences between catalog descriptions, paper pseudocode, and particular code paths; those distinctions matter for any later implementation.

Local comparison used the accepted fixed-SwiGLU 2B release and accepted ablation release. Their tiller_swiglu.py, public core.py, and TILLER_SWIGLU.md are byte-identical; hashes are recorded in frozen-source-comparison.json. The ablation audit was read separately. Withdrawn workspace edits were not used as evidence.

**What our optimizer actually offers as a combination point**

Fixed-SwiGLU has congruence chi=1, so the GRAIN-dependent adaptive response-drift rotation is zero. The loss ledger and coordination remain active. Rejected coordination falls back to grouped MLP and head-block attention parents; it does not become the native whole-matrix Muon baseline. Thus low acceptance alone cannot identify what produced the advantage.

For the 1.09B model, native gate/value matrices are 4608×1536, but TILLER polarizes 256×1536 blocks. Down-projection blocks are 1536×256. The core applies a block/global rank compensation before the native matrix RMS scaling. These exact axes and scale factors must survive any controlled parent substitution.

The clean conceptual composition is: true loss gradient → parent direction construction → response measurements, gradient contractions and budget → TILLER coordination → update. The implementation is not already an unrestricted wrapper for arbitrary optimizers. A new parent must supply consistent scores, energies, decay cross terms, and momentum semantics before the existing acceptance calculation.

Changing a direction after acceptance would invalidate the calculation that accepted it. The ledger must use the actual proposed directions on refresh steps and the matching gradient surrogates between refreshes. A transformed gradient estimate must never masquerade as the raw loss gradient in the descent guards. New states require explicit checkpoint identities and restoration tests.

**Candidate assessment**

| Candidate | What it adds | Assessment for our fixed-SwiGLU optimizer |
|---|---|---|
| MuonEq | Diagonal conditioning before polar approximation | First inexpensive parent candidate; isolates within-group conditioning without a new history clock. |
| HTMuon | Non-flat singular-value response | Strongest targeted hypothesis for within-block spectral effects later in training; changes the parent method. |
| NorMuon / Muon+ | Row/column balancing after polar approximation | Plausible, but check overlap with existing block geometry before spending a run. |
| Aurora | Leverage-balanced polar directions for tall matrices | Particularly relevant to down-projection blocks; its wide-matrix branch already reduces to ordinary polar. |
| SOAP–Muon / Newton–Muon | Additional curvature/input geometry | Conceptually complementary, but materially greater implementation and state costs. |
| MARS-M / MONA | Gradient-difference correction before polar | Possible temporal complement; extra state and raw-gradient guard semantics are significant. |
| TEON | Polarization across tensors from consecutive layers | Important structural prior art and alternative parent geometry; not the same operation as our head blocks. |
| SGG / Magma | Another group allocation or masking rule | Overlaps or conflicts with the existing coordination budget; comparison is cleaner than blind stacking. |
| Nexus | Gradient similarity through a microbatch inner trajectory | Directly relevant to the benchmark/loss puzzle, but incompatible with our current fixed-parameter accumulation assumptions without redesign. |
| Gram Newton–Schulz | Faster computation of a selected polynomial | Systems option; does not by itself answer why the optimizer learns better. |

MuonEq computes current row/column scales before orthogonalization; its main transformation adds no persistent state. Its paper separates conditioning benefits from the fact that preprocessing changes the exact polar direction. The default is row conditioning, not universally row-and-column conditioning. It reports LLaMA2/C4 comparisons through nominal 1B models; those configurations and hardware differ from ours. [Paper](https://arxiv.org/html/2603.28254v2).

My inference: this is a relatively clean test of whether the parent benefits from balancing inside a TILLER group. Keep our grouping, momentum, NS5, compensation, auxiliary optimizer, and coordination rules fixed. Do not also enable the author's optional spectral-proxy backend or change normalization axes in the same first test. This is a proposed extension, not unchanged TILLER.

HTMuon replaces perfectly flat singular values with a small positive power of momentum singular values. Its 1B C4 experiment reports PPL 14.17 versus Muon's 14.33, using a different recipe, including matrix-optimizer treatment of embedding/output layers. Its expensive exact implementation and accelerated variants are distinct. [Paper](https://arxiv.org/html/2603.10067v1).

The relevant hypothesis is specific: our coordinator can scale groups, but one coefficient cannot change relative singular-direction strengths inside a block. Spectral shaping can. This would matter even when c=1. It does not show that our tail singular modes are noisy, or that spectral shaping preserves the early advantage. A copied-state comparison would need actual finite-step held-out loss, raw-gradient contractions, and matched update budget; spectra alone cannot establish usefulness.

NorMuon tracks row means of squared orthogonalized updates and uses that history to rebalance them. Its statistic is not TILLER's existing gradient-based factorized adaptive history. Muon+ applies post-polar row/column normalization without that EMA. They therefore make different interventions. [NorMuon](https://arxiv.org/html/2510.05491v1), [Muon+](https://arxiv.org/html/2602.21545v3).

There is an important geometric overlap. For an exact polar factor of a full-row-rank wide block, U Uᵀ=I and every row already has unit norm. Additional row balancing is then trivial up to a common scale. Our NS5 approximation is not exact, and tall down blocks differ, so this is not an argument that normalization cannot help. It is a reason to measure the affected axes first. A small synthetic float64 illustration is saved separately: post-row normalization changes an 8×32 exact-polar direction only at roundoff, whereas pre-row conditioning changes it materially. This is algebraic illustration, not production or GPU evidence.

Aurora's inspected implementation explicitly calls standard polar for square/wide matrices and iteratively adjusts diagonal scales for tall matrices. This makes the proposed application to our down blocks more concrete than applying its headline neuron-balancing story uniformly to every TILLER block. It can require multiple polar calls. [Author code](https://github.com/tilde-research/aurora-release/blob/12e30810e1a2226fba77dcb7d6e23f3c4c79726b/src/aurora.py), [author explanation](https://blog.tilderesearch.com/blog/aurora).

SOAP–Muon combines basis-aware adaptive conditioning with a polar update. The authors label the report preliminary. Newton–Muon uses an input second-moment inverse to condition gradients before momentum/polar construction; its reported runtime evidence is from reproduced historical modded-NanoGPT settings, not our dense 2B model. [SOAP–Muon report](https://nikhilvyas.github.io/SOAP_Muon.pdf), [Newton–Muon](https://arxiv.org/html/2604.01472v1).

Both offer local geometry beyond our scalar group allocation. However, having activation hooks does not make covariance estimation or inversion free. Our K32 response samples are not automatically an adequate full covariance estimator. A 256-dimensional factor is cheaper than a width-dimensional factor, but factor storage across all groups, basis changes, update timing, and state restoration still need accounting. These are later candidates, not the first small modification.

MARS-M already combines gradient-difference correction with matrix orthogonalization. Its exact and approximate modes have different gradient-computation requirements. MONA adds an EMA of gradient differences before the momentum/polar path and evaluates MoE models; nominal total parameter counts should not be treated as comparable dense sizes. [MARS-M code and protocol](https://github.com/AGI-Arena/MARS/tree/4831e28eba863a4e69d7e9474ff8e73bc1b410fe/MARS_M), [MONA](https://arxiv.org/html/2605.26842v2).

The attraction is temporal information that group allocation cannot create. The expense is extra gradient/history tensors, distinct initialization, and potentially different clipping. Gradient differences across changing minibatches are not pure curvature measurements. MONA's convergence discussion also assumes positive expected alignment; preservation of a column space alone does not establish positive inner products. I would not import that claim as a guarantee for our acceptance guard.

TEON combines momentum tensors from consecutive layers before orthogonalization; its selected attention grouping is not our within-layer grouping of heads. Its FineWeb table gives nominal 1B PPL 10.84 versus 11.19 for Muon with PolarExpress, under that paper's much longer token budget. This is useful evidence that matrix assembly itself can matter. [Paper](https://arxiv.org/html/2601.23261v2).

This makes TEON important prior art if our attention-parent cell explains much of the gain. A cross-layer parent would also change the response basis and coupling assumptions. It is a substantial structural extension, not a drop-in way to retain original attention TILLER unchanged.

SGG dynamically clusters gradient statistics and calibrates group-specific learning rates. Magma combines stochastic update masking with momentum–gradient alignment damping; its reported gains include different base optimizers and MoE settings. [SGG](https://arxiv.org/html/2506.01049v1), [Magma](https://arxiv.org/html/2602.15322v1).

These are relevant comparators to our coefficient allocation. Applying another mask or group multiplier after our accepted transaction changes its budget and descent calculation. A zeroed role can also fail an existing strict-positive guard. Their existence motivates comparing allocation mechanisms, not weakening our rejection predicates merely to obtain more acceptances.

Nexus explicitly studies better downstream generalization at similar pretraining loss. Its engineering version updates an inner model between microbatches and feeds the resulting displacement to an outer optimizer; forward/backward counts can remain unchanged even though parameters move during accumulation. [Paper](https://arxiv.org/html/2604.09258v1).

That result is relevant context, not an explanation of our benchmark pattern. Our exact global-gradient contractions and whole-update probes currently assume one parameter state across the accumulated update. Nexus would break that assumption. Preserving a comparable true-gradient certificate could require additional work, and copying/offloading an inner model has real bandwidth/state cost. It is a longer-term generalization direction, not a trivial wrapper for our trainer.

Gram Newton–Schulz reorganizes polynomial evaluation around a small Gram matrix. Its authors demonstrate numerical instability in naive low-precision recurrences and use restarts to control it. [Author article](https://dao-lab.ai/blog/2026/gram-newton-schulz/).

An algebraically equivalent implementation still needs BF16 direction and acceptance parity checks. Changing the polynomial to PolarExpress is a separate numerical-method intervention. Since forward/backward dominates our recorded step time, a large speedup of orthogonalization cannot be advertised as the same speedup of training.

Other screened directions remain useful context: AdaMuon combines a sign-transformed polar input with elementwise adaptation of polar updates; Muon² applies a gradient second-moment preconditioner before polar. Both differ from NorMuon's reduced row history. [AdaMuon](https://arxiv.org/abs/2507.11005), [Muon²](https://arxiv.org/html/2604.09967v1). SPECTRA clips spectra rather than imposing exact flatness and reports benefits for several vector optimizers; clipping an already flat polar spectrum can be redundant. [SPECTRA](https://arxiv.org/html/2603.14315v1). Weight-manifold methods such as Mano alter constraints/retraction and are more intrusive than a parent substitution. [Mano](https://arxiv.org/abs/2601.23000).

**What close source inspection changed**

These findings are about the inspected author code, not alleged bugs in our running release:

| Source | Verified implementation detail | Consequence for a future port |
|---|---|---|
| MuonEq, GPU muoneq.py | Default auxiliary AdamW betas are (.95,.95); optional backend and phase behavior also exist. | Import the selected parent transformation, not the whole optimizer's defaults; retain our (.9,.95), no-decay auxiliary behavior. |
| NorMuon, normuon_update | Squared statistics come from the polar output; code preserves its measured pre-normalization Frobenius norm; Nesterov uses an in-place grad lerp. | Keep separate state and explicit norm convention; protect raw gradients needed by our guards. Paper target-RMS and this code's measured-norm preservation are not identical prescriptions. |
| Muon+, apply_post_polar_norm | The inspected helper accepts eps but uses literal 1e-7 inside each square root. | Resolve executed arithmetic, not just configuration fields. Other polar backends must not silently enter the comparison. |
| SOAP–Muon, nanogpt_optimizer.py | Initialization updates a preconditioner then skips the parameter update; post-polar code also applies an elementwise signed square root and renormalizes. | This is more than a minimal “SOAP then Muon” recipe. Counter alignment and the extra nonlinear transform require explicit choices. The OLMo/nanoGPT scaling conventions differ. |
| MARS-M, mars_m.py | The inspected path clips c_t when its norm exceeds one despite exposing clip_c; approximate last_grad is assigned by reference. | Audit actual clipping and gradient-buffer lifetime. Copying a class would introduce confounds beyond a history correction. |
| Aurora, aurora.py | Wide/square branch bypasses the leverage-balancing iteration. | Its additional mechanism would not affect our wide gate/value blocks under that orientation. |

These claims can be checked against the pinned files in author-source-manifest.json. They do not establish that the author experiments used every inspected default or code path. No third-party code was executed or installed.

**Connection to scaling and the late gap**

The hyperparameter-transfer study is especially relevant: it analyzes blocking, update normalization, and width/depth scaling, and reports that optimizer rankings can change with the scaling prescription. It does not diagnose our method. [Primary study](https://arxiv.org/html/2512.05620v1).

Our model-size comparisons also change block aspect ratios relative to native matrices. Thus “larger models intrinsically favor TILLER” and “these shapes and scales favor this parent construction” remain different explanations. The scheduled four-cell ablation helps separate parent structure from coordination at one size; it does not by itself prove a scaling law.

The 1000-step ablation is well suited to explaining the early advantage and proposal failures. It cannot establish that a candidate fixes deterioration thousands of steps later. Also, a finite 3B-token endpoint on the retained 12B learning-rate horizon is not evidence of fully annealed convergence. These are interpretation limits, not requests to change existing runs.

**Recommended decision after the existing ablation**

1. If grouped MLP explains most of the advantage, retain that parent geometry. MuonEq is the lowest-cost complementary direction; HTMuon is the more targeted separate test of within-group spectra. Do not combine both in the first comparison.
2. If head-block attention explains most of the advantage, treat attention matrix construction as central and use TEON as relevant prior art. Keep MLP changes separate.
3. If full coordination beats both-parent training, preserve the coordinator when testing an improved parent. If it does not, do not attribute the observed advantage to proposal acceptance.
4. Use the new rejection records to distinguish no surrogate gain, descent failures by role/layer, budget residuals, hard cases, and nonfinite arithmetic. An optimizer that improves a different mechanism need not fix the dominant rejection reason. Higher acceptance is not itself the objective.
5. Any later interaction claim needs both the new parent alone and the same parent with coordination under matched settings. A hybrid beating native Muon alone would not identify a contribution from TILLER coordination.

My current priority is MuonEq for a restrained implementation hypothesis, HTMuon for the late-stage spectral hypothesis, and Nexus as conceptual guidance for studying downstream generalization. All remain proposals pending our ablation evidence; no additional run has been queued.

Artifacts alongside this report: optimizer-screening.csv, paper-index.json, author-source-manifest.json, review-manifest.json, frozen-source-comparison.json, and synthetic-polar-geometry.json. They preserve the coverage, source versions, and limits of this review.
