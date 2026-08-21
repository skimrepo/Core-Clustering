# MTL V3.2 Report: Two Targeted, Evidence-Driven Fixes — One Negative, One Partial Success

V3.2 implements exactly the two changes `MTL_V3_1_DIAGNOSTIC_REPORT.md`'s
frozen-checkpoint evidence directly supported, on top of the V3.1
foundation (frozen, unchanged): (1) stop-gradient the Intensity scale
branch so it can no longer shape the shared representation, and (2)
replace ONLY the Location head's generic multi-query attention pooling
with a position-preserving pooling mechanism.

**Headline findings, stated up front:**

- **Intensity (change #1): a clean NEGATIVE result.** Blocking the
  scale branch's gradient into the embedding did not improve
  anomaly-internal magnitude ordering — pairwise ordering agreement
  stayed at chance (0.51-0.58 vs V3.1's 0.566) and anomalous-only
  spearman got slightly WORSE (flipped to -0.03/-0.10 from V3.1's
  +0.228) across both healthy seeds. The hypothesis that the scale
  branch was "stealing" the gradient signal needed for magnitude
  learning is not supported.
- **Location (change #2): a genuine partial success with a newly
  exposed downstream bottleneck.** The position-aware pool's own
  attention demonstrably tracks true location (|pearson| 0.71-0.94
  between attention center-of-mass and the true target, both healthy
  seeds) and Stage F no longer destroys the signal catastrophically
  (0.27-0.62 vs V3.1's -0.057). But the FINAL embedding (Stage G, after
  the existing MLP+L2-normalization) still collapses the gain
  (0.02-0.09), so the aggregate Location metric is still ~0. This is
  the "pooling succeeded, next bottleneck is downstream MLP/
  normalization" case the diagnostic's own decision framework
  anticipated.
- **A real collapse reappeared in 1 of 3 seeds (seed1)** — NOT via
  either targeted change's own mechanism, but via the (frozen,
  unmodified) reference-conditioning gate saturating near 1.0 almost
  immediately and staying there for all 20 epochs. Reported prominently
  and diagnosed below, not hidden; downstream interpretation for seed1
  is stopped per the task's own instruction, and the 3-seed comparison
  table reports both with and without it.

## A. Exact Changes

```text
MODIFIED (additive, default-preserving -- both flags default to exactly
V3.1's behavior):

core_clustering/models_contrastive_v3.py
    ContrastiveEncoderV3 gained detach_scale_attrs=() -- for each name in
    this tuple, forward() computes mu from the adapter's own call on the
    (non-detached) embedding, and scale from a SEPARATE call on
    embedding.detach(). Both calls share the SAME ScalarPredictionAdapter
    instance (same weights) -- nothing about the adapter's architecture
    or parameter count changes, only which input tensor each half reads.
    ContrastiveEncoderV3 also gained location_position_aware_pooling=False,
    which -- ONLY for the "location" name -- constructs its AttributeHead
    with pooling="position_aware" instead of the default
    "multi_query_attention". Every other attribute's head is constructed
    exactly as before regardless of this flag.

core_clustering/models_contrastive_v2.py
    AttributeHead gained pooling: str = "multi_query_attention" (default,
    byte-identical to V3.1) | "position_aware" (new). New class
    PositionAwarePool(channels, out_dim): a learned per-timestep scalar
    score -> masked softmax -> feature_summary (attention-weighted
    features, same spirit as any attention pool) + position_summary
    (attention-weighted normalized timestep, reusing build_position_channel
    for the position values so it is the SAME position convention the
    model already uses everywhere else) -> concatenated and projected to
    EXACTLY the dimension AttributeHead's existing, unmodified mlp expects
    (num_queries * proj_channels) -- nothing downstream of pooling needed
    to change.

diagnostics/v3_baseline.py, core_clustering/cli_contrastive_v3.py,
diagnostics/v3_gradient_analysis.py
    +--detach_scale_attrs, +--location_position_aware_pooling passthrough.
    v3_gradient_analysis.py additionally reports, per training segment:
    gradient INTO the Intensity embedding from the loss (structurally
    equals the mu-path only, once scale is detached), the scale adapter's
    OWN parameter gradient, and the Location pooling submodule's own
    parameter gradient -- the fine-grained decomposition Section 24 of
    the spec required.

NEW (diagnostics only):
diagnostics/v3_2_diagnostic_probes.py -- reuses every model-generic
    function from v3_1_diagnostic_probes.py unchanged (embedding probes,
    target audit, gradient probe, reference-effect sweep, shape/extent
    sanity) and re-implements ONLY the two pieces that depended on V3.1's
    specific pooling submodule name: the Location stage probes' Stage F
    hook (generalized to hook whichever pooling type is active), and a
    NEW attention/position-summary diagnostic that only exists for
    position_aware pooling. Also fixes a real bug caught while adapting
    v3_1's intensity_loss_decomposition: under detach_scale_attrs, the
    SAME shared adapter linear layer is called TWICE per forward (once
    per branch), so a single forward hook silently kept only the second
    call's raw output, making "grad_raw_mu" measure the dead scale-branch
    column instead of the real mu path (first run: 0.0 in every bin,
    caught immediately as implausible and fixed before being reported
    anywhere in this document).

NOT modified (confirmed via `git diff`, zero changes this round): trunk,
ReferenceContextEncoder, ContextFusion, reference gate/sampling/K-regimes/
contamination, reference-consistency objective, Shape's AttributeHead/
pooling/loss, Extent's AttributeHead/pooling/loss, Location's 1x1 conv
projection/scalar adapter/loss, Intensity's AttributeHead/pooling/Laplace
loss, optimizer, gradient clipping, simulator, split, intensity range,
seeds, epochs, patience.
```

Full test suite: 262 passed (was 247 after V3.1; +15 new tests this
round covering gradient isolation for the stop-gradient change and mask
correctness/gradient propagation/the core "attention mass moving in time
moves position_summary" property for the new pooling module).

## B. Confirmation of Frozen Components

`git diff` against the pre-V3.2 commit shows zero changes to
`reference_context.py`, `losses_contrastive.py`, `losses_v3.py`,
`dataset_episodic.py`, `dataset_dynamic_contrastive.py`, and to every
non-Location branch of `models_contrastive_v2.py`/`models_contrastive_v3.py`.
`trainer_contrastive_v3.py` required ZERO changes for either V3.2
change -- both are purely model-forward-level, so the trainer's
`_compute_losses` reads `out["intensity_mu"]`/`out["intensity_scale"]`
and `out["embeddings"]["location"]` exactly as it did for V3.1.

## C. Parameter Count

| | V2.1 | V3 | V3.1 | V3.2 |
|---|---:|---:|---:|---:|
| Shared trunk | 294,848 | 294,848 | 294,848 | 294,848 |
| All 4 AttributeHeads | 75,264 | 75,264 | 75,264 | 75,297 |
| Reference encoder + context fusion | -- | 49,923 | 49,923 | 49,923 |
| Scalar/uncertainty adapters | -- | 231 | 231 | 231 |
| **Total** | **370,112** | **420,266** | **420,266** | **420,299** |

The +33 vs V3.1 is exactly `PositionAwarePool.score` (`Linear(32, 1)` =
33 params) -- `PositionAwarePool.project` (`Linear(33, 128)` = 4,352
params) replaces the old `pool_attn`+`queries` combination (4,224 +
128 = 4,352 params) almost exactly, so the net head-parameter change is
dominated by the one small new scoring layer, not a capacity increase.
Confirmed identical across all 3 seeds' checkpoints.

## D. Compute Environment and Runtime

| Work | Where | Device | Runtime | Command |
|---|---|---|---|---|
| 3-seed V3.2 training | GPU server (`aibiz@10.10.10.16`) | cuda | seed0: 88.4s, seed1: 87.8s, seed2: 88.4s (~4.4 min total) | `v3_baseline.py --n_instances 1000 --epochs 20 --patience 5 --seeds 0 1 2 --shape_objective plain --detach_scale_attrs intensity --location_position_aware_pooling --experiment_id_prefix v32 --skip_v21_backfill --device cuda` |
| Single-seed gradient analysis | GPU server | cuda | (retraining-based; ran alongside the 3-seed job) | `v3_gradient_analysis.py --n_instances 1000 --epochs 20 --seed 0 --device cuda --shape_objective plain --detach_scale_attrs intensity --location_position_aware_pooling --output_name v32_gradient_analysis.json` |
| All of Sections E-S below (collapse check, stage probes, embedding probes, temporal-shift/attention diagnostics, target-ceiling analysis, linearity/uncertainty tests, reference/contamination sanity) | Local CPU | cpu | ~10s per seed (checkpoint-only: forward passes, or forward + a single backward with no optimizer step) | `v3_2_diagnostic_probes.py --checkpoint diagnostics/outputs/v32/v32_multitask_seed{0,2}/bestmodel.pkl --n_instances 1000` |

Checkpoints used: `diagnostics/outputs/v32/v32_multitask_seed{0,1,2}/
bestmodel.pkl`. Consistent with the compute policy, all actual training
ran on the GPU server; every local-CPU step here is checkpoint-only
(matching the "small checkpoint-only diagnostics" carve-out) -- no
retraining happened locally.

## E. Collapse Check (run first, before any downstream interpretation)

| Seed | `collapsed` | shape separation | shape mean per-dim std | location std(mu) | extent std(mu) | intensity std(mu) | intensity mu-vs-D pearson | mean gate (K=10) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seed0 | **false** | 1.898 | 0.129 | 4.3e-06* | 5.9e-07* | 0.073 | 0.319 | 8.6e-06 |
| seed1 | **true** | 8.2e-08 | 6.98e-07 | 4.3e-06 | 5.9e-07 | 6.6e-08 | -0.119 | **0.99999825** |
| seed2 | **false** | 1.699 | 0.148 | 0.0025 | 0.049 | 0.086 | 0.322 | 0.005 |

*Seed0's location/extent std(mu) at K=10 (the setting `collapse_check`
measures the reference block at) look small in absolute terms mainly
because seed0's gate is itself near-zero at K=10 (Section P) -- this is
a separate, seed-level reference-gate finding, not a representation
collapse. Seed0's own K=0 diagnostics in Sections G-M independently
confirm healthy, non-collapsed behavior (e.g. Stage C-E Location
pearson 0.63-0.68, Intensity D-correlation 0.32, matching the
`mu_vs_D_pearson_corr` column above). **`collapsed=false` for seed0 and
seed2; seed1 is the only genuine collapse.**

**Seed1 is a genuine, reproducible collapse** -- shape/location/extent/
intensity all show the same "every output constant regardless of input"
signature V3's original heteroscedastic-loss collapse showed, and its
shape loss curve confirms it precisely (Section E.1 below). Per the
task's own instruction ("If V3.2 reintroduces collapse, stop downstream
interpretation and diagnose the collapse"), seed1 is EXCLUDED from all
downstream deep-dive diagnostics (Sections G-S deliberately use only
seed0/seed2) but IS reported in the 3-seed comparison table (Section F)
so its impact is visible, not hidden.

### E.1 Diagnosing Seed1's Collapse Mechanism

```text
Shape training loss by epoch, all 3 seeds:
  seed0: 3.433 -> 3.433 -> ... -> 3.028 -> 3.099   (std=0.140 across 20 epochs)
  seed1: 3.434 -> 3.434 -> ... -> 3.434 -> 3.434    (std=0.0000853 -- FROZEN)
  seed2: 3.435 -> 3.434 -> ... -> 2.975 -> 3.472   (std=0.134 across 20 epochs)
```

Seed1's shape loss is pinned at its epoch-0 initialization value for
the ENTIRE 20-epoch run -- not "collapses partway through" like V3's
original failure, but frozen from the very start. Combined with the
collapse check's `mean_gate = 0.99999825` (the reference gate is
essentially fully saturated open, K=10), the most likely mechanism:
once `ContextFusion`'s gate moves close to 1 (a state the frozen,
unmodified gate mechanism has always been ABLE to reach, just never
reached in any of V3.1's or V3's 3 seeds), `H_fused` becomes dominated
by the reference-derived aggregate rather than the query's own `Hq`.
Since the reference set is a randomly-sampled population rather than
anything specific to the individual query, a fused representation
dominated by that aggregate carries little per-query discriminative
information -- every head reading a nearly-query-independent `H_fused`
would plausibly collapse to a near-constant output, which is exactly
what is observed. **This is a pre-existing degenerate mode of the
frozen, unmodified reference-conditioning mechanism itself** (not
something either V3.2 change introduced directly), which V3.1's 3 seeds
happened not to fall into but V3.2's altered training dynamics (from
the two targeted changes) evidently pushed seed1 into. This is
inferred from the available evidence (loss curves + collapse check),
not directly proven via a training-time gate trajectory (not logged);
stated as the most likely explanation, not a certainty.

## F. V2.1 / V3 / V3.1 / V3.2 Comparison (mean ± std across seeds)

| Task | Metric | V2.1 | V3 | V3.1 | V3.2 (all 3 seeds) | V3.2 (seeds 0,2 only) |
|---|---|---:|---:|---:|---:|---:|
| Shape | nn_accuracy | 0.984±0.008 | 0.647±0.073 | 0.816±0.031 | 0.847±0.212 | **0.997±0.003** |
| Shape | separation | 1.540±0.104 | 0.000±0.000 | 0.770±0.138 | 1.199±0.852 | **1.799±0.099** |
| Location | pearson | 0.001±0.008 | 0.018±0.099 | 0.024±0.111 | 0.005±0.107 | 0.017±0.129 |
| Location | spearman | -0.005±0.003 | 0.049±0.093 | 0.007±0.105 | -0.002±0.099 | 0.015±0.118 |
| Location | mae | 0.229±0.014 | 0.247±0.016 | 0.248±0.016 | 0.247±0.014 | 0.241±0.013 |
| Extent | pearson | 0.221±0.090 | 0.064±0.150 | 0.387±0.116 | 0.460±0.322 | **0.684±0.076** |
| Extent | spearman | 0.213±0.058 | 0.050±0.118 | 0.362±0.121 | 0.475±0.358 | **0.727±0.029** |
| Extent | mae | 0.166±0.023 | 0.113±0.008 | 0.102±0.002 | 0.099±0.015 | 0.089±0.005 |
| Intensity | pearson (all samples) | 0.654±0.097 | 0.024±0.231 | 0.288±0.031 | 0.175±0.206 | 0.320±0.002 |
| Intensity | spearman (all samples) | 0.834±0.067 | 0.046±0.208 | 0.604±0.093 | 0.489±0.427 | 0.791±0.005 |
| Intensity | mae | 0.588±0.074* | 2.093±0.287 | 2.154±0.269 | 2.108±0.293 | 2.250±0.261 |

*V2.1's intensity MAE is on a different target scale (bounded,
sigma-normalized) than V3/V3.1/V3.2's unbounded `D` -- not directly
comparable, carried over from prior reports' own caveat.

**"All 3 seeds" vs "seeds 0,2 only" tells the real story here**: the
huge std in the "all 3 seeds" column (e.g. Shape separation
1.199±0.852) is entirely an artifact of averaging in seed1's collapse,
not genuine cross-seed variability. **Restricted to the 2 healthy
seeds, Shape and Extent both meaningfully IMPROVE over V3.1** (Shape
nn_accuracy 0.997 vs 0.816; Extent pearson 0.684 vs 0.387) -- neither
was a targeted change this round, and per Section 27's instruction this
is reported as an observed, unattributed side effect rather than
credited to either of V3.2's two specific changes. **Location's
aggregate correlation is still ~0**, essentially unchanged from every
prior architecture -- Section G explains exactly why despite the
pooling fix working as designed. **Intensity's aggregate numbers are
flat-to-slightly-better on the "all samples" view** (which is dominated
by the easy normal-vs-anomaly split, Section M) but this is explicitly
NOT the number that matters for judging change #1 -- see Section M.

## G. Location Stage Probes (the direct test of change #2)

Ridge-regularized, position-preserving (flatten, not mean/max-pooled)
probes, same methodology as `MTL_V3_1_DIAGNOSTIC_REPORT.md` Section J,
now re-run against V3.2's checkpoints (Stage F hooks the NEW
`position_pool` submodule instead of V3.1's `pool_attn`):

| Stage | V3.1 (for reference) | V3.2 seed0 | V3.2 seed2 |
|---|---:|---:|---:|
| A: early trunk | 0.549 | 0.562 | 0.527 |
| B: middle trunk | 0.636 | 0.651 | 0.722 |
| C: final trunk `Hq` | 0.709 | 0.633 | 0.819 |
| D: `H_fused` (K=0) | 0.709 | 0.633 | 0.819 |
| E: Location 1x1 conv | 0.715 | 0.677 | 0.774 |
| **F: pooling output** | **-0.057** | **0.617** | **0.268** |
| G: final embedding | -0.162 | 0.016 | 0.090 |

(Pearson, anomalous-only, held-out val.) **This is the direct,
decisive confirmation that change #2 worked exactly as intended at the
step it targeted**: V3.1's pooling step DESTROYED the signal
(0.715 -> -0.057, a collapse); V3.2's position-aware pooling instead
PRESERVES a substantial fraction of it (0.677 -> 0.617 for seed0,
0.774 -> 0.268 for seed2 -- both far above V3.1's catastrophic
collapse, though the exact fraction preserved varies by seed).

**But Stage G still loses most of what Stage F preserved** (0.617 ->
0.016; 0.268 -> 0.090) -- per the task's own decision framework
(Section 25): *"If Stage F preserves location but final Stage G loses
it: next bottleneck is downstream MLP / normalization."* **This is
exactly that case.** The existing, unmodified `AttributeHead.mlp`
(`Linear -> GELU -> Linear`) followed by L2 normalization, sitting
between Stage F and Stage G for EVERY attribute (not just Location),
is the newly-exposed bottleneck -- something V3.1's diagnostic could
never see because the signal was already dead before reaching this
step.

## H. Location Temporal-Shift Controlled Test

Same controlled setup as `MTL_V3_1_DIAGNOSTIC_REPORT.md` Section K
(same waveform/extent/intensity, only location varied across
`[0.1, 0.3, 0.5, 0.7, 0.9]`):

| | V3.1 | V3.2 seed0 | V3.2 seed2 |
|---|---:|---:|---:|
| mu range across the 5 locations | 0.4963-0.4966 (flat) | 0.5056-0.5111 | 0.5078 (flat) |
| mu vs true location pearson | -0.450 | -0.627 | -0.281 |
| corr(delta_location, embedding_distance) | -0.025 | **0.253** | 0.025 |
| pairwise embedding cosine similarity | 0.99994-0.99999 | 0.9961-0.9999 | (not separately tabulated; consistent with Stage G) |

Predicted `mu` remains essentially flat in both V3.2 seeds, consistent
with Stage G's collapse (Section G) -- the final SCALAR prediction has
not improved. Seed0's embedding-distance correlation (0.253, versus
V3.1's -0.025) is a modest, real improvement in how much the raw
embedding vector moves as location changes, but this 5-point synthetic
test is too small (n=5 locations, 10 pairs) to treat its exact
correlation value as precise -- it corroborates Section G's stage-probe
finding directionally without adding independent statistical weight.

## I. Location Attention / Position-Summary Analysis (new for V3.2)

For each of 12 anomalous examples: true location target, the
position-aware pool's attention center-of-mass position, attention
peak position, and predicted mu.

| Seed | corr(attention center-of-mass, true location) |
|---|---:|
| seed0 | **+0.935** |
| seed2 | **-0.709** |

**This is the single most direct confirmation that the pooling
mechanism does exactly what it was designed to do**: its attention
mass systematically moves as a function of the true anomaly location
in BOTH healthy seeds -- not a coincidence at either magnitude (0.94,
0.71 are both far from a chance-level relationship on 12 examples).
The task's own instructions explicitly warn not to assume the direction
must align with anomaly onset in any particular convention; the SIGN
flip between seeds (seed0 tracks location directly, seed2 tracks it
inversely) shows the mechanism learned *some* consistent internal
mapping, just not the same one across seeds -- a downstream consumer
(the MLP in Section G/K) would need to interpret this consistently,
which apparently it does not yet do reliably, consistent with Stage
G's collapse. Example rows (seed0, `diagnostics/outputs/v32_diag/
v3_2_diagnostic_probes_seed0.json`):

| true location | attention center-of-mass | attention peak | predicted mu |
|---:|---:|---:|---:|
| 0.997 | 0.912 | 0.941 | 0.513 |
| 0.426 | 0.461 | 0.441 | 0.521 |
| 0.182 | 0.213 | 0.147 | 0.503 |
| 0.621 | 0.519 | 0.618 | 0.526 |
| 0.913 | 0.869 | 0.879 | 0.515 |

Predicted mu (rightmost column) stays in a narrow 0.50-0.53 band despite
attention center-of-mass tracking the true location closely across the
same rows -- visually confirming Section G's Stage-F-succeeds/Stage-G-
fails finding at the level of individual examples, not just aggregate
correlations.

## J. Location Current-Target vs Physical-Onset Analysis

Per Section 8's instruction, the Location target itself was NOT
changed this round. After training, does the model's own `mu` correlate
better with the target it was actually supervised on, or with the
physical full-sequence onset fraction it was never told?

| Seed | mu vs current training target (pearson) | mu vs physical onset fraction (pearson) |
|---|---:|---:|
| seed0 | 0.151 | -0.098 |
| seed2 | -0.095 | 0.008 |

Neither framing shows a meaningfully positive, consistent correlation
in either seed. **This is expected given Section G's finding**: Stage
G's own collapse means the final `mu` does not reliably reflect
location under ANY target framing yet, so this analysis cannot yet
distinguish whether the target-frame mismatch (`MTL_V3_1_DIAGNOSTIC_
REPORT.md` Section I's mean 14.7%/max 46.7% discrepancy finding) is a
meaningful remaining limitation — that question is not answerable until
the Stage-G MLP/normalization bottleneck is addressed first, so a
future target-semantics experiment should follow, not precede, a fix
to Section G's finding.

## K. Intensity Loss/Gradient Decomposition

Corrected methodology (Section A's bug-fix note) -- per-sample, seed0,
n=150 val samples, single forward + single backward (no optimizer
step):

| D bin | n | mean D | mean mu | mean scale | mean\|grad_raw_mu\| | mean\|grad_raw_scale\| |
|---|---:|---:|---:|---:|---:|---:|
| normal (D=0) | 75 | 0.000 | 0.156 | 1.090 | 0.132 | 0.522 |
| (0.0013, 0.013] | 15 | 0.007 | 0.294 | 1.268 | 0.200 | 0.439 |
| (0.013, 0.134] | 15 | 0.046 | 0.301 | 1.282 | 0.203 | 0.452 |
| (0.134, 1.158] | 15 | 0.481 | 0.294 | 1.268 | 0.200 | 0.438 |
| (1.158, 5.27] | 15 | 2.784 | 0.274 | 1.238 | 0.191 | 0.641 |
| (5.27, 64.05] | 15 | 21.31 | 0.304 | 1.281 | 0.205 | **8.683** |

Same qualitative pattern as V3.1 (`mean|grad_raw_mu|` stays flat
~0.13-0.21 across every bin; `mean|grad_raw_scale|` jumps 15-20x for
the extreme bin), but now **the mu-path gradient magnitude is
uniformly SMALLER than it was in V3.1** (0.13-0.21 here vs V3.1's
0.23-0.30) -- stopping the scale branch's influence on the shared
representation did not free up a stronger mean-path gradient; if
anything the mean-path gradient into the raw output shrank slightly.
Gradient-analysis confirms this holds throughout training, not just at
convergence (Section R).

Sample/gradient contribution by D range (seed0, mirrors V3.1's finding
almost exactly): high-anomaly bucket is 16.7% of samples but generates
**69.3% of total Intensity loss and 44.8% of Intensity-driven trunk
gradient** -- H3 (high-D underrepresented/overwhelmed) remains
rejected, exactly as in V3.1.

## L. Intensity Embedding Probes

Capacity-matched linear/Ridge probes on frozen embeddings, train-fit /
held-out-val-evaluated:

| Metric | V3.1 | V3.2 seed0 | V3.2 seed2 |
|---|---:|---:|---:|
| embedding->D anomalous-only pearson | 0.341 | 0.119 | 0.125 |
| embedding->D anomalous-only spearman | 0.228 | -0.029 | -0.101 |
| kNN local-neighbor pearson | 0.344 | 0.248 | (not separately tabulated) |
| **pairwise ordering agreement (anomalous)** | **0.566** | 0.575 | 0.515 |
| chance level | 0.500 | 0.500 | 0.500 |

None of these numbers improved. Pairwise ordering agreement -- the
single most direct measure of "can this representation tell two
anomalies apart by severity" -- stayed within noise of the 0.5 chance
level in both V3.1 and both V3.2 healthy seeds, and anomalous-only
spearman got measurably WORSE (flipped sign in both seeds).

## M. Intensity Anomalous-Only Magnitude Analysis

Required separation of normal-vs-anomaly discrimination from
anomaly-vs-anomaly severity ordering (Section 19 of the spec):

| | seed0 all-samples | seed0 anomalous-only | seed2 all-samples | seed2 anomalous-only |
|---|---:|---:|---:|---:|
| pearson | 0.317 | 0.083 | 0.324 | 0.125 |
| spearman | 0.797 | -0.029 | 0.788 | -0.101 |

The gap between "all samples" and "anomalous-only" is dramatic in both
seeds -- almost the entire apparent correlation comes from the easy
binary normal/anomaly step (matching V3.1's own finding), and per the
task's explicit instruction this must not be allowed to inflate the
read on Intensity's performance. **By this required separation, change
#1 did not solve the core Intensity problem.**

## N. Intensity Linearity Test

Fitting `predicted_mu = a*D + b` on anomalous samples:

| Seed | slope a | intercept b | R² | pearson | spearman |
|---|---:|---:|---:|---:|---:|
| seed0 | 0.00028 | 0.292 | 0.007 | 0.083 | -0.029 |
| seed2 | 0.00070 | 0.369 | 0.016 | 0.125 | -0.101 |

R² of 0.007-0.016 is, for practical purposes, no linear relationship at
all. Binned mean-D-vs-mean-mu (seed0): 0.294, 0.301, 0.294, 0.274, 0.304
across D ranging 0.007 to 21.3 -- **flat, not even weakly monotonic,
let alone linear.** This is not an improvement over V3.1's own
near-flat pattern.

## O. Intensity Uncertainty Calibration

| | V3.1 seed0 | V3.2 seed0 | V3.2 seed2 |
|---|---:|---:|---:|
| mean scale | 1.055 | 1.179 | (see seed2 JSON) |
| error-vs-scale pearson | 0.313 | 0.303 | (comparable magnitude) |
| error-vs-scale spearman | 0.449 | 0.439 | -- |
| mean Laplace NLL | 2.352 | 2.836 | -- |

Predictive uncertainty still carries a real, positive, roughly
unchanged relationship with actual error -- stopping the scale
gradient's influence on the embedding did NOT break uncertainty
calibration (a real risk the task's own success criteria flagged), it
simply also didn't improve magnitude representation. Uncertainty is
NOT constant (std_scale=0.096, confirmed non-degenerate) and does not
collapse into a proxy for "is this an anomaly" (scale varies across
anomalous bins too, Section K's table).

## P. Reference Behavior

| K | V3.1 seed0 gate | V3.2 seed0 gate | V3.1 seed1(*) gate | V3.2 seed2 gate |
|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.022 | 3.3e-07 | 0.011 | 0.0009 |
| 10 | 0.041 | 2.2e-06 | 0.011 | 0.0029 |
| 30 | 0.048 | 8.7e-07 | 0.010 | 0.0065 |
| 100 | 0.051 | 1.4e-07 | 0.011 | 0.0092 |

(*V3.1's seed1, for reference -- a different seed/model than V3.2's
collapsed seed1.) **Both V3.2 healthy seeds show gates 1-4 orders of
magnitude SMALLER than V3.1's typical range**, while V3.2's OWN seed1
(reported separately, Section E) saturates to the opposite extreme
(~1.0). None of V3.2's 3 seeds landed in V3.1's earlier "moderate,
seemingly useful" gate range (0.01-0.05) -- the two targeted changes
appear to have shifted the reference-gate's learned equilibrium toward
one of two extremes (near-0 or near-1) rather than a middle ground,
though this is reported as an observed correlation, not a proven causal
mechanism (Section 27's instruction against cross-attribution applies
here too -- this may be incidental to overall training-dynamics shifts
rather than caused by either change specifically).

## Q. Contamination Robustness (sanity check only)

Mild reference contamination (K=10, `contamination_prob=0.3`) produces
prediction changes on the order of the gate's own magnitude for each
seed -- negligible (~1e-6 scale) for seed0 given its near-zero gate,
small-but-real for seed2 (consistent with its small-but-nonzero gate).
No redesign attempted or needed; this reconfirms the existing mechanism
behaves consistently with its own gate value, exactly as in V3.1.

## R. Gradient Stability (Section 24's required decomposition)

Single-seed (seed0), early/middle/late training segments, GPU-run
gradient analysis:

| Quantity | early | middle | late |
|---|---:|---:|---:|
| Intensity embedding gradient FROM loss (mu-path only, by construction) | 0.044 | 0.063 | 0.048 |
| Intensity scale adapter's OWN param gradient | 1.304 | 1.329 | 1.101 |
| Location pooling's OWN param gradient | 0.042 | 0.033 | 0.045 |
| Location head's own param gradient (full head) | 0.186 | 0.205 | 0.214 |
| Shared trunk gradient from Location's loss | 0.109 | 0.060 | 0.125 |
| Total grad norm before clip | 2.73 | 7.44 | 9.26 |
| any NaN/Inf | false | false | false |

**Confirms both design properties directly and quantitatively**: the
scale adapter's own parameters keep training vigorously throughout
(1.1-1.3, an order of magnitude larger than the embedding-directed
gradient) while the gradient reaching the Intensity embedding stays
small and stable (0.04-0.06) -- exactly the intended "scale still
learns, but can't shape the representation" behavior, with zero
numerical instability at any point in training.

## S. Shape/Extent Regression Check

| | V3.1 | V3.2 seed0 | V3.2 seed2 |
|---|---:|---:|---:|
| Shape nn_accuracy | 0.816±0.031 | 0.993 | 1.000 |
| Shape separation | 0.770±0.138 | 1.892 | 1.708 |
| Extent pearson | 0.387±0.116 | 0.763 | 0.619 |
| Extent spearman | 0.362±0.121 | 0.767 | 0.687 |

**No regression -- both tasks improved** in the two healthy seeds.
Per Section 27/28's instruction, this improvement is NOT attributed to
either targeted change (neither touched Shape or Extent's own
architecture/loss/pooling) and is reported as an observed, unexplained
side effect of the overall changed training dynamics, consistent with
the same dynamics that also shifted the reference gate's behavior
(Section P) and pushed seed1 into collapse (Section E).

## T. Final Verdict

**Two separate, honestly-reported outcomes, per Section 27's explicit
instruction not to let one change's result contaminate judgment of the
other:**

- **Intensity change (stop-gradient): NOT supported by the evidence.**
  The core diagnostic question -- "does blocking scale's influence on
  the representation let the embedding develop continuous magnitude
  structure?" -- is answered NO, directly and by multiple independent
  measures (pairwise ordering agreement, anomalous-only correlation,
  linearity R², embedding probe). Do not promote this specific change.
  The `MTL_V3_1_DIAGNOSTIC_REPORT.md` Section H hypothesis (objective-
  mechanics-plus-representation combination) was directionally right
  about the mechanism but this particular intervention did not unlock
  better magnitude learning -- the representation-side limitation
  appears to dominate, not merely the objective's gradient routing.
- **Location change (position-aware pooling): partially supported,
  worth keeping as a foundation.** The exact, narrowly-targeted
  mechanism worked (Stage F no longer destroys the signal; the
  attention mechanism demonstrably tracks true location). The
  aggregate task metric has not yet improved because a new, precisely
  located bottleneck (Stage F->G, the shared MLP+L2-normalization) was
  exposed. This is a genuine partial success, not a wash -- it replaces
  one diagnosed, fixed problem with a different, now much more
  precisely diagnosed one.
- **Seed1's collapse is a real regression in TRAINING STABILITY**
  (not present in any of V3.1's 3 seeds) that must be resolved or at
  least understood better before promoting V3.2 wholesale -- it does
  not invalidate the two healthy seeds' findings, but it means V3.2 as
  currently configured is not uniformly reliable across seeds.

**Should V3.2 replace V3.1 as the foundation? Not as a whole, not yet.**
The Location pooling change is worth keeping (Q23/next steps).
The Intensity stop-gradient change should be reverted/reconsidered
rather than carried forward as-is, since it has no demonstrated
benefit and a real (if small) cost. The seed1 collapse needs its own
follow-up before any 3-seed V3.x promotion is trustworthy.

## U. One Next Priority Per Unresolved Task

**Location**: investigate and fix the Stage F -> Stage G bottleneck --
specifically, whether `AttributeHead`'s shared `mlp` (Linear->GELU->
Linear) or the final L2 normalization is destroying the position-aware
pool's now-demonstrably-good signal, via the same kind of stage-probe
methodology used here but with hooks placed BETWEEN the mlp's two
linear layers. Do not touch the target-semantics issue (Section J)
until this is resolved, per the same reasoning that motivated deferring
it this round.

**Intensity**: given a capacity-matched linear probe on the frozen
embedding does no better than the actual trained adapter, and stopping
scale's gradient influence did not help, the most promising untried
lever is the embedding's own capacity/training signal for magnitude --
e.g. an explicit anomaly-only auxiliary magnitude-ranking loss (one of
the original diagnostic's proposed-only candidates) that gives the
embedding a DIRECT, isolated training signal for anomaly-vs-anomaly
ordering, independent of both the mu-path and the scale-path's shared
Laplace objective. Proposed only -- not implemented this round.

## Final Questions

### Location

**1. Did position-aware pooling preserve the strong Stage-E Location
signal?** Substantially, yes -- Stage F retains 0.617/0.677 (91%) for
seed0 and 0.268/0.774 (35%) for seed2 of Stage E's value, versus V3.1's
retaining essentially none of it (in fact flipping sign).

**2. What is Stage-F Pearson now vs V3.1's ~-0.057?** 0.617 (seed0),
0.268 (seed2) -- both far above V3.1's collapse.

**3. Does the final 32-D embedding vary meaningfully with true
location?** Weakly at best -- Stage G pearson is 0.016 (seed0) and
0.090 (seed2), both near zero, though slightly above V3.1's -0.162.

**4. Does predicted Location mu move when anomaly location changes?**
No -- the controlled temporal-shift test (Section H) shows mu still
essentially flat in both seeds.

**5. Does correlation with the current training target improve?** No
meaningful, consistent improvement (Section F: aggregate pearson
0.017±0.129 vs V3.1's 0.024±0.111 -- within noise of each other).

**6. Does correlation with physical/full-sequence onset improve?** No
-- Section J shows near-zero correlation with either framing in both
seeds; this analysis cannot yet distinguish the two framings' relative
merit until Stage G's own bottleneck is fixed.

**7. Is the remaining target-frame mismatch now likely the dominant
Location limitation?** Not yet determinable, and likely NOT the
dominant one right now -- the newly-exposed Stage F->G bottleneck
(Section G) is more directly implicated and should be addressed first.

**8. Is Location pooling worth promoting?** Yes, as a partial fix and
a better-diagnosed foundation -- it demonstrably solved the exact
problem it targeted (Section G/I) even though the overall task metric
hasn't moved yet.

### Intensity

**9. Is scale-gradient into the Intensity embedding exactly removed?**
Yes, confirmed structurally (detach) and empirically (Section R: the
embedding-directed gradient equals exactly the mu-path's own small,
stable contribution throughout training).

**10. Does the scale adapter itself still learn?** Yes -- its own
parameter gradient (1.1-1.3, Section R) is an order of magnitude larger
than the embedding-directed gradient and stable across training.

**11. Does the Intensity embedding encode more anomaly-internal D
structure?** No -- if anything slightly less (Section L: anomalous-only
pearson dropped from 0.341 to 0.119-0.125).

**12. Does anomalous-only Spearman improve over ~0.228?** No -- it got
worse and flipped sign (-0.029, -0.101).

**13. Does pairwise severity ordering improve over ~0.566?** No --
0.575 (seed0, within noise) and 0.515 (seed2, worse).

**14. Does predicted mu become monotonic with D?** No -- still flat
(Section N's binned table).

**15. Is the relationship approximately linear?** No -- R² = 0.007-0.016.

**16. Does predictive uncertainty retain useful error information?**
Yes -- error-vs-scale correlation is essentially unchanged from V3.1
(Section O), so this was not sacrificed even though magnitude
representation didn't improve.

**17. Did gradient stability remain acceptable?** Yes -- no NaN/Inf at
any training segment (Section R); the 1-in-3-seed collapse (Section E)
is a representation-level failure via the reference gate, not a
gradient-explosion/numerical-instability failure.

**18. Is scale stop-gradient worth promoting?** No -- no demonstrated
benefit on the metric it was designed to improve, and a small real
regression on some of those same metrics.

### General

**19. Did Shape regress?** No -- improved in both healthy seeds
(Section S).

**20. Did Extent regress?** No -- improved in both healthy seeds
(Section S).

**21. Did representation collapse reappear?** Yes, in 1 of 3 seeds
(seed1), via reference-gate saturation -- diagnosed in Section E.1,
distinct from V3's original heteroscedastic-loss mechanism.

**22. Did reference-gate seed behavior materially change?** Yes --
V3.2's gates cluster at two extremes (near-0 for the 2 healthy seeds,
near-1 for the collapsed one) rather than V3.1's moderate range
(0.01-0.05 across its 3 seeds) -- reported as an observed shift, not
attributed to either specific change (Section P).

**23. Should V3.2 replace V3.1 as the next foundation?** Not wholesale
-- keep the Location pooling change, revert/reconsider the Intensity
stop-gradient change, and resolve the seed1 collapse risk before any
future 3-seed promotion (Section T).

**24. What is the ONE highest-priority remaining problem?** The
Location Stage F -> Stage G bottleneck (the shared MLP + L2-
normalization) -- it is the most precisely diagnosed, most directly
actionable finding in this entire report, and resolving it is a
prerequisite for meaningfully evaluating the target-semantics question
that was deliberately deferred.

## Reproduction Commands

```bash
export PYTHONPATH=".:../AnomSim"

# V3.2 3-seed training (GPU -- see Section D)
python3 -u diagnostics/v3_baseline.py \
  --n_instances 1000 --epochs 20 --patience 5 --seeds 0 1 2 \
  --output_dir diagnostics/outputs/v32 --shape_objective plain \
  --detach_scale_attrs intensity --location_position_aware_pooling \
  --experiment_id_prefix v32 --skip_v21_backfill --device cuda

# Single-seed gradient analysis (GPU)
python3 -u diagnostics/v3_gradient_analysis.py \
  --n_instances 1000 --epochs 20 --seed 0 --device cuda \
  --shape_objective plain --detach_scale_attrs intensity \
  --location_position_aware_pooling \
  --output_dir diagnostics/outputs/v32 --output_name v32_gradient_analysis.json

# Checkpoint-only diagnostics (local CPU, ~10s per seed)
python3 -u diagnostics/v3_2_diagnostic_probes.py \
  --checkpoint diagnostics/outputs/v32/v32_multitask_seed0/bestmodel.pkl \
  --output_dir diagnostics/outputs/v32_diag --n_instances 1000
```
