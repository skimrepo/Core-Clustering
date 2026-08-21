# MTL V3 Report

V3 adds OPTIONAL local reference-set conditioning (K in {0,3,10,30} at
training time; K up to 100 tested at inference-only) and probabilistic
(Laplace mean+scale) outputs for Location/Extent/Intensity on top of
V2.1's architecture, plus a redefined, deliberately unbounded Intensity
target D = RMS(realized deviation), replacing V2.3's ordinal objective.
Same shared trunk, same Generic AttributeHeads, same final L2
normalization, Shift-only, same simulator config/range/split as every
prior report in this series. Single seed=0/1/2, n_instances=1000,
epochs=20, CPU.

**Headline finding, stated up front**: V3's shared representation
COLLAPSED during training -- every attribute's output (Shape's embedding,
and every scalar head's predicted mean AND predicted uncertainty) became
essentially constant regardless of input, across all 3 seeds. This is
reported prominently and diagnosed below, not hidden.

## A. Exact Code Changes

```text
NEW:
core_clustering/reference_context.py   -- ReferenceContextEncoder, ContextFusion
core_clustering/prob_heads.py          -- ScalarPredictionAdapter, ShapeUncertaintyAdapter,
                                           laplace_nll, heteroscedastic_weight
core_clustering/models_contrastive_v3.py -- ContrastiveEncoderV3 (reuses V2.1's trunk
                                           and AttributeHead classes UNCHANGED)
core_clustering/dataset_episodic.py    -- EpisodicContrastiveDataset (subclasses
                                           DynamicContrastiveDataset, query generation
                                           untouched), episodic_pad_collate
core_clustering/losses_v3.py           -- ReferenceConsistencyLoss
core_clustering/trainer_contrastive_v3.py -- ContrastiveTrainerV3 (single AdamW)
core_clustering/cli_contrastive_v3.py  -- V3 CLI entry point
diagnostics/v3_baseline.py             -- 3-seed training + evaluation, V2.1 backfill
diagnostics/v3_reeval.py               -- re-evaluation without retraining (quantile-
                                           binned intensity table fix)
diagnostics/v3_gradient_analysis.py    -- early/mid/late trunk+head gradient norms
diagnostics/v3_eval_diagnostics.py     -- reference-sensitivity/contamination/
                                           uncertainty/clustering diagnostics

MODIFIED (additive only):
core_clustering/dataset_dynamic_contrastive.py -- +sigma_ref field (used to derive D)
core_clustering/losses_contrastive.py  -- ShapeContrastiveLoss gained an OPTIONAL
                                           return_per_sample=False param (default
                                           unchanged, used only by V3's heteroscedastic
                                           weighting)

BUG FOUND AND FIXED during this work (see Section K/L):
core_clustering/models_contrastive_v3.py -- fully-padded reference slots (K=0 items,
                                           or K<batch-max-K items) were being run
                                           through the trunk's self-attention, which
                                           produces NaN under torch.no_grad() specifically
                                           for an all-masked row (a real PyTorch fused-
                                           attention-kernel behavior, confirmed via
                                           direct A/B testing). Fixed by never running
                                           genuinely all-padding slots through the trunk.
                                           This was NOT the cause of the collapse
                                           reported below -- collapse happens with
                                           finite losses throughout training.

V1/V2/V2.1/V2.2/V2.2a/V2.3 -- all unchanged, still fully reproducible.
```

## B. Parameter-Count Change

| | V2.1 | V3 |
|---|---:|---:|
| Shared trunk | 294,848 | 294,848 (identical, reused unchanged) |
| All 4 AttributeHeads | 75,264 | 75,264 (identical, reused unchanged) |
| Reference encoder + context fusion | -- | 49,923 |
| Scalar/uncertainty adapters | -- | 231 |
| **Total** | **370,112** | **420,266** (+13.5%) |

## C/D. Baseline vs New Performance (mean ± std, 3 seeds)

| Task | Metric | V2.1 | V3 | Change |
|---|---|---:|---:|---|
| Shape | nn_accuracy | 0.984 ± 0.008 | **0.647 ± 0.073** | large regression |
| Shape | separation | 1.540 ± 0.104 | **0.000 ± 0.000** | **total collapse** |
| Location | pearson | 0.001 ± 0.008 | 0.018 ± 0.099 | still no signal, now noisier |
| Location | spearman | -0.005 ± 0.003 | 0.049 ± 0.093 | still no signal, now noisier |
| Extent | pearson | 0.221 ± 0.090 | 0.064 ± 0.150 | regression |
| Extent | spearman | 0.213 ± 0.058 | 0.050 ± 0.118 | regression |
| Intensity | pearson | 0.654 ± 0.097 | 0.024 ± 0.231 | large regression, sign-inconsistent across seeds |
| Intensity | spearman | 0.834 ± 0.067 | 0.046 ± 0.208 | large regression |
| Intensity | mae | 0.588 ± 0.074 | 2.093 ± 0.287 | worse (different scale -- D is unbounded, see Section I) |

Every task regressed. Shape's separation is EXACTLY 0.000 (to displayed
precision) at all 3 seeds -- not "weaker," but a complete, reproducible
collapse.

## E/F. Reference-Count Robustness / Reference-Subset Sensitivity

K in {0, 3, 10, 30, 100}, 40 val queries, 5 independent reference draws
each (seed0 checkpoint):

| K | std(location_mu) across draws | std(extent_mu) | std(intensity_mu) | mean gate | mean pred. change from resampling |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | 0.0 | 3.7e-9 | 1.9e-9 | 1.4e-9 | 6.6e-9 |
| 10 | 0.0 | 2.1e-9 | 1.0e-9 | 3.2e-9 | 3.3e-9 |
| 30 | 0.0 | 3.3e-9 | 2.2e-9 | 3.8e-9 | 5.7e-9 |
| 100 | 0.0 | 2.4e-9 | 4.2e-10 | 4.1e-9 | 3.0e-9 |

**This is NOT meaningful robustness.** Every value here is floating-point
noise (1e-9 scale). The model is "insensitive to reference resampling"
only because it is ALSO insensitive to the query itself (Section K) -- the
learned gate is essentially zero even for K>0, so references contribute
nothing regardless of which ones are drawn. K=0 "remains usable" only in
the degenerate sense that it is indistinguishable from every other K.

## G. Reference Contamination Robustness

K=10, contamination_prob=0.3 (mean 2.8/10 references contaminated), 40
queries: mean absolute prediction change from contamination is
2.98e-9-4.28e-9 across all three scalar heads -- again floating-point
noise, not evidence of genuine robustness to contaminated references
(there is no learned signal for contamination to disturb in the first
place).

## H. Uncertainty Calibration Diagnostics

| Task | error-vs-uncertainty pearson | spearman | mean uncertainty | mean NLL | 50%/80%/95% coverage |
|---|---:|---:|---:|---:|---|
| Location | 0.088 | 0.063 | 0.250 | 0.329 | 0.36 / 0.76 / 1.00 |
| Extent | 0.049 | 0.091 | 0.113 | -0.449 | -- |
| Intensity | -0.132 | 0.139 | 1.726 | 2.684 | 0.81 / 0.87 / 0.90 |

All error-vs-uncertainty correlations are near zero (no real relationship
-- consistent with Section K: uncertainty is a constant number per task,
so it cannot correlate with anything). Location's 50% coverage (0.36,
should be ~0.5) reflects a constant mu poorly centered against location's
actual uniform[0,1] distribution, not a calibration property worth trusting.

## I. Intensity Target-vs-Prediction Analysis

D (realized deviation, unbounded, scales with each instance's own signal
amplitude -- observed range in this val set: 0 to ~64) vs predicted mu,
quantile-binned (seed0):

| D bin | count | mean target D | mean predicted mu | mean predicted scale |
|---|---:|---:|---:|---:|
| normal (D=0) | 75 | 0.0 | 0.4669 | 0.9392 |
| (0.056, 0.28] | 15 | 0.126 | 0.1589 | 1.974 |
| (0.28, 2.83] | 15 | 1.187 | 0.1589 | 1.974 |
| (2.83, 78.85] | 15 | 16.37 | 0.1589 | 1.974 |

(Seed0's full 6-bin table also splits the small end further; all bins show
the same pattern.) **mean_predicted_mu is flat to 4+ significant figures
across bins spanning D=0 to D=16+.** The old 0-2 normalized-distance
ceiling from V1/V2 is indeed gone (mu is unbounded in principle, confirmed
at the unit level in Section 9's tests) -- but this is moot, because mu
never moves at all in response to D. **Intensity is not "compressed", it
is simply not being predicted.**

## J. Gradient Stability Comparison

Shared-trunk / own-head gradient norm, mean (max) over 15 sampled batches
per segment, seed0:

| Task | Trunk early | Trunk middle | Trunk late | Head early | Head middle | Head late |
|---|---:|---:|---:|---:|---:|---:|
| Shape | 0.0066 (0.013) | 0.0050 (0.013) | **0.0015 (0.0028)** | 0.057 (0.067) | 0.070 (0.114) | **0.0011 (0.0016)** |
| Location | 0.031 (0.053) | 0.056 (0.493) | 0.0015 (0.0038) | 0.192 (0.467) | 0.291 (1.243) | 0.139 (0.363) |
| Extent | 0.024 (0.070) | 0.030 (0.075) | 0.0018 (0.0031) | 0.184 (0.432) | 0.414 (1.039) | 0.266 (0.437) |
| Intensity | 0.015 (0.057) | 0.071 (0.470) | 0.0010 (0.0029) | 0.294 (0.769) | 0.419 (1.911) | 0.173 (0.598) |

No NaN/Inf at any point (confirmed separately from the training run's own
epoch-by-epoch finite, smoothly-decreasing loss across all 20 epochs, all
3 seeds -- the collapse is a stable convergence, not a numerical blowup).
The reference-conditioning-related NaN bug found and fixed earlier
(Section A) is unrelated to this collapse.

**Important nuance, stated precisely because it does NOT simply confirm
the intuitive story**: Shape's own trunk gradient is NOT the largest at
any sampled point -- if anything it is consistently the SMALLEST of the
four (0.0066/0.0050/0.0015 vs Location/Extent/Intensity's 0.015-0.071 at
early/middle). What IS distinctive about Shape is that BOTH its trunk
gradient AND its head gradient collapse to near-zero by late training
(0.0015 and 0.0011) while Location/Extent/Intensity's HEAD gradients stay
substantial even at late training (0.14-0.27) -- their heads are still
being pushed to fit something, even as ALL FOUR tasks' TRUNK gradients
converge to a similarly small ~0.001-0.002 by late training. This is
consistent with Shape settling into a stable, low-local-gradient
degenerate optimum (Section K) while the shared trunk stops carrying a
strong differentiating signal from any task by late training -- but the
raw gradient-MAGNITUDE data alone does NOT show Shape's own contribution
as the dominant one at the three sampled points, so the causal chain in
Section K should be read as a plausible, partially-evidenced mechanism,
not a fully proven one.

## K. Regressions and Root-Cause Diagnosis

**Every task regressed. Root cause, confirmed by direct inspection of the
seed0 checkpoint's actual outputs (not just aggregate metrics):**

```text
Shape embedding: cosine similarity between ANY two val instances ≈ 1.0
  (std ≈ 5e-8) -- every shape embedding is the same point on the unit sphere.
Shape uncertainty scale: mean 2.514, std 7.3e-8 -- also constant.
Location scale: mean 0.250, std 2.0e-6 -- constant.
Extent scale: mean 0.113, std 4.5e-7 -- constant.
Intensity mu: mean 0.103, std 6.3e-8 -- constant, REGARDLESS of D (0 to 64+).
Intensity scale: mean 1.726, std 4.4e-7 -- constant.
Reference gate (K>0): ≈0 -- the model learned to ignore references entirely.
```

Every head's mean AND uncertainty output is a constant, to 6-8 decimal
places, independent of the actual input. This is not "weak learning" --
it is a complete collapse of the shared representation that every head
reads.

**Mechanistic explanation.** The Shape heteroscedastic loss
(`base_shape_loss_i / b_i + log(b_i)`, exactly as specified) has an
UNBOUNDED degenerate minimum: for a FIXED per-anchor base loss `c`, the
optimal `b_i = c`, giving a minimized value of `1 + log(c)` -- which
diverges to **negative infinity as c -> 0**. Since embedding COLLAPSE
(every embedding identical) trivially drives the underlying contrastive
loss `c` toward a small constant (positive and negative pair distances
both -> 0), the heteroscedastic wrapper rewards collapsing the Shape
embedding with an UNBOUNDED loss decrease, not a bounded one -- there is
no floor. This gives Shape's gradient an extremely strong, likely
dominant pull toward collapsing its own embedding.

Because Section 6's context-fusion design (explicitly requested) feeds
the SAME fused trunk representation (`H_fused`) into ALL FOUR heads before
branching, ANY mechanism that drives `H_fused` toward a degenerate,
input-independent state would corrupt every head's input simultaneously
-- which is consistent with what is actually observed (Location, Extent,
and Intensity all constant too, despite their own Laplace NLL losses
having no reason to individually want a collapsed representation).

**This part of the causal chain is a plausible hypothesis, not a fully
proven one** -- Section J's direct gradient measurements do NOT show
Shape's raw trunk-gradient magnitude as the largest of the four at any
sampled point (it is consistently the smallest at early/middle, and by
late training ALL FOUR tasks' trunk gradients have converged to a
similarly small ~0.001-0.002). What Section J DOES show distinctively for
Shape is that both its trunk AND head gradients collapse to near-zero by
late training, while Location/Extent/Intensity's head gradients remain
substantial throughout even as their trunk gradients also shrink. Two
readings are both consistent with the full evidence and cannot be
distinguished with the measurements taken this round: (a) Shape's
degenerate optimum was reached early via a large gradient THIS session's
10%/50%/90% sampling simply did not land on, pulling the shared trunk
into collapse before the "early" checkpoint was even measured, or (b) the
shared trunk's gradient was already generically weak across all four
tasks for reasons independent of Shape (a milder version of the
persistent "trunk gradient dilution across 4 tasks" pattern this whole
report series has tracked since V2.1, `MTL_V2_REPORT.md` Section 10), and
Shape's unbounded degenerate minimum simply reached a stable (low-
gradient) collapsed point fastest once in that weak-signal regime, without
being the primary CAUSE of the trunk's weak signal in the first place.
The heteroscedastic Shape loss's unbounded minimum (an unambiguous,
verified mathematical property, and the only one of the four tasks with
such a property) remains the most likely SINGLE differentiating factor
given no other task's loss has this risk, but this report stops short of
claiming the gradient data alone proves the full causal chain from
Shape's loss to every other task's collapse.

The reference-context/gate mechanism did not protect against this: the
gate learned to shrink toward zero (Section E/F), which is CONSISTENT with
collapse (once `H_fused` carries no useful information regardless of
source, there is no benefit to attending to references either) but not
informative on its own as a separate finding.

## L. Overall Verdict and Answers

**1. Did probabilistic predictions improve or hurt each task?** Hurt --
every task's metric regressed, and the probabilistic heads' own outputs
(mu and scale) are both constant, so "probabilistic" in name only for this
run.

**2. Is Intensity now approximately monotonic/linear with raw realized
deviation?** No. mu is flat regardless of D (Section I).

**3. Did the old 0-2 normalized-distance ceiling disappear?** Structurally
yes (the adapter has no such ceiling, confirmed at the unit level) -- but
this is moot given Finding K.

**4. Does uncertainty correlate with actual prediction difficulty/error?**
No (Section H, all correlations near zero) -- expected, since uncertainty
is a constant number per task.

**5. Does uncertainty react sensibly to stochastic waveform families?**
Not assessed in depth -- not meaningful to assess given Finding K (a
constant cannot react to anything). Marked NOT MEANINGFULLY TESTABLE
this round, not "passed" or "failed."

**6. How sensitive is the model to which reference subset was selected?**
Not meaningfully sensitive (Section E/F) -- but only because it is equally
insensitive to everything else, not because it achieved genuine
reference-robustness.

**7. Does K=0 remain usable?** Trivially yes, in the sense that it is
indistinguishable from any other K -- not a meaningful pass.

**8. Does increasing K generally help?** No measurable effect in either
direction (Section E/F).

**9. Does mild reference contamination cause graceful degradation?**
No measurable effect (Section G) -- again because there is nothing left
to degrade.

**10. Did gradient stability remain acceptable?** See Section J once
complete; preliminarily, training losses stayed finite and decreased
smoothly for all 20 epochs at all 3 seeds (no NaN/Inf, no divergence) --
the collapse is a smooth, stable convergence to a degenerate optimum, not
a numerical blowup.

**11. Is the optional reference context worth keeping?** Cannot be judged
on this run -- the mechanism itself (K=0 hard-gated, weighted mean/
variance, gated fusion) was unit-tested and works correctly in isolation
(`tests/test_reference_context.py`, all passing); it was never given a
fair test here because the shared representation it operates on collapsed
for an unrelated reason (Shape's loss). Worth re-testing once that is
addressed.

**12. Is there evidence the latent space supports future unlabeled
normal-cluster discovery?** No, not from this run -- the clustering sanity
check's 0.62 label agreement (barely above chance for 2 clusters) reflects
KMeans splitting a cloud of near-identical points, not a meaningful
discovery result.

**PROMOTE / KEEP / VERDICT: KEEP V2.1. Do not promote V3 in its current
form.** The architecture additions (reference conditioning, probabilistic
heads) are plausibly sound in isolation (all pass unit tests, including
adversarial edge cases like K=0/mixed-K/all-padding batches), but the
specific heteroscedastic Shape loss formulation used here has an unbounded
degenerate minimum that collapsed the ENTIRE shared representation across
all 3 seeds, taking every other task down with it. This is diagnosed with
reasonable confidence (direct inspection of collapsed outputs + a
mechanistic explanation consistent with this whole report series' prior
findings about Shape's gradient dominance), but per the task's own
instructions, no fix (bounding the heteroscedastic term, removing it,
detaching Shape's contribution to the shared trunk, etc.) has been
implemented or even selected as "the" next step below -- only diagnosed.

## Next Single Priority

*(Proposed only, per instructions not to auto-implement a fix or start a
new architecture/hyperparameter search.)* Re-run this exact experiment
with ONLY the Shape loss reverted to its non-heteroscedastic form (plain
`ShapeContrastiveLoss`, no `heteroscedastic_weight` wrapper, everything
else in V3 unchanged) to directly test whether the collapse is specifically
attributable to the heteroscedastic Shape term, isolating this ONE
variable before considering any other change to the reference-conditioning
or probabilistic-head machinery, both of which pass their own unit tests
and have not yet been evaluated on a non-collapsed shared representation.

## Files Changed

See Section A.

## Reproduction Command

```bash
export PYTHONPATH=".:../AnomSim"

# V3 3-seed baseline + V2.1 backfill (K capped at 30 for training, per
# Section 61's resource note -- K=100 tested only at eval time)
python3 -u diagnostics/v3_baseline.py \
  --n_instances 1000 --epochs 20 --patience 5 --seeds 0 1 2 \
  --k_regimes 0 3 10 30 --device cpu --output_dir diagnostics/outputs/v3 --force

# Re-evaluate with fixed quantile-based intensity binning (no retraining)
python3 -u diagnostics/v3_reeval.py --seeds 0 1 2 --output_dir diagnostics/outputs/v3

# Gradient stability (Section J)
python3 -u diagnostics/v3_gradient_analysis.py \
  --n_instances 1000 --epochs 20 --seed 0 --device cpu --k_regimes 0 3 10 30 \
  --output_dir diagnostics/outputs/v3

# Reference-sensitivity / contamination / uncertainty / clustering (Sections E-H, L.12)
python3 -u diagnostics/v3_eval_diagnostics.py \
  --checkpoint diagnostics/outputs/v3/v3_multitask_seed0/bestmodel.pkl \
  --n_instances 1000 --output_dir diagnostics/outputs/v3
```

Full V3 training run: ~510s/seed x 3 seeds (~25 min total) on CPU, plus
V2.1 backfill (~15-45s x 2 seeds). All diagnostics complete in under 5
minutes combined.
