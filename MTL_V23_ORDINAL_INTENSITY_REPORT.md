# MTL V2.3 Ordinal Intensity Report

Tests whether replacing intensity's bounded-transform absolute regression
(V2.2a's `I_raw` -> `I/(1+I)` -> distance regression) with a generic,
ORDER-only radial loss lets the model learn severity ordering directly,
without ever being told to encode `I_raw`'s numeric scale as embedding
distance. Architecture identical to V2.1/V2.2a. Shift-only. Seed=0,
n_instances=1000, epochs=20/patience=5, CPU -- same as every prior V2.x
baseline in this line of experiments.

## 1. Exact Change

```text
Unchanged: architecture (ContrastiveEncoderV2, all 4 AttributeHeads,
           embedding_dim=32, final L2 normalization on ALL heads including
           intensity), optimizer, LR, gradient clipping, loss weights,
           Shape/Location/Extent losses, universal I_raw definition
           (RMS(delta over support)/sigma_ref), Shift-only training,
           intensity sampling range (0.2-4.0, log-uniform).

Changed:   intensity's loss module only.
           V2.2a: NormalRelativeRegressionLoss, target = I_raw/(1+I_raw)
           V2.3:  RadialOrdinalLoss (new, generic), target = I_raw directly
                  (intensity_metric_transform="identity" -- the bounded
                  transform infrastructure is NOT deleted, just unused here)
```

Config-selectable via `intensity_objective` ("radial_regression" default =
V1-V2.2a unchanged, "radial_ordinal" = V2.3) and
`intensity_metric_transform` (None = auto-derived as before, "identity" =
V2.3's explicit override). V1/V2/V2.1/V2.2/V2.2a all remain reproducible
via their own flag combination.

## 2. RadialOrdinalLoss

```python
centroid = normal_emb.mean(dim=0)                      # same convention as NormalRelativeRegressionLoss
normal_pull = ((normal_emb - centroid) ** 2).sum(-1).mean()

y = zeros(batch); y[is_anomalous] = value[is_anomalous]  # normal y=0 by construction
s = (embeddings - centroid.detach()).norm(dim=-1)        # severity score, ALL samples

y_diff = y_i - y_j     # for every pair (i,j)
s_diff = s_i - s_j
valid  = |y_diff| > eps                                  # excludes ties (all normal-normal pairs)
rank_loss = mean( softplus( -sign(y_diff) * s_diff ) )   # over valid pairs

loss = rank_loss + normal_pull
```

Implemented in `core_clustering/losses_contrastive.py`, generic (not named
or scoped to "intensity") -- reusable for any future positive-unbounded
scalar attribute with a normal reference class. Same stop-gradient
centroid convention as `NormalRelativeRegressionLoss`
(`centroid.detach()` for the severity score; live `centroid` for
`normal_pull`, mirroring that loss's own `centroid`/`centroid.detach()`
split exactly). No margin, no scale hyperparameter, no
`|I_i - I_j|` weighting -- only the SIGN of each pair's raw gap is used.

## 3. Scale-Invariance Tests

`tests/test_losses_contrastive.py`, 9 new tests, all passing:

- Correct ordering (severity increasing with value) gives strictly lower
  loss than the same embeddings with severity/value pairing reversed.
- **Raw-scale invariance**: targets `[0,1,2]` and `[0,100,200]` (same
  order, same embeddings) give `torch.isclose` loss values -- proves the
  ranking term depends only on `sign(y_i - y_j)`, never on `|y_i - y_j|`.
- Tied targets (two anomalies sharing the same raw value) contribute
  nothing to the ranking term regardless of how far apart they sit in
  embedding space.
- Normal-clustering behavior matches `NormalRelativeRegressionLoss`
  (tighter normal cluster -> lower loss, all else fixed).
- No NaN/Inf across a batch spanning 3 orders of magnitude (0.1 to 50)
  plus an exact tie.
- Gradient reaches the embeddings tensor directly.
- `MultiHeadContrastiveLoss(intensity_objective=...)` correctly selects
  `NormalRelativeRegressionLoss` (default) or `RadialOrdinalLoss`.

Full suite: 193/193 passing (no regression to V1/V2/V2.1/V2.2/V2.2a).

## 4. Learned Severity Geometry

See `diagnostics/outputs/v23/v23_severity_curve.png`. **The learned curve
is essentially FLAT across the entire I_raw range (0.2-4.0)**, sitting at
~1.45-1.47 for every anomaly regardless of severity, with normal severity
near 0. Visually, V2.3 learned something close to a two-level STEP function
(normal ≈0, anomaly ≈1.46) rather than a smooth monotonic ramp -- the
opposite of the "0.2σ < 1σ < 5σ < 50σ" fine-grained ordering the design was
aiming for.

## 5. Intensity Ordering

| Metric | Value |
|---|---:|
| Pearson (I_raw vs severity) | 0.199 |
| Spearman | 0.355 |
| Kendall tau | 0.277 |
| Overall pairwise ordering accuracy | 0.639 (2775 pairs) |

Spearman/Kendall are both positive and clearly above chance (0), so SOME
real monotonic signal exists -- but see Section 8: it lives inside an
extremely narrow severity band, which is why the ordering accuracy is much
weaker than V2.1's (Section 10).

## 6. Quantile Ordering

Val I_raw split into tertiles (boundaries 0.525, 1.411):

| Group pair | Accuracy | N pairs |
|---|---:|---:|
| low-low | 0.593 | 300 |
| mid-mid | 0.530 | 300 |
| **high-high** | **0.497** (chance) | 300 |
| low-mid | 0.616 | 625 |
| mid-high | 0.746 | 625 |
| low-high | 0.696 | 625 |

Within-tertile ordering is weak to non-existent (high-high is literally at
chance, 49.7%) -- almost all resolvable signal is in distinguishing BROAD
tertiles from each other (mid-high, low-high), not in fine-grained ordering
within a severity band.

## 7. Gap Resolution

|I_i - I_raw_j| split into tertiles (boundaries 0.412, 1.323):

| Gap bucket | Accuracy | N pairs |
|---|---:|---:|
| small_gap | 0.579 | 925 |
| medium_gap | 0.625 | 925 |
| large_gap | 0.711 | 925 |

Monotonically improving with gap size (as expected -- bigger real
differences are easier to catch even through a compressed mapping), but
even the LARGEST-gap tertile only reaches 71.1% -- well below V2.1's
large-gap accuracy of 93.5% (`MTL_INTENSITY_GEOMETRY_REPORT.md` Section 7).

## 8. Normal vs Anomaly Geometry

| | Normal (n=75) | Anomaly (n=75) |
|---|---:|---:|
| min | 0.00005 | 1.396 |
| median | 0.00154 | 1.453 |
| max | 0.00697 | 1.522 |
| mean | 0.00180 | 1.467 |
| std | 0.00136 | 0.027 |

**fraction(anomaly severity > normal severity) = 1.000 (perfect
separation).** But the anomaly distribution's own spread (std=0.027 over a
[1.396, 1.522] range, span 0.126) is almost as tight, in absolute terms, as
the normal cluster's own spread -- ALL anomalies, from the weakest (I_raw
≈0.2) to the strongest (I_raw ≈4.0) sampled, collapsed into a single narrow
band. This is the single clearest number in this whole diagnostic.

## 9. Geometry Utilization

Embedding norms confirmed unit-norm to float precision (mean 1.0, std
~4e-8), as designed. Centroid norm 0.99999750. Severity (anomaly) distance
range [1.396, 1.522] uses only ~6% of the theoretical [0,2] unit-sphere
range -- less absolute range than EITHER V2.1 (used 75%, `MTL_
INTENSITY_GEOMETRY_REPORT.md` Section 8) or V2.2a (used 37%). V2.3 uses
the LEAST of the available geometry of any variant measured, despite (or
perhaps because of) having no absolute-scale target constraining it at all.

## 10. Other Task Performance

| Task | Metric | V2.1 | V2.2a | V2.3 |
|---|---|---:|---:|---:|
| Shape | nn_accuracy | 0.993 | 0.987 | **1.000** |
| Shape | separation | 1.526 | 1.662 | 1.967 |
| Location | pearson | -0.006 | 0.005 | 0.014 |
| Extent | pearson | 0.340 | 0.353 | **0.453** |

Shape and extent both look BEST under V2.3 by their headline metrics --
but see Section 11's gradient data before reading this as an unqualified
win.

## 11. Gradient Scale

Shared-trunk gradient norm, mean over 15 sampled batches per segment:

| Task | V2.1 e/m/l | V2.2a e/m/l | V2.3 e/m/l |
|---|---|---|---|
| **Shape** | 0.375 / 10.075 / 17.281 | 0.108 / 4.896 / 0.681 | 0.858 / 0.549 / **26.465** |
| Location | 0.495 / 0.117 / 0.143 | 0.457 / 0.440 / 0.242 | 0.668 / 1.282 / 0.587 |
| Extent | 1.840 / 0.155 / 0.118 | 0.665 / 0.681 / 0.044 | 0.365 / 0.265 / 0.220 |
| Intensity | 1.563 / 4.589 / 1.409 | 0.617 / 2.997 / 0.052 | 0.853 / 0.104 / **3.136** |

**Intensity gradient does NOT blow up with raw unbounded I_raw** (early
0.85 -> mid 0.10 -> late 3.14) -- confirms the ordinal design's core
premise that using only the SIGN of the raw gap, never its magnitude,
prevents the ~300x/10,000x runaway seen in V2/earlier. This part of the
hypothesis holds.

**But shape's late-training gradient (26.46) is the LARGEST seen across
EVERY V2.x variant measured so far** (V2.1: 17.28, V2.2: 16.14, V2.2a:
0.68). Shape's own metric didn't regress (Section 10), but the underlying
gradient-dominance pathology this whole line of reports has tracked is not
resolved by the ordinal objective -- if anything it is worse here than in
any prior variant except the original V2.1/V2.2.

## 12. Hypothesis Evaluation

**"Unbounded physical deviation does not need to be numerically encoded as
embedding distance. A bounded latent severity geometry can learn its
monotonic ordering directly."**

**PARTIAL.**

Supporting evidence: a real, positive, statistically distinguishable-from-
zero monotonic relationship IS learned (Spearman 0.355, Kendall 0.277,
gap-resolution accuracy rising monotonically with gap size 0.579->0.711),
without any absolute-scale target and without the gradient-runaway failure
mode of naive raw-value regression (Section 11).

Against full support: the geometry that emerged is not the graded
"0.2σ<1σ<5σ<50σ" severity ladder the design intended -- it is closer to a
binary step (normal vs. anomalous, perfectly separated) with the
CONTINUOUS severity-among-anomalies signal almost entirely collapsed
(Section 8's std=0.027 across the full sampled range). And critically, this
weaker, collapsed ordering is measurably WORSE by every ordering metric
than V2.1's ORIGINAL radial-regression approach achieves as a side effect
of its own (unrelated) absolute-calibration objective (Section 10 of
`MTL_INTENSITY_GEOMETRY_REPORT.md`: V2.1 overall accuracy 0.799 vs V2.3's
0.639; V2.1 low-high 0.975 vs V2.3's 0.696).

## 13. Overall Verdict

**KEEP RADIAL REGRESSION.**

The generic ordinal loss is correctly implemented, scale-invariant as
designed, and does avoid intensity's own gradient-runaway failure mode --
real, verified wins for the MECHANISM. But the actual severity geometry it
produces is a worse ordering tool than V2.1's original absolute-distance
regression, which -- despite being unable to satisfy its own target above
I_raw≈2 (`MTL_INTENSITY_GEOMETRY_REPORT.md` Section 9) -- still separates
and orders intensity levels substantially better in every metric measured
here. Given this project's own principle of reporting honest, sometimes
disappointing results rather than promoting a mechanism because it is
newer or more principled, V2.1's simpler approach is the one to keep as
the reference point for intensity specifically, at least until Section 15's
proposed next step is investigated.

## 14. Most Important Finding

**The ordinal objective solved the EASY sub-problem (is this anomalous at
all?) essentially perfectly (fraction_anomaly_gt_normal = 1.000) while
almost completely failing to solve the HARD sub-problem this whole
intensity investigation exists for (how severe is this anomaly?) --
anomaly severities collapsed into a band spanning only ~6% of the available
embedding-distance geometry, regardless of whether the true I_raw was 0.2
or 4.0.** This is a specific, measurable form of failure distinct from
V2.2/V2.2a's "transform compression" story (there is no transform here at
all) -- it looks more like an optimization-landscape effect: normal-vs-
anomaly pairs (all of which are individually "easy" once any anomaly drifts
away from the normal cluster) may dominate what the pairwise ranking
objective actually optimizes for, at the expense of the finer, harder
anomaly-vs-anomaly ordering sub-problem.

## 15. Next Single Priority

**Diagnostic only, proposed but NOT implemented**: instrument
`RadialOrdinalLoss` (or a standalone analysis script) to separately report
the mean per-pair-type ranking loss and its relative contribution to the
gradient for normal-vs-anomaly pairs versus anomaly-vs-anomaly pairs across
training, to test the mechanism proposed in Section 14 -- if normal-vs-
anomaly pairs' loss saturates near zero early while contributing a large
share of total pairs, that would directly confirm they are "solved for
free" and crowding out the anomaly-vs-anomaly signal, motivating (in a
LATER, separate step) something like reporting the two pair types' losses
independently rather than only their sum. No such instrumentation,
reweighting, or loss change was implemented in this report.

## 16. Files Changed

```text
core_clustering/losses_contrastive.py      -- +RadialOrdinalLoss (generic);
                                               MultiHeadContrastiveLoss gained
                                               intensity_objective param
                                               (default "radial_regression"
                                               unchanged; "radial_ordinal" new)
core_clustering/trainer_contrastive_v2.py  -- +intensity_objective passthrough
core_clustering/cli_contrastive_v2.py      -- +--intensity_objective flag
core_clustering/dataset_dynamic_contrastive.py -- +intensity_metric_transform
                                               override (None=auto, preserves
                                               V2.2/V2.2a; "identity" for V2.3)
diagnostics/phase1_baselines.py            -- build_loaders passthrough list
                                               gained intensity_metric_transform
diagnostics/v2_baseline.py                 -- +--intensity_objective,
                                               --intensity_metric_transform;
                                               fixed a real bug in
                                               evaluate_intensity_dual (the
                                               [0,1) clip was being applied
                                               even for identity transform,
                                               producing NaN Pearson/Spearman
                                               for V2.3's raw-space eval --
                                               caught and fixed this session)
diagnostics/v2_gradient_analysis.py        -- +--intensity_objective,
                                               --intensity_metric_transform
diagnostics/v23_ordinal_intensity_diagnostic.py -- NEW: Sections 5-9's
                                               quantile/percentile-based
                                               ordering diagnostics (no
                                               fixed-threshold assumptions,
                                               no reference curve)
tests/test_losses_contrastive.py           -- +9 tests for RadialOrdinalLoss
tests/test_dataset_dynamic_contrastive.py  -- +1 test for intensity_metric_transform
diagnostics/outputs/v2/v23_multitask_seed0/*, v23_gradient_analysis.json -- this run
diagnostics/outputs/v23/*                  -- ordinal diagnostic outputs + plot
MTL_V23_ORDINAL_INTENSITY_REPORT.md        -- this file

Full test suite: 193/193 passing. V1/V2/V2.1/V2.2/V2.2a all unchanged and
reproducible via their own intensity_objective/intensity_metric_transform
defaults.
```

## 17. Reproduction Command

```bash
export PYTHONPATH=".:../AnomSim"

# V2.3 multitask seed0 baseline
python3 -u diagnostics/v2_baseline.py \
  --normalize_embedding --intensity_mode universal_deviation_intensity \
  --intensity_min 0.2 --intensity_max 4.0 --intensity_metric_transform identity \
  --intensity_objective radial_ordinal --experiment_id_prefix v23 \
  --modes multitask --n_instances 1000 --epochs 20 --patience 5 --seed 0 \
  --device cpu --output_dir diagnostics/outputs/v2 --force

# V2.3 gradient norm re-measurement (Section 11)
python3 -u diagnostics/v2_gradient_analysis.py \
  --normalize_embedding --intensity_mode universal_deviation_intensity \
  --intensity_min 0.2 --intensity_max 4.0 --intensity_metric_transform identity \
  --intensity_objective radial_ordinal --experiment_id_prefix v23 \
  --n_instances 1000 --epochs 20 --seed 0 --device cpu \
  --output_dir diagnostics/outputs/v2

# V2.3 ordinal intensity geometry diagnostic (Sections 4-9)
python3 -u diagnostics/v23_ordinal_intensity_diagnostic.py \
  --output_dir diagnostics/outputs/v23
```

All three completed in under a minute total on CPU -- no GPU or remote
server needed.
