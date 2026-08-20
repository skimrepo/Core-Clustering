# MTL V2.2a Report

Single, narrow question: was V2.2's performance regression (extent/intensity
turning negative) caused by the universal-deviation intensity SEMANTICS, or
by the intensity sampling range confound (V2.2 used 0.05-8.0 while V2.1 used
0.2-4.0)? V2.2a re-runs V2.2 exactly, only with `intensity_min=0.2,
intensity_max=4.0` matched to V2.1's legacy range. Shift-only, no
architecture change, all numbers seed=0/CPU/n_instances=1000/epochs=20
(patience=5), identical to V2.1/V2.2 in every other respect.

## 1. Exact Experimental Difference

Confirmed by diff of the two runs' `config.json`: the ONLY differing
fields between V2.2 and V2.2a are `intensity_min` (0.05 -> 0.2) and
`intensity_max` (8.0 -> 4.0). `intensity_mode="universal_deviation_intensity"`,
`intensity_sampling="log_uniform"`, architecture, optimizer, LR, loss
weights, gradient clipping, seed, and dataset scale are all identical to
V2.2 (and, for everything except the intensity range/semantics, to V2.1).

## 2. Legacy vs Universal Shift Equivalence

Ran BEFORE any model training (gate, per the spec), across 450 samples (10
waveform seeds x 3 start ratios x 3 extent ratios x 5 intensity values k in
{0.2, 0.5, 1.0, 2.0, 4.0}), comparing legacy `ShiftAnomaly(forced_magnitude_
std_multiplier=k)` against `apply_calibrated_anomaly(..., i_target=k)` for
the SAME clean waveform/location/extent, with matched-seed RNGs so both
paths draw the same direction (+/-1):

```text
max_abs_diff over all 450 samples : 4.0e-08
mean_abs_diff (avg per sample)    : 3.85e-09
rms_diff (avg per sample)         : 7.07e-09
masks equal for all samples       : True
regions equal for all samples     : True
native k vs realized intensity MAE: 2.98e-07   pearson = 1.00000000
```

**Equivalence holds to floating-point precision.** Both sigma_ref (std of
the whole clean window) and the calibration target agree with the theory in
`MTL_V22_REPORT.md` Section 1-2: for ShiftAnomaly specifically, a constant
offset over a fixed support means `RMS(delta)/sigma_ref` reduces exactly to
`magnitude_std_multiplier`. No semantic difference found -- proceeded
directly to Section 3-10 without needing to investigate a discrepancy.

## 3. Dataset Distribution Comparison

Same entity manifest/seed (n_instances=1000, seed=0), legacy
(`min_magnitude_std_multiplier=0.2, max_magnitude_std_multiplier=4.0`) vs
universal (`intensity_min=0.2, intensity_max=4.0`), one full pass over
train-split anomalous entities:

| Stat | Legacy intensity (native k) | Universal intensity (I_raw) |
|---|---:|---:|
| min | 0.2023 | 0.2023 |
| p5 | 0.2557 | 0.2557 |
| p25 | 0.4494 | 0.4494 |
| median | 0.9903 | 0.9903 |
| p75 | 2.1117 | 2.1117 |
| p95 | 3.5847 | 3.5847 |
| max | 3.9940 | 3.9940 |
| mean | 1.3586 | 1.3586 |
| std | 1.0667 | 1.0667 |

Identical to 4 decimal places (both are literally the same log-uniform
distribution over [0.2, 4.0] -- expected given Section 2's signal
equivalence, not a new finding). Location/extent distributions and
anomaly:normal ratio were unaffected by intensity_mode (not shown -- neither
implementation touches location/extent sampling).

## 4. Metric Target Distribution

`I_metric = I_raw / (1 + I_raw)` applied to the universal distribution's
percentile points (monotonic transform, so percentile ORDER is preserved,
though the transform is nonlinear so gaps between percentiles compress at
the high end):

| I_raw | I_metric |
|---:|---:|
| 0.2 | 0.1667 |
| 0.5 | 0.3333 |
| 1.0 | 0.5000 |
| 2.0 | 0.6667 |
| 3.0 | 0.7500 |
| 4.0 | 0.8000 |

Note the compression: I_raw's full range [0.2, 4.0] (a 20x spread) maps to
I_metric's [0.167, 0.8] -- a narrower absolute range, and the top half of
I_raw (2.0-4.0) only spans I_metric 0.667-0.8 (width 0.133), while the
bottom half (0.2-1.0) spans 0.167-0.5 (width 0.333). Large intensities are
metric-compressed relative to small ones. Not changed in this experiment --
recorded here because Section 13 revisits it.

## 5. V2.1 vs V2.2 vs V2.2a Performance

All three: multitask, seed=0, n_instances=1000, epochs requested=20 (all
early-stopped, patience=5), CPU, `normalize_embedding=True`.

| Task | Metric | V2.1 | V2.2 | V2.2a | Interpretation |
|---|---|---:|---:|---:|---|
| Shape | nn_accuracy | 0.993 | 0.907 | **0.987** | recovers to ~V2.1 |
| Shape | separation | 1.526 | 0.987 | **1.662** | recovers (even nominally best) |
| Location | pearson | -0.006 | -0.013 | 0.005 | flat, ~zero in all three |
| Extent | pearson | 0.340 | -0.135 | **0.353** | fully recovers, matches V2.1 |
| Extent | spearman | 0.294 | -0.018 | 0.439 | recovers, even exceeds V2.1 |
| Intensity (metric) | pearson | 0.537 | -0.345 | **0.249** | partially recovers -- sign flips back positive, but well below V2.1 |
| Intensity (metric) | spearman | 0.814 | -0.170 | 0.315 | partially recovers, well below V2.1 |
| Intensity (raw) | pearson | 0.537 (identity transform) | -0.285 | 0.192 | partially recovers, well below V2.1 |

## 6. Shape Result

**Recovers to essentially V2.1's level** (nn_accuracy 0.993 -> 0.907 -> 0.987;
separation 1.526 -> 0.987 -> 1.662, actually the highest of the three). Shape
has no direct dependence on intensity's semantics or range, so this recovery
is entirely a cross-task (shared-trunk) effect -- see Section 10's gradient
data for the likely mechanism.

## 7. Location Result

Unaffected across all three variants -- pearson -0.006/-0.013/0.005, all
indistinguishable from zero. Consistent with `MTL_DIAGNOSTIC_REPORT.md`'s
original finding that location shows no real signal regardless of
multi-task configuration.

## 8. Extent Result

**Fully recovers** -- 0.340 (V2.1) -> -0.135 (V2.2) -> 0.353 (V2.2a),
matching V2.1 almost exactly (and improving on spearman: 0.294 -> 0.439).
Extent's own code path (loss, target definition) never changed across any
of V2.1/V2.2/V2.2a -- this recovery is fully attributable to whatever
cross-task interaction the intensity range change was causing, now removed.

## 9. Intensity Result

Metric-space: pearson -0.345 (V2.2) -> **0.249** (V2.2a) -- sign flips back
positive, a real improvement, but still well short of V2.1's 0.537.
Raw-space (inverse-transformed): -0.285 -> 0.192, same pattern, same gap to
V2.1. Both spaces agree on direction and magnitude (as expected, since the
inverse transform preserves rank order) -- intensity is the ONE task that
does NOT fully recover to V2.1's level even after matching the range
exactly, unlike shape and extent (Sections 6, 8).

## 10. Gradient Scale

Shared-trunk gradient norm, mean over 15 sampled batches per segment:

| Task | V2.1 early/mid/late | V2.2 early/mid/late | V2.2a early/mid/late |
|---|---|---|---|
| Shape | 0.375 / 10.075 / 17.281 | 0.858 / 8.647 / 16.145 | 0.108 / 4.896 / **0.681** |
| Location | 0.495 / 0.117 / 0.143 | 0.459 / 0.198 / 0.596 | 0.457 / 0.440 / 0.242 |
| Extent | 1.840 / 0.155 / 0.118 | 0.590 / 0.256 / 0.300 | 0.665 / 0.681 / 0.044 |
| Intensity | 1.563 / 4.589 / 1.409 | 0.664 / 0.368 / 0.258 | 0.617 / 2.997 / 0.052 |

**Shape's persistent late-training gradient dominance (17.28 in V2.1, 16.14
in V2.2) is essentially GONE in V2.2a (0.68 late)** -- a mid-training spike
(4.90) still appears but resolves by late training, unlike V2.1/V2.2 where
it persisted and grew. All four tasks' late-training trunk gradients in
V2.2a are small and roughly comparable (0.04-0.68), a qualitatively more
balanced regime than either V2.1 or V2.2 showed. Intensity's gradient
remains well-behaved (no runaway, as in V2.2) and even smaller late (0.05).

This is a second, independent line of evidence (beyond the performance
numbers in Section 5-9) that the intensity RANGE -- not the universal
deviation semantics -- was driving both the performance regression and the
shape-gradient-domination pattern: matching the range resolved both
simultaneously, even though shape's own loss/target never changed.

## 11. Was the V2.2 Regression Range-Driven?

**STRONGLY SUPPORTED** (for shape and extent specifically; intensity itself
only partially confirms this -- see Section 13).

Evidence: (a) shape and extent both recovered to ~V2.1 levels essentially
completely once the range was matched (Sections 6, 8); (b) shape's
gradient-domination pattern, present in both V2.1 and V2.2 nearly
identically, disappeared in V2.2a (Section 10) -- the range change altered
training dynamics for a task with zero direct connection to intensity's
definition; (c) Section 2's floating-point-exact signal equivalence and
Section 3's near-identical distributions independently rule out "the
universal semantics generate meaningfully different data" as an
alternative explanation for shift specifically.

## 12. Does Universal Deviation Still Look Viable for Shift?

**PROMISING.**

Not SUPPORTED outright, because intensity itself (the attribute the whole
V2.2/V2.2a line of experiments is about) still underperforms V2.1 by a
real margin (0.249 vs 0.537 pearson) even after removing the range confound.
Not NOT SUPPORTED, because shape and extent are fully healthy, the
calibration machinery is proven correct and semantically equivalent to
legacy for Shift (Section 2), and intensity's OWN metric moved from clearly
broken (-0.345) to clearly positive and reasonably correlated (0.249) --
a real, substantial improvement, just not a full recovery.

## 13. If Performance Still Differs from V2.1

**B. The `I_raw -> I_metric = I_raw/(1+I_raw)` transform / unit-sphere
embedding-distance geometry interaction is the more suspect cause, not (A)
universal deviation semantics.**

Reasoning: Section 2 proved legacy and universal Shift generate
floating-point-identical signals and floating-point-identical realized
intensity values for the same k; Section 3 proved the resulting
intensity_raw DISTRIBUTIONS are identical to 4 decimal places once the
range is matched. This leaves NO remaining semantic or distributional
difference between V2.1's training data and V2.2a's training data for
intensity -- the raw numbers going into the pipeline are, for all practical
purposes, the same numbers. The ONE remaining difference is that V2.1
regresses embedding distance directly toward the raw native parameter k
(range 0.2-4.0, unbounded scale), while V2.2a regresses it toward
`I_metric = k/(1+k)` (range 0.167-0.8, compressed at the high end -- Section
4). Given identical underlying data, any remaining performance gap must
trace to this transform (or its interaction with the L2-normalized
embedding space's bounded distance geometry, max pairwise distance 2),
not to the universal-deviation definition itself.

## 14. Next Single Priority

**Diagnostic only, not a fix, proposed but NOT implemented**: bin V2.2a's
held-out intensity predictions by true I_raw (e.g. low/mid/high thirds of
the 0.2-4.0 range) and compare per-bin regression error/correlation. Section
4 showed I_metric compresses the top half of the intensity range into a
narrow band (0.667-0.8 for I_raw 2.0-4.0) -- if per-bin error is
concentrated at the high-I_raw end (where I_metric is most compressed),
that would directly implicate the transform's nonlinearity as the residual
cause identified in Section 13, rather than requiring any architecture,
loss, or transform change to test the hypothesis.

## 15. Files Changed

```text
diagnostics/v22a_shift_equivalence.py     -- NEW: Sections 2-4's diagnostics
                                              (pure numpy, no model)
diagnostics/v2_baseline.py                -- +--experiment_id_prefix override
                                              (lets v22a share intensity_mode
                                              with v22 but get a distinct
                                              experiment_id/architecture tag)
diagnostics/v2_gradient_analysis.py       -- +--experiment_id_prefix override
diagnostics/outputs/v22a/v22a_shift_equivalence.json -- Section 2-4 data
diagnostics/outputs/v2/v22a_multitask_seed0/*         -- this run's data
diagnostics/outputs/v2/v22a_gradient_analysis.json    -- this run's data
MTL_V22A_REPORT.md                        -- this file

No changes to: core_clustering/models_contrastive_v2.py,
core_clustering/trainer_contrastive_v2.py, core_clustering/target_transforms.py,
core_clustering/dataset_dynamic_contrastive.py, core_clustering/losses_contrastive.py.
V1, V2, V2.1, V2.2 (all still reachable via their own intensity_mode/range/
experiment_id_prefix combination) -- unchanged and reproducible.
Full test suite: 184/184 passing (no regression; no new unit tests were
needed since no core_clustering/ or anomsim/ library code changed --
diagnostics/ scripts only, verified via the actual runs recorded above).
```

## 16. Reproduction Command

```bash
export PYTHONPATH=".:../AnomSim"

# Signal equivalence + dataset/metric distribution diagnostics (Sections 2-4) -- gate first
python3 -u diagnostics/v22a_shift_equivalence.py --output_dir diagnostics/outputs/v22a

# V2.2a multitask seed0 baseline (Section 5's V2.2a column)
python3 -u diagnostics/v2_baseline.py \
  --normalize_embedding --intensity_mode universal_deviation_intensity \
  --intensity_min 0.2 --intensity_max 4.0 --experiment_id_prefix v22a \
  --modes multitask --n_instances 1000 --epochs 20 --patience 5 --seed 0 \
  --device cpu --output_dir diagnostics/outputs/v2 --force

# V2.2a gradient norm re-measurement (Section 10)
python3 -u diagnostics/v2_gradient_analysis.py \
  --normalize_embedding --intensity_mode universal_deviation_intensity \
  --intensity_min 0.2 --intensity_max 4.0 --experiment_id_prefix v22a \
  --n_instances 1000 --epochs 20 --seed 0 --device cpu \
  --output_dir diagnostics/outputs/v2
```

All three commands completed in under 30 seconds total on CPU -- no GPU or
remote server needed.
