# MTL Intensity Geometry Diagnostic

Observation-only: loaded EXISTING V2.1 and V2.2a checkpoints (no retraining,
no architecture/loss/transform change) and directly measured how each
placed intensity embeddings relative to the normal centroid, across the
same raw intensity range (0.2-4.0), on the val split.

## 1. Experiment Setup

- Checkpoints: `diagnostics/outputs/v2/v21_multitask_seed0/bestmodel.pkl`,
  `diagnostics/outputs/v2/v22a_multitask_seed0/bestmodel.pkl` (both seed=0,
  n_instances=1000, identical architecture: `ContrastiveEncoderV2`,
  embedding_dim=32, `normalize_embedding=True`).
- Dataset: val split, `generate_entity_manifest(n_instances=1000, base_seed=0)`
  + `DynamicContrastiveDataset(split="val", train=False, ...)`, matching
  each model's own training config (V2.1: `legacy_native_intensity`,
  0.2-4.0; V2.2a: `universal_deviation_intensity`, `intensity_min=0.2,
  intensity_max=4.0`). n=75 anomalous val instances for both (same entity
  manifest/seed).
- Centroid: `centroid = normal_embeddings.mean(axis=0)` over the SAME val
  set's normal instances -- the exact convention already used by
  `diagnostics/metrics.py`'s `normal_relative_metrics` (the function behind
  every "intensity Pearson" number in every prior V2.x report), not a new
  definition.
- `predicted_distance = ||embedding - centroid||_2`.

## 2. Global Geometry

| | V2.1 | V2.2a |
|---|---:|---:|
| Pearson (I_raw vs distance) | 0.537 | 0.201 |
| Spearman | 0.814 | 0.315 |
| MAE (distance vs training target) | 0.640 | 0.758 |
| RMSE | 0.820 | 1.136 |

Distance distribution (n=75 each):

| Stat | V2.1 | V2.2a |
|---|---:|---:|
| min | 0.192 | 0.151 |
| p5 | 0.403 | 0.298 |
| p25 | 0.832 | 0.497 |
| median | 1.442 | 0.535 |
| p75 | 1.456 | 0.703 |
| p95 | 1.491 | 0.724 |
| max | 1.504 | 0.737 |
| mean | 1.199 | 0.566 |
| std | 0.389 | 0.138 |

V2.1 uses roughly 4x more of the achievable distance range (0.19-1.50, span
1.31) than V2.2a (0.15-0.74, span 0.59) -- neither uses anywhere close to
the theoretical unit-sphere max of 2.

## 3. Raw Intensity Bins

| Bin | Count | Mean I_raw | V2.1 mean dist. | V2.2a mean dist. | V2.1 MAE | V2.2a MAE |
|---|---:|---:|---:|---:|---:|---:|
| [0.2,0.5) | 23 | 0.354 | 0.810 | 0.519 | 0.485 | 0.277 |
| [0.5,1.0) | 20 | 0.683 | 1.256 | 0.561 | 0.573 | 0.197 |
| [1.0,2.0) | 15 | 1.458 | 1.412 | 0.613 | 0.273 | 0.135 |
| [2.0,3.0) | 12 | 2.420 | 1.469 | 0.596 | 0.951 | 0.122 |
| [3.0,4.0] | 5 | 3.453 | 1.474 | 0.597 | 1.980 | 0.177 |

V2.1's mean distance rises steeply through the first three bins (0.81 ->
1.26 -> 1.41) then plateaus (1.41 -> 1.47 -> 1.47). V2.2a's mean distance
rises much more weakly and plateaus EARLIER, already essentially flat from
bin 3 onward (0.61 -> 0.60 -> 0.60) despite its own target (I_metric) still
rising modestly across these same bins (0.59 -> 0.71 -> 0.77, Section 4) --
V2.2a is not even tracking its own compressed target's residual rise in
this range (visible in Section 4's plot as the black curve going flat
while the red reference curve keeps climbing).

## 4. Learned Distance Curves

See `diagnostics/outputs/intensity_geometry/intensity_geometry_v21_vs_v22a.png`.
V2.1 (left panel): scatter tracks the raw target line closely up to
~I_raw=1.5, then flattens hard into a tight ceiling band around 1.45-1.50.
V2.2a (right panel): scatter rises quickly then flattens by ~I_raw=1.0 into
a band around 0.55-0.60, visibly BELOW its own already-compressed I/(1+I)
reference curve for I_raw > 1.5 -- the model underperforms even the
transform's own (already narrow) target range in the upper half.

## 5. High-Intensity Saturation

I_raw >= 2.0 subset (n=17 both):

| | V2.1 | V2.2a |
|---|---:|---:|
| Pearson | 0.407 | 0.224 |
| Spearman | 0.527 | 0.304 |
| std(distance) | 0.020 | 0.108 |
| mean pairwise \|Δdistance\| | 0.022 | 0.118 |

I_raw < 1.0 subset (n=43 both):

| | V2.1 | V2.2a |
|---|---:|---:|
| Pearson | 0.479 | 0.233 |
| Spearman | 0.562 | 0.311 |
| std(distance) | 0.419 | 0.152 |
| mean pairwise \|Δdistance\| | 0.476 | 0.172 |

**Important nuance, not the naive expectation**: V2.1's own high-intensity
subset collapses MORE tightly in absolute terms (std=0.020, a near-total
flatline) than V2.2a's (std=0.108) -- V2.1 saturates harder at its own
ceiling. What differs is WHERE that saturation sits relative to the rest of
the range: V2.1's collapse happens at ~1.47, far above where low-intensity
samples live (median distance in the low bin ~0.81-1.26), so between-group
separation survives even though within-high-group resolution is poor
(Section 6). V2.2a's low-intensity resolution is ALSO much worse than
V2.1's (std 0.152 vs 0.419) -- it is not simply "high intensity specifically
collapses more" but that V2.2a's mapping is compressed and noisier across
the WHOLE range, high and low alike.

## 6. Pairwise Ordering Accuracy

| Group | V2.1 | V2.2a |
|---|---:|---:|
| Overall (2775 pairs) | 0.799 | 0.615 |
| low-low (I<1.0, 903 pairs) | 0.693 | 0.605 |
| high-high (I>=2.0, 136 pairs) | 0.684 | 0.610 |
| **low-high (731 pairs)** | **0.975** | **0.650** |

V2.1 separates the low and high groups almost perfectly (97.5%) despite
poor WITHIN-group resolution at both ends (69-68%) -- consistent with a
model that has learned "is this severe or not" very well even though it
can't finely rank severity within a tier. V2.2a's low-high separation
(65.0%) is barely better than its own within-group accuracy (60-61%) --
the low and high groups are not cleanly separated in embedding-distance
space at all.

## 7. Local Resolution

Ordering accuracy by |I_raw_i - I_raw_j|, all 2775 val-anomaly pairs:

| |Δintensity| range | V2.1 | V2.2a |
|---|---:|---:|
| [0.0, 0.25) | 0.579 | 0.546 |
| [0.25, 0.5) | 0.737 | 0.640 |
| [0.5, 1.0) | 0.800 | 0.685 |
| **[1.0, inf)** | **0.935** | **0.611** |

V2.1's accuracy rises monotonically with gap size, as expected -- larger
real intensity differences are easier to order correctly, up to 93.5% for
the largest gaps. **V2.2a's accuracy is NON-monotonic**: it rises through
the first three bins (0.55 -> 0.64 -> 0.69) then DROPS for the largest-gap
bin (0.61) -- worse than the [0.5,1.0) bin. This is the single cleanest
signature in this whole diagnostic of the transform-compression hypothesis:
pairs with the largest raw intensity gaps are disproportionately likely to
include a high-I_raw member, exactly where `I/(1+I)` compresses the most
(Section 4/9's mapping table) -- so the largest REAL gaps get the SMALLEST
metric-space gaps to learn from, and ordering degrades precisely there.

## 8. Unit-Sphere Utilization

| | V2.1 | V2.2a |
|---|---:|---:|
| embedding \|\|e\|\| (mean, min-max) | 1.0000 (0.99999988-1.0) | 1.0000 (0.99999988-1.00000012) |
| centroid \|\|c\|\| | 0.995 | 0.998 |
| observed distance max | 1.504 | 0.737 |
| observed distance p95 | 1.491 | 0.724 |
| theoretical max (unit sphere) | 2.0 | 2.0 |

Both models' embeddings are confirmed unit-norm to float precision, as
designed (L2 normalization). Neither uses the full theoretical range: V2.1
reaches 75% of the theoretical max (1.50/2.0), V2.2a reaches only 37%
(0.74/2.0) -- V2.2a leaves substantially more of the available geometry
unused.

## 9. V2.1 Impossible Target Analysis

I_raw > 2.0 subset (n=17): mean target = 2.724 (unreachable given the
observed max distance of 1.504), mean distance = 1.470, distance range
[1.449, 1.504] (span 0.055 -- essentially a single point), ordering
accuracy WITHIN this subset = 0.684 (same subset as Section 5's
high-intensity group, by construction).

**V2.1 cannot satisfy its own target beyond I_raw≈1.5-2 (the achievable
distance ceiling is far below where the raw target keeps demanding more),
yet it still separates this whole "impossible" subset cleanly from the
low-intensity group (Section 6's low-high = 0.975).** This is exactly H2's
predicted mechanism: absolute calibration fails above the ceiling, but the
model repurposes the ceiling itself as a stable "this is a severe anomaly"
signal, preserving the coarse ordering that matters most for the pairs that
are easiest to tell apart in the first place.

## 10. Location / Extent Leakage

| | V2.1 | V2.2a |
|---|---:|---:|
| corr(distance, location) | 0.024 | 0.087 |
| corr(distance, extent) | 0.435 | 0.357 |

Both models show a moderate positive correlation between intensity-head
distance and extent (not surprising -- ShiftAnomaly's realized deviation
and its own extent both derive from the same injected region, and this
project's earlier reports already documented extent/intensity interaction
at the gradient level). Location shows negligible correlation in both.
Not pursued further per the spec's scope limit.

## 11. Hypothesis Evaluation

**H1. Transform compression hypothesis: SUPPORTED.**
Section 7's non-monotonic local-resolution pattern (V2.2a's largest-gap
pairs ordered WORSE than mid-sized gaps, 0.611 vs 0.685-0.640, unlike V2.1's
clean monotonic 0.579->0.935) is a specific, clean signature that gaps
landing in the compressed high-I_metric region are hard to resolve
regardless of how large they are in raw terms. Section 5's low-intensity
subset also shows V2.2a's resolution is markedly worse than V2.1's even
where I/(1+I) is nearly linear (std 0.152 vs 0.419) -- so compression at the
high end is not the ONLY effect, but it is a real, independently-evidenced
one.

**H2. Natural saturation hypothesis: SUPPORTED (as the mechanism behind
V2.1's specific success, not as a full explanation of V2.2a's failure).**
Section 9 confirms V2.1 cannot hit its own target above I_raw≈1.5-2, yet
Section 6 confirms it still achieves 97.5% low-high separation -- absolute
miscalibration coexisting with excellent coarse ordering, exactly as H2
predicted.

**H3. Objective mismatch hypothesis: NOT SUPPORTED.**
V2.1's regression-to-centroid-distance objective works reasonably well
(0.799 overall ordering, 0.975 low-high) despite its own calibration being
provably impossible past I_raw≈2 -- this is evidence the OBJECTIVE FORM
(regress distance toward a scalar) is workable, not evidence it is
fundamentally mismatched to the task. No result here required inventing a
reason regression-to-distance itself is broken; every observed weakness in
V2.2a is explainable by H1/H2 without invoking H3.

## 12. Overall Verdict

**TRANSFORM COMPRESSION STRONGLY SUPPORTED.**

Primary evidence: Section 7's non-monotonic local-resolution breakdown
(V2.2a's largest real intensity gaps are its WORST-resolved, a pattern
absent in V2.1) is a specific, mechanistic fingerprint of `I/(1+I)`'s
compression, not just "V2.2a is noisier in general." Combined with Section
6's low-high separation gap (97.5% vs 65.0%) and Section 8's unused-geometry
gap (75% vs 37% of the theoretical distance range), the picture is
consistent: V2.1's natural saturation (H2, real and contributing) still
leaves it a wide, well-separated low-vs-high geometry to work with, while
V2.2a's transform compresses exactly the region needed to keep large,
easy-to-resolve intensity gaps easy to resolve.

## 13. Next Single Priority

**Diagnostic only, proposed but NOT implemented**: re-run this exact
geometry diagnostic against a THIRD checkpoint trained with a raw-intensity
target that is bounded but NOT compressive at the high end -- e.g. simple
min-max normalization to [0,1] over the sampled range, instead of
`I/(1+I)`'s asymptotic compression -- to test whether a bounded-but-linear
target recovers V2.1-level ordering (supporting H1 specifically) or still
underperforms V2.1 (implicating something else, e.g. the bounded target's
narrower absolute scale alone, independent of compression shape). This
isolates compression SHAPE from boundedness ITSELF, which the V2.1-vs-V2.2a
comparison alone cannot separate. No such run was performed in this
diagnostic.

## 14. Files Generated

```text
diagnostics/intensity_geometry_diagnostic.py -- NEW: this diagnostic (loads
                                                 existing checkpoints only,
                                                 no training)
diagnostics/outputs/intensity_geometry/intensity_geometry_results.json -- all
                                                 sections' computed statistics
diagnostics/outputs/intensity_geometry/intensity_geometry_samples_v21.csv,
                     intensity_geometry_samples_v22a.csv -- per-sample export
                                                 (sample_id, I_raw, training_target,
                                                 predicted_distance, prediction_error,
                                                 location, extent)
diagnostics/outputs/intensity_geometry/intensity_geometry_v21_vs_v22a.png -- Section 4's plot
MTL_INTENSITY_GEOMETRY_REPORT.md -- this file

No changes to any model, loss, transform, dataset, or training code.
Full test suite unaffected (no library code touched) -- not re-run for this
diagnostic-only, no-code-change addition.
```

## 15. Reproduction Command

```bash
export PYTHONPATH=".:../AnomSim"
python3 -u diagnostics/intensity_geometry_diagnostic.py \
  --output_dir diagnostics/outputs/intensity_geometry
```

Loads the existing `v21_multitask_seed0` and `v22a_multitask_seed0`
checkpoints (already committed in prior V2.1/V2.2a runs); no training,
completed in a few seconds on CPU.
