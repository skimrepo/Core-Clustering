# MTL V2.2 Report

Single change tested: replace ShiftAnomaly's native generator parameter
(magnitude_std_multiplier) with a type-agnostic, post-hoc-measured
"universal realized deviation" as the intensity training label. Architecture
is byte-for-byte V2.1 (final embedding L2 normalization on). All numbers
below are seed=0, CPU, n_instances=1000, epochs=20/patience=5 -- identical
to V2.1's own baseline in every respect except intensity label semantics
(and, as an acknowledged confound, the intensity sampling range -- see
Section 3/10).

## 1. Universal Intensity Definition

```text
delta[t]   = x_anom[t] - x_clean[t]            (only defined/used over the anomaly's own support M)
sigma_ref  = std(x_clean over its valid region)  (see Section 2)

I_raw      = sqrt( mean_{t in M}( (delta[t] / sigma_ref)^2 ) )
           = RMS(delta over M) / sigma_ref
```

Implemented exactly as `anomsim.anomalies.base.compute_realized_intensity`
(RMS via `compute_rms`, divided by `sigma_ref`). RMS (not SUM) over the
support region only -- deliberately duration-independent, see Section 7.

## 2. Reference Scale

```python
def compute_reference_scale(clean, valid_mask=None, eps=1e-8):
    x = clean.reshape(-1)
    if valid_mask is not None:
        x = x[valid_mask.reshape(-1) > 0.5]
    return max(std(x), eps)
```

Investigated existing simulator conventions first, per the spec: ShiftAnomaly
(the only type actually wired into `DynamicContrastiveDataset`) already
computed `clean_std = float(window[0].std())` where `window` is the FULL
clean series passed to `.apply()` -- i.e. std over the entire clean baseline,
not just the anomaly's support region. This is exactly the convention the
spec asked to prefer ("clean waveform의 valid 영역 전체에서 계산한 standard
deviation"), so `compute_reference_scale` formalizes what ShiftAnomaly
already did as a shared, type-agnostic helper rather than introducing a new
convention. `apply_calibrated_anomaly` computes `sigma_ref` from `Z` (the
clean counterfactual `.apply()` itself returns for that window), never from
the anomaly-injected signal -- not circular.

## 3. Native Parameter vs Universal Intensity

| Anomaly type | Native parameter (generation-time) | Used as training target? |
|---|---|---|
| shift (only type wired into the model pipeline) | `magnitude_std_multiplier` (log-uniform, legacy default 0.2-4.0) | Legacy mode: YES. V2.2: NO |
| spike | `scale` (stddev of one point's perturbation) | No (not used in this pipeline) |
| noise | `scale` (stddev over a region) | No |
| cutoff | none (uniform-random replacement value) | No |
| scale | `scale` (multiplicative-ratio stddev) | No |
| wander | `scale` (ramp target stddev) | No |
| contextual | `scale` (affine coefficient stddev) | No |
| upsidedown | none (reflects about local mean) | No |

Only `shift` is used by `DynamicContrastiveDataset` (the actual training
pipeline); the other 7 types above were exercised only by the calibration
diagnostics (Section 5-8) to confirm the calibration machinery is genuinely
type-agnostic, not because the model pipeline uses them. In V2.2,
`ShiftAnomaly` is still constructed with a native parameter
(`forced_magnitude_std_multiplier=1.0`), but this value is now an arbitrary
shape-only placeholder -- since ShiftAnomaly's perturbation is a constant
offset with no temporal-shape variation to preserve, its specific numeric
value is irrelevant after calibration overwrites the actual magnitude. The
REAL magnitude comes entirely from the sampled `I_target` and
`apply_calibrated_anomaly`'s rescaling, never from this native parameter.

## 4. Calibration Procedure

```python
Y_raw, Z, mask = anomaly.apply(Y, start, end, rng)      # type-specific shape only
support = mask < 0.5
sigma_ref = compute_reference_scale(Z)
delta_raw = Y_raw - Z
r_raw = compute_rms(delta_raw, support)
scale = sigma_ref * i_target / (r_raw + eps)
delta_scaled = delta_raw * scale                          # single positive scalar -- shape preserved exactly
Y_calibrated = Z + delta_scaled
i_realized = compute_realized_intensity(delta_scaled, support, sigma_ref)
```

`apply_calibrated_anomaly` (anomsim/anomalies/base.py) wraps this generically
around ANY `BaseAnomaly` subclass's `.apply()` output -- no anomaly-type
branching anywhere in the calibration code itself.

## 5. Calibration Accuracy

Per type, across 75 samples each (5 I_target buckets: very_small=0.1,
small=0.5, medium=1.5, large=4.0, very_large=8.0, x 5 base-waveform seeds x
3 apply seeds):

| Type | MAE | Mean rel. error | Pearson | Spearman | Mean ratio realized/target |
|---|---:|---:|---:|---:|---:|
| spike | 0.00000 | 0.00000 | 1.000000 | 1.000000 | 1.00000 |
| noise | 0.00000 | 0.00000 | 1.000000 | 1.000000 | 1.00000 |
| cutoff | 0.00000 | 0.00000 | 1.000000 | 1.000000 | 1.00000 |
| scale | 0.00000 | 0.00000 | 1.000000 | 1.000000 | 1.00000 |
| wander | 0.00000 | 0.00000 | 1.000000 | 1.000000 | 1.00000 |
| contextual | 0.00000 | 0.00000 | 1.000000 | 1.000000 | 1.00000 |
| upsidedown | 0.00000 | 0.00000 | 1.000000 | 1.000000 | 1.00000 |
| shift | 0.00000 | 0.00000 | 1.000000 | 1.000000 | 1.00000 |

**Important framing**: this near-perfect match is an ALGEBRAIC GUARANTEE of
the calibration formula (Section 4 solves for `scale` exactly so that
`RMS(delta_scaled)/sigma_ref == i_target` up to floating-point precision),
not an empirically discovered result -- it would be true for any anomaly
type with a nonzero raw perturbation regardless of how "good" the underlying
generator is. What this diagnostic actually validates is the IMPLEMENTATION
(no degenerate-`r_raw` failure, no sign/shape bug, works uniformly across 8
structurally different generation mechanisms without any type-specific
code), not a "surprising" empirical property.

## 6. Cross-Type Consistency

At I_target=1.0 and I_target=3.0, across all 8 types (5 waveform seeds
each): cross-type std of per-type mean realized intensity = 0.00000 for
both targets; max absolute deviation from target across any type = 0.00000.
Same caveat as Section 5 -- this equality is the direct algebraic
consequence of calibration, not a discovered property, and it confirms
"same I_target -> same realized deviation strength regardless of anomaly
type" holds in the implementation exactly as designed.

## 7. Extent Leakage

Correlational (extent varies naturally across random samples for the 7
types without a `forced_region` option; Pearson between realized extent
ratio and relative bias `(I_realized - I_target)/I_target`, 32 samples/type
at I_target=2.0): all types show mean relative bias of 0.00000 (to 5
decimals) regardless of extent, with correlation coefficients (-0.21 to
+0.36) that are not meaningful given the near-zero, noise-floor bias values
they're computed against.

Controlled (`shift`, the only type supporting `forced_region`, extent ratio
0.033 / 0.100 / 0.133 of a 300-step series, I_target=2.0): mean realized
intensity = 2.00000 at ALL three extents. **No extent leakage** -- exactly
as the RMS-based (not SUM-based) definition is designed to guarantee.

## 8. Shape Preservation

Cosine similarity between `delta_raw` (pre-calibration) and `delta_scaled`
(post-calibration), over the support region, across all 8 types (5 samples
each): mean and min cosine similarity = **1.00000000** for every type.
Exactly as expected -- calibration is a pure positive-scalar rescale, so
temporal shape/sign is preserved identically; this is a correctness check,
not a new finding.

## 9. Intensity Target Transform

`core_clustering/target_transforms.py`, `ScalarMetricTargetTransform`:

```python
forward(raw):   raw / (1 + raw)          # [0, inf) -> [0, 1)
inverse(metric): d / (1 - d)   where d = min(metric, 1 - eps)
```

Applied inside `DynamicContrastiveDataset._inject` via a per-attribute mode
lookup (`"identity"` for location/extent, `"positive_unbounded_to_unit"` for
intensity when `intensity_mode="universal_deviation_intensity"`) -- no
`if attribute == "intensity"` branching anywhere in architecture/trainer/
loss code. Applied to the REALIZED intensity (`I_raw`, what actually ended
up in the data), not the sampled target -- the model must be trained on what
is actually true of the generated instance, not what generation intended.
`losses_contrastive.py` is completely unchanged: `NormalRelativeRegressionLoss`
just regresses embedding distance toward whatever scalar value the dataset
hands it, with zero awareness of this transform.

## 10. V2.1 vs V2.2 Performance

Both rows: multitask, seed=0, n_instances=1000, epochs requested=20 (both
early-stopped, patience=5), CPU, `normalize_embedding=True` in both.

| Task | Metric | V2.1 | V2.2 | Change |
|---|---|---:|---:|---|
| Shape | nn_accuracy | 0.993 | 0.907 | worse |
| Shape | separation | 1.526 | 0.987 | worse |
| Location | pearson | -0.006 | -0.013 | roughly flat (both ~zero) |
| Location | mae | 0.241 | 0.273 | slightly worse |
| Extent | pearson | 0.340 | **-0.135** | **large regression, sign flip** |
| Extent | spearman | 0.294 | -0.018 | large regression |
| Intensity (metric-space) | pearson | 0.537 | -0.345 | **large regression, sign flip** |
| Intensity (raw-space) | pearson | n/a (identity transform in V2.1) | -0.285 | -- |

**Acknowledged confound**: V2.2's intensity sampling range
(`intensity_min=0.05, intensity_max=8.0`, log-uniform) differs from V2.1's
legacy range (`min_magnitude_std_multiplier=0.2, max_magnitude_std_multiplier=4.0`)
-- this was NOT held constant, so some of the difference above could be
range-driven rather than semantics-driven. Flagged, not hidden; see
Section 15.

## 11. Intensity Evaluation

**A. Metric-space** (predicted embedding distance vs `I_metric`): pearson
-0.345, spearman -0.170, mae 0.379, rmse 0.468 (n=75 val anomalous
instances).

**B. Raw-space** (`I_pred_raw = d/(1-d)`, safe-clamped at `d <= 1-1e-6`, vs
true `I_raw`): pearson -0.285, spearman -0.170 (identical to metric-space's
spearman -- rank order is invariant under the monotonic inverse transform,
as expected), mae 1.384, rmse 2.260. Both directions of evaluation agree:
intensity is NOT being learned well under V2.2, in either space.

## 12. Gradient Scale

Shared-trunk gradient norm, mean over 15 sampled batches per segment:

| Task | V2.1 early/mid/late | V2.2 early/mid/late |
|---|---|---|
| Shape | 0.375 / 10.075 / 17.281 | 0.858 / 8.647 / **16.145** |
| Location | 0.495 / 0.117 / 0.143 | 0.459 / 0.198 / 0.596 |
| Extent | 1.840 / 0.155 / 0.118 | 0.590 / 0.256 / 0.300 |
| **Intensity** | 1.563 / 4.589 / 1.409 | **0.664 / 0.368 / 0.258** |

**Intensity's gradient is now well-behaved AND monotonically shrinking**
(0.664 -> 0.368 -> 0.258), a real improvement over V2.1's non-monotonic
1.56/4.59/1.41 -- the bounded [0,1) metric target gives intensity's
regression loss a much better-conditioned gradient than either V2's
unbounded native parameter (which grew to 10,002.7) or even V2.1's already-
normalized-embedding version. **But shape's gradient blowup is essentially
unchanged** (17.28 in V2.1 vs 16.14 in V2.2, ~46x and ~19x growth from early
respectively, both far larger than any other task by late training) --
confirming shape's instability is independent of intensity's label
semantics, a separate and still-unresolved issue (see `MTL_V21_REPORT.md`
Section 4/10).

## 13. Overall Verdict

**NO BENEFIT.**

Intensity's own gradient dynamics genuinely improved (bounded, monotonically
decreasing, no runaway -- Section 12), and the calibration machinery itself
is implemented correctly and generalizes cleanly across 8 anomaly types
(Sections 5-8). But on the metrics that actually matter -- downstream task
performance -- V2.2 is worse than V2.1 on 3 of 4 tasks (shape, extent,
intensity all declined; only location stayed roughly flat, both near zero
either way), with extent and intensity both flipping to NEGATIVE pearson.
Shape's independent gradient-scale problem (Section 12, unchanged from
V2.1) persists and is not explained away by this change. A cleaner gradient
for intensity did not translate into better representations for intensity
itself, let alone for the other tasks.

## 14. Most Important Finding

**Extent's performance collapsed (0.340 -> -0.135) even though NOTHING
about extent's own definition, loss, or the shared trunk architecture
changed between V2.1 and V2.2 -- the only difference between the two runs
is intensity's label semantics (and its sampling range).** Since shape's
gradient-domination pattern is nearly identical in both runs (17.28 vs
16.14 late) yet V2.1 still achieved good extent performance under that same
domination, shape's gradient alone cannot explain extent's collapse here.
This points to intensity's NEW label distribution (or its new min/max
range) interacting with the shared trunk in a way that specifically hurts
extent -- a genuine, single-seed, but mechanistically plausible instance of
cross-task interference introduced by a change that, on paper, only
touched one attribute's target semantics. Reported as-is, not explained
away, per the project's standing principle of surfacing surprising results
honestly.

## 15. Next Single Priority

**Re-run V2.2 with `intensity_min=0.2, intensity_max=4.0`** (matching
V2.1/legacy's exact range) to isolate whether Section 10-14's regression is
driven by the universal-intensity SEMANTICS or by the acknowledged
range confound (Section 10) -- proposed only, NOT implemented this round.
This is the single most direct way to determine whether "NO BENEFIT" is a
property of the new label definition itself or an artifact of also widening
the sampled intensity range at the same time, before drawing any stronger
conclusion or attempting any fix.

## 16. Files Changed

```text
AnomSim:
anomsim/anomalies/base.py                 -- +compute_reference_scale, compute_rms,
                                              compute_realized_intensity,
                                              calibrate_delta_to_target_intensity,
                                              apply_calibrated_anomaly, sample_log_uniform
tests/test_anomaly_calibration.py          -- NEW: 17 tests for the above
Full AnomSim suite: 216/216 passing (no regression).

Core-Clustering:
core_clustering/target_transforms.py       -- NEW: ScalarMetricTargetTransform
core_clustering/dataset_dynamic_contrastive.py -- +intensity_mode/intensity_min/
                                              intensity_max/intensity_sampling
                                              (default legacy_native_intensity,
                                              V1/V2/V2.1 behavior unchanged);
                                              +intensity_value_raw field on every
                                              returned item
diagnostics/phase1_baselines.py            -- build_loaders additively passes
                                              through the new intensity_* args
                                              (getattr-guarded, zero behavior
                                              change for callers without them)
diagnostics/v2_baseline.py                 -- +--intensity_mode/min/max/sampling;
                                              +evaluate_intensity_dual (metric-space
                                              + inverse-transformed raw-space eval);
                                              experiment_id gets a v22_ prefix
diagnostics/v2_gradient_analysis.py        -- +--intensity_mode/min/max/sampling;
                                              output file v22_gradient_analysis.json
diagnostics/v22_intensity_calibration.py   -- NEW: Sections 5-8's calibration
                                              diagnostics (pure numpy, no model)
tests/test_target_transforms.py            -- NEW: 6 tests
tests/test_dataset_dynamic_contrastive.py  -- +5 tests for universal_deviation_intensity
diagnostics/outputs/v22/v22_intensity_calibration.json -- this run's calibration data
diagnostics/outputs/v2/v22_multitask_seed0/*, v22_gradient_analysis.json -- this run's data
MTL_V22_REPORT.md                          -- this file

Full Core-Clustering suite: 184/184 passing (no regression).
V1, V2, V2.1 (intensity_mode=legacy_native_intensity, the default) --
unchanged, still fully functional and reproducible.
```

## 17. Reproduction Command

```bash
export PYTHONPATH=".:../AnomSim"

# Calibration diagnostics (Sections 5-8) -- gate before any model run
python3 -u diagnostics/v22_intensity_calibration.py --output_dir diagnostics/outputs/v22

# V2.2 multitask seed0 baseline (Section 10's V2.2 column)
python3 -u diagnostics/v2_baseline.py \
  --normalize_embedding --intensity_mode universal_deviation_intensity \
  --modes multitask --n_instances 1000 --epochs 20 --patience 5 --seed 0 \
  --device cpu --output_dir diagnostics/outputs/v2 --force

# V2.2 gradient norm/embedding-norm re-measurement (Section 12)
python3 -u diagnostics/v2_gradient_analysis.py \
  --normalize_embedding --intensity_mode universal_deviation_intensity \
  --n_instances 1000 --epochs 20 --seed 0 --device cpu \
  --output_dir diagnostics/outputs/v2
```

All three commands completed in under 30 seconds total on CPU, same as
V2/V2.1's own runs -- no GPU or remote server needed.
