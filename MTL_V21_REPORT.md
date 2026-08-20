# MTL V2.1 Report

Single change tested: does constraining every AttributeHead's final embedding
to unit L2 norm reduce the ~300x intensity gradient-norm runaway found in
V2? All numbers below are from ONE seed (seed=0), identical config to V2's
own baseline (n_instances=1000, epochs=20/patience=5, CPU) -- same dataset,
sampler, split, optimizer, LR, loss weights, gradient clipping. Only
`normalize_embedding=True` differs. Run locally this session; see Section
12 for the exact commands.

## 1. Exact Change

```text
Added:
- AttributeHead.normalize_embedding: bool = False (new constructor arg,
  default False = exact V2 behavior)
- When True: raw_embedding = self.mlp(flat); return F.normalize(raw_embedding, p=2, dim=-1, eps=1e-8)
- ContrastiveEncoderV2 gained the same normalize_embedding flag, passed
  through uniformly to all 4 AttributeHeads (never per-attribute)
- AttributeHead.last_raw_embedding: diagnostic-only stashed (detached)
  pre-normalization embedding, read by v2_gradient_analysis.py for Section
  5's raw-norm tracking. Does not affect forward()'s return value or the
  backward graph.

Unchanged:
- Shared trunk (Stem/Stage0-3), AttributeHead's proj/queries/pool_attn/mlp
  architecture and sizes, embedding_dim=32
- ShapeContrastiveLoss / PairwiseGapRegressionLoss / NormalRelativeRegressionLoss
  (extent, intensity) -- formulas, pair construction, centroid calc,
  stop-gradient, loss weights: byte-for-byte identical to V2
- ContrastiveTrainerV2 (single AdamW, same LR, same clip_grad_norm_=1.0 on
  the combined gradient)
- No new LayerNorm/BatchNorm/GroupNorm/RMSNorm anywhere, no embedding norm
  penalty, no learnable temperature/scale added, no target scaling/
  normalization on location/extent/intensity
- V1 and plain V2 (models_contrastive_v2.py, trainer_contrastive_v2.py,
  cli_contrastive_v2.py) all still run exactly as before via
  normalize_embedding=False (the default)
```

## 2. Embedding Norm Sanity

`tests/test_models_contrastive_v2.py` (new tests, all passing):

- `normalize_embedding=True` -> every attribute's output norm is 1.0 to
  float precision (`torch.allclose(norms, ones, atol=1e-5)`), verified for
  all 4 attributes uniformly via `ContrastiveEncoderV2`.
- Forcing the MLP's last layer to output an exact zero vector still
  produces a finite, all-zero output (`F.normalize`'s `eps` guards the
  0/0 case) -- no NaN/Inf.
- `normalize_embedding=False` (plain V2) is unaffected -- output is NOT
  unit-norm, confirming no accidental behavior change for existing V2 runs.

From the actual V2.1 seed0 run's sampled batches (`v21_gradient_analysis.json`),
observed normalized-embedding norm: mean 0.9999999776-0.9999999925, std
~1e-8-1e-7 across all 4 attributes at every segment -- unit norm holds
throughout training, not just at init.

## 3. Gradient Flow Sanity

Re-verified with `normalize_embedding=True` (`test_encoder_v2_gradient_flow_isolated_still_holds_with_normalize_embedding`, passing):

| Loss | Shared trunk | Own head | Other heads |
|---|---|---|---|
| Shape | O | O | X |
| Location | O | O | X |
| Extent | O | O | X |
| Intensity | O | O | X |

Identical to plain V2 (Section 4 of `MTL_V2_REPORT.md`) -- the normalization
sits entirely inside each head's own forward path, so it does not create or
remove any cross-head connection.

## 4. Gradient Scale Comparison

Shared-trunk gradient norm, mean over 15 sampled batches per segment:

| Task | V2 early/mid/late | V2.1 early/mid/late |
|---|---|---|
| Shape | 0.031 / 0.013 / 0.016 | 0.375 / **10.075** / **17.281** |
| Location | 0.344 / 0.142 / 0.296 | 0.495 / 0.117 / 0.143 |
| Extent | 1.950 / 0.407 / 0.449 | 1.840 / 0.155 / 0.118 |
| **Intensity** | 33.86 / 342.53 / **10,002.66** | 1.563 / 4.589 / **1.409** |

**Intensity's specific runaway is gone.** V2's intensity trunk gradient grew
monotonically by ~300x (33.9 -> 342.5 -> 10,002.7); V2.1's stays bounded and
roughly flat (1.6 -> 4.6 -> 1.4), never approaching V2's late-training
values. This directly answers the report's motivating question for
intensity specifically.

**But shape's gradient now grows by ~46x instead** (0.375 -> 10.08 ->
17.28), becoming the LARGEST shared-trunk gradient of any task by late
training (17.28 vs location's 0.143, extent's 0.118, intensity's 1.41 --
roughly 12-150x larger than the other three). This did not happen in V2
(shape stayed flat at 0.013-0.031 there) or in V1. See Section 8 -- this is
reported as a new, distinct finding, not folded into "intensity fixed."

## 5. Raw Embedding Norm Dynamics

Pre-normalization (`last_raw_embedding`) norm, mean over sampled batches:

| Task | Raw norm early | Raw norm mid | Raw norm late |
|---|---:|---:|---:|
| Shape | 1.92 ± 0.24 | 28.14 ± 21.79 | 39.86 ± 33.54 |
| Location | 1.87 ± 0.76 | 18.30 ± 3.38 | 14.05 ± 2.42 |
| Extent | 4.33 ± 2.88 | 56.18 ± 14.07 | 56.86 ± 9.11 |
| Intensity | 9.62 ± 3.94 | 46.83 ± 12.43 | 64.74 ± 15.37 |

Normalized (final) embedding norm for every task/segment: 1.0000 ± ~1e-7 (Section 2).

**Important, explicitly flagged per the report template's own warning**: the
raw pre-normalization embedding is NOT stabilized -- every attribute's raw
norm grows 10-30x from early to late training (shape: 1.9->40, location:
1.9->14, extent: 4.3->57, intensity: 9.6->65). L2 normalization fixed the
OUTPUT scale, but the activation feeding into it keeps growing throughout
training for all four attributes, not just intensity. **This must not be
read as "gradient instability is fully resolved"** -- it has changed shape
(literally: shape's gradient is now the one growing unboundedly, Section 4)
rather than disappearing from the system entirely.

## 6. Performance Comparison

Both rows: multitask, seed=0, n_instances=1000, epochs requested=20 (both
early-stopped, patience=5), CPU.

| Task | Metric | V2 | V2.1 | Change |
|---|---|---:|---:|---|
| Shape | nn_accuracy | 0.793 | **0.993** | large improvement |
| Shape | positive_pair_dist | 0.038 | 0.259 | (scale changes under unit-norm; see note) |
| Shape | negative_pair_dist | 0.044 | 1.785 | (scale changes under unit-norm; see note) |
| Shape | separation | 0.0067 | 1.526 | large improvement |
| Location | pearson | 0.062 | -0.006 | worse, crossed to ~zero/slightly negative |
| Location | spearman | 0.017 | -0.007 | worse |
| Location | mae | 0.283 | 0.241 | improved |
| Location | rmse | 0.363 | 0.302 | improved |
| Extent | pearson | 0.137 | **0.340** | large improvement (best of V1/V2/V2.1: 0.207/0.137/0.340) |
| Extent | spearman | 0.206 | 0.294 | improved |
| Extent | mae | 0.281 | 0.141 | improved |
| Extent | rmse | 0.386 | 0.172 | improved |
| Intensity | pearson | 0.676 | 0.537 | worse (V1 was 0.909 -- still well below V1 either way) |
| Intensity | spearman | 0.789 | 0.814 | slightly improved |
| Intensity | mae | 0.667 | 0.640 | roughly flat |
| Intensity | rmse | 0.967 | 0.820 | improved |

(Note on shape's pos/neg pair distances: under unit-norm embeddings the
maximum possible pairwise distance is 2, vs an unbounded scale in V2 -- the
absolute distance values are not directly comparable across the two rows,
only `separation` and `nn_accuracy`, which are scale-relative, are.)

### Q1: Intensity가 V2의 Pearson 0.676에서 회복되는가?

**No.** 0.676 -> 0.537, a further decline (though its gradient no longer
runs away, its final metric got worse, not better). Spearman/RMSE moved
slightly the other way, so this isn't a uniform collapse, but the headline
Pearson metric is worse.

### Q2: Shape가 V2의 NN accuracy 0.793보다 악화되지 않는가?

**No regression -- clearly improved** (0.793 -> 0.993). But this comes
alongside the new gradient-scale finding in Section 4, so it should not yet
be read as an unqualified win independent of that.

### Q3: Location이 V2의 Pearson 0.062에서 개선/유지되는가?

**Not maintained -- slightly worse**, and crossed from weakly positive to
weakly negative (0.062 -> -0.006). Both values are close enough to zero
that this is plausibly just noise on top of "no real signal either way,"
consistent with `MTL_DIAGNOSTIC_REPORT.md`'s original finding that location
is bad in both single- and multi-task conditions.

### Q4: Extent가 V2의 Pearson 0.137에서 개선/유지되는가?

**Improved, substantially** -- 0.137 -> 0.340, now the best extent result
across V1 (0.207), V2 (0.137), and V2.1 (0.340).

## 7. Gradient Interaction

Same 15-batch-per-segment sampling as Section 4. Mean cosine similarity,
shared-trunk gradient, per pair:

| Pair | Early | Middle | Late |
|---|---:|---:|---:|
| shape vs location | 0.075 | -0.072 | 0.301 |
| shape vs extent | -0.151 | 0.214 | 0.208 |
| shape vs intensity | 0.472 | -0.040 | 0.014 |
| location vs extent | 0.013 | -0.033 | 0.148 |
| location vs intensity | 0.032 | 0.043 | 0.001 |
| **extent vs intensity** | -0.255 | 0.355 | -0.078 |

extent-vs-intensity is mixed (negative/positive/negative across the three
segments, not consistently one direction), a different pattern from V1's
consistently-negative -0.28/-0.52/-0.26 and from V2's mostly-positive
+0.24/-0.01/+0.08. **Per the report's own scope, this is recorded as a
measurement only** -- no architecture conclusion is drawn from it, because
(Section 4-5) intensity's and shape's gradient magnitudes are both still
shifting substantially across segments, and a cosine computed against a
still-changing-scale vector is not a stable basis for a directional-conflict
claim this round.

## 8. Primary Question

**GRADIENT EXPLOSION REDUCED BUT REMAINS** (with an important qualifier).

Evidence:
- Intensity's specific, motivating problem -- a ~300x runaway to a trunk
  gradient norm of 10,002.7 -- is gone. V2.1's intensity trunk gradient
  stays within 1.4-4.6 across all three segments (Section 4).
- However, the underlying mechanism (Section 5: raw pre-normalization
  embedding norm growing unboundedly across training) was NOT fixed by
  final-embedding L2 normalization -- it still grows 10-30x for every
  attribute, shape included.
- A NEW large gradient-scale imbalance appeared: shape's shared-trunk
  gradient grows ~46x (0.375->17.28) and becomes the single largest
  trunk gradient by late training, a pattern absent in both V1 and V2.

This is reported as "reduced but remains," not "fixed," specifically
because the instability that motivated V2.1 (unconstrained embedding scale
feeding a runaway gradient) relocated to a different task rather than
disappearing -- exactly the failure mode Section 5's own instructions
warned against over-claiming.

## 9. Does V2.1 Improve the Overall Baseline?

**INCONCLUSIVE.**

Shape and extent both improved substantially and intensity's specific
gradient pathology is resolved -- real, positive signal. But location did
not improve (arguably very slightly worse, within noise), intensity's own
headline metric got worse despite its gradient stabilizing, and a new
gradient-scale imbalance (shape) appeared that this 1-seed run cannot yet
distinguish from a genuine new problem vs. training-run noise. Promoting
V2.1 as the new baseline outright would be premature given this new,
unexplained instability; discarding it outright would ignore extent's and
shape's clear, large improvements. A 3-seed confirmation (matching how
location_only/extent_only/multitask were confirmed in Phase 2) is the
natural next step before either promoting or discarding V2.1 -- not
attempted here, since the spec asked for a single seed0 run and an explicit
stop after reporting.

## 10. Next Single Priority

**Per-task gradient clipping before summing, instead of clipping only the
combined total.** (Proposed only -- NOT implemented this round.)

Both V2's intensity blowup and V2.1's shape blowup share the same
downstream amplifier: `ContrastiveTrainerV2`/`SimpleTrainer` call
`clip_grad_norm_` on the SUM of all four weighted losses' gradients, so
whichever task's raw gradient happens to be largest at a given step
dominates the clipped update direction almost completely (quantified for
V2 in `MTL_V2_REPORT.md` Section 10, and structurally identical here just
with a different task in the dominant role, Section 4). Clipping each
task's own gradient to a fixed norm BEFORE summing would cap any single
task's contribution to the shared trunk update regardless of which task's
raw scale happens to spike in a given training run -- a training-loop
change, not an architecture, loss, or target change, so it stays within
the same category of minimal intervention this V2/V2.1 line of experiments
has used throughout. This is offered as the single most-evidenced next
candidate; no other change is recommended alongside it.

## 11. Files Changed

```text
core_clustering/models_contrastive_v2.py  -- AttributeHead gained normalize_embedding
                                              (default False) + last_raw_embedding
                                              introspection attribute;
                                              ContrastiveEncoderV2 passes the flag
                                              through uniformly to all 4 heads
core_clustering/cli_contrastive_v2.py     -- +--normalize_embedding flag
diagnostics/v2_baseline.py                -- +--normalize_embedding flag;
                                              experiment_id now prefixed v2_/v21_
diagnostics/v2_gradient_analysis.py       -- +--normalize_embedding flag; now also
                                              records raw/normalized embedding norm
                                              per segment per task (Section 5);
                                              output file v21_gradient_analysis.json
                                              when the flag is set
tests/test_models_contrastive_v2.py       -- +6 tests for normalize_embedding
                                              (unit-norm output, zero-input safety,
                                              raw-embedding stash, uniform application,
                                              gradient-flow isolation still holds)
diagnostics/outputs/v2/v21_multitask_seed0/* -- this run's checkpoint/config/metrics
diagnostics/outputs/v2/v21_gradient_analysis.json -- this run's gradient/embedding data
MTL_V21_REPORT.md                          -- this file

V1, plain V2 (normalize_embedding=False) -- unchanged, still fully functional.
Full test suite: 173/173 passing.
```

## 12. Reproduction Command

```bash
export PYTHONPATH=".:../AnomSim"

# V2.1 multitask seed0 baseline (Section 6's V2.1 column)
python3 -u diagnostics/v2_baseline.py \
  --normalize_embedding --modes multitask --n_instances 1000 --epochs 20 --patience 5 \
  --seed 0 --device cpu --output_dir diagnostics/outputs/v2 --force

# V2.1 gradient norm/cosine/embedding-norm re-measurement (Sections 4-5, 7)
python3 -u diagnostics/v2_gradient_analysis.py \
  --normalize_embedding --n_instances 1000 --epochs 20 --seed 0 --device cpu \
  --output_dir diagnostics/outputs/v2

# Standalone V2.1 training via the production CLI
python3 -m core_clustering.cli_contrastive_v2 \
  --normalize_embedding --output_dir outputs/v21 --run_id run0 \
  --n_instances 1000 --epochs 100 --patience 10 --seed 0 --gpu -1
```

Both diagnostic runs completed in under 30 seconds on CPU, same as V2's own
runs -- no GPU or remote server needed.
