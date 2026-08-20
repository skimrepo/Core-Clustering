# V2.3 Radial Ordinal Pair Contribution Diagnostic

Diagnostic only. No change to `RadialOrdinalLoss`, the model, the
optimizer, the dataset, or the training procedure. V2.3's exact seed=0
training (Shift-only, universal deviation intensity, range 0.2-4.0,
identity transform, `radial_ordinal` objective, n_instances=1000,
epochs=20, batch_size=32, CPU) was reproduced with additional, measurement-
only instrumentation at sampled batches; every actual `optimizer.step()`
still used the real, unmodified combined 4-task loss.

## 1. Setup

`diagnostics/v23_pair_contribution_diagnostic.py` (new): replicates
`RadialOrdinalLoss.forward`'s exact centroid/severity/pair/direction/
softplus math in a separate function (`decompose_intensity_pairs`) that
additionally exposes NA (normal-anomaly) and AA (anomaly-anomaly) masks,
per-pair margins, and the two sub-losses. **Verified equivalent to the
real loss module on every one of the 45 sampled batches (15 each at
early/middle/late): `|reconstructed - real_intensity_loss|` = exactly
`0.0` in all three segments** -- the decomposition changes nothing about
what the model actually trains on. NA/AA sub-loss gradients were measured
via a separate `torch.autograd.grad` call per batch (same "measure then
still take the real step" pattern as `phase2_gradient_analysis.py`/
`v2_gradient_analysis.py`); the hypothetical "balanced" aggregate
(Section 8) was computed as a number only and never backpropagated.
Batch composition: `BalancedBatchSampler`, batch_size=32 -> 16 normal +
16 anomalous per batch always, so `n_NA=512`, `n_AA=240` on every sampled
batch (fixed by the sampler, not something that drifts during training).

## 2. Pair Counts

| | Count | Fraction of valid pairs |
|---|---:|---:|
| NA (normal-anomaly) | 512 | 0.681 |
| AA (anomaly-anomaly) | 240 | 0.319 |

Constant across all segments (determined entirely by
`BalancedBatchSampler`'s fixed 16/16 split, not by training progress).
NA outnumbers AA by **2.13x** -- a real but moderate imbalance, not an
extreme one.

## 3. Loss Contribution

### Table A

| Segment | Pair Type | Count | Count % | Mean Loss | Loss Contribution % |
|---|---|---:|---:|---:|---:|
| early | NA | 512 | 68.1% | 0.518 | 63.6% |
| early | AA | 240 | 31.9% | 0.631 | 36.4% |
| middle | NA | 512 | 68.1% | 0.681 | 67.7% |
| middle | AA | 240 | 31.9% | 0.691 | 32.3% |
| late | NA | 512 | 68.1% | 0.598 | 65.2% |
| late | AA | 240 | 31.9% | 0.679 | 34.8% |

Loss contribution % tracks count % closely (within a few points) at every
segment -- AA's per-pair mean loss is consistently AS HIGH OR HIGHER than
NA's (0.631 vs 0.518 early; 0.691 vs 0.681 middle; 0.679 vs 0.598 late), so
AA pairs are not being trivially "solved away" -- if anything they are
individually HARDER on average. The count imbalance mechanically caps AA's
share of the total near ~32-36%, but this is a moderate dilution, not a
near-total erasure.

## 4. Ordering Progression

### Table C

| Segment | NA Ordering Acc | AA Ordering Acc | NA Margin (mean) | AA Margin (mean) |
|---|---:|---:|---:|---:|
| early | 0.936 | 0.689 | 0.537 | 0.372 |
| middle | 0.898 | 0.600 | 0.025 | 0.004 |
| late | 0.962 | 0.635 | 0.204 | 0.030 |

**NA is >90% correctly ordered at every single segment, including the
very first one measured (early = 10% into training).** AA never exceeds
69% and is essentially flat-to-slightly-worse across training (0.689 ->
0.600 -> 0.635) -- it does not meaningfully improve as training proceeds.
NA's margin (confidence of correct ordering) is consistently several times
larger than AA's at every segment (e.g. late: 0.204 vs 0.030, a 6.8x gap).
This is the single most consistent finding in this diagnostic.

## 5. Gradient Contribution

### Table B

| Segment | Pair Type | Trunk Grad Norm | Head Grad Norm |
|---|---|---:|---:|
| early | NA | 1.902 | 0.426 |
| early | AA | 2.494 | 0.485 |
| middle | NA | 0.148 | 0.255 |
| middle | AA | 0.107 | 0.254 |
| late | NA | 2.007 | 0.551 |
| late | AA | 0.508 | 0.061 |

**Gradient magnitude dominance is NOT constant across training.** Early on,
AA's trunk gradient is actually LARGER than NA's (2.494 vs 1.902) --the
opposite of naive "NA dominates" intuition. At middle they are comparable
(0.107 vs 0.148). Only by LATE training does NA's gradient clearly exceed
AA's -- 4.0x at the trunk (2.007 vs 0.508), 9.0x at the intensity head
(0.551 vs 0.061). Gradient dominance by NA is a LATE-emerging phenomenon,
not a constant, from-the-start property of this loss.

## 6. Gradient Direction

### Table D

| Segment | cos(g_NA, g_AA) on shared trunk |
|---|---:|
| early | **-0.760** (std 0.539) |
| middle | -0.006 (std 0.701, high variance) |
| late | **-0.456** (std 0.708) |

NA and AA gradients on the shared trunk are substantially, not just
mildly, ANTI-ALIGNED at both early and late training -- this is not simply
"one is bigger than the other," the two pair types are frequently pulling
the trunk in genuinely conflicting directions. Middle training's near-zero
mean cosine comes with very high variance (std 0.70, comparable to the
mean's own magnitude), meaning individual batches vary between aligned and
conflicting rather than settling into a stable near-zero relationship.

## 7. Severity Collapse Timeline

### Table E

| Segment | Normal Severity (mean) | Anomaly Severity (mean) | Anomaly Severity (std) |
|---|---:|---:|---:|
| early | 0.023 | 0.561 | **0.709** |
| middle | 0.006 | 0.031 | **0.017** |
| late | 0.051 | 0.254 | 0.055 |

**The anomaly-severity spread (std) collapses by two orders of magnitude
between early (0.709) and middle (0.017) training, and never recovers back
to anywhere near its early value (late: 0.055, still 13x smaller than
early).** This collapse happens in the SAME window where AA's gradient
share and margin are at their weakest relative to NA (Table B/C, middle
segment) and where NA/AA trunk gradients are most nearly balanced in
magnitude but (per Table D) highly variable in direction batch-to-batch --
consistent with AA's already-weaker, sign-inconsistent signal losing a
"tug of war" against NA's more numerous, more confidently-signed pairs
during exactly this phase. The partial rebound by late (std 0.017 -> 0.055)
coincides with NA's renewed gradient dominance (Table B) rather than any
recovery of AA's own signal (AA's ordering accuracy stays flat, 0.600 ->
0.635, and its margin stays small, 0.004 -> 0.030) -- i.e. the late-stage
change in severity SCALE is not evidence of AA-driven re-differentiation.

`fraction(anomaly_severity > normal_severity)` = 0.936 / 0.898 / 0.962
across the three segments -- numerically identical to NA ordering accuracy
by construction (every anomaly-vs-normal comparison IS an NA pair), a
useful internal consistency check that the instrumentation is correct.

## 8. Current vs Hypothetical Balanced Aggregation

| Segment | L_current (actual, all-pair mean) | L_balanced (hypothetical, 0.5·NA + 0.5·AA) | Relative difference |
|---|---:|---:|---:|
| early | 0.554 | 0.575 | +3.7% |
| middle | 0.684 | 0.686 | +0.3% |
| late | 0.624 | 0.639 | +2.4% |

Numbers only -- never used for any optimizer step, no new checkpoint
created. **The hypothetical balanced aggregate differs from the actual
current loss value by only 0.3-3.7% at any measured segment.** Simply
re-weighting NA/AA to be numerically equal would barely move the scalar
loss value itself -- most of the gap between what the model achieves for
NA vs AA ordering is not explained by this arithmetic dilution.

## 9. Hypothesis Evaluation

**H1. Pair-count dominance: PARTIAL.**
Real (2.13x more NA pairs, fixed by batch composition) and it does cap
AA's loss-contribution share near 32-36% (Section 3), but AA's per-pair
loss is comparable to or higher than NA's throughout, and Section 8 shows
correcting the count imbalance would change the aggregate loss value by
under 4% at any segment. Count imbalance is real but small in its
direct numerical effect.

**H2. Gradient dominance: PARTIAL.**
True specifically at LATE training (4-9x, Table B) but NOT true early
(AA's trunk gradient is actually larger than NA's) or at middle (roughly
even). This is a real, but time-localized, not a constant, mechanism.

**H3. Easy-task saturation: SUPPORTED.**
The cleanest, most consistent finding: NA ordering accuracy is >90% at
every single segment measured, including the earliest (10% into
training), while AA accuracy never exceeds 69% and does not meaningfully
improve over the whole run (0.689 -> 0.600 -> 0.635). NA's margin is
several times larger than AA's throughout.

**H4. Directional interference: SUPPORTED.**
`cos(g_NA, g_AA)` on the shared trunk is substantially negative at early
(-0.760) and late (-0.456) training -- not just a magnitude difference,
a real directional conflict, independently measured from both the count
and magnitude analyses above.

## 10. Primary Mechanism

**Primary: EASY NORMAL-ANOMALY SHORTCUT.**
**Secondary: DIRECTIONAL INTERFERENCE.**

NA being solved almost immediately and staying solved (H3) is the most
direct explanation for why AA's continuous severity-ordering signal never
gets a chance to shape the representation on its own terms -- there is
essentially no point in training where AA is "still being actively
learned while NA competes for gradient," because NA's confident,
high-accuracy signal is present from the earliest segment measured.
Directional interference (H4) compounds this: even in the window where AA's
own gradient magnitude is comparable to NA's (middle segment), the two are
not reliably pointing the same way, so AA's signal doesn't accumulate
cleanly across batches the way a consistently-aligned gradient would.
Pair-count dominance (H1) and gradient-magnitude dominance (H2) are real
but secondary/contextual effects, not selected as primary or secondary
here given Section 8's small (<4%) numeric impact and H2's inconsistency
across training stages.

## 11. Is Balanced NA/AA Aggregation Justified?

**WEAK EVIDENCE.**

Section 8 shows a straight 0.5/0.5 reweighting would move the scalar loss
value by under 4% at every measured segment -- it would not meaningfully
change how much AA "counts" in practice, because AA's per-pair loss is
already comparable to or larger than NA's; the imbalance is not primarily
an arithmetic dilution problem. The stronger mechanisms identified (H3, H4)
are about WHEN/WHETHER AA's signal is learnable at all and whether it
fights NA directionally -- neither is fixed by changing the relative
WEIGHT of two already-computed means. A reweighting fix would be treating
a symptom (count ratio) that this diagnostic's own numbers show is not the
dominant lever.

## 12. Next Single Priority

**Diagnostic only, proposed but NOT implemented**: measure AA's INTERNAL
gradient consistency -- split each batch's AA pairs into random halves and
compute `cos(g_AA_half1, g_AA_half2)` on the shared trunk, across the same
early/middle/late segments. This distinguishes two different stories that
look identical from this report's data alone: (a) AA conflicts mainly WITH
NA (this report's finding), vs (b) AA pairs also conflict substantially
WITH EACH OTHER, independent of NA, meaning the anomaly-severity-ordering
sub-problem may be intrinsically harder to satisfy simultaneously across
many pairs regardless of what NA is doing. If (b) turns out to be true
too, that would suggest the difficulty is not solely "NA crowds out AA"
but that continuous multi-way ranking among anomalies is itself a harder
optimization target than binary normal/anomaly separation -- a materially
different conclusion. No such measurement, reweighting, or loss change was
implemented in this report.

## 13. Files Generated

```text
diagnostics/v23_pair_contribution_diagnostic.py -- NEW: this diagnostic
                                                    (decompose_intensity_pairs,
                                                    verified equivalent to
                                                    RadialOrdinalLoss.forward
                                                    on every sampled batch)
diagnostics/outputs/v23/v23_pair_contribution.json -- full early/middle/late
                                                    aggregated statistics
MTL_V23_PAIR_CONTRIBUTION_REPORT.md              -- this file

No changes to core_clustering/losses_contrastive.py, any model file, the
optimizer, the dataset, or any training configuration. No new checkpoint
was created (this run reproduces V2.3's own training exactly, purely for
measurement -- it was not saved as a new artifact distinct from the
existing v23_multitask_seed0 checkpoint).
```

## 14. Reproduction Command

```bash
export PYTHONPATH=".:../AnomSim"
python3 -u diagnostics/v23_pair_contribution_diagnostic.py \
  --n_instances 1000 --epochs 20 --seed 0 --device cpu \
  --output_dir diagnostics/outputs/v23
```

Completed in under a minute on CPU. Reproduces V2.3's exact training run
(same seed, same config) with measurement-only instrumentation added.
