# MTL V2 Report

All 14 sections below are backed by real, locally-run numbers (seed=0, CPU,
n_instances=1000, epochs=20/patience=5 -- identical config to Phase 1's
`phase1_multitask_seed0` baseline, so the V1/V2 comparison in Section 5 is
apples-to-apples on everything except the architecture and embedding_dim,
which is a deliberate, spec-mandated difference, not a confound -- see
Section 2). This is a single seed. Given the magnitude and mechanistic
clarity of what Section 9/10 found (a reproducible, order-of-magnitude
gradient blowup with a clear causal story), a 1-seed result was judged
sufficient to write up now rather than delay for a 3-seed confirmation --
but the "Overall Verdict" and "Next Single Priority" below should be treated
as informed by ONE run, and re-checked with more seeds before any further
architecture change is made.

## 1. Implemented Architecture

```text
Input                                   (B, 1, T)          -- raw signal only
  + build_position_channel(x, pad_mask) (B, 1, T)          -- per-sample t/(L-1)
  -> concat                             (B, 2, T)

Stem: Conv1d(2->16, k=3, s=1, pad=1) x1                     (B, 16, T)      [T unchanged, stride=1]

Stage0: ConvEncoderBlock(16->16,  k=3, s=2, pad=1)          (B, 16,  275)
Stage1: ConvEncoderBlock(16->32,  k=3, s=2, pad=1) + SelfAttn(32,  heads=4)  (B, 32,  138)
Stage2: ConvEncoderBlock(32->64,  k=3, s=2, pad=1) + SelfAttn(64,  heads=4)  (B, 64,   69)
Stage3: ConvEncoderBlock(64->128, k=3, s=2, pad=1) + SelfAttn(128, heads=4)  (B, 128,  35)

  == shared representation H = (B, 128, 35), NO squeeze, NO shared pool ==

  +-- AttributeHead("shape")     -> (B, 32)
  +-- AttributeHead("location")  -> (B, 32)
  +-- AttributeHead("extent")    -> (B, 32)
  +-- AttributeHead("intensity") -> (B, 32)

AttributeHead(feat=(B,128,35)):
  Conv1d(128->32, k=1)                   (B, 32, 35)
  4 learned queries, MultiheadAttention(embed_dim=32, heads=1), key/value=feat
                                          (B, 4, 32)
  flatten                                (B, 128)
  Linear(128->64) -> GELU -> Linear(64->32)
                                          (B, 32)  == final embedding for this attribute
```

Stage lengths (550 -> 550 -> 275 -> 138 -> 69 -> 35) are the SAME halving
sequence as V1 (self-attention at stages 1/2/3, nominal lengths <=256,
matching `attention_max_resolution=256`'s existing threshold) -- Stem/Stage0-3
were explicitly left unchanged per the spec, only their input channel count
(1->2) and the removal of the post-Stage3 squeeze differ from V1.

## 2. Changes from V1

```text
Removed:
- shared Conv1d(128->4) "squeeze" bottleneck after Stage3
- shared single-query sinusoidal-positional-encoding attention pool (pool_query/pool_attn)
- shared z=4 latent vector
- 4x Linear(4->16) per-attribute heads reading that shared z

Added:
- second input channel: per-sample normalized temporal position,
  position[t] = t/(L-1) within each sample's own valid length L
  (build_position_channel in models_contrastive_v2.py)
- Generic AttributeHead (identical architecture, independent parameters,
  ModuleDict keyed by attribute name) -- 1x1 proj(128->32) -> 4 learned-query
  MultiheadAttention pool -> flatten -> Linear(128->64)->GELU->Linear(64->32)
- ConvBottleneckEncoder gained an additive include_squeeze=True/False
  constructor flag (default True = V1 behavior unchanged) so V2 could reuse
  Stem/Stage0-3 without duplicating that code
- ContrastiveTrainerV2: single AdamW over model+loss params (was already
  true of the SimpleTrainer used for Phase 1's own baselines -- see caveat
  below)
- embedding_dim default changed 16 -> 32 (per spec Section 10) -- NOT
  isolated from the architecture change in this comparison; a real
  confound acknowledged here, not hidden

Unchanged:
- Stem/Stage0-3 conv+self-attention architecture and channel widths
  ([16,32,64,128]), kernel_size/stride/padding, downsampling ratio
- ShapeContrastiveLoss / PairwiseGapRegressionLoss / NormalRelativeRegressionLoss
  formulas, unchanged (only architecture changed, not loss math)
- BalancedBatchSampler, contrastive_pad_collate, DynamicContrastiveDataset --
  no dataset-side changes; the position channel is built inside the model's
  forward() from pad_mask, not in the dataloader
- V1 (ContrastiveEncoder/ContrastiveTrainer/cli_contrastive.py) untouched
  and still importable/runnable exactly as before
```

**Caveat on the optimizer claim in the original spec**: the spec's Section
12 asked to remove "4 independent AdamW, each including the shared trunk"
and replace it with a single AdamW. Checking `diagnostics/phase1_baselines.py`
(the script that actually produced V1's Phase 1/2 multitask numbers) shows
it already used `diagnostics/simple_trainer.py`'s single-AdamW `SimpleTrainer`,
not `core_clustering/trainer_contrastive.py`'s 4-optimizer `ContrastiveTrainer`
(that class is used by the separate `cli_contrastive.py` production path, not
by the diagnostic baselines this report compares against). So the V1-vs-V2
numbers in Section 5 already share the same single-optimizer setup on both
sides -- ContrastiveTrainerV2 does NOT introduce a new optimizer confound
relative to the specific V1 numbers being compared here, though it does
still matter for the separate `cli_contrastive.py` production entry point,
which is why ContrastiveTrainerV2 + cli_contrastive_v2.py were still built
as specified.

## 3. Parameter Count

Measured on the production-scale default config (`ConvBottleneckConfig(n_time_max=550, n_features=2, attention_max_resolution=256)`, `embedding_dim=32`):

```text
Shared trunk (encoder, Stem+Stage0-3+self-attn): 294,848
One AttributeHead:                                18,816
All 4 AttributeHeads:                             75,264
Total:                                            370,112
Shared ratio:         79.7%
Task-specific ratio:  20.3%
```

Matches the spec's expectation ("shared parameter가 전체의 대부분을 차지해야 합니다").
Reproducible via `count_parameters()` in `core_clustering/models_contrastive_v2.py`.

## 4. Gradient Flow Sanity Check

Verified programmatically (`tests/test_models_contrastive_v2.py::test_gradient_flow_isolated_per_attribute_loss`,
passing) by backward()-ing each attribute's own embedding sum independently and
checking `torch.autograd.grad(..., allow_unused=True)` on every other head's
parameters:

| Loss | Shared trunk | Own head | Other heads |
|---|---|---|---|
| Shape | O | O | X |
| Location | O | O | X |
| Extent | O | O | X |
| Intensity | O | O | X |

No cross-head gradient leakage in any direction -- the four AttributeHeads
never share parameters and are only connected through the shared trunk
feature map H, so a single attribute's loss graph never touches another
head's weights at all (not just zero gradient -- structurally absent, hence
`None` under `allow_unused=True`). Also confirmed: at least one shared-trunk
parameter receives a nonzero gradient for every one of the four losses (the
trunk is reachable from every attribute, as designed).

## 5. V1 vs V2 Performance

Both rows: multitask, seed=0, n_instances=1000, epochs requested=20 (both
early-stopped on patience=5), CPU. V1 numbers from
`diagnostics/outputs/phase1/phase1_multitask_seed0/metrics.json` (already
in `MTL_DIAGNOSTIC_REPORT.md` Section 3). V2 numbers from
`diagnostics/outputs/v2/v2_multitask_seed0/metrics.json` (this session).

| Task | Metric | V1 | V2 | Change |
|---|---|---:|---:|---|
| Shape | nn_accuracy | 0.833 | 0.793 | slightly worse |
| Shape | pos/neg separation | 0.062 | 0.0067 | ~9x smaller (see Section 8) |
| Location | pearson | 0.018 | 0.062 | ~3.4x larger, still very weak |
| Extent | pearson | 0.207 | 0.137 | worse, not recovered |
| Intensity | pearson | 0.909 | 0.676 | clearly worse (see Section 9-10) |

### Q1: Location이 실제로 학습되기 시작했는가?

Pearson went from 0.018 to 0.062 -- a real, reproducible improvement in
direction and a >3x increase in magnitude, consistent with Section 4's
diagnosis that the OLD shared attention-pool (now removed) was where
location's signal was being destroyed. But 0.062 is still a very weak
correlation (explains under 1% of variance) -- this is "a crack of light,"
not "location now works." See Section 6 verdict: INCONCLUSIVE, leaning
toward a small real improvement.

### Q2: Extent의 multi-task 성능이 회복되었는가?

No. 0.207 -> 0.137, a further ~34% relative drop, not a recovery. See
Section 7.

### Q3: Shape/Intensity가 새 구조 때문에 손해를 봤는가?

Shape: roughly flat on nn_accuracy (0.833->0.793, within normal 1-seed
noise), but its already-small pos/neg separation shrank further (0.062 ->
0.0067). Intensity: clearly regressed (0.909->0.676) and, per Section 9-10,
this is NOT random noise -- there is a specific, large, mechanistic cause
(a gradient-norm blowup unique to V2, absent in V1 under the identical
config). See Section 8.

## 6. Location Result

**INCONCLUSIVE** (leaning toward a small real improvement, not a fix).

Pearson 0.018 -> 0.062 (seed0 only, both sides). Direction and magnitude
both moved the way the Section 4 diagnosis (location info dies specifically
at the old shared attention-pool) predicted removing that bottleneck should
move them. But the absolute magnitude is still far from "location works" --
0.062 is barely distinguishable from noise on N=2775 pairwise gaps at 1
seed. This needs a 3-seed re-check (matching how location_only/extent_only/
multitask were confirmed in Phase 2) before calling it either SUPPORTED or
a dead end.

## 7. Extent Result

**NOT IMPROVED.**

Pearson 0.207 (V1 multitask) -> 0.137 (V2 multitask), a further decline, not
a recovery. Removing the shared pooling bottleneck did not help extent the
way it was hoped to -- consistent with the original diagnostic's Hypothesis
F (trunk capacity/interference across all 4 attributes simultaneously, not
one specific antagonist) still being live: giving extent its own head and
own pooling did not, by itself, fix its multitask degradation, which points
toward the interference being upstream in the shared trunk's Stage0-3
representation itself, not in the (now-removed) shared pooling step for
extent specifically.

## 8. Shape / Intensity Regression Check

Shape: nn_accuracy 0.833->0.793 is within normal single-seed variation for
this metric (compare Phase 1's own seed0 screening vs 3-seed spreads
elsewhere in `MTL_DIAGNOSTIC_REPORT.md`) -- not flagged as a real
regression on this metric alone. Its pos/neg separation shrinking further
(0.062->0.0067) is consistent with (not independent evidence beyond) the
same embedding-norm dynamics discussed for intensity below, since all four
heads' outputs are summed into one clipped gradient.

Intensity: pearson 0.909->0.676 IS a real regression, and Section 9-10 give
a specific, mechanistic reason: intensity's own trunk gradient norm grows
33.9 (early) -> 342.5 (middle) -> **10,002.7 (late)**, roughly 300x over the
course of one 11-epoch run -- a pattern completely ABSENT in V1's identical
multitask config (V1's intensity loss stayed bounded between 0.6-2.1 across
all 10 of its own training epochs, per
`diagnostics/outputs/phase1/phase1_multitask_seed0/epoch_history.json`,
re-checked this session). V2's own training log shows the same signature
directly: `loss_intensity` climbs from ~2.6 (epoch0) to 257.5 (epoch8) to
346.9 (epoch9) before partially recovering -- a real, visible instability,
not a measurement artifact.

**This is the most important, surprising finding in this report and is
reported as-is, not smoothed over**: removing the shared bottleneck did not
just fail to fix extent -- it introduced a NEW, severe gradient-scale
pathology for intensity that was not present in V1 under the same config,
seed, and data scale.

## 9. Trunk vs Head Gradient

Sampled 15 batches each at early(10%)/middle(50%)/late(90%) of a 20-epoch
multitask run (n_instances=1000, seed=0, CPU) -- same sampling convention as
Phase 2's gradient analysis, adapted for V2 (trunk = `model.encoder` only;
no `pool_attn`/`pool_query` exist in V2).

| Task | Trunk grad norm (late) | Head grad norm (late) | Ratio (trunk/head) |
|---|---:|---:|---:|
| Shape | 0.016 ± 0.005 | 0.097 ± 0.028 | 0.165 |
| Location | 0.296 ± 0.085 | 1.509 ± 0.548 | 0.196 |
| Extent | 0.449 ± 0.130 | 3.907 ± 1.141 | 0.115 |
| Intensity | **10,002.7 ± 6,046.4** | **38,431.8 ± 25,091.2** | 0.260 |

(Early/middle segments in `diagnostics/outputs/v2/v2_gradient_analysis.json`
show the same monotonic blowup for intensity: trunk 33.9 -> 342.5 ->
10,002.7; every other task's trunk-gradient norm stays flat or shrinks
across the same three segments, matching V1's own qualitative pattern from
`MTL_DIAGNOSTIC_REPORT.md` Section 8.)

The trunk-to-head ratio itself (~0.12-0.26, fairly stable across tasks)
shows the trunk IS still receiving a real, nonzero, proportionate update
signal relative to each head throughout training -- it is not the case that
"only the heads learn." The pathology is intensity's ABSOLUTE gradient
scale exploding by ~300x over training, not the trunk being starved
relative to its own head.

## 10. MTL Gradient Interaction

Same 15-batch-per-segment sampling, cosine similarity of each task pair's
gradient on the shared trunk:

| Pair | Early | Middle | Late |
|---|---:|---:|---:|
| shape vs location | 0.045 | 0.013 | 0.043 |
| shape vs extent | -0.005 | 0.018 | 0.033 |
| shape vs intensity | 0.017 | -0.008 | -0.034 |
| location vs extent | 0.059 | -0.019 | 0.070 |
| location vs intensity | 0.041 | 0.016 | -0.055 |
| **extent vs intensity** | 0.235 | -0.009 | **0.083** |

(Mean cosine per pair per segment, n=15 sampled batches each; full stats
incl. std/frac_negative in `diagnostics/outputs/v2/v2_gradient_analysis.json`.)

**extent vs intensity flipped from consistently NEGATIVE in V1 (mean -0.28
early / -0.52 middle / -0.26 late, majority-conflicting in 67-73% of
sampled batches -- V1's most consistent conflicting pair) to no longer
consistently negative in V2 (+0.24 early / -0.01 middle / +0.08 late --
weakly positive at 2 of 3 segments, essentially zero at middle, and no
segment majority-conflicting)**. Taken at face value this looks like an
improvement in directional conflict. **However, this number should be
treated with real caution, not celebrated**: intensity's gradient vector at
this late-segment point is
~625,000x larger in norm than shape's and ~22,000x larger than extent's
(Section 9) -- a cosine computed against a vector of that scale is
dominated by whatever directions intensity's own runaway regression happens
to push, and the practical effect of `clip_grad_norm_` on the COMBINED
gradient (used by both V1 and V2's trainer) is that once one task's raw
gradient norm exceeds the others' by orders of magnitude, the clipped
update direction is effectively hijacked to be almost pure intensity,
regardless of what the other three tasks' gradients look like. The
apparent "reduced conflict" may simply be an artifact of intensity's
gradient becoming so dominant that cosine similarity against it stops being
a meaningful measurement of genuine multi-task interaction. This is flagged
explicitly as a result that should NOT be read as "V2 fixed the extent-
intensity conflict" without first fixing Section 9's gradient-scale
pathology and re-measuring.

## 11. Overall Verdict

**V2 PROMISING BUT NEEDS ONE FIX.**

Three most important pieces of evidence:

1. The gradient-flow sanity check (Section 4) and parameter-count balance
   (Section 3, 80% shared / 20% task-specific) confirm the architecture
   itself was implemented correctly and matches the spec's intent -- no
   structural bug is dragging these numbers down.
2. Location moved in the predicted direction (Section 6) after removing the
   shared-pooling bottleneck that Phase 2 diagnostics specifically
   implicated -- weak evidence the core architectural hypothesis (shared
   pooling was destroying task-specific information too early) has some
   truth to it, though not yet strong enough to call SUPPORTED outright.
3. Intensity's gradient norm exploding ~300x over one training run
   (Section 8-9), a pattern completely absent from V1 under the identical
   config/seed/data, is a specific, large, and well-evidenced NEW problem
   introduced by V2 -- and it plausibly explains BOTH intensity's own
   regression AND shape's shrinking separation (via gradient-clipping
   direction-hijacking, Section 10), meaning the current V1-vs-V2 numbers
   likely understate V2's true potential until this one issue is fixed.

## 12. Next Single Priority

**Target scaling for intensity's regression loss.**

Not gradient balancing algorithms (GradNorm/PCGrad are still explicitly
out of scope), not a head-capacity or trunk-capacity change, not
re-touching Stage3 resolution or the position channel -- Section 8-9 give a
specific, mechanistically-understood target: `NormalRelativeRegressionLoss`
regresses the anomaly's embedding-space distance-from-normal-centroid
directly toward intensity's raw value (range ~0.2-4.0, per that loss's own
docstring), with no learnable scale on either side (deliberately, so the
head's own Linear/MLP weights are the only degree of freedom for scale --
see that loss's docstring in `losses_contrastive.py`). Under V2's larger,
more expressive AttributeHead (vs V1's single Linear(4->16) reading a
4-dim shared bottleneck), that degree of freedom appears able to grow the
achievable embedding-distance scale much further before any correcting
force pushes back, which is consistent with the specific ~300x blowup
observed only for intensity (the attribute whose regression target has the
widest range and whose achievable embedding norm is least constrained).
Simple deterministic normalization of intensity's target (e.g. z-scoring
log-intensity, or clamping the regression target range) -- not a learned
weighting scheme -- is the smallest, most targeted next change, and is
exactly the contingency the original V2 spec itself flagged in advance
("V2에서도 수십-수백 배 imbalance가 유지될 경우... target normalization... 적용할
예정"). Re-run this Section 5-10 comparison after that one change, at the
same seed=0 first, before considering anything else.

## 13. Files Changed

```text
core_clustering/models_conv_bottleneck.py   -- additive include_squeeze flag on
                                                ConvBottleneckEncoder (default True,
                                                V1 behavior/params unchanged)
core_clustering/models_contrastive_v2.py    -- NEW: build_position_channel,
                                                AttributeHead, ContrastiveEncoderV2,
                                                count_parameters
core_clustering/trainer_contrastive_v2.py   -- NEW: ContrastiveTrainerV2 (single AdamW)
core_clustering/cli_contrastive_v2.py       -- NEW: V2 training CLI
diagnostics/v2_baseline.py                  -- NEW: V1-comparable multitask seed0
                                                baseline runner (reuses
                                                phase1_baselines.build_loaders /
                                                evaluate_all_metrics directly)
diagnostics/v2_gradient_analysis.py         -- NEW: trunk/head gradient norm +
                                                cosine re-measurement for V2
tests/test_models_conv_bottleneck.py        -- +2 tests for include_squeeze
tests/test_models_contrastive_v2.py         -- NEW: position channel, AttributeHead,
                                                ContrastiveEncoderV2, gradient-flow
                                                sanity check, parameter-count tests
tests/test_trainer_contrastive_v2.py        -- NEW: mirrors test_trainer_contrastive.py
                                                for the single-optimizer trainer
tests/test_cli_contrastive_v2.py            -- NEW: mirrors test_cli_contrastive.py
MTL_V2_REPORT.md                            -- this file

V1 (models_contrastive.py, trainer_contrastive.py, cli_contrastive.py) --
UNCHANGED, still fully functional, not deleted per spec Section 22.
```

Full test suite: 167/167 passing (`PYTHONPATH=".:../AnomSim" python3 -m
pytest tests/ -q`), including all pre-existing V1 tests (no regression).

## 14. Reproduction Command

```bash
export PYTHONPATH=".:../AnomSim"

# V2 multitask seed0 baseline (Section 5's V2 column)
python3 -u diagnostics/v2_baseline.py \
  --modes multitask --n_instances 1000 --epochs 20 --patience 5 --seed 0 \
  --device cpu --output_dir diagnostics/outputs/v2 --force

# V2 gradient norm/cosine re-measurement (Sections 9-10)
python3 -u diagnostics/v2_gradient_analysis.py \
  --n_instances 1000 --epochs 20 --seed 0 --device cpu \
  --output_dir diagnostics/outputs/v2

# Standalone V2 training via the production CLI (equivalent model/trainer,
# full argument surface, e.g. for a real held-out run)
python3 -m core_clustering.cli_contrastive_v2 \
  --output_dir outputs/v2 --run_id run0 \
  --n_instances 1000 --epochs 100 --patience 10 --seed 0 --gpu -1
```

Both diagnostic runs above completed in under 30 seconds on CPU (this
session, this machine) -- no GPU or remote server was needed for this
1-seed baseline; a 3-seed confirmation (for location, per Section 6) would
still be cheap enough to run the same way.
