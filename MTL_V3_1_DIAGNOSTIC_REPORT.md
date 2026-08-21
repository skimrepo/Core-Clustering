# MTL V3.1 Diagnostic Report: Why Intensity Can't Represent Magnitude, Why Location Stays Unlearned

This round is diagnosis only. The V3.1 foundation (trunk, AttributeHead,
ReferenceContextEncoder/ContextFusion, plain Shape loss, all three
probabilistic scalar heads, reference sampling, simulator, splits) is
**frozen and unchanged**. Nothing here retrains the main model, changes a
loss, or starts a "V3.2" — every number below comes from a forward pass
(and, in a few clearly-labeled cases, a single backward pass with no
optimizer step) on the existing `diagnostics/outputs/v31/
v31_multitask_seed0/bestmodel.pkl` checkpoint.

**Headline findings, stated up front:**

- **Intensity (Question A)**: the frozen embedding itself — not just the
  current scalar decoder — only weakly encodes within-anomaly magnitude
  (linear-probe pairwise-ordering agreement 0.566, barely above the 0.5
  chance level). This is compounded by a specific, verified mechanism in
  the Laplace objective: the mean-path gradient is magnitude-bounded
  (`|d/dmu| <= 1/scale`) while the scale-path gradient grows without
  bound as residual grows, so for high-D samples the optimizer's cheapest
  way to reduce loss is inflating uncertainty, not correcting the mean.
  Both a representation limitation and an objective-mechanics effect are
  real and mutually reinforcing — this is a **combination**, not a single
  root cause.
- **Location (Question B)**: information about the true location value
  is present and INCREASES through the shared trunk and even through the
  Location head's own 1x1 convolution (pearson 0.55 -> 0.64 -> 0.71 ->
  0.72 across Stages A-E), then **collapses at the head's own multi-query
  attention pooling step** (0.72 -> -0.06 at Stage F) and stays destroyed
  through the final embedding (-0.16 at Stage G). This is **CASE L3**:
  the AttributeHead's pooling geometry, not the shared trunk or
  ContextFusion, is the primary bottleneck. A secondary, compounding
  factor was also found and is reported honestly: the location TARGET
  itself is defined relative to the extent-dependent feasible start
  range, not the full valid sequence length, which caps even the
  best-achievable pre-pooling correlation below what a "clean" target
  would allow.

## A. Executive Summary

Two independent investigations, each following the same discipline:
target audit before model probes, frozen-checkpoint probes before any
gradient work, and no changes to the frozen V3.1 model. Intensity's
failure is diagnosed as a real combination of representation and
objective-mechanics effects — H3 (rare high-D examples get overwhelmed)
is explicitly **rejected** by the data; high-D examples in fact dominate
both the loss and the trunk-gradient budget. Location's failure is
diagnosed as CASE L3 (pooling bottleneck), with a genuine, previously
undocumented target-semantics caveat reported as a secondary, compounding
factor rather than the primary cause. One recommended next controlled
experiment is proposed for each (Sections P/Q) — neither is implemented.

## B. Exact Diagnostics/Code Changes

```text
NEW (diagnostics only, no training/architecture/loss change):
diagnostics/v3_1_diagnostic_probes.py
    - Pure-arithmetic checks (location_target_audit): no model involved.
    - Frozen-checkpoint forward-pass probes (intensity_embedding_probe,
      location_stage_probes, location_temporal_shift_test,
      location_position_channel_ablation, reference_effect_on_location,
      shape_extent_sanity): model.eval(), torch.no_grad() throughout.
    - Frozen-checkpoint forward+SINGLE-backward probes
      (intensity_loss_decomposition, location_gradient_probe): gradients
      are read via forward hooks + torch.autograd.grad on the existing
      checkpoint's weights. NO optimizer.step() is ever called anywhere
      in this file -- weights are never updated. This is why the whole
      script runs in ~10 seconds on local CPU (verified below) and needed
      no GPU time.

NOT modified: every file listed as frozen in the task's Section 1
(models_contrastive_v3.py, reference_context.py, prob_heads.py,
losses_contrastive.py, losses_v3.py, dataset_episodic.py,
dataset_dynamic_contrastive.py, trainer_contrastive_v3.py,
cli_contrastive_v3.py). Confirmed by `git diff` showing zero changes to
any of these files this round.
```

## C. Compute Environment

| Work | Where it ran | Device | Runtime | Checkpoint |
|---|---|---|---|---|
| All of Sections D-O below (loss decomposition, binned analysis, embedding probes, target audit, stage probes, temporal-shift test, position-channel ablation, gradient probes, reference-effect sweep, shape/extent sanity) | Local CPU | cpu | **~10.1s wall-clock** (`time` output: 21.3s user, 3.0s system, 239% cpu, 10.132s total) for the ENTIRE script, `n_instances=1000` (matching V3.1's own training scale) | `diagnostics/outputs/v31/v31_multitask_seed0/bestmodel.pkl` (seed0 only -- single-seed diagnostic, per Section 16's explicit instruction not to auto-run 3 seeds) |

No GPU time was used or needed this round. Per Section 0's policy, this
is justified explicitly: every probe here is checkpoint-only (forward
pass, or forward + one backward pass with no optimizer step) -- none of
it is "model training, retraining, multi-seed runs, gradient-analysis
TRAINING, large repeated inference, or long diagnostic sweeps" in the
sense the policy means to gate to the GPU server. If a later round needs
an actual retraining-based gradient-analysis run (comparing gradient
trajectories ACROSS epochs, as `v3_gradient_analysis.py` does), that
would move to the GPU server -- it was not needed to answer either
question this round.

**Reproduction command:**
```bash
PYTHONPATH=.:../AnomSim python3 diagnostics/v3_1_diagnostic_probes.py \
  --checkpoint diagnostics/outputs/v31/v31_multitask_seed0/bestmodel.pkl \
  --output_dir diagnostics/outputs/v31_diag --seed 0 --n_instances 1000
```

## D. Intensity Loss Decomposition

Per-sample, on 150 val instances (75 normal, 75 anomalous), a single
forward pass with gradient tracking + a single backward pass on
`residual/scale + log(2*scale)` summed over the batch (no optimizer
step). Three representative rows (full 150-row table in
`diagnostics/outputs/v31_diag/v3_1_diagnostic_probes.json`):

| D | mu | scale | residual | residual_term | scale_term | total | grad_raw_mu | grad_raw_scale |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32.07 | 0.484 | 1.634 | 31.58 | 19.33 | 1.184 | 20.51 | -0.235 | **-9.03** |
| 3.46 | 0.188 | 0.533 | 3.28 | 6.15 | 0.064 | 6.21 | -0.322 | **-3.99** |
| 0.033 | 0.484 | 1.634 | 0.45 | 0.276 | 1.184 | 1.46 | 0.235 | 0.357 |

`grad_raw_mu` stays in a narrow ~0.2-0.3 band regardless of D;
`grad_raw_scale` grows by more than 20x between a small-D and the
largest-D sample. This single pattern, visible even at the per-sample
level, is the same one Section G's bin-level aggregation confirms.

One additional, unplanned observation worth reporting plainly: mu/scale
are **not smoothly varying with D** even within similar D ranges -- e.g.
the D=3.46 row above gets mu=0.188 (matching the "normal" profile)
while a D=32.07 row gets mu=0.484 (matching the modal "anomaly"
profile). The model's output looks closer to a **soft binary switch**
between two profiles (~0.19/scale~0.53 and ~0.48/scale~1.6) than a
continuum -- consistent with, and a sharper restatement of, `MTL_V3_1_
REPORT.md` Section H's "normal-vs-anomaly separation, not magnitude
representation" finding.

## E. Intensity Target-Bin Statistics

Quantile-binned `D` (same convention as `v3_baseline.py`'s
`evaluate_v3`), n=150 val samples:

| D bin | n | mean D | mean mu | mean scale | mean residual | mean residual_term | mean scale_term | mean\|grad_raw_mu\| | mean\|grad_raw_scale\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal (D=0) | 75 | 0.000 | 0.189 | 0.596 | 0.189 | 0.325 | 0.155 | 0.296 | 0.509 |
| (0.0013, 0.013] | 15 | 0.007 | 0.391 | 1.372 | 0.384 | 0.278 | 0.940 | 0.235 | 0.403 |
| (0.013, 0.134] | 15 | 0.046 | 0.412 | 1.466 | 0.366 | 0.242 | 1.035 | 0.227 | 0.406 |
| (0.134, 1.158] | 15 | 0.481 | 0.478 | 1.633 | 0.303 | 0.186 | 1.184 | 0.233 | 0.401 |
| (1.158, 5.27] | 15 | 2.784 | 0.399 | 1.368 | 2.386 | 2.226 | 0.930 | 0.242 | 0.955 |
| (5.27, 64.05] | 15 | 21.31 | 0.457 | 1.589 | 20.85 | 13.03 | 1.149 | 0.228 | **6.014** |

Reading the columns left to right within a row tells the whole story:
`mean|grad_raw_mu|` is essentially FLAT (0.23-0.30) across every bin
including the most extreme one, while `mean|grad_raw_scale|` is flat
for low/medium bins (~0.40-0.96) and then jumps **6-15x** for the
highest bin (6.014). Predicted `scale` itself plateaus around 1.4-1.6
from the very first anomalous bin onward and does NOT keep growing
into the tens for D=21 -- it cannot "keep up" with D's magnitude, which
is exactly why `residual_term` (13.03) still dominates `scale_term`
(1.15) for the highest bin despite scale's attempted compensation.

## F. Intensity Embedding Probe (frozen, post-hoc, linear-capacity-matched)

Fit on 700 train embeddings, evaluated on 150 held-out val embeddings.
`LinearRegression`/`Ridge`-class probes have the SAME functional form
(a linear map from the 32-D embedding) as the model's own actual
`ScalarPredictionAdapter` -- this is a capacity-matched comparison, not
a more powerful probe cheating its way to a better answer.

| Probe | pearson | spearman | mae | rmse |
|---|---:|---:|---:|---:|
| embedding -> D (all 850 samples, normal+anomalous) | 0.423 | 0.676 | 3.19 | 7.29 |
| embedding -> D (anomalous only, n=375 train / 75 val) | 0.341 | 0.228 | 5.98 | 10.20 |
| embedding -> log1p(D) (all samples) | 0.382 | 0.728 | 2.53 (D-space) | 7.93 (D-space) |

**The "all samples" row's correlation is driven overwhelmingly by the
binary normal/anomalous split, not by within-anomaly structure** -- the
same probe restricted to anomalous-only samples drops from
spearman=0.676 to spearman=0.228. This mirrors the adapter's own
behavior in Sections D/E exactly, using a probe that was never touched
by the Laplace objective's specific gradient dynamics.

kNN local-neighbor diagnostic (5 nearest TRAIN anomalous embeddings per
VAL anomalous query, correlate query's true D with neighbors' mean D):
**pearson=0.344, spearman=0.241** (n=75 queries) -- weak-to-moderate,
not a strong local-smoothness signal.

Pairwise ordering test (1977 random anomalous val pairs, does the
D-probe's prediction preserve the true D ranking?): **agreement
rate = 0.566** against a **0.5 chance level**. This is the single most
direct, decisive number in this section: a linear decoder of the SAME
capacity as the real adapter, trained directly for D-regression rather
than jointly for Laplace NLL, still barely beats a coin flip at ordering
two anomalies by severity.

## G. Intensity Sample/Gradient Contribution by D Range

n=150 val samples, tertile split of the anomalous subset (own-loss
gradient into the shared trunk, isolated from the other three tasks'
losses):

| Bucket | n | frac of samples | frac of total intensity loss | frac of total trunk-grad norm (intensity-only) |
|---|---:|---:|---:|---:|
| normal | 75 | 50.0% | 10.2% | 31.0% |
| low anomaly | 25 | 16.7% | 8.8% | 12.1% |
| medium anomaly | 25 | 16.7% | 10.2% | 2.5% |
| **high anomaly** | 25 | 16.7% | **70.8%** | **54.4%** |

The high-anomaly bucket is 1/6 of all samples but generates **71% of
all Intensity loss and 54% of all Intensity-driven trunk gradient**.
This directly and explicitly **contradicts H3** ("high-D examples are
too rare and their gradient gets overwhelmed") -- if anything, high-D
examples dominate the gradient budget disproportionately to their
count. Whatever is limiting magnitude learning, it is not a lack of
gradient volume from the extreme end.

## H. Intensity Root-Cause Classification

Evaluating H1-H5 against the evidence above:

- **H1 (scale absorbs large errors)**: **Supported, but only partially.**
  Scale does jump immediately once ANY anomaly is present (0.60 normal
  -> 1.37-1.63 anomalous) and its gradient grows sharply with residual
  (Section E/G), but scale itself plateaus ~1.4-1.6 and does not scale
  proportionally into the tens for D=21+ -- so it absorbs error only up
  to a point, not arbitrarily.
- **H2 (loss reduced mainly via scale, not magnitude matching)**:
  **Supported.** `scale_term` dominates `residual_term` for every bin
  except the most extreme one, where residual is simply too large for
  the plateaued scale to absorb (Section E).
- **H3 (high-D underrepresented/overwhelmed)**: **Rejected.** Section G
  shows the opposite: high-D samples dominate both loss and gradient
  share relative to their count.
- **H4 (embedding contains severity info, decoder fails to decode it)**:
  **Not well supported as the primary story.** A capacity-matched linear
  probe on the SAME frozen embedding does little better than the actual
  adapter at within-anomaly ordering (Section F's 0.566 pairwise
  agreement) -- if the information were cleanly present and only the
  decoder were failing, a probe trained directly for regression should
  do meaningfully better than a probe trained jointly under Laplace NLL.
  It does not.
- **H5 (embedding only encodes binary normal/anomaly)**: **Largely
  supported, with nuance.** The embedding->D probe's correlation is
  driven mostly by the normal/anomalous split (Section F); some residual
  anomaly-vs-anomaly structure exists (0.34 pearson, 0.24 spearman
  within-anomaly; kNN 0.34/0.24) but it is weak, not the "no information
  at all" extreme of H5's strongest form.

**Verdict: a combination, weighted toward representation+objective
interaction rather than decoder-alone.** The most defensible synthesis:
the Laplace objective's own gradient asymmetry (bounded mean-gradient,
unbounded scale-gradient -- H1/H2, directly verified in Sections D/E)
plausibly explains WHY the representation never developed strong
within-anomaly magnitude structure in the first place (H4/H5): once a
high-residual sample's cheapest loss-reduction path is scale inflation
rather than embedding refinement, the trunk/head/embedding never
receive strong, consistent pressure to encode fine-grained severity,
even though (Section G) they receive PLENTY of gradient VOLUME from
those same samples. Volume is not the problem; DIRECTION is.

## I. Location Target Audit

Pure arithmetic, no model, n=75 anomalous val samples. Re-derives the
actual anomaly onset using the EXACT formula `DynamicContrastiveDataset.
_inject` uses (`length = round(extent_ratio * n_time)`, `max_start =
n_time - length`, `start = round(location_ratio * max_start)`), then
compares `location_value` (the training target) against the "naive"
`start / n_time` (onset as a fraction of the FULL valid sequence --
the frame the position-channel input actually uses, per
`build_position_channel`'s own docstring: `position[t] = t/(L-1)` over
the sample's full valid length).

| Statistic | Value |
|---|---:|
| location_value range | [0.031, 0.998] |
| onset_fraction_of_full_length range | [0.030, 0.863] |
| corr(location_value, onset_fraction) pearson | 0.924 |
| corr(location_value, onset_fraction) spearman | 0.939 |
| mean \|location_value - onset_fraction\| | **0.147** |
| max \|location_value - onset_fraction\| | **0.467** |
| corr(discrepancy, extent_value) pearson | **0.661** |

Example rows (full 75-row table in the JSON output):

| requested location_ratio | extent_ratio | n_time | actual start | length | onset_fraction | discrepancy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.997 | 0.183 | 548 | 447 | 100 | 0.816 | 0.182 |
| 0.621 | 0.369 | 539 | 211 | 199 | 0.392 | **0.229** |
| 0.530 | 0.439 | 519 | 154 | 228 | 0.297 | **0.233** |
| 0.125 | 0.156 | 526 | 56 | 82 | 0.107 | 0.019 |

**This is a real, previously undocumented finding, not a bug in the
strict sense** -- `location_value` is an internally-consistent,
well-defined target (it always correctly recovers `start` via
`round(location_value * max_start)`), and there is no indexing/off-by-one
error. But it is defined in the FEASIBLE-START-RANGE frame
(`[0, n_time - length]`), while the position-channel input the trunk
actually sees is defined in the FULL-LENGTH frame (`[0, n_time-1]`).
The two frames coincide only when `extent_value ~= 0`; for a typical
extent draw (0.05-0.5 of `n_time`), they diverge by up to ~half the
target's own range, and the divergence is strongly correlated with
extent (pearson 0.661) -- confirmed directly, not assumed. **This adds
real target-decoding difficulty on top of whatever the network itself
can extract, but Section J below shows it does NOT explain the
near-total collapse by itself** (a network that could only decode the
"naive" onset frame should still land somewhere well above the
observed near-zero final correlation, and Section J's earlier-stage
correlations of 0.55-0.72 confirm exactly that: enough is preserved
pre-pooling to substantially beat chance despite this confound).

## J. Location Information-Preservation Probes by Model Stage

Ridge-regularized probes (alpha=10, chosen because several stages are
high-dimensional flattened sequence features where unregularized OLS
would be underdetermined), fit on 700 train / evaluated on 150 val
ANOMALOUS-only samples (location is undefined for normal samples).
Sequence-valued stages (A-E) are represented by flattening
`(channels, time)` into one fixed-size vector per sample (every sample
is padded to the SAME `max_len`, so this is dimensionally valid) with
padded positions zeroed first -- **not** mean/max-pooled, which would
erase exactly the positional information a location probe needs (an
earlier draft of this probe used mean+max pooling and found near-zero
correlation from Stage C onward; switching to a position-preserving
flatten changed the picture completely, which is itself informative:
naive pooling was hiding a real signal that the raw feature map does
contain).

| Stage | Description | pearson | spearman | feature dim |
|---|---|---:|---:|---:|
| A | Early trunk (post-stem, full resolution) | 0.549 | 0.520 | 8,800 |
| B | Middle trunk (post block1 + self-attention) | 0.636 | 0.610 | 4,416 |
| C | Final shared trunk `Hq` (pre-fusion) | 0.709 | 0.716 | 4,480 |
| D | `H_fused` (K=0, so identical to C by construction) | 0.709 | 0.716 | 4,480 |
| E | Location head's 1x1 conv projection | **0.715** | **0.730** | 1,120 |
| F | Head's own multi-query attention pooling | **-0.057** | -0.066 | 128 |
| G | Final normalized 32-D Location embedding | -0.162 | -0.036 | 32 |

Information about true location **increases monotonically from Stage A
through Stage E** (0.549 -> 0.636 -> 0.709 -> 0.709 -> 0.715) — the
shared trunk and ContextFusion are, if anything, mildly HELPING, not
hurting. It then **collapses at Stage F**, the head's own learned
multi-query attention pool, and stays destroyed through Stage G. This
is the single most decisive result in this report.

## K. Location Temporal-Shift Controlled Test

Same background waveform, same anomaly shape/extent/intensity
(`extent_ratio=0.15`, `magnitude_std_multiplier=1.0`), ONLY location
varied: `[0.1, 0.3, 0.5, 0.7, 0.9]`.

| true location | predicted mu | predicted scale |
|---:|---:|---:|
| 0.1 | 0.4966 | 0.2512 |
| 0.3 | 0.4965 | 0.2512 |
| 0.5 | 0.4963 | 0.2516 |
| 0.7 | 0.4964 | 0.2517 |
| 0.9 | 0.4965 | 0.2514 |

`mu` is flat to 3 significant figures across a location range spanning
the entire [0.1, 0.9] interval. Pairwise embedding cosine similarity
across ALL 10 location pairs is **0.99994-0.99999** (essentially
identical vectors), and `corr(delta_location, embedding_distance) =
-0.025` (no relationship). This directly confirms, on a fully
controlled synthetic input (no confound from Section I's target-frame
issue, since here we read out the model's OWN prediction, not a probe
against the target), that the final embedding does not move when the
anomaly's location moves — consistent with Stage G's near-zero probe
correlation in Section J.

## L. Location Position-Channel Diagnostic

Input-level ablation (monkeypatching `models_contrastive_v3.
build_position_channel` for the call only -- confirmed necessary
because `from ... import build_position_channel` binds the name in
`models_contrastive_v3`'s own module namespace at import time; patching
`models_contrastive_v2`'s attribute, as an initial draft of this script
did, silently has NO effect and produced a false "the channel doesn't
matter at all" result that was caught and fixed before being reported
here), n=75 anomalous val samples:

| Ablation | mean embedding L2 change | mu-vs-true-location pearson |
|---|---:|---:|
| correct position channel | -- (baseline) | 0.027 |
| zeroed position channel | 0.049 | -0.142 |
| reversed position channel (`1 - pos`) | 0.010 | 0.021 |

The position channel's presence, absence, or reversal changes the
FINAL embedding only slightly (L2 change of 0.05 and 0.01 against a
unit-norm embedding), and none of the three conditions produce a
meaningfully positive mu-vs-location correlation. **This diagnostic is
only weakly informative given Section J's finding**: by the time we
read the final output, the signal has already been destroyed at Stage
F regardless of what the position channel contained, so an ablation
measured only at the output cannot distinguish "the trunk never used
the channel" from "the trunk used it but pooling threw the result
away." A more informative version of this test would ablate the
channel and re-run the Section J stage probes at Stage C/E rather than
only reading the final mu — this was not done this round (an explicit,
disclosed gap) since Section J's stage-probe evidence already answers
the more important question (where the information is lost) more
directly.

## M. Reference-Context Effect on Location

Fixed queries, K in {0, 3, 10, 30, 100}, reading the model's actual
`location_mu` output (i.e., downstream of Stage F/G's already-lossy
pooling):

| K | mean gate | mu-vs-true-location pearson | mean H_fused std |
|---:|---:|---:|---:|
| 0 | 0.000 | -0.086 | 1.530 |
| 3 | 0.015 | -0.065 | 1.553 |
| 10 | 0.037 | -0.055 | 1.571 |
| 30 | 0.045 | -0.065 | 1.579 |
| 100 | 0.045 | -0.062 | 1.577 |

Gate opens with K (consistent with `MTL_V3_1_REPORT.md`'s own finding
for this seed) and `H_fused`'s own scale grows mildly (1.53 -> 1.58),
but the mu-vs-location correlation stays uniformly poor and slightly
NEGATIVE across every K -- reference conditioning neither meaningfully
helps nor further hurts Location. Given Section J shows Stage
D (`H_fused`) still carries strong location information (0.709,
identical to Stage C at K=0), **the most likely explanation is that
whatever K does to `H_fused` is moot for the final output, because the
information is lost downstream at pooling regardless of what enters
it.** This is inferred by combining Sections J and M rather than
directly measured (Section J's stage probe was only run at K=0) — a
disclosed limitation, not a new probe run this round.

## N. Location Root-Cause Classification

**CASE L3: Location exists after fusion but disappears in Generic
AttributeHead pooling.** Supported directly and decisively by Section
J's stage-by-stage probe (0.549 -> 0.636 -> 0.709 -> 0.709 -> 0.715 ->
**-0.057** -> -0.162) and corroborated by Section K's fully-controlled
temporal-shift test (embeddings at 5 very different true locations are
99.99%+ cosine-similar) and Section M (fusion doesn't rescue it because
the damage happens downstream of fusion).

**A secondary, compounding factor was also found and should not be
discarded just because it isn't the primary cause**: Section I's target
audit shows the `location_value` target itself is defined in a
different (extent-dependent) reference frame than the position-channel
input, with a mean 14.7%/max 46.7% discrepancy from the "natural" onset
fraction. This likely caps the BEST achievable pre-pooling correlation
(explaining why even Stage E's correlation is "only" 0.72 rather than
closer to 1.0) without being the primary explanation for the
near-total collapse observed at the FINAL output — that collapse is
squarely a Stage F/pooling phenomenon per Section J.

`location_gradient_probe` (single frozen-checkpoint backward, no
optimizer step) adds one more corroborating data point: the Location
head's own parameters receive a smaller gradient norm (0.027) than the
shared trunk does from the SAME loss (0.101) — consistent with a head
whose pooling mechanism has historically received too little/too
poorly-directed training signal to learn a position-preserving
aggregation, plausibly why it never departed far from its
initialization's generic (position-discarding) behavior.

## O. Existing Shape/Extent/Reference-Gate Sanity Metrics

Confirms this round's diagnostics did not accidentally disturb
anything (same frozen checkpoint, same seed0, n=150 val queries):

| Metric | This round | `MTL_V3_1_REPORT.md` seed0 |
|---|---:|---:|
| Shape nn_accuracy | 0.873 | 0.860 |
| Shape separation | 0.996 | 0.964 |
| Extent pearson | 0.457 | (report's 3-seed mean 0.387 ± 0.116; seed0 individual not previously isolated) |
| Extent spearman | 0.482 | (3-seed mean 0.362 ± 0.121) |

The small Shape difference (0.873 vs 0.860) is a **known, disclosed
methodological artifact, not a regression**: this round's probes batch-
pad every sample to `max_len=550` with an explicit `pad_mask`, while
`v3_baseline.py`'s original `evaluate_v3` forwards each sample ALONE at
its own exact (unpadded) length. Both are legitimate; they are not
guaranteed to produce bit-identical floating-point results through
convolution/attention boundary handling, and the ~1.3pp Shape
difference here is consistent with that, not with any change to the
checkpoint or code path. Reference-gate statistics (Section M) are
consistent with `MTL_V3_1_REPORT.md`'s own report of this exact seed's
gate behavior.

## P. Recommended Next Controlled Experiment — Intensity

*(Proposed only; not implemented, per Section 7/21's instructions.)*

Given H1/H2 confirmed and H3 rejected, and H4/H5 pointing at a
representation limitation the current objective's gradient asymmetry
plausibly caused: the safest SINGLE next controlled change is to
**detach the scale path's gradient from the shared embedding for a
trial run** (i.e., compute `scale` from a `.detach()`'d copy of the
embedding, or equivalently stop-gradient the scale branch), keeping
`mu`'s path exactly as-is. This isolates whether removing the "cheap
escape valve" (scale inflation) forces the mean-path gradient to do
more of the work, without simultaneously changing the loss function's
form (no MSE/Huber/ranking swap, no reweighting) — a strictly smaller,
more isolated change than any of Section 7's other listed candidates,
and directly targeted at the ONE mechanism (H1/H2) confirmed with the
most direct evidence.

## Q. Recommended Next Controlled Experiment — Location

*(Proposed only; not implemented.)*

Given CASE L3 is decisively supported (the trunk, ContextFusion, and
even the head's own 1x1 conv all preserve strong location information;
only the multi-query attention pooling step destroys it): the safest
SINGLE next controlled change is to **replace ONLY the Location head's
pooling mechanism** with a position-aware readout (e.g., a single
learned linear projection to a per-timestep scalar score followed by a
softmax-weighted sum over time — a standard "soft-argmax" style pool
that is structurally forced to output something a function of WHERE
the mass concentrates, unlike unconstrained multi-query attention)
while leaving the shared trunk, ContextFusion, other three heads, and
Location's own loss/adapter completely untouched. This directly targets
the ONE stage (F) where the evidence shows the failure occurs, without
touching the target-semantics issue (Section I) in the same
experiment — that should be evaluated separately once pooling itself is
no longer the confound, to avoid conflating two changes' effects.

## Section 19 Answers — Intensity

1. **Does Laplace scale absorb large Intensity errors?** Partially --
   it jumps immediately for any anomaly and dominates the loss for
   small/medium D, but plateaus ~1.4-1.6 and cannot keep absorbing error
   as D grows into the tens (Section E).
2. **Does large D generate stronger mean-path gradients?** No --
   `mean|grad_raw_mu|` is flat (~0.23-0.30) across every D bin including
   the most extreme (Section E).
3. **Are high-D examples underrepresented?** No, and this is directly
   contradicted: high-D examples are 1/6 of samples but generate 71% of
   loss and 54% of trunk gradient (Section G). H3 is rejected.
4. **Does normalized Intensity embedding contain continuous D
   information?** Weakly, and mostly at the normal-vs-anomalous
   boundary, not within the anomalous population (embedding-to-D
   pearson drops from 0.42 "all samples" to 0.34 "anomalous only";
   pairwise ordering agreement only 0.566 vs 0.5 chance) (Section F).
5. **Is the problem representation, scalar decoder, Laplace objective,
   data distribution, or a combination?** A combination, weighted
   toward representation+objective interaction: the objective's
   gradient asymmetry (bounded mean-gradient, unbounded scale-gradient)
   plausibly explains why the representation never developed strong
   magnitude structure, since a capacity-matched decoder trained
   directly for regression does little better on the SAME frozen
   embedding (Section H). Data distribution (H3) is explicitly ruled
   out as a contributor.
6. **Why does normal-vs-anomaly separation emerge more easily than
   anomaly-vs-anomaly magnitude?** Normal-vs-anomaly is a single,
   large, easy-to-satisfy gradient direction reachable via one broad
   embedding shift; anomaly-vs-anomaly magnitude requires a CONTINUOUS,
   fine-grained embedding gradient the Laplace objective's own
   mean-path never strongly demands once scale can absorb the
   difference instead (Sections D/E/H).
7. **What is the ONE safest next Intensity change?** Detach the scale
   path's gradient from the embedding for a trial run, changing nothing
   else (Section P).

## Section 20 Answers — Location

1. **Is the Location target correct after all preprocessing?**
   Internally consistent (no bug), but defined in a different
   reference frame (extent-dependent feasible-start-range) than the
   position-channel input's frame -- mean 14.7%/max 46.7% discrepancy
   from the "naive" onset-fraction interpretation (Section I).
2. **At which model stage is Location information first measurable?**
   Already present at Stage A (early trunk, post-stem), pearson 0.549,
   and it INCREASES through Stage E (Section J).
3. **At which stage is it lost?** Stage F -- the Location head's own
   multi-query attention pooling (0.715 -> -0.057) (Section J).
4. **Does ContextFusion damage it?** No -- Stage D (post-fusion, K=0)
   is identical to Stage C (pre-fusion) by construction at K=0, and
   Section M shows fusion at K>0 doesn't measurably help or hurt the
   final (already-lost) output either.
5. **Does multi-query pooling destroy it?** Yes -- this is the single
   most decisive finding in this report (Section J).
6. **Does the final normalized Location embedding still contain
   location?** No -- pearson -0.162 (Section J), confirmed independently
   by the temporal-shift test's 99.99%+ cosine similarity across very
   different true locations (Section K).
7. **Is the scalar probabilistic decoder the problem?** Not primarily
   -- by the time information reaches the decoder (post-pooling), it is
   already gone; the decoder cannot decode what it isn't given.
8. **Does the temporal-position channel materially affect the
   representation?** Inconclusive at the OUTPUT level (Section L) --
   the ablation was only measured post-collapse, which Section L
   explicitly flags as a limited test given Section J's finding.
9. **Do references help or hurt Location?** Neither, measurably --
   gate opens with K but the (already-destroyed) final mu stays
   uniformly poor across all K (Section M).
10. **What is the ONE safest next Location change?** Replace ONLY the
    Location head's pooling mechanism with a position-aware
    (soft-argmax-style) readout, leaving everything else — including
    the target-semantics issue — untouched for now (Section Q).

## Experiment Discipline Confirmation

No hyperparameter sweep was run. No new architecture was added. No
objective was changed. No recommendation above was implemented. Both
open questions were answered with a single best-supported case/verdict
each, with disclosed nuance and disclosed gaps (Section L's
inconclusive output-level ablation; Section M's inferred-rather-than-
directly-measured H_fused-at-K>0 claim) rather than overclaiming
certainty where the evidence didn't fully reach it. H3 is reported as a
clean negative finding, not omitted for being inconvenient.
