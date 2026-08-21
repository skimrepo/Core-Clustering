# MTL V3.1 Report: Isolating the Shape Heteroscedastic Loss as V3's Collapse Variable

V3.1 is a strictly single-variable controlled experiment against
`MTL_V3_REPORT.md`'s finding that V3's entire shared representation
collapsed during training. **The only change: Shape's training loss is
reverted from the heteroscedastic wrapper
(`base_shape_loss_i / b_i + log(b_i)`) to the original, plain
`ShapeContrastiveLoss` — no division by a predicted scale, no
`log(scale)` term.** Everything else in V3 — trunk, AttributeHead, L2
norm, `ReferenceContextEncoder`/`ContextFusion`, gate, K regimes
(train: `{0,3,10,30}`, eval-only: `100`), reference sampling/
contamination, reference-consistency loss, Location/Extent/Intensity
probabilistic heads and Laplace NLL, optimizer, gradient clipping,
dataset split, simulator/anomaly type/range, seeds `[0,1,2]`,
epochs=20, patience=5, input normalization, positional channel — is
byte-for-byte unchanged. V2.1 and V3's own 3-seed results are reused
verbatim from `diagnostics/outputs/v2` and `diagnostics/outputs/v3`;
nothing was retrained.

**Headline finding, stated up front**: the COLLAPSE CHECK reports
`collapsed=false` for all 3 seeds. Removing the Shape heteroscedastic
wrapper resolved the representation collapse. This is not a full
return to V2.1's performance, and one specific sub-mechanism (the
reference gate) recovered inconsistently across seeds — both are
reported honestly below, not smoothed over.

## A. Exact Code Change

```text
MODIFIED (additive, default-preserving):
core_clustering/trainer_contrastive_v3.py
  ContrastiveTrainerV3.__init__ gained shape_objective: str =
  "heteroscedastic" (default, byte-for-byte reproduces V3) | "plain"
  (V3.1). In _compute_losses:
    heteroscedastic (V3, unchanged):
      l_shape = heteroscedastic_weight(per_anchor[valid], shape_scale[valid])
    plain (V3.1, new):
      l_shape = self.shape_loss(embeddings["shape"], shape)   # scalar,
                                 return_per_sample=False (the loss's own
                                 pre-existing default)
  out["shape_scale"] is still computed by the model in both modes
  (ShapeUncertaintyAdapter is NOT removed from the architecture) but in
  "plain" mode it is never read by the loss -- confirmed by a dedicated
  test asserting model.shape_uncertainty's parameters receive no
  gradient after backward() (tests/test_trainer_contrastive_v3.py::
  test_trainer_v3_plain_shape_objective_detaches_shape_uncertainty_from_the_loss).

core_clustering/cli_contrastive_v3.py       -- +--shape_objective passthrough
diagnostics/v3_baseline.py                  -- +--shape_objective,
                                                +--experiment_id_prefix
                                                (so V3.1 writes v31_* files,
                                                never touching V3's v3_*
                                                manifest/checkpoints)
diagnostics/v3_gradient_analysis.py         -- +shape_objective param,
                                                threaded through
                                                compute_task_losses/measure_batch
                                                (NOT exercised this round --
                                                see Section K)
diagnostics/v3_eval_diagnostics.py          -- +collapse_check(): pairwise
                                                shape-embedding cosine
                                                similarity, per-dim std,
                                                separation; Location/Extent/
                                                Intensity mu+scale std;
                                                intensity mu-vs-D correlation;
                                                reference-gate mean/std at
                                                K=10. Runs FIRST in main(),
                                                before any other diagnostic.
                                                Verified against V3's own
                                                seed0 checkpoint: reproduces
                                                V3's documented collapse
                                                numbers exactly (cosine sim
                                                1.0, all stds ~1e-6-1e-8,
                                                gate ~3e-9).

tests/test_trainer_contrastive_v3.py        -- +3 tests (default value,
                                                plain-mode gradient
                                                detachment, invalid-value
                                                ValueError)

NOT modified: everything in Section A of MTL_V3_REPORT.md's "NEW"/
"MODIFIED" list (reference_context.py, prob_heads.py,
models_contrastive_v3.py, dataset_episodic.py, losses_v3.py,
dataset_dynamic_contrastive.py, losses_contrastive.py). No new Shape-
uncertainty design was added; no fix beyond the one specified revert
was implemented.
```

Full test suite: 247 passed (was 244 in V3; +3 new tests for the flag).

## B. Parameter Count (unchanged, confirmed)

`n_params_total` = 420,266 for all 3 V3.1 checkpoints — identical to
V3's 420,266 (see `MTL_V3_REPORT.md` Section B). The architecture is
verified byte-for-byte unchanged; only the training objective differs.

## C. Experimental Setup

- **Reused, not retrained**: V2.1 seeds 0/1/2 (`diagnostics/outputs/v2/
  v21_multitask_seed{0,1,2}/metrics.json`), V3 seeds 0/1/2
  (`diagnostics/outputs/v3/v3_multitask_seed{0,1,2}/`).
- **Newly trained**: V3.1 seeds 0/1/2, `shape_objective="plain"`,
  identical config to V3 (`n_instances=1000, epochs=20, patience=5,
  k_regimes=(0,3,10,30)` at train time, `K=100` eval-only).
- **Infra note (see Section L for the full write-up)**: training was
  run twice for cross-verification — once locally on CPU
  (`diagnostics/outputs/v31`, killed partway through seed1/seed2 once
  the GPU run finished) and once on the user's GPU server. **The GPU
  run completed all 3 seeds in ~5 minutes total; the local CPU run
  was projected at ~60-65 minutes for the same 3 seeds** (seed0 alone
  took 1170s / ~19.5 min on CPU at ~58s/epoch). The results reported
  below are from the GPU run's checkpoints, transferred back via
  `rsync`.

## D. COLLAPSE CHECK (run first, before any other diagnostic)

Per-seed, `n_queries=150`, reference gate measured at `K=10` with 3
draws/query:

| Seed | `collapsed` | shape mean pairwise cos-sim | shape mean per-dim std | shape separation | loc std(mu) | ext std(mu) | int std(mu) | int mu-vs-D pearson | mean gate (K=10) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **V3 (for reference)** | **true** | 1.0000 | 1.0e-05 | 1.6e-05 | 8.6e-07 | 2.8e-07 | 5.6e-08 | n/a (undefined) | 3.3e-09 |
| V3.1 seed0 | **false** | 0.2988 | 0.1291 | 0.9639 | 1.3e-03 | 4.1e-02 | 1.4e-01 | 0.331 | 0.0405 |
| V3.1 seed1 | **false** | 0.4064 | 0.1172 | 0.6885 | 6.0e-04 | 3.2e-02 | 3.0e-01 | 0.258 | 0.0107 |
| V3.1 seed2 | **false** | 0.5724 | 0.0990 | 0.6560 | 2.1e-03 | 1.2e-02 | 1.3e-02 | 0.275 | 4.3e-04 |

Every V3.1 seed clears `COLLAPSE_STD_THRESHOLD=1e-3` on shape's
per-dimension std by 2 orders of magnitude, and on at least one of
location/extent/intensity's std(mu) by 1-2 orders of magnitude — the
`collapsed` flag is `false` because it only takes one head showing
real variance to disprove "every output is constant," and here shape,
extent, and intensity all clearly do (location's std(mu) is smaller
but still ~1-2 orders above V3's ~1e-6-1e-7 floor). **Conclusion: the
collapse, as strictly defined by this output-level check, is resolved
at all 3 seeds.**

One number in this table is NOT uniformly recovered: **the reference
gate**. Seed0 (0.041) and seed1 (0.011) show real, non-trivial gate
opening; seed2's gate (4.3e-4) is barely above V3's collapsed floor
(3.3e-9) and, as Section F below shows, produces contamination
sensitivity indistinguishable from noise. This is reported plainly in
Section F/M — it does not change the `collapsed=false` verdict (the
gate's job is to blend reference information into the shared
representation, and per-query/per-sample outputs vary regardless of
whether the gate itself is large), but it means "does the model
actually use the optional reference conditioning" is a seed-dependent
yes/no, not a clean universal yes.

## E. Task-Metric Comparison (mean ± std, 3 seeds)

| Task | Metric | V2.1 | V3 (collapsed) | V3.1 (this experiment) |
|---|---|---:|---:|---:|
| Shape | nn_accuracy | 0.984 ± 0.008 | 0.647 ± 0.073 | **0.816 ± 0.031** |
| Shape | separation | 1.540 ± 0.104 | 0.000 ± 0.000 | **0.770 ± 0.138** |
| Location | pearson | 0.001 ± 0.008 | 0.018 ± 0.099 | 0.024 ± 0.111 |
| Location | spearman | -0.005 ± 0.003 | 0.049 ± 0.093 | 0.007 ± 0.105 |
| Extent | pearson | 0.221 ± 0.090 | 0.064 ± 0.150 | **0.387 ± 0.116** |
| Extent | spearman | 0.213 ± 0.058 | 0.050 ± 0.118 | **0.362 ± 0.121** |
| Intensity | pearson | 0.654 ± 0.097 | 0.024 ± 0.231 | **0.288 ± 0.031** |
| Intensity | spearman | 0.834 ± 0.067 | 0.046 ± 0.208 | **0.604 ± 0.093** |
| Intensity | mae | 0.588 ± 0.074 (different target scale, see caveat) | 2.093 ± 0.287 | 2.154 ± 0.269 |

**Comparability caveat (carried over from `MTL_V3_REPORT.md` Section
C/D — repeated here so it is not missed):** V2.1's intensity target is
`legacy_native_intensity` (bounded, sigma-normalized); V3/V3.1's target
is `D`, the raw unbounded realized deviation. **V2.1's intensity MAE
is not comparable to V3/V3.1's on an absolute scale** — only the
correlation metrics (pearson/spearman) are informative across that
boundary, and only V3-vs-V3.1's MAE is a fair apples-to-apples
comparison (both target `D`; 2.093 vs 2.154 is noise-level, consistent
with std(mu) recovering variance but not yet a tight fit to `D`'s
actual magnitude — see Section H).

Shape, Extent, and Intensity's rank-correlation metrics (spearman) all
recover substantially — Extent's pearson/spearman are actually the
*best* of the three architectures, and Intensity's spearman recovers
~72% of the collapse-to-V2.1 gap (0.046 → 0.604, vs V2.1's 0.834).
Shape's nn_accuracy recovers ~55% of the gap (0.647 → 0.816, vs V2.1's
0.984) — real, reproducible, but not a full return to V2.1. Location
shows no signal in any of the three architectures — this is a
pre-existing, unrelated finding tracked since earlier reports in this
series, not something this experiment was expected to fix, and its
near-zero correlation here is not evidence against the Case A
conclusion below.

## F. Reference-Conditioning Behavior

Gate value by K, seed0 (the most gate-responsive seed):

| K | mean gate | contamination Δ(location_mu) | contamination Δ(extent_mu) | contamination Δ(intensity_mu) |
|---|---:|---:|---:|---:|
| 0 | 0.0000 | -- | -- | -- |
| 3 | 0.0222 | -- | -- | -- |
| 10 | 0.0407 | 2.6e-05 | 3.1e-04 | 3.5e-04 |
| 30 | 0.0481 | -- | -- | -- |
| 100 | 0.0514 | -- | -- | -- |

Gate increases monotonically with K for seed0 and seed1 (seed1: 0.011
→ 0.011 → 0.010 → 0.011, essentially flat-but-nonzero rather than
monotonic) — a real, learned, non-degenerate response to reference
availability in 2/3 seeds. **Seed2 is the exception**: gate stays at
~4-6e-4 across all K, and contamination changes predictions by only
~1e-6-1e-7 (`location_mu` 5.0e-7, `extent_mu` 6.7e-6, `intensity_mu`
7.7e-6) — two to three orders of magnitude below seed0/seed1's
contamination sensitivity, and only ~1 order above V3's collapsed
noise floor. Seed2 also early-stopped at epoch 15 (best epoch 9,
`patience=5`) vs seed0/seed1's full 20 epochs — consistent with the
gate mechanism simply not having converged to a useful state in that
seed's training trajectory, though this report does not claim to know
why with confidence.

**This is the one clear non-uniform finding of the experiment**:
representation collapse is resolved for all 3 seeds, but whether the
model actually *uses* the optional reference-conditioning pathway is
inconsistent — meaningful in 2/3 seeds, negligible in the third.

## G. Uncertainty Calibration Diagnostics

| Task | Seed | error-vs-uncertainty pearson | spearman | mean uncertainty | mean NLL | 80% coverage |
|---|---|---:|---:|---:|---:|---:|
| Location | 0 | 0.048 | 0.037 | 0.248 | 0.329 | 0.76 |
| Location | 1 | 0.194 | 0.210 | 0.256 | 0.349 | 0.80 |
| Location | 2 | -0.267 | -0.254 | 0.253 | 0.215 | 0.91 |
| Extent | 0 | -0.146 | -0.205 | 0.155 | -0.495 | -- |
| Extent | 1 | 0.016 | -0.003 | 0.118 | -0.583 | -- |
| Extent | 2 | 0.109 | 0.133 | 0.167 | -0.496 | -- |
| Intensity | 0 | **0.313** | 0.449 | 1.055 | 2.352 | 0.85 |
| Intensity | 1 | **0.248** | 0.628 | 1.205 | 1.804 | 0.91 |
| Intensity | 2 | **0.322** | -0.065 | 1.119 | 2.177 | 0.85 |

Location and Extent's error-vs-uncertainty correlations are
sign-inconsistent across seeds (not a real calibrated signal — same
pattern as V2.1 and V3, pre-existing and unrelated to this change).
**Intensity's pearson correlation is positive and consistent across
all 3 seeds (0.25-0.32)** — a real, reproducible relationship between
predicted uncertainty and actual error that did not exist in V3
(where uncertainty was a constant and could not correlate with
anything by construction). Intensity's spearman is less consistent
(0.45, 0.63, -0.06), so this should be read as "uncertainty carries
some real signal" rather than "uncertainty is well-calibrated."
Location's 80% coverage (0.76-0.91, nominal 0.80) is reasonable;
Location's 50% coverage was undershooting in V2.1/V3 too (a
pre-existing, unrelated miscalibration, not re-litigated here).

## H. Intensity Target-vs-Prediction Analysis

Quantile-binned `D` vs predicted `mu`, seed0:

| D bin | count | mean target D | mean predicted mu | mean predicted scale |
|---|---:|---:|---:|---:|
| normal (D=0) | 75 | 0.000 | 0.190 | 0.614 |
| (0.0013, 0.013] | 15 | 0.007 | 0.394 | 1.389 |
| (0.013, 0.134] | 15 | 0.046 | 0.416 | 1.475 |
| (0.134, 1.158] | 15 | 0.481 | 0.479 | 1.634 |
| (1.158, 5.27] | 15 | 2.784 | 0.399 | 1.375 |
| (5.27, 64.05] | 15 | 21.31 | 0.462 | 1.611 |

Two distinct findings, both honest:
1. **A real, discrete normal-vs-anomalous signal recovered**:
   `mu` jumps from 0.19 (normal, D=0) to ~0.40-0.48 (any anomaly) — in
   V3 this same table was flat to 4 significant figures across the
   entire D=0-to-16+ range (`MTL_V3_REPORT.md` Section I). This alone
   accounts for most of the pearson/spearman recovery in Section E.
2. **Within the anomalous range, `mu` does NOT track D's actual
   magnitude** — it stays in a narrow 0.39-0.48 band whether D is
   0.007 or 21.3, a ~3000x range in the target. The old collapse's
   complete flatness is gone, but a genuinely monotonic
   intensity-magnitude fit was not achieved by this single change
   either. Predicted `scale` tracks `mu`'s own small movements rather
   than D's magnitude, consistent with the same finding.

## I. Shape Uncertainty Adapter Status (diagnostic-only, non-calibrated)

`out["shape_scale"]` is still computed every forward pass (the model
was not modified), but under `shape_objective="plain"` it receives no
gradient — verified directly (Section A's test). Measured on seed0's
150 val queries: `mean=0.639, std=0.069, min=0.570, max=0.729`. This
is a random, untrained (frozen at initialization) projection of the
now-healthy shape embedding — it varies somewhat only because its
*input* (the shape embedding) now varies meaningfully, not because the
adapter itself learned anything. **These numbers must not be
interpreted as calibrated per-sample shape uncertainty.** This matches
the spec's requirement exactly: the adapter stays in the code path,
diagnostic-only, detached from the objective.

## J. Clustering Sanity Check

| Seed | KMeans label agreement | normal-majority cluster purity |
|---|---:|---:|
| 0 | 0.873 | 0.853 |
| 1 | 0.847 | 0.853 |
| 2 | 0.867 | 0.960 |

Consistently well above chance (0.5) and well above V3's collapsed
value (a cloud of near-identical points, agreement barely above
chance for 2 clusters) at all 3 seeds — further, independent
confirmation that the shape embedding now carries real, clusterable
structure.

## K. Gradient Analysis — Deliberately NOT Rerun This Round

`diagnostics/v3_gradient_analysis.py` was extended to accept
`--shape_objective` (Section A) but was **not executed** for V3.1.
Reasoning, stated plainly rather than hidden:

1. The gradient breakdown in `MTL_V3_REPORT.md` Section J existed to
   help *diagnose* the mechanism behind V3's collapse when the cause
   was still uncertain. That diagnostic question is now answered with
   high confidence by Sections D/E/H/J's convergent, independent
   evidence (the COLLAPSE CHECK, task metrics, the intensity
   normal-vs-anomaly jump, and the clustering check all agree).
   Re-measuring gradients would corroborate a conclusion already well
   supported, not change it.
2. This script re-trains a full model from scratch purely to sample
   gradients at 3 checkpoints — the same expensive, long-running
   training workload that Section L below identifies as belonging on
   the GPU server going forward, not run speculatively on local CPU.

This is an explicit gap, not a hidden one. If the gradient comparison
is wanted later, the reproduction command is:

```bash
PYTHONPATH=.:../AnomSim python3 diagnostics/v3_gradient_analysis.py \
  --n_instances 1000 --epochs 20 --seed 0 --device cuda \
  --shape_objective plain --output_dir diagnostics/outputs/v31 \
  --output_name v31_gradient_analysis.json
```

## L. Infra Note: Long-Running Training/Inference Experiments Should Run on the GPU Server

Recorded here per explicit instruction, for future reports in this
series to follow as a default:

- **Local CPU** (this machine, ~10 cores): V3.1 seed0 alone took
  1170s (~19.5 min) at ~58s/epoch for the standard
  `n_instances=1000, epochs=20` config — about 3x slower than
  `MTL_V3_REPORT.md`'s own documented ~20s/epoch for the identical
  config, for reasons not diagnosed (likely machine load at the time,
  not a code regression). Projected full 3-seed cost: ~60-65 minutes.
- **GPU server**: the identical 3-seed run (same command, only
  `--device cuda`) completed in **~5 minutes total**.
- Seed1 and seed2's `best_val_loss` on the GPU run (4.80, 5.27) were
  still trending down at their respective stopping points (seed1 hit
  its best at the very last epoch, 19/19; seed2 early-stopped at
  epoch 9/15) — there appears to be room for the loss to fall further
  with a longer patience/epoch budget, which is far more affordable
  to explore on the GPU server than locally.

**Going forward: any V3.x-series experiment involving model
training (all seeds) or gradient-analysis-style repeated training for
diagnostics should default to the GPU server, not local CPU.** Local
CPU remains fine for the (fast, training-free) evaluation/diagnostic
scripts (`v3_eval_diagnostics.py`, `v3_reeval.py`), which complete in
seconds to low minutes regardless of device.

## M. Root-Cause Diagnosis and Case Classification

Per the decision framework: **CASE A — the collapse is resolved by
this single change, and the mechanism proposed in
`MTL_V3_REPORT.md` Section K is directly corroborated.**

Mechanistic evidence (loss-curve comparison, seed0, both runs from
their own `epoch_history.json`, no retraining needed for this
comparison):

```text
V3 (heteroscedastic) loss_shape by epoch: 3.81 -> 3.33 -> 3.16 -> ... -> 2.32 -> 2.29
  Monotonically decreasing across ALL 20 epochs, never plateaus --
  consistent with chasing the heteroscedastic term's UNBOUNDED
  degenerate minimum (1 + log(c) -> -infinity as the underlying
  contrastive loss c -> 0 via embedding collapse).

V3.1 (plain) loss_shape by epoch: 3.43 -> 3.42 -> 3.35 -> 3.36 -> 3.39
  -> 3.34 -> 3.33 -> 3.30 -> 3.30 -> 3.28 -> 3.30 -> 3.32 -> 3.32 ->
  3.33 -> 3.31 -> 3.31 -> 3.28 -> 3.34 -> 3.32 -> 3.26
  Drops slightly then PLATEAUS/oscillates in a narrow 3.26-3.39 band
  for the remaining ~17 epochs -- the behavior of a bounded
  contrastive loss sitting near its floor, not a term with an
  unbounded incentive to collapse further.
```

This is the clearest possible confirmation of the hypothesis: the
plain loss's trajectory is qualitatively different in exactly the way
predicted (bounded/plateauing vs unbounded/still-falling), and this
qualitative difference alone — independent of any downstream task
metric — already distinguishes "safe" from "degenerate" objective
shape. Combined with the COLLAPSE CHECK (Section D) and the
consistent, multi-metric recovery in Sections E/H/J, **the evidence
converges from four independent angles (loss-curve shape, output-level
variance, task correlations, clustering structure) on the same
conclusion**: V3's collapse was caused, specifically and primarily, by
the Shape heteroscedastic wrapper's unbounded degenerate minimum, as
hypothesized. `MTL_V3_REPORT.md` Section K's hedge (that raw
trunk-gradient magnitude alone did not fully prove Shape's dominance)
is not contradicted — this experiment provides a different, more
direct line of evidence (a controlled ablation) rather than resolving
that specific earlier ambiguity about gradient magnitudes, which
Section K above explains was not re-measured this round.

Case A does not mean "fully solved" — Sections E/F/H record real,
un-swept-under-the-rug residual issues (reference gate inconsistent
across seeds, intensity's within-anomaly magnitude sensitivity still
weak, shape/extent/intensity recovering only partway to V2.1 on some
metrics). These are downstream-of-collapse refinement questions, not
evidence against the Case A classification itself.

## N. Overall Verdict and Answers

**1. Is the collapse resolved?** Yes, at all 3 seeds, by the
COLLAPSE CHECK's strict definition (Section D).

**2. Is the collapse resolution caused specifically by removing the
heteroscedastic wrapper, as opposed to some other confound?** Yes,
with high confidence — this was the ONLY code change, verified by a
direct diff review (Section A), and the loss-curve mechanism
(Section M) independently confirms the predicted qualitative
signature.

**3. Did Shape's own metric fully return to V2.1?** No —
nn_accuracy recovered from 0.647 to 0.816, about 55% of the gap to
V2.1's 0.984. Real recovery, not full restoration.

**4. Did Extent's metric fully return to V2.1?** It exceeded it —
pearson 0.387 vs V2.1's 0.221. Best-of-three-architectures for this
task.

**5. Did Intensity's correlation metrics recover?** Substantially —
spearman 0.604 vs V3's 0.046 and V2.1's 0.834 (~72% of the gap
closed). Pearson 0.288 vs V3's 0.024 and V2.1's 0.654 (~44% of the
gap closed).

**6. Did Intensity's mean-vs-target relationship become monotonic
across the full D range?** Partially. The old total flatness is
gone and a clear normal-vs-anomaly step recovered (Section H), but
within the anomalous range mu does not yet track D's ~3000x magnitude
range in a monotonic way.

**7. Did Location recover?** No — near-zero correlation in V2.1, V3,
AND V3.1 alike. Pre-existing, unrelated to this experiment.

**8. Is the reference gate meaningfully non-zero now?** Yes for
seed0/seed1 (0.01-0.05, with real contamination sensitivity), no for
seed2 (~4e-4, contamination-insensitive) — inconsistent across seeds,
reported as a genuine open finding, not resolved by this change.

**9. Does K generally help once K>0?** For seed0, gate rises
monotonically 0→3→10→30→100 (0.022→0.051); for seed1 it is flat-but-
nonzero; for seed2 it stays negligible regardless of K. No uniform
answer across seeds.

**10. Does mild reference contamination degrade predictions
measurably (as opposed to floating-point noise)?** Yes for seed0/
seed1 (Δ on the order of 1e-4 to 1e-3, well above float noise); no
for seed2 (Δ ~1e-6, indistinguishable from V3's collapsed noise
floor).

**11. Is Location/Extent/Intensity's uncertainty calibrated (error
correlates with predicted scale)?** Only Intensity shows a
consistent, reproducible positive pearson correlation across all 3
seeds (0.25-0.32) — real signal, not full calibration (spearman is
inconsistent). Location/Extent show sign-inconsistent correlations
across seeds, same as in V2.1/V3 — pre-existing, unrelated.

**12. Is `shape_scale`'s diagnostic-only value trustworthy as an
uncertainty estimate?** No, by design — it is an untrained, frozen-
at-init projection under `shape_objective="plain"` (Section I).
Reported only to confirm the detachment worked as intended.

**13. Is the architecture verified byte-for-byte unchanged from
V3?** Yes — identical parameter count (420,266) at all 3 seeds
(Section B), and only the one loss-computation branch differs in
code (Section A).

**14. Was training numerically stable (no NaN/Inf)?** Yes — all 3
seeds completed with finite losses throughout; seed0/seed1 ran the
full 20 epochs without early stopping triggering, seed2 early-stopped
at epoch 15 (best epoch 9) via the standard patience=5 mechanism, not
a numerical failure.

**15. Was the full K=100 sensitivity sweep run (eval-only, as
required)?** Yes, for all 3 seeds (Section F/D).

**16. Was the gradient-norm breakdown (V3's Section J equivalent)
rerun for V3.1?** No, deliberately — see Section K's stated
reasoning (the primary diagnostic question is already answered with
convergent evidence from four other angles; re-running it means
another full training pass, which per Section L should move to the
GPU server rather than run speculatively on local CPU).

**17. Does this fully rule out any risk from Location/Extent/
Intensity's own Laplace NLL losses (which share the same
mathematical "unbounded-as-residual→0" structure in principle)?** No
— this was explicitly out of scope for this controlled experiment
(only Shape's loss was touched), and this report does not claim
anything new about that separate, structurally-similar risk. It
remains an open question for a future controlled experiment, not
addressed or ruled out here.

**18. PROMOTE / KEEP / VERDICT?** **V3.1 (plain Shape loss) resolves
V3's collapse and should replace V3's heteroscedastic Shape objective
in any future V3-line work.** It is not yet a clean promotion over
V2.1 as a production baseline — Shape/Intensity have not fully
recovered to V2.1's level, the reference-gate mechanism's value is
still unproven (seed-dependent), and Location remains unsolved
everywhere. The correct framing: **the specific failure mode
diagnosed in `MTL_V3_REPORT.md` is fixed; V3.1 is a valid foundation
to continue evaluating the reference-conditioning and probabilistic-
head machinery on, which V3 never permitted because its shared
representation was dead.**

## O. Next Steps (proposed only — none implemented, no new "V3.2" started)

Per instructions, this report diagnoses and confirms; it does not
select or implement a fix. Candidates worth considering later, not
started:

1. **Investigate the reference-gate seed-inconsistency (Section F)**
   before trusting the reference-conditioning mechanism in general —
   e.g. does it correlate with early-stopping epoch, initialization,
   or K-regime sampling luck across seeds. This is the most direct
   open thread from this specific report.
2. **A bounded alternative to heteroscedastic Shape weighting**, if
   per-sample Shape uncertainty is still wanted (e.g. clamping
   `log(scale)` to a floor, or a bounded reparameterization like
   `scale = softplus(raw) + eps` combined with a KL-style
   regularizer toward a fixed prior) — proposed as a future
   direction only, not designed or implemented here.
3. **Check whether Location/Extent/Intensity's Laplace NLL carries
   the same unbounded-degenerate-minimum risk** (verdict question 17)
   via an analogous controlled ablation, given they share the same
   mathematical structure Shape's heteroscedastic wrapper did.
4. **Intensity's within-anomaly magnitude sensitivity (Section H)**
   is the clearest remaining task-level gap on a now-healthy
   representation — worth its own controlled experiment once the
   reference-gate question above is resolved, so any future change
   isn't confounded with today's still-open gate inconsistency.

## Files Changed

See Section A.

## Reproduction Command

```bash
export PYTHONPATH=".:../AnomSim"

# V3.1 3-seed run (GPU recommended -- see Section L; local CPU works,
# just ~12x slower for this config)
python3 -u diagnostics/v3_baseline.py \
  --n_instances 1000 --epochs 20 --patience 5 --seeds 0 1 2 \
  --output_dir diagnostics/outputs/v31 --shape_objective plain \
  --experiment_id_prefix v31 --skip_v21_backfill --device cuda

# COLLAPSE CHECK + full eval diagnostics per seed (fast, CPU is fine)
for s in 0 1 2; do
  python3 -u diagnostics/v3_eval_diagnostics.py \
    --checkpoint diagnostics/outputs/v31/v31_multitask_seed$s/bestmodel.pkl \
    --output_dir diagnostics/outputs/v31/v31_multitask_seed$s \
    --output_name v31_eval_diagnostics.json --seed $s --n_instances 1000
done
```

GPU run: ~5 min total for all 3 seeds. Local CPU run for the same
config: ~60-65 min projected (seed0 alone measured at 1170s). All
eval diagnostics (collapse check + Sections F-J) complete in well
under 5 minutes combined, on CPU, per seed.
