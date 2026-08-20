# MTL Diagnostic Report

All planned Phase 1 + Phase 2 items are now complete (3-seed confirmation for
location_only/extent_only/multitask landed -- see Section 3/6 for the resulting
correction to the originally-reported extent numbers).

## 1. Architecture Verification

```
Actual architecture (n_time_max=550 default):

Input (B,1,550)
Stem: Conv1d(1->16,k=3,s=1,pad=1,reflect) + GroupNorm(8,16)          -> (B,16,550)
Stage0: Conv1d(16->16,k=3,s=2,pad=1) + GroupNorm                     -> (B,16,275)   [no attention]
Stage1: Conv1d(16->32,s=2) + GroupNorm + SelfAttention(32ch,4head)   -> (B,32,138)
Stage2: Conv1d(32->64,s=2) + GroupNorm + SelfAttention(64ch,4head)   -> (B,64,69)
Stage3: Conv1d(64->128,s=2)+ GroupNorm + SelfAttention(128ch,4head)  -> (B,128,35)
Squeeze: Conv1d(128->4, k=1)                                         -> (B,4,35)
[sinusoidal positional encoding added here, ONLY place it appears]  -> (B,4,35)
Attention Pooling: learned query(1,1,4) + MultiheadAttention(4ch,1head) -> z:(B,4)
4x Linear(4->16) heads (independent) -> shape/location/extent/intensity embedding (B,16) each
```

Differences / confirmations vs. the provided description:

- Attention attached only at Stage1/2/3 (nominal length <= attention_max_resolution=256).
  Stage0 (275) excluded. VERIFIED by reading `compute_num_stages`/`ConvBottleneckEncoder.__init__`.
- **Stage1-3 SelfAttentionBlock has NO positional encoding of its own** -- grepped
  `models_conv_bottleneck.py` for "positional/pos_enc" -- zero matches. The ONLY
  positional encoding in the whole model is added once, in `models_contrastive.py`,
  right before the pooling attention (sinusoidal, on a continuous fractional
  position so it stays valid across variable lengths).
- **key_padding_mask IS correctly propagated and used in Stage1-3 self-attention.**
  VERIFIED by reading `ConvBottleneckEncoder.forward` (models_conv_bottleneck.py:227-238):
  the pad_mask is downsampled via `F.max_pool1d` with the SAME kernel_size/stride/
  padding as the feature conv at every stride-2 stage (so mask and feature time-length
  always match exactly), and `key_padding_mask = (m[:,0,:] < 0.5)` (True=ignore,
  matching PyTorch's convention) is passed into every `SelfAttentionBlock` call.
  This is NOT an architecture issue -- flagged as requested, but the mask handling
  itself is correct.
- Optimizer: the MAIN pipeline (`core_clustering/trainer_contrastive.py`) uses FOUR
  per-attribute AdamW optimizers, each covering the shared trunk (encoder + pool_attn +
  pool_query, trunk lr = base_lr/4) plus only that attribute's own head, with
  sequential per-attribute backward+step every batch (via `torch.autograd.grad`,
  computed all-at-once before any step to avoid corrupting the retained graph, then
  applied shape -> location -> extent -> intensity in order). **This diagnostic
  campaign (Phase 1) deliberately does NOT use that trainer** -- it uses a new
  `diagnostics/simple_trainer.py` with ONE plain AdamW over all model+loss
  parameters, to avoid a real confound: 4 separate optimizers each independently
  apply AdamW's decoupled weight decay to the same shared trunk every step, so a
  naive `weights=(1,0,0,0)` single-task run on the per-attribute-optimizer trainer
  would still get ~4x the weight-decay shrinkage a true single-task setup would.
- Head: `Linear(4->16)`, bias, 4 independent instances (ModuleDict). Not MLP.
- z_dim = bottleneck_channels = 4 (config default).

## 2. Key Findings (Phase 1, screening, 1 seed)

1. **Location fails to learn even in SINGLE-TASK mode** (pearson=-0.050,
   spearman=-0.049, i.e. no better than chance) -- this was NOT expected going in;
   prior assumption was that location-only should learn fine and multi-task
   competition was the problem. This single-task result says the location problem
   is likely NOT primarily a gradient-conflict/MTL issue -- something is wrong
   before multi-task sharing even enters the picture (representation/architecture
   candidate, to be tested in Phase 2).
2. **Extent learns well alone (pearson=0.773 at seed0; 3-seed-confirmed mean
   0.493±0.198, see Section 3) but degrades severely under multi-task
   (pearson=0.207 at seed0; 3-seed mean 0.168±0.031)** -- a genuine, large
   single-task-to-multi-task gap, though the seed0-only 0.773 number was an
   outlier (Section 3 has the corrected picture). This IS consistent with real
   MTL interference (gradient conflict or shared-
   trunk capacity competition) specifically for extent.
3. **Intensity is robust to multi-task** (single: pearson=0.897 vs multi:
   pearson=0.909 -- no degradation, if anything marginally better).
4. **Shape's nearest-neighbor classification accuracy is roughly unaffected by
   multi-task** (single: 0.827 vs multi: 0.833), but the absolute positive/
   negative pair separation shrinks drastically (single: 1.199 vs multi: 0.062)
   -- the embedding's absolute scale compresses under multi-task, but the
   relative/rank structure needed for 1-NN classification survives.
5. All 5 screening runs converged (early-stopped by patience=5) in well under a
   minute of wall-clock each on the server (7-15 epochs run out of a 20-epoch
   budget) -- no divergence observed in this run (contrast with the earlier,
   pre-diagnostic finding of extent loss exploding to 78 under the OLD
   learned-query-pool-without-warmup setup; this SimpleTrainer + current
   architecture did not reproduce that divergence at this scale/budget).
6. **Cross-task metrics inside a single-task run are meaningless and must be
   ignored** -- e.g. `shape_only`'s reported "extent" metrics (mae=3.56) reflect
   an UNTRAINED, randomly-initialized extent head (weight=0 means that head
   never received gradient), not evidence about extent's learnability. Only the
   diagonal (task's own metric under its own single-task run) is meaningful.

## 3. Single-task vs Multi-task

Only each task's OWN metric is shown (see Finding 6 -- off-diagonal metrics from
single-task runs are from an untrained head and are not meaningful). Shape and
intensity single-task were screened at 1 seed only (per the user's own
instruction -- no meaningful single-vs-multi difference was seen at screening,
so no 3-seed confirmation was warranted). Location and extent single-task,
plus multitask, were confirmed at 3 seeds (0,1,2) -- shown as mean ± std below.

| Task | Metric | Single-task | Multi-task | Change |
|---|---|---|---|---|
| Shape | nn_accuracy (seed0 only) | 0.827 | 0.833 | ~unchanged |
| Shape | pos/neg separation (seed0 only) | 1.199 | 0.062 | large drop (scale compresses; classification unaffected) |
| Location | pearson, 3-seed mean±std | -0.011 ± 0.062 | not separately re-aggregated (screening: 0.018) | no signal in either condition |
| **Extent** | **pearson, 3-seed mean±std** | **0.493 ± 0.198** | **0.168 ± 0.031** | **large drop, ~66% relative -- but see correction below** |
| Intensity | pearson (seed0 only) | 0.897 | 0.909 | no degradation |

**Correction from the original screening report**: the originally reported
extent_only pearson of 0.773 was seed0 only and turned out to be an outlier --
seeds 1,2 gave 0.349 and 0.358 (3-seed mean 0.493, std 0.198, i.e. extent_only
itself is highly seed-sensitive: it ranges from 0.35 to 0.77 depending on
seed). Multitask's extent pearson, by contrast, is comparatively STABLE
across seeds (0.207, 0.130, 0.167 -- std only 0.031). **Net effect: the
direction of the original finding holds (multi-task degrades extent, ~66%
relative drop in the mean) but the originally-reported magnitude (0.773 ->
0.207, a 73% drop) was inflated by one lucky single-task seed.** This
seed-instability of extent_only itself (short training: early-stopped after
just 6-9 of 20 epochs each time, patience=5) is a new finding in its own
right, separate from the multi-task question.

Full metrics (including the non-meaningful off-diagonal single-task numbers) are
preserved in `diagnostics/outputs/phase1/experiment_results.json` and per-experiment
`metrics.json` files -- nothing was deleted or summarized away.

**Answering the key question ("Single-task에서는 잘 되는데 Multi-task에서만
악화되는 task가 있는가?"): YES for extent, confirmed across 3 seeds (0.49 mean
single-task -> 0.17 mean multi-task) -- a real, large relative drop (~66%),
though less dramatic than the initial single-seed 0.773-vs-0.207 comparison
suggested. Location is
bad in BOTH conditions (not an MTL-specific degradation -- a prior, more basic
problem). Shape and intensity show no meaningful MTL degradation on their primary
metrics.**

## 4. Representation Probe

**Location probe (target=location): COMPLETE.**

| Representation | Probe | Pearson | Spearman |
|---|---|---:|---:|
| Stage2 | linear | -0.212 | -0.190 |
| Stage2 | mlp | 0.132 | 0.149 |
| Stage3 | linear | 0.211 | 0.233 |
| Stage3 | mlp | 0.148 | 0.143 |
| **Squeeze** | linear | 0.179 | 0.122 |
| **Squeeze** | **mlp** | **0.261 (best of all 8)** | 0.235 |
| **Pool z** | linear | -0.033 | 0.007 |
| **Pool z** | mlp | -0.029 | 0.045 |

(Checkpoint: `phase1_location_only_seed0`, n_val=75 anomalous instances; full
metrics incl. MAE/RMSE in `diagnostics/outputs/phase2/location_probe_results.json`.)

**Interpretation -- this localizes the failure precisely.** Location
information is weak but genuinely PRESENT at Stage2/Stage3, and actually
strongest at Squeeze (mlp pearson=0.261, the best result across all 8
representation/probe combinations). It then collapses to ~0 at Pool z
(-0.029 to -0.033) -- **the information loss happens specifically at the
attention-pooling step**, not in the conv/attention trunk (which retains
some signal) and not at the squeeze (which has the cleanest signal of any
representation tested). The trunk is not blind to location; the single-
query learned attention-pool is failing to preserve/aggregate it. This is
consistent with the pool's learned query having converged toward "attend to
the largest deviation" (useful for intensity) rather than anything
positionally discriminative, even in this location-ONLY run with zero
competing tasks.

**Extent probe (target=extent): COMPLETE**, run on TWO checkpoints (Phase 1's
`extent_only` and `multitask` bestmodel.pkl) to localize where extent's
information is lost between single-task and multi-task training.

| Representation | Probe | Extent-only checkpoint | Multitask checkpoint |
|---|---|---:|---:|
| Stage2 | linear | -0.080 | -0.161 |
| Stage2 | mlp | 0.014 | 0.042 |
| Stage3 | linear | -0.365 | -0.009 |
| Stage3 | mlp | 0.520 | 0.074 |
| Squeeze | linear | 0.615 | 0.187 |
| Squeeze | mlp | 0.544 | 0.437 |
| Pool z | linear | **0.630** | **0.043** |
| Pool z | mlp | **0.764** | **-0.011** |

(Values are Pearson correlation between the probe's prediction and true extent
value, on held-out val instances; full metrics incl. Spearman/MAE/RMSE in
`diagnostics/outputs/phase2/extent_probe_{extentonly,multitask}/extent_probe_results.json`.)

**Interpretation**: in the extent-only checkpoint, extent information is
present but not cleanly LINEARLY accessible until Squeeze/Pool (Stage3 is messy:
linear=-0.365, mlp=0.520 -- present but nonlinearly tangled; Squeeze/Pool clean
it up to 0.54-0.76). In the MULTITASK checkpoint, this same distillation
collapses -- Pool z drops to ~0 for BOTH probe types (0.043 linear, -0.011
mlp). Since even a freshly-trained, full-capacity MLP probe cannot recover
extent information from the multitask trunk's Pool z, this is not "the extent
head just isn't reading available info" (which would point at head capacity,
Hypothesis C) -- **the shared trunk's representation itself loses extent
information under multi-task training** (supports Hypothesis F/E, not C).

## 5. Latent Dimension Ablation

NOT RUN. Phase 3, conditional on Phase 2 findings per the adaptive-ablation principle.

## 6. Head Ablation

NOT RUN. Phase 4, conditional.

## 7. Head LR Ablation

NOT RUN. Phase 5, conditional.

## 8. Gradient Norms

Sampled 15 batches each at early (10%), middle (50%), late (90%) of a 20-epoch
multitask training run (n_instances=1000, seed=0, `SimpleTrainer`'s single
combined AdamW). Norms are of each task's weighted-loss gradient flattened
across the shared trunk (encoder + pool_attn + pool_query).

| Task | Early (mean±std) | Middle (mean±std) | Late (mean±std) |
|---|---:|---:|---:|
| shape | 0.215 ± 0.083 | 0.084 ± 0.027 | 0.104 ± 0.019 |
| location | 4.013 ± 3.678 | 1.217 ± 0.833 | 0.747 ± 0.352 |
| extent | 6.069 ± 3.957 | 13.134 ± 9.658 | **59.858 ± 32.384** |
| intensity | 21.109 ± 16.089 | 15.514 ± 8.206 | **56.046 ± 31.521** |

**Large, real magnitude imbalance**: shape's gradient stays tiny (~0.08-0.22)
throughout, while extent and intensity are 30-700x larger and, notably,
extent/intensity's norms GROW substantially from early to late (extent:
6->60, intensity: 21->56) rather than shrinking as training converges --
location shrinks (4.0->0.75) and shape stays flat. In a single shared
optimizer that sums all four weighted losses before backward, shape's signal
is numerically dwarfed by extent/intensity's by 2-3 orders of magnitude.
**Supports Hypothesis D (gradient magnitude imbalance exists) -- strongly.**

## 9. Gradient Cosine Similarity

Same sampling as Section 8. Mean cosine similarity per pair, per training segment:

| Pair | Early | Middle | Late |
|---|---:|---:|---:|
| shape vs location | 0.401 | -0.153 | 0.254 |
| shape vs extent | 0.105 | -0.008 | -0.179 |
| shape vs intensity | -0.160 | 0.129 | 0.036 |
| location vs extent | 0.176 | **0.593** | -0.166 |
| location vs intensity | -0.365 | -0.446 | 0.223 |
| **extent vs intensity** | **-0.278** | **-0.516** | **-0.264** |

Conflict rate (fraction of the 15 sampled batches with NEGATIVE cosine) for
extent vs intensity: early 0.67, middle 0.73, late 0.67 -- **consistently
majority-conflicting across all three training phases**, the only pair with
this property. location vs extent is a striking outlier in the other
direction: strongly ALIGNED at middle training (0.593, 0/15 negative) but
flips to conflicting by late (-0.166) -- a non-monotonic relationship, not a
stable alignment or a stable conflict.

**Supports Hypothesis E (directional conflict) specifically for extent vs
intensity** -- the most consistent negative-cosine pair across all segments,
and (per Section 8) also two of the largest-magnitude gradients, so this
conflict acts on a large fraction of the trunk's actual movement, not a
minor component.

**However**, this does not fully explain Section 3's pairwise ablation
(Section 4-adjacent, see below): extent+shape (paired with the SMALLEST,
least-consistently-conflicting gradient) degraded extent MORE than
extent+intensity did. Directional conflict is real and evidenced, but is not
established as the SOLE or even primary driver of extent's degradation --
some capacity-sharing or other mechanism may also be contributing (see
Section 12).

## 10. Loss Scale

Raw loss magnitude was not separately instrumented in Section 8/9's run;
gradient norm (which already reflects both loss scale and its sensitivity to
trunk parameters) is used as the proxy per the report template's intent.
See Section 8 -- extent/intensity's gradients are both far larger than
shape's/location's AND growing late in training, consistent with these two
raw-value regression losses (targets in real units: extent 0.05-0.5,
intensity 0.2-4.0) producing much larger loss curvature than shape's
softmax-normalized contrastive loss or location's bounded pairwise-gap
regression.

## Problem B.3: Cheap Pairwise Task Ablation (extent + one other task)

Same SimpleTrainer/config as Phase 1, seed=0 only (not 3-seed confirmed --
given extent_only's seed-sensitivity discovered in Section 3, treat this
table's exact values with the same caution; the ranking/direction is still
informative). Compare against Phase 1's own seed0 extent_only (0.773, now
known to be an outlier -- see Section 3) and seed0 4-task multitask (0.207):

| Combination | Extent pearson | Extent spearman | best_val_loss |
|---|---:|---:|---:|
| extent_only (Phase 1) | 0.773 | 0.761 | 0.018 |
| extent + location | 0.280 | 0.339 | 0.131 |
| extent + intensity | 0.225 | 0.299 | 0.745 |
| 4-task multitask (Phase 1) | 0.207 | 0.277 | 4.731 |
| **extent + shape** | **0.085** | **0.154** | 3.503 |

**Surprising finding**: extent+shape is the WORST pairwise combination --
worse than the full 4-task multitask, and worse than extent+intensity despite
shape carrying the gradient that is smallest in magnitude (Section 8) and
only inconsistently anti-correlated with extent (Section 9: +0.105 early,
~0 middle, -0.179 late -- not a strong, consistent conflict like extent-vs-
intensity). This is NOT fully explained by either the magnitude-imbalance
story (shape's gradient is tiny, so a pure combined-magnitude argument would
predict shape gets swamped BY extent, not the reverse) or the cosine-conflict
story (shape-extent conflict is weak/inconsistent, unlike extent-intensity's
strong consistent conflict). ANY pairing degrades extent substantially
relative to solo training -- multi-task sharing itself, not one specific
antagonist, may be the dominant effect, though this pairwise result is
1-seed screening and could partly reflect run-to-run noise at this small
scale (candidate for 3-seed confirmation if pursued further).

## 11. Embedding Diagnostics

**Location sanity check (Problem A.1) -- COMPLETE**, on the `location_only`
seed0 checkpoint (75 anomalous val instances, 2775 pairs):

- **Oracle embedding check**: a hand-built "perfect" embedding
  (`e_i = [location_i, 0, ..., 0]`) gives loss = 9.5e-12 (~0) under
  `PairwiseGapRegressionLoss`. **The loss formula itself is sane** -- it is
  not structurally broken or unsatisfiable.
- **Tiny network optimizability check**: a 2-layer MLP taking the RAW
  location value as its ONLY input (bypassing the encoder entirely) trained
  with the SAME loss goes from initial loss 0.0022 to final loss 8.2e-7 in
  300 steps. **The loss is readily optimizable via gradient descent** given a
  maximally-informative input -- optimization difficulty is not the problem
  either.
- **Embedding collapse stats**: mean_per_dim_std=0.038, min_per_dim_std=0.0089
  (nonzero -- not collapsed), mean_embedding_norm=3.39, mean_pairwise_distance
  =0.206. Predicted-distance distribution spans 0.006 to 0.784 (real,
  non-degenerate spread) -- **not a collapse failure**.
- Yet regression_metrics(predicted_distance, true_gap): pearson=-0.050.

**Conclusion for Problem A**: the encoder's location-only model produces
embeddings that vary (not collapsed) and could in principle satisfy the loss
(which is sane and optimizable given the right input) -- but the VARIATION
the encoder actually produces is uncorrelated with true location. This
narrows Problem A specifically to **"the encoder fails to extract usable
location information from the raw series"** -- not a loss bug, not
optimization failure, not embedding collapse. The location representation
probe (Section 4, still pending) is what would localize WHERE in the trunk
this information is lost, but the failure point is now known to be encoder-
side, not loss/optimizer-side.

## 12. Diagnosis

| Hypothesis | Verdict | Evidence (one line) |
|---|---|---|
| A. z_dim=4 bottleneck is the main problem | INCONCLUSIVE | No ablation run yet (Phase 3 not started) |
| B. Shared pooling is the main information-loss cause (for location) | **SUPPORTED** | Section 4: probe pearson is 0.13-0.26 at Stage2/Stage3/Squeeze (best at Squeeze, 0.261) and collapses to ~-0.03 at Pool z -- the loss is localized specifically to the attention-pooling step, in a location-ONLY run with zero task competition |
| C. Linear head capacity is insufficient (for extent) | NOT SUPPORTED | Section 4: a fresh MLP probe (full capacity) on the multitask trunk's Pool z still gets pearson=-0.011 -- the info isn't there for ANY probe to find, not a head-capacity limitation |
| D. Gradient magnitude imbalance exists | **SUPPORTED** | Section 8: shape's gradient (~0.1) is 30-700x smaller than extent/intensity's (6-60), and extent/intensity's norms GROW late in training rather than converging |
| E. Gradient directional conflict exists (extent vs intensity specifically) | **SUPPORTED** | Section 9: extent-vs-intensity cosine is negative in all 3 sampled segments (mean -0.28/-0.52/-0.26), majority-conflicting (67-73% of batches) -- the only pair with this consistency |
| F. Trunk lacks capacity for all 4 attributes simultaneously (for extent) | SUPPORTED as a contributor, not sole cause | Section 4: extent info is lost from the trunk's OWN representation under MTL (not just unread by the head); Section B.3: EVERY pairwise combination hurts extent substantially, not just the extent-vs-intensity conflict pair, suggesting general capacity/interference beyond one specific antagonist |
| F (for location) | NOT SUPPORTED as sole cause | Location fails even with ZERO competition (single-task) -- capacity-sharing cannot be the reason location fails, since there's nothing to share with in that condition |
| G. Multi-task sharing provides positive transfer over single-task | NOT SUPPORTED (extent, location) / INCONCLUSIVE (shape, intensity) | Extent got worse under MTL (3-seed mean 0.49->0.17, worse still in 3 of the pairwise combos); location never worked either way; shape/intensity roughly flat |

**Open/surprising result not yet explained**: extent+shape (Section B.3) degrades
extent MORE than extent+intensity, despite shape's gradient being far smaller in
magnitude (Hypothesis D would predict shape gets dominated, not the reverse) and
only weakly/inconsistently anti-correlated with extent (Hypothesis E is weak for
this specific pair). Neither magnitude imbalance nor directional conflict alone
explains this ranking -- flagged as INCONCLUSIVE pending further investigation,
not glossed over.

## 13. Recommended Next Architecture

For LOCATION: the failure point is now localized to the attention-pooling step
specifically (Section 4/12, Hypothesis B supported) -- a pooling redesign
(e.g. multi-query attention pool, or restoring/strengthening positional
encoding INTO the pooling step specifically rather than the trunk) is a
well-evidenced candidate. Not implemented here per the diagnostic principle
of confirming before changing, but this is no longer speculative -- it is
the one part of the architecture directly shown (via frozen-representation
probing, not just circumstantial reasoning) to be where location's already-
present signal gets destroyed.

For EXTENT, gradient
magnitude imbalance (D) and extent-vs-intensity directional conflict (E) are
now both supported by direct measurement (Sections 8-9), and Section 4 shows
the multitask trunk's own representation loses extent info (not a head-
capacity issue, C not supported) -- but Section B.3's extent+shape result is
not yet explained by either D or E alone, so a specific fix (e.g. gradient
normalization only on extent-vs-intensity) would be premature without
understanding why extent+shape is even worse. Recommend resolving the
open/surprising result (Section 12) before proposing an architecture change.

## 14. What NOT to Change Yet

- PCGrad / GradNorm / any gradient balancing: not introduced, per instructions.
  Extent's single-vs-multi gap is suggestive but not yet confirmed as a
  directional-conflict problem specifically (could also be capacity-sharing,
  which gradient balancing wouldn't fix).
- z_dim, head architecture, head LR: no ablation run yet, nothing to base a
  change on.
- Positional encoding in Stage1-3: the representation probe (Section 4) shows
  location info is already present (weakly) at Stage2/Stage3/Squeeze and is
  lost specifically AT Pool z -- so adding positional encoding to the Stage1-3
  self-attention blocks is not indicated by the evidence; the trunk isn't
  where the information is lost. A pooling redesign (Section 13) is the
  evidence-backed candidate instead. Still not implemented here, per the
  principle of confirming before changing -- this note is updated from the
  earlier (correct at the time) "still needs localizing" state now that
  Section 4 has localized it.

## 15. Files Changed

- `diagnostics/simple_trainer.py` (new): single-optimizer trainer for fair
  single-task/multi-task comparison (see Section 1 rationale).
- `diagnostics/metrics.py` (new): location/normal-relative/shape/collapse metrics.
- `diagnostics/phase1_baselines.py` (new): Phase 1 experiment runner.
- `diagnostics/phase2_location_sanity.py` (new): Problem A.1 -- oracle embedding
  check, tiny-network optimizability check, distance-distribution/collapse stats.
- `diagnostics/representation_probe.py` (new): frozen Stage2/Stage3/Squeeze/
  Pool-z extraction (replicates the model's own submodules read-only) +
  Linear/MLP probe training. Reused for both location and extent probes.
- `diagnostics/phase2_location_probe.py` (new): Problem A.2/B.5 runner
  (`--target location|extent|intensity`).
- `diagnostics/phase2_pairs.py` (new): Problem B.3, extent+one-other-task
  pairwise ablation, reuses `phase1_baselines.run_experiment` directly.
- `diagnostics/phase2_gradient_analysis.py` (new): Problem B.4, per-task
  gradient norm + pairwise cosine similarity sampled at early/mid/late
  training segments during a normal multitask run.
- Two bugs found and fixed after the first server run: (1)
  `cache_all_representations` fed raw un-padded variable-length series into
  the model (no pad_mask), causing a shape mismatch across instances --
  fixed by explicitly padding to `max_len` with a matching pad_mask,
  mirroring `contrastive_pad_collate`'s convention; (2) a gradient tensor's
  `.numpy()` call was missing `.cpu()`, working on CPU but failing on CUDA.
- Main pipeline (`core_clustering/`) untouched by this diagnostic work.

## 16. Reproduction Commands

```bash
export PYTHONPATH=".:../AnomSim"
mkdir -p diagnostics/outputs

for mode in shape_only location_only extent_only intensity_only multitask; do
  python3 -u diagnostics/phase1_baselines.py \
    --modes $mode \
    --n_instances 1000 --epochs 20 --patience 5 --seed 0 \
    --device cpu \
    --output_dir diagnostics/outputs/phase1 \
    > diagnostics/outputs/phase1_${mode}.log 2>&1 &
done
wait
cat diagnostics/outputs/phase1/experiment_results.json
```

Screening: 1 seed (seed=0), all 5 modes. Confirmatory 3-seed re-verification for
location_only/extent_only/multitask (seeds 1,2 in addition to seed0) is COMPLETE
-- see Section 3 for the corrected, seed-averaged extent numbers. Command used:

```bash
for seed in 1 2; do
  for mode in location_only extent_only multitask; do
    python3 -u diagnostics/phase1_baselines.py \
      --modes $mode --n_instances 1000 --epochs 20 --patience 5 --seed $seed --device cuda \
      --output_dir diagnostics/outputs/phase1 \
      > diagnostics/outputs/phase1_${mode}_seed${seed}.log 2>&1 &
  done
  wait
done
```

Phase 2 commands (Problem A.1/A.2, Problem B.3/B.4/B.5):

```bash
export PYTHONPATH=".:../AnomSim"
mkdir -p diagnostics/outputs/phase2

# Problem A: location sanity + probe
python3 -u diagnostics/phase2_location_sanity.py \
  --checkpoint diagnostics/outputs/phase1/phase1_location_only_seed0/bestmodel.pkl \
  --n_instances 1000 --seed 0 --output_dir diagnostics/outputs/phase2 &
python3 -u diagnostics/phase2_location_probe.py \
  --checkpoint diagnostics/outputs/phase1/phase1_location_only_seed0/bestmodel.pkl \
  --n_instances 1000 --seed 0 --target location \
  --output_dir diagnostics/outputs/phase2 &
wait

# Problem B: pairwise ablation
for pair in extent_shape extent_location extent_intensity; do
  python3 -u diagnostics/phase2_pairs.py \
    --pairs $pair --n_instances 1000 --epochs 20 --patience 5 --seed 0 --device cuda \
    --output_dir diagnostics/outputs/phase2/pairs &
done
wait

# Problem B: gradient analysis + extent probes (2 checkpoints)
python3 -u diagnostics/phase2_gradient_analysis.py \
  --n_instances 1000 --epochs 20 --seed 0 --device cuda \
  --output_dir diagnostics/outputs/phase2 &
python3 -u diagnostics/phase2_location_probe.py \
  --checkpoint diagnostics/outputs/phase1/phase1_extent_only_seed0/bestmodel.pkl \
  --n_instances 1000 --seed 0 --target extent \
  --output_dir diagnostics/outputs/phase2/extent_probe_extentonly &
python3 -u diagnostics/phase2_location_probe.py \
  --checkpoint diagnostics/outputs/phase1/phase1_multitask_seed0/bestmodel.pkl \
  --n_instances 1000 --seed 0 --target extent \
  --output_dir diagnostics/outputs/phase2/extent_probe_multitask &
wait
```

## Resource Efficiency

```
Hardware:
- Server: 10.10.10.16, single NVIDIA H100 NVL (95.8GB VRAM). At time of Phase 2,
  ~89.6GB already in use by two unrelated production VLLM inference processes --
  only ~6.2GB free. GPU jobs were run in staggered batches of <=3 concurrent
  (not all-at-once) specifically to avoid risking OOM on a shared production GPU.
- Local dev machine: 10-core Mac, no CUDA, MPS available (used only for smoke
  tests / bug reproduction, never for the actual reported numbers).

Parallelization:
- Phase 1: 5 modes launched as independent background processes.
- Phase 2: batched in groups of <=3 concurrent GPU jobs (location sanity+probe;
  3 pairwise ablations; gradient analysis + 2 extent probes), `wait` between
  batches, per the shared-GPU memory caution above.
```

| Experiment | Device | Runtime |
|---|---|---|
| Phase 1: shape_only (seed 0) | CPU | 12.3s |
| Phase 1: location_only (seed 0) | CPU | 7.6s |
| Phase 1: extent_only (seed 0) | CPU | 9.3s |
| Phase 1: intensity_only (seed 0) | CPU | 8.7s |
| Phase 1: multitask (seed 0) | CPU | 10.0s |
| Phase 2: location sanity + probe | CPU (script has no --device) | not recorded |
| Phase 2: extent+shape/location/intensity | CUDA | not recorded (server-side) |
| Phase 2: gradient analysis (20 epochs + 45 sampled batches) | CUDA | not recorded |
| Phase 2: extent probe x2 checkpoints | CPU (script has no --device) | not recorded |

Runtimes for the Phase 2 GPU batch were not captured in what was shared back --
can be pulled from the `phase2_*.log` files' timestamps if needed for the
final report.

Experiments skipped for efficiency: z_dim/head/head-LR ablations (Phases 3-5)
-- deferred per the adaptive phase-gating principle, pending resolution of
Section 12's open finding; not run, not assumed.
