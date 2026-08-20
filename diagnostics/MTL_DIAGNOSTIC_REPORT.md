# MTL Diagnostic Report

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
2. **Extent learns well alone (pearson=0.773) but degrades severely under
   multi-task (pearson=0.207)** -- a genuine, large single-task-to-multi-task gap.
   This IS consistent with real MTL interference (gradient conflict or shared-
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
single-task runs are from an untrained head and are not meaningful).

| Task | Metric | Single-task | Multi-task | Change |
|---|---|---|---|---|
| Shape | nn_accuracy | 0.827 | 0.833 | ~unchanged |
| Shape | pos/neg separation | 1.199 | 0.062 | large drop (scale compresses; classification unaffected) |
| Location | pearson (pair-dist vs gap) | -0.050 | 0.018 | no signal in either condition |
| Location | spearman | -0.049 | 0.006 | no signal in either condition |
| Extent | pearson (dist-to-centroid vs value) | 0.773 | 0.207 | **large drop** |
| Extent | spearman | 0.761 | 0.277 | **large drop** |
| Intensity | pearson | 0.897 | 0.909 | no degradation |
| Intensity | spearman | 0.884 | 0.906 | no degradation |

Full metrics (including the non-meaningful off-diagonal single-task numbers) are
preserved in `diagnostics/outputs/phase1/experiment_results.json` and per-experiment
`metrics.json` files -- nothing was deleted or summarized away.

**Answering the key question ("Single-task에서는 잘 되는데 Multi-task에서만
악화되는 task가 있는가?"): YES for extent (0.773 -> 0.207), clearly. Location is
bad in BOTH conditions (not an MTL-specific degradation -- a prior, more basic
problem). Shape and intensity show no meaningful MTL degradation on their primary
metrics.**

## 4. Representation Probe

NOT RUN. Planned for Phase 2, pending decision to proceed.

## 5. Latent Dimension Ablation

NOT RUN. Phase 3, conditional on Phase 2 findings per the adaptive-ablation principle.

## 6. Head Ablation

NOT RUN. Phase 4, conditional.

## 7. Head LR Ablation

NOT RUN. Phase 5, conditional.

## 8. Gradient Norms

NOT RUN. Planned for Phase 2 (sampled early/middle/late training, 10-20 batches
per segment, on the multi-task run specifically).

## 9. Gradient Cosine Similarity

NOT RUN. Same phase as above.

## 10. Loss Scale

NOT RUN. Will be recorded alongside the Phase 2 gradient analysis.

## 11. Embedding Diagnostics

NOT RUN as a dedicated step, but partially available as a byproduct of Section 3's
task metrics (e.g. shape's positive/negative pair distances, which are one of the
requested collapse diagnostics). Full `embedding_collapse_stats` (per-dim std,
mean norm, mean pairwise distance) not yet computed for Phase 1 checkpoints --
can be added cheaply from the saved `bestmodel.pkl` checkpoints without retraining
if useful.

## 12. Diagnosis

| Hypothesis | Verdict | Evidence (one line) |
|---|---|---|
| A. z_dim=4 bottleneck is the main problem | INCONCLUSIVE | No ablation run yet (Phase 3 not started) |
| B. Shared pooling is the main information-loss cause | INCONCLUSIVE | Plausible given location fails even single-task, but no representation probe yet to localize WHERE the info is lost (Phase 2) |
| C. Linear head capacity is insufficient | INCONCLUSIVE | No head ablation run yet (Phase 4 not started) |
| D. Gradient magnitude imbalance exists | INCONCLUSIVE | No gradient analysis run yet (Phase 2) |
| E. Gradient directional conflict exists | INCONCLUSIVE | No gradient analysis run yet (Phase 2), though extent's large single->multi drop is CONSISTENT with (not proof of) this |
| F. Trunk lacks capacity for all 4 attributes simultaneously | INCONCLUSIVE | Partially consistent with extent's degradation, but location fails even without any competition, so capacity-sharing isn't the whole story |
| G. Multi-task sharing provides positive transfer over single-task | NOT SUPPORTED (for extent, location) / INCONCLUSIVE (shape, intensity) | Extent got worse under MTL (0.773->0.207); location never worked either way; shape/intensity roughly flat -- no evidence of positive transfer for any task so far |

## 13. Recommended Next Architecture

Not yet -- per the diagnostic principle, no architecture change is recommended
until Phase 2 (representation probing + gradient analysis) identifies WHERE
location's information is lost (if it even exists in Stage2/3/Squeeze) and
WHETHER extent's degradation is gradient-conflict-driven or capacity-driven.
Recommending an architecture change now would be premature.

## 14. What NOT to Change Yet

- PCGrad / GradNorm / any gradient balancing: not introduced, per instructions.
  Extent's single-vs-multi gap is suggestive but not yet confirmed as a
  directional-conflict problem specifically (could also be capacity-sharing,
  which gradient balancing wouldn't fix).
- z_dim, head architecture, head LR: no ablation run yet, nothing to base a
  change on.
- Positional encoding in Stage1-3: location fails even in single-task, so the
  "does location-only already work" screening question from Section 11 has been
  directly answered -- **NO, it does not** -- which now makes positional encoding
  in the self-attention stages a legitimate candidate, but Phase 2's probe should
  localize the loss point (Stage2/3 vs Squeeze vs Pool) before changing anything,
  per the stated principle of not jumping to a fix.

## 15. Files Changed

- `diagnostics/simple_trainer.py` (new): single-optimizer trainer for fair
  single-task/multi-task comparison (see Section 1 rationale).
- `diagnostics/metrics.py` (new): location/normal-relative/shape/collapse metrics.
- `diagnostics/phase1_baselines.py` (new): Phase 1 experiment runner.
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

Screening: 1 seed (seed=0), all 5 modes. Confirmatory (3-seed) re-verification
NOT YET RUN -- per Section 17.2's rule, extent's single-vs-multi gap (0.773 vs
0.207) is large enough to warrant 3-seed confirmation before treating it as
settled; recommended before/alongside Phase 2.

## Resource Efficiency

```
Hardware:
- Server: 10.10.10.16 (GPU present per user; this Phase 1 run used --device cpu,
  GPU not yet exercised)
- Local dev machine: 10-core Mac, no CUDA, MPS available (not used for this run)

Parallelization:
- 5 modes launched as independent background processes (bash `&` + `wait`),
  fully independent (no shared state) -- matches Section 17.4's guidance.
```

| Experiment | Runtime (server, CPU) |
|---|---|
| shape_only (seed 0) | 12.3s |
| location_only (seed 0) | 7.6s |
| extent_only (seed 0) | 9.3s |
| intensity_only (seed 0) | 8.7s |
| multitask (seed 0) | 10.0s |

Experiments skipped for efficiency: representation probing, gradient analysis,
z_dim/head/head-LR ablations -- all deferred to later phases per the adaptive
phase-gating principle; not run yet, not assumed.
