"""MTL_V3_1_DIAGNOSTIC_REPORT.md: checkpoint-only diagnostics (no retraining,
no optimizer steps) investigating two unresolved V3.1 questions:

A. Why does Intensity distinguish normal-vs-anomaly but not represent
   anomaly MAGNITUDE?
B. Why has Location remained essentially unlearned across V2.1/V3/V3.1?

Every probe here either (a) does pure arithmetic on already-generated
dataset fields (no model), or (b) does a forward pass plus AT MOST one
backward pass on a frozen, already-trained checkpoint to read out
intermediate activations/gradients -- never an optimizer.step(), never a
training loop. This satisfies the compute policy's "checkpoint-only small
diagnostics" carve-out; nothing here needed the GPU server.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v3_1_diagnostic_probes.py \\
        --checkpoint diagnostics/outputs/v31/v31_multitask_seed0/bestmodel.pkl \\
        --output_dir diagnostics/outputs/v31_diag
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(line_buffering=True)

from core_clustering.dataset_dynamic_contrastive import generate_entity_manifest
from core_clustering.dataset_episodic import EpisodicContrastiveDataset
from core_clustering.models_conv_bottleneck import ConvBottleneckConfig
from core_clustering.models_contrastive_v3 import ContrastiveEncoderV3
from core_clustering.prob_heads import laplace_nll

from diagnostics.metrics import shape_metrics


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------

def load_model(checkpoint_path, device="cpu", embedding_dim=32, max_len=550, attention_max_resolution=256):
    config = ConvBottleneckConfig(n_time_max=max_len, n_features=2, attention_max_resolution=attention_max_resolution)
    model = ContrastiveEncoderV3(config, embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def build_dataset(split, seed=0, n_instances=1000, length_range=(500, 550), intensity_min=0.2, intensity_max=4.0):
    entities = generate_entity_manifest(n_instances=n_instances, anomaly_ratio=0.5, base_seed=seed)
    return EpisodicContrastiveDataset(
        entities, split=split, train=False, base_seed=seed, length_range=length_range,
        intensity_mode="universal_deviation_intensity", intensity_min=intensity_min, intensity_max=intensity_max,
        intensity_metric_transform="identity",
    )


def _pad_batch(items, max_len=550, device="cpu"):
    B = len(items)
    Y = torch.zeros(B, 1, max_len)
    pad_mask = torch.zeros(B, 1, max_len)
    for i, item in enumerate(items):
        n = item["Y"].shape[-1]
        Y[i, :, :n] = item["Y"]
        pad_mask[i, :, :n] = 1.0
    return Y.to(device), pad_mask.to(device)


def _quantile_bins(d_vals, is_anom, n_bins=5):
    """Same convention as v3_baseline.py's evaluate_v3: normal (D=0) gets
    its own explicit bin; anomalous subset is split into n_bins quantile
    bins so the bin edges track the (unbounded, per-instance-scaled)
    distribution instead of a fixed absolute range."""
    bins = [{"bin": "normal (D=0)", "mask": ~is_anom}]
    anom_d = d_vals[is_anom]
    if len(anom_d) >= n_bins:
        edges = np.percentile(anom_d, np.linspace(0, 100, n_bins + 1))
        edges[0] -= 1e-9
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = is_anom & (d_vals > lo) & (d_vals <= hi)
            bins.append({"bin": f"({lo:.4g},{hi:.4g}]", "mask": mask})
    return bins


# --------------------------------------------------------------------------
# Section D/E/G: Intensity loss decomposition, binned analysis, contribution
# --------------------------------------------------------------------------

def intensity_loss_decomposition(model, dataset, device="cpu", n_samples=300, max_len=550):
    """Single forward + single backward pass on a frozen checkpoint (no
    optimizer step) to read out, per validation sample: D, mu, scale,
    residual, residual_term, scale_term, total loss, and the gradient of
    that per-sample loss w.r.t. the adapter's raw (pre-link) outputs, the
    Intensity embedding, and the shared trunk feature Hq."""
    items = [dataset[i] for i in range(min(n_samples, len(dataset)))]
    Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
    D = torch.tensor([it["D"] for it in items], dtype=torch.float32, device=device)
    shape_label = np.array([it["shape_label"] for it in items])
    is_anom = shape_label == 1

    captured = {}

    def _hook(name):
        def _fn(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t.retain_grad()
            captured[name] = t
        return _fn

    h_trunk = model.encoder.register_forward_hook(_hook("Hq"))
    h_emb = model.attribute_heads["intensity"].register_forward_hook(_hook("embedding"))
    h_raw = model.scalar_adapters["intensity"].linear.register_forward_hook(_hook("raw"))
    try:
        out = model(Y, query_pad_mask=pad_mask)
        mu, scale = out["intensity_mu"], out["intensity_scale"]
        residual = torch.abs(D - mu)
        residual_term = residual / scale
        scale_term = torch.log(2 * scale)
        per_sample_loss = residual_term + scale_term
        per_sample_loss.sum().backward()
    finally:
        h_trunk.remove()
        h_emb.remove()
        h_raw.remove()

    raw_grad = captured["raw"].grad  # (N, 2): [:,0] wrt raw_mean (pre-softplus mu), [:,1] wrt raw_scale
    emb_grad_norm = captured["embedding"].grad.norm(dim=-1)
    trunk_grad_norm = captured["Hq"].grad.flatten(1).norm(dim=-1)

    rows = []
    for i, it in enumerate(items):
        rows.append({
            "D": float(D[i]), "mu": float(mu[i]), "scale": float(scale[i]),
            "residual": float(residual[i]), "residual_term": float(residual_term[i]),
            "scale_term": float(scale_term[i]), "total_intensity_loss": float(per_sample_loss[i]),
            "grad_raw_mu": float(raw_grad[i, 0]), "grad_raw_scale": float(raw_grad[i, 1]),
            "grad_embedding_norm": float(emb_grad_norm[i]), "grad_trunk_norm": float(trunk_grad_norm[i]),
            "is_anom": bool(is_anom[i]),
        })
    return rows


def intensity_binned_analysis(rows):
    D = np.array([r["D"] for r in rows])
    is_anom = np.array([r["is_anom"] for r in rows])
    bins = _quantile_bins(D, is_anom)
    result = []
    for b in bins:
        mask = b["mask"]
        n = int(mask.sum())
        entry = {"bin": b["bin"], "count": n}
        if n > 0:
            for key in ("D", "mu", "scale", "residual", "residual_term", "scale_term"):
                vals = np.array([r[key] for r, m in zip(rows, mask) if m])
                entry[f"mean_{key}"] = float(vals.mean())
                entry[f"median_{key}"] = float(np.median(vals))
            for key in ("grad_raw_mu", "grad_raw_scale"):
                vals = np.abs(np.array([r[key] for r, m in zip(rows, mask) if m]))
                entry[f"mean_abs_{key}"] = float(vals.mean())
        result.append(entry)
    return result


def intensity_sample_contribution(rows):
    D = np.array([r["D"] for r in rows])
    is_anom = np.array([r["is_anom"] for r in rows])
    loss = np.array([r["total_intensity_loss"] for r in rows])
    trunk_grad = np.array([r["grad_trunk_norm"] for r in rows])

    anom_d = D[is_anom]
    buckets = {"normal": ~is_anom}
    if len(anom_d) >= 3:
        lo_edge, hi_edge = np.percentile(anom_d, [100 / 3, 200 / 3])
        buckets["low_anomaly"] = is_anom & (D <= lo_edge)
        buckets["medium_anomaly"] = is_anom & (D > lo_edge) & (D <= hi_edge)
        buckets["high_anomaly"] = is_anom & (D > hi_edge)

    total_loss = loss.sum()
    total_grad = trunk_grad.sum()
    n_total = len(rows)
    result = {}
    for name, mask in buckets.items():
        n = int(mask.sum())
        result[name] = {
            "n": n, "frac_samples": n / n_total,
            "frac_total_loss": float(loss[mask].sum() / total_loss) if total_loss > 0 else float("nan"),
            "frac_total_trunk_grad_norm": float(trunk_grad[mask].sum() / total_grad) if total_grad > 0 else float("nan"),
        }
    return result


# --------------------------------------------------------------------------
# Section F: Intensity embedding probe (frozen, post-hoc, linear only)
# --------------------------------------------------------------------------

def _collect_intensity_embeddings(model, dataset, device="cpu", n_samples=500, max_len=550):
    items = [dataset[i] for i in range(min(n_samples, len(dataset)))]
    Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
    with torch.no_grad():
        out = model(Y, query_pad_mask=pad_mask)
    emb = out["embeddings"]["intensity"].cpu().numpy()
    D = np.array([it["D"] for it in items])
    is_anom = np.array([it["shape_label"] for it in items]) == 1
    return emb, D, is_anom


def intensity_embedding_probe(model, train_ds, val_ds, device="cpu", n_train=500, n_val=300, max_len=550):
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import NearestNeighbors

    emb_tr, D_tr, anom_tr = _collect_intensity_embeddings(model, train_ds, device, n_train, max_len)
    emb_va, D_va, anom_va = _collect_intensity_embeddings(model, val_ds, device, n_val, max_len)

    def _eval_probe(feat_tr, target_tr, feat_va, target_va, inverse=lambda x: x):
        reg = LinearRegression().fit(feat_tr, target_tr)
        pred_va = inverse(reg.predict(feat_va))
        true_va = inverse(target_va)
        return {
            "pearson": float(pearsonr(pred_va, true_va)[0]),
            "spearman": float(spearmanr(pred_va, true_va)[0]),
            "mae": float(np.mean(np.abs(pred_va - true_va))),
            "rmse": float(np.sqrt(np.mean((pred_va - true_va) ** 2))),
        }, reg

    probe_D_all, reg_D_all = _eval_probe(emb_tr, D_tr, emb_va, D_va)
    if anom_tr.sum() >= 5 and anom_va.sum() >= 5:
        probe_D_anom, reg_D_anom = _eval_probe(emb_tr[anom_tr], D_tr[anom_tr], emb_va[anom_va], D_va[anom_va])
    else:
        probe_D_anom, reg_D_anom = {"note": "insufficient anomalous samples"}, None
    probe_log1p, reg_log = _eval_probe(emb_tr, np.log1p(D_tr), emb_va, np.log1p(D_va), inverse=np.expm1)

    # kNN local-neighbor diagnostic: for each val ANOMALOUS embedding, find
    # its 5 nearest TRAIN anomalous-embedding neighbors; correlate the
    # query's true D with its neighbors' mean D. High correlation = nearby
    # embeddings really do encode similar severity (local smoothness);
    # low/near-zero = no local D structure in the embedding geometry.
    knn_result = {"note": "insufficient anomalous samples"}
    if anom_tr.sum() >= 6 and anom_va.sum() >= 5:
        nn = NearestNeighbors(n_neighbors=min(5, int(anom_tr.sum()))).fit(emb_tr[anom_tr])
        _, idx = nn.kneighbors(emb_va[anom_va])
        neighbor_mean_D = D_tr[anom_tr][idx].mean(axis=1)
        query_D = D_va[anom_va]
        knn_result = {
            "pearson": float(pearsonr(neighbor_mean_D, query_D)[0]),
            "spearman": float(spearmanr(neighbor_mean_D, query_D)[0]),
            "n_queries": int(anom_va.sum()),
        }

    # Pairwise ordering: for random anomalous pairs (val), does the probe's
    # prediction preserve the true D ordering more often than chance?
    pairwise_result = {"note": "insufficient anomalous samples"}
    if anom_va.sum() >= 10:
        rng = np.random.default_rng(0)
        va_idx = np.where(anom_va)[0]
        n_pairs = min(2000, len(va_idx) * (len(va_idx) - 1) // 2)
        i_idx = rng.choice(va_idx, size=n_pairs)
        j_idx = rng.choice(va_idx, size=n_pairs)
        keep = i_idx != j_idx
        i_idx, j_idx = i_idx[keep], j_idx[keep]
        true_order = np.sign(D_va[i_idx] - D_va[j_idx])
        pred_va_D = reg_D_all.predict(emb_va)
        pred_order = np.sign(pred_va_D[i_idx] - pred_va_D[j_idx])
        valid = true_order != 0
        agreement = float(np.mean(pred_order[valid] == true_order[valid])) if valid.any() else float("nan")
        pairwise_result = {"n_pairs": int(valid.sum()), "agreement_rate": agreement,
                            "chance_level": 0.5}

    return {
        "linear_probe_embedding_to_D_all": probe_D_all,
        "linear_probe_embedding_to_D_anomalous_only": probe_D_anom,
        "linear_probe_embedding_to_log1p_D": probe_log1p,
        "knn_local_neighbor_diagnostic": knn_result,
        "pairwise_ordering_agreement": pairwise_result,
    }


# --------------------------------------------------------------------------
# Section I/J/K: Location target audit, stage probes, temporal-shift test
# --------------------------------------------------------------------------

def location_target_audit(dataset, n_samples=300):
    """Pure arithmetic on already-generated dataset fields -- no model
    forward needed. Recomputes the actual anomaly onset position implied by
    (location_value, extent_value, n_time) using the EXACT formula
    DynamicContrastiveDataset._inject uses, then compares it against the
    naive "onset / valid_length" interpretation the spec's own working
    hypothesis assumed."""
    rows = []
    for i in range(min(n_samples, len(dataset))):
        item = dataset[i]
        if item["shape_label"] != 1:
            continue
        n_time = item["n_time"]
        loc, ext = item["location_value"], item["extent_value"]
        length = max(1, int(round(ext * n_time)))
        length = min(length, n_time)
        max_start = n_time - length
        start = int(round(loc * max_start)) if max_start > 0 else 0
        onset_fraction_of_full_length = start / n_time if n_time > 0 else 0.0
        rows.append({
            "n_time": n_time, "location_value": loc, "extent_value": ext,
            "length": length, "max_start": max_start, "start": start,
            "onset_fraction_of_full_length": onset_fraction_of_full_length,
            "discrepancy_loc_vs_onset_fraction": loc - onset_fraction_of_full_length,
        })

    loc_arr = np.array([r["location_value"] for r in rows])
    onset_arr = np.array([r["onset_fraction_of_full_length"] for r in rows])
    ext_arr = np.array([r["extent_value"] for r in rows])
    disc = loc_arr - onset_arr

    return {
        "n_anomalous_samples_audited": len(rows),
        "location_value_min": float(loc_arr.min()), "location_value_max": float(loc_arr.max()),
        "location_value_mean": float(loc_arr.mean()), "location_value_std": float(loc_arr.std()),
        "onset_fraction_min": float(onset_arr.min()), "onset_fraction_max": float(onset_arr.max()),
        "corr_location_value_vs_onset_fraction_pearson": float(pearsonr(loc_arr, onset_arr)[0]),
        "corr_location_value_vs_onset_fraction_spearman": float(spearmanr(loc_arr, onset_arr)[0]),
        "mean_abs_discrepancy": float(np.mean(np.abs(disc))),
        "max_abs_discrepancy": float(np.max(np.abs(disc))),
        "corr_discrepancy_vs_extent_pearson": float(pearsonr(disc, ext_arr)[0]),
        "example_table": [
            {"requested_location_ratio": round(r["location_value"], 4),
             "extent_ratio": round(r["extent_value"], 4),
             "n_time": r["n_time"], "actual_start": r["start"], "length": r["length"],
             "onset_fraction_of_full_length": round(r["onset_fraction_of_full_length"], 4),
             "discrepancy": round(r["discrepancy_loc_vs_onset_fraction"], 4)}
            for r in rows[:12]
        ],
    }


def _fit_eval_probe(feat_tr, y_tr, feat_va, y_va):
    """Ridge, not plain OLS: several stages below are flattened
    (channels*time) sequence features whose dimensionality can exceed the
    probe's own training-sample count, where unregularized LinearRegression
    degenerates into an underdetermined interpolator (unstable, misleading
    held-out correlations). A small fixed ridge penalty keeps every stage's
    probe comparably well-posed without introducing per-stage tuning as a
    confound."""
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha=10.0).fit(feat_tr, y_tr)
    pred = reg.predict(feat_va)
    return {
        "pearson": float(pearsonr(pred, y_va)[0]),
        "spearman": float(spearmanr(pred, y_va)[0]),
        "mae": float(np.mean(np.abs(pred - y_va))),
        "rmse": float(np.sqrt(np.mean((pred - y_va) ** 2))),
        "feature_dim": feat_tr.shape[1],
    }


def location_stage_probes(model, train_ds, val_ds, device="cpu", n_train=400, n_val=250, max_len=550):
    """Stages A-G, captured via forward hooks on a K=0 forward pass (so
    model.encoder fires exactly once per call -- unambiguous to hook).
    Sequence-valued stages (A/B/C/D) are reduced via masked mean+max pool
    (a fixed, non-learned readout) before the linear probe, so we are
    probing the REPRESENTATION's content, not fitting a network capable of
    solving the task on its own."""
    captured = {}

    def _hook(name):
        def _fn(module, inp, out):
            captured[name] = out[0] if isinstance(out, tuple) else out
        return _fn

    hooks = [
        model.encoder.stem[-1].register_forward_hook(_hook("stage_A_early_trunk")),
        model.encoder.attn_by_stage["1"].register_forward_hook(_hook("stage_B_middle_trunk")),
        model.attribute_heads["location"].proj.register_forward_hook(_hook("stage_E_head_1x1conv")),
        model.attribute_heads["location"].pool_attn.register_forward_hook(_hook("stage_F_pooled")),
    ]

    def _run(dataset, n):
        raw_items = [dataset[i] for i in range(min(n, len(dataset)))]
        items = [it for it in raw_items if it["shape_label"] == 1]
        Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
        # Mirror ConvBottleneckEncoder.forward's own mask propagation for
        # block0 (stride 1, mask unchanged) and block1 (stride 2, one
        # max_pool1d hop) so stage A/B pooling excludes padded timesteps at
        # THEIR OWN resolution, not the final trunk's.
        mask_stage_A = pad_mask  # stem: stride=1, mask unchanged
        mask_after_block0 = F.max_pool1d(pad_mask, kernel_size=3, stride=2, padding=1)
        mask_stage_B = F.max_pool1d(mask_after_block0, kernel_size=3, stride=2, padding=1)
        with torch.no_grad():
            Hq, Hq_mask = model._trunk_forward(Y, pad_mask)
            zeros = torch.zeros_like(Hq)
            has_ref = torch.zeros(Hq.shape[0], device=device)
            H_fused, gate = model.context_fusion(Hq, zeros, zeros, 0.0, has_ref, query_mask=Hq_mask)
            out = model(Y, query_pad_mask=pad_mask)
        loc_val = np.array([it["location_value"] for it in items])

        def _seq_flatten(t, mask=None):
            # t: (B, C, T'). Every sample in a batch is padded to the SAME
            # max_len, so T' is identical across the batch regardless of
            # each sample's own valid length -- flattening (channel, time)
            # into one fixed-size vector per sample is dimensionally valid
            # AND preserves WHICH timestep each value came from (mean/max
            # pooling over time would erase exactly the positional
            # information a location probe needs). Padded positions are
            # zeroed first so they contribute a constant (uninformative,
            # not noisy) value.
            if mask is not None:
                t = t * mask
            return t.reshape(t.shape[0], -1).cpu().numpy()

        feats = {
            "stage_A_early_trunk": _seq_flatten(captured["stage_A_early_trunk"], mask_stage_A),
            "stage_B_middle_trunk": _seq_flatten(captured["stage_B_middle_trunk"], mask_stage_B),
            "stage_C_final_Hq": _seq_flatten(Hq, Hq_mask),
            "stage_D_H_fused": _seq_flatten(H_fused, Hq_mask),
            "stage_E_head_1x1conv": _seq_flatten(captured["stage_E_head_1x1conv"], Hq_mask),
            "stage_F_pooled_flat": captured["stage_F_pooled"].reshape(len(items), -1).cpu().numpy(),
            "stage_G_final_embedding": out["embeddings"]["location"].cpu().numpy(),
        }
        return feats, loc_val

    try:
        feats_tr, loc_tr = _run(train_ds, n_train)
        feats_va, loc_va = _run(val_ds, n_val)
    finally:
        for h in hooks:
            h.remove()

    result = {}
    for stage in feats_tr:
        result[stage] = _fit_eval_probe(feats_tr[stage], loc_tr, feats_va[stage], loc_va)
    return result


def location_temporal_shift_test(model, device="cpu", max_len=550, n_time=520, base_seed=777):
    """Same background waveform, same anomaly shape/extent/intensity,
    varying ONLY the anomaly's location -- does the frozen model's
    representation/prediction actually move when the injected onset moves?
    Uses AnomSim's ShiftAnomaly directly (same class the dataset uses) so
    the injection mechanics are identical to training."""
    from anomsim.anomalies.base import apply_calibrated_anomaly
    from anomsim.anomalies.redlamp_types import ShiftAnomaly
    from anomsim.waveforms.basic import WhiteNoiseWaveform

    rng_bg = np.random.default_rng(base_seed)
    wf_params = WhiteNoiseWaveform.random_params(rng_bg, n_time)
    Z = WhiteNoiseWaveform(**wf_params).generate(n_time=n_time, rng=rng_bg)
    extent_ratio, i_target = 0.15, 1.0
    length = max(1, int(round(extent_ratio * n_time)))
    max_start = n_time - length

    locations = [0.1, 0.3, 0.5, 0.7, 0.9]
    items = []
    for loc in locations:
        start = int(round(loc * max_start))
        rng_inj = np.random.default_rng(base_seed + 1)
        anomaly = ShiftAnomaly(forced_region=(start, start + length), forced_magnitude_std_multiplier=1.0)
        Y_injected, _, _, meta = apply_calibrated_anomaly(anomaly, Z, 0, n_time, rng_inj, i_target)
        clean_mean, clean_std = Z.mean(), Z.std()
        Y_norm = (Y_injected - clean_mean) / (clean_std + 1e-8)
        items.append({"Y": torch.from_numpy(Y_norm).float(), "location_value": loc, "start": start})

    Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
    captured = {}

    def _hook(name):
        def _fn(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t
        return _fn

    h1 = model.encoder.register_forward_hook(_hook("Hq_raw"))
    try:
        with torch.no_grad():
            out = model(Y, query_pad_mask=pad_mask)
    finally:
        h1.remove()

    emb = out["embeddings"]["location"].cpu().numpy()
    mu = out["location_mu"].cpu().numpy()
    scale = out["location_scale"].cpu().numpy()

    rows = []
    for i, loc in enumerate(locations):
        rows.append({"location": loc, "location_mu": float(mu[i]), "location_scale": float(scale[i])})

    pair_stats = []
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            d_loc = abs(locations[i] - locations[j])
            emb_dist = float(np.linalg.norm(emb[i] - emb[j]))
            cos_sim = float(np.dot(emb[i], emb[j]) / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[j]) + 1e-12))
            pair_stats.append({"delta_location": d_loc, "embedding_l2_distance": emb_dist, "cosine_similarity": cos_sim})

    d_loc_arr = np.array([p["delta_location"] for p in pair_stats])
    dist_arr = np.array([p["embedding_l2_distance"] for p in pair_stats])
    corr = float(pearsonr(d_loc_arr, dist_arr)[0]) if len(set(d_loc_arr)) > 1 else float("nan")

    return {
        "locations_tested": locations, "per_location_prediction": rows, "pairwise": pair_stats,
        "corr_delta_location_vs_embedding_distance": corr,
        "mu_vs_true_location_pearson": float(pearsonr([r["location_mu"] for r in rows], locations)[0]),
    }


def location_position_channel_ablation(model, dataset, device="cpu", n_samples=100, max_len=550):
    """Input-level diagnostic only: zero or reverse the SECOND input channel
    (the position channel build_position_channel constructs) before it
    would normally be built -- but build_position_channel is called INSIDE
    _trunk_forward from the raw 1-channel Y, so to override it we monkeypatch
    build_position_channel for the duration of this call only (never touches
    model weights or training code)."""
    # models_contrastive_v3._trunk_forward resolves the name
    # "build_position_channel" against ITS OWN module globals (bound once at
    # import time via `from ... import build_position_channel`) -- patching
    # models_contrastive_v2's attribute would silently do nothing, since v3
    # never looks the name up on that module again after import.
    import core_clustering.models_contrastive_v3 as v3mod

    raw_items = [dataset[i] for i in range(min(n_samples, len(dataset)))]
    items = [it for it in raw_items if it["shape_label"] == 1][:n_samples]
    Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
    loc_val = np.array([it["location_value"] for it in items])

    original_fn = v3mod.build_position_channel

    def _get_embeddings(pos_fn):
        v3mod.build_position_channel = pos_fn
        try:
            with torch.no_grad():
                out = model(Y, query_pad_mask=pad_mask)
        finally:
            v3mod.build_position_channel = original_fn
        return out["embeddings"]["location"].cpu().numpy(), out["location_mu"].cpu().numpy()

    emb_correct, mu_correct = _get_embeddings(original_fn)
    emb_zero, mu_zero = _get_embeddings(lambda x, pad_mask=None: torch.zeros_like(original_fn(x, pad_mask)))
    emb_rev, mu_rev = _get_embeddings(lambda x, pad_mask=None: 1.0 - original_fn(x, pad_mask))

    def _diff_stats(a, b):
        d = np.linalg.norm(a - b, axis=-1)
        return {"mean_embedding_l2_change": float(d.mean()), "std_embedding_l2_change": float(d.std())}

    return {
        "n_samples": len(items),
        "correct_vs_zeroed": _diff_stats(emb_correct, emb_zero),
        "correct_vs_reversed": _diff_stats(emb_correct, emb_rev),
        "mu_pearson_correct_vs_true_location": float(pearsonr(mu_correct, loc_val)[0]),
        "mu_pearson_zeroed_vs_true_location": float(pearsonr(mu_zero, loc_val)[0]),
        "mu_pearson_reversed_vs_true_location": float(pearsonr(mu_rev, loc_val)[0]),
    }


def location_gradient_probe(model, dataset, device="cpu", n_samples=200, max_len=550):
    """Single forward + single backward (no optimizer step) isolating
    L_location's own gradient (masked to anomalous samples, matching the
    trainer's own masking) into four parameter groups: scalar adapter,
    Location head, context fusion, shared trunk."""
    items = [dataset[i] for i in range(min(n_samples, len(dataset)))]
    Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
    loc = torch.tensor([it["location_value"] for it in items], dtype=torch.float32, device=device)
    is_anom = torch.tensor([it["shape_label"] == 1 for it in items], dtype=torch.bool, device=device)

    out = model(Y, query_pad_mask=pad_mask)
    l_loc = laplace_nll(loc[is_anom], out["location_mu"][is_anom], out["location_scale"][is_anom])

    groups = {
        "scalar_adapter": list(model.scalar_adapters["location"].parameters()),
        "location_head": list(model.attribute_heads["location"].parameters()),
        "context_fusion": list(model.context_fusion.parameters()),
        "shared_trunk": list(model.encoder.parameters()),
    }
    result = {}
    for name, params in groups.items():
        grads = torch.autograd.grad(l_loc, params, retain_graph=True, allow_unused=True)
        flat = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])
        result[name] = {"grad_norm": float(flat.norm().item()), "n_params": int(flat.numel())}
    result["l_location_value"] = float(l_loc.item())
    result["n_anomalous_in_batch"] = int(is_anom.sum().item())
    return result


# --------------------------------------------------------------------------
# Section M: reference-context effect on Location
# --------------------------------------------------------------------------

def reference_effect_on_location(model, dataset, device="cpu", n_queries=30, max_len=550,
                                  k_values=(0, 3, 10, 30, 100)):
    items = [dataset[i] for i in range(min(n_queries, len(dataset))) if dataset[i]["shape_label"] == 1]
    loc_vals = np.array([it["location_value"] for it in items])

    captured = {}

    def _hook(module, inp, out):
        captured["H_fused"], captured["gate"] = out

    h = model.context_fusion.register_forward_hook(_hook)
    result = {}
    try:
        for K in k_values:
            mus, gates, hfused_std = [], [], []
            for it in items:
                Y, pad_mask = _pad_batch([it], max_len=max_len, device=device)
                if K == 0:
                    ref_x = ref_pad_mask = ref_k_valid_mask = None
                else:
                    refs, _ = dataset.sample_alternate_references(0, K=K)
                    ref_x = torch.zeros(1, K, 1, max_len)
                    ref_pad_mask = torch.zeros(1, K, 1, max_len)
                    ref_k_valid_mask = torch.ones(1, K)
                    for k, (Y_ref, n_ref) in enumerate(refs):
                        ref_x[0, k, 0, :n_ref] = torch.from_numpy(Y_ref[0]).float()
                        ref_pad_mask[0, k, 0, :n_ref] = 1.0
                    ref_x, ref_pad_mask, ref_k_valid_mask = (
                        ref_x.to(device), ref_pad_mask.to(device), ref_k_valid_mask.to(device))
                with torch.no_grad():
                    out = model(Y, query_pad_mask=pad_mask, ref_x=ref_x, ref_pad_mask=ref_pad_mask,
                                ref_k_valid_mask=ref_k_valid_mask)
                mus.append(float(out["location_mu"][0]))
                gates.append(float(out["gate"][0]))
                hfused_std.append(float(captured["H_fused"].std().item()))
            mus = np.array(mus)
            result[f"K={K}"] = {
                "mean_gate": float(np.mean(gates)),
                "mu_pearson_vs_true_location": float(pearsonr(mus, loc_vals)[0]) if len(set(mus)) > 1 else float("nan"),
                "mean_H_fused_std": float(np.mean(hfused_std)),
            }
    finally:
        h.remove()
    return result


# --------------------------------------------------------------------------
# Section O: existing Shape/Extent/reference-gate sanity metrics (reuse only)
# --------------------------------------------------------------------------

def shape_extent_sanity(model, dataset, device="cpu", n_queries=150, max_len=550):
    items = [dataset[i] for i in range(min(n_queries, len(dataset)))]
    Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
    with torch.no_grad():
        out = model(Y, query_pad_mask=pad_mask)
    shape_emb = out["embeddings"]["shape"].cpu().numpy()
    shape_label = np.array([it["shape_label"] for it in items])
    is_anom = shape_label == 1
    ext_mu = out["extent_mu"].cpu().numpy()
    ext_val = np.array([it["extent_value"] for it in items])
    return {
        "shape": shape_metrics(shape_emb, shape_label),
        "extent_pearson": float(pearsonr(ext_mu[is_anom], ext_val[is_anom])[0]) if is_anom.sum() > 2 else float("nan"),
        "extent_spearman": float(spearmanr(ext_mu[is_anom], ext_val[is_anom])[0]) if is_anom.sum() > 2 else float("nan"),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="diagnostics/outputs/v31_diag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_instances", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, device=args.device)
    train_ds = build_dataset("train", seed=args.seed, n_instances=args.n_instances)
    val_ds = build_dataset("val", seed=args.seed, n_instances=args.n_instances)

    result = {}

    print("=== Intensity: loss decomposition + binned analysis + contribution ===")
    rows = intensity_loss_decomposition(model, val_ds, device=args.device, n_samples=len(val_ds))
    result["intensity_loss_decomposition_sample"] = rows[:20]
    result["intensity_binned"] = intensity_binned_analysis(rows)
    result["intensity_sample_contribution"] = intensity_sample_contribution(rows)
    print(json.dumps(result["intensity_binned"], indent=2))
    print(json.dumps(result["intensity_sample_contribution"], indent=2))

    print("\n=== Intensity: embedding probe ===")
    result["intensity_embedding_probe"] = intensity_embedding_probe(
        model, train_ds, val_ds, device=args.device, n_train=len(train_ds), n_val=len(val_ds))
    print(json.dumps(result["intensity_embedding_probe"], indent=2))

    print("\n=== Location: target audit ===")
    result["location_target_audit"] = location_target_audit(val_ds, n_samples=len(val_ds))
    print(json.dumps({k: v for k, v in result["location_target_audit"].items() if k != "example_table"}, indent=2))

    print("\n=== Location: stage probes ===")
    result["location_stage_probes"] = location_stage_probes(
        model, train_ds, val_ds, device=args.device, n_train=len(train_ds), n_val=len(val_ds))
    print(json.dumps(result["location_stage_probes"], indent=2))

    print("\n=== Location: temporal-shift test ===")
    result["location_temporal_shift"] = location_temporal_shift_test(model, device=args.device)
    print(json.dumps(result["location_temporal_shift"], indent=2))

    print("\n=== Location: position-channel ablation ===")
    result["location_position_channel_ablation"] = location_position_channel_ablation(
        model, val_ds, device=args.device, n_samples=len(val_ds))
    print(json.dumps(result["location_position_channel_ablation"], indent=2))

    print("\n=== Location: gradient probe (single frozen-checkpoint backward, no optimizer step) ===")
    result["location_gradient_probe"] = location_gradient_probe(model, val_ds, device=args.device, n_samples=len(val_ds))
    print(json.dumps(result["location_gradient_probe"], indent=2))

    print("\n=== Location: reference-context effect ===")
    result["reference_effect_on_location"] = reference_effect_on_location(model, val_ds, device=args.device)
    print(json.dumps(result["reference_effect_on_location"], indent=2))

    print("\n=== Shape/Extent sanity ===")
    result["shape_extent_sanity"] = shape_extent_sanity(model, val_ds, device=args.device)
    print(json.dumps(result["shape_extent_sanity"], indent=2))

    out_path = os.path.join(args.output_dir, "v3_1_diagnostic_probes.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
