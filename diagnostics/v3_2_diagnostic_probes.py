"""MTL_V3_2_REPORT.md diagnostics: checkpoint-only (no retraining, no
optimizer steps -- exactly the same discipline as v3_1_diagnostic_probes.py)
probes against a trained V3.2 checkpoint (detach_scale_attrs=("intensity",),
location_position_aware_pooling=True).

Reuses everything from v3_1_diagnostic_probes.py that is generic w.r.t. the
model (intensity loss decomposition/binning/contribution/embedding-probe,
location target audit, location gradient probe, reference-effect sweep,
shape/extent sanity) and re-implements ONLY the two pieces that depended on
V3.1's specific Location pooling submodule name (pool_attn): the stage
probes' Stage F hook, and a new attention/position-summary diagnostic that
only exists when pooling="position_aware".

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v3_2_diagnostic_probes.py \\
        --checkpoint diagnostics/outputs/v32/v32_multitask_seed0/bestmodel.pkl \\
        --output_dir diagnostics/outputs/v32_diag
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

from core_clustering.models_contrastive_v2 import build_position_channel
from core_clustering.prob_heads import laplace_nll

from diagnostics.v3_eval_diagnostics import load_model as _load_model_base
from diagnostics.v3_1_diagnostic_probes import (
    _fit_eval_probe,
    _pad_batch,
    build_dataset,
    intensity_binned_analysis,
    intensity_embedding_probe,
    intensity_sample_contribution,
    location_gradient_probe,
    location_target_audit,
    location_temporal_shift_test,
    reference_effect_on_location,
    shape_extent_sanity,
)


def intensity_loss_decomposition(model, dataset, device="cpu", n_samples=300, max_len=550):
    """V3.2 variant of v3_1_diagnostic_probes.intensity_loss_decomposition.

    Under detach_scale_attrs=("intensity",), the SAME shared
    scalar_adapters["intensity"].linear layer is called TWICE per forward
    (once with the embedding for mu, once with a DETACHED copy for scale --
    see models_contrastive_v3.py's forward). A single forward hook that
    just overwrites one captured tensor would silently keep only the
    SECOND (scale) call's raw output, making grad_raw_mu measure the dead
    scale-branch's raw_mean column instead of the real mu path -- this is
    exactly the bug that produced "mean_abs_grad_raw_mu = 0.0" in every bin
    on the first run of this diagnostic against a V3.2 checkpoint, caught
    and fixed here rather than silently reported. This version appends
    EVERY call to a list and grads each one separately."""
    items = [dataset[i] for i in range(min(n_samples, len(dataset)))]
    Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
    D = torch.tensor([it["D"] for it in items], dtype=torch.float32, device=device)
    shape_label = np.array([it["shape_label"] for it in items])
    is_anom = shape_label == 1

    captured = {"Hq": None, "embedding": None, "raw_calls": []}

    def _hook_single(name):
        def _fn(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            t.retain_grad()
            captured[name] = t
        return _fn

    def _hook_append_raw(module, inp, out):
        out.retain_grad()
        captured["raw_calls"].append(out)

    h_trunk = model.encoder.register_forward_hook(_hook_single("Hq"))
    h_emb = model.attribute_heads["intensity"].register_forward_hook(_hook_single("embedding"))
    h_raw = model.scalar_adapters["intensity"].linear.register_forward_hook(_hook_append_raw)
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

    is_detached_intensity = "intensity" in model.detach_scale_attrs
    if is_detached_intensity:
        assert len(captured["raw_calls"]) == 2, \
            f"expected exactly 2 calls to the intensity adapter's linear layer, got {len(captured['raw_calls'])}"
        raw_mu_call, raw_scale_call = captured["raw_calls"]
        grad_raw_mu_col = raw_mu_call.grad[:, 0]
        grad_raw_scale_col = raw_scale_call.grad[:, 1]
    else:
        (raw_call,) = captured["raw_calls"]
        grad_raw_mu_col = raw_call.grad[:, 0]
        grad_raw_scale_col = raw_call.grad[:, 1]

    emb_grad_norm = captured["embedding"].grad.norm(dim=-1)
    trunk_grad_norm = captured["Hq"].grad.flatten(1).norm(dim=-1)

    rows = []
    for i, it in enumerate(items):
        rows.append({
            "D": float(D[i]), "mu": float(mu[i]), "scale": float(scale[i]),
            "residual": float(residual[i]), "residual_term": float(residual_term[i]),
            "scale_term": float(scale_term[i]), "total_intensity_loss": float(per_sample_loss[i]),
            "grad_raw_mu": float(grad_raw_mu_col[i]), "grad_raw_scale": float(grad_raw_scale_col[i]),
            "grad_embedding_norm": float(emb_grad_norm[i]), "grad_trunk_norm": float(trunk_grad_norm[i]),
            "is_anom": bool(is_anom[i]),
        })
    return rows


def load_model(checkpoint_path, device="cpu", embedding_dim=32, max_len=550, attention_max_resolution=256,
               detach_scale_attrs=("intensity",), location_position_aware_pooling=True):
    return _load_model_base(checkpoint_path, device=device, embedding_dim=embedding_dim, max_len=max_len,
                             attention_max_resolution=attention_max_resolution,
                             detach_scale_attrs=detach_scale_attrs,
                             location_position_aware_pooling=location_position_aware_pooling)


# --------------------------------------------------------------------------
# Section G: Location stage probes, generalized Stage F hook
# --------------------------------------------------------------------------

def location_stage_probes(model, train_ds, val_ds, device="cpu", n_train=400, n_val=250, max_len=550):
    captured = {}

    def _hook(name):
        def _fn(module, inp, out):
            captured[name] = out[0] if isinstance(out, tuple) else out
        return _fn

    location_head = model.attribute_heads["location"]
    hooks = [
        model.encoder.stem[-1].register_forward_hook(_hook("stage_A_early_trunk")),
        model.encoder.attn_by_stage["1"].register_forward_hook(_hook("stage_B_middle_trunk")),
        location_head.proj.register_forward_hook(_hook("stage_E_head_1x1conv")),
    ]
    if location_head.pooling == "position_aware":
        # Stage F is the position-aware pool's OWN output (the projected
        # feature_summary+position_summary vector) -- hook the submodule
        # directly rather than assuming V3.1's pool_attn name.
        hooks.append(location_head.position_pool.register_forward_hook(_hook("stage_F_pooled")))
    else:
        hooks.append(location_head.pool_attn.register_forward_hook(_hook("stage_F_pooled")))

    def _run(dataset, n):
        raw_items = [dataset[i] for i in range(min(n, len(dataset)))]
        items = [it for it in raw_items if it["shape_label"] == 1]
        Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
        mask_stage_A = pad_mask
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


# --------------------------------------------------------------------------
# Section I: Location attention / position-summary diagnostic (new for V3.2)
# --------------------------------------------------------------------------

def location_attention_diagnostic(model, dataset, device="cpu", n_examples=12, max_len=550):
    """Only meaningful when pooling="position_aware". For each example:
    true anomaly onset (in the position-channel's own full-length frame),
    the attention weights' center-of-mass position, and predicted mu."""
    location_head = model.attribute_heads["location"]
    if location_head.pooling != "position_aware":
        return {"note": "not applicable -- pooling is not position_aware"}

    items = [dataset[i] for i in range(len(dataset)) if dataset[i]["shape_label"] == 1][:n_examples]
    Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
    with torch.no_grad():
        Hq, Hq_mask = model._trunk_forward(Y, pad_mask)
        zeros = torch.zeros_like(Hq)
        has_ref = torch.zeros(Hq.shape[0], device=device)
        H_fused, gate = model.context_fusion(Hq, zeros, zeros, 0.0, has_ref, query_mask=Hq_mask)
        h = location_head.proj(H_fused)
        h = h * Hq_mask
        h_t = h.transpose(1, 2)
        pad_mask_t = Hq_mask[:, 0, :]
        location_head.position_pool(h_t, pad_mask_t)
        a_t = location_head.position_pool.last_attention_weights  # (B, T')
        pos = build_position_channel(torch.zeros_like(pad_mask_t).unsqueeze(1), Hq_mask)[:, 0, :]
        out = model(Y, query_pad_mask=pad_mask)
    mu = out["location_mu"].cpu().numpy()

    rows = []
    for i, it in enumerate(items):
        com = float((a_t[i] * pos[i]).sum().item())  # attention center-of-mass, position-channel frame
        peak_idx = int(a_t[i].argmax().item())
        rows.append({
            "true_location_target": it["location_value"],
            "attention_center_of_mass_position": com,
            "attention_peak_position": float(pos[i, peak_idx].item()),
            "predicted_mu": float(mu[i]),
        })

    coms = np.array([r["attention_center_of_mass_position"] for r in rows])
    targets = np.array([r["true_location_target"] for r in rows])
    corr = float(pearsonr(coms, targets)[0]) if len(set(coms)) > 1 else float("nan")
    return {"examples": rows, "corr_attention_com_vs_location_target": corr}


# --------------------------------------------------------------------------
# Section J: current target vs physical-onset ceiling analysis (post-training)
# --------------------------------------------------------------------------

def location_target_ceiling_analysis(model, dataset, device="cpu", n_samples=300, max_len=550):
    """After training: does the model's own mu correlate better with the
    training target it was actually supervised on, or with the physical
    full-sequence onset fraction it was never directly told? Reuses
    location_target_audit's onset-fraction recomputation."""
    audit = location_target_audit(dataset, n_samples=n_samples)

    raw_items = [dataset[i] for i in range(min(n_samples, len(dataset)))]
    items = [it for it in raw_items if it["shape_label"] == 1]
    Y, pad_mask = _pad_batch(items, max_len=max_len, device=device)
    with torch.no_grad():
        out = model(Y, query_pad_mask=pad_mask)
    mu = out["location_mu"].cpu().numpy()

    onset_fracs = []
    for it in items:
        n_time = it["n_time"]
        length = max(1, int(round(it["extent_value"] * n_time)))
        length = min(length, n_time)
        max_start = n_time - length
        start = int(round(it["location_value"] * max_start)) if max_start > 0 else 0
        onset_fracs.append(start / n_time if n_time > 0 else 0.0)
    onset_fracs = np.array(onset_fracs)
    targets = np.array([it["location_value"] for it in items])

    return {
        "target_audit": {k: v for k, v in audit.items() if k != "example_table"},
        "n_samples": len(items),
        "mu_vs_current_target": {
            "pearson": float(pearsonr(mu, targets)[0]), "spearman": float(spearmanr(mu, targets)[0]),
        },
        "mu_vs_physical_onset_fraction": {
            "pearson": float(pearsonr(mu, onset_fracs)[0]), "spearman": float(spearmanr(mu, onset_fracs)[0]),
        },
    }


# --------------------------------------------------------------------------
# Section N: Intensity linearity test (new for V3.2)
# --------------------------------------------------------------------------

def intensity_linearity_test(rows):
    """rows: the list from intensity_loss_decomposition. Fits predicted_mu
    = a*D + b on anomalous samples and reports slope/intercept/R^2 alongside
    correlation, plus a coarse binned mean-D vs mean-mu table."""
    anom = [r for r in rows if r["is_anom"]]
    D = np.array([r["D"] for r in anom])
    mu = np.array([r["mu"] for r in anom])

    a, b = np.polyfit(D, mu, 1)
    pred = a * D + b
    ss_res = float(np.sum((mu - pred) ** 2))
    ss_tot = float(np.sum((mu - mu.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n_bins = 5
    edges = np.percentile(D, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-9
    binned = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (D > lo) & (D <= hi)
        if mask.any():
            binned.append({"mean_D": float(D[mask].mean()), "mean_mu": float(mu[mask].mean()), "n": int(mask.sum())})

    return {
        "slope_a": float(a), "intercept_b": float(b), "r_squared": r2,
        "pearson": float(pearsonr(D, mu)[0]), "spearman": float(spearmanr(D, mu)[0]),
        "binned_mean_D_vs_mean_mu": binned,
    }


# --------------------------------------------------------------------------
# Section O: Intensity uncertainty test (adapted from v3_eval_diagnostics)
# --------------------------------------------------------------------------

def intensity_uncertainty_test(rows):
    anom = [r for r in rows if r["is_anom"]]
    D = np.array([r["D"] for r in rows])
    mu = np.array([r["mu"] for r in rows])
    scale = np.array([r["scale"] for r in rows])
    err = np.abs(D - mu)
    nlls = [float(laplace_nll(torch.tensor([r["D"]]), torch.tensor([r["mu"]]), torch.tensor([r["scale"]])).item())
            for r in rows]

    corr = {"pearson": float("nan"), "spearman": float("nan")}
    if len(set(err)) > 2 and len(set(scale)) > 2:
        corr = {"pearson": float(pearsonr(err, scale)[0]), "spearman": float(spearmanr(err, scale)[0])}

    return {
        "mean_scale": float(scale.mean()), "std_scale": float(scale.std()),
        "error_vs_scale_corr": corr, "mean_laplace_nll": float(np.mean(nlls)),
        "n_anomalous": len(anom), "n_total": len(rows),
    }


# --------------------------------------------------------------------------
# Section M/N (Intensity): normal-vs-anomaly vs within-anomaly split
# --------------------------------------------------------------------------

def intensity_normal_vs_anomaly_split(rows):
    D = np.array([r["D"] for r in rows])
    mu = np.array([r["mu"] for r in rows])
    is_anom = np.array([r["is_anom"] for r in rows])
    all_corr = {"pearson": float(pearsonr(D, mu)[0]), "spearman": float(spearmanr(D, mu)[0])}
    anom_corr = {"pearson": float("nan"), "spearman": float("nan")}
    if is_anom.sum() > 5:
        anom_corr = {"pearson": float(pearsonr(D[is_anom], mu[is_anom])[0]),
                     "spearman": float(spearmanr(D[is_anom], mu[is_anom])[0])}
    return {"all_samples": all_corr, "anomalous_only": anom_corr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="diagnostics/outputs/v32_diag")
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

    print("\n=== Intensity: normal-vs-anomaly vs within-anomaly ===")
    result["intensity_normal_vs_anomaly_split"] = intensity_normal_vs_anomaly_split(rows)
    print(json.dumps(result["intensity_normal_vs_anomaly_split"], indent=2))

    print("\n=== Intensity: linearity test ===")
    result["intensity_linearity"] = intensity_linearity_test(rows)
    print(json.dumps(result["intensity_linearity"], indent=2))

    print("\n=== Intensity: uncertainty test ===")
    result["intensity_uncertainty"] = intensity_uncertainty_test(rows)
    print(json.dumps(result["intensity_uncertainty"], indent=2))

    print("\n=== Intensity: embedding probe ===")
    result["intensity_embedding_probe"] = intensity_embedding_probe(
        model, train_ds, val_ds, device=args.device, n_train=len(train_ds), n_val=len(val_ds))
    print(json.dumps(result["intensity_embedding_probe"], indent=2))

    print("\n=== Location: target audit + ceiling analysis ===")
    result["location_target_ceiling"] = location_target_ceiling_analysis(
        model, val_ds, device=args.device, n_samples=len(val_ds))
    print(json.dumps(result["location_target_ceiling"], indent=2))

    print("\n=== Location: stage probes ===")
    result["location_stage_probes"] = location_stage_probes(
        model, train_ds, val_ds, device=args.device, n_train=len(train_ds), n_val=len(val_ds))
    print(json.dumps(result["location_stage_probes"], indent=2))

    print("\n=== Location: temporal-shift test ===")
    result["location_temporal_shift"] = location_temporal_shift_test(model, device=args.device)
    print(json.dumps(result["location_temporal_shift"], indent=2))

    print("\n=== Location: attention/position-summary diagnostic ===")
    result["location_attention_diagnostic"] = location_attention_diagnostic(model, val_ds, device=args.device)
    print(json.dumps(result["location_attention_diagnostic"], indent=2))

    print("\n=== Location: gradient probe (single frozen-checkpoint backward, no optimizer step) ===")
    result["location_gradient_probe"] = location_gradient_probe(model, val_ds, device=args.device, n_samples=len(val_ds))
    print(json.dumps(result["location_gradient_probe"], indent=2))

    print("\n=== Location: reference-context effect ===")
    result["reference_effect_on_location"] = reference_effect_on_location(model, val_ds, device=args.device)
    print(json.dumps(result["reference_effect_on_location"], indent=2))

    print("\n=== Shape/Extent sanity ===")
    result["shape_extent_sanity"] = shape_extent_sanity(model, val_ds, device=args.device)
    print(json.dumps(result["shape_extent_sanity"], indent=2))

    out_path = os.path.join(args.output_dir, "v3_2_diagnostic_probes.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
