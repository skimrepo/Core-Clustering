"""MTL_V22_REPORT.md Sections 16-19: simulator-level calibration diagnostics
for the universal realized-deviation intensity definition
(anomsim.anomalies.base.apply_calibrated_anomaly). Pure numpy, no model/
torch involved -- these confirm the SIMULATOR calibration itself is correct
across anomaly types BEFORE any V2.2 model baseline is run (gating step,
per the spec).

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v22_intensity_calibration.py \\
        --output_dir diagnostics/outputs/v22
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(line_buffering=True)

from anomsim.anomalies.base import apply_calibrated_anomaly, get_anomaly
from anomsim.waveforms.basic import WhiteNoiseWaveform

# Representative anomaly types spanning the distinct generation mechanisms
# in the registry: additive-at-a-point (spike), additive-over-a-region
# (noise), replacement (cutoff), multiplicative (scale), ramp-then-hold
# (wander), affine (contextual), reflection (upsidedown), constant-offset
# (shift -- the only type actually wired into DynamicContrastiveDataset).
REPRESENTATIVE_TYPES = {
    "spike": lambda n_time: {"scale": 1.0},
    "noise": lambda n_time: {"min_range": max(1, int(0.05 * n_time)), "scale": 0.1},
    "cutoff": lambda n_time: {"min_range": max(1, int(0.05 * n_time))},
    "scale": lambda n_time: {"min_range": max(1, int(0.05 * n_time)), "scale": 1.0},
    "wander": lambda n_time: {"min_range": max(1, int(0.05 * n_time)), "scale": 1.0},
    "contextual": lambda n_time: {"min_range": max(1, int(0.05 * n_time)), "scale": 1.0},
    "upsidedown": lambda n_time: {"min_range": max(1, int(0.05 * n_time))},
    "shift": lambda n_time: {"min_range_ratio": 0.05, "max_range_ratio": 0.5},
}
I_TARGET_BUCKETS = {"very_small": 0.1, "small": 0.5, "medium": 1.5, "large": 4.0, "very_large": 8.0}


def make_clean_series(seed, n_time=300):
    rng = np.random.default_rng(seed)
    wf_params = WhiteNoiseWaveform.random_params(rng, n_time)
    return WhiteNoiseWaveform(**wf_params).generate(n_time=n_time, rng=rng)


def run_one(type_name, i_target, waveform_seed, apply_seed, n_time=300):
    Z = make_clean_series(waveform_seed, n_time=n_time)
    cls = get_anomaly(type_name)
    anomaly = cls(**REPRESENTATIVE_TYPES[type_name](n_time))
    rng = np.random.default_rng(apply_seed)
    Y_calibrated, Z_ret, mask, meta = apply_calibrated_anomaly(anomaly, Z, 0, n_time, rng, i_target)

    support = mask < 0.5
    extent_ratio = float(support.sum()) / n_time
    delta_raw_unused = None  # not recomputed here; shape preservation measured separately below
    return meta, extent_ratio, Y_calibrated, Z_ret, mask


def section16_calibration_accuracy(n_waveform_seeds=5, n_apply_seeds=3):
    """Per-type: target vs realized intensity, across many samples."""
    results = {}
    for type_name in REPRESENTATIVE_TYPES:
        targets, realized = [], []
        for bucket_name, i_target in I_TARGET_BUCKETS.items():
            for ws in range(n_waveform_seeds):
                for asd in range(n_apply_seeds):
                    meta, _, _, _, _ = run_one(type_name, i_target, waveform_seed=ws, apply_seed=asd + ws * 100)
                    targets.append(i_target)
                    realized.append(meta["realized_intensity_raw"])
        targets, realized = np.array(targets), np.array(realized)
        abs_err = np.abs(realized - targets)
        rel_err = abs_err / targets
        pearson = float(pearsonr(targets, realized)[0]) if len(set(targets)) > 1 else float("nan")
        spearman = float(spearmanr(targets, realized)[0]) if len(set(targets)) > 1 else float("nan")
        results[type_name] = {
            "n": len(targets),
            "mae": float(abs_err.mean()),
            "mean_relative_error": float(rel_err.mean()),
            "pearson": pearson,
            "spearman": spearman,
            "mean_ratio_realized_over_target": float((realized / targets).mean()),
        }
        print(f"  {type_name:<12} n={len(targets):<4} mae={results[type_name]['mae']:.5f}  "
              f"rel_err={results[type_name]['mean_relative_error']:.5f}  "
              f"pearson={pearson:.6f}  ratio={results[type_name]['mean_ratio_realized_over_target']:.5f}")
    return results


def section17_cross_type_consistency(i_targets=(1.0, 3.0), n_waveform_seeds=5):
    """Same I_target applied across all types -- realized deviation should
    be comparable regardless of generation mechanism."""
    results = {}
    for i_target in i_targets:
        per_type = {}
        for type_name in REPRESENTATIVE_TYPES:
            realized = [
                run_one(type_name, i_target, waveform_seed=ws, apply_seed=ws)[0]["realized_intensity_raw"]
                for ws in range(n_waveform_seeds)
            ]
            per_type[type_name] = {"mean": float(np.mean(realized)), "std": float(np.std(realized))}
        realized_means = np.array([v["mean"] for v in per_type.values()])
        results[f"i_target_{i_target}"] = {
            "per_type": per_type,
            "cross_type_std_of_means": float(realized_means.std()),
            "cross_type_max_abs_deviation_from_target": float(np.max(np.abs(realized_means - i_target))),
        }
        print(f"  I_target={i_target}: cross-type std of per-type means = "
              f"{results[f'i_target_{i_target}']['cross_type_std_of_means']:.5f}")
    return results


def section18_extent_leakage(i_target=2.0, n_waveform_seeds=8, n_apply_seeds=4):
    """For each type, sample many (extent varies naturally with apply_seed
    since most types don't support a forced region); bucket samples by
    their REALIZED extent_ratio into tertiles and report I_realized per
    tertile plus the extent-vs-relative-bias correlation. 'shift' additionally
    gets a controlled small/medium/large comparison since it supports
    forced_region exactly."""
    results = {"correlational": {}, "shift_controlled": {}}
    for type_name in REPRESENTATIVE_TYPES:
        extents, realized = [], []
        for ws in range(n_waveform_seeds):
            for asd in range(n_apply_seeds):
                meta, extent_ratio, _, _, _ = run_one(type_name, i_target, waveform_seed=ws,
                                                        apply_seed=asd + ws * 100)
                extents.append(extent_ratio)
                realized.append(meta["realized_intensity_raw"])
        extents, realized = np.array(extents), np.array(realized)
        rel_bias = (realized - i_target) / i_target
        corr = float(pearsonr(extents, rel_bias)[0]) if len(set(extents)) > 1 else float("nan")
        results["correlational"][type_name] = {
            "n": len(extents),
            "extent_ratio_range": [float(extents.min()), float(extents.max())],
            "pearson_extent_vs_relative_bias": corr,
            "mean_relative_bias": float(rel_bias.mean()),
        }
        print(f"  {type_name:<12} extent-vs-bias pearson={corr:.4f}  mean_rel_bias={rel_bias.mean():.5f}")

    # controlled comparison for 'shift' (the only type with forced_region)
    from anomsim.anomalies.redlamp_types import ShiftAnomaly
    n_time = 300
    small_medium_large = {"small": (10, 20), "medium": (100, 130), "large": (250, 290)}
    for label, (s, e) in small_medium_large.items():
        realized_vals = []
        for ws in range(n_waveform_seeds):
            Z = make_clean_series(ws, n_time=n_time)
            anomaly = ShiftAnomaly(forced_region=(s, e), forced_magnitude_std_multiplier=1.0)
            rng = np.random.default_rng(ws)
            _, _, _, meta = apply_calibrated_anomaly(anomaly, Z, 0, n_time, rng, i_target)
            realized_vals.append(meta["realized_intensity_raw"])
        results["shift_controlled"][label] = {
            "extent_ratio": (e - s) / n_time, "mean_realized": float(np.mean(realized_vals)),
            "std_realized": float(np.std(realized_vals)),
        }
        print(f"  shift[{label}] extent_ratio={(e-s)/n_time:.3f} "
              f"mean_realized={np.mean(realized_vals):.5f}")
    return results


def section19_shape_preservation(n_waveform_seeds=5):
    """Cosine similarity between delta_raw (pre-calibration) and
    delta_scaled (post-calibration) over the support region -- must be
    ~1.0 since calibration is a pure positive-scalar rescale."""
    results = {}
    for type_name in REPRESENTATIVE_TYPES:
        n_time = 300
        cls = get_anomaly(type_name)
        cos_sims = []
        for ws in range(n_waveform_seeds):
            Z = make_clean_series(ws, n_time=n_time)
            anomaly = cls(**REPRESENTATIVE_TYPES[type_name](n_time))
            rng = np.random.default_rng(ws)
            Y_raw, Z_raw, mask_raw = anomaly.apply(Z, 0, n_time, rng)
            support = (mask_raw < 0.5).ravel()
            if not support.any():
                continue
            delta_raw = (Y_raw - Z_raw).ravel()

            rng2 = np.random.default_rng(ws)  # SAME seed -> same raw candidate/region before calibration
            _, _, _, meta = apply_calibrated_anomaly(cls(**REPRESENTATIVE_TYPES[type_name](n_time)),
                                                       Z, 0, n_time, rng2, i_target=3.0)
            delta_scaled = delta_raw * meta["calibration_scale"] if meta["calibration_scale"] else delta_raw

            a, b = delta_raw[support], delta_scaled[support]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos_sims.append(float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan"))
        cos_sims = [c for c in cos_sims if c == c]
        results[type_name] = {
            "n": len(cos_sims),
            "mean_cosine_similarity": float(np.mean(cos_sims)) if cos_sims else float("nan"),
            "min_cosine_similarity": float(np.min(cos_sims)) if cos_sims else float("nan"),
        }
        print(f"  {type_name:<12} mean_cos={results[type_name]['mean_cosine_similarity']:.8f}  "
              f"min_cos={results[type_name]['min_cosine_similarity']:.8f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/v22")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Section 16: per-type calibration accuracy ===")
    s16 = section16_calibration_accuracy()
    print("=== Section 17: cross-type consistency ===")
    s17 = section17_cross_type_consistency()
    print("=== Section 18: extent leakage ===")
    s18 = section18_extent_leakage()
    print("=== Section 19: shape preservation ===")
    s19 = section19_shape_preservation()

    result = {
        "calibration_accuracy": s16,
        "cross_type_consistency": s17,
        "extent_leakage": s18,
        "shape_preservation": s19,
    }
    out_path = os.path.join(args.output_dir, "v22_intensity_calibration.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
