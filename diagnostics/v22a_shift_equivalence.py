"""MTL_V22A_REPORT.md Sections 2-4: gates V2.2a's model baseline on
confirming legacy ShiftAnomaly and universal-calibrated ShiftAnomaly
produce (near-)identical signals for the same (waveform, location, extent,
intensity k) -- both use the SAME clean-baseline reference scale (whole
clean series std) and the SAME direction draw (matched-seed RNGs), so they
should agree up to floating point. Also compares the two datasets' scalar
metadata distributions and reports the I_raw -> I_metric mapping. Pure
numpy, no torch/model -- run BEFORE any V2.2a training.

Usage:
    PYTHONPATH=.:../AnomSim python3 diagnostics/v22a_shift_equivalence.py \\
        --output_dir diagnostics/outputs/v22a
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.stdout.reconfigure(line_buffering=True)

from anomsim.anomalies.base import apply_calibrated_anomaly
from anomsim.anomalies.redlamp_types import ShiftAnomaly
from anomsim.waveforms.basic import WhiteNoiseWaveform

from core_clustering.dataset_dynamic_contrastive import DynamicContrastiveDataset, generate_entity_manifest
from core_clustering.target_transforms import ScalarMetricTargetTransform

DIRECTION_SEED = 987654321  # shared seed for BOTH legacy/universal direction draws -- makes them agree exactly


def make_clean_series(seed, n_time=300):
    rng = np.random.default_rng(seed)
    wf_params = WhiteNoiseWaveform.random_params(rng, n_time)
    return WhiteNoiseWaveform(**wf_params).generate(n_time=n_time, rng=rng)


def run_pair(waveform_seed, start_ratio, extent_ratio, k, n_time=300):
    Z = make_clean_series(waveform_seed, n_time)
    length = max(1, int(round(extent_ratio * n_time)))
    max_start = n_time - length
    start = int(round(start_ratio * max_start))
    end = start + length

    rng_legacy = np.random.default_rng(DIRECTION_SEED)
    legacy = ShiftAnomaly(forced_region=(start, end), forced_magnitude_std_multiplier=k)
    Y_legacy, Z_legacy, mask_legacy = legacy.apply(Z, 0, n_time, rng_legacy)

    rng_universal = np.random.default_rng(DIRECTION_SEED)
    universal = ShiftAnomaly(forced_region=(start, end), forced_magnitude_std_multiplier=1.0)
    Y_universal, Z_universal, mask_universal, meta = apply_calibrated_anomaly(
        universal, Z, 0, n_time, rng_universal, i_target=k
    )

    diff = Y_legacy - Y_universal
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.mean(diff ** 2))),
        "mask_equal": bool(np.array_equal(mask_legacy, mask_universal)),
        "region_equal": (start, end) == (start, end),  # both use the SAME forced_region by construction
        "native_k": k,
        "realized_intensity": meta["realized_intensity_raw"],
    }


def section2_signal_equivalence(n_waveform_seeds=10, start_ratios=(0.1, 0.5, 0.9),
                                 extent_ratios=(0.05, 0.2, 0.5), ks=(0.2, 0.5, 1.0, 2.0, 4.0)):
    rows = []
    for ws in range(n_waveform_seeds):
        for sr in start_ratios:
            for er in extent_ratios:
                for k in ks:
                    rows.append(run_pair(ws, sr, er, k))

    max_abs_diffs = np.array([r["max_abs_diff"] for r in rows])
    mean_abs_diffs = np.array([r["mean_abs_diff"] for r in rows])
    rms_diffs = np.array([r["rms_diff"] for r in rows])
    native_ks = np.array([r["native_k"] for r in rows])
    realized = np.array([r["realized_intensity"] for r in rows])
    all_masks_equal = all(r["mask_equal"] for r in rows)
    all_regions_equal = all(r["region_equal"] for r in rows)

    intensity_abs_err = np.abs(realized - native_ks)
    summary = {
        "n_samples": len(rows),
        "signal_diff": {
            "max_abs_diff_over_all_samples": float(max_abs_diffs.max()),
            "mean_of_mean_abs_diff": float(mean_abs_diffs.mean()),
            "mean_of_rms_diff": float(rms_diffs.mean()),
        },
        "mask_equal_for_all_samples": all_masks_equal,
        "region_equal_for_all_samples": all_regions_equal,
        "intensity_equivalence": {
            "mae_native_k_vs_realized": float(intensity_abs_err.mean()),
            "max_abs_err": float(intensity_abs_err.max()),
            "pearson": float(pearsonr(native_ks, realized)[0]),
            "spearman": float(spearmanr(native_ks, realized)[0]),
        },
    }
    print(f"n_samples={summary['n_samples']}")
    print(f"signal max_abs_diff (over all samples) = {summary['signal_diff']['max_abs_diff_over_all_samples']:.3e}")
    print(f"signal mean_abs_diff (avg per sample)   = {summary['signal_diff']['mean_of_mean_abs_diff']:.3e}")
    print(f"signal rms_diff (avg per sample)        = {summary['signal_diff']['mean_of_rms_diff']:.3e}")
    print(f"masks all equal: {all_masks_equal}   regions all equal: {all_regions_equal}")
    print(f"intensity MAE(native_k, realized) = {summary['intensity_equivalence']['mae_native_k_vs_realized']:.3e}"
          f"  pearson={summary['intensity_equivalence']['pearson']:.8f}")
    return summary


def _distribution_stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "min": float(values.min()), "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)), "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)), "p95": float(np.percentile(values, 95)),
        "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std()),
    }


def section3_dataset_distribution(n_instances=1000, seed=0, length_range=(500, 550)):
    entities = generate_entity_manifest(n_instances=n_instances, anomaly_ratio=0.5, base_seed=seed)

    ds_legacy = DynamicContrastiveDataset(
        entities, split="train", train=True, base_seed=seed, length_range=length_range,
        intensity_mode="legacy_native_intensity", min_magnitude_std_multiplier=0.2, max_magnitude_std_multiplier=4.0,
    )
    ds_universal = DynamicContrastiveDataset(
        entities, split="train", train=True, base_seed=seed, length_range=length_range,
        intensity_mode="universal_deviation_intensity", intensity_min=0.2, intensity_max=4.0,
    )

    def collect(ds):
        loc, ext, inten_raw = [], [], []
        n_anom, n_normal = 0, 0
        for i, e in enumerate(ds.entities):
            item = ds[i]
            if e.is_anomalous:
                n_anom += 1
                loc.append(item["location_value"])
                ext.append(item["extent_value"])
                inten_raw.append(item["intensity_value_raw"])
            else:
                n_normal += 1
        return {
            "location": _distribution_stats(loc), "extent": _distribution_stats(ext),
            "intensity_raw": _distribution_stats(inten_raw),
            "n_anomalous": n_anom, "n_normal": n_normal, "anomaly_ratio": n_anom / (n_anom + n_normal),
        }

    legacy_stats = collect(ds_legacy)
    universal_stats = collect(ds_universal)
    print("legacy   intensity_raw:", json.dumps(legacy_stats["intensity_raw"], indent=2))
    print("universal intensity_raw:", json.dumps(universal_stats["intensity_raw"], indent=2))
    return {"legacy": legacy_stats, "universal": universal_stats}


def section4_metric_target_distribution(universal_dist):
    transform = ScalarMetricTargetTransform(mode="positive_unbounded_to_unit")
    raw_stats = universal_dist["intensity_raw"]
    # reconstruct approximate metric-space distribution stats by transforming
    # the same percentile points (monotonic transform preserves percentile order)
    metric_stats = {k: transform.forward(v) for k, v in raw_stats.items() if k != "std"}
    example_values = [0.2, 0.5, 1.0, 2.0, 3.0, 4.0]
    mapping_table = [{"I_raw": v, "I_metric": transform.forward(v)} for v in example_values]
    print("raw->metric mapping:", json.dumps(mapping_table, indent=2))
    return {"metric_space_percentiles_from_raw": metric_stats, "example_mapping": mapping_table}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="diagnostics/outputs/v22a")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Section 2: legacy vs universal Shift signal equivalence ===")
    s2 = section2_signal_equivalence()
    print("\n=== Section 3: dataset distribution comparison ===")
    s3 = section3_dataset_distribution()
    print("\n=== Section 4: metric target distribution ===")
    s4 = section4_metric_target_distribution(s3["universal"])

    result = {"signal_equivalence": s2, "dataset_distribution": s3, "metric_target_distribution": s4}
    out_path = os.path.join(args.output_dir, "v22a_shift_equivalence.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
