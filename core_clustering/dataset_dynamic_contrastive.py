import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

try:
    from anomsim.anomalies.redlamp_types import ShiftAnomaly
    from anomsim.waveforms.basic import WhiteNoiseWaveform
except ImportError:
    # Reuse AnomSim's actual injection/waveform classes directly (not a
    # vendored copy) so the two repos' logic can never silently drift apart
    # -- same convention as online_dataset.py.
    _sibling_anomsim = os.path.join(os.path.dirname(__file__), "..", "..", "AnomSim")
    if os.path.isdir(_sibling_anomsim):
        import sys

        sys.path.insert(0, _sibling_anomsim)
    try:
        from anomsim.anomalies.redlamp_types import ShiftAnomaly
        from anomsim.waveforms.basic import WhiteNoiseWaveform
    except ImportError as e:
        raise ImportError(
            "core_clustering.dataset_dynamic_contrastive requires the AnomSim package to be "
            "importable. Clone AnomSim as a sibling directory next to Core-Clustering, or add "
            "its repo root to PYTHONPATH."
        ) from e

from anomsim.anomalies.base import apply_calibrated_anomaly, sample_log_uniform

from core_clustering.dataset_contrastive import NORMAL_SENTINEL, SHAPE_LABELS
from core_clustering.target_transforms import ScalarMetricTargetTransform

INTENSITY_MODE_LEGACY = "legacy_native_intensity"
INTENSITY_MODE_UNIVERSAL = "universal_deviation_intensity"
_VALID_INTENSITY_MODES = (INTENSITY_MODE_LEGACY, INTENSITY_MODE_UNIVERSAL)

SEED_BLOCK = 10_000_000
EVAL_SEED_OFFSET = 500_000
WORKER_SEED_OFFSET = 1_000


@dataclass
class EntitySpec:
    """A fixed background series identity -- split/role/seed never change
    across epochs. Everything else (n_time, waveform scale, and for
    anomalous entities the injected shift's location/extent/intensity) is
    regenerated deterministically FROM this seed at __getitem__ time, not
    stored here -- so the manifest itself stays tiny regardless of dataset
    size."""

    entity_id: int
    split: str
    seed: int
    is_anomalous: bool


def generate_entity_manifest(
    n_instances: int = 400,
    anomaly_ratio: float = 0.5,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    base_seed: int = 0,
) -> List[EntitySpec]:
    """Assigns each entity a fixed role (normal/anomalous) and split, same
    balancing convention as anomsim_v3_dataset.generate_anomsim_v3_dataset
    (each role split independently so every split's class ratio matches) --
    minus the bucket-stratified injection-parameter assignment, since
    injection parameters are no longer decided here at all: they're sampled
    fresh (train) or cached once (val/test) inside DynamicContrastiveDataset."""
    if not (0.0 <= anomaly_ratio <= 1.0):
        raise ValueError("anomaly_ratio must be in [0, 1]")
    if abs(sum(split_ratios) - 1.0) > 1e-9:
        raise ValueError("split_ratios must sum to 1.0")

    n_anomalous = round(anomaly_ratio * n_instances)
    is_anomalous_flags = [True] * n_anomalous + [False] * (n_instances - n_anomalous)
    assign_rng = np.random.default_rng(base_seed + SEED_BLOCK)
    assign_rng.shuffle(is_anomalous_flags)

    anom_indices = [i for i, f in enumerate(is_anomalous_flags) if f]
    normal_indices = [i for i, f in enumerate(is_anomalous_flags) if not f]
    split_rng = np.random.default_rng(base_seed + 2 * SEED_BLOCK)
    split_by_index = {}
    for indices in (anom_indices, normal_indices):
        indices = list(indices)
        split_rng.shuffle(indices)
        n = len(indices)
        n_train = round(split_ratios[0] * n)
        n_val = round(split_ratios[1] * n)
        for j, idx in enumerate(indices):
            split_by_index[idx] = "train" if j < n_train else ("val" if j < n_train + n_val else "test")

    return [
        EntitySpec(entity_id=i, split=split_by_index[i], seed=base_seed + i, is_anomalous=is_anomalous_flags[i])
        for i in range(n_instances)
    ]


class DynamicContrastiveDataset(torch.utils.data.Dataset):
    """One item = one whole series, injected on the fly instead of loaded
    from disk. Background (clean, pre-injection) series is regenerated
    deterministically from the entity's own fixed seed every call -- never
    stored -- so reproducing a specific instance only requires its seed.

    train=True: injection parameters (location/extent/intensity, all
    continuous) are drawn fresh from a persistent RNG on every __getitem__
    call -- effectively unlimited augmentation diversity, matching RedLamp's
    own per-batch injection convention.
    train=False (val/test): injection parameters are drawn ONCE per entity
    at construction time and cached, so val/test loss is comparable across
    epochs (a moving target would make early stopping meaningless).
    """

    def __init__(
        self,
        entities: List[EntitySpec],
        split: str,
        train: bool,
        base_seed: int = 0,
        length_range: Tuple[int, int] = (500, 550),
        min_range_ratio: float = 0.05,
        max_range_ratio: float = 0.5,
        min_magnitude_std_multiplier: float = 0.2,
        max_magnitude_std_multiplier: float = 4.0,
        intensity_mode: str = INTENSITY_MODE_LEGACY,
        intensity_min: float = 0.05,
        intensity_max: float = 8.0,
        intensity_sampling: str = "log_uniform",
        intensity_metric_transform: str = None,
    ):
        if intensity_mode not in _VALID_INTENSITY_MODES:
            raise ValueError(f"intensity_mode must be one of {_VALID_INTENSITY_MODES}, got {intensity_mode!r}")
        if intensity_sampling != "log_uniform":
            raise ValueError(f"intensity_sampling only supports 'log_uniform', got {intensity_sampling!r}")

        self.entities = [e for e in entities if e.split == split]
        self.train = train
        self.base_seed = base_seed
        self.length_range = length_range
        self.min_range_ratio = min_range_ratio
        self.max_range_ratio = max_range_ratio
        self.min_magnitude_std_multiplier = min_magnitude_std_multiplier
        self.max_magnitude_std_multiplier = max_magnitude_std_multiplier
        # V2.2 (MTL_V22_REPORT.md): intensity_mode="universal_deviation_intensity"
        # replaces the type-specific native generator parameter (here,
        # ShiftAnomaly's magnitude_std_multiplier) as the model's training
        # target with a type-agnostic, post-hoc-measured realized deviation
        # (RMS of the actual injected perturbation over its support, divided
        # by the clean signal's own reference scale) -- see
        # anomsim.anomalies.base.apply_calibrated_anomaly. Default stays
        # INTENSITY_MODE_LEGACY so V1/V2/V2.1 reproducibility is untouched.
        self.intensity_mode = intensity_mode
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.intensity_sampling = intensity_sampling
        # V2.3 (MTL_V23_ORDINAL_INTENSITY_REPORT.md): intensity_metric_transform
        # lets a caller decouple "which deviation definition" (intensity_mode)
        # from "whether a bounded metric transform is applied to it" -- e.g.
        # universal_deviation_intensity + explicit "identity" gives an
        # unbounded raw I_raw target for RadialOrdinalLoss to consume, unlike
        # V2.2/V2.2a's implicit positive_unbounded_to_unit. None (default)
        # preserves the original auto-derived behavior exactly.
        transform_mode = intensity_metric_transform
        if transform_mode is None:
            transform_mode = "positive_unbounded_to_unit" if intensity_mode == INTENSITY_MODE_UNIVERSAL else "identity"
        self._intensity_transform = ScalarMetricTargetTransform(
            mode=transform_mode
        )

        self._rng = np.random.default_rng(base_seed)
        self._eval_cache = {}
        if not train:
            self._build_eval_cache()

    def __len__(self):
        return len(self.entities)

    def _generate_background(self, entity: EntitySpec):
        rng_bg = np.random.default_rng(entity.seed)
        n_time = int(rng_bg.integers(self.length_range[0], self.length_range[1] + 1))
        wf_params = WhiteNoiseWaveform.random_params(rng_bg, n_time)
        Z = WhiteNoiseWaveform(**wf_params).generate(n_time=n_time, rng=rng_bg)
        return Z, n_time

    def _inject(self, Z: np.ndarray, n_time: int, rng: np.random.Generator):
        location_ratio = float(rng.uniform(0.0, 1.0))
        extent_ratio = float(rng.uniform(self.min_range_ratio, self.max_range_ratio))

        length = max(1, int(round(extent_ratio * n_time)))
        length = min(length, n_time)
        max_start = n_time - length
        # location_ratio indexes the FEASIBLE start range directly (not the
        # full series then clamped) -- always geometrically consistent
        # regardless of extent, unlike the old bucket-edge approach where a
        # long extent could force a start outside its assigned bucket.
        start = int(round(location_ratio * max_start))

        if self.intensity_mode == INTENSITY_MODE_UNIVERSAL:
            i_target = sample_log_uniform(rng, self.intensity_min, self.intensity_max)
            # forced_magnitude_std_multiplier=1.0 here is an arbitrary,
            # shape-only placeholder -- ShiftAnomaly's perturbation is a
            # constant offset (no temporal shape variation to preserve
            # either way), and apply_calibrated_anomaly's rescaling
            # determines the ACTUAL realized magnitude, not this value.
            anomaly = ShiftAnomaly(forced_region=(start, start + length), forced_magnitude_std_multiplier=1.0)
            Y_injected, _, _, meta = apply_calibrated_anomaly(anomaly, Z, 0, n_time, rng, i_target)
            intensity_raw = meta["realized_intensity_raw"]
            sigma_ref = meta["sigma_ref"]
        else:
            intensity_raw = float(10 ** rng.uniform(
                np.log10(self.min_magnitude_std_multiplier), np.log10(self.max_magnitude_std_multiplier)
            ))
            anomaly = ShiftAnomaly(forced_region=(start, start + length),
                                    forced_magnitude_std_multiplier=intensity_raw)
            Y_injected, _, _ = anomaly.apply(Z, 0, n_time, rng)
            sigma_ref = float(Z[0].std())  # same clean-baseline scale ShiftAnomaly itself uses internally

        intensity_metric = self._intensity_transform.forward(intensity_raw)
        return Y_injected, location_ratio, extent_ratio, intensity_metric, intensity_raw, sigma_ref

    def _build_eval_cache(self):
        for entity in self.entities:
            if not entity.is_anomalous:
                continue
            Z, n_time = self._generate_background(entity)
            rng_inj = np.random.default_rng(self.base_seed + EVAL_SEED_OFFSET + entity.entity_id)
            self._eval_cache[entity.entity_id] = self._inject(Z, n_time, rng_inj)

    def __getitem__(self, idx):
        entity = self.entities[idx]
        Z, n_time = self._generate_background(entity)

        if not entity.is_anomalous:
            Y = Z
            loc = ext = inten = inten_raw = NORMAL_SENTINEL
            sigma_ref = float(Z[0].std())
        elif self.train:
            Y, loc, ext, inten, inten_raw, sigma_ref = self._inject(Z, n_time, self._rng)
        else:
            Y, loc, ext, inten, inten_raw, sigma_ref = self._eval_cache[entity.entity_id]

        clean_mean, clean_std = Z.mean(), Z.std()
        Y_norm = (Y - clean_mean) / (clean_std + 1e-8)
        return {
            "Y": torch.from_numpy(Y_norm).float(),
            "shape_label": SHAPE_LABELS["shift"] if entity.is_anomalous else SHAPE_LABELS["normal"],
            "location_value": float(loc),
            "extent_value": float(ext),
            "intensity_value": float(inten),
            # Raw (pre-metric-transform) intensity -- always populated for
            # both modes (mirrors intensity_value in legacy mode, since
            # legacy semantics have no separate raw/metric split) so
            # diagnostics scripts can read it uniformly regardless of mode.
            "intensity_value_raw": float(inten_raw),
            # Clean-baseline reference scale (V3, MTL_V3_REPORT.md): lets a
            # caller derive D = RMS(delta) = intensity_value_raw * sigma_ref
            # (universal mode) without dividing by sigma_ref again -- purely
            # additive metadata, unused by any existing V1-V2.3 code path.
            "sigma_ref": sigma_ref,
            "n_time": n_time,
        }


def dynamic_worker_init_fn(worker_id: int) -> None:
    """Without this, DataLoader(num_workers>0) forks the dataset object
    into each worker process with an IDENTICAL RNG state, so every worker
    would draw the same 'random' injection parameters -- a well-known
    PyTorch DataLoader pitfall. Re-seeds each worker's copy uniquely."""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._rng = np.random.default_rng(dataset.base_seed + WORKER_SEED_OFFSET + worker_id)
