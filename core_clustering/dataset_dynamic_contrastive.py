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

from core_clustering.dataset_contrastive import NORMAL_SENTINEL, SHAPE_LABELS

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
    ):
        self.entities = [e for e in entities if e.split == split]
        self.train = train
        self.base_seed = base_seed
        self.length_range = length_range
        self.min_range_ratio = min_range_ratio
        self.max_range_ratio = max_range_ratio
        self.min_magnitude_std_multiplier = min_magnitude_std_multiplier
        self.max_magnitude_std_multiplier = max_magnitude_std_multiplier

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
        intensity = float(10 ** rng.uniform(
            np.log10(self.min_magnitude_std_multiplier), np.log10(self.max_magnitude_std_multiplier)
        ))

        length = max(1, int(round(extent_ratio * n_time)))
        length = min(length, n_time)
        max_start = n_time - length
        # location_ratio indexes the FEASIBLE start range directly (not the
        # full series then clamped) -- always geometrically consistent
        # regardless of extent, unlike the old bucket-edge approach where a
        # long extent could force a start outside its assigned bucket.
        start = int(round(location_ratio * max_start))

        anomaly = ShiftAnomaly(forced_region=(start, start + length), forced_magnitude_std_multiplier=intensity)
        Y_injected, _, _ = anomaly.apply(Z, 0, n_time, rng)
        return Y_injected, location_ratio, extent_ratio, intensity

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
            loc = ext = inten = NORMAL_SENTINEL
        elif self.train:
            Y, loc, ext, inten = self._inject(Z, n_time, self._rng)
        else:
            Y, loc, ext, inten = self._eval_cache[entity.entity_id]

        clean_mean, clean_std = Z.mean(), Z.std()
        Y_norm = (Y - clean_mean) / (clean_std + 1e-8)
        return {
            "Y": torch.from_numpy(Y_norm).float(),
            "shape_label": SHAPE_LABELS["shift"] if entity.is_anomalous else SHAPE_LABELS["normal"],
            "location_value": float(loc),
            "extent_value": float(ext),
            "intensity_value": float(inten),
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
