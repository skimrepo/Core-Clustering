import numpy as np
import torch

from core_clustering.dataset_dynamic_contrastive import (
    INTENSITY_MODE_UNIVERSAL,
    DynamicContrastiveDataset,
    ShiftAnomaly,
    WhiteNoiseWaveform,
)

DEFAULT_K_REGIMES = (0, 3, 10, 30, 100)
# Distinct from DynamicContrastiveDataset's own SEED_BLOCK(1e7)/EVAL_SEED_OFFSET(5e5)/
# WORKER_SEED_OFFSET(1e3) -- reference-set randomness is drawn from a fully
# separate stream so it can never perturb the parent class's own query-
# generation RNG sequence (reproducibility of the base dataset is untouched).
EPISODE_SEED_BLOCK = 20_000_000
EPISODE_EVAL_SEED_OFFSET = 700_000
# Fixed (not swept), mild contamination setting: a short, weak Shift
# anomaly -- just enough to make a reference "mistaken", not a severe one.
DEFAULT_CONTAMINATION_EXTENT_RATIO = 0.15
DEFAULT_CONTAMINATION_INTENSITY = 1.0


class EpisodicContrastiveDataset(DynamicContrastiveDataset):
    """V3 (MTL_V3_REPORT.md): wraps DynamicContrastiveDataset's EXISTING
    query generation (background + Shift injection, unchanged, same anomaly
    type/range/split as V1-V2.3) and additionally attaches, per item:

    - D: the new, deliberately UNBOUNDED intensity target (RMS(delta over
      the anomaly support), NOT divided by sigma_ref -- see Section 10 of
      the spec). D=0 for normal queries. Derived from fields
      DynamicContrastiveDataset already exposes (intensity_value_raw *
      sigma_ref) when intensity_mode=universal_deviation_intensity; no
      change to how the underlying anomaly is generated.
    - K reference sequences (independently-generated normal backgrounds,
      occasionally mildly contaminated with the SAME existing ShiftAnomaly
      at one fixed, small setting), K drawn from a small regime list
      (default {0,3,10,30,100}) rather than swept.

    The paired clean signal is NEVER exposed as a model input anywhere in
    this class -- it is used only internally (by the parent class and by
    _generate_reference) to compute labels/targets."""

    def __init__(self, entities, split, train, base_seed=0, length_range=(500, 550),
                 k_regimes=DEFAULT_K_REGIMES, contamination_prob=0.05,
                 contamination_extent_ratio=DEFAULT_CONTAMINATION_EXTENT_RATIO,
                 contamination_intensity=DEFAULT_CONTAMINATION_INTENSITY,
                 include_alternate_references: bool = False, **kwargs):
        super().__init__(entities, split, train, base_seed=base_seed, length_range=length_range, **kwargs)
        self.k_regimes = tuple(k_regimes)
        self.contamination_prob = contamination_prob
        self.contamination_extent_ratio = contamination_extent_ratio
        self.contamination_intensity = contamination_intensity
        # When True, __getitem__ also attaches a SECOND, independently-drawn
        # reference set (same K) for the SAME query -- used by the
        # trainer's reference-consistency loss (Section 7). Kept per-item
        # (not per-batch-index-tracked) specifically so this dataset never
        # needs the DataLoader to thread original indices back to it.
        self.include_alternate_references = include_alternate_references

        self._episode_rng = np.random.default_rng(base_seed + EPISODE_SEED_BLOCK)
        self._episode_cache = {}
        if not train:
            self._build_episode_cache()

    def _sample_k(self, rng):
        return int(rng.choice(self.k_regimes))

    def _generate_reference(self, ref_seed: int, contaminate: bool, rng: np.random.Generator):
        rng_bg = np.random.default_rng(ref_seed)
        n_time = int(rng_bg.integers(self.length_range[0], self.length_range[1] + 1))
        wf_params = WhiteNoiseWaveform.random_params(rng_bg, n_time)
        Z = WhiteNoiseWaveform(**wf_params).generate(n_time=n_time, rng=rng_bg)

        if contaminate:
            length = max(1, int(round(self.contamination_extent_ratio * n_time)))
            length = min(length, n_time)
            start = int(rng.integers(0, max(1, n_time - length + 1)))
            anomaly = ShiftAnomaly(forced_region=(start, start + length),
                                    forced_magnitude_std_multiplier=self.contamination_intensity)
            Y, _, _ = anomaly.apply(Z, 0, n_time, rng)
        else:
            Y = Z

        clean_mean, clean_std = Z.mean(), Z.std()
        Y_norm = (Y - clean_mean) / (clean_std + 1e-8)
        return Y_norm, n_time

    def _draw_reference_set(self, K: int, rng: np.random.Generator):
        if K == 0:
            return [], []
        contaminate_flags = (rng.random(K) < self.contamination_prob).tolist()
        ref_seeds = rng.integers(0, 2 ** 31 - 1, size=K)
        refs = [
            self._generate_reference(int(s), bool(c), rng)
            for s, c in zip(ref_seeds, contaminate_flags)
        ]
        return refs, contaminate_flags

    def _build_episode_cache(self):
        for entity in self.entities:
            rng = np.random.default_rng(self.base_seed + EPISODE_EVAL_SEED_OFFSET + entity.entity_id)
            K = self._sample_k(rng)
            refs, contam = self._draw_reference_set(K, rng)
            self._episode_cache[entity.entity_id] = (K, refs, contam)

    def sample_alternate_references(self, idx: int, K: int = None):
        """Draws a SECOND, independent reference subset for the SAME query
        index -- used by the trainer's reference-consistency loss (Section
        7), never by __getitem__ itself. Uses the shared episode RNG stream
        (train mode) so it never repeats what __getitem__ already drew."""
        if K is None:
            K = self._sample_k(self._episode_rng)
        return self._draw_reference_set(K, self._episode_rng)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        entity = self.entities[idx]

        is_universal = self.intensity_mode == INTENSITY_MODE_UNIVERSAL
        if entity.is_anomalous and is_universal:
            item["D"] = float(item["intensity_value_raw"] * item["sigma_ref"])
        else:
            item["D"] = 0.0

        if self.train:
            K = self._sample_k(self._episode_rng)
            refs, contam = self._draw_reference_set(K, self._episode_rng)
        else:
            K, refs, contam = self._episode_cache[entity.entity_id]

        item["K"] = K
        item["references"] = refs
        item["reference_contaminated"] = contam

        if self.include_alternate_references:
            refs_b, contam_b = self.sample_alternate_references(idx, K=K)
            item["references_b"] = refs_b
            item["reference_contaminated_b"] = contam_b

        return item


def episodic_pad_collate(batch, max_len=550):
    """Like contrastive_pad_collate, plus batched, K-padded reference
    tensors. Different items may have different K (their own episode's
    reference-count regime) -- references are right-padded up to the
    BATCH's own max K, with ref_k_valid_mask marking which slots are real
    vs pure batch padding (see ReferenceContextEncoder's k_valid_mask)."""
    lengths = [item["n_time"] for item in batch]
    too_long = [n for n in lengths if n > max_len]
    if too_long:
        raise ValueError(f"sample length(s) {too_long} exceed max_len={max_len}")

    B, T = len(batch), max_len
    Y_padded = torch.zeros(B, 1, T)
    pad_mask = torch.zeros(B, 1, T)
    for i, item in enumerate(batch):
        n = item["n_time"]
        Y_padded[i, :, :n] = item["Y"]
        pad_mask[i, :, :n] = 1.0

    max_k = max((item["K"] for item in batch), default=0)
    max_k = max(max_k, 1)  # keep tensor shapes well-defined even in an all-K=0 batch
    ref_x = torch.zeros(B, max_k, 1, T)
    ref_pad_mask = torch.zeros(B, max_k, 1, T)
    ref_k_valid_mask = torch.zeros(B, max_k)
    for i, item in enumerate(batch):
        for k, (Y_ref, n_ref) in enumerate(item["references"]):
            ref_x[i, k, 0, :n_ref] = torch.from_numpy(Y_ref[0]).float()
            ref_pad_mask[i, k, 0, :n_ref] = 1.0
            ref_k_valid_mask[i, k] = 1.0

    out = {
        "Y": Y_padded,
        "pad_mask": pad_mask,
        "shape_label": torch.tensor([item["shape_label"] for item in batch], dtype=torch.long),
        "location_value": torch.tensor([item["location_value"] for item in batch], dtype=torch.float32),
        "extent_value": torch.tensor([item["extent_value"] for item in batch], dtype=torch.float32),
        "intensity_value": torch.tensor([item["intensity_value"] for item in batch], dtype=torch.float32),
        "D": torch.tensor([item["D"] for item in batch], dtype=torch.float32),
        "ref_x": ref_x,
        "ref_pad_mask": ref_pad_mask,
        "ref_k_valid_mask": ref_k_valid_mask,
        "K": torch.tensor([item["K"] for item in batch], dtype=torch.long),
        "lengths": torch.tensor(lengths),
    }

    if all("references_b" in item for item in batch):
        ref_x_b = torch.zeros(B, max_k, 1, T)
        ref_pad_mask_b = torch.zeros(B, max_k, 1, T)
        ref_k_valid_mask_b = torch.zeros(B, max_k)
        for i, item in enumerate(batch):
            for k, (Y_ref, n_ref) in enumerate(item["references_b"]):
                ref_x_b[i, k, 0, :n_ref] = torch.from_numpy(Y_ref[0]).float()
                ref_pad_mask_b[i, k, 0, :n_ref] = 1.0
                ref_k_valid_mask_b[i, k] = 1.0
        out["ref_x_b"] = ref_x_b
        out["ref_pad_mask_b"] = ref_pad_mask_b
        out["ref_k_valid_mask_b"] = ref_k_valid_mask_b

    return out
