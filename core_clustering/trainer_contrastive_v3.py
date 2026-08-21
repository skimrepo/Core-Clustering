import json
import os
import time
from typing import List, Optional

import numpy as np
import torch

from core_clustering.losses_contrastive import (
    NormalRelativeRegressionLoss,
    PairwiseGapRegressionLoss,
    ShapeContrastiveLoss,
)
from core_clustering.losses_v3 import ReferenceConsistencyLoss
from core_clustering.prob_heads import heteroscedastic_weight, laplace_nll

ATTRS = ("shape", "location", "extent", "intensity")

# Fixed, conservative, NOT swept (per MTL_V3_REPORT.md's explicit instruction).
DEFAULT_LAMBDA_GEOM = 0.1
DEFAULT_LAMBDA_REF = 0.1
DEFAULT_CONSISTENCY_PROB = 0.2


class ContrastiveTrainerV3:
    """Single AdamW over the ENTIRE model (trunk + heads + reference
    context/fusion + scalar/uncertainty adapters) plus the loss modules'
    own learnable parameters (Shape's temperature, the low-weight geometry
    auxiliaries) -- no per-task optimizer split, no duplicated trunk
    parameters across optimizers, matching V2.1-V2.3's established
    single-optimizer convention.

    Total loss (Section 12 of the spec):
        L = L_shape (heteroscedastic-weighted contrastive)
          + L_location_prob + L_extent_prob + L_intensity_prob (Laplace NLL)
          + lambda_geom * (existing low-weight geometry auxiliaries)
          + lambda_ref * L_ref_consistency (only on a random subset of
            batches, via EpisodicContrastiveDataset's optional second
            reference draw)

    Location/Extent's probabilistic loss is masked to anomalous samples
    only (their existing validity semantics -- undefined for normal).
    Intensity's D target is defined for EVERY sample (D=0 for normal), so
    its Laplace NLL uses the full batch."""

    def __init__(self, model, device: str = "cpu", lr: float = 0.001, max_grad_norm: float = 1.0,
                 patience: int = 10, output_dir: Optional[str] = None,
                 lambda_geom: float = DEFAULT_LAMBDA_GEOM, lambda_ref: float = DEFAULT_LAMBDA_REF,
                 consistency_prob: float = DEFAULT_CONSISTENCY_PROB, seed: int = 0,
                 shape_objective: str = "heteroscedastic"):
        if shape_objective not in ("heteroscedastic", "plain"):
            raise ValueError(f"shape_objective must be 'heteroscedastic' or 'plain', got {shape_objective!r}")
        self.shape_objective = shape_objective
        self.device = device
        self.model = model.to(device)
        self.shape_loss = ShapeContrastiveLoss().to(device)
        self.location_geom_loss = PairwiseGapRegressionLoss().to(device)
        self.extent_geom_loss = NormalRelativeRegressionLoss().to(device)
        self.ref_consistency_loss = ReferenceConsistencyLoss()

        all_params = (
            list(self.model.parameters()) + list(self.shape_loss.parameters())
            + list(self.location_geom_loss.parameters()) + list(self.extent_geom_loss.parameters())
        )
        self.optimizer = torch.optim.AdamW(all_params, lr=lr)

        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.output_dir = output_dir
        self.lambda_geom = lambda_geom
        self.lambda_ref = lambda_ref
        self.consistency_prob = consistency_prob
        self._rng = np.random.default_rng(seed)
        self.best_val_loss = float("inf")
        self.best_epoch: Optional[int] = None

    def _all_params(self):
        return (
            list(self.model.parameters()) + list(self.shape_loss.parameters())
            + list(self.location_geom_loss.parameters()) + list(self.extent_geom_loss.parameters())
        )

    def _compute_losses(self, batch, use_consistency: bool):
        device = self.device
        Y = batch["Y"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        shape = batch["shape_label"].to(device)
        loc = batch["location_value"].to(device)
        ext = batch["extent_value"].to(device)
        D = batch["D"].to(device)
        ref_x = batch["ref_x"].to(device)
        ref_pad_mask = batch["ref_pad_mask"].to(device)
        ref_k_valid_mask = batch["ref_k_valid_mask"].to(device)

        out = self.model(Y, query_pad_mask=pad_mask, ref_x=ref_x, ref_pad_mask=ref_pad_mask,
                          ref_k_valid_mask=ref_k_valid_mask)
        is_anom = shape == 1

        if self.shape_objective == "heteroscedastic":
            mean_shape_loss, per_anchor, valid_anchor = self.shape_loss(
                out["embeddings"]["shape"], shape, return_per_sample=True
            )
            if valid_anchor.any():
                l_shape = heteroscedastic_weight(per_anchor[valid_anchor], out["shape_scale"][valid_anchor])
            else:
                l_shape = mean_shape_loss
        else:  # "plain" -- V3.1: original ShapeContrastiveLoss, no scale weighting.
            # out["shape_scale"] is still computed by the model (diagnostic-only,
            # non-calibrated) but intentionally not read here.
            l_shape = self.shape_loss(out["embeddings"]["shape"], shape)

        if is_anom.any():
            l_loc = laplace_nll(loc[is_anom], out["location_mu"][is_anom], out["location_scale"][is_anom])
            l_ext = laplace_nll(ext[is_anom], out["extent_mu"][is_anom], out["extent_scale"][is_anom])
        else:
            l_loc = Y.new_tensor(0.0)
            l_ext = Y.new_tensor(0.0)

        l_int = laplace_nll(D, out["intensity_mu"], out["intensity_scale"])  # every sample, D=0 for normal

        anomaly_pair_mask = is_anom.unsqueeze(0) & is_anom.unsqueeze(1)
        geom_loc = self.location_geom_loss(out["embeddings"]["location"], loc, anomaly_pair_mask)
        geom_ext = self.extent_geom_loss(out["embeddings"]["extent"], is_anom, ext)

        total = l_shape + l_loc + l_ext + l_int + self.lambda_geom * (geom_loc + geom_ext)

        ref_consistency = None
        if use_consistency and "ref_x_b" in batch:
            ref_x_b = batch["ref_x_b"].to(device)
            ref_pad_mask_b = batch["ref_pad_mask_b"].to(device)
            ref_k_valid_mask_b = batch["ref_k_valid_mask_b"].to(device)
            out_b = self.model(Y, query_pad_mask=pad_mask, ref_x=ref_x_b, ref_pad_mask=ref_pad_mask_b,
                                ref_k_valid_mask=ref_k_valid_mask_b)
            ref_consistency = self.ref_consistency_loss(out, out_b)
            total = total + self.lambda_ref * ref_consistency

        components = {
            "shape": float(l_shape.item()), "location": float(l_loc.item()), "extent": float(l_ext.item()),
            "intensity": float(l_int.item()), "geom_location": float(geom_loc.item()),
            "geom_extent": float(geom_ext.item()),
            "ref_consistency": float(ref_consistency.item()) if ref_consistency is not None else None,
        }
        return total, components, out

    def _run_epoch(self, dataloader, train: bool):
        self.model.train(train)
        self.shape_loss.train(train)
        total_loss = 0.0
        component_sums = {k: 0.0 for k in ("shape", "location", "extent", "intensity",
                                            "geom_location", "geom_extent")}
        step_count = 0
        for batch in dataloader:
            use_consistency = train and bool(self._rng.random() < self.consistency_prob)
            if train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                total, components, _ = self._compute_losses(batch, use_consistency=use_consistency)
            if torch.isnan(total) or torch.isinf(total):
                continue
            if train:
                total.backward()
                torch.nn.utils.clip_grad_norm_(self._all_params(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += float(total.item())
            for k in component_sums:
                component_sums[k] += components[k]
            step_count += 1

        if step_count == 0:
            return float("nan"), {k: float("nan") for k in component_sums}
        avg_components = {k: v / step_count for k, v in component_sums.items()}
        return total_loss / step_count, avg_components

    def train(self, train_dataloader, val_dataloader=None, epochs: int = 100) -> List[dict]:
        history: List[dict] = []
        stop_counter = 0

        for epoch in range(epochs):
            start = time.time()
            train_loss, train_components = self._run_epoch(train_dataloader, train=True)
            epoch_seconds = time.time() - start

            val_loss = None
            is_best = False
            if val_dataloader is not None:
                val_loss, _ = self._run_epoch(val_dataloader, train=False)
                if val_loss != val_loss:
                    stop_counter += 10
                elif val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    stop_counter = 0
                    is_best = True
                    if self.output_dir:
                        os.makedirs(self.output_dir, exist_ok=True)
                        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "bestmodel.pkl"))
                else:
                    stop_counter += 1

            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                      "epoch_seconds": epoch_seconds, "is_best": is_best}
            record.update({f"loss_{k}": v for k, v in train_components.items()})
            history.append(record)

            if self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                with open(os.path.join(self.output_dir, "epoch_history.json"), "w") as f:
                    json.dump(history, f, indent=2)

            line = f"epoch {epoch:4d}  train_loss {train_loss:.4f}"
            if val_loss is not None:
                line += f"  val_loss {val_loss:.4f}"
            line += f"  time {epoch_seconds:.1f}s"
            if is_best:
                line += "  *best*"
            print(line, flush=True)

            if val_dataloader is not None and stop_counter > self.patience - 1:
                break

        if self.best_epoch is None and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, "bestmodel.pkl"))

        return history
