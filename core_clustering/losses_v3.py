import torch
import torch.nn as nn

SCALAR_MU_KEYS = ("location_mu", "extent_mu", "intensity_mu")


class ReferenceConsistencyLoss(nn.Module):
    """Penalizes disagreement between two model OUTPUT dicts for the SAME
    query under two independently-sampled valid reference subsets (see
    MTL_V3_REPORT.md Section 7) -- operates on predictions/embeddings, not
    on raw reference embeddings, and reads whichever scalar_mu keys are
    actually present in the output dict (so it composes with however many
    scalar heads a given model exposes, without hardcoding Location/Extent/
    Intensity by name beyond the shared key convention)."""

    def forward(self, out_a: dict, out_b: dict) -> torch.Tensor:
        terms = []
        for key in SCALAR_MU_KEYS:
            if key in out_a and key in out_b:
                terms.append(torch.abs(out_a[key] - out_b[key]).mean())
        if "shape" in out_a.get("embeddings", {}) and "shape" in out_b.get("embeddings", {}):
            terms.append((out_a["embeddings"]["shape"] - out_b["embeddings"]["shape"]).norm(dim=-1).mean())
        if not terms:
            return torch.tensor(0.0)
        return torch.stack(terms).mean()
