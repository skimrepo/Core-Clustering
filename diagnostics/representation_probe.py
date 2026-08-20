"""Frozen-representation extraction + linear/MLP probes (MTL_DIAGNOSTIC
Phase 2, Problems A.2 and B.5). Does NOT modify core_clustering/ -- this
replicates ContrastiveEncoder's forward pass using its own submodules
(frozen, no_grad) purely to capture intermediate representations that
ContrastiveEncoder.forward() doesn't expose (it only returns per-head
outputs). If this replication ever drifts from the real forward pass,
the model's own outputs (recomputed here too) are compared as a sanity
check in extract_representations' caller-facing tests.

Representations captured (see report Section 4):
  stage2:   (B, 64, 69)  post stage-2 conv+attn, pad-mask-zeroed
  stage3:   (B, 128, 35) post stage-3 conv+attn, pad-mask-zeroed
  squeeze:  (B, 4, 35)   post 1x1 squeeze conv, pad-mask-zeroed
  pool_z:   (B, 4)       final pooled vector (what heads actually read)

Sequence representations (stage2/stage3/squeeze) are flattened directly
(channels*time) rather than mean/max-pooled -- mean-pooling would destroy
positional information by the same mechanism under diagnosis, so a probe
built on a mean-pooled sequence rep couldn't distinguish "info is in the
sequence but pooling loses it" from "info was never in the sequence at
all". Flatten preserves per-timestep values; padded positions are exactly
zero (the encoder re-zeros them after every stage), so this is already
"mask-aware" without extra logic -- and since every input is padded to
the SAME fixed max_len, every instance's stage2/stage3/squeeze shape is
identical, so flatten produces a fixed-size vector.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def extract_representations(model, x, pad_mask=None):
    """Replicates ContrastiveEncoder.forward()'s internal computation,
    capturing stage2/stage3/squeeze/pool_z along the way. model must be in
    eval mode; call under torch.no_grad() (already applied here)."""
    encoder = model.encoder
    h, m = x, pad_mask

    for block in encoder.stem:
        h = block(h)
        if m is not None:
            h = h * m

    captured = {}
    for i, block in enumerate(encoder.blocks):
        h = block(h)
        if m is not None:
            m = F.max_pool1d(m, kernel_size=encoder.kernel_size, stride=encoder.stride, padding=encoder.padding)
            h = h * m
        if str(i) in encoder.attn_by_stage:
            key_padding_mask = (m[:, 0, :] < 0.5) if m is not None else None
            h = encoder.attn_by_stage[str(i)](h, key_padding_mask=key_padding_mask)
            if m is not None:
                h = h * m
        if i == 2:
            captured["stage2"] = (h.clone(), m.clone() if m is not None else None)
        if i == 3:
            captured["stage3"] = (h.clone(), m.clone() if m is not None else None)

    h = encoder.squeeze(h)
    if m is not None:
        h = h * m
    captured["squeeze"] = (h.clone(), m.clone() if m is not None else None)

    # pooling (mirrors ContrastiveEncoder.forward's "attention" branch)
    batch, channels, time = h.shape
    pos = torch.linspace(0.0, 1.0, time, device=h.device).view(1, time, 1).expand(batch, time, 1)
    from core_clustering.models_contrastive import _sinusoidal_position_encoding
    pos_enc = _sinusoidal_position_encoding(pos, channels)
    feat_t = h.transpose(1, 2) + pos_enc
    key_padding_mask = (m[:, 0, :] < 0.5) if m is not None else None
    query = model.pool_query.expand(batch, -1, -1)
    pool_z, _ = model.pool_attn(query, feat_t, feat_t, key_padding_mask=key_padding_mask, need_weights=False)
    pool_z = pool_z.squeeze(1)
    captured["pool_z"] = pool_z

    return captured


def flatten_rep(rep):
    """rep is either (feat, mask) for sequence reps or a plain tensor for
    pool_z. Flattens to (batch, D). Padded positions are already zero
    (see module docstring), so flatten alone is mask-aware."""
    if isinstance(rep, tuple):
        feat, _ = rep
        return feat.reshape(feat.shape[0], -1)
    return rep


@torch.no_grad()
def cache_all_representations(model, dataset, device="cpu", max_len=550):
    """Runs the frozen encoder once over the whole dataset, returns a dict
    of {rep_name: (N, D) numpy array} plus per-instance labels -- so probes
    never need to re-run the encoder (Section 17.9).

    dataset[i]["Y"] is the RAW, un-padded series (length varies per
    instance, e.g. 500-550) -- unlike training (which goes through
    contrastive_pad_collate), so it must be explicitly right-padded to a
    fixed max_len here too, with a matching pad_mask. Without this, stage2/
    stage3/squeeze end up a DIFFERENT compressed length per instance
    (conv output length depends on input length), and np.stack across
    instances fails with a shape mismatch."""
    model.eval()
    reps = {"stage2": [], "stage3": [], "squeeze": [], "pool_z": []}
    shape_labels, loc_vals, ext_vals, int_vals = [], [], [], []

    for i in range(len(dataset)):
        item = dataset[i]
        n_time = item["Y"].shape[1]
        x = torch.zeros(1, 1, max_len)
        x[:, :, :n_time] = item["Y"]
        pad_mask = torch.zeros(1, 1, max_len)
        pad_mask[:, :, :n_time] = 1.0
        x, pad_mask = x.to(device), pad_mask.to(device)

        captured = extract_representations(model, x, pad_mask=pad_mask)
        for name in reps:
            reps[name].append(flatten_rep(captured[name])[0].cpu().numpy())
        shape_labels.append(item["shape_label"])
        loc_vals.append(item["location_value"])
        ext_vals.append(item["extent_value"])
        int_vals.append(item["intensity_value"])

    out = {name: np.stack(vals) for name, vals in reps.items()}
    out["shape_label"] = np.array(shape_labels)
    out["location_value"] = np.array(loc_vals)
    out["extent_value"] = np.array(ext_vals)
    out["intensity_value"] = np.array(int_vals)
    return out


class LinearProbe(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_probe(X_train, y_train, X_val, y_val, probe_type="linear", epochs=100, lr=0.01, patience=10):
    """X: (N, D) float32 numpy, y: (N,) float32 numpy. Trains a tiny probe
    with plain MSE, encoder-independent (X is already-extracted, frozen
    features) -- from core_clustering.metrics import regression_metrics is
    used by the caller to report MAE/RMSE/Pearson/Spearman on the probe's
    val predictions."""
    device = "cpu"
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)

    in_dim = X_train.shape[1]
    probe = LinearProbe(in_dim) if probe_type == "linear" else MLPProbe(in_dim)
    probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    best_train_loss = float("inf")
    stop_counter = 0
    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        pred = probe(X_train_t)
        loss = torch.nn.functional.mse_loss(pred, y_train_t)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_train_loss - 1e-6:
            best_train_loss = loss_val
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter > patience:
            break

    probe.eval()
    with torch.no_grad():
        pred_val = probe(X_val_t).cpu().numpy()
    return pred_val
