"""Task-specific evaluation metrics for the MTL diagnostic campaign
(diagnostics/MTL_DIAGNOSTIC_REPORT.md, Section 2). Kept separate from
core_clustering/ -- these are diagnostic-only, not part of the training
pipeline."""
import numpy as np
from scipy.stats import spearmanr


def regression_metrics(pred, target):
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    if len(pred) > 1 and np.std(pred) > 0 and np.std(target) > 0:
        pearson = float(np.corrcoef(pred, target)[0, 1])
        spearman = float(spearmanr(pred, target).correlation)
    else:
        pearson = float("nan")
        spearman = float("nan")
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "spearman": spearman, "n": len(pred)}


def location_metrics(embeddings: np.ndarray, values: np.ndarray) -> dict:
    """embeddings: (N, D) location embeddings for ANOMALOUS instances only.
    values: (N,) true location values in [0,1]. Compares every pairwise
    embedding distance against the true |value_i - value_j| gap."""
    n = len(values)
    if n < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"),
                "spearman": float("nan"), "n": 0}
    D = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)
    gap = np.abs(values[:, None] - values[None, :])
    iu = np.triu_indices(n, k=1)
    return regression_metrics(D[iu], gap[iu])


def normal_relative_metrics(embeddings: np.ndarray, is_anomalous: np.ndarray, values: np.ndarray) -> dict:
    """embeddings: (N,D) for ALL instances (normal+anomalous) of one head.
    is_anomalous: (N,) bool. values: (N,) true value (only meaningful where
    is_anomalous is True). Distance from each anomaly's embedding to the
    normal-cluster centroid, compared against its true value."""
    normal = embeddings[~is_anomalous]
    anomaly = embeddings[is_anomalous]
    if len(normal) == 0 or len(anomaly) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"),
                "spearman": float("nan"), "n": 0}
    centroid = normal.mean(axis=0)
    d = np.linalg.norm(anomaly - centroid, axis=1)
    return regression_metrics(d, values[is_anomalous])


def shape_metrics(embeddings: np.ndarray, shape_labels: np.ndarray) -> dict:
    """embeddings: (N,D). shape_labels: (N,) int (0=normal,1=shift).
    Positive/negative pair mean distance, separation, and leave-one-out
    1-NN classification accuracy."""
    n = len(shape_labels)
    D = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=-1)
    same = shape_labels[:, None] == shape_labels[None, :]
    iu = np.triu_indices(n, k=1)
    pos_mask, neg_mask = same[iu], ~same[iu]
    pos_dist = float(D[iu][pos_mask].mean()) if pos_mask.any() else float("nan")
    neg_dist = float(D[iu][neg_mask].mean()) if neg_mask.any() else float("nan")

    D_noself = D.copy()
    np.fill_diagonal(D_noself, np.inf)
    nn_idx = D_noself.argmin(axis=1)
    pred = shape_labels[nn_idx]
    nn_accuracy = float((pred == shape_labels).mean())

    return {
        "positive_pair_mean_distance": pos_dist,
        "negative_pair_mean_distance": neg_dist,
        "separation": neg_dist - pos_dist,
        "nn_accuracy": nn_accuracy,
        "n": n,
    }


def embedding_collapse_stats(embeddings: np.ndarray) -> dict:
    """Generic collapse diagnostics for any (N,D) embedding set (Section 13)."""
    per_dim_std = embeddings.std(axis=0)
    norms = np.linalg.norm(embeddings, axis=1)
    n = len(embeddings)
    if n > 500:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=500, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings
    D = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    iu = np.triu_indices(len(sample), k=1)
    return {
        "mean_per_dim_std": float(per_dim_std.mean()),
        "min_per_dim_std": float(per_dim_std.min()),
        "mean_embedding_norm": float(norms.mean()),
        "mean_pairwise_distance": float(D[iu].mean()) if len(sample) > 1 else float("nan"),
    }
