import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from core_clustering.colors import AXIS, CHART_COLORS, CHART_MARKERS, GRID, INK, INK2, MUTED, SURFACE

Z_COLOR = CHART_COLORS[1]
Y_COLOR = CHART_COLORS[5]
SHADE_COLOR = "#e34948"


def plot_example_window(ax, Y, Z, mask, anomaly_type, waveform_type=None, seed=None, show_legend=True):
    """Single-window before/after renderer (RedLamp/AnomSim style): dashed Z
    (normal counterfactual) vs solid Y (observed), anomaly region shaded.
    Y/Z/mask: shape (window_size,) or (1, window_size)."""
    Y = np.asarray(Y).reshape(-1)
    Z = np.asarray(Z).reshape(-1)
    mask = np.asarray(mask).reshape(-1)
    t = np.arange(len(Y))

    if anomaly_type == "normal":
        ax.plot(t, Y, color=Y_COLOR, linewidth=1.2, label="Y (no injection)")
    else:
        ax.plot(t, Z, linestyle="--", color=Z_COLOR, linewidth=1.2, label="Z (normal)")
        ax.plot(t, Y, color=Y_COLOR, linewidth=1.2, label="Y (injected)")

        zero_idx = np.where(mask == 0)[0]
        if len(zero_idx) == 1:
            ax.axvline(zero_idx[0], color=SHADE_COLOR, alpha=0.4, linewidth=1.5)
        elif len(zero_idx) > 1:
            # Contiguous shading from first to last flagged index -- this
            # also correctly covers a mask that runs to the series end
            # (e.g. AnomSim's "wander" type) without needing a name-based
            # special case: zero_idx[-1] + 1 == len(Y) in that case already.
            ax.axvspan(zero_idx[0], zero_idx[-1] + 1, color=SHADE_COLOR, alpha=0.15)

    label_parts = []
    if waveform_type:
        label_parts.append(str(waveform_type))
    label_parts.append(str(anomaly_type))
    if seed is not None:
        label_parts.append(f"seed={seed}")
    ax.set_ylabel("\n".join(label_parts), color=INK, fontsize=8)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.tick_params(colors=MUTED, labelsize=7)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7, facecolor=SURFACE, edgecolor=AXIS)


def plot_tsne_by_class(embeddings, class_idx, class_names, save_path, title=None, perplexity=30, seed=0):
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(np.asarray(embeddings))

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, name in enumerate(class_names):
        mask = class_idx == i
        if not mask.any():
            continue
        color = CHART_COLORS[i % len(CHART_COLORS)]
        marker = CHART_MARKERS[(i // len(CHART_COLORS)) % len(CHART_MARKERS)]
        ax.scatter(coords[mask, 0], coords[mask, 1], s=16, alpha=0.85, edgecolors="none", color=color, marker=marker, label=name)

    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), labelcolor=INK2)
    if title:
        ax.set_title(title, color=INK)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_tsne_by_domain(embeddings, class_idx, domain_idx, class_names, domain_names, save_path, title=None, perplexity=30, seed=0):
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(np.asarray(embeddings))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(right=0.72)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for j, _dname in enumerate(domain_names):
        marker = CHART_MARKERS[j % len(CHART_MARKERS)]
        for i, _cname in enumerate(class_names):
            mask = (class_idx == i) & (domain_idx == j)
            if mask.any():
                ax.scatter(
                    coords[mask, 0], coords[mask, 1], color=CHART_COLORS[i % len(CHART_COLORS)],
                    marker=marker, s=16, alpha=0.85, edgecolors="none",
                )

    type_handles = [
        ax.scatter([], [], color=CHART_COLORS[i % len(CHART_COLORS)], s=16, label=name)
        for i, name in enumerate(class_names)
    ]
    domain_handles = [
        ax.scatter([], [], color=INK2, marker=CHART_MARKERS[j % len(CHART_MARKERS)], s=16, label=dname)
        for j, dname in enumerate(domain_names)
    ]
    type_legend = ax.legend(
        handles=type_handles, title="anomaly type", loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.85,
    )
    ax.add_artist(type_legend)
    ax.legend(
        handles=domain_handles, title="domain", loc="lower left", bbox_to_anchor=(1.02, 0.0),
        frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.85,
    )

    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, color=INK)
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_representative_samples(dataset, indices, out_dir, n_per_domain=3, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    domains = sorted(set(dataset.domain[indices].tolist()))
    for domain in domains:
        domain_idx = indices[dataset.domain[indices] == domain]
        k = min(n_per_domain, len(domain_idx))
        if k < n_per_domain:
            print(f"Note: domain '{domain}' has only {len(domain_idx)} example(s) (< requested {n_per_domain}); using all available.")
        chosen = rng.choice(domain_idx, size=k, replace=False)
        for i, idx in enumerate(chosen):
            fig, ax = plt.subplots(figsize=(9, 2.3))
            fig.patch.set_facecolor(SURFACE)
            plot_example_window(
                ax, dataset.Y[idx], dataset.Z[idx], dataset.labels[idx], dataset.anomaly_type[idx],
                waveform_type=dataset.domain[idx],
            )
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{domain}_{i + 1}.png"), dpi=150, facecolor=SURFACE)
            plt.close(fig)
