"""Consistent plotting style and reusable diagnostic figures.
by Andre R. Barbosa, April - October 2026
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


MODE_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
MODE_LABELS = [
    "Mode 1  (X-trans, ~2.16 Hz)",
    "Mode 2  (Y-trans, ~2.24 Hz)",
    "Mode 3  (Y east-wing, ~2.46 Hz)",
]


def set_paper_style() -> None:
    """Apply a consistent rcParam style for paper-figure-quality plots."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.style": "italic",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def posterior_marginals_grid(
    draws: np.ndarray,            # (n_draws, n_params)
    param_names: list[str],
    param_labels: list[str],
    truth: dict[str, float] | None = None,  # for overlay (e.g., prior mean)
    fig_size: tuple[float, float] = (13, 9),
) -> plt.Figure:
    """Draw a grid of histograms with 95 % CI markers per parameter."""
    n_params = draws.shape[1]
    ncols = 4
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
    axes = axes.flatten()
    for i in range(n_params):
        ax = axes[i]
        x = draws[:, i]
        ax.hist(x, bins=40, color="#4477AA", alpha=0.85, edgecolor="white")
        med = np.median(x)
        lo, hi = np.quantile(x, [0.025, 0.975])
        ax.axvline(med, color="black", lw=1.0, ls="-")
        ax.axvline(lo, color="black", lw=0.6, ls=":")
        ax.axvline(hi, color="black", lw=0.6, ls=":")
        if truth is not None and param_names[i] in truth:
            ax.axvline(truth[param_names[i]], color="crimson", lw=1.2, ls="--",
                       alpha=0.8, label="prior mean")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(param_labels[i], fontsize=9.5)
        ax.tick_params(axis="both", labelsize=7.5)
    for j in range(n_params, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def predictive_overlay(
    times: np.ndarray,            # (N,) datetimes
    f_obs: np.ndarray,            # (N,) observed
    f_pred_median: np.ndarray,    # (N,)
    f_pred_lo: np.ndarray,        # (N,) 2.5 %
    f_pred_hi: np.ndarray,        # (N,) 97.5 %
    mode_label: str,
    color: str = "#1f77b4",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlay 95 % predictive band on observations for one mode."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 3))
    ax.scatter(times, f_obs, s=3, alpha=0.30, color=color, edgecolor="none",
               label=f"obs ({mode_label})")
    ax.fill_between(times, f_pred_lo, f_pred_hi, color=color, alpha=0.15,
                    edgecolor="none", label="95 % predictive band")
    ax.plot(times, f_pred_median, "-", color=color, lw=1.0, alpha=0.9,
            label="predictive median")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(fontsize=7.5, loc="upper right", ncol=3)
    return ax
