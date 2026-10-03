"""
Paper Fig. 6 — Stiffness/mass EMC decomposition.
by Andre R. Barbosa, April - October 2026

Usage:
    python analysis/fig05_split_decomposition.py

Outputs:
    analysis/output/Fig_Level1b_Split_Decomposition.png
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import truncnorm, gaussian_kde

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
DPI = 160
FIGSIZE = (11, 4.2)

MODE_COLORS = {
    "Mode 1": "#2166ac",   # blue
    "Mode 2": "#4dac26",   # green
    "Mode 3": "#d7191c",   # red
}
ALPHA_M_COLOR = "#888888"  # grey

BW_METHOD = "scott"        # KDE bandwidth selector
N_MC_SAMPLES = 20_000      # Monte Carlo draws for α_m and α_k
RNG_SEED = 20260608        # fixed seed so the figure is reproducible
_RNG = np.random.default_rng(RNG_SEED)

# Fallback prior params if not present in JSON
_ALPHA_M_MEAN_DEFAULT = 0.00456
_ALPHA_M_SIGMA_DEFAULT = 0.0015
_ALPHA_M_LOWER_DEFAULT = 0.0

# ── PATHS ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "analysis" / "output" / "Fig_Level1b_Split_Decomposition.png"
POST_JSON = ROOT / "aux" / "level1_adim" / "v3_per_mode_tau" / "posterior.json"
SPLIT_JSON = ROOT / "aux" / "level1b_split" / "decomposition_results.json"


# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_alpha_eff_draws(post_json: Path) -> dict:
    """Return α_eff posterior draws for each mode (raw units, not ×1e3).

    Column layout: alpha_i is at column index 6*i + 1, i=0,1,2.
    """
    with open(post_json) as fh:
        data = json.load(fh)
    arr = np.asarray(data["thinned_draws"])   # shape (N, 18)
    return {
        "Mode 1": arr[:, 1],
        "Mode 2": arr[:, 7],
        "Mode 3": arr[:, 13],
    }


def load_split_results(split_json: Path) -> dict:
    """Return decomposition_results dict from JSON."""
    with open(split_json) as fh:
        return json.load(fh)


def sample_alpha_m(split_data: dict, n: int) -> np.ndarray:
    """Draw n samples from the truncated-normal α_m prior.

    Prior parameters read from split_data['meta']['prior']; falls back to
    defaults if keys are absent.
    """
    prior = split_data.get("meta", {}).get("prior", {})
    mean  = prior.get("alpha_m_mean",  _ALPHA_M_MEAN_DEFAULT)
    sigma = prior.get("alpha_m_sigma", _ALPHA_M_SIGMA_DEFAULT)
    lower = prior.get("alpha_m_lower", _ALPHA_M_LOWER_DEFAULT)

    a = (lower - mean) / sigma   # standardised lower bound
    b = np.inf                   # no upper truncation
    return truncnorm.rvs(a, b, loc=mean, scale=sigma, size=n, random_state=_RNG)


# ── KDE HELPERS ────────────────────────────────────────────────────────────────

def _kde_line(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Evaluate a KDE on x_grid and normalise peak to 1.
    Reflected at zero (all three quantities are non-negative), so no density below 0."""
    kde = gaussian_kde(samples, bw_method=BW_METHOD)
    density = kde(x_grid) + kde(-x_grid)
    density[x_grid < 0] = 0.0
    density /= density.max()
    return density


def _grid(samples: np.ndarray, pad: float = 0.25) -> np.ndarray:
    """Return a 400-point grid spanning [min-pad*range, max+pad*range]."""
    lo, hi = samples.min(), samples.max()
    rng = hi - lo
    return np.linspace(max(0.0, lo - pad * rng), hi + pad * rng, 400)


# ── PANEL DRAWING ──────────────────────────────────────────────────────────────

def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_a(ax: plt.Axes, alpha_eff: dict) -> None:
    """Panel (a): α_eff posterior KDE for all three modes."""
    for mode, draws in alpha_eff.items():
        color = MODE_COLORS[mode]
        x = _grid(draws)
        y = _kde_line(draws, x)
        ax.plot(x * 1e3, y, color=color, lw=1.8, label=mode)
        median = np.median(draws)
        ax.axvline(median * 1e3, color=color, lw=1.0, ls="--", alpha=0.75)

    ax.set_xlabel(r"$\alpha_\mathrm{eff}$  ($\times10^{-3}$  %$^{-1}$)", fontsize=9)
    ax.set_ylabel("Normalized density", fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.35)
    _despine(ax)

    # Legend in panel (a)
    handles = [
        Line2D([0], [0], color=MODE_COLORS[m], lw=1.8, label=m)
        for m in MODE_COLORS
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper right")

    ax.text(0.5, -0.22, "(a)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, style="italic")


def draw_panel_b(ax: plt.Axes, alpha_m_samples: np.ndarray,
                 alpha_m_mean: float) -> None:
    """Panel (b): α_m prior KDE (grey fill)."""
    x = _grid(alpha_m_samples)
    y = _kde_line(alpha_m_samples, x)

    ax.fill_between(x * 1e3, y, color=ALPHA_M_COLOR, alpha=0.30)
    ax.plot(x * 1e3, y, color=ALPHA_M_COLOR, lw=1.8)
    ax.axvline(alpha_m_mean * 1e3, color=ALPHA_M_COLOR, lw=1.0, ls="--", alpha=0.85)

    ax.set_xlabel(r"$\alpha_m$  ($\times10^{-3}$  %$^{-1}$)", fontsize=9)
    ax.set_ylabel("Normalized density", fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.35)
    _despine(ax)

    ax.text(0.5, -0.22, "(b)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, style="italic")


def draw_panel_c(ax: plt.Axes, alpha_k: dict) -> None:
    """Panel (c): α_k derived stiffness sensitivity KDE for all three modes."""
    for mode, draws in alpha_k.items():
        color = MODE_COLORS[mode]
        x = _grid(draws)
        y = _kde_line(draws, x)
        ax.plot(x * 1e3, y, color=color, lw=1.8)
        median = np.median(draws)
        ax.axvline(median * 1e3, color=color, lw=1.0, ls="--", alpha=0.75)

    ax.axvline(0.0, color="black", lw=0.9, ls=":", alpha=0.65)

    ax.set_xlabel(r"$\alpha_k$  ($\times10^{-3}$  %$^{-1}$)", fontsize=9)
    ax.set_ylabel("Normalized density", fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.35)
    _despine(ax)

    ax.text(0.5, -0.22, "(c)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, style="italic")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main() -> None:
    matplotlib.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.style":      "italic",
        "axes.titlesize":  9,
        "axes.labelsize":  9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # ── Load data ──────────────────────────────────────────────────────────────
    alpha_eff = load_alpha_eff_draws(POST_JSON)
    split_data = load_split_results(SPLIT_JSON)

    prior = split_data.get("meta", {}).get("prior", {})
    alpha_m_mean  = prior.get("alpha_m_mean",  _ALPHA_M_MEAN_DEFAULT)
    alpha_m_sigma = prior.get("alpha_m_sigma", _ALPHA_M_SIGMA_DEFAULT)
    alpha_m_lower = prior.get("alpha_m_lower", _ALPHA_M_LOWER_DEFAULT)

    # ── Sample α_m prior ──────────────────────────────────────────────────────
    alpha_m_samples = sample_alpha_m(split_data, N_MC_SAMPLES)

    # ── Derive α_k = α_eff + α_m for each mode ────────────────────────────────
    # Each mode's α_eff draw set may be smaller than N_MC_SAMPLES; resample α_m
    # to match length, then add.
    alpha_k = {}
    for mode, eff_draws in alpha_eff.items():
        n = len(eff_draws)
        am = sample_alpha_m(split_data, n)
        alpha_k[mode] = eff_draws + am

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    draw_panel_a(axes[0], alpha_eff)
    draw_panel_b(axes[1], alpha_m_samples, alpha_m_mean)
    draw_panel_c(axes[2], alpha_k)

    fig.tight_layout(pad=1.2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
