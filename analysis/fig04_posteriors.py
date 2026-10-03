"""
fig04_posteriors.py
===================
by Andre R. Barbosa, April - October 2026

Paper Fig. 4 — Level-1 posterior marginals: α, β, τ_EMC, τ_T across three modes.

Data source:
  aux/level1_adim/v3_per_mode_tau/posterior.json
  — 18-parameter per-mode-τ emcee run (64 walkers × 4000 production steps,
    2000-step warmup, seed 20260609).  All four panels (α, β, τ_EMC, τ_T)
    use actual thinned MCMC draws.

Parameter layout (6 per mode, indices 6*i + k for i=0..2):
  k=0  f_ref_i       k=1  alpha_i       k=2  beta_i
  k=3  log_sigma_obs_i    k=4  log_tau_EMC_h_i    k=5  log_tau_T_h_i

Usage:
    python analysis/fig04_posteriors.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

ROOT        = Path(__file__).resolve().parent.parent
POST_JSON   = ROOT / "aux" / "level1_adim" / "v3_per_mode_tau" / "posterior.json"
OUT_PATH    = ROOT / "analysis" / "output" / "Fig_Level1_Posteriors_Violin.png"

DPI = 160

MODE_LABELS = ["Mode 1", "Mode 2", "Mode 3"]
COLORS      = ["#2166AC", "#4DAC26", "#D6604D"]   # blue / green / red

_STYLE = {
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.style":      "italic",
    "font.size":       9.5,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def _violin(ax, data_list, x_positions, colors, width=0.35):
    """
    Draw half-violin (mirrored) + median marker for each mode.
    data_list : list of 1-D arrays
    """
    for i, (data, xp, c) in enumerate(zip(data_list, x_positions, colors)):
        kde = gaussian_kde(data, bw_method=0.25)
        lo, hi = np.percentile(data, [0.5, 99.5])
        ys = np.linspace(lo, hi, 300)
        ks = kde(ys)
        ks = ks / ks.max() * (width / 2)   # normalise to half-width

        # Filled violin (both sides)
        ax.fill_betweenx(ys, xp - ks, xp + ks, color=c, alpha=0.45, zorder=3)
        ax.plot(xp - ks, ys, color=c, lw=0.7, alpha=0.7, zorder=4)
        ax.plot(xp + ks, ys, color=c, lw=0.7, alpha=0.7, zorder=4)

        # Median line
        med = np.median(data)
        ax.scatter([xp], [med], color=c, s=28, zorder=5, edgecolors="white", lw=0.8)

        # 95% CI whisker
        q025, q975 = np.percentile(data, [2.5, 97.5])
        ax.plot([xp, xp], [q025, q975], color=c, lw=1.2, zorder=4, solid_capstyle="round")


def main():
    plt.rcParams.update(_STYLE)

    # ── Load per-mode-τ posterior draws ──────────────────────────────────────
    # Layout: 6 params per mode (i=0..2), index = 6*i + k
    #   k=0 f_ref, k=1 alpha, k=2 beta, k=3 log_sigma_obs,
    #   k=4 log_tau_EMC_h, k=5 log_tau_T_h
    with open(POST_JSON) as f:
        post = json.load(f)

    draws = np.array(post["thinned_draws"])   # (N, 18)

    alpha   = [draws[:, 6*i + 1]                    for i in range(3)]
    beta    = [draws[:, 6*i + 2]                    for i in range(3)]
    tau_emc = [np.exp(draws[:, 6*i + 4]) / 24.0     for i in range(3)]  # h → days
    tau_t   = [np.exp(draws[:, 6*i + 5]) / 24.0     for i in range(3)]  # h → days

    # ── Layout: 2 × 2 ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.35})

    x_pos = [1, 2, 3]

    # Panel (0,0) — α_i  [× 10^{-3} %^{-1}]
    ax = axes[0, 0]
    _violin(ax, [a * 1e3 for a in alpha], x_pos, COLORS)
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax.set_ylabel(r"$\alpha_i$  ($\times 10^{-3}$ %$^{-1}$)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODE_LABELS)
    ax.set_xlim(0.4, 3.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="gray", alpha=0.35)
    ax.text(0.5, -0.16, "(a)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, style="italic")

    # Panel (0,1) — β_i  [× 10^{-3} °C^{-1}]
    ax = axes[0, 1]
    _violin(ax, [b * 1e3 for b in beta], x_pos, COLORS)
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax.set_ylabel(r"$\beta_i$  ($\times 10^{-3}$ °C$^{-1}$)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODE_LABELS)
    ax.set_xlim(0.4, 3.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="gray", alpha=0.35)
    # Annotate the suppressed Mode-2 temperature sensitivity
    ax.text(2, 0.05, r"$|\beta_2| \ll |\beta_1|,|\beta_3|$", ha="center", fontsize=8,
            color=COLORS[1], style="italic")
    ax.text(0.5, -0.16, "(b)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, style="italic")

    # Panel (1,0) — τ_EMC,i  [days]
    ax = axes[1, 0]
    _violin(ax, tau_emc, x_pos, COLORS)
    # Theory band for CLT through-thickness (25–55 d)
    ax.axhspan(25, 55, color="#AAAAAA", alpha=0.18, label="CLT bilateral diffusion (theory)")
    ax.set_ylabel(r"$\tau_{\mathrm{EMC},i}$  (days)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODE_LABELS)
    ax.set_xlim(0.4, 3.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="gray", alpha=0.35)
    ax.legend(fontsize=7.8, loc="upper left", framealpha=0.8)
    pass  # no annotation for Mode 3
    ax.text(0.5, -0.16, "(c)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, style="italic")

    # Panel (1,1) — τ_T,i  [days]
    ax = axes[1, 1]
    _violin(ax, tau_t, x_pos, COLORS)
    ax.set_ylabel(r"$\tau_{T,i}$  (days)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODE_LABELS)
    ax.set_xlim(0.4, 3.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, color="gray", alpha=0.35)
    ax.text(0.5, -0.16, "(d)", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, style="italic")

    # ── Shared legend (mode colours) ──────────────────────────────────────────
    handles = [mpatches.Patch(color=c, alpha=0.65, label=lbl)
               for c, lbl in zip(COLORS, MODE_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.015))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
