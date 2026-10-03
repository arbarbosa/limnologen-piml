"""
fig03_architecture_v2.py
========================
by Andre R. Barbosa, April - October 2026

Paper Fig. 1 — Three-level PIML architecture flowchart.

Uses matplotlib patches + annotations only — no external dependencies.

Layout (top → bottom):
  ┌─────────────────────────────────────────┐
  │  ENVIRONMENTAL INPUTS                   │  (T, RH → EMC)
  │  DIFFUSION-LAG FILTER                   │  (τ_EMC, τ_T per mode)
  ├──────────────┬──────────────────────────┤
  │ Level 1      │ Level 2 (MDOF twin)      │  physics prediction f̂_phys
  │ Adim twin    │ Eigenvalue f₀(k)         │
  ├──────────────┴──────────────────────────┤
  │         ⊖ subtract physics              │
  │  Level 3: GP residual g(z)              │
  ├─────────────────────────────────────────┤
  │  PIML prediction: f̂ = f̂_phys + g       │
  └─────────────────────────────────────────┘

Usage:
    python analysis/fig03_architecture_v2.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

ROOT     = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "analysis" / "output" / "Fig_Architecture_PIML_v2.png"

DPI = 160

# ── colour palette ─────────────────────────────────────────────────────────────
C_ENV   = "#E8F4FD"   # light blue — environmental inputs
C_LAG   = "#D0EAF8"   # blue — lag filter
C_L1    = "#FFF3CD"   # amber — Level 1
C_L2    = "#D4EDDA"   # green — Level 2
C_L3    = "#F8D7DA"   # pink — Level 3
C_OUT   = "#EDE7F6"   # purple — output
C_ARROW = "#555555"
EDGE    = "#444444"
TXT     = "#111111"

BOLD = dict(fontweight="bold")

_STYLE = {
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.style":      "italic",
    "font.size":       9.5,
}


def _box(ax, x, y, w, h, fc, label, sublabel=None,
         fontsize=9.5, bold=False, radius=0.025, ls="solid"):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         fc=fc, ec=EDGE, lw=1.0, ls=ls, zorder=3)
    ax.add_patch(box)
    kw = dict(ha="center", va="center", fontsize=fontsize,
              color=TXT, zorder=4, linespacing=1.4)
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.025, label, fontweight="bold" if bold else "normal", **kw)
        ax.text(cx, cy - 0.030, sublabel, fontsize=fontsize - 0.5,
                style="italic", **{k: v for k, v in kw.items() if k != "fontsize"})
    else:
        ax.text(cx, cy, label, fontweight="bold" if bold else "normal", **kw)


def _arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.2, mutation_scale=12),
                zorder=5)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.01, my, label, fontsize=8, color="#555555",
                va="center", style="italic")


def main():
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.045, 1)
    ax.axis("off")

    # ── Row coordinates (y from top) ──────────────────────────────────────────
    M   = 0.04   # left/right margin
    W   = 0.92   # total usable width
    GAP = 0.020  # gap between side-by-side boxes

    # Box heights; the Level 3 GP box is taller to fit its text
    # Positions computed top-down with gap=0.026 between rows

    # Environmental inputs (single full-width box — outdoor only)
    h0 = 0.09;  y0 = 0.884
    w_env = W
    _box(ax, M, y0, W, h0, C_ENV,
         "Outdoor weather station  (SMHI Växjö A, ID 64510)",
         r"$T_\mathrm{out}(t)$,  RH$_\mathrm{out}(t)$   [hourly]",
         fontsize=9.5)

    # EMC conversion
    h1 = 0.070;  y1 = y0 - 0.026 - h1   # 0.788
    _box(ax, M, y1, W, h1, C_ENV,
         "EMC(t)  via Hailwood-Horrobin sorption isotherm",
         fontsize=9.5)

    # Lag filter
    h2 = 0.085;  y2 = y1 - 0.026 - h2   # 0.683
    _box(ax, M, y2, W, h2, C_LAG,
         "Diffusion-lag filter  (exponential moving average)",
         r"$\widetilde{\mathrm{EMC}}_{\tau,i}(t)$,  $\widetilde{T}_{\tau,i}(t)$"
         r"  with per-mode lag times $\tau_{\mathrm{EMC},i}$,  $\tau_{T,i}$  [MCMC]",
         fontsize=9.2)

    # Physics layer (two columns)
    h3 = 0.113;  y3 = y2 - 0.026 - h3   # 0.554
    w_L1 = (W - GAP) * 0.50
    w_L2 = (W - GAP) * 0.50
    _box(ax, M, y3, w_L1, h3, C_L1,
         "Level 1 — Adimensional twin   ·   Step 1",
         r"$f_i(t) = f_{\mathrm{ref},i}\sqrt{1 + \alpha_i\,\widetilde{\mathrm{EMC}}_{\tau,i}"
         r" + \beta_i\,\widetilde{T}_{\tau,i}} + \varepsilon$"
         "\n" r"$\alpha_i,\,\beta_i,\,\tau_{\mathrm{EMC},i},\,\tau_{T,i}$  [emcee MCMC]",
         fontsize=8.8, bold=True)
    _box(ax, M + w_L1 + GAP, y3, w_L2, h3, C_L2,
         "Level 2 — Dimensional MDOF twin   ·   Step 2",
         r"7-DOF shear-stack:  $\mathbf{K}(k_X,k_Y)\,\mathbf{v} = \omega^2\mathbf{M}\mathbf{v}$"
         "\n" r"$f_{0,i}(k_X,k_Y)$   [NumPyro NUTS]",
         fontsize=8.8, bold=True)

    # Mid — subtract physics / residual label
    y_sub = y3 - 0.032   # 0.522
    ax.text(0.515, y_sub - 0.004,
            r"$\ominus$  residual:  $r_i(t) = f_i(t) - \hat{f}_{\mathrm{phys},i}(t)$",
            ha="left", va="center", fontsize=9.5, color="#333333", zorder=4)

    # GP residual
    h4 = 0.128;  y4 = y_sub - 0.024 - h4   # ~0.380
    _box(ax, M, y4, W, h4, C_L3,
         "Level 3 — Gaussian process residual layer   ·   Step 3",
         r"$g_i \sim \mathcal{GP}(0,\;\sigma_\mathrm{GP}^2\,k_{\mathrm{M5/2}}(\mathbf{z},\mathbf{z}'))$"
         r"     covariates: $[\mathrm{EMC}_{24\mathrm{h}},\;T_{24\mathrm{h}},"
         r"\;\sin(2\pi h_d/24),\;\cos(2\pi h_d/24)]$"
         "\n"
         r"Variance cap $\sigma_\mathrm{GP}^2 \leq \kappa\,\sigma_\mathrm{obs}^2$"
         r"  +  length-scale box bounds   [tinygp MLE, 12 restarts]",
         fontsize=8.8, bold=True)

    # Output
    h5 = 0.082;  y5 = y4 - 0.026 - h5
    _box(ax, M, y5, W, h5, C_OUT,
         "PIML prediction  +  95 % prediction interval",
         r"$\hat{f}_i(t) = \hat{f}_{\mathrm{phys},i}(t) + g_i(\mathbf{z}(t))$"
         r"     CRPS-validated on windstorm-free hold-out set",
         fontsize=9.2, bold=True)

    # Sequential SHM note
    h6 = 0.118;  y6 = y5 - 0.060 - h6
    _box(ax, M, y6, W, h6, "#F3E5F5",
         "Operational pathway — sequential Bayesian updating",
         r"Trailing 30 d window: $p(k_X,k_Y \mid f_{\mathrm{obs},t-W:t})$"
         r"  with $\alpha,\beta,\tau$ fixed   $\rightarrow$  storey-stiffness drift detection"
         "\n" r"(50 % detection at ≈2.3 % from a single record;  ≈0.8 % with a 30 d window)",
         fontsize=8.8, ls=(0, (5, 3)))

    # ── Vertical arrows ────────────────────────────────────────────────────────
    cx = 0.5
    _arrow(ax, cx, y0,            cx, y1 + h1)
    _arrow(ax, cx, y1,            cx, y2 + h2)
    _arrow(ax, M + w_L1/2,              y2, M + w_L1/2,              y3 + h3)
    _arrow(ax, M + w_L1+GAP+w_L2/2,    y2, M + w_L1+GAP+w_L2/2,    y3 + h3)
    _arrow(ax, M + w_L1/2,              y3, cx - 0.004, y_sub + 0.012)
    _arrow(ax, M + w_L1+GAP+w_L2/2,    y3, cx + 0.004, y_sub + 0.012)
    _arrow(ax, cx, y_sub + 0.012,  cx, y4 + h4)
    _arrow(ax, cx, y4,             cx, y5 + h5)
    ax.add_patch(FancyArrowPatch((cx, y5 - 0.002), (cx, y6 + h6 + 0.002),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=C_ARROW, lw=1.4, ls=(0, (4, 3)), zorder=5))
    ax.text(cx + 0.015, y5 - 0.030,
            "operational use (after baseline identification)",
            ha="left", va="center", fontsize=8.2, color="#555555")

    # ── Legend / annotation ───────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(fc=C_ENV,  ec=EDGE, label="Environmental inputs"),
        mpatches.Patch(fc=C_LAG,  ec=EDGE, label="Lag filter"),
        mpatches.Patch(fc=C_L1,   ec=EDGE, label="Level 1 — Adim twin (emcee)"),
        mpatches.Patch(fc=C_L2,   ec=EDGE, label="Level 2 — MDOF twin (NUTS)"),
        mpatches.Patch(fc=C_L3,   ec=EDGE, label="Level 3 — GP residual (MLE)"),
        mpatches.Patch(fc=C_OUT,  ec=EDGE, label="PIML prediction"),
        mpatches.Patch(fc="#F3E5F5", ec=EDGE, ls=(0, (5, 3)), label="Operational SHM pathway"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
          fontsize=7.8, framealpha=0.9,
          ncol=2,
          bbox_to_anchor=(0.5, -0.045))


    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
