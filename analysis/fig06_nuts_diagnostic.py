"""
fig06_nuts_diagnostic.py
========================
by Andre R. Barbosa, April - October 2026

Paper Fig. 7 — MDOF NUTS posterior diagnostic (3 panels):

  (a) Marginal posteriors for k_X and k_Y
  (b) Level 1 (emcee) vs Level 2 (NUTS) cross-check of environmental
      sensitivity parameters (α₁, α₂, β₁, β₂) — forest plot
  (c) Theoretical MDOF shear-stack mode shape vs SSI-cov OMA observations
      at sensor floors (3, 5, 7) for Mode 1 (X) and Mode 2 (Y)

Data sources
------------
  aux/level2_mdof_nuts/posterior.json
  aux/level1_adim/v3_per_mode_tau/posterior.json
  analysis/oma_mode_shapes.json

Output
------
  analysis/output/Fig_Level2_NUTS_Diagnostic.png

Usage
-----
    python analysis/fig06_nuts_diagnostic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.linalg import eigh

# ── PATHS ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
DATA_PATH  = ROOT / "aux" / "level2_mdof_nuts" / "posterior.json"
POST1_PATH = ROOT / "aux" / "level1_adim" / "v3_per_mode_tau" / "posterior.json"
OMA_PATH   = ROOT / "analysis" / "oma_mode_shapes.json"
OUT_PATH   = ROOT / "analysis" / "output" / "Fig_Level2_NUTS_Diagnostic.png"

# ── CONSTANTS ────────────────────────────────────────────────────────────────
DPI      = 160
FIGSIZE  = (13, 5.0)
N_DRAWS  = 5_000
RNG_SEED = 42

N_STOREYS = 7            # 7 stories: Story 1 (concrete FL1→FL2) + Stories 2–7 (CLT)
M_CENTRAL = 1e5          # kg / storey
STOREY_H  = 3.0          # m per storey (concrete: 2.7+0.3, CLT: 2.49+0.51)

# Floor level elevations (Hus 6, Limnologen)
#   FL1 =  0.0 m  (ground / fixed base)
#   FL2 =  3.0 m  (top of concrete basement)
#   FL3 =  6.0 m  (sensor)   FL4 =  9.0 m
#   FL5 = 12.0 m  (sensor)   FL6 = 15.0 m
#   FL7 = 18.0 m  (sensor)   FL8 = 21.0 m  (roof)

# Sensor channel layout — 12 channels, 0-based
# Fl3 → ch 0-3, Fl5 → ch 4-7, Fl7 → ch 8-11
OBS_FLOORS  = [3, 5, 7]
OBS_HEIGHTS = [6.0, 12.0, 18.0]     # metres: FL3=6m, FL5=12m, FL7=18m
CH_X = {3: (0, 2), 5: (4, 6), 7: (8, 10)}
CH_Y = {3: (1, 3), 5: (5, 7), 7: (9, 11)}

# Colors
COL_KX   = "#2166ac"   # blue  — Mode 1 / k_X / Level 1
COL_KY   = "#1a9850"   # green — Mode 2 / k_Y
COL_L2   = "#d6604d"   # orange-red — Level 2 NUTS
COL_GRID = "0.75"

# Panel (c) — sensitivity study colours
COL_R     = {1: "#999999", 10: "#f4a582", 50: "#a50026"}
COL_6     = "#1a9641"   # 6-DOF limit (best fit, dark green)
COL_OMA_Y = "#d95f02"   # OMA Mode 2 Y (orange)

# DOF indices for sensor floors (FL3=6m, FL5=12m, FL7=18m)
#   7-DOF model (DOFs 0–6 at FL2–FL8 = 3,6,…,21 m):
#     FL3=DOF1, FL5=DOF3, FL7=DOF5
#   6-DOF limit (base@FL2=3m, DOFs 0–5 at FL3–FL8 = 6,…,21 m):
#     FL3=DOF0, FL5=DOF2, FL7=DOF4
SENSOR_DOFS_7 = [1, 3, 5]
SENSOR_DOFS_6 = [0, 2, 4]

# OMA runs to exclude
OMA_EXCLUDE = {"Hus6_Jan24"}   # anomalous: FL5 > FL7 amplitude

FONT_SIZE = 10
SCALE     = 1e8        # k displayed in units of 10^8 N/m

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.style":      "italic",
    "font.size":       FONT_SIZE,
    "axes.titlesize":  FONT_SIZE,
    "axes.labelsize":  FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
})


# ── HELPERS ──────────────────────────────────────────────────────────────────

def remove_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, alpha=0.25, color=COL_GRID, linewidth=0.6, axis=axis)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.5, -0.20, label, transform=ax.transAxes,
            fontsize=FONT_SIZE + 1, style="italic", va="top", ha="center")


def build_K_nonuniform(k_springs: np.ndarray) -> np.ndarray:
    """
    Stiffness matrix for a non-uniform shear stack.

    k_springs[j] = stiffness of spring j (j=0 connects fixed base to DOF 0).
    K[i,i] = k_springs[i] + k_springs[i+1]  for i < n-1
    K[n-1, n-1] = k_springs[n-1]
    K[i, i+1] = K[i+1, i] = -k_springs[i+1]
    """
    n = len(k_springs)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = k_springs[i]
        if i < n - 1:
            K[i, i]     += k_springs[i + 1]
            K[i, i + 1]  = -k_springs[i + 1]
            K[i + 1, i]  = -k_springs[i + 1]
    return K


def first_mode_norm(k_springs: np.ndarray, m: float, ref_dof: int) -> np.ndarray:
    """First mode shape normalised so amplitude at ref_dof = +1."""
    K = build_K_nonuniform(k_springs)
    n = len(k_springs)
    _, vecs = eigh(K, m * np.eye(n))
    phi = vecs[:, 0]
    if phi[ref_dof] < 0:
        phi = -phi
    return phi / phi[ref_dof]


def shear_building_mode1(k_uniform: float,
                         m_uniform: float = M_CENTRAL,
                         n: int = N_STOREYS) -> np.ndarray:
    """
    First mode shape of an n-DOF uniform shear building (fixed base, free top).

    Returns normalised shape vector (length n), positive-pointing, peak = 1.
    """
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = (2.0 if i < n - 1 else 1.0) * k_uniform
        if i < n - 1:
            K[i, i + 1] = -k_uniform
            K[i + 1, i] = -k_uniform
    M = m_uniform * np.eye(n)
    _, vecs = eigh(K, M)          # ascending eigenvalue order
    phi = vecs[:, 0]              # first (lowest-frequency) mode
    phi /= np.max(np.abs(phi))    # peak = 1
    if phi[-1] < 0:
        phi = -phi                # ensure positive top-floor direction
    return phi


def phase_normalize(phi_complex: np.ndarray) -> np.ndarray:
    """
    Rotate a complex mode shape so the channel with maximum magnitude is
    real and positive, then return the real part.
    """
    idx = np.argmax(np.abs(phi_complex))
    phi_rot = phi_complex * np.exp(-1j * np.angle(phi_complex[idx]))
    phi_real = np.real(phi_rot)
    if phi_real[idx] < 0:
        phi_real = -phi_real
    return phi_real


def extract_oma_shapes(oma_data: list,
                       mode_idx: int,
                       ch_dict: dict,
                       exclude: set | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract mode-shape amplitudes at sensor floors from OMA data.

    Only Hus6 entries (not Hus8) are included.  Runs in *exclude* are skipped.
    Amplitudes at each floor = average of the two sensor channels.
    Each run is normalised so Floor 7 (index 2) = 1.0  (explicit physical reference).

    Returns
    -------
    heights   : (3,) measured floor heights in m
    mean_phi  : (3,) mean normalised shape across runs
    std_phi   : (3,) std across runs
    all_shapes: (n_runs, 3) per-run shapes for scatter / credible band
    """
    if exclude is None:
        exclude = set()
    all_shapes: list[np.ndarray] = []
    for entry in oma_data:
        lbl = entry["label"]
        if "Hus6" not in lbl or "Hus8" in lbl:
            continue
        if lbl in exclude:
            continue
        phi_r = np.array(entry["modes"][mode_idx]["mode_shape_real"])
        phi_i = np.array(entry["modes"][mode_idx]["mode_shape_imag"])
        phi   = phase_normalize(phi_r + 1j * phi_i)

        vals = np.array([
            0.5 * (phi[ch_dict[fl][0]] + phi[ch_dict[fl][1]])
            for fl in OBS_FLOORS
        ])
        # Normalise to Floor 7 (index 2 = top sensor floor)
        if abs(vals[2]) > 1e-9:
            vals /= vals[2]
        all_shapes.append(vals)

    arr = np.array(all_shapes)       # (n_runs, 3)
    return np.array(OBS_HEIGHTS), arr.mean(axis=0), arr.std(axis=0), arr


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Load NUTS results ────────────────────────────────────────────────────
    with open(DATA_PATH) as fh:
        nuts = json.load(fh)
    ps  = nuts["param_summary"]
    cfg = nuts["config"]

    rng = np.random.default_rng(RNG_SEED)
    kX_draws = np.exp(rng.normal(ps["log_k_X"]["mean"],
                                  ps["log_k_X"]["std"], N_DRAWS)) / SCALE
    kY_draws = np.exp(rng.normal(ps["log_k_Y"]["mean"],
                                  ps["log_k_Y"]["std"], N_DRAWS)) / SCALE
    kX_med = np.median(kX_draws)
    kY_med = np.median(kY_draws)

    # ── Load Level 1 (emcee) draws ───────────────────────────────────────────
    # Layout: 6 params per mode i: index = 6*i + k
    #   k=0 f_ref, k=1 alpha, k=2 beta, ...
    with open(POST1_PATH) as fh:
        d1 = json.load(fh)
    draws1 = np.array(d1["thinned_draws"])   # (N, 18)
    alpha1_l1 = draws1[:, 1]    # Mode 1
    beta1_l1  = draws1[:, 2]
    alpha2_l1 = draws1[:, 7]    # Mode 2
    beta2_l1  = draws1[:, 8]

    # ── Load OMA mode shapes ─────────────────────────────────────────────────
    with open(OMA_PATH) as fh:
        oma_data = json.load(fh)

    obs_h, phi1_mean, phi1_std, phi1_runs = extract_oma_shapes(
        oma_data, 0, CH_X, exclude=OMA_EXCLUDE)
    obs_h, phi2_mean, phi2_std, phi2_runs = extract_oma_shapes(
        oma_data, 1, CH_Y, exclude=OMA_EXCLUDE)

    # ── Sensitivity family (mode shapes are scale-invariant → use k_ref=1) ────
    # 7-DOF model: DOFs at FL2–FL8 = [3,6,…,21 m]; Story 1 spring = r×k_ref
    k_ref = 1.0
    heights_plot = np.arange(N_STOREYS + 1) * STOREY_H   # [0,3,6,…,21] m (8 nodes)

    sens_shapes = {}
    for r in [1, 10, 50]:
        ks = np.array([r * k_ref] + [k_ref] * (N_STOREYS - 1))   # 7 springs
        phi = first_mode_norm(ks, M_CENTRAL, SENSOR_DOFS_7[2])
        sens_shapes[r] = np.concatenate([[0.0], phi])

    # 6-DOF limit: Story 1 rigid → base at FL2 (3 m); DOFs at FL3–FL8 (6–21 m)
    ks6 = np.array([k_ref] * (N_STOREYS - 1))   # 6 uniform CLT springs
    phi6 = first_mode_norm(ks6, M_CENTRAL, SENSOR_DOFS_6[2])
    shape_6dof = np.concatenate([[0.0, 0.0], phi6])  # zeros at 0 m and 3 m

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92,
                        bottom=0.20, wspace=0.44)

    # ════════════════════════════════════════════════════════════════════════
    # Panel (a) — marginal posteriors k_X, k_Y
    # ════════════════════════════════════════════════════════════════════════
    ax = axes[0]
    ax.hist(kX_draws, bins=50, color=COL_KX, alpha=0.65, edgecolor="none",
            label=rf"$k_X$ — median {kX_med:.3f}")
    ax.hist(kY_draws, bins=50, color=COL_KY, alpha=0.55, edgecolor="none",
            label=rf"$k_Y$ — median {kY_med:.3f}")
    ax.axvline(kX_med, color=COL_KX, lw=1.4, ls="--")
    ax.axvline(kY_med, color=COL_KY, lw=1.4, ls="--")
    ax.set_xlabel(r"Storey stiffness  ($\times 10^8$ N/m)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, loc="upper left")
    remove_spines(ax)
    add_grid(ax)
    panel_label(ax, "(a)")

    # ════════════════════════════════════════════════════════════════════════
    # Panel (b) — forest plot: Level 1 vs Level 2 environmental parameters
    # Shows that the adimensional (emcee) and dimensional (NUTS) models
    # recover the same environmental sensitivities.
    # ════════════════════════════════════════════════════════════════════════
    ax = axes[1]

    # (label, L1 draws, L2 mean, L2 std, display scale)
    # Bottom-to-top order on y-axis
    param_rows = [
        (r"$\beta_2$",   beta2_l1,  ps["beta_2"]["mean"],  ps["beta_2"]["std"],  1e3),
        (r"$\beta_1$",   beta1_l1,  ps["beta_1"]["mean"],  ps["beta_1"]["std"],  1e3),
        (r"$\alpha_2$",  alpha2_l1, ps["alpha_2"]["mean"], ps["alpha_2"]["std"], 1e3),
        (r"$\alpha_1$",  alpha1_l1, ps["alpha_1"]["mean"], ps["alpha_1"]["std"], 1e3),
    ]

    YOFF = 0.18   # vertical offset between Level 1 / Level 2 dots

    for row, (lbl, l1_draws, l2_mean, l2_std, sc) in enumerate(param_rows):
        # Light horizontal guide
        ax.axhline(row, color="0.90", lw=0.8, zorder=0)

        # Level 1 — emcee (blue circle, 95 % CI from draws)
        med = np.median(l1_draws) * sc
        lo, hi = np.percentile(l1_draws * sc, [2.5, 97.5])
        ax.errorbar(med, row + YOFF,
                    xerr=[[med - lo], [hi - med]],
                    fmt="o", color=COL_KX, markersize=6,
                    capsize=3, capthick=1.1, lw=1.1, zorder=3)

        # Level 2 — NUTS (orange diamond, 95 % CI ≈ ±1.96 σ)
        val = l2_mean * sc
        ci  = 1.96 * l2_std * sc
        ax.errorbar(val, row - YOFF,
                    xerr=ci,
                    fmt="D", color=COL_L2, markersize=5,
                    capsize=3, capthick=1.1, lw=1.1, zorder=3)

    ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax.set_yticks(range(len(param_rows)))
    ax.set_yticklabels([r[0] for r in param_rows])
    ax.set_ylim(-0.55, len(param_rows) - 0.45)
    ax.set_xlabel(r"Parameter value  ($\times 10^{-3}$)")

    legend_handles = [
        Line2D([0], [0], marker="o", color=COL_KX, lw=0,
               markersize=6, label="Level 1 (emcee)"),
        Line2D([0], [0], marker="D", color=COL_L2, lw=0,
               markersize=5, label="Level 2 (NUTS)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    remove_spines(ax)
    add_grid(ax, axis="x")
    panel_label(ax, "(b)")

    # ════════════════════════════════════════════════════════════════════════
    # Panel (c) — Story 1 stiffness sensitivity + OMA observations
    #
    # Shows how the mode shape changes as the concrete podium (Story 1)
    # stiffness ratio r increases from 1 (uniform CLT) toward infinity
    # (equivalent to the 7-DOF model with base at 3 m).  The 6-DOF limit
    # (base at 6 m, both Stories 1+2 rigid) is shown as the best-fit model.
    # OMA shapes: Jan24 excluded, normalised to Floor 7.
    # ════════════════════════════════════════════════════════════════════════
    ax = axes[2]

    # ── Concrete Story 1 shading (FL1→FL2 = 0–3 m) ──────────────────────────
    ax.axhspan(0, 3.0, alpha=0.13, color="#8B6914", linewidth=0, zorder=0)
    ax.axhline(3.0, color="#8B6914", lw=0.9, ls=":", alpha=0.55, zorder=1)
    ax.text(1.22, 1.5, "Storey 1\n(conc.)", ha="right", va="center",
            fontsize=6.5, color="#7a5800", style="italic")

    # ── Sensitivity family: r = 1, 10, 50 ────────────────────────────────────
    r_styles = {
        1:  ("#999999", "-",  1.0, "r = 1 (uniform)"),
        10: ("#f4a582", "--", 1.3, "r = 10"),
        50: ("#a50026", "--", 1.5, "r = 50"),
    }
    for r, (col, ls, lw, lbl) in r_styles.items():
        ax.plot(sens_shapes[r], heights_plot,
                color=col, ls=ls, lw=lw, label=lbl, zorder=2)

    # ── 6-DOF limit (Story 1 rigid → base at FL2=3 m) — best fit ─────────────
    ax.plot(shape_6dof, heights_plot,
            color=COL_6, ls="-", lw=2.1,
            label=r"6-DOF limit ($r{\to}\infty$, base@FL2)", zorder=4)

    # ── OMA observations (FL7-normalised, Jan24 excluded) ────────────────────
    ax.errorbar(phi1_mean, obs_h, xerr=phi1_std,
                fmt="o", color=COL_KX, markersize=7,
                capsize=3, capthick=1.2, lw=1.2,
                label="OMA Mode 1 ($X$)", zorder=5)
    ax.errorbar(phi2_mean, obs_h, xerr=phi2_std,
                fmt="s", color=COL_OMA_Y, markersize=6,
                capsize=3, capthick=1.2, lw=1.2,
                label="OMA Mode 2 ($Y$)", zorder=5)

    # ── Ground datum and sensor-floor guides ─────────────────────────────────
    ax.axhline(0, color="0.5", lw=0.7, alpha=0.5)
    ax.plot(0, 0, "k^", markersize=7, zorder=6, clip_on=False)
    for fl_h in OBS_HEIGHTS:
        ax.axhline(fl_h, color=COL_GRID, lw=0.5, ls=":", zorder=0)

    ax.set_xlabel("Normalized amplitude (ref. FL7)")
    ax.set_ylabel("Height (m)")
    ax.set_ylim(-1.5, 23.5)
    ax.set_xlim(-0.05, 1.27)

    # ── Right axis: floor labels FL1–FL8 ─────────────────────────────────────
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    fl_ticks  = [i * STOREY_H for i in range(N_STOREYS + 1)]
    fl_labels = (["FL1\n(Base)"] +
                 [f"FL{i+2}" for i in range(N_STOREYS - 1)] +
                 ["FL8\n(Roof)"])
    ax_r.set_yticks(fl_ticks)
    ax_r.set_yticklabels(fl_labels, fontsize=FONT_SIZE - 2)
    ax_r.tick_params(length=0)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["left"].set_visible(False)

    ax.legend(frameon=False, loc="upper left", fontsize=FONT_SIZE - 2,
              handlelength=1.6, labelspacing=0.30)
    remove_spines(ax)
    add_grid(ax)
    panel_label(ax, "(c)")

    # ── Save ────────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
