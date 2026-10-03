"""
fig07_piml_composite.py
────────────────────────────────────────────────────────────────────────────────
by Andre R. Barbosa, April - October 2026

Paper Fig. 8 — PIML time-series composite, one PNG per mode.

For each mode (1, 2, 3) produces a 3-panel figure:
  Panel 0: f_obs (grey dots) | f_adim (blue) | PIML mean ± 95 % PI (red)
  Panel 1: residuals  f_obs − f_adim (blue) and f_obs − PIML (red)
  Panel 2: GP correction mean ± 95 % GP uncertainty (green)

GP hyperparameters are loaded directly from the saved JSON — no re-fitting.

Usage
─────
  cd <repository root>
  python analysis/fig07_piml_composite.py          # DPI=160
  python analysis/fig07_piml_composite.py --dpi 300

Outputs
───────
  analysis/output/Fig_Level3_PIML_Mode1.png
  analysis/output/Fig_Level3_PIML_Mode2.png
  analysis/output/Fig_Level3_PIML_Mode3.png
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.linalg import cho_factor, cho_solve

# ── project path ───────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # analysis/
ROOT = HERE.parent                              # repository root
sys.path.insert(0, str(ROOT))

from analysis.common.data_loader import (
    load_continuous_hus6,
    per_mode_view,
    get_lag_matrix,
    chronological_split,
)
from analysis.common.forward_model import adim_forward_single_mode

# ── CONSTANTS — edit these to adjust appearance ────────────────────────────────
DPI         = 160
FIGSIZE     = (14, 9)
HSPACE      = 0.10
BURST_START = pd.Timestamp("2026-04-04")
BURST_END   = pd.Timestamp("2026-04-11")
COL_OBS     = "#444444"    # observed dots
COL_ADIM    = "#1f77b4"    # physics layer (blue)
COL_PIML    = "#d62728"    # PIML (red)
COL_GP      = "#2ca02c"    # GP correction (green)
ALPHA_OBS   = 0.35
ALPHA_FILL  = 0.11
MODES       = [1, 2, 3]

# ── paths ──────────────────────────────────────────────────────────────────────
AUX_DIR = ROOT / "aux" / "level3_gp"
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib style (matches rest of paper) ──────────────────────────────────
_STYLE = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "DejaVu Sans", "Helvetica"],
    "font.style":       "italic",
    "font.size":        10,
    "axes.labelsize":   10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  9,
    "axes.titlesize":   10,
}


# ── GP helpers (pure numpy/scipy — no jax/tinygp required) ───────────────────

def _scale_X(X: np.ndarray, lp: np.ndarray) -> np.ndarray:
    """ARD scaling: shared length-scale for sin/cos diurnal channels."""
    ell = np.exp(lp[1:])                               # [ell_EMC, ell_T, ell_diel]
    return X / np.array([ell[0], ell[1], ell[2], ell[2]])[None, :]


def _matern52(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Matérn-5/2 kernel matrix between pre-scaled row-arrays X1 and X2."""
    diff   = X1[:, None, :] - X2[None, :, :]          # (n1, n2, d)
    r2     = np.sum(diff ** 2, axis=-1)                # (n1, n2)
    r      = np.sqrt(np.maximum(r2, 0.0))
    sqrt5r = np.sqrt(5.0) * r
    return (1.0 + sqrt5r + (5.0 / 3.0) * r2) * np.exp(-sqrt5r)


def predict_gp(
    lp: np.ndarray,
    X_train: np.ndarray,
    r_train: np.ndarray,
    X_pred: np.ndarray,
    noise_var: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GP posterior mean and variance at prediction points.

    Parameters
    ----------
    lp : (4,) array — [log_sigma_gp, log_ell_EMC, log_ell_T, log_ell_diel]
    X_train : (n_train, 4)
    r_train : (n_train,) — residuals f_obs - f_adim on training set
    X_pred  : (n_pred,  4)
    noise_var : float — observation variance (Hz²)

    Returns
    -------
    gp_mean : (n_pred,)
    gp_var  : (n_pred,)
    """
    sigma2   = np.exp(lp[0]) ** 2
    X_tr_sc  = _scale_X(X_train.astype(np.float64), lp)
    X_pr_sc  = _scale_X(X_pred.astype(np.float64),  lp)

    K_tr     = sigma2 * _matern52(X_tr_sc, X_tr_sc)
    np.fill_diagonal(K_tr, K_tr.diagonal() + noise_var)
    K_pr_tr  = sigma2 * _matern52(X_pr_sc, X_tr_sc)  # (n_pred, n_train)

    c, low   = cho_factor(K_tr, lower=True)
    alpha    = cho_solve((c, low), r_train.astype(np.float64))
    gp_mean  = K_pr_tr @ alpha

    # Posterior variance: σ² - diag(K_pr_tr @ K_tr⁻¹ @ K_pr_tr.T)
    v        = cho_solve((c, low), K_pr_tr.T)         # (n_train, n_pred)
    gp_var   = np.maximum(sigma2 - np.einsum("ij,ji->i", K_pr_tr, v), 0.0)

    return gp_mean, gp_var


# ── data helpers ───────────────────────────────────────────────────────────────

def downsample_to_6h(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one record per 6-hour floor slot (matches level3_gp_residual.py)."""
    df = df.copy().sort_values("DateTime").reset_index(drop=True)
    df["_slot"] = pd.to_datetime(df["DateTime"]).dt.floor("6h")
    out = df.groupby("_slot", sort=True).first().reset_index(drop=True)
    return out.drop(columns=["_slot"], errors="ignore")


def build_gp_covariates(df: pd.DataFrame) -> np.ndarray:
    """Return (N, 4) covariate array: [EMC_tau24h, T_tau24h, sin_h, cos_h]."""
    hour = pd.to_datetime(df["DateTime"]).dt.hour.to_numpy(dtype=float)
    return np.column_stack([
        df["EMC_tau24h"].to_numpy(dtype=float),
        df["T_tau24h"].to_numpy(dtype=float),
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
    ])


def compute_adim(df: pd.DataFrame, theta: dict) -> np.ndarray:
    """Adimensional physics prediction for a dataframe."""
    return adim_forward_single_mode(
        theta["f_ref"],
        theta["alpha"],
        theta["beta"],
        theta["log_tau_EMC_h"],
        theta["log_tau_T_h"],
        get_lag_matrix(df, "EMC"),
        get_lag_matrix(df, "T"),
    )


# ── per-mode pipeline ──────────────────────────────────────────────────────────

def process_mode(
    mode: int,
    df_all: pd.DataFrame,
) -> dict:
    """Run the PIML prediction pipeline for one mode.

    Returns a dict with all arrays needed for plotting.
    """
    # Load saved hyperparameters
    json_path = AUX_DIR / f"level3_Mode{mode}_result.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing {json_path}\n"
            f"Run: python analysis/level3_gp_residual.py --mode {mode} --restarts 6"
        )
    result    = json.load(open(json_path))
    theta     = result["step01_theta"]
    noise_var = result["noise_var_Hz2"]
    sigma_obs = result["sigma_obs_Hz"]
    primary_k = result["primary_k"]

    # Select the k-sweep entry for primary_k
    k_row = next(
        (x for x in result["k_sweep"] if x["k"] == primary_k),
        result["k_sweep"][0],
    )
    lp = np.array([
        k_row["log_sigma_gp"],
        np.log(k_row["ell_EMC"]),
        np.log(k_row["ell_T"]),
        np.log(k_row["ell_diel"]),
    ])

    # Per-mode data (pole-count + damping filtered)
    df_mode         = per_mode_view(df_all, mode_idx=mode)
    train, val      = chronological_split(df_mode)
    train_ds        = downsample_to_6h(train)   # GP conditioning set

    # Physics predictions
    f_adim_ds       = compute_adim(train_ds, theta)
    f_obs_ds        = train_ds[f"f{mode}"].to_numpy(float)
    r_train_ds      = f_obs_ds - f_adim_ds

    f_adim_tr       = compute_adim(train, theta)
    f_adim_va       = compute_adim(val,   theta)
    f_obs_tr        = train[f"f{mode}"].to_numpy(float)
    f_obs_va        = val[f"f{mode}"].to_numpy(float)

    # GP covariate matrices
    X_ds            = build_gp_covariates(train_ds)
    X_tr_full       = build_gp_covariates(train)
    X_va            = build_gp_covariates(val)

    # GP predictions — condition on downsampled train, predict everywhere
    gm_tr, gv_tr = predict_gp(lp, X_ds, r_train_ds, X_tr_full, noise_var)
    gm_va, gv_va = predict_gp(lp, X_ds, r_train_ds, X_va,      noise_var)

    # Concatenate train + val for display
    dt_all      = pd.to_datetime(
        pd.concat([train["DateTime"], val["DateTime"]], ignore_index=True)
    )
    f_obs_all   = np.concatenate([f_obs_tr,   f_obs_va])
    f_adim_all  = np.concatenate([f_adim_tr,  f_adim_va])
    gm_all      = np.concatenate([gm_tr,      gm_va])
    gv_all      = np.concatenate([gv_tr,      gv_va])
    piml_all    = f_adim_all + gm_all
    ts_all      = np.sqrt(np.maximum(gv_all + noise_var, 1e-16))   # total pred. std
    ts_gp_all   = np.sqrt(np.maximum(gv_all, 1e-16))               # GP-only std

    split_dt = pd.to_datetime(val["DateTime"].iloc[0])

    return dict(
        mode       = mode,
        dt_all     = dt_all,
        f_obs_all  = f_obs_all,
        f_adim_all = f_adim_all,
        piml_all   = piml_all,
        gm_all     = gm_all,
        gv_all     = gv_all,
        ts_all     = ts_all,
        ts_gp_all  = ts_gp_all,
        sigma_obs  = sigma_obs,
        noise_var  = noise_var,
        split_dt   = split_dt,
    )


# ── plotting ───────────────────────────────────────────────────────────────────

def make_figure(data: dict) -> plt.Figure:
    """Create the 3-panel figure for one mode."""
    mode       = data["mode"]
    dt_all     = data["dt_all"]
    f_obs_all  = data["f_obs_all"]
    f_adim_all = data["f_adim_all"]
    piml_all   = data["piml_all"]
    gm_all     = data["gm_all"]
    ts_all     = data["ts_all"]
    ts_gp_all  = data["ts_gp_all"]
    sigma_obs  = data["sigma_obs"]
    split_dt   = data["split_dt"]

    res_adim = f_obs_all - f_adim_all
    res_piml = f_obs_all - piml_all

    fig, axes = plt.subplots(
        3, 1, figsize=FIGSIZE, sharex=True,
        gridspec_kw={"hspace": HSPACE},
    )

    # ── Panel 0: frequency time series ────────────────────────────────────────
    ax = axes[0]
    ax.scatter(dt_all, f_obs_all,
               s=3.5, c=COL_OBS, alpha=ALPHA_OBS, zorder=2,
               label=r"$f_{\mathrm{obs}}$")
    ax.plot(dt_all, f_adim_all,
            lw=0.9, c=COL_ADIM, alpha=0.80, zorder=3,
            label=r"$\hat{f}_{\mathrm{phys}}$ (Level 1)")
    ax.plot(dt_all, piml_all,
            lw=1.2, c=COL_PIML, alpha=0.92, zorder=4,
            label="PIML mean")
    ax.fill_between(
        dt_all,
        piml_all - 1.96 * ts_all,
        piml_all + 1.96 * ts_all,
        color=COL_PIML, alpha=ALPHA_FILL, zorder=1,
        label=r"PIML $\pm 1.96\,\sigma_{\mathrm{total}}$",
    )
    ax.axvline(split_dt, c="k", lw=0.9, ls="--", alpha=0.50, zorder=5)
    ax.axvspan(BURST_START, BURST_END, color="#f5a623", alpha=0.16,
               label="wind burst (val)", zorder=0)
    ax.set_ylabel(rf"$f_{mode}$ (Hz)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.16)

    # ── Panel 1: residuals ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(dt_all, res_adim,
               s=2.5, c=COL_ADIM, alpha=ALPHA_OBS, zorder=2,
               label=r"$f_{\mathrm{obs}} - \hat{f}_{\mathrm{adim}}$")
    ax.scatter(dt_all, res_piml,
               s=2.5, c=COL_PIML, alpha=ALPHA_OBS, zorder=3,
               label=r"$f_{\mathrm{obs}} - f_{\mathrm{PIML}}$")
    ax.axhline(+sigma_obs, c="k", lw=0.8, ls=":", alpha=0.55)
    ax.axhline(-sigma_obs, c="k", lw=0.8, ls=":", alpha=0.55,
               label=rf"$\pm\sigma_{{\mathrm{{obs}}}}={sigma_obs:.4f}$ Hz")
    ax.axhline(0.0, c="k", lw=0.6, alpha=0.30)
    ax.axvline(split_dt, c="k", lw=0.9, ls="--", alpha=0.50)
    ax.axvspan(BURST_START, BURST_END, color="#f5a623", alpha=0.16, zorder=0)
    ax.set_ylabel("Residual (Hz)")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.16)

    # ── Panel 2: GP correction ─────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(dt_all, gm_all,
            lw=1.1, c=COL_GP, alpha=0.90, zorder=3,
            label="GP correction mean")
    ax.fill_between(
        dt_all,
        gm_all - 1.96 * ts_gp_all,
        gm_all + 1.96 * ts_gp_all,
        color=COL_GP, alpha=ALPHA_FILL, zorder=1,
        label=r"GP $\pm 1.96\,\sigma_{\mathrm{GP}}$",
    )
    ax.axhline(0.0, c="k", lw=0.6, alpha=0.30)
    ax.axvline(split_dt, c="k", lw=0.9, ls="--", alpha=0.50)
    ax.axvspan(BURST_START, BURST_END, color="#f5a623", alpha=0.16, zorder=0)
    ax.set_ylabel("GP correction (Hz)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.16)

    # ── shared x-axis formatting ───────────────────────────────────────────────
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=20, ha="right")

    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main(dpi: int = DPI) -> None:
    plt.rcParams.update(_STYLE)

    print("Loading dataset ...", flush=True)
    df_all = load_continuous_hus6()

    for mode in MODES:
        print(f"Mode {mode} ...", flush=True)
        data     = process_mode(mode, df_all)
        fig      = make_figure(data)
        out_path = OUT_DIR / f"Fig_Level3_PIML_Mode{mode}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {out_path}", flush=True)

    print("Done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Fig 7 — PIML time-series composite (one PNG per mode)."
    )
    parser.add_argument(
        "--dpi", type=int, default=DPI,
        help=f"Output DPI (default {DPI}; use 300 for print quality).",
    )
    args = parser.parse_args()
    main(dpi=args.dpi)
