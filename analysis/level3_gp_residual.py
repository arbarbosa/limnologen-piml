"""
level3_gp_residual.py  —  Level 3: physics-informed ML (GP residual layer)
==========================================================================
by Andre R. Barbosa, April - October 2026

PIML architecture (per mode):

  f_obs(t) = f̂_adim(t; θ*) + g(x_t) + ε
  g   ~ GP(0,  σ²_GP · Matérn-5/2(ARD))
  ε   ~ Normal(0, σ²_obs)          [from Level 1 posterior]

where θ* = Level 1 posterior medians (empirical Bayes).

GP covariates
─────────────
  x_t = [EMC_tau24h, T_tau24h, sin(2π·h/24), cos(2π·h/24)]

Two-knob bounding strategy
──────────────────────────
  Knob 1 — length-scale bounds (hard box during optimisation):
    ℓ_EMC  ∈ [1, 20] % EMC
    ℓ_T    ∈ [1, 30] °C
    ℓ_diel ∈ [0.05, 5]

  Knob 2 — variance cap:
    σ²_GP ≤ k · σ²_obs;   k ∈ {0.25, 0.5, 1, 2, 3, 5, ∞};  primary k = 1

Usage
─────
  cd <repository root>
  python analysis/level3_gp_residual.py --mode 1
  python analysis/level3_gp_residual.py --mode 2
  python analysis/level3_gp_residual.py --mode 3

Outputs  (all in output/ and aux/level3_gp/)
────────────────────────────────────────────
  Fig_Level3_PIML_Mode{n}.png          – 3-panel PIML time-series
  Fig_Level3_PIML_Mode{n}_settings.txt – figure settings for journal reporting
  Fig_Level3_kSweep_Mode{n}.png        – CRPS / coverage / variance vs k
  Fig_Level3_kSweep_Mode{n}_settings.txt
  aux/level3_gp/level3_Mode{n}_result.json
  aux/level3_gp/level3_Mode{n}_sanity.md
  aux/level3_gp/level3_Mode{n}_run_log.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import minimize
from scipy.stats import norm as stats_norm

import jax
import jax.numpy as jnp
import tinygp

jax.config.update("jax_enable_x64", True)

# ── project paths ──────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from analysis.common.data_loader import (
    load_continuous_hus6, per_mode_view, get_lag_matrix, chronological_split,
)
from analysis.common.forward_model import adim_forward_single_mode

AUX_DIR    = ROOT / "aux" / "level3_gp"
OUTPUT_DIR = ROOT / "output"
AUX_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

STEP01_PATH = ROOT / "aux" / "level1_adim" / "v3_per_mode_tau" / "posterior.json"

# ── GP / optimisation config ───────────────────────────────────────────────────
LL_EMC_LB,  LL_EMC_UB  = np.log(1.0),  np.log(20.0)
LL_T_LB,    LL_T_UB    = np.log(1.0),  np.log(30.0)
LL_DIEL_LB, LL_DIEL_UB = np.log(0.05), np.log(5.0)
K_VALUES  = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
K_PRIMARY = 1.0
N_RESTARTS = 12   # number of multi-start optimizations
RNG_SEED   = 20260608
DPI        = 160

# Burst-recording configuration ─────────────────────────────────────────────
# Training bursts (Aug 5–9 and Oct 4–6 2025) are down-sampled to the nominal
# 6 h grid so those 3–7 day windows don't receive 6× the GP-fitting weight.
# The Apr 2026 validation burst (windstorm) is kept in the val set but metrics
# are reported separately so we can see how much it inflates CRPS / coverage.
VAL_BURST_PERIODS = [("2026-04-04", "2026-04-11")]  # end is exclusive midnight

GP_COVARIATES = ["EMC_tau24h", "T_tau24h", "sin(2π·h/24)", "cos(2π·h/24)"]

# ── figure style ──────────────────────────────────────────────────────────────
_STYLE = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "DejaVu Sans", "Helvetica"],
    "font.style":       "italic",
    "font.size":        10,
    "axes.labelsize":   10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "axes.titlesize":   10,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Data helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_step01_medians(mode_idx: int) -> dict:
    with open(STEP01_PATH) as f:
        d = json.load(f)
    s = d["summary"]
    i = mode_idx + 1
    key_emc = f"log_tau_EMC_h_{i}" if f"log_tau_EMC_h_{i}" in s else "log_tau_EMC_h"
    key_t   = f"log_tau_T_h_{i}" if f"log_tau_T_h_{i}" in s else "log_tau_T_h"
    return {
        "f_ref":         s[f"f_ref_{i}"]["median"],
        "alpha":         s[f"alpha_{i}"]["median"],
        "beta":          s[f"beta_{i}"]["median"],
        "log_sigma_obs": s[f"log_sigma_obs_{i}"]["median"],
        "log_tau_EMC_h": s[key_emc]["median"],
        "log_tau_T_h":   s[key_t]["median"],
    }


def build_gp_covariates(df: pd.DataFrame) -> np.ndarray:
    hour = pd.to_datetime(df["DateTime"]).dt.hour.to_numpy(dtype=float)
    return np.column_stack([
        df["EMC_tau24h"].to_numpy(dtype=float),
        df["T_tau24h"].to_numpy(dtype=float),
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
    ])


def downsample_to_6h(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one record per 6-hour slot (0 / 6 / 12 / 18 h, floor-aligned).

    Burst events produce clusters of 1 h observations.  Retaining all of them
    gives those 3–7 day windows 6× the influence of a normal week during GP
    marginal-likelihood optimisation.  Down-sampling to the nominal 6 h cadence
    restores uniform temporal weight across the training period.
    """
    df = df.copy().sort_values("DateTime").reset_index(drop=True)
    df["_slot"] = pd.to_datetime(df["DateTime"]).dt.floor("6h")
    out = df.groupby("_slot", sort=True).first().reset_index(drop=True)
    return out.drop(columns=["_slot"], errors="ignore")


def label_val_bursts(val: pd.DataFrame) -> np.ndarray:
    """Return boolean mask (length N_val): True for records inside a burst period.

    Burst records in the validation set have inflated OMA scatter
    (windstorm-induced noise floor rise).  Splitting metrics by burst / no-burst
    lets us assess how much the event contaminates CRPS and coverage.
    """
    dt   = pd.to_datetime(val["DateTime"])
    mask = np.zeros(len(val), dtype=bool)
    for start, end in VAL_BURST_PERIODS:
        mask |= (dt >= start) & (dt < pd.Timestamp(end))
    return mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. GP model  (log_params = [log_σ_GP, log_ℓ_EMC, log_ℓ_T, log_ℓ_diel])
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _scale_X(X: jnp.ndarray, lp: jnp.ndarray) -> jnp.ndarray:
    ell = jnp.exp(lp[1:])
    return X / jnp.array([ell[0], ell[1], ell[2], ell[2]])[None, :]


def make_nlml_fn(X_jax, y_jax, noise_var):
    @jax.jit
    def _nlml(lp):
        gp = tinygp.GaussianProcess(
            jnp.exp(lp[0]) ** 2 * tinygp.kernels.Matern52(),
            _scale_X(X_jax, lp), diag=noise_var)
        return -gp.log_probability(y_jax)
    return jax.jit(jax.value_and_grad(_nlml))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Optimiser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fit_gp(X_train, r_train, noise_var, log_sg_ub, rng, n_restarts, pfx="", x_warm=None):
    r_std = float(np.std(r_train)) + 1e-8
    bounds = [
        (np.log(r_std * 0.01), log_sg_ub),
        (LL_EMC_LB, LL_EMC_UB),
        (LL_T_LB,   LL_T_UB),
        (LL_DIEL_LB, LL_DIEL_UB),
    ]
    X_j = jnp.array(X_train, dtype=jnp.float64)
    y_j = jnp.array(r_train, dtype=jnp.float64)
    fn  = make_nlml_fn(X_j, y_j, noise_var)

    best_p, best_f = None, np.inf
    # Warm start: the best solution from the previous (smaller) k is
    # feasible here, so starting from it guarantees the optimum cannot get worse as k grows.
    if x_warm is not None:
        xw = np.clip(np.asarray(x_warm, float), [bb[0] for bb in bounds],
                     [bb[1] if bb[1] is not None else np.inf for bb in bounds])
        res = minimize(
            lambda p: (lambda v, g: (float(v), np.array(g, dtype=np.float64)))(
                *fn(jnp.array(p, dtype=jnp.float64))),
            xw, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
        )
        best_f, best_p = res.fun, res.x.copy()
        print(f"  {pfx}warm start: nlml={res.fun:.4f}", flush=True)
    for k in range(n_restarts):
        lo = max(np.log(r_std * 0.01), np.log(1e-8))
        hi = log_sg_ub if log_sg_ub is not None else np.log(r_std * 5)
        hi = max(hi, lo + 0.01)
        x0 = np.array([
            rng.uniform(lo, hi),
            rng.uniform(LL_EMC_LB, LL_EMC_UB),
            rng.uniform(LL_T_LB,   LL_T_UB),
            rng.uniform(LL_DIEL_LB, LL_DIEL_UB),
        ])
        res = minimize(
            lambda p: (lambda v, g: (float(v), np.array(g, dtype=np.float64)))(
                *fn(jnp.array(p, dtype=jnp.float64))),
            x0, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-8},
        )
        if res.fun < best_f:
            best_f, best_p = res.fun, res.x.copy()
        print(f"  {pfx}restart {k+1}/{n_restarts}: "
              f"nlml={res.fun:.4f}  σ_GP={np.exp(res.x[0]):.5f}  "
              f"ℓ=[{np.exp(res.x[1]):.2f},{np.exp(res.x[2]):.2f},{np.exp(res.x[3]):.3f}]",
              flush=True)
    return best_p, best_f


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Prediction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict_gp(lp, X_train, r_train, X_pred, noise_var):
    lp_j    = jnp.array(lp, dtype=jnp.float64)
    X_tr_sc = _scale_X(jnp.array(X_train, dtype=jnp.float64), lp_j)
    X_pr_sc = _scale_X(jnp.array(X_pred,  dtype=jnp.float64), lp_j)
    gp      = tinygp.GaussianProcess(
        jnp.exp(lp_j[0]) ** 2 * tinygp.kernels.Matern52(),
        X_tr_sc, diag=noise_var)
    _, cond = gp.condition(jnp.array(r_train, dtype=jnp.float64), X_pr_sc)
    return np.array(cond.loc), np.array(cond.variance)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Diagnostics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def crps_gaussian(mu, sigma, y):
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats_norm.cdf(z) - 1) +
                    2 * stats_norm.pdf(z) - 1 / np.sqrt(np.pi))


def compute_diag(f_adim, gm, gv, f_obs, noise_var, label=""):
    pm   = f_adim + gm
    ts   = np.sqrt(np.maximum(gv + noise_var, 1e-16))
    cov  = float(np.mean(np.abs(f_obs - pm) <= 1.96 * ts))
    d    = {
        "n":               int(len(f_obs)),
        "CRPS_mean":       float(np.mean(crps_gaussian(pm, ts, f_obs))),
        "CRPS_median":     float(np.median(crps_gaussian(pm, ts, f_obs))),
        "coverage_95":     cov,
        "RMSE_adim_Hz":    float(np.sqrt(np.mean((f_obs - f_adim) ** 2))),
        "RMSE_piml_Hz":    float(np.sqrt(np.mean((f_obs - pm) ** 2))),
        "var_ratio_gp_obs": float(np.mean(gv) / noise_var),
    }
    if label:
        print(f"  [{label}]  n={d['n']}  CRPS={d['CRPS_mean']:.5f}  "
              f"cov95={cov:.3f}  RMSE {d['RMSE_adim_Hz']:.5f}→{d['RMSE_piml_Hz']:.5f} Hz",
              flush=True)
    return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Settings file
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def write_settings(out_path: Path, fig_name: str, mode: int,
                   theta: dict, sigma_obs: float, noise_var: float,
                   train: pd.DataFrame, val: pd.DataFrame,
                   n_restarts: int, pri_rec: dict, label: str = "",
                   n_train_raw: int = 0) -> None:
    sg      = pri_rec["sigma_gp_Hz"]
    tr_d    = pri_rec["train"]
    va_d    = pri_rec["val"]
    lines   = [
        f"FIGURE: {fig_name}",
        f"LABEL:  {label}" if label else "",
        f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"SCRIPT: analysis/level3_gp_residual.py --mode {mode}",
        "",
        "=== DATA ===",
        f"Dataset: Limnologen_continuous_OMA_aligned.csv",
        f"Building: Limnologen Hus 6",
        f"Mode: {mode}",
        f"Pole-count filter: >= 5",
        f"N_train: {len(train)} records"
            + (f"  (down-sampled from {n_train_raw} raw; burst 1 h → 6 h)" if n_train_raw > len(train) else ""),
        f"N_val:   {len(val)} records  ({VAL_BURST_PERIODS[0][0]} – {VAL_BURST_PERIODS[0][1]} = burst period)",
        f"Train period: {train['DateTime'].iloc[0]}  to  {train['DateTime'].iloc[-1]}",
        f"Val period:   {val['DateTime'].iloc[0]}  to  {val['DateTime'].iloc[-1]}",
        "",
        "=== PARAMETRIC MODEL — Level 1 posterior medians ===",
        f"f_ref_{mode}:         {theta['f_ref']:.5f} Hz",
        f"alpha_{mode}:         {theta['alpha']:.6f}  [per %EMC]",
        f"beta_{mode}:          {theta['beta']:.6f}  [per °C]",
        f"sigma_obs_{mode}:     {sigma_obs:.5f} Hz  (sigma^2 = {noise_var:.4e} Hz^2)",
        f"tau_EMC_{mode}:       {np.exp(theta['log_tau_EMC_h']):.1f} h  "
            f"({np.exp(theta['log_tau_EMC_h'])/24:.1f} d)",
        f"tau_T_{mode}:         {np.exp(theta['log_tau_T_h']):.1f} h  "
            f"({np.exp(theta['log_tau_T_h'])/24:.1f} d)",
        "",
        "=== GP CONFIGURATION ===",
        "Covariates: EMC_tau24h [%], T_tau24h [degC], sin(2pi*h/24), cos(2pi*h/24)",
        "Kernel: Matern-5/2, ARD (one length scale per covariate; sin/cos share one scale)",
        "Parameterisation: log_sigma_GP, log_ell_EMC, log_ell_T, log_ell_diel",
        f"Length-scale bounds:",
        f"  ell_EMC  in [exp({LL_EMC_LB:.3f}), exp({LL_EMC_UB:.3f})] "
            f"= [{np.exp(LL_EMC_LB):.2f}, {np.exp(LL_EMC_UB):.2f}] %EMC",
        f"  ell_T    in [exp({LL_T_LB:.3f}), exp({LL_T_UB:.3f})] "
            f"= [{np.exp(LL_T_LB):.2f}, {np.exp(LL_T_UB):.2f}] degC",
        f"  ell_diel in [exp({LL_DIEL_LB:.3f}), exp({LL_DIEL_UB:.3f})] "
            f"= [{np.exp(LL_DIEL_LB):.3f}, {np.exp(LL_DIEL_UB):.2f}]",
        f"Variance cap: sigma_GP^2 <= k * sigma_obs^2",
        f"  k_primary = {K_PRIMARY}  (reported hyperparameters below)",
        f"  k_sweep   = {[v if not np.isinf(v) else 'inf' for v in K_VALUES]}",
        f"Observation noise: sigma_obs fixed at Level 1 posterior median (empirical Bayes)",
        f"Optimizer: scipy.optimize.minimize  method=L-BFGS-B",
        f"  max_iter=400  ftol=1e-12  gtol=1e-8",
        f"Random restarts: {n_restarts}",
        f"RNG seed: {RNG_SEED}",
        "",
        f"=== OPTIMISED HYPERPARAMETERS  (k = {K_PRIMARY}) ===",
        f"sigma_GP:  {sg:.5f} Hz   (sigma_GP/sigma_obs = {sg/sigma_obs:.4f})",
        f"ell_EMC:   {pri_rec['ell_EMC']:.4f} %EMC",
        f"ell_T:     {pri_rec['ell_T']:.4f} degC",
        f"ell_diel:  {pri_rec['ell_diel']:.5f}",
        f"neg_log_ML:{pri_rec['neg_log_ml']:.4f}",
        "",
        f"=== DIAGNOSTICS  (k = {K_PRIMARY}) ===",
        f"Train  CRPS (mean):    {tr_d['CRPS_mean']:.5f} Hz",
        f"Train  coverage 95%:   {tr_d['coverage_95']:.4f}",
        f"Train  RMSE adim:      {tr_d['RMSE_adim_Hz']:.5f} Hz",
        f"Train  RMSE PIML:      {tr_d['RMSE_piml_Hz']:.5f} Hz",
        f"Val    CRPS (mean):    {va_d['CRPS_mean']:.5f} Hz",
        f"Val    coverage 95%:   {va_d['coverage_95']:.4f}",
        f"Val    RMSE adim:      {va_d['RMSE_adim_Hz']:.5f} Hz",
        f"Val    RMSE PIML:      {va_d['RMSE_piml_Hz']:.5f} Hz",
        f"GP/obs variance ratio: {va_d['var_ratio_gp_obs']:.5f}  "
            f"(sigma_GP^2 / sigma_obs^2)",
        "",
        f"=== VAL BURST-SPLIT DIAGNOSTICS  (k = {K_PRIMARY}) ===",
        f"Burst period: {VAL_BURST_PERIODS}",
        f"Val (no burst) n:        {pri_rec.get('val_noburst', {}).get('n', 'n/a')}",
        f"Val (no burst) CRPS:     {pri_rec.get('val_noburst', {}).get('CRPS_mean', float('nan')):.5f} Hz",
        f"Val (no burst) cov 95%:  {pri_rec.get('val_noburst', {}).get('coverage_95', float('nan')):.4f}",
        f"Val (no burst) RMSE adim:{pri_rec.get('val_noburst', {}).get('RMSE_adim_Hz', float('nan')):.5f} Hz",
        f"Val (no burst) RMSE PIML:{pri_rec.get('val_noburst', {}).get('RMSE_piml_Hz', float('nan')):.5f} Hz",
        f"Val (burst)    n:        {pri_rec.get('val_burst', {}).get('n', 'n/a')}",
        f"Val (burst)    CRPS:     {pri_rec.get('val_burst', {}).get('CRPS_mean', float('nan')):.5f} Hz",
        f"Val (burst)    cov 95%:  {pri_rec.get('val_burst', {}).get('coverage_95', float('nan')):.4f}",
        f"Val (burst)    RMSE adim:{pri_rec.get('val_burst', {}).get('RMSE_adim_Hz', float('nan')):.5f} Hz",
        f"Val (burst)    RMSE PIML:{pri_rec.get('val_burst', {}).get('RMSE_piml_Hz', float('nan')):.5f} Hz",
        "",
        "=== FIGURE STYLE ===",
        "Font: Arial Italic, 10pt",
        "No figure title",
        f"DPI: {DPI}",
        "Colors: obs=#444444, adim=#1f77b4, PIML=#d62728, GP correction=#2ca02c",
        "CI: +/- 1.96 * sqrt(sigma_GP^2 + sigma_obs^2)",
        "Train/val split: vertical dashed black line",
        "Uncertainty fill: alpha=0.11",
    ]
    out_path.write_text("\n".join(l for l in lines if l is not None))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_piml(dates_tr, f_obs_tr, f_adim_tr, gm_tr, gv_tr,
              dates_va, f_obs_va, f_adim_va, gm_va, gv_va,
              noise_var, k, mode, out_path):
    plt.rcParams.update(_STYLE)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"hspace": 0.10})

    dates_all  = pd.concat([dates_tr, dates_va], ignore_index=True)
    f_obs_a    = np.concatenate([f_obs_tr,  f_obs_va])
    f_adim_a   = np.concatenate([f_adim_tr, f_adim_va])
    gm_a       = np.concatenate([gm_tr, gm_va])
    gv_a       = np.concatenate([gv_tr, gv_va])
    piml_a     = f_adim_a + gm_a
    ts_a       = np.sqrt(np.maximum(gv_a + noise_var, 1e-16))
    dt_a       = pd.to_datetime(dates_all)
    split_dt   = pd.to_datetime(dates_va.iloc[0])
    sigma_obs  = float(np.sqrt(noise_var))

    # Panel 0: f_obs + f_adim + PIML
    ax = axes[0]
    ax.scatter(dt_a, f_obs_a, s=3.5, c="#444444", alpha=0.35, zorder=2,
               label=r"$f_\mathrm{obs}$")
    ax.plot(dt_a, f_adim_a, lw=0.9, c="#1f77b4", alpha=0.80,
            label=r"$\hat{f}_\mathrm{adim}$ (Level 1)")
    ax.plot(dt_a, piml_a, lw=1.3, c="#d62728", alpha=0.90,
            label="PIML mean")
    ax.fill_between(dt_a, piml_a - 1.96*ts_a, piml_a + 1.96*ts_a,
                    color="#d62728", alpha=0.11,
                    label=r"PIML $\pm1.96\sigma_\mathrm{total}$")
    ax.axvline(split_dt, c="k", lw=1.0, ls="--", alpha=0.50)
    ax.set_ylabel(f"$f_{mode}$  (Hz)")
    ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.18)

    # Panel 1: residuals
    ax = axes[1]
    ax.scatter(dt_a, f_obs_a - f_adim_a, s=3, c="#1f77b4", alpha=0.30,
               label="adim residual")
    ax.scatter(dt_a, f_obs_a - piml_a,   s=3, c="#d62728", alpha=0.30,
               label="PIML residual")
    for m, ls in [(1, "--"), (2, ":")]:
        ax.axhline( m*sigma_obs, c="gray", lw=0.7, ls=ls, alpha=0.60)
        ax.axhline(-m*sigma_obs, c="gray", lw=0.7, ls=ls, alpha=0.60)
    ax.axhline(0, c="k", lw=0.8, alpha=0.40)
    ax.axvline(split_dt, c="k", lw=1.0, ls="--", alpha=0.50)
    ax.set_ylabel("Residual  (Hz)")
    ax.text(0.01, 0.94, f"dashed: ±{sigma_obs:.4f} Hz ($\\sigma_\\mathrm{{obs}}$)",
            transform=ax.transAxes, fontsize=8, color="gray", va="top",
            style="italic")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.18)

    # Panel 2: GP correction
    ax = axes[2]
    gp_std_a = np.sqrt(np.maximum(gv_a, 0.0))
    ax.plot(dt_a, gm_a, lw=1.1, c="#2ca02c", label="GP correction mean")
    ax.fill_between(dt_a, gm_a - 1.96*gp_std_a, gm_a + 1.96*gp_std_a,
                    color="#2ca02c", alpha=0.17,
                    label=r"GP $\pm1.96\sigma_\mathrm{GP}$")
    ax.axhline(0, c="k", lw=0.8, alpha=0.40)
    ax.axvline(split_dt, c="k", lw=1.0, ls="--", alpha=0.50)
    ax.set_ylabel("GP correction  (Hz)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.18)

    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}", flush=True)


def plot_ksweep(k_vals, train_diags, val_diags, val_noburst_diags, mode, out_path):
    plt.rcParams.update(_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    k_labels = [str(v) if not np.isinf(v) else "∞" for v in k_vals]
    x = np.arange(len(k_vals))
    k1 = next((i for i, v in enumerate(k_vals) if v == K_PRIMARY), None)

    for ax, key, ylabel in [
        (axes[0], "CRPS_mean",        "CRPS (Hz)"),
        (axes[1], "coverage_95",      "Coverage"),
        (axes[2], "var_ratio_gp_obs", r"$\sigma^2_\mathrm{GP}\ /\ \sigma^2_\mathrm{obs}$"),
    ]:
        ax.plot(x, [d[key] for d in train_diags],      "o-",  c="#1f77b4",
                lw=1.5, ms=5, label="train")
        ax.plot(x, [d[key] for d in val_diags],         "s--", c="#d62728",
                lw=1.5, ms=5, label="val (all)")
        ax.plot(x, [d[key] for d in val_noburst_diags], "^:",  c="#ff7f0e",
                lw=1.5, ms=5, label="val (no burst)")
        ax.set_xticks(x)
        ax.set_xticklabels(k_labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=9)
        if k1 is not None:
            ax.axvline(k1, c="gray", lw=1.0, ls=":", alpha=0.60)
        if key == "coverage_95":
            ax.axhline(0.95, c="gray", lw=0.8, ls="--", alpha=0.60)
            ax.set_ylim(0.78, 1.02)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}", flush=True)


def write_ksweep_settings(out_path, fig_name, mode, k_vals,
                          sweep_results, n_restarts):
    lines = [
        f"FIGURE: {fig_name}",
        f"GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"SCRIPT: analysis/level3_gp_residual.py --mode {mode}",
        "",
        "=== PURPOSE ===",
        "Knob-2 (variance-cap) sweep: sigma_GP^2 <= k * sigma_obs^2",
        "Shows how CRPS, 95% coverage, and GP/obs variance ratio vary with k.",
        "",
        f"k values: {[v if not np.isinf(v) else 'inf' for v in k_vals]}",
        f"k_primary (highlighted): {K_PRIMARY}",
        f"Random restarts per k: {n_restarts}",
        f"RNG seed: {RNG_SEED}",
        "",
        "=== PER-k RESULTS ===",
        "(CRPS_nb / cov95_nb = val excluding burst period; burst = windstorm event)",
        f"{'k':>6}  {'sigma_GP':>10}  {'ell_EMC':>8}  {'ell_T':>8}  "
        f"{'ell_diel':>9}  {'CRPS_all':>10}  {'cov95_all':>10}  "
        f"{'CRPS_nb':>10}  {'cov95_nb':>10}",
    ]
    for r in sweep_results:
        k_s = str(r['k']) if r['k'] != 'inf' else '∞'
        nb  = r.get('val_noburst') or r['val']
        lines.append(
            f"{k_s:>6}  {r['sigma_gp_Hz']:>10.5f}  {r['ell_EMC']:>8.3f}  "
            f"{r['ell_T']:>8.3f}  {r['ell_diel']:>9.4f}  "
            f"{r['val']['CRPS_mean']:>10.5f}  {r['val']['coverage_95']:>10.4f}  "
            f"{nb['CRPS_mean']:>10.5f}  {nb['coverage_95']:>10.4f}"
        )
    lines += [
        "",
        "=== FIGURE STYLE ===",
        "Font: Arial Italic, 10pt",
        "No figure title",
        f"DPI: {DPI}",
        "Colors: train=#1f77b4, val=#d62728",
        "k_primary marker: vertical dotted gray line",
        "Coverage reference: horizontal dashed gray line at 0.95",
    ]
    out_path.write_text("\n".join(lines))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="Level 3: GP residual layer")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3],
                        help="Mode number (1, 2, or 3)")
    parser.add_argument("--restarts", type=int, default=N_RESTARTS,
                        help=f"Optimizer restarts per k (default {N_RESTARTS})")
    args = parser.parse_args()

    mode     = args.mode
    mode_idx = mode - 1
    n_rest   = args.restarts

    # Tee stdout to log
    log_buf     = StringIO()
    orig_stdout = sys.stdout

    class _Tee:
        def __init__(self, *s): self.streams = s
        def write(self, d):
            for s in self.streams: s.write(d)
        def flush(self):
            for s in self.streams: s.flush()

    sys.stdout = _Tee(orig_stdout, log_buf)

    t0  = time.time()
    rng = np.random.default_rng(RNG_SEED)

    print("=" * 70)
    print(f"Level 3 — GP residual layer  (Mode {mode})")
    print("=" * 70)

    # ── Level 1 posterior ──────────────────────────────────────────────────
    theta     = load_step01_medians(mode_idx)
    sigma_obs = float(np.exp(theta["log_sigma_obs"]))
    noise_var = float(sigma_obs ** 2)
    print(f"\nLevel 1 posteriors (Mode {mode}):")
    print(f"  f_ref={theta['f_ref']:.5f} Hz  α={theta['alpha']:.6f}  β={theta['beta']:.6f}")
    print(f"  σ_obs={sigma_obs:.5f} Hz  τ_EMC={np.exp(theta['log_tau_EMC_h']):.1f}h  "
          f"τ_T={np.exp(theta['log_tau_T_h']):.1f}h")

    # ── Data ──────────────────────────────────────────────────────────────
    df_all   = load_continuous_hus6()
    df_mode  = per_mode_view(df_all, mode_idx=mode)
    train, val = chronological_split(df_mode)

    # Downsample training bursts to the nominal 6 h cadence
    n_train_raw    = len(train)
    train          = downsample_to_6h(train)
    burst_mask_val = label_val_bursts(val)
    print(f"\nMode {mode}: total={len(df_mode):,}  "
          f"train={n_train_raw:,} → {len(train):,} (6 h down-sample, "
          f"{n_train_raw - len(train)} burst records removed)  "
          f"val={len(val):,} ({int(burst_mask_val.sum())} burst records)")

    def _adim(df):
        return adim_forward_single_mode(
            f_ref=theta["f_ref"], alpha=theta["alpha"], beta=theta["beta"],
            log_tau_EMC_h=theta["log_tau_EMC_h"], log_tau_T_h=theta["log_tau_T_h"],
            emc_lag_matrix=get_lag_matrix(df, "EMC"),
            t_lag_matrix=get_lag_matrix(df, "T"))

    f_adim_tr = _adim(train);  f_adim_va = _adim(val)
    f_obs_tr  = train[f"f{mode}"].to_numpy(float)
    f_obs_va  = val[f"f{mode}"].to_numpy(float)
    r_train   = f_obs_tr - f_adim_tr

    print(f"Residuals (train): mean={r_train.mean():.5f}  std={r_train.std():.5f} Hz  "
          f"(std/σ_obs={r_train.std()/sigma_obs:.3f})")

    X_tr = build_gp_covariates(train)
    X_va = build_gp_covariates(val)

    # Warm-up JIT
    X_j = jnp.array(X_tr, dtype=jnp.float64)
    y_j = jnp.array(r_train, dtype=jnp.float64)
    _fn = make_nlml_fn(X_j, y_j, noise_var)
    _fn(jnp.array([np.log(sigma_obs * 0.5), np.log(5.), np.log(5.), np.log(1.)]))

    # ── k-sweep ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}\nKnob-2 sweep (Mode {mode})\n{'─'*60}")
    sweep  = []
    pri_data = {}

    prev_best = None
    for k in K_VALUES:
        log_sg_ub = (np.log(sigma_obs) + 0.5 * np.log(k)
                     if not np.isinf(k) else None)
        k_s = f"k={k}" if not np.isinf(k) else "k=inf"
        ub_s = (f"{log_sg_ub:.4f} (σ_GP≤{np.exp(log_sg_ub):.5f})"
                if log_sg_ub else "none")
        print(f"\n[{k_s}]  log_σ_GP ub: {ub_s}", flush=True)

        best_p, best_nlml = fit_gp(
            X_tr, r_train, noise_var, log_sg_ub, rng, n_rest, pfx=f"{k_s} ",
            x_warm=prev_best)
        prev_best = best_p

        gm_tr, gv_tr = predict_gp(best_p, X_tr, r_train, X_tr, noise_var)
        gm_va, gv_va = predict_gp(best_p, X_tr, r_train, X_va, noise_var)
        d_tr = compute_diag(f_adim_tr, gm_tr, gv_tr, f_obs_tr, noise_var,
                            f"{k_s} train")
        d_va = compute_diag(f_adim_va, gm_va, gv_va, f_obs_va, noise_var,
                            f"{k_s} val (all)")

        # burst-split validation metrics
        nb = ~burst_mask_val;  br = burst_mask_val
        d_va_nb = compute_diag(f_adim_va[nb], gm_va[nb], gv_va[nb],
                               f_obs_va[nb], noise_var, f"{k_s} val-noburst") \
                  if nb.any() else {}
        d_va_br = compute_diag(f_adim_va[br], gm_va[br], gv_va[br],
                               f_obs_va[br], noise_var, f"{k_s} val-burst") \
                  if br.any() else {}

        rec = {
            "k":            k if not np.isinf(k) else "inf",
            "sigma_gp_Hz":  float(np.exp(best_p[0])),
            "log_sigma_gp": float(best_p[0]),
            "ell_EMC":      float(np.exp(best_p[1])),
            "ell_T":        float(np.exp(best_p[2])),
            "ell_diel":     float(np.exp(best_p[3])),
            "neg_log_ml":   float(best_nlml),
            "train":        d_tr,
            "val":          d_va,
            "val_noburst":  d_va_nb,
            "val_burst":    d_va_br,
        }
        sweep.append(rec)

        if k == K_PRIMARY:
            pri_data = dict(p=best_p,
                            gm_tr=gm_tr, gv_tr=gv_tr,
                            gm_va=gm_va, gv_va=gv_va)

    # Summary table
    pri_rec = next(r for r in sweep if r["k"] == K_PRIMARY)
    print(f"\n{'─'*72}")
    print(f"{'k':>6} | {'σ_GP':>9} | {'ℓ_EMC':>7} | {'ℓ_T':>6} | "
          f"{'ℓ_diel':>7} | {'CRPS_val':>10} | {'cov95_val':>10}")
    print("─" * 72)
    for r in sweep:
        k_s = str(r["k"]) if r["k"] != "inf" else "∞"
        print(f"{k_s:>6} | {r['sigma_gp_Hz']:>9.5f} | {r['ell_EMC']:>7.3f} | "
              f"{r['ell_T']:>6.3f} | {r['ell_diel']:>7.4f} | "
              f"{r['val']['CRPS_mean']:>10.5f} | {r['val']['coverage_95']:>10.4f}")

    # ── Figures + settings ─────────────────────────────────────────────────
    print("\nGenerating figures …", flush=True)

    piml_fig = OUTPUT_DIR / f"Fig_Level3_PIML_Mode{mode}.png"
    plot_piml(
        pd.Series(train["DateTime"].values), f_obs_tr, f_adim_tr,
        pri_data["gm_tr"], pri_data["gv_tr"],
        pd.Series(val["DateTime"].values),   f_obs_va, f_adim_va,
        pri_data["gm_va"], pri_data["gv_va"],
        noise_var, K_PRIMARY, mode, piml_fig,
    )
    write_settings(
        OUTPUT_DIR / f"Fig_Level3_PIML_Mode{mode}_settings.txt",
        piml_fig.name, mode, theta, sigma_obs, noise_var, train, val,
        n_rest, pri_rec,
        label=f"PIML time-series: f_obs, adim prediction, PIML mean +/- 1.96sigma_total, GP correction",
        n_train_raw=n_train_raw,
    )

    ksweep_fig = OUTPUT_DIR / f"Fig_Level3_kSweep_Mode{mode}.png"
    plot_ksweep(K_VALUES, [r["train"] for r in sweep],
                [r["val"] for r in sweep],
                [r["val_noburst"] or r["val"] for r in sweep],
                mode, ksweep_fig)
    write_ksweep_settings(
        OUTPUT_DIR / f"Fig_Level3_kSweep_Mode{mode}_settings.txt",
        ksweep_fig.name, mode, K_VALUES, sweep, n_rest,
    )

    # ── JSON + sanity checklist ────────────────────────────────────────────
    result = {
        "mode": mode,
        "n_train_raw": n_train_raw, "n_train": int(len(train)),
        "n_val": int(len(val)), "n_val_burst": int(burst_mask_val.sum()),
        "step01_theta": theta, "noise_var_Hz2": noise_var,
        "sigma_obs_Hz": float(sigma_obs),
        "gp_covariates": GP_COVARIATES,
        "primary_k": K_PRIMARY, "k_sweep": sweep,
        "elapsed_total_s": round(time.time() - t0, 1),
    }
    json_path = AUX_DIR / f"level3_Mode{mode}_result.json"
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved → {json_path}", flush=True)

    sg_ratio = pri_rec["sigma_gp_Hz"] / sigma_obs
    chk = [
        f"# Level 3 Mode {mode} — Sanity Checklist",
        f"k_primary={K_PRIMARY}  n_train={len(train)}  n_val={len(val)}\n",
        "## Hyperparameters (k=1)",
        f"- σ_GP   = {pri_rec['sigma_gp_Hz']:.5f} Hz  (σ_obs={sigma_obs:.5f}  ratio={sg_ratio:.3f})",
        f"- ℓ_EMC  = {pri_rec['ell_EMC']:.4f} %",
        f"- ℓ_T    = {pri_rec['ell_T']:.4f} °C",
        f"- ℓ_diel = {pri_rec['ell_diel']:.5f}\n",
        "## Val diagnostics (all)",
        f"- CRPS     = {pri_rec['val']['CRPS_mean']:.5f} Hz",
        f"- Coverage = {pri_rec['val']['coverage_95']:.4f}",
        f"- RMSE: adim={pri_rec['val']['RMSE_adim_Hz']:.5f} → PIML={pri_rec['val']['RMSE_piml_Hz']:.5f} Hz\n",
        "## Val burst split",
        f"- No-burst  CRPS     = {pri_rec['val_noburst'].get('CRPS_mean', float('nan')):.5f} Hz"
            f"  (n={pri_rec['val_noburst'].get('n','?')})",
        f"- No-burst  Coverage = {pri_rec['val_noburst'].get('coverage_95', float('nan')):.4f}",
        f"- No-burst  RMSE: adim={pri_rec['val_noburst'].get('RMSE_adim_Hz', float('nan')):.5f}"
            f" → PIML={pri_rec['val_noburst'].get('RMSE_piml_Hz', float('nan')):.5f} Hz",
        f"- Burst     CRPS     = {pri_rec['val_burst'].get('CRPS_mean', float('nan')):.5f} Hz"
            f"  (n={pri_rec['val_burst'].get('n','?')})",
        f"- Burst     Coverage = {pri_rec['val_burst'].get('coverage_95', float('nan')):.4f}",
        f"- Burst     RMSE: adim={pri_rec['val_burst'].get('RMSE_adim_Hz', float('nan')):.5f}"
            f" → PIML={pri_rec['val_burst'].get('RMSE_piml_Hz', float('nan')):.5f} Hz\n",
        "## Flags",
        f"- σ_GP/σ_obs = {sg_ratio:.3f} " +
            ("✓" if sg_ratio < 0.8 else "⚠ large GP variance"),
        f"- k-cap binding: " +
            ("no (self-regularised)" if sg_ratio < 0.99 else "yes — k-cap active"),
        f"- Val coverage: " +
            ("✓" if 0.90 < pri_rec['val']['coverage_95'] < 0.99 else "⚠"),
        f"- RMSE improved: " +
            ("✓" if pri_rec['val']['RMSE_piml_Hz'] < pri_rec['val']['RMSE_adim_Hz']
             else "⚠ PIML no better on val"),
    ]
    (AUX_DIR / f"level3_Mode{mode}_sanity.md").write_text("\n".join(chk))

    sys.stdout = orig_stdout
    (AUX_DIR / f"level3_Mode{mode}_run_log.txt").write_text(log_buf.getvalue())
    print(f"\n✓ Mode {mode} done — {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
