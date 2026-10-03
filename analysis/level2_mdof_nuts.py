"""Level 2 — Dimensional MDOF Bayesian fit via NumPyro NUTS.
by Andre R. Barbosa, April - October 2026

Model
-----
A fixed-base lumped-mass shear stack with N_STOREYS = 7 degrees of freedom
(floor levels FL2 to FL8) and story mass M_CENTRAL fixed at 100,000 kg. The first
natural frequency in each direction replaces the Level 1 reference frequency:
Mode 1 (X-translation) depends on k_X and Mode 2 (Y-translation) on k_Y.

Variants
--------
  - compact, fixed tau (default; the model reported in the paper): one stiffness
    per direction, (alpha_i, beta_i, sigma_obs_i) per mode, and the lag times
    fixed at the Level 1 posterior medians — 8 parameters;
  - per-story profile, fixed tau (--no-compact): log k_j = log k_geo + profile_j,
    profile_j ~ Normal(0, SIGMA_PROFILE) — 24 parameters;
  - per-story profile with sampled lag times (--no-fixed-tau) — 28 parameters.
With one frequency per direction the per-story profile is not identifiable, so
the profile variants are retained for reference only.

Forward model (JAX-JIT)
-----------------------
  - tridiagonal K per direction from the story stiffnesses
  - first natural frequency via jnp.linalg.eigvalsh — fully differentiable
  - lag interpolation via jax.vmap over jnp.interp on the pre-gridded lag matrix

Sampler
-------
  NumPyro NUTS, 4 chains; the paper run uses --warmup 1000 --samples 10000.

Output written to aux/level2_mdof_nuts/:
  posterior.json, diagnostic.png and run diagnostics
"""

from __future__ import annotations

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
import arviz as az

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from analysis.common.data_loader import (
    load_continuous_hus6, per_mode_view, get_lag_matrix,
    chronological_split, FREQ_COLS,
)
from analysis.common.plotting import set_paper_style, MODE_COLORS, MODE_LABELS

# ── Physical constants ─────────────────────────────────────────────────────
M_CENTRAL = 100_000.0          # kg / storey, fixed (Reynolds 2014 apt A + JCSS Part 2)
SIGMA_LOG_K = 0.15             # prior SD on log(k_geo) — ~15% uncertainty on stiffness scale
SIGMA_PROFILE = 0.05           # prior SD on per-storey log-deviation — ~5% per storey
N_STOREYS = 7                  # 7 DOFs at floor levels FL2 (3 m) to FL8 (21 m); fixed base at FL1

# Pre-computed lag grid (hours, log-spaced)
LAG_HOURS = np.array([1., 3., 6., 12., 24., 48., 72., 168., 336., 504.,
                       720., 1080., 1440., 2160.])
LOG_TAU_GRID = jnp.array(np.log(LAG_HOURS), dtype=jnp.float32)

# Priors (material physics, per mode)
ALPHA_PRIOR_MEAN  = 0.002    # Normal mean (1/%EMC)  — JCSS d(E)/dMC
ALPHA_PRIOR_SD    = 0.002    # broad enough to accommodate the mode-specific range
BETA_PRIOR_MEAN   = -0.0025  # Normal mean (1/°C)
BETA_PRIOR_SD     = 0.001
LOG_SIGMA_OBS_MU  = float(np.log(0.015))   # LogNormal centre ≈ 15 mHz
LOG_SIGMA_OBS_SD  = 0.30
LOG_TAU_EMC_MU    = float(np.log(1080.))   # 45 days in hours
LOG_TAU_EMC_SD    = 0.60
LOG_TAU_T_MU      = float(np.log(168.))    # 7 days in hours
LOG_TAU_T_SD      = 0.70


# ── JAX forward model (JIT-compiled) ──────────────────────────────────────

def build_K_jax(k: jnp.ndarray) -> jnp.ndarray:
    """n×n tridiagonal shear-stack stiffness matrix. k: (n,)

    Convention (matches build_shear_K in forward_model.py):
      k[0]   = ground spring (storey 1 ↔ fixed base)
      k[j]   = inter-storey spring (storey j ↔ storey j+1)
      K[j,j] = k[j] + k[j+1]  for j=0..n-2
      K[n-1,n-1] = k[n-1]
      K[j,j±1]   = -k[j+1]
    """
    main_diag = jnp.append(k[:-1] + k[1:], k[-1:])   # (8,)
    off_diag  = -k[1:]                                  # (7,)
    return jnp.diag(main_diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)


def first_freq_jax(k: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    """First natural frequency (Hz) of n-DOF shear stack. k, m: (n,)"""
    K = build_K_jax(k)
    # Generalised eigenvalue K v = ω² M v with M = diag(m).
    # Transform: M^{-1/2} K M^{-1/2} x = ω² x, then eigvalsh gives ascending λ.
    inv_sqrt_m = 1.0 / jnp.sqrt(m)
    K_tilde = K * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]
    lambda_min = jnp.linalg.eigvalsh(K_tilde)[0]
    return jnp.sqrt(jnp.maximum(lambda_min, 0.0)) / (2.0 * jnp.pi)


def interp_lag_jax(log_tau: jnp.ndarray, lag_matrix: jnp.ndarray) -> jnp.ndarray:
    """Differentiable lag interpolation along the pre-gridded tau axis.

    log_tau  : scalar (log-hours)
    lag_matrix: (N, 14) — pre-computed lag values at LOG_TAU_GRID
    returns   : (N,)   — interpolated lag time-series
    """
    def _row(row):
        return jnp.interp(log_tau, LOG_TAU_GRID, row)
    return jax.vmap(_row)(lag_matrix)


# ── NumPyro probabilistic models ──────────────────────────────────────────

def numpyro_model(emc_mats, t_mats, f_obs_list, k_anchor_X, k_anchor_Y):
    """28-parameter dimensional MDOF model — tau jointly sampled.

    emc_mats  : list of 2 JAX arrays (N_i, 14) — lag matrix at LAG_HOURS grid
    t_mats    : list of 2 JAX arrays (N_i, 14)
    f_obs_list: list of 2 JAX arrays (N_i,)
    k_anchor_X, k_anchor_Y: float — period-anchored prior centre (N/m)
    """
    m_uniform = jnp.full(N_STOREYS, M_CENTRAL)

    # ── X-direction stiffness (Mode 1) ──
    log_k_X_geo     = numpyro.sample("log_k_X_geo",
                          dist.Normal(float(np.log(k_anchor_X)), SIGMA_LOG_K))
    log_k_X_profile = numpyro.sample("log_k_X_profile",
                          dist.Normal(jnp.zeros(N_STOREYS),
                                      jnp.full(N_STOREYS, SIGMA_PROFILE)).to_event(1))
    k_X = jnp.exp(log_k_X_geo + log_k_X_profile)
    f0_X = first_freq_jax(k_X, m_uniform)

    # ── Y-direction stiffness (Mode 2) ──
    log_k_Y_geo     = numpyro.sample("log_k_Y_geo",
                          dist.Normal(float(np.log(k_anchor_Y)), SIGMA_LOG_K))
    log_k_Y_profile = numpyro.sample("log_k_Y_profile",
                          dist.Normal(jnp.zeros(N_STOREYS),
                                      jnp.full(N_STOREYS, SIGMA_PROFILE)).to_event(1))
    k_Y = jnp.exp(log_k_Y_geo + log_k_Y_profile)
    f0_Y = first_freq_jax(k_Y, m_uniform)

    f0 = [f0_X, f0_Y]

    for i, lbl in enumerate(["1", "2"]):
        alpha = numpyro.sample(f"alpha_{lbl}", dist.Normal(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD))
        beta  = numpyro.sample(f"beta_{lbl}",  dist.Normal(BETA_PRIOR_MEAN, BETA_PRIOR_SD))
        log_sigma_obs = numpyro.sample(f"log_sigma_obs_{lbl}",
                    dist.Normal(LOG_SIGMA_OBS_MU, LOG_SIGMA_OBS_SD))
        log_tau_EMC   = numpyro.sample(f"log_tau_EMC_h_{lbl}",
                    dist.Normal(LOG_TAU_EMC_MU, LOG_TAU_EMC_SD))
        log_tau_T     = numpyro.sample(f"log_tau_T_h_{lbl}",
                    dist.Normal(LOG_TAU_T_MU, LOG_TAU_T_SD))

        emc_lag = interp_lag_jax(log_tau_EMC, emc_mats[i])
        t_lag   = interp_lag_jax(log_tau_T,   t_mats[i])

        one_plus = jnp.maximum(1.0 + alpha * emc_lag + beta * t_lag, 1e-6)
        f_hat    = f0[i] * jnp.sqrt(one_plus)
        sigma    = jnp.exp(log_sigma_obs)
        numpyro.sample(f"f_obs_{lbl}", dist.Normal(f_hat, sigma), obs=f_obs_list[i])


def numpyro_model_compact_fixed_tau(emc_lags, t_lags, f_obs_list, k_anchor_X, k_anchor_Y):
    """8-parameter compact dimensional model — uniform-k per direction, tau fixed.

    With 1 frequency per direction, per-storey profile parameters are not identifiable
    (rank 1 data cannot constrain rank 8 profile parameters — they show Rhat>2 even
    with 1000 warmup steps in 4-chain tests). This compact model uses a single scalar
    stiffness per direction; per-storey resolution requires Mode 3+ for independent
    constraints. The 28-parameter per-storey variant is retained for reference; the
    compact model identifies what the data actually supports.

    emc_lags  : list of 2 JAX arrays (N_i,) — lag series pre-interpolated at Level 1 tau
    t_lags    : list of 2 JAX arrays (N_i,) — same for temperature
    f_obs_list: list of 2 JAX arrays (N_i,)
    """
    m_uniform = jnp.full(N_STOREYS, M_CENTRAL)

    log_k_X = numpyro.sample("log_k_X",
                  dist.Normal(float(np.log(k_anchor_X)), SIGMA_LOG_K))
    k_X  = jnp.full(N_STOREYS, jnp.exp(log_k_X))
    f0_X = first_freq_jax(k_X, m_uniform)

    log_k_Y = numpyro.sample("log_k_Y",
                  dist.Normal(float(np.log(k_anchor_Y)), SIGMA_LOG_K))
    k_Y  = jnp.full(N_STOREYS, jnp.exp(log_k_Y))
    f0_Y = first_freq_jax(k_Y, m_uniform)

    f0 = [f0_X, f0_Y]

    for i, lbl in enumerate(["1", "2"]):
        alpha = numpyro.sample(f"alpha_{lbl}", dist.Normal(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD))
        beta  = numpyro.sample(f"beta_{lbl}",  dist.Normal(BETA_PRIOR_MEAN, BETA_PRIOR_SD))
        log_sigma_obs = numpyro.sample(f"log_sigma_obs_{lbl}",
                    dist.Normal(LOG_SIGMA_OBS_MU, LOG_SIGMA_OBS_SD))

        one_plus = jnp.maximum(1.0 + alpha * emc_lags[i] + beta * t_lags[i], 1e-6)
        f_hat    = f0[i] * jnp.sqrt(one_plus)
        sigma    = jnp.exp(log_sigma_obs)
        numpyro.sample(f"f_obs_{lbl}", dist.Normal(f_hat, sigma), obs=f_obs_list[i])


def numpyro_model_fixed_tau(emc_lags, t_lags, f_obs_list, k_anchor_X, k_anchor_Y):
    """24-parameter dimensional MDOF model — tau fixed to Level 1 medians.

    Two-stage approach: tau is treated as a known quantity from Level 1.
    This removes the tau-geometry that causes mixing failures.

    emc_lags  : list of 2 JAX arrays (N_i,) — lag series pre-interpolated at Level 1 tau
    t_lags    : list of 2 JAX arrays (N_i,) — same for temperature
    f_obs_list: list of 2 JAX arrays (N_i,)
    """
    m_uniform = jnp.full(N_STOREYS, M_CENTRAL)

    log_k_X_geo     = numpyro.sample("log_k_X_geo",
                          dist.Normal(float(np.log(k_anchor_X)), SIGMA_LOG_K))
    log_k_X_profile = numpyro.sample("log_k_X_profile",
                          dist.Normal(jnp.zeros(N_STOREYS),
                                      jnp.full(N_STOREYS, SIGMA_PROFILE)).to_event(1))
    k_X  = jnp.exp(log_k_X_geo + log_k_X_profile)
    f0_X = first_freq_jax(k_X, m_uniform)

    log_k_Y_geo     = numpyro.sample("log_k_Y_geo",
                          dist.Normal(float(np.log(k_anchor_Y)), SIGMA_LOG_K))
    log_k_Y_profile = numpyro.sample("log_k_Y_profile",
                          dist.Normal(jnp.zeros(N_STOREYS),
                                      jnp.full(N_STOREYS, SIGMA_PROFILE)).to_event(1))
    k_Y  = jnp.exp(log_k_Y_geo + log_k_Y_profile)
    f0_Y = first_freq_jax(k_Y, m_uniform)

    f0 = [f0_X, f0_Y]

    for i, lbl in enumerate(["1", "2"]):
        alpha = numpyro.sample(f"alpha_{lbl}", dist.Normal(ALPHA_PRIOR_MEAN, ALPHA_PRIOR_SD))
        beta  = numpyro.sample(f"beta_{lbl}",  dist.Normal(BETA_PRIOR_MEAN, BETA_PRIOR_SD))
        log_sigma_obs = numpyro.sample(f"log_sigma_obs_{lbl}",
                    dist.Normal(LOG_SIGMA_OBS_MU, LOG_SIGMA_OBS_SD))

        one_plus = jnp.maximum(1.0 + alpha * emc_lags[i] + beta * t_lags[i], 1e-6)
        f_hat    = f0[i] * jnp.sqrt(one_plus)
        sigma    = jnp.exp(log_sigma_obs)
        numpyro.sample(f"f_obs_{lbl}", dist.Normal(f_hat, sigma), obs=f_obs_list[i])


# ── Stiffness anchor (period-matching) ────────────────────────────────────

def compute_k_anchor(f_med_hz: float, M_central_kg: float = M_CENTRAL,
                     n_storeys: int = N_STOREYS) -> float:
    """Period-anchored prior mean for the geometric-mean per-storey k.

    For a uniform n-DOF shear building (all k_j = k, all m_j = M_central),
    the first natural frequency of the fixed-base chain is:
      f_1 = (1/2π) × 2 sin(π / (4n+2)) × √(k / M_central)
    → k = M_central × [2π f_1 / (2 sin(π / (4n+2)))]²

    n=7: shape_factor = 2sin(π/30) ≈ 0.2091
    n=8: shape_factor = 2sin(π/34) ≈ 0.1844
    """
    shape_factor = 2.0 * np.sin(np.pi / (4 * n_storeys + 2))
    omega        = 2.0 * np.pi * f_med_hz
    return float(M_central_kg * (omega / shape_factor) ** 2)


# ── Posterior predictive (NumPy, post-sampling) ───────────────────────────

def posterior_predictive_numpy(samples_dict: dict, emc_mat: np.ndarray, t_mat: np.ndarray,
                                mode_idx: int, n_draws: int = 200,
                                rng: np.random.Generator | None = None,
                                fixed_tau: bool = True,
                                fixed_tau_vals: dict | None = None,
                                compact: bool = True) -> np.ndarray:
    """Draw posterior-predictive f_hat trajectories for one mode (CPU NumPy).

    samples_dict: dict[str, np.ndarray] — each value already flattened to (S,) or (S, K).
    compact=True:  uses scalar log_k_X / log_k_Y (uniform per-storey k).
    compact=False: uses log_k_X_geo + log_k_X_profile (per-storey profile).
    fixed_tau=True:  emc_mat, t_mat are (N,) pre-interpolated lag series.
    fixed_tau=False: emc_mat, t_mat are (N, 14) lag matrices at LAG_HOURS.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lbl = str(mode_idx + 1)

    if compact:
        k_key = "log_k_X" if mode_idx == 0 else "log_k_Y"
        n_post = samples_dict[k_key].shape[0]
    else:
        geo_key     = "log_k_X_geo"     if mode_idx == 0 else "log_k_Y_geo"
        profile_key = "log_k_X_profile" if mode_idx == 0 else "log_k_Y_profile"
        n_post = samples_dict[geo_key].shape[0]

    n_draws = min(n_draws, n_post)
    idx = rng.choice(n_post, size=n_draws, replace=False)

    log_tau_grid_np = np.log(LAG_HOURS)   # (14,)
    N   = emc_mat.shape[0]
    out = np.empty((n_draws, N))

    from scipy.linalg import eigh as _eigh
    from analysis.common.forward_model import build_shear_K

    for ii, draw in enumerate(idx):
        if compact:
            log_k = float(samples_dict[k_key][draw])
            k_per_storey = np.full(N_STOREYS, np.exp(log_k))
        else:
            log_k_geo     = float(samples_dict[geo_key][draw])
            log_k_profile = np.array(samples_dict[profile_key][draw])   # (N_STOREYS,)
            k_per_storey  = np.exp(log_k_geo + log_k_profile)

        m    = np.full(N_STOREYS, M_CENTRAL)
        K    = build_shear_K(k_per_storey)
        evals = _eigh(K, np.diag(m), eigvals_only=True)
        f0   = float(np.sqrt(max(evals[0], 0.0)) / (2 * np.pi))

        alpha     = float(samples_dict[f"alpha_{lbl}"][draw])
        beta      = float(samples_dict[f"beta_{lbl}"][draw])
        sigma_obs = float(np.exp(samples_dict[f"log_sigma_obs_{lbl}"][draw]))

        if fixed_tau and fixed_tau_vals:
            # Lag series already pre-interpolated; emc_mat and t_mat are (N,)
            emc_lag = np.asarray(emc_mat)
            t_lag   = np.asarray(t_mat)
        else:
            log_tau_EMC = float(samples_dict[f"log_tau_EMC_h_{lbl}"][draw])
            log_tau_T   = float(samples_dict[f"log_tau_T_h_{lbl}"][draw])
            emc_lag = np.array([np.interp(log_tau_EMC, log_tau_grid_np, emc_mat[t]) for t in range(N)])
            t_lag   = np.array([np.interp(log_tau_T,   log_tau_grid_np,   t_mat[t]) for t in range(N)])

        one_plus = np.maximum(1.0 + alpha * emc_lag + beta * t_lag, 1e-6)
        f_hat    = f0 * np.sqrt(one_plus)
        out[ii]  = f_hat + rng.normal(0.0, sigma_obs, size=N)

    return out


# ── Sanity checklist ───────────────────────────────────────────────────────

def evaluate_sanity(idata, k_anchor_X, k_anchor_Y, f_med,
                    fixed_tau: bool = True,
                    fixed_tau_vals: dict | None = None,
                    compact: bool = True):
    """Pre-registered sanity checks for Level 2 NUTS."""
    post = idata.posterior
    checks = []

    def _stack(var):
        return np.array(post[var]).reshape(-1, *post[var].shape[2:])

    has_profile = "log_k_X_profile" in post.data_vars

    if has_profile:
        # Per-storey k_X medians: positive, within 3× anchor
        k_X_meds = np.median(np.exp(_stack("log_k_X_geo")[:, None] + _stack("log_k_X_profile")), axis=0)
        checks.append({
            "name": "k_X per-storey medians all positive & finite",
            "value": [round(float(v), 2) for v in k_X_meds],
            "passed": bool(np.all(np.isfinite(k_X_meds)) and np.all(k_X_meds > 0)),
            "note": "Physical sanity",
        })
        k_X_geo = float(np.exp(np.median(_stack("log_k_X_geo"))))
        k_Y_geo = float(np.exp(np.median(_stack("log_k_Y_geo"))))
    else:
        k_X_geo = float(np.exp(np.median(_stack("log_k_X"))))
        k_Y_geo = float(np.exp(np.median(_stack("log_k_Y"))))
    checks.append({
        "name": "Geometric-mean k_X within 3× of period anchor",
        "value": [round(k_X_geo, 2), round(k_anchor_X, 2)],
        "passed": bool(k_anchor_X / 3 <= k_X_geo <= 3 * k_anchor_X),
        "note": "Prior consistency",
    })
    checks.append({
        "name": "Geometric-mean k_Y within 3× of period anchor",
        "value": [round(k_Y_geo, 2), round(k_anchor_Y, 2)],
        "passed": bool(k_anchor_Y / 3 <= k_Y_geo <= 3 * k_anchor_Y),
        "note": "Prior consistency",
    })

    # alpha signs; tau checks depend on model variant
    for i, lbl in enumerate(["1", "2"]):
        a_med = float(np.median(_stack(f"alpha_{lbl}")))
        checks.append({
            "name": f"alpha_{lbl} > 0 (stiffness increases with EMC)",
            "value": round(a_med, 5),
            "passed": bool(a_med > 0),
            "note": "Expected positive from Level 1",
        })
        checks.append({
            "name": f"alpha_{lbl} in (0.0005, 0.010)",
            "value": round(a_med, 5),
            "passed": bool(0.0005 < a_med < 0.010),
            "note": f"Level 1 posterior: mode 1 ≈ 0.0037, mode 2 ≈ 0.0014",
        })
        b_med = float(np.median(_stack(f"beta_{lbl}")))
        checks.append({
            "name": f"beta_{lbl} < 0 (stiffness decreases with temperature)",
            "value": round(b_med, 5),
            "passed": bool(b_med < 0),
            "note": "Expected negative (thermal softening)",
        })
        if fixed_tau and fixed_tau_vals:
            tau_EMC = float(np.exp(fixed_tau_vals[f"log_tau_EMC_h_{lbl}"]) / 24)
            checks.append({
                "name": f"tau_EMC_{lbl} in (2, 60) days [FIXED from Level 1]",
                "value": round(tau_EMC, 1),
                "passed": bool(2 < tau_EMC < 60),
                "note": "Fixed at Level 1 median — passes by construction",
            })
        else:
            tau_EMC = float(np.exp(np.median(_stack(f"log_tau_EMC_h_{lbl}"))) / 24)
            checks.append({
                "name": f"tau_EMC_{lbl} in (2, 60) days",
                "value": round(tau_EMC, 1),
                "passed": bool(2 < tau_EMC < 60),
                "note": f"Level 1: mode 1 ≈ 12 d, mode 2 ≈ 9 d",
            })

    # Convergence
    rhat_ds   = az.rhat(idata)
    rhat_max  = float(max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars))
    ess_ds    = az.ess(idata)
    ess_min   = float(min(float(ess_ds[v].min())  for v in ess_ds.data_vars))
    checks.append({
        "name": "Rhat_max < 1.01 (NUTS target)",
        "value": round(rhat_max, 4),
        "passed": bool(rhat_max < 1.01),
        "note": "Paper-quality convergence criterion",
    })
    checks.append({
        "name": "ESS_min > 400 per parameter",
        "value": round(ess_min, 0),
        "passed": bool(ess_min > 400),
        "note": "Adequate NUTS sampling",
    })

    return {
        "checks": checks,
        "n_passed": int(sum(c["passed"] for c in checks)),
        "n_total":  len(checks),
        "rhat_max": rhat_max,
        "ess_min":  ess_min,
    }


def write_sanity_md(path: Path, sanity: dict) -> None:
    lines = [
        "# Level 2 NUTS — Sanity Check Status\n",
        f"**{sanity['n_passed']}/{sanity['n_total']} checks passed.**  "
        f"Rhat_max = {sanity['rhat_max']:.4f}  |  ESS_min = {sanity['ess_min']:.0f}\n",
    ]
    for c in sanity["checks"]:
        mark = "✓" if c["passed"] else "✗"
        val_str = (
            "[" + ", ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in c["value"]) + "]"
            if isinstance(c["value"], list)
            else f"{c['value']:.4g}"
        )
        lines.append(f"- **{mark} {c['name']}** | value = {val_str} | {c['note']}")
    path.write_text("\n".join(lines))


# ── Diagnostic figure ─────────────────────────────────────────────────────

def build_diagnostic_figure(idata, per_mode_train_dfs, emc_mats_train, t_mats_train,
                             samples_dict: dict, f_med, rng,
                             fixed_tau: bool = True,
                             fixed_tau_vals: dict | None = None,
                             compact: bool = True) -> plt.Figure:
    set_paper_style()
    fig = plt.figure(figsize=(16, 18))
    gs  = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 0.9, 2.0], hspace=0.50)

    post = idata.posterior

    def _stack(var):
        return np.array(post[var]).reshape(-1, *post[var].shape[2:])

    # ── stiffness posterior ──
    gs_k = gs[0].subgridspec(1, 2, wspace=0.25)
    has_profile = "log_k_X_profile" in post.data_vars
    for di, (direction, geo_key, profile_key, k_key, color) in enumerate([
        ("X (Mode 1)", "log_k_X_geo", "log_k_X_profile", "log_k_X", MODE_COLORS[0]),
        ("Y (Mode 2)", "log_k_Y_geo", "log_k_Y_profile", "log_k_Y", MODE_COLORS[1]),
    ]):
        ax = fig.add_subplot(gs_k[0, di])
        if has_profile:
            log_k_geo  = _stack(geo_key)
            log_k_prof = _stack(profile_key)
            k_draws    = np.exp(log_k_geo[:, None] + log_k_prof)
            for j in range(N_STOREYS):
                k_j = k_draws[:, j]
                med = np.median(k_j)
                lo, hi = np.quantile(k_j, [0.025, 0.975])
                ax.errorbar([j], [med/1e8], yerr=[[((med-lo)/1e8)], [((hi-med)/1e8)]],
                            fmt="o", color=color, capsize=3, lw=1.5, ms=5)
            ax.set_xlabel(f"Storey j  (0 = ground spring, {N_STOREYS-1} = top)")
            ax.set_xticks(range(N_STOREYS))
        else:
            # Compact model: histogram of scalar k
            k_draws_flat = np.exp(_stack(k_key))
            ax.hist(k_draws_flat / 1e8, bins=40, color=color, alpha=0.75, edgecolor="white")
            ax.axvline(np.median(k_draws_flat) / 1e8, color="k", lw=1.5, ls="--",
                       label=f"median={np.median(k_draws_flat):.3e} N/m")
            ax.set_xlabel("Uniform per-storey k  (×10⁸ N/m)")
            ax.legend(fontsize=7.5)
        ax.set_ylabel("k  (×10⁸ N/m)")
        ax.set_title(f"Stiffness posterior — {direction}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # ── tau posteriors (sampled) or sigma_obs + alpha (fixed-tau) ──
    gs_tau = gs[1].subgridspec(1, 4, wspace=0.40)
    if not fixed_tau:
        for i, (lbl, color) in enumerate([("1", MODE_COLORS[0]), ("2", MODE_COLORS[1])]):
            for j, (tau_key, title_suf, unit_factor) in enumerate([
                (f"log_tau_EMC_h_{lbl}", f"τ_EMC mode {lbl}", 1/24),
                (f"log_tau_T_h_{lbl}",   f"τ_T   mode {lbl}", 1/24),
            ]):
                ax = fig.add_subplot(gs_tau[0, i * 2 + j])
                vals = np.exp(_stack(tau_key)) * unit_factor  # days
                ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor="white")
                ax.axvline(np.median(vals), color="k", lw=1.2, ls="--")
                ax.set_xlabel("days")
                ax.set_title(f"{title_suf}\nmed={np.median(vals):.1f} d", fontsize=9)
                ax.tick_params(labelsize=7)
    else:
        # Fixed-tau: plot alpha and sigma_obs posteriors instead
        for i, (lbl, color) in enumerate([("1", MODE_COLORS[0]), ("2", MODE_COLORS[1])]):
            for j, (key, xlabel, scale, title) in enumerate([
                (f"alpha_{lbl}",         "×10⁻³ per %EMC", 1e3, f"α mode {lbl}"),
                (f"log_sigma_obs_{lbl}", "mHz",            1e3, f"σ_obs mode {lbl}"),
            ]):
                ax = fig.add_subplot(gs_tau[0, i * 2 + j])
                if j == 0:
                    vals = _stack(key) * scale
                else:
                    vals = np.exp(_stack(key)) * scale
                ax.hist(vals, bins=30, color=color, alpha=0.75, edgecolor="white")
                ax.axvline(np.median(vals), color="k", lw=1.2, ls="--")
                ax.set_xlabel(xlabel)
                ax.set_title(f"{title}\nmed={np.median(vals):.2f}", fontsize=9)
                ax.tick_params(labelsize=7)
        # Annotate fixed tau values
        if fixed_tau_vals:
            tau_txt = (f"τ fixed from Level 1:  "
                       f"τ_EMC M1={np.exp(fixed_tau_vals['log_tau_EMC_h_1'])/24:.1f}d  "
                       f"τ_EMC M2={np.exp(fixed_tau_vals['log_tau_EMC_h_2'])/24:.1f}d  "
                       f"τ_T M1={np.exp(fixed_tau_vals['log_tau_T_h_1'])/24:.1f}d  "
                       f"τ_T M2={np.exp(fixed_tau_vals['log_tau_T_h_2'])/24:.1f}d")
            fig.text(0.5, 0.595, tau_txt, ha="center", fontsize=7.5, color="gray", style="italic")

    # ── alpha and beta per mode ──
    gs_ab = gs[2].subgridspec(1, 4, wspace=0.40)
    for i, (lbl, color) in enumerate([("1", MODE_COLORS[0]), ("2", MODE_COLORS[1])]):
        for j, (key, title) in enumerate([
            (f"alpha_{lbl}", f"α mode {lbl}  (×10³)"),
            (f"beta_{lbl}",  f"β mode {lbl}  (×10³)"),
        ]):
            ax = fig.add_subplot(gs_ab[0, i * 2 + j])
            vals = _stack(key) * 1e3
            ax.hist(vals, bins=40, color=color, alpha=0.75, edgecolor="white")
            ax.axvline(np.median(vals), color="k", lw=1.2, ls="--")
            ax.set_xlabel(f"×10⁻³ per %EMC" if j == 0 else "×10⁻³ per °C")
            ax.set_title(f"{title}\nmed={np.median(vals):.2f}", fontsize=9)
            ax.tick_params(labelsize=7)

    # ── posterior predictive overlays ──
    gs_pp = gs[3].subgridspec(2, 1, hspace=0.45)

    for i in range(2):
        ax = fig.add_subplot(gs_pp[i, 0])
        sub_df  = per_mode_train_dfs[i]
        f_obs_i = sub_df[FREQ_COLS[i]].to_numpy(float)
        emc_mat = emc_mats_train[i]    # (N, 14) — pre-gridded at LAG_HOURS
        t_mat   = t_mats_train[i]      # (N, 14)

        pp = posterior_predictive_numpy(samples_dict, emc_mat, t_mat, mode_idx=i,
                                         n_draws=150, rng=rng,
                                         fixed_tau=fixed_tau, fixed_tau_vals=fixed_tau_vals,
                                         compact=compact)

        pp_lo  = np.quantile(pp, 0.025, axis=0)
        pp_hi  = np.quantile(pp, 0.975, axis=0)
        pp_med = np.quantile(pp, 0.50,  axis=0)
        times  = pd.to_datetime(sub_df.DateTime)

        ax.scatter(times, f_obs_i, s=3, alpha=0.30, color=MODE_COLORS[i],
                   edgecolor="none", label="Observed")
        ax.fill_between(times, pp_lo, pp_hi, color=MODE_COLORS[i],
                        alpha=0.18, edgecolor="none", label="95% predictive band")
        ax.plot(times, pp_med, "-", color=MODE_COLORS[i], lw=1.0, alpha=0.9,
                label="Predictive median")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Level 2 NUTS — posterior predictive ({MODE_LABELS[i]})", fontsize=10)
        ax.legend(fontsize=7.5, ncol=3, loc="upper right")
        ax.set_xlabel("Date")

    fig.suptitle(f"Level 2 — Dimensional MDOF NUTS (7-DOF, M_central fixed, {N_STOREYS}-storey shear stack)",
                 fontsize=12, weight="bold", y=0.997)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────

def run(out_dir: Path,
        num_warmup:  int = 600,
        num_samples: int = 600,
        num_chains:  int = 4,
        rng_seed:    int = 20260608,
        fixed_tau:   bool = True,
        compact:     bool = True) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run_log.txt"

    def log(msg: str) -> None:
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(run_log, "a") as fh:
            fh.write(line + "\n")

    log(f"=== level2_mdof_nuts START ===")
    log(f"num_warmup={num_warmup} num_samples={num_samples} num_chains={num_chains} rng_seed={rng_seed} fixed_tau={fixed_tau}")
    log(f"JAX backend: {jax.default_backend()}, devices: {jax.device_count()}")

    # ── Load data ──
    df_raw = load_continuous_hus6(apply_filters=True)
    train_df, val_df = chronological_split(df_raw)
    log(f"Dataset: {len(df_raw)} rows; train={len(train_df)} val={len(val_df)}")

    per_mode_train = []
    f_meds         = []
    for mode_idx in (1, 2):
        sub = per_mode_view(train_df, mode_idx=mode_idx, apply_pole_filter=True)
        t0  = sub.DateTime.min()
        t1  = t0 + pd.Timedelta(days=30)
        f_m = float(sub.loc[sub.DateTime <= t1][FREQ_COLS[mode_idx - 1]].median())
        f_meds.append(f_m)
        per_mode_train.append({
            "df":          sub,
            "f_obs":       sub[FREQ_COLS[mode_idx - 1]].to_numpy(float),
            "emc_mat":     get_lag_matrix(sub, "EMC").astype(np.float32),
            "t_mat":       get_lag_matrix(sub, "T").astype(np.float32),
        })
        log(f"  Mode {mode_idx}: N_train={len(sub)}, first-month median={f_m:.4f} Hz")

    # ── Compute stiffness anchors ──
    k_anchor_X = compute_k_anchor(f_meds[0])
    k_anchor_Y = compute_k_anchor(f_meds[1])
    log(f"k_anchor_X = {k_anchor_X:.3e} N/m  (f_med={f_meds[0]:.4f} Hz)")
    log(f"k_anchor_Y = {k_anchor_Y:.3e} N/m  (f_med={f_meds[1]:.4f} Hz)")

    # ── Level 1 tau medians (used for fixed-tau model and as init point) ──
    # From aux/level1_adim/v3_per_mode_tau/posterior.json
    LOG_TAU_EMC_H1  = 5.9940  # 401 h ≈ 16.7 d (Level 1 posterior median)
    LOG_TAU_T_H1    = 6.1203  # 455 h ≈ 19.0 d (Level 1 posterior median)
    LOG_TAU_EMC_H2  = 5.5872  # 267 h ≈ 11.1 d (Level 1 posterior median)
    LOG_TAU_T_H2    = 4.9767  # 145 h ≈  6.0 d (Level 1 posterior median)

    log_tau_np = np.log(LAG_HOURS)   # (14,) grid for np.interp

    def _interp_lag(mat, log_tau):
        """CPU interpolation of (N,14) lag matrix at fixed log_tau."""
        return np.array([np.interp(log_tau, log_tau_np, mat[t]) for t in range(len(mat))])

    # ── Convert to JAX arrays ──
    emc_mats_np = [d["emc_mat"] for d in per_mode_train]
    t_mats_np   = [d["t_mat"]   for d in per_mode_train]
    f_jax       = [jnp.array(d["f_obs"], dtype=jnp.float32) for d in per_mode_train]

    if fixed_tau:
        n_params = 8 if compact else 24
        model_tag = f"{'COMPACT 8-param' if compact else 'PROFILE 24-param'} fixed-tau"
        log(f"Running {model_tag}: tau_EMC={[LOG_TAU_EMC_H1,LOG_TAU_EMC_H2]} tau_T={[LOG_TAU_T_H1,LOG_TAU_T_H2]}")
        emc_lags_np = [
            _interp_lag(emc_mats_np[0], LOG_TAU_EMC_H1).astype(np.float32),
            _interp_lag(emc_mats_np[1], LOG_TAU_EMC_H2).astype(np.float32),
        ]
        t_lags_np = [
            _interp_lag(t_mats_np[0], LOG_TAU_T_H1).astype(np.float32),
            _interp_lag(t_mats_np[1], LOG_TAU_T_H2).astype(np.float32),
        ]
        emc_input = [jnp.array(x) for x in emc_lags_np]
        t_input   = [jnp.array(x) for x in t_lags_np]
        model_fn  = numpyro_model_compact_fixed_tau if compact else numpyro_model_fixed_tau
        # Record fixed tau in result later
        fixed_tau_vals = {
            "log_tau_EMC_h_1": LOG_TAU_EMC_H1, "log_tau_T_h_1": LOG_TAU_T_H1,
            "log_tau_EMC_h_2": LOG_TAU_EMC_H2, "log_tau_T_h_2": LOG_TAU_T_H2,
        }
    else:
        emc_lags_np = [d["emc_mat"] for d in per_mode_train]   # (N,14) for full model
        t_lags_np   = [d["t_mat"]   for d in per_mode_train]
        emc_input = [jnp.array(d["emc_mat"]) for d in per_mode_train]
        t_input   = [jnp.array(d["t_mat"])   for d in per_mode_train]
        model_fn  = numpyro_model
        fixed_tau_vals = None

    # ── Run NUTS ──
    # When fixed_tau=True: 24 params, no tau geometry → fast convergence.
    # When fixed_tau=False: 28 params, tau initialization from Level 1 needed.
    from numpyro.infer.initialization import init_to_value
    if compact and fixed_tau:
        base_init = {
            "log_k_X":          float(np.log(k_anchor_X)),
            "log_k_Y":          float(np.log(k_anchor_Y)),
            "alpha_1":  0.003671, "beta_1": -0.002094, "log_sigma_obs_1": -4.448,
            "alpha_2":  0.001434, "beta_2": -0.000277, "log_sigma_obs_2": -4.325,
        }
    else:
        base_init = {
            "log_k_X_geo":      float(np.log(k_anchor_X)),
            "log_k_X_profile":  np.zeros(N_STOREYS),
            "log_k_Y_geo":      float(np.log(k_anchor_Y)),
            "log_k_Y_profile":  np.zeros(N_STOREYS),
            "alpha_1":  0.003671, "beta_1": -0.002094, "log_sigma_obs_1": -4.448,
            "alpha_2":  0.001434, "beta_2": -0.000277, "log_sigma_obs_2": -4.325,
        }
    if not fixed_tau:
        base_init.update({
            "log_tau_EMC_h_1": LOG_TAU_EMC_H1, "log_tau_T_h_1": LOG_TAU_T_H1,
            "log_tau_EMC_h_2": LOG_TAU_EMC_H2, "log_tau_T_h_2": LOG_TAU_T_H2,
        })

    kernel = NUTS(model_fn,
                  target_accept_prob=0.80,
                  init_strategy=init_to_value(values=base_init))
    mcmc   = MCMC(kernel,
                  num_warmup=num_warmup,
                  num_samples=num_samples,
                  num_chains=num_chains,
                  chain_method="sequential",
                  progress_bar=True)

    log(f"Starting NUTS warmup+sampling ({'fixed-tau 24-param' if fixed_tau else 'full 28-param'})...")
    t0_mcmc = time.time()
    rng_key = jax.random.PRNGKey(rng_seed)
    if fixed_tau:
        mcmc.run(rng_key,
                 emc_lags=emc_input, t_lags=t_input, f_obs_list=f_jax,
                 k_anchor_X=k_anchor_X, k_anchor_Y=k_anchor_Y)
    else:
        mcmc.run(rng_key,
                 emc_mats=emc_input, t_mats=t_input, f_obs_list=f_jax,
                 k_anchor_X=k_anchor_X, k_anchor_Y=k_anchor_Y)
    elapsed = time.time() - t0_mcmc
    log(f"NUTS complete in {elapsed:.1f} s ({elapsed/60:.1f} min)")

    # ── Diagnostics via ArviZ ──
    idata = az.from_numpyro(mcmc)
    rhat_ds  = az.rhat(idata)
    ess_ds   = az.ess(idata)
    rhat_max = float(max(float(rhat_ds[v].max()) for v in rhat_ds.data_vars))
    ess_min  = float(min(float(ess_ds[v].min())  for v in ess_ds.data_vars))
    log(f"Rhat_max = {rhat_max:.4f}  |  ESS_min = {ess_min:.0f}")

    # ── Build posterior summary ──
    post      = idata.posterior
    param_names = list(post.data_vars)

    def _flat(var):
        arr = np.array(post[var])
        return arr.reshape(-1, *arr.shape[2:])

    summary_rows = {}
    for pn in param_names:
        draws = _flat(pn)
        if draws.ndim == 1:
            summary_rows[pn] = {
                "mean": float(np.mean(draws)),
                "std":  float(np.std(draws)),
                "q05":  float(np.quantile(draws, 0.05)),
                "q50":  float(np.median(draws)),
                "q95":  float(np.quantile(draws, 0.95)),
            }
        else:  # per-storey array
            for j in range(draws.shape[1]):
                key = f"{pn}[{j}]"
                d_j = draws[:, j]
                summary_rows[key] = {
                    "mean": float(np.mean(d_j)),
                    "std":  float(np.std(d_j)),
                    "q05":  float(np.quantile(d_j, 0.05)),
                    "q50":  float(np.median(d_j)),
                    "q95":  float(np.quantile(d_j, 0.95)),
                }

    # Derived: geometric-mean k per direction and τ in days
    if compact and fixed_tau:
        k_X_geo = np.exp(_flat("log_k_X"))
        k_Y_geo = np.exp(_flat("log_k_Y"))
    else:
        log_k_X_geo_draws  = _flat("log_k_X_geo")
        log_k_X_prof_draws = _flat("log_k_X_profile")
        log_k_Y_geo_draws  = _flat("log_k_Y_geo")
        log_k_Y_prof_draws = _flat("log_k_Y_profile")
        k_X_draws = np.exp(log_k_X_geo_draws[:, None] + log_k_X_prof_draws)
        k_Y_draws = np.exp(log_k_Y_geo_draws[:, None] + log_k_Y_prof_draws)
        k_X_geo   = np.exp(np.mean(np.log(k_X_draws), axis=1))
        k_Y_geo   = np.exp(np.mean(np.log(k_Y_draws), axis=1))

    headline_bits = []
    for i, lbl in enumerate(["1", "2"]):
        a_med   = float(np.median(_flat(f"alpha_{lbl}")))
        b_med   = float(np.median(_flat(f"beta_{lbl}")))
        so_med  = float(np.exp(np.median(_flat(f"log_sigma_obs_{lbl}"))))
        if fixed_tau:
            tau_EMC_keys = [LOG_TAU_EMC_H1, LOG_TAU_EMC_H2]
            tau_T_keys   = [LOG_TAU_T_H1,   LOG_TAU_T_H2]
            tE_med = float(np.exp(tau_EMC_keys[i]) / 24)
            tT_med = float(np.exp(tau_T_keys[i])   / 24)
            tau_tag = f"[τ_EMC={tE_med:.1f}d τ_T={tT_med:.1f}d FIXED]"
        else:
            tE_med = float(np.exp(np.median(_flat(f"log_tau_EMC_h_{lbl}"))) / 24)
            tT_med = float(np.exp(np.median(_flat(f"log_tau_T_h_{lbl}")))   / 24)
            tau_tag = f"τ_EMC={tE_med:.1f}d τ_T={tT_med:.1f}d"
        headline_bits.append(
            f"α_{lbl}={a_med:.4f} β_{lbl}={b_med:.4f} {tau_tag} σ_obs_{lbl}={so_med*1e3:.1f}mHz"
        )
    kX_m = float(np.median(k_X_geo)); kY_m = float(np.median(k_Y_geo))
    headline = (f"<k_X>={kX_m:.3e} N/m  <k_Y>={kY_m:.3e} N/m  |  "
                + "  |  ".join(headline_bits))
    log(f"Headline: {headline}")

    # ── Save posterior.json ──
    result = {
        "step": "level2_mdof_nuts",
        "sampler": f"NumPyro NUTS {num_chains}×{num_warmup}w+{num_samples}s (sequential)",
        "n_draws_total": num_chains * num_samples,
        "chain_diagnostics": {
            "rhat_max": rhat_max,
            "ess_min":  ess_min,
            "elapsed_s": round(elapsed, 1),
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
        },
        "config": {
            "M_central_kg":        M_CENTRAL,
            "k_anchor_X_Nm":       k_anchor_X,
            "k_anchor_Y_Nm":       k_anchor_Y,
            "f_med_mode1_Hz":      f_meds[0],
            "f_med_mode2_Hz":      f_meds[1],
            "sigma_log_k":         SIGMA_LOG_K,
            "sigma_profile":       SIGMA_PROFILE,
            "rng_seed":            rng_seed,
            "fixed_tau":           fixed_tau,
            "fixed_tau_values":    fixed_tau_vals,
        },
        "param_summary": summary_rows,
        "headline": headline,
        "notes": (
            f"Level 2 NUTS. N_STOREYS={N_STOREYS} (7 DOFs at floor levels FL2–FL8, fixed base at FL1). "
            "M_central=100t/storey fixed. "
            "Modes 1 (X-translation) + 2 (Y-translation). "
            + ("Two-stage: tau fixed to Level 1 medians (fixed_tau=True)."
               if fixed_tau else "Full 28-param joint model.")
        ),
    }

    out_json = out_dir / "posterior.json"
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    log(f"Wrote {out_json}")

    # ── Sanity checklist ──
    rng_np = np.random.default_rng(rng_seed + 1)
    sanity = evaluate_sanity(idata, k_anchor_X, k_anchor_Y, f_meds,
                             fixed_tau=fixed_tau, fixed_tau_vals=fixed_tau_vals,
                             compact=compact)
    write_sanity_md(out_dir / "sanity_checklist.md", sanity)
    status = "PASS" if sanity["n_passed"] == sanity["n_total"] else \
             ("PASS-WITH-NOTE" if sanity["n_passed"] >= sanity["n_total"] - 2 else "FAIL")
    log(f"Sanity: {sanity['n_passed']}/{sanity['n_total']} passed → {status}")

    # ── Decision file ──
    decision_lines = [
        f"# Level 2 — Dimensional MDOF (NUTS) — Decision\n",
        f"**Status:** {status}\n",
        f"**Headline:** {headline}\n",
        f"**Rhat_max:** {rhat_max:.4f}   **ESS_min:** {ess_min:.0f}   **Time:** {elapsed/60:.1f} min\n",
        f"**Sampler:** NumPyro NUTS, {num_chains} chains × ({num_warmup}w + {num_samples}s)\n",
        f"\n## Sanity: {sanity['n_passed']}/{sanity['n_total']}\n",
    ]
    for c in sanity["checks"]:
        mark = "✓" if c["passed"] else "✗"
        decision_lines.append(f"- {mark} {c['name']}: {c['value']}")
    (out_dir / "LEVEL_2_decision.md").write_text("\n".join(decision_lines))

    # ── Build flat samples dict for posterior predictive ──
    samples_dict = {}
    for pn in param_names:
        samples_dict[pn] = _flat(pn)

    # ── Diagnostic figure ──
    # For posterior predictive, need the lag series used for sampling
    emc_mats_for_fig = emc_lags_np if fixed_tau else [d["emc_mat"] for d in per_mode_train]
    t_mats_for_fig   = t_lags_np   if fixed_tau else [d["t_mat"]   for d in per_mode_train]

    fig = build_diagnostic_figure(
        idata,
        per_mode_train_dfs=[d["df"] for d in per_mode_train],
        emc_mats_train    =emc_mats_for_fig,
        t_mats_train      =t_mats_for_fig,
        samples_dict=samples_dict,
        f_med=f_meds,
        rng=rng_np,
        fixed_tau=fixed_tau,
        fixed_tau_vals=fixed_tau_vals,
        compact=compact,
    )
    fig_path = out_dir / "diagnostic.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {fig_path}")

    # ── Copy figure to output/ ──
    import shutil
    out_fig = ROOT / "analysis" / "output" / "Fig_Level2_NUTS_Diagnostic.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fig_path, out_fig)
    log(f"Copied to {out_fig}")

    log(f"=== level2_mdof_nuts END ===")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup",        type=int,  default=400)
    ap.add_argument("--samples",       type=int,  default=400)
    ap.add_argument("--chains",        type=int,  default=4)
    ap.add_argument("--seed",          type=int,  default=20260608)
    ap.add_argument("--out-dir",       type=str,  default=None)
    ap.add_argument("--no-fixed-tau",  action="store_true",
                    help="Use full 28-param joint model (requires more warmup)")
    ap.add_argument("--no-compact",    action="store_true",
                    help="Use per-storey profile model (8+8 params; needs Mode 3+ data)")
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent.parent / "aux" / "level2_mdof_nuts"
    )
    result = run(out, num_warmup=args.warmup, num_samples=args.samples,
                 num_chains=args.chains, rng_seed=args.seed,
                 fixed_tau=not args.no_fixed_tau,
                 compact=not args.no_compact)
    print("\n=== HEADLINE ===")
    print(result["headline"])
