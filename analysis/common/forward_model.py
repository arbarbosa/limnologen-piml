"""Forward models for the analysis.
by Andre R. Barbosa, April - October 2026

The adimensional model (Level 1) is implemented here; the dimensional MDOF
eigenvalue forward map is added in Level 2.

Adimensional model (Model 1)
----------------------------
For each mode i in {1, 2, 3} and time t:

    f_hat_i(t) = f_ref_i * sqrt(1 + alpha_i * EMC_tilde_tau_EMC(t)
                                  + beta_i  * T_tilde_tau_T(t))

where:
    - EMC = Equivalent Moisture Content of the outdoor air (derived from
            T and RH via Hailwood-Horrobin / Simpson formula in the
            SMHI-aligned data).
    - EMC_tilde_tau_EMC(t) = first-order low-pass filter of EMC(s) with time
            constant tau_EMC. Represents how far back the EMC history is
            integrated into the building's current effective moisture context.
    - T_tilde_tau_T(t)     = same for outdoor temperature with time
            constant tau_T.

alpha_i = alpha_eff_i is the *effective* EMC sensitivity, conflating stiffness
and mass contributions (see split model below).

Stiffness/mass split model (Model 2)
-------------------------------------
Separates the EMC effect into two physical mechanisms:

    f_hat_i(t) = f_ref_i * sqrt(
                    (1 + alpha_k_i * EMC_tilde_tau_EMC(t) + beta_i * T_tilde_tau_T(t))
                    / (1 + alpha_m * EMC_tilde_tau_EMC(t))
                 )

where:
    - alpha_k_i  = stiffness-EMC sensitivity for mode i (per %EMC).
                   Captures both MOE softening (negative contribution) and
                   swelling-induced connection stiffening (positive). In CLT
                   buildings, the latter tends to dominate, giving alpha_k_i > 0.
    - alpha_m    = mass-EMC sensitivity (shared across modes, per %EMC).
                   Wood mass increases ~1 % per %MC increase; for a building
                   with CLT mass fraction r_CLT, alpha_m ≈ r_CLT * 0.009.
                   Always positive; bounded by wood science.
    - alpha_eff_i ≈ alpha_k_i - alpha_m  (first-order Taylor approximation).

The parameters alpha_k_i and alpha_m are not separately identifiable from
frequency data alone. Identification requires an informative prior on alpha_m
from wood science (see priors.SplitPriorConfig). This is the Bayesian
identifiability-with-physics approach: the prior on alpha_m acts as an
external measurement, enabling decomposition into the two mechanisms.

We interpolate the lag values linearly in log(tau) over the 14 pre-computed
columns in the aligned CSV.

This is fast and gives < 1 % interpolation error against on-the-fly
exponential moving averages.
"""

from __future__ import annotations

import numpy as np

from analysis.common.data_loader import LAG_TAU_HOURS

# Pre-computed once: log of the lag-tau grid
LOG_LAG_TAU_HOURS = np.log(LAG_TAU_HOURS)


def interp_lag(log_tau: float, lag_matrix: np.ndarray) -> np.ndarray:
    """Evaluate the lag-filtered series at continuous log(tau).

    Linear interpolation in log(tau) over the 14-column pre-computed lag grid.
    Edge clamping at the grid extremes.

    Parameters
    ----------
    log_tau : float
        log(tau) in hours.
    lag_matrix : (N, 14) array
        Pre-computed EMC_tau or T_tau values at the 14 grid points.

    Returns
    -------
    (N,) array of interpolated values at log_tau.
    """
    # Clamp to grid range
    log_tau = max(LOG_LAG_TAU_HOURS[0], min(LOG_LAG_TAU_HOURS[-1], float(log_tau)))
    # Locate segment
    idx_hi = int(np.searchsorted(LOG_LAG_TAU_HOURS, log_tau, side="right"))
    idx_hi = max(1, min(len(LOG_LAG_TAU_HOURS) - 1, idx_hi))
    idx_lo = idx_hi - 1
    x_lo = LOG_LAG_TAU_HOURS[idx_lo]
    x_hi = LOG_LAG_TAU_HOURS[idx_hi]
    w = (log_tau - x_lo) / (x_hi - x_lo) if x_hi > x_lo else 0.0
    return (1.0 - w) * lag_matrix[:, idx_lo] + w * lag_matrix[:, idx_hi]


def adim_forward_single_mode(
    f_ref: float,
    alpha: float,
    beta: float,
    log_tau_EMC_h: float,
    log_tau_T_h: float,
    emc_lag_matrix: np.ndarray,
    t_lag_matrix: np.ndarray,
) -> np.ndarray:
    """Compute f_hat_i(t) for one mode given its parameters.

    Returns
    -------
    (N,) array of predicted frequencies.
    """
    emc_lag = interp_lag(log_tau_EMC_h, emc_lag_matrix)
    t_lag = interp_lag(log_tau_T_h, t_lag_matrix)
    one_plus = 1.0 + alpha * emc_lag + beta * t_lag
    # Numerical safety: clip below at small positive to avoid sqrt of negative
    # during MCMC excursions; should never be triggered at the posterior mode.
    one_plus = np.maximum(one_plus, 1e-6)
    return f_ref * np.sqrt(one_plus)


def log_likelihood_adim(
    theta: np.ndarray,
    per_mode_data: list[dict],
) -> float:
    """Compute the joint log-likelihood across the three modes.

    Parameters
    ----------
    theta : array (14,)
        Parameter vector (see analysis.common.priors).
    per_mode_data : list of length 3
        Each entry a dict with keys {f_obs, emc_lag_matrix, t_lag_matrix}.

    Returns
    -------
    Scalar log-likelihood.
    """
    log_tau_EMC_h_val = theta[12]
    log_tau_T_h_val = theta[13]
    ll_total = 0.0
    for i, d in enumerate(per_mode_data):
        base = 4 * i
        f_ref = theta[base + 0]
        alpha = theta[base + 1]
        beta = theta[base + 2]
        log_sigma_obs = theta[base + 3]
        sigma_obs = np.exp(log_sigma_obs)
        f_hat = adim_forward_single_mode(
            f_ref, alpha, beta, log_tau_EMC_h_val, log_tau_T_h_val,
            d["emc_lag_matrix"], d["t_lag_matrix"],
        )
        resid = d["f_obs"] - f_hat
        # Normal log-likelihood, vectorized
        ll_i = (
            -0.5 * resid.size * np.log(2.0 * np.pi)
            - resid.size * log_sigma_obs
            - 0.5 * np.sum((resid / sigma_obs) ** 2)
        )
        ll_total += ll_i
    return float(ll_total)


def log_likelihood_adim_per_mode(
    theta: np.ndarray,
    per_mode_data: list[dict],
) -> float:
    """Joint log-likelihood for the per-mode-tau variant (18 params).

    Layout: for each mode i ∈ {0,1,2}, theta[6i:6i+6] =
        [f_ref_i, alpha_i, beta_i, log_sigma_obs_i, log_tau_EMC_h_i, log_tau_T_h_i]
    """
    ll_total = 0.0
    for i, d in enumerate(per_mode_data):
        base = 6 * i
        f_ref = theta[base + 0]
        alpha = theta[base + 1]
        beta = theta[base + 2]
        log_sigma_obs = theta[base + 3]
        log_tau_EMC_h_i = theta[base + 4]
        log_tau_T_h_i = theta[base + 5]
        sigma_obs = np.exp(log_sigma_obs)
        f_hat = adim_forward_single_mode(
            f_ref, alpha, beta, log_tau_EMC_h_i, log_tau_T_h_i,
            d["emc_lag_matrix"], d["t_lag_matrix"],
        )
        resid = d["f_obs"] - f_hat
        ll_i = (
            -0.5 * resid.size * np.log(2.0 * np.pi)
            - resid.size * log_sigma_obs
            - 0.5 * np.sum((resid / sigma_obs) ** 2)
        )
        ll_total += ll_i
    return float(ll_total)


# ----------------------------------------------------------------------------
# Dimensional MDOF model (Level 2)
# ----------------------------------------------------------------------------

from scipy.linalg import eigh as _scipy_eigh


def build_shear_K(k: np.ndarray) -> np.ndarray:
    """Build an n-DOF shear-stack tridiagonal stiffness matrix.

    k[0]   is the ground spring (between fixed base and storey 1).
    k[1..n-1] are the inter-storey springs (between storey j and storey j+1).

    For storey 1 (idx 0):
      K[0,0] = k[0] + k[1]
      K[0,1] = K[1,0] = -k[1]
    For middle storeys j (idx 1..n-2):
      K[j,j] = k[j] + k[j+1]
      K[j,j+1] = K[j+1,j] = -k[j+1]
    For top storey (idx n-1):
      K[n-1,n-1] = k[n-1]

    Works for any n ≥ 1. Hus 6 uses n=7 (FL2–FL8, Story 1 concrete podium as fixed base).
    """
    n = len(k)
    K = np.zeros((n, n))
    K[0, 0] = k[0] + k[1]
    K[0, 1] = K[1, 0] = -k[1]
    for j in range(1, n - 1):
        K[j, j] = k[j] + k[j + 1]
        K[j, j + 1] = K[j + 1, j] = -k[j + 1]
    K[n - 1, n - 1] = k[n - 1]
    return K


def first_natural_frequency(k: np.ndarray, m: np.ndarray) -> float:
    """Smallest natural frequency (Hz) of the n-DOF shear stack.

    Solves generalized eigenvalue problem (K - omega^2 M) v = 0.
    Works for any n; Hus 6 uses n=7.
    """
    K = build_shear_K(k)
    M = np.diag(m)
    eigenvals = _scipy_eigh(K, M, eigvals_only=True)
    omega_sq = eigenvals[0]  # smallest
    if omega_sq <= 0:
        return float("nan")
    return float(np.sqrt(omega_sq) / (2 * np.pi))


def dim_forward_two_modes(theta: np.ndarray,
                           emc_lag_matrix_per_mode: list[np.ndarray],
                           t_lag_matrix_per_mode: list[np.ndarray]) -> list[np.ndarray]:
    """Compute f_hat(t) per mode for the dimensional 2-mode model.

    theta layout (37 params): see DIM_PARAM_NAMES in priors.
    Returns: list of 2 arrays, [f_hat_1, f_hat_2] of length N_i per mode.
    """
    # Unpack stiffnesses
    k_X = np.exp(theta[0:8])
    k_Y = np.exp(theta[8:16])
    # Unpack mass
    M_central = float(np.exp(theta[16]))
    eta = theta[19:27]
    m = M_central * np.exp(eta)  # 8-vector
    # First natural frequencies (one per mode)
    f0_X = first_natural_frequency(k_X, m)
    f0_Y = first_natural_frequency(k_Y, m)
    out = []
    for i in range(2):
        base = 27 + 5 * i
        alpha = theta[base + 0]
        beta = theta[base + 1]
        log_tau_EMC = theta[base + 3]
        log_tau_T = theta[base + 4]
        f0_i = f0_X if i == 0 else f0_Y
        emc_lag = interp_lag(log_tau_EMC, emc_lag_matrix_per_mode[i])
        t_lag = interp_lag(log_tau_T, t_lag_matrix_per_mode[i])
        one_plus = 1.0 + alpha * emc_lag + beta * t_lag
        one_plus = np.maximum(one_plus, 1e-6)
        f_hat_i = f0_i * np.sqrt(one_plus)
        out.append(f_hat_i)
    return out


def dim_forward_two_modes_perstorey(theta: np.ndarray,
                                      emc_lag_matrix_per_mode: list[np.ndarray],
                                      t_lag_matrix_per_mode: list[np.ndarray],
                                      M_central_fixed: float = 100_000.0) -> list[np.ndarray]:
    """Forward map for the per-storey-k reparameterized model (28 params, M fixed)."""
    log_k_X_geo = theta[0]
    log_k_X_profile = theta[1:9]
    log_k_Y_geo = theta[9]
    log_k_Y_profile = theta[10:18]
    k_X_per_storey = np.exp(log_k_X_geo + log_k_X_profile)   # (8,)
    k_Y_per_storey = np.exp(log_k_Y_geo + log_k_Y_profile)   # (8,)
    m_uniform = np.full(8, M_central_fixed)
    f0_X = first_natural_frequency(k_X_per_storey, m_uniform)
    f0_Y = first_natural_frequency(k_Y_per_storey, m_uniform)
    out = []
    for i in range(2):
        base = 18 + 5 * i
        alpha = theta[base + 0]
        beta = theta[base + 1]
        log_tau_EMC = theta[base + 3]
        log_tau_T = theta[base + 4]
        f0_i = f0_X if i == 0 else f0_Y
        emc_lag = interp_lag(log_tau_EMC, emc_lag_matrix_per_mode[i])
        t_lag = interp_lag(log_tau_T, t_lag_matrix_per_mode[i])
        one_plus = np.maximum(1.0 + alpha * emc_lag + beta * t_lag, 1e-6)
        out.append(f0_i * np.sqrt(one_plus))
    return out


def log_likelihood_dim_two_modes_perstorey(theta: np.ndarray,
                                             per_mode_data: list[dict],
                                             M_central_fixed: float = 100_000.0) -> float:
    """Joint log-likelihood for the per-storey-k reparameterized model."""
    try:
        f_hat_list = dim_forward_two_modes_perstorey(
            theta,
            emc_lag_matrix_per_mode=[d["emc_lag_matrix"] for d in per_mode_data],
            t_lag_matrix_per_mode=[d["t_lag_matrix"] for d in per_mode_data],
            M_central_fixed=M_central_fixed,
        )
    except Exception:
        return -np.inf
    ll = 0.0
    for i in range(2):
        f_hat = f_hat_list[i]
        if not np.all(np.isfinite(f_hat)):
            return -np.inf
        log_sigma_obs = theta[18 + 5 * i + 2]
        sigma_obs = np.exp(log_sigma_obs)
        resid = per_mode_data[i]["f_obs"] - f_hat
        ll_i = (
            -0.5 * resid.size * np.log(2.0 * np.pi)
            - resid.size * log_sigma_obs
            - 0.5 * np.sum((resid / sigma_obs) ** 2)
        )
        ll += ll_i
    return float(ll)


def posterior_predictive_dim_two_modes_perstorey(theta_draws: np.ndarray,
                                                   emc_lag_matrix: np.ndarray,
                                                   t_lag_matrix: np.ndarray,
                                                   mode_idx: int,
                                                   include_obs_noise: bool = True,
                                                   rng: np.random.Generator | None = None,
                                                   M_central_fixed: float = 100_000.0) -> np.ndarray:
    """Posterior predictive for the per-storey-k variant."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_draws = theta_draws.shape[0]
    N = emc_lag_matrix.shape[0]
    out = np.empty((n_draws, N))
    for k in range(n_draws):
        theta = theta_draws[k]
        if mode_idx == 0:
            k_per_storey = np.exp(theta[0] + theta[1:9])
        else:
            k_per_storey = np.exp(theta[9] + theta[10:18])
        f0 = first_natural_frequency(k_per_storey, np.full(8, M_central_fixed))
        base = 18 + 5 * mode_idx
        alpha = theta[base + 0]
        beta = theta[base + 1]
        log_tau_EMC = theta[base + 3]
        log_tau_T = theta[base + 4]
        emc_lag = interp_lag(log_tau_EMC, emc_lag_matrix)
        t_lag = interp_lag(log_tau_T, t_lag_matrix)
        one_plus = np.maximum(1.0 + alpha * emc_lag + beta * t_lag, 1e-6)
        f_hat = f0 * np.sqrt(one_plus)
        if include_obs_noise:
            sigma_obs = np.exp(theta[base + 2])
            f_hat = f_hat + rng.normal(0.0, sigma_obs, size=N)
        out[k] = f_hat
    return out


def dim_forward_two_modes_uniform(theta: np.ndarray,
                                    emc_lag_matrix_per_mode: list[np.ndarray],
                                    t_lag_matrix_per_mode: list[np.ndarray]) -> list[np.ndarray]:
    """Forward map for the uniform-k dimensional 2-mode model (13 params).

    theta = [log_k_X, log_k_Y, log_M_central,
             alpha_1, beta_1, log_sigma_obs_1, log_tau_EMC_h_1, log_tau_T_h_1,
             alpha_2, beta_2, log_sigma_obs_2, log_tau_EMC_h_2, log_tau_T_h_2]
    """
    k_X = float(np.exp(theta[0]))
    k_Y = float(np.exp(theta[1]))
    M_central = float(np.exp(theta[2]))
    k_uniform = np.full(8, np.nan)  # placeholder
    m_uniform = np.full(8, M_central)
    # First natural frequencies via eigh()
    f0_X = first_natural_frequency(np.full(8, k_X), m_uniform)
    f0_Y = first_natural_frequency(np.full(8, k_Y), m_uniform)
    out = []
    for i in range(2):
        base = 3 + 5 * i
        alpha = theta[base + 0]
        beta = theta[base + 1]
        log_tau_EMC = theta[base + 3]
        log_tau_T = theta[base + 4]
        f0_i = f0_X if i == 0 else f0_Y
        emc_lag = interp_lag(log_tau_EMC, emc_lag_matrix_per_mode[i])
        t_lag = interp_lag(log_tau_T, t_lag_matrix_per_mode[i])
        one_plus = 1.0 + alpha * emc_lag + beta * t_lag
        one_plus = np.maximum(one_plus, 1e-6)
        out.append(f0_i * np.sqrt(one_plus))
    return out


def log_likelihood_dim_two_modes_uniform(theta: np.ndarray,
                                          per_mode_data: list[dict]) -> float:
    """Joint log-likelihood for the uniform-k dimensional 2-mode model."""
    try:
        f_hat_list = dim_forward_two_modes_uniform(
            theta,
            emc_lag_matrix_per_mode=[d["emc_lag_matrix"] for d in per_mode_data],
            t_lag_matrix_per_mode=[d["t_lag_matrix"] for d in per_mode_data],
        )
    except Exception:
        return -np.inf
    ll = 0.0
    for i in range(2):
        f_hat = f_hat_list[i]
        if not np.all(np.isfinite(f_hat)):
            return -np.inf
        log_sigma_obs = theta[3 + 5 * i + 2]
        sigma_obs = np.exp(log_sigma_obs)
        resid = per_mode_data[i]["f_obs"] - f_hat
        ll_i = (
            -0.5 * resid.size * np.log(2.0 * np.pi)
            - resid.size * log_sigma_obs
            - 0.5 * np.sum((resid / sigma_obs) ** 2)
        )
        ll += ll_i
    return float(ll)


def posterior_predictive_dim_two_modes_uniform(theta_draws: np.ndarray,
                                                emc_lag_matrix: np.ndarray,
                                                t_lag_matrix: np.ndarray,
                                                mode_idx: int,
                                                include_obs_noise: bool = True,
                                                rng: np.random.Generator | None = None) -> np.ndarray:
    """Posterior predictive for the uniform-k variant."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_draws = theta_draws.shape[0]
    N = emc_lag_matrix.shape[0]
    out = np.empty((n_draws, N))
    for k in range(n_draws):
        theta = theta_draws[k]
        k_dir = float(np.exp(theta[0] if mode_idx == 0 else theta[1]))
        M_central = float(np.exp(theta[2]))
        f0 = first_natural_frequency(np.full(8, k_dir), np.full(8, M_central))
        base = 3 + 5 * mode_idx
        alpha = theta[base + 0]
        beta = theta[base + 1]
        log_tau_EMC = theta[base + 3]
        log_tau_T = theta[base + 4]
        emc_lag = interp_lag(log_tau_EMC, emc_lag_matrix)
        t_lag = interp_lag(log_tau_T, t_lag_matrix)
        one_plus = np.maximum(1.0 + alpha * emc_lag + beta * t_lag, 1e-6)
        f_hat = f0 * np.sqrt(one_plus)
        if include_obs_noise:
            sigma_obs = np.exp(theta[base + 2])
            f_hat = f_hat + rng.normal(0.0, sigma_obs, size=N)
        out[k] = f_hat
    return out


def log_likelihood_dim_two_modes(theta: np.ndarray,
                                  per_mode_data: list[dict]) -> float:
    """Joint log-likelihood for the dimensional 2-mode model."""
    # Sanity: stiffness and mass must give a real frequency
    try:
        f_hat_list = dim_forward_two_modes(
            theta,
            emc_lag_matrix_per_mode=[d["emc_lag_matrix"] for d in per_mode_data],
            t_lag_matrix_per_mode=[d["t_lag_matrix"] for d in per_mode_data],
        )
    except Exception:
        return -np.inf
    ll = 0.0
    for i in range(2):
        f_hat = f_hat_list[i]
        if not np.all(np.isfinite(f_hat)):
            return -np.inf
        log_sigma_obs = theta[27 + 5 * i + 2]
        sigma_obs = np.exp(log_sigma_obs)
        resid = per_mode_data[i]["f_obs"] - f_hat
        ll_i = (
            -0.5 * resid.size * np.log(2.0 * np.pi)
            - resid.size * log_sigma_obs
            - 0.5 * np.sum((resid / sigma_obs) ** 2)
        )
        ll += ll_i
    return float(ll)


def posterior_predictive_dim_two_modes(theta_draws: np.ndarray,
                                        emc_lag_matrix: np.ndarray,
                                        t_lag_matrix: np.ndarray,
                                        mode_idx: int,
                                        include_obs_noise: bool = True,
                                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate posterior predictive draws for one mode of the dimensional model."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_draws = theta_draws.shape[0]
    N = emc_lag_matrix.shape[0]
    out = np.empty((n_draws, N))
    for k in range(n_draws):
        theta = theta_draws[k]
        # Compute mode i's predicted frequency
        k_per_dir = np.exp(theta[0:8] if mode_idx == 0 else theta[8:16])
        M_central = float(np.exp(theta[16]))
        eta = theta[19:27]
        m = M_central * np.exp(eta)
        f0 = first_natural_frequency(k_per_dir, m)
        base = 27 + 5 * mode_idx
        alpha = theta[base + 0]
        beta = theta[base + 1]
        log_tau_EMC = theta[base + 3]
        log_tau_T = theta[base + 4]
        emc_lag = interp_lag(log_tau_EMC, emc_lag_matrix)
        t_lag = interp_lag(log_tau_T, t_lag_matrix)
        one_plus = np.maximum(1.0 + alpha * emc_lag + beta * t_lag, 1e-6)
        f_hat = f0 * np.sqrt(one_plus)
        if include_obs_noise:
            sigma_obs = np.exp(theta[base + 2])
            f_hat = f_hat + rng.normal(0.0, sigma_obs, size=N)
        out[k] = f_hat
    return out


def posterior_predictive_adim(
    theta_draws: np.ndarray,        # (n_draws, 14) or (n_draws, 18)
    emc_lag_matrix: np.ndarray,     # (N, 14)
    t_lag_matrix: np.ndarray,       # (N, 14)
    mode_idx: int,                  # 0, 1, 2
    include_obs_noise: bool = True,
    rng: np.random.Generator | None = None,
    per_mode_tau: bool = False,
) -> np.ndarray:
    """Generate posterior predictive draws of f_i(t).

    Supports both shared-tau (14 param) and per-mode-tau (18 param) layouts.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_draws = theta_draws.shape[0]
    N = emc_lag_matrix.shape[0]
    out = np.empty((n_draws, N))
    if per_mode_tau:
        base = 6 * mode_idx
        for k in range(n_draws):
            theta = theta_draws[k]
            f_hat = adim_forward_single_mode(
                theta[base + 0], theta[base + 1], theta[base + 2],
                theta[base + 4], theta[base + 5],
                emc_lag_matrix, t_lag_matrix,
            )
            if include_obs_noise:
                sigma_obs = np.exp(theta[base + 3])
                f_hat = f_hat + rng.normal(0.0, sigma_obs, size=N)
            out[k] = f_hat
    else:
        base = 4 * mode_idx
        for k in range(n_draws):
            theta = theta_draws[k]
            f_hat = adim_forward_single_mode(
                theta[base + 0], theta[base + 1], theta[base + 2],
                theta[12], theta[13],
                emc_lag_matrix, t_lag_matrix,
            )
            if include_obs_noise:
                sigma_obs = np.exp(theta[base + 3])
                f_hat = f_hat + rng.normal(0.0, sigma_obs, size=N)
            out[k] = f_hat
    return out


# ============================================================================
# Stiffness / mass split model (Model 2)
# ============================================================================

def split_forward_single_mode(
    f_ref: float,
    alpha_k: float,
    alpha_m: float,
    beta: float,
    log_tau_EMC_h: float,
    log_tau_T_h: float,
    emc_lag_matrix: np.ndarray,
    t_lag_matrix: np.ndarray,
) -> np.ndarray:
    """Exact stiffness/mass-split forward model for one mode.

    f_hat(t) = f_ref * sqrt( (1 + alpha_k*EMC_tau + beta*T_tau)
                             / (1 + alpha_m*EMC_tau) )

    Parameters
    ----------
    f_ref : float
        Reference natural frequency (Hz).  Absorbs steady-state ratio
        sqrt(k_0 / m_0) / (2*pi) at the dataset's reference EMC.
    alpha_k : float
        Stiffness EMC sensitivity (per %EMC).  Positive → stiffness
        increases with EMC (e.g. swelling-induced connection stiffening).
        Negative → stiffness decreases (MOE softening dominates).
    alpha_m : float
        Mass EMC sensitivity (per %EMC, shared across modes).  Always
        positive: wood mass increases ~1 % per %MC increase.  Bounded
        from above by the CLT mass fraction in the floor assembly
        (~r_CLT * 0.009 per %EMC for Norway spruce CLT).
    beta : float
        Temperature sensitivity (per °C).  Captures stiffness change
        with temperature; thermal mass change is negligible (beta_m ≈ 0).
    log_tau_EMC_h : float
        log(tau_EMC) in hours — time constant of the EMC low-pass filter.
    log_tau_T_h : float
        log(tau_T) in hours — time constant of the temperature low-pass filter.
    emc_lag_matrix : (N, 14) array
        Pre-computed exponential-moving-average of EMC at the 14 lag-tau
        grid points.  Values are in absolute %EMC.
    t_lag_matrix : (N, 14) array
        Same for outdoor temperature (°C).

    Returns
    -------
    (N,) array of predicted natural frequencies in Hz.
    """
    emc_lag = interp_lag(log_tau_EMC_h, emc_lag_matrix)   # absolute %EMC
    t_lag   = interp_lag(log_tau_T_h,   t_lag_matrix)     # °C
    numerator   = 1.0 + alpha_k * emc_lag + beta * t_lag
    denominator = 1.0 + alpha_m * emc_lag
    # Safety clips: avoid sqrt of zero or negative in MCMC excursions
    numerator   = np.maximum(numerator,   1e-6)
    denominator = np.maximum(denominator, 1e-6)
    return f_ref * np.sqrt(numerator / denominator)


def alpha_eff_from_split(
    alpha_k: float | np.ndarray,
    alpha_m: float | np.ndarray,
    emc_ref: float = 19.0,
) -> np.ndarray:
    """First-order approximation of alpha_eff from the split parameters.

    alpha_eff ≈ (alpha_k - alpha_m) / (1 + alpha_m * emc_ref)

    The denominator correction accounts for the nonlinear interaction at
    the reference EMC level.  For emc_ref ≈ 19 % and alpha_m ≈ 0.005,
    the correction is ~1/(1.095) ≈ 4.5 % — modest but non-negligible.

    Parameters
    ----------
    alpha_k, alpha_m : float or array
        Stiffness and mass EMC sensitivities (per %EMC).
    emc_ref : float
        Reference absolute EMC (%); default is the dataset mean (~19 %).

    Returns
    -------
    alpha_eff (float or array, same shape as inputs).
    """
    return (np.asarray(alpha_k) - np.asarray(alpha_m)) / (1.0 + np.asarray(alpha_m) * emc_ref)


def log_likelihood_split_per_mode(
    theta: np.ndarray,
    per_mode_data: list[dict],
) -> float:
    """Joint log-likelihood for the stiffness/mass split model (19 params).

    Parameter layout:
        For each mode i in {0,1,2}, theta[7*i : 7*i+7] =
            [f_ref_i, alpha_k_i, beta_i, log_sigma_obs_i,
             log_tau_EMC_h_i, log_tau_T_h_i,   (per-mode tau)
             <placeholder — alpha_m is at index 18>]
        theta[18] = alpha_m  (shared mass sensitivity)

    Concretely:
        theta[0]  = f_ref_1
        theta[1]  = alpha_k_1
        theta[2]  = beta_1
        theta[3]  = log_sigma_obs_1
        theta[4]  = log_tau_EMC_h_1
        theta[5]  = log_tau_T_h_1
        theta[6]  = f_ref_2
        theta[7]  = alpha_k_2
        theta[8]  = beta_2
        theta[9]  = log_sigma_obs_2
        theta[10] = log_tau_EMC_h_2
        theta[11] = log_tau_T_h_2
        theta[12] = f_ref_3
        theta[13] = alpha_k_3
        theta[14] = beta_3
        theta[15] = log_sigma_obs_3
        theta[16] = log_tau_EMC_h_3
        theta[17] = log_tau_T_h_3
        theta[18] = alpha_m   (shared)
    Total: 19 parameters.
    """
    alpha_m = theta[18]
    ll_total = 0.0
    for i, d in enumerate(per_mode_data):
        base = 6 * i
        f_ref         = theta[base + 0]
        alpha_k       = theta[base + 1]
        beta          = theta[base + 2]
        log_sigma_obs = theta[base + 3]
        log_tau_EMC   = theta[base + 4]
        log_tau_T     = theta[base + 5]
        sigma_obs = np.exp(log_sigma_obs)
        f_hat = split_forward_single_mode(
            f_ref, alpha_k, alpha_m, beta, log_tau_EMC, log_tau_T,
            d["emc_lag_matrix"], d["t_lag_matrix"],
        )
        resid = d["f_obs"] - f_hat
        ll_i = (
            -0.5 * resid.size * np.log(2.0 * np.pi)
            - resid.size * log_sigma_obs
            - 0.5 * np.sum((resid / sigma_obs) ** 2)
        )
        ll_total += ll_i
    return float(ll_total)


def posterior_predictive_split(
    theta_draws: np.ndarray,
    emc_lag_matrix: np.ndarray,
    t_lag_matrix: np.ndarray,
    mode_idx: int,
    include_obs_noise: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Posterior predictive draws for one mode of the split model.

    Parameters
    ----------
    theta_draws : (n_draws, 19) array
        Posterior samples from the split model (19-param layout).
    emc_lag_matrix, t_lag_matrix : (N, 14) arrays
    mode_idx : int
        0, 1, or 2.
    include_obs_noise : bool
        If True, add N(0, sigma_obs) noise to each draw.
    rng : numpy Generator or None

    Returns
    -------
    (n_draws, N) array of predictive samples.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_draws = theta_draws.shape[0]
    N = emc_lag_matrix.shape[0]
    out = np.empty((n_draws, N))
    for k in range(n_draws):
        theta = theta_draws[k]
        base = 6 * mode_idx
        f_hat = split_forward_single_mode(
            theta[base + 0],  # f_ref
            theta[base + 1],  # alpha_k
            theta[18],        # alpha_m (shared)
            theta[base + 2],  # beta
            theta[base + 4],  # log_tau_EMC_h
            theta[base + 5],  # log_tau_T_h
            emc_lag_matrix,
            t_lag_matrix,
        )
        if include_obs_noise:
            sigma_obs = np.exp(theta[base + 3])
            f_hat = f_hat + rng.normal(0.0, sigma_obs, size=N)
        out[k] = f_hat
    return out


# Parameter names for the split model
SPLIT_PARAM_NAMES = [
    "f_ref_1", "alpha_k_1", "beta_1", "log_sigma_obs_1", "log_tau_EMC_h_1", "log_tau_T_h_1",
    "f_ref_2", "alpha_k_2", "beta_2", "log_sigma_obs_2", "log_tau_EMC_h_2", "log_tau_T_h_2",
    "f_ref_3", "alpha_k_3", "beta_3", "log_sigma_obs_3", "log_tau_EMC_h_3", "log_tau_T_h_3",
    "alpha_m",
]
SPLIT_PARAM_LATEX = [
    r"$f_{\mathrm{ref},1}$", r"$\alpha_{k,1}$", r"$\beta_1$", r"$\log\sigma_{\mathrm{obs},1}$",
    r"$\log\tau_{\mathrm{EMC},1}$", r"$\log\tau_{T,1}$",
    r"$f_{\mathrm{ref},2}$", r"$\alpha_{k,2}$", r"$\beta_2$", r"$\log\sigma_{\mathrm{obs},2}$",
    r"$\log\tau_{\mathrm{EMC},2}$", r"$\log\tau_{T,2}$",
    r"$f_{\mathrm{ref},3}$", r"$\alpha_{k,3}$", r"$\beta_3$", r"$\log\sigma_{\mathrm{obs},3}$",
    r"$\log\tau_{\mathrm{EMC},3}$", r"$\log\tau_{T,3}$",
    r"$\alpha_m$",
]
