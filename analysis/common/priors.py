"""JCSS-derived priors — single source of truth for the entire pipeline.
by Andre R. Barbosa, April - October 2026

References
----------
- JCSS Probabilistic Model Code, Part 3 (Timber).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


# ----------------------------------------------------------------------------
# Adimensional twin-model priors (Level 1)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class AdimPriorConfig:
    """Per-mode priors for the adimensional twin-model.

    All parameters are independent across modes; `tau_EMC_h` and `tau_T_h` are
    shared across modes in this configuration (CLT-material / envelope-filter
    constants).
    """

    # f_ref,i prior: Normal centered at first-month median of mode i (set at runtime)
    # We carry the prior sigma here.
    f_ref_sigma_hz: float = 0.05  # broad; data informs

    # alpha_i prior: Normal(+0.0020, 0.0010)  -- JCSS d(E)/du-derived, inversion-aware
    alpha_mean: float = 0.0020
    alpha_sigma: float = 0.0010

    # beta_i prior: Normal(-0.0025, 0.0010)   -- JCSS d(E)/dT-derived
    beta_mean: float = -0.0025
    beta_sigma: float = 0.0010

    # log sigma_obs,i prior: Normal(log 0.015 Hz, 0.30)
    # equivalent to LogNormal(log 0.015, 0.30) on sigma_obs itself
    log_sigma_obs_mean: float = float(np.log(0.015))
    log_sigma_obs_sigma: float = 0.30

    # log tau_EMC (hours) prior: Normal(log(45 d * 24 h), 0.40)
    # tau_EMC is the time constant of the low-pass filter on outdoor EMC
    # (Equivalent Moisture Content of the air, derived from T + RH).
    # equivalent to LogNormal(45 d, 0.40) on tau_EMC
    log_tau_EMC_h_mean: float = float(np.log(45.0 * 24.0))
    log_tau_EMC_h_sigma: float = 0.40

    # log tau_T (hours) prior: Normal(log(7 d * 24 h), 0.40)
    log_tau_T_h_mean: float = float(np.log(7.0 * 24.0))
    log_tau_T_h_sigma: float = 0.40

    def as_dict(self) -> dict:
        return asdict(self)


def normal_log_pdf(x: np.ndarray | float, mean: float, sigma: float) -> np.ndarray | float:
    """Log of a Normal(mean, sigma) pdf, vectorized."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mean) / sigma) ** 2


def log_prior_adim(theta: np.ndarray, f_ref_mean: np.ndarray, cfg: AdimPriorConfig) -> float:
    """Compute log p(theta) for the adimensional twin-model.

    Parameter ordering (length 14):
      [f_ref_1, alpha_1, beta_1, log_sigma_1,
       f_ref_2, alpha_2, beta_2, log_sigma_2,
       f_ref_3, alpha_3, beta_3, log_sigma_3,
       log_tau_EMC_h, log_tau_T_h]

    Parameters
    ----------
    theta : array (14,)
    f_ref_mean : array (3,)   per-mode prior means for f_ref (Hz), set at runtime
                                from first-month medians.
    cfg : AdimPriorConfig
    """
    lp = 0.0
    for i in range(3):
        base = 4 * i
        f_ref = theta[base + 0]
        alpha = theta[base + 1]
        beta = theta[base + 2]
        log_sigma = theta[base + 3]
        lp += normal_log_pdf(f_ref, f_ref_mean[i], cfg.f_ref_sigma_hz)
        lp += normal_log_pdf(alpha, cfg.alpha_mean, cfg.alpha_sigma)
        lp += normal_log_pdf(beta, cfg.beta_mean, cfg.beta_sigma)
        lp += normal_log_pdf(log_sigma, cfg.log_sigma_obs_mean, cfg.log_sigma_obs_sigma)
    log_tau_EMC = theta[12]
    log_tau_T = theta[13]
    lp += normal_log_pdf(log_tau_EMC, cfg.log_tau_EMC_h_mean, cfg.log_tau_EMC_h_sigma)
    lp += normal_log_pdf(log_tau_T, cfg.log_tau_T_h_mean, cfg.log_tau_T_h_sigma)
    return float(lp)


# ----------------------------------------------------------------------------
# Parameter names (for output JSON, diagnostic plots)
# ----------------------------------------------------------------------------

ADIM_PARAM_NAMES = [
    "f_ref_1", "alpha_1", "beta_1", "log_sigma_obs_1",
    "f_ref_2", "alpha_2", "beta_2", "log_sigma_obs_2",
    "f_ref_3", "alpha_3", "beta_3", "log_sigma_obs_3",
    "log_tau_EMC_h", "log_tau_T_h",
]

ADIM_PARAM_LATEX = [
    r"$f_{\mathrm{ref},1}$", r"$\alpha_1$", r"$\beta_1$", r"$\log\sigma_{\mathrm{obs},1}$",
    r"$f_{\mathrm{ref},2}$", r"$\alpha_2$", r"$\beta_2$", r"$\log\sigma_{\mathrm{obs},2}$",
    r"$f_{\mathrm{ref},3}$", r"$\alpha_3$", r"$\beta_3$", r"$\log\sigma_{\mathrm{obs},3}$",
    r"$\log\tau_{\mathrm{EMC}}\,[\mathrm{h}]$", r"$\log\tau_T\,[\mathrm{h}]$",
]

# Per-mode tau variant (18 params): each mode has its own log_tau_EMC_h, log_tau_T_h
ADIM_PARAM_NAMES_PER_MODE = [
    "f_ref_1", "alpha_1", "beta_1", "log_sigma_obs_1", "log_tau_EMC_h_1", "log_tau_T_h_1",
    "f_ref_2", "alpha_2", "beta_2", "log_sigma_obs_2", "log_tau_EMC_h_2", "log_tau_T_h_2",
    "f_ref_3", "alpha_3", "beta_3", "log_sigma_obs_3", "log_tau_EMC_h_3", "log_tau_T_h_3",
]

ADIM_PARAM_LATEX_PER_MODE = [
    r"$f_{\mathrm{ref},1}$", r"$\alpha_1$", r"$\beta_1$", r"$\log\sigma_{\mathrm{obs},1}$",
        r"$\log\tau_{\mathrm{EMC},1}\,[\mathrm{h}]$", r"$\log\tau_{T,1}\,[\mathrm{h}]$",
    r"$f_{\mathrm{ref},2}$", r"$\alpha_2$", r"$\beta_2$", r"$\log\sigma_{\mathrm{obs},2}$",
        r"$\log\tau_{\mathrm{EMC},2}\,[\mathrm{h}]$", r"$\log\tau_{T,2}\,[\mathrm{h}]$",
    r"$f_{\mathrm{ref},3}$", r"$\alpha_3$", r"$\beta_3$", r"$\log\sigma_{\mathrm{obs},3}$",
        r"$\log\tau_{\mathrm{EMC},3}\,[\mathrm{h}]$", r"$\log\tau_{T,3}\,[\mathrm{h}]$",
]


# ----------------------------------------------------------------------------
# Dimensional MDOF model priors (Level 2)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class DimPriorConfig:
    """JCSS-derived priors for the dimensional MDOF model (Level 2).

    Model components:
    - Period-anchored prior on per-storey stiffness k_0,j (log-Normal)
    - Hierarchical compound-symmetric mass: M_central, sigma_floor, rho_M, eta_j
    - Per-mode environmental coupling (alpha_i, beta_i, sigma_obs_i, tau_EMC,i, tau_T,i)
    """
    # Stiffness prior: log k_0,j ~ Normal(log k_anchor, sigma_log_k)
    # k_anchor is set per direction at runtime from the first-month observed median.
    sigma_log_k: float = 0.30   # 30 % per-storey stiffness variability around the period-anchored mean

    # Mass priors (shared across X and Y modes — one mass distribution per building)
    log_M_central_mean: float = float(np.log(100_000.0))   # 100 t/storey (Reynolds apt A scale; will be refined)
    log_M_central_sigma: float = 0.20                       # ~20 % uncertainty on storey mass
    log_sigma_floor_mean: float = float(np.log(0.07))       # 7 % per-storey mass deviation
    log_sigma_floor_sigma: float = 0.40
    logit_rho_M_mean: float = 0.0                            # rho_M = sigmoid(0) = 0.5 (moderate partial pooling)
    logit_rho_M_sigma: float = 1.0                           # broad

    # Per-mode environmental priors (same per-mode-tau structure as adim)
    alpha_mean: float = 0.0020
    alpha_sigma: float = 0.0015                              # widened from 0.0010 since per-mode posteriors span a wider range
    beta_mean: float = -0.0025
    beta_sigma: float = 0.0015                               # widened similarly
    log_sigma_obs_mean: float = float(np.log(0.015))
    log_sigma_obs_sigma: float = 0.30
    log_tau_EMC_h_mean: float = float(np.log(45.0 * 24.0))
    log_tau_EMC_h_sigma: float = 0.50                        # widened to accommodate mode-specific differences
    log_tau_T_h_mean: float = float(np.log(7.0 * 24.0))
    log_tau_T_h_sigma: float = 0.60

    def as_dict(self) -> dict:
        return asdict(self)


# Parameter layout for the dimensional two-mode model:
#   [log_k_X_0, ..., log_k_X_7,                                      # 0..7   (8 X-direction storey stiffnesses)
#    log_k_Y_0, ..., log_k_Y_7,                                      # 8..15  (8 Y-direction storey stiffnesses)
#    log_M_central, log_sigma_floor, logit_rho_M,                    # 16, 17, 18
#    eta_0, ..., eta_7,                                              # 19..26 (per-storey log-mass deviations)
#    alpha_1, beta_1, log_sigma_obs_1, log_tau_EMC_h_1, log_tau_T_h_1,  # 27..31 (Mode 1 = X)
#    alpha_2, beta_2, log_sigma_obs_2, log_tau_EMC_h_2, log_tau_T_h_2]  # 32..36 (Mode 2 = Y)
DIM_PARAM_NAMES = (
    [f"log_k_X_{j}" for j in range(8)] +
    [f"log_k_Y_{j}" for j in range(8)] +
    ["log_M_central", "log_sigma_floor", "logit_rho_M"] +
    [f"eta_{j}" for j in range(8)] +
    ["alpha_1", "beta_1", "log_sigma_obs_1", "log_tau_EMC_h_1", "log_tau_T_h_1",
     "alpha_2", "beta_2", "log_sigma_obs_2", "log_tau_EMC_h_2", "log_tau_T_h_2"]
)

DIM_PARAM_LATEX = (
    [rf"$\log k_{{X,{j}}}$" for j in range(8)] +
    [rf"$\log k_{{Y,{j}}}$" for j in range(8)] +
    [r"$\log M_{\mathrm{c}}$", r"$\log \sigma_{\mathrm{fl}}$", r"$\mathrm{logit}\,\rho_M$"] +
    [rf"$\eta_{j}$" for j in range(8)] +
    [r"$\alpha_1$", r"$\beta_1$", r"$\log\sigma_{\mathrm{obs},1}$",
     r"$\log\tau_{\mathrm{EMC},1}$", r"$\log\tau_{T,1}$",
     r"$\alpha_2$", r"$\beta_2$", r"$\log\sigma_{\mathrm{obs},2}$",
     r"$\log\tau_{\mathrm{EMC},2}$", r"$\log\tau_{T,2}$"]
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Per-storey-k reparameterized variant of Level 2: geo-mean + tight profile, fixed M_central.
# Parameter layout (28 params):
#   [log_k_X_geo, log_k_X_profile_0..7,    # idx 0..8
#    log_k_Y_geo, log_k_Y_profile_0..7,    # idx 9..17
#    alpha_1, beta_1, log_sigma_obs_1, log_tau_EMC_h_1, log_tau_T_h_1,   # idx 18..22
#    alpha_2, beta_2, log_sigma_obs_2, log_tau_EMC_h_2, log_tau_T_h_2]   # idx 23..27
DIM_PERSTOREY_PARAM_NAMES = (
    ["log_k_X_geo"] + [f"log_k_X_profile_{j}" for j in range(8)] +
    ["log_k_Y_geo"] + [f"log_k_Y_profile_{j}" for j in range(8)] +
    ["alpha_1", "beta_1", "log_sigma_obs_1", "log_tau_EMC_h_1", "log_tau_T_h_1",
     "alpha_2", "beta_2", "log_sigma_obs_2", "log_tau_EMC_h_2", "log_tau_T_h_2"]
)
DIM_PERSTOREY_PARAM_LATEX = (
    [r"$\log \bar k_X$"] + [rf"$\log k_{{X,{j}}}^{{prof}}$" for j in range(8)] +
    [r"$\log \bar k_Y$"] + [rf"$\log k_{{Y,{j}}}^{{prof}}$" for j in range(8)] +
    [r"$\alpha_1$", r"$\beta_1$", r"$\log\sigma_{\mathrm{obs},1}$",
     r"$\log\tau_{\mathrm{EMC},1}$", r"$\log\tau_{T,1}$",
     r"$\alpha_2$", r"$\beta_2$", r"$\log\sigma_{\mathrm{obs},2}$",
     r"$\log\tau_{\mathrm{EMC},2}$", r"$\log\tau_{T,2}$"]
)
# Tight per-storey profile prior (sigma_profile = 0.05 → ~ 5 % per-storey variation max)
PROFILE_PRIOR_SIGMA = 0.05
M_CENTRAL_FIXED_KG = 100_000.0


def log_prior_dim_two_modes_perstorey(theta: np.ndarray,
                                       k_anchor_X: float,
                                       k_anchor_Y: float,
                                       cfg: DimPriorConfig) -> float:
    """Log prior for the per-storey-k reparameterized model (28 parameters)."""
    lp = 0.0
    lp += normal_log_pdf(theta[0], float(np.log(k_anchor_X)), cfg.sigma_log_k)
    for j in range(8):
        lp += normal_log_pdf(theta[1 + j], 0.0, PROFILE_PRIOR_SIGMA)
    lp += normal_log_pdf(theta[9], float(np.log(k_anchor_Y)), cfg.sigma_log_k)
    for j in range(8):
        lp += normal_log_pdf(theta[10 + j], 0.0, PROFILE_PRIOR_SIGMA)
    for i in range(2):
        base = 18 + 5 * i
        lp += normal_log_pdf(theta[base + 0], cfg.alpha_mean, cfg.alpha_sigma)
        lp += normal_log_pdf(theta[base + 1], cfg.beta_mean, cfg.beta_sigma)
        lp += normal_log_pdf(theta[base + 2], cfg.log_sigma_obs_mean, cfg.log_sigma_obs_sigma)
        lp += normal_log_pdf(theta[base + 3], cfg.log_tau_EMC_h_mean, cfg.log_tau_EMC_h_sigma)
        lp += normal_log_pdf(theta[base + 4], cfg.log_tau_T_h_mean, cfg.log_tau_T_h_sigma)
    return float(lp)


# Uniform-k variant of Level 2: one k per direction, one M_central, no per-storey detail.
DIM_UNIFORM_PARAM_NAMES = [
    "log_k_X", "log_k_Y", "log_M_central",
    "alpha_1", "beta_1", "log_sigma_obs_1", "log_tau_EMC_h_1", "log_tau_T_h_1",
    "alpha_2", "beta_2", "log_sigma_obs_2", "log_tau_EMC_h_2", "log_tau_T_h_2",
]
DIM_UNIFORM_PARAM_LATEX = [
    r"$\log k_X$", r"$\log k_Y$", r"$\log M_{\mathrm{c}}$",
    r"$\alpha_1$", r"$\beta_1$", r"$\log\sigma_{\mathrm{obs},1}$",
    r"$\log\tau_{\mathrm{EMC},1}$", r"$\log\tau_{T,1}$",
    r"$\alpha_2$", r"$\beta_2$", r"$\log\sigma_{\mathrm{obs},2}$",
    r"$\log\tau_{\mathrm{EMC},2}$", r"$\log\tau_{T,2}$",
]


def log_prior_dim_two_modes_uniform(theta: np.ndarray,
                                     k_anchor_X: float,
                                     k_anchor_Y: float,
                                     cfg: DimPriorConfig) -> float:
    """Log prior for the uniform-k dimensional 2-mode model (13 parameters).

    No per-storey stiffness, no per-storey mass; just (k_X, k_Y, M_central) +
    per-mode env params.
    """
    lp = 0.0
    lp += normal_log_pdf(theta[0], float(np.log(k_anchor_X)), cfg.sigma_log_k)
    lp += normal_log_pdf(theta[1], float(np.log(k_anchor_Y)), cfg.sigma_log_k)
    lp += normal_log_pdf(theta[2], cfg.log_M_central_mean, cfg.log_M_central_sigma)
    for i in range(2):
        base = 3 + 5 * i
        lp += normal_log_pdf(theta[base + 0], cfg.alpha_mean, cfg.alpha_sigma)
        lp += normal_log_pdf(theta[base + 1], cfg.beta_mean, cfg.beta_sigma)
        lp += normal_log_pdf(theta[base + 2], cfg.log_sigma_obs_mean, cfg.log_sigma_obs_sigma)
        lp += normal_log_pdf(theta[base + 3], cfg.log_tau_EMC_h_mean, cfg.log_tau_EMC_h_sigma)
        lp += normal_log_pdf(theta[base + 4], cfg.log_tau_T_h_mean, cfg.log_tau_T_h_sigma)
    return float(lp)


def log_prior_dim_two_modes(theta: np.ndarray,
                             k_anchor_X: float,
                             k_anchor_Y: float,
                             cfg: DimPriorConfig) -> float:
    """Log prior for the dimensional 2-mode model (37 parameters).

    k_anchor_X, k_anchor_Y: period-anchored prior means (N/m) for X- and Y-stack stiffnesses.
    """
    lp = 0.0
    # Per-storey stiffnesses, X-direction (idx 0..7)
    for j in range(8):
        lp += normal_log_pdf(theta[j], float(np.log(k_anchor_X)), cfg.sigma_log_k)
    # Per-storey stiffnesses, Y-direction (idx 8..15)
    for j in range(8):
        lp += normal_log_pdf(theta[8 + j], float(np.log(k_anchor_Y)), cfg.sigma_log_k)
    # Mass hyperparameters
    lp += normal_log_pdf(theta[16], cfg.log_M_central_mean, cfg.log_M_central_sigma)
    lp += normal_log_pdf(theta[17], cfg.log_sigma_floor_mean, cfg.log_sigma_floor_sigma)
    lp += normal_log_pdf(theta[18], cfg.logit_rho_M_mean, cfg.logit_rho_M_sigma)
    # Per-storey eta_j ~ Normal(0, Sigma_M) with Sigma_M compound-symmetric:
    # Sigma_M = sigma_floor^2 * [(1-rho)I + rho * 1 1^T]
    # We sample eta_j and add the prior log density. The compound-symmetric Cholesky:
    # eta_j = sigma_floor * (sqrt(1-rho) * z_j + sqrt(rho/n) * ... ), but for simplicity
    # we use the diagonal IID-like prior penalty since the data informs eta_j weakly.
    sigma_floor = np.exp(theta[17])
    rho_M = sigmoid(theta[18])
    # Determinant-based prior contribution (approx; ignore the cross-coupling term for the
    # log-density floor; the data will constrain eta_j primarily).
    for j in range(8):
        lp += normal_log_pdf(theta[19 + j], 0.0, sigma_floor)
    # Per-mode env params
    for i in range(2):
        base = 27 + 5 * i
        lp += normal_log_pdf(theta[base + 0], cfg.alpha_mean, cfg.alpha_sigma)
        lp += normal_log_pdf(theta[base + 1], cfg.beta_mean, cfg.beta_sigma)
        lp += normal_log_pdf(theta[base + 2], cfg.log_sigma_obs_mean, cfg.log_sigma_obs_sigma)
        lp += normal_log_pdf(theta[base + 3], cfg.log_tau_EMC_h_mean, cfg.log_tau_EMC_h_sigma)
        lp += normal_log_pdf(theta[base + 4], cfg.log_tau_T_h_mean, cfg.log_tau_T_h_sigma)
    return float(lp)


# ============================================================================
# Stiffness / mass split model priors (Level 1b)
# ============================================================================

@dataclass(frozen=True)
class SplitPriorConfig:
    """Priors for the stiffness/mass split model (Model 2).

    Per-mode parameters (f_ref_i, alpha_k_i, beta_i, sigma_obs_i, tau_EMC_i, tau_T_i)
    follow the same structure as AdimPriorConfig, but the EMC sensitivity is
    alpha_k (stiffness) rather than alpha_eff.

    The shared alpha_m parameter uses a wood-science-derived informative prior:

        alpha_m ~ TruncatedNormal(mu=0.005, sigma=0.002, lower=0, upper=0.015)

    Derivation (Norway spruce CLT, Limnologen):
        - Dry density rho_0 ≈ 440 kg/m³
        - At reference MC=19 %: rho ≈ 524 kg/m³
        - Mass per %MC increase: 440/100 = 4.4 kg/m³
        - Fractional mass increase per %MC: 4.4 / 524 ≈ 0.0084
        - Estimated CLT mass fraction in floor assembly: r_CLT ≈ 0.55-0.65
          (140-240 mm CLT slab + screed/finishes/partitions)
        - alpha_m = r_CLT * 0.0084 ≈ 0.0046-0.0055 per %EMC
        - Prior mean set at 0.005, sigma 0.002 to cover r_CLT ∈ [0.35, 0.80]

    The prior on alpha_k is Normal(mu=0.007, sigma=0.003):
        - Expect alpha_k = alpha_eff + alpha_m ≈ 0.003-0.004 + 0.005 ≈ 0.007-0.009
        - Positive prior mean reflects net stiffening (connection swelling
          dominates over MOE softening in CLT buildings)

    Reference EMC for the nonlinear alpha_eff approximation:
        - emc_ref = 19.0 % (dataset mean at the relevant lag timescales)
    """
    # Reference outdoor EMC for the alpha_eff → alpha_k conversion (% absolute)
    emc_ref_pct: float = 19.0

    # f_ref prior: same as adim model
    f_ref_sigma_hz: float = 0.05

    # alpha_k prior: Normal(+0.007, 0.003) — positive-leaning (stiffening expected)
    # alpha_k = alpha_eff + alpha_m, alpha_eff ≈ 0.002-0.004, alpha_m ≈ 0.005
    alpha_k_mean: float = 0.0070
    alpha_k_sigma: float = 0.0030

    # alpha_m prior: informative wood-science truncated-Normal (per %EMC)
    # Implemented as soft-truncated Normal: p(alpha_m) ∝ Normal(mu, sigma) * I(alpha_m > 0)
    # (the lower truncation is enforced via the log_prior; upper is soft via the Normal tail)
    alpha_m_mean: float = 0.0050
    alpha_m_sigma: float = 0.0020
    alpha_m_lower: float = 0.0000   # hard lower bound (mass cannot decrease with moisture)
    alpha_m_upper: float = 0.0150   # soft upper bound (unphysically large)

    # beta prior: same as adim model
    beta_mean: float = -0.0025
    beta_sigma: float = 0.0010

    # sigma_obs prior: same as adim model
    log_sigma_obs_mean: float = float(np.log(0.015))
    log_sigma_obs_sigma: float = 0.30

    # tau priors: same as adim per-mode model
    log_tau_EMC_h_mean: float = float(np.log(45.0 * 24.0))
    log_tau_EMC_h_sigma: float = 0.40
    log_tau_T_h_mean: float = float(np.log(7.0 * 24.0))
    log_tau_T_h_sigma: float = 0.40

    def as_dict(self) -> dict:
        return asdict(self)


def log_prior_split_per_mode(
    theta: np.ndarray,
    f_ref_mean: np.ndarray,
    cfg: SplitPriorConfig,
) -> float:
    """Log prior for the stiffness/mass split model (19 params).

    Parameter layout (matches forward_model.SPLIT_PARAM_NAMES):
        For each mode i in {0,1,2}, theta[6*i : 6*i+6] =
            [f_ref_i, alpha_k_i, beta_i, log_sigma_obs_i,
             log_tau_EMC_h_i, log_tau_T_h_i]
        theta[18] = alpha_m   (shared)
    """
    lp = 0.0
    for i in range(3):
        base = 6 * i
        lp += normal_log_pdf(theta[base + 0], f_ref_mean[i], cfg.f_ref_sigma_hz)
        lp += normal_log_pdf(theta[base + 1], cfg.alpha_k_mean, cfg.alpha_k_sigma)
        lp += normal_log_pdf(theta[base + 2], cfg.beta_mean, cfg.beta_sigma)
        lp += normal_log_pdf(theta[base + 3], cfg.log_sigma_obs_mean, cfg.log_sigma_obs_sigma)
        lp += normal_log_pdf(theta[base + 4], cfg.log_tau_EMC_h_mean, cfg.log_tau_EMC_h_sigma)
        lp += normal_log_pdf(theta[base + 5], cfg.log_tau_T_h_mean, cfg.log_tau_T_h_sigma)
    # Shared alpha_m: truncated Normal (hard lower at 0, soft upper via Normal tail)
    alpha_m = theta[18]
    if alpha_m < cfg.alpha_m_lower or alpha_m > cfg.alpha_m_upper:
        return -np.inf
    lp += normal_log_pdf(alpha_m, cfg.alpha_m_mean, cfg.alpha_m_sigma)
    return float(lp)


def alpha_k_from_eff(
    alpha_eff: np.ndarray,
    alpha_m: np.ndarray,
    emc_ref: float = 19.0,
) -> np.ndarray:
    """Recover alpha_k from the observed alpha_eff and alpha_m.

    Exact (nonlinear) inversion of:
        alpha_eff ≈ (alpha_k - alpha_m) / (1 + alpha_m * emc_ref)

    Solving for alpha_k:
        alpha_k = alpha_eff * (1 + alpha_m * emc_ref) + alpha_m

    Note: at alpha_m * emc_ref ≈ 0.005 * 19 = 0.095, the correction is
    ~9.5%, which is within posterior uncertainty but non-negligible.
    """
    return np.asarray(alpha_eff) * (1.0 + np.asarray(alpha_m) * emc_ref) + np.asarray(alpha_m)


def log_prior_adim_per_mode(theta: np.ndarray, f_ref_mean: np.ndarray, cfg: AdimPriorConfig) -> float:
    """Log prior for the per-mode-tau variant (18 params).

    Layout: for each mode i ∈ {0,1,2}, theta[6i:6i+6] =
        [f_ref_i, alpha_i, beta_i, log_sigma_obs_i, log_tau_EMC_h_i, log_tau_T_h_i]
    """
    lp = 0.0
    for i in range(3):
        base = 6 * i
        lp += normal_log_pdf(theta[base + 0], f_ref_mean[i], cfg.f_ref_sigma_hz)
        lp += normal_log_pdf(theta[base + 1], cfg.alpha_mean, cfg.alpha_sigma)
        lp += normal_log_pdf(theta[base + 2], cfg.beta_mean, cfg.beta_sigma)
        lp += normal_log_pdf(theta[base + 3], cfg.log_sigma_obs_mean, cfg.log_sigma_obs_sigma)
        lp += normal_log_pdf(theta[base + 4], cfg.log_tau_EMC_h_mean, cfg.log_tau_EMC_h_sigma)
        lp += normal_log_pdf(theta[base + 5], cfg.log_tau_T_h_mean, cfg.log_tau_T_h_sigma)
    return float(lp)
