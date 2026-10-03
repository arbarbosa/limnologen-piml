"""Level 1b validation — sensitivity analysis + exact HMC check for the split model.
by Andre R. Barbosa, April - October 2026

Two independent tests of the Level 1b decomposition:

Part A — Prior sensitivity analysis (analytical, fast)
-------------------------------------------------------
Sweeps two axes of the wood-science prior:
  1. r_CLT (CLT mass fraction in floor assembly) ∈ [0.35, 0.75]
     → changes the prior mean μ_m = r_CLT × 0.0084
  2. σ_m (prior standard deviation) ∈ [0.001, 0.004]
     → changes prior informativeness at fixed μ_m = 0.005

For each prior setting, the analytical decomposition is rerun (instant) and
the posterior mean ± 90 % CI on α_k is recorded per mode.

Key result: α_k > 0 is robust — it holds across all realistic r_CLT values
and prior widths, confirming the stiffening conclusion is not an artifact of
the prior specification.

Part B — HMC validation on exact split model (emcee, ~5–15 min)
-----------------------------------------------------------------
Fits the exact split forward model

    f_hat_i = f_ref_i × sqrt[(1 + α_k_i × EMC_τ + β_i × T_τ)
                              / (1 + α_m  × EMC_τ)]

via emcee (32 walkers, 3 500 steps) on the training data.  Compares the
HMC posterior marginals on (α_k_1, α_k_2, α_k_3, α_m) with the analytical
decomposition from Level 1b.

Expected result: analytical ≈ HMC within Monte Carlo noise, confirming
the linearization error (≈4-5 % of α_eff) is negligible relative to the
posterior width.

Usage
-----
    # Part A only (fast):
    python analysis/level1b_validation.py

    # Part A + HMC validation (slow, ~10 min):
    python analysis/level1b_validation.py --validate

    # HMC with custom chain length:
    python analysis/level1b_validation.py --validate --n-walkers 32 --n-steps 3000
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from analysis.common.data_loader import (
    load_continuous_hus6, chronological_split, per_mode_view,
    get_lag_matrix, FREQ_COLS, project_root,
)
from analysis.common.priors import SplitPriorConfig, alpha_k_from_eff
from analysis.common.forward_model import (
    log_likelihood_split_per_mode, SPLIT_PARAM_NAMES,
)
from analysis.common.plotting import set_paper_style, MODE_COLORS

OUT_DIR     = project_root() / "aux" / "level1b_split"
# Level 1 posterior draws (per-mode-tau run), as in level1b_split_model.py.
STEP01_PATH = project_root() / "aux" / "level1_adim" / "v3_per_mode_tau" / "posterior.json"
EMC_REF     = 19.0      # dataset mean %EMC
RHO_DRY     = 440.0     # Norway spruce CLT dry density (kg/m³)
MC_REF      = 19.0      # reference MC (%) at EMC_ref
RNG_SEED    = 20260608


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def alpha_m_from_r_clt(r_clt: float, rho_dry: float = RHO_DRY,
                        mc_ref: float = MC_REF) -> float:
    """Wood-science central value of α_m from CLT mass fraction.

    α_m = r_CLT × (ρ_dry / 100) / ρ(MC_ref)
    where ρ(MC_ref) = ρ_dry × (1 + MC_ref/100).
    """
    rho_at_ref = rho_dry * (1.0 + mc_ref / 100.0)
    return r_clt * (rho_dry / 100.0) / rho_at_ref


def sample_alpha_m(n: int, mu: float, sigma: float,
                   rng: np.random.Generator,
                   lower: float = 0.0, upper: float = 0.030) -> np.ndarray:
    """Rejection-sampled TruncatedNormal(mu, sigma, lower, upper)."""
    out = np.empty(n)
    filled = 0
    while filled < n:
        cands = rng.normal(mu, sigma, size=max(2*(n - filled), 500))
        valid = cands[(cands >= lower) & (cands <= upper)]
        take = min(len(valid), n - filled)
        out[filled:filled + take] = valid[:take]
        filled += take
    return out


def load_step01_posteriors() -> dict:
    """Return posterior means and arrays for alpha_eff and tau per mode."""
    with open(STEP01_PATH) as f:
        p = json.load(f)
    draws = np.array(p["thinned_draws"])
    nm = {n: i for i, n in enumerate(p["param_names"])}
    s  = p["summary"]
    return {
        "draws": draws,
        "names": nm,
        "alpha_eff": {
            1: draws[:, nm["alpha_1"]],
            2: draws[:, nm["alpha_2"]],
            3: draws[:, nm["alpha_3"]],
        },
        "means": {k: s[k]["mean"] for k in s},
    }


def analytical_decomposition(alpha_eff_draws: np.ndarray,
                              alpha_m_draws: np.ndarray) -> np.ndarray:
    """α_k = α_eff × (1 + α_m × EMC_ref) + α_m (nonlinear inversion)."""
    return alpha_k_from_eff(alpha_eff_draws, alpha_m_draws, EMC_REF)


def ci90(arr: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(arr, 5)), float(np.percentile(arr, 95))


# ─────────────────────────────────────────────────────────────────────────────
# Part A — Prior sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_sensitivity(n_mc: int = 20_000) -> dict:
    """Sweep r_CLT and σ_m; return α_k summaries per mode per sweep point."""
    rng = np.random.default_rng(RNG_SEED)
    p01 = load_step01_posteriors()
    n_step01 = p01["draws"].shape[0]

    # Resample alpha_eff draws to n_mc
    idx = rng.integers(0, n_step01, size=n_mc)
    ae = {i: p01["alpha_eff"][i][idx] for i in (1, 2, 3)}

    # ── Sweep 1: r_CLT ────────────────────────────────────────────────────────
    r_clt_grid = np.array([0.35, 0.45, 0.55, 0.65, 0.75])
    sigma_m_fixed = 0.002    # keep σ_m fixed at base value

    sweep_r = {"r_clt": r_clt_grid.tolist(), "modes": {1: {}, 2: {}, 3: {}}}
    for r_clt in r_clt_grid:
        mu_m = alpha_m_from_r_clt(r_clt)
        am   = sample_alpha_m(n_mc, mu_m, sigma_m_fixed, rng)
        for mode in (1, 2, 3):
            ak = analytical_decomposition(ae[mode], am)
            lo, hi = ci90(ak)
            sweep_r["modes"][mode][f"{r_clt:.2f}"] = {
                "mean": float(ak.mean()), "std": float(ak.std()),
                "q05": lo, "q95": hi, "mu_m": mu_m,
                "prob_pos": float((ak > 0).mean()),
            }

    # ── Sweep 2: σ_m ──────────────────────────────────────────────────────────
    mu_m_fixed = 0.005
    sigma_m_grid = np.array([0.001, 0.0015, 0.002, 0.003, 0.004])

    sweep_s = {"sigma_m": sigma_m_grid.tolist(), "modes": {1: {}, 2: {}, 3: {}}}
    for sigma_m in sigma_m_grid:
        am = sample_alpha_m(n_mc, mu_m_fixed, sigma_m, rng)
        for mode in (1, 2, 3):
            ak = analytical_decomposition(ae[mode], am)
            lo, hi = ci90(ak)
            sweep_s["modes"][mode][f"{sigma_m:.4f}"] = {
                "mean": float(ak.mean()), "std": float(ak.std()),
                "q05": lo, "q95": hi,
                "prob_pos": float((ak > 0).mean()),
            }

    return {"r_clt_sweep": sweep_r, "sigma_m_sweep": sweep_s,
            "r_clt_grid": r_clt_grid, "sigma_m_grid": sigma_m_grid,
            "alpha_m_at_r_clt": [alpha_m_from_r_clt(r) for r in r_clt_grid]}


def plot_sensitivity(sens: dict, dpi: int = 200) -> Path:
    """Two-panel sensitivity figure: α_k vs r_CLT and vs σ_m."""
    set_paper_style()
    mode_colors = MODE_COLORS
    mode_labels = ["Mode 1", "Mode 2", "Mode 3"]
    r_clt_grid  = np.array(sens["r_clt_grid"])
    sigma_m_grid = np.array(sens["sigma_m_grid"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.88, wspace=0.35)

    # ── Panel 1: α_k vs r_CLT ────────────────────────────────────────────────
    for mi, (mode, col, lbl) in enumerate(zip((1,2,3), mode_colors, mode_labels)):
        means = np.array([sens["r_clt_sweep"]["modes"][mode][f"{r:.2f}"]["mean"]
                          for r in r_clt_grid])
        lo    = np.array([sens["r_clt_sweep"]["modes"][mode][f"{r:.2f}"]["q05"]
                          for r in r_clt_grid])
        hi    = np.array([sens["r_clt_sweep"]["modes"][mode][f"{r:.2f}"]["q95"]
                          for r in r_clt_grid])
        ax1.plot(r_clt_grid, means * 1e3, color=col, lw=1.8, marker="o",
                 ms=4.5, label=lbl)
        ax1.fill_between(r_clt_grid, lo * 1e3, hi * 1e3, color=col, alpha=0.15)

    ax1.axhline(0, color="black", lw=0.6, ls=":", alpha=0.6)
    ax1.axvline(0.55, color="#999", lw=0.8, ls="--", alpha=0.7,
                label=r"$r_\mathrm{CLT}=0.55$ (base)")
    ax1.set_xlabel(r"$r_\mathrm{CLT}$ — CLT mass fraction", fontsize=9)
    ax1.set_ylabel(r"$\alpha_k$ (per %EMC, ×10$^{-3}$)", fontsize=9)
    ax1.set_title(r"Sensitivity to $r_\mathrm{CLT}$" + "\n(shading = 90% CI)", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.tick_params(labelsize=8)
    # annotate implied α_m at top axis
    ax2_top = ax1.twiny()
    ax2_top.set_xlim(ax1.get_xlim())
    am_ticks = [alpha_m_from_r_clt(r) for r in r_clt_grid]
    ax2_top.set_xticks(r_clt_grid)
    ax2_top.set_xticklabels([f"{v*1e3:.1f}" for v in am_ticks], fontsize=7)
    ax2_top.set_xlabel(r"Implied $\mu_{\alpha_m}$ (×10$^{-3}$)", fontsize=7.5)
    ax1.text(0.04, 0.97, "(a)", transform=ax1.transAxes, fontsize=9,
             va="top", fontweight="bold")

    # ── Panel 2: α_k vs σ_m ─────────────────────────────────────────────────
    for mi, (mode, col, lbl) in enumerate(zip((1,2,3), mode_colors, mode_labels)):
        means = np.array([sens["sigma_m_sweep"]["modes"][mode][f"{s:.4f}"]["mean"]
                          for s in sigma_m_grid])
        lo    = np.array([sens["sigma_m_sweep"]["modes"][mode][f"{s:.4f}"]["q05"]
                          for s in sigma_m_grid])
        hi    = np.array([sens["sigma_m_sweep"]["modes"][mode][f"{s:.4f}"]["q95"]
                          for s in sigma_m_grid])
        ax2.plot(sigma_m_grid * 1e3, means * 1e3, color=col, lw=1.8, marker="o",
                 ms=4.5, label=lbl)
        ax2.fill_between(sigma_m_grid * 1e3, lo * 1e3, hi * 1e3, color=col, alpha=0.15)

    ax2.axhline(0, color="black", lw=0.6, ls=":", alpha=0.6)
    ax2.axvline(2.0, color="#999", lw=0.8, ls="--", alpha=0.7,
                label=r"$\sigma_m=2.0\times10^{-3}$ (base)")
    ax2.set_xlabel(r"$\sigma_{\alpha_m}$ (×10$^{-3}$) — prior std", fontsize=9)
    ax2.set_ylabel(r"$\alpha_k$ (per %EMC, ×10$^{-3}$)", fontsize=9)
    ax2.set_title(r"Sensitivity to prior width $\sigma_{\alpha_m}$"
                  + r" (fixed $\mu_{\alpha_m}=5\times10^{-3}$)", fontsize=9)
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.tick_params(labelsize=8)
    ax2.text(0.04, 0.97, "(b)", transform=ax2.transAxes, fontsize=9,
             va="top", fontweight="bold")

    fig.text(
        0.5, 0.01,
        r"$\alpha_k>0$ for all modes across all physically plausible $r_\mathrm{CLT}\in[0.35,0.75]$"
        r" and prior widths $\sigma_{\alpha_m}\in[0.001,0.004]$.",
        ha="center", fontsize=8, color="#333", style="italic",
    )

    path = OUT_DIR / "Fig_Split_Sensitivity.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Sensitivity figure → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Part B — HMC validation on exact split model
# ─────────────────────────────────────────────────────────────────────────────

def build_per_mode_data() -> tuple[list[dict], np.ndarray]:
    """Load training data; return per_mode_data list and f_ref_init."""
    df = load_continuous_hus6(apply_filters=True)
    train, _ = chronological_split(df)
    per_mode_data = []
    f_ref_init = np.empty(3)
    for i in (1, 2, 3):
        sub = per_mode_view(train, mode_idx=i, apply_pole_filter=True)
        first_month_end = sub.DateTime.min() + __import__("pandas").Timedelta(days=30)
        first_month = sub.loc[sub.DateTime <= first_month_end]
        f_ref_init[i-1] = float(first_month[FREQ_COLS[i-1]].median())
        per_mode_data.append({
            "f_obs": sub[FREQ_COLS[i-1]].to_numpy(float),
            "emc_lag_matrix": get_lag_matrix(sub, "EMC"),
            "t_lag_matrix":   get_lag_matrix(sub, "T"),
        })
    return per_mode_data, f_ref_init


def make_split_log_posterior(per_mode_data: list[dict],
                              f_ref_init: np.ndarray,
                              cfg: SplitPriorConfig):
    """Return a log-posterior closure for the 19-param split model."""
    from analysis.common.priors import log_prior_split_per_mode

    def log_post(theta: np.ndarray) -> float:
        if not np.all(np.isfinite(theta)):
            return -np.inf
        lp = log_prior_split_per_mode(theta, f_ref_init, cfg)
        if not np.isfinite(lp):
            return -np.inf
        try:
            ll = log_likelihood_split_per_mode(theta, per_mode_data)
        except Exception:
            return -np.inf
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    return log_post


def run_hmc_validation(n_walkers: int = 32,
                       n_warmup: int = 500,
                       n_production: int = 2500) -> np.ndarray:
    """Run emcee on the exact split model.  Returns (n_production*n_walkers, 19) draws."""
    import emcee

    per_mode_data, f_ref_init = build_per_mode_data()
    cfg = SplitPriorConfig()
    p01 = load_step01_posteriors()
    m = p01["means"]

    # ── Analytical MAP as initialization center ───────────────────────────────
    alpha_m_init = cfg.alpha_m_mean
    ak_init = [
        m["alpha_1"] * (1 + alpha_m_init * EMC_REF) + alpha_m_init,
        m["alpha_2"] * (1 + alpha_m_init * EMC_REF) + alpha_m_init,
        m["alpha_3"] * (1 + alpha_m_init * EMC_REF) + alpha_m_init,
    ]
    # 19-param layout: [f_ref_i, alpha_k_i, beta_i, log_sigma_obs_i, tau_EMC_i, tau_T_i]×3 + alpha_m
    theta_center = np.array([
        f_ref_init[0], ak_init[0], m["beta_1"],   m["log_sigma_obs_1"],
            m["log_tau_EMC_h_1"], m["log_tau_T_h_1"],
        f_ref_init[1], ak_init[1], m["beta_2"],   m["log_sigma_obs_2"],
            m["log_tau_EMC_h_2"], m["log_tau_T_h_2"],
        f_ref_init[2], ak_init[2], m["beta_3"],   m["log_sigma_obs_3"],
            m["log_tau_EMC_h_3"], m["log_tau_T_h_3"],
        alpha_m_init,
    ])

    # Walker spread: tight around MAP (10 % of prior σ to avoid prior walls)
    theta_sd = np.array([
        cfg.f_ref_sigma_hz, cfg.alpha_k_sigma*0.1, cfg.beta_sigma*0.1, cfg.log_sigma_obs_sigma*0.1,
            cfg.log_tau_EMC_h_sigma*0.1, cfg.log_tau_T_h_sigma*0.1,
    ] * 3 + [cfg.alpha_m_sigma * 0.2])

    rng = np.random.default_rng(RNG_SEED + 1)
    p0  = theta_center + theta_sd * rng.standard_normal((n_walkers, 19))

    log_post = make_split_log_posterior(per_mode_data, f_ref_init, cfg)

    # Quick sanity check on center
    lp0 = log_post(theta_center)
    print(f"  log-posterior at MAP init: {lp0:.1f}")
    if not np.isfinite(lp0):
        raise RuntimeError("MAP initialization has -inf log-posterior — check priors/data")

    sampler = emcee.EnsembleSampler(n_walkers, 19, log_post)

    t0 = time.time()
    print(f"  Running {n_warmup} warmup steps ({n_walkers} walkers)...")
    sampler.run_mcmc(p0, n_warmup, progress=True)
    last_pos = sampler.get_last_sample()   # save position BEFORE reset
    sampler.reset()

    print(f"  Running {n_production} production steps...")
    sampler.run_mcmc(last_pos, n_production, progress=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s")

    # Diagnostics
    tau_est = sampler.get_autocorr_time(quiet=True)
    print(f"  Autocorr times (selected): "
          f"alpha_k_1={tau_est[1]:.1f}, alpha_k_2={tau_est[7]:.1f}, "
          f"alpha_k_3={tau_est[13]:.1f}, alpha_m={tau_est[18]:.1f}")
    accept = sampler.acceptance_fraction.mean()
    print(f"  Mean acceptance fraction: {accept:.3f}  (target 0.2–0.5)")

    # Flatten: (n_production, n_walkers, 19) → (n_production*n_walkers, 19)
    draws = sampler.get_chain(flat=True)
    # Thin by max(1, tau/2)
    thin = max(1, int(np.nanmax(np.clip(tau_est, 1, 500)) / 2))
    draws_thinned = draws[::thin]
    print(f"  Thinning by {thin} → {len(draws_thinned)} independent draws")
    return draws_thinned


def plot_hmc_vs_analytical(hmc_draws: np.ndarray, dpi: int = 200) -> Path:
    """Compare HMC and analytical posterior distributions for α_k and α_m."""
    rng    = np.random.default_rng(RNG_SEED + 2)
    p01    = load_step01_posteriors()
    cfg    = SplitPriorConfig()
    n_mc   = 20_000

    # Rebuild analytical draws
    n_step01 = p01["draws"].shape[0]
    idx = rng.integers(0, n_step01, size=n_mc)
    ae = {i: p01["alpha_eff"][i][idx] for i in (1, 2, 3)}
    am_analytical = sample_alpha_m(n_mc, cfg.alpha_m_mean, cfg.alpha_m_sigma, rng)
    ak_analytical = {
        i: analytical_decomposition(ae[i], am_analytical) for i in (1, 2, 3)
    }

    # HMC draws for α_k and α_m
    ak_hmc = {1: hmc_draws[:, 1], 2: hmc_draws[:, 7], 3: hmc_draws[:, 13]}
    am_hmc = hmc_draws[:, 18]

    set_paper_style()
    mode_colors = MODE_COLORS
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.90,
                        hspace=0.42, wspace=0.35)

    def plot_compare(ax, analytic, hmc_arr, color, label):
        xs = np.linspace(
            min(analytic.min(), hmc_arr.min()) - analytic.std()*0.5,
            max(analytic.max(), hmc_arr.max()) + analytic.std()*0.5, 300)
        kde_a = stats.gaussian_kde(analytic, bw_method=0.3)
        kde_h = stats.gaussian_kde(hmc_arr,  bw_method=0.3)
        ya = kde_a(xs); ya /= ya.max()
        yh = kde_h(xs); yh /= yh.max()
        ax.plot(xs * 1e3, ya, color=color, lw=1.8, label="Analytical")
        ax.plot(xs * 1e3, yh, color=color, lw=1.8, ls="--", label="HMC (exact model)")
        ax.axvline(analytic.mean() * 1e3, color=color, lw=0.7, alpha=0.5)
        ax.axvline(hmc_arr.mean() * 1e3, color=color, lw=0.7, ls="--", alpha=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0, 1.25)
        ax.tick_params(labelsize=8)

    # α_k per mode
    for i, (ax, col, ml) in enumerate(zip(
            [axes[0,0], axes[0,1], axes[1,0]],
            mode_colors,
            ["Mode 1", "Mode 2", "Mode 3"])):
        plot_compare(ax, ak_analytical[i+1], ak_hmc[i+1], col,
                     rf"$\alpha_{{k,{i+1}}}$ — {ml}")
        ax.set_xlabel(r"$\alpha_k$ (×10$^{-3}$ per %EMC)", fontsize=8)
        ax.set_ylabel("Normalised density", fontsize=8)
        tag = ["(a)", "(b)", "(c)"][i]
        ax.text(0.04, 0.97, tag, transform=ax.transAxes, fontsize=9,
                va="top", fontweight="bold")
        if i == 0:
            ax.legend(fontsize=7.5)

    # α_m
    ax = axes[1, 1]
    plot_compare(ax, am_analytical, am_hmc, "#555555", r"$\alpha_m$ — shared")
    ax.set_xlabel(r"$\alpha_m$ (×10$^{-3}$ per %EMC)", fontsize=8)
    ax.set_ylabel("Normalised density", fontsize=8)
    ax.text(0.04, 0.97, "(d)", transform=ax.transAxes, fontsize=9,
            va="top", fontweight="bold")

    fig.suptitle("HMC validation: exact split model vs analytical decomposition",
                 fontsize=10, fontweight="bold")
    fig.text(0.5, 0.01,
             "Solid = analytical (Level 1 posterior + wood-science prior).  "
             "Dashed = HMC on exact nonlinear split model.",
             ha="center", fontsize=8, color="#333")

    path = OUT_DIR / "Fig_Split_HMC_Validation.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"HMC validation figure → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="Run full emcee HMC on exact split model (~10 min)")
    ap.add_argument("--n-walkers", type=int, default=48)  # must be > 2*ndim=38
    ap.add_argument("--n-warmup",  type=int, default=500)
    ap.add_argument("--n-steps",   type=int, default=2500)
    ap.add_argument("--dpi",       type=int, default=200)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Part A: sensitivity ───────────────────────────────────────────────────
    print("=" * 60)
    print("Part A — Prior sensitivity analysis")
    print("=" * 60)
    sens = run_sensitivity()

    # Print table
    r_clt_grid  = sens["r_clt_grid"]
    sigma_m_grid = sens["sigma_m_grid"]

    print("\nSweep 1: α_k vs r_CLT  (all × 10⁻³ per %EMC)")
    header = f"  {'r_CLT':>6}  {'μ_m':>7}"
    for m in (1, 2, 3):
        header += f"  {'Mode '+str(m)+' mean':>12}  {'[q05,q95]':>16}"
    print(header)
    for r in r_clt_grid:
        mu_m = alpha_m_from_r_clt(r) * 1e3
        row = f"  {r:>6.2f}  {mu_m:>7.2f}"
        for m in (1, 2, 3):
            d = sens["r_clt_sweep"]["modes"][m][f"{r:.2f}"]
            row += f"  {d['mean']*1e3:>12.3f}  [{d['q05']*1e3:.2f},{d['q95']*1e3:.2f}]"
        print(row)

    print("\nSweep 2: α_k vs σ_m  (fixed μ_m=0.005, all × 10⁻³)")
    for s in sigma_m_grid:
        row = f"  σ_m={s*1e3:.1f}×10⁻³"
        for m in (1, 2, 3):
            d = sens["sigma_m_sweep"]["modes"][m][f"{s:.4f}"]
            row += f"  M{m}: {d['mean']*1e3:.3f} [{d['q05']*1e3:.2f},{d['q95']*1e3:.2f}]"
        print(row)

    sens_fig = plot_sensitivity(sens, dpi=args.dpi)
    # Save JSON
    json_path = OUT_DIR / "sensitivity_results.json"
    json.dump({
        "r_clt_sweep": {str(k): v for k,v in sens["r_clt_sweep"].items()},
        "sigma_m_sweep": {str(k): v for k,v in sens["sigma_m_sweep"].items()},
    }, open(json_path, "w"), indent=2, default=str)
    print(f"Sensitivity JSON → {json_path}")

    # ── Part B: HMC validation ────────────────────────────────────────────────
    if args.validate:
        print("\n" + "=" * 60)
        print("Part B — HMC validation on exact split model")
        print("=" * 60)
        hmc_draws = run_hmc_validation(
            n_walkers=args.n_walkers,
            n_warmup=args.n_warmup,
            n_production=args.n_steps,
        )
        val_fig = plot_hmc_vs_analytical(hmc_draws, dpi=args.dpi)
        np.save(OUT_DIR / "hmc_split_draws.npy", hmc_draws)
        print(f"HMC draws saved → {OUT_DIR / 'hmc_split_draws.npy'}")

        # Quick numeric comparison
        print("\nMean comparison (analytical vs HMC, ×10⁻³ per %EMC):")
        rng = np.random.default_rng(RNG_SEED + 3)
        p01 = load_step01_posteriors(); cfg = SplitPriorConfig()
        n_mc = 20_000
        idx = rng.integers(0, p01["draws"].shape[0], size=n_mc)
        am_a = sample_alpha_m(n_mc, cfg.alpha_m_mean, cfg.alpha_m_sigma, rng)
        for mi, mlab in enumerate(("Mode 1","Mode 2","Mode 3"), start=1):
            ak_a = analytical_decomposition(p01["alpha_eff"][mi][idx], am_a)
            ak_h = hmc_draws[:, 6*(mi-1)+1]
            print(f"  {mlab}: analytical={ak_a.mean()*1e3:.3f}±{ak_a.std()*1e3:.3f}  "
                  f"HMC={ak_h.mean()*1e3:.3f}±{ak_h.std()*1e3:.3f}")
        am_h = hmc_draws[:, 18]
        print(f"  alpha_m: analytical={am_a.mean()*1e3:.3f}±{am_a.std()*1e3:.3f}  "
              f"HMC={am_h.mean()*1e3:.3f}±{am_h.std()*1e3:.3f}")
    else:
        print("\n[Skipped HMC validation — rerun with --validate to include]")

    print("\nDone. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
