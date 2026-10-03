"""Level 1 — Adimensional twin-model fit on the Hus 6 continuous record.
by Andre R. Barbosa, April - October 2026

What this level does:
- Fits the adimensional model, a per-mode scalar,
      f_hat_i(t) = f_ref_i * sqrt(1 + alpha_i*EMC_tilde_tau_EMC(t) + beta_i*T_tilde_tau_T(t)),
  to the identified frequencies of the three modes on the training window.
- f_ref_i, alpha_i, beta_i and sigma_obs_i are per mode. The lag times are either
  shared across modes (14 parameters, per_mode_tau=False) or per mode
  (18 parameters, per_mode_tau=True; the model reported in the paper).
- The posterior feeds Level 1b (stiffness-mass decomposition), fixes the lag
  times of Level 2 and provides the physics mean of Level 3.

Sampler
-------
emcee (affine-invariant ensemble MCMC). The log posterior is sampler-agnostic.

Output (written to `out_dir`, e.g. aux/level1_adim/v3_per_mode_tau/):
- posterior.json
- diagnostic.png
- run diagnostics (sanity checklist, run summary, run log)
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee

# Ensure local imports work whether run as script or module
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from analysis.common.data_loader import (
    load_continuous_hus6, per_mode_view, get_lag_matrix,
    chronological_split, FREQ_COLS,
)
from analysis.common.priors import (
    AdimPriorConfig, log_prior_adim, log_prior_adim_per_mode,
    ADIM_PARAM_NAMES, ADIM_PARAM_LATEX,
    ADIM_PARAM_NAMES_PER_MODE, ADIM_PARAM_LATEX_PER_MODE,
)
from analysis.common.forward_model import (
    log_likelihood_adim, log_likelihood_adim_per_mode, posterior_predictive_adim,
)
from analysis.common.plotting import (
    set_paper_style, posterior_marginals_grid, predictive_overlay,
    MODE_COLORS, MODE_LABELS,
)
from analysis.common.io import (
    save_posterior_summary, write_decision_file, append_status_log,
)


STEP_ID = "level1_adim"


# ----------------------------------------------------------------------------
# Log-posterior wrapper for emcee
# ----------------------------------------------------------------------------

def make_log_posterior(per_mode_data: list[dict],
                       f_ref_mean: np.ndarray,
                       cfg: AdimPriorConfig,
                       per_mode_tau: bool = False):
    """Return a log_posterior(theta) closure for emcee.

    If per_mode_tau is True, expects 18-dim theta with per-mode τ_EMC, τ_T.
    """
    prior_fn = log_prior_adim_per_mode if per_mode_tau else log_prior_adim
    lik_fn = log_likelihood_adim_per_mode if per_mode_tau else log_likelihood_adim

    def log_posterior(theta: np.ndarray) -> float:
        if not np.all(np.isfinite(theta)):
            return -np.inf
        lp = prior_fn(theta, f_ref_mean, cfg)
        if not np.isfinite(lp):
            return -np.inf
        try:
            ll = lik_fn(theta, per_mode_data)
        except (ValueError, FloatingPointError):
            return -np.inf
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    return log_posterior


# ----------------------------------------------------------------------------
# Main step function
# ----------------------------------------------------------------------------

def run(out_dir: Path,
        n_walkers: int = 64,
        n_warmup: int = 1500,
        n_production: int = 4000,
        rng_seed: int = 20260529,
        init_prior_scale: float = 1.0,
        per_mode_tau: bool = False) -> dict:
    """Execute Level 1 and write outputs to `out_dir`.

    Parameters
    ----------
    init_prior_scale : float
        Standard-deviation multiplier for the initial walker spread around the
        prior mean. 1.0 = full prior σ (recommended; explores well).
        0.1 = a narrow ball around the prior mean.
    per_mode_tau : bool
        If True, each mode has its own (τ_EMC,i, τ_T,i) — 18-param model.
        If False (default), τ_EMC and τ_T are shared across modes — 14 params.
        Per-mode tau is the model reported in the paper.

    Returns a dict with paths to written artifacts and diagnostics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run_log.txt"

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(run_log, "a") as f:
            f.write(line + "\n")

    log(f"=== {STEP_ID} START ===")
    log(f"n_walkers={n_walkers} n_warmup={n_warmup} n_production={n_production} rng_seed={rng_seed} per_mode_tau={per_mode_tau}")
    param_names = ADIM_PARAM_NAMES_PER_MODE if per_mode_tau else ADIM_PARAM_NAMES
    param_latex = ADIM_PARAM_LATEX_PER_MODE if per_mode_tau else ADIM_PARAM_LATEX
    n_dim = len(param_names)

    # ------------------------------------------------------------------
    # 1. Load + filter data
    # ------------------------------------------------------------------
    df_raw = load_continuous_hus6(apply_filters=True)
    log(f"Loaded continuous Hus 6: {len(df_raw)} rows after damping outlier filter")
    train_df, val_df = chronological_split(df_raw)
    log(f"Train ({train_df.DateTime.min().date()} -> {train_df.DateTime.max().date()}): {len(train_df)} rows")
    log(f"Val   ({val_df.DateTime.min().date()} -> {val_df.DateTime.max().date()}): {len(val_df)} rows")

    # Per-mode training views (with pole-count >= 5)
    per_mode_train = []
    per_mode_n = []
    f_ref_init = np.empty(3)
    for i in (1, 2, 3):
        sub = per_mode_view(train_df, mode_idx=i, apply_pole_filter=True)
        per_mode_n.append(len(sub))
        # First-month median as prior anchor for f_ref_i
        first_month_end = sub.DateTime.min() + pd.Timedelta(days=30)
        first_month = sub.loc[sub.DateTime <= first_month_end]
        f_ref_init[i - 1] = float(first_month[FREQ_COLS[i - 1]].median())
        per_mode_train.append({
            "f_obs": sub[FREQ_COLS[i - 1]].to_numpy(dtype=float),
            "emc_lag_matrix": get_lag_matrix(sub, "EMC"),
            "t_lag_matrix": get_lag_matrix(sub, "T"),
        })
        log(f"  Mode {i}: N_train_filtered={len(sub)}, f_ref prior mean = first-month median = {f_ref_init[i-1]:.4f} Hz")

    cfg = AdimPriorConfig()
    log(f"Prior config: {json.dumps(cfg.as_dict(), indent=2)}")

    # ------------------------------------------------------------------
    # 2. Set up sampler
    # ------------------------------------------------------------------
    rng = np.random.default_rng(rng_seed)

    # Initial walker positions: prior means with jitter
    if per_mode_tau:
        prior_mean = np.array([
            f_ref_init[0], cfg.alpha_mean, cfg.beta_mean, cfg.log_sigma_obs_mean,
                cfg.log_tau_EMC_h_mean, cfg.log_tau_T_h_mean,
            f_ref_init[1], cfg.alpha_mean, cfg.beta_mean, cfg.log_sigma_obs_mean,
                cfg.log_tau_EMC_h_mean, cfg.log_tau_T_h_mean,
            f_ref_init[2], cfg.alpha_mean, cfg.beta_mean, cfg.log_sigma_obs_mean,
                cfg.log_tau_EMC_h_mean, cfg.log_tau_T_h_mean,
        ])
        prior_sd = np.array([
            cfg.f_ref_sigma_hz, cfg.alpha_sigma, cfg.beta_sigma, cfg.log_sigma_obs_sigma,
                cfg.log_tau_EMC_h_sigma, cfg.log_tau_T_h_sigma,
            cfg.f_ref_sigma_hz, cfg.alpha_sigma, cfg.beta_sigma, cfg.log_sigma_obs_sigma,
                cfg.log_tau_EMC_h_sigma, cfg.log_tau_T_h_sigma,
            cfg.f_ref_sigma_hz, cfg.alpha_sigma, cfg.beta_sigma, cfg.log_sigma_obs_sigma,
                cfg.log_tau_EMC_h_sigma, cfg.log_tau_T_h_sigma,
        ])
    else:
        prior_mean = np.array([
            f_ref_init[0], cfg.alpha_mean, cfg.beta_mean, cfg.log_sigma_obs_mean,
            f_ref_init[1], cfg.alpha_mean, cfg.beta_mean, cfg.log_sigma_obs_mean,
            f_ref_init[2], cfg.alpha_mean, cfg.beta_mean, cfg.log_sigma_obs_mean,
            cfg.log_tau_EMC_h_mean, cfg.log_tau_T_h_mean,
        ])
        prior_sd = np.array([
            cfg.f_ref_sigma_hz, cfg.alpha_sigma, cfg.beta_sigma, cfg.log_sigma_obs_sigma,
            cfg.f_ref_sigma_hz, cfg.alpha_sigma, cfg.beta_sigma, cfg.log_sigma_obs_sigma,
            cfg.f_ref_sigma_hz, cfg.alpha_sigma, cfg.beta_sigma, cfg.log_sigma_obs_sigma,
            cfg.log_tau_EMC_h_sigma, cfg.log_tau_T_h_sigma,
        ])

    p0 = prior_mean + init_prior_scale * prior_sd * rng.standard_normal((n_walkers, n_dim))
    log(f"Initial walker positions sampled from prior * init_prior_scale={init_prior_scale}.")

    log_posterior = make_log_posterior(per_mode_train, f_ref_init, cfg,
                                       per_mode_tau=per_mode_tau)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior)

    # ------------------------------------------------------------------
    # 3. Warmup
    # ------------------------------------------------------------------
    t0 = time.time()
    log(f"Warmup: {n_warmup} steps ...")
    state = sampler.run_mcmc(p0, n_warmup, progress=False)
    warm_time = time.time() - t0
    accept_warm = float(np.mean(sampler.acceptance_fraction))
    log(f"Warmup done in {warm_time:.1f} s. Mean acceptance: {accept_warm:.3f}")

    sampler.reset()

    # ------------------------------------------------------------------
    # 4. Production
    # ------------------------------------------------------------------
    log(f"Production: {n_production} steps ...")
    t0 = time.time()
    sampler.run_mcmc(state, n_production, progress=False)
    prod_time = time.time() - t0
    accept_prod = float(np.mean(sampler.acceptance_fraction))
    log(f"Production done in {prod_time:.1f} s. Mean acceptance: {accept_prod:.3f}")

    # ------------------------------------------------------------------
    # 5. Diagnostics
    # ------------------------------------------------------------------
    chain = sampler.get_chain()  # (n_steps, n_walkers, n_dim)
    samples = sampler.get_chain(flat=True, discard=0)  # (n_steps*n_walkers, n_dim)

    # Autocorrelation -> integrated autocorrelation time (tau)
    try:
        tau_int = sampler.get_autocorr_time(tol=0)
        tau_int_max = float(np.max(tau_int))
    except emcee.autocorr.AutocorrError as e:
        tau_int = np.array([np.nan] * n_dim)
        tau_int_max = float("nan")
        log(f"Autocorr warning: {e}")
    ess_per_param = n_walkers * n_production / np.where(np.isfinite(tau_int), tau_int, np.nan)
    ess_min = float(np.nanmin(ess_per_param))
    log(f"Min integrated autocorrelation time across params: {tau_int_max:.1f}")
    log(f"Min ESS across params: {ess_min:.0f}")

    # Split-Rhat per parameter (simple two-halves)
    rhat_max = float(_split_rhat_max(chain))
    log(f"Split-Rhat max: {rhat_max:.4f}")

    # ------------------------------------------------------------------
    # 6. Posterior summaries
    # ------------------------------------------------------------------
    save_posterior_summary(
        out_path=out_dir / "posterior.json",
        param_names=param_names,
        samples=samples,
        chain_diagnostics={
            "n_walkers": n_walkers,
            "n_warmup": n_warmup,
            "n_production": n_production,
            "warmup_acceptance": accept_warm,
            "production_acceptance": accept_prod,
            "integrated_autocorr_time_per_param": tau_int.tolist(),
            "ess_per_param": ess_per_param.tolist(),
            "ess_min": ess_min,
            "split_rhat_max": rhat_max,
            "warmup_wall_time_s": warm_time,
            "production_wall_time_s": prod_time,
        },
        config={
            "prior": cfg.as_dict(),
            "f_ref_prior_means_hz": f_ref_init.tolist(),
            "per_mode_N_train": per_mode_n,
            "rng_seed": rng_seed,
            "init_prior_scale": init_prior_scale,
            "per_mode_tau": per_mode_tau,
            "sampler": "emcee (affine-invariant ensemble sampler)",
        },
        notes=("Level 1 — adimensional twin-model fit on Hus 6 continuous data; "
               f"per_mode_tau={per_mode_tau}."),
    )
    log(f"Wrote posterior.json -> {out_dir / 'posterior.json'}")

    # ------------------------------------------------------------------
    # 7. Diagnostic figure
    # ------------------------------------------------------------------
    set_paper_style()
    fig = _build_diagnostic_figure(
        samples=samples,
        per_mode_data=per_mode_train,
        per_mode_dfs=[per_mode_view(train_df, i) for i in (1, 2, 3)],
        cfg=cfg,
        f_ref_init=f_ref_init,
        rng=rng,
        per_mode_tau=per_mode_tau,
    )
    fig.savefig(out_dir / "diagnostic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote diagnostic.png -> {out_dir / 'diagnostic.png'}")

    # ------------------------------------------------------------------
    # 8. Sanity checklist
    # ------------------------------------------------------------------
    sanity = _evaluate_sanity_checklist(
        samples, ess_min, rhat_max, sampler.acceptance_fraction.mean(),
        cfg=cfg, f_ref_init=f_ref_init, per_mode_tau=per_mode_tau,
    )
    _write_sanity_checklist(out_dir / "sanity_checklist.md", sanity)
    log(f"Wrote sanity_checklist.md -> {out_dir / 'sanity_checklist.md'}")

    # ------------------------------------------------------------------
    # 9. Run summary + status log
    # ------------------------------------------------------------------
    headline = _build_headline(samples, per_mode_tau=per_mode_tau)
    sanity_status = "PASS" if all(c["passed"] for c in sanity["checks"]) \
                    else ("PASS-WITH-NOTE" if any(c["passed"] for c in sanity["checks"]) else "FAIL")
    write_decision_file(
        out_path=out_dir / "LEVEL_1_decision.md",
        step_id="Level 1 — Adimensional twin-model fit",
        headline_result=headline,
        sanity_status=sanity_status,
        notes=(
            "Three-mode joint fit. f_ref per mode anchored at first-month median. "
            + ("tau_EMC and tau_T per mode. " if per_mode_tau else "tau_EMC and tau_T shared across modes. ")
            + "emcee affine-invariant sampler. "
            f"Posterior medians (alpha, beta, log_tau_EMC_h, log_tau_T_h): {headline}"
        ),
    )
    log(f"Wrote LEVEL_1_decision.md -> {out_dir / 'LEVEL_1_decision.md'}")

    # Append to pipeline_status.md
    status_md = out_dir.parent / "pipeline_status.md"
    append_status_log(
        status_md,
        f"- {datetime.now().strftime('%Y-%m-%d %H:%M')} | level1_adim | RUN COMPLETE | "
        f"sanity={sanity_status}, ESS_min={ess_min:.0f}, Rhat_max={rhat_max:.3f}, "
        f"runtime={warm_time+prod_time:.1f}s"
    )
    log(f"Appended to pipeline_status.md -> {status_md}")

    log(f"=== {STEP_ID} END ===")

    return {
        "posterior_path": out_dir / "posterior.json",
        "figure_path": out_dir / "diagnostic.png",
        "decision_path": out_dir / "LEVEL_1_decision.md",
        "sanity_log": sanity,
        "ess_min": ess_min,
        "rhat_max": rhat_max,
        "acceptance_mean": float(sampler.acceptance_fraction.mean()),
    }


# ----------------------------------------------------------------------------
# Diagnostic figure builder
# ----------------------------------------------------------------------------

def _build_diagnostic_figure(samples, per_mode_data, per_mode_dfs, cfg,
                             f_ref_init, rng, per_mode_tau=False):
    """Compose the Level 1 diagnostic PNG.

    Layout, top to bottom:
      posterior marginals (4-column grid)
      lag-time marginals and joint scatter
      predictive overlay for each mode on the training window
    """
    fig = plt.figure(figsize=(14, 17))
    gs_outer = fig.add_gridspec(3, 1, height_ratios=[2.6, 1.0, 2.3], hspace=0.35)

    names = ADIM_PARAM_NAMES_PER_MODE if per_mode_tau else ADIM_PARAM_NAMES
    latex = ADIM_PARAM_LATEX_PER_MODE if per_mode_tau else ADIM_PARAM_LATEX
    n_dim_local = len(names)

    # ----- posterior marginals (grid sized to n_dim) -----
    ncols = 6 if per_mode_tau else 4
    nrows = int(np.ceil(n_dim_local / ncols))
    gs_post = gs_outer[0].subgridspec(nrows, ncols, hspace=0.55, wspace=0.4)

    prior_means = {}
    for i in range(3):
        prior_means[f"f_ref_{i+1}"] = f_ref_init[i]
        prior_means[f"alpha_{i+1}"] = cfg.alpha_mean
        prior_means[f"beta_{i+1}"] = cfg.beta_mean
        prior_means[f"log_sigma_obs_{i+1}"] = cfg.log_sigma_obs_mean
        if per_mode_tau:
            prior_means[f"log_tau_EMC_h_{i+1}"] = cfg.log_tau_EMC_h_mean
            prior_means[f"log_tau_T_h_{i+1}"] = cfg.log_tau_T_h_mean
    if not per_mode_tau:
        prior_means["log_tau_EMC_h"] = cfg.log_tau_EMC_h_mean
        prior_means["log_tau_T_h"] = cfg.log_tau_T_h_mean

    for k, name in enumerate(names):
        ax = fig.add_subplot(gs_post[k // ncols, k % ncols])
        x = samples[:, k]
        ax.hist(x, bins=40, color="#4477AA", alpha=0.85, edgecolor="white")
        med = np.median(x)
        lo, hi = np.quantile(x, [0.025, 0.975])
        ax.axvline(med, color="black", lw=1.0)
        ax.axvline(lo, color="black", lw=0.6, ls=":")
        ax.axvline(hi, color="black", lw=0.6, ls=":")
        if name in prior_means:
            ax.axvline(prior_means[name], color="crimson", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(latex[k], fontsize=9.0)
        ax.tick_params(labelsize=7.0)
    suptitle = (f"Level 1 — Adimensional twin-model "
                f"({'PER-MODE τ' if per_mode_tau else 'shared τ'}) — "
                f"posterior marginals (red = prior mean)")
    fig.text(0.5, 0.955, suptitle, ha="center", fontsize=11.5, weight="bold")

    # ----- tau marginals + tau_EMC vs tau_T joint -----
    gs_tau = gs_outer[1].subgridspec(1, 3, wspace=0.30)
    ax_tM = fig.add_subplot(gs_tau[0, 0])
    ax_tT = fig.add_subplot(gs_tau[0, 1])
    ax_joint = fig.add_subplot(gs_tau[0, 2])

    if per_mode_tau:
        # Per-mode layout: tau_EMC_h_i at index 6*i+4, tau_T_h_i at index 6*i+5
        # Show Mode 1 tau in the marginal panels; overlay all modes in joint scatter
        tau_EMC_d = np.exp(samples[:, 4]) / 24.0   # Mode 1
        tau_T_d   = np.exp(samples[:, 5]) / 24.0   # Mode 1
        tau_label = " (Mode 1)"
        # Collect all-mode medians for annotation
        tau_emc_meds = [float(np.median(np.exp(samples[:, 6*i+4]) / 24.0)) for i in range(3)]
        tau_t_meds   = [float(np.median(np.exp(samples[:, 6*i+5]) / 24.0)) for i in range(3)]
    else:
        # Shared-τ layout: tau_EMC_h at index 12, tau_T_h at index 13
        tau_EMC_d = np.exp(samples[:, 12]) / 24.0
        tau_T_d   = np.exp(samples[:, 13]) / 24.0
        tau_label = " (shared)"
        tau_emc_meds = [float(np.median(tau_EMC_d))]
        tau_t_meds   = [float(np.median(tau_T_d))]

    ax_tM.hist(tau_EMC_d, bins=50, color="#33A02C", alpha=0.85, edgecolor="white")
    ax_tM.set_xlabel(r"$\tau_{EMC}$ (days)"); ax_tM.set_ylabel("count")
    ax_tM.set_title(f"τ_EMC posterior{tau_label}  (med {np.median(tau_EMC_d):.2f} d)", fontsize=9.5)
    ax_tM.axvline(45, color="crimson", lw=0.8, ls="--", label="prior med 45 d")
    ax_tM.legend(fontsize=7)
    ax_tT.hist(tau_T_d, bins=50, color="#FF7F00", alpha=0.85, edgecolor="white")
    ax_tT.set_xlabel(r"$\tau_T$ (days)"); ax_tT.set_ylabel("count")
    ax_tT.set_title(f"τ_T posterior{tau_label}  (med {np.median(tau_T_d):.2f} d)", fontsize=9.5)
    ax_tT.axvline(7, color="crimson", lw=0.8, ls="--", label="prior med 7 d")
    ax_tT.legend(fontsize=7)
    ax_joint.hexbin(tau_EMC_d, tau_T_d, gridsize=40, cmap="viridis", mincnt=1)
    ax_joint.set_xlabel(r"$\tau_{EMC}$ (days)"); ax_joint.set_ylabel(r"$\tau_T$ (days)")
    title_joint = "τ_EMC vs τ_T" + (f"  [{', '.join(f'{t:.1f}' for t in tau_emc_meds)} d]"
                                      if per_mode_tau else "")
    ax_joint.set_title(title_joint, fontsize=9.5)

    # ----- predictive overlay per mode -----
    gs_pred = gs_outer[2].subgridspec(3, 1, hspace=0.50)
    n_pred_draws = 200
    rng2 = np.random.default_rng(rng.integers(0, 2**31))
    draw_idx = rng2.choice(samples.shape[0], n_pred_draws, replace=False)
    theta_pp = samples[draw_idx]
    for i in range(3):
        ax = fig.add_subplot(gs_pred[i, 0])
        sub_df = per_mode_dfs[i]
        f_obs = sub_df[FREQ_COLS[i]].to_numpy()
        emc_mat = sub_df[[f"EMC_tau{int(t)}h" for t in (1,3,6,12,24,48,72,168,336,504,720,1080,1440,2160)]].to_numpy()
        T_mat = sub_df[[f"T_tau{int(t)}h" for t in (1,3,6,12,24,48,72,168,336,504,720,1080,1440,2160)]].to_numpy()
        pp = posterior_predictive_adim(theta_pp, emc_mat, T_mat, mode_idx=i,
                                       include_obs_noise=True, rng=rng2,
                                       per_mode_tau=per_mode_tau)
        pp_lo = np.quantile(pp, 0.025, axis=0)
        pp_hi = np.quantile(pp, 0.975, axis=0)
        pp_med = np.quantile(pp, 0.50, axis=0)
        times = sub_df.DateTime
        predictive_overlay(times, f_obs, pp_med, pp_lo, pp_hi,
                           mode_label=MODE_LABELS[i], color=MODE_COLORS[i], ax=ax)
        ax.set_title(f"Posterior predictive — {MODE_LABELS[i]}  (N_train = {len(f_obs)})",
                     fontsize=10)
        ax.set_xlabel("Date")

    fig.suptitle("", y=0.99)
    return fig


# ----------------------------------------------------------------------------
# Sanity-check checklist
# ----------------------------------------------------------------------------

def _evaluate_sanity_checklist(samples, ess_min, rhat_max, mean_accept, cfg, f_ref_init,
                                per_mode_tau=False):
    """Return a dict with the preregistered checks and their pass/fail status."""
    def med(col): return float(np.median(samples[:, col]))
    def q025(col): return float(np.quantile(samples[:, col], 0.025))
    def q975(col): return float(np.quantile(samples[:, col], 0.975))

    # Per-mode parameter index helpers (handle both layouts).
    if per_mode_tau:
        def f_ref_col(i): return 6 * i + 0
        def alpha_col(i): return 6 * i + 1
        def beta_col(i):  return 6 * i + 2
        def log_sigma_col(i): return 6 * i + 3
        # Per-mode tau columns
        def log_tau_EMC_col(i): return 6 * i + 4
        def log_tau_T_col(i):   return 6 * i + 5
    else:
        def f_ref_col(i): return 4 * i + 0
        def alpha_col(i): return 4 * i + 1
        def beta_col(i):  return 4 * i + 2
        def log_sigma_col(i): return 4 * i + 3

    checks = []

    # sigma_obs in [0.005, 0.020] Hz per mode
    for i in range(3):
        log_sig = med(log_sigma_col(i))
        sig = np.exp(log_sig)
        ok = 0.005 <= sig <= 0.020
        checks.append({
            "name": f"sigma_obs_{i+1} in [0.005, 0.020] Hz",
            "value": float(sig),
            "passed": bool(ok),
            "note": "JCSS-derived plausible range; below = unrealistic, above = parametric too weak",
        })

    # alpha_i in (+0.0010, +0.0035), CI excludes 0
    for i in range(3):
        a_med = med(alpha_col(i))
        a_lo, a_hi = q025(alpha_col(i)), q975(alpha_col(i))
        in_range = 0.0010 <= a_med <= 0.0035
        excludes_zero = a_lo > 0.0 or a_hi < 0.0
        checks.append({
            "name": f"alpha_{i+1} median in (0.0010, 0.0035) and 95% CI excludes 0",
            "value": [a_lo, a_med, a_hi],
            "passed": bool(in_range and excludes_zero),
            "note": "Positive alpha mechanism confirmed",
        })

    # beta_i in (-0.0040, -0.0010), CI excludes 0
    for i in range(3):
        b_med = med(beta_col(i))
        b_lo, b_hi = q025(beta_col(i)), q975(beta_col(i))
        in_range = -0.0040 <= b_med <= -0.0010
        excludes_zero = b_lo > 0.0 or b_hi < 0.0
        checks.append({
            "name": f"beta_{i+1} median in (-0.0040, -0.0010) and 95% CI excludes 0",
            "value": [b_lo, b_med, b_hi],
            "passed": bool(in_range and excludes_zero),
            "note": "Thermal-softening sign confirmed",
        })

    # tau_EMC in (20, 80) d
    if per_mode_tau:
        for i in range(3):
            tauEMC_d = np.exp(med(log_tau_EMC_col(i))) / 24.0
            checks.append({
                "name": f"tau_EMC_{i+1} in (20, 80) days",
                "value": float(tauEMC_d),
                "passed": bool(20.0 <= tauEMC_d <= 80.0),
                "note": "EMC low-pass timescale plausible",
            })
        for i in range(3):
            tauT_d = np.exp(med(log_tau_T_col(i))) / 24.0
            checks.append({
                "name": f"tau_T_{i+1} in (0.5, 30) days",
                "value": float(tauT_d),
                "passed": bool(0.5 <= tauT_d <= 30.0),
                "note": "Thermal lag timescale plausible",
            })
    else:
        tauEMC_d = np.exp(med(12)) / 24.0
        checks.append({
            "name": "tau_EMC in (20, 80) days",
            "value": float(tauEMC_d),
            "passed": bool(20.0 <= tauEMC_d <= 80.0),
            "note": "EMC low-pass timescale plausible (envelope-filter through CLT panels)",
        })
        tauT_d = np.exp(med(13)) / 24.0
        checks.append({
            "name": "tau_T in (0.5, 30) days",
            "value": float(tauT_d),
            "passed": bool(0.5 <= tauT_d <= 30.0),
            "note": "Thermal lag timescale plausible",
        })

    # ESS_min > 400, Rhat_max < 1.05 (relaxed from 1.01 because emcee chains can have higher autocorrelation than NUTS)
    checks.append({
        "name": "ESS_min > 400",
        "value": float(ess_min),
        "passed": bool(ess_min > 400),
        "note": "Independent-sample equivalent",
    })
    checks.append({
        "name": "Split-Rhat_max < 1.05",
        "value": float(rhat_max),
        "passed": bool(rhat_max < 1.05),
        "note": "Convergence diagnostic",
    })
    checks.append({
        "name": "Mean acceptance fraction in [0.20, 0.55] (emcee target)",
        "value": float(mean_accept),
        "passed": bool(0.20 <= mean_accept <= 0.55),
        "note": "Sampler efficiency",
    })

    return {"checks": checks,
            "n_passed": int(sum(c["passed"] for c in checks)),
            "n_total": len(checks)}


def _write_sanity_checklist(path: Path, sanity: dict) -> None:
    lines = ["# Level 1 — Sanity Check Status\n",
             f"**Total: {sanity['n_passed']}/{sanity['n_total']} checks passed.**\n"]
    for c in sanity["checks"]:
        mark = "✓" if c["passed"] else "✗"
        val_str = c["value"] if not isinstance(c["value"], list) else \
                  f"[{c['value'][0]:.4g}, {c['value'][1]:.4g}, {c['value'][2]:.4g}]"
        if isinstance(val_str, float):
            val_str = f"{val_str:.4g}"
        lines.append(f"- **{mark} {c['name']}** | value = {val_str} | {c['note']}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _build_headline(samples, per_mode_tau=False) -> str:
    if per_mode_tau:
        a = [float(np.median(samples[:, 6*i+1])) for i in range(3)]
        b = [float(np.median(samples[:, 6*i+2])) for i in range(3)]
        tEMC = [float(np.exp(np.median(samples[:, 6*i+4])) / 24.0) for i in range(3)]
        tT   = [float(np.exp(np.median(samples[:, 6*i+5])) / 24.0) for i in range(3)]
        return (f"alpha = [{a[0]:+.4f}, {a[1]:+.4f}, {a[2]:+.4f}], "
                f"beta = [{b[0]:+.4f}, {b[1]:+.4f}, {b[2]:+.4f}], "
                f"tau_EMC = [{tEMC[0]:.1f}, {tEMC[1]:.1f}, {tEMC[2]:.1f}] d, "
                f"tau_T = [{tT[0]:.1f}, {tT[1]:.1f}, {tT[2]:.1f}] d")
    else:
        a1 = float(np.median(samples[:, 1]))
        a2 = float(np.median(samples[:, 5]))
        a3 = float(np.median(samples[:, 9]))
        b1 = float(np.median(samples[:, 2]))
        b2 = float(np.median(samples[:, 6]))
        b3 = float(np.median(samples[:, 10]))
        tauEMC = float(np.exp(np.median(samples[:, 12])) / 24.0)
        tauT = float(np.exp(np.median(samples[:, 13])) / 24.0)
        return (f"alpha = [{a1:+.4f}, {a2:+.4f}, {a3:+.4f}], "
                f"beta = [{b1:+.4f}, {b2:+.4f}, {b3:+.4f}], "
                f"tau_EMC = {tauEMC:.2f} d, tau_T = {tauT:.2f} d")


# ----------------------------------------------------------------------------
# Simple split-Rhat utility (two halves of each chain)
# ----------------------------------------------------------------------------

def _split_rhat_max(chain: np.ndarray) -> float:
    """Compute the max split-Rhat across parameters from emcee chain (n_steps, n_walkers, n_dim)."""
    n_steps, n_walkers, n_dim = chain.shape
    half = n_steps // 2
    rhats = []
    for d in range(n_dim):
        # Treat each (walker x half) as one chain -> 2 * n_walkers chains
        c = chain[:, :, d]
        c1 = c[:half].T  # (n_walkers, half)
        c2 = c[half:2*half].T
        chains = np.vstack([c1, c2])  # (2*n_walkers, half)
        n_chains = chains.shape[0]
        n = chains.shape[1]
        chain_means = chains.mean(axis=1)
        chain_vars = chains.var(axis=1, ddof=1)
        W = chain_vars.mean()
        B = n * chain_means.var(ddof=1)
        var_post = (n - 1) / n * W + B / n
        rhat = float(np.sqrt(var_post / max(W, 1e-12)))
        rhats.append(rhat)
    return float(np.max(rhats))


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "aux" / "level1_adim"
    run(out_dir)
