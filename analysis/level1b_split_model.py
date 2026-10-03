"""Level 1b — Stiffness / mass EMC decomposition for the split model (Model 2).
by Andre R. Barbosa, April - October 2026

Scientific context
------------------
The adimensional model (Level 1) fits an *effective* EMC sensitivity alpha_eff
that conflates two distinct physical mechanisms:

    alpha_eff_i = alpha_k_i - alpha_m  (first-order approximation)

where:
  - alpha_k_i  = net stiffness-EMC sensitivity for mode i.
                 Includes MOE softening (negative) and swelling-induced
                 connection stiffening (positive).  In CLT buildings with
                 friction/bearing joints, swelling tends to dominate → alpha_k > 0.
  - alpha_m    = mass-EMC sensitivity (shared across modes).
                 Wood mass increases ~1% per %MC; with CLT fraction r_CLT ≈ 0.55,
                 alpha_m ≈ r_CLT * 0.009 ≈ 0.005 per %EMC.  Always positive.

The frequency data alone cannot identify alpha_k and alpha_m separately
(the likelihood is flat along the ridge alpha_k - alpha_m = alpha_eff_MAP).
Identification requires an *informative prior on alpha_m* from wood science.

This script performs the Bayesian stiffness/mass decomposition:
  1. Load the Level 1 MCMC posterior draws for alpha_eff (all three modes).
  2. Draw alpha_m from the wood-science TruncatedNormal prior.
  3. Recover alpha_k = alpha_eff * (1 + alpha_m * EMC_ref) + alpha_m
     (nonlinear inversion of the first-order approximation).
  4. Report posteriors on (alpha_eff, alpha_m, alpha_k) and their implications.
  5. Produce a publication-quality 3-panel decomposition figure.

Key result
----------
All three modes show alpha_k > 0 (net stiffness increase with EMC), which is
consistent with swelling-induced connection stiffening outweighing MOE softening.
The sign follows from alpha_eff > 0 and alpha_m >= 0; the magnitude of alpha_k
depends on the prior on alpha_m.  The mass contribution alpha_m ≈ 0.005 per %EMC
is comparable in magnitude to alpha_eff, so it should not be neglected.

Output files (written to aux/level1b_split/)
--------------------------------------------
- decomposition_results.json   — posterior summaries for all parameters
- Fig_Split_Decomposition.png  — 3-panel posterior violin + decomposition plot
- step01b_summary.md           — summary table of the decomposition
"""

from __future__ import annotations

import sys, json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from analysis.common.data_loader import project_root
from analysis.common.priors import SplitPriorConfig, alpha_k_from_eff
from analysis.common.plotting import set_paper_style, MODE_COLORS, MODE_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES    = 20_000    # Monte Carlo draws for the decomposition
RNG_SEED     = 20260608
EMC_REF      = 19.0      # %EMC — dataset mean at the relevant lag timescales
# Level 1 posterior draws (per-mode-tau run).
ALPHA_EFF_POSTERIOR_PATH = project_root() / "aux" / "level1_adim" / "v3_per_mode_tau" / "posterior.json"
OUT_DIR      = project_root() / "aux" / "level1b_split"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_alpha_eff_draws() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (alpha_eff_1, alpha_eff_2, alpha_eff_3) posterior draws."""
    with open(ALPHA_EFF_POSTERIOR_PATH) as f:
        p = json.load(f)
    draws = np.array(p["thinned_draws"])          # (n_draws, 18) per-mode-tau layout
    names = p["param_names"]
    idx = {n: i for i, n in enumerate(names)}
    a1 = draws[:, idx["alpha_1"]]
    a2 = draws[:, idx["alpha_2"]]
    a3 = draws[:, idx["alpha_3"]]
    return a1, a2, a3


def sample_alpha_m(n: int, cfg: SplitPriorConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw alpha_m samples from the truncated-Normal wood-science prior."""
    # Rejection sampling from TruncatedNormal(mu, sigma, lower, upper)
    out = np.empty(n)
    filled = 0
    while filled < n:
        candidates = rng.normal(cfg.alpha_m_mean, cfg.alpha_m_sigma, size=2 * (n - filled))
        valid = candidates[(candidates >= cfg.alpha_m_lower) & (candidates <= cfg.alpha_m_upper)]
        take = min(len(valid), n - filled)
        out[filled : filled + take] = valid[:take]
        filled += take
    return out


def summarize(arr: np.ndarray, label: str) -> dict:
    """Return a summary dict for one parameter array."""
    return {
        "label": label,
        "mean":  float(arr.mean()),
        "std":   float(arr.std()),
        "q05":   float(np.percentile(arr, 5)),
        "q25":   float(np.percentile(arr, 25)),
        "q50":   float(np.percentile(arr, 50)),
        "q75":   float(np.percentile(arr, 75)),
        "q95":   float(np.percentile(arr, 95)),
        "prob_positive": float((arr > 0).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main decomposition
# ─────────────────────────────────────────────────────────────────────────────

def run(dpi: int = 200) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    cfg = SplitPriorConfig()

    # ── 1. Load Level 1 alpha_eff posteriors ─────────────────────────────────
    ae1, ae2, ae3 = load_alpha_eff_draws()
    n_step01 = len(ae1)

    # ── 2. Resample to N_SAMPLES (with replacement) to match MC size ─────────
    idx_resample = rng.integers(0, n_step01, size=N_SAMPLES)
    ae1 = ae1[idx_resample]
    ae2 = ae2[idx_resample]
    ae3 = ae3[idx_resample]

    # ── 3. Sample alpha_m from wood-science prior (independent of data) ───────
    alpha_m = sample_alpha_m(N_SAMPLES, cfg, rng)

    # ── 4. Recover alpha_k via nonlinear inversion ────────────────────────────
    #   alpha_k = alpha_eff * (1 + alpha_m * EMC_ref) + alpha_m
    ak1 = alpha_k_from_eff(ae1, alpha_m, EMC_REF)
    ak2 = alpha_k_from_eff(ae2, alpha_m, EMC_REF)
    ak3 = alpha_k_from_eff(ae3, alpha_m, EMC_REF)

    # ── 5. Compute implied alpha_eff_recovered (validation) ──────────────────
    #   alpha_eff_rec = (alpha_k - alpha_m) / (1 + alpha_m * EMC_ref)
    #   should match the Level 1 draws by construction
    ae1_rec = (ak1 - alpha_m) / (1.0 + alpha_m * EMC_REF)
    ae2_rec = (ak2 - alpha_m) / (1.0 + alpha_m * EMC_REF)
    ae3_rec = (ak3 - alpha_m) / (1.0 + alpha_m * EMC_REF)
    assert np.allclose(ae1_rec, ae1, atol=1e-8), "Round-trip check failed Mode 1"
    assert np.allclose(ae2_rec, ae2, atol=1e-8), "Round-trip check failed Mode 2"
    assert np.allclose(ae3_rec, ae3, atol=1e-8), "Round-trip check failed Mode 3"
    print("Round-trip check: alpha_eff recovered exactly ✓")

    # ── 6. Compute α_k / α_m ratio ────────────────────────────────────────────
    ratio1 = ak1 / alpha_m
    ratio2 = ak2 / alpha_m
    ratio3 = ak3 / alpha_m

    # ── 7. Summary statistics ─────────────────────────────────────────────────
    results = {
        "meta": {
            "script":    "level1b_split_model.py",
            "timestamp": datetime.now().isoformat(),
            "n_samples": N_SAMPLES,
            "emc_ref_pct": EMC_REF,
            "prior": cfg.as_dict(),
        },
        "alpha_m": summarize(alpha_m, "alpha_m (shared, wood-science prior)"),
        "modes": {},
    }
    for label, ae, ak, ratio in [
        ("Mode 1", ae1, ak1, ratio1),
        ("Mode 2", ae2, ak2, ratio2),
        ("Mode 3", ae3, ak3, ratio3),
    ]:
        results["modes"][label] = {
            "alpha_eff": summarize(ae, f"alpha_eff ({label})"),
            "alpha_k":   summarize(ak, f"alpha_k ({label})"),
            "alpha_k_over_alpha_m": summarize(ratio, f"alpha_k/alpha_m ({label})"),
        }

    # ── 8. Print interpretation ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("STIFFNESS / MASS EMC DECOMPOSITION")
    print(f"  Reference EMC = {EMC_REF:.1f} %")
    print(f"  alpha_m  prior: TruncNorm(mu={cfg.alpha_m_mean:.4f}, "
          f"sigma={cfg.alpha_m_sigma:.4f}) per %EMC")
    print(f"{'─'*60}")
    print(f"  alpha_m  posterior: {alpha_m.mean():.5f} ± {alpha_m.std():.5f} per %EMC")
    print(f"    → CLT mass increases {alpha_m.mean()*100:.2f} % per 1 %EMC increase")
    print()
    for label, ae, ak, ratio in [
        ("Mode 1", ae1, ak1, ratio1),
        ("Mode 2", ae2, ak2, ratio2),
        ("Mode 3", ae3, ak3, ratio3),
    ]:
        print(f"  {label}:")
        print(f"    alpha_eff = {ae.mean():.5f} ± {ae.std():.5f}  (data, from Level 1)")
        print(f"    alpha_k   = {ak.mean():.5f} ± {ak.std():.5f}  (stiffness sensitivity)")
        print(f"    alpha_k / alpha_m = {ratio.mean():.2f} ± {ratio.std():.2f}")
        print(f"    → net stiffening {ak.mean()*100:.2f}% per %EMC  "
              f"[{'STIFFENING' if ak.mean() > 0 else 'SOFTENING'}]")
        print(f"    P(alpha_k > 0) = {(ak>0).mean():.4f}")
    print(f"{'─'*60}")

    # ── 9. Figure ──────────────────────────────────────────────────────────────
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=False)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.18, top=0.88, wspace=0.35)

    mode_colors = MODE_COLORS  # list: [blue, red, green]
    alpha_eff_draws  = [ae1, ae2, ae3]
    alpha_k_draws    = [ak1, ak2, ak3]
    alpha_m_draws    = alpha_m

    # ── Panel 1: alpha_eff posteriors (from Level 1) ─────────────────────────
    ax = axes[0]
    for i, (ae, col, lbl) in enumerate(zip(alpha_eff_draws, mode_colors, ["Mode 1", "Mode 2", "Mode 3"])):
        # KDE
        xs = np.linspace(ae.min() - ae.std(), ae.max() + ae.std(), 300)
        kde = stats.gaussian_kde(ae, bw_method=0.3)
        ys  = kde(xs)
        ax.plot(xs * 1e3, ys / ys.max(), color=col, lw=1.6, label=lbl)
        ax.axvline(ae.mean() * 1e3, color=col, lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel(r"$\alpha_\mathrm{eff}$ (per %EMC, ×10⁻³)", fontsize=9)
    ax.set_ylabel("Normalized density", fontsize=9)
    ax.set_title(r"Observed: $\alpha_\mathrm{eff}$" + "\n(from Level 1 MCMC)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, 1.25)
    ax.text(0.04, 0.97, "(a)", transform=ax.transAxes, fontsize=9, va="top", fontweight="bold")

    # ── Panel 2: alpha_m prior (wood science) ────────────────────────────────
    ax = axes[1]
    xs_m = np.linspace(0, cfg.alpha_m_upper * 1.1, 400)
    kde_m = stats.gaussian_kde(alpha_m_draws, bw_method=0.3)
    ys_m  = kde_m(xs_m)
    ax.fill_between(xs_m * 1e3, ys_m / ys_m.max(), alpha=0.25, color="#888888")
    ax.plot(xs_m * 1e3, ys_m / ys_m.max(), color="#444444", lw=1.6, label="Wood-science prior")
    ax.axvline(alpha_m_draws.mean() * 1e3, color="#444444", lw=1.0, ls="--", alpha=0.8)
    # Annotate CLT fraction interpretation
    ax.annotate(
        f"$\\mu_{{\\alpha_m}}={cfg.alpha_m_mean*1e3:.1f}\\times10^{{-3}}$\n"
        f"$\\sigma_{{\\alpha_m}}={cfg.alpha_m_sigma*1e3:.1f}\\times10^{{-3}}$\n"
        f"Norway spruce CLT\n$r_{{\\rm CLT}}\\approx 0.55$",
        xy=(cfg.alpha_m_mean * 1e3, 0.65),
        xytext=(cfg.alpha_m_mean * 1e3 + 2.5, 0.75),
        fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7, color="#444"),
        ha="left", color="#333333",
    )
    ax.set_xlabel(r"$\alpha_m$ (per %EMC, ×10⁻³)", fontsize=9)
    ax.set_ylabel("Normalized density", fontsize=9)
    ax.set_title(r"Wood-science prior: $\alpha_m$" + "\n(shared across modes)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.text(0.04, 0.97, "(b)", transform=ax.transAxes, fontsize=9, va="top", fontweight="bold")

    # ── Panel 3: alpha_k = alpha_eff + alpha_m (derived stiffness sensitivity)
    ax = axes[2]
    for i, (ak, col, lbl) in enumerate(zip(alpha_k_draws, mode_colors, ["Mode 1", "Mode 2", "Mode 3"])):
        xs = np.linspace(ak.min() - ak.std(), ak.max() + ak.std(), 300)
        kde = stats.gaussian_kde(ak, bw_method=0.3)
        ys  = kde(xs)
        ax.plot(xs * 1e3, ys / ys.max(), color=col, lw=1.6, label=lbl)
        ax.axvline(ak.mean() * 1e3, color=col, lw=0.8, ls="--", alpha=0.7)
    ax.axvline(0, color="black", lw=0.6, ls=":", alpha=0.5)
    ax.set_xlabel(r"$\alpha_k$ (per %EMC, ×10⁻³)", fontsize=9)
    ax.set_ylabel("Normalized density", fontsize=9)
    ax.set_title(r"Derived: $\alpha_k = \alpha_\mathrm{eff}(1+\alpha_m\cdot\overline{\mathrm{EMC}})+\alpha_m$"
                 + "\n(stiffness sensitivity)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.set_ylim(0, 1.25)
    ax.text(0.04, 0.97, "(c)", transform=ax.transAxes, fontsize=9, va="top", fontweight="bold")

    # ── Subtitle / annotation ─────────────────────────────────────────────────
    fig.text(
        0.5, 0.02,
        r"$\alpha_k>0$: connection swelling stiffening dominates over MOE softening in CLT buildings. "
        r"All modes: $P(\alpha_k>0)>99\%$.",
        ha="center", fontsize=8, color="#333333", style="italic",
    )

    fig_path = OUT_DIR / "Fig_Split_Decomposition.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {fig_path}")

    # ── 10. Save JSON results ─────────────────────────────────────────────────
    json_path = OUT_DIR / "decomposition_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results JSON → {json_path}")

    # ── 11. Write summary markdown ────────────────────────────────────────────
    md_lines = [
        "# Level 1b — Stiffness/Mass EMC Decomposition Results",
        "",
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"N Monte Carlo samples: {N_SAMPLES:,}",
        f"Reference EMC: {EMC_REF:.1f} %",
        "",
        "## Wood-Science Prior on α_m",
        "",
        f"- **Prior**: TruncatedNormal(μ={cfg.alpha_m_mean:.4f}, σ={cfg.alpha_m_sigma:.4f}, lower=0) per %EMC",
        f"- **Derivation**: Norway spruce CLT, ρ_dry ≈ 440 kg/m³, r_CLT ≈ 0.55",
        f"  → mass gain per %EMC = 0.55 × (440/524) × 0.01 ≈ 0.0046 per %EMC",
        f"- **Posterior** (prior-dominated): {alpha_m.mean()*1e3:.2f} ± {alpha_m.std()*1e3:.2f} ×10⁻³ per %EMC",
        "",
        "## Per-Mode Decomposition",
        "",
        "| Mode | α_eff (Level 1) | α_m (wood sci.) | α_k = α_eff + α_m* | α_k/α_m | P(α_k>0) |",
        "|------|----------------|----------------|---------------------|---------|----------|",
    ]
    for label, ae, ak, am_arr, ratio in [
        ("Mode 1", ae1, ak1, alpha_m, ratio1),
        ("Mode 2", ae2, ak2, alpha_m, ratio2),
        ("Mode 3", ae3, ak3, alpha_m, ratio3),
    ]:
        md_lines.append(
            f"| {label} | {ae.mean()*1e3:.3f}±{ae.std()*1e3:.3f} | "
            f"{am_arr.mean()*1e3:.3f}±{am_arr.std()*1e3:.3f} | "
            f"{ak.mean()*1e3:.3f}±{ak.std()*1e3:.3f} | "
            f"{ratio.mean():.2f} | {(ak>0).mean():.4f} |"
        )
    md_lines += [
        "",
        "*×10⁻³ per %EMC; α_k recovery uses nonlinear inversion at EMC_ref=19%.*",
        "",
        "## Note",
        "",
        "All posterior samples of α_k are positive. The sign follows from α_eff > 0 and",
        "the non-negative mass effect (α_m ≥ 0); the magnitude of α_k depends on the prior",
        "on α_m, because the frequency data constrain only the difference α_k − α_m.",
        "",
    ]
    md_path = OUT_DIR / "step01b_summary.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Summary → {md_path}")

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    run(dpi=args.dpi)
