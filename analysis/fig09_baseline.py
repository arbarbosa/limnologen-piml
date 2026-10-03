"""
fig09_baseline.py
=================
by Andre R. Barbosa, April - October 2026

Paper Fig. 5 — Concurrent vs lagged environmental covariates.
Shows that modal frequency correlates weakly with concurrent EMC/T but
strongly with the diffusion-lagged covariates (the "unaccounted factors"
of Entezami et al. are the time-domain memory).

Outputs: figures/fig09.png  (paper)  +  figures/Fig_Baseline_Contrast.png
Usage:   python analysis/fig09_baseline.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.plotting import set_paper_style
MODE_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]  # same mode colours as the other figures (Mode 2 green, Mode 3 red)

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "Limnologen_continuous_OMA_aligned.csv"
FIGDIR = ROOT / "figures"
GREY = "#9e9e9e"

# lags taken from the Level 1 posterior (per-mode medians), covariates log-interpolated
import json as _json
_P = _json.load(open(ROOT / "aux/level1_adim/v3_per_mode_tau/posterior.json"))
_D = np.array(_P["thinned_draws"]); _IX = {n: i for i, n in enumerate(_P["param_names"])}
TAU_EMC_H = {m: float(np.exp(np.median(_D[:, _IX[f"log_tau_EMC_h_{m}"]]))) for m in (1, 2, 3)}
TAU_T_H = {m: float(np.exp(np.median(_D[:, _IX[f"log_tau_T_h_{m}"]]))) for m in (1, 2, 3)}
_GRID = np.array([1, 3, 6, 12, 24, 48, 72, 168, 336, 504, 720, 1080, 1440, 2160])
def lagged(df, var, tau_h):
    j = np.searchsorted(_GRID, tau_h); lo, hi = _GRID[j - 1], _GRID[j]
    w = (np.log(tau_h) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (1 - w) * df[f"{var}_tau{lo}h"].to_numpy() + w * df[f"{var}_tau{hi}h"].to_numpy()


def r2_multi(y, X):
    X = np.column_stack([np.ones(len(X))] + [X[:, i] for i in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ beta
    return 1.0 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)


LAGS = np.array([1, 3, 6, 12, 24, 48, 72, 168, 336, 504, 720, 1080, 1440, 2160])
TAU_EMC_D = {m: TAU_EMC_H[m] / 24 for m in (1, 2, 3)}


def main():
    set_paper_style()
    df = pd.read_csv(CSV)
    r2c, r2l = [], []
    for m in (1, 2, 3):
        f = f"f{m}"
        d2 = df.assign(_E=lagged(df, "EMC", TAU_EMC_H[m]), _T=lagged(df, "T", TAU_T_H[m]))
        sub = d2[[f, "EMC", "T_out", "_E", "_T"]].dropna()
        y = sub[f].to_numpy()
        r2c.append(r2_multi(y, sub[["EMC", "T_out"]].to_numpy()))
        r2l.append(r2_multi(y, sub[["_E", "_T"]].to_numpy()))

    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
    x = np.arange(3)
    w = 0.38
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    # (a) variance explained: concurrent vs lagged
    ax[0].bar(x - w / 2, r2c, w, color=GREY, edgecolor="white", label="concurrent (no lag)")
    print("rho profile peaks and values at identified tau printed below")
    for i in range(3):
        ax[0].bar(x[i] + w / 2, r2l[i], w, color=MODE_COLORS[i], edgecolor="white",
                  label="lagged (identified $\\tau$)" if i == 0 else None)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["Mode 1", "Mode 2", "Mode 3"])
    ax[0].set_ylabel(r"$R^2$ of $f$ vs (EMC, $T$)")
    ax[0].legend(frameon=False, loc="upper right")

    # (b) moisture lag profile: |rho(f, EMC_lag)| vs lag, per mode
    ld = LAGS / 24.0
    for i, m in enumerate((1, 2, 3)):
        f = f"f{m}"
        rr = []
        for h in LAGS:
            s = df[[f, f"EMC_tau{h}h"]].dropna()
            rr.append(abs(np.corrcoef(s[f"EMC_tau{h}h"], s[f])[0, 1]))
        ax[1].semilogx(ld, rr, "o-", ms=3, color=MODE_COLORS[i], label=f"Mode {m}")
        print(f"Mode {m}: |rho| lag0={rr[0]:.2f} peak={max(rr):.2f} at {ld[int(np.argmax(rr))]:.1f} d; tau_EMC={TAU_EMC_D[m]:.1f} d")
        ax[1].axvline(TAU_EMC_D[m], ls=":", color=MODE_COLORS[i], lw=1.0)
    ax[1].set_xlabel("EMC lag (days)")
    ax[1].set_ylabel(r"$|\rho(f,\widetilde{\mathrm{EMC}})|$")
    ax[1].legend(frameon=False, loc="lower right")

    fig.tight_layout(pad=1.2)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for name in ("fig09.png", "Fig_Baseline_Contrast.png"):
        fig.savefig(FIGDIR / name, dpi=200, bbox_inches="tight")
    # individual panels for LaTeX subfigures (letters set by \subcaption)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for a, suf in zip(ax, "ab"):
        bb = a.get_tightbbox(rend).transformed(fig.dpi_scale_trans.inverted()).padded(0.05)
        fig.savefig(FIGDIR / f"fig09{suf}.png", dpi=200, bbox_inches=bb)
    plt.close(fig)
    print("Saved fig09 (baseline contrast). R2c=%s R2l=%s" %
          ([f"{v:.2f}" for v in r2c], [f"{v:.2f}" for v in r2l]))


if __name__ == "__main__":
    main()
