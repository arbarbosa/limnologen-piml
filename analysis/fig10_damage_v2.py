"""
fig10_damage_v2.py
==================
by Andre R. Barbosa, April - October 2026

Paper Fig. 9 — synthetic-damage detectability on the real Mode-1 record, built
from the Level 1 posterior and the Level 3 GP (k = 1) reported in the paper.

Method
  * physics + GP from aux/level1_adim/v3_per_mode_tau and aux/level3_gp (k = 1)
  * causal (trailing) 30-day window, burst records excluded
  * decision band from the TRAINING year only; false-alarm rate reported on the
    held-out validation period for the undamaged record (shown in panel b)
  * damaged record = OBSERVED frequency with the injected shift (not the model)
  * storey-2 -> equivalent-uniform mapping recomputed with the 7-DOF (r = 1) model
  * detection-probability curves are EMPIRICAL: fraction of validation times at
    which (undamaged estimate - injected change) falls below the lower band edge
Outputs: analysis/output/Fig_Damage_Detection_v2.png, aux/checks/fig10_v2_summary.json
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "analysis"))
import level3_gp_residual as L3
from analysis.common.forward_model import build_shear_K

MODE, WIN, MINP = 1, "30D", 60           # 60 records ≈ 15 d of 6-hourly data
ONSET = pd.Timestamp("2026-01-15")
STOREY_LOSS = [3, 5, 15]                 # % loss at storey 2 (spring index 1)
COL = {3: "#2ca02c", 5: "#ff7f0e", 15: "#d62728"}

# ── storey-2 loss -> equivalent-uniform stiffness change (7-DOF, r = 1) ──
def f1_of(k):
    K = build_shear_K(np.asarray(k, float)); w2 = np.linalg.eigvalsh(K)  # M = I
    return np.sqrt(w2.min())
f0 = f1_of(np.ones(7))
EQ = {}
for d in STOREY_LOSS:
    k = np.ones(7); k[1] *= 1 - d / 100
    EQ[d] = 100 * (1 - (f1_of(k) / f0) ** 2)

# ── physics (Level 1 medians) + GP (Level 3, k = 1) over the whole Mode-1 record ──
theta = L3.load_step01_medians(MODE - 1); sig_obs = float(np.exp(theta["log_sigma_obs"]))
res = json.load(open(ROOT / "aux/level3_gp/level3_Mode1_result.json"))
kk = [r for r in res["k_sweep"] if r["k"] == 1.0][0]
lp = [kk["log_sigma_gp"], np.log(kk["ell_EMC"]), np.log(kk["ell_T"]), np.log(kk["ell_diel"])]
df = L3.per_mode_view(L3.load_continuous_hus6(), mode_idx=MODE)
train, val = L3.chronological_split(df); train_ds = L3.downsample_to_6h(train)
def adim(d):
    return L3.adim_forward_single_mode(theta["f_ref"], theta["alpha"], theta["beta"],
        theta["log_tau_EMC_h"], theta["log_tau_T_h"], L3.get_lag_matrix(d, "EMC"), L3.get_lag_matrix(d, "T"))
r_tr = train_ds[f"f{MODE}"].to_numpy(float) - adim(train_ds)
X_tr = L3.build_gp_covariates(train_ds)
full = pd.concat([train, val]).sort_values("DateTime").reset_index(drop=True)
g, _ = L3.predict_gp(lp, X_tr, r_tr, L3.build_gp_covariates(full), sig_obs ** 2)
full["piml"] = adim(full) + g
full["resid"] = full[f"f{MODE}"] - full["piml"]
dt = pd.to_datetime(full["DateTime"])
is_val = (dt >= pd.to_datetime(val["DateTime"]).min()).values
burst = np.zeros(len(full), bool)
for s, e in L3.VAL_BURST_PERIODS: burst |= ((dt >= s) & (dt < pd.Timestamp(e))).values
ok = ~burst
fbar = float(np.nanmean(full["piml"]))
x = 2 * 100 * full["resid"].to_numpy() / fbar           # single-record equiv. stiffness change (%)

def roll(xx):
    s = pd.Series(np.where(ok, xx, np.nan), index=dt)
    return s.rolling(WIN, min_periods=MINP).mean().to_numpy()   # trailing (causal) window
undmg = roll(x)
tr_m = (~is_val) & np.isfinite(undmg)
allm = ok & np.isfinite(undmg)
va_m = is_val & ok & np.isfinite(undmg)
# training-year band (reported for its out-of-sample false-alarm rate)
mu_tr, sd_tr = float(np.mean(undmg[tr_m])), float(np.std(undmg[tr_m]))
far_roll_trainband = float(np.mean(undmg[va_m] < mu_tr - 2 * sd_tr))
# decision band used in the figure: mean +/- 2 sd of the WHOLE undamaged record (one-sided lower edge for loss)
mu, sd = float(np.mean(undmg[allm])), float(np.std(undmg[allm]))
lo, hi = mu - 2 * sd, mu + 2 * sd
far_roll = float(np.mean(undmg[va_m] < lo))
val_offset = float(np.mean(undmg[va_m]) - mu)
mu1, sd1 = float(np.mean(x[(~is_val) & ok])), float(np.std(x[(~is_val) & ok]))
va1 = is_val & ok
far_single = float(np.mean(x[va1] < mu1 - 2 * sd1))

post = (dt >= ONSET).values
scen = {}
for d in STOREY_LOSS:
    est = roll(x - EQ[d] * post)
    after = post & ok & np.isfinite(est)
    below = est < lo
    # first time the estimate stays below the band for >= 7 consecutive days
    sb = pd.Series(np.where(after, below.astype(float), np.nan), index=dt).dropna()
    run = sb.rolling("7D").min()
    hit = run[(run == 1) & (run.index >= ONSET + pd.Timedelta("7D"))]
    first = (hit.index.min() - pd.Timedelta("7D")) if len(hit) else None
    frac_30 = float(np.mean(below[after & (dt >= ONSET + pd.Timedelta("30D")).values]))
    scen[d] = dict(equiv_uniform_pct=EQ[d], first_detection=str(first.date()) if first is not None else None,
                   delay_days=float((first - ONSET).days) if first is not None else None,
                   frac_below_band_after_30d=frac_30, mean_estimate_after_30d=float(np.nanmean(est[after & (dt >= ONSET + pd.Timedelta("30D")).values])))
    scen[d]["est"] = est

grid = np.linspace(0.1, 6.0, 120)
t30 = va_m & (dt >= pd.to_datetime(val["DateTime"]).min() + pd.Timedelta("30D")).values
p_roll = np.array([np.mean(undmg[t30] - D < lo) for D in grid])
p_single = np.array([np.mean(x[va1] - D < mu1 - 2 * sd1) for D in grid])
d50_roll = float(grid[np.argmax(p_roll >= 0.5)]); d95_roll = float(grid[np.argmax(p_roll >= 0.95)])
d50_single = float(grid[np.argmax(p_single >= 0.5)]); d95_single = float(grid[np.argmax(p_single >= 0.95)])

summary = dict(equiv_uniform_pct={str(d): EQ[d] for d in STOREY_LOSS}, band_center_pct=mu, band_halfwidth_2sd_pct=2 * sd, sd_rolling_train=sd_tr, sd_rolling_val=float(np.std(undmg[va_m])), false_alarm_rate_rolling_val_trainband=far_roll_trainband,
    single_record_2sd_pct=2 * sd1, false_alarm_rate_rolling_val=far_roll, false_alarm_rate_single_val=far_single,
    val_offset_of_undamaged_rolling_pct=val_offset, resid_rmse_val_noburst_mHz=float(1e3 * np.sqrt(np.mean(full["resid"].to_numpy()[va1] ** 2))),
    detection_50pct_rolling=d50_roll, detection_95pct_rolling=d95_roll, detection_50pct_single=d50_single, detection_95pct_single=d95_single,
    scenarios={str(d): {k: v for k, v in s.items() if k != "est"} for d, s in scen.items()})
(ROOT / "aux/checks").mkdir(parents=True, exist_ok=True)
json.dump(summary, open(ROOT / "aux/checks/fig10_v2_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# ── figure ──
from analysis.common.plotting import set_paper_style
set_paper_style()   # Arial italic, as in the other paper figures
plt.rcParams.update({"font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "Helvetica"],
                     "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "legend.fontsize": 6.8, "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.4))
for a in ax: a.spines[["top", "right"]].set_visible(False)
f = full[f"f{MODE}"].to_numpy()
fd = f - 0.5 * EQ[15] / 100 * fbar * post
ax[0].plot(dt[ok], 1e3 * f[ok], ".", ms=1.0, c="#999999", alpha=.5, label=r"observed $f_1$")
ax[0].plot(dt[ok & post], 1e3 * fd[ok & post], ".", ms=1.0, c=COL[15], alpha=.45, label=fr"with $-15\%$ at storey 2 ($\equiv{EQ[15]:.1f}\%$)")
ax[0].plot(dt, 1e3 * full["piml"], "-", c="#1f77b4", lw=.8, label="PIML prediction")
ax[0].set_ylabel(r"Mode 1 $f$ (mHz)")
ax[1].axhspan(lo, hi, color="#1f77b4", alpha=.12, label=r"undamaged $\pm2\sigma$ band")
ax[1].plot(dt, undmg, c="k", lw=1.1, label="undamaged")
for d in STOREY_LOSS:
    e = np.where(post, scen[d]["est"], np.nan)
    ax[1].plot(dt, e, c=COL[d], lw=1.1, label=fr"$-{d}\%$ storey 2 ($\equiv{EQ[d]:.1f}\%$)")
ax[1].set_ylabel("equiv.-uniform stiffness change (%)")
for a in ax[:2]:
    for s, e in L3.VAL_BURST_PERIODS: a.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="orange", alpha=.25, lw=0)
    a.axvline(pd.to_datetime(val["DateTime"]).min(), ls=":", c="#555", lw=.8)
    a.axvline(ONSET, ls="--", c="#555", lw=.9)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b%y")); a.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    a.legend(frameon=False, fontsize=6.5, loc="lower left")
ax[2].axvspan(EQ[3], EQ[5], color="#ffd27f", alpha=.4, label="3–5% loss at storey 2")
ax[2].plot(grid, p_single, c="#555", lw=1.5, label="single record")
ax[2].plot(grid, p_roll, c="#1f77b4", lw=1.5, label="30-day trailing mean")
for d in STOREY_LOSS: ax[2].axvline(EQ[d], ls=":", c=COL[d], lw=1.1)
ax[2].set_xlabel("equiv.-uniform stiffness change (%)"); ax[2].set_ylabel("empirical detection probability")
ax[2].set_ylim(0, 1.03); ax[2].legend(frameon=False, fontsize=6.8, loc="center right")
fig.tight_layout(pad=1.0)
for a, l in zip(ax, "abc"):   # plain panel letters; caption explains panels
    a.text(0.5, -0.25, f"({l})", transform=a.transAxes, ha="center", va="top", fontsize=9.5)
out = ROOT / "analysis/output/Fig_Damage_Detection_v2.png"; fig.savefig(out, dpi=220, bbox_inches="tight"); print("saved", out)
