"""Synthetic identifiability study for the Level 1 per-mode lag model.
by Andre R. Barbosa, April - October 2026

Synthetic frequency records are generated on the real training design (same
timestamps, same lagged EMC/T covariates, same pole filter) from known parameters,
with Gaussian noise at the identified sigma_obs, and re-fitted with exactly the
Level 1 prior, likelihood and emcee settings (64 walkers, 1500 warm-up, 4000 steps).

Cases (tau in days; other parameters = real posterior medians):
  A realistic : tau_EMC = real medians, tau_T = real medians
  B spread    : tau_EMC = (10, 20, 40) -- 2x separations, one value inside 25-55 d band
  C null      : tau_EMC = (17, 17, 17), tau_T = (7, 7, 7) -- tests spurious separation
Two noise seeds per case.  Output: aux/checks/synthetic_lag_identifiability.json
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, emcee
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from analysis.common.data_loader import load_continuous_hus6, per_mode_view, get_lag_matrix, chronological_split, FREQ_COLS
from analysis.common.priors import AdimPriorConfig, log_prior_adim_per_mode, ADIM_PARAM_NAMES_PER_MODE as NAMES
from analysis.common.forward_model import adim_forward_single_mode, log_likelihood_adim_per_mode
from analysis.level1_adimensional_twin import make_log_posterior

OUT = ROOT / "aux" / "checks"; OUT.mkdir(parents=True, exist_ok=True)
post = json.load(open(ROOT / "aux/level1_adim/v3_per_mode_tau/posterior.json"))
D = np.array(post["thinned_draws"]); med = np.median(D, axis=0); ix = {n: i for i, n in enumerate(post["param_names"])}

df = load_continuous_hus6(apply_filters=True); tr, _ = chronological_split(df)
design, f_ref_init = [], np.empty(3)
for i in (1, 2, 3):
    sub = per_mode_view(tr, mode_idx=i, apply_pole_filter=True)
    fm = sub.loc[sub.DateTime <= sub.DateTime.min() + pd.Timedelta(days=30)]
    f_ref_init[i-1] = float(fm[FREQ_COLS[i-1]].median())
    design.append({"emc_lag_matrix": get_lag_matrix(sub, "EMC"), "t_lag_matrix": get_lag_matrix(sub, "T")})

real_tE = [np.exp(med[ix[f"log_tau_EMC_h_{i}"]])/24 for i in (1,2,3)]
real_tT = [np.exp(med[ix[f"log_tau_T_h_{i}"]])/24 for i in (1,2,3)]
CASES = {"A_realistic": (real_tE, real_tT), "B_spread": ([10., 20., 40.], real_tT), "C_null": ([17.]*3, [7.]*3)}
cfg = AdimPriorConfig()
_out = OUT / "synthetic_lag_identifiability.json"
results = json.load(open(_out)) if _out.exists() else {"prior": cfg.as_dict(), "cases": {}}
ONLY = sys.argv[1:]  # optional: case names to run

def fit(data, seed):
    rng = np.random.default_rng(seed)
    pm = np.concatenate([[f_ref_init[k], cfg.alpha_mean, cfg.beta_mean, cfg.log_sigma_obs_mean, cfg.log_tau_EMC_h_mean, cfg.log_tau_T_h_mean] for k in range(3)])
    ps = np.tile([cfg.f_ref_sigma_hz, cfg.alpha_sigma, cfg.beta_sigma, cfg.log_sigma_obs_sigma, cfg.log_tau_EMC_h_sigma, cfg.log_tau_T_h_sigma], 3)
    p0 = pm + ps * rng.standard_normal((64, 18))
    s = emcee.EnsembleSampler(64, 18, make_log_posterior(data, f_ref_init, cfg, per_mode_tau=True))
    st = s.run_mcmc(p0, 1500, progress=False); s.reset(); s.run_mcmc(st, 4000, progress=False)
    return s.get_chain(flat=True, thin=50)

for cname, (tE, tT) in CASES.items():
    if ONLY and cname not in ONLY: continue
    results["cases"][cname] = {"truth_tau_EMC_d": tE, "truth_tau_T_d": tT, "fits": []}
    for seed in (1, 2):
        t0 = time.time(); rng = np.random.default_rng(1000 + seed); data = []
        for k in range(3):
            th = dict(f_ref=med[ix[f"f_ref_{k+1}"]], alpha=med[ix[f"alpha_{k+1}"]], beta=med[ix[f"beta_{k+1}"]])
            f = adim_forward_single_mode(th["f_ref"], th["alpha"], th["beta"], np.log(tE[k]*24), np.log(tT[k]*24), **design[k])
            sig = np.exp(med[ix[f"log_sigma_obs_{k+1}"]])
            data.append({**design[k], "f_obs": f + sig * rng.standard_normal(f.size)})
        ch = fit(data, 20260609 + seed)
        r = {"seed": seed, "modes": []}
        tEs = [np.exp(ch[:, 6*k+4])/24 for k in range(3)]; tTs = [np.exp(ch[:, 6*k+5])/24 for k in range(3)]
        for k in range(3):
            def summ(x, truth): q = np.percentile(x, [2.5, 50, 97.5]); return {"truth": float(truth), "median": float(q[1]), "ci95": [float(q[0]), float(q[2])], "covered": bool(q[0] <= truth <= q[2])}
            r["modes"].append({"tau_EMC_d": summ(tEs[k], tE[k]), "tau_T_d": summ(tTs[k], tT[k]),
                               "alpha": summ(ch[:, 6*k+1], med[ix[f"alpha_{k+1}"]]), "beta": summ(ch[:, 6*k+2], med[ix[f"beta_{k+1}"]]),
                               "corr_alpha_beta": float(np.corrcoef(ch[:, 6*k+1], ch[:, 6*k+2])[0, 1]),
                               "sd_log_tau_EMC": float(np.log(tEs[k]).std())})
        r["P_tauEMC"] = {"1>2": float((tEs[0] > tEs[1]).mean()), "1>3": float((tEs[0] > tEs[2]).mean()), "3>2": float((tEs[2] > tEs[1]).mean())}
        r["P_tauT"] = {"1>2": float((tTs[0] > tTs[1]).mean()), "1>3": float((tTs[0] > tTs[2]).mean())}
        r["seconds"] = time.time() - t0
        results["cases"][cname]["fits"].append(r)
        json.dump(results, open(OUT / "synthetic_lag_identifiability.json", "w"), indent=1)
        print(cname, seed, f"{r['seconds']:.0f}s", flush=True)
# real-data alpha-beta posterior correlations
results["real_corr_alpha_beta"] = [float(np.corrcoef(D[:, ix[f"alpha_{k}"]], D[:, ix[f"beta_{k}"]])[0, 1]) for k in (1,2,3)]
json.dump(results, open(OUT / "synthetic_lag_identifiability.json", "w"), indent=1); print("DONE", flush=True)
