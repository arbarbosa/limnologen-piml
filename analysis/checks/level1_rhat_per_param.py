"""Independent long Level 1 chain with per-parameter split-R-hat; saves chain for pooling.
by Andre R. Barbosa, April - October 2026
"""
import sys, json, numpy as np, pandas as pd, emcee
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from analysis.common.data_loader import load_continuous_hus6, per_mode_view, get_lag_matrix, chronological_split, FREQ_COLS
from analysis.common.priors import AdimPriorConfig, ADIM_PARAM_NAMES_PER_MODE as NAMES
from analysis.level1_adimensional_twin import make_log_posterior
seed = int(sys.argv[1]); OUT = ROOT / "aux/checks/level1_long_chain"; OUT.mkdir(parents=True, exist_ok=True)
df = load_continuous_hus6(apply_filters=True); tr, _ = chronological_split(df); data, fr = [], np.empty(3)
for i in (1, 2, 3):
    sub = per_mode_view(tr, mode_idx=i, apply_pole_filter=True)
    fr[i-1] = float(sub.loc[sub.DateTime <= sub.DateTime.min() + pd.Timedelta(days=30), FREQ_COLS[i-1]].median())
    data.append({"f_obs": sub[FREQ_COLS[i-1]].to_numpy(float), "emc_lag_matrix": get_lag_matrix(sub, "EMC"), "t_lag_matrix": get_lag_matrix(sub, "T")})
cfg = AdimPriorConfig(); rng = np.random.default_rng(seed)
pm = np.concatenate([[fr[k], cfg.alpha_mean, cfg.beta_mean, cfg.log_sigma_obs_mean, cfg.log_tau_EMC_h_mean, cfg.log_tau_T_h_mean] for k in range(3)])
ps = np.tile([cfg.f_ref_sigma_hz, cfg.alpha_sigma, cfg.beta_sigma, cfg.log_sigma_obs_sigma, cfg.log_tau_EMC_h_sigma, cfg.log_tau_T_h_sigma], 3)
s = emcee.EnsembleSampler(64, 18, make_log_posterior(data, fr, cfg, per_mode_tau=True))
st = s.run_mcmc(pm + ps * rng.standard_normal((64, 18)), 4000, progress=False); s.reset(); s.run_mcmc(st, 16000, progress=False)
ch = s.get_chain()[::10]   # thin by 10 to keep file small: (1600, 64, 18)
np.save(OUT / f"chain_seed{seed}.npy", ch)
print("saved", ch.shape, "tau_max", float(np.max(s.get_autocorr_time(tol=0))))
