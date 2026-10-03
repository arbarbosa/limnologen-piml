"""Level 1 re-fit with a sign-neutral alpha prior N(0, 0.003^2); all else as reported.
by Andre R. Barbosa, April - October 2026
"""
import sys, json, numpy as np, pandas as pd, emcee
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from analysis.common.data_loader import load_continuous_hus6, per_mode_view, get_lag_matrix, chronological_split, FREQ_COLS
from analysis.common.priors import AdimPriorConfig
from analysis.level1_adimensional_twin import make_log_posterior
df = load_continuous_hus6(apply_filters=True); tr, _ = chronological_split(df); data, fr = [], np.empty(3)
for i in (1, 2, 3):
    sub = per_mode_view(tr, mode_idx=i, apply_pole_filter=True)
    fr[i-1] = float(sub.loc[sub.DateTime <= sub.DateTime.min() + pd.Timedelta(days=30), FREQ_COLS[i-1]].median())
    data.append({"f_obs": sub[FREQ_COLS[i-1]].to_numpy(float), "emc_lag_matrix": get_lag_matrix(sub, "EMC"), "t_lag_matrix": get_lag_matrix(sub, "T")})
import dataclasses; cfg = dataclasses.replace(AdimPriorConfig(), alpha_mean=0.0, alpha_sigma=0.003)
rng = np.random.default_rng(20260609)
pm = np.concatenate([[fr[k], cfg.alpha_mean, cfg.beta_mean, cfg.log_sigma_obs_mean, cfg.log_tau_EMC_h_mean, cfg.log_tau_T_h_mean] for k in range(3)])
ps = np.tile([cfg.f_ref_sigma_hz, cfg.alpha_sigma, cfg.beta_sigma, cfg.log_sigma_obs_sigma, cfg.log_tau_EMC_h_sigma, cfg.log_tau_T_h_sigma], 3)
s = emcee.EnsembleSampler(64, 18, make_log_posterior(data, fr, cfg, per_mode_tau=True))
st = s.run_mcmc(pm + ps * rng.standard_normal((64, 18)), 3000, progress=False); s.reset(); s.run_mcmc(st, 8000, progress=False)
ch = s.get_chain(flat=True, thin=20)
ref = json.load(open(ROOT / "aux/level1_adim/v3_per_mode_tau/posterior.json")); D = np.array(ref["thinned_draws"]); ix = {n: i for i, n in enumerate(ref["param_names"])}
out = {}
for k in range(3):
    a = ch[:, 6*k+1]; b = ch[:, 6*k+2]; tE = np.exp(ch[:, 6*k+4]) / 24; tT = np.exp(ch[:, 6*k+5]) / 24
    ra = D[:, ix[f"alpha_{k+1}"]]
    q = lambda x: [float(v) for v in np.percentile(x, [2.5, 50, 97.5])]
    out[f"Mode {k+1}"] = dict(alpha_neutral=q(a*1e3), alpha_reported=q(ra*1e3), P_alpha_pos=float((a > 0).mean()),
                             beta_neutral=q(b*1e3), tauEMC_d=q(tE), tauT_d=q(tT))
    print(f"Mode {k+1}: alpha neutral {q(a*1e3)[1]:.2f} [{q(a*1e3)[0]:.2f},{q(a*1e3)[2]:.2f}] vs reported {q(ra*1e3)[1]:.2f} [{q(ra*1e3)[0]:.2f},{q(ra*1e3)[2]:.2f}] e-3; P(alpha>0)={(a>0).mean():.3f}; beta {q(b*1e3)[1]:.2f}; tauEMC {q(tE)[1]:.1f} d; tauT {q(tT)[1]:.1f} d")
(ROOT / "aux/checks").mkdir(exist_ok=True); json.dump(out, open(ROOT / "aux/checks/alpha_prior_sensitivity.json", "w"), indent=1)
