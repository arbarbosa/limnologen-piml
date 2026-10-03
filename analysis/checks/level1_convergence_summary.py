"""Per-parameter convergence and comparison with the reported Level 1 posterior (Table 1).
by Andre R. Barbosa, April - October 2026
"""
import json, numpy as np
from pathlib import Path
from scipy.stats import rankdata, norm
D = Path("aux/checks/level1_long_chain"); names = json.load(open("aux/level1_adim/v3_per_mode_tau/posterior.json"))["param_names"]
rep = np.array(json.load(open("aux/level1_adim/v3_per_mode_tau/posterior.json"))["thinned_draws"])
c1, c2 = np.load(D/"chain_seed20260610.npy"), np.load(D/"chain_seed20260611.npy")
def split_rhat(x):  # x: (n, m) draws x chains -> rank-normalised split-Rhat (Vehtari et al. 2021, bulk)
    n = x.shape[0] // 2; z = np.hstack([x[:n], x[n:2*n]])
    z = norm.ppf((rankdata(z).reshape(z.shape) - 0.375) / (z.size + 0.25))
    W = z.var(axis=0, ddof=1).mean(); B = n * z.mean(axis=0).var(ddof=1)
    return np.sqrt(((n-1)/n*W + B/n) / W)
def between_runs(a, b):  # two runs as two chains (walkers pooled)
    x = np.column_stack([a.reshape(-1), b.reshape(-1)]); return split_rhat(x)
res = {}
print(f"{'param':18s} Rhat_run1 Rhat_run2 Rhat_pooled(128 walkers) Rhat_between_runs | median reported -> long")
tf = lambda n, v: np.exp(v)/24 if "tau" in n else (np.exp(v)*1e3 if "sigma" in n else (v*1e3 if n.split('_')[0] in ("alpha","beta") else v))
for k, n in enumerate(names):
    r1, r2 = split_rhat(c1[:, :, k]), split_rhat(c2[:, :, k])
    rp = split_rhat(np.concatenate([c1[:, :, k], c2[:, :, k]], axis=1)); rb = between_runs(c1[:, :, k], c2[:, :, k])
    long = np.concatenate([c1[:, :, k].ravel(), c2[:, :, k].ravel()])
    mr, ml = np.median(tf(n, rep[:, k])), np.median(tf(n, long))
    ql = np.percentile(tf(n, long), [2.5, 97.5]); qr = np.percentile(tf(n, rep[:, k]), [2.5, 97.5])
    res[n] = dict(rhat_run1=float(r1), rhat_run2=float(r2), rhat_pooled=float(rp), rhat_between=float(rb),
                  median_reported=float(mr), ci_reported=[float(q) for q in qr], median_long=float(ml), ci_long=[float(q) for q in ql])
    print(f"{n:18s} {r1:8.4f} {r2:8.4f} {rp:10.4f} {rb:12.4f}      | {mr:9.4g} [{qr[0]:.4g},{qr[1]:.4g}] -> {ml:9.4g} [{ql[0]:.4g},{ql[1]:.4g}]")
json.dump(res, open(D/"convergence_summary.json", "w"), indent=1)
print("max pooled Rhat:", max(v['rhat_pooled'] for v in res.values()), " max between-run Rhat:", max(v['rhat_between'] for v in res.values()))
