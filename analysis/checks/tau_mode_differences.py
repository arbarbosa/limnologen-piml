"""Between-mode differences of the lag times from the Level 1 posterior
by Andre R. Barbosa, April - October 2026
(same thinned draws as Table 1)."""
import json, numpy as np
p = json.load(open("aux/level1_adim/v3_per_mode_tau/posterior.json"))
d = np.array(p["thinned_draws"]); n = {k: i for i, k in enumerate(p["param_names"])}
for name in ("tau_EMC_h", "tau_T_h"):
    t = [np.exp(d[:, n[f"log_{name}_{i}"]]) / 24 for i in (1, 2, 3)]   # days
    print(f"\n{name} (days): medians", [f"{np.median(x):.1f} [{np.percentile(x,2.5):.1f}, {np.percentile(x,97.5):.1f}]" for x in t])
    for i, j in [(0, 1), (0, 2), (2, 1)]:
        r = np.log(t[i] / t[j]); q = np.exp(np.percentile(r, [2.5, 50, 97.5]))
        print(f"  mode{i+1}/mode{j+1}: ratio median {q[1]:.2f}, 95% CI [{q[0]:.2f}, {q[2]:.2f}], P(mode{i+1} > mode{j+1}) = {(r>0).mean():.3f}")
    print("  corr(log tau) 1-2, 1-3, 2-3:", [round(np.corrcoef(np.log(t[a]), np.log(t[b]))[0,1], 2) for a, b in [(0,1),(0,2),(1,2)]])
# also: fraction of posterior mass inside the 25-55 d theory band
t = [np.exp(d[:, n[f"log_tau_EMC_h_{i}"]]) / 24 for i in (1, 2, 3)]
print("\nP(tau_EMC in 25-55 d):", [round(((x >= 25) & (x <= 55)).mean(), 3) for x in t])
# prior vs posterior width (log sd): prior sd 0.5
print("posterior sd(log tau_EMC):", [round(np.log(x).std(), 2) for x in t], "(prior 0.5)")
