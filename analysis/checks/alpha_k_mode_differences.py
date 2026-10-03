"""Between-mode differences of alpha_k from the analytical split model
by Andre R. Barbosa, April - October 2026
(same draws, seed and inversion as level1b_split_model.run(), without writing outputs)."""
import sys, numpy as np
sys.path.insert(0, "analysis")
import level1b_split_model as m
rng = np.random.default_rng(m.RNG_SEED); cfg = m.SplitPriorConfig()
ae = m.load_alpha_eff_draws(); idx = rng.integers(0, len(ae[0]), size=m.N_SAMPLES)
ae = [a[idx] for a in ae]; am = m.sample_alpha_m(m.N_SAMPLES, cfg, rng)
ak = [m.alpha_k_from_eff(a, am, m.EMC_REF) for a in ae]
print("alpha_k mean±sd (x1e-3):", [f"{a.mean()*1e3:.2f}±{a.std()*1e3:.2f}" for a in ak], " alpha_m:", f"{am.mean()*1e3:.2f}±{am.std()*1e3:.2f}")
print("corr(ak1,ak2)=%.2f corr(ak1,am)=%.2f" % (np.corrcoef(ak[0],ak[1])[0,1], np.corrcoef(ak[0],am)[0,1]))
for (i,j) in [(0,1),(0,2),(1,2)]:
    d = ak[i]-ak[j]; q = np.percentile(d,[2.5,97.5])*1e3
    print(f"ak{i+1}-ak{j+1}: {d.mean()*1e3:.2f}±{d.std()*1e3:.2f}  95%CI [{q[0]:.2f},{q[1]:.2f}]  P(>0)={(d>0).mean():.4f}")
