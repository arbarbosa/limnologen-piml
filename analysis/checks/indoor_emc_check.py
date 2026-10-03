"""Indoor climate is estimated from
by Andre R. Barbosa, April - October 2026
outdoor data with the EN ISO 13788 humidity-class approach (no measured indoor data yet).
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "analysis")
from level0_build_dataset import hailwood_emc

wx = pd.read_csv("data/smhi_weather_Vaxjo_64510.csv", parse_dates=["DateTime"]).set_index("DateTime")
wx = wx[~wx.index.duplicated()].asfreq("h").interpolate(limit=6)
T, RH = wx["T_out"].values, wx["RH_out"].values

def vsat(Tc):  # saturation vapour density, g/m3
    p = 610.94*np.exp(17.625*Tc/(Tc+243.04))
    return p/(461.5*(Tc+273.15))*1000
def indoor(dv0):
    Tin = np.maximum(21.0, T)                      # heated to 21 C, free-floating above
    dv  = dv0*np.clip((20-T)/20, 0, 1)             # ISO 13788: dv0 at T<=0, 0 at T>=20
    v   = RH/100*vsat(T) + dv
    return Tin, np.clip(100*v/vsat(Tin), 5, 100)

def ema(x, tau_h):
    lam = 1-1/tau_h; out = np.empty_like(x); acc = x[0]
    for i, xi in enumerate(x):
        acc = lam*acc + (1-lam)*xi; out[i] = acc
    return out

emc_out = hailwood_emc(T, RH)
res = {}
for name, dv0 in [("class2 (dv=4 g/m3)", 4.0), ("class3 (dv=6 g/m3)", 6.0)]:
    Tin, RHin = indoor(dv0); res[name] = (hailwood_emc(Tin, RHin), RHin)

oma = pd.read_csv("Limnologen_continuous_OMA_aligned.csv", parse_dates=["DateTime"]).set_index("DateTime")
idx = wx.index.get_indexer(oma.index.round("h"), method="nearest")

print("Monthly means (Nov 2024 - May 2026):")
m = pd.DataFrame({"EMC_out": emc_out, "EMC_in_c2": res["class2 (dv=4 g/m3)"][0],
                  "RH_in_c2": res["class2 (dv=4 g/m3)"][1], "T_out": T}, index=wx.index).resample("QS").mean()
print(m.round(1).to_string())
print()
print("corr(EMC_out, EMC_in) hourly:", {k: round(np.corrcoef(emc_out, v[0])[0,1], 2) for k, v in res.items()})
print()
for tau_d in (11, 17):
    tau = tau_d*24
    Tl = ema(T, tau); Eo = ema(emc_out, tau)
    print(f"--- lag tau = {tau_d} d ---")
    for mode in ("f1", "f2", "f3"):
        ok = oma[mode].notna().values
        f = oma[mode].values[ok]; ii = idx[ok]
        row = [f"{mode}: r(f,EMC_out)={np.corrcoef(f, Eo[ii])[0,1]:+.2f}"]
        for k, (Ein, _) in res.items():
            Ei = ema(Ein, tau)[ii]
            row.append(f"r(f,EMC_in {k[:6]})={np.corrcoef(f, Ei)[0,1]:+.2f}")
            # linear fit f ~ a + b*EMC + c*T for sign of alpha
        X = np.column_stack([np.ones(ok.sum()), Eo[ii], Tl[ii]]); bo = np.linalg.lstsq(X, f, rcond=None)[0]
        Ei = ema(res["class2 (dv=4 g/m3)"][0], tau)[ii]
        X = np.column_stack([np.ones(ok.sum()), Ei, Tl[ii]]); bi = np.linalg.lstsq(X, f, rcond=None)[0]
        r2 = lambda X,b: 1-np.var(f-X@b)/np.var(f)
        row.append(f"| joint fit with T: b_EMCout={bo[1]*1e3:+.2f} mHz/%  b_EMCin={bi[1]*1e3:+.2f} mHz/%")
        print("  ".join(row))

print("\nCollinearity with lagged T (tau=17 d), at OMA timestamps:")
tau = 17*24; Tl = ema(T, tau)[idx]
for k, E in [("EMC_out", emc_out), ("EMC_in class2", res["class2 (dv=4 g/m3)"][0]), ("EMC_in class3", res["class3 (dv=6 g/m3)"][0])]:
    El = ema(E, tau)[idx]; r = np.corrcoef(El, Tl)[0,1]
    print(f"  r({k}, T) = {r:+.2f}   VIF = {1/(1-r**2):.1f}")
