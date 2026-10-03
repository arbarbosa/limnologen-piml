"""
level0_build_dataset.py
=======================
by Andre R. Barbosa, April - October 2026

Build  Limnologen_continuous_OMA_aligned.csv  from two raw inputs:

  1. OMA results table   — one row per recording window, columns:
        DateTime, f1, f2, f3,
        Mode_1_Damping(%), Mode_2_Damping(%), Mode_3_Damping(%),
        Mode_1_PoleCount,  Mode_2_PoleCount,  Mode_3_PoleCount

  2. SMHI weather table  — hourly observations from Växjö A
     (station 64510), columns: DateTime, T_out (°C), RH_out (%)

Both tables can be supplied as CSV file paths on the command line.
If the weather file is omitted the script fetches it from the SMHI
open-data API (requires internet access).

Usage
-----
From the project root:

    cd <repository root>
    python analysis/level0_build_dataset.py \
        --oma   data/oma_continuous_raw.csv  \
        --smhi  data/smhi_weather_raw.csv    \
        --out   Limnologen_continuous_OMA_aligned.csv

Or to fetch SMHI automatically:

    python analysis/level0_build_dataset.py \
        --oma   data/oma_continuous_raw.csv

If you already have the aligned CSV and just need to re-add the lag
columns (e.g., after changing the lag grid):

    python analysis/level0_build_dataset.py --relag existing_aligned.csv

Output
------
  Limnologen_continuous_OMA_aligned.csv   (at project root)

Columns in the output CSV
-------------------------
  DateTime                    — UTC or local timestamp (parsed as-is)
  f1, f2, f3                  — modal natural frequencies (Hz)
  Mode_{1,2,3}_Damping(%)     — modal damping ratios (%)
  Mode_{1,2,3}_PoleCount      — SSI stable pole counts
  T_out                       — outdoor air temperature (°C)
  RH_out                      — outdoor relative humidity (%)
  EMC                         — equilibrium moisture content (%, Hailwood-Horrobin)
  EMC_tau{t}h, T_tau{t}h     — EMA-filtered EMC and T at 14 lag values
                               t ∈ {1,3,6,12,24,48,72,168,336,504,720,1080,1440,2160}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests  # only needed for --smhi fetch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT  = PROJECT_ROOT / "Limnologen_continuous_OMA_aligned.csv"

# Lag grid in hours (matches analysis/common/data_loader.py)
LAG_TAU_HOURS = [1, 3, 6, 12, 24, 48, 72, 168, 336, 504, 720, 1080, 1440, 2160]

# SMHI open-data station — Växjö A (station 64510, corrected archive)
SMHI_STATION = 64510
SMHI_PARAM_T  = 1   # air temperature (°C), hourly
SMHI_PARAM_RH = 6   # relative humidity (%), hourly

# Required columns in the OMA input table
OMA_REQUIRED_COLS = [
    "DateTime",
    "f1", "f2", "f3",
    "Mode_1_Damping(%)", "Mode_2_Damping(%)", "Mode_3_Damping(%)",
    "Mode_1_PoleCount",  "Mode_2_PoleCount",  "Mode_3_PoleCount",
]


# ---------------------------------------------------------------------------
# EMC — Hailwood-Horrobin sorption isotherm for wood
# ---------------------------------------------------------------------------
def hailwood_emc(T_C: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
    """Equilibrium moisture content of wood (%).

    Uses the Hailwood-Horrobin model as tabulated in the Wood Handbook
    (USDA, 2010, Table 4-2) and widely used in European timber standards.

    Note: the coefficient set below (W = 349 + 1.29 T + ...) is the Celsius
    parameterization (Simpson 1998 / Wood Handbook metric form), so T is used in
    degC (e.g. EMC = 12.0% at 21 degC / 65% RH, cf. Wood Handbook Table 4-2).

    Parameters
    ----------
    T_C   : air temperature (°C)
    RH_pct: relative humidity (%)

    Returns
    -------
    EMC in % (dry-basis)
    """
    T = np.asarray(T_C, dtype=float)   # coefficients below expect °C
    h = np.clip(np.asarray(RH_pct, dtype=float) / 100.0, 0.0, 0.99)

    W  = 349.0 + 1.29  * T + 0.0135  * T**2
    K  = 0.805 + 7.36e-4 * T - 2.73e-6 * T**2
    K1 = 6.27  - 9.38e-3 * T - 3.03e-4 * T**2
    K2 = 1.91  + 4.07e-2 * T - 2.93e-4 * T**2

    Kh   = K * h
    denom1 = 1.0 - Kh
    denom2 = 1.0 + K1 * Kh + K1 * K2 * Kh**2

    # Guard against division by zero near saturation
    denom1 = np.where(np.abs(denom1) < 1e-9, 1e-9, denom1)
    denom2 = np.where(np.abs(denom2) < 1e-9, 1e-9, denom2)

    # Standard Wood Handbook / Simpson (1973) Hailwood-Horrobin form:
    # second numerator term is (K1 K h + 2 K1 K2 K^2 h^2) = K1 Kh (1 + 2 K2 Kh).
    emc = (1800.0 / W) * (
        Kh / denom1
        + K1 * Kh * (1.0 + 2.0 * K2 * Kh) / denom2
    )
    return np.clip(emc, 0.0, 30.0)   # physical range: 0–30 %


# ---------------------------------------------------------------------------
# EMA lag computation
# ---------------------------------------------------------------------------
def add_ema_lags(
    df: pd.DataFrame,
    col: str,
    tau_hours: list[int],
    prefix: str,
) -> pd.DataFrame:
    """Add EMA-filtered columns for `col` at each lag in `tau_hours`.

    The filter alpha is:  alpha = 1 - exp(-dt_h / tau_h)
    Applied with pandas ewm(alpha=alpha, adjust=False).mean() on an hourly
    resampled + interpolated series, then joined back to the original index.

    Parameters
    ----------
    df         : DataFrame with DateTime column (not necessarily hourly spaced).
    col        : column name to filter (e.g., "EMC" or "T_out").
    tau_hours  : list of lag time constants in hours.
    prefix     : column prefix for output (e.g., "EMC" or "T").

    Returns
    -------
    df with new columns f"{prefix}_tau{t}h" for each t in tau_hours.
    """
    # Work on a 1-hour resample to make the EMA dt uniform.
    dt_h = df.set_index("DateTime")[col]
    hourly = dt_h.resample("1h").mean().interpolate("time")

    for tau_h in tau_hours:
        alpha = 1.0 - np.exp(-1.0 / tau_h)   # dt = 1 h
        filtered = hourly.ewm(alpha=alpha, adjust=False).mean()
        # Map back to the original (non-uniform) timestamps
        lag_col = f"{prefix}_tau{tau_h}h"
        df[lag_col] = filtered.reindex(df["DateTime"], method="nearest", tolerance="2h").values

    return df


# ---------------------------------------------------------------------------
# SMHI API fetch
# ---------------------------------------------------------------------------
def fetch_smhi(station: int, param: int, label: str) -> pd.Series:
    """Fetch a single hourly parameter from the SMHI open-data API.

    Returns a pandas Series indexed by UTC datetime.
    """
    url = (
        f"https://opendata-download-metobs.smhi.se/api/version/1.0/"
        f"parameter/{param}/station/{station}/period/corrected-archive/data.csv"
    )
    print(f"  Fetching SMHI {label} from: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    lines = resp.text.splitlines()
    # SMHI CSV has a multi-line header; data starts after the line with 'Datum'
    data_start = next(
        i for i, ln in enumerate(lines) if ln.startswith("Datum")
    )
    from io import StringIO
    raw = pd.read_csv(
        StringIO("\n".join(lines[data_start:])),
        sep=";",
        parse_dates={"DateTime": ["Datum", "Tid (UTC)"]},
        dayfirst=False,
    )
    val_col = [c for c in raw.columns if c not in ("DateTime", "Kvalitet")][0]
    s = pd.to_numeric(raw[val_col], errors="coerce")
    s.index = raw["DateTime"]
    s.name = label
    return s.dropna()


def load_or_fetch_smhi(smhi_path: Optional[Path]) -> pd.DataFrame:
    """Load SMHI data from CSV or fetch from API.

    Expected CSV columns: DateTime, T_out (°C), RH_out (%).
    """
    if smhi_path is not None:
        print(f"[level0] Loading SMHI weather from: {smhi_path}")
        wdf = pd.read_csv(smhi_path, parse_dates=["DateTime"])
        assert "T_out"  in wdf.columns, "SMHI CSV missing 'T_out' column"
        assert "RH_out" in wdf.columns, "SMHI CSV missing 'RH_out' column"
        return wdf.sort_values("DateTime").reset_index(drop=True)

    print("[level0] Fetching SMHI weather from open-data API …")
    T_series  = fetch_smhi(SMHI_STATION, SMHI_PARAM_T,  "T_out")
    RH_series = fetch_smhi(SMHI_STATION, SMHI_PARAM_RH, "RH_out")

    wdf = pd.DataFrame({"T_out": T_series, "RH_out": RH_series}).dropna()
    wdf.index.name = "DateTime"
    wdf = wdf.reset_index()
    return wdf.sort_values("DateTime").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------
def build_aligned_dataset(
    oma_path: Path,
    smhi_path: Optional[Path],
    out_path: Path,
) -> pd.DataFrame:
    """Build and save the aligned dataset.

    Steps
    -----
    1. Load OMA results (one row per recording window).
    2. Load / fetch SMHI weather (hourly T, RH).
    3. Align: for each OMA timestamp find the nearest hourly weather obs
       within ±1.5 h.
    4. Compute EMC via Hailwood-Horrobin.
    5. Add 14 EMA lag columns for EMC and T.
    6. Save CSV.
    """
    # ── 1. OMA results ──────────────────────────────────────────────────────
    print(f"[level0] Loading OMA results from: {oma_path}")
    oma = pd.read_csv(oma_path, parse_dates=["DateTime"])
    missing = [c for c in OMA_REQUIRED_COLS if c not in oma.columns]
    if missing:
        raise ValueError(
            f"OMA CSV is missing required columns: {missing}\n"
            f"  Expected: {OMA_REQUIRED_COLS}\n"
            f"  Found:    {list(oma.columns)}"
        )
    oma = oma.sort_values("DateTime").reset_index(drop=True)
    print(f"  {len(oma)} records, "
          f"{oma.DateTime.min().date()} – {oma.DateTime.max().date()}")

    # ── 2. SMHI weather ─────────────────────────────────────────────────────
    weather = load_or_fetch_smhi(smhi_path)
    print(f"  SMHI: {len(weather)} hourly observations, "
          f"{weather.DateTime.min().date()} – {weather.DateTime.max().date()}")

    # ── 3. Align OMA ↔ weather ──────────────────────────────────────────────
    weather_idx = weather.set_index("DateTime")
    weather_hourly = weather_idx.resample("1h").mean().interpolate("time")

    aligned_T  = []
    aligned_RH = []
    for ts in oma["DateTime"]:
        # Find nearest hourly slot within ±1.5 h
        slot = ts.round("h")
        if slot in weather_hourly.index:
            aligned_T.append(float(weather_hourly.loc[slot, "T_out"]))
            aligned_RH.append(float(weather_hourly.loc[slot, "RH_out"]))
        else:
            aligned_T.append(np.nan)
            aligned_RH.append(np.nan)

    oma["T_out"]  = aligned_T
    oma["RH_out"] = aligned_RH

    n_missing = oma["T_out"].isna().sum()
    if n_missing > 0:
        print(f"  WARNING: {n_missing} OMA records have no SMHI match "
              f"(gap > 1.5 h or outside SMHI coverage).")

    # ── 4. EMC ──────────────────────────────────────────────────────────────
    oma["EMC"] = hailwood_emc(oma["T_out"].values, oma["RH_out"].values)
    print(f"  EMC range: {oma.EMC.min():.2f} – {oma.EMC.max():.2f} %")

    # ── 5. EMA lag columns ──────────────────────────────────────────────────
    print(f"[level0] Computing {len(LAG_TAU_HOURS)} EMA lag columns for EMC and T …")
    oma = add_ema_lags(oma, "EMC",   LAG_TAU_HOURS, "EMC")
    oma = add_ema_lags(oma, "T_out", LAG_TAU_HOURS, "T")

    # ── 6. Save ─────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oma.to_csv(out_path, index=False)
    print(f"\n[level0] Saved {len(oma)} rows × {len(oma.columns)} columns")
    print(f"         → {out_path}")
    return oma


# ---------------------------------------------------------------------------
# Re-lag only mode (fast re-add of lag columns to existing aligned CSV)
# ---------------------------------------------------------------------------
def relag_existing(csv_path: Path, out_path: Path) -> None:
    """Re-compute EMA lag columns for an existing aligned CSV.

    Use this if you change the lag grid or fix the EMA formula without
    needing to re-run OMA or re-fetch SMHI.
    """
    print(f"[level0] Re-lag mode: loading {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["DateTime"])
    assert "EMC"   in df.columns, "CSV missing 'EMC' column"
    assert "T_out" in df.columns, "CSV missing 'T_out' column"

    # Drop old lag cols (re-add fresh)
    old_lag = [c for c in df.columns
               if c.startswith("EMC_tau") or c.startswith("T_tau")]
    df = df.drop(columns=old_lag)

    df = add_ema_lags(df, "EMC",   LAG_TAU_HOURS, "EMC")
    df = add_ema_lags(df, "T_out", LAG_TAU_HOURS, "T")

    df.to_csv(out_path, index=False)
    print(f"[level0] Re-lagged CSV saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oma",  type=Path, default=None,
        help="Path to OMA results CSV (DateTime, f1, f2, f3, damping, pole counts).",
    )
    p.add_argument(
        "--smhi", type=Path, default=None,
        help="Path to SMHI weather CSV (DateTime, T_out, RH_out). "
             "If omitted, fetches from SMHI open-data API.",
    )
    p.add_argument(
        "--out",  type=Path, default=DEFAULT_OUT,
        help=f"Output path. Default: {DEFAULT_OUT}",
    )
    p.add_argument(
        "--relag", type=Path, default=None, metavar="EXISTING_CSV",
        help="Re-compute lag columns for an existing aligned CSV (skips OMA/SMHI steps).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.relag is not None:
        relag_existing(args.relag, args.out)
        return

    if args.oma is None:
        print("ERROR: --oma is required unless --relag is used.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Typical usage:", file=sys.stderr)
        print("  python analysis/level0_build_dataset.py \\", file=sys.stderr)
        print("      --oma  data/oma_continuous_raw.csv \\", file=sys.stderr)
        print("      --smhi data/smhi_weather_raw.csv", file=sys.stderr)
        print("", file=sys.stderr)
        print("OMA CSV format (one row per recording window):", file=sys.stderr)
        print("  DateTime, f1, f2, f3,", file=sys.stderr)
        print("  Mode_1_Damping(%), Mode_2_Damping(%), Mode_3_Damping(%),", file=sys.stderr)
        print("  Mode_1_PoleCount, Mode_2_PoleCount, Mode_3_PoleCount", file=sys.stderr)
        print("", file=sys.stderr)
        print("SMHI CSV format (hourly, Växjö A station 64510):", file=sys.stderr)
        print("  DateTime, T_out, RH_out", file=sys.stderr)
        print("  (If omitted the script fetches from SMHI open-data API)", file=sys.stderr)
        sys.exit(1)

    if not args.oma.exists():
        print(f"ERROR: OMA file not found: {args.oma}", file=sys.stderr)
        sys.exit(1)

    build_aligned_dataset(
        oma_path=args.oma,
        smhi_path=args.smhi,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
