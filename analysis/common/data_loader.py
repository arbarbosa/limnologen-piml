"""Canonical data loaders for the analysis.
by Andre R. Barbosa, April - October 2026

Applies the record filters in a single place so every step uses the
same definition of "valid data". See `aux/pipeline_design.md` for the rationale.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants of the analysis configuration
# ----------------------------------------------------------------------------

# Pre-computed lag columns in the aligned CSV. All in hours.
LAG_TAU_HOURS = np.array(
    [1, 3, 6, 12, 24, 48, 72, 168, 336, 504, 720, 1080, 1440, 2160], dtype=float
)
EMC_LAG_COLS = [f"EMC_tau{int(t)}h" for t in LAG_TAU_HOURS]
T_LAG_COLS = [f"T_tau{int(t)}h" for t in LAG_TAU_HOURS]

FREQ_COLS = ["f1", "f2", "f3"]
DAMP_COLS = [f"Mode_{i}_Damping(%)" for i in (1, 2, 3)]
POLE_COLS = [f"Mode_{i}_PoleCount" for i in (1, 2, 3)]

# Record filter thresholds
POLE_COUNT_MIN = 5  # minimum number of stable SSI poles per mode
DAMPING_MAX_PCT = 6.0  # ζ < 6 % outlier cutoff used in EDA + downstream fits


def project_root() -> Path:
    """Return the project root (the repository root)."""
    here = Path(__file__).resolve()
    # analysis/common/data_loader.py -> project root is two levels up
    return here.parent.parent.parent


def load_continuous_hus6(
    apply_filters: bool = True,
    csv_path: Path | None = None,
) -> pd.DataFrame:
    """Load the continuous Hus 6 OMA dataset, aligned to SMHI weather + lag cols.

    The file is assumed to be `Limnologen_continuous_OMA_aligned.csv`
    (Limnologen Hus 6 only).

    Parameters
    ----------
    apply_filters : bool
        If True (default), apply the pole-count and damping
        filters. If False, return the raw aligned data.
    csv_path : Path or None
        Override the default path (project_root / `Limnologen_continuous_OMA_aligned.csv`).

    Returns
    -------
    DataFrame indexed by integer with parsed DateTime column. Has fields
    `f1,f2,f3` for frequencies, `Mode_{i}_Damping(%)` and `Mode_{i}_PoleCount`
    for damping and SSI pole counts, environmental fields, and the 14 EMC_tau
    + 14 T_tau lagged columns.
    """
    if csv_path is None:
        csv_path = project_root() / "Limnologen_continuous_OMA_aligned.csv"
    df = pd.read_csv(csv_path, parse_dates=["DateTime"])
    if apply_filters:
        # Damping outlier filter: any mode > DAMPING_MAX_PCT -> mask that mode's
        # frequency too, so per-mode filters downstream see consistent missingness.
        for fcol, dcol in zip(FREQ_COLS, DAMP_COLS):
            bad = df[dcol] > DAMPING_MAX_PCT
            df.loc[bad, [fcol, dcol]] = np.nan
    return df


def per_mode_view(
    df: pd.DataFrame,
    mode_idx: int,  # 1, 2, 3
    apply_pole_filter: bool = True,
) -> pd.DataFrame:
    """Return the rows valid for fitting a single mode.

    Drops rows where the mode's frequency is missing or pole count < `POLE_COUNT_MIN`.

    Parameters
    ----------
    df : DataFrame
        From `load_continuous_hus6()`.
    mode_idx : int
        1, 2, or 3 (which mode to filter to).
    apply_pole_filter : bool
        Apply pole-count >= 5 filter (default True).

    Returns
    -------
    DataFrame, indexed by integer, of valid records for that mode.
    """
    if mode_idx not in (1, 2, 3):
        raise ValueError("mode_idx must be 1, 2, or 3.")
    fcol = FREQ_COLS[mode_idx - 1]
    pcol = POLE_COLS[mode_idx - 1]
    mask = df[fcol].notna()
    if apply_pole_filter:
        mask &= df[pcol] >= POLE_COUNT_MIN
    return df.loc[mask].reset_index(drop=True)


def get_lag_matrix(df: pd.DataFrame, kind: str) -> np.ndarray:
    """Extract the (N, 14) lag matrix for EMC or T.

    Parameters
    ----------
    df : DataFrame
    kind : "EMC" or "T"

    Returns
    -------
    (N, 14) float array, columns indexed by `LAG_TAU_HOURS`.
    """
    if kind == "EMC":
        return df[EMC_LAG_COLS].to_numpy(dtype=float)
    if kind == "T":
        return df[T_LAG_COLS].to_numpy(dtype=float)
    raise ValueError("kind must be 'EMC' or 'T'.")


def chronological_split(
    df: pd.DataFrame,
    train_end: str = "2025-10-31",
    val_end: str = "2026-05-27",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, val) DataFrames split at the training/validation boundary.

    Train: Nov 2024 -- Oct 2025 (~ one full annual cycle).
    Val:   Nov 2025 -- May 2026 (second cycle).
    """
    if "DateTime" not in df.columns:
        raise KeyError("Expected a 'DateTime' column on the DataFrame.")
    train_end_ts = pd.Timestamp(train_end) + pd.Timedelta(hours=23, minutes=59)
    val_end_ts = pd.Timestamp(val_end) + pd.Timedelta(hours=23, minutes=59)
    train = df.loc[df.DateTime <= train_end_ts].reset_index(drop=True)
    val = df.loc[(df.DateTime > train_end_ts) & (df.DateTime <= val_end_ts)].reset_index(drop=True)
    return train, val


def load_spot_tests(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the Jan-Feb 2024 spot tests aligned to weather + lag cols.

    Contains 4 Hus 6 + 2 Hus 8 spot tests.
    """
    if csv_path is None:
        csv_path = project_root() / "spot_tests_aligned.csv"
    return pd.read_csv(csv_path, parse_dates=["DateTime"])
