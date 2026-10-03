"""
fig02_oma_timeseries.py
=======================
by Andre R. Barbosa, April - October 2026

Paper Fig. 3 — Natural frequency time series for Modes 1, 2, 3 of Limnologen Hus 6.

Marks:
  - Monthly x-ticks
  - Vertical dashed line at the training/validation split (November 1, 2025)
  - Shaded bands for all three wind-burst windows:
      Aug 5–9 2025 and Oct 4–6 2025  (training bursts, down-sampled to 6 h)
      Apr 4–11 2026                   (validation windstorm, σ ×2.3 Mode 3)

Usage:
    cd <repo root>
    python analysis/fig02_oma_timeseries.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "Limnologen_continuous_OMA_aligned.csv"
OUT_PATH = ROOT / "analysis" / "output" / "Fig_OMA_TimeSeries.png"

# ── style — matches Level 3 ─────────────────────────────────────────────────────
DPI = 160
_STYLE = {
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.style":      "italic",
    "font.size":       10,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.titlesize":  10,
}

# ── burst windows ─────────────────────────────────────────────────────────────
# Training bursts: down-sampled to 6 h cadence in Level 3 (ERA5-confirmed)
TRAIN_BURSTS = [
    (pd.Timestamp("2025-08-05"), pd.Timestamp("2025-08-09")),   # Aug 2025
    (pd.Timestamp("2025-10-04"), pd.Timestamp("2025-10-06")),   # Oct 2025
]
# Validation windstorm: Apr 2026 (24.2 m/s ERA5, σ ×2.3 Mode 3)
VAL_BURST_START = pd.Timestamp("2026-04-04")
VAL_BURST_END   = pd.Timestamp("2026-04-11")

# ── mode labels ───────────────────────────────────────────────────────────────
MODES = [
    ("f1", r"Mode 1 — $X$-translation",  r"$f_1$ (Hz)",  "#2166AC"),   # blue
    ("f2", r"Mode 2 — $Y$-translation",  r"$f_2$ (Hz)",  "#4DAC26"),   # green
    ("f3", r"Mode 3 — Torsion",           r"$f_3$ (Hz)",  "#D6604D"),   # red
]

# ── y-axis limits — set from actual data range + small margin ─────────────────
YLIM = {
    "f1": (2.00, 2.22),   # data: 2.006–2.202 Hz
    "f2": (2.17, 2.37),   # data: 2.189–2.356 Hz
    "f3": (2.32, 2.69),   # data: 2.340–2.664 Hz
}


def main():
    plt.rcParams.update(_STYLE)

    # Load data
    df = pd.read_csv(DATA_CSV, parse_dates=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Training / validation split
    t_start = df["DateTime"].min()
    t_split = pd.Timestamp("2025-11-01")   # pipeline split (common/data_loader.chronological_split)
    t_end   = df["DateTime"].max()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0), sharex=True,
                             gridspec_kw={"hspace": 0.12})

    for letter, ax, (col, label, ylabel, color) in zip("abc", axes, MODES):
        # Panel letter to the left of each subplot
        ax.text(-0.075, 1.0, f"({letter})", transform=ax.transAxes,
                fontsize=11, va="top", ha="right")
        mask = df[col].notna()
        t    = df.loc[mask, "DateTime"]
        f    = df.loc[mask, col]

        # Scatter — training
        tr = t < t_split
        ax.scatter(t[tr],  f[tr],  s=2.5, color=color,      alpha=0.55,
                   linewidths=0, rasterized=True, zorder=3)
        # Scatter — validation (darker)
        ax.scatter(t[~tr], f[~tr], s=2.5, color=color,      alpha=0.80,
                   linewidths=0, rasterized=True, zorder=3)

        # Burst shading — training bursts (lighter blue-grey)
        for tb_start, tb_end in TRAIN_BURSTS:
            ax.axvspan(tb_start, tb_end, color="#5B8DB8", alpha=0.28, zorder=2)
        # Burst shading — validation windstorm (orange)
        ax.axvspan(VAL_BURST_START, VAL_BURST_END, color="#FF7F00", alpha=0.28, zorder=2)

        # Train/val split line
        ax.axvline(t_split, color="black", linewidth=1.1, linestyle="--",
                   zorder=4, label="Train/val split")

        # 30-day rolling median for visual reference
        df_mode = df.loc[mask, ["DateTime", col]].set_index("DateTime").sort_index()
        roll = df_mode[col].rolling("30D", center=True, min_periods=12).median()
        ax.plot(roll.index, roll.values, color="black", lw=0.9, alpha=0.7, zorder=5)

        # y-axis label carries both unit and mode ID
        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_ylim(*YLIM[col])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))

        # Mode label — lower left, away from legend and top edge
        ax.text(0.01, 0.05, label, transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        ax.grid(axis="y", lw=0.4, color="gray", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # Legend — bottom panel, upper left
    train_label = f"Training set ({t_start.strftime('%b %Y')} – {(t_split - pd.Timedelta(days=1)).strftime('%b %Y')})"
    val_label   = f"Validation set ({t_split.strftime('%b %Y')} – {t_end.strftime('%b %Y')})"
    handles = [
        mpatches.Patch(color="#888888", alpha=0.55, label=train_label),
        mpatches.Patch(color="#333333", alpha=0.80, label=val_label),
        plt.Line2D([0], [0], color="black", lw=0.9, alpha=0.7,
                   label="30-day rolling median"),
        plt.Line2D([0], [0], color="black", lw=1.1, ls="--",
                   label="Train/val split"),
        mpatches.Patch(color="#5B8DB8", alpha=0.45, label="Wind burst (train; Aug & Oct 2025)"),
        mpatches.Patch(color="#FF7F00", alpha=0.45, label="Wind burst (val; Apr 2026)"),
    ]
    axes[2].legend(handles=handles, loc="upper left", fontsize=8.5,
                   framealpha=0.88, ncol=1)

    # x-axis: month ticks
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=20, ha="right")
    axes[-1].set_xlim(t_start - pd.Timedelta(days=5),
                      t_end   + pd.Timedelta(days=5))

    # No suptitle — caption serves as the figure title in the manuscript
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
