"""
fig08_ksweep.py
===============
by Andre R. Barbosa, April - October 2026

Fig. S1 (Supplementary Material) — GP variance-cap (k) sensitivity sweep for
Modes 1, 2, 3.

Data sources:
    aux/level3_gp/level3_Mode1_result.json
    aux/level3_gp/level3_Mode2_result.json
    aux/level3_gp/level3_Mode3_result.json

Outputs:
    analysis/output/Fig_Level3_kSweep_allModes.png   — 3-panel composite
    analysis/output/Fig_Level3_kSweep_Mode1.png      — single-panel (0.33 linewidth)
    analysis/output/Fig_Level3_kSweep_Mode2.png
    analysis/output/Fig_Level3_kSweep_Mode3.png

Usage
-----
    python analysis/fig08_ksweep.py

Requirements: numpy, matplotlib, json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "aux" / "level3_gp"
OUT_DIR = ROOT / "analysis" / "output"

OUT_COMPOSITE = OUT_DIR / "Fig_Level3_kSweep_allModes.png"
OUT_SINGLE_TMPL = OUT_DIR / "Fig_Level3_kSweep_Mode{n}.png"

DPI = 160
FIGSIZE_COMPOSITE = (13, 4.2)
FIGSIZE_SINGLE = (4.5, 4.0)

# Line styles / colors per split
SPLITS: list[dict[str, Any]] = [
    {"key": "train",       "label": "train",          "color": "#2166ac", "marker": "o", "linestyle": "-"},
    {"key": "val",         "label": "val (all)",       "color": "#d73027", "marker": "s", "linestyle": "--"},
    {"key": "val_noburst", "label": "val (no burst)",  "color": "#f4a736", "marker": "^", "linestyle": ":"},
]

FONT_SIZE = 10
FONT_FAMILY = "Arial"
COL_GRID = "0.75"

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.style": "italic",
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
    }
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def remove_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, color=COL_GRID, linewidth=0.6)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10, 1.04, label,
        transform=ax.transAxes,
        fontsize=FONT_SIZE + 1,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def k_to_x(k_values: list) -> tuple[list[float], list[str]]:
    """Convert raw k values (may include 'inf') to numeric x positions and tick labels."""
    x_pos: list[float] = []
    labels: list[str] = []
    for v in k_values:
        if v == "inf" or v == float("inf"):
            x_pos.append(float("inf"))
            labels.append(r"$\infty$")
        else:
            x_pos.append(float(v))
            labels.append("" if float(v) == 0.5 else str(v))  # 0.5 label dropped (overlaps 0.25)
    return x_pos, labels


def load_mode(mode_n: int) -> dict[str, Any]:
    path = DATA_DIR / f"level3_Mode{mode_n}_result.json"
    with open(path, "r") as fh:
        return json.load(fh)


def extract_sweep(data: dict[str, Any]) -> dict[str, Any]:
    """Return dict with x positions, labels, and per-split CRPS arrays."""
    sweep = data["k_sweep"]
    raw_k = [entry["k"] for entry in sweep]
    x_pos, labels = k_to_x(raw_k)

    # Replace inf with a display position one step beyond last finite value
    finite_vals = [v for v in x_pos if not (isinstance(v, float) and v == float("inf"))]
    step = (finite_vals[-1] - finite_vals[-2]) if len(finite_vals) >= 2 else 1.0
    x_display = [
        v if not (isinstance(v, float) and v == float("inf")) else finite_vals[-1] + step
        for v in x_pos
    ]

    result: dict[str, Any] = {
        "x_display": x_display,
        "x_labels": labels,
        "primary_k": data["primary_k"],
    }
    for sp in SPLITS:
        key = sp["key"]
        result[key] = [entry[key]["CRPS_mean"] for entry in sweep]

    return result


def draw_mode_panel(
    ax: plt.Axes,
    mode_n: int,
    sweep: dict[str, Any],
    show_legend: bool = False,
    show_ylabel: bool = True,
) -> None:
    """Draw one mode's k-sweep CRPS panel onto ax."""
    x = sweep["x_display"]
    pk = sweep["primary_k"]

    # Find display x position corresponding to primary_k
    finite_k = [
        (i, v) for i, v in enumerate(x)
        if not (isinstance(v, float) and v == float("inf"))
    ]
    # primary_k display x is simply pk (it is in the finite range)
    pk_x = float(pk)

    for sp in SPLITS:
        ax.plot(
            x,
            [v * 1e3 for v in sweep[sp["key"]]],
            color=sp["color"],
            marker=sp["marker"],
            linestyle=sp["linestyle"],
            linewidth=1.6,
            markersize=5,
            label=sp["label"],
        )

    # Vertical line at selected k
    ax.axvline(pk_x, color="0.50", linewidth=1.0, linestyle=":", zorder=0,
               label=rf"selected $\kappa$ = {pk:.1f}")

    # x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(sweep["x_labels"], fontsize=FONT_SIZE - 1)

    # no title; mode given by LaTeX subcaption/caption
    ax.set_xlabel(r"Variance cap $\kappa$")
    if show_ylabel:
        ax.set_ylabel("CRPS (mHz)")

    remove_spines(ax)
    add_grid(ax)

    if show_legend:
        ax.legend(frameon=True, framealpha=0.9, loc="center right", ncol=1, fontsize=FONT_SIZE - 2)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all three modes
    modes_data = {n: load_mode(n) for n in [1, 2, 3]}
    sweeps = {n: extract_sweep(modes_data[n]) for n in [1, 2, 3]}

    # ==================================================================
    # Composite 3-panel figure
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_COMPOSITE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.16, wspace=0.35)

    panel_labels_abc = ["(a)", "(b)", "(c)"]
    for idx, mode_n in enumerate([1, 2, 3]):
        ax = axes[idx]
        draw_mode_panel(
            ax,
            mode_n,
            sweeps[mode_n],
            show_legend=(idx == 0),
            show_ylabel=(idx == 0),
        )
        panel_label(ax, panel_labels_abc[idx])

    fig.savefig(OUT_COMPOSITE, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_COMPOSITE}")

    # ==================================================================
    # Individual single-panel figures (for 0.33\linewidth subfigures)
    # ==================================================================
    for mode_n in [1, 2, 3]:
        fig_s, ax_s = plt.subplots(1, 1, figsize=FIGSIZE_SINGLE)
        fig_s.subplots_adjust(left=0.16, right=0.95, top=0.88, bottom=0.16)

        draw_mode_panel(
            ax_s,
            mode_n,
            sweeps[mode_n],
            show_legend=True,
            show_ylabel=True,
        )

        out_path = OUT_DIR / f"Fig_Level3_kSweep_Mode{mode_n}.png"
        fig_s.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig_s)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
