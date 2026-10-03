"""
wdd_oma.py
----------
by Andre R. Barbosa, April - October 2026

PSD + SSI-cov OMA analysis of a WebDAQ-504 .wdd raw acceleration file.

Workflow
--------
  1. Load .wdd (unfiltered, Fs=2048 Hz) via wdd_loader.load_wdd()
  2. Compute Welch PSD at full 2048 Hz — two panels (0–10 Hz and full Nyquist)
  3. Decimate to 64 Hz (factor 32, IIR anti-alias, zero-phase)
  4. Run pyOMA2 FDD + SSI-cov on decimated data
  5. Plot combined FDD singular values + stabilisation diagram
  6. Print stable-pole histogram in 1–5 Hz band

Usage
-----
    cd <repository root>
    python analysis/wdd_oma.py
    python analysis/wdd_oma.py data/other_file.wdd
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend, decimate

from pyoma2.algorithms.fdd import FDD, FDDRunParams
from pyoma2.algorithms.ssi import SSI, SSIRunParams
from pyoma2.setup.single import SingleSetup
from pyoma2.functions.plot import stab_plot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wdd_loader import load_wdd

# ── configuration ──────────────────────────────────────────────────────────
DEFAULT_WDD   = os.path.join(HERE, "..", "data",
                             "Limnologen_2025-05-02T07-00-00-008.wdd")
DEC_FACTOR    = 32          # 2048 → 64 Hz
FS_DEC        = 64          # Hz after decimation

# PSD (at full 2048 Hz)
NPERSEG_PSD   = 4*65536       # df = 2048/65536/4 = 0.031/4 Hz

# SSI-cov (on decimated 64 Hz data)
BR            = 40          # block rows
ORDMIN        = 2
ORDMAX        = 60
STEP          = 2
SSI_SC        = {"err_fn": 0.01, "err_xi": 0.10, "err_phi": 0.05}
SSI_HC        = {"xi_max": 0.10, "mpc_lim": 0.5, "mpd_lim": 0.5, "CoV_max": 0.2}

# FDD (on decimated data)
NXSEG_FDD     = 2048        # df = 64/2048 = 0.031 Hz, ~61 averages
FMAX_PLOT     = 10.0        # Hz for structural-range plots

# Known/expected frequency markers
MODES = [
    # (freq_Hz, color, label, linestyle, alpha, stagger_row)
    (1.667, "#999999", "1.667 Hz (harmonic)",  ":",  0.70, 1),
    (2.130, "#2ca02c", "2.13 Hz (Mode 1)",     "-",  0.92, 2),
    (2.260, "#8c564b", "2.26 Hz (Mode 2)",     "-",  0.92, 0),
    (2.480, "#e377c2", "2.48 Hz (Mode 3)",     "-",  0.92, 1),
]
ROW_FRAC = [0.06, 0.18, 0.30]   # stagger fractions from top of log y-axis

OUTPUT_DIR    = os.path.join(HERE, "..", "output")
DPI           = 160


# ── helpers ────────────────────────────────────────────────────────────────
def add_vlines(ax, modes, fmax=None, label=True, row_y=None):
    for fv, col, lbl, ls, alpha, row in modes:
        if fmax is not None and fv > fmax:
            continue
        ax.axvline(fv, color=col, lw=1.1, ls=ls, alpha=alpha, zorder=3)
        if label and row_y is not None:
            ax.text(fv + 0.07, row_y[row], lbl,
                    color=col, fontsize=8.5, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=col, alpha=0.88, linewidth=0.7))


# ═══════════════════════════════════════════════════════════════════════════
def main():
    fpath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WDD
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────
    data, FS, ch_names, _ = load_wdd(fpath)
    ch1, ch2 = data[:, 0], data[:, 1]
    N = len(ch1)
    print(f"Loaded: {N} samples, {N/FS:.1f} s @ {FS} Hz")
    print(f"  {ch_names[0]}: std={ch1.std()*1e3:.3f} mg")
    print(f"  {ch_names[1]}: std={ch2.std()*1e3:.3f} mg")

    # ── 2. PSD at full Fs ─────────────────────────────────────────────────
    n_avg = (N - NPERSEG_PSD) // (NPERSEG_PSD // 2) + 1
    print(f"\nPSD: df={FS/NPERSEG_PSD:.4f} Hz, ~{n_avg} averages")
    f_psd, P1 = welch(detrend(ch1), fs=FS, window="hann",
                      nperseg=NPERSEG_PSD, noverlap=NPERSEG_PSD//2,
                      scaling="density")
    _,     P2 = welch(detrend(ch2), fs=FS, window="hann",
                      nperseg=NPERSEG_PSD, noverlap=NPERSEG_PSD//2,
                      scaling="density")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10.5})

    fig_psd, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig_psd.suptitle(
        f"Limnologen Hus 6 — Unfiltered PSD (WebDAQ-504, g)  |  {os.path.basename(fpath)}\n"
        rf"$F_s$={FS} Hz, N={N:,}, Welch $n_{{seg}}$={NPERSEG_PSD}, "
        rf"$\Delta f$={FS/NPERSEG_PSD:.3f} Hz, ~{n_avg} averages",
        fontsize=10.5, weight="bold")

    for ax, fmax, title in zip(axes,
            [FMAX_PLOT, FS / 2],
            [f"0 – {FMAX_PLOT:.0f} Hz (structural range)",
             f"0 – {FS//2} Hz (full Nyquist)"]):
        mask = f_psd <= fmax
        ax.semilogy(f_psd[mask], P1[mask], "#1f77b4", lw=1.2, alpha=0.9, label="Ch1 X")
        ax.semilogy(f_psd[mask], P2[mask], "#d62728", lw=1.2, alpha=0.8, label="Ch2 Y")
        ax.set_xlim(0, fmax)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD  (g² / Hz)")
        ax.set_title(title, fontsize=10.5, weight="bold")
        ax.grid(True, which="both", alpha=0.20)
        ax.legend(fontsize=9, loc="upper right")
        if fmax <= FMAX_PLOT:
            ax.axvspan(1.5, 3.0, color="#FFFFCC", alpha=0.25, zorder=0)
            ylo, yhi = ax.get_ylim()
            log_span = np.log10(yhi) - np.log10(ylo)
            row_y = [10 ** (np.log10(yhi) - f * log_span) for f in ROW_FRAC]
            add_vlines(ax, MODES, fmax=fmax, label=True, row_y=row_y)
        else:
            for k in range(1, int(fmax / 1.667) + 1):
                fv = k * 1.667
                if fv <= fmax:
                    ax.axvline(fv, color="#AAAAAA", lw=0.7, ls=":", alpha=0.55)
        ax.xaxis.set_major_locator(
            plt.MultipleLocator(1.0 if fmax <= 10 else 100))

    fig_psd.tight_layout()
    psd_out = os.path.join(OUTPUT_DIR, "Fig_TrackA_WDD_PSD.png")
    fig_psd.savefig(psd_out, dpi=DPI, bbox_inches="tight")
    plt.close(fig_psd)
    print(f"Saved → {psd_out}")

    # ── 3. Decimate to 64 Hz ──────────────────────────────────────────────
    print(f"\nDecimating {FS}→{FS_DEC} Hz (factor {DEC_FACTOR}) …")
    d1 = decimate(ch1, DEC_FACTOR, ftype="iir", zero_phase=True)
    d2 = decimate(ch2, DEC_FACTOR, ftype="iir", zero_phase=True)
    data_dec = np.column_stack([d1, d2])
    print(f"Decimated: {len(d1)} samples @ {FS_DEC} Hz")

    # ── 4. pyOMA2 FDD + SSI-cov ───────────────────────────────────────────
    print("\nRunning FDD + SSI-cov …")
    ss  = SingleSetup(data_dec, fs=FS_DEC)
    fdd = FDD(run_params=FDDRunParams(nxseg=NXSEG_FDD, method_SD="cor", pov=0.5))
    ssi = SSI(run_params=SSIRunParams(
        br=BR, method="cov", ordmin=ORDMIN, ordmax=ORDMAX, step=STEP,
        sc=SSI_SC, hc=SSI_HC))
    ss.add_algorithms(fdd, ssi)
    ss.run_by_name("FDD")
    ss.run_by_name("SSI")

    freq  = np.array(fdd.result.freq)
    S_val = np.array(fdd.result.S_val)
    sv1, sv2 = S_val[0, 0, :], S_val[1, 1, :]

    # ── 5. OMA figure: FDD + stabilisation diagram ─────────────────────────
    fig_oma, (ax_sv, ax_stab) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.5], "hspace": 0.04})
    fig_oma.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.07)
    fig_oma.suptitle(
        f"Limnologen Hus 6 — OMA  |  {os.path.basename(fpath)}  |  "
        f"decimated {FS}→{FS_DEC} Hz\n"
        "Top: FDD singular values    "
        f"Bottom: SSI-cov stabilisation (br={BR}, ord {ORDMIN}–{ORDMAX}, step {STEP})",
        fontsize=10.5, weight="bold")

    fmask = freq <= FMAX_PLOT
    ax_sv.semilogy(freq[fmask], sv1[fmask], "#1f77b4", lw=1.5, label="SV₁")
    ax_sv.semilogy(freq[fmask], sv2[fmask], "#d62728", lw=1.0, alpha=0.7, label="SV₂")
    ax_sv.set_ylabel("Singular value")
    ax_sv.legend(fontsize=9, loc="upper right")
    ax_sv.grid(True, which="both", alpha=0.20)
    ax_sv.tick_params(labelbottom=False)
    ax_sv.axvspan(1.5, 3.0, color="#FFFFCC", alpha=0.25, zorder=0)
    add_vlines(ax_sv, MODES, fmax=FMAX_PLOT, label=False)

    stab_plot(Fn=ssi.result.Fn_poles, Lab=ssi.result.Lab,
              step=STEP, ordmax=ORDMAX, ordmin=ORDMIN,
              freqlim=(0, FMAX_PLOT), hide_poles=False,
              fig=fig_oma, ax=ax_stab, color_scheme="classic")
    ax_stab.set_xlim(0, FMAX_PLOT)
    ax_stab.set_title("")
    ax_stab.set_xlabel("Frequency (Hz)")
    ax_stab.set_ylabel("Model order")
    ax_stab.grid(True, which="major", alpha=0.20)
    ax_stab.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax_stab.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    ax_stab.axvspan(1.5, 3.0, color="#FFFFCC", alpha=0.15, zorder=0)

    ymax = ax_stab.get_ylim()[1]
    row_y_stab = [ymax * f for f in [0.93, 0.83, 0.73, 0.63]]
    for (fv, col, lbl, ls, alpha, _), ry in zip(MODES, row_y_stab):
        ax_sv.axvline(fv, color=col, lw=1.0, ls=ls, alpha=alpha)
        ax_stab.axvline(fv, color=col, lw=1.0, ls=ls, alpha=alpha)
        ax_stab.text(fv + 0.07, ry, lbl, color=col, fontsize=8.5,
                     va="center", ha="left",
                     bbox=dict(boxstyle="round,pad=0.18", fc="white",
                               ec=col, alpha=0.88, linewidth=0.7))

    oma_out = os.path.join(OUTPUT_DIR, "Fig_TrackA_WDD_OMA.png")
    fig_oma.savefig(oma_out, dpi=DPI, bbox_inches="tight")
    plt.close(fig_oma)
    print(f"Saved → {oma_out}")

    # ── 6. Stable-pole summary ────────────────────────────────────────────
    Fn_flat = np.where(ssi.result.Lab == 1,
                       ssi.result.Fn_poles, np.nan).flatten()
    Fn_flat = Fn_flat[(~np.isnan(Fn_flat)) & (Fn_flat > 1.0) & (Fn_flat < 5.0)]
    print("\nStable poles in 1–5 Hz:")
    vals, edges = np.histogram(Fn_flat, bins=np.arange(1.0, 5.05, 0.05))
    for v, e in zip(vals, edges):
        if v >= 3:
            print(f"  {e:.2f}–{e+0.05:.2f} Hz : {v:2d} occurrences")


if __name__ == "__main__":
    main()
