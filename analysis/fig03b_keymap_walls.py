"""
fig03b_keymap_walls.py
by Andre R. Barbosa, April - October 2026
Panel (b) of the building figure: colour-coded wall key-map of the standard
floor (Cantisani 2026 thesis Fig. 48; drawing (c) Martinsons Byggsystem AB)
with the paper's X/Y/N arrows and the two sensor locations, plus schematic
horizontal sections (to scale) of three wall build-ups redrawn from the
layer lists of thesis Sec. 6 / Fig. 47:
  X walls : YV-01-02 exterior, 85 mm CLT
  Y walls : IV-01-01 interior partition, 95 mm CLT
            LSV-06-02 apartment-separating double-leaf timber frame (no CLT)
All new text is Arial italic (Liberation Sans fallback) at >= 8 pt print size.
Output: figures/fig03b_keymap_walls.png
"""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "analysis" / "input" / "fig01"
OUT = ROOT / "figures"
CM = 1 / 2.54
BLUE, RED = "#1f5fbf", "#d62728"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.style": "italic", "mathtext.default": "it"})


def arrow(ax, x0, y0, dx, dy, c, lw=1.6, ms=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x0 + dx, y0 + dy), arrowstyle="-|>",
                                 mutation_scale=ms, color=c, lw=lw, shrinkA=0, shrinkB=0, zorder=5))


# ------------------------------------------------------------ key-map plan
km = Image.open(IN / "walls" / "keymap_standard_floor.png").convert("RGB")
px = km.load(); W0, H0 = km.size
for x in range(0, 160):                      # remove the thesis legend (redrawn below in Arial)
    for y in range(715, H0):
        px[x, y] = (255, 255, 255)
km = km.crop((6, 0, W0, 1304)).rotate(90, expand=True, fillcolor="white")   # long axis horizontal, as in the paper
P = np.array(km); ph, pw = P.shape[:2]               # 1326 x 746

H = 6.6                      # printed panel height, cm
WP = H * pw / ph             # ~11.7 cm
WS = 4.6                     # wall-section strip, cm
W = WP + WS
fig = plt.figure(figsize=(W * CM, H * CM))
ax = fig.add_axes([0, 0, WP / W, 1])
ax.imshow(P, interpolation="lanczos"); ax.set_xlim(0, pw); ax.set_ylim(ph, 0); ax.set_axis_off()

# global axes + north, bottom-right free area (where the thesis legend was)
ox, oy, L = 1205, 580, 80
arrow(ax, ox, oy, L, 0, BLUE, lw=2.0, ms=12); arrow(ax, ox, oy, 0, -L, RED, lw=2.0, ms=12)
ax.text(ox + L + 6, oy, "X", color=BLUE, fontsize=10, va="center", ha="left")
ax.text(ox, oy - L - 6, "Y", color=RED, fontsize=10, va="bottom", ha="center")
nx, ny, NL, th = 1140, 580, 70, math.radians(30)
arrow(ax, nx, ny, -NL * math.sin(th), -NL * math.cos(th), "k", lw=1.6, ms=11)
ax.text(nx - (NL + 14) * math.sin(th), ny - (NL + 14) * math.cos(th), "N", fontsize=10, ha="center", va="center")

# sensor locations at the two lift-shaft cores
for (x, y, lab, tx, ty) in [(356, 384, "Loc 1", 374, 420), (835, 338, "Loc 2", 853, 374)]:
    ax.plot(x, y, "o", ms=4.5, color="k", zorder=6)
    arrow(ax, x, y, 50, 0, BLUE); arrow(ax, x, y, 0, -50, RED)
    ax.text(tx, ty, lab, fontsize=8, ha="left", va="center",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=7)

# wall-type legend (colours as in the thesis key-map), top-left free area
legend = [("YV-01-02 ext., CLT 85", (131, 146, 188)),
          ("YV-01-03 ext., CLT 95", (70, 120, 80)),
          ("YV-02-03 ext., CLT 85+glulam", (191, 142, 101)),
          ("IV-01-01 int., CLT 95", (207, 167, 178)),
          ("IV-01-02 int., CLT 85", (110, 170, 175)),
          ("IV-01-03 attic, CLT 73", (127, 180, 148)),
          ("LSV-06-02 separator, TF", (240, 205, 110))]
ax.add_patch(Rectangle((465, 596), 830, 128, fc="white", ec="none", alpha=0.92, zorder=5))
for i, (lab, col) in enumerate(legend):
    cx, cy = (476, 930)[i // 4], 614 + (i % 4) * 29
    ax.add_patch(Rectangle((cx, cy - 9), 26, 18, fc=np.array(col) / 255, ec="none", zorder=6))
    ax.text(cx + 34, cy, lab, fontsize=8, va="center", ha="left", zorder=7)

# ------------------------------------------------------------ schematic wall sections (to scale)
MAT = {"clt": ("#c8925a", "CLT"), "ins": ("#f3e6a8", "insulation / stud cavity"),
       "gyp": ("#c9c9c9", "plasterboard"), "osb": ("#e0c088", "OSB"), "air": ("#ffffff", "air gap"),
       "fac": ("#f3e6a8", None), "paint": ("#404040", None)}
walls = [("YV-01-02", "X walls, exterior", BLUE,
          [("paint", 5), ("fac", 180), ("clt", 85), ("ins", 45), ("gyp", 15)], ("out", "in")),
         ("IV-01-01", "Y walls, interior partition", RED,
          [("gyp", 13), ("clt", 95), ("gyp", 13)], None),
         ("LSV-06-02", "Y walls, separator", RED,
          [("gyp", 30), ("ins", 28), ("osb", 8), ("ins", 120), ("air", 20), ("ins", 120), ("osb", 8), ("ins", 28), ("gyp", 30)], None)]
axs = fig.add_axes([WP / W + 0.012, 0.0, 1 - WP / W - 0.02, 1.0])
axs.set_xlim(0, 400); axs.set_ylim(0, 100); axs.set_axis_off()
scale = 392 / 392            # 1 mm = 1 unit; 392 units ~ 4.1 cm at this strip width
ybar, hbar, gap = 84, 9, 27
for k, (code, title, col, layers, sides) in enumerate(walls):
    yb = ybar - k * gap
    axs.text(0, yb + hbar + 2.5, code, fontsize=8, color=col, va="bottom", ha="left")
    axs.text(400, yb + hbar + 2.5, title, fontsize=8, va="bottom", ha="right")
    x = 0
    for mat, t in layers:
        fc, _ = MAT[mat]
        axs.add_patch(Rectangle((x, yb), t, hbar, fc=fc, ec="k", lw=0.4))
        if mat == "clt":                                   # 3-layer CLT: lamella lines
            for f in (1 / 3, 2 / 3):
                axs.plot([x + t * f, x + t * f], [yb, yb + hbar], color="#7a4f24", lw=0.5)
            axs.text(x + t / 2, yb + hbar / 2, f"CLT {t}", fontsize=8, ha="center", va="center", color="k")
        elif mat in ("fac", "ins") and t >= 100:
            axs.text(x + t / 2, yb + hbar / 2, f"{t}", fontsize=8, ha="center", va="center")
        elif mat in ("ins",) and 40 <= t < 100:
            axs.plot([x + t / 2, x + t / 2], [yb + 1, yb + hbar - 1], color="#8c6a3c", lw=1.2)   # stud
        x += t
    # thickness string below the bar (major layers only)
    axs.text(x, yb - 1.5, f"{x} mm", fontsize=8, ha="right", va="top")
    if sides:
        axs.text(0, yb - 1.5, "outside", fontsize=8, ha="left", va="top")
# material legend (explicit positions)
for mat, lab, xx, ly0 in [("clt", "CLT", 0, 16), ("ins", "insulation / studs", 130, 16),
                          ("gyp", "plasterboard", 0, 9), ("osb", "OSB", 190, 9), ("air", "air gap", 0, 2)]:
    axs.add_patch(Rectangle((xx, ly0 - 2.5), 14, 5, fc=MAT[mat][0], ec="k", lw=0.4))
    axs.text(xx + 18, ly0, lab, fontsize=8, va="center", ha="left")
fig.add_artist(plt.Line2D([WP / W + 0.004, WP / W + 0.004], [0.03, 0.97], color="0.6", lw=0.6))

fig.savefig(OUT / "fig03b_keymap_walls.png", dpi=600, facecolor="white")
print("saved", OUT / "fig03b_keymap_walls.png", "size cm", round(W, 2), "x", H)
