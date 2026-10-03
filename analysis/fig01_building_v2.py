"""
fig01_building_v2.py
by Andre R. Barbosa, April - October 2026
Panels for the Limnologen Hus 6 building figure (paper Fig. 2), assembled in
LaTeX as subfigures (a) photo, (b) plan, (c)-(f) the four AVT set-ups.

 * (b) plan rotated 90 deg ccw; global X/Y axes and the two sensor locations
   (Loc 1, Loc 2) with local X/Y arrows, following Cantisani (2026) thesis
   Fig. 58 (X along the long axis of the plan, Y across it; Loc 1 at the
   west-wing stair/lift core, Loc 2 at the east-wing lift shaft).
 * (c)-(f) original tiny "Floor #n Loc #m" labels removed and replaced by
   8-pt Arial italic floor labels (F3...F7) and Loc 1 / Loc 2 column labels;
   the unreadable Y/X legend is removed (explained in the caption).
All new text is Arial italic (Liberation Sans fallback) at >= 8 pt print size.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "analysis" / "input" / "fig01"
OUT = ROOT / "figures"
CM = 1 / 2.54
BLUE, RED = "#1f5fbf", "#d62728"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.style": "italic", "mathtext.default": "it"})


def canvas(img, width_cm):
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(width_cm * CM, width_cm * CM * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="lanczos")
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_axis_off()
    return fig, ax


def arrow(ax, x0, y0, dx, dy, c, lw=1.6, ms=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x0 + dx, y0 + dy), arrowstyle="-|>",
                                 mutation_scale=ms, color=c, lw=lw, shrinkA=0, shrinkB=0, zorder=5))


# ---------------------------------------------------------------- (a) photo
ph = Image.open(IN / "hus6_photo.jpeg"); w, h = ph.size
ph.crop((int(0.12 * w), int(0.08 * h), w, int(0.90 * h))).save(OUT / "fig03a_photo.png")   # trim left neighbour + foreground grass

# ---------------------------------------------------------------- (b) plan
pl = Image.open(IN / "hus6_floorplan.png").convert("RGB"); w, h = pl.size
pl = pl.crop((int(.03 * w), int(.02 * h), int(.97 * w), int(.98 * h))).rotate(90, expand=True, fillcolor="white")
P = np.array(pl)
fig, ax = canvas(P, 10.2)           # printed width in the paper (height 6.8 cm)
# global axes (bottom-right, free area of the drawing)
ox, oy, L = 885, 655, 105
arrow(ax, ox, oy, L, 0, BLUE, lw=2.0, ms=12); arrow(ax, ox, oy, 0, -L, RED, lw=2.0, ms=12)
ax.text(ox + L + 6, oy, "X", color=BLUE, fontsize=10, va="center", ha="left")
ax.text(ox, oy - L - 6, "Y", color=RED, fontsize=10, va="bottom", ha="center")
# north arrow, as in Cantisani (2026) Fig. 58 (north ~30 deg anticlockwise from +Y)
import math
nx, ny, NL, th = 805, 655, 95, math.radians(30)
arrow(ax, nx, ny, -NL * math.sin(th), -NL * math.cos(th), "k", lw=1.6, ms=11)
ax.text(nx - (NL + 14) * math.sin(th), ny - (NL + 14) * math.cos(th), "N",
        fontsize=10, ha="center", va="center")
# sensor locations with local X/Y arrows
for (x, y, lab, tx, ty, ha) in [(300, 336, "Loc 1", 318, 372, "left"),
                                (665, 302, "Loc 2", 683, 338, "left")]:
    ax.plot(x, y, "o", ms=4.5, color="k", zorder=6)
    arrow(ax, x, y, 55, 0, BLUE); arrow(ax, x, y, 0, -55, RED)
    ax.text(tx, ty, lab, fontsize=8, ha=ha, va="center",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=7)
fig.savefig(OUT / "fig03b_plan.png", dpi=600, facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- (c)-(f) set-ups
S = np.array(Image.open(IN / "hus6_sensors.png").convert("RGB"))
rows = [(3, 396), (441, 837)]; cols = [(3, 653), (735, 1386)]
crops = [S[r0:r1, c0:c1].copy() for (r0, r1) in rows for (c0, c1) in cols]
# three panels: layout 3 (first phase) has the same sensor positions as layout 2
crops = [crops[0], crops[1], crops[3]]
BAND = 60
crops_orig = {k: c.copy() for k, c in zip("cde", crops)}
for k, A in zip("cde", crops):
    a = A.astype(int)
    red = (a[..., 0] > 170) & (a[..., 1] < 90) & (a[..., 2] < 90)
    lab, n = ndimage.label(red)
    cms = ndimage.center_of_mass(red, lab, range(1, n + 1))
    szs = ndimage.sum(red, lab, range(1, n + 1))
    dots = [(int(cy), int(cx)) for (cy, cx), s in zip(cms, szs) if s > 15 and cy < 350]
    # remove legend (coloured pixels, bottom-left)
    pass
    box = A[355:, :110]; bi = box.astype(int)
    box[((bi.max(2) - bi.min(2)) > 12) | (bi.sum(2) > 600)] = 255     # coloured Y/X legend + its halo
    dark = A.astype(int).sum(2) < 745          # includes anti-aliased grey text
    x1 = sorted({x for _, x in dots})[0]
    for (dy, dx) in dots:
        if dx < 300:   # Loc 1 text is left of the dot
            xs, xe = 40, dx - 9
        else:          # Loc 2 text is right of the dot
            xs, xe = dx + 40, 630
        ys, ye = dy - 11, dy + 9
        if dx >= 300:
            xe = 612                           # stop short of the right-hand wall
        keep_cols = (dark[ys - 6:ye + 1, xs:xe].mean(0) > 0.9)   # vertical drawing lines
        saved = A[ys:ye, xs:xe][:, keep_cols].copy()
        A[ys:ye, xs:xe] = 255
        A[ys:ye, xs:xe][:, keep_cols] = saved
        # restore horizontal floor lines (rows that are mostly dark in the original)
        orig = crops_orig[k][ys:ye, xs:xe]
        keep_rows = (orig.astype(int).sum(2) < 600).mean(1) > 0.6
        keep_rows[:16] = False                  # text occupies rows up to ~dy+4
        A[ys:ye, xs:xe][keep_rows] = orig[keep_rows]
    A = np.vstack([A, np.full((BAND, A.shape[1], 3), 255, np.uint8)])
    fig, ax = canvas(A, 5.2)
    # floor number from vertical position: ground floor line ~ y 343, 43 px / floor
    FLOORS = {"c": [7], "d": [7, 5, 3], "e": [7, 6, 4]}   # Cantisani thesis Fig. 57
    left = sorted((dy, dx) for dy, dx in dots if dx < 300)
    for (dy, dx), fl in zip(left, FLOORS[k]):
        if True:
            ax.text(dx - 13, dy - 1, f"F{fl}", fontsize=8, ha="right", va="center")
    xl = [dx for _, dx in dots if dx < 300][0]; xr = [dx for _, dx in dots if dx >= 300][0]
    yb = A.shape[0] - BAND / 2 - 4
    ax.text(xl, yb, "Loc 1", fontsize=8, ha="center", va="center")
    ax.text(xr, yb, "Loc 2", fontsize=8, ha="center", va="center")
    fig.savefig(OUT / f"fig03{k}_setup.png", dpi=600, facecolor="white")
    plt.close(fig)
    print(k, sorted(dots))
print("done")
