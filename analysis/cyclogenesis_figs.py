"""Cyclogenesis figure in the handbook style.

Two panels on one colour scale: every cool-season low that passes the Fritzen
et al. (2021) filters, then only those that deepen at least 4 hPa after they
are first drawn. Zones are outlined and numbered on the second. Reads
outputs/cyclogenesis.npz, cyclogenesis_deep4.npz and the matching events
files (run cyclogenesis.py with and without --deepen 4 first).
"""
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from common import ARGS, NO_ANCHORS, OUT, REPO, B, box_extent, g, plt

INK, MUTED, RULE = "#15202B", "#566374", "#C9D2DC"
LOW, BOMB = "#C8281F", "#1F4FB4"
mpl.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": RULE, "savefig.facecolor": "white"})
GRAT = re.compile(r"^\d+[NW]$")
CELL = 150.0
GRID = g.Grid(cell_km=CELL)
FULL = (GRID.x_min, GRID.x_max, GRID.y_min, GRID.y_max)
EXT = box_extent(-126, -60, 25, 58, cell_km=CELL, pad_km=60)
FWD = GRID.transformer()


def ramp(hexcol):
    base = LinearSegmentedColormap.from_list("b", ["#FFFFFF", hexcol, "#0B0B0B"])
    return LinearSegmentedColormap.from_list("r", [base(x) for x in np.linspace(0.03, 0.78, 16)])


def mapax(ax, lw=0.8):
    n0, c0 = len(ax.lines), len(ax.collections)
    B._draw_map_furniture(ax, g.Grid(cell_km=50.0), ARGS, EXT)
    for ln in ax.lines[n0:]:
        ln.set_linewidth(ln.get_linewidth() * lw)
    for co in ax.collections[c0:]:
        co.set_linewidth(np.asarray(co.get_linewidth()) * lw)
    for t in list(ax.texts):
        if GRAT.match(t.get_text().strip()):
            t.remove()
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(RULE)


all_ = np.load(OUT / "cyclogenesis.npz")
dev = np.load(OUT / "cyclogenesis_deep4.npz")
ev_all = pd.read_parquet(OUT / "cyclogenesis_events.parquet")
ev_dev = pd.read_parquet(OUT / "cyclogenesis_events_deep4.parquet")
vmax = float(np.percentile(all_["smooth"][all_["smooth"] > 0], 99.5))

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"wspace": 0.03})
panels = [(axes[0], all_, ev_all, "Every low passing the paper's filters", False),
          (axes[1], dev, ev_dev, "Only lows that deepen 4 hPa or more after first drawn", True)]
for ax, z, ev, title, zones in panels:
    sm = z["smooth"]
    im = ax.imshow(np.where(sm > 0, sm, np.nan), origin="lower", extent=FULL, cmap=ramp(LOW),
                   vmin=0, vmax=vmax, interpolation="bilinear", zorder=1)
    mapax(ax)
    b = ev[ev.bomb]
    bx, by = FWD.transform(b.genesis_lon.to_numpy(), b.genesis_lat.to_numpy())
    ax.scatter(bx, by, s=9, color=BOMB, edgecolors="white", linewidths=0.4, zorder=6)
    if zones:
        lab = z["labels"]
        xs = np.linspace(FULL[0], FULL[1], GRID.nx); ys = np.linspace(FULL[2], FULL[3], GRID.ny)
        for k in range(1, int(lab.max()) + 1):
            ax.contour(xs, ys, (lab == k).astype(float), levels=[0.5], colors=INK, linewidths=1.2, zorder=5)
            rr, cc = np.argwhere(lab == k).mean(axis=0)
            ax.text(xs[int(round(cc))], ys[int(round(rr))], str(k), fontsize=8.5, fontweight="bold",
                    ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec=INK, lw=0.8))
    ax.set_title(f"{title}  ({len(ev) / int(z['n_seasons']):.0f} per season)",
                 loc="left", fontsize=10, fontweight="semibold", color=INK, pad=4)
cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.045, pad=0.03, aspect=50)
cb.set_label("new lows per 150 km cell per cool season (3 x 3 mean)", fontsize=8, color=MUTED)
cb.outline.set_edgecolor(RULE); cb.ax.tick_params(labelsize=7.5, length=2)
axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=BOMB, markersize=4.5,
                               label="bomb cyclone, where first drawn")],
               loc="lower left", fontsize=8, framealpha=0.96, edgecolor=RULE)
for dest in (OUT, REPO / "private" / "handbook" / "img"):
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "hb_cyclogenesis.png", dpi=150, bbox_inches="tight", pad_inches=0.06)
print("wrote hb_cyclogenesis.png")
