"""Two maps the project has not made yet.

1. Where the trough increase happened, as a ratio centred honestly on 1.
2. Which boundary type is drawn most often in each cell -- a categorical map
   rather than another density field.
"""
import sys
from pathlib import Path
from types import SimpleNamespace


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import OUT
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

import build as B
from codsus import grid as g

T = OUT
ARGS = SimpleNamespace(anchors=False, ne_dir="data/ne", no_basemap=False)


def base_map(cell_km, extent):
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=140)
    grid = g.Grid(cell_km=cell_km)
    B._draw_map_furniture(ax, grid, ARGS, extent)
    return fig, ax


# ---------------------------------------------------------------- trend map
a = np.load(f"{T}/trof_early.npz")
b = np.load(f"{T}/trof_late.npz")
ok = (a["counts"] >= 50) & (b["counts"] >= 50)
ratio = np.where(ok, b["freq"] / np.where(a["freq"] > 0, a["freq"], np.nan), np.nan)
ext = tuple(a["extent"])

fig, ax = plt.subplots(figsize=(11, 8.5), dpi=140)
grid = g.Grid(cell_km=float(a["cell_km"]))
mesh = ax.imshow(
    ratio, origin="lower", extent=ext, cmap="RdBu_r", interpolation="nearest",
    # Centred on 1.0 so "unchanged" is white. Without this the diverging
    # colormap puts its neutral colour at the midpoint of the data range and
    # every unchanged cell reads as a decrease.
    norm=TwoSlopeNorm(vcenter=1.0, vmin=0.5, vmax=3.0),
)
B._draw_map_furniture(ax, grid, ARGS, ext)
ax.set_title(
    "Troughs drawn per analysis: 2019-22 vs 2007-10\n"
    "ratio of frequency, cells with 50+ crossings in both eras"
)
fig.colorbar(mesh, ax=ax, shrink=0.7, label="late / early  (1.0 = no change)",
             extend="both")
fig.savefig("trof_trend.png", bbox_inches="tight")
print("wrote trof_trend.png")

# ------------------------------------------------------- dominant type map
TYPES = ["COLD", "WARM", "STNRY", "OCFNT", "TROF", "DRYLN"]
COLORS = ["#2166ac", "#d6604d", "#7b3294", "#8c6d31", "#f1a340", "#000000"]
LABELS = ["Cold", "Warm", "Stationary", "Occluded", "Trough", "Dryline"]

stack, counts = [], []
for t in TYPES:
    z = np.load(f"{T}/dom_{t}.npz")
    stack.append(z["freq"])
    counts.append(z["counts"])
stack = np.stack(stack)
counts = np.stack(counts)

total = counts.sum(axis=0)
dom = np.where(total >= 100, stack.argmax(axis=0), np.nan)

fig, ax = plt.subplots(figsize=(11, 8.5), dpi=140)
z0 = np.load(f"{T}/dom_COLD.npz")
ext = tuple(z0["extent"])
grid = g.Grid(cell_km=float(z0["cell_km"]))
cmap = ListedColormap(COLORS)
mesh = ax.imshow(
    dom, origin="lower", extent=ext, cmap=cmap, interpolation="nearest",
    norm=BoundaryNorm(np.arange(-0.5, len(TYPES)), cmap.N),
)
B._draw_map_furniture(ax, grid, ARGS, ext)
ax.set_title(
    "Which boundary does WPC draw most often here?\n"
    "most frequent analysed type per cell, 2006-2022, "
    f"{grid.cell_km:g} km cells"
)
cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, ticks=range(len(TYPES)))
cbar.ax.set_yticklabels(LABELS)
fig.savefig("dominant_type.png", bbox_inches="tight")
print("wrote dominant_type.png")

frac = {LABELS[i]: float(np.nanmean(dom == i)) for i in range(len(TYPES))}
print("share of cells where each type dominates:")
for k, v in sorted(frac.items(), key=lambda kv: -kv[1]):
    print(f"  {k:12s} {100*v:5.1f}%")
