"""Where is one front type most common, absolutely and relatively?

Usage: python analysis/front_share.py [COLD|WARM|STNRY|OCFNT]   (default STNRY)

Original question, for stationary fronts:

The dominant-type map hinted that stationary fronts trace terrain. A heat map
says exactly where. Share is included for the same reason as the warm-front
pair: absolute frequency finds where they are drawn most, share finds where
they win the competition against the other front types.
"""
import pickle
import sys
from types import SimpleNamespace


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import OUT
from pyproj import Transformer

import build as B
from codsus import grid as g

with open(OUT / "dom_counts.pkl", "rb") as fh:
    COUNTS = pickle.load(fh)

CELL = 50
FRONTS = ["COLD", "WARM", "STNRY", "OCFNT"]
ARGS = SimpleNamespace(anchors=True, ne_dir="data/ne", no_basemap=False)

grid = g.Grid(cell_km=CELL)
ext = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
back = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)

TYPE = sys.argv[1] if len(sys.argv) > 1 else "STNRY"
stn = COUNTS[CELL][TYPE].astype(float)
allf = np.stack([COUNTS[CELL][t] for t in FRONTS]).astype(float).sum(axis=0)

N_ANALYSES = 46743
freq = stn / N_ANALYSES
share = np.where(allf >= 200, stn / np.where(allf > 0, allf, np.nan), np.nan)


def where(field, label):
    r, c = np.unravel_index(np.nanargmax(field), field.shape)
    x = ext[0] + (c + 0.5) * CELL * 1000
    y = ext[2] + (r + 0.5) * CELL * 1000
    lon, lat = back.transform(x, y)
    print(f"  {label}: {np.nanmax(field):.4f} at {lat:.1f}N {abs(lon):.1f}W")


def top(field, n=8):
    """Rank the distinct local maxima so the terrain claim can be checked."""
    flat = np.argsort(np.nan_to_num(field, nan=-1), axis=None)[::-1]
    picked = []
    for idx in flat:
        r, c = np.unravel_index(idx, field.shape)
        if any(abs(r - pr) < 8 and abs(c - pc) < 8 for pr, pc in picked):
            continue
        picked.append((r, c))
        x = ext[0] + (c + 0.5) * CELL * 1000
        y = ext[2] + (r + 0.5) * CELL * 1000
        lon, lat = back.transform(x, y)
        print(f"    {field[r, c]:.4f}  {lat:5.1f}N {abs(lon):6.1f}W")
        if len(picked) == n:
            break


def draw(field, out, title, subtitle, cbar, cmap, vmax):
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    m = ax.imshow(np.where(field > 0, field, np.nan), origin="lower", extent=ext,
                  cmap=cmap, interpolation="nearest", vmin=0, vmax=vmax)
    B._draw_map_furniture(ax, grid, ARGS, ext)
    ax.set_title(f"{title}\n{subtitle}")
    fig.colorbar(m, ax=ax, shrink=0.7, label=cbar, extend="max")
    fig.savefig(OUT / out, bbox_inches="tight")
    print(f"wrote {out}")


print("peaks:")
where(freq, "absolute frequency")
where(share, "share of all fronts")

print("\nseparated local maxima, absolute frequency:")
top(freq)

occ = share[np.isfinite(share)]
print(f"\n{TYPE} share: median {np.nanmedian(occ):.3f}, "
      f"p90 {np.nanpercentile(occ, 90):.3f}, max {np.nanmax(occ):.3f}")
print(f"cells where {TYPE} fronts are the plurality (>40%): "
      f"{int(np.nansum(occ > 0.40)):,} of {occ.size:,}")

draw(freq, f"{TYPE.lower()}_freq.png", f"{TYPE} front frequency",
     f"crossings per analysis, 2006-2022, {CELL} km cells",
     "stationary front crossings per analysis", "magma_r",
     float(np.nanpercentile(freq[freq > 0], 99)))
draw(share, f"{TYPE.lower()}_share.png", f"{TYPE} fronts as a share of all analysed fronts",
     f"stationary / (cold + warm + stationary + occluded), {CELL} km cells",
     "stationary front share", "viridis", float(np.nanpercentile(occ, 99.5)))
