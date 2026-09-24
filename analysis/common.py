"""Shared plumbing for the scripts in analysis/.

Every script here needs the same four things: the package and build script on
sys.path, a filtered read of the fronts store, a projected map with the
project's furniture on it, and a consistent look. Keeping them in one place
means the figures are visibly one family.

Run scripts from the repo root as `python analysis/<name>.py`. Everything they
generate goes to outputs/, which is gitignored.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pads

import build as B  # noqa: E402
from codsus import fronts_xml as fx  # noqa: E402
from codsus import grid as g  # noqa: E402

POINTS = str(REPO / "data/parquet/points")
CENTERS = str(REPO / "data/parquet/centers")
ARGS = SimpleNamespace(anchors=True, ne_dir=str(REPO / "data/ne"), no_basemap=False)
NO_ANCHORS = SimpleNamespace(anchors=False, ne_dir=str(REPO / "data/ne"), no_basemap=False)

# Total analyses in the fronts store, used as the denominator everywhere a
# frequency per analysis is quoted.
N_ANALYSES = 46743

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 11,
})


def read_fronts(ftypes, columns=("valid_time", "ftype", "stage", "lat_e2", "lon_e2")):
    """Polyline rows for the given feature types, straight out of Parquet."""
    dset = pads.dataset(POINTS, format="parquet", partitioning="hive")
    if isinstance(ftypes, str):
        ftypes = [ftypes]
    filt = pads.field("ftype").isin(list(ftypes))
    return dset.to_table(columns=list(columns), filter=filt).to_pandas()


def n_analyses_by(series):
    """Distinct analysis times per group -- the correct denominator."""
    return series.nunique()


def basemap(extent=None, cell_km=50.0, figsize=(11, 8.5), anchors=True):
    """A projected axes with the project's standard furniture drawn on it."""
    grid = g.Grid(cell_km=cell_km)
    full = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
    extent = extent or full
    fig, ax = plt.subplots(figsize=figsize)
    fwd = B._draw_map_furniture(ax, grid, ARGS if anchors else NO_ANCHORS, extent)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    return fig, ax, grid, fwd, extent


def box_extent(lon_min, lon_max, lat_min, lat_max, cell_km=50.0, pad_km=200):
    """Projected extent enclosing a lon/lat box, with a little margin."""
    grid = g.Grid(cell_km=cell_km)
    fwd = grid.transformer()
    lons = np.concatenate([
        np.linspace(lon_min, lon_max, 60), np.linspace(lon_min, lon_max, 60),
        np.full(60, lon_min), np.full(60, lon_max)])
    lats = np.concatenate([
        np.full(60, lat_min), np.full(60, lat_max),
        np.linspace(lat_min, lat_max, 60), np.linspace(lat_min, lat_max, 60)])
    x, y = fwd.transform(lons, lats)
    p = pad_km * 1000
    return (x.min() - p, x.max() + p, y.min() - p, y.max() + p)


def save(fig, name):
    out = OUT / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {name}")
    return str(out)
