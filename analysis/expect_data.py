"""Data for the "what to expect this month" page.

Everything is computed from synoptic-hour analyses (00, 06, 12, 18Z) so each
layer is sampled evenly in space and type. Front layers are the fraction of
synoptic maps with that feature drawn through the 50 km cell; centre layers
are the fraction with a low or high analysed in the 100 km cell (high-
resolution bulletins, 2009-2018). Pressure ranges are seasonal, because a
single month leaves too few centres per cell to estimate a 1st percentile.

Output: outputs/expect_data.npz, read by expect_page.py.
"""
import pickle

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from scipy.ndimage import gaussian_filter
from common import CENTERS, OUT, POINTS, fx, g, read_fronts

SYN = [0, 6, 12, 18]
SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
G50, G100, G200 = g.Grid(cell_km=50.0), g.Grid(cell_km=100.0), g.Grid(cell_km=200.0)
out = {}

# ------------------------------------------------ fronts, troughs, drylines
with open(OUT / "handbook_data2.pkl", "rb") as fh:
    H = pickle.load(fh)
den_m = H["den_month"]
for ft in ["COLD", "WARM", "STNRY", "OCFNT"]:
    out[f"m_{ft}"] = H[f"month_{ft}"].astype(np.float32)

for ft in ["TROF", "DRYLN"]:
    raw = read_fronts(ft)
    t = pd.to_datetime(raw.valid_time, utc=True)
    keep = t.dt.hour.isin(SYN).to_numpy()
    raw, month = raw[keep].reset_index(drop=True), t.dt.month.to_numpy()[keep]
    cube = np.zeros((12,) + G50.shape, dtype=np.float32)
    for m in range(1, 13):
        cube[m - 1] = g.frequency_grid(fx.explode(raw[month == m]), G50) / den_m[m - 1]
    out[f"m_{ft}"] = cube
    print(f"{ft} monthly done", flush=True)

# how often a dryline or squall line appears at all, by month
for ft in ["DRYLN", "SQLN"]:
    raw = read_fronts(ft, columns=("valid_time",))
    t = pd.to_datetime(raw.valid_time, utc=True)
    t = t[t.dt.hour.isin(SYN)].drop_duplicates()
    out[f"present_{ft}"] = (t.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
                            .to_numpy() / den_m).astype(np.float32)

# ------------------------------------------------------------- centres
c = pads.dataset(CENTERS, format="parquet", partitioning="hive").to_table(
    columns=["valid_time", "res", "kind", "lat", "lon", "pressure_hpa"]).to_pandas()
c = c[c.res == "HR"].copy()
c["t"] = pd.to_datetime(c.valid_time, utc=True)
c = c[c.t.dt.hour.isin(SYN)]
c["month"] = c.t.dt.month
lows = c[(c.kind == "L") & c.pressure_hpa.between(940, 1030)]
highs = c[(c.kind == "H") & c.pressure_hpa.between(990, 1070)]
an = c.drop_duplicates("valid_time")
den_c = an.month.value_counts().reindex(range(1, 13)).to_numpy()
for k, df in (("L", lows), ("H", highs)):
    cube = np.zeros((12,) + G100.shape, dtype=np.float32)
    for m in range(1, 13):
        cube[m - 1] = gaussian_filter(g.point_grid(df[df.month == m], G100) / den_c[m - 1], 0.8)
    out[f"m_{k}"] = cube


def pct_grid(df, q, grid, min_n=30):
    row, col = grid.to_cells(df.lon.to_numpy(), df.lat.to_numpy())
    ok = row >= 0
    tab = pd.DataFrame({"cell": row[ok] * grid.nx + col[ok], "p": df.pressure_hpa.to_numpy()[ok]})
    agg = tab.groupby("cell").p.agg(n="size", q=lambda s: np.percentile(s, q))
    agg = agg[agg.n >= min_n]
    f = np.full(grid.ny * grid.nx, np.nan, dtype=np.float32)
    f[agg.index.to_numpy()] = agg.q.to_numpy()
    return f.reshape(grid.shape)


for s, months in SEASONS.items():
    out[f"p_L_{s}"] = pct_grid(lows[lows.month.isin(months)], 1, G200)
    out[f"p_H_{s}"] = pct_grid(highs[highs.month.isin(months)], 99, G200)
    print(f"pressure {s}: lows p1 median {np.nanmedian(out[f'p_L_{s}']):.0f}, "
          f"highs p99 median {np.nanmedian(out[f'p_H_{s}']):.0f}")

# ------------------------------------------------------- dryline position
dl = H["dryline"]
lines = np.full((12, 4, 2), np.nan, dtype=np.float32)      # month, latitude cut, (lat, lon)
for m in range(1, 13):
    for i, cut in enumerate([31.0, 33.0, 35.0, 37.0]):
        s = dl[(dl.lat == cut) & (dl.month == m)].lon
        if len(s) >= 25:
            lines[m - 1, i] = (cut, s.median())
out["dryline_lines"] = lines

np.savez_compressed(OUT / "expect_data.npz", **out)
print("saved expect_data.npz")
