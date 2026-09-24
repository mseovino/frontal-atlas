import sys
import numpy as np, pandas as pd, pyarrow.dataset as pads
from common import CENTERS, g

YEARS = range(2009, 2019)
GRID = g.Grid(cell_km=50.0); FWD = GRID.transformer()
cds = pads.dataset(CENTERS, format="parquet", partitioning="hive")
lows = cds.to_table(columns=["valid_time","res","kind","lat","lon","pressure_hpa"],
                    filter=(pads.field("kind") == "L")).to_pandas()
lows = lows[lows.res == "HR"].copy()
lows["atime"] = pd.to_datetime(lows.valid_time, utc=True)
lows = lows[lows.atime.dt.year.isin(YEARS)]
lows = lows[lows.lat.between(25, 60) & lows.lon.between(-150, -40)]
lows = lows.dropna(subset=["pressure_hpa"])
lows = lows[lows.pressure_hpa.between(940, 1030)].reset_index(drop=True)
x, y = FWD.transform(lows.lon.to_numpy(), lows.lat.to_numpy())
lows = lows.assign(x=x, y=y)

nn = np.full(len(lows), np.inf)
for _, idx in lows.groupby("atime", sort=False).indices.items():
    if idx.size == 1:
        continue
    xa, ya = lows.x.to_numpy()[idx], lows.y.to_numpy()[idx]
    d = np.hypot(xa[:, None] - xa[None, :], ya[:, None] - ya[None, :]) / 1000.0
    np.fill_diagonal(d, np.inf)
    nn[idx] = d.min(axis=1)

fin = nn[np.isfinite(nn)]
print(f"{len(lows):,} lows;  lows per analysis: mean "
      f"{len(lows)/lows.atime.nunique():.1f}")
print(f"nearest other low in the same analysis: median {np.median(fin):.0f} km")
for cut in (300, 500, 750, 1000, 1500):
    print(f"  another low within {cut:5d} km: {100*np.mean(nn <= cut):5.1f}%")
