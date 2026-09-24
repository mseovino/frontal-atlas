"""Frontal atlas for the training handbook.

Built from the four synoptic analyses only (00, 06, 12, 18Z). Two reasons.
Classical fronts are drawn more often at those hours and troughs/stationary
fronts less often, so pooling all eight hours biases each type differently.
And the off-hour charts stop near 169W, 31W and 18N, so pooling undercounts
anything beyond those edges by about half. Synoptic-only sampling is uniform
in both time and space.
"""
import sys


import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from common import OUT, POINTS, REPO, fx, g, read_fronts

SYN = [0, 6, 12, 18]
TYPES = ["COLD", "WARM", "STNRY", "OCFNT", "TROF"]
SEASONS = {"ANN": list(range(1, 13)), "DJF": [12, 1, 2], "JJA": [6, 7, 8],
           "MAM": [3, 4, 5], "SON": [9, 10, 11]}
GRID = g.Grid(cell_km=50.0)

# Denominators: distinct synoptic analyses per season.
vt = pads.dataset(POINTS, format="parquet", partitioning="hive").to_table(
    columns=["valid_time"]).to_pandas().valid_time.drop_duplicates()
vt = pd.to_datetime(vt, utc=True)
vt = vt[vt.dt.hour.isin(SYN)]
den = {s: int(vt.dt.month.isin(m).sum()) for s, m in SEASONS.items()}
print("synoptic analyses:", den)

out = {}
for ft in TYPES:
    raw = read_fronts(ft)
    t = pd.to_datetime(raw.valid_time, utc=True)
    raw = raw[t.dt.hour.isin(SYN).to_numpy()].reset_index(drop=True)
    month = pd.to_datetime(raw.valid_time, utc=True).dt.month.to_numpy()
    for s, months in SEASONS.items():
        sub = raw[np.isin(month, months)]
        out[f"{ft}_{s}"] = g.frequency_grid(fx.explode(sub), GRID) / den[s]
    print(f"  {ft}: {len(raw):,} synoptic-hour features", flush=True)

np.savez_compressed(OUT / "handbook_atlas.npz", **out,
                    **{f"den_{s}": v for s, v in den.items()})
print("saved handbook_atlas.npz")
