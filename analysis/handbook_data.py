"""Data for the second pass of handbook figures.

Computes, once, everything the figure script needs so that styling can be
iterated without touching the archive again:
  * monthly synoptic-hour frequency for the four front types,
  * where lows and highs are analysed, by season,
  * a central-pressure plausibility field (1st percentile of lows, 99th of highs),
  * dryline latitude crossings,
  * squall line crossing counts,
  * four real cyclone cases chosen by a fixed rule, not by eye.
"""
import pickle
import sys


import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from common import OUT, CENTERS, POINTS, REPO, fx, g, read_fronts

SYN = [0, 6, 12, 18]
GRID50 = g.Grid(cell_km=50.0)
GRID100 = g.Grid(cell_km=100.0)
GRID200 = g.Grid(cell_km=200.0)
out = {}

# ------------------------------------------------------ monthly fronts
vt = pads.dataset(POINTS, format="parquet", partitioning="hive").to_table(
    columns=["valid_time"]).to_pandas().valid_time.drop_duplicates()
vt = pd.to_datetime(vt, utc=True)
vt = vt[vt.dt.hour.isin(SYN)]
den_m = vt.dt.month.value_counts().reindex(range(1, 13)).to_numpy()
out["den_month"] = den_m
for ft in ["COLD", "WARM", "STNRY", "OCFNT"]:
    raw = read_fronts(ft)
    t = pd.to_datetime(raw.valid_time, utc=True)
    keep = t.dt.hour.isin(SYN).to_numpy()
    raw, month = raw[keep].reset_index(drop=True), t.dt.month.to_numpy()[keep]
    cube = np.zeros((12,) + GRID50.shape)
    for m in range(1, 13):
        cube[m - 1] = g.frequency_grid(fx.explode(raw[month == m]), GRID50) / den_m[m - 1]
    out[f"month_{ft}"] = cube.astype(np.float32)
    print(f"monthly {ft} done", flush=True)

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
for s, months in {"DJF": [12, 1, 2], "JJA": [6, 7, 8]}.items():
    n = int(an.month.isin(months).sum())
    out[f"den_c_{s}"] = n
    for k, df in (("L", lows), ("H", highs)):
        sub = df[df.month.isin(months)]
        out[f"centres_{k}_{s}"] = g.point_grid(sub, GRID100) / n
print("centre analyses used:", len(an))

def percentile_grid(df, q, grid, min_n=40):
    row, col = grid.to_cells(df.lon.to_numpy(), df.lat.to_numpy())
    ok = row >= 0
    tab = pd.DataFrame({"cell": row[ok] * grid.nx + col[ok],
                        "p": df.pressure_hpa.to_numpy()[ok]})
    agg = tab.groupby("cell").p.agg(["size", lambda s: np.percentile(s, q)])
    agg.columns = ["n", "q"]
    agg = agg[agg.n >= min_n]
    field = np.full(grid.ny * grid.nx, np.nan)
    field[agg.index.to_numpy()] = agg.q.to_numpy()
    return field.reshape(grid.shape)

out["low_p01"] = percentile_grid(lows, 1, GRID200)
out["high_p99"] = percentile_grid(highs, 99, GRID200)
out["low_quantiles"] = np.percentile(lows.pressure_hpa, [0.1, 1, 5, 50, 95, 99])
out["high_quantiles"] = np.percentile(highs.pressure_hpa, [1, 5, 50, 95, 99, 99.9])
print("low q:", out["low_quantiles"], "\nhigh q:", out["high_quantiles"])

# ------------------------------------------------------------- dryline
d = read_fronts("DRYLN")
dt = pd.to_datetime(d.valid_time, utc=True)
rows = []
for cut in (31.0, 33.0, 35.0, 37.0):
    for la, lo, mo, hr in zip(d.lat_e2, d.lon_e2, dt.dt.month, dt.dt.hour):
        lat = np.asarray(la, float) / 100; lon = np.asarray(lo, float) / 100
        if lat.size < 2:
            continue
        a, b = lat[:-1], lat[1:]
        hit = ((a <= cut) & (b > cut)) | ((b <= cut) & (a > cut))
        if hit.any():
            f = (cut - a[hit]) / (b[hit] - a[hit])
            rows.append((cut, float(np.median(lon[:-1][hit] + f * (lon[1:][hit] - lon[:-1][hit]))), mo, hr))
out["dryline"] = pd.DataFrame(rows, columns=["lat", "lon", "month", "hour"])

# --------------------------------------------------------- squall lines
sq = read_fronts("SQLN")
out["sqln_grid"] = g.frequency_grid(fx.explode(sq), GRID100)
st = pd.to_datetime(sq.valid_time, utc=True)
out["sqln_month"] = st.dt.month.value_counts().reindex(range(1, 13), fill_value=0).to_numpy()
out["sqln_hour"] = st.dt.hour.value_counts().reindex([0, 3, 6, 9, 12, 15, 18, 21], fill_value=0).to_numpy()
out["sqln_n"] = len(sq)

# ---------------------------------------------------------- real cases
# Fixed rule: 12Z, Oct-Apr, central US, no other low within 900 km, central
# pressure <= 996 hPa. Take the deepest, then the deepest in a different year
# and month, and so on, so the four cases are not one storm or one season.
L = lows[(lows.t.dt.hour == 12) & lows.month.isin([10, 11, 12, 1, 2, 3, 4])].copy()
fwd = GRID50.transformer()
L["x"], L["y"] = fwd.transform(L.lon.to_numpy(), L.lat.to_numpy())
nn = np.full(len(L), np.inf)
allL = lows[lows.t.dt.hour == 12].copy()
allL["x"], allL["y"] = fwd.transform(allL.lon.to_numpy(), allL.lat.to_numpy())
grp = allL.groupby("valid_time")
for i, (vtime, x, y) in enumerate(zip(L.valid_time, L.x, L.y)):
    o = grp.get_group(vtime)
    dd = np.hypot(o.x - x, o.y - y) / 1000.0
    dd = dd[dd > 1.0]
    nn[i] = dd.min() if len(dd) else np.inf
L["nn"] = nn
cand = L[L.lat.between(36, 48) & L.lon.between(-100, -82) & (L.nn > 900)
         & (L.pressure_hpa <= 996)].sort_values("pressure_hpa")
picked, years, months = [], set(), set()
for _, r in cand.iterrows():
    if r.t.year in years or r.month in months:
        continue
    picked.append(r); years.add(r.t.year); months.add(r.month)
    if len(picked) == 4:
        break
cases = []
dset = pads.dataset(POINTS, format="parquet", partitioning="hive")
for r in picked:
    fr = dset.to_table(columns=["valid_time", "ftype", "stage", "lat_e2", "lon_e2"],
                       filter=pads.field("year") == int(r.t.year)).to_pandas()
    fr = fr[pd.to_datetime(fr.valid_time, utc=True) == r.t]
    cen = c[c.t == r.t][["kind", "lat", "lon", "pressure_hpa"]]
    cases.append({"time": r.t, "lat": r.lat, "lon": r.lon, "p": r.pressure_hpa,
                  "fronts": fr, "centres": cen})
    print(f"case {r.t:%Y-%m-%d %HZ}  {r.pressure_hpa:.0f} hPa  {r.lat:.1f}N {abs(r.lon):.1f}W  "
          f"{len(fr)} boundaries on the chart")
out["cases"] = cases

with open(OUT / "handbook_data2.pkl", "wb") as fh:
    pickle.dump(out, fh)
print("saved handbook_data2.pkl")
