"""Why does the passage count come out four times too high?

Hypothesis: a hard distance threshold flickers. Analysts redraw a front every
cycle and its plotted position wobbles, so one system crosses the threshold
several times and is counted as several passages. Diagnose by looking at the
gap distribution between present-analyses, then sweep radius and gap tolerance.
"""
import sys
import numpy as np, pandas as pd
from common import read_fronts

CLAT, CLON = 39.10, -94.60
YEARS = (2007, 2022); N = 16

raw = read_fronts(["COLD"])
t = pd.to_datetime(raw.valid_time, utc=True)
raw = raw.assign(atime=t, year=t.dt.year)
raw = raw[(raw.year >= YEARS[0]) & (raw.year <= YEARS[1])].reset_index(drop=True)

kx = 111.32 * np.cos(np.radians(CLAT))

def mindist(lat_e2, lon_e2):
    lat = np.asarray(lat_e2, dtype=float)/100.0
    lon = np.asarray(lon_e2, dtype=float)/100.0
    x = (lon - CLON) * kx; y = (lat - CLAT) * 111.32
    if x.size == 1:
        return float(np.hypot(x[0], y[0]))
    ax, ay = x[:-1], y[:-1]; dx, dy = x[1:]-ax, y[1:]-ay
    seg = dx*dx + dy*dy
    with np.errstate(invalid="ignore", divide="ignore"):
        tt = np.where(seg > 0, -(ax*dx + ay*dy)/np.where(seg > 0, seg, 1), 0.0)
    tt = np.clip(tt, 0, 1)
    return float(np.hypot(ax + tt*dx, ay + tt*dy).min())

raw["d"] = [mindist(a, b) for a, b in zip(raw.lat_e2, raw.lon_e2)]
per_an = raw.groupby("atime").d.min()
all_times = np.array(sorted(raw.atime.unique()))
print(f"{len(all_times):,} analyses with any cold front drawn anywhere")
print(f"nearest cold front to the point: median {per_an.median():.0f} km\n")

def count(radius, gap_h):
    ts = np.array(sorted(per_an[per_an <= radius].index), dtype="datetime64[ns]")
    if ts.size == 0:
        return 0, 0.0, 0.0
    gaps = np.diff(ts).astype("timedelta64[m]").astype(float)/60.0
    runs = np.split(np.arange(ts.size), np.flatnonzero(gaps > gap_h + 0.5) + 1)
    dur = np.array([(ts[r[-1]]-ts[r[0]])/np.timedelta64(1,"h") + 3.0 for r in runs])
    return len(runs), len(runs)/N, float(np.median(dur))

print("gap between consecutive present-analyses, radius 150 km")
ts = np.array(sorted(per_an[per_an <= 150].index), dtype="datetime64[ns]")
gaps = np.diff(ts).astype("timedelta64[m]").astype(float)/60.0
for lo, hi in [(3,3),(6,6),(9,12),(15,24),(27,48),(51,1e9)]:
    m = (gaps >= lo) & (gaps <= hi)
    print(f"  {lo:3.0f}-{hi if hi<1e9 else 999:3.0f} h: {100*m.mean():5.1f}%")

print(f"\n{'radius':>8s}" + "".join(f"{f'gap {g}h':>12s}" for g in [3,6,12,24]))
for radius in (50, 100, 150, 200, 300):
    cells = []
    for gap in (3, 6, 12, 24):
        n, per_yr, med = count(radius, gap)
        cells.append(f"{per_yr:6.1f}/yr")
    print(f"{radius:6.0f}km" + "".join(f"{c:>12s}" for c in cells))

print(f"\n{'radius':>8s}" + "".join(f"{f'gap {g}h':>12s}" for g in [3,6,12,24]))
for radius in (50, 100, 150, 200, 300):
    cells = []
    for gap in (3, 6, 12, 24):
        n, per_yr, med = count(radius, gap)
        cells.append(f"{med:5.0f} h")
    print(f"{radius:6.0f}km" + "".join(f"{c:>12s}" for c in cells))
