"""Which ingredient, if any, gives the damming proxy a cold-season cycle?

Rather than tuning thresholds until the answer looks right, sweep each
ingredient separately and print the seasonal ratio it produces. Cold air
damming is strongly cold-season, so a definition that is actually selecting
damming should show a cold-to-warm ratio well above 1. If none of them do,
that is the finding.
"""
import sys


import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from common import CENTERS, MONTHS, read_fronts

YEARS = (2007, 2018)
CREST_LAT0, CREST_LON0, CREST_SLOPE = 34.5, -84.0, 7.0 / 6.0
COLD_M, WARM_M = [10, 11, 12, 1, 2, 3, 4], [5, 6, 7, 8, 9]


def crest_lon(lat):
    return CREST_LON0 + (lat - CREST_LAT0) * CREST_SLOPE


raw = read_fronts(["STNRY", "TROF"])
t = pd.to_datetime(raw.valid_time, utc=True)
raw = raw.assign(atime=t, month=t.dt.month, year=t.dt.year)
raw = raw[(raw.year >= YEARS[0]) & (raw.year <= YEARS[1])].reset_index(drop=True)


def geometry(lat_e2, lon_e2, east_max, lat_hi, min_km):
    lat = np.asarray(lat_e2, dtype=float) / 100.0
    lon = np.asarray(lon_e2, dtype=float) / 100.0
    off = lon - crest_lon(lat)
    m = (lat >= 33.0) & (lat <= lat_hi) & (off >= -0.5) & (off <= east_max)
    if m.sum() < 2:
        return False, np.nan
    la, lo = lat[m], lon[m]
    dy = (la[-1] - la[0]) * 111.32
    dx = (lo[-1] - lo[0]) * 111.32 * np.cos(np.radians(la.mean()))
    if np.hypot(dx, dy) < min_km:
        return False, np.nan
    return True, np.degrees(np.arctan2(dx, dy)) % 180.0


# ------------------------------------------------------------------ centers
cds = pads.dataset(CENTERS, format="parquet", partitioning="hive")
hi = cds.to_table(columns=["valid_time", "res", "kind", "lat", "lon", "pressure_hpa"],
                  filter=(pads.field("kind") == "H")).to_pandas()
hi = hi[hi.res == "HR"].copy()
hi["atime"] = pd.to_datetime(hi.valid_time, utc=True)
hi = hi[(hi.atime.dt.year >= YEARS[0]) & (hi.atime.dt.year <= YEARS[1])]

shared = sorted(set(raw.atime) & set(hi.atime))
shared_s = pd.Series(shared)
denom = shared_s.dt.month.value_counts().reindex(range(1, 13), fill_value=0)
shared_set = set(shared)
print(f"{len(shared):,} shared analyses\n")


def ratio(times, label, show=False):
    times = [x for x in times if x in shared_set]
    if not times:
        print(f"{label:52s}  nothing"); return
    n = pd.Series(pd.DatetimeIndex(times)).dt.month.value_counts().reindex(
        range(1, 13), fill_value=0)
    rate = 100 * n / denom
    r = rate[COLD_M].mean() / rate[WARM_M].mean()
    print(f"{label:52s}  {100*len(times)/len(shared):5.1f}% of analyses   "
          f"cold/warm {r:5.2f}")
    if show:
        print("      " + "  ".join(f"{MONTHS[m-1]} {rate[m]:4.1f}" for m in range(1, 13)))


# ------------------------------------------- ingredient 1: the high alone
for box, name in [((38, 52, -80, -58), "wide NE box"),
                  ((41, 50, -78, -63), "tight New England / Quebec"),
                  ((42, 50, -76, -64), "tighter, inland")]:
    la0, la1, lo0, lo1 = box
    sub = hi[hi.lat.between(la0, la1) & hi.lon.between(lo0, lo1)]
    ratio(sub.atime.unique(), f"HIGH only: {name}")
    for p in [1018, 1022, 1026]:
        s2 = sub[sub.pressure_hpa >= p]
        ratio(s2.atime.unique(), f"HIGH only: {name}, >= {p} hPa")
print()

# --------------------------------- ingredient 2: the boundary geometry alone
for east_max, lat_hi, min_km, gname in [
    (4.5, 39.5, 150, "corridor to coast, 33-39.5N, 150 km"),
    (2.5, 38.5, 150, "inner Piedmont only, 33-38.5N, 150 km"),
    (2.5, 38.5, 300, "inner Piedmont only, 33-38.5N, 300 km"),
]:
    ok, bear = zip(*[geometry(a, b, east_max, lat_hi, min_km)
                     for a, b in zip(raw.lat_e2, raw.lon_e2)])
    ok = np.array(ok); bear = np.array(bear)
    delta = np.abs((bear - 43.0 + 90) % 180 - 90)
    for tol in [35.0, 25.0]:
        sel = raw[ok & (delta <= tol)]
        for types in [("STNRY", "TROF"), ("STNRY",)]:
            s = sel[sel.ftype.isin(types)]
            tag = "STNRY+TROF" if len(types) == 2 else "STNRY"
            ratio(s.atime.unique(), f"GEOM: {gname}, +/-{tol:g}deg, {tag}")
    globals()[f"cache_{east_max}_{lat_hi}_{min_km}"] = (ok, delta)
print()

# ------------------------------------------------- both, tightest of each
ok, delta = globals()["cache_2.5_38.5_300"]
geom_sel = raw[ok & (delta <= 25.0) & (raw.ftype == "STNRY")]
for box, name in [((41, 50, -78, -63), "tight NE"), ((42, 50, -76, -64), "inland NE")]:
    la0, la1, lo0, lo1 = box
    for p in [0, 1018, 1022]:
        sub = hi[hi.lat.between(la0, la1) & hi.lon.between(lo0, lo1)
                 & (hi.pressure_hpa >= p)]
        both = set(geom_sel.atime) & set(sub.atime)
        ratio(sorted(both), f"BOTH: inner STNRY 300km +/-25 + {name} >= {p}",
              show=True)
