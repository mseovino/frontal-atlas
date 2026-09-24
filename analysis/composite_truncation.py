"""How many composite lows have their surroundings clipped by the chart edge?

The off-hour analyses stop near 169W, 31W, 18N and 78N. A low close enough to
one of those edges has part of its neighbourhood outside the analysed area, so
the composite records "no front drawn there" when the truth is "not analysed
there". That is a false zero, and it is not spread evenly -- it concentrates in
exactly the regions where mature occluded cyclones live.
"""
import sys
import numpy as np, pandas as pd, pyarrow.dataset as pads
from common import CENTERS

YEARS = range(2009, 2019)
SYN = [0, 6, 12, 18]
# Median analysed extent per hour, measured from the archive.
OFF_BOUNDS = dict(w=-169.0, e=-31.0, s=18.0, n=78.5)
REACH_KM = 900.0

cds = pads.dataset(CENTERS, format="parquet", partitioning="hive")
lows = cds.to_table(columns=["valid_time","res","kind","lat","lon","pressure_hpa"],
                    filter=(pads.field("kind") == "L")).to_pandas()
lows = lows[lows.res == "HR"].copy()
lows["atime"] = pd.to_datetime(lows.valid_time, utc=True)
lows = lows[lows.atime.dt.year.isin(YEARS)]
lows = lows[lows.lat.between(25, 60) & lows.lon.between(-150, -40)]
lows = lows.dropna(subset=["pressure_hpa"])
lows = lows[lows.pressure_hpa.between(940, 1030)].reset_index(drop=True)
lows["hour"] = lows.atime.dt.hour
lows["syn"] = lows.hour.isin(SYN)

# Reach in degrees, at the low's own latitude.
dlat = REACH_KM / 111.32
dlon = REACH_KM / (111.32 * np.cos(np.radians(lows.lat.to_numpy())))
clipped = ((lows.lon - dlon < OFF_BOUNDS["w"]) | (lows.lon + dlon > OFF_BOUNDS["e"])
           | (lows.lat - dlat < OFF_BOUNDS["s"]) | (lows.lat + dlat > OFF_BOUNDS["n"]))
lows["clipped_offhour"] = clipped

print(f"{len(lows):,} composite lows, {REACH_KM:g} km reach\n")
print(f"neighbourhood crosses an off-hour chart edge: "
      f"{100*clipped.mean():.1f}% of all lows")
print(f"  of those, {100*(~lows.syn[clipped]).mean():.0f}% are on an off-hour "
      f"analysis, where the clipping is real")
bad = lows[clipped & ~lows.syn]
print(f"lows actually affected (off-hour AND clipped): {len(bad):,} "
      f"({100*len(bad)/len(lows):.1f}% of the composite)")

print("\nwhere the affected lows are, by longitude band")
bands = [(-150,-130,"Gulf of Alaska / NE Pacific"), (-130,-110,"West Coast"),
         (-110,-90,"Plains"), (-90,-70,"East / Appalachians"),
         (-70,-55,"western Atlantic"), (-55,-40,"Labrador / Greenland side")]
for lo, hi, name in bands:
    s = lows[(lows.lon >= lo) & (lows.lon < hi)]
    if len(s) == 0:
        continue
    aff = s.clipped_offhour & ~s.syn
    print(f"  {name:30s} {len(s):7,} lows   {100*aff.mean():5.1f}% affected")

print("\nsame, restricted to deep lows (<= 990 hPa), the mature systems")
deep = lows[lows.pressure_hpa <= 990]
for lo, hi, name in bands:
    s = deep[(deep.lon >= lo) & (deep.lon < hi)]
    if len(s) < 200:
        continue
    aff = s.clipped_offhour & ~s.syn
    print(f"  {name:30s} {len(s):7,} lows   {100*aff.mean():5.1f}% affected")
