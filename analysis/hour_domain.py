"""Is the synoptic-hour feature surplus a domain effect rather than a detail effect?

Claim under test: 00/06/12/18Z analyses carry more features because they cover
more ground, not because they are drawn in more detail. If so, restricting to a
fixed North American box should equalise the per-analysis feature counts.
"""
import sys
import numpy as np, pandas as pd
from common import read_fronts

BOXES = {
    "everything drawn":      None,
    "NA land box  25-50N, 125-67W":  (25, 50, -125, -67),
    "wider NA     20-60N, 140-60W":  (20, 60, -140, -60),
}
HOURS = [0, 3, 6, 9, 12, 15, 18, 21]
SYN, OFF = [0, 6, 12, 18], [3, 9, 15, 21]

raw = read_fronts(["COLD", "WARM", "STNRY", "OCFNT", "TROF"])
t = pd.to_datetime(raw.valid_time, utc=True)
raw = raw.assign(hour=t.dt.hour)

# Per-feature extremes, computed once.
raw["lat_lo"] = [min(a)/100.0 for a in raw.lat_e2]
raw["lat_hi"] = [max(a)/100.0 for a in raw.lat_e2]
raw["lon_lo"] = [min(b)/100.0 for b in raw.lon_e2]
raw["lon_hi"] = [max(b)/100.0 for b in raw.lon_e2]

n_an = raw.groupby("hour").valid_time.nunique().reindex(HOURS)
print("analyses per hour:", dict(n_an))

for name, box in BOXES.items():
    if box is None:
        sub = raw
    else:
        la0, la1, lo0, lo1 = box
        # Feature overlaps the box if its bounding range intersects it.
        sub = raw[(raw.lat_hi >= la0) & (raw.lat_lo <= la1)
                  & (raw.lon_hi >= lo0) & (raw.lon_lo <= lo1)]
    per = sub.groupby("hour").size().reindex(HOURS, fill_value=0) / n_an
    ratio = per[SYN].mean() / per[OFF].mean()
    print(f"\n{name}   features per analysis")
    print("   " + "  ".join(f"{h:02d}Z {per[h]:5.1f}" for h in HOURS))
    print(f"   synoptic/off-hour ratio: {ratio:.2f}")

# How far do the analyses actually reach, by hour?
print("\nmedian westernmost and easternmost vertex per analysis, by hour")
ext = raw.groupby(["valid_time", "hour"]).agg(
    w=("lon_lo", "min"), e=("lon_hi", "max"),
    s=("lat_lo", "min"), n=("lat_hi", "max")).reset_index()
for h in HOURS:
    s = ext[ext.hour == h]
    print(f"  {h:02d}Z  west {s.w.median():7.1f}  east {s.e.median():7.1f}  "
          f"south {s.s.median():5.1f}  north {s.n.median():5.1f}")
