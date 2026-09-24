"""Quantify statements in the WPC Surface Analysis Manual against the archive.

1. "Dry lines are usually not located east of the 94th meridian."
2. Per-type synoptic/off-hour ratio in a box that is WPC's own area at every
   hour, away from the ocean fronts OPC supplied at synoptic times and the
   area south of 31N that NHC/TAFB drew. If the ratio survives here, the
   effect is WPC-internal; if it vanishes, it came from the merge.
"""
import sys
import numpy as np, pandas as pd
from common import MONTHS, read_fronts

# ------------------------------------------------------------ 1. drylines
d = read_fronts("DRYLN")
t = pd.to_datetime(d.valid_time, utc=True)
d = d.assign(month=t.dt.month.to_numpy())
lon = [np.asarray(b) / 100.0 for b in d.lon_e2]
east_any = np.array([(x > -94.0).any() for x in lon])
east_mean = np.array([x.mean() > -94.0 for x in lon])
allv = np.concatenate(lon)
print("DRYLINES vs the 94W rule")
print(f"  features: {len(d):,}")
print(f"  vertices east of 94W: {100*np.mean(allv > -94):.1f}%")
print(f"  features with any part east of 94W: {100*east_any.mean():.1f}%")
print(f"  features centred east of 94W: {100*east_mean.mean():.1f}%")
print(f"  easternmost vertex ever: {abs(allv.max()):.1f}W")
for pct in (99, 99.9):
    print(f"  {pct}th percentile easternmost-per-feature: "
          f"{abs(np.percentile([x.max() for x in lon], pct)):.1f}W")
by = pd.Series(east_any).groupby(d.month).mean().reindex(range(1, 13))
print("  any part east of 94W, by month: " +
      "  ".join(f"{MONTHS[m-1]} {100*v:.0f}%" for m, v in by.items() if np.isfinite(v)))

# --------------------------------------------- 2. hour ratio, WPC-only box
TYPES = ["COLD", "WARM", "STNRY", "OCFNT", "TROF", "DRYLN", "SQLN"]
HRS, SYN, OFF = [0,3,6,9,12,15,18,21], [0,6,12,18], [3,9,15,21]
BOXES = {"NA box 25-50N 125-67W": (25, 50, -125, -67),
         "interior 33-48N 115-80W": (33, 48, -115, -80)}
raw = read_fronts(TYPES)
t = pd.to_datetime(raw.valid_time, utc=True)
raw = raw.assign(hour=t.dt.hour.to_numpy(), year=t.dt.year.to_numpy())
den = pd.Series(t.unique()).dt.hour.value_counts().reindex(HRS)
mlat = np.array([np.mean(a) for a in raw.lat_e2]) / 100.0
mlon = np.array([np.mean(b) for b in raw.lon_e2]) / 100.0
print("\nSYNOPTIC / OFF-HOUR RATIO, features centred in box")
print(f"  {'type':<7s}" + "".join(f"{k:>26s}" for k in BOXES))
for ft in TYPES:
    cells = []
    for la0, la1, lo0, lo1 in BOXES.values():
        m = (raw.ftype == ft).to_numpy() & (mlat >= la0) & (mlat <= la1) \
            & (mlon >= lo0) & (mlon <= lo1)
        per = raw[m].groupby("hour").size().reindex(HRS, fill_value=0) / den
        cells.append(per[SYN].mean() / per[OFF].mean() if per[OFF].mean() else np.nan)
    print(f"  {ft:<7s}" + "".join(f"{c:26.2f}" for c in cells))
