"""Is the trough increase a step (a tool or policy change) or a ramp (drift)?

Synoptic hours only, features counted by centroid in a Lower 48 box, per
synoptic analysis, by year and by half-year so a mid-record step would show.
"""
import sys
import numpy as np, pandas as pd
from common import read_fronts

SYN = [0, 6, 12, 18]
BOX = (25, 50, -125, -67)
raw = read_fronts(["TROF", "STNRY", "COLD"])
t = pd.to_datetime(raw.valid_time, utc=True)
keep = t.dt.hour.isin(SYN).to_numpy()
raw, t = raw[keep].reset_index(drop=True), t[keep].reset_index(drop=True)
mlat = np.array([np.mean(a) for a in raw.lat_e2]) / 100
mlon = np.array([np.mean(b) for b in raw.lon_e2]) / 100
inb = (mlat >= BOX[0]) & (mlat <= BOX[1]) & (mlon >= BOX[2]) & (mlon <= BOX[3])
raw, t = raw[inb].reset_index(drop=True), t[inb].reset_index(drop=True)
raw["year"] = t.dt.year.to_numpy()
raw["half"] = (t.dt.year + (t.dt.month > 6) * 0.5).to_numpy()
an = pd.Series(t.unique())
den_y = an.dt.year.value_counts()
den_h = (an.dt.year + (an.dt.month > 6) * 0.5).value_counts()

tab = raw.groupby(["year", "ftype"], observed=True).size().unstack().div(den_y, axis=0)
tab = tab.loc[2007:2022]
print("features per synoptic analysis, Lower 48 box")
print(tab.round(2).to_string())
y = tab.index.to_numpy(float)
for ft in tab.columns:
    b = np.polyfit(y, tab[ft].to_numpy(), 1)[0]
    print(f"  {ft:6s} trend {b:+.3f}/yr  ({100*b*15/tab[ft].iloc[:3].mean():+.0f}% over 15 yr)")

h = raw[raw.ftype == "TROF"].groupby("half").size().div(den_h).sort_index()
h = h[(h.index >= 2007) & (h.index < 2023)]
d = h.diff()
print("\nlargest half-year jumps in trough rate:")
for k in d.abs().sort_values(ascending=False).index[:5]:
    print(f"  {k:.1f}  {h[k]:.2f}  (change {d[k]:+.2f})")
print(f"typical half-year change (median abs): {d.abs().median():.2f}")

# ------------------------- more boundaries, or the same ones in more pieces?
from common import g
raw["km"] = [g.polyline_length_km(np.asarray(a) / 100, np.asarray(b) / 100)
             for a, b in zip(raw.lat_e2, raw.lon_e2)]
L = raw.groupby(["year", "ftype"], observed=True).km.sum().unstack().div(den_y, axis=0).loc[2007:2022]
M = raw.groupby(["year", "ftype"], observed=True).km.median().unstack().loc[2007:2022]
print("\ntotal drawn length per synoptic analysis (1000 km)")
print((L / 1000).round(1).to_string())
print("\nmedian length of one feature (km)")
print(M.round(0).to_string())
for ft in L.columns:
    a, b = L[ft].loc[2007:2010].mean(), L[ft].loc[2019:2022].mean()
    c, d = tab[ft].loc[2007:2010].mean(), tab[ft].loc[2019:2022].mean()
    print(f"  {ft:6s} 2007-10 -> 2019-22:  count {100*(d/c-1):+.0f}%   total length {100*(b/a-1):+.0f}%")
