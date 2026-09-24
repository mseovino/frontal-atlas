"""Is the dissipating-to-forming imbalance a detection asymmetry?

Proposed explanation: a dissipating front is already drawn, so the analyst is
tracking a feature they can see weakening. A forming front is not drawn yet, so
marking it means recognising incipient frontogenesis where nothing exists on
the chart. The second task is harder.

That predicts the ratio should fall where observations are dense, because
formation becomes easier to spot. Over the data-sparse ocean it should be
worse. Tested here against a CONUS box and two open-ocean boxes.
"""
import sys
import numpy as np, pandas as pd
from common import g, read_fronts

TYPES = ["COLD", "WARM", "STNRY", "OCFNT"]
BOXES = {
    "CONUS (dense obs)":        (25, 50, -125, -67),
    "N Pacific (sparse)":       (30, 55, -175, -135),
    "N Atlantic (sparse)":      (30, 55, -60, -25),
    "Canada interior (medium)": (50, 65, -120, -75),
}

raw = read_fronts(TYPES)
raw["stage"] = raw.stage.astype(object)

def ratio(sub, label):
    d = int((sub.stage == "DISS").sum()); f = int((sub.stage == "FORM").sum())
    tot = len(sub)
    if f < 30:
        print(f"  {label:28s} too few"); return
    print(f"  {label:28s}{tot:9,}{d:9,}{f:8,}{d/f:9.2f}"
          f"{100*(d+f)/tot:9.1f}%")

print("by feature type")
print(f"  {'':28s}{'total':>9s}{'DISS':>9s}{'FORM':>8s}{'ratio':>9s}{'tagged':>10s}")
for ft in TYPES:
    ratio(raw[raw.ftype == ft], ft)
ratio(raw, "ALL")

# Mean position of each feature, for the regional split.
raw["mlat"] = [np.mean(a)/100.0 for a in raw.lat_e2]
raw["mlon"] = [np.mean(b)/100.0 for b in raw.lon_e2]

print("\nby region -- the test of the detection explanation")
print(f"  {'':28s}{'total':>9s}{'DISS':>9s}{'FORM':>8s}{'ratio':>9s}{'tagged':>10s}")
for name, (la0, la1, lo0, lo1) in BOXES.items():
    sub = raw[raw.mlat.between(la0, la1) & raw.mlon.between(lo0, lo1)]
    ratio(sub, name)

print("\ncold fronts only, same regions (controls for type mix)")
print(f"  {'':28s}{'total':>9s}{'DISS':>9s}{'FORM':>8s}{'ratio':>9s}{'tagged':>10s}")
cold = raw[raw.ftype == "COLD"]
for name, (la0, la1, lo0, lo1) in BOXES.items():
    sub = cold[cold.mlat.between(la0, la1) & cold.mlon.between(lo0, lo1)]
    ratio(sub, name)

# Second prediction: a nascent front should be drawn short, and so should a
# decaying one, so both tagged states should be shorter than untagged.
print("\nmedian drawn length by stage (km), all types")
lens = np.array([g.polyline_length_km(np.asarray(a)/100.0, np.asarray(b)/100.0)
                 for a, b in zip(raw.lat_e2, raw.lon_e2)])
raw["km"] = lens
for st in ["FORM", "DISS", None]:
    s = raw[raw.stage.isna()] if st is None else raw[raw.stage == st]
    print(f"  {str(st or 'untagged'):10s} n={len(s):8,}  median {s.km.median():6.0f} km"
          f"  p25 {s.km.quantile(.25):5.0f}  p75 {s.km.quantile(.75):6.0f}")
