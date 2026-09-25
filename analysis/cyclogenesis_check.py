"""Does each genesis zone behave like real cyclogenesis?

Real genesis starts shallow and deepens. A mature low that is merely picked up
-- entering the analysed area, or redrawn after being dropped for a few maps --
starts deep and does not. So for each zone: central pressure at the first
analysis, change over the next 24 h, and whether any low was analysed within
UP_KM upstream (west) on the maps 3-12 h before, which a first-drawn low
should not have.
Reads outputs/cyclogenesis_events.parquet and outputs/cyclogenesis.npz.
"""
import numpy as np
import pandas as pd
from common import OUT, REPO, g

UP_KM = 1200.0
ev = pd.read_parquet(OUT / "cyclogenesis_events.parquet")
z = np.load(OUT / "cyclogenesis.npz")
tr = pd.read_parquet(REPO / "data/tracks/tracks.parquet",
                     columns=["valid_time", "lat", "lon", "pressure_hpa", "track_id"])
tr["t"] = tr.valid_time.dt.tz_localize(None)

# pressure at genesis and 24 h later
first = tr.sort_values("t").groupby("track_id").first()
tr_i = tr.set_index(["track_id", "t"]).pressure_hpa
ev["p0"] = ev.track_id.map(first.pressure_hpa)
t24 = ev.genesis_time.dt.tz_localize(None) + pd.Timedelta(hours=24)
ev["dp24"] = [tr_i.get((k, t), np.nan) for k, t in zip(ev.track_id, t24)] - ev.p0
ev["hour"] = ev.genesis_time.dt.hour

# upstream low on the preceding maps
by_t = {t: s for t, s in tr.groupby("t")}
up = []
for tid, t0, la, lo in zip(ev.track_id, ev.genesis_time.dt.tz_localize(None), ev.genesis_lat, ev.genesis_lon):
    hit = False
    for h in (3, 6, 9, 12):
        s = by_t.get(t0 - pd.Timedelta(hours=h))
        if s is None:
            continue
        s = s[(s.track_id != tid) & (s.lon < lo)]          # upstream only
        if len(s):
            _, _, d = g.GEOD.inv(np.full(len(s), lo), np.full(len(s), la), s.lon.to_numpy(), s.lat.to_numpy())
            if (d / 1000 <= UP_KM).any():
                hit = True
                break
    up.append(hit)
ev["upstream_low"] = up

NAMES = {}
lab = z["labels"]
for k in sorted(ev.zone.unique()):
    s = ev[ev.zone == k]
    NAMES[k] = f"{s.genesis_lat.median():.1f}N {abs(s.genesis_lon.median()):.1f}W" if k else "outside zones"
print(f"{'zone':<16s}{'n':>5s}{'p at genesis':>14s}{'24h change':>12s}{'deepens':>9s}"
      f"{'upstream low':>14s}{'synoptic hr':>13s}")
for k, s in ev.groupby("zone"):
    print(f"{NAMES[k]:<16s}{len(s):5d}{s.p0.median():14.0f}{s.dp24.median():+12.1f}"
          f"{100 * (s.dp24 < 0).mean():8.0f}%{100 * s.upstream_low.mean():13.0f}%"
          f"{100 * s.hour.isin([0, 6, 12, 18]).mean():12.0f}%")
ev.to_parquet(OUT / "cyclogenesis_events_checked.parquet")
