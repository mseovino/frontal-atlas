"""Cool-season cyclogenesis zones from analysed lows, after Fritzen et al. (2021).

Fritzen, Lang and Gensini (2021, JAMC 60, 1319-1331) tracked NARR sea-level
pressure minima for October-April 1979-2019, kept cyclones lasting at least
24 h, travelling over 500 km, with a net-to-total path ratio above 0.6, and
defined genesis zones from smoothed equal-area genesis counts. This does the
same with the lows WPC analysts drew (high-resolution bulletins, 2009-2018).

"Genesis" here means the first map a low was drawn on. A track can also start
because a low was skipped for a map or two and the tracker could not bridge
the gap; those are counted below as likely continuations (a track that ended
within CONT_KM and CONT_H of the new start) and reported separately.

Writes outputs/cyclogenesis.npz and outputs/cyclogenesis_*.png.
"""
import numpy as np
import pandas as pd
from scipy import ndimage
from common import OUT, REPO, basemap, box_extent, g, plt

COOL = [10, 11, 12, 1, 2, 3, 4]
MIN_H, MIN_PATH, MIN_RATIO = 24.0, 500.0, 0.6       # the paper's filters
CONT_KM, CONT_H = 400.0, 9.0
# Genesis box kept inside the coverage of the analysed lows. The east edge of
# the bulletin is 30W and the north edge about 82N. In the west the density of
# analysed lows triples between 155W and 150W; whether that is Gulf of Alaska
# cyclogenesis or a change in coverage cannot be told from this archive, so
# the box stops at 140W, the same caution as the paper's 30-gridpoint buffer.
BOX = (25.0, 72.0, -140.0, -45.0)
CELL = 150.0
SMOOTH = 1                                           # 3x3 mean, as in the paper
PEAK_Q, HALF = 90.0, 0.5                             # peaks above this percentile; zone = cells >= HALF x peak

tr = pd.read_parquet(REPO / "data/tracks/tracks.parquet").sort_values(["track_id", "valid_time"])
st = pd.read_parquet(REPO / "data/tracks/track_stats.parquet")
st["gmonth"] = st.genesis_time.dt.month
st["season"] = np.where(st.gmonth >= 10, st.genesis_time.dt.year + 1, st.genesis_time.dt.year)
st["ratio"] = st.net_km / st.path_km.replace(0, np.nan)
cool = st[st.gmonth.isin(COOL) & (st.season.between(2010, 2018))]      # nine full cool seasons
etc = cool[(cool.duration_hours >= MIN_H) & (cool.path_km > MIN_PATH) & (cool.ratio > MIN_RATIO)
           & cool.genesis_lat.between(BOX[0], BOX[1]) & cool.genesis_lon.between(BOX[2], BOX[3])].copy()
n_seasons = etc.season.nunique()
print(f"{len(cool):,} cool-season tracks; {len(etc):,} pass the paper's filters in the genesis box "
      f"({len(etc) / n_seasons:.0f} per season over {n_seasons} seasons)")

# --------------------------------------------------- likely continuations
ends = st[["track_id", "lysis_time", "lysis_lat", "lysis_lon"]].sort_values("lysis_time")
geod = g.GEOD
cont = np.zeros(len(etc), dtype=bool)
lt = ends.lysis_time.dt.tz_localize(None).to_numpy()
for i, (tid, t0, la, lo) in enumerate(zip(etc.track_id, etc.genesis_time, etc.genesis_lat, etc.genesis_lon)):
    t0 = np.datetime64(t0.tz_localize(None))
    a, b = np.searchsorted(lt, t0 - np.timedelta64(int(CONT_H * 60), "m")), np.searchsorted(lt, t0)
    if b <= a:
        continue
    c = ends.iloc[a:b]
    c = c[c.track_id != tid]
    if len(c):
        _, _, d = geod.inv(np.full(len(c), lo), np.full(len(c), la), c.lysis_lon.to_numpy(), c.lysis_lat.to_numpy())
        cont[i] = (d / 1000 <= CONT_KM).any()
etc["continuation"] = cont
print(f"likely continuations of an earlier track: {cont.mean():.0%}")
gen = etc[~etc.continuation].copy()
print(f"{len(gen):,} genesis events kept ({len(gen) / n_seasons:.0f} per season)")

# ----------------------------------------------------------------- bombs
tr = tr[tr.track_id.isin(gen.track_id)]
tr = tr.set_index("valid_time")
bomb = {}
for tid, s in tr.groupby("track_id"):
    p = s.pressure_hpa
    p24 = p.reindex(p.index - pd.Timedelta(hours=24))
    p24.index = p.index
    lat = s.lat.to_numpy()
    b = (p24.to_numpy() - p.to_numpy()) / 24.0 * np.sin(np.radians(60)) / np.sin(np.radians(np.abs(lat)))
    bomb[tid] = np.nanmax(b) if np.isfinite(b).any() else np.nan
gen["bergeron"] = gen.track_id.map(bomb)
gen["bomb"] = gen.bergeron >= 1.0
print(f"bomb cyclones: {gen.bomb.sum():,} ({gen.bomb.sum() / n_seasons:.1f} per season)")

# ------------------------------------------------------- genesis density
grid = g.Grid(cell_km=CELL)
counts = g.point_grid(gen.rename(columns={"genesis_lat": "lat", "genesis_lon": "lon"}), grid).astype(float)
per_season = counts / n_seasons
sm = ndimage.uniform_filter(per_season, size=2 * SMOOTH + 1, mode="constant")
# Zones. The paper thresholded the smoothed field and hand-trimmed the result.
# A single threshold cannot separate neighbouring maxima (Colorado and Alberta
# merge), so instead: take each local maximum above the PEAK_Q percentile,
# strongest first, and grow its zone over connected cells at least HALF of its
# peak value that no stronger zone has already claimed.
thr = np.percentile(sm[sm > 0], PEAK_Q)
peaks = np.argwhere((sm == ndimage.maximum_filter(sm, size=5)) & (sm >= thr))
peaks = peaks[np.argsort(-sm[tuple(peaks.T)])]
lab = np.zeros(sm.shape, dtype=int); nz = 0
for r, c in peaks:
    if lab[r, c]:
        continue
    comp, _ = ndimage.label((sm >= HALF * sm[r, c]) & (lab == 0))
    if comp[r, c] == 0:
        continue
    nz += 1
    lab[comp == comp[r, c]] = nz
lon_c, lat_c = grid.cell_centers()
row, col = grid.to_cells(gen.genesis_lon.to_numpy(), gen.genesis_lat.to_numpy())
gen["zone"] = np.where(row >= 0, lab[np.clip(row, 0, None), np.clip(col, 0, None)], 0)

zones = []
for z in range(1, nz + 1):
    m = lab == z
    w = per_season[m]
    zones.append({"zone": z, "cells": int(m.sum()),
                  "lat": float(np.average(lat_c[m], weights=w + 1e-9)),
                  "lon": float(np.average(lon_c[m], weights=w + 1e-9))})
zones = pd.DataFrame(zones)
zs = gen[gen.zone > 0].groupby("zone").agg(n=("track_id", "size"), bombs=("bomb", "sum"),
                                              min_p=("min_pressure_hpa", "median"))
zones = zones.join(zs, on="zone")
zones["per_season"] = zones.n / n_seasons
zones["per_1e5km2"] = zones.per_season / (zones.cells * CELL ** 2 / 1e5)
zones = zones.sort_values("per_season", ascending=False)
print(f"\n{nz} zones: local maxima above the {PEAK_Q:g}th percentile, each grown to {HALF:g} of its peak "
      f"({thr:.2f} per {CELL:g} km cell per season)")
print(f"{'lat':>6s}{'lon':>8s}{'cells':>7s}{'per season':>12s}{'per 1e5 km2':>13s}{'bombs':>7s}{'median min p':>14s}")
for _, z in zones.iterrows():
    print(f"{z.lat:6.1f}{z.lon:8.1f}{z.cells:7d}{z.per_season:12.1f}{z.per_1e5km2:13.2f}"
          f"{int(z.bombs):7d}{z.min_p:14.0f}")
print(f"\noutside all zones: {100 * (gen.zone == 0).mean():.0f}% of genesis events "
      f"(the paper: 70% 'other')")

np.savez_compressed(OUT / "cyclogenesis.npz", per_season=per_season, smooth=sm, labels=lab,
                    n_seasons=n_seasons)
gen.to_parquet(OUT / "cyclogenesis_events.parquet")

# ----------------------------------------------------------------- figure
ext = box_extent(-165, -40, 22, 75, cell_km=CELL, pad_km=100)
fig, ax, grid_, fwd, ext = basemap(ext, cell_km=CELL, figsize=(10.5, 8.2), anchors=True)
full = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
im = ax.imshow(np.where(sm > 0, sm, np.nan), origin="lower", extent=full, cmap="YlOrRd",
               vmin=0, vmax=np.percentile(sm[sm > 0], 99.5), interpolation="bilinear", zorder=1)
ax.contour(np.linspace(full[0], full[1], grid.nx), np.linspace(full[2], full[3], grid.ny),
           (lab > 0).astype(float), levels=[0.5], colors="#15202B", linewidths=1.3, zorder=5)
bx, by = fwd.transform(gen.genesis_lon[gen.bomb].to_numpy(), gen.genesis_lat[gen.bomb].to_numpy())
ax.scatter(bx, by, s=7, color="#1F4FB4", zorder=6, label="bomb cyclone genesis")
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.legend(loc="lower left", fontsize=8)
fig.colorbar(im, ax=ax, shrink=0.7, label=f"genesis per {CELL:g} km cell per cool season (3x3 mean)")
ax.set_title("Cool-season cyclogenesis from analysed lows, Oct-Apr 2009/10-2017/18\n"
             "Fritzen et al. (2021) filters; outlined: genesis zones")
fig.savefig(OUT / "cyclogenesis_zones.png", dpi=140, bbox_inches="tight")
print("wrote cyclogenesis_zones.png")
