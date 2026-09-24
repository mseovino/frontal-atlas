"""Reframe: the Piedmont boundary is the Appalachian lee trough.

Given that, two questions are worth answering. First, characterise the lee
trough properly, since it is a real and useful climatology in its own right.
Second, test whether damming is separable from it at all in this archive: if
CAD is present, the boundary during a strong New England high should sit in a
measurably different place from the ordinary lee trough, because the damming
cold dome pushes the boundary east into the coastal plain.
"""
import sys
import numpy as np, pandas as pd, pyarrow.dataset as pads
from common import CENTERS, MONTHS, read_fronts

YEARS = (2009, 2018)
crest = lambda lat: -84.0 + (lat - 34.5) * (7.0 / 6.0)
COLD_M = [10, 11, 12, 1, 2, 3, 4]

raw = read_fronts(["STNRY", "TROF"])
t = pd.to_datetime(raw.valid_time, utc=True)
raw = raw.assign(atime=t, month=t.dt.month, hour=t.dt.hour, year=t.dt.year)
raw = raw[(raw.year >= YEARS[0]) & (raw.year <= YEARS[1])].reset_index(drop=True)


def screen(lat_e2, lon_e2):
    """Mean offset east of the crest, for a mountain-parallel Piedmont line."""
    lat = np.asarray(lat_e2, dtype=float) / 100.0
    lon = np.asarray(lon_e2, dtype=float) / 100.0
    off = lon - crest(lat)
    m = (lat >= 33.0) & (lat <= 38.5) & (off >= -1.0) & (off <= 6.0)
    if m.sum() < 2:
        return False, np.nan, np.nan
    la, lo = lat[m], lon[m]
    dy = (la[-1] - la[0]) * 111.32
    dx = (lo[-1] - lo[0]) * 111.32 * np.cos(np.radians(la.mean()))
    if np.hypot(dx, dy) < 300.0:
        return False, np.nan, np.nan
    bearing = np.degrees(np.arctan2(dx, dy)) % 180.0
    if abs((bearing - 43.0 + 90) % 180 - 90) > 25.0:
        return False, np.nan, np.nan
    # Distance east of the crest, in km, averaged along the in-corridor part.
    off_km = (lo - crest(la)) * 111.32 * np.cos(np.radians(la))
    return True, float(np.mean(off_km)), float(np.mean(la))


ok, offkm, mlat = zip(*[screen(a, b) for a, b in zip(raw.lat_e2, raw.lon_e2)])
raw = raw.assign(ok=np.array(ok), off_km=np.array(offkm), mlat=np.array(mlat))
lt = raw[raw.ok & (raw.ftype == "TROF")].copy()
n_an = raw.atime.nunique()
print(f"{len(lt):,} Piedmont lee-trough features in {n_an:,} analyses")
print(f"present in {100*lt.atime.nunique()/n_an:.1f}% of analyses")
print(f"mean distance east of the crest: median {lt.off_km.median():.0f} km, "
      f"p25 {lt.off_km.quantile(.25):.0f}, p75 {lt.off_km.quantile(.75):.0f}")

md = raw.groupby("month").atime.nunique()
lm = lt.groupby("month").atime.nunique().reindex(range(1, 13), fill_value=0)
print("\nlee trough presence by month (% of analyses)")
for m in range(1, 13):
    r = 100 * lm[m] / md[m]
    print(f"  {MONTHS[m-1]} {r:5.1f}%  {'#' * int(round(r))}")

lh = lt.groupby("hour").atime.nunique()
hd = raw.groupby("hour").atime.nunique()
print("\nby analysis hour (% of analyses): " +
      "  ".join(f"{h:02d}Z {100*lh.get(h,0)/hd[h]:.1f}%" for h in [0,3,6,9,12,15,18,21]))

# ------------------------------------------------- is damming separable?
cds = pads.dataset(CENTERS, format="parquet", partitioning="hive")
hi = cds.to_table(columns=["valid_time","res","kind","lat","lon","pressure_hpa"],
                  filter=(pads.field("kind") == "H")).to_pandas()
hi = hi[hi.res == "HR"].copy()
hi["atime"] = pd.to_datetime(hi.valid_time, utc=True)
ne = hi[hi.lat.between(41, 50) & hi.lon.between(-78, -63)]

cold = lt[lt.month.isin(COLD_M)]
print(f"\ncold-season Piedmont troughs: {len(cold):,}")
print("\ndoes a New England high displace the boundary east?")
print(f"{'New England high':<28s}{'n':>7s}{'median km E of crest':>23s}")
for label, sel in [
    ("none in box", ~cold.atime.isin(set(ne.atime))),
    ("any", cold.atime.isin(set(ne.atime))),
    (">= 1022 hPa", cold.atime.isin(set(ne[ne.pressure_hpa >= 1022].atime))),
    (">= 1026 hPa", cold.atime.isin(set(ne[ne.pressure_hpa >= 1026].atime))),
]:
    s = cold[sel]
    if len(s) > 30:
        print(f"{label:<28s}{len(s):7,}{s.off_km.median():23.0f}")

a = cold[~cold.atime.isin(set(ne.atime))].off_km
b = cold[cold.atime.isin(set(ne[ne.pressure_hpa >= 1026].atime))].off_km
if len(b) > 30:
    pooled = np.sqrt((a.var() + b.var()) / 2)
    print(f"\nshift with a >= 1026 hPa high: {b.median()-a.median():+.0f} km "
          f"(pooled sd {pooled:.0f} km, Cohen d {(b.mean()-a.mean())/pooled:+.2f})")

# ------------------------------------------------------------------ figures
from common import basemap, box_extent, fx, g, plt, save

rate_m = (100 * lm / md).reindex(range(1, 13))
rate_h = pd.Series({h: 100 * lh.get(h, 0) / hd[h] for h in [0,3,6,9,12,15,18,21]})

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

axes[0].bar(range(1, 13), rate_m.values, color="#7A5C2E", width=0.72)
axes[0].set_xticks(range(1, 13)); axes[0].set_xticklabels(MONTHS, fontsize=7)
axes[0].set_ylabel("% of analyses")
axes[0].set_title("two maxima, not one")

axes[1].bar(range(8), rate_h.values, color="#7A5C2E", width=0.7)
axes[1].set_xticks(range(8))
axes[1].set_xticklabels([f"{h:02d}Z" for h in [0,3,6,9,12,15,18,21]], fontsize=7)
axes[1].set_ylabel("% of analyses"); axes[1].set_ylim(0, 30)
axes[1].set_title("afternoon maximum")

no_high = cold[~cold.atime.isin(set(ne.atime))].off_km
big_high = cold[cold.atime.isin(set(ne[ne.pressure_hpa >= 1026].atime))].off_km
bins = np.arange(-50, 500, 25)
axes[2].hist(no_high, bins=bins, density=True, color="#B9A57E", alpha=0.85,
             label=f"no New England high  (n={len(no_high):,})")
axes[2].hist(big_high, bins=bins, density=True, histtype="step", lw=2,
             color="#1F4E79", label=f"high >= 1026 hPa  (n={len(big_high):,})")
axes[2].axvline(no_high.median(), color="#7A5C2E", lw=1.2, ls="--")
axes[2].axvline(big_high.median(), color="#1F4E79", lw=1.2, ls="--")
axes[2].set_xlabel("km east of the Blue Ridge crest")
axes[2].set_ylabel("density"); axes[2].legend(fontsize=6.5, frameon=False)
axes[2].set_title("a strong high pulls it 52 km west")
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("The Appalachian lee trough as WPC analyses it, 2009-2018\n"
             "mountain-parallel Piedmont troughs, present in 21.6% of analyses",
             y=1.11, fontsize=12)
save(fig, "leetrough_timing.png")

pts = fx.explode(lt.drop(columns=["atime", "ok", "off_km", "mlat"]))
grid = g.Grid(cell_km=50.0)
counts = g.frequency_grid(pts, grid)
ext = box_extent(-86, -74, 31, 41, cell_km=50.0, pad_km=60)
fig, ax, grid, fwd, ext = basemap(ext, cell_km=50.0, figsize=(7.2, 7.8))
full = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
m = ax.imshow(np.where(counts > 0, counts, np.nan), origin="lower", extent=full,
              cmap="YlOrBr", interpolation="nearest", zorder=1,
              vmax=np.percentile(counts[counts > 0], 99))
cl = np.linspace(33.0, 40.0, 60)
cx, cy = fwd.transform(crest(cl), cl)
ax.plot(cx, cy, color="#3B5323", lw=1.8, ls="--", zorder=6)
ax.annotate("Blue Ridge crest", (cx[40], cy[40]), color="#3B5323", fontsize=8,
            xytext=(-84, 4), textcoords="offset points", zorder=7)
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.set_title("The Appalachian lee trough\n"
             f"{len(lt):,} mountain-parallel Piedmont troughs, 2009-2018, "
             "median 133 km east of the crest")
fig.colorbar(m, ax=ax, shrink=0.7, label="troughs crossing cell", extend="max")
save(fig, "leetrough_map.png")
