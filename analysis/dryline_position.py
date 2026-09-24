"""Where is the dryline likely to be today?

The heat map answers "where do drylines live". A forecaster needs the other
question: given that it is late May and 21Z, what longitude has the dryline
historically sat at? So this reduces every analysed dryline to its longitude
where it crosses a set of fixed latitudes, then builds the month-by-hour
distribution of that longitude.
"""
import sys


import numpy as np
import pandas as pd
from common import (MONTHS, N_ANALYSES, basemap, box_extent, fx, g, plt,
                    read_fronts, save)

# Latitudes to cut the dryline at. Chosen to span the southern Plains from the
# Rio Grande to the Kansas border without extending into latitudes where
# drylines are rare enough that the quantiles stop meaning anything.
CUTS = [31.0, 33.0, 35.0, 37.0]

raw = read_fronts("DRYLN")
t = pd.to_datetime(raw.valid_time, utc=True)
raw = raw.assign(month=t.dt.month, hour=t.dt.hour, year=t.dt.year)
print(f"{len(raw):,} DRYLN features, "
      f"{raw.valid_time.nunique():,} analyses containing one "
      f"({100*raw.valid_time.nunique()/N_ANALYSES:.1f}%)")


def crossings(lat_e2, lon_e2, cut):
    """Longitudes where a polyline crosses a latitude, linearly interpolated.

    A dryline is drawn roughly north-south, so it normally crosses each cut
    once; the occasional bulge crosses three times, and taking the median of
    the crossings is the stable summary of "where the line is" at that
    latitude.
    """
    lat = np.asarray(lat_e2, dtype=float) / 100.0
    lon = np.asarray(lon_e2, dtype=float) / 100.0
    if lat.size < 2:
        return np.nan
    a, b = lat[:-1], lat[1:]
    hit = ((a <= cut) & (b > cut)) | ((b <= cut) & (a > cut))
    if not hit.any():
        return np.nan
    f = (cut - a[hit]) / (b[hit] - a[hit])
    return float(np.median(lon[:-1][hit] + f * (lon[1:][hit] - lon[:-1][hit])))


rows = []
for cut in CUTS:
    lons = np.array([crossings(a, b, cut) for a, b in zip(raw.lat_e2, raw.lon_e2)])
    ok = np.isfinite(lons)
    rows.append(pd.DataFrame({
        "lat": cut, "lon": lons[ok], "month": raw.month.to_numpy()[ok],
        "hour": raw.hour.to_numpy()[ok], "year": raw.year.to_numpy()[ok],
    }))
cross = pd.concat(rows, ignore_index=True)
print(f"{len(cross):,} latitude crossings extracted\n")

# ------------------------------------------------------------ month table
print("Median dryline longitude by month (degrees W), n in brackets")
print(f"{'':>5s}" + "".join(f"{f'{c:g}N':>14s}" for c in CUTS))
for m in range(1, 13):
    cells = []
    for c in CUTS:
        s = cross[(cross.month == m) & (cross.lat == c)].lon
        cells.append(f"{abs(s.median()):.1f} [{len(s)}]" if len(s) >= 10 else "-")
    print(f"{MONTHS[m-1]:>5s}" + "".join(f"{x:>14s}" for x in cells))

# -------------------------------------------------------------- hour table
peak = cross[cross.month.isin([4, 5, 6])]
print("\nApr-Jun median longitude by analysis hour (the diurnal excursion)")
print(f"{'':>5s}" + "".join(f"{f'{c:g}N':>14s}" for c in CUTS))
for h in [0, 3, 6, 9, 12, 15, 18, 21]:
    cells = []
    for c in CUTS:
        s = peak[(peak.hour == h) & (peak.lat == c)].lon
        cells.append(f"{abs(s.median()):.1f} [{len(s)}]" if len(s) >= 10 else "-")
    print(f"{h:02d}Z" + "".join(f"{x:>14s}" for x in cells))

for c in CUTS:
    s = peak[peak.lat == c]
    a = s[s.hour == 12].lon.median()
    b = s[s.hour == 0].lon.median()
    if np.isfinite(a) and np.isfinite(b):
        km = (b - a) * 111.32 * np.cos(np.radians(c))
        print(f"  {c:g}N  12Z -> 00Z excursion: {km:+.0f} km "
              f"({abs(a):.1f}W -> {abs(b):.1f}W)")

# ------------------------------------------------------------------ plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)

# Monthly quartile envelope, one line per latitude.
colors = ["#8C2D04", "#CC4C02", "#EC7014", "#FE9929"]
for c, col in zip(CUTS, colors):
    med, lo, hi, ms = [], [], [], []
    for m in range(1, 13):
        s = cross[(cross.month == m) & (cross.lat == c)].lon
        if len(s) >= 10:
            ms.append(m); med.append(s.median())
            lo.append(s.quantile(0.25)); hi.append(s.quantile(0.75))
    axes[0].fill_between(ms, lo, hi, color=col, alpha=0.18, lw=0)
    axes[0].plot(ms, med, color=col, marker="o", ms=3.5, label=f"{c:g}N")
axes[0].set_xticks(range(1, 13)); axes[0].set_xticklabels(MONTHS, fontsize=7)
axes[0].set_title("by month (all hours)")
axes[0].set_ylabel("longitude")
axes[0].legend(fontsize=8, title="latitude", title_fontsize=8)

hours = [0, 3, 6, 9, 12, 15, 18, 21]
for c, col in zip(CUTS, colors):
    med, lo, hi, hs = [], [], [], []
    for h in hours:
        s = peak[(peak.hour == h) & (peak.lat == c)].lon
        if len(s) >= 10:
            hs.append(h); med.append(s.median())
            lo.append(s.quantile(0.25)); hi.append(s.quantile(0.75))
    axes[1].fill_between(hs, lo, hi, color=col, alpha=0.18, lw=0)
    axes[1].plot(hs, med, color=col, marker="o", ms=3.5, label=f"{c:g}N")
axes[1].set_xticks(hours); axes[1].set_xticklabels([f"{h:02d}Z" for h in hours], fontsize=7)
axes[1].set_title("by analysis hour (April-June only)")

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.yaxis.set_major_formatter(lambda v, p: f"{abs(v):.0f}W")
    a.grid(axis="y", color="0.9", lw=0.6)
fig.suptitle("Climatological dryline position: median with interquartile range",
             y=1.02, fontsize=11)
save(fig, "dryline_position.png")

# --------------------------------------------------- map of monthly medians
fig, ax, grid, fwd, ext = basemap(
    box_extent(-106, -93, 28, 40, cell_km=50.0, pad_km=120),
    cell_km=50.0, figsize=(7.6, 7.6))
cmap = plt.get_cmap("turbo")
for m in [3, 4, 5, 6, 7, 8]:
    lons, lats = [], []
    for c in CUTS:
        s = cross[(cross.month == m) & (cross.lat == c)].lon
        if len(s) >= 10:
            lons.append(s.median()); lats.append(c)
    if len(lons) >= 2:
        x, y = fwd.transform(np.array(lons), np.array(lats))
        ax.plot(x, y, color=cmap((m - 3) / 5), lw=2.4, marker="o", ms=5,
                zorder=6, label=MONTHS[m - 1])
ax.legend(fontsize=8, loc="upper left", title="median dryline", title_fontsize=8)
ax.set_title("Where the dryline sits, month by month\n"
             "median analysed longitude at four latitudes, 2006-2022")
save(fig, "dryline_march.png")
