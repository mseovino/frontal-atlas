"""What does WPC actually call a squall line?

3,921 features in sixteen years is roughly one every thirty-six analyses, so
this is a description of analysis practice rather than a climatology of squall
lines. Framed that way it still answers a real question: when does a squall
line show up on the national surface product instead of being folded into a
cold front or a trough?
"""
import sys


import numpy as np
import pandas as pd
from common import (MONTHS, N_ANALYSES, basemap, box_extent, fx, g, np, plt,
                    read_fronts, save)

raw = read_fronts("SQLN")
print(f"{len(raw):,} SQLN features")

pts = fx.explode(raw)
t = pd.to_datetime(raw.valid_time, utc=True)
raw = raw.assign(month=t.dt.month, hour=t.dt.hour, year=t.dt.year, date=t.dt.date)

# ---------------------------------------------------------------- when
per_month = raw.groupby("month").size().reindex(range(1, 13), fill_value=0)
per_hour = raw.groupby("hour").size().reindex([0, 3, 6, 9, 12, 15, 18, 21], fill_value=0)
per_year = raw.groupby("year").size()

# Analyses carrying at least one squall line -- the honest "how often does
# this appear at all" number, since one analysis can hold several.
an_with = raw.valid_time.nunique()
days_with = raw.date.nunique()
print(f"analyses containing a squall line: {an_with:,} of {N_ANALYSES:,} "
      f"({100*an_with/N_ANALYSES:.1f}%)")
print(f"distinct days: {days_with:,} of ~5,840 ({100*days_with/5840:.1f}%)")
print(f"features per analysis when present: {len(raw)/an_with:.2f}")

print("\nby month:")
for m, n in per_month.items():
    print(f"  {MONTHS[m-1]} {n:5,}  {'#' * int(60*n/per_month.max())}")

print("\nby analysis hour (UTC):")
for h, n in per_hour.items():
    print(f"  {h:02d}Z {n:5,}  {'#' * int(50*n/per_hour.max())}")

print("\nby year:")
for y, n in per_year.items():
    print(f"  {y} {n:5,}")

# ---------------------------------------------------------------- how long
lengths = np.array([
    g.polyline_length_km(np.asarray(a) / 100.0, np.asarray(b) / 100.0)
    for a, b in zip(raw.lat_e2, raw.lon_e2)
])
print(f"\nlength km: median {np.median(lengths):.0f}, "
      f"p25 {np.percentile(lengths, 25):.0f}, p75 {np.percentile(lengths, 75):.0f}, "
      f"max {lengths.max():.0f}")

# ---------------------------------------------------------------- where
grid = g.Grid(cell_km=100.0)
counts = g.frequency_grid(pts, grid)
ext = box_extent(-107, -66, 24, 50, cell_km=100.0, pad_km=150)

fig, ax, grid, fwd, ext = basemap(ext, cell_km=100.0, figsize=(10, 6.6))
full = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
m = ax.imshow(np.where(counts > 0, counts, np.nan), origin="lower", extent=full,
              cmap="inferno_r", interpolation="nearest", zorder=1,
              vmax=np.percentile(counts[counts > 0], 99))
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.set_title("Squall lines as analysed by WPC, 2006-2022\n"
             f"{len(raw):,} features, 100 km cells -- analysis practice, not a "
             "squall line climatology")
fig.colorbar(m, ax=ax, shrink=0.75, label="analysed squall lines crossing cell",
             extend="max")
save(fig, "sqln_map.png")

# ---------------------------------------------------------------- timing panel
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
axes[0].bar(range(1, 13), per_month.values, color="#7B2D26", width=0.72)
axes[0].set_xticks(range(1, 13)); axes[0].set_xticklabels(MONTHS, fontsize=7)
axes[0].set_title("by month"); axes[0].set_ylabel("features")

axes[1].bar(range(len(per_hour)), per_hour.values, color="#7B2D26", width=0.72)
axes[1].set_xticks(range(len(per_hour)))
axes[1].set_xticklabels([f"{h:02d}Z" for h in per_hour.index], fontsize=7)
axes[1].set_title("by analysis hour")

axes[2].plot(per_year.index, per_year.values, color="#7B2D26", marker="o", ms=3)
axes[2].set_title("by year"); axes[2].set_ylim(bottom=0)
for a in axes:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("When WPC draws a squall line", y=1.04, fontsize=11)
save(fig, "sqln_timing.png")
