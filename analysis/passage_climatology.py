"""Frontal passage climatology for a single forecast area.

The unit here is a *passage*, not a front-present analysis. A front that sits
over the area for two days is one event, not sixteen. So presence is evaluated
at every analysis, consecutive present-analyses are grouped into episodes, and
each episode counts as one passage. That is the quantity a forecaster means by
"how many fronts came through last month", and it is the one a raw frequency
grid does not give you.

The area is a disc around the office. A county warning area is an irregular
polygon roughly 200-400 km across, and synoptic fronts are an order of
magnitude longer than that, so the exact boundary barely changes a crossing
count -- "did a front pass within R km of the office" is the same statistic to
within noise, and it needs no external boundary file.

Two parameters decide the answer and neither has an obviously correct value,
so both are reported rather than chosen quietly. The radius sets what counts
as "here": 50 km means the front passed over the office, 150 km means it
crossed somewhere in the county warning area. The merge tolerance sets what
counts as one event: a hand-drawn front wobbles by tens of kilometres as it is
redrawn each cycle, so a hard threshold with no tolerance splits one system
into several. At 150 km, moving the tolerance from 3 to 24 hours halves the
count, which is the size of the effect being corrected. 24 hours is the
default, on the grounds that one synoptic system should count once.

Usage:  cwa.py <name> <lat> <lon> [radius_km]
"""
from __future__ import annotations

import sys


import numpy as np
import pandas as pd
from common import MONTHS, basemap, box_extent, fx, g, plt, read_fronts, save

TYPES = ["COLD", "WARM", "STNRY", "OCFNT", "TROF", "DRYLN", "SQLN"]
FRONTS = ["COLD", "WARM", "STNRY", "OCFNT"]
COLORS = {"COLD": "#2166AC", "WARM": "#D14A32", "STNRY": "#3F7D3A",
          "OCFNT": "#7B3294", "TROF": "#C8952A", "DRYLN": "#7A6A55",
          "SQLN": "#8C2D26"}
LABEL = {"COLD": "Cold", "WARM": "Warm", "STNRY": "Stationary",
         "OCFNT": "Occluded", "TROF": "Trough", "DRYLN": "Dryline",
         "SQLN": "Squall line"}
STEP_H = 3.0
MERGE_GAP_H = 24.0     # one synoptic system counts once
R_OFFICE, R_CWA = 50.0, 150.0
YEARS = (2007, 2022)          # complete years in the fronts store
N_YEARS = YEARS[1] - YEARS[0] + 1


def mindist(lat_e2, lon_e2, clat, clon):
    """Closest approach of the polyline to the centre, in km.

    Distance is measured to the nearest point on each leg, not just to the
    vertices, so a long straight front that steps over the area between two
    vertices is still counted.
    """
    lat = np.asarray(lat_e2, dtype=float) / 100.0
    lon = np.asarray(lon_e2, dtype=float) / 100.0
    # Local flat-earth metres about the centre. Over a few hundred km this is
    # well inside the error already present in a hand-drawn front.
    kx = 111.32 * np.cos(np.radians(clat))
    x = (lon - clon) * kx
    y = (lat - clat) * 111.32
    if x.size == 1:
        return float(np.hypot(x[0], y[0]))
    ax, ay = x[:-1], y[:-1]
    bx, by = x[1:], y[1:]
    dx, dy = bx - ax, by - ay
    seg = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(seg > 0, -(ax * dx + ay * dy) / np.where(seg > 0, seg, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return float(np.hypot(ax + t * dx, ay + t * dy).min())


def episodes(times, merge_h=MERGE_GAP_H):
    """Present-analyses grouped into passage events.

    Runs are broken only by a gap longer than `merge_h`, not by a single
    missing analysis, because a gap of one cycle almost always means the front
    was redrawn slightly outside the threshold rather than that it left and
    came back.
    """
    ts = np.array(sorted({pd.Timestamp(x).tz_localize(None) for x in times}),
                  dtype="datetime64[ns]")
    if ts.size == 0:
        return pd.DataFrame(columns=["start", "end", "hours"])
    gap = np.diff(ts).astype("timedelta64[m]").astype(float) / 60.0
    runs = np.split(np.arange(ts.size), np.flatnonzero(gap > merge_h + 0.5) + 1)
    return pd.DataFrame([{
        "start": ts[r[0]], "end": ts[r[-1]],
        "hours": (ts[r[-1]] - ts[r[0]]) / np.timedelta64(1, "h") + STEP_H,
    } for r in runs])


def main(name, clat, clon, radius_km):
    raw = read_fronts(TYPES)
    t = pd.to_datetime(raw.valid_time, utc=True)
    raw = raw.assign(atime=t, year=t.dt.year)
    raw = raw[(raw.year >= YEARS[0]) & (raw.year <= YEARS[1])].reset_index(drop=True)

    dist = np.array([mindist(a, b, clat, clon)
                     for a, b in zip(raw.lat_e2, raw.lon_e2)])
    raw = raw.assign(dist=dist)
    near = raw[dist <= radius_km].copy()
    n_an = raw.atime.nunique()
    print(f"{name}: {clat:.2f}N {abs(clon):.2f}W, {radius_km:g} km radius")
    print(f"{n_an:,} analyses, {YEARS[0]}-{YEARS[1]} ({N_YEARS} years)")
    print(f"{len(near):,} features cross the area\n")

    # The two choices that move the answer, shown before any of it is used.
    print("sensitivity of the cold-front count to the two free parameters")
    print(f"{'radius':>9s}" + "".join(f"{f'merge {m:g}h':>12s}"
                                      for m in (3, 6, 12, 24)))
    cold = raw[raw.ftype == "COLD"]
    for r in (R_OFFICE, 100.0, R_CWA, 250.0):
        row = []
        for m in (3, 6, 12, 24):
            e = episodes(cold[cold.dist <= r].atime, m)
            row.append(f"{len(e)/N_YEARS:6.1f}/yr")
        print(f"{r:7.0f}km" + "".join(f"{c:>12s}" for c in row))
    for r, tag in ((R_OFFICE, "over the office"), (R_CWA, "anywhere in the CWA")):
        e = episodes(cold[cold.dist <= r].atime)
        print(f"  {tag:22s} r={r:g} km, merge {MERGE_GAP_H:g} h: "
              f"{len(e)/N_YEARS:.1f} cold fronts per year, "
              f"median {e.hours.median():.0f} h over the area")
    print()

    ep = {}
    for ft in TYPES:
        sub = near[near.ftype == ft]
        e = episodes(sub.atime)
        if len(e):
            e["month"] = pd.DatetimeIndex(e.start).month
            e["year"] = pd.DatetimeIndex(e.start).year
            e["hour"] = pd.DatetimeIndex(e.start).hour
        ep[ft] = e

    # ------------------------------------------------------------- table
    print("passages per year, and the month they favour")
    print(f"{'type':<13s}{'per year':>10s}{'median hrs':>12s}{'peak month':>12s}")
    for ft in TYPES:
        e = ep[ft]
        if not len(e):
            print(f"{LABEL[ft]:<13s}{0:>10}"); continue
        per_month = e.groupby("month").size().reindex(range(1, 13), fill_value=0)
        print(f"{LABEL[ft]:<13s}{len(e)/N_YEARS:10.1f}{e.hours.median():12.0f}"
              f"{MONTHS[per_month.idxmax()-1]:>12s}")

    print("\ncold frontal passages per month (mean per year, and range)")
    e = ep["COLD"]
    tab = e.groupby(["year", "month"]).size().unstack(fill_value=0).reindex(
        columns=range(1, 13), fill_value=0)
    for m in range(1, 13):
        col = tab[m]
        print(f"  {MONTHS[m-1]}  {col.mean():4.1f}   "
              f"(min {col.min()}, max {col.max()})  "
              f"{'#' * int(round(col.mean() * 3))}")
    print(f"  ALL  {tab.sum(axis=1).mean():.1f} cold frontal passages per year")

    gaps = np.diff(np.sort(pd.DatetimeIndex(e.start).values)) / np.timedelta64(1, "D")
    print(f"\ndays between cold frontal passages: median {np.median(gaps):.1f}, "
          f"p25 {np.percentile(gaps,25):.1f}, p75 {np.percentile(gaps,75):.1f}")
    for season, months in [("winter DJF", [12,1,2]), ("spring MAM", [3,4,5]),
                           ("summer JJA", [6,7,8]), ("autumn SON", [9,10,11])]:
        s = e[e.month.isin(months)]
        print(f"  {season}: {len(s)/N_YEARS:4.1f} per season, "
              f"one every {90*N_YEARS/max(len(s),1):.1f} days")

    # ------------------------------------------------------------ figures
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.8))

    ax = axes[0]
    bottom = np.zeros(12)
    for ft in FRONTS:
        v = (ep[ft].groupby("month").size().reindex(range(1, 13), fill_value=0)
             / N_YEARS).to_numpy() if len(ep[ft]) else np.zeros(12)
        ax.bar(range(1, 13), v, 0.74, bottom=bottom, color=COLORS[ft],
               label=LABEL[ft])
        bottom += v
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS, fontsize=7)
    ax.set_ylabel("passages per month"); ax.set_title("how many, and what kind")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    box = [ep["COLD"][ep["COLD"].month == m].groupby("year").size()
           .reindex(range(YEARS[0], YEARS[1]+1), fill_value=0).to_numpy()
           for m in range(1, 13)]
    bp = ax.boxplot(box, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=1.4))
    for p in bp["boxes"]:
        p.set_facecolor("#2166AC"); p.set_edgecolor("#12395F")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS, fontsize=7)
    ax.set_ylabel("cold fronts per month")
    ax.set_title(f"year-to-year spread ({N_YEARS} years)")

    # Arrival hour is computed from the four synoptic analyses only. Fronts
    # are drawn more often at 00/06/12/18Z than at the off-hours even inside a
    # fixed domain -- cold fronts by a factor of 1.23, while troughs and
    # stationary fronts run the other way -- so mixing all eight hours puts a
    # spurious twice-a-day oscillation into any arrival-time statistic.
    ax = axes[2]
    hrs = [0, 6, 12, 18]
    syn = near[(near.ftype == "COLD") & (near.atime.dt.hour.isin(hrs))]
    e6 = episodes(syn.atime)
    v = pd.Series(pd.DatetimeIndex(e6.start).hour).value_counts().reindex(
        hrs, fill_value=0)
    ax.bar(range(4), 100 * v.to_numpy() / max(v.sum(), 1), 0.66, color="#2166AC")
    ax.axhline(25, color="0.5", lw=1.0, ls="--")
    ax.annotate("even", (3.42, 25), fontsize=7, color="0.4", va="bottom", ha="right")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{h:02d}Z" for h in hrs], fontsize=8)
    ax.set_ylabel("% of cold frontal passages")
    ax.set_title("when they arrive\n(synoptic analyses only)", fontsize=10)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Frontal passage climatology  -  {name}  "
                 f"({radius_km:g} km radius, {YEARS[0]}-{YEARS[1]})",
                 y=1.07, fontsize=12)
    save(fig, f"cwa_{name.lower().replace(' ', '_')}_timing.png")

    # ---------------------------------------------------------------- map
    pts = fx.explode(near[near.ftype == "COLD"].drop(columns=["atime"]))
    grid = g.Grid(cell_km=50.0)
    counts = g.frequency_grid(pts, grid)
    ext = box_extent(clon - 14, clon + 14, clat - 10, clat + 10, cell_km=50.0)
    fig, ax, grid, fwd, ext = basemap(ext, cell_km=50.0, figsize=(7.6, 7.2))
    full = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
    m = ax.imshow(np.where(counts > 0, counts, np.nan), origin="lower",
                  extent=full, cmap="Blues", interpolation="nearest", zorder=1,
                  vmax=np.percentile(counts[counts > 0], 99))
    th = np.linspace(0, 2 * np.pi, 361)
    kx = 111.32 * np.cos(np.radians(clat))
    rx, ry = fwd.transform(clon + radius_km * np.cos(th) / kx,
                           clat + radius_km * np.sin(th) / 111.32)
    ax.plot(rx, ry, color="#B03030", lw=2.0, zorder=7)
    cx, cy = fwd.transform(clon, clat)
    ax.plot(cx, cy, marker="*", ms=13, color="#B03030", zorder=8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"Cold fronts that passed within {radius_km:g} km of {name}\n"
                 f"where they were when they did  -  50 km cells, "
                 f"{YEARS[0]}-{YEARS[1]}")
    fig.colorbar(m, ax=ax, shrink=0.7, label="crossings per cell", extend="max")
    save(fig, f"cwa_{name.lower().replace(' ', '_')}_map.png")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]),
         float(sys.argv[4]) if len(sys.argv) > 4 else 150.0)
