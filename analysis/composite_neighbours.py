"""The Norwegian cyclone model, rebuilt from operational analyses.

For every analysed low, every front drawn in the same analysis is expressed in
coordinates centred on that low, and the results are stacked. What comes out
is the average frontal structure of a real surface cyclone, with the spread
that the 1919 cartoon does not have.

Three choices matter.

Constant-arc-length resampling. Analysts put vertices where a line bends, so
raw vertices would map pen habits. Every polyline is densified to a fixed
spacing in the projected plane before compositing, the same correction the
gridded density maps use.

Equal-area coordinates. Offsets are computed in the project's Lambert
azimuthal equal-area plane and then differenced, so a bin 1,000 km north of
the low covers the same ground as one 1,000 km south of it.

North-up, not motion-relative. The classical composite is often rotated onto
the cyclone's heading. This one is not, because that requires tracks and would
narrow the sample; since most of these cyclones move roughly eastward the
canonical structure survives, somewhat blurred by the spread in heading. That
blurring is a known cost, not an accident.

The high-resolution centers begin in 2009 and end in 2018, so this uses those
ten complete overlap years with the fronts store.
"""
import sys


import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from common import OUT, CENTERS, POINTS, REPO, fx, g, plt, save

YEARS = range(2009, 2019)
FRONTS = ["COLD", "WARM", "OCFNT", "STNRY"]
SPACING_KM = 25.0
BIN_KM = 25.0
REACH_KM = 1500.0
NB = int(2 * REACH_KM / BIN_KM)          # bins per side
EDGES = np.linspace(-REACH_KM, REACH_KM, NB + 1)

# Lows are restricted to the well-analysed core of the domain, so a composite
# is not built partly from cyclones whose surroundings fall off the chart.
LOW_LAT = (25.0, 60.0)
LOW_LON = (-150.0, -40.0)

GRID = g.Grid(cell_km=50.0)
FWD = GRID.transformer()

STYLE = {"COLD": "#2166AC", "WARM": "#D6604D",
         "OCFNT": "#7B3294", "STNRY": "#4D7C3A"}
NAME = {"COLD": "Cold front", "WARM": "Warm front",
        "OCFNT": "Occluded front", "STNRY": "Stationary front"}


def densify(df):
    """Polyline rows -> projected points at roughly SPACING_KM along-track."""
    pts = fx.explode(df)
    if pts.empty:
        return None
    x, y = FWD.transform(pts.lon.to_numpy(), pts.lat.to_numpy())
    feat = pts.feature_id.to_numpy()
    tim = pts.valid_time.to_numpy()

    same = feat[1:] == feat[:-1]
    x0, y0, x1, y1 = x[:-1][same], y[:-1][same], x[1:][same], y[1:][same]
    t0 = tim[:-1][same]
    good = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    x0, y0, x1, y1, t0 = x0[good], y0[good], x1[good], y1[good], t0[good]
    if x0.size == 0:
        return None

    n = np.maximum(1, np.ceil(np.hypot(x1 - x0, y1 - y0) /
                              (SPACING_KM * 1000.0)).astype(np.int64))
    leg = np.repeat(np.arange(n.size), n)
    off = np.arange(n.sum()) - np.repeat(
        np.concatenate([[0], np.cumsum(n)[:-1]]), n)
    f = off / n[leg]
    px = x0[leg] + f * (x1[leg] - x0[leg])
    py = y0[leg] + f * (y1[leg] - y0[leg])
    return t0[leg], px, py


# --------------------------------------------------------------------- lows
cds = pads.dataset(CENTERS, format="parquet", partitioning="hive")
lows = cds.to_table(
    columns=["valid_time", "res", "kind", "lat", "lon", "pressure_hpa"],
    filter=(pads.field("kind") == "L"),
).to_pandas()
lows = lows[lows.res == "HR"].copy()
lows["atime"] = pd.to_datetime(lows.valid_time, utc=True)
lows = lows[lows.atime.dt.year.isin(YEARS)]
lows = lows[lows.lat.between(*LOW_LAT) & lows.lon.between(*LOW_LON)]
lows = lows.dropna(subset=["pressure_hpa"])
lows = lows[lows.pressure_hpa.between(940, 1030)]
lx, ly = FWD.transform(lows.lon.to_numpy(), lows.lat.to_numpy())
lows = lows.assign(x=lx, y=ly, year=lows.atime.dt.year)
print(f"{len(lows):,} analysed lows in {min(YEARS)}-{max(YEARS)}")
print(f"central pressure: median {lows.pressure_hpa.median():.0f} hPa, "
      f"terciles at {lows.pressure_hpa.quantile(1/3):.0f} and "
      f"{lows.pressure_hpa.quantile(2/3):.0f}")

# --------------------------------------------------- neighbouring lows
# The composite attributes no front to any particular low, so a frontal
# system with both a primary and an analysed triple-point low enters twice,
# once centred on each. Whether that happens depends on whether the analyst
# drew the secondary low, which is not consistent. Flagging each low by its
# neighbourhood lets the composite be rebuilt without that ambiguity.
NEIGH_KM = 750.0

def neighbours(df):
    nn = np.full(len(df), np.inf)
    primary = np.ones(len(df), dtype=bool)
    for _, idx in df.groupby("atime", sort=False).indices.items():
        if idx.size == 1:
            continue
        x = df.x.to_numpy()[idx]; y = df.y.to_numpy()[idx]
        p = df.pressure_hpa.to_numpy()[idx]
        d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) / 1000.0
        np.fill_diagonal(d, np.inf)
        nn[idx] = d.min(axis=1)
        near = d <= NEIGH_KM
        # Primary = no deeper low within NEIGH_KM. Ties broken by first index,
        # so exactly one low in a cluster is ever the primary.
        for k in range(idx.size):
            comp = np.flatnonzero(near[k])
            if comp.size and (p[comp] < p[k]).any():
                primary[idx[k]] = False
    return nn, primary

nn_km, is_primary = neighbours(lows)
lows = lows.assign(nn_km=nn_km, primary=is_primary)
print(f"nearest other low: median {np.median(nn_km[np.isfinite(nn_km)]):.0f} km")
for cut in (500, 750, 1000, 1500):
    print(f"  another low within {cut:5d} km: {100*np.mean(nn_km <= cut):5.1f}%")
print(f"lows that are the deepest within {NEIGH_KM:g} km: "
      f"{100*is_primary.mean():.1f}%")

T1, T2 = lows.pressure_hpa.quantile(1 / 3), lows.pressure_hpa.quantile(2 / 3)
STRATA = {
    "all": lambda d: np.ones(len(d), dtype=bool),
    "isolated": lambda d: d.nn_km > NEIGH_KM,
    "primary": lambda d: d.primary.to_numpy(),
    "secondary": lambda d: ~d.primary.to_numpy(),
}

hist = {s: {f: np.zeros((NB, NB), dtype=np.float64) for f in FRONTS}
        for s in STRATA}
n_low = {s: 0 for s in STRATA}

dset = pads.dataset(POINTS, format="parquet", partitioning="hive")

for year in YEARS:
    ysub = lows[lows.year == year]
    if ysub.empty:
        continue
    # Index the year's lows by analysis time once.
    ysub = ysub.sort_values("atime")
    keys = ysub.atime.to_numpy()
    bounds = {}
    uniq, starts = np.unique(keys, return_index=True)
    ends = np.append(starts[1:], len(keys))
    for u, a, b in zip(uniq, starts, ends):
        bounds[u] = (a, b)
    lxa, lya = ysub.x.to_numpy(), ysub.y.to_numpy()
    strata_mask = {s: fn(ysub).to_numpy() if hasattr(fn(ysub), "to_numpy")
                   else fn(ysub) for s, fn in STRATA.items()}
    for s in STRATA:
        n_low[s] += int(np.sum(strata_mask[s]))

    for ftype in FRONTS:
        raw = dset.to_table(
            columns=["valid_time", "ftype", "stage", "lat_e2", "lon_e2"],
            filter=(pads.field("ftype") == ftype) & (pads.field("year") == year),
        ).to_pandas()
        if raw.empty:
            continue
        dens = densify(raw)
        if dens is None:
            continue
        ptime, px, py = dens
        order = np.argsort(ptime, kind="stable")
        ptime, px, py = ptime[order], px[order], py[order]
        pu, pstart = np.unique(ptime, return_index=True)
        pend = np.append(pstart[1:], len(ptime))

        flat = {s: [] for s in STRATA}
        for u, a, b in zip(pu, pstart, pend):
            if u not in bounds:
                continue
            i, j = bounds[u]
            dx = (px[a:b, None] - lxa[None, i:j]) / 1000.0
            dy = (py[a:b, None] - lya[None, i:j]) / 1000.0
            inr = (np.abs(dx) < REACH_KM) & (np.abs(dy) < REACH_KM)
            if not inr.any():
                continue
            cx = np.floor((dx + REACH_KM) / BIN_KM).astype(np.int32)
            cy = np.floor((dy + REACH_KM) / BIN_KM).astype(np.int32)
            idx = cy * NB + cx
            for s in STRATA:
                m = inr & strata_mask[s][None, i:j]
                if m.any():
                    flat[s].append(idx[m])
        for s in STRATA:
            if flat[s]:
                v = np.concatenate(flat[s])
                hist[s][ftype] += np.bincount(
                    v, minlength=NB * NB).reshape(NB, NB)
    print(f"  {year} done", flush=True)

np.savez_compressed(
    OUT / "norwegian_neighbour.npz",
    **{f"{s}_{f}": hist[s][f] for s in STRATA for f in FRONTS},
    n_low=np.array([n_low[s] for s in STRATA]),
    strata=np.array(list(STRATA)), bin_km=BIN_KM, reach_km=REACH_KM,
)
print("\nlows per stratum: " + "  ".join(f"{s} {n_low[s]:,}" for s in STRATA))
print("saved norwegian_neighbour.npz")
