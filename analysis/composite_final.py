"""The cyclone-relative frontal composite, with the three known problems fixed.

1. Triple points are detected from the drawn fronts, not guessed from pressure.
   A triple point is the end of an occluded front where a cold-front end and a
   warm-front end both lie within JOIN_KM. A low within TRIPLE_KM of a triple
   point, and closer to it than to the other end of that occlusion, is a
   triple-point low: it sits on the frontal intersection of a parent cyclone,
   so centring a composite on it is centring on the wrong point. These lows are
   composited separately instead of mixed in.
2. The full storm tracks are included. Synoptic hours only, where the chart
   covers the whole hemisphere, so the Gulf of Alaska and Irminger Sea lows
   can come in without running off the edge of an intermediate map.
3. Each case is rotated so the low's direction of travel points east (+x).
   Heading comes from the low's own track, 6 hours either side; lows that
   moved less than MIN_MOVE_KM in that time have no meaningful heading and are
   left out of the rotated composites. "North" in the unrotated version is
   grid north of the equal-area projection, which is true north at 100W.

Writes outputs/composite_final.npz; figures come from composite_final_figs.py.
"""
import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from common import OUT, POINTS, REPO, fx, g

YEARS = range(2009, 2019)
SYN = [0, 6, 12, 18]
FRONTS = ["COLD", "WARM", "OCFNT", "STNRY"]
SPACING_KM, BIN_KM, REACH_KM = 25.0, 25.0, 1500.0
NB = int(2 * REACH_KM / BIN_KM)
JOIN_KM, TRIPLE_KM = 150.0, 250.0
MIN_MOVE_KM = 60.0
LOW_LAT, LOW_LON = (25.0, 70.0), (-178.0, -25.0)
GRID = g.Grid(cell_km=50.0)
FWD = GRID.transformer()

# ----------------------------------------------------------- lows + heading
tr = pd.read_parquet(REPO / "data" / "tracks" / "tracks.parquet")
tr = tr.sort_values(["track_id", "valid_time"]).reset_index(drop=True)
tr["x"], tr["y"] = FWD.transform(tr.lon.to_numpy(), tr.lat.to_numpy())
grp = tr.groupby("track_id", sort=False)
for k in (2, 1):
    for side, sh in (("p", k), ("n", -k)):
        tr[f"{side}x{k}"] = grp.x.shift(sh)
        tr[f"{side}y{k}"] = grp.y.shift(sh)
        tr[f"{side}t{k}"] = grp.valid_time.shift(sh)
px = tr.px2.fillna(tr.px1); py = tr.py2.fillna(tr.py1); pt = tr.pt2.fillna(tr.pt1)
nx = tr.nx2.fillna(tr.nx1); ny = tr.ny2.fillna(tr.ny1); nt = tr.nt2.fillna(tr.nt1)
# fall back to one-sided differences at the ends of a track
px, py, pt = px.fillna(tr.x), py.fillna(tr.y), pt.fillna(tr.valid_time)
nx, ny, nt = nx.fillna(tr.x), ny.fillna(tr.y), nt.fillna(tr.valid_time)
move = np.hypot(nx - px, ny - py) / 1000.0
span = (nt - pt).dt.total_seconds() / 3600.0
tr["heading"] = np.arctan2(ny - py, nx - px)
tr["moving"] = (move >= MIN_MOVE_KM) & (span.between(3, 15))
tr["speed"] = np.where(span > 0, move / span, np.nan)

lows = tr[tr.valid_time.dt.hour.isin(SYN) & tr.valid_time.dt.year.isin(YEARS)
          & tr.lat.between(*LOW_LAT) & tr.lon.between(*LOW_LON)
          & tr.pressure_hpa.between(940, 1030)].copy()
lows["year"] = lows.valid_time.dt.year
T1, T2 = lows.pressure_hpa.quantile([1 / 3, 2 / 3])
print(f"{len(lows):,} lows; {lows.moving.mean():.0%} have a heading; "
      f"median speed {np.nanmedian(lows.speed[lows.moving]):.0f} km/h; terciles {T1:.0f} / {T2:.0f} hPa")

STRATA = ["north", "rot", "rot_deep", "rot_mod", "rot_weak", "triple", "rot_land", "rot_ocean"]
hist = {s: {f: np.zeros((NB, NB)) for f in FRONTS} for s in STRATA}
n_low = {s: 0 for s in STRATA}
dset = pads.dataset(POINTS, format="parquet", partitioning="hive")
n_triple_pts = 0


def densify(df):
    pts = fx.explode(df)
    x, y = FWD.transform(pts.lon.to_numpy(), pts.lat.to_numpy())
    feat, tim = pts.feature_id.to_numpy(), pts.valid_time.to_numpy()
    same = feat[1:] == feat[:-1]
    x0, y0, x1, y1, t0 = x[:-1][same], y[:-1][same], x[1:][same], y[1:][same], tim[:-1][same]
    n = np.maximum(1, np.ceil(np.hypot(x1 - x0, y1 - y0) / (SPACING_KM * 1000)).astype(np.int64))
    leg = np.repeat(np.arange(n.size), n)
    f = (np.arange(n.sum()) - np.repeat(np.concatenate([[0], np.cumsum(n)[:-1]]), n)) / n[leg]
    return t0[leg], x0[leg] + f * (x1[leg] - x0[leg]), y0[leg] + f * (y1[leg] - y0[leg])


def ends(df):
    """Projected first and last vertex of every feature, with its time."""
    la0 = np.array([a[0] for a in df.lat_e2]) / 100; lo0 = np.array([b[0] for b in df.lon_e2]) / 100
    la1 = np.array([a[-1] for a in df.lat_e2]) / 100; lo1 = np.array([b[-1] for b in df.lon_e2]) / 100
    x0, y0 = FWD.transform(lo0, la0); x1, y1 = FWD.transform(lo1, la1)
    return pd.DataFrame({"t": df.valid_time.to_numpy(), "x0": x0, "y0": y0, "x1": x1, "y1": y1})


for year in YEARS:
    ly = lows[lows.year == year].sort_values("valid_time")
    fr = {}
    for ft in FRONTS:
        d = dset.to_table(columns=["valid_time", "ftype", "stage", "lat_e2", "lon_e2"],
                          filter=(pads.field("ftype") == ft) & (pads.field("year") == year)).to_pandas()
        fr[ft] = d[pd.to_datetime(d.valid_time, utc=True).dt.hour.isin(SYN).to_numpy()].reset_index(drop=True)

    # ---- triple points: occlusion ends with a cold end and a warm end nearby
    E = {ft: ends(fr[ft]) for ft in ("COLD", "WARM", "OCFNT")}
    cw = {}
    for ft in ("COLD", "WARM"):
        e = E[ft]
        cw[ft] = {t: np.column_stack([np.r_[s.x0, s.x1], np.r_[s.y0, s.y1]]) for t, s in e.groupby("t")}
    triple_rows = []            # (time, Tx, Ty, Sx, Sy)
    for t, s in E["OCFNT"].groupby("t"):
        if t not in cw["COLD"] or t not in cw["WARM"]:
            continue
        C_, W_ = cw["COLD"][t], cw["WARM"][t]
        for a, b in ((("x0", "y0"), ("x1", "y1")), (("x1", "y1"), ("x0", "y0"))):
            tx, ty = s[a[0]].to_numpy(), s[a[1]].to_numpy()
            dc = np.hypot(tx[:, None] - C_[:, 0], ty[:, None] - C_[:, 1]).min(axis=1) / 1000
            dw = np.hypot(tx[:, None] - W_[:, 0], ty[:, None] - W_[:, 1]).min(axis=1) / 1000
            hit = (dc <= JOIN_KM) & (dw <= JOIN_KM)
            for i in np.flatnonzero(hit):
                triple_rows.append((t, tx[i], ty[i], s[b[0]].iloc[i], s[b[1]].iloc[i]))
    tp = pd.DataFrame(triple_rows, columns=["t", "tx", "ty", "sx", "sy"])
    n_triple_pts += len(tp)
    tpg = {t: s[["tx", "ty", "sx", "sy"]].to_numpy() for t, s in tp.groupby("t")}

    is_triple = np.zeros(len(ly), dtype=bool)
    lt = ly.valid_time.to_numpy(); lx = ly.x.to_numpy(); lyy = ly.y.to_numpy()
    for i in range(len(ly)):
        arr = tpg.get(lt[i])
        if arr is None:
            continue
        dT = np.hypot(arr[:, 0] - lx[i], arr[:, 1] - lyy[i]) / 1000
        dS = np.hypot(arr[:, 2] - lx[i], arr[:, 3] - lyy[i]) / 1000
        is_triple[i] = bool(np.any((dT <= TRIPLE_KM) & (dT < dS)))
    ly = ly.assign(triple=is_triple)

    p = ly.pressure_hpa.to_numpy(); mv = ly.moving.to_numpy(); tri = ly.triple.to_numpy()
    ocean = (ly.lon.to_numpy() < -125) | (ly.lon.to_numpy() > -65) | (ly.lat.to_numpy() > 55)
    masks = {
        "north": ~tri, "rot": ~tri & mv,
        "rot_deep": ~tri & mv & (p <= T1), "rot_mod": ~tri & mv & (p > T1) & (p <= T2),
        "rot_weak": ~tri & mv & (p > T2), "triple": tri & mv,
        "rot_land": ~tri & mv & ~ocean, "rot_ocean": ~tri & mv & ocean,
    }
    for s in STRATA:
        n_low[s] += int(masks[s].sum())
    cos_h, sin_h = np.cos(ly.heading.to_numpy()), np.sin(ly.heading.to_numpy())
    keys = pd.Series(np.arange(len(ly))).groupby(lt).apply(np.array).to_dict()

    for ft in FRONTS:
        if fr[ft].empty:
            continue
        ptime, ppx, ppy = densify(fr[ft])
        order = np.argsort(ptime, kind="stable")
        ptime, ppx, ppy = ptime[order], ppx[order], ppy[order]
        uu, st = np.unique(ptime, return_index=True)
        en = np.append(st[1:], len(ptime))
        flat = {s: [] for s in STRATA}
        for u, a, b in zip(uu, st, en):
            idx = keys.get(u)
            if idx is None:
                continue
            dx = (ppx[a:b, None] - lx[None, idx]) / 1000
            dy = (ppy[a:b, None] - lyy[None, idx]) / 1000
            # rotate so the direction of travel is +x
            rx = dx * cos_h[None, idx] + dy * sin_h[None, idx]
            ry = -dx * sin_h[None, idx] + dy * cos_h[None, idx]
            for s in STRATA:
                m = masks[s][idx]
                if not m.any():
                    continue
                X, Y = (dx, dy) if s == "north" else (rx, ry)
                X, Y = X[:, m], Y[:, m]
                inr = (np.abs(X) < REACH_KM) & (np.abs(Y) < REACH_KM)
                if inr.any():
                    cx = ((X[inr] + REACH_KM) // BIN_KM).astype(np.int32)
                    cy = ((Y[inr] + REACH_KM) // BIN_KM).astype(np.int32)
                    flat[s].append(cy * NB + cx)
        for s in STRATA:
            if flat[s]:
                hist[s][ft] += np.bincount(np.concatenate(flat[s]), minlength=NB * NB).reshape(NB, NB)
    print(f"  {year}: {len(tp):,} triple points, {is_triple.sum():,} triple-point lows of {len(ly):,}", flush=True)

np.savez_compressed(OUT / "composite_final.npz",
                    **{f"{s}_{f}": hist[s][f] for s in STRATA for f in FRONTS},
                    n_low=np.array([n_low[s] for s in STRATA]), strata=np.array(STRATA),
                    bin_km=BIN_KM, reach_km=REACH_KM, t1=T1, t2=T2)
print(f"\n{n_triple_pts:,} triple points detected")
print("lows per stratum: " + "  ".join(f"{s} {n_low[s]:,}" for s in STRATA))
