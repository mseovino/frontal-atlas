"""Cyclone composites by lifecycle stage, north up.

Stages come from the fronts drawn around each low (Matthew's framing):
  open         a cold front and a warm front end within ATTACH_KM, and no
               occluded front comes within WRAP_NEAR_KM
  attached     an occluded front ends within ATTACH_KM -- the early occlusion,
               drawn running into the low
  wrapped      no occluded front ends near the low, but one passes within
               WRAP_NEAR_KM and spans at least WRAP_DEG of azimuth around it --
               the mature stage, where the occlusion curls round the centre
               without being connected to it
Attached lows are split again by the track: still deepening over the next 12 h,
or already filling.

Occlusion type. The archive has one occluded symbol, but at the triple point a
cold-type occlusion should continue roughly in line with the cold front, and a
warm-type one in line with the warm front. Each triple point is classified by
which of the two its occlusion is more nearly collinear with, and the parent
low (at the other end of that occlusion) inherits the type. The Unified
Manual's regional description -- warm occlusions on the east sides of ocean
basins and in the lee of the Divide, cold ones on the west sides of basins --
is the check on whether that classification means anything.

Writes outputs/composite_stages.npz and outputs/occlusion_types.parquet.
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
ATTACH_KM, JOIN_KM, TRIPLE_KM = 250.0, 150.0, 250.0
WRAP_NEAR_KM, WRAP_FAR_KM, WRAP_DEG = 400.0, 900.0, 120.0
DIR_KM = 150.0          # length of the stretch used to measure a front's direction at the triple point
LOW_LAT, LOW_LON = (25.0, 70.0), (-178.0, -25.0)
FWD = g.Grid(cell_km=50.0).transformer()

# ------------------------------------------------------------------- lows
tr = pd.read_parquet(REPO / "data/tracks/tracks.parquet")
tr["t"] = tr.valid_time.dt.tz_localize(None)
p_at = tr.set_index(["track_id", "t"]).pressure_hpa
lows = tr[tr.t.dt.hour.isin(SYN) & tr.t.dt.year.isin(YEARS)
          & tr.lat.between(*LOW_LAT) & tr.lon.between(*LOW_LON)
          & tr.pressure_hpa.between(940, 1030)].copy()
lows["x"], lows["y"] = FWD.transform(lows.lon.to_numpy(), lows.lat.to_numpy())
lows["dp12"] = p_at.reindex(list(zip(lows.track_id, lows.t + pd.Timedelta(hours=12)))).to_numpy() - lows.pressure_hpa.to_numpy()
lows["year"] = lows.t.dt.year
print(f"{len(lows):,} lows; 12-h tendency known for {lows.dp12.notna().mean():.0%}")

STRATA = ["open", "attached", "attached_deepening", "attached_filling", "wrapped",
          "occ_cold", "occ_warm", "triple", "neither"]
hist = {s: {f: np.zeros((NB, NB)) for f in FRONTS} for s in STRATA}
n_low = {s: 0 for s in STRATA}
types_rows = []
dset = pads.dataset(POINTS, format="parquet", partitioning="hive")


def densify(df):
    pts = fx.explode(df)
    x, y = FWD.transform(pts.lon.to_numpy(), pts.lat.to_numpy())
    feat = pts.feature_id.to_numpy()
    tim = pd.to_datetime(pts.valid_time, utc=True).dt.tz_localize(None).to_numpy()
    same = feat[1:] == feat[:-1]
    x0, y0, x1, y1, t0 = x[:-1][same], y[:-1][same], x[1:][same], y[1:][same], tim[:-1][same]
    n = np.maximum(1, np.ceil(np.hypot(x1 - x0, y1 - y0) / (SPACING_KM * 1000)).astype(np.int64))
    leg = np.repeat(np.arange(n.size), n)
    f = (np.arange(n.sum()) - np.repeat(np.concatenate([[0], np.cumsum(n)[:-1]]), n)) / n[leg]
    return t0[leg], x0[leg] + f * (x1[leg] - x0[leg]), y0[leg] + f * (y1[leg] - y0[leg])


def projected(df):
    """Per feature: time and projected vertex array (metres)."""
    out = []
    for t, la, lo in zip(pd.to_datetime(df.valid_time, utc=True).dt.tz_localize(None), df.lat_e2, df.lon_e2):
        x, y = FWD.transform(np.asarray(lo) / 100.0, np.asarray(la) / 100.0)
        out.append((np.datetime64(t, "ns"), np.column_stack([x, y])))
    return out


def direction_from_end(v, at_start):
    """Unit vector pointing from a polyline end back along the line, DIR_KM in."""
    pts = v if at_start else v[::-1]
    d = np.hypot(*(pts - pts[0]).T) / 1000
    k = np.searchsorted(d, DIR_KM)
    k = min(max(k, 1), len(pts) - 1)
    u = pts[k] - pts[0]
    n = np.hypot(*u)
    return u / n if n > 0 else None


def angle(u, v):
    return np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1)))


for year in YEARS:
    ly = lows[lows.year == year].reset_index(drop=True)
    fr = {}
    for ft in FRONTS:
        d = dset.to_table(columns=["valid_time", "ftype", "stage", "lat_e2", "lon_e2"],
                          filter=(pads.field("ftype") == ft) & (pads.field("year") == year)).to_pandas()
        fr[ft] = d[pd.to_datetime(d.valid_time, utc=True).dt.hour.isin(SYN).to_numpy()].reset_index(drop=True)
    P = {ft: projected(fr[ft]) for ft in ("COLD", "WARM", "OCFNT")}
    byt = {ft: {} for ft in P}
    for ft in P:
        for t, v in P[ft]:
            byt[ft].setdefault(t, []).append(v)

    # ---- per low: attached ends, occlusion proximity and wrap
    lt = ly.t.to_numpy(dtype="datetime64[ns]"); lx = ly.x.to_numpy(); lyy = ly.y.to_numpy()
    dend = {ft: np.full(len(ly), np.inf) for ft in P}
    occ_near = np.full(len(ly), np.inf); occ_span = np.zeros(len(ly))
    for i in range(len(ly)):
        c = np.array([lx[i], lyy[i]])
        for ft in P:
            vs = byt[ft].get(lt[i])
            if not vs:
                continue
            ends = np.array([p for v in vs for p in (v[0], v[-1])])
            dend[ft][i] = np.hypot(*(ends - c).T).min() / 1000
            if ft == "OCFNT":
                allv = np.vstack(vs)
                dd = np.hypot(*(allv - c).T) / 1000
                occ_near[i] = dd.min()
                sel = allv[dd <= WRAP_FAR_KM]
                if len(sel):
                    az = (np.degrees(np.arctan2(*(sel - c).T[::-1])) % 360) // 30
                    occ_span[i] = 30.0 * len(np.unique(az))

    # ---- triple points and occlusion type
    tri = np.zeros(len(ly), dtype=bool); otype = np.array([""] * len(ly), dtype=object)
    for t, vs in byt["OCFNT"].items():
        C_, W_ = byt["COLD"].get(t), byt["WARM"].get(t)
        if not C_ or not W_:
            continue
        for v in vs:
            for at_start in (True, False):
                T = v[0] if at_start else v[-1]
                S = v[-1] if at_start else v[0]
                def nearest(lines):
                    best = (np.inf, None, None)
                    for w in lines:
                        for s_end, p in ((True, w[0]), (False, w[-1])):
                            d = np.hypot(*(p - T)) / 1000
                            if d < best[0]:
                                best = (d, w, s_end)
                    return best
                dc, wc, sc = nearest(C_)
                dw, ww, sw = nearest(W_)
                if dc > JOIN_KM or dw > JOIN_KM:
                    continue
                o_in = direction_from_end(v, at_start)          # from T back toward the low
                c_out = direction_from_end(wc, sc); w_out = direction_from_end(ww, sw)
                if o_in is None or c_out is None or w_out is None:
                    continue
                # continuation of the occlusion beyond T points along -o_in
                a_c, a_w = angle(-o_in, c_out), angle(-o_in, w_out)
                kind = "cold" if a_c < a_w else "warm"
                tlon, tlat = g.Grid(cell_km=50.0).transformer().transform(*T, direction="INVERSE")
                types_rows.append((t, tlat, tlon, kind, a_c, a_w, np.hypot(*(S - T)) / 1000))
                # lows at the two ends of this occlusion
                m = lt == t
                if m.any():
                    idx = np.flatnonzero(m)
                    dT = np.hypot(lx[idx] - T[0], lyy[idx] - T[1]) / 1000
                    dS = np.hypot(lx[idx] - S[0], lyy[idx] - S[1]) / 1000
                    tri[idx[(dT <= TRIPLE_KM) & (dT < dS)]] = True
                    par = idx[(dS <= ATTACH_KM) & (dS <= dT)]
                    otype[par] = kind

    attached = (dend["OCFNT"] <= ATTACH_KM) & ~tri
    wrapped = (dend["OCFNT"] > ATTACH_KM) & (occ_near <= WRAP_NEAR_KM) & (occ_span >= WRAP_DEG) & ~tri
    open_ = (dend["COLD"] <= ATTACH_KM) & (dend["WARM"] <= ATTACH_KM) & (occ_near > WRAP_NEAR_KM) & ~tri
    dp = ly.dp12.to_numpy()
    masks = {
        "open": open_, "attached": attached,
        "attached_deepening": attached & (dp < 0), "attached_filling": attached & (dp > 0),
        "wrapped": wrapped,
        "occ_cold": (otype == "cold") & ~tri, "occ_warm": (otype == "warm") & ~tri,
        "triple": tri, "neither": ~tri & ~open_ & ~attached & ~wrapped,
    }
    for s in STRATA:
        n_low[s] += int(masks[s].sum())

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
            for s in STRATA:
                mm = masks[s][idx]
                if not mm.any():
                    continue
                X, Y = dx[:, mm], dy[:, mm]
                inr = (np.abs(X) < REACH_KM) & (np.abs(Y) < REACH_KM)
                if inr.any():
                    flat[s].append(((Y[inr] + REACH_KM) // BIN_KM).astype(np.int32) * NB
                                   + ((X[inr] + REACH_KM) // BIN_KM).astype(np.int32))
        for s in STRATA:
            if flat[s]:
                hist[s][ft] += np.bincount(np.concatenate(flat[s]), minlength=NB * NB).reshape(NB, NB)
    print(f"  {year}: open {open_.mean():.0%}  attached {attached.mean():.0%}  wrapped {wrapped.mean():.0%}  "
          f"triple {tri.mean():.1%}", flush=True)

np.savez_compressed(OUT / "composite_stages.npz",
                    **{f"{s}_{f}": hist[s][f] for s in STRATA for f in FRONTS},
                    n_low=np.array([n_low[s] for s in STRATA]), strata=np.array(STRATA),
                    bin_km=BIN_KM, reach_km=REACH_KM)
ty = pd.DataFrame(types_rows, columns=["t", "lat", "lon", "kind", "angle_cold", "angle_warm", "occl_km"])
ty.to_parquet(OUT / "occlusion_types.parquet")

print("\nlows per stage: " + "  ".join(f"{s} {n_low[s]:,}" for s in STRATA))
print(f"\n{len(ty):,} triple points typed: {100 * (ty.kind == 'cold').mean():.0f}% cold-type, "
      f"{100 * (ty.kind == 'warm').mean():.0f}% warm-type")
print(f"median angle to the front it continues: {np.median(np.minimum(ty.angle_cold, ty.angle_warm)):.0f} deg; "
      f"to the other: {np.median(np.maximum(ty.angle_cold, ty.angle_warm)):.0f} deg")
REG = [("NE Pacific, east side of basin", -165, -125), ("West, lee of the Divide", -125, -100),
       ("Plains and Midwest", -100, -85), ("Great Lakes and East", -85, -70),
       ("Western Atlantic, west side of basin", -70, -40)]
print("\nwarm-type share of occlusions by region (Unified Manual: warm on east sides of basins "
      "and lee of the Divide, cold on west sides of basins)")
for name, lo0, lo1 in REG:
    s = ty[ty.lon.between(lo0, lo1) & ty.lat.between(30, 65)]
    if len(s) > 50:
        print(f"  {name:<38s} n={len(s):6,}   warm-type {100 * (s.kind == 'warm').mean():4.0f}%")
