"""Real archived analyses drawn as charts: fronts with inferred pips, H and L.

Shared by the handbook's real-case figures. The archive stores only the lines,
so pip sides are inferred: cold and warm fronts from their motion between the
analyses three hours either side, occlusions by continuity through the triple
point, everything else by the usual convention.
"""
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
from common import ARGS, B, CENTERS, NO_ANCHORS, POINTS, g
import re
import front_symbols as fs

G50 = g.Grid(cell_km=50.0)
FWD = G50.transformer()
RULE = "#C9D2DC"
GRAT = re.compile(r"^\d+[NW]$")
C = {"COLD": "#1F4FB4", "WARM": "#C8281F", "OCFNT": "#7A2F9E", "STNRY": "#2F7A4A",
     "TROF": "#D2711A", "DRYLN": "#8B5A2B", "SQLN": "#A3231C", "TRPWV": "#D2711A",
     "LOW": "#C8281F", "HIGH": "#1F4FB4"}


def mapax(ax, extent, labels=False, anchors=False, lw=1.0):
    n_lines, n_coll = len(ax.lines), len(ax.collections)
    B._draw_map_furniture(ax, G50, ARGS if anchors else NO_ANCHORS, extent)
    # Furniture weights are tuned for a full-page map; scale them with panel
    # size so coastlines do not swamp the data on small multiples.
    if lw != 1.0:
        for ln in ax.lines[n_lines:]:
            ln.set_linewidth(ln.get_linewidth() * lw)
        for co in ax.collections[n_coll:]:
            co.set_linewidth(np.asarray(co.get_linewidth()) * lw)
    if not labels:
        for t in list(ax.texts):
            if GRAT.match(t.get_text().strip()):
                t.remove()
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(RULE)


def naive(t):
    t = pd.Timestamp(t)
    return t.tz_convert(None) if t.tzinfo else t


@lru_cache(maxsize=4)
def year_fronts(year):
    tab = pads.dataset(POINTS, format="parquet", partitioning="hive").to_table(
        columns=["valid_time", "ftype", "stage", "lat_e2", "lon_e2"],
        filter=pads.field("year") == int(year)).to_pandas()
    vt = pd.to_datetime(tab.valid_time, utc=True).dt.tz_localize(None)
    return tab.assign(t=vt)


@lru_cache(maxsize=1)
def all_centres():
    c = pads.dataset(CENTERS, format="parquet", partitioning="hive").to_table(
        columns=["valid_time", "res", "kind", "lat", "lon", "pressure_hpa"]).to_pandas()
    c = c[c.res == "HR"].copy()
    c["t"] = pd.to_datetime(c.valid_time, utc=True).dt.tz_localize(None)
    return c


def fronts_at(t):
    t = naive(t)
    tab = year_fronts(t.year)
    return tab[tab.t == t]


def centres_at(t):
    c = all_centres()
    return c[c.t == naive(t)]


PIPPED = ("COLD", "WARM", "OCFNT", "STNRY", "DRYLN")
# Where motion is too small to say, pips follow the usual convention: cold
# toward the east-southeast, warm toward the north, occluded toward the
# east-northeast, stationary triangles toward the (warm) south, dryline
# scallops toward the moist east. Compass bearings, degrees clockwise from north.
CONVENTION = {"COLD": 115, "WARM": 10, "OCFNT": 70, "STNRY": 180, "DRYLN": 90}
MOVED_KM = 15.0          # net 3-hour motion needed before trusting it


def neighbour_maps(t):
    """Fronts on the analyses three hours before and after t, keyed by offset."""
    t0 = naive(t)
    out = {}
    for h in (-3, 3):
        sub = fronts_at(t0 + pd.Timedelta(hours=h))
        lines = {}
        for la, lo, ft in zip(sub.lat_e2, sub.lon_e2, sub.ftype):
            x, y = FWD.transform(np.asarray(lo) / 100.0, np.asarray(la) / 100.0)
            if len(x) > 1:
                lines.setdefault(str(ft), []).append(fs.densify(x, y, 20_000.0))
        out[h] = lines
    return out


def pip_side(x, y, lat, lon, ft, nb, tally):
    """Which side of the line the pips go on, and whether motion decided it."""
    if ft in ("COLD", "WARM"):
        m = [fs.normal_motion(x, y, nb[3].get(ft), 300_000.0),
             fs.normal_motion(x, y, nb[-3].get(ft), 300_000.0)]
        moved = [v for v in (m[0], None if m[1] is None else -m[1]) if v is not None]
        if moved and abs(np.mean(moved)) > MOVED_KM * 1000:
            side = 1 if np.mean(moved) > 0 else -1
            tally["motion"] += 1
            tally["agree"] += side == conventional_side(x, y, lat, lon, ft)
            return side
    tally["convention"] += 1
    return conventional_side(x, y, lat, lon, ft)


def conventional_side(x, y, lat, lon, ft):
    k = len(lat) // 2
    br = np.radians(CONVENTION[ft])
    e = np.subtract(FWD.transform(lon[k] + 0.5, lat[k]), FWD.transform(lon[k], lat[k]))
    n = np.subtract(FWD.transform(lon[k], lat[k] + 0.5), FWD.transform(lon[k], lat[k]))
    e, n = e / np.hypot(*e), n / np.hypot(*n)
    v = np.sin(br) * e + np.cos(br) * n
    return fs.side_toward(x, y, v[0], v[1])


JOIN_M = 120_000.0       # an occlusion end this close to a front end is a triple point


def occlusion_side(x, y, joined):
    """Pips on the same side as the warm (else cold) front they continue into.

    Walking from the low along the occlusion, through the triple point and on
    along the warm or cold front, the pips stay on one side. Occlusions are
    not placed by their own motion: between maps they lengthen and wrap round
    the low, which reads as sideways motion that is not there.
    """
    for want in ("WARM", "COLD"):
        for fx_, fy_, fside, fft in joined:
            if fft != want:
                continue
            for oe in (0, -1):
                for fe in (0, -1):
                    if np.hypot(x[oe] - fx_[fe], y[oe] - fy_[fe]) < JOIN_M:
                        # occlusion in low-to-junction order ends at oe; the
                        # partner runs away from the junction starting at fe
                        s_partner = fside if fe == 0 else -fside
                        return s_partner if oe == -1 else -s_partner
    return None


def draw_line(ax, lat, lon, ft, nb=None, tally=None, joined=None, lw=1.8,
              spacing_km=150.0, size_km=62.0):
    x, y = FWD.transform(lon, lat)
    if ft in PIPPED:
        side = occlusion_side(x, y, joined) if ft == "OCFNT" and joined else None
        if side is not None:
            tally["joined"] += 1
        else:
            side = pip_side(x, y, lat, lon, ft, nb, tally)
        if joined is not None and ft in ("COLD", "WARM"):
            joined.append((x, y, side, ft))
        fs.draw_front(ax, x, y, ft, side=side, unit=1000.0, lw=lw,
                      spacing_km=spacing_km, size_km=size_km)
        return
    style = {"TROF": dict(color=C["TROF"], lw=1.5, ls=(0, (5, 3))),
             "SQLN": dict(color=C["SQLN"], lw=1.6, ls=(0, (6, 2, 1, 2, 1, 2))),
             "TRPWV": dict(color=C["TRPWV"], lw=1.2)}.get(ft)
    if style:
        ax.plot(x, y, zorder=6, solid_capstyle="round", **style)


def draw_chart(ax, fr, t, lw=1.8, spacing_km=150.0, size_km=62.0):
    """Every boundary on one analysis; returns how each pip side was decided."""
    nb, tally = neighbour_maps(t), {"motion": 0, "agree": 0, "convention": 0, "joined": 0}
    joined = []
    # cold and warm fronts first, so occlusions can take their side from them
    order = sorted(range(len(fr)), key=lambda i: str(fr.ftype.iloc[i]) == "OCFNT")
    for i in order:
        la, lo, ft = fr.lat_e2.iloc[i], fr.lon_e2.iloc[i], str(fr.ftype.iloc[i])
        draw_line(ax, np.asarray(la) / 100.0, np.asarray(lo) / 100.0, ft, nb, tally, joined,
                  lw=lw, spacing_km=spacing_km, size_km=size_km)
    return tally


def draw_centres(ax, cen, ext, size=13, dy=110_000):
    for _, c in cen.iterrows():
        x, y = FWD.transform(c.lon, c.lat)
        pad = 0.04 * (ext[1] - ext[0])
        if not (ext[0] + pad < x < ext[1] - pad and ext[2] + dy + pad < y < ext[3] - pad):
            continue                     # keep the letter and its pressure inside the frame
        col = C["LOW"] if c.kind == "L" else C["HIGH"]
        ax.text(x, y, c.kind, color=col, fontsize=size, fontweight="bold", ha="center",
                va="center", zorder=8)
        ax.text(x, y - dy, f"{c.pressure_hpa:.0f}", color=col, fontsize=size * 0.58,
                ha="center", va="top", zorder=8,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
