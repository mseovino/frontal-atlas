"""Drawing-consistency monitor: does the analysis product change when nothing
in the atmosphere says it should?

The trough doubling in this archive went unnoticed for years because nobody
was measuring it. This script measures the handful of quantities that would
have caught it, every month, and flags any month that departs from the
previous three years:

  length_<TYPE>   drawn length per synoptic map over the Lower 48, divided by
                  the same calendar month's mean over the reference years, so
                  the seasonal cycle is removed
  pieces_<TYPE>   median length of one drawn feature, in km. A fall means the
                  same boundaries are being drawn in more, shorter pieces
  int_syn_<TYPE>  features per intermediate map divided by features per
                  synoptic map, counted by centroid in WPC's interior area.
                  About 1.0 for classical fronts; about 1.2-1.25 for
                  stationary fronts and troughs in 2006-2022
  stage_share     fraction of fronts carrying a developing or weakening symbol
  diss_form       weakening : developing ratio

A flag means "look at this", not "this is wrong". Real weather can move any
of these for a month; a run of flags in one direction is a practice change.

Usage:  python analysis/consistency_monitor.py [--ref 2007 2016] [--window 36]
Writes outputs/consistency_monthly.csv and outputs/consistency_monitor.png.
"""
import argparse

import numpy as np
import pandas as pd
from pyproj import Geod
from common import OUT, MONTHS, plt, read_fronts

TYPES = ["COLD", "WARM", "STNRY", "OCFNT", "TROF"]
SYN, INTER = [0, 6, 12, 18], [3, 9, 15, 21]
L48 = (25.0, 50.0, -125.0, -67.0)      # length trends
INTERIOR = (33.0, 48.0, -115.0, -80.0)  # WPC's own area at every hour
GEOD = Geod(ellps="WGS84")


def per_feature(df):
    """Centroid, great-circle length and vertex count for every polyline, vectorised."""
    n = df.lat_e2.map(len).to_numpy()
    lat = np.concatenate(df.lat_e2.to_numpy()).astype(float) / 100.0
    lon = np.concatenate(df.lon_e2.to_numpy()).astype(float) / 100.0
    fid = np.repeat(np.arange(len(df)), n)
    same = fid[1:] == fid[:-1]
    _, _, d = GEOD.inv(lon[:-1][same], lat[:-1][same], lon[1:][same], lat[1:][same])
    km = np.bincount(fid[:-1][same], weights=d / 1000.0, minlength=len(df))
    return (np.bincount(fid, weights=lat, minlength=len(df)) / n,
            np.bincount(fid, weights=lon, minlength=len(df)) / n, km)


def inside(la, lo, box):
    return (la >= box[0]) & (la <= box[1]) & (lo >= box[2]) & (lo <= box[3])


def main(ref, window):
    raw = read_fronts(TYPES)
    t = pd.to_datetime(raw.valid_time, utc=True).dt.tz_localize(None)
    raw["ym"] = t.dt.to_period("M").to_numpy()
    raw["hour"] = t.dt.hour.to_numpy()
    raw["stage"] = raw.stage.astype(object)
    raw["ftype"] = raw.ftype.astype(str)
    raw["mlat"], raw["mlon"], raw["km"] = per_feature(raw)
    print(f"{len(raw):,} features measured", flush=True)

    an = pd.DataFrame({"ym": raw.ym, "hour": raw.hour, "t": t}).drop_duplicates("t")
    n_syn = an[an.hour.isin(SYN)].groupby("ym").size()
    n_int = an[an.hour.isin(INTER)].groupby("ym").size()

    cols = {}
    syn = raw[raw.hour.isin(SYN)]
    l48 = syn[inside(syn.mlat, syn.mlon, L48)]
    inter = raw[inside(raw.mlat, raw.mlon, INTERIOR)]
    for ft in TYPES:
        s = l48[l48.ftype == ft]
        cols[f"length_{ft}"] = s.groupby("ym").km.sum() / n_syn
        cols[f"pieces_{ft}"] = s.groupby("ym").km.median()
        i = inter[inter.ftype == ft]
        per_int = i[i.hour.isin(INTER)].groupby("ym").size() / n_int
        per_syn = i[i.hour.isin(SYN)].groupby("ym").size() / n_syn
        cols[f"int_syn_{ft}"] = per_int / per_syn
    fr = syn[syn.ftype.isin(["COLD", "WARM", "STNRY", "OCFNT"])]
    tagged = fr.stage.notna()
    cols["stage_share"] = tagged.groupby(fr.ym).mean()
    cols["diss_form"] = ((fr.stage == "DISS").groupby(fr.ym).sum()
                         / (fr.stage == "FORM").groupby(fr.ym).sum().replace(0, np.nan))
    m = pd.DataFrame(cols).sort_index()
    m = m[(m.index >= pd.Period("2007-01", "M"))]

    # Remove the seasonal cycle from drawn length: ratio to the reference
    # years' mean for the same calendar month.
    month = np.array([p.month for p in m.index])
    in_ref = np.array([ref[0] <= p.year <= ref[1] for p in m.index])
    for ft in TYPES:
        c = f"length_{ft}"
        clim = pd.Series(m[c].to_numpy()[in_ref]).groupby(month[in_ref]).mean()
        m[c] = m[c].to_numpy() / clim.reindex(month).to_numpy()

    # Flag: departure from the trailing window, in robust standard deviations.
    flags = pd.DataFrame(index=m.index)
    for c in m.columns:
        med = m[c].shift(1).rolling(window, min_periods=window // 2).median()
        mad = (m[c].shift(1) - med).abs().rolling(window, min_periods=window // 2).median()
        flags[c] = (m[c] - med) / (1.4826 * mad)
    m.to_csv(OUT / "consistency_monthly.csv")

    last = m.index[-1]
    print(f"\nlatest month {last}: robust z against the previous {window} months")
    for c in m.columns:
        z = flags[c].iloc[-1]
        mark = "  <-- look at this" if abs(z) > 3 else ""
        print(f"  {c:16s} {m[c].iloc[-1]:8.3f}   z {z:+5.1f}{mark}")
    runs = (flags.abs() > 3).rolling(6).sum()
    print("\nmonths with 3+ flags in the preceding six, per metric:")
    for c in m.columns:
        hot = runs.index[runs[c] >= 3]
        if len(hot):
            print(f"  {c:16s} first {hot[0]}, {len(hot)} months")

    # Drift check. A trailing window cannot see a slow ramp: the baseline
    # climbs with it, which is exactly how the trough doubling would have
    # slipped past. So also compare the 12-month mean against the fixed
    # reference years, in units of the reference years' own spread of
    # 12-month means, and report the first month it stays beyond 3 of them.
    roll = m.rolling(12).mean()
    ref_roll = roll[in_ref]
    drift = (roll - ref_roll.mean()) / ref_roll.std()
    print(f"\ndrift of the 12-month mean from {ref[0]}-{ref[1]}, in reference SDs"
          " (first month beyond 3 for 6 months running)")
    for c in m.columns:
        beyond = (drift[c].abs() > 3).rolling(6).sum() >= 6
        first = drift.index[beyond.to_numpy()]
        note = f"first {first[0]}" if len(first) else "never"
        print(f"  {c:16s} latest {drift[c].iloc[-1]:+6.1f}   {note}")
    drift.to_csv(OUT / "consistency_drift.csv")

    show = ["length_TROF", "length_STNRY", "length_COLD", "pieces_COLD",
            "int_syn_STNRY", "int_syn_TROF", "stage_share", "diss_form"]
    fig, axes = plt.subplots(4, 2, figsize=(12, 10.5), sharex=True)
    x = m.index.to_timestamp()
    for ax, c in zip(axes.ravel(), show):
        ax.plot(x, m[c], color="#9AA5B1", lw=0.8)
        ax.plot(x, m[c].rolling(12, center=True).mean(), color="#15202B", lw=1.8)
        hot = flags[c].abs() > 3
        ax.scatter(x[hot], m[c][hot], s=10, color="#C8281F", zorder=3)
        ax.set_title(c, loc="left", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E6EAEE", lw=0.7)
    fig.suptitle("Drawing consistency by month (grey), 12-month mean (black), "
                 f"flags beyond 3 robust SD of the previous {window} months (red)",
                 fontsize=10.5, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "consistency_monitor.png", dpi=140)
    print("\nwrote consistency_monthly.csv and consistency_monitor.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs=2, type=int, default=[2007, 2016],
                    help="reference years for the seasonal cycle of drawn length")
    ap.add_argument("--window", type=int, default=36, help="trailing months for flags")
    a = ap.parse_args()
    main(tuple(a.ref), a.window)
