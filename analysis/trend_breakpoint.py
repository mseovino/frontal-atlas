"""When exactly did trough drawing step up? Monthly, deseasonalised.

Drawn trough length per synoptic analysis over the Lower 48, divided by the
2007-2016 mean for the same calendar month, so the summer trough maximum does
not masquerade as a step. Then a single mean-shift breakpoint search over
2017-2022: the split month that best separates 'before' from 'after'.
"""
import sys
import numpy as np, pandas as pd
from common import g, read_fronts

SYN = [0, 6, 12, 18]; BOX = (25, 50, -125, -67)
out = {}
for ft in ["TROF", "STNRY", "COLD"]:
    raw = read_fronts(ft)
    t = pd.to_datetime(raw.valid_time, utc=True)
    k = t.dt.hour.isin(SYN).to_numpy()
    raw, t = raw[k].reset_index(drop=True), t[k].reset_index(drop=True)
    mlat = np.array([np.mean(a) for a in raw.lat_e2]) / 100
    mlon = np.array([np.mean(b) for b in raw.lon_e2]) / 100
    ib = (mlat >= BOX[0]) & (mlat <= BOX[1]) & (mlon >= BOX[2]) & (mlon <= BOX[3])
    raw, t = raw[ib].reset_index(drop=True), t[ib].reset_index(drop=True)
    km = np.array([g.polyline_length_km(np.asarray(a) / 100, np.asarray(b) / 100)
                   for a, b in zip(raw.lat_e2, raw.lon_e2)])
    ym = t.dt.tz_localize(None).dt.to_period("M")
    out[ft] = pd.Series(km).groupby(ym.values).sum()

an = pd.to_datetime(read_fronts(["COLD", "TROF", "STNRY"], columns=("valid_time",)).valid_time
                    .drop_duplicates(), utc=True)
an = an[an.dt.hour.isin(SYN)]
den = an.dt.tz_localize(None).dt.to_period("M").value_counts().sort_index()

for ft, s in out.items():
    s = (s / den).dropna()
    s = s[(s.index >= pd.Period("2007-01", "M")) & (s.index <= pd.Period("2022-12", "M"))]
    clim = s[s.index.year <= 2016].groupby(s[s.index.year <= 2016].index.month).mean()
    anom = s / clim.reindex(s.index.month).to_numpy()
    seg = anom[anom.index.year >= 2017]
    best, bk = -1, None
    for i in range(6, len(seg) - 6):
        a, b = seg.iloc[:i], seg.iloc[i:]
        tstat = (b.mean() - a.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
        if tstat > best:
            best, bk = tstat, seg.index[i]
    print(f"\n{ft}: ratio to 2007-16 same-month mean")
    yr = anom.groupby(anom.index.year).mean()
    print("  by year: " + "  ".join(f"{y}:{v:.2f}" for y, v in yr.loc[2015:].items()))
    q = anom[(anom.index >= pd.Period("2019-07", "M")) & (anom.index <= pd.Period("2021-12", "M"))]
    print("  monthly Jul 2019 - Dec 2021:")
    for i in range(0, len(q), 6):
        print("    " + "  ".join(f"{p.strftime('%b%y')} {v:.2f}" for p, v in q.iloc[i:i+6].items()))
    print(f"  best single break (2017-22): starts {bk}  (t = {best:.1f})")
    if ft == "TROF":
        pre = anom[(anom.index >= pd.Period("2019-03", "M")) & (anom.index <= pd.Period("2020-02", "M"))].mean()
        post = anom[(anom.index >= pd.Period("2020-04", "M")) & (anom.index <= pd.Period("2021-03", "M"))].mean()
        print(f"  12 months before Mar 2020: {pre:.2f}   12 months after: {post:.2f}")
