"""A real analysis for each lifecycle composite: the chart that matches it best.

The composites say where fronts are drawn around lows at each stage, but they
are averages and do not look like a chart. Here, for every low of a stage
over or near the United States, the fronts on that one analysis are binned
exactly as the composite was (north up, 25 km bins, smoothed the same way)
and compared with the composite 150-900 km from the low by cosine
similarity. The best match (the deepest low, among charts within 0.02 of
the best score) is drawn as it was analysed, in the same frame as
the composite, so the two can be read side by side.

Only lows with no other low within 700 km are used, so the chart shows one
system; the match rewards a chart whose fronts sit where the composite's do
and penalises fronts where the composite has none.

Usage: python analysis/composite_examples.py
Needs outputs/low_stages.parquet and composite_stages.npz (composite_stages.py).
Writes outputs/composite_examples.csv and the handbook figure
hb_composite_examples.png.
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from common import OUT, REPO, fx, plt
import chart_fronts as cf
from composite_render import Composite, DISP, INK, MUTED

STAGES = [("open", "Open wave"), ("attached_deepening", "Occluding, still deepening"),
          ("attached_filling", "Occluded, filling"), ("wrapped", "Occlusion wrapped, not attached")]
FRONTS = ["COLD", "WARM", "OCFNT", "STNRY"]
LAT, LON = (28.0, 52.0), (-115.0, -60.0)       # the US and its coastal waters
ALONE_KM, MATCH_KM, SPACING_KM = 700.0, 900.0, 25.0
TIE = 0.02                                      # match scores this close count as equal
HB = REPO / "private" / "handbook" / "img"

comp = Composite(OUT / "composite_stages.npz")
reach, bin_ = comp.reach, comp.bin
nb = int(2 * reach / bin_)
# every stage has fronts drawn touching the low, so the first 150 km says
# nothing about which chart is typical and would favour small systems
disc = (comp.rad >= 150) & (comp.rad <= MATCH_KM)

st = pd.read_parquet(OUT / "low_stages.parquet")
st["t"] = pd.to_datetime(st.t)
# isolation from every analysed low at the same time, whatever its stage
xy = st.groupby("t")[["x", "y"]].apply(lambda d: d.to_numpy())
alone = np.array([np.sort(np.hypot(*(xy[t] - [x, y]).T))[1:2].min(initial=np.inf) / 1000 > ALONE_KM
                  for t, x, y in zip(st.t, st.x, st.y)])
cand = st[st.stage.isin([s for s, _ in STAGES]) & st.lat.between(*LAT) & st.lon.between(*LON) & alone]
print(f"{len(cand):,} isolated lows over or near the US:",
      cand.stage.value_counts().reindex([s for s, _ in STAGES]).to_dict())


def target(s):
    F = comp.fields(s)
    v = np.concatenate([F[f][disc] for f in FRONTS])
    return v / np.linalg.norm(v)


def case_vector(fr, x0, y0):
    """This analysis's fronts binned around the low as the composite was."""
    out = []
    for f in FRONTS:
        sub = fr[fr.ftype == f]
        h = np.zeros(nb * nb)
        if len(sub):
            pts = fx.explode(sub.assign(valid_time=sub.t))
            X, Y = cf.FWD.transform(pts.lon.to_numpy(), pts.lat.to_numpy())
            feat = pts.feature_id.to_numpy()
            dx, dy = [], []
            for k in np.unique(feat):
                m = feat == k
                d = cf.fs.densify(X[m], Y[m], SPACING_KM * 1000)
                dx.append((d[:, 0] - x0) / 1000); dy.append((d[:, 1] - y0) / 1000)
            dx, dy = np.concatenate(dx), np.concatenate(dy)
            inr = (np.abs(dx) < reach) & (np.abs(dy) < reach)
            idx = ((dy[inr] + reach) // bin_).astype(int) * nb + ((dx[inr] + reach) // bin_).astype(int)
            h = np.bincount(idx, minlength=nb * nb).astype(float)
        out.append(gaussian_filter(h.reshape(nb, nb), 2.0)[disc])
    v = np.concatenate(out)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


rows = []
for s, _ in STAGES:
    tv = target(s)
    c = cand[cand.stage == s]
    for year, cy in c.groupby(c.t.dt.year):
        tab = cf.year_fronts(year)
        tab = tab[tab.ftype.isin(FRONTS)]
        byt = dict(tuple(tab.groupby("t")))
        for r in cy.itertuples():
            fr = byt.get(r.t)
            if fr is None:
                continue
            rows.append((s, r.t, r.lat, r.lon, r.x, r.y, r.pressure_hpa, r.dp12,
                         float(case_vector(fr, r.x, r.y) @ tv)))
    print(f"  scored {s}", flush=True)
sc = pd.DataFrame(rows, columns=["stage", "t", "lat", "lon", "x", "y", "pressure_hpa", "dp12", "match"])
sc.to_csv(OUT / "composite_examples.csv", index=False)
# Scores this close are a tie; among them, show the deepest low.
near = sc[sc.match >= sc.groupby("stage").match.transform("max") - TIE]
best = near.sort_values("pressure_hpa").groupby("stage").head(1).set_index("stage")
for s, _ in STAGES:
    top = sc[sc.stage == s].nlargest(5, "match")
    print(f"\n{s}: median match {sc[sc.stage == s].match.median():.2f}; best")
    for r in top.itertuples():
        print(f"   {r.t:%Y-%m-%d %HZ}  {r.lat:5.1f}N {abs(r.lon):5.1f}W  {r.pressure_hpa:4.0f} hPa  match {r.match:.2f}")

# ------------------------------------------------------------------ figure
mpl.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "text.color": INK})
fig, axes = plt.subplots(2, 2, figsize=(11.2, 11.4), gridspec_kw={"wspace": 0.1, "hspace": 0.16})
R = DISP * 1000
th = np.linspace(0, 2 * np.pi, 361)
for ax, (s, title) in zip(axes.ravel(), STAGES):
    r = best.loc[s]
    ext = (r.x - R, r.x + R, r.y - R, r.y + R)
    cf.mapax(ax, ext, lw=0.8)
    tally = cf.draw_chart(ax, cf.fronts_at(r.t), r.t, lw=2.0, spacing_km=120.0, size_km=52.0)
    for rr in (250, 500, 750):
        ax.plot(r.x + rr * 1000 * np.cos(th), r.y + rr * 1000 * np.sin(th), color="#9AA5B1",
                lw=0.6, ls=(0, (3, 3)), zorder=3)
    cf.draw_centres(ax, cf.centres_at(r.t), ext, size=14, dy=75_000)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"{title}", loc="left", fontsize=9.5, fontweight="semibold")
    ax.set_title(f"{r.t:%HZ %d %b %Y}  ·  {r.pressure_hpa:.0f} hPa", loc="right",
                 fontsize=8.5, color=MUTED)
    print(f"{s}: {r.t:%Y-%m-%d %HZ} pips {tally}")
h = [Line2D([], [], color=cf.C[f], lw=2.2, label=lab) for f, lab in
     [("COLD", "Cold"), ("WARM", "Warm"), ("OCFNT", "Occluded")]]
h += [Line2D([], [], color=cf.C["COLD"], lw=2.2, ls=(0, (4, 4)), gapcolor=cf.C["WARM"], label="Stationary"),
      Line2D([], [], color=cf.C["TROF"], lw=1.6, ls=(0, (5, 3)), label="Trough"),
      Line2D([], [], color="#9AA5B1", lw=0.8, ls=(0, (3, 3)), label="250, 500, 750 km from the low")]
fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.06))
for d in (OUT, HB):
    fig.savefig(d / "hb_composite_examples.png", dpi=150, bbox_inches="tight", pad_inches=0.06,
                facecolor="white")
print("wrote hb_composite_examples.png")
