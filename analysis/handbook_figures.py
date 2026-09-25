"""Second-pass handbook figures, in one consistent style.

Rules applied to every figure:
  * Appendix A colours (cold royal blue, warm red, occluded purple, trough
    orange, dryline brown, squall line red); stationary takes green because a
    filled field cannot alternate red and blue, and the page says so.
  * One font (Segoe UI), one set of sizes, no in-image titles -- the page
    captions carry the words, so the images carry only data and panel labels.
  * Maps cropped to where the analysis actually draws things, graticule labels
    kept only on single large maps, horizontal colour bars under the map.
"""
import pickle
import re
import sys


import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
from common import OUT, ARGS, NO_ANCHORS, REPO, B, box_extent, fx, g, plt
import chart_fronts as cf

# Handbook figures go beside the (private) handbook source that embeds them.
IMG = REPO / "private" / "handbook" / "img"
IMG.mkdir(parents=True, exist_ok=True)

INK, MUTED, RULE = "#15202B", "#566374", "#C9D2DC"
mpl.rcParams.update({
    "font.family": "Segoe UI", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": RULE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10, "axes.titleweight": "semibold",
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06, "savefig.facecolor": "white",
})

C = {"COLD": "#1F4FB4", "WARM": "#C8281F", "OCFNT": "#7A2F9E", "STNRY": "#2F7A4A",
     "TROF": "#D2711A", "DRYLN": "#8B5A2B", "SQLN": "#A3231C", "TRPWV": "#D2711A",
     "LOW": "#C8281F", "HIGH": "#1F4FB4"}
NAME = {"COLD": "Cold", "WARM": "Warm", "OCFNT": "Occluded", "STNRY": "Stationary",
        "TROF": "Surface trough", "DRYLN": "Dryline", "SQLN": "Squall line"}
FRONTS = ["COLD", "WARM", "STNRY", "OCFNT"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

G50, G100, G200 = g.Grid(cell_km=50.0), g.Grid(cell_km=100.0), g.Grid(cell_km=200.0)
FULL = (G50.x_min, G50.x_max, G50.y_min, G50.y_max)
NA = (-4.35e6, 4.25e6, -2.9e6, 3.95e6)          # where fronts are drawn
FWD = G50.transformer()
GRAT = re.compile(r"^\d+[NW]$")

z = np.load(OUT / "handbook_atlas.npz")
with open(OUT / "handbook_data2.pkl", "rb") as fh:
    D = pickle.load(fh)


def ramp(hexcol, lo=0.03, hi=0.78):
    base = LinearSegmentedColormap.from_list("b", ["#FFFFFF", hexcol, "#0B0B0B"])
    return LinearSegmentedColormap.from_list("r", [base(x) for x in np.linspace(lo, hi, 16)])


def mapax(ax, extent=NA, **kw):
    cf.mapax(ax, extent, **kw)


def pct(vmax):
    return FuncFormatter(lambda v, p: f"{100*v:.1f}%" if vmax < 0.05 else f"{100*v:.0f}%")


def hbar(fig, m, ax, vmax, label=None):
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02,
                      aspect=38, format=pct(vmax))
    cb.outline.set_edgecolor(RULE); cb.ax.tick_params(labelsize=7.5, length=2)
    if label:
        cb.set_label(label, fontsize=8, color=MUTED)
    return cb


def field(ax, arr, cmap, vmax, extent_grid=FULL):
    return ax.imshow(np.where(arr > 0, arr, np.nan), origin="lower", extent=extent_grid,
                     cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", zorder=1)


def ptitle(ax, text, color=INK):
    ax.set_title(text, loc="left", color=color, pad=4)


def save(fig, name):
    fig.savefig(IMG / name)
    plt.close(fig)
    print("wrote", name)


def vmax_of(*arrs, q=99.5):
    v = np.concatenate([a.ravel() for a in arrs])
    return float(np.percentile(v[v > 0], q))


# ============================================================ 1. annual atlas
fig, axes = plt.subplots(2, 2, figsize=(11, 9.8), gridspec_kw={"hspace": 0.16, "wspace": 0.04})
for ax, ft in zip(axes.ravel(), FRONTS):
    f = z[f"{ft}_ANN"]; vm = vmax_of(f)
    m = field(ax, f, ramp(C[ft]), vm); mapax(ax, lw=0.75)
    ptitle(ax, f"{NAME[ft]} fronts", C[ft]); hbar(fig, m, ax, vm)
save(fig, "hb_atlas_annual.png")

# ========================================================== 2. seasonal atlas
fig, axes = plt.subplots(4, 2, figsize=(9.6, 16.2), gridspec_kw={"hspace": 0.22, "wspace": 0.03})
for i, ft in enumerate(FRONTS):
    vm = vmax_of(z[f"{ft}_DJF"], z[f"{ft}_JJA"])
    for j, (s, lab) in enumerate([("DJF", "December-February"), ("JJA", "June-August")]):
        m = field(axes[i, j], z[f"{ft}_{s}"], ramp(C[ft]), vm); mapax(axes[i, j], lw=0.65)
        ptitle(axes[i, j], f"{NAME[ft]}  \u00b7  {lab}", C[ft])
    hbar(fig, m, axes[i, :], vm)
save(fig, "hb_atlas_seasonal.png")

# =========================================================== 3. month by month
cold = D["month_COLD"]; vm = vmax_of(cold)
fig, axes = plt.subplots(3, 4, figsize=(12.5, 8.6), gridspec_kw={"hspace": 0.1, "wspace": 0.03})
for k, ax in enumerate(axes.ravel()):
    m = field(ax, cold[k], ramp(C["COLD"]), vm); mapax(ax, lw=0.4)
    ptitle(ax, MONTHS[k])
hbar(fig, m, axes, vm, "cold front drawn through the cell, % of synoptic analyses")
save(fig, "hb_months_cold.png")

# ================================================================== 4. troughs
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={"wspace": 0.03})
vm = vmax_of(z["TROF_DJF"], z["TROF_JJA"])
for ax, s, lab in zip(axes, ["DJF", "JJA"], ["December-February", "June-August"]):
    m = field(ax, z[f"TROF_{s}"], ramp(C["TROF"]), vm); mapax(ax, anchors=True)
    ptitle(ax, lab, C["TROF"])
hbar(fig, m, axes, vm)
save(fig, "hb_troughs.png")

# ============================================================ 5. dominant type
n_ann = int(z["den_ANN"])


def dominant(types, name):
    stack = np.stack([z[f"{t}_ANN"] * n_ann for t in types])
    support = stack.sum(axis=0) >= 100
    srt = np.sort(stack, axis=0)
    decided = support & ((srt[-1] - srt[-2]) > 2.0 * np.sqrt(srt[-1] + srt[-2]))
    win = np.where(decided, stack.argmax(axis=0), np.nan)
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.imshow(np.where(support & ~decided, 1.0, np.nan), origin="lower", extent=FULL,
              cmap=ListedColormap(["#DADDE1"]), interpolation="nearest", zorder=1)
    cm = ListedColormap([C[t] for t in types])
    ax.imshow(win, origin="lower", extent=FULL, cmap=cm,
              norm=BoundaryNorm(np.arange(-0.5, len(types)), cm.N),
              interpolation="nearest", zorder=1, alpha=0.9)
    mapax(ax, labels=True)
    h = [Patch(facecolor=C[t], label=NAME[t] + ("" if t == "TROF" else " front")) for t in types]
    h.append(Patch(facecolor="#DADDE1", label="no clear leader"))
    ax.legend(handles=h, loc="lower left", fontsize=8, framealpha=0.96, edgecolor=RULE)
    save(fig, name)


dominant(FRONTS, "hb_dominant_fronts.png")
dominant(FRONTS + ["TROF"], "hb_dominant_all.png")

# ======================================================== 6. highs and lows
fig, axes = plt.subplots(2, 2, figsize=(11, 9.9), gridspec_kw={"hspace": 0.3, "wspace": 0.04})
for i, (k, lab, col) in enumerate([("L", "Lows", C["LOW"]), ("H", "Highs", C["HIGH"])]):
    a1 = gaussian_filter(D[f"centres_{k}_DJF"], 0.8); a2 = gaussian_filter(D[f"centres_{k}_JJA"], 0.8)
    vm = vmax_of(a1, a2, q=99.7)
    for j, (arr, s) in enumerate([(a1, "December-February"), (a2, "June-August")]):
        m = axes[i, j].imshow(np.where(arr > 1e-4, arr, np.nan), origin="lower",
                              extent=(G100.x_min, G100.x_max, G100.y_min, G100.y_max),
                              cmap=ramp(col), vmin=0, vmax=vm, interpolation="bilinear", zorder=1)
        mapax(axes[i, j], lw=0.75); ptitle(axes[i, j], f"{lab}  \u00b7  {s}", col)
    hbar(fig, m, axes[i, :], vm, f"{lab.lower()} analysed in the 100 km cell, % of synoptic analyses")
save(fig, "hb_centres.png")

# ================================================= 7. central-pressure check
fig, axes = plt.subplots(1, 2, figsize=(12, 5.9), gridspec_kw={"wspace": 0.04})
ext200 = (G200.x_min, G200.x_max, G200.y_min, G200.y_max)
lvl_l = np.arange(960, 1012, 4)
lvl_h = np.arange(1024, 1064, 4)
for ax, arr, lv, cmap, lab in [
        (axes[0], D["low_p01"], lvl_l, "magma", "Deepest 1% of lows analysed in the cell (hPa)"),
        (axes[1], D["high_p99"], lvl_h, "viridis", "Strongest 1% of highs analysed in the cell (hPa)")]:
    cm = plt.get_cmap(cmap, len(lv) + 1)
    m = ax.imshow(arr, origin="lower", extent=ext200, cmap=cm,
                  norm=BoundaryNorm(lv, cm.N, extend="both"), interpolation="nearest", zorder=1)
    mapax(ax, anchors=True); ptitle(ax, lab)
    cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=38)
    cb.outline.set_edgecolor(RULE); cb.ax.tick_params(labelsize=7.5, length=2)
save(fig, "hb_pressure_check.png")

# ============================================================ 8. composite
zc = np.load(OUT / "norwegian_neighbour.npz", allow_pickle=True)
STR = [str(s) for s in zc["strata"]]; NLOW = dict(zip(STR, zc["n_low"]))
BIN, REACH = float(zc["bin_km"]), float(zc["reach_km"])
NB = int(2 * REACH / BIN); ax_ = -REACH + (np.arange(NB) + 0.5) * BIN
XX, YY = np.meshgrid(ax_, ax_); RAD = np.hypot(XX, YY)
RIDX = np.clip((RAD / BIN).astype(int), 0, NB); EXTC = (-REACH, REACH, -REACH, REACH)
CFR = ["COLD", "WARM", "OCFNT", "STNRY"]


def anomaly(stratum, ft):
    a = gaussian_filter(zc[f"{stratum}_{ft}"].astype(float) / max(NLOW[stratum], 1), 1.6)
    tot = np.bincount(RIDX.ravel(), weights=a.ravel(), minlength=NB + 1)
    cnt = np.bincount(RIDX.ravel(), minlength=NB + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a / (tot / cnt)[RIDX]
    disc = RAD <= 900.0
    ok = disc & (a >= np.nanpercentile(a[disc], 65)) & np.isfinite(r)
    return np.nan_to_num(np.where(ok, r, np.nan))


def compax(ax, stratum, lw=1.6, rings=True, lab=True):
    for ft in CFR:
        r = anomaly(stratum, ft)
        ax.contourf(r, levels=[1.5, 1e9], extent=EXTC, colors=[C[ft]], alpha=0.28, zorder=2)
        ax.contour(r, levels=[1.5], extent=EXTC, colors=[C[ft]], linewidths=lw, zorder=3)
    th = np.linspace(0, 2 * np.pi, 361)
    for rr in (250, 500, 750):
        ax.plot(rr * np.cos(th), rr * np.sin(th), color="#9AA5B1", lw=0.6, ls=(0, (3, 3)), zorder=4)
        if lab:
            ax.text(rr * 0.72, -rr * 0.72, f"{rr} km", fontsize=7, color=MUTED, ha="left", va="top", zorder=5)
    ax.axhline(0, color="#D5DBE2", lw=0.6, zorder=1); ax.axvline(0, color="#D5DBE2", lw=0.6, zorder=1)
    ax.plot(0, 0, "o", ms=11, mfc="white", mec=C["LOW"], mew=1.6, zorder=9)
    ax.text(0, 0, "L", ha="center", va="center", fontsize=9, fontweight="bold", color=C["LOW"], zorder=10)
    ax.set_xlim(-950, 950); ax.set_ylim(-950, 950); ax.set_aspect("equal")
    ax.set_xticks([-750, -500, -250, 0, 250, 500, 750]); ax.set_yticks([-750, -500, -250, 0, 250, 500, 750])
    ax.tick_params(labelsize=7, length=2)
    for s in ax.spines.values():
        s.set_edgecolor(RULE)


# The lifecycle composite (hb_composite.png) is drawn by composite_render.py.
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), gridspec_kw={"wspace": 0.12})
for ax, s, lab in zip(axes, ["isolated", "primary", "secondary"],
                      ["No other low within 750 km", "Deepest low within 750 km",
                       "A deeper low within 750 km"]):
    compax(ax, s, lw=1.3, lab=(s == "isolated"))
    ptitle(ax, f"{lab}  ({NLOW[s]:,})")
axes[0].legend(handles=[Line2D([], [], color=C[f], lw=3, label=NAME[f]) for f in CFR],
               loc="lower left", fontsize=7.5, framealpha=0.96, edgecolor=RULE)
save(fig, "hb_neighbours.png")

# ============================================================== 9. real cases


fig, axes = plt.subplots(2, 2, figsize=(11, 10.4), gridspec_kw={"hspace": 0.12, "wspace": 0.04})
R = 1700_000.0
for ax, case in zip(axes.ravel(), D["cases"]):
    cx, cy = FWD.transform(case["lon"], case["lat"])
    ext = (cx - R, cx + R, cy - R * 0.9, cy + R * 0.9)
    mapax(ax, extent=ext, lw=0.8)
    tally = cf.draw_chart(ax, case["fronts"], case["time"])
    print(f"{case['time']:%Y-%m-%d} pips: {tally}")
    cf.draw_centres(ax, case["centres"], ext)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ptitle(ax, f"{case['time']:%d %b %Y}  12Z   \u00b7   {case['p']:.0f} hPa low")
h = [Line2D([], [], color=C["COLD"], lw=2.2, label="Cold"),
     Line2D([], [], color=C["WARM"], lw=2.2, label="Warm"),
     Line2D([], [], color=C["OCFNT"], lw=2.2, label="Occluded"),
     Line2D([], [], color=C["COLD"], lw=2.2, ls=(0, (4, 4)), gapcolor=C["WARM"], label="Stationary"),
     Line2D([], [], color=C["TROF"], lw=1.6, ls=(0, (5, 3)), label="Trough"),
     Line2D([], [], color=C["DRYLN"], lw=1.8, label="Dryline"),
     Line2D([], [], color=C["SQLN"], lw=1.6, ls=(0, (6, 2, 1, 2, 1, 2)), label="Squall line")]
fig.legend(handles=h, loc="lower center", ncol=7, fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.5, 0.03))
save(fig, "hb_cases.png")

# ============================================================ 10. drylines
dl = D["dryline"]
LATS = [31.0, 33.0, 35.0, 37.0]
BR = ["#4E3219", "#7A4E27", "#A87543", "#CFA27A"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, gridspec_kw={"wspace": 0.06})
for cut, col in zip(LATS, BR):
    s = dl[dl.lat == cut]
    for ax, key, xs in [(axes[0], "month", range(1, 13)), (axes[1], "hour", [0, 3, 6, 9, 12, 15, 18, 21])]:
        sub = s if key == "month" else s[s.month.isin([4, 5, 6])]
        med, lo, hi, xx = [], [], [], []
        for v in xs:
            q = sub[sub[key] == v].lon
            if len(q) >= 25:
                xx.append(v); med.append(q.median()); lo.append(q.quantile(.25)); hi.append(q.quantile(.75))
        ax.fill_between(xx, lo, hi, color=col, alpha=0.13, lw=0)
        ax.plot(xx, med, color=col, lw=1.9, marker="o", ms=3.2, label=f"{cut:.0f}\u00b0N")
axes[0].set_xticks(range(1, 13)); axes[0].set_xticklabels([m[0] for m in MONTHS])
axes[1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
axes[1].set_xticklabels([f"{h:02d}Z" for h in [0, 3, 6, 9, 12, 15, 18, 21]])
ptitle(axes[0], "By month, all hours"); ptitle(axes[1], "By analysis hour, April-June")
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{abs(v):.0f}\u00b0W"))
axes[0].legend(fontsize=7.5, frameon=False, ncol=4, loc="upper center")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6EAEE", lw=0.7); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
save(fig, "hb_dryline_position.png")

ext = box_extent(-107, -94, 28.5, 39.5, cell_km=50.0, pad_km=60)
fig, ax = plt.subplots(figsize=(6.2, 6.2))
mapax(ax, extent=ext, labels=True, anchors=True)
mc = LinearSegmentedColormap.from_list("m", ["#E3C9A8", "#8B5A2B", "#3A2210"])
show = [3, 4, 5, 6, 10]
for i, mth in enumerate(show):
    pts = [(cut, dl[(dl.lat == cut) & (dl.month == mth)].lon.median()) for cut in LATS]
    pts = [(a, b) for a, b in pts if np.isfinite(b)]
    x, y = FWD.transform([p[1] for p in pts], [p[0] for p in pts])
    col = mc(i / (len(show) - 1)) if mth != 10 else "#6B7785"
    ax.plot(x, y, color=col, lw=2.6, marker="o", ms=4.5, zorder=7,
            ls="-" if mth != 10 else (0, (4, 2)), label=MONTHS[mth - 1])
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.legend(fontsize=8, loc="lower left", title="median position", title_fontsize=8,
          framealpha=0.96, edgecolor=RULE)
save(fig, "hb_dryline_map.png")

# ========================================================== 11. squall lines
ext = box_extent(-106, -68, 25, 49, cell_km=100.0, pad_km=100)
fig = plt.figure(figsize=(11.5, 5.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], hspace=0.55, wspace=0.12)
ax = fig.add_subplot(gs[:, 0])
sq = D["sqln_grid"]; vm = float(np.percentile(sq[sq > 0], 99))
m = ax.imshow(np.where(sq > 0, sq, np.nan), origin="lower",
              extent=(G100.x_min, G100.x_max, G100.y_min, G100.y_max),
              cmap=ramp(C["SQLN"]), vmin=0, vmax=vm, interpolation="nearest", zorder=1)
mapax(ax, extent=ext, anchors=True)
cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=38)
cb.set_label("squall lines drawn through the 100 km cell, 2006-2022", fontsize=8, color=MUTED)
cb.outline.set_edgecolor(RULE); cb.ax.tick_params(labelsize=7.5, length=2)
a1 = fig.add_subplot(gs[0, 1]); a2 = fig.add_subplot(gs[1, 1])
a1.bar(range(1, 13), D["sqln_month"], color=C["SQLN"], width=0.72)
a1.set_xticks(range(1, 13)); a1.set_xticklabels([m[0] for m in MONTHS])
ptitle(a1, "By month")
a2.bar(range(8), D["sqln_hour"], color=C["SQLN"], width=0.72)
a2.set_xticks(range(8)); a2.set_xticklabels([f"{h:02d}Z" for h in [0, 3, 6, 9, 12, 15, 18, 21]])
ptitle(a2, "By analysis hour")
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8)
    a.grid(axis="y", color="#E6EAEE", lw=0.7); a.set_axisbelow(True)
save(fig, "hb_sqln.png")

print("\nlow quantiles  (0.1,1,5,50,95,99):", np.round(D["low_quantiles"], 0))
print("high quantiles (1,5,50,95,99,99.9):", np.round(D["high_quantiles"], 0))
