"""Figures and page for "what to expect this month".

One figure per month (six panels on scales shared across all twelve months, so
months compare directly) and one pressure-range figure per season. The page
embeds them and switches with a month picker. Reads outputs/expect_data.npz
from expect_data.py; writes private/expect/.
"""
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from pyproj import Transformer
from common import ARGS, NO_ANCHORS, OUT, REPO, B, g, plt

DEST = REPO / "private" / "expect"
IMG = DEST / "img"
IMG.mkdir(parents=True, exist_ok=True)
Z = np.load(OUT / "expect_data.npz")

INK, MUTED, RULE = "#15202B", "#566374", "#C9D2DC"
mpl.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": RULE, "savefig.facecolor": "white"})
C = {"COLD": "#1F4FB4", "WARM": "#C8281F", "STNRY": "#2F7A4A", "OCFNT": "#7A2F9E",
     "TROF": "#D2711A", "DRYLN": "#8B5A2B", "L": "#C8281F", "H": "#1F4FB4"}
TITLE = {"COLD": "Cold fronts", "WARM": "Warm fronts", "STNRY": "Stationary fronts",
         "OCFNT": "Occluded fronts", "TROF": "Troughs, with median dryline", "LH": "Lows (red) and highs (blue)"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
SEASON_OF = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
             6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASON_NAME = {"DJF": "December-February", "MAM": "March-May", "JJA": "June-August", "SON": "September-November"}

G50, G100, G200 = g.Grid(cell_km=50.0), g.Grid(cell_km=100.0), g.Grid(cell_km=200.0)
E50 = (G50.x_min, G50.x_max, G50.y_min, G50.y_max)
E100 = (G100.x_min, G100.x_max, G100.y_min, G100.y_max)
E200 = (G200.x_min, G200.x_max, G200.y_min, G200.y_max)
NA = (-3.9e6, 3.5e6, -2.7e6, 2.9e6)       # the part of the domain WPC draws most
FWD = G50.transformer()
GRAT = re.compile(r"^\d+[NW]$")


def ramp(hexcol):
    base = LinearSegmentedColormap.from_list("b", ["#FFFFFF", hexcol, "#0B0B0B"])
    return LinearSegmentedColormap.from_list("r", [base(x) for x in np.linspace(0.03, 0.78, 16)])


def mapax(ax, anchors=False, lw=0.6):
    n0 = len(ax.lines); c0 = len(ax.collections)
    B._draw_map_furniture(ax, G50, ARGS if anchors else NO_ANCHORS, NA)
    for ln in ax.lines[n0:]:
        ln.set_linewidth(ln.get_linewidth() * lw)
    for co in ax.collections[c0:]:
        co.set_linewidth(np.asarray(co.get_linewidth()) * lw)
    for t in list(ax.texts):
        if GRAT.match(t.get_text().strip()):
            t.remove()
    ax.set_xlim(NA[0], NA[1]); ax.set_ylim(NA[2], NA[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(RULE)


def vmax(key, q=99.5):
    v = Z[key].ravel(); v = v[v > 0]
    return float(np.percentile(v, q))


def pctfmt(v, p):
    return f"{100 * v:.1f}%"


VM = {k: vmax(f"m_{k}") for k in ["COLD", "WARM", "STNRY", "OCFNT", "TROF"]}
VM["L"], VM["H"] = vmax("m_L", 99.7), vmax("m_H", 99.7)


def month_figure(m):
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.9), gridspec_kw={"hspace": 0.14, "wspace": 0.03})
    for ax, k in zip(axes.ravel()[:5], ["COLD", "WARM", "STNRY", "OCFNT", "TROF"]):
        a = Z[f"m_{k}"][m - 1]
        im = ax.imshow(np.where(a > 0, a, np.nan), origin="lower", extent=E50, cmap=ramp(C[k]),
                       vmin=0, vmax=VM[k], interpolation="nearest", zorder=1)
        mapax(ax)
        ax.set_title(TITLE[k], loc="left", color=C[k], fontsize=10, fontweight="semibold", pad=3)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.015, aspect=30,
                          format=mpl.ticker.FuncFormatter(pctfmt))
        cb.outline.set_edgecolor(RULE); cb.ax.tick_params(labelsize=6.5, length=2)
        cb.locator = mpl.ticker.MaxNLocator(4); cb.update_ticks()
        if k == "TROF":
            line = Z["dryline_lines"][m - 1]
            ok = np.isfinite(line[:, 1])
            if ok.sum() >= 2:
                x, y = FWD.transform(line[ok, 1], line[ok, 0])
                ax.plot(x, y, color="white", lw=5.0, zorder=6, solid_capstyle="round")
                ax.plot(x, y, color=C["DRYLN"], lw=3.0, zorder=7, solid_capstyle="round",
                        marker="o", ms=3.5)
    ax = axes.ravel()[5]
    # Square-root scaling: a few semi-permanent thermal lows are analysed in
    # the same cell map after map, and on a linear scale they hide the
    # travelling storm tracks entirely.
    for k, cm_hex in (("H", C["H"]), ("L", C["L"])):
        a = Z[f"m_{k}"][m - 1]
        cmap = LinearSegmentedColormap.from_list(k, [(1, 1, 1, 0), mpl.colors.to_rgba(cm_hex, 0.9)])
        im = ax.imshow(np.where(a > 1e-4, a, np.nan), origin="lower", extent=E100, cmap=cmap,
                       norm=mpl.colors.PowerNorm(0.5, vmin=0, vmax=VM[k]),
                       interpolation="bilinear", zorder=1 if k == "H" else 2)
    mapax(ax, anchors=True)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.015, aspect=30,
                      format=mpl.ticker.FuncFormatter(pctfmt))
    cb.outline.set_edgecolor(RULE); cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_ticks([0, VM["L"] / 16, VM["L"] / 4, VM["L"]])
    ax.set_title(TITLE["LH"], loc="left", color=INK, fontsize=10, fontweight="semibold", pad=3)
    fig.savefig(IMG / f"month_{m:02d}.png", dpi=105, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def season_figure(s):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"wspace": 0.04})
    for ax, key, lv, cmap, lab in [
            (axes[0], f"p_L_{s}", np.arange(956, 1012, 4), "magma", "Deepest 1% of lows (hPa)"),
            (axes[1], f"p_H_{s}", np.arange(1020, 1064, 4), "viridis", "Strongest 1% of highs (hPa)")]:
        cm = plt.get_cmap(cmap, len(lv) + 1)
        im = ax.imshow(Z[key], origin="lower", extent=E200, cmap=cm,
                       norm=BoundaryNorm(lv, cm.N, extend="both"), interpolation="nearest", zorder=1)
        mapax(ax, anchors=True, lw=0.8)
        ax.set_title(lab, loc="left", fontsize=10, fontweight="semibold", pad=3)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, aspect=36)
        cb.outline.set_edgecolor(RULE); cb.ax.tick_params(labelsize=7, length=2)
    fig.savefig(IMG / f"pressure_{s}.png", dpi=105, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


for m in range(1, 13):
    month_figure(m)
for s in SEASON_NAME:
    season_figure(s)
print("figures written")

# ------------------------------------------------------------- key numbers
to200 = Transformer.from_crs("EPSG:4326", G200.crs, always_xy=True)
PLACES = {"Seattle": (47.6, -122.3), "Northern Plains": (46.8, -100.8), "Chicago": (41.9, -87.6),
          "New York": (40.7, -74.0), "Houston": (29.8, -95.4)}


def at(field, lat, lon):
    x, y = to200.transform(lon, lat)
    c = int((x - G200.x_min) // 200_000); r = int((y - G200.y_min) // 200_000)
    win = field[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
    return float(np.nanmedian(win)) if np.isfinite(win).any() else float("nan")


facts = []
for m in range(1, 13):
    s = SEASON_OF[m]
    line = Z["dryline_lines"][m - 1]
    lon35 = line[2, 1]
    facts.append({
        "month": MONTHS[m - 1], "season": SEASON_NAME[s], "s": s,
        "dryline": round(100 * float(Z["present_DRYLN"][m - 1]), 1),
        "sqln": round(100 * float(Z["present_SQLN"][m - 1]), 1),
        "dry35": None if not np.isfinite(lon35) else round(abs(float(lon35)), 1),
        "press": [{"place": k, "low": round(at(Z[f"p_L_{s}"], *v)), "high": round(at(Z[f"p_H_{s}"], *v))}
                  for k, v in PLACES.items()],
    })
for f in facts:
    for p in f["press"]:
        p["low"] = None if p["low"] != p["low"] else p["low"]
        p["high"] = None if p["high"] != p["high"] else p["high"]
print(json.dumps(facts[0], indent=1)[:600])


def uri(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


imgs = {f"m{m}": uri(IMG / f"month_{m:02d}.png") for m in range(1, 13)}
imgs.update({s: uri(IMG / f"pressure_{s}.png") for s in SEASON_NAME})

HB = (REPO / "private" / "handbook" / "index.html").read_text(encoding="utf-8")
STYLE = HB[HB.index("<style>"):HB.index("</style>")]
FONTS = HB[HB.index('<link rel="preconnect"'):HB.index("<style>")]
REF = "https://claude.ai/artifact/NFSN9utCyojVrYv9W2B48t"

page = f"""<title>What to Expect This Month</title>
{FONTS}{STYLE}
  .page {{ grid-template-columns: 1fr !important; max-width: 70rem; }}
  .months {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin: 0 0 1.4rem; }}
  .months button {{
    font: 500 0.9rem "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
    padding: 0.45rem 0.8rem; border-radius: 4px; cursor: pointer;
    background: var(--surface); color: var(--ink); border: 1px solid var(--rule);
  }}
  .months button:hover {{ background: var(--accent-2); }}
  .months button[aria-pressed="true"] {{ background: var(--accent); color: var(--surface); border-color: var(--accent); }}
  .months button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
  #mname {{ font-size: 1.9rem; font-weight: 700; margin: 0 0 0.9rem; }}
  .ptable td.n {{ min-width: 6rem; }}
</style>

<div class="page">
  <header class="masthead" style="padding-block: 2.6rem 1.6rem">
    <div class="docline"><span>WPC Surface Analysis</span><span>Training reference</span><span>Companion to <a href="{REF}">What the Surface Analysis Shows</a></span></div>
    <h1>What to Expect This Month</h1>
    <p class="standfirst">Pick a month to see where each boundary is usually drawn, where lows and highs sit, and which central pressures are ordinary or rare for that season. Use it during practice shifts: when your map puts something where these maps are pale, look at the evidence again.</p>
  </header>

  <main>
    <nav class="months" aria-label="Month" id="months"></nav>
    <h2 id="mname">January</h2>

    <div class="keys" id="keys"></div>

    <figure>
      <div class="frame"><img id="mimg" alt="Six maps for the selected month: cold, warm, stationary and occluded front frequency, troughs with the median dryline, and where lows and highs are analysed."></div>
      <figcaption><b>Percent of synoptic maps with the feature drawn through each 50&nbsp;km cell, 2006&ndash;2022.</b> Each panel keeps the same scale in every month, so a pale month really is quieter. The brown line is the median analysed dryline position. Lows and highs are 2009&ndash;2018, 100&nbsp;km cells, on a square-root scale (colour bar shows lows) so the storm tracks show beside the fixed thermal lows. Stationary fronts are green here because a filled map cannot alternate red and blue.</figcaption>
    </figure>

    <section class="topic" style="margin-top:1.2rem">
      <h3>Central pressure, <span id="sname">December-February</span></h3>
      <p>Only 1% of lows analysed in a region were deeper than the left map shows, and only 1% of highs were stronger than the right. A value past these is not wrong, but it is rare enough to check the label before the map goes out (SOP 8.3).</p>
      <figure>
        <div class="frame"><img id="pimg" alt="Two maps of the seasonal 1st-percentile low and 99th-percentile high central pressure."></div>
      </figure>
      <div class="tablewrap"><table class="ptable">
        <thead><tr><th>Region</th><th class="n">Deepest 1% of lows</th><th class="n">Strongest 1% of highs</th></tr></thead>
        <tbody id="ptab"></tbody>
      </table></div>
    </section>

    <footer class="end"><p>From the NOAA Unified Surface Analysis archive (fronts, 2006&ndash;2022) and the coded surface bulletin (centres, 2009&ndash;2018), synoptic analyses only. Current practice may differ, especially for troughs, which are drawn more often now than the 16-year average shows.</p></footer>
  </main>
</div>

<script>
const IMGS = {json.dumps(imgs)};
const FACTS = {json.dumps(facts)};
const ABBR = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"];
const nav = document.getElementById("months");
function fmt(v, unit) {{ return v === null ? "&ndash;" : v + unit; }}
function show(i) {{
  const f = FACTS[i];
  document.getElementById("mname").textContent = f.month;
  document.getElementById("sname").textContent = f.season;
  document.getElementById("mimg").src = IMGS["m" + (i + 1)];
  document.getElementById("pimg").src = IMGS[f.s];
  document.getElementById("keys").innerHTML =
    `<div class="key"><b>${{f.dryline}}%</b><span>of synoptic maps have a dryline drawn</span></div>` +
    `<div class="key"><b>${{f.dry35 === null ? "&ndash;" : f.dry35 + "&deg;W"}}</b><span>median dryline position at 35&deg;N</span></div>` +
    `<div class="key"><b>${{f.sqln}}%</b><span>of synoptic maps have a squall line drawn</span></div>`;
  document.getElementById("ptab").innerHTML = f.press.map(p =>
    `<tr><td>${{p.place}}</td><td class="n">${{fmt(p.low, " hPa")}}</td><td class="n">${{fmt(p.high, " hPa")}}</td></tr>`).join("");
  [...nav.children].forEach((b, j) => b.setAttribute("aria-pressed", j === i ? "true" : "false"));
  try {{ history.replaceState(null, "", "#" + ABBR[i]); }} catch (e) {{}}
}}
FACTS.forEach((f, i) => {{
  const b = document.createElement("button");
  b.type = "button"; b.textContent = f.month.slice(0, 3); b.id = "m-" + ABBR[i];
  b.addEventListener("click", () => show(i));
  nav.appendChild(b);
}});
const h = ABBR.indexOf((location.hash || "").slice(1).toLowerCase());
show(h >= 0 ? h : new Date().getMonth());
</script>
"""
(DEST / "what-to-expect.html").write_text(page, encoding="utf-8")
print(f"wrote what-to-expect.html, {len(page) / 1e6:.1f} MB")
