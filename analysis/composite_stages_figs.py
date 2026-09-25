"""Figures for composite_stages.py (reads outputs/composite_stages.npz).

Shading marks where a front type is at least LEVEL times more common than its
own average at the same distance from the low. Dividing by the azimuthal mean
removes both the pile-up at the centre (fronts are drawn touching the low) and
the isotropic background of unrelated fronts within reach.
"""
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from common import OUT, REPO, plt

IMG = OUT
IMG.mkdir(parents=True, exist_ok=True)
z = np.load(OUT / "composite_stages.npz", allow_pickle=True)
STR = [str(s) for s in z["strata"]]
N = dict(zip(STR, z["n_low"]))
BIN, REACH = float(z["bin_km"]), float(z["reach_km"])
NB = int(2 * REACH / BIN)
ax_ = -REACH + (np.arange(NB) + 0.5) * BIN
XX, YY = np.meshgrid(ax_, ax_)
RAD = np.hypot(XX, YY)
RIDX = np.clip((RAD / BIN).astype(int), 0, NB)
EXT = (-REACH, REACH, -REACH, REACH)
FRONTS = ["COLD", "WARM", "OCFNT", "STNRY"]
C = {"COLD": "#1F4FB4", "WARM": "#C8281F", "OCFNT": "#7A2F9E", "STNRY": "#2F7A4A"}
NAME = {"COLD": "Cold", "WARM": "Warm", "OCFNT": "Occluded", "STNRY": "Stationary"}
LEVEL, DISC = 1.5, 900.0
INK, MUTED, RULE = "#15202B", "#566374", "#C9D2DC"
mpl.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": RULE, "xtick.color": MUTED, "ytick.color": MUTED,
                     "savefig.facecolor": "white"})


def raw(s, f):
    return gaussian_filter(z[f"{s}_{f}"].astype(float) / max(N[s], 1), 1.6)


def anomaly(s, f):
    a = raw(s, f)
    tot = np.bincount(RIDX.ravel(), weights=a.ravel(), minlength=NB + 1)
    cnt = np.bincount(RIDX.ravel(), minlength=NB + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a / (tot / cnt)[RIDX]
    d = RAD <= DISC
    ok = d & (a >= np.nanpercentile(a[d], 65)) & np.isfinite(r)
    return np.nan_to_num(np.where(ok, r, np.nan))


def claimed(s, f):
    return float((anomaly(s, f) >= LEVEL).sum() / (RAD <= DISC).sum())


def peak(s, f):
    """Bearing (deg clockwise from the direction of travel) and range of the
    densest ring-normalised bin between 200 and 800 km."""
    r = anomaly(s, f) * raw(s, f)
    r[(RAD < 200) | (RAD > 800)] = 0
    i, j = np.unravel_index(np.argmax(r), r.shape)
    x, y = ax_[j], ax_[i]
    return (np.degrees(np.arctan2(-y, x)) % 360), float(np.hypot(x, y))


def draw(ax, s, lw=1.6, labels=True, arrow=True):
    for f in FRONTS:
        a = anomaly(s, f)
        ax.contourf(a, levels=[LEVEL, 1e9], extent=EXT, colors=[C[f]], alpha=0.28, zorder=2)
        ax.contour(a, levels=[LEVEL], extent=EXT, colors=[C[f]], linewidths=lw, zorder=3)
    th = np.linspace(0, 2 * np.pi, 361)
    for r in (250, 500, 750):
        ax.plot(r * np.cos(th), r * np.sin(th), color="#9AA5B1", lw=0.6, ls=(0, (3, 3)), zorder=4)
        if labels:
            ax.text(r * 0.72, -r * 0.72, f"{r} km", fontsize=7, color=MUTED, va="top", zorder=5)
    ax.axhline(0, color="#DDE2E8", lw=0.6, zorder=1); ax.axvline(0, color="#DDE2E8", lw=0.6, zorder=1)
    if arrow:
        ax.annotate("", xy=(880, 880), xytext=(620, 880),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4), zorder=9)
        ax.text(750, 830, "direction\nof travel", ha="center", va="top", fontsize=7, color=MUTED)
    ax.plot(0, 0, "o", ms=11, mfc="white", mec="#C8281F", mew=1.6, zorder=9)
    ax.text(0, 0, "L", ha="center", va="center", fontsize=9, fontweight="bold", color="#C8281F", zorder=10)
    ax.set_xlim(-950, 950); ax.set_ylim(-950, 950); ax.set_aspect("equal")
    ax.set_xticks([-750, -500, -250, 0, 250, 500, 750]); ax.set_yticks([-750, -500, -250, 0, 250, 500, 750])
    ax.tick_params(labelsize=7, length=2)
    for sp in ax.spines.values():
        sp.set_edgecolor(RULE)


def legend(ax):
    ax.legend(handles=[Line2D([], [], color=C[f], lw=3, label=NAME[f]) for f in FRONTS],
              loc="lower left", fontsize=7.5, framealpha=0.96, edgecolor=RULE)


def save(fig, name):
    fig.savefig(IMG / name, dpi=150, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)



print("lows: " + "  ".join(f"{s} {N[s]:,}" for s in STR))
TH = np.degrees(np.arctan2(YY, XX)) % 360
ring = (RAD >= 100) & (RAD < 700)
print("occluded front around the low, 100-700 km: amount per low, and peak/mean over 8 sectors")
for s in ["attached_deepening", "attached_filling", "wrapped", "occ_warm", "occ_cold"]:
    a = raw(s, "OCFNT")
    sect = np.array([a[ring & (TH >= k) & (TH < k + 45)].mean() for k in range(0, 360, 45)])
    print(f"  {s:<20s} {a[ring].sum():6.2f}   peak/mean {sect.max() / sect.mean():.2f}   "
          f"favoured {int(np.argmax(sect)) * 45}-{int(np.argmax(sect)) * 45 + 45} deg CCW from east")

fig, axes = plt.subplots(2, 2, figsize=(11, 11), gridspec_kw={"wspace": 0.1, "hspace": 0.14})
for ax, s, lab in [(axes[0, 0], "open", "Open wave"),
                   (axes[0, 1], "attached_deepening", "Occlusion attached, still deepening"),
                   (axes[1, 0], "attached_filling", "Occlusion attached, filling"),
                   (axes[1, 1], "wrapped", "Occlusion near but not attached")]:
    draw(ax, s, lw=1.4, labels=(s == "open"), arrow=False)
    ax.set_title(f"{lab}  ({N[s]:,})", loc="left", fontsize=9.5, fontweight="semibold")
legend(axes[0, 0])
save(fig, "stages_lifecycle.png")

fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.8), gridspec_kw={"wspace": 0.1})
for ax, s, lab in [(axes[0], "occ_warm", "Occlusion continues the warm front"),
                   (axes[1], "occ_cold", "Occlusion continues the cold front")]:
    draw(ax, s, lw=1.4, labels=(s == "occ_warm"), arrow=False)
    ax.set_title(f"{lab}  ({N[s]:,})", loc="left", fontsize=9.5, fontweight="semibold")
legend(axes[0])
save(fig, "stages_occlusion_type.png")
