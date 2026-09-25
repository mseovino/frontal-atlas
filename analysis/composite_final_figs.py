"""Figures for the finished composite (reads outputs/composite_final.npz).

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

IMG = REPO / "private" / "handbook" / "img"
IMG.mkdir(parents=True, exist_ok=True)
z = np.load(OUT / "composite_final.npz", allow_pickle=True)
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


# -------------------------------------------------------------- numbers
print("lows: " + "  ".join(f"{s} {N[s]:,}" for s in STR))
print(f"\nshare of the {DISC:.0f} km disc each type claims (sharper = larger, more coherent lobe)")
print(f"{'stratum':<11s}" + "".join(f"{NAME[f]:>12s}" for f in FRONTS))
for s in STR:
    print(f"{s:<11s}" + "".join(f"{100 * claimed(s, f):11.1f}%" for f in FRONTS))
print("\npeak bearing from direction of travel (clockwise) and range, rotated, all lows")
for f in FRONTS:
    b, r = peak("rot", f)
    print(f"  {NAME[f]:<11s} {b:5.0f} deg  {r:4.0f} km")

# -------------------------------------------------------------- main
fig, ax = plt.subplots(figsize=(7.2, 7.0))
draw(ax, "rot", lw=1.9)
box = dict(boxstyle="round,pad=0.3", fc="white", ec=RULE, lw=0.7)
for f in FRONTS:
    b, r = peak("rot", f)
    t = np.radians(b); x, y = r * np.cos(t), -r * np.sin(t)
    ax.text(x * 1.0, y * 1.0 + (70 if y >= 0 else -70), NAME[f].lower() + " front" if f != "OCFNT" else "occlusion",
            color=C[f], fontsize=8.5, fontweight="semibold", ha="center", va="center", bbox=box, zorder=12)
ax.set_xlabel("km along the direction of travel", fontsize=8, color=MUTED)
ax.set_ylabel("km to the left of travel", fontsize=8, color=MUTED)
save(fig, "rot_composite.png")

# ------------------------------------------------------- what rotation buys
fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), gridspec_kw={"wspace": 0.12})
draw(axes[0], "north", arrow=False); axes[0].set_title("Grid north up", loc="left", fontweight="semibold")
draw(axes[1], "rot", labels=False); axes[1].set_title("Rotated so every low travels to the right", loc="left", fontweight="semibold")
legend(axes[0])
save(fig, "rot_composite_rotation.png")

# ------------------------------------------------------------- by depth
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"wspace": 0.12})
t1, t2 = float(z["t1"]), float(z["t2"])
for ax, s, lab in zip(axes, ["rot_deep", "rot_mod", "rot_weak"],
                      [f"Deep, {t1:.0f} hPa or lower", f"Moderate, {t1:.0f}-{t2:.0f} hPa", f"Weak, above {t2:.0f} hPa"]):
    draw(ax, s, lw=1.3, labels=(s == "rot_deep"), arrow=(s == "rot_deep"))
    ax.set_title(f"{lab}  ({N[s]:,})", loc="left", fontsize=9.5, fontweight="semibold")
legend(axes[0])
save(fig, "rot_composite_depth.png")

# ------------------------------------------- triple points, land and ocean
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"wspace": 0.12})
for ax, s, lab in zip(axes, ["triple", "rot_land", "rot_ocean"],
                      ["Triple-point lows", "Parent lows over land", "Parent lows over the oceans"]):
    draw(ax, s, lw=1.3, labels=(s == "triple"), arrow=(s == "triple"))
    ax.set_title(f"{lab}  ({N[s]:,})", loc="left", fontsize=9.5, fontweight="semibold")
legend(axes[0])
save(fig, "rot_composite_triple.png")


# ------------------------------------------------------ lifecycle stages
if "r_occl" in N:
    print("\nlifecycle strata (share of disc claimed)")
    print(f"{'stratum':<13s}{'lows':>9s}" + "".join(f"{NAME[f]:>12s}" for f in FRONTS))
    for s in ["n_mov", "rot", "n_open", "r_open", "n_occl", "r_occl", "n_occl_land", "r_occl_land"]:
        print(f"{s:<13s}{N[s]:9,}" + "".join(f"{100 * claimed(s, f):11.1f}%" for f in FRONTS))

    TH = np.degrees(np.arctan2(YY, XX)) % 360
    print("\nwhere the occlusion sits around occluded lows, 100-600 km "
          "(peak/mean over eight 45-degree sectors; 1.0 = no preferred side)")
    for s in ["n_occl", "r_occl", "n_occl_land", "r_occl_land"]:
        a = raw(s, "OCFNT")
        ring = (RAD >= 100) & (RAD < 600)
        sect = np.array([a[ring & (TH >= k) & (TH < k + 45)].mean() for k in range(0, 360, 45)])
        k = int(np.argmax(sect))
        frame = "compass, 0 = east, counter-clockwise" if s.startswith("n") else "0 = direction of travel, counter-clockwise"
        print(f"  {s:<13s} peak/mean {sect.max() / sect.mean():.2f}   favoured sector {k * 45}-{k * 45 + 45} deg ({frame})")

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 10.4), gridspec_kw={"wspace": 0.1, "hspace": 0.16})
    for ax, s, lab in [(axes[0, 0], "n_open", "Open waves, grid north up"),
                       (axes[0, 1], "r_open", "Open waves, rotated to travel"),
                       (axes[1, 0], "n_occl", "Occluded lows, grid north up"),
                       (axes[1, 1], "r_occl", "Occluded lows, rotated to travel")]:
        draw(ax, s, lw=1.4, labels=(s == "n_open"), arrow=s.startswith("r"))
        ax.set_title(f"{lab}  ({N[s]:,})", loc="left", fontsize=9.5, fontweight="semibold")
    legend(axes[0, 0])
    save(fig, "rot_composite_stages.png")


# ------------------------------------------ handbook: north-up, by stage
def note(ax, xy, text, col):
    ax.text(*xy, text, color=col, fontsize=8.5, fontweight="semibold", ha="center", va="center",
            zorder=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RULE, lw=0.7))


if "n_triple" in N:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.1), gridspec_kw={"wspace": 0.1})
    draw(axes[0], "n_open", arrow=False)
    note(axes[0], (-430, -470), "cold front", C["COLD"])
    note(axes[0], (470, -220), "warm front", C["WARM"])
    note(axes[0], (-20, 830), "occlusions of older\ncyclones to the north", C["OCFNT"])
    axes[0].set_title(f"Open waves  ({N['n_open']:,} lows)", loc="left", fontsize=10, fontweight="semibold")
    draw(axes[1], "n_occl", labels=False, arrow=False)
    note(axes[1], (300, -300), "occlusion", C["OCFNT"])
    note(axes[1], (-330, -640), "cold front", C["COLD"])
    note(axes[1], (690, -560), "warm front", C["WARM"])
    note(axes[1], (560, 520), "stationary", C["STNRY"])
    axes[1].set_title(f"Occluded lows  ({N['n_occl']:,} lows)", loc="left", fontsize=10, fontweight="semibold")
    for ax in axes:
        ax.set_xlabel("km east of the low", fontsize=8, color=MUTED)
    axes[0].set_ylabel("km north of the low", fontsize=8, color=MUTED)
    save(fig, "hb_composite.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.1), gridspec_kw={"wspace": 0.1})
    draw(axes[0], "n_occl", arrow=False)
    axes[0].set_title(f"Occluded parent lows  ({N['n_occl']:,})", loc="left", fontsize=10, fontweight="semibold")
    draw(axes[1], "n_triple", labels=False, arrow=False)
    axes[1].set_title(f"Triple-point lows  ({N['n_triple']:,})", loc="left", fontsize=10, fontweight="semibold")
    legend(axes[0])
    for ax in axes:
        ax.set_xlabel("km east of the low", fontsize=8, color=MUTED)
    axes[0].set_ylabel("km north of the low", fontsize=8, color=MUTED)
    save(fig, "hb_composite_triple.png")
