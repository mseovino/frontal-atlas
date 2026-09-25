"""A cleaner way to draw the cyclone composites.

The earlier figures shaded, for each front type separately, where it was at
least 1.5 times its own ring average. That marks direction only, overlapping
translucent lobes turn to mud exactly where the occlusion meets the other
fronts, and the ratio speckles where counts are small.

Here each point is coloured by the mix of front types drawn there (each type's
share of all front length at that point, sharpened so the leading type
dominates) and faded by how much front is drawn there at all. Nothing
overlaps, mixed zones show as blends, and a feature that surrounds the low
still shows, because nothing is divided by the ring average.

Usage: python analysis/composite_render.py
Writes outputs/composite_*.png and the handbook figures.
"""
import numpy as np
import matplotlib as mpl
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, zoom
from common import OUT, REPO, plt

INK, MUTED, RULE = "#15202B", "#566374", "#C9D2DC"
C = {"COLD": "#1F4FB4", "WARM": "#C8281F", "OCFNT": "#7A2F9E", "STNRY": "#2F7A4A"}
NAME = {"COLD": "Cold", "WARM": "Warm", "OCFNT": "Occluded", "STNRY": "Stationary"}
FRONTS = ["COLD", "WARM", "OCFNT", "STNRY"]
SHARPEN = 3.0        # share ** SHARPEN before mixing colours
STYLE = "steps"      # "steps" (three levels) or "smooth" (continuous blend)
STEPS = [0.2, 0.45, 0.75]        # fractions of the shared scale: sometimes / often / most often
TINTS = [0.28, 0.58, 1.0]        # how much of the type colour each step shows
OUTLINES = False                 # outline each type's own extent on top of the fills
OUTLINE_AT = 0.2                 # outline level, as a fraction of the shared scale
DISP = 950.0
mpl.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": RULE, "xtick.color": MUTED, "ytick.color": MUTED,
                     "savefig.facecolor": "white"})


class Composite:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.z = z
        self.N = dict(zip([str(s) for s in z["strata"]], z["n_low"]))
        self.reach = float(z["reach_km"]); self.bin = float(z["bin_km"])
        nb = int(2 * self.reach / self.bin)
        ax_ = -self.reach + (np.arange(nb) + 0.5) * self.bin
        X, Y = np.meshgrid(ax_, ax_)
        self.rad = np.hypot(X, Y)

    def fields(self, s, sigma=2.0):
        """Per-low density of each type, minus its far-field background.

        Beyond ~1,100 km the fronts around a composite low belong to other
        systems and are spread evenly, so their mean there is a flat haze
        under the whole panel. Subtracting it (not dividing by a ring average)
        removes the haze and still keeps features that surround the low.
        """
        far = (self.rad >= 1100) & (self.rad <= 1450)
        out = {}
        for f in FRONTS:
            a = gaussian_filter(self.z[f"{s}_{f}"].astype(float) / max(self.N[s], 1), sigma)
            out[f] = np.clip(a - a[far].mean(), 0, None)
        return out

    def image(self, s, vmax=None):
        F = self.fields(s)
        total = sum(F.values())
        w = {f: (F[f] / np.where(total > 0, total, 1)) ** SHARPEN for f in FRONTS}
        wsum = sum(w.values())
        rgb = sum(np.multiply.outer(w[f] / np.where(wsum > 0, wsum, 1), to_rgb(C[f])) for f in FRONTS)
        if vmax is None:
            vmax = self.scale(s)
        # Fade by total drawn length; a square root keeps weaker features visible.
        a = np.clip(total / vmax, 0, 1) ** 0.6
        white = np.ones_like(rgb)
        return rgb * a[..., None] + white * (1 - a[..., None]), vmax

    def image_steps(self, s, vmax):
        """Three steps of how much front is drawn, each in the leading type's colour.

        Steps are fractions of the shared scale; blending colours inside a step
        would only make muddy bands, so each point takes its leading type.
        """
        # Interpolate onto a 4x finer grid first, so step edges are smooth
        # curves rather than 25 km staircases.
        F = {f: zoom(a, 4, order=1) for f, a in self.fields(s).items()}
        total = sum(F.values())
        lead = np.argmax(np.stack([F[f] for f in FRONTS]), axis=0)
        level = np.digitize(total / vmax, STEPS)                       # 0 = below the first step
        rgb = np.ones(total.shape + (3,))
        for k, f in enumerate(FRONTS):
            base = np.array(to_rgb(C[f]))
            for lv, tint in enumerate(TINTS, start=1):
                m = (lead == k) & (level == lv)
                rgb[m] = base * tint + (1 - tint)
        self._last_fields = F
        return rgb, total

    def scale(self, s):
        # Fade scaled to the ring 150-900 km out, not the spike at the centre,
        # which every stage has because fronts are drawn touching the low.
        total = sum(self.fields(s).values())
        return np.percentile(total[(self.rad >= 150) & (self.rad <= 900)], 97)

    def draw(self, ax, s, vmax=None, labels=True):
        if STYLE == "steps":
            vmax = vmax or self.scale(s)
            img, total = self.image_steps(s, vmax)
        else:
            img, vmax = self.image(s, vmax)
        ax.imshow(img, origin="lower", extent=(-self.reach, self.reach, -self.reach, self.reach),
                  interpolation="nearest" if STYLE == "steps" else "bilinear", zorder=1)
        if STYLE == "steps":
            ext = np.linspace(-self.reach + self.bin / 2, self.reach - self.bin / 2, total.shape[0])
            ax.contour(ext, ext, total / vmax, levels=STEPS, colors="white", linewidths=0.6, zorder=2)
            if OUTLINES:
                # Each type's own extent as a thin outline, so a type that is
                # second at a point still shows where it runs under the leader.
                for f in FRONTS:
                    ax.contour(ext, ext, self._last_fields[f] / vmax, levels=[OUTLINE_AT],
                               colors=[C[f]], linewidths=1.1, zorder=3)
        th = np.linspace(0, 2 * np.pi, 361)
        for r in (250, 500, 750):
            ax.plot(r * np.cos(th), r * np.sin(th), color="#9AA5B1", lw=0.6, ls=(0, (3, 3)), zorder=3)
            if labels:
                ax.text(r * 0.72, -r * 0.72, f"{r} km", fontsize=7, color=MUTED, va="top", zorder=4)
        ax.plot(0, 0, "o", ms=11, mfc="white", mec="#C8281F", mew=1.6, zorder=9)
        ax.text(0, 0, "L", ha="center", va="center", fontsize=9, fontweight="bold", color="#C8281F", zorder=10)
        ax.set_xlim(-DISP, DISP); ax.set_ylim(-DISP, DISP); ax.set_aspect("equal")
        ticks = [-750, -500, -250, 0, 250, 500, 750]
        ax.set_xticks(ticks); ax.set_yticks(ticks); ax.tick_params(labelsize=7, length=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(RULE)
        return vmax


def legend(fig_or_ax, **kw):
    h = [Patch(facecolor=C[f], label=NAME[f]) for f in FRONTS]
    if STYLE == "steps":
        grey = np.array(to_rgb("#4A5563"))
        h += [Patch(facecolor="none", edgecolor="none", label="")]
        h += [Patch(facecolor=grey * t + (1 - t), edgecolor=RULE, label=lab)
              for t, lab in zip(TINTS, ["sometimes drawn", "often", "most often"])]
    fig_or_ax.legend(handles=h, fontsize=7.5, framealpha=0.96, edgecolor=RULE, **kw)


def panels(comp, specs, name, ncols=2, dest=(OUT,)):
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.7 * nrows),
                             gridspec_kw={"wspace": 0.1, "hspace": 0.16}, squeeze=False)
    # one shared fade scale per figure, so panels compare directly
    vmax = max(comp.scale(s) for s, _ in specs)
    for k, (s, title) in enumerate(specs):
        ax = axes.ravel()[k]
        comp.draw(ax, s, vmax=vmax, labels=(k == 0))
        ax.set_title(f"{title}  ({comp.N[s]:,})", loc="left", fontsize=9.5, fontweight="semibold")
        if k % ncols == 0:
            ax.set_ylabel("km north of the low", fontsize=8, color=MUTED)
        if k // ncols == nrows - 1:
            ax.set_xlabel("km east of the low", fontsize=8, color=MUTED)
    for ax in axes.ravel()[len(specs):]:
        ax.axis("off")
    legend(axes.ravel()[0], loc="lower left")
    for d in dest:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    HB = REPO / "private" / "handbook" / "img"
    st = Composite(OUT / "composite_stages.npz")
    panels(st, [("open", "Open wave"), ("attached_deepening", "Occluding, still deepening"),
                ("attached_filling", "Occluded, filling"), ("wrapped", "Occlusion wrapped, not attached")],
           "hb_composite.png", dest=(OUT, HB))
    panels(st, [("occ_warm", "Occlusion continues the warm front"),
                ("occ_cold", "Occlusion continues the cold front")], "composite_types_mix.png")
    fi = Composite(OUT / "composite_final.npz")
    panels(fi, [("n_occl", "Occluded parent lows"), ("n_triple", "Triple-point lows")],
           "hb_composite_triple.png", dest=(OUT, HB))
    OUTLINES = True   # trial only; the handbook uses the plain steps
    panels(st, [("open", "Open wave"), ("attached_deepening", "Occluding, still deepening"),
                ("attached_filling", "Occluded, filling"), ("wrapped", "Occlusion wrapped, not attached")],
           "trial_composite_outlines.png")
