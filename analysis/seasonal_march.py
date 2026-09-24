"""The seasonal march of frontal frequency, animated.

Daily or weekly? Neither, quite. There are 16 years times 8 analyses for each
calendar day, which is only ~128 analyses -- far too few for a 50 km map. But
frames do not have to be independent. Using a centred running window of +/- 15
days, every day of the year gets a frame built from ~3,970 analyses, which is
the same sample a monthly map has. So the animation runs at daily resolution
while each frame carries monthly-strength statistics; what daily framing buys
is smooth motion, not extra information.

The window is circular, so 31 December and 1 January are neighbours and the
loop closes without a seam.

Leap days are folded onto 28 February. Otherwise one frame in the year would
be built from a quarter of the sample of its neighbours and would flicker.
"""
import sys


import numpy as np
import pandas as pd
from PIL import Image
from common import REPO, basemap, fx, g, plt, read_fronts

CELL_KM = 50.0
HALF_WINDOW = 15          # +/- days
STEP_DAYS = 2             # frames per year
NDAY = 365

FRONTS = ["COLD", "WARM", "STNRY", "OCFNT"]


def day_index(ts):
    """Day of year on a common 365-day calendar, leap day folded onto 28 Feb."""
    doy = ts.dt.dayofyear.to_numpy().astype(int)
    leap = ts.dt.is_leap_year.to_numpy()
    after = leap & (ts.dt.month.to_numpy() > 2)
    doy = np.where(after, doy - 1, doy)
    return np.clip(doy, 1, NDAY) - 1


def daily_counts(ftypes, grid):
    """(365, ny, nx) crossing counts, and analyses per day."""
    raw = read_fronts(ftypes)
    ts = pd.to_datetime(raw.valid_time, utc=True)
    raw = raw.assign(di=day_index(ts))

    cube = np.zeros((NDAY,) + grid.shape, dtype=np.int32)
    for di, sub in raw.groupby("di", sort=True):
        cube[di] = g.frequency_grid(fx.explode(sub), grid)

    per_day = (pd.DataFrame({"di": raw.di, "t": raw.valid_time})
               .groupby("di").t.nunique().reindex(range(NDAY), fill_value=0)
               .to_numpy())
    return cube, per_day


def smooth(cube, per_day):
    """Circular +/- HALF_WINDOW running sum, then divide by analyses in window."""
    k = 2 * HALF_WINDOW + 1
    pad = np.concatenate([cube[-HALF_WINDOW:], cube, cube[:HALF_WINDOW]], axis=0)
    csum = np.cumsum(np.concatenate([np.zeros((1,) + cube.shape[1:]), pad]), axis=0)
    num = csum[k:] - csum[:-k]
    padn = np.concatenate([per_day[-HALF_WINDOW:], per_day, per_day[:HALF_WINDOW]])
    cn = np.cumsum(np.concatenate([[0], padn]))
    den = (cn[k:] - cn[:-k]).astype(float)
    return num / den[:, None, None], den


def cached_counts(ftypes, grid, tag):
    """Gridding the whole archive by day takes minutes; encoding does not.

    Caching the day cube separately means the GIF can be re-encoded at a
    different size, frame rate or palette without paying for the gridding
    again.
    """
    path = REPO / f"march_cube_{tag}.npz"
    if path.exists():
        z = np.load(path)
        return z["cube"], z["per_day"]
    cube, per_day = daily_counts(ftypes, grid)
    np.savez_compressed(path, cube=cube, per_day=per_day)
    return cube, per_day


def animate(ftypes, out, title, cmap, tag):
    grid = g.Grid(cell_km=CELL_KM)
    cube, per_day = cached_counts(ftypes, grid, tag)
    freq, den = smooth(cube, per_day)
    print(f"  window sample: {den.min():.0f} to {den.max():.0f} analyses per frame")

    # One colour scale for the whole loop, or the march is invisible because
    # every frame renormalises to its own maximum.
    vmax = float(np.percentile(freq[freq > 0], 99.5))
    print(f"  vmax {vmax:.4f}")

    labels = pd.date_range("2001-01-01", periods=NDAY, freq="D")
    frames = []
    for d in range(0, NDAY, STEP_DAYS):
        fig, ax, grid, fwd, ext = basemap(cell_km=CELL_KM, figsize=(6.1, 5.1),
                                          anchors=False)
        full = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)
        ax.imshow(np.where(freq[d] > 0, freq[d], np.nan), origin="lower",
                  extent=full, cmap=cmap, interpolation="bilinear",
                  vmin=0, vmax=vmax, zorder=1)
        ax.set_title(f"{title}\n{labels[d]:%d %B}   "
                     f"({2*HALF_WINDOW+1}-day window, 2006-2022)", fontsize=10)
        # Progress bar along the bottom, so the loop position is readable.
        x0, x1 = full[0], full[1]
        y = full[2] + 0.02 * (full[3] - full[2])
        ax.plot([x0, x1], [y, y], color="0.75", lw=2.5, zorder=8,
                solid_capstyle="butt")
        ax.plot([x0, x0 + (x1 - x0) * (d + 1) / NDAY], [y, y],
                color="#B03030", lw=2.5, zorder=9, solid_capstyle="butt")
        fig.canvas.draw()
        frames.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).convert(
                "P", palette=Image.ADAPTIVE, colors=64))
        plt.close(fig)
        if d % 60 == 0:
            print(f"    frame {d}/{NDAY}", flush=True)

    path = REPO / out
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=True, disposal=2)
    mb = path.stat().st_size / 1e6
    print(f"wrote {out}  ({len(frames)} frames, {mb:.1f} MB)")


plt.rcParams["figure.dpi"] = 100

print("all fronts:")
animate(FRONTS, "march_all_fronts.gif",
        "Analysed fronts: the seasonal march", "magma_r", "all")
print("\ncold fronts:")
animate(["COLD"], "march_cold_fronts.gif",
        "Cold fronts: the seasonal march", "Blues", "cold")
