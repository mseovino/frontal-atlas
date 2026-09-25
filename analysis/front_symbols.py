"""Draw fronts with chart symbols: triangles, semicircles, alternating pips.

Works in whatever units the axes use (map metres or composite kilometres);
`unit` is the size of one kilometre in those units. `side` is +1 to put the
symbols on the left of the line (looking along it in vertex order), -1 for
the right. A stationary front takes the side of its blue triangles and puts
the red semicircles opposite.

The archive stores only the lines, not which side the pips were drawn on, so
any side passed here is inferred; see normal_motion() for one way.
"""
import numpy as np
from matplotlib.patches import Polygon

C = {"COLD": "#1F4FB4", "WARM": "#C8281F", "OCFNT": "#7A2F9E", "DRYLN": "#8B5A2B",
     "TROF": "#D2711A", "SQLN": "#A3231C"}
SPACING_KM, SIZE_KM = 190.0, 62.0


def _frame(x, y, d, s):
    """Positions, unit tangents and left normals at arclengths s."""
    px, py = np.interp(s, d, x), np.interp(s, d, y)
    eps = d[-1] * 1e-3 + 1e-9
    hi, lo = np.minimum(s + eps, d[-1]), np.maximum(s - eps, 0)
    tx = np.interp(hi, d, x) - np.interp(lo, d, x)
    ty = np.interp(hi, d, y) - np.interp(lo, d, y)
    n = np.hypot(tx, ty) + 1e-12
    tx, ty = tx / n, ty / n
    return px, py, tx, ty, -ty, tx


def _triangle(ax, p, sgn, size, col, z):
    cx, cy, tx, ty, nx, ny = p
    b, h = 0.5 * size, 0.85 * size
    pts = [(cx - tx * b, cy - ty * b), (cx + tx * b, cy + ty * b),
           (cx + sgn * nx * h, cy + sgn * ny * h)]
    ax.add_patch(Polygon(pts, closed=True, fc=col, ec=col, lw=0.4, zorder=z))


def _semicircle(ax, p, sgn, size, col, z, filled=True):
    cx, cy, tx, ty, nx, ny = p
    r = 0.5 * size
    a = np.linspace(0, np.pi, 14)
    pts = np.column_stack([cx + r * np.cos(a) * tx + sgn * r * np.sin(a) * nx,
                           cy + r * np.cos(a) * ty + sgn * r * np.sin(a) * ny])
    ax.add_patch(Polygon(pts, closed=filled, fc=col if filled else "none", ec=col,
                         lw=0.4 if filled else 1.1, zorder=z))


def draw_front(ax, x, y, kind, side=1, unit=1.0, lw=2.0, z=6, spacing_km=SPACING_KM,
               size_km=SIZE_KM):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    if d[-1] <= 0:
        return
    step, size = spacing_km * unit, size_km * unit
    if kind == "STNRY":
        # alternating blue and red lengths, one pip in the middle of each
        edges = np.append(np.arange(0, d[-1], step), d[-1])
        xs, ys = np.interp(edges, d, x), np.interp(edges, d, y)
        for k in range(len(xs) - 1):
            ax.plot(xs[k:k + 2], ys[k:k + 2], color=C["COLD"] if k % 2 == 0 else C["WARM"],
                    lw=lw, solid_capstyle="butt", zorder=z)
    else:
        ax.plot(x, y, color=C[kind], lw=lw, solid_capstyle="round", zorder=z)
    s = np.arange(step * 0.5, d[-1] - step * 0.25, step)
    if len(s) == 0 and d[-1] > size * 1.5:
        s = np.array([d[-1] / 2])              # short front: still one pip
    frame = _frame(x, y, d, s)
    for k in range(len(s)):
        p = tuple(a[k] for a in frame)
        if kind == "COLD":
            _triangle(ax, p, side, size, C["COLD"], z + 1)
        elif kind == "WARM":
            _semicircle(ax, p, side, size, C["WARM"], z + 1)
        elif kind == "OCFNT":
            if k % 2 == 0:
                _triangle(ax, p, side, size, C["OCFNT"], z + 1)
            else:
                _semicircle(ax, p, side, size, C["OCFNT"], z + 1)
        elif kind == "STNRY":
            if k % 2 == 0:
                _triangle(ax, p, side, size, C["COLD"], z + 1)
            else:
                _semicircle(ax, p, -side, size, C["WARM"], z + 1)
        elif kind == "DRYLN":
            _semicircle(ax, p, side, size, C["DRYLN"], z + 1, filled=False)


def side_toward(x, y, vx, vy):
    """+1 if the line's left side faces (vx, vy) on average, else -1."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    tx, ty = np.diff(x), np.diff(y)
    return 1 if np.sum(-ty * vx + tx * vy) >= 0 else -1


def densify(x, y, step):
    """Vertices every `step` along a polyline, as an (N, 2) array."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    s = np.append(np.arange(0, d[-1], step), d[-1])
    return np.column_stack([np.interp(s, d, x), np.interp(s, d, y)])


def normal_motion(x, y, others, max_dist):
    """How far to its left a front lies on another map, or None.

    `others` is a list of densified (N, 2) vertex arrays of the same front
    type on another map. For each segment of this front, the nearest point on
    those lines (within `max_dist`, in axis units) gives a displacement; its
    component along the segment's left normal, averaged along the front by
    length, is returned. With the map three hours later, a positive value
    means the front moved toward its left side.
    """
    if not others:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    other = np.vstack(others)
    tx, ty = np.diff(x), np.diff(y)
    ln = np.hypot(tx, ty) + 1e-12
    mx, my = (x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2
    dd = np.hypot(other[:, 0][None] - mx[:, None], other[:, 1][None] - my[:, None])
    j = dd.argmin(axis=1)
    near = dd[np.arange(len(j)), j] < max_dist
    if ln[near].sum() < 0.4 * ln.sum():
        return None
    proj = (-ty / ln) * (other[j, 0] - mx) + (tx / ln) * (other[j, 1] - my)
    return float(np.sum((proj * ln)[near]) / ln[near].sum())
