"""Equal-area gridding and great-circle resampling for frontal density.

Two ideas do most of the work here.

Constant-arc-length resampling. Analysts place vertices where the front bends,
so a wiggly front carries more vertices per unit length than a straight one.
Binning raw vertices would therefore map analyst pen habits, not frontal
frequency. Resampling every polyline to a fixed along-track spacing removes
that bias.

Count each cell once per feature. After resampling, a front that meanders
within one grid cell would still deposit several points there. Taking the set
of unique cells per feature makes the statistic "did a front cross this cell",
which is the quantity with a physical meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pyproj import CRS, Geod, Transformer

GEOD = Geod(ellps="WGS84")

# Lambert Azimuthal Equal Area centered on North America. Equal-area matters:
# on a plate carree grid, a cell at 60N covers half the ground of one at 25N,
# so frontal frequency would be inflated toward the pole for free.
NA_LAEA = CRS.from_proj4(
    "+proj=laea +lat_0=45 +lon_0=-100 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)


@dataclass(frozen=True)
class Grid:
    """Regular grid in projected metres."""

    cell_km: float = 50.0
    x_min: float = -4.5e6
    x_max: float = 4.5e6
    y_min: float = -4.0e6
    y_max: float = 4.0e6
    crs: CRS = NA_LAEA

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / (self.cell_km * 1000)))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / (self.cell_km * 1000)))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def transformer(self) -> Transformer:
        return Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    def to_cells(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project lon/lat to integer (row, col). Out-of-domain -> -1."""
        x, y = self.transformer().transform(np.asarray(lon), np.asarray(lat))
        step = self.cell_km * 1000
        col = np.floor((x - self.x_min) / step).astype(np.int64)
        row = np.floor((y - self.y_min) / step).astype(np.int64)
        bad = (
            ~np.isfinite(x)
            | ~np.isfinite(y)
            | (col < 0)
            | (col >= self.nx)
            | (row < 0)
            | (row >= self.ny)
        )
        col[bad] = -1
        row[bad] = -1
        return row, col

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lon, lat) arrays of cell centers, shaped like the grid."""
        step = self.cell_km * 1000
        xs = self.x_min + step * (np.arange(self.nx) + 0.5)
        ys = self.y_min + step * (np.arange(self.ny) + 0.5)
        xx, yy = np.meshgrid(xs, ys)
        inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lon, lat = inv.transform(xx, yy)
        return lon, lat


def resample_polyline(
    lats: np.ndarray, lons: np.ndarray, spacing_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline to roughly constant great-circle spacing.

    Densifies each leg with pyproj's geodesic interpolation, so the result
    follows the great circle between vertices rather than a straight line in
    lat/lon space. That distinction matters at high latitude and for the long
    legs common in low-resolution (1 degree) bulletins.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if lats.size < 2:
        return lats, lons

    out_lat: list[float] = [lats[0]]
    out_lon: list[float] = [lons[0]]

    for i in range(len(lats) - 1):
        lon1, lat1, lon2, lat2 = lons[i], lats[i], lons[i + 1], lats[i + 1]
        _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
        n_extra = int(dist_m / (spacing_km * 1000)) - 1
        if n_extra > 0:
            for lon_i, lat_i in GEOD.npts(lon1, lat1, lon2, lat2, n_extra):
                out_lon.append(lon_i)
                out_lat.append(lat_i)
        out_lat.append(lat2)
        out_lon.append(lon2)

    return np.array(out_lat), np.array(out_lon)


def polyline_length_km(lats: np.ndarray, lons: np.ndarray) -> float:
    """Total great-circle length of a polyline, in km."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    if lats.size < 2:
        return 0.0
    _, _, dist = GEOD.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    return float(np.sum(dist) / 1000.0)


def frequency_grid(
    points: pd.DataFrame,
    grid: Grid | None = None,
    spacing_km: float | None = None,
) -> np.ndarray:
    """Count, per cell, how many distinct features crossed it.

    `points` is the tidy vertex table from codsus.load, already filtered to
    whatever subset you care about (one front type, one month, one decade).
    Divide the result by the number of bulletins in that subset to get a
    frequency per analysis.

    Densification happens in the projected plane rather than by calling the
    geodesic routine per leg. A season of troughs is ~130k features and ~600k
    legs, and a Python loop with a pyproj call inside it takes minutes; this
    is the same computation done once over flat arrays.

    The accuracy cost is nil at this scale. `resample_polyline` follows the
    great circle because a straight line in *lat/lon* diverges badly from one
    at high latitude, but a straight line in an equal-area projection does
    not: over the 10-100 km legs the bulletins contain, the two agree to well
    under a kilometre, against 50 km cells. `resample_polyline` remains the
    reference implementation and is what the resampling tests exercise.
    """
    grid = grid or Grid()
    # Sub-cell sampling so a front cannot step over a cell without landing in it.
    spacing_km = spacing_km or grid.cell_km / 4.0
    counts = np.zeros(grid.shape, dtype=np.int64)
    if points.empty:
        return counts

    df = points.sort_values(["bulletin_id", "feature_id", "ord"])
    feature = df.groupby(["bulletin_id", "feature_id"], sort=False).ngroup().to_numpy()
    x, y = grid.transformer().transform(df.lon.to_numpy(), df.lat.to_numpy())

    # Legs are consecutive rows belonging to the same feature.
    same = feature[1:] == feature[:-1]
    x0, y0 = x[:-1][same], y[:-1][same]
    x1, y1 = x[1:][same], y[1:][same]
    leg_feature = feature[:-1][same]

    finite = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    x0, y0 = x0[finite], y0[finite]
    x1, y1 = x1[finite], y1[finite]
    leg_feature = leg_feature[finite]
    if x0.size == 0:
        return counts

    # Split every leg into whole steps no longer than the sampling spacing.
    n_sub = np.maximum(
        1, np.ceil(np.hypot(x1 - x0, y1 - y0) / (spacing_km * 1000.0)).astype(np.int64)
    )
    leg = np.repeat(np.arange(n_sub.size), n_sub)
    offset = np.arange(n_sub.sum()) - np.repeat(
        np.concatenate([[0], np.cumsum(n_sub)[:-1]]), n_sub
    )
    t = offset / n_sub[leg]

    # t runs [0, 1) per leg, so leg endpoints are appended once rather than
    # being sampled twice where one leg ends and the next begins.
    px = np.concatenate([x0[leg] + t * (x1[leg] - x0[leg]), x1])
    py = np.concatenate([y0[leg] + t * (y1[leg] - y0[leg]), y1])
    pf = np.concatenate([leg_feature[leg], leg_feature])

    step = grid.cell_km * 1000.0
    col = np.floor((px - grid.x_min) / step).astype(np.int64)
    row = np.floor((py - grid.y_min) / step).astype(np.int64)
    keep = (col >= 0) & (col < grid.nx) & (row >= 0) & (row < grid.ny)
    if not keep.any():
        return counts

    cell = row[keep] * grid.nx + col[keep]
    # Unique (feature, cell) pairs: one crossing counts once, however densely
    # sampled. Packing both into one integer keeps this a 1-D unique.
    pairs = np.unique(pf[keep] * (grid.ny * grid.nx) + cell)
    flat = pairs % (grid.ny * grid.nx)

    return np.bincount(flat, minlength=grid.ny * grid.nx).reshape(grid.shape)


def point_grid(centers: pd.DataFrame, grid: Grid | None = None) -> np.ndarray:
    """Count pressure centers per cell.

    Centers are points, not lines, so there is nothing to resample and nothing
    to deduplicate -- two lows in one cell in one analysis are two lows. That
    makes this a plain 2-D histogram in the equal-area plane, and it is a
    different statistic from `frequency_grid`: centers per analysis, not the
    fraction of analyses with a crossing.
    """
    grid = grid or Grid()
    counts = np.zeros(grid.shape, dtype=np.int64)
    if centers.empty:
        return counts

    row, col = grid.to_cells(centers.lon.to_numpy(), centers.lat.to_numpy())
    keep = row >= 0
    if not keep.any():
        return counts

    flat = row[keep] * grid.nx + col[keep]
    return np.bincount(flat, minlength=grid.ny * grid.nx).reshape(grid.shape)

def intensity_grid(
    centers: pd.DataFrame,
    grid: "Grid | None" = None,
    min_count: int = 5,
) -> np.ndarray:
    """Mean central pressure (hPa) per cell, NaN where too few samples to trust.

    Unlike point_grid (a count), this averages pressure_hpa within each cell.
    A genuinely different statistic from frequency: this answers "how strong
    is a system here", not "how often does one pass through."
    """
    grid = grid or Grid()
    if centers.empty:
        return np.full(grid.shape, np.nan)

    df = centers.dropna(subset=["pressure_hpa"])
    if df.empty:
        return np.full(grid.shape, np.nan)

    row, col = grid.to_cells(df.lon.to_numpy(), df.lat.to_numpy())
    keep = row >= 0
    if not keep.any():
        return np.full(grid.shape, np.nan)

    flat = row[keep] * grid.nx + col[keep]
    pressures = df.pressure_hpa.to_numpy()[keep]

    # bincount rather than np.add.at, which falls back to an unbuffered
    # element-by-element loop and is an order of magnitude slower.
    size = grid.ny * grid.nx
    sums = np.bincount(flat, weights=pressures, minlength=size)
    counts = np.bincount(flat, minlength=size)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts
    mean[counts < min_count] = np.nan
    return mean.reshape(grid.shape)