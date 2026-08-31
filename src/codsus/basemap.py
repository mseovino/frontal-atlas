"""Coastlines and political boundaries for maps, without Cartopy.

Cartopy is the usual answer and it is a heavy one: no wheel on every platform
(Python 3.14 on Windows wants a source build against GEOS), plus a runtime
fetch of Natural Earth data into a user cache directory. All this module needs
from that stack is line geometry, and Natural Earth publishes it as GeoJSON,
which is a dict of coordinate lists. Parsing it costs nothing but stdlib json
and projecting it costs nothing but the pyproj transformer the grid already
carries.

Fetch the data once with `python scripts/build.py basemap`, which writes to
data/ne/. Nothing here downloads anything on its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Natural Earth 1:50m. Public domain, no attribution required, though it is
# polite. Line layers rather than polygons: nothing here needs a fill, and the
# admin_0 "boundary_lines_land" layer avoids drawing maritime claims.
LAYERS = {
    "coastline": "ne_50m_coastline.geojson",
    "countries": "ne_50m_admin_0_boundary_lines_land.geojson",
    "states": "ne_50m_admin_1_states_provinces_lines.geojson",
}

BASE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
)

# Generous bound on the analysis domain. Clipping before projecting keeps the
# work proportional to what will actually be drawn -- the global 50m coastline
# is a lot of vertices, most of them in the wrong hemisphere.
DOMAIN = (-180.0, -8.0, -8.0, 88.0)  # lon_min, lon_max, lat_min, lat_max


def _geometry_lines(geom: dict) -> list[list]:
    """Every coordinate ring in a geometry, as plain lists of [lon, lat]."""
    kind = geom.get("type")
    coords = geom.get("coordinates") or []
    if kind == "LineString":
        return [coords]
    if kind == "MultiLineString":
        return list(coords)
    if kind == "Polygon":
        return list(coords)
    if kind == "MultiPolygon":
        return [ring for poly in coords for ring in poly]
    return []


def read_layer(path: Path) -> list[np.ndarray]:
    """Read a GeoJSON line layer into arrays of shape (N, 2) in lon/lat."""
    with Path(path).open(encoding="utf-8") as fh:
        doc = json.load(fh)

    out: list[np.ndarray] = []
    for feature in doc.get("features") or []:
        geom = feature.get("geometry") or {}
        for line in _geometry_lines(geom):
            arr = np.asarray(line, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2:
                out.append(arr[:, :2])
    return out


def _clip_runs(line: np.ndarray, domain: tuple) -> list[np.ndarray]:
    """Split a line into the contiguous runs that fall inside the domain.

    Splitting rather than masking matters: a line that leaves the domain and
    comes back must not be reconnected by a straight segment across the gap.
    """
    lon_min, lon_max, lat_min, lat_max = domain
    inside = (
        (line[:, 0] >= lon_min)
        & (line[:, 0] <= lon_max)
        & (line[:, 1] >= lat_min)
        & (line[:, 1] <= lat_max)
    )
    if not inside.any():
        return []
    if inside.all():
        return [line]

    runs: list[np.ndarray] = []
    edges = np.flatnonzero(np.diff(inside.astype(np.int8)))
    start = 0
    for edge in list(edges + 1) + [len(inside)]:
        if inside[start] and edge - start >= 2:
            runs.append(line[start:edge])
        start = edge
    return runs


def projected_segments(
    ne_dir: Path,
    transformer,
    layers: tuple[str, ...] = ("coastline", "countries", "states"),
    domain: tuple = DOMAIN,
    max_jump_km: float = 2000.0,
) -> dict[str, list[np.ndarray]]:
    """Load the requested layers and project them into grid metres.

    Returns {layer: [array of shape (N, 2) in projected metres]}. Runs are
    broken wherever a step exceeds `max_jump_km`, which removes the seam
    artifacts a line gets when it wraps the far side of the projection.
    """
    ne_dir = Path(ne_dir)
    result: dict[str, list[np.ndarray]] = {}

    for name in layers:
        path = ne_dir / LAYERS[name]
        if not path.exists():
            continue

        pieces: list[np.ndarray] = []
        for line in read_layer(path):
            for run in _clip_runs(line, domain):
                x, y = transformer.transform(run[:, 0], run[:, 1])
                xy = np.column_stack([x, y])
                good = np.isfinite(xy).all(axis=1)
                if good.sum() < 2:
                    continue
                xy = xy[good]

                step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
                breaks = np.flatnonzero(step > max_jump_km * 1000.0) + 1
                for chunk in np.split(xy, breaks):
                    if len(chunk) >= 2:
                        pieces.append(chunk)

        result[name] = pieces

    return result


def available(ne_dir: Path) -> bool:
    """True if any layer has been fetched into `ne_dir`."""
    return any((Path(ne_dir) / f).exists() for f in LAYERS.values())
