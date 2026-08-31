"""Load NWS Coded Surface Bulletins into tidy tables.

Source archive: NCICS / NC State, "National Weather Service Coded Surface
Bulletins, 2003- (JSON format)" on Zenodo. One JSON object per bulletin with
keys: bulletinType, createDate, validDate, Highs, Lows, ColdFronts,
WarmFronts, OccludedFronts, StationaryFronts, Troughs.

Pressure-center groups are objects of parallel arrays (lats, lons, pressures).
Front groups are objects of arrays where each *front* is one polyline. The
published description does not pin down whether polylines arrive nested
(list-of-lists) or flat, so `_polylines` accepts either and normalizes.

Output is two tidy tables:

  centers  one row per pressure center
  points   one row per vertex of every front/trough polyline

Vertex-level storage keeps everything queryable from SQL (DuckDB reads the
Parquet directly) and lets you rebuild polylines with a groupby on
(bulletin_id, feature_id).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

# Bulletin key -> short feature type code used everywhere downstream.
FRONT_KEYS = {
    "ColdFronts": "COLD",
    "WarmFronts": "WARM",
    "StationaryFronts": "STNRY",
    "OccludedFronts": "OCFNT",
    "Troughs": "TROF",
}

CENTER_KEYS = {"Highs": "H", "Lows": "L"}


@dataclass
class LoadReport:
    """What actually came out of a load, so silent drops are visible."""

    bulletins: int = 0
    centers: int = 0
    polylines: int = 0
    skipped: int = 0
    problems: list[str] | None = None

    def __post_init__(self) -> None:
        if self.problems is None:
            self.problems = []


def _polylines(group: dict) -> list[tuple[list[float], list[float]]]:
    """Normalize a front group into a list of (lats, lons) polylines.

    Handles both nested (one inner list per front) and flat (single front)
    layouts, since the archive documentation is ambiguous on this point.
    """
    lats, lons = group.get("lats"), group.get("lons")
    if not lats or not lons:
        return []

    # Nested: [[lat, lat, ...], [lat, ...]] -> one entry per front.
    if isinstance(lats[0], (list, tuple)):
        return [
            (list(a), list(b))
            for a, b in zip(lats, lons)
            if len(a) == len(b) and len(a) >= 2
        ]

    # Flat: a single polyline.
    if len(lats) == len(lons) and len(lats) >= 2:
        return [(list(lats), list(lons))]
    return []


def _normalize_lon(lon: float) -> float:
    """Coded bulletins express longitude as degrees WEST, positive.

    Convert to signed degrees east in [-180, 180], which is what pyproj,
    Cartopy and every sane downstream tool expect. Values already negative
    are assumed to be signed east and passed through.
    """
    lon = float(lon)
    if lon > 180.0:
        return lon - 360.0
    if lon > 0.0:
        return -lon
    return lon


def iter_bulletins(root: Path) -> Iterator[tuple[Path, dict]]:
    """Yield (path, parsed) for every .json under root, recursively."""
    for path in sorted(root.rglob("*.json")):
        try:
            with path.open() as fh:
                yield path, json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            # Surfaced by the caller rather than raised: a 20-year archive
            # will contain a handful of malformed files and stopping the
            # whole load for one of them is useless behavior.
            yield path, {"__error__": str(exc)}


def load(paths: Iterable[Path] | Path) -> tuple[pd.DataFrame, pd.DataFrame, LoadReport]:
    """Parse an archive directory (or list of files) into (centers, points)."""
    root = Path(paths) if isinstance(paths, (str, Path)) else None
    source = iter_bulletins(root) if root else ((p, json.loads(Path(p).read_text())) for p in paths)

    center_rows: list[dict] = []
    point_rows: list[dict] = []
    report = LoadReport()

    for path, doc in source:
        if "__error__" in doc:
            report.skipped += 1
            report.problems.append(f"{path.name}: {doc['__error__']}")
            continue

        valid = pd.to_datetime(doc.get("validDate"), utc=True, errors="coerce")
        if pd.isna(valid):
            report.skipped += 1
            report.problems.append(f"{path.name}: unparseable validDate")
            continue

        created = pd.to_datetime(doc.get("createDate"), utc=True, errors="coerce")
        res = doc.get("bulletinType", "LR")
        # One bulletin per (valid time, resolution): LR and HR versions of the
        # same analysis coexist in the archive and must not be double counted.
        bulletin_id = f"{valid.strftime('%Y%m%d%H%M')}_{res}"

        for key, kind in CENTER_KEYS.items():
            group = doc.get(key) or {}
            lats = group.get("lats") or []
            lons = group.get("lons") or []
            pres = group.get("pressures") or [None] * len(lats)
            for lat, lon, p in zip(lats, lons, pres):
                center_rows.append(
                    {
                        "bulletin_id": bulletin_id,
                        "valid_time": valid,
                        "create_time": created,
                        "res": res,
                        "kind": kind,
                        "lat": float(lat),
                        "lon": _normalize_lon(lon),
                        "pressure_hpa": float(p) if p is not None else None,
                    }
                )
                report.centers += 1

        for key, ftype in FRONT_KEYS.items():
            group = doc.get(key) or {}
            for feature_id, (lats, lons) in enumerate(_polylines(group)):
                for ordinal, (lat, lon) in enumerate(zip(lats, lons)):
                    point_rows.append(
                        {
                            "bulletin_id": bulletin_id,
                            "valid_time": valid,
                            "res": res,
                            "ftype": ftype,
                            "feature_id": f"{ftype}{feature_id}",
                            "ord": ordinal,
                            "lat": float(lat),
                            "lon": _normalize_lon(lon),
                        }
                    )
                report.polylines += 1

        report.bulletins += 1

    centers = pd.DataFrame(center_rows)
    points = pd.DataFrame(point_rows)
    return centers, points, report


def write_parquet(centers: pd.DataFrame, points: pd.DataFrame, out: Path) -> None:
    """Persist to Parquet, partitioned by year for cheap subsetting."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not centers.empty:
        centers.assign(year=centers.valid_time.dt.year).to_parquet(
            out / "centers", partition_cols=["year"], index=False
        )
    if not points.empty:
        points.assign(year=points.valid_time.dt.year).to_parquet(
            out / "points", partition_cols=["year"], index=False
        )
