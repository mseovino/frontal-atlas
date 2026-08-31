"""Load NWS Coded Surface Bulletins into tidy tables.

Source archive: NCICS / NC State, "National Weather Service Coded Surface
Bulletins, 2003- (JSON format)" on Zenodo (record 2646544, CC-BY-SA-4.0).
One JSON object per bulletin with keys: bulletinType, createDate, validDate,
Highs, Lows, ColdFronts, WarmFronts, OccludedFronts, StationaryFronts,
Troughs.

Two things about the real archive differ from what the published description
implies, and both were checked against the files rather than assumed:

  * Pressure-center groups ARE objects of parallel arrays
    ({"lats": [...], "lons": [...], "pressures": [...]}), or null.
  * Front groups are NOT. Each is a *list*, one entry per front, and each
    entry is an object {"lats": [...], "lons": [...], "strength": "..."},
    or null when no front of that type was analyzed. `_polylines` accepts
    this layout plus the nested and flat object layouts the documentation
    suggested, so either producer parses.

Output is two tidy tables:

  centers  one row per pressure center
  points   one row per vertex of every front/trough polyline

Vertex-level storage keeps everything queryable from SQL (DuckDB reads the
Parquet directly) and lets you rebuild polylines with a groupby on
(bulletin_id, feature_id).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Bulletin key -> short feature type code used everywhere downstream.
FRONT_KEYS = {
    "ColdFronts": "COLD",
    "WarmFronts": "WARM",
    "StationaryFronts": "STNRY",
    "OccludedFronts": "OCFNT",
    "Troughs": "TROF",
}

CENTER_KEYS = {"Highs": "H", "Lows": "L"}

# Column types are pinned so every Parquet chunk written during a streaming
# ingest carries an identical schema. Without this, a chunk in which every
# `strength` happens to be null lands as null-typed and the dataset will not
# read back as a single table.
CENTER_DTYPES = {
    "bulletin_id": "string",
    "res": "string",
    "kind": "string",
    "lat": "float64",
    "lon": "float64",
    "pressure_hpa": "float64",
}
POINT_DTYPES = {
    "bulletin_id": "string",
    "res": "string",
    "ftype": "string",
    "feature_id": "string",
    "strength": "string",
    "ord": "int32",
    "lat": "float64",
    "lon": "float64",
}


@dataclass
class LoadReport:
    """What actually came out of a load, so silent drops are visible."""

    bulletins: int = 0
    centers: int = 0
    polylines: int = 0
    skipped: int = 0
    degenerate: int = 0
    lon_flipped: int = 0
    out_of_range: int = 0
    problems: list[str] = field(default_factory=list)


def _polylines(group: object) -> list[tuple[list[float], list[float], str | None]]:
    """Normalize a front group into a list of (lats, lons, strength).

    Three layouts are accepted:

      list of objects   the real archive: one object per front, each with its
                        own lats/lons and a `strength` label
      nested object     {"lats": [[...], [...]], "lons": [[...], [...]]}
      flat object       {"lats": [...], "lons": [...]} -- a single front

    Anything with fewer than two vertices, or with mismatched lat/lon lengths,
    is dropped: it cannot contribute a segment and would only add noise.
    """
    out: list[tuple[list[float], list[float], str | None]] = []

    if isinstance(group, list):
        for item in group:
            if not isinstance(item, dict):
                continue
            lats, lons = item.get("lats"), item.get("lons")
            if not lats or not lons or len(lats) != len(lons) or len(lats) < 2:
                continue
            out.append((list(lats), list(lons), item.get("strength")))
        return out

    if not isinstance(group, dict):
        return out

    lats, lons = group.get("lats"), group.get("lons")
    if not lats or not lons:
        return out

    # Nested: [[lat, lat, ...], [lat, ...]] -> one entry per front.
    if isinstance(lats[0], (list, tuple)):
        return [
            (list(a), list(b), None)
            for a, b in zip(lats, lons)
            if len(a) == len(b) and len(a) >= 2
        ]

    # Flat: a single polyline.
    if len(lats) == len(lons) and len(lats) >= 2:
        out.append((list(lats), list(lons), group.get("strength")))
    return out

def center_frequency_grid(centers: pd.DataFrame, grid: Grid | None = None) -> np.ndarray:
    """Count how many analyses placed a center (H or L) in each cell."""
    grid = grid or Grid()
    counts = np.zeros(grid.shape, dtype=np.int64)
    if centers.empty:
        return counts
    row, col = grid.to_cells(centers.lon.to_numpy(), centers.lat.to_numpy())
    keep = row >= 0
    if not keep.any():
        return counts
    flat = row[keep] * grid.nx + col[keep]
    np.add.at(counts.reshape(-1), flat, 1)
    return counts

def _normalize_lon(lon: float, report: LoadReport | None = None) -> float:
    """Return longitude as signed degrees east in [-180, 180].

    The JSON archive already stores signed east -- every longitude in it is
    negative, spanning 180W to about 1W -- so for that source this is a
    pass-through. The raw ASCII bulletins encode positive degrees west, and a
    positive value here is taken to be that convention and flipped.

    The flip is counted in the report rather than done silently: a genuinely
    east-of-Greenwich analysis point would be mangled by it, and you want that
    visible as a number you can check, not buried in a map that looks fine.
    """
    lon = float(lon)
    if lon > 180.0:
        return lon - 360.0
    if lon > 0.0:
        if report is not None:
            report.lon_flipped += 1
        return -lon
    return lon


def _in_range(lat: float, lon: float) -> bool:
    """Reject coordinates that cannot exist.

    The archive carries a small number of transcription errors -- latitudes of
    91 to 98 degrees, all in early LR bulletins. Ten vertices in 12.8 million,
    so they change nothing statistically, but they project to infinity and are
    better dropped at the door with a count than carried as NaN.
    """
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _read(path: Path) -> dict:
    try:
        with Path(path).open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        # Surfaced by the caller rather than raised: a 20-year archive will
        # contain a handful of malformed files, and stopping the whole load
        # for one of them is useless behavior.
        return {"__error__": str(exc)}


def iter_bulletins(root: Path) -> Iterator[tuple[Path, dict]]:
    """Yield (path, parsed) for every .json under root, recursively."""
    for path in sorted(Path(root).rglob("*.json")):
        yield path, _read(path)


def parse_bulletin(
    path: Path,
    doc: dict,
    center_rows: list[dict],
    point_rows: list[dict],
    report: LoadReport,
) -> None:
    """Parse one bulletin, appending rows to the accumulators."""
    if "__error__" in doc:
        report.skipped += 1
        report.problems.append(f"{Path(path).name}: {doc['__error__']}")
        return

    valid = pd.to_datetime(doc.get("validDate"), utc=True, errors="coerce")
    if pd.isna(valid):
        report.skipped += 1
        report.problems.append(f"{Path(path).name}: unparseable validDate")
        return

    created = pd.to_datetime(doc.get("createDate"), utc=True, errors="coerce")
    res = doc.get("bulletinType") or "LR"
    # One bulletin per (valid time, resolution): LR and HR versions of the
    # same analysis coexist in the archive and must not be double counted.
    bulletin_id = f"{valid.strftime('%Y%m%d%H%M')}_{res}"

    for key, kind in CENTER_KEYS.items():
        group = doc.get(key) or {}
        if not isinstance(group, dict):
            continue
        lats = group.get("lats") or []
        lons = group.get("lons") or []
        pres = group.get("pressures") or [None] * len(lats)
        for lat, lon, p in zip(lats, lons, pres):
            lat, lon = float(lat), _normalize_lon(lon, report)
            if not _in_range(lat, lon):
                report.out_of_range += 1
                continue
            center_rows.append(
                {
                    "bulletin_id": bulletin_id,
                    "valid_time": valid,
                    "create_time": created,
                    "res": res,
                    "kind": kind,
                    "lat": lat,
                    "lon": lon,
                    "pressure_hpa": float(p) if p is not None else None,
                }
            )
            report.centers += 1

    for key, ftype in FRONT_KEYS.items():
        group = doc.get(key)
        parsed = _polylines(group)
        if isinstance(group, list):
            # Fronts present in the file that carried too few vertices to be
            # a line. Counted so the drop rate is a number you can look at.
            report.degenerate += len(group) - len(parsed)
        for feature_id, (lats, lons, strength) in enumerate(parsed):
            kept: list[dict] = []
            # `ord` keeps the original vertex index, so a dropped bad vertex
            # leaves a visible gap rather than silently renumbering the line.
            for ordinal, (lat, lon) in enumerate(zip(lats, lons)):
                lat, lon = float(lat), _normalize_lon(lon, report)
                if not _in_range(lat, lon):
                    report.out_of_range += 1
                    continue
                kept.append(
                    {
                        "bulletin_id": bulletin_id,
                        "valid_time": valid,
                        "res": res,
                        "ftype": ftype,
                        "feature_id": f"{ftype}{feature_id}",
                        "strength": strength,
                        "ord": ordinal,
                        "lat": lat,
                        "lon": lon,
                    }
                )
            if len(kept) < 2:
                # Nothing left that can make a segment.
                report.degenerate += 1
                continue
            point_rows.extend(kept)
            report.polylines += 1

    report.bulletins += 1


def _frame(rows: list[dict], dtypes: dict[str, str]) -> pd.DataFrame:
    """Build a DataFrame with a pinned schema, whether or not there are rows."""
    if not rows:
        times = ["valid_time", "create_time"] if "kind" in dtypes else ["valid_time"]
        empty = pd.DataFrame(
            {c: pd.Series(dtype="datetime64[ns, UTC]") for c in times}
        )
        for col, dt in dtypes.items():
            empty[col] = pd.Series(dtype=dt)
        return empty
    df = pd.DataFrame(rows)
    return df.astype({c: d for c, d in dtypes.items() if c in df.columns})


def load(paths: Iterable[Path] | Path) -> tuple[pd.DataFrame, pd.DataFrame, LoadReport]:
    """Parse an archive directory (or list of files) into (centers, points).

    Everything is held in memory. The full archive is roughly 12.7M vertex
    rows, which does not fit comfortably as Python objects -- use `ingest`
    for that, which streams.
    """
    root = Path(paths) if isinstance(paths, (str, Path)) else None
    source = iter_bulletins(root) if root else ((p, _read(Path(p))) for p in paths)

    center_rows: list[dict] = []
    point_rows: list[dict] = []
    report = LoadReport()

    for path, doc in source:
        parse_bulletin(path, doc, center_rows, point_rows, report)

    return _frame(center_rows, CENTER_DTYPES), _frame(point_rows, POINT_DTYPES), report


def _write_chunk(df: pd.DataFrame, out: Path, name: str, chunk: int) -> None:
    if df.empty:
        return
    df = df.assign(year=df.valid_time.dt.year.astype("int32"))
    pq.write_to_dataset(
        pa.Table.from_pandas(df, preserve_index=False),
        root_path=str(Path(out) / name),
        partition_cols=["year"],
        basename_template=f"part-{chunk:05d}-{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )


def ingest(root: Path, out: Path, chunk_files: int = 4000, progress=None) -> LoadReport:
    """Stream the archive to year-partitioned Parquet in bounded memory.

    Files are parsed in chunks and each chunk is appended to the dataset.
    Chunk boundaries do not align with year partitions, which is fine: each
    chunk writes its own part file into every year it touches.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(root).rglob("*.json"))
    report = LoadReport()

    for chunk, start in enumerate(range(0, len(files), chunk_files)):
        center_rows: list[dict] = []
        point_rows: list[dict] = []
        for path in files[start : start + chunk_files]:
            parse_bulletin(path, _read(path), center_rows, point_rows, report)

        _write_chunk(_frame(center_rows, CENTER_DTYPES), out, "centers", chunk)
        _write_chunk(_frame(point_rows, POINT_DTYPES), out, "points", chunk)
        if progress is not None:
            progress(min(start + chunk_files, len(files)), len(files), report)

    return report


def write_parquet(centers: pd.DataFrame, points: pd.DataFrame, out: Path) -> None:
    """Persist to Parquet, partitioned by year for cheap subsetting."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    _write_chunk(centers, out, "centers", 0)
    _write_chunk(points, out, "points", 0)
