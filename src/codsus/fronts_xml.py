"""Load NOAA Unified Surface Analysis front XML into a compact Parquet store.

Source archive: "NOAA Unified Surface Analysis Fronts" on Zenodo (record
7505022, CC-BY-4.0), December 2006 through December 2022, one XML per
3-hourly analysis, 46,786 files.

This is the PGEN/VGF export the analyst actually draws in, not a text
derivative of it, and that matters in two ways:

  * Drylines survive. The coded surface bulletin collapses trough, outflow
    boundary, squall line and dryline into a single TROF keyword, per WPC's
    own format spec. Here each line carries a `pgenType`, and DRY_LINE is
    its own value.
  * Coordinates are full precision, against 0.1 degrees (about 11 km) for
    even the high-resolution coded bulletin.

What it does not carry is pressure centers: every element in the archive is
a Line of pgenCategory "Front". Highs and lows still come from CODSUS.

STORAGE. One row per polyline, with the vertices held as Arrow list columns,
rather than one row per vertex. The archive has 2.46M polylines and 12.9M
vertices, so a vertex-per-row table repeats the analysis time, feature type
and feature id twelve million times; that redundancy, not the coordinates,
was the bulk of a first attempt that came to 94 MB. Call `read_points` to
explode it back to the tidy vertex frame the grid code expects.
"""

from __future__ import annotations

import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

# pgenType -> (short ftype, stage). The _FORM and _DISS variants are the same
# feature annotated as forming or dissipating; splitting that annotation into
# its own column keeps `ftype` a small closed vocabulary matching the CODSUS
# tables, so both sources answer the same query.
PGEN_TYPES = {
    "COLD_FRONT": ("COLD", None),
    "COLD_FRONT_FORM": ("COLD", "FORM"),
    "COLD_FRONT_DISS": ("COLD", "DISS"),
    "WARM_FRONT": ("WARM", None),
    "WARM_FRONT_FORM": ("WARM", "FORM"),
    "WARM_FRONT_DISS": ("WARM", "DISS"),
    "STATIONARY_FRONT": ("STNRY", None),
    "STATIONARY_FRONT_FORM": ("STNRY", "FORM"),
    "STATIONARY_FRONT_DISS": ("STNRY", "DISS"),
    "OCCLUDED_FRONT": ("OCFNT", None),
    "OCCLUDED_FRONT_FORM": ("OCFNT", "FORM"),
    "OCCLUDED_FRONT_DISS": ("OCFNT", "DISS"),
    "TROF": ("TROF", None),
    "TROUGH": ("TROF", None),
    # Tropical waves, kept distinct from mid-latitude troughs: they are a
    # different phenomenon and lumping them into TROF would contaminate every
    # trough statistic with 44,465 easterly waves.
    "TROPICAL_TROF": ("TRPWV", None),
    "TROPICAL_WAVE": ("TRPWV", None),
    "DRY_LINE": ("DRYLN", None),
    "SQUALL_LINE": ("SQLN", None),
    "INSTABILITY": ("SQLN", None),
}

# Degrees are stored scaled by 100 in int32 -- a grid step of 0.01 degrees,
# about 1.1 km. Ten times finer than the 0.1-degree coded bulletin, and still
# far finer than the real placement accuracy of a boundary drawn by hand on a
# continental map. The source's six decimals assert a tenth of a metre, which
# is storage spent on noise: consecutive vertices sit a median 1.39 degrees
# apart, so the low-order digits carry no spatial coherence to exploit and
# compress as pure entropy. Quantising here is what takes the store from 79 MB
# to 48 MB, and measurement said to stop at absolute values -- delta-encoding
# along the polyline made it *larger* at this precision, and would have left
# raw DuckDB queries reading differences instead of coordinates.
COORD_SCALE = 100

# Thirteen files in the archive carry corrupt bytes inside an unrelated
# aviation-turbulence element: invalid UTF-8 sequences plus control characters
# that XML 1.0 forbids. Decoding with replacement and stripping these recovers
# the fronts in those analyses instead of discarding the whole file.
_ILLEGAL_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")

_STAMP = re.compile(r"(\d{10})f\d+\.xml$")


@dataclass
class FrontsReport:
    """What actually came out of a load, so silent drops stay visible."""

    analyses: int = 0
    polylines: int = 0
    vertices: int = 0
    skipped: int = 0
    degenerate: int = 0
    repaired: int = 0
    unknown_types: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)


def _valid_time(name: str):
    """Analysis time comes from the filename: pres_pmsl_YYYYMMDDHHf000.xml."""
    m = _STAMP.search(name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%d%H", utc=True, errors="coerce")


def _parse_xml(blob: bytes, report: FrontsReport):
    """Parse, retrying once with control characters stripped."""
    try:
        return ET.fromstring(blob)
    except ET.ParseError:
        text = _ILLEGAL_XML.sub("", blob.decode("utf-8", errors="replace"))
        root = ET.fromstring(text.encode("utf-8"))
        report.repaired += 1
        return root


def parse_analysis(name: str, blob: bytes, report: FrontsReport) -> list:
    """Parse one analysis XML into polyline rows."""
    valid = _valid_time(name)
    if valid is None or pd.isna(valid):
        report.skipped += 1
        report.problems.append(f"{name}: no parseable timestamp")
        return []

    try:
        root = _parse_xml(blob, report)
    except ET.ParseError as exc:
        # A 16-year archive will contain a few malformed files, and one of
        # them must not stop the whole ingest.
        report.skipped += 1
        report.problems.append(f"{name}: {exc}")
        return []

    rows = []
    for line in root.iter("Line"):
        pgen = line.get("pgenType")
        mapped = PGEN_TYPES.get(pgen)
        if mapped is None:
            report.unknown_types[pgen] = report.unknown_types.get(pgen, 0) + 1
            continue
        ftype, stage = mapped

        lats, lons = [], []
        for pt in line.findall("Point"):
            try:
                lat = float(pt.get("Lat"))
                lon = float(pt.get("Lon"))
            except (TypeError, ValueError):
                continue
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            lats.append(int(round(lat * COORD_SCALE)))
            lons.append(int(round(lon * COORD_SCALE)))

        if len(lats) < 2:
            # A single point is not a boundary and yields no segment.
            report.degenerate += 1
            continue

        rows.append({
            "valid_time": valid,
            "ftype": ftype,
            "stage": stage,
            "lat_e2": lats,
            "lon_e2": lons,
        })
        report.vertices += len(lats)

    report.analyses += 1
    report.polylines += len(rows)
    return rows


def iter_members(archive: Path) -> Iterator:
    """Stream (name, bytes) straight out of the .tar.gz, with no extract step.

    Reading the tarball in place avoids materialising 46,786 loose files,
    which on Windows costs more in directory overhead than the archive itself.
    """
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            fh = tf.extractfile(member)
            if fh is not None:
                yield member.name, fh.read()


SCHEMA = pa.schema([
    ("valid_time", pa.timestamp("us", tz="UTC")),
    ("ftype", pa.dictionary(pa.int8(), pa.string())),
    ("stage", pa.dictionary(pa.int8(), pa.string())),
    ("lat_e2", pa.list_(pa.int32())),
    ("lon_e2", pa.list_(pa.int32())),
    ("year", pa.int16()),
])

# Plain zstd on absolute values. Both DELTA_BINARY_PACKED and hand-rolled
# delta encoding were measured and both came out larger at this precision.
_WRITE_OPTS = dict(
    compression="zstd",
    compression_level=19,
    use_dictionary=["ftype", "stage"],
    write_statistics=True,
)


def _frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["year"] = df.valid_time.dt.year.astype("int16")
    # Sorting groups a type's features together and orders them in time, which
    # is what lets run-length and delta encoding do their work.
    df = df.sort_values(["year", "ftype", "valid_time"], kind="stable")
    df["ftype"] = df.ftype.astype("category")
    df["stage"] = df.stage.astype("category")
    return df


def ingest(archive: Path, dest: Path, chunk_analyses: int = 4000, progress=None):
    """Parse the tarball into a year-partitioned Parquet dataset of polylines.

    Written in two stages: streamed staging chunks, then one compacted file per
    year. Streaming alone leaves a few hundred small files whose per-file
    dictionaries and row-group headers cost more than the data they describe.
    """
    report = FrontsReport()
    dest = Path(dest)
    staging = dest.parent / (dest.name + "__staging")
    for path in (dest, staging):
        if path.exists():
            for child in sorted(path.rglob("*"), reverse=True):
                child.unlink() if child.is_file() else child.rmdir()
            path.rmdir()
    staging.mkdir(parents=True)

    buf, part, seen = [], 0, 0

    def flush():
        nonlocal buf, part
        if not buf:
            return
        table = pa.Table.from_pandas(_frame(buf), schema=SCHEMA, preserve_index=False)
        pq.write_table(table, staging / f"part-{part:04d}.parquet", **_WRITE_OPTS)
        part += 1
        buf = []

    for name, blob in iter_members(archive):
        buf.extend(parse_analysis(name, blob, report))
        seen += 1
        if seen % chunk_analyses == 0:
            flush()
            if progress:
                progress(seen, report)
    flush()

    # Compact: one file per year, each written as a single sorted row group.
    dest.mkdir(parents=True)
    staged = pads.dataset(staging, format="parquet")
    years = sorted(
        pa.compute.unique(staged.to_table(columns=["year"])["year"]).to_pylist()
    )
    # `year` is dropped from the file body: it is the partition key, and
    # storing it as a column too makes the reader see an int16 column and a
    # dictionary-typed partition field of the same name, which will not merge.
    leaf_schema = pa.schema([f for f in SCHEMA if f.name != "year"])
    for year in years:
        table = staged.to_table(filter=pads.field("year") == year)
        df = table.to_pandas().sort_values(["ftype", "valid_time"], kind="stable")
        df["ftype"] = df.ftype.astype("category")
        df["stage"] = df.stage.astype("category")
        out = pa.Table.from_pandas(
            df.drop(columns="year"), schema=leaf_schema, preserve_index=False
        )
        (dest / f"year={year}").mkdir(parents=True, exist_ok=True)
        pq.write_table(out, dest / f"year={year}" / "part-0.parquet", **_WRITE_OPTS)

    for child in sorted(staging.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    staging.rmdir()

    if progress:
        progress(seen, report)
    return report


def read_points(dest, columns=None, **kwargs) -> pd.DataFrame:
    """Read the polyline store and explode it into the tidy vertex frame."""
    return explode(pd.read_parquet(dest, **kwargs))


def explode(df: pd.DataFrame) -> pd.DataFrame:
    """Explode polyline rows into the tidy vertex frame.

    Returns the same shape the CODSUS points table has -- one row per vertex,
    with `bulletin_id`, `feature_id`, `ord` and float degrees -- so the grid
    and density code takes either source unchanged. Separate from
    `read_points` so a caller that has already pushed its filters into the
    Parquet read can explode the result without a second pass over the store.
    """
    if df.empty:
        return df

    lengths = df.lat_e2.map(len).to_numpy()
    feature_idx = np.repeat(np.arange(len(df), dtype=np.int64), lengths)

    out = pd.DataFrame({
        "valid_time": df.valid_time.to_numpy()[feature_idx],
        "ftype": pd.Categorical.from_codes(
            df.ftype.cat.codes.to_numpy()[feature_idx], df.ftype.cat.categories
        ) if hasattr(df.ftype, "cat") else df.ftype.to_numpy()[feature_idx],
        "stage": df.stage.to_numpy()[feature_idx],
        "feature_id": feature_idx.astype(np.int32),
        "ord": np.concatenate([np.arange(n, dtype=np.int32) for n in lengths]),
        "lat": np.concatenate(df.lat_e2.to_numpy()) / COORD_SCALE,
        "lon": np.concatenate(df.lon_e2.to_numpy()) / COORD_SCALE,
    })
    # bulletin_id is derivable from the analysis time, so it is not stored;
    # rebuilding it here keeps the two sources interchangeable downstream.
    out["bulletin_id"] = (
        pd.to_datetime(out.valid_time, utc=True).dt.strftime("%Y%m%d%H%M") + "_UA"
    )
    out["res"] = "UA"
    return out
