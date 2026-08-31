#!/usr/bin/env python3
"""Ingest the Coded Surface Bulletin archive and build a frequency grid.

    python scripts/build.py ingest  data/raw/json  data/parquet
    python scripts/build.py density data/parquet --ftype COLD --month 3 --res HR

The ingest step is the slow one and you run it once. Everything after that
reads Parquet, which DuckDB will happily query directly:

    SELECT ftype, count(DISTINCT bulletin_id)
    FROM 'data/parquet/points/**/*.parquet'
    WHERE year BETWEEN 2010 AND 2025 GROUP BY 1;
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codsus import grid as g  # noqa: E402
from codsus import load as ld  # noqa: E402


def cmd_ingest(args: argparse.Namespace) -> None:
    centers, points, report = ld.load(Path(args.source))
    print(
        f"parsed {report.bulletins} bulletins: "
        f"{report.centers} centers, {report.polylines} polylines, "
        f"{report.skipped} skipped"
    )
    for problem in (report.problems or [])[:10]:
        print(f"  ! {problem}")

    ld.write_parquet(centers, points, Path(args.dest))
    print(f"wrote {args.dest}/centers and {args.dest}/points")


def cmd_density(args: argparse.Namespace) -> None:
    points = pd.read_parquet(Path(args.source) / "points")

    # Resolution matters: 1-degree LR bulletins and 0.1-degree HR bulletins
    # produce visibly different density fields, so never mix them in one map.
    points = points[points.res == args.res]
    if args.ftype:
        points = points[points.ftype == args.ftype]
    if args.month:
        points = points[points.valid_time.dt.month == args.month]
    if args.start:
        points = points[points.valid_time >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        points = points[points.valid_time <= pd.Timestamp(args.end, tz="UTC")]

    n_bulletins = points.bulletin_id.nunique()
    if n_bulletins == 0:
        sys.exit("no bulletins match that selection")

    grid = g.Grid(cell_km=args.cell_km)
    counts = g.frequency_grid(points, grid)

    # Per-analysis frequency, so subsets of different length compare directly.
    freq = counts / n_bulletins
    out = Path(args.out)
    np.savez_compressed(
        out,
        freq=freq,
        counts=counts,
        n_bulletins=n_bulletins,
        cell_km=grid.cell_km,
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        crs=grid.crs.to_proj4(),
    )
    print(
        f"{n_bulletins} bulletins -> {out}  "
        f"(peak {freq.max():.3f} crossings per analysis)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="parse JSON archive into Parquet")
    p.add_argument("source", help="directory of bulletin JSON files")
    p.add_argument("dest", help="output directory for Parquet tables")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("density", help="build a frontal frequency grid")
    p.add_argument("source", help="Parquet directory from ingest")
    p.add_argument("--ftype", choices=list(ld.FRONT_KEYS.values()))
    p.add_argument("--month", type=int)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--res", default="HR", choices=["HR", "LR"])
    p.add_argument("--cell-km", type=float, default=50.0)
    p.add_argument("--out", default="density.npz")
    p.set_defaults(func=cmd_density)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
