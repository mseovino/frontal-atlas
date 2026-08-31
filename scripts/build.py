#!/usr/bin/env python3
"""Ingest the Coded Surface Bulletin archive and build a frequency grid.

    python scripts/build.py ingest  data/raw/json  data/parquet --progress
    python scripts/build.py density data/parquet --ftype COLD --month 3 --res HR
    python scripts/build.py plot    density.npz --out density.png

The ingest step is the slow one and you run it once. Everything after that
reads Parquet, which DuckDB will happily query directly:

    SELECT ftype, count(DISTINCT bulletin_id)
    FROM 'data/parquet/points/**/*.parquet'
    WHERE year BETWEEN 2010 AND 2018 GROUP BY 1;
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codsus import basemap as bm  # noqa: E402
from codsus import grid as g  # noqa: E402
from codsus import load as ld  # noqa: E402


def cmd_ingest(args: argparse.Namespace) -> None:
    # Streamed rather than loaded whole: the full archive is ~12.7M vertex
    # rows, which does not fit comfortably in memory as Python objects.
    def progress(done: int, total: int, report: ld.LoadReport) -> None:
        print(
            f"  {done:>6}/{total} files  "
            f"{report.polylines:>9} polylines  {report.centers:>9} centers",
            flush=True,
        )

    report = ld.ingest(
        Path(args.source),
        Path(args.dest),
        chunk_files=args.chunk_files,
        progress=progress if args.progress else None,
    )
    print(
        f"parsed {report.bulletins} bulletins: "
        f"{report.centers} centers, {report.polylines} polylines, "
        f"{report.skipped} skipped, {report.degenerate} degenerate fronts"
    )
    if report.out_of_range:
        print(f"  ! {report.out_of_range} vertices dropped as impossible coordinates")
    if report.lon_flipped:
        # Nonzero means positive longitudes appeared. The JSON archive is
        # signed east throughout, so this is worth looking at before trusting
        # any map built from the result.
        print(f"  ! {report.lon_flipped} longitudes were flipped to signed east")
    for problem in report.problems[:10]:
        print(f"  ! {problem}")
    if len(report.problems) > 10:
        print(f"  ... and {len(report.problems) - 10} more")

    print(f"wrote {args.dest}/centers and {args.dest}/points")

def cmd_centers(args: argparse.Namespace) -> None:
    centers = pd.read_parquet(Path(args.source) / "centers")
    centers = centers[centers.kind == args.kind]

    season_months = {"winter": [12, 1, 2], "summer": [6, 7, 8]}
    if args.season != "all":
        centers = centers[centers.valid_time.dt.month.isin(season_months[args.season])]

    n_bulletins = centers.bulletin_id.nunique()
    if n_bulletins == 0:
        sys.exit("no bulletins match that selection")

    grid = g.Grid(cell_km=args.cell_km)
    counts = g.center_frequency_grid(centers, grid)
    freq = counts / n_bulletins
    np.savez_compressed(
        args.out,
        freq=freq, counts=counts, n_bulletins=n_bulletins,
        cell_km=grid.cell_km,
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        crs=grid.crs.to_proj4(),
    )
    print(f"{n_bulletins} bulletins -> {args.out} (peak {freq.max():.3f} per analysis)")

def cmd_basemap(args: argparse.Namespace) -> None:
    """Fetch Natural Earth line layers for plotting.

    An explicit step rather than a runtime download, so a plot never reaches
    the network behind your back and the data lands somewhere you can see.
    """
    from urllib.request import urlopen

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, filename in bm.LAYERS.items():
        target = dest / filename
        if target.exists() and not args.force:
            print(f"  have {name:<10} {target}")
            continue
        url = f"{bm.BASE_URL}/{filename}"
        with urlopen(url) as response:
            payload = response.read()
        target.write_bytes(payload)
        print(f"  got  {name:<10} {target}  ({len(payload) / 1e6:.1f} MB)")
    print(f"Natural Earth 1:50m, public domain. {dest}")


# Meteorological seasons. Winter is the awkward one: it straddles the year
# boundary, which is why it is a month set and not a date range.
SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def cmd_density(args: argparse.Namespace) -> None:
    # The points table is ~12.7M rows for the full archive, so the selection is
    # pushed into the Parquet read rather than applied after loading it all.
    # Resolution matters: 1-degree LR bulletins and 0.1-degree HR bulletins
    # produce visibly different density fields, so never mix them in one map.
    dataset = ds.dataset(
        Path(args.source) / "points", format="parquet", partitioning="hive"
    )

    period = ds.field("res") == args.res
    if args.start:
        period = period & (ds.field("year") >= pd.Timestamp(args.start).year)
    if args.end:
        period = period & (ds.field("year") <= pd.Timestamp(args.end).year)

    months = SEASONS[args.season] if args.season else ()
    if args.month:
        months = (args.month,)

    def in_window(df: pd.DataFrame) -> pd.DataFrame:
        if months:
            df = df[df.valid_time.dt.month.isin(months)]
        if args.start:
            df = df[df.valid_time >= pd.Timestamp(args.start, tz="UTC")]
        if args.end:
            df = df[df.valid_time <= pd.Timestamp(args.end, tz="UTC")]
        return df

    # The denominator is every analysis in the window, not just the analyses
    # that happened to contain this front type. Dividing by the latter asks
    # "how far did a cold front reach, given that one was drawn at all", which
    # inflates the field wherever the type is intermittent and quietly makes
    # types and seasons non-comparable.
    n_analyses = in_window(
        dataset.to_table(columns=["bulletin_id", "valid_time"], filter=period)
        .to_pandas()
    ).bulletin_id.nunique()

    selection = period
    if args.ftype:
        selection = selection & (ds.field("ftype") == args.ftype)
    points = in_window(
        dataset.to_table(
            columns=["bulletin_id", "feature_id", "ord", "lat", "lon", "valid_time"],
            filter=selection,
        ).to_pandas()
    )

    n_bulletins = points.bulletin_id.nunique()
    if n_analyses == 0:
        sys.exit("no bulletins match that selection")

    # LR vertices are snapped to whole degrees, so many legs are exactly zonal
    # and land entirely inside one row of cells. Grid finer than that spacing
    # and the field breaks into latitude stripes that are pure quantization
    # artifact -- one degree of latitude is about 111 km.
    if args.res == "LR" and args.cell_km < 111.0:
        print(
            f"  ! {args.cell_km:g} km cells on LR data: 1-degree vertex snapping "
            "will alias into latitude stripes. Use --cell-km 150 or larger.",
            file=sys.stderr,
        )

    grid = g.Grid(cell_km=args.cell_km)
    counts = g.frequency_grid(points, grid)

    # Per-analysis frequency, so subsets of different length compare directly.
    freq = counts / n_analyses
    out = Path(args.out)
    # The selection travels with the array. A density field is uninterpretable
    # without knowing which front type, resolution and period produced it, and
    # that is exactly what gets lost between building a grid and plotting it.
    label = _selection_label(args)
    np.savez_compressed(
        out,
        freq=freq,
        counts=counts,
        n_analyses=n_analyses,
        n_bulletins=n_bulletins,
        cell_km=grid.cell_km,
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        crs=grid.crs.to_proj4(),
        label=label,
    )
    print(
        f"{n_analyses} analyses in window, {n_bulletins} containing "
        f"{args.ftype or 'any'} -> {out}  "
        f"(peak {freq.max():.3f} crossings per analysis)"
    )


# Enough of a skeleton to place a continental-scale pattern without shipping
# or downloading a coastline dataset.
ANCHORS = [
    ("Seattle", 47.6, -122.3),
    ("Los Angeles", 34.1, -118.2),
    ("Denver", 39.7, -105.0),
    ("Winnipeg", 49.9, -97.1),
    ("Minneapolis", 45.0, -93.3),
    ("Chicago", 41.9, -87.6),
    ("Dallas", 32.8, -96.8),
    ("Houston", 29.8, -95.4),
    ("Atlanta", 33.7, -84.4),
    ("New York", 40.7, -74.0),
    ("Miami", 25.8, -80.2),
    ("Mexico City", 19.4, -99.1),
    ("Bermuda", 32.3, -64.8),
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _selection_label(args: argparse.Namespace) -> str:
    """Human-readable description of what a density field actually contains."""
    bits = [f"{args.ftype or 'all'} fronts", args.res]
    if args.month:
        bits.append(MONTHS[args.month - 1])
    elif args.season:
        bits.append(args.season)
    if args.start or args.end:
        bits.append(f"{args.start or 'start'} to {args.end or 'end'}")
    return ", ".join(bits)


def cmd_plot(args: argparse.Namespace) -> None:
    """Render a density .npz to PNG.

    Drawn in the grid's own equal-area projection, with a labeled lon/lat
    graticule computed from the CRS and a handful of city reference points.
    Deliberately free of Cartopy: it has no wheel on every platform (including
    Python 3.14 on Windows, where it wants a source build against GEOS) and it
    fetches Natural Earth data at runtime, which is a lot of machinery to make
    a first map legible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.source)
    freq = z["freq"]
    x_min, x_max, y_min, y_max = z["extent"]
    label = str(z["label"]) if "label" in z else Path(args.source).stem
    n = int(z["n_analyses"]) if "n_analyses" in z else int(z["n_bulletins"])

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=args.dpi)
    mesh = ax.imshow(
        np.where(freq > 0, freq, np.nan),
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        cmap=args.cmap,
        interpolation="nearest",
    )

    # Graticule: project constant-lat and constant-lon lines into grid metres.
    # Labeled, because an unlabeled equal-area map of an unfamiliar domain is
    # very hard to read a pattern off.
    grid = g.Grid(cell_km=float(z["cell_km"]))
    fwd = grid.transformer()

    def inside(gx, gy):
        return (gx > x_min) & (gx < x_max) & (gy > y_min) & (gy < y_max)

    for lat in range(20, 81, 10):
        lons = np.linspace(-179, -21, 400)
        gx, gy = fwd.transform(lons, np.full_like(lons, float(lat)))
        ax.plot(gx, gy, color="0.6", lw=0.35, ls=(0, (4, 3)), zorder=2)
        vis = np.flatnonzero(inside(gx, gy))
        if vis.size:
            ax.annotate(
                f"{lat}N", (gx[vis[0]], gy[vis[0]]), color="0.45", fontsize=7,
                xytext=(3, 2), textcoords="offset points", zorder=5,
            )
    for lon in range(-160, -39, 20):
        lats = np.linspace(10, 84, 400)
        gx, gy = fwd.transform(np.full_like(lats, float(lon)), lats)
        ax.plot(gx, gy, color="0.6", lw=0.35, ls=(0, (4, 3)), zorder=2)
        vis = np.flatnonzero(inside(gx, gy))
        if vis.size:
            ax.annotate(
                f"{abs(lon)}W", (gx[vis[0]], gy[vis[0]]), color="0.45", fontsize=7,
                xytext=(2, 3), textcoords="offset points", zorder=5,
            )

    # Coastlines and political boundaries, if they have been fetched.
    from matplotlib.collections import LineCollection

    ne_dir = Path(args.ne_dir)
    have_basemap = bm.available(ne_dir) and not args.no_basemap
    if have_basemap:
        # Weighted so the hierarchy reads at a glance: coast heaviest, then
        # national borders, then states. All dark enough to survive the dark
        # end of the colormap, which the faint end of a grey ramp does not.
        styles = {
            "coastline": dict(color="0.1", lw=0.8),
            "countries": dict(color="0.15", lw=0.6),
            "states": dict(color="0.3", lw=0.45),
        }
        segments = bm.projected_segments(ne_dir, fwd)
        for name, pieces in segments.items():
            if pieces:
                ax.add_collection(
                    LineCollection(pieces, zorder=4, **styles[name])
                )

    # City anchors are the fallback when there are no outlines to draw, and
    # clutter when there are -- so they follow the basemap unless asked for.
    if args.anchors or not have_basemap:
        for name, lat, lon in ANCHORS:
            gx, gy = fwd.transform(lon, lat)
            if not (x_min < gx < x_max and y_min < gy < y_max):
                continue
            ax.plot(gx, gy, "o", ms=2.5, color="0.15", zorder=6)
            ax.annotate(
                name, (gx, gy), color="0.15", fontsize=7,
                xytext=(4, -1), textcoords="offset points", zorder=6,
            )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{label}\n{n:,} analyses, {grid.cell_km:g} km equal-area cells")
    fig.colorbar(mesh, ax=ax, shrink=0.7, label="crossings per analysis")
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="parse JSON archive into Parquet")
    p.add_argument("source", help="directory of bulletin JSON files")
    p.add_argument("dest", help="output directory for Parquet tables")
    p.add_argument("--chunk-files", type=int, default=4000)
    p.add_argument("--progress", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("density", help="build a frontal frequency grid")
    p.add_argument("source", help="Parquet directory from ingest")
    p.add_argument("--ftype", choices=list(ld.FRONT_KEYS.values()))
    p.add_argument("--month", type=int)
    p.add_argument("--season", choices=list(SEASONS), help="DJF, MAM, JJA or SON")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--res", default="HR", choices=["HR", "LR"])
    p.add_argument("--cell-km", type=float, default=50.0)
    p.add_argument("--out", default="density.npz")
    p.set_defaults(func=cmd_density)

    p = sub.add_parser("centers", help="build a high/low pressure center frequency grid")
    p.add_argument("source", help="Parquet directory from ingest")
    p.add_argument("--kind", required=True, choices=["H", "L"])
    p.add_argument("--season", default="all", choices=["all", "winter", "summer"])
    p.add_argument("--cell-km", type=float, default=50.0)
    p.add_argument("--out", default="centers.npz")
    p.set_defaults(func=cmd_centers)    

    p = sub.add_parser("basemap", help="fetch Natural Earth outlines for plotting")
    p.add_argument("--dest", default="data/ne")
    p.add_argument("--force", action="store_true", help="re-download existing layers")
    p.set_defaults(func=cmd_basemap)

    p = sub.add_parser("plot", help="render a density .npz to PNG")
    p.add_argument("source", help=".npz written by density")
    p.add_argument("--out", default="density.png")
    p.add_argument("--cmap", default="magma_r")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--anchors", action="store_true", help="draw city reference points")
    p.add_argument("--ne-dir", default="data/ne", help="Natural Earth GeoJSON directory")
    p.add_argument("--no-basemap", action="store_true", help="omit coastlines and borders")
    p.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
