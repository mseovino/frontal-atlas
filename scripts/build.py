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

def _load_centers(args: argparse.Namespace) -> pd.DataFrame:
    """Read the centers table for one kind, resolution and season.

    Filtering on `res` is not optional. LR and HR carry the same analysis at
    two resolutions, so without it every analysis is counted twice and at two
    slightly different positions, which both doubles the denominator and
    smears the field.
    """
    centers = (
        ds.dataset(Path(args.source) / "centers", format="parquet", partitioning="hive")
        .to_table(
            columns=["bulletin_id", "valid_time", "kind", "lat", "lon", "pressure_hpa"],
            filter=(ds.field("res") == args.res) & (ds.field("kind") == args.kind),
        )
        .to_pandas()
    )
    if args.season:
        centers = centers[centers.valid_time.dt.month.isin(SEASONS[args.season])]

    centers, dropped = ld.drop_implausible_pressure(centers)
    if dropped:
        print(f"  dropped {dropped} centers whose pressure contradicts their label")
    return centers


def cmd_centers(args: argparse.Namespace) -> None:
    centers = _load_centers(args)
    n_analyses = centers.bulletin_id.nunique()
    if n_analyses == 0:
        sys.exit("no bulletins match that selection")

    grid = g.Grid(cell_km=args.cell_km)
    raw = g.point_grid(centers, grid)

    # Mask on the raw counts, before smoothing: after a Gaussian pass every
    # cell near an occupied one holds a fractional count, so thresholding the
    # smoothed field would be testing the support of the kernel rather than
    # how much data actually stands behind each cell.
    counts = raw.astype(float)
    if args.smooth_km > 0:
        from scipy.ndimage import gaussian_filter

        counts = gaussian_filter(counts, sigma=args.smooth_km / args.cell_km)

    freq = counts / n_analyses
    freq[raw < args.min_count] = np.nan

    np.savez_compressed(
        args.out,
        freq=freq,
        counts=raw,
        n_analyses=n_analyses,
        n_bulletins=n_analyses,
        cell_km=grid.cell_km,
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        crs=grid.crs.to_proj4(),
        label=_centers_label(args),
        unit="centers per analysis",
    )
    print(
        f"{n_analyses} analyses -> {args.out} "
        f"(peak {np.nanmax(freq):.3f} per analysis)"
    )

def cmd_track(args: argparse.Namespace) -> None:
    from codsus import track as tk

    centers = (
        ds.dataset(Path(args.source) / "centers", format="parquet", partitioning="hive")
        .to_table(
            columns=["bulletin_id", "valid_time", "res", "kind",
                     "lat", "lon", "pressure_hpa"],
            filter=(ds.field("res") == args.res) & (ds.field("kind") == args.kind),
        )
        .to_pandas()
    )
    # A center whose pressure contradicts its label would either break a track
    # in two on the pressure-jump cap or drag the cost of a wrong match down.
    centers, dropped = ld.drop_implausible_pressure(centers)
    if dropped:
        print(f"  dropped {dropped} centers whose pressure contradicts their label")

    params = tk.TrackParams(max_gap_hours=args.max_gap_hours)
    tracks = tk.build_tracks(centers, kind=args.kind, res=args.res, params=params)
    stats = tk.track_stats(tracks)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(out / "tracks.parquet")
    stats.to_parquet(out / "track_stats.parquet")

    print(f"{len(stats)} tracks from {len(tracks)} centers "
          f"({args.kind}, {args.res})")
    if not stats.empty:
        longest = stats.loc[stats.n_steps.idxmax()]
        deepest = stats.loc[stats.min_pressure_hpa.idxmin()]
        print(f"  longest track: {longest.n_steps} steps over "
              f"{longest.duration_hours / 24:.1f} days, "
              f"net displacement {longest.net_km:.0f} km")
        print(f"  deepest track: {deepest.min_pressure_hpa:.0f} hPa, "
              f"genesis {deepest.genesis_time}")

        # Quasi-stationary features link into very long tracks that are not
        # cyclones. Reported here so the number is seen before anyone
        # computes a lifetime statistic over the whole set.
        durable = stats[stats.n_steps >= 8]  # a day or more at 3-hourly
        stationary = durable[durable.net_km < 500.0]
        if len(durable):
            print(f"  {len(durable)} tracks last a day or more; "
                  f"{len(stationary)} of those displace under 500 km "
                  f"({100 * len(stationary) / len(durable):.0f}%) and are "
                  f"likely quasi-stationary, not travelling systems")

def cmd_intensity(args: argparse.Namespace) -> None:
    centers = _load_centers(args)
    n_analyses = centers.bulletin_id.nunique()
    if n_analyses == 0:
        sys.exit("no bulletins match that selection")

    grid = g.Grid(cell_km=args.cell_km)
    mean_hpa = g.intensity_grid(centers, grid, min_count=args.min_count)
    np.savez_compressed(
        args.out,
        mean_hpa=mean_hpa,
        n_analyses=n_analyses,
        n_bulletins=n_analyses,
        cell_km=grid.cell_km,
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        crs=grid.crs.to_proj4(),
        label=_centers_label(args, "mean pressure"),
    )
    valid = mean_hpa[~np.isnan(mean_hpa)]
    if valid.size == 0:
        sys.exit(f"every cell fell below --min-count {args.min_count}")
    print(
        f"{n_analyses} analyses -> {args.out} "
        f"(range {valid.min():.0f}-{valid.max():.0f} hPa)"
    )

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


def _centers_label(args: argparse.Namespace, what: str = "density") -> str:
    """Describe a centers or intensity field so the plot can title itself."""
    kind = "Lows" if args.kind == "L" else "Highs"
    bits = [f"{kind} {what}", args.res]
    if args.season:
        bits.append(args.season)
    return ", ".join(bits)


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


def cmd_trackgrid(args: argparse.Namespace) -> None:
    """Grid tracks into a density field the ordinary `plot` command can render.

    Three statistics, and they answer different questions:

      path     how often a track passes over each cell -- the storm track
      genesis  where systems are first analysed -- cyclogenesis regions
      lysis    where they are last analysed

    `path` reuses `frequency_grid`, because a track is a polyline exactly like
    a front is: the same "count each feature once per cell" rule applies, and
    a system that loiters in one cell should count once, not once per analysis.
    """
    source = Path(args.source)
    tracks = pd.read_parquet(source / "tracks.parquet")
    stats = pd.read_parquet(source / "track_stats.parquet")

    keep = (stats.n_steps >= args.min_steps) & (stats.net_km >= args.min_net_km)
    if args.season:
        keep &= stats.genesis_time.dt.month.isin(SEASONS[args.season])
    selected = stats[keep]
    if selected.empty:
        sys.exit("no tracks match that selection")

    grid = g.Grid(cell_km=args.cell_km)
    n_tracks = len(selected)

    if args.mode == "path":
        paths = tracks[tracks.track_id.isin(set(selected.track_id))]
        # frequency_grid keys on (bulletin_id, feature_id) and orders by `ord`.
        # Built column by column rather than renamed: the tracks table already
        # has its own bulletin_id, and renaming onto it makes the label
        # ambiguous rather than replacing it.
        counts = g.frequency_grid(
            pd.DataFrame(
                {
                    "bulletin_id": paths.track_id.to_numpy(),
                    "feature_id": "track",
                    "ord": paths.track_step.to_numpy(),
                    "lat": paths.lat.to_numpy(),
                    "lon": paths.lon.to_numpy(),
                }
            ),
            grid,
        )
        unit = "track passages per track"
    else:
        lat = selected.genesis_lat if args.mode == "genesis" else selected.lysis_lat
        lon = selected.genesis_lon if args.mode == "genesis" else selected.lysis_lon
        counts = g.point_grid(
            pd.DataFrame({"lat": lat.to_numpy(), "lon": lon.to_numpy()}), grid
        )
        unit = f"{args.mode} points per track"

    freq = counts / n_tracks
    kind = "Lows" if args.kind == "L" else "Highs"
    season = f", {args.season}" if args.season else ""
    np.savez_compressed(
        args.out,
        freq=freq,
        counts=counts,
        n_analyses=n_tracks,
        n_bulletins=n_tracks,
        cell_km=grid.cell_km,
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        crs=grid.crs.to_proj4(),
        label=f"{kind} track {args.mode}{season}",
        unit=unit,
        n_label="tracks",
    )
    print(
        f"{n_tracks} tracks -> {args.out} "
        f"(peak {freq.max():.3f} {unit})"
    )


def _draw_map_furniture(ax, grid, args, extent):
    """Draw graticule, coastlines, borders and anchors; return the transformer.

    Shared by every map this script makes, so a density field and a bundle of
    cyclone tracks land on visibly the same map rather than on two maps that
    merely resemble each other.
    """
    from matplotlib.collections import LineCollection

    x_min, x_max, y_min, y_max = extent
    fwd = grid.transformer()

    def inside(gx, gy):
        return (gx > x_min) & (gx < x_max) & (gy > y_min) & (gy < y_max)

    # Graticule: project constant-lat and constant-lon lines into grid metres.
    # Labeled, because an unlabeled equal-area map of an unfamiliar domain is
    # very hard to read a pattern off.
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
        for name, pieces in bm.projected_segments(ne_dir, fwd).items():
            if pieces:
                ax.add_collection(LineCollection(pieces, zorder=4, **styles[name]))

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
    return fwd


def cmd_plot_tracks(args: argparse.Namespace) -> None:
    """Draw individual track paths on the map.

    A spaghetti plot rather than a density: the point is to look at the
    tracks themselves, one line per system, and see whether they are shaped
    like cyclones. Quasi-stationary features are excluded by default, since
    otherwise several thousand knots of thermal low sit over the Southwest
    and hide everything that actually travels.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    source = Path(args.source)
    tracks = pd.read_parquet(source / "tracks.parquet")
    stats = pd.read_parquet(source / "track_stats.parquet")

    keep = (stats.n_steps >= args.min_steps) & (stats.net_km >= args.min_net_km)
    if args.season:
        keep &= stats.genesis_time.dt.month.isin(SEASONS[args.season])
    selected = stats[keep]
    if selected.empty:
        sys.exit("no tracks match that selection")

    # A deterministic sample: 90k lines is neither legible nor quick, and a
    # fixed seed keeps successive runs comparable.
    if args.limit and len(selected) > args.limit:
        selected = selected.sample(args.limit, random_state=0)

    paths = tracks[tracks.track_id.isin(set(selected.track_id))]
    paths = paths.sort_values(["track_id", "track_step"])

    grid = g.Grid(cell_km=args.cell_km)
    extent = (grid.x_min, grid.x_max, grid.y_min, grid.y_max)

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=args.dpi)
    ax.set_facecolor("white")
    fwd = _draw_map_furniture(ax, grid, args, extent)

    x, y = fwd.transform(paths.lon.to_numpy(), paths.lat.to_numpy())
    ids = paths.track_id.to_numpy()
    # Split at track boundaries so consecutive systems are not joined by a
    # line across the continent.
    breaks = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    segments = [
        seg for seg in np.split(np.column_stack([x, y]), breaks) if len(seg) >= 2
    ]

    # Colour by minimum central pressure: the deep systems are the ones worth
    # picking out of the tangle.
    depth = selected.set_index("track_id").min_pressure_hpa
    order = pd.unique(ids)
    colours = depth.reindex(order).to_numpy()

    lines = LineCollection(
        segments, cmap=args.cmap, linewidths=args.linewidth,
        alpha=args.alpha, zorder=3,
    )
    lines.set_array(colours[: len(segments)])
    ax.add_collection(lines)

    if args.genesis:
        gx, gy = fwd.transform(
            selected.genesis_lon.to_numpy(), selected.genesis_lat.to_numpy()
        )
        ax.plot(gx, gy, ".", ms=1.6, color="0.15", alpha=0.5, zorder=5)

    kind = "Lows" if args.kind == "L" else "Highs"
    season = f", {args.season}" if args.season else ""
    ax.set_title(
        f"{kind} tracks{season}\n"
        f"{len(segments):,} of {keep.sum():,} tracks with "
        f"{args.min_steps}+ steps and {args.min_net_km:g}+ km displacement"
    )
    fig.colorbar(
        lines, ax=ax, shrink=0.7, label="minimum central pressure (hPa)"
    )
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(segments)} tracks drawn)")


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
    is_intensity = "mean_hpa" in z
    field = z["mean_hpa"] if is_intensity else z["freq"]
    x_min, x_max, y_min, y_max = z["extent"]
    label = str(z["label"]) if "label" in z else Path(args.source).stem
    n = int(z["n_analyses"]) if "n_analyses" in z else int(z["n_bulletins"])
    unit = str(z["unit"]) if "unit" in z else "crossings per analysis"
    # Track grids are counted in tracks, not analyses; saying "analyses" for
    # them would misstate the sample size in the one place a reader checks it.
    n_label = str(z["n_label"]) if "n_label" in z else "analyses"

    if is_intensity:
        # Pressure is not a density: it has no meaningful zero, empty cells
        # are already NaN, and the scale should follow the data rather than
        # start at the origin.
        display, vmin, vmax = field, None, None
        cbar_label = "mean central pressure (hPa)"
    else:
        # Density fields are heavy-tailed -- a handful of cells sit far above
        # the rest and, left alone, compress everything else into one flat
        # colour. Clipping at a high percentile of the occupied cells shows
        # the pattern, and the peak goes in the title so nothing is hidden.
        display = np.where(field > 0, field, np.nan)
        occupied = field[field > 0]
        vmin = 0.0
        vmax = (
            float(np.percentile(occupied, args.vmax_pct))
            if occupied.size
            else None
        )
        cbar_label = unit

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=args.dpi)
    cmap = args.cmap
    if is_intensity and args.cmap == "magma_r":
        cmap = "RdBu_r"  # neutral diverging default for pressure, not density
    mesh = ax.imshow(
        display,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    grid = g.Grid(cell_km=float(z["cell_km"]))
    fwd = _draw_map_furniture(ax, grid, args, (x_min, x_max, y_min, y_max))

    subtitle = f"{n:,} {n_label}, {grid.cell_km:g} km equal-area cells"
    if not is_intensity:
        subtitle += f", peak {np.nanmax(field):.3f}"
    ax.set_title(f"{label}\n{subtitle}")
    fig.colorbar(
        mesh, ax=ax, shrink=0.7, label=cbar_label,
        extend="max" if vmax is not None else "neither",
    )
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
    p.add_argument("--res", default="HR", choices=["HR", "LR"])
    p.add_argument("--season", choices=list(SEASONS), help="DJF, MAM, JJA or SON")
    p.add_argument("--cell-km", type=float, default=150.0)
    p.add_argument("--out", default="centers.npz")
    p.add_argument("--smooth-km", type=float, default=0.0)
    p.add_argument("--min-count", type=int, default=5)
    p.set_defaults(func=cmd_centers)

    p = sub.add_parser("intensity", help="mean central pressure grid")
    p.add_argument("source", help="Parquet directory from ingest")
    p.add_argument("--kind", required=True, choices=["H", "L"])
    p.add_argument("--res", default="HR", choices=["HR", "LR"])
    p.add_argument("--season", choices=list(SEASONS), help="DJF, MAM, JJA or SON")
    p.add_argument("--cell-km", type=float, default=250.0)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--out", default="intensity.npz")
    p.set_defaults(func=cmd_intensity)

    p = sub.add_parser("track", help="link pressure centers into tracks over time")
    p.add_argument("source", help="Parquet directory from ingest")
    p.add_argument("--kind", required=True, choices=["H", "L"])
    p.add_argument("--res", default="HR", choices=["HR", "LR"])
    p.add_argument(
        "--max-gap-hours", type=float, default=6.0,
        help="never link centers across a longer gap than this",
    )
    p.add_argument("--out", default="data/tracks")
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("trackgrid", help="grid tracks into a density .npz")
    p.add_argument("source", default="data/tracks", nargs="?",
                   help="directory written by track")
    p.add_argument("--mode", default="path", choices=["path", "genesis", "lysis"])
    p.add_argument("--kind", default="L", choices=["H", "L"], help="for the title")
    p.add_argument("--season", choices=list(SEASONS))
    p.add_argument("--min-steps", type=int, default=8)
    p.add_argument("--min-net-km", type=float, default=500.0)
    p.add_argument("--cell-km", type=float, default=150.0)
    p.add_argument("--out", default="trackgrid.npz")
    p.set_defaults(func=cmd_trackgrid)

    p = sub.add_parser("plot-tracks", help="draw track paths on the map")
    p.add_argument("source", default="data/tracks", nargs="?",
                   help="directory written by track")
    p.add_argument("--kind", default="L", choices=["H", "L"], help="for the title")
    p.add_argument("--season", choices=list(SEASONS))
    p.add_argument("--min-steps", type=int, default=8,
                   help="at 3-hourly, 8 steps is a day")
    p.add_argument("--min-net-km", type=float, default=500.0,
                   help="exclude quasi-stationary features")
    p.add_argument("--limit", type=int, default=1500,
                   help="sample this many tracks; 0 draws all")
    p.add_argument("--genesis", action="store_true", help="mark genesis points")
    p.add_argument("--cell-km", type=float, default=50.0, help="map extent only")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--linewidth", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--out", default="tracks.png")
    p.add_argument("--anchors", action="store_true")
    p.add_argument("--ne-dir", default="data/ne")
    p.add_argument("--no-basemap", action="store_true")
    p.set_defaults(func=cmd_plot_tracks)

    p = sub.add_parser("basemap", help="fetch Natural Earth outlines for plotting")
    p.add_argument("--dest", default="data/ne")
    p.add_argument("--force", action="store_true", help="re-download existing layers")
    p.set_defaults(func=cmd_basemap)

    p = sub.add_parser("plot", help="render a density .npz to PNG")
    p.add_argument("source", help=".npz written by density")
    p.add_argument("--out", default="density.png")
    p.add_argument("--cmap", default="magma_r")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument(
        "--vmax-pct", type=float, default=99.0,
        help="clip the colour scale at this percentile of occupied cells",
    )
    p.add_argument("--anchors", action="store_true", help="draw city reference points")
    p.add_argument("--ne-dir", default="data/ne", help="Natural Earth GeoJSON directory")
    p.add_argument("--no-basemap", action="store_true", help="omit coastlines and borders")
    p.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
