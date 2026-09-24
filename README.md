# frontal-atlas

A climatology of North American fronts and pressure centers built from what
WPC's surface analysts actually drew: sixteen years of hand analyses, counted.

Every published frontal climatology is objective, derived from reanalysis
thermal-gradient fields. This one is built from the analyses themselves, which
makes it two things at once: a climatology of the atmosphere, and a record of
analysis practice. Several findings here turned out to be the second kind, and
`analysis/` keeps the scripts that separate them.

## Getting the data

Two archives, one per job.

**Fronts** come from the NOAA Unified Surface Analysis front archive (Zenodo
[7505022](https://zenodo.org/records/7505022), CC-BY-4.0): one XML per
analysis, December 2006 to December 2022, exported from the graphical analysis
files with each feature's type intact. Unlike the coded bulletin below, it keeps
drylines and squall lines separate from troughs and marks fronts as developing
or weakening. `ingest-fronts` reads the 148 MB tarball in place and writes a
55 MB row-per-polyline Parquet store at 0.01 degree precision.

**Pressure centers** come from the NWS Coded Surface Bulletin archive NCICS
published on Zenodo:

- JSON, ready to use: https://zenodo.org/records/2646544
- Raw ASCII, if you want to parse it yourself: https://zenodo.org/records/2642801

The JSON archive is a single 77 MB tarball, `CODSUS_JSON_2003-2018.tgz`,
CC-BY-SA-4.0, covering 2003-2018 at 1 degree precision with a 0.1 degree
high-resolution version from 2009. Its fronts are not used any more: the
bulletin format codes troughs, outflow boundaries, squall lines and drylines
all as `TROF`, so no dryline can be recovered from it.

```bash
mkdir -p data/raw
curl -L -o data/raw/front_xmls.tar.gz   "https://zenodo.org/api/records/7505022/files/front_xmls.tar.gz/content"
curl -L -o data/raw/CODSUS_JSON_2003-2018.tgz   https://zenodo.org/api/records/2646544/files/CODSUS_JSON_2003-2018.tgz/content
mkdir -p data/raw/json && tar xzf data/raw/CODSUS_JSON_2003-2018.tgz -C data/raw/json
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

Without `uv`, the stdlib route works the same. On Windows the venv puts
executables in `.venv/Scripts` rather than `.venv/bin`:

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -e ".[dev]"
```

## Running it

```bash
python scripts/build.py basemap                     # once: coastlines and borders
python scripts/build.py ingest-fronts data/raw/front_xmls.tar.gz data/parquet/points --progress
python scripts/build.py ingest data/raw/json data/parquet --progress   # pressure centers
python scripts/build.py density data/parquet --ftype COLD --month 3 --out mar_cold.npz
python scripts/build.py plot mar_cold.npz --out mar_cold.png
```

Pressure centers have their own three commands, all of which take
`--kind H|L`, `--res`, and `--season`:

```bash
python scripts/build.py centers   data/parquet --kind L --res HR --season DJF --out lows_djf.npz
python scripts/build.py intensity data/parquet --kind L --res HR --season DJF --out lows_djf_p.npz
python scripts/build.py track     data/parquet --kind L --res HR --out data/tracks
```

`centers` counts centers per cell, `intensity` averages central pressure, and
`track` links centers across analyses into tracks (Hungarian assignment on
distance plus a pressure-jump penalty, under a speed cap). Tracking the full
HR low record takes about 18 seconds and yields ~90,600 tracks.

Once tracks exist there are two ways to look at them:

```bash
# the tracks themselves, one line per system, coloured by depth
python scripts/build.py plot-tracks data/tracks --season DJF --out tracks.png

# tracks as a griddable field, rendered by the ordinary plot command
python scripts/build.py trackgrid data/tracks --mode path    --season DJF --out storm_track.npz
python scripts/build.py trackgrid data/tracks --mode genesis --season DJF --out genesis.npz
python scripts/build.py plot storm_track.npz --out storm_track.png
```

`trackgrid --mode` is `path` (how often a track crosses each cell — the storm
track), `genesis`, or `lysis`. Both commands default to tracks of 8+ steps
and 500+ km displacement, which is what excludes the quasi-stationary
features described below; pass `--min-net-km 0` to see everything.

`density` takes `--month N` or `--season DJF|MAM|JJA|SON`. `basemap` fetches
3.3 MB of Natural Earth 1:50m line layers (coastline, national borders,
states and provinces) into `data/ne/`. It is a separate command on purpose:
plotting never reaches the network, and if you skip it, `plot` falls back to
a labeled graticule and a dozen city markers.

Ingest is the only slow step: roughly 12 minutes for the full archive,
producing about 12.7M vertex rows and 1.7M pressure centers. It streams in
file chunks and appends year-partitioned Parquet, because the whole thing does
not fit in memory as Python objects. After that everything reads Parquet, and
DuckDB queries it in place without a load step.

## Analysis scripts

`scripts/build.py` builds the stores and the standard grids. Everything past
that lives in `analysis/`, one script per question, each run from the repo root
as `python analysis/<name>.py`. Generated figures, grids and pickles go to
`outputs/`, which is gitignored.

| Script | Question |
|---|---|
| `atlas_synoptic.py` | Frequency of each front type by season, synoptic hours only |
| `handbook_data.py`, `handbook_figures.py` | Monthly fronts, highs and lows, central-pressure ranges, real cases, dryline and squall-line data; one consistent figure style |
| `type_counts.py`, `front_share.py` | Per-type counts at several cell sizes; where one type is common absolutely and as a share of all fronts |
| `composite_build.py`, `composite_neighbours.py` | Front positions relative to the analysed low, split by depth and by whether another low is nearby |
| `composite_final.py`, `composite_final_figs.py` | The finished composite: triple points found from the drawn fronts, the full ocean storm tracks, each low rotated onto its direction of travel from its track |
| `composite_neighbour_stats.py`, `composite_truncation.py` | How crowded analysed lows are; how many composite lows sit near a chart edge |
| `dryline_position.py` | Dryline longitude by month and hour at four latitudes |
| `squall_lines.py` | Where and when squall lines are drawn |
| `lee_trough.py`, `cad_proxy_sweep.py`, `cad_trough_test.py` | The Appalachian lee trough, and why damming cannot be separated from it in this archive |
| `seasonal_march.py` | 31-day running-window animation of front frequency |
| `passage_climatology.py`, `passage_sensitivity.py` | Frontal passages at one point, and how much the count depends on radius and merge tolerance |
| `hour_domain.py`, `manual_checks.py` | Synoptic versus intermediate hours: coverage, and the per-type effect counted by feature centroid |
| `stage_detection.py` | Why weakening fronts are tagged three times as often as developing ones |
| `trend_drawn_length.py`, `trend_breakpoint.py`, `trough_trend_map.py` | How much is drawn per map over time, and when the trough increase happened |
| `consistency_monitor.py` | Monthly drawing-consistency check: drawn length, piece length, hour ratios and stage use, flagged against both the recent past and a fixed reference period |
| `expect_data.py`, `expect_page.py` | Month-by-month maps and seasonal central-pressure ranges for the training page |

Three rules came out of this work and every script follows them. Count
features by where they are centred, not by whether they touch a region:
overlap counting pulls in long ocean fronts. Use the four synoptic hours for
anything per-type, because intermediate maps cover less area and carry more
stationary fronts and troughs. And read every trend against the dates of known
changes at the desk before calling it atmospheric.

## Design notes

**Longitude sign.** The raw ASCII bulletins encode longitude as positive
degrees west. The NCICS **JSON archive has already converted it** — every
longitude in it is negative, spanning 180W to about 1W — so for that source
the loader passes values through untouched. It still flips a positive value,
on the assumption that it came from the ASCII convention, but it counts every
flip in the load report rather than doing it silently: a genuinely
east-of-Greenwich point would be mangled by the same rule, and that belongs in
a number you can check, not in a map that looks plausible. A full ingest of
the JSON archive reports zero flips. If yours doesn't, find out why before
trusting the output.

**Legacy CODSUS notes.** The next few notes describe the coded-bulletin ingest,
which now supplies pressure centers only.

**Front groups are lists, not arrays.** Pressure-center groups really are
objects of parallel arrays, `{lats, lons, pressures}`. Front groups are not,
despite what the schema description suggests: each is a *list* with one object
per front, `{lats, lons, strength}`, and it is `null` when no front of that
type was analyzed — routine, not an error. The loader accepts that layout plus
the nested and flat object forms, so fixtures written from the documentation
still parse. The `strength` label is carried through to the points table and
is not used by anything yet.

**Equal-area grid.** Density is accumulated on a Lambert Azimuthal Equal Area
grid centered at 45N/100W. On a plate carrée grid, a cell at 60N covers about
half the ground of one at 25N, which inflates apparent frontal frequency
toward the pole for free.

**Constant-arc-length resampling.** Analysts place vertices where a front
bends, so vertex density tracks pen habits rather than frontal frequency.
Every polyline is resampled to fixed great-circle spacing before gridding.

**The denominator is every analysis, not every hit.** Frequency is crossings
divided by the number of analyses in the window, including the analyses in
which no front of that type was drawn anywhere. Dividing instead by the
analyses that contained the type answers a different and much less useful
question — "how far did a cold front reach, given that one was drawn at all" —
and it inflates exactly those types and seasons that are intermittent, which
makes seasons and front types non-comparable. `density` prints both counts so
the gap between them is visible.

**Gridding is vectorized; resampling is the reference.** `frequency_grid`
densifies in the projected plane over flat arrays rather than looping features
and calling pyproj per leg — on one March of HR cold fronts that is 0.12 s
against 34 s, and a season of troughs is 130k features, where the loop simply
does not finish. The two agree to a correlation of 0.997; the small residual
is the vectorized version catching *more* crossings, because the old
`int(dist/spacing) - 1` could leave a leg unsampled and step over a cell.
Straight lines in an equal-area projection sit well under a kilometre from the
great circle over the 10–100 km legs bulletins contain, so nothing is lost;
`resample_polyline` stays as the geodesic reference and is what the resampling
tests exercise.

**One count per cell per feature.** After resampling, a meandering front would
deposit several points in the same cell. Taking unique cells makes the
statistic "did a front cross this cell," which is the quantity that means
something physically.

**Match the cell size to the resolution.** LR vertices are snapped to whole
degrees, so a great many legs are drawn exactly along a parallel and fall
entirely inside one row of cells. Grid LR at 50 km and the field breaks into
latitude stripes that are pure quantization artifact — visibly so, and the
striping vanishes at 150 km, where the large-scale pattern matches HR. One
degree of latitude is about 111 km, so that is the floor for LR. HR (0.1°)
is fine at 50 km. `density` warns if you ask for LR below 111 km.

**Impossible coordinates exist.** Ten vertices and one pressure center in the
archive carry latitudes between 91 and 98 degrees, all in LR bulletins from
2003–2005 near the eastern edge of the domain — transcription errors in the
source. They project to infinity. The loader drops them at the door and counts
them in `LoadReport.out_of_range`; `ord` keeps the original vertex index so
the gap in a line is visible rather than renumbered away.

**A long track is not necessarily a cyclone.** Nearest-neighbour linking
assumes a center is a travelling system, and for quasi-stationary features
that assumption fails completely. The longest low track in the HR record runs
**45 days and displaces 14 km**: it is the Mojave thermal low, re-analysed in
the same place every three hours all summer. It is a real track of a real
analysed feature and it is not a cyclone. Of the 12,768 low tracks lasting a
day or more, **36% displace under 500 km**. `track_stats` therefore reports
`net_km`, `path_km` and `mean_speed_kmh`, and `track` prints the stationary
fraction — filter on motion before computing anything about lifetimes,
deepening rates, or genesis density.

**Central pressure needs a plausibility filter.** The archive holds 1,115
"lows" above 1030 hPa, 800 "highs" below 1000, and a handful beyond anything
ever observed on Earth, including a low at 1188 hPa — about 0.11% of centers.
The same bad values appear in both the LR and HR copies of an analysis, so
these are errors in the source analysis, not the encoding. They are kept at
ingest and dropped at analysis time by `load.drop_implausible_pressure`, so a
reader can still see that they exist. The low floor is 880 hPa to admit
tropical cyclones, which means it also admits a few bad extratropical values —
an 883 hPa "low" over interior Alaska survives the filter and is certainly
wrong.

**Never mix LR and HR.** The same analysis appears in the archive at both
resolutions, and they carry distinct `bulletin_id`s so they cannot silently
merge. The 2009 resolution change is a real discontinuity in the record —
restrict fine-scale work to HR (2009+), or degrade everything to LR for the
full period. Do not straddle it.

## Roadmap

- [x] Loader, tidy tables, Parquet output
- [x] Equal-area frontal density
- [x] Unified Surface Analysis front ingest, with drylines, squall lines and
      developing/weakening stages kept separate
- [x] Cyclone tracking: link centers across bulletins (~400 km/3h cap plus a
      pressure-continuity term), track and genesis density
- [x] Cyclone-relative frontal composite, finished: triple-point handling,
      the full ocean storm-track termini, rotation onto storm motion
- [x] Drawing-consistency monitor: per-type rates by hour, drawn length and
      stage use over time, so practice changes show up as they happen
- [ ] Fronts and heavy rain: how often extreme precipitation falls near an
      analysed stationary front or trough (needs Stage IV)
- [ ] Front motion: match segments between consecutive analyses, compute
      normal displacement, map where boundaries stall by season — the
      climatological prior for excessive rainfall that doesn't currently exist
- [ ] Type transitions: where cold fronts get relabeled stationary, which is
      essentially a picture of terrain blocking
- [ ] Frontal waves: cyclogenesis points against preexisting boundaries
- [ ] Front rose: passage frequency by type, direction, and speed at a point
- [ ] Web front end (MapLibre over precomputed tiles)

The segment-matching code for front motion is the same operation as scoring a
trainee's hand analysis against the official one. Write it once and the
analysis trainer becomes a second front end on this library rather than a
separate project.

## Caveats worth stating in anything you publish

The record is a record of human analysis. Analyst rosters, training, tooling,
and conventions all changed over 20+ years, and the 2009 resolution jump is a
step change in the data itself. Any trend you find is a mixture of atmospheric
signal and analysis practice, and separating them is a research question, not
a preprocessing step.

Two practice effects are already measured. Trough drawing over the Lower 48
roughly doubled between 2007-10 and 2019-22, in total length and not just
count, with most of the rise after 2017; it is not tied to any known tool
change. And inside WPC's own area, stationary fronts and troughs are drawn
about 20-25% more often on intermediate maps than synoptic ones, while the
classical front types are flat across hours.
