# frontal-atlas

A climatology of North American fronts and pressure centers built from the
NWS Coded Surface Bulletin — the machine-readable record of every surface
analysis WPC has drawn since 2003.

Every published frontal climatology is objective, derived from reanalysis
thermal-gradient fields. This one is built from what human analysts actually
drew. The difference between the two is the interesting part.

## Getting the data

The bulletins are ASCII files giving locations of fronts, troughs, and high
and low centers, broadcast on NOAAPort since 2003 at 1° lat/lon precision,
with a 0.1° high-resolution version added in 2009. WPC serves only a rolling
two-week window, so use the archive NCICS published on Zenodo:

- JSON, ready to use: https://zenodo.org/records/2646544
- Raw ASCII, if you want to parse it yourself: https://zenodo.org/records/2642801

The JSON archive is a single 77 MB tarball, `CODSUS_JSON_2003-2018.tgz`,
CC-BY-SA-4.0, unpacking to 75,336 files under `CODSUS_JSON/{LR,HR}/YYYY/MM/`.
Note the end date: **the archive stops at 2018-12-31**, so "2003–" means
2003–2018 here, and extending to the present means a separate ingest from
another source. For anything current, MetPy has `parse_wpc_surface_bulletin`.

```bash
mkdir -p data/raw
curl -L -o data/raw/CODSUS_JSON_2003-2018.tgz \
  https://zenodo.org/api/records/2646544/files/CODSUS_JSON_2003-2018.tgz/content
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
python scripts/build.py ingest data/raw/json data/parquet --progress
python scripts/build.py density data/parquet --ftype COLD --month 3 --out mar_cold.npz
python scripts/build.py plot mar_cold.npz --out mar_cold.png
```

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

**Never mix LR and HR.** The same analysis appears in the archive at both
resolutions, and they carry distinct `bulletin_id`s so they cannot silently
merge. The 2009 resolution change is a real discontinuity in the record —
restrict fine-scale work to HR (2009+), or degrade everything to LR for the
full period. Do not straddle it.

## Roadmap

- [x] Loader, tidy tables, Parquet output
- [x] Equal-area frontal density
- [ ] Cyclone tracking: link centers across bulletins (~400 km/3h cap plus a
      pressure-continuity term), then track/genesis/lysis density and
      deepening rates
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
