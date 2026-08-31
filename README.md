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

The whole thing is about 55 MB. For anything current, MetPy has
`parse_wpc_surface_bulletin`.

```bash
mkdir -p data/raw && cd data/raw
# download and unpack the JSON archive here
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

## Running it

```bash
python scripts/build.py ingest data/raw/json data/parquet
python scripts/build.py density data/parquet --ftype COLD --month 3 --out mar_cold.npz
```

Ingest is the only slow step. After that everything reads Parquet, and DuckDB
queries it in place without a load step.

## Design notes

**Longitude sign.** Coded bulletins encode longitude as positive degrees west.
The loader converts to signed degrees east on the way in. Getting this wrong
puts your entire climatology in Asia, and the failure is obvious enough that
you would catch it — the subtler version is mixing conventions between the
centers and points tables.

**Equal-area grid.** Density is accumulated on a Lambert Azimuthal Equal Area
grid centered at 45N/100W. On a plate carrée grid, a cell at 60N covers about
half the ground of one at 25N, which inflates apparent frontal frequency
toward the pole for free.

**Constant-arc-length resampling.** Analysts place vertices where a front
bends, so vertex density tracks pen habits rather than frontal frequency.
Every polyline is resampled to fixed great-circle spacing before gridding.

**One count per cell per feature.** After resampling, a meandering front would
deposit several points in the same cell. Taking unique cells makes the
statistic "did a front cross this cell," which is the quantity that means
something physically.

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
