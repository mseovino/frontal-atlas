"""Dominant-boundary-type maps, with and without troughs, across cell sizes.

For a categorical argmax map the thing that degrades with resolution is not
the count in a cell but whether the winner is really winning. A cell where
cold fronts beat stationary fronts 41 to 39 is a coin flip being drawn as a
hard colour. So the resolution question is answered by the margin between
first and second place, not by the raw sample size.

A cell counts as decided when the winner leads the runner-up by more than two
standard deviations of the difference of two Poisson counts, sqrt(n1 + n2).
"""
import pickle
import sys


import numpy as np
from common import OUT
import pyarrow.dataset as pads

from codsus import fronts_xml as fx
from codsus import grid as g

TYPES = ["COLD", "WARM", "STNRY", "OCFNT", "TROF", "DRYLN"]
CELLS = [100, 75, 50, 35, 25]

dset = pads.dataset("data/parquet/points", format="parquet", partitioning="hive")

counts = {km: {} for km in CELLS}
for t in TYPES:
    df = dset.to_table(
        columns=["valid_time", "ftype", "stage", "lat_e2", "lon_e2"],
        filter=pads.field("ftype") == t,
    ).to_pandas()
    pts = fx.explode(df)
    print(f"{t}: {len(pts):,} vertices", flush=True)
    for km in CELLS:
        counts[km][t] = g.frequency_grid(pts, g.Grid(cell_km=km))
    del df, pts

with open(OUT / "dom_counts.pkl", "wb") as fh:
    pickle.dump(counts, fh)

print(f"\n{'cells':>6s} {'variant':>12s} {'mapped':>9s} {'decided':>9s} {'median n1':>10s}")
for km in CELLS:
    for variant, keep in [("with trough", TYPES),
                          ("no trough", [t for t in TYPES if t != "TROF"])]:
        stack = np.stack([counts[km][t] for t in keep]).astype(float)
        total = stack.sum(axis=0)
        # Enough data in the cell for the question to mean anything at all.
        support = total >= 100
        srt = np.sort(stack, axis=0)
        n1, n2 = srt[-1], srt[-2]
        decided = support & ((n1 - n2) > 2.0 * np.sqrt(n1 + n2))
        print(f"{km:5d}km {variant:>12s} {support.sum():9,} "
              f"{100*decided.sum()/max(support.sum(),1):8.0f}% {np.median(n1[support]):10.0f}")
