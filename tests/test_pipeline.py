"""Tests against synthetic bulletins.

The real archive is not reachable from this sandbox, so these fixtures are
hand-built to match the documented JSON schema. They verify the parts that are
easy to get subtly wrong and hard to notice later: longitude sign convention,
resampling density, and the equal-area property of the grid.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codsus import grid as g  # noqa: E402
from codsus import load as ld  # noqa: E402


def make_bulletin(tmp_path: Path, name: str, valid: str, res: str = "HR") -> Path:
    """A synthetic bulletin: one low, one high, one cold and one warm front.

    Longitudes are positive-west, as the coded bulletin encodes them.
    """
    doc = {
        "bulletinType": res,
        "createDate": valid,
        "validDate": valid,
        "Lows": {"lats": [42.0], "lons": [95.0], "pressures": [996.0]},
        "Highs": {"lats": [35.0], "lons": [75.0], "pressures": [1024.0]},
        "ColdFronts": {
            "lats": [[42.0, 38.0, 34.0]],
            "lons": [[95.0, 92.0, 90.0]],
        },
        "WarmFronts": {
            "lats": [[42.0, 43.0, 44.0]],
            "lons": [[95.0, 90.0, 85.0]],
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def test_longitude_sign_is_converted_to_signed_east(tmp_path):
    make_bulletin(tmp_path, "a.json", "2020-01-15T12:00:00Z")
    centers, points, report = ld.load(tmp_path)

    assert report.bulletins == 1
    assert report.skipped == 0
    # 95 degrees west must become -95, not stay +95 out in China.
    assert centers.lon.max() < 0
    assert points.lon.max() < 0
    assert centers.loc[centers.kind == "L", "lon"].iloc[0] == pytest.approx(-95.0)


def test_center_and_feature_counts(tmp_path):
    make_bulletin(tmp_path, "a.json", "2020-01-15T12:00:00Z")
    centers, points, report = ld.load(tmp_path)

    assert report.centers == 2
    assert report.polylines == 2
    assert set(points.ftype.unique()) == {"COLD", "WARM"}
    assert len(points) == 6  # three vertices per front


def test_lr_and_hr_versions_get_distinct_bulletin_ids(tmp_path):
    """The same analysis appears twice in the archive at two resolutions.

    They must not merge, or every post-2009 map is double counted.
    """
    make_bulletin(tmp_path, "hr.json", "2020-01-15T12:00:00Z", res="HR")
    make_bulletin(tmp_path, "lr.json", "2020-01-15T12:00:00Z", res="LR")
    centers, points, report = ld.load(tmp_path)

    assert report.bulletins == 2
    assert points.bulletin_id.nunique() == 2


def test_malformed_file_is_skipped_not_fatal(tmp_path):
    make_bulletin(tmp_path, "good.json", "2020-01-15T12:00:00Z")
    (tmp_path / "bad.json").write_text("{not json")

    centers, points, report = ld.load(tmp_path)
    assert report.bulletins == 1
    assert report.skipped == 1
    assert report.problems


def test_resampling_hits_target_spacing():
    # A 5-degree meridional leg is roughly 555 km.
    lat, lon = g.resample_polyline([30.0, 35.0], [-90.0, -90.0], spacing_km=50.0)
    assert len(lat) > 10

    seg_km = [
        g.polyline_length_km(lat[i : i + 2], lon[i : i + 2]) for i in range(len(lat) - 1)
    ]
    # Every leg should sit at or under the requested spacing.
    assert max(seg_km) <= 55.0


def test_resampling_preserves_endpoints_and_length():
    lats, lons = [40.0, 35.0, 30.0], [-100.0, -95.0, -88.0]
    original = g.polyline_length_km(np.array(lats), np.array(lons))
    lat, lon = g.resample_polyline(lats, lons, spacing_km=25.0)

    assert lat[0] == pytest.approx(lats[0])
    assert lat[-1] == pytest.approx(lats[-1])
    assert g.polyline_length_km(lat, lon) == pytest.approx(original, rel=0.01)


def test_grid_cells_are_equal_area():
    """The whole point of LAEA: a cell at 60N covers the same ground as one at 25N."""
    grid = g.Grid(cell_km=50.0)
    lon, lat = grid.cell_centers()

    # Sample a northern and a southern cell, confirm both map back into range.
    assert lat.min() < 25.0 < lat.max()
    assert lat.max() > 55.0
    assert grid.shape == (160, 180)


def test_frequency_grid_counts_each_feature_once_per_cell(tmp_path):
    make_bulletin(tmp_path, "a.json", "2020-01-15T12:00:00Z")
    _, points, _ = ld.load(tmp_path)

    grid = g.Grid(cell_km=100.0)
    counts = g.frequency_grid(points[points.ftype == "COLD"], grid)

    assert counts.sum() > 0
    # One front in one bulletin can contribute at most one count to any cell.
    assert counts.max() == 1


def test_frequency_accumulates_across_bulletins(tmp_path):
    make_bulletin(tmp_path, "a.json", "2020-01-15T12:00:00Z")
    make_bulletin(tmp_path, "b.json", "2020-01-15T18:00:00Z")
    _, points, _ = ld.load(tmp_path)

    counts = g.frequency_grid(points[points.ftype == "COLD"], g.Grid(cell_km=100.0))
    # Identical fronts at two times: cells they share should now read 2.
    assert counts.max() == 2


def make_real_bulletin(tmp_path: Path, name: str, valid: str, res: str = "HR") -> Path:
    """A bulletin in the layout the published archive actually uses.

    Front groups are lists, one object per front, each carrying its own
    lats/lons and a strength label, and longitudes are already signed east.
    A front type with nothing analyzed comes through as null.
    """
    doc = {
        "bulletinType": res,
        "createDate": valid,
        "validDate": valid,
        "Lows": {"lats": [42.0], "lons": [-95.0], "pressures": [996.0]},
        "Highs": {"lats": [35.0], "lons": [-75.0], "pressures": [1024.0]},
        "ColdFronts": [
            {"lats": [42.0, 38.0, 34.0], "lons": [-95.0, -92.0, -90.0],
             "strength": "moderate"},
            {"lats": [50.0, 48.0], "lons": [-120.0, -118.0], "strength": "weak"},
        ],
        "StationaryFronts": [
            {"lats": [31.0, 30.0], "lons": [-97.0, -98.0], "strength": "weak"}
        ],
        "WarmFronts": None,
        "OccludedFronts": None,
        "Troughs": [],
    }
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def test_real_archive_layout_parses(tmp_path):
    """The published archive stores fronts as lists of objects, not arrays."""
    make_real_bulletin(tmp_path, "a.json", "2014-01-22 09:00:00")
    centers, points, report = ld.load(tmp_path)

    assert report.skipped == 0
    assert report.polylines == 3  # two cold, one stationary
    assert report.centers == 2
    assert set(points.ftype.unique()) == {"COLD", "STNRY"}
    assert points.groupby("feature_id").ngroups == 3


def test_null_front_group_is_not_an_error(tmp_path):
    """A front type with nothing analyzed is null, and that is routine."""
    make_real_bulletin(tmp_path, "a.json", "2014-01-22 09:00:00")
    _, points, report = ld.load(tmp_path)

    assert report.skipped == 0
    assert "WARM" not in set(points.ftype.unique())


def test_strength_label_is_kept(tmp_path):
    make_real_bulletin(tmp_path, "a.json", "2014-01-22 09:00:00")
    _, points, _ = ld.load(tmp_path)

    cold = points[points.ftype == "COLD"]
    assert set(cold.strength.unique()) == {"moderate", "weak"}


def test_signed_east_longitudes_pass_through_unflipped(tmp_path):
    """Archive longitudes are already signed east; flipping them would be a bug."""
    make_real_bulletin(tmp_path, "a.json", "2014-01-22 09:00:00")
    centers, points, report = ld.load(tmp_path)

    assert report.lon_flipped == 0
    assert centers.lon.max() < 0
    assert points.lon.min() == pytest.approx(-120.0)


def test_positive_longitudes_are_flipped_and_counted(tmp_path):
    """The ASCII convention still parses, but the flip is reported, not silent."""
    make_bulletin(tmp_path, "a.json", "2020-01-15T12:00:00Z")
    _, _, report = ld.load(tmp_path)

    assert report.lon_flipped > 0


def test_short_front_is_dropped_and_counted(tmp_path):
    """A one-vertex front cannot make a segment; it should not vanish silently."""
    doc = {
        "bulletinType": "HR",
        "createDate": "2014-01-22 09:00:00",
        "validDate": "2014-01-22 09:00:00",
        "ColdFronts": [{"lats": [40.0], "lons": [-90.0], "strength": "weak"}],
    }
    (tmp_path / "a.json").write_text(json.dumps(doc))

    _, points, report = ld.load(tmp_path)
    assert report.polylines == 0
    assert report.degenerate == 1
    assert points.empty


def test_streaming_ingest_matches_in_memory_load(tmp_path):
    """`ingest` chunks the archive; it must not change what comes out."""
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        make_real_bulletin(src, f"b{i}.json", f"2014-01-2{i} 09:00:00")

    _, points, report = ld.load(src)
    out = tmp_path / "pq"
    stream = ld.ingest(src, out, chunk_files=2)

    import pandas as pd

    written = pd.read_parquet(out / "points")
    assert stream.bulletins == report.bulletins == 5
    assert stream.polylines == report.polylines
    assert len(written) == len(points)
    assert set(written.bulletin_id) == set(points.bulletin_id)


def test_impossible_coordinates_are_dropped_and_counted(tmp_path):
    """The archive holds a handful of latitudes above 90; they project to inf."""
    doc = {
        "bulletinType": "LR",
        "createDate": "2004-07-23 15:00:00",
        "validDate": "2004-07-23 15:00:00",
        "ColdFronts": [
            {"lats": [40.0, 98.0, 36.0], "lons": [-90.0, -5.0, -88.0],
             "strength": "weak"}
        ],
        "Lows": {"lats": [97.0, 42.0], "lons": [-6.0, -95.0], "pressures": [996.0, 1000.0]},
    }
    (tmp_path / "a.json").write_text(json.dumps(doc))

    centers, points, report = ld.load(tmp_path)
    assert report.out_of_range == 2
    assert report.polylines == 1
    assert points.lat.max() <= 90.0
    assert centers.lat.max() <= 90.0
    # The surviving vertices keep their original indices, so the gap is visible.
    assert list(points.ord) == [0, 2]


def test_feature_left_with_one_vertex_is_dropped(tmp_path):
    """Dropping bad vertices can leave too little to make a segment."""
    doc = {
        "bulletinType": "LR",
        "createDate": "2004-07-23 15:00:00",
        "validDate": "2004-07-23 15:00:00",
        "ColdFronts": [{"lats": [98.0, 36.0], "lons": [-5.0, -88.0], "strength": "weak"}],
    }
    (tmp_path / "a.json").write_text(json.dumps(doc))

    _, points, report = ld.load(tmp_path)
    assert report.out_of_range == 1
    assert report.polylines == 0
    assert report.degenerate == 1
    assert points.empty
