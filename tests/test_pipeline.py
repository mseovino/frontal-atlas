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
