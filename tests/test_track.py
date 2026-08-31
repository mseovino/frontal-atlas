"""Tests for center tracking, intensity gridding, and the pressure filter.

The tracking fixtures are hand-built center tables: a synthetic low walking
east at a known speed, so what should and should not link is arithmetic
rather than judgement.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codsus import grid as g  # noqa: E402
from codsus import load as ld  # noqa: E402
from codsus import track as tk  # noqa: E402


def centers_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a centers table from (hours, lat, lon, pressure) rows."""
    base = pd.Timestamp("2015-01-01T00:00:00Z")
    return pd.DataFrame(
        [
            {
                "bulletin_id": f"b{i}",
                "valid_time": base + pd.Timedelta(hours=r["h"]),
                "res": r.get("res", "HR"),
                "kind": r.get("kind", "L"),
                "lat": r["lat"],
                "lon": r["lon"],
                "pressure_hpa": r.get("p", 1000.0),
            }
            for i, r in enumerate(rows)
        ]
    )


def walking_low(n_steps: int, km_per_step: float, hours: float = 3.0) -> pd.DataFrame:
    """A single low marching east at a fixed rate, one center per analysis."""
    deg_per_step = km_per_step / 78.0  # ~78 km per degree of longitude at 45N
    return centers_frame(
        [
            {"h": i * hours, "lat": 45.0, "lon": -100.0 + i * deg_per_step, "p": 1000.0}
            for i in range(n_steps)
        ]
    )


def test_a_steady_low_becomes_one_track():
    tracks = tk.build_tracks(walking_low(6, 200.0), kind="L", res="HR")

    assert tracks.track_id.nunique() == 1
    assert list(tracks.sort_values("valid_time").track_step) == [0, 1, 2, 3, 4, 5]


def test_a_low_moving_too_fast_does_not_link():
    """The speed cap is 133 km/h; 900 km in 3 hours is 300 km/h."""
    tracks = tk.build_tracks(walking_low(4, 900.0), kind="L", res="HR")

    assert tracks.track_id.nunique() == 4
    assert set(tracks.track_step) == {0}


def test_no_link_across_a_gap_longer_than_the_cap():
    """A 12-hour gap must break the track even though the speed cap alone
    would permit 1,596 km of travel across it."""
    df = centers_frame(
        [
            {"h": 0, "lat": 45.0, "lon": -100.0},
            {"h": 12, "lat": 45.0, "lon": -99.0},
        ]
    )
    tracks = tk.build_tracks(df, kind="L", res="HR")

    assert tracks.track_id.nunique() == 2
    # Raising the ceiling reconnects it, so the gap is what split it.
    generous = tk.build_tracks(
        df, kind="L", res="HR", params=tk.TrackParams(max_gap_hours=24.0)
    )
    assert generous.track_id.nunique() == 1


def test_a_large_pressure_jump_breaks_the_track():
    df = centers_frame(
        [
            {"h": 0, "lat": 45.0, "lon": -100.0, "p": 1000.0},
            {"h": 3, "lat": 45.0, "lon": -99.0, "p": 960.0},
        ]
    )
    tracks = tk.build_tracks(df, kind="L", res="HR")
    assert tracks.track_id.nunique() == 2


def test_resolutions_are_never_linked_to_each_other():
    """The same analysis at two resolutions must not read as a track."""
    df = centers_frame(
        [
            {"h": 0, "lat": 45.0, "lon": -100.0, "res": "HR"},
            {"h": 0, "lat": 45.0, "lon": -100.0, "res": "LR"},
            {"h": 3, "lat": 45.0, "lon": -99.0, "res": "HR"},
        ]
    )
    tracks = tk.build_tracks(df, kind="L", res="HR")

    assert set(tracks.res) == {"HR"}
    assert len(tracks) == 2
    assert tracks.track_id.nunique() == 1


def test_two_lows_keep_separate_identities():
    """Assignment must not swap two systems that pass near each other."""
    df = centers_frame(
        [
            {"h": 0, "lat": 30.0, "lon": -100.0, "p": 1000.0},
            {"h": 0, "lat": 50.0, "lon": -100.0, "p": 990.0},
            {"h": 3, "lat": 30.0, "lon": -98.0, "p": 1000.0},
            {"h": 3, "lat": 50.0, "lon": -98.0, "p": 990.0},
        ]
    )
    tracks = tk.build_tracks(df, kind="L", res="HR")

    assert tracks.track_id.nunique() == 2
    for _, leg in tracks.groupby("track_id"):
        # Each track stays on its own latitude rather than jumping 20 degrees.
        assert leg.lat.nunique() == 1


def test_track_ids_survive_a_duplicated_index():
    """Parquet read back from several part files can repeat index labels."""
    df = walking_low(4, 200.0)
    df.index = [0, 0, 1, 1]

    tracks = tk.build_tracks(df, kind="L", res="HR")
    assert tracks.track_id.nunique() == 1
    assert sorted(tracks.track_step) == [0, 1, 2, 3]


def test_track_stats_reports_endpoints_and_extremes():
    df = centers_frame(
        [
            {"h": 0, "lat": 45.0, "lon": -100.0, "p": 1000.0},
            {"h": 3, "lat": 45.0, "lon": -98.0, "p": 990.0},
            {"h": 6, "lat": 45.0, "lon": -96.0, "p": 995.0},
        ]
    )
    stats = tk.track_stats(tk.build_tracks(df, kind="L", res="HR"))

    assert len(stats) == 1
    row = stats.iloc[0]
    assert row.n_steps == 2
    assert row.min_pressure_hpa == pytest.approx(990.0)
    assert row.genesis_lon == pytest.approx(-100.0)
    assert row.lysis_lon == pytest.approx(-96.0)


def test_empty_input_returns_empty_tracks():
    empty = centers_frame([{"h": 0, "lat": 45.0, "lon": -100.0, "kind": "H"}])
    tracks = tk.build_tracks(empty, kind="L", res="HR")
    assert tracks.empty
    assert tk.track_stats(tracks).empty


def test_implausible_pressures_are_flagged_by_label_not_value_alone():
    """1040 hPa is ordinary for a high and contradicts the label on a low.

    A weak low at 1020 is entirely normal, so the threshold sits at 1030 and
    the filter must not reach down and take ordinary systems with it.
    """
    df = centers_frame(
        [
            {"h": 0, "lat": 45.0, "lon": -100.0, "kind": "L", "p": 1040.0},
            {"h": 0, "lat": 40.0, "lon": -100.0, "kind": "H", "p": 1040.0},
            {"h": 0, "lat": 35.0, "lon": -100.0, "kind": "L", "p": 1188.0},
            {"h": 0, "lat": 30.0, "lon": -100.0, "kind": "H", "p": 980.0},
            {"h": 0, "lat": 25.0, "lon": -100.0, "kind": "L", "p": 1020.0},
        ]
    )
    kept, dropped = ld.drop_implausible_pressure(df)

    assert dropped == 3  # the 1040 low, the 1188 low, the 980 high
    assert sorted(kept.pressure_hpa) == [1020.0, 1040.0]
    assert set(kept.kind) == {"H", "L"}


def test_missing_pressure_is_absent_not_implausible():
    df = centers_frame([{"h": 0, "lat": 45.0, "lon": -100.0, "p": None}])
    kept, dropped = ld.drop_implausible_pressure(df)
    assert dropped == 0
    assert len(kept) == 1


def test_intensity_grid_averages_and_respects_min_count():
    grid = g.Grid(cell_km=200.0)
    # Three centers in one place, one center far away on its own.
    df = centers_frame(
        [
            {"h": 0, "lat": 45.0, "lon": -100.0, "p": 990.0},
            {"h": 3, "lat": 45.0, "lon": -100.0, "p": 1000.0},
            {"h": 6, "lat": 45.0, "lon": -100.0, "p": 1010.0},
            {"h": 9, "lat": 30.0, "lon": -80.0, "p": 950.0},
        ]
    )
    mean = g.intensity_grid(df, grid, min_count=3)

    finite = mean[np.isfinite(mean)]
    assert finite.size == 1
    assert finite[0] == pytest.approx(1000.0)


def test_intensity_grid_is_all_nan_without_pressures():
    df = centers_frame([{"h": 0, "lat": 45.0, "lon": -100.0, "p": None}])
    mean = g.intensity_grid(df, g.Grid(cell_km=200.0), min_count=1)
    assert np.isnan(mean).all()


def test_stats_separate_a_traveller_from_a_stationary_feature():
    """A 45-day track that never moves is the archive's real failure mode."""
    moving = centers_frame(
        [{"h": i * 3, "lat": 45.0, "lon": -100.0 + i * 2.0} for i in range(6)]
    )
    parked = centers_frame(
        [{"h": i * 3, "lat": 34.7, "lon": -114.6} for i in range(6)]
    )
    parked["bulletin_id"] = [f"p{i}" for i in range(6)]

    moving_stats = tk.track_stats(tk.build_tracks(moving, kind="L", res="HR"))
    parked_stats = tk.track_stats(tk.build_tracks(parked, kind="L", res="HR"))

    # Same number of steps, entirely different motion.
    assert moving_stats.iloc[0].n_steps == parked_stats.iloc[0].n_steps == 5
    assert moving_stats.iloc[0].net_km > 700
    assert parked_stats.iloc[0].net_km == pytest.approx(0.0, abs=1.0)
    assert parked_stats.iloc[0].mean_speed_kmh == pytest.approx(0.0, abs=0.1)
    assert moving_stats.iloc[0].mean_speed_kmh > 40


def test_path_length_exceeds_net_displacement_when_a_track_doubles_back():
    df = centers_frame(
        [
            {"h": 0, "lat": 45.0, "lon": -100.0},
            {"h": 3, "lat": 45.0, "lon": -98.0},
            {"h": 6, "lat": 45.0, "lon": -100.0},
        ]
    )
    stats = tk.track_stats(tk.build_tracks(df, kind="L", res="HR"))
    row = stats.iloc[0]

    assert row.net_km == pytest.approx(0.0, abs=1.0)
    assert row.path_km > 300  # out and back, roughly 2 x 157 km
