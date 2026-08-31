"""Track pressure centers across consecutive analyses.

A track links the same physical low (or high) across time: the same
center reappearing in each subsequent bulletin, allowed to drift in space
and deepen/fill in pressure, but not to teleport or swap identity with an
unrelated system on the other side of the domain.

Matching between consecutive analyses is a linear assignment problem:
minimize total cost (distance plus a pressure-jump penalty) subject to a
hard displacement-speed cap, solved with the Hungarian algorithm. Centers
with no acceptable match are track endpoints -- genesis if nothing
matched before them, lysis if nothing matches after.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .grid import GEOD


@dataclass(frozen=True)
class TrackParams:
    max_speed_kmh: float = 133.0     # ~400 km / 3h, per the roadmap
    max_pressure_jump: float = 15.0  # hPa, between consecutive matched centers
    pressure_weight: float = 10.0    # km-equivalent cost per hPa of jump
    # Never link across a gap longer than this. The HR series is 3-hourly with
    # 482 six-hour gaps and 9 nine-hour gaps in 2009-2018, and the speed cap
    # scales with elapsed time -- so without a ceiling a nine-hour gap quietly
    # authorizes a 1,197 km jump, and a longer outage in future data would
    # authorize a match clean across the continent.
    max_gap_hours: float = 6.0


def _pairwise_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (km) between every pair across two coordinate sets."""
    lon1g, lon2g = np.meshgrid(np.asarray(lon1), np.asarray(lon2), indexing="ij")
    lat1g, lat2g = np.meshgrid(np.asarray(lat1), np.asarray(lat2), indexing="ij")
    _, _, dist_m = GEOD.inv(lon1g.ravel(), lat1g.ravel(), lon2g.ravel(), lat2g.ravel())
    return dist_m.reshape(lon1g.shape) / 1000.0


def _match_step(
    prev: pd.DataFrame, cur: pd.DataFrame, hours: float, params: TrackParams
) -> list[tuple[int, int]]:
    """Match centers in `prev` (positional) to centers in `cur`."""
    if prev.empty or cur.empty:
        return []

    dist = _pairwise_km(prev.lat.to_numpy(), prev.lon.to_numpy(),
                         cur.lat.to_numpy(), cur.lon.to_numpy())
    dp = np.abs(prev.pressure_hpa.to_numpy()[:, None]
                - cur.pressure_hpa.to_numpy()[None, :])
    dp = np.nan_to_num(dp, nan=0.0)  # a missing pressure shouldn't forbid a match

    cost = dist + params.pressure_weight * dp
    max_km = params.max_speed_kmh * hours
    infeasible = (dist > max_km) | (dp > params.max_pressure_jump)
    BIG = 1e7
    cost = np.where(infeasible, BIG, cost)

    row_ind, col_ind = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(row_ind, col_ind) if cost[r, c] < BIG]


def build_tracks(
    centers: pd.DataFrame, kind: str, res: str, params: TrackParams | None = None
) -> pd.DataFrame:
    """Link centers of one kind ('H' or 'L') into tracks across time.

    One resolution at a time -- LR and HR must never be mixed here, same as
    everywhere else in this codebase: a spurious HR/LR pairing of the same
    analysis would read as a center that displaced 0 km and deepened
    instantly, which is exactly the kind of artifact the pressure-jump and
    speed caps exist to reject, but simplest to just never offer it a match.

    Adds two columns to `centers`: `track_id` and `track_step` (0 at
    genesis, incrementing along the track).
    """
    params = params or TrackParams()
    df = centers[(centers.kind == kind) & (centers.res == res)].copy()
    if df.empty:
        return df.assign(
            track_id=pd.Series(dtype="string"),
            track_step=pd.Series(dtype="int32"),
        )

    # reset_index so row labels are positions: a Parquet dataset read back from
    # several part files can carry duplicate index labels, and any label-based
    # bookkeeping would then silently cross-assign tracks.
    df = df.sort_values("valid_time").reset_index(drop=True)

    # Group once rather than filtering the frame per timestep. `df[df.valid_time
    # == t]` inside the loop is a full scan per analysis, which makes the whole
    # thing quadratic: fine for a month, roughly 75 minutes for ten years of
    # HR lows, which is what the CLI runs by default.
    groups = list(df.groupby("valid_time", sort=True))

    # Positional arrays, not .loc per row: scalar label assignment into a
    # Series 410k times is its own slow path.
    track_ids = np.empty(len(df), dtype=object)
    track_step = np.zeros(len(df), dtype=np.int64)

    next_track_num = 0
    prev_time, prev_slice, prev_pos = None, None, None

    for t, cur_slice in groups:
        cur_pos = cur_slice.index.to_numpy()

        matches = []
        if prev_slice is not None:
            hours = (t - prev_time).total_seconds() / 3600.0
            if 0 < hours <= params.max_gap_hours:
                matches = _match_step(
                    prev_slice.reset_index(drop=True),
                    cur_slice.reset_index(drop=True),
                    hours, params,
                )

        matched_positions = set()
        for r, c in matches:
            source = prev_pos[r]
            target = cur_pos[c]
            track_ids[target] = track_ids[source]
            track_step[target] = track_step[source] + 1
            matched_positions.add(c)

        for pos, target in enumerate(cur_pos):
            if pos in matched_positions:
                continue
            track_ids[target] = f"{kind}{next_track_num:07d}"
            track_step[target] = 0
            next_track_num += 1

        prev_time, prev_slice, prev_pos = t, cur_slice, cur_pos

    df["track_id"] = pd.Series(track_ids, index=df.index, dtype="string")
    df["track_step"] = pd.Series(track_step, index=df.index, dtype="int32")
    return df


def track_stats(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per track: genesis/lysis time+place, lifetime, pressure, motion.

    The motion columns are not decoration. Nearest-neighbour linking is only
    as meaningful as the assumption that a center is a travelling system, and
    in this archive that assumption fails hard for quasi-stationary features:
    the longest low track in the HR record runs 45 days and displaces 14 km,
    because it is the Mojave thermal low being re-analysed in the same place
    every three hours all summer. It is a real track of a real analysed
    feature, and it is not a cyclone. Filter on `net_km` or `mean_speed_kmh`
    before computing anything about lifetimes or deepening rates.
    """
    if tracks.empty:
        return pd.DataFrame()

    ordered = tracks.sort_values(["track_id", "track_step"])
    stats = (
        ordered.groupby("track_id")
        .agg(
            genesis_time=("valid_time", "first"),
            lysis_time=("valid_time", "last"),
            genesis_lat=("lat", "first"),
            genesis_lon=("lon", "first"),
            lysis_lat=("lat", "last"),
            lysis_lon=("lon", "last"),
            n_steps=("track_step", "max"),
            min_pressure_hpa=("pressure_hpa", "min"),
            max_pressure_hpa=("pressure_hpa", "max"),
        )
        .reset_index()
    )

    # Straight-line genesis-to-lysis distance.
    _, _, net_m = GEOD.inv(
        stats.genesis_lon.to_numpy(), stats.genesis_lat.to_numpy(),
        stats.lysis_lon.to_numpy(), stats.lysis_lat.to_numpy(),
    )
    stats["net_km"] = net_m / 1000.0

    # Distance actually walked, summed leg by leg within each track.
    lat, lon = ordered.lat.to_numpy(), ordered.lon.to_numpy()
    same = ordered.track_id.to_numpy()[1:] == ordered.track_id.to_numpy()[:-1]
    leg_km = np.zeros(len(ordered) - 1)
    if same.any():
        _, _, leg_m = GEOD.inv(
            lon[:-1][same], lat[:-1][same], lon[1:][same], lat[1:][same]
        )
        leg_km[same] = leg_m / 1000.0
    walked = pd.Series(leg_km, index=ordered.index[:-1]).groupby(
        ordered.track_id.iloc[:-1].to_numpy()
    ).sum()
    stats["path_km"] = stats.track_id.map(walked).fillna(0.0)

    hours = (stats.lysis_time - stats.genesis_time).dt.total_seconds() / 3600.0
    stats["duration_hours"] = hours
    with np.errstate(invalid="ignore", divide="ignore"):
        stats["mean_speed_kmh"] = np.where(hours > 0, stats.path_km / hours, 0.0)

    return stats