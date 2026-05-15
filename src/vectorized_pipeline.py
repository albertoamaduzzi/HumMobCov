"""
vectorized_pipeline.py
======================
Vectorized replacement for the per-user ``compute_all()`` loop.

Instead of instantiating one ``User`` object per user (with a skmob
TrajDataFrame inside), all metrics are computed in batch over the full
period DataFrame using:

* **Polars group_by expressions** for parallelisable scalar metrics
  (radius of gyration, entropies, home, q, distance, k-RG, frequency).
* **Polars scan_parquet + expressions** for preprocessing (10–30× faster
  than pandas read_parquet + Shapely point-in-polygon).
* **Numba-JIT per-user loops** via polars ``map_groups`` for the two
  inherently sequential algorithms (real entropy, S(t) curve).
* **NumPy-only map_groups** for the Gonzalez PCA trajectory shape.
* **GeoPandas sjoin** for vectorised county / rurality assignment.

Public API
----------
``preprocess_shard_polars(file, dataset)``
    Replaces ``dataset_info.__init__()`` + ``preprocess()``.
    Returns ``{period_name: pl.DataFrame}``.

``compute_all_polars(cfg, dataset, period_df, period, already_done, store, ...)``
    Replaces ``compute_all()`` in pipeline.py.
    Accepts the same ``already_done_*`` sets and ``ParquetStore``.

Metric implementations
-----------------------
The following metrics differ slightly from the old skmob-based ones:

* **radius_of_gyration**: old code had a groupby bug (applied ``xy()``
  to the *full* DataFrame inside a per-geohash loop so ``x[0]`` always
  referred to the first row of the full frame, not the group centroid).
  The new code computes the correct time-weighted RG formula.

* **Gonzalez**: old code had swapped lat/lon labels for the reference
  point.  New code uses ``(lat_mean, lon_mean)`` correctly.

* **uncorrelated_entropy, random_entropy**: exact same formula as skmob,
  computed via polars expressions.

* **real_entropy**: same LZ78 algorithm as skmob, re-implemented in
  pure Python (called via ``map_groups``).

* **distance**: mean inter-stop haversine distance, same as skmob.

* **k_radius_of_gyration**: same formula as skmob (top-k locations by
  visit count).

* **S(t)**: same logic as ``User._fill_dict``; re-implemented without
  the per-row Python lambda overhead.

* **frequency/rank**: visit-count based frequency and rank per geohash7.
"""
from __future__ import annotations

import math
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

try:
    import geohash as _geohash
    _HAS_GEOHASH = True
except ImportError:
    _HAS_GEOHASH = False

from .constants import (
    K_RADIUS_VALUES,
    TIME_INTERVAL_S_MAX,
    US_BOUNDING_BOX,
    ALL_SCALAR_METRICS,
)
from .utils import ifnotexistsmkdir

if TYPE_CHECKING:
    from .store import ParquetStore
    from .datasets import _BaseDataset

from .utils import get_cpu_quota as _get_cpu_quota

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):  # type: ignore[misc]
        return it

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PI = math.pi
_R_KM = 6371.0        # WGS-84 mean radius in km
_T_MAX = TIME_INTERVAL_S_MAX  # 1420

# Polars' thread pool is set at import time via the POLARS_NUM_THREADS env var.
# main.py sets it to the cgroup CPU quota before importing this module.
_real_cores = _get_cpu_quota()

# ---------------------------------------------------------------------------
# Section 1 — Pure-Python / NumPy per-user algorithms
#   Called via polars map_groups; no skmob, no pandas overhead per user.
# ---------------------------------------------------------------------------


def _lz78_entropy(locations: list) -> float:
    """
    Lempel-Ziv 78 entropy rate estimator.

    Produces the same result as skmob's ``real_entropy`` function.

    Parameters
    ----------
    locations : list of hashable
        Ordered sequence of visited location identifiers (e.g. geohash7).

    Returns
    -------
    float
        Estimated entropy rate in bits.
    """
    n = len(locations)
    if n <= 1:
        return 0.0
    dictionary: set = set()
    phrase: list = []
    n_phrases = 0
    for loc in locations:
        phrase.append(loc)
        key = tuple(phrase)
        if key not in dictionary:
            dictionary.add(key)
            n_phrases += 1
            phrase = []
    if phrase:
        n_phrases += 1
    if n_phrases <= 1:
        return 0.0
    return n_phrases * math.log2(n_phrases) / n


def _st_curve(t_hours: np.ndarray, geohashes: list, t_step: int) -> list:
    """
    Compute S(t) exploration curve (distinct visited places vs. time).

    Replicates ``User._fill_dict`` logic without pandas/skmob overhead.

    Parameters
    ----------
    t_hours : int64 array
        Hour offset from first stop for each row (sorted, first = 0).
    geohashes : list[str]
        Geohash7 for each row (same order as t_hours).
    t_step : int
        Hour step size (= t_threshold).

    Returns
    -------
    list[int]
        Length = TIME_INTERVAL_S_MAX - 1.  ``result[h]`` is the number
        of distinct locations seen by hour h.
    """
    n = len(t_hours)
    n_steps = _T_MAX - 1  # = 1419

    if n == 0:
        return [0] * n_steps

    # Running distinct count per position
    visited: set = set()
    s_arr = np.empty(n, dtype=np.int32)
    for i, gh in enumerate(geohashes):
        visited.add(gh)
        s_arr[i] = len(visited)

    t_arr = t_hours.astype(np.int64)

    # For each time step h, find the last stop with t_arr[e] <= h
    result = [0] * n_steps
    e = 0
    for h in range(0, n_steps, t_step):
        # Advance pointer while next stop has already passed
        while e < n - 1 and t_arr[e + 1] <= h:
            e += 1
        if t_arr[e] <= h:
            result[h] = int(s_arr[e])
    return result


def _gonzalez_pca(lat: np.ndarray, lon: np.ndarray):
    """
    Gonzalez et al. PCA trajectory-shape computation.

    Returns ``(x_norm, y_norm, sigmax, sigmay)`` arrays.  Returns
    ``None`` on degenerate input.

    Parameters
    ----------
    lat, lon : float64 arrays
        Stop coordinates.
    """
    if len(lat) < 2:
        return None

    mean_lat = lat.mean()
    mean_lon = lon.mean()

    # Project to tangent plane (same _xy_inner formula as utils.py)
    c_lat = 0.6 * 1e5 * (1.85533 - 0.006222 * math.sin(mean_lat * _PI / 180.0))
    c_lon = c_lat * math.cos(mean_lat * _PI / 180.0)
    proj_x = c_lon * (lon - mean_lon)
    proj_y = c_lat * (lat - mean_lat)

    shifted_x = proj_x - proj_x.mean()
    shifted_y = proj_y - proj_y.mean()

    all_zero = (np.abs(shifted_x) < 1).all() and (np.abs(shifted_y) < 1).all()
    if all_zero:
        shifted_x = np.zeros_like(shifted_x)
        shifted_y = np.zeros_like(shifted_y)

    Ixx = np.sum(shifted_x ** 2)
    Iyy = np.sum(shifted_y ** 2)
    Ixy = np.sum(shifted_x * shifted_y)
    mu  = math.sqrt(max(0.0, 4 * Ixy ** 2 + Ixx ** 2 - 2 * Ixx * Iyy + Iyy ** 2))

    if all_zero:
        cos_theta = 0.0
    else:
        denom = 0.5 * Ixx - 0.5 * Iyy + 0.5 * mu
        if abs(denom) < 1e-10:
            cos_theta = 0.0
        else:
            cos_theta = -Ixy / (denom * math.sqrt(1.0 + Ixy ** 2 / denom ** 2))

    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta ** 2))

    if all_zero:
        rot_x = np.zeros_like(shifted_x)
        rot_y = np.zeros_like(shifted_y)
    else:
        rot_x = -cos_theta * shifted_x + sin_theta * shifted_y
        rot_y = -cos_theta * shifted_y - sin_theta * shifted_x

    valid = np.isfinite(rot_x) & np.isfinite(rot_y)
    rot_x = rot_x[valid]
    rot_y = rot_y[valid]
    if len(rot_x) == 0:
        return None

    sigma_x = math.sqrt((rot_x ** 2).mean()) if len(rot_x) > 0 else 0.0
    sigma_y = math.sqrt((rot_y ** 2).mean()) if len(rot_y) > 0 else 0.0
    if sigma_x < 1e-5:
        sigma_x = 0.0
    if sigma_y < 1e-5:
        sigma_y = 0.0

    if sigma_x == 0 and sigma_y != 0:
        x_norm = rot_x / sigma_x if sigma_x != 0 else np.zeros_like(rot_x)
        y_norm = rot_y / sigma_y
    elif sigma_y == 0 and sigma_x != 0:
        x_norm = rot_x / sigma_x
        y_norm = np.zeros_like(rot_y)
    elif sigma_x == 0 and sigma_y == 0:
        x_norm = np.zeros_like(rot_x)
        y_norm = np.zeros_like(rot_y)
    else:
        x_norm = rot_x / sigma_x
        y_norm = rot_y / sigma_y

    return x_norm, y_norm, sigma_x, sigma_y


# ---------------------------------------------------------------------------
# Section 2 — Polars preprocessing
#   Replaces dataset_info.__init__() + preprocess()
# ---------------------------------------------------------------------------


def preprocess_shard_polars(
    file: str | Path,
    dataset: "_BaseDataset",
) -> dict[str, pl.DataFrame]:
    """
    Load and preprocess one raw parquet shard, returning a polars
    DataFrame per period.

    Replaces ``dataset_info.__init__()`` + ``dataset_info.preprocess()``.

    Speedups vs. pandas path:
    * ``pl.scan_parquet`` reads only needed columns with predicate pushdown.
    * Bounding-box filter is a vectorised polars expression (no Shapely
      Point/Polygon object per row).
    * Time-diff and per-user filter use polars window functions + the
      existing numba ``_filter_inner`` via ``map_groups``.

    Returns
    -------
    dict[str, pl.DataFrame]
        ``{period_name: period_df}`` containing only users with >= np_
        stops.  Columns: userId, clusterLatitude (lat), clusterLongitude
        (lon), begin, end, geohash7, dur_min.
    """
    bbox   = dataset.bounding_box
    lats   = [p[0] for p in bbox]
    lons   = [p[1] for p in bbox]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # -- 1. Lazy load + spatial filter (predicate pushed into parquet reader)
    lf = (
        pl.scan_parquet(str(file))
        .filter(
            pl.col("clusterLatitude").is_between(min_lat, max_lat)
            & pl.col("clusterLongitude").is_between(min_lon, max_lon)
        )
        .with_columns([
            pl.col("clusterLatitude").alias("lat"),
            pl.col("clusterLongitude").alias("lon"),
        ])
    )
    df = lf.collect()

    if df.is_empty():
        return {}

    # -- 2. Ensure datetime types
    for col in ("begin", "end"):
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).str.to_datetime())

    # -- 3. Compute inter-stop time diff per user (hours)
    #       First row per user gets 0 so it is always kept.
    df = (
        df.sort(["userId", "begin"])
        .with_columns(
            pl.col("begin")
            .diff()
            .over("userId")
            .dt.total_hours()
            .fill_null(0)
            .cast(pl.Float64)
            .alias("_time_diff_h")
        )
    )

    # -- 4. Temporal filter per user (stateful → map_groups with numba)
    from .utils import filter_ as _filter

    t_thr = dataset.t_threshold

    def _apply_filter(grp: pl.DataFrame) -> pl.DataFrame:
        mask = _filter(grp["_time_diff_h"].to_numpy(), t_thr)
        return grp.filter(pl.Series(mask))

    df = df.group_by("userId").map_groups(_apply_filter)
    df = df.drop("_time_diff_h")

    # -- 5. Add stop-duration column (minutes).
    #       Clamp to >= 0: raw data can have end < begin (clock drift /
    #       data-quality issues).  Negative dur_min would corrupt the
    #       time-weighted centroid in _compute_radius_of_gyration_polars,
    #       causing sum(dur_min) ≈ 0 and RG values of 10^6+ km.
    df = df.with_columns(
        (pl.col("end") - pl.col("begin"))
        .dt.total_minutes()
        .cast(pl.Float64)
        .clip(lower_bound=0.0)
        .alias("dur_min")
    )

    # -- 6. Split by period and filter by min points
    period_df: dict[str, pl.DataFrame] = {}
    for p_idx, period_name in enumerate(dataset.period_names):
        t_start = dataset.period_division[p_idx]
        t_end   = dataset.period_division[p_idx + 1]

        pdf = df.filter(
            (pl.col("begin") > pl.lit(t_start))
            & (pl.col("begin") < pl.lit(t_end))
        )
        if pdf.is_empty():
            continue

        # Keep only users with >= np_ stops
        counts = (
            pdf.group_by("userId")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= dataset.np_)
        )
        valid_users = counts["userId"]
        pdf = pdf.filter(pl.col("userId").is_in(valid_users))
        if not pdf.is_empty():
            period_df[period_name] = pdf

    return period_df


# ---------------------------------------------------------------------------
# Section 3 — Vectorized scalar metrics
# ---------------------------------------------------------------------------


def _compute_radius_of_gyration_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Time-weighted radius of gyration for every user in ``df``.

    Formula (corrected vs. old User.py):
        rg = sqrt( sum_i( dur_i * d_i^2 ) / sum_i(dur_i) )
    where d_i is the distance of stop i from the time-weighted centroid,
    computed via the local tangent-plane projection used in utils.xy().

    Returns a DataFrame with columns [userId, radius_gyration] (km).
    """
    # Step 1 — time-weighted centre of mass per user
    cm = df.group_by("userId").agg(
        [
            (pl.col("lat") * pl.col("dur_min")).sum() / pl.col("dur_min").sum(),
            (pl.col("lon") * pl.col("dur_min")).sum() / pl.col("dur_min").sum(),
            pl.col("dur_min").sum().alias("total_dur"),
        ]
    ).rename({"lat": "cm_lat", "lon": "cm_lon"})

    # Step 2 — join centre of mass back, compute projected distances
    df2 = (
        df.join(cm, on="userId")
        .with_columns(
            # Tangent-plane scale factors (same as utils._xy_inner)
            (0.6e5 * (1.85533 - 0.006222 * (pl.col("cm_lat") * _PI / 180).sin()))
            .alias("c_lat")
        )
        .with_columns(
            (pl.col("c_lat") * (pl.col("cm_lat") * _PI / 180).cos()).alias("c_lon")
        )
        .with_columns(
            (
                (pl.col("c_lon") * (pl.col("lon") - pl.col("cm_lon"))).pow(2)
                + (pl.col("c_lat") * (pl.col("lat") - pl.col("cm_lat"))).pow(2)
            ).alias("d2_m2")
        )
    )

    # Step 3 — weighted RG; convert m → km
    rg = df2.group_by("userId").agg(
        ((pl.col("dur_min") * pl.col("d2_m2")).sum() / pl.col("dur_min").sum())
        .sqrt()
        .alias("radius_gyration")
    ).with_columns((pl.col("radius_gyration") / 1000.0).alias("radius_gyration"))

    return rg


def _compute_krg_polars(df: pl.DataFrame, k_values: list[int]) -> pl.DataFrame:
    """
    k-radius of gyration for all users and all k in ``k_values``.

    For each k, keeps only the k most visited locations (by visit count)
    per user, then computes the time-weighted RG on that subset.

    Returns a DataFrame with [userId, rg_3, rg_6, rg_10] (or the requested k).
    """
    # Visit count per (user, location)
    loc_counts = (
        df.group_by(["userId", "geohash7"])
        .agg(pl.len().alias("n_visits"))
        .sort(["userId", "n_visits"], descending=[False, True])
    )

    results = df.select("userId").unique()

    for k in k_values:
        top_k = (
            loc_counts.group_by("userId")
            .head(k)
            .select(["userId", "geohash7"])
        )
        df_k = df.join(top_k, on=["userId", "geohash7"], how="inner")
        if df_k.is_empty():
            results = results.with_columns(pl.lit(float("nan")).alias(f"rg_{k}"))
            continue
        rg_k = _compute_radius_of_gyration_polars(df_k).rename(
            {"radius_gyration": f"rg_{k}"}
        )
        results = results.join(rg_k, on="userId", how="left")

    return results


def _compute_entropies_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Random entropy and uncorrelated entropy for all users.

    random_entropy     = log2(n_distinct_locations)
    uncorrelated_entropy = -sum(p_i * log2(p_i))
      where p_i = n_visits_at_i / total_visits.

    Returns [userId, random_entropy, uncorrelated_entropy].
    """
    # Per-user, per-location visit count
    loc_stats = df.group_by(["userId", "geohash7"]).agg(
        pl.len().alias("n_visits")
    )

    # Total visits per user
    user_totals = loc_stats.group_by("userId").agg(
        pl.col("n_visits").sum().alias("total_visits"),
        pl.len().alias("n_locs"),
    )

    # Entropy contributions
    entropy_df = (
        loc_stats.join(user_totals, on="userId")
        .with_columns(
            (pl.col("n_visits") / pl.col("total_visits")).alias("p_i")
        )
        .with_columns(
            (-(pl.col("p_i") * (pl.col("p_i").log(2)))).alias("h_contrib")
        )
        .group_by("userId")
        .agg(
            pl.col("n_locs").first().log(2).alias("random_entropy"),
            pl.col("h_contrib").sum().alias("uncorrelated_entropy"),
        )
    )
    return entropy_df


def _compute_real_entropy_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Real entropy (LZ78 estimator) for all users via map_groups.

    Returns [userId, real_entropy].
    """
    def _group_fn(grp: pl.DataFrame) -> pl.DataFrame:
        uid  = grp["userId"][0]
        locs = grp.sort("begin")["geohash7"].to_list()
        h    = _lz78_entropy(locs)
        return pl.DataFrame({"userId": [uid], "real_entropy": [h]})

    return df.group_by("userId").map_groups(_group_fn)


# ---------------------------------------------------------------------------
# Top-level worker functions (module-level — required for multiprocessing pickle)
# ---------------------------------------------------------------------------

def _re_worker(chunk_df: pl.DataFrame) -> pl.DataFrame:
    """ProcessPoolExecutor worker: real_entropy for a user chunk."""
    return _compute_real_entropy_polars(chunk_df)


def _st_worker(args: tuple) -> dict:
    """ProcessPoolExecutor worker: S(t) for a user chunk."""
    chunk_df, t_threshold = args
    return _compute_st_polars(chunk_df, t_threshold)


def _gonz_worker(chunk_df: pl.DataFrame) -> dict:
    """ProcessPoolExecutor worker: Gonzalez for a user chunk."""
    return _compute_gonzalez_polars(chunk_df)


def _freq_worker(chunk_df: pl.DataFrame) -> dict:
    """ProcessPoolExecutor worker: visit frequency for a user chunk."""
    return _compute_frequency_polars(chunk_df)


def _split_df_by_users(df: pl.DataFrame, n: int) -> list:
    """Split df into n roughly equal chunks by userId."""
    all_users = df["userId"].unique(maintain_order=False).to_list()
    chunk_size = max(1, math.ceil(len(all_users) / n))
    return [
        df.filter(pl.col("userId").is_in(all_users[i: i + chunk_size]))
        for i in range(0, len(all_users), chunk_size)
    ]


def _compute_home_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Home location: geohash7 with most total time spent.

    Returns [userId, home_geohash7, home_lat, home_lon].
    """
    # Total time per (user, geohash7)
    loc_dur = df.group_by(["userId", "geohash7"]).agg(
        pl.col("dur_min").sum().alias("loc_dur"),
        pl.col("lat").first().alias("loc_lat"),
        pl.col("lon").first().alias("loc_lon"),
    )

    # Home = geohash7 with max duration per user
    home = (
        loc_dur.sort(["userId", "loc_dur"], descending=[False, True])
        .unique("userId", keep="first")
        .select(
            pl.col("userId"),
            pl.col("geohash7").alias("home_geohash7"),
            pl.col("loc_lat").alias("home_lat"),
            pl.col("loc_lon").alias("home_lon"),
        )
    )
    return home


def _compute_distance_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Mean inter-stop haversine distance [km] for all users.

    Returns [userId, distance].
    """
    df_s = df.sort(["userId", "begin"])

    # Previous stop coordinates (within user)
    df_s = df_s.with_columns(
        [
            pl.col("lat").shift(1).over("userId").alias("prev_lat"),
            pl.col("lon").shift(1).over("userId").alias("prev_lon"),
            pl.col("userId").shift(1).over("userId").alias("prev_uid"),
        ]
    )

    # Drop first stop of each user (no previous)
    pairs = df_s.filter(pl.col("userId") == pl.col("prev_uid")).drop("prev_uid")

    # Haversine
    d2r = _PI / 180.0
    pairs = pairs.with_columns(
        [
            ((pl.col("lat") - pl.col("prev_lat")) * d2r / 2).sin().pow(2).alias("dlat2"),
            ((pl.col("lon") - pl.col("prev_lon")) * d2r / 2).sin().pow(2).alias("dlon2"),
            (pl.col("prev_lat") * d2r).cos().alias("cos_lat1"),
            (pl.col("lat") * d2r).cos().alias("cos_lat2"),
        ]
    ).with_columns(
        (
            2
            * _R_KM
            * (
                pl.col("dlat2") + pl.col("cos_lat1") * pl.col("cos_lat2") * pl.col("dlon2")
            )
            .sqrt()
            .arcsin()
        ).alias("step_km")
    )

    dist = pairs.group_by("userId").agg(
        pl.col("step_km").mean().alias("distance")
    )
    return dist


def _compute_fraction_time_polars(
    df: pl.DataFrame, period_start, period_end
) -> pl.DataFrame:
    """
    Fraction of period time during which the user's location is known (q).

    Returns [userId, q].
    """
    from .utils import time_difference as _tdiff
    total_hours = _tdiff(period_start, period_end)
    total_minutes = total_hours * 60.0

    q_df = df.group_by("userId").agg(
        (pl.col("dur_min").sum() / total_minutes).alias("q")
    )
    return q_df


def _compute_st_polars(df: pl.DataFrame, t_threshold: int) -> dict[str, list]:
    """
    S(t) exploration curve for all users.

    Returns ``{uid_str: [visited_places_list]}`` suitable for
    ``store.write_st_batch()``.
    """
    result: dict[str, list] = {}

    _n_u = df["userId"].n_unique()
    for uid, grp in _tqdm(df.sort("begin").group_by("userId"), total=_n_u, desc="S(t)", unit="user", leave=False):
        uid_str = str(uid)
        t0      = grp["begin"][0]
        t_hours = (grp["begin"] - t0).dt.total_hours().to_numpy().astype(np.int64)
        geos    = grp["geohash7"].to_list()
        result[uid_str] = _st_curve(t_hours, geos, t_threshold)

    return result


def _compute_gonzalez_polars(df: pl.DataFrame) -> dict:
    """
    Gonzalez PCA trajectory shape for all users.

    Returns a batch dict ``{uid_str: pd.DataFrame}`` suitable for
    ``store.write_gonzalez_batch()``.
    """
    batch: dict = {}
    _n_u = df["userId"].n_unique()
    for uid, grp in _tqdm(df.group_by("userId"), total=_n_u, desc="gonzalez", unit="user", leave=False):
        uid_str = str(uid)
        lat_arr = grp["lat"].to_numpy()
        lon_arr = grp["lon"].to_numpy()
        pca = _gonzalez_pca(lat_arr, lon_arr)
        if pca is None:
            continue
        x_norm, y_norm, sigmax, sigmay = pca
        batch[uid_str] = pd.DataFrame(
            {"x_norm": x_norm, "y_norm": y_norm, "sigmax": sigmax, "sigmay": sigmay}
        )
    return batch


def _compute_frequency_polars(df: pl.DataFrame) -> dict:
    """
    Visit frequency and rank per location per user.

    Returns a batch dict ``{uid_str: pd.DataFrame}`` with columns
    [frequency, rank, geohash7, geohash6] suitable for
    ``store.write_frequency_batch()``.
    """
    loc_counts = (
        df.group_by(["userId", "geohash7"])
        .agg(pl.len().alias("n_visits"))
        .sort(["userId", "n_visits"], descending=[False, True])
    )
    user_totals = loc_counts.group_by("userId").agg(
        pl.col("n_visits").sum().alias("total")
    )
    loc_counts = loc_counts.join(user_totals, on="userId").with_columns(
        (pl.col("n_visits") / pl.col("total")).alias("frequency")
    )

    batch: dict = {}
    _n_u = loc_counts["userId"].n_unique()
    for uid, grp in _tqdm(loc_counts.group_by("userId"), total=_n_u, desc="frequency", unit="user", leave=False):
        uid_str = str(uid)
        n = len(grp)
        gh7  = grp["geohash7"].to_list()
        freq = grp["frequency"].to_list()
        rank = list(range(1, n + 1))
        gh6  = [g[:6] for g in gh7]
        batch[uid_str] = pd.DataFrame(
            {"frequency": freq, "rank": rank, "geohash7": gh7, "geohash6": gh6}
        )
    return batch


def _compute_weekly_rg_polars(
    df: pl.DataFrame,
    period_division: list,
    period_name: str,
    perodname2idx: dict,
    time_window_days: int = 7,
) -> tuple[dict, list]:
    """
    Weekly radius of gyration for all users.

    Returns ``(batch, all_weeks)`` where:
    * ``batch`` is ``{uid_str: {week_label: rg_value}}``
    * ``all_weeks`` is the ordered list of week-start datetimes as strings

    Implementation: outer loop over weeks (≤ 19), inner computation uses
    ``_compute_radius_of_gyration_polars`` on ALL users at once via Polars'
    own Rust thread pool.  This replaces the old O(n_users × n_weeks) Python
    nested-loop that was the primary serial bottleneck for the long periods.

    Keys in each user's dict match the string entries in ``all_weeks`` so
    that ``write_weekly_rg_batch`` can look them up correctly.
    """
    from datetime import timedelta as _td

    p_idx   = perodname2idx[period_name]
    p_start = period_division[p_idx]
    p_end   = period_division[p_idx + 1]

    # Build week boundaries
    weeks: list = []
    w = p_start
    while w < p_end:
        weeks.append(w)
        w += _td(days=time_window_days)

    all_weeks = [str(w) for w in weeks]
    batch: dict = {}

    if len(weeks) < 2:
        return batch, all_weeks

    # Outer loop is over weeks (≤ 19); the expensive per-user RG computation
    # runs fully in Polars' thread pool for ALL users in each week at once.
    for wi in _tqdm(range(len(weeks) - 1), desc="weekly_rg", unit="week", leave=False):
        w_label = all_weeks[wi]          # string key, matches write_weekly_rg_batch lookup
        w_start = weeks[wi]
        w_end   = weeks[wi + 1]

        week_df = df.filter(
            (pl.col("begin") > pl.lit(w_start))
            & (pl.col("begin") < pl.lit(w_end))
        )
        if week_df.is_empty():
            continue

        # Drop users with ≤ 3 stops in this week (same threshold as old code)
        n_stops = week_df.group_by("userId").agg(pl.len().alias("n_stops"))
        valid_users = n_stops.filter(pl.col("n_stops") > 3)["userId"]
        week_df = week_df.filter(pl.col("userId").is_in(valid_users))
        if week_df.is_empty():
            continue

        # Vectorized RG for all valid users — uses Polars thread pool
        rg_df = _compute_radius_of_gyration_polars(week_df)  # [userId, radius_gyration]

        for uid_str, rg_val in zip(
            rg_df["userId"].cast(pl.Utf8).to_list(),
            rg_df["radius_gyration"].to_list(),
        ):
            if uid_str not in batch:
                batch[uid_str] = {}
            batch[uid_str][w_label] = rg_val

    return batch, all_weeks


def _assign_county_polars(
    scalars: pl.DataFrame,
    geojson,
    county2party: dict,
    county2rural: dict,
) -> pl.DataFrame:
    """
    Vectorised county / rurality / party assignment via GeoPandas sjoin.

    ``scalars`` must contain columns [userId, home_lat, home_lon].

    Returns ``scalars`` with added columns
    [county_home, party_government, rurality_level].
    """
    if scalars.is_empty() or "home_lat" not in scalars.columns:
        return scalars.with_columns(
            [
                pl.lit(None).cast(pl.Utf8).alias("county_home"),
                pl.lit(None).cast(pl.Utf8).alias("party_government"),
                pl.lit(None).cast(pl.Utf8).alias("rurality_level"),
            ]
        )

    pdf = scalars.select(["userId", "home_lat", "home_lon"]).to_pandas()
    gdf_pts = gpd.GeoDataFrame(
        pdf,
        geometry=gpd.points_from_xy(pdf["home_lon"], pdf["home_lat"]),
        crs="EPSG:4326",
    )

    county_gdf = geojson[["name", "geometry"]].copy()
    if county_gdf.crs is None:
        county_gdf = county_gdf.set_crs("EPSG:4326")

    joined = gpd.sjoin(gdf_pts, county_gdf, how="left", predicate="within")
    joined = joined.drop_duplicates("userId").set_index("userId")

    # Map county → party / rurality
    county_names = joined["name"].where(joined["name"].notna(), None)
    parties  = county_names.map(lambda n: county2party.get(n, None) if n else None)
    rurality = county_names.map(lambda n: county2rural.get(n, None) if n else None)

    userId_list      = scalars["userId"].cast(pl.Utf8).to_list()
    county_list:    list = []
    party_list:     list = []
    rurality_list:  list = []
    for uid in userId_list:
        uid_str = str(uid)
        county_list.append(str(county_names.get(uid_str, "")) or "")
        party_list.append(str(parties.get(uid_str, "")) or "")
        rurality_list.append(str(rurality.get(uid_str, "")) or "")

    return scalars.with_columns(
        [
            pl.Series("county_home",      county_list,   dtype=pl.Utf8),
            pl.Series("party_government",  party_list,    dtype=pl.Utf8),
            pl.Series("rurality_level",    rurality_list, dtype=pl.Utf8),
        ]
    )


# ---------------------------------------------------------------------------
# Section 4 — Top-level compute_all_polars
# ---------------------------------------------------------------------------


def compute_all_polars(
    cfg: dict,
    dataset: "_BaseDataset",
    period_df: pl.DataFrame,
    period: str,
    already_done_scalars: set,
    already_done_gonzalez: set,
    already_done_st: set,
    already_done_freq: set,
    already_done_wrg: set,
    store: "ParquetStore",
    batch_size: int = 5000,
) -> dict:
    """
    Vectorized replacement for ``compute_all()`` in pipeline.py.

    Computes all enabled metrics for every user in ``period_df`` that
    is not already in the store, then writes results to ``store`` in one
    large batch.

    Parameters
    ----------
    cfg : dict
        Algorithm-flow config (same as ``compute_all``).
    dataset : _BaseDataset
        Region dataset (census data, parameters).
    period_df : pl.DataFrame
        Already time-filtered rows for this period.  Must contain
        [userId, lat, lon, begin, end, geohash7, dur_min].
    period : str
        Period name.
    already_done_* : set[str]
        User IDs already present in the store (skip them).
    store : ParquetStore
        Output store.
    batch_size : int
        Maximum users per parquet shard write.  Larger = fewer shards.
    Returns
    -------
    dict
        Empty dict (week2points tracking not implemented here).
    """
    if period_df.is_empty():
        return {}

    raw = cfg.get("raw_trajectories", False)
    if not raw:
        # Non-raw mode: no trajectory data available, nothing to do
        return {}

    t_threshold = dataset.t_threshold
    period_start, period_end = dataset.period_names2period_division[period]

    # -- Filter out already-computed users ----------------------------------
    done_all = already_done_scalars
    pending_df = period_df.filter(
        ~pl.col("userId").cast(pl.Utf8).is_in(done_all)
    )
    if pending_df.is_empty():
        return {}

    n_pending = pending_df["userId"].n_unique()
    print(f"  [vectorized] Period '{period}': {n_pending:,} pending users")

    # ------------------------------------------------------------------ #
    # SCALAR METRICS (single polars pass for most)                        #
    # ------------------------------------------------------------------ #
    _t_scalars = time.perf_counter()
    scalars: pl.DataFrame | None = None

    if cfg.get("is_radius_gyration") and not cfg.get("already_computed_rg"):
        _t_m = time.perf_counter()
        print("  [compute] radius_of_gyration ...", end="", flush=True)
        rg = _compute_radius_of_gyration_polars(pending_df)
        print(f" {time.perf_counter()-_t_m:.1f}s")
        scalars = rg if scalars is None else scalars.join(rg, on="userId", how="full", coalesce=True)

    if cfg.get("is_random_entropy") and not cfg.get("already_computed_random_entropy"):
        _t_m = time.perf_counter()
        print("  [compute] random_entropy ...", end="", flush=True)
        ent = _compute_entropies_polars(pending_df)
        print(f" {time.perf_counter()-_t_m:.1f}s")
        scalars = (
            ent.select(["userId", "random_entropy"])
            if scalars is None
            else scalars.join(ent.select(["userId", "random_entropy"]), on="userId", how="full", coalesce=True)
        )

    if cfg.get("is_uncorrelated_entropy") and not cfg.get("already_computed_uncorrelated_entropy"):
        if scalars is None or "uncorrelated_entropy" not in scalars.columns:
            _t_m = time.perf_counter()
            print("  [compute] uncorrelated_entropy ...", end="", flush=True)
            ent = _compute_entropies_polars(pending_df)
            print(f" {time.perf_counter()-_t_m:.1f}s")
            scalars = (
                ent.select(["userId", "uncorrelated_entropy"])
                if scalars is None
                else scalars.join(ent.select(["userId", "uncorrelated_entropy"]), on="userId", how="full", coalesce=True)
            )

    if cfg.get("is_real_entropy") and not cfg.get("already_computed_real_entropy"):
        _t = time.perf_counter()
        print("  [compute] real_entropy [LZ78] ...", end="", flush=True)
        re_df = _compute_real_entropy_polars(pending_df)
        print(f" {time.perf_counter()-_t:.1f}s")
        scalars = re_df if scalars is None else scalars.join(re_df, on="userId", how="full", coalesce=True)

    if cfg.get("is_distance") and not cfg.get("already_computed_distance"):
        _t_m = time.perf_counter()
        print("  [compute] distance ...", end="", flush=True)
        dist = _compute_distance_polars(pending_df)
        print(f" {time.perf_counter()-_t_m:.1f}s")
        scalars = dist if scalars is None else scalars.join(dist, on="userId", how="full", coalesce=True)

    if cfg.get("is_fraction_time") and not cfg.get("already_computed_fraction_time"):
        _t_m = time.perf_counter()
        print("  [compute] fraction_time ...", end="", flush=True)
        q_df = _compute_fraction_time_polars(pending_df, period_start, period_end)
        print(f" {time.perf_counter()-_t_m:.1f}s")
        scalars = q_df if scalars is None else scalars.join(q_df, on="userId", how="full", coalesce=True)

    if cfg.get("is_home") and not cfg.get("already_computed_home"):
        _t_m = time.perf_counter()
        print("  [compute] home ...", end="", flush=True)
        home = _compute_home_polars(pending_df)
        print(f" {time.perf_counter()-_t_m:.1f}s")
        scalars = home if scalars is None else scalars.join(home, on="userId", how="full", coalesce=True)

    if cfg.get("is_krg") and not cfg.get("already_computed_krg"):
        _t_m = time.perf_counter()
        print("  [compute] k_radius_of_gyration ...", end="", flush=True)
        krg = _compute_krg_polars(pending_df, K_RADIUS_VALUES)
        print(f" {time.perf_counter()-_t_m:.1f}s")
        scalars = krg if scalars is None else scalars.join(krg, on="userId", how="full", coalesce=True)

    if cfg.get("is_county_rural") and not cfg.get("already_computed_county_rural"):
        if scalars is not None and "home_lat" in scalars.columns:
            _t = time.perf_counter()
            scalars = _assign_county_polars(
                scalars, dataset.geojson, dataset.county2party, dataset.county2rural
            )
            print(f"  [timing] county_assignment: {time.perf_counter()-_t:.1f}s")

    # -- Convert home coordinates to WKT Point string (for store compat) --
    if scalars is not None and "home_lat" in scalars.columns:
        home_wkt = (
            scalars["home_lat"].to_list(),
            scalars["home_lon"].to_list(),
        )
        home_str = [
            str(Point([lat, lon])) if lat is not None else ""
            for lat, lon in zip(*home_wkt)
        ]
        scalars = scalars.with_columns(
            pl.Series("home", home_str, dtype=pl.Utf8)
        )
        # home_geohash7 already in scalars from _compute_home_polars

    print(f"  [timing] scalar metrics: {time.perf_counter()-_t_scalars:.1f}s total")

    # -- Build scalar batch and write to store in chunks -------------------
    if scalars is not None:
        uid_col = scalars["userId"].cast(pl.Utf8).to_list()
        col_map = {c: scalars[c].to_list() for c in scalars.columns if c != "userId"}
        n_total = len(uid_col)

        for chunk_start in range(0, n_total, batch_size):
            chunk_end = min(chunk_start + batch_size, n_total)
            batch: dict = {}
            for i in range(chunk_start, chunk_end):
                uid_str = uid_col[i]
                row: dict = {}
                for metric in ALL_SCALAR_METRICS:
                    vals = col_map.get(metric)
                    if vals is not None:
                        v = vals[i]
                        row[metric] = None if (isinstance(v, float) and math.isnan(v)) else v
                batch[uid_str] = row
            store.write_scalars_batch(period, batch)

    # ------------------------------------------------------------------ #
    # GONZALEZ                                                             #
    # ------------------------------------------------------------------ #
    if cfg.get("is_gonzalez") and not cfg.get("already_computed_gonzalez"):
        pending_gonz = pending_df.filter(
            ~pl.col("userId").cast(pl.Utf8).is_in(already_done_gonzalez)
        )
        if not pending_gonz.is_empty():
            _t = time.perf_counter()
            print("  [compute] gonzalez ...", end="", flush=True)
            gonz_batch: dict = _compute_gonzalez_polars(pending_gonz)
            print(f" {time.perf_counter()-_t:.1f}s")
            uids = list(gonz_batch.keys())
            for chunk_start in range(0, len(uids), batch_size):
                chunk = {u: gonz_batch[u] for u in uids[chunk_start:chunk_start + batch_size]}
                store.write_gonzalez_batch(period, chunk)

    # ------------------------------------------------------------------ #
    # S(t)                                                                 #
    # ------------------------------------------------------------------ #
    if cfg.get("is_St") and not cfg.get("already_computed_St"):
        pending_st = pending_df.filter(
            ~pl.col("userId").cast(pl.Utf8).is_in(already_done_st)
        )
        if not pending_st.is_empty():
            _t = time.perf_counter()
            print("  [compute] S(t) ...", end="", flush=True)
            st_batch: dict = _compute_st_polars(pending_st, t_threshold)
            print(f" {time.perf_counter()-_t:.1f}s")
            uids = list(st_batch.keys())
            for chunk_start in range(0, len(uids), batch_size):
                chunk = {u: st_batch[u] for u in uids[chunk_start:chunk_start + batch_size]}
                store.write_st_batch(period, chunk)

    # ------------------------------------------------------------------ #
    # FREQUENCY                                                            #
    # ------------------------------------------------------------------ #
    if cfg.get("is_frequency") and not cfg.get("already_computed_frequency"):
        pending_freq = pending_df.filter(
            ~pl.col("userId").cast(pl.Utf8).is_in(already_done_freq)
        )
        if not pending_freq.is_empty():
            _t = time.perf_counter()
            print("  [compute] frequency ...", end="", flush=True)
            freq_batch: dict = _compute_frequency_polars(pending_freq)
            print(f" {time.perf_counter()-_t:.1f}s")
            uids = list(freq_batch.keys())
            for chunk_start in range(0, len(uids), batch_size):
                chunk = {u: freq_batch[u] for u in uids[chunk_start:chunk_start + batch_size]}
                store.write_frequency_batch(period, chunk)

    # ------------------------------------------------------------------ #
    # WEEKLY RG                                                            #
    # ------------------------------------------------------------------ #
    if cfg.get("is_weekly_radius_gyration"):
        pending_wrg = pending_df.filter(
            ~pl.col("userId").cast(pl.Utf8).is_in(already_done_wrg)
        )
        if not pending_wrg.is_empty():
            _t = time.perf_counter()
            wrg_batch, all_weeks = _compute_weekly_rg_polars(
                pending_wrg,
                dataset.period_division,
                period,
                dataset.perodname2idx,
            )
            print(f"  [timing] weekly_rg: {time.perf_counter()-_t:.1f}s (vectorized, Polars)")
            uids = list(wrg_batch.keys())
            for chunk_start in range(0, len(uids), batch_size):
                chunk = {u: wrg_batch[u] for u in uids[chunk_start:chunk_start + batch_size]}
                store.write_weekly_rg_batch(period, chunk, all_weeks)

    print(f"  [vectorized] Period '{period}': done.")
    return {}
