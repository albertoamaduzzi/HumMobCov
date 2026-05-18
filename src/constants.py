"""
constants.py
============
All project-wide constants and path definitions for the HumMobCov analysis.

Paths use pathlib.Path and are anchored to PROJECT_ROOT so the project
is portable.  Paths that point to external server storage (raw Cuebiq data,
legacy output directories) are defined separately and can be overridden
via environment variables or a local config file if needed.
"""

import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout  (HumMobCov/)
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent   # HumMobCov/
DIR_SRC        = PROJECT_ROOT / "src"
DIR_OUTPUT     = PROJECT_ROOT / "output"
DIR_DATA       = PROJECT_ROOT / "data"
DIR_CENSUS     = PROJECT_ROOT / "census_data"
DIR_CONFIG     = DIR_DATA / "config"

# ---------------------------------------------------------------------------
# External / server-side data paths  (raw Cuebiq stops)
# ---------------------------------------------------------------------------
DIR_CUEBIQ_BASE  = Path("/data/shared/cuebiq/MOBS")
DIR_RAW_DATA_CA  = DIR_CUEBIQ_BASE / "urban_rural_flow_stops_cali_urban_rural_v3"
DIR_RAW_DATA_MA  = DIR_CUEBIQ_BASE / "20220330_stops_hq_users_MA"

# Legacy output root — per-user result files from the old pipeline live here.
# Defaults to the local milestones_analysis/ folder inside the project.
# Override by setting the environment variable MILESTONES_DIR if the data
# lives elsewhere (e.g. on a mounted server volume).
import os as _os
DIR_MILESTONES_SERVER = Path(
    _os.environ.get("MILESTONES_DIR", str(PROJECT_ROOT / "milestones_analysis"))
)

# ---------------------------------------------------------------------------
# S3 / object-store configuration
# Raw Cuebiq stop-point shards live on an S3-compatible store.
# All values can be overridden via environment variables.
# ---------------------------------------------------------------------------
S3_ENDPOINT_URL: str = _os.environ.get("S3_ENDPOINT_URL", "https://s3.atlas.fbk.eu")
S3_BUCKET:       str = _os.environ.get("S3_BUCKET",       "chub-datalake")

# S3 key prefixes for raw Cuebiq stop-point shards (one parquet file per shard)
S3_RAW_PREFIX: dict[str, str] = {
    "CA": _os.environ.get(
        "S3_PREFIX_CA",
        "shared/cuebiq/MOBS/urban_rural_flow_stops_cali_urban_rural_v3",
    ),
    "MA": _os.environ.get(
        "S3_PREFIX_MA",
        "shared/cuebiq/MOBS/20220330_stops_hq_users_MA",
    ),
}

# Local temp directory for downloaded shards — deleted immediately after processing
DIR_SHARD_TEMP = Path(
    _os.environ.get("SHARD_TEMP_DIR", str(PROJECT_ROOT / ".shard_tmp"))
)

# ---------------------------------------------------------------------------
# S3 output paths — final_pipeline output folder
# ---------------------------------------------------------------------------
# All computed results (parquet store, transition matrices, …) are
# synchronised here after local computation.  The folder name
# ``final_pipeline`` sits at the root of the bucket so it is easy to
# identify and separate from raw input data.
#
# Override via environment variables to change bucket / prefix:
#   S3_OUTPUT_BUCKET     (defaults to S3_BUCKET)
#   S3_OUTPUT_PREFIX_CA
#   S3_OUTPUT_PREFIX_MA
# ---------------------------------------------------------------------------
S3_OUTPUT_BUCKET: str = _os.environ.get("S3_OUTPUT_BUCKET", "chub-datalake")

S3_OUTPUT_PREFIX: dict[str, str] = {
    "CA": _os.environ.get(
        "S3_OUTPUT_PREFIX_CA",
        "final_pipeline/CA",
    ),
    "MA": _os.environ.get(
        "S3_OUTPUT_PREFIX_MA",
        "final_pipeline/MA",
    ),
}

# Sub-prefix for transition matrices within final_pipeline/
S3_TRANSITION_PREFIX: dict[str, str] = {
    region: f"{prefix}/transition_matrices"
    for region, prefix in S3_OUTPUT_PREFIX.items()
}

# Whether to upload computed results to S3 by default.
# Set S3_UPLOAD_DEFAULT=0 to keep results local only.
S3_UPLOAD_DEFAULT: bool = _os.environ.get("S3_UPLOAD_DEFAULT", "1") != "0"

del _os

# External library path
DIR_LIBRARIES = Path("/data/rgallotti/libraries/PythonScripts")

# ---------------------------------------------------------------------------
# Supported regions
# ---------------------------------------------------------------------------
LIST_REGIONS = ["CA", "MA"]

# Raw data directory per region
DIR_RAW_DATA = {
    "CA": DIR_RAW_DATA_CA,
    "MA": DIR_RAW_DATA_MA,
}

# MA comes as fixed named parquet shards
LIST_FILES_MA = [
    "subset_1.snappy.parquet",
    "subset_2.snappy.parquet",
    "subset_3.snappy.parquet",
    "subset_4.snappy.parquet",
    "subset_5.snappy.parquet",
    "subset_6.snappy.parquet",
    "subset_7.snappy.parquet",
    "subset_8.snappy.parquet",
    "subset_9.snappy.parquet",
    "subset_a.snappy.parquet",
    "subset_b.snappy.parquet",
    "subset_c.snappy.parquet",
    "subset_d.snappy.parquet",
    "subset_e.snappy.parquet",
    "subset_f.snappy.parquet",
]

# ---------------------------------------------------------------------------
# Census / reference files per region
# ---------------------------------------------------------------------------
CENSUS_FILES = {
    "CA": {
        "urban_info":   DIR_CENSUS / "California" / "urban_info_threshold_urbanity_500.csv",
        "party_county": DIR_CENSUS / "California" / "political_government_per_county.csv",
        "geojson":      DIR_CENSUS / "California" / "geometry_census_new.geojson",
    },
    "MA": {
        "urban_info":   DIR_CENSUS / "Massachusets" / "Massachusets.csv",
        "party_county": DIR_CENSUS / "Massachusets" / "political_government_per_county.csv",
        "geojson":      DIR_CENSUS / "Massachusets" / "geometry_census_new.geojson",
    },
}

# ---------------------------------------------------------------------------
# Time periods (COVID-19 phases, year 2020)
# ---------------------------------------------------------------------------
PERIOD_NAMES = [
    "15 jan - 15 march",
    "15 march - 15 may",
    "15 may - sept",
]

PERIOD_DIVISION = [
    datetime.datetime(2020,  1, 15),
    datetime.datetime(2020,  3, 15),
    datetime.datetime(2020,  5, 15),
    datetime.datetime(2020,  9, 30),
]

# Convenience mapping:  period_name -> (start_dt, end_dt)
PERIOD_NAMES_TO_DIVISION = {
    PERIOD_NAMES[p]: [PERIOD_DIVISION[p], PERIOD_DIVISION[p + 1]]
    for p in range(len(PERIOD_NAMES))
}

# ---------------------------------------------------------------------------
# Preprocessing parameters
# ---------------------------------------------------------------------------
MIN_POINTS_PER_USER   = 20   # np_ — minimum stop-points a user must have per period
TIME_THRESHOLD_HOURS  = 1    # t_threshold — minimum hours between successive stops

# ---------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------
# Bounding box that covers the contiguous USA + AK/HI
US_BOUNDING_BOX = [
    (18.91619,  -171.791110603),
    (71.3577635769, -171.791110603),
    (71.3577635769,  -66.96466),
    (18.91619,  -66.96466),
]

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
RURALITY_LEVELS  = ["rural", "urban"]
PARTY_NAMES      = ["Democratic", "Republican"]
K_RADIUS_VALUES  = [3, 6, 10]          # k values for k-radius of gyration

# S(t) exploration curve — time axis runs from 0 to this value (minutes)
TIME_INTERVAL_S_MAX = 1420

# ---------------------------------------------------------------------------
# Output file kinds (used by get_already_saved_user_per_period)
# ---------------------------------------------------------------------------
METRIC_FILE_KINDS = ["all_scalars", "gonzalez", "S", "frequency", "weekly_rg"]

# ---------------------------------------------------------------------------
# Scalar metric definitions
# ---------------------------------------------------------------------------
# All scalar metric names stored in the all_scalars parquet table.
# Float metrics (stored as Float64 / NaN when missing).
SCALAR_METRICS_FLOAT: list[str] = [
    "radius_gyration",        # radius of gyration [km]
    "random_entropy",         # S_rand — Boltzmann entropy of visits
    "uncorrelated_entropy",   # S_unc  — entropy ignoring temporal order
    "real_entropy",           # S_real — true entropy of trajectory
    "distance",               # total haversine path length [km] (sum of inter-stop distances)
    "q",                      # predictability limit
] + [f"rg_{k}" for k in K_RADIUS_VALUES]   # rg_3, rg_6, rg_10

# String / categorical metrics (stored as Utf8 / empty-string when missing).
SCALAR_METRICS_STR: list[str] = [
    "home",              # home location geohash7
    "home_geohash7",     # alias kept for backward compatibility
    "county_home",       # county of home location
    "party_government",  # political party of home county
    "rurality_level",    # "urban" | "rural"
]

# Full ordered list — float fields first, then string fields.
# This is the canonical column order for the all_scalars parquet.
ALL_SCALAR_METRICS: list[str] = SCALAR_METRICS_FLOAT + SCALAR_METRICS_STR

# ---------------------------------------------------------------------------
# Gonzalez (PCA trajectory shape) parquet schema
# ---------------------------------------------------------------------------
# Long-format table: one row per (user × visited_location).
# Columns written to gonzalez parquet shards.
GONZALEZ_COLUMNS: list[str] = [
    "user_id",   # str  — user identifier
    "x_norm",    # float — normalised x-coord along 1st principal axis (= x / sigmax)
    "y_norm",    # float — normalised y-coord along 2nd principal axis (= y / sigmay)
    "sigmax",    # float — std dev of locations along 1st axis [km]
    "sigmay",    # float — std dev of locations along 2nd axis [km]
]

# ---------------------------------------------------------------------------
# Frequency / rank parquet schema
# ---------------------------------------------------------------------------
# Long-format table: one row per (user × visited_location rank).
FREQUENCY_COLUMNS: list[str] = [
    "user_id",           # str  — user identifier
    "rank",              # int  — rank of location (1 = most visited)
    "frequency",         # float — relative visit frequency
    "geohash6",          # str  — geohash6 of location
    "geohash7",          # str  — geohash7 of location
]

# ---------------------------------------------------------------------------
# Output file name templates  (format with .format(user, period, np_, t))
# ---------------------------------------------------------------------------
FNAME_SCALARS      = "all_scalars_{user}_period_{period}_np_{np_}_t_{t}.csv.gz"
FNAME_GONZALEZ     = "gonzalez_{user}_period_{period}_np_{np_}_t_{t}.csv.gz"
FNAME_ST           = "S_t_{user}_period_{period}_np_{np_}_t_{t}.csv.gz"
FNAME_FREQ_RANK    = "frequnecy_rank_{user}_period_{period}_np_{np_}_t_{t}.csv.gz"
FNAME_WEEKLY_RG    = "weekly_rg_{user}_period_{period}_np_{np_}_t_{t}.json"
FNAME_WEEK_NPEOPLE = "number_users_period_{period}.json"

# ---------------------------------------------------------------------------
# Legacy shard-level metric output directories
# (produced by the old per-parquet-shard pipeline)
#
# Two layouts exist:
#
#   CA (dataxuser/)
#     dataxuser/{metric}_{uid}_period_{period}_np_{np_}_t_{t}.csv.gz
#     One file per user per metric.
#
#   MA (metric-specific directories)
#     {metric_dir}/{np_}/{t}/{prefix}_{period}_{np_}_threshold_{t}_hour_{shard}.csv
#     One file per shard containing all users in that shard.
#
# Use get_legacy_metric_dir(metric, np_, t) to build the full path.
# ---------------------------------------------------------------------------

# MA metric folder names and per-metric CSV/JSON file-name prefixes.
# Layout:  milestones_analysis/MA/{folder}/{np_}/{t}/{prefix}_{period}...
MA_LEGACY_METRIC_DIRS: dict[str, dict] = {
    "rg": {
        "folder": "radius_gyration_measures_new_threshold",
        "prefix": "rg",                       # rg_{period}_{np_}_threshold_{t}_hour_...csv
        "file_ext": ".csv",
        "scalar_col": "radius_gyration",       # target all_scalars column name
    },
    "distance": {
        "folder": "distance_measures_new_threshold",
        "prefix": "dist",                      # dist_{period}_{np_}_threshold_{t}_hour_...csv
        "file_ext": ".csv",
        "scalar_col": "distance",
    },
    "k_rg": {
        "folder": "k_radius_gyration_measures_new_threshold",
        # k is encoded as prefix: 3k_rg, 6k_rg, 10k_rg
        "k_prefixes": {3: "3k_rg", 6: "6k_rg", 10: "10k_rg"},
        "file_ext": ".csv",
    },
    "entropic": {
        "folder": "entropic_measures_new_threshold",
        # Aggregate files — one file per period.
        # real_entropy_{period}_{np_}_threshold_{t}_hour.csv  (cols: ;values;uid)
        # uncorr_entropy_{period}_{np_}_threshold_{t}_hour.json  ({values, uid})
        # rdm_entropy_{period}_{np_}_threshold_{t}_hour.json   ({values, uid})
        "file_ext_csv": ".csv",
        "file_ext_json": ".json",
    },
    "gonzalez": {
        "folder": "gonzalez_new_threshold",
        "prefix": "gonzalez",
        "file_ext": ".json",   # {x_sigmax:{values,uid}, y_sigmay:{values,uid}, ...}
    },
    "S": {
        "folder": "st_new_threshold",
        "prefix": "dict_s",    # dict_s_{period}_{np_}_part-{shard}_hour_{t}_CA.csv
        "file_ext": ".csv",    # rows=users (no uid!), cols=time steps
    },
    "frequency": {
        "folder": "location_frequency_new_threshold",
        "file_ext": ".csv",
    },
    "home": {
        "folder": "home_new_threshold",
        "prefix": "home",      # home_{period}_{np_}_threshold_{t}_hour_subset_N.csv
        "file_ext": ".csv",    # cols: ;home_lat;home_lon;uid
    },
}

# Convenience mapping:  metric_key → legacy folder path (for CA dataxuser style too)
_LEGACY_METRIC_DIRS: dict = {
    "gonzalez":  "gonzalez_new_threshold",
    "rg":        "radius_gyration_measures_new_threshold",
    "k_rg":      "k_radius_gyration_measures_new_threshold",
    "entropic":  "entropic_measures_new_threshold",
    "S":         "st_new_threshold",
    "frequency": "location_frequency_new_threshold",
    "distance":  "distance_measures_new_threshold",
    "home":      "home_new_threshold",
}


def get_legacy_metric_dir(region: str, metric: str, np_: int, t: int) -> Path:
    """Return the shard-output directory for *metric* with given parameters.

    Parameters
    ----------
    region : str
        ``"CA"`` or ``"MA"``.
    metric : str
        One of ``_LEGACY_METRIC_DIRS`` keys, e.g. ``"rg"``, ``"gonzalez"``.
    np_ : int
        Minimum-points-per-user threshold (sub-directory level).
    t : int
        Time threshold in hours (sub-directory level).

    Returns
    -------
    Path
        For MA: ``DIR_MILESTONES_SERVER / "MA" / "{folder}" / str(np_) / str(t)``
        For CA: ``DIR_MILESTONES_SERVER / "CA" / "dataxuser"``
    """
    if region == "CA":
        return DIR_MILESTONES_SERVER / "CA" / "dataxuser"
    try:
        if region == "MA":
            folder = MA_LEGACY_METRIC_DIRS[metric]["folder"]
        else:
            folder = _LEGACY_METRIC_DIRS[metric]
    except KeyError:
        raise ValueError(f"Unknown metric {metric!r}. Choose from: {list(_LEGACY_METRIC_DIRS)}")
    return DIR_MILESTONES_SERVER / region / folder / str(np_) / str(t)
