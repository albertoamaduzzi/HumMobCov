"""
utils.py
========
Stateless utility functions shared across the project.

Performance notes
-----------------
* ``filter_()``  — the inner time-filter loop is compiled with **Numba JIT**
  on the first call (bytecode cached to ``__pycache__`` via ``cache=True``).
  Subsequent calls run at near-C speed. Falls back to a plain NumPy loop when
  Numba is not installed.

* ``xy()``  — the coordinate-projection arithmetic is compiled with Numba for
  fast element-wise array operations. Falls back to NumPy.

* ``t_stop()``  — datetime subtraction uses **Polars** columnar arithmetic
  (Rust-backed, SIMD-accelerated) when available; otherwise uses integer
  NumPy arithmetic (avoids float division).

* ``get_already_saved_user_per_period()``  — the directory tree is scanned
  with a ``ThreadPoolExecutor`` so multiple subdirectories are listed
  concurrently (important on NFS / network-mounted storage).
"""

import json
import math
import concurrent.futures
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ── optional fast backends ──────────────────────────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:                         # noqa: BLE001
    _NUMBA = False

try:
    import polars as pl
    _POLARS = True
except ImportError:                         # noqa: BLE001
    _POLARS = False

from .constants import (
    METRIC_FILE_KINDS,
    PERIOD_NAMES,
    DIR_OUTPUT,
    DIR_MILESTONES_SERVER,
    K_RADIUS_VALUES,
    FNAME_SCALARS,
    FNAME_GONZALEZ,
    MIN_POINTS_PER_USER,
    TIME_THRESHOLD_HOURS,
    get_legacy_metric_dir,
)


# ---------------------------------------------------------------------------
# Numba-compiled inner loops (or pure-NumPy fallbacks)
# ---------------------------------------------------------------------------

if _NUMBA:
    @_njit(cache=True, fastmath=True)
    def _filter_inner(x_arr: np.ndarray, t_threshold: float) -> np.ndarray:  # type: ignore[misc]
        """Forward scan compiled to native code. NOT parallelisable (stateful)."""
        n = len(x_arr)
        result = np.empty(n, dtype=np.bool_)
        if n == 0:
            return result
        result[0] = True
        cum_time = 0.0
        for i in range(1, n):
            val = x_arr[i]
            if val >= t_threshold:
                result[i] = True
                cum_time = 0.0
            else:
                cum_time += val
                if cum_time >= t_threshold:
                    result[i] = True
                    cum_time = 0.0
                else:
                    result[i] = False
        return result

    @_njit(cache=True, fastmath=True)
    def _xy_inner(                          # type: ignore[misc]
        lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
    ):
        """Element-wise tangent-plane projection compiled to native code."""
        PI = math.pi
        c_lat = 0.6 * 100_000 * (1.85533 - 0.006222 * math.sin(lat0 * PI / 180.0))
        c_lon = c_lat * math.cos(lat0 * PI / 180.0)
        return c_lon * (lon - lon0), c_lat * (lat - lat0)

else:
    # ── pure-NumPy fallbacks (same logic, no JIT) ───────────────────────────
    def _filter_inner(x_arr: np.ndarray, t_threshold: float) -> np.ndarray:
        n = len(x_arr)
        result = np.empty(n, dtype=np.bool_)
        if n == 0:
            return result
        result[0] = True
        cum_time = 0.0
        for i in range(1, n):
            val = x_arr[i]
            if val >= t_threshold:
                result[i] = True
                cum_time = 0.0
            else:
                cum_time += val
                if cum_time >= t_threshold:
                    result[i] = True
                    cum_time = 0.0
                else:
                    result[i] = False
        return result

    def _xy_inner(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float):
        PI = math.pi
        c_lat = 0.6 * 100_000 * (1.85533 - 0.006222 * math.sin(lat0 * PI / 180.0))
        c_lon = c_lat * math.cos(lat0 * PI / 180.0)
        return c_lon * (lon - lon0), c_lat * (lat - lat0)


# ---------------------------------------------------------------------------
# Trajectory / spatial helpers
# ---------------------------------------------------------------------------

def filter_(x, t_threshold: int) -> list:
    """
    Time-filter a series of inter-stop time differences.

    Keeps a row only when the *cumulative* time since the last kept row
    is >= ``t_threshold`` hours.  The very first row is always kept.

    The inner loop is compiled by Numba on the first call and cached;
    subsequent calls run at near-C speed.

    Parameters
    ----------
    x : array-like
        Series of inter-stop time differences (hours).
    t_threshold : int
        Minimum elapsed hours before accepting the next stop.

    Returns
    -------
    list[bool]
        Boolean mask aligned with ``x``.
    """
    arr = np.asarray(x, dtype=np.float64)
    return _filter_inner(arr, float(t_threshold)).tolist()


def xy(lat, lon, lat0: float, lon0: float):
    """
    Project geographic coordinates onto a local tangent plane anchored at
    ``(lat0, lon0)``.

    The arithmetic kernel is compiled by Numba when available.

    Parameters
    ----------
    lat, lon : array-like
        Coordinates to project.
    lat0, lon0 : float
        Origin of the tangent plane.

    Returns
    -------
    x, y : np.ndarray
        Projected coordinates in metres.
    """
    return _xy_inner(
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        float(lat0),
        float(lon0),
    )


def t_stop(df) -> list:
    """
    Compute stop durations in minutes for each row of a TrajDataFrame.

    Uses **Polars** columnar arithmetic (Rust / SIMD) when available;
    otherwise falls back to integer NumPy arithmetic.

    Parameters
    ----------
    df : skmob.TrajDataFrame
        Must contain ``datetime`` (start) and ``end`` columns.

    Returns
    -------
    list[int]
    """
    if _POLARS:
        try:
            # Convert to ms-epoch int64 arrays, build Polars Duration series
            start_ms = df["datetime"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
            end_ms   = df["end"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
            duration_min = (
                pl.Series(end_ms - start_ms, dtype=pl.Duration("ms"))
                .dt.total_minutes()
            )
            return duration_min.cast(pl.Int32).to_list()
        except Exception:                   # noqa: BLE001
            pass                            # fall through to NumPy

    # NumPy fallback — integer arithmetic avoids float division
    start_s = df["datetime"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    end_s   = df["end"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    return ((end_s - start_s) // 60).astype(np.int32).tolist()


def time_difference(start, end) -> float:
    """Return elapsed time in hours between two datetime objects."""
    return (end - start).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# File-system helpers
def get_cpu_quota() -> int:
    """
    Return the number of CPUs actually available to this process.

    On Linux containers (Docker / Kubernetes) ``os.cpu_count()`` reports
    *all host CPUs*, not the cgroup quota.  This function reads the real
    quota from the cgroup v2 ``cpu.max`` file (``<quota> <period>``).
    Falls back to ``os.cpu_count()`` if the file is absent or unparseable.
    """
    import os as _os
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota_str, period_str = f.read().split()
        if quota_str != "max":
            return max(1, int(float(quota_str) / float(period_str)))
    except Exception:
        pass
    return _os.cpu_count() or 1


# ---------------------------------------------------------------------------

def ifnotexistsmkdir(dir_: Path | str) -> Path:
    """Create ``dir_`` if it does not exist and return it as a Path."""
    p = Path(dir_)
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_pth(save_dir: Path | str, np_: int, t_threshold: int) -> Path:
    """
    Build and create the nested ``save_dir/<np_>/<t_threshold>/`` directory.

    Returns
    -------
    Path
    """
    return ifnotexistsmkdir(Path(save_dir) / str(np_) / str(t_threshold))


# ---------------------------------------------------------------------------
# Per-period checkpoint helpers
# ---------------------------------------------------------------------------

def _parse_fname(fname: str) -> tuple[str | None, str | None, str | None]:
    """
    Derive ``(period, metric_kind, user_id)`` from an output filename, or
    ``(None, None, None)`` if the name does not match any known pattern.
    """
    if "jan - " in fname:
        period = "15 jan - 15 march"
    elif "march - " in fname:
        period = "15 march - 15 may"
    elif "may - " in fname:
        period = "15 may - sept"
    else:
        return None, None, None

    for kind in METRIC_FILE_KINDS:
        if kind in fname:
            parts = fname.split("_")
            user  = parts[1] if "gonzalez" in fname else parts[2]
            return period, kind, user

    return None, None, None


def _list_files_in_dir(d: Path) -> list[str]:
    """Return names of all files directly inside ``d`` (non-recursive)."""
    try:
        return [f.name for f in d.iterdir() if f.is_file()]
    except (PermissionError, NotADirectoryError):
        return []


def get_already_saved_user_per_period(directory: Path | str) -> dict:
    """
    Scan ``directory`` and reconstruct the checkpoint dict::

        {period: {metric_kind: [user_id, ...]}}

    by parsing file names that encode period, metric kind, and user id.

    The directory tree is listed with a ``ThreadPoolExecutor`` so multiple
    subdirectories are enumerated concurrently (beneficial on NFS storage).
    """
    directory = Path(directory)
    period2user: dict = {p: {k: [] for k in METRIC_FILE_KINDS} for p in PERIOD_NAMES}

    # Collect all directories (root + every subdirectory)
    all_dirs = [directory] + [d for d in directory.rglob("*") if d.is_dir()]

    # Parallel I/O: list each directory in its own thread
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fname_lists = list(executor.map(_list_files_in_dir, all_dirs))

    # Parse filenames (pure string ops — no benefit from further parallelism)
    for fnames in fname_lists:
        for fname in fnames:
            period, kind, user = _parse_fname(fname)
            if period is not None:
                period2user[period][kind].append(user)

    return period2user


def update_already_saved_users(
    already_saved: dict,
    period: str,
    user: str,
    dataset_id: str,
    output_dir: Path | str | None = None,
) -> None:
    """
    Persist the updated ``already_saved`` checkpoint as a compressed CSV.

    Parameters
    ----------
    already_saved : dict
        Mapping ``{period: [user_ids]}`` updated in place by the caller.
    period : str
        Period key that was updated.
    user : str
        The user that was just processed.
    dataset_id : str
        Region identifier, e.g. ``"CA"`` or ``"MA"``.
    output_dir : Path or str, optional
        Override for the base output directory.  Defaults to
        ``DIR_OUTPUT / dataset_id``.
    """
    base = Path(output_dir) if output_dir else DIR_MILESTONES_SERVER / dataset_id
    ifnotexistsmkdir(base)

    already_saved[period] = list(np.unique(already_saved[period]))
    df_users = pd.DataFrame(already_saved[period], columns=["users"])
    checkpoint_path = base / "already_saved_users_per_period.csv.gz"
    df_users.to_csv(checkpoint_path, index=False, compression="gzip")


def upload_already_saved_users(
    period_names: list,
    dataset_id: str,
    output_dir: Path | str | None = None,
) -> dict:
    """
    Load the user checkpoint from disk.  Returns an empty list per period
    if no checkpoint file exists.
    """
    base = Path(output_dir) if output_dir else DIR_MILESTONES_SERVER / dataset_id
    csv_path  = base / "already_saved_users_per_period.csv.gz"
    json_path = base / "already_saved_users_per_period.json"

    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    if csv_path.exists():
        df    = pd.read_csv(csv_path, compression="gzip")
        users = df["users"].tolist()
        return {p: users for p in period_names}
    return {p: [] for p in period_names}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def init_compare_periods_dict(period_names: dict, measures: list, type_: str) -> dict:
    """
    Initialise a nested dict for pairwise period comparisons::

        {
          "period_i-period_j": {
              measure_0: 0.  or [],
              ...
          }
        }

    Parameters
    ----------
    period_names : dict
        Keys are period name strings.
    measures : list[str]
        Measure names to include.
    type_ : str
        ``"scalar"`` initialises values to ``0.``; ``"list"`` to ``[]``.
    """
    tmp: dict = defaultdict()
    for k in period_names:
        for k1 in period_names:
            if k1 > k:
                tmp[f"{k}-{k1}"] = defaultdict()
                for m in measures:
                    tmp[f"{k}-{k1}"][m] = 0.0 if type_ == "scalar" else []
    return tmp


# ---------------------------------------------------------------------------
# Extraction: build dataxuser/ from legacy per-shard metric files
# ---------------------------------------------------------------------------

def _read_scalar_shard_dir(
    shard_dir: Path,
    period: str,
    prefix: str = "",
) -> dict[str, float]:
    """
    Read all ``;values;uid`` CSV/JSON files inside *shard_dir* whose names
    contain *period* (and optionally start with *prefix*).

    Returns
    -------
    dict
        ``{uid: value}`` — one scalar per user (last shard wins on collision,
        but each shard file should cover a disjoint set of users).
    """
    uid2val: dict[str, float] = {}
    if not shard_dir.exists():
        return uid2val
    for f in sorted(shard_dir.iterdir()):
        if not f.is_file():
            continue
        if period not in f.name:
            continue
        if prefix and not f.name.startswith(prefix):
            continue
        try:
            df = pd.read_csv(f, sep=";", index_col=0)
            if {"uid", "values"} <= set(df.columns):
                for uid, val in zip(df["uid"].values, df["values"].values):
                    uid2val[str(uid)] = float(val)
        except Exception:  # noqa: BLE001
            pass
    return uid2val


def _read_gonzalez_shard_dir(
    gon_dir: Path,
    period: str,
) -> dict[str, pd.DataFrame]:
    """
    Read all gonzalez JSON shard files for *period* in *gon_dir*.

    Each JSON has the structure::

        {
          "x_sigmax": {"values": [...], "uid": [...]},
          "y_sigmay": {"values": [...], "uid": [...]},
          "sigmax":   {"values": [...], "uid": [...]},
          "sigmay":   {"values": [...], "uid": [...]},
          ...
        }

    Returns
    -------
    dict
        ``{uid: DataFrame}`` with columns
        ``["x_norm", "y_norm", "sigmax", "sigmay"]``
        (one row per visit / stop-point).
    """
    uid2rows: dict[str, list] = defaultdict(list)
    if not gon_dir.exists():
        return {}
    for f in sorted(gon_dir.iterdir()):
        if not f.is_file() or f.suffix != ".json":
            continue
        if period not in f.name:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
            uids   = d["x_sigmax"]["uid"]
            xnorm  = d["x_sigmax"]["values"]
            ynorm  = d["y_sigmay"]["values"]
            sigx   = d["sigmax"]["values"]
            sigy   = d["sigmay"]["values"]
            for i, uid in enumerate(uids):
                uid2rows[str(uid)].append({
                    "x_norm": xnorm[i],
                    "y_norm": ynorm[i],
                    "sigmax": sigx[i],
                    "sigmay": sigy[i],
                })
        except Exception:  # noqa: BLE001
            pass
    return {uid: pd.DataFrame(rows) for uid, rows in uid2rows.items()}


def extract_dataxuser_from_shards(
    region: str,
    np_: int = MIN_POINTS_PER_USER,
    t: int = TIME_THRESHOLD_HOURS,
    output_dir: Path | str | None = None,
) -> dict:
    """
    Read legacy per-shard metric files from
    ``milestones_analysis/{metric}_new_threshold/{np_}/{t}/``
    and write per-user CSV.gz files into the ``dataxuser/`` directory.

    This function bridges the old per-parquet-shard pipeline output and the
    new per-user pipeline that ``pipeline.py`` and ``plotter.py`` expect.

    Metrics extracted
    -----------------
    * **all_scalars** CSV.gz — columns: ``radius_gyration``,
      ``random_entropy``, ``uncorrelated_entropy``, ``real_entropy``,
      ``distance``, ``rg_{k}`` for each k in K_RADIUS_VALUES.
      Fields that are unavailable for the given np_/t combination are
      written as ``NaN``.  ``party_government`` and ``rurality_level``
      require a census join and are always ``NaN`` here.
    * **gonzalez** CSV.gz — columns: ``x_norm``, ``y_norm``,
      ``sigmax``, ``sigmay`` (one row per stop-point visit).

    Parameters
    ----------
    region : str
        ``"CA"`` or ``"MA"``.
    np_ : int
        Minimum-points threshold (used for directory lookup and file names).
    t : int
        Time-threshold in hours (used for directory lookup and file names).
    output_dir : Path or str, optional
        Override for the ``dataxuser/`` base directory.  Defaults to
        ``DIR_MILESTONES_SERVER / region / "dataxuser"``.

    Returns
    -------
    dict
        Checkpoint dict ``{period: {kind: [user_ids, ...]}}``.
    """
    base_out = (
        Path(output_dir)
        if output_dir
        else DIR_MILESTONES_SERVER / region / "dataxuser"
    )
    ifnotexistsmkdir(base_out)

    checkpoint = {p: {k: [] for k in METRIC_FILE_KINDS} for p in PERIOD_NAMES}

    for period in PERIOD_NAMES:
        print(f"  [{region}] Extracting period: {period!r} ...")

        # ── scalar metrics ─────────────────────────────────────────────────
        uid2rg   = _read_scalar_shard_dir(get_legacy_metric_dir("rg",       np_, t), period)
        uid2dist = _read_scalar_shard_dir(get_legacy_metric_dir("distance",  np_, t), period)
        uid2rdm  = _read_scalar_shard_dir(get_legacy_metric_dir("entropic",  np_, t), period,
                                          prefix="rdm_entropy")
        uid2unc  = _read_scalar_shard_dir(get_legacy_metric_dir("entropic",  np_, t), period,
                                          prefix="uncorr_entropy")
        uid2real = _read_scalar_shard_dir(get_legacy_metric_dir("entropic",  np_, t), period,
                                          prefix="real_entropy")

        uid2krg: dict[int, dict[str, float]] = {}
        for k in K_RADIUS_VALUES:
            uid2krg[k] = _read_scalar_shard_dir(
                get_legacy_metric_dir("k_rg", np_, t), period,
                prefix=f"{k}k_rg",
            )

        all_scalar_users = (
            set(uid2rg) | set(uid2rdm) | set(uid2unc)
            | set(uid2real) | set(uid2dist)
        )
        for k in K_RADIUS_VALUES:
            all_scalar_users |= set(uid2krg[k])

        for uid in all_scalar_users:
            row: dict = {
                "radius_gyration":      uid2rg.get(uid, float("nan")),
                "random_entropy":       uid2rdm.get(uid, float("nan")),
                "uncorrelated_entropy": uid2unc.get(uid, float("nan")),
                "real_entropy":         uid2real.get(uid, float("nan")),
                "distance":             uid2dist.get(uid, float("nan")),
                "party_government":     float("nan"),
                "rurality_level":       float("nan"),
            }
            for k in K_RADIUS_VALUES:
                row[f"rg_{k}"] = uid2krg[k].get(uid, float("nan"))

            fname = base_out / FNAME_SCALARS.format(
                user=uid, period=period, np_=np_, t=t
            )
            pd.DataFrame([row]).to_csv(fname, index=False, compression="gzip")
            checkpoint[period]["all_scalars"].append(uid)

        print(f"    → {len(all_scalar_users)} users written (all_scalars).")

        # ── gonzalez (per-visit) ───────────────────────────────────────────
        uid2visits = _read_gonzalez_shard_dir(
            get_legacy_metric_dir("gonzalez", np_, t), period
        )
        for uid, visits_df in uid2visits.items():
            fname = base_out / FNAME_GONZALEZ.format(
                user=uid, period=period, np_=np_, t=t
            )
            visits_df.to_csv(fname, index=False, compression="gzip")
            checkpoint[period]["gonzalez"].append(uid)

        print(f"    → {len(uid2visits)} users written (gonzalez).")

    return checkpoint
