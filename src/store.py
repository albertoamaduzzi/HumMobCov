"""
store.py
========
High-throughput columnar parquet storage for per-user mobility metrics.

Storage format — "users as columns"
------------------------------------
Each (period, metric_kind) maps to a **shard directory** that accumulates
write-once parquet shard files.  Users are columns; the row index is a
fixed semantic dimension:

    all_scalars  →  index col ``"metric"``  (one row per scalar name)
    S            →  index col ``"time"``    (one row per time step 0…1419)
    weekly_rg    →  index col ``"week"``    (one row per week key)

Variable-length kinds (gonzalez, frequency) are stored as **long-format**
tables with a ``"user_id"`` column.

Resume logic
------------
Fixed-length kinds: ``get_computed_users()`` reads only the parquet
*footer* (schema metadata) to obtain column names — no row data loaded.

Variable-length kinds: ``get_computed_users_long()`` reads only the
``user_id`` column.

Shard files are write-once (never modified after creation).
``consolidate()`` merges all shards into one ``consolidated.parquet``.

Migration
---------
``migrate_from_legacy()`` reads the old per-user ``*.csv.gz`` / ``*.json``
files from ``dataxuser/`` and writes them into the new parquet store,
skipping already-migrated users.

Plotter reading
---------------
``read_scalars(period)`` returns a Polars DataFrame shaped
``[n_users × (n_metrics + 1)]`` (transposed from the on-disk format).

``read_st_matrix(period)`` / ``read_weekly_rg_matrix(period)`` return the
raw ``[n_rows × (1 + n_users)]`` matrices.
"""

from __future__ import annotations

import json as _json
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import re as _re

from .constants import (
    K_RADIUS_VALUES,
    TIME_INTERVAL_S_MAX,
    MA_LEGACY_METRIC_DIRS,
)
from .utils import ifnotexistsmkdir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_LENGTH_KINDS: frozenset[str] = frozenset({"all_scalars", "weekly_rg", "S"})
LONG_FORMAT_KINDS:  frozenset[str] = frozenset({"gonzalez", "frequency"})
ALL_KINDS: frozenset[str]          = FIXED_LENGTH_KINDS | LONG_FORMAT_KINDS

ALL_SCALAR_METRICS: list[str] = [
    "radius_gyration",
    "random_entropy",
    "uncorrelated_entropy",
    "real_entropy",
    "distance",
    "q",
    "home",
    "home_geohash7",
    "county_home",
    "party_government",
    "rurality_level",
] + [f"rg_{k}" for k in K_RADIUS_VALUES]

# Metrics stored as strings (not floats)
_STR_SCALAR_METRICS: frozenset[str] = frozenset({
    "home", "home_geohash7", "county_home", "party_government", "rurality_level",
})

_INDEX_COL: dict[str, str] = {
    "all_scalars": "metric",
    "weekly_rg":   "week",
    "S":           "time",
}

CONSOLIDATED_FNAME = "consolidated.parquet"


def _safe_period(period: str) -> str:
    """Replace characters unsafe for directory names."""
    return period.replace(" ", "_").replace("/", "-")


# ---------------------------------------------------------------------------
# ParquetStore
# ---------------------------------------------------------------------------

class ParquetStore:
    """
    Columnar parquet storage for per-user mobility metrics.

    Directory layout::

        base_dir/
            {kind}_period_{period_safe}_np_{np_}_t_{t}/
                shard_{monotonic_ns}.parquet   (users as columns)
                consolidated.parquet           (merged, created on demand)

    Parameters
    ----------
    base_dir : Path or str
        Root directory for shard sub-directories.
    np_ : int
        Min-points threshold (embedded in directory names).
    t_threshold : int
        Time threshold in hours (embedded in directory names).
    """

    def __init__(
        self,
        base_dir: Path | str,
        np_: int,
        t_threshold: int,
    ) -> None:
        self.base_dir    = Path(base_dir)
        self.np_         = np_
        self.t_threshold = t_threshold

    # ------------------------------------------------------------------
    # Internal path helpers
    # ------------------------------------------------------------------

    def _shard_dir(self, period: str, kind: str) -> Path:
        name = (
            f"{kind}_period_{_safe_period(period)}"
            f"_np_{self.np_}_t_{self.t_threshold}"
        )
        return ifnotexistsmkdir(self.base_dir / name)

    def _consolidated_path(self, period: str, kind: str) -> Path:
        return self._shard_dir(period, kind) / CONSOLIDATED_FNAME

    def _new_shard_path(self, period: str, kind: str) -> Path:
        ts = _time.monotonic_ns()
        return self._shard_dir(period, kind) / f"shard_{ts}.parquet"

    def _shard_files(self, period: str, kind: str) -> list[Path]:
        return sorted(self._shard_dir(period, kind).glob("shard_*.parquet"))

    def _all_files(self, period: str, kind: str) -> list[Path]:
        """Consolidated (if exists) + all shard files."""
        cp = self._consolidated_path(period, kind)
        return ([cp] if cp.exists() else []) + self._shard_files(period, kind)

    # ------------------------------------------------------------------
    # Resume / checkpoint
    # ------------------------------------------------------------------

    def get_computed_users(self, period: str, kind: str) -> set[str]:
        """
        Return user IDs already stored for (period, kind).

        Reads only parquet *footer metadata* — no row data is loaded.
        """
        index_col = _INDEX_COL.get(kind)
        users: set[str] = set()
        for f in self._all_files(period, kind):
            try:
                schema = pl.read_parquet_schema(f)
                users.update(n for n in schema if n != index_col)
            except Exception:
                continue
        return users

    def get_computed_users_long(self, period: str, kind: str) -> set[str]:
        """
        Resume check for long-format kinds (gonzalez, frequency).
        Reads only the ``user_id`` column.
        """
        users: set[str] = set()
        for f in self._all_files(period, kind):
            try:
                users.update(
                    pl.scan_parquet(f)
                    .select("user_id")
                    .collect()["user_id"]
                    .cast(pl.Utf8)
                    .to_list()
                )
            except Exception:
                continue
        return users

    def get_s3_computed_users(
        self,
        period: str,
        kind: str,
        s3_bucket: str,
        s3_prefix: str,
        endpoint_url: str,
    ) -> set[str]:
        """
        Return user IDs already stored for (period, kind) on S3.

        Downloads the consolidated parquet to a temporary file, reads user
        identifiers from its metadata (fixed-length kinds) or ``user_id``
        column (long-format kinds), then deletes the temp file.

        Returns an empty set if the S3 object does not exist or the download
        fails.
        """
        from .s3_io import s3_read_parquet, s3_read_parquet_schema

        shard_dir_name = self._shard_dir(period, kind).name
        s3_key = f"{s3_prefix}/{shard_dir_name}/{CONSOLIDATED_FNAME}"

        users: set[str] = set()
        try:
            if kind in FIXED_LENGTH_KINDS:
                index_col = _INDEX_COL.get(kind)
                schema = s3_read_parquet_schema(s3_bucket, s3_key, endpoint_url)
                if schema is not None:
                    users.update(n for n in schema if n != index_col)
            else:
                df = s3_read_parquet(s3_bucket, s3_key, endpoint_url)
                if df is not None:
                    users.update(
                        df.select("user_id")["user_id"].cast(pl.Utf8).to_list()
                    )
        except Exception:
            pass
        return users

    def get_s3_computed_users_long(
        self,
        period: str,
        kind: str,
        s3_bucket: str,
        s3_prefix: str,
        endpoint_url: str,
    ) -> set[str]:
        """Alias kept for symmetry; delegates to ``get_s3_computed_users``."""
        return self.get_s3_computed_users(
            period, kind, s3_bucket, s3_prefix, endpoint_url
        )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_scalars_batch(
        self,
        period: str,
        batch: dict[str, dict[str, Any]],
    ) -> None:
        """
        Write a batch of scalar results for ``period``.

        Parameters
        ----------
        batch : {user_id: {metric_name: value}}
        """
        if not batch:
            return

        # Store every value as Utf8 to avoid type-inference errors when
        # a single user column contains both float and string metrics.
        # Numeric values are cast back to Float64 in read_scalars().
        data: dict[str, list[str]] = {"metric": ALL_SCALAR_METRICS}
        for uid, metrics in batch.items():
            col: list[str] = []
            for m in ALL_SCALAR_METRICS:
                v = metrics.get(m)
                if v is None:
                    col.append("")
                else:
                    col.append(str(v))
            data[str(uid)] = col

        pl.DataFrame(data).write_parquet(
            self._new_shard_path(period, "all_scalars"),
            compression="snappy",
        )

    def write_st_batch(
        self,
        period: str,
        batch: dict[str, list],
    ) -> None:
        """
        Write S(t) exploration curves for a batch of users.

        Parameters
        ----------
        batch : {user_id: list[int]}  length == TIME_INTERVAL_S_MAX - 1
        """
        if not batch:
            return

        n_steps = TIME_INTERVAL_S_MAX - 1
        data: dict[str, list] = {"time": list(range(n_steps))}
        for uid, values in batch.items():
            if len(values) >= n_steps:
                data[str(uid)] = [int(v) for v in values[:n_steps]]
            else:
                last = int(values[-1]) if values else 0
                data[str(uid)] = (
                    [int(v) for v in values] + [last] * (n_steps - len(values))
                )

        pl.DataFrame(data).write_parquet(
            self._new_shard_path(period, "S"),
            compression="snappy",
        )

    def write_weekly_rg_batch(
        self,
        period: str,
        batch: dict[str, dict],
        all_weeks: list,
    ) -> None:
        """
        Write weekly RG for a batch of users.

        Parameters
        ----------
        batch : {user_id: {week_key: rg_value}}
        all_weeks : list
            Ordered list of all week keys for this period (determines rows).
        """
        if not batch:
            return

        data: dict[str, list] = {"week": [str(w) for w in all_weeks]}
        for uid, week2rg in batch.items():
            col: list = []
            for w in all_weeks:
                v = week2rg.get(w, week2rg.get(str(w)))
                try:
                    col.append(float(v) if v is not None else float("nan"))
                except (TypeError, ValueError):
                    col.append(float("nan"))
            data[str(uid)] = col

        pl.DataFrame(data).write_parquet(
            self._new_shard_path(period, "weekly_rg"),
            compression="snappy",
        )

    def write_gonzalez_batch(
        self,
        period: str,
        batch: dict[str, Any],  # {user_id: pd.DataFrame}
    ) -> None:
        """Write Gonzalez PCA data for a batch of users (long format)."""
        import pandas as _pd

        if not batch:
            return

        frames: list[_pd.DataFrame] = []
        for uid, df in batch.items():
            if not isinstance(df, _pd.DataFrame):
                continue
            _GON_COLS = {"x_norm", "y_norm", "sigmax", "sigmay"}
            if not _GON_COLS.issubset(df.columns) or df.empty:
                continue   # skip malformed / empty gonzalez frames
            sub = df[["x_norm", "y_norm", "sigmax", "sigmay"]].copy()
            sub["user_id"] = str(uid)
            frames.append(sub)

        if frames:
            combined = _pd.concat(frames, ignore_index=True)
            pl.from_pandas(combined).write_parquet(
                self._new_shard_path(period, "gonzalez"),
                compression="snappy",
            )

    def write_frequency_batch(
        self,
        period: str,
        batch: dict[str, Any],  # {user_id: pd.DataFrame}
    ) -> None:
        """Write frequency/rank data for a batch of users (long format)."""
        import pandas as _pd

        if not batch:
            return

        frames: list[_pd.DataFrame] = []
        for uid, df in batch.items():
            sub = df.copy()
            sub["user_id"] = str(uid)
            frames.append(sub)

        if frames:
            combined = _pd.concat(frames, ignore_index=True)
            pl.from_pandas(combined).write_parquet(
                self._new_shard_path(period, "frequency"),
                compression="snappy",
            )

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, period: str, kind: str) -> Path:
        """
        Merge all shard files into ``consolidated.parquet`` and delete shards.

        For fixed-length kinds: horizontal concat (shared index column).
        For long-format kinds: vertical concat.

        Returns the path to the consolidated file.
        """
        shards = self._shard_files(period, kind)
        if not shards:
            return self._consolidated_path(period, kind)

        cp = self._consolidated_path(period, kind)

        if kind in FIXED_LENGTH_KINDS:
            frames: list[pl.DataFrame] = []
            if cp.exists():
                try:
                    frames.append(pl.read_parquet(cp))
                except Exception as _e:
                    print(f"  [store] WARNING: dropping corrupt consolidated file {cp.name}: {_e}")
                    cp.unlink(missing_ok=True)
            for f in shards:
                try:
                    frames.append(pl.read_parquet(f))
                except Exception as _e:
                    print(f"  [store] WARNING: dropping corrupt shard {f.name}: {_e}")
                    f.unlink(missing_ok=True)
            if not frames:
                return cp
            merged = _hconcat_fixed(frames, _INDEX_COL[kind])
            merged.write_parquet(cp, compression="snappy")
        else:
            frames = []
            if cp.exists():
                try:
                    frames.append(pl.read_parquet(cp))
                except Exception as _e:
                    print(f"  [store] WARNING: dropping corrupt consolidated file {cp.name}: {_e}")
                    cp.unlink(missing_ok=True)
            for f in shards:
                try:
                    frames.append(pl.read_parquet(f))
                except Exception as _e:
                    print(f"  [store] WARNING: dropping corrupt shard {f.name}: {_e}")
                    f.unlink(missing_ok=True)
            if not frames:
                return cp
            pl.concat(frames, how="diagonal_relaxed").write_parquet(
                cp, compression="snappy"
            )

        for f in shards:
            f.unlink(missing_ok=True)

        return cp

    def consolidate_all(self, period: str) -> None:
        """Consolidate all metric kinds for ``period``."""
        for kind in list(FIXED_LENGTH_KINDS) + list(LONG_FORMAT_KINDS):
            if self._shard_files(period, kind):
                self.consolidate(period, kind)

    # ------------------------------------------------------------------
    # Reading (used by plotter)
    # ------------------------------------------------------------------

    def read_scalars(self, period: str) -> pl.DataFrame:
        """
        Load all users' scalar metrics for ``period``.

        Returns a Polars DataFrame shaped ``[n_users × (n_metrics + 1)]``
        with a ``"user_id"`` column.  The on-disk format is transposed
        (metrics × users) for efficient append; this method transposes it
        back for analytical use.
        """
        raw = self._read_fixed(period, "all_scalars")
        if raw is None or raw.is_empty():
            return pl.DataFrame()
        return _transpose_metrics_to_users(raw)

    def read_st_matrix(self, period: str) -> pl.DataFrame:
        """
        S(t) matrix: ``[n_time_steps × (1 + n_users)]`` with ``"time"`` col.
        Users are columns.
        """
        result = self._read_fixed(period, "S")
        return result if result is not None else pl.DataFrame()

    def read_weekly_rg_matrix(self, period: str) -> pl.DataFrame:
        """
        Weekly-RG matrix: ``[n_weeks × (1 + n_users)]`` with ``"week"`` col.
        Users are columns.
        """
        result = self._read_fixed(period, "weekly_rg")
        return result if result is not None else pl.DataFrame()

    def read_gonzalez(self, period: str) -> pl.DataFrame:
        """Gonzalez long-format table: [n_visits, 5]."""
        result = self._read_long(period, "gonzalez")
        return result if result is not None else pl.DataFrame()

    def read_frequency(self, period: str) -> pl.DataFrame:
        """Frequency/rank long-format table: [n_locations, 3]."""
        result = self._read_long(period, "frequency")
        return result if result is not None else pl.DataFrame()

    def _read_fixed(self, period: str, kind: str) -> pl.DataFrame | None:
        cp = self._consolidated_path(period, kind)
        if cp.exists():
            return pl.read_parquet(cp)
        shards = self._shard_files(period, kind)
        if not shards:
            return None
        return _hconcat_fixed(
            [pl.read_parquet(f) for f in shards],
            _INDEX_COL[kind],
        )

    def _read_long(self, period: str, kind: str) -> pl.DataFrame | None:
        cp = self._consolidated_path(period, kind)
        if cp.exists():
            return pl.read_parquet(cp)
        shards = self._shard_files(period, kind)
        if not shards:
            return None
        return pl.concat(
            [pl.read_parquet(f) for f in shards],
            how="diagonal_relaxed",
        )

    # ------------------------------------------------------------------
    # Migration from legacy per-user CSV.gz / JSON files
    # ------------------------------------------------------------------

    def migrate_from_legacy(
        self,
        legacy_dir: Path | str,
        period: str,
        np_: int,
        t: int,
        batch_size: int = 500,
        verbose: bool = True,
    ) -> None:
        """
        Read legacy per-user files from ``legacy_dir`` and write them into
        the new columnar parquet store.  Already-migrated users are skipped.

        This is a **one-time, non-production** migration step.

        Parameters
        ----------
        legacy_dir : Path or str
            ``dataxuser/`` directory containing the old per-user files.
        period : str
            Period name (e.g. ``"15 jan - 15 march"``).
        np_, t : int
            Parameters encoded in legacy file names (for filtering).
        batch_size : int
            Number of users to accumulate before writing one shard file.
        verbose : bool
            Print progress messages.
        """
        import pandas as _pd

        legacy_dir = Path(legacy_dir)

        # Resume: read only parquet footers — no data loaded
        done_scalars  = self.get_computed_users(period, "all_scalars")
        done_gonzalez = self.get_computed_users_long(period, "gonzalez")
        done_st       = self.get_computed_users(period, "S")
        done_freq     = self.get_computed_users_long(period, "frequency")
        done_wrg      = self.get_computed_users(period, "weekly_rg")

        scalar_batch:   dict[str, dict]       = {}
        gonzalez_batch: dict[str, _pd.DataFrame] = {}
        st_batch:       dict[str, list]       = {}
        freq_batch:     dict[str, _pd.DataFrame] = {}
        wrg_batch:      dict[str, dict]       = {}

        count = 0

        def _flush(force: bool = False) -> None:
            if force or len(scalar_batch) >= batch_size:
                self.write_scalars_batch(period, scalar_batch)
                scalar_batch.clear()
            if force or len(gonzalez_batch) >= batch_size:
                self.write_gonzalez_batch(period, gonzalez_batch)
                gonzalez_batch.clear()
            if force or len(st_batch) >= batch_size:
                self.write_st_batch(period, st_batch)
                st_batch.clear()
            if force or len(freq_batch) >= batch_size:
                self.write_frequency_batch(period, freq_batch)
                freq_batch.clear()

        for f in legacy_dir.iterdir():
            if not f.is_file():
                continue
            fname = f.name
            if period not in fname:
                continue

            # ── scalars ──────────────────────────────────────────────
            if fname.startswith("all_scalars_") and fname.endswith(".csv.gz"):
                uid = _extract_uid(fname, "all_scalars_", ".csv.gz")
                if uid is None or uid in done_scalars:
                    continue
                try:
                    row = _pd.read_csv(f, compression="gzip").iloc[0].to_dict()
                    scalar_batch[uid] = row
                    count += 1
                except Exception:
                    continue

            # ── gonzalez ─────────────────────────────────────────────
            elif fname.startswith("gonzalez_") and fname.endswith(".csv.gz"):
                uid = _extract_uid(fname, "gonzalez_", ".csv.gz")
                if uid is None or uid in done_gonzalez:
                    continue
                try:
                    gonzalez_batch[uid] = _pd.read_csv(f, compression="gzip")
                    count += 1
                except Exception:
                    continue

            # ── S(t) ──────────────────────────────────────────────────
            elif fname.startswith("S_t_") and fname.endswith(".csv.gz"):
                uid = _extract_uid(fname, "S_t_", ".csv.gz")
                if uid is None or uid in done_st:
                    continue
                try:
                    df = _pd.read_csv(f, compression="gzip")
                    st_batch[uid] = df["visited_places"].tolist()
                    count += 1
                except Exception:
                    continue

            # ── frequency ────────────────────────────────────────────
            elif fname.startswith("frequnecy_rank_") and fname.endswith(".csv.gz"):
                uid = _extract_uid(fname, "frequnecy_rank_", ".csv.gz")
                if uid is None or uid in done_freq:
                    continue
                try:
                    freq_batch[uid] = _pd.read_csv(f, compression="gzip")
                    count += 1
                except Exception:
                    continue

            if count > 0 and count % batch_size == 0:
                _flush()
                if verbose:
                    print(f"    [{period}] flushed {count} users …")

        _flush(force=True)

        # ── weekly RG (JSON) ──────────────────────────────────────────
        all_weeks_set: set = set()
        for f in legacy_dir.iterdir():
            if not f.is_file() or f.suffix != ".json":
                continue
            if period not in f.name or not f.name.startswith("weekly_rg_"):
                continue
            uid = _extract_uid(f.name, "weekly_rg_", ".json")
            if uid is None or uid in done_wrg:
                continue
            try:
                with open(f) as fh:
                    week2rg = _json.load(fh)
                wrg_batch[uid] = week2rg
                all_weeks_set.update(week2rg.keys())
            except Exception:
                continue

        if wrg_batch:
            all_weeks = sorted(all_weeks_set)
            self.write_weekly_rg_batch(period, wrg_batch, all_weeks)
            if verbose:
                print(
                    f"    [{period}] {len(wrg_batch)} users migrated (weekly_rg)."
                )

        if verbose:
            print(f"  Period '{period}': {count} users migrated to parquet store.")

    def migrate_all_periods(
        self,
        legacy_dir: Path | str,
        period_names: list[str],
        np_: int,
        t: int,
        batch_size: int = 500,
        consolidate: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Migrate all periods from ``legacy_dir`` to the parquet store.

        Parameters
        ----------
        legacy_dir : Path or str
            ``dataxuser/`` directory with the old per-user files.
        period_names : list[str]
            Period names to migrate (typically ``PERIOD_NAMES``).
        np_, t : int
            Parameters encoded in legacy file names.
        batch_size : int
            Users per shard.
        consolidate : bool
            If True, run ``consolidate_all()`` after each period to merge
            shards into a single file.
        verbose : bool
            Print progress messages.
        """
        for period in period_names:
            if verbose:
                print(f"\nMigrating period: {period!r} …")
            self.migrate_from_legacy(
                legacy_dir, period, np_, t,
                batch_size=batch_size, verbose=verbose,
            )
            if consolidate:
                if verbose:
                    print(f"  Consolidating shards for '{period}' …")
                self.consolidate_all(period)
        if verbose:
            print("\nMigration complete.")

    # ------------------------------------------------------------------
    # MA-specific migration (metric-folder layout)
    # ------------------------------------------------------------------

    def migrate_from_legacy_MA(
        self,
        ma_legacy_base: Path | str,
        period: str,
        np_: int,
        t: int,
        batch_size: int = 2000,
        verbose: bool = True,
    ) -> None:
        """
        Migrate one period of legacy MA data into the parquet store.

        MA data is organised into per-metric directories rather than a
        single ``dataxuser/`` folder.  Each directory contains CSV/JSON
        *shard* files (one shard = one original parquet partition) covering
        all users in that shard.

        Metric-folder layout::

            ma_legacy_base/
                radius_gyration_measures_new_threshold/{np_}/{t}/
                    rg_{period}_{np_}_threshold_{t}_hour_{shard}.csv
                distance_measures_new_threshold/{np_}/{t}/
                    dist_{period}_{np_}_threshold_{t}_hour_{shard}.csv
                k_radius_gyration_measures_new_threshold/{np_}/{t}/
                    3k_rg_{period}_{np_}_threshold_{t}_hour_{shard}.csv
                    6k_rg_{period}_{np_}_threshold_{t}_hour_{shard}.csv
                    10k_rg_{period}_{np_}_threshold_{t}_hour_{shard}.csv
                entropic_measures_new_threshold/{np_}/{t}/
                    real_entropy_{period}_{np_}_threshold_{t}_hour.csv
                    uncorr_entropy_{period}_{np_}_threshold_{t}_hour.json
                    rdm_entropy_{period}_{np_}_threshold_{t}_hour.json
                gonzalez_new_threshold/{np_}/{t}/
                    gonzalez_{period}_{np_}_*_{t}_{shard}.json
                st_new_threshold/{np_}/{t}/
                    dict_s_{period}_{np_}_{shard}_hour_{t}*.csv
                home_new_threshold/{np_}/{t}/
                    home_{period}_{np_}_threshold_{t}_hour_{shard}.csv

        S(t) files do **not** contain a uid column; row order is matched to
        the corresponding rg shard file using the shared Spark TID string
        embedded in both filenames (``part-NNNNN-tid-…-c000``).

        Parameters
        ----------
        ma_legacy_base : Path or str
            ``milestones_analysis/MA/`` directory.
        period : str
            Period name, e.g. ``"15 jan - 15 march"``.
        np_, t : int
            Sub-directory selectors (must match legacy computation params).
        batch_size : int
            Users accumulated before each shard write.
        verbose : bool
        """
        import pandas as _pd

        ma_legacy_base = Path(ma_legacy_base)

        done_scalars  = self.get_computed_users(period, "all_scalars")
        done_gonzalez = self.get_computed_users_long(period, "gonzalez")
        done_st       = self.get_computed_users(period, "S")

        # ── helpers ──────────────────────────────────────────────────────

        def _metric_dir(key: str) -> Path:
            folder = MA_LEGACY_METRIC_DIRS[key]["folder"]
            return ma_legacy_base / folder / str(np_) / str(t)

        def _period_in_file(fname: str) -> bool:
            """Return True when *period* appears in the filename."""
            return period in fname

        def _read_uid_value_csv(path: Path) -> dict[str, float]:
            """Read a ;-separated CSV with uid and values cols → {uid: float}."""
            try:
                df = _pd.read_csv(path, sep=";", usecols=["uid", "values"])
                return dict(zip(df["uid"].astype(str), df["values"].astype(float)))
            except Exception:
                return {}

        def _read_uid_value_json(path: Path) -> dict[str, float]:
            """Read a {values:[...], uid:[...]} JSON → {uid: float}."""
            try:
                with open(path) as fh:
                    d = _json.load(fh)
                return {str(u): float(v) for u, v in zip(d["uid"], d["values"])}
            except Exception:
                return {}

        def _extract_tid(fname: str) -> str | None:
            """Extract the Spark TID token (part-NNNNN-tid-…-c000) from filename."""
            m = _re.search(r"(part-\d+-tid-[^_\s.]+)", fname)
            return m.group(1) if m else None

        # ── 1. SCALARS ────────────────────────────────────────────────────
        # uid → {metric: value}  (merged from multiple source folders)
        uid_scalars: dict[str, dict[str, Any]] = {}

        def _accumulate(src_dict: dict[str, float], metric_name: str) -> None:
            for uid, val in src_dict.items():
                uid_scalars.setdefault(uid, {})[metric_name] = val

        # radius of gyration
        rg_dir = _metric_dir("rg")
        rg_uid_map: dict[str, list[str]] = {}   # tid → ordered uid list (for S(t))
        if rg_dir.exists():
            for f in sorted(rg_dir.glob("*.csv")):
                if not _period_in_file(f.name):
                    continue
                mapping = _read_uid_value_csv(f)
                _accumulate(mapping, "radius_gyration")
                tid = _extract_tid(f.name)
                if tid:
                    rg_uid_map[tid] = list(mapping.keys())

        # distance
        dist_dir = _metric_dir("distance")
        if dist_dir.exists():
            for f in sorted(dist_dir.glob("*.csv")):
                if _period_in_file(f.name):
                    _accumulate(_read_uid_value_csv(f), "distance")

        # k-radius of gyration (k = 3, 6, 10)
        krg_dir = _metric_dir("k_rg")
        if krg_dir.exists():
            k_prefix_map = MA_LEGACY_METRIC_DIRS["k_rg"]["k_prefixes"]
            for k, prefix in k_prefix_map.items():
                col = f"rg_{k}"
                for f in sorted(krg_dir.glob(f"{prefix}_*.csv")):
                    if _period_in_file(f.name):
                        _accumulate(_read_uid_value_csv(f), col)

        # entropic measures (may not exist for all t values)
        entropic_dir = _metric_dir("entropic")
        if entropic_dir.exists():
            for f in sorted(entropic_dir.iterdir()):
                if not _period_in_file(f.name):
                    continue
                bn = f.name
                if bn.startswith("real_entropy") and bn.endswith(".csv"):
                    _accumulate(_read_uid_value_csv(f), "real_entropy")
                elif bn.startswith("uncorr_entropy") and bn.endswith(".json"):
                    _accumulate(_read_uid_value_json(f), "uncorrelated_entropy")
                elif bn.startswith("rdm_entropy") and bn.endswith(".json"):
                    _accumulate(_read_uid_value_json(f), "random_entropy")

        # home (lat/lon → stored as "lat;lon" string in the `home` field)
        home_dir = _metric_dir("home")
        if home_dir.exists():
            for f in sorted(home_dir.glob("*.csv")):
                if not _period_in_file(f.name):
                    continue
                try:
                    hdf = _pd.read_csv(f, sep=";", usecols=["uid", "home_lat", "home_lon"])
                    for _, row in hdf.iterrows():
                        uid_scalars.setdefault(str(row["uid"]), {})["home"] = (
                            f"{row['home_lat']:.6f};{row['home_lon']:.6f}"
                        )
                except Exception:
                    continue

        # Write scalars in batches (skip already-done users)
        scalar_batch: dict[str, dict] = {}
        scalar_count = 0
        for uid, metrics in uid_scalars.items():
            if uid in done_scalars:
                continue
            scalar_batch[uid] = metrics
            scalar_count += 1
            if len(scalar_batch) >= batch_size:
                self.write_scalars_batch(period, scalar_batch)
                scalar_batch.clear()
                if verbose:
                    print(f"    [{period}] scalars: flushed {scalar_count} …")
        if scalar_batch:
            self.write_scalars_batch(period, scalar_batch)
        if verbose:
            print(f"  [{period}] scalars: {scalar_count} users written.")

        # ── 2. GONZALEZ ────────────────────────────────────────────────────
        # MA gonzalez JSON format:
        #   {x_sigmax: {values:[...], uid:[...]}, y_sigmay: {...}, sigmax: {...}, sigmay: {...}}
        # x_sigmax = x / sigmax (normalised), y_sigmay = y / sigmay (normalised)
        gonzalez_dir = _metric_dir("gonzalez")
        gonzalez_count = 0
        if gonzalez_dir.exists():
            gonzalez_batch: dict[str, _pd.DataFrame] = {}
            for f in sorted(gonzalez_dir.glob("*.json")):
                if not _period_in_file(f.name):
                    continue
                try:
                    with open(f) as fh:
                        raw = _json.load(fh)
                    # Build per-user DataFrames from the shard-level JSON
                    # Each key has {values: [...], uid: [...]}.
                    # Only x_sigmax, y_sigmay, sigmax, sigmay are reliable;
                    # 'x' and 'y' are often empty in the legacy files.
                    x_norm_map  = dict(zip(raw["x_sigmax"]["uid"], raw["x_sigmax"]["values"]))
                    y_norm_map  = dict(zip(raw["y_sigmay"]["uid"], raw["y_sigmay"]["values"]))
                    sigmax_map  = dict(zip(raw.get("sigmax", {}).get("uid", []),
                                          raw.get("sigmax", {}).get("values", [])))
                    sigmay_map  = dict(zip(raw.get("sigmay", {}).get("uid", []),
                                          raw.get("sigmay", {}).get("values", [])))

                    all_uids = set(x_norm_map) | set(y_norm_map)
                    for uid in all_uids:
                        uid = str(uid)
                        if uid in done_gonzalez:
                            continue
                        df = _pd.DataFrame({
                            "x_norm":  [x_norm_map.get(uid, float("nan"))],
                            "y_norm":  [y_norm_map.get(uid, float("nan"))],
                            "sigmax":  [sigmax_map.get(uid, float("nan"))],
                            "sigmay":  [sigmay_map.get(uid, float("nan"))],
                        })
                        gonzalez_batch[uid] = df
                        gonzalez_count += 1
                        if len(gonzalez_batch) >= batch_size:
                            self.write_gonzalez_batch(period, gonzalez_batch)
                            gonzalez_batch.clear()
                            if verbose:
                                print(f"    [{period}] gonzalez: flushed {gonzalez_count} …")
                except Exception:
                    continue
            if gonzalez_batch:
                self.write_gonzalez_batch(period, gonzalez_batch)
        if verbose:
            print(f"  [{period}] gonzalez: {gonzalez_count} users written.")

        # ── 3. S(t) ────────────────────────────────────────────────────────
        # MA S(t) files: rows=users (no uid), cols=time steps (0..1419).
        # Uid is recovered by matching the Spark TID from the filename to the
        # corresponding rg shard file (same TID → same row order).
        st_dir = _metric_dir("S")
        st_count = 0
        if st_dir.exists() and rg_uid_map:
            st_batch: dict[str, list] = {}
            for f in sorted(st_dir.glob("*.csv")):
                if not _period_in_file(f.name):
                    continue
                tid = _extract_tid(f.name)
                if tid is None or tid not in rg_uid_map:
                    continue
                uid_list = rg_uid_map[tid]
                try:
                    st_df = _pd.read_csv(f, sep=";", index_col=0)
                    if len(st_df) != len(uid_list):
                        # Row count mismatch — skip this shard
                        continue
                    for row_idx, uid in enumerate(uid_list):
                        uid = str(uid)
                        if uid in done_st:
                            continue
                        st_batch[uid] = st_df.iloc[row_idx].tolist()
                        st_count += 1
                        if len(st_batch) >= batch_size:
                            self.write_st_batch(period, st_batch)
                            st_batch.clear()
                            if verbose:
                                print(f"    [{period}] S(t): flushed {st_count} …")
                except Exception:
                    continue
            if st_batch:
                self.write_st_batch(period, st_batch)
        if verbose:
            print(f"  [{period}] S(t): {st_count} users written.")

    def migrate_all_periods_MA(
        self,
        ma_legacy_base: Path | str,
        period_names: list[str],
        np_: int,
        t: int,
        batch_size: int = 2000,
        consolidate: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Migrate all periods of MA legacy data into the parquet store.

        Parameters
        ----------
        ma_legacy_base : Path or str
            ``milestones_analysis/MA/`` directory.
        period_names : list[str]
            Periods to migrate (typically ``PERIOD_NAMES``).
        np_, t : int
            Sub-directory selectors.
        batch_size : int
            Users per shard write.
        consolidate : bool
            If True merge shards after each period.
        verbose : bool
        """
        for period in period_names:
            if verbose:
                print(f"\nMigrating MA period: {period!r} …")
            self.migrate_from_legacy_MA(
                ma_legacy_base, period, np_, t,
                batch_size=batch_size, verbose=verbose,
            )
            if consolidate:
                if verbose:
                    print(f"  Consolidating shards for '{period}' …")
                self.consolidate_all(period)
        if verbose:
            print("\nMA migration complete.")

    # ------------------------------------------------------------------
    # S3 synchronisation
    # ------------------------------------------------------------------

    def pull_period_from_s3(
        self,
        period: str,
        kind: str,
        s3_bucket: str,
        s3_prefix: str,
        endpoint_url: str,
        verbose: bool = True,
    ) -> bool:
        """
        Ensure (period, kind) data is available locally.

        Strategy (in order):
        1. ``consolidated.parquet`` exists on S3 → download it directly.
        2. Per-shard files exist under ``shards/`` on S3 → download,
           merge in memory, write ``consolidated.parquet`` locally.
        3. Nothing on S3 → return False.

        Unlike ``consolidate_s3_shards`` this method **never re-uploads**
        anything; it is read-only with respect to S3.

        Returns
        -------
        bool
            True if the local store now has data for (period, kind).
        """
        from .s3_io import s3_exists, s3_download, s3_list, s3_read_parquet

        shard_dir_name = self._shard_dir(period, kind).name
        s3_consolidated_key = f"{s3_prefix}/{shard_dir_name}/{CONSOLIDATED_FNAME}"
        local_cp = self._consolidated_path(period, kind)

        # ── Try consolidated.parquet first ──────────────────────────────────
        if s3_exists(s3_bucket, s3_consolidated_key, endpoint_url):
            if verbose:
                print(
                    f"  [S3 pull] {kind} {period!r}: downloading"
                    f" consolidated.parquet …",
                    end=" ",
                )
            ok = s3_download(s3_bucket, s3_consolidated_key, local_cp, endpoint_url)
            if verbose:
                print("OK" if ok else "FAILED")
            return ok

        # ── Fall back: merge per-shard files ────────────────────────────────
        s3_shards_prefix = f"{s3_prefix}/{shard_dir_name}/shards"
        s3_keys = s3_list(s3_bucket, s3_shards_prefix, endpoint_url, suffix=".parquet")
        if not s3_keys:
            if verbose:
                print(
                    f"  [S3 pull] {kind} {period!r}: not found on S3."
                )
            return False

        if verbose:
            print(
                f"  [S3 pull] {kind} {period!r}: merging"
                f" {len(s3_keys)} shard(s) …",
                end=" ",
            )

        frames: list[pl.DataFrame] = []
        for s3_key in s3_keys:
            df = s3_read_parquet(s3_bucket, s3_key, endpoint_url)
            if df is not None:
                frames.append(df)

        if not frames:
            if verbose:
                print("FAILED (no readable shards)")
            return False

        index_col = _INDEX_COL.get(kind)
        if kind in FIXED_LENGTH_KINDS:
            merged = _hconcat_fixed(frames, index_col)
        else:
            merged = pl.concat(frames, how="diagonal_relaxed")

        merged.write_parquet(local_cp, compression="snappy")
        if verbose:
            print("OK")
        return True

    def pull_all_from_s3(
        self,
        period_names: list[str],
        s3_bucket: str,
        s3_prefix: str,
        endpoint_url: str,
        kinds: list[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, dict[str, bool]]:
        """
        Pull all missing (period, kind) data from S3 to the local store.

        Only downloads kinds that are **not already present locally**,
        so it is safe to call repeatedly.

        Parameters
        ----------
        period_names : list[str]
            Periods to check (typically ``dataset.period_names``).
        kinds : list[str] or None
            Subset of metric kinds to check.  Defaults to all kinds.

        Returns
        -------
        dict[str, dict[str, bool]]
            ``{period: {kind: pulled}}`` mapping where ``pulled`` is True
            if the data is now available locally (either pre-existing or
            just downloaded).
        """
        if kinds is None:
            kinds = list(FIXED_LENGTH_KINDS) + list(LONG_FORMAT_KINDS)

        results: dict[str, dict[str, bool]] = {}
        for period in period_names:
            results[period] = {}
            for kind in kinds:
                # Check local first
                if kind in FIXED_LENGTH_KINDS:
                    already = bool(self.get_computed_users(period, kind))
                else:
                    already = bool(self.get_computed_users_long(period, kind))

                if already:
                    if verbose:
                        print(f"  [{period}] {kind}: already local — skip")
                    results[period][kind] = True
                    continue

                # Pull from S3
                ok = self.pull_period_from_s3(
                    period, kind, s3_bucket, s3_prefix, endpoint_url,
                    verbose=verbose,
                )
                results[period][kind] = ok

        return results

    def upload_shard_to_s3_unique(
        self,
        period: str,
        shard_label: str,
        s3_bucket: str,
        s3_prefix: str,
        endpoint_url: str,
        delete_after: bool = True,
        verbose: bool = True,
    ) -> dict[str, bool]:
        """
        Consolidate local shard files for (period) and upload each metric
        kind to a **unique per-shard key** on S3, preserving outputs from
        all previous shards.

        Key pattern::

            {s3_prefix}/{kind}_period_{period}_np_{np_}_t_{t}/shards/{shard_label}.parquet

        Use this during incremental S3-progressive computation so that each
        shard's output is written immediately and local files can be deleted.
        Call ``consolidate_s3_shards()`` once at the end to merge everything
        into the final ``consolidated.parquet``.

        Parameters
        ----------
        shard_label : str
            Unique, stable identifier for this shard (e.g. an MD5 hash of
            the raw S3 shard key).
        delete_after : bool
            Delete the local consolidated file after a successful upload.

        Returns
        -------
        dict[str, bool]
            ``{kind: success}`` mapping.
        """
        import traceback as _tb
        results: dict[str, bool] = {}
        for kind in list(FIXED_LENGTH_KINDS) + list(LONG_FORMAT_KINDS):
            try:
                local_files = self._shard_files(period, kind)
                cp_existing = self._consolidated_path(period, kind)
                if not local_files and not cp_existing.exists():
                    continue

                # Merge all local shard_*.parquet → consolidated.parquet
                cp = self.consolidate(period, kind)
                if not cp.exists():
                    results[kind] = False
                    continue

                shard_dir_name = cp.parent.name
                s3_key = f"{s3_prefix}/{shard_dir_name}/shards/{shard_label}.parquet"

                if verbose:
                    print(
                        f"  [S3 upload] {kind} {period!r}"
                        f" → s3://{s3_bucket}/{s3_key}"
                    )

                ok = self.upload_file_to_s3(
                    cp, s3_bucket, s3_key, endpoint_url,
                    delete_after=delete_after,
                )
                results[kind] = ok
                if verbose:
                    print(f"    {'OK' if ok else 'FAILED (local copy kept)'}")
            except Exception as _kind_exc:
                print(
                    f"  [S3 upload] WARNING: {kind} {period!r} failed —"
                    f" skipping this kind, others continue."
                    f"\n  {_kind_exc}\n{_tb.format_exc()}"
                )
                results[kind] = False

        return results

    def consolidate_s3_shards(
        self,
        period: str,
        s3_bucket: str,
        s3_prefix: str,
        endpoint_url: str,
        delete_shards_after: bool = True,
        verbose: bool = True,
    ) -> dict[str, bool]:
        """
        Download all per-shard parquet files for (period) from S3, merge
        them into a single ``consolidated.parquet``, and upload the result
        to S3 under the canonical key::

            {s3_prefix}/{kind}_period_{period}_np_{np_}_t_{t}/consolidated.parquet

        Call this **once** after all shards have been processed and their
        individual outputs have been uploaded via ``upload_shard_to_s3_unique``.

        Parameters
        ----------
        delete_shards_after : bool
            Delete the individual per-shard S3 files after a successful merge.

        Returns
        -------
        dict[str, bool]
            ``{kind: success}`` mapping.
        """
        from .s3_io import s3_list, s3_read_parquet, s3_delete
        import tempfile as _tf

        results: dict[str, bool] = {}

        for kind in list(FIXED_LENGTH_KINDS) + list(LONG_FORMAT_KINDS):
            shard_dir_name = self._shard_dir(period, kind).name
            s3_shards_prefix = f"{s3_prefix}/{shard_dir_name}/shards"
            s3_consolidated_key = f"{s3_prefix}/{shard_dir_name}/{CONSOLIDATED_FNAME}"

            # List per-shard files on S3
            s3_keys = s3_list(
                s3_bucket, s3_shards_prefix, endpoint_url, suffix=".parquet"
            )
            if not s3_keys:
                if verbose:
                    print(
                        f"  [S3 consolidate] No per-shard files on S3 for"
                        f" {kind} {period!r} — skipping."
                    )
                continue

            if verbose:
                print(
                    f"  [S3 consolidate] {kind} {period!r}:"
                    f" {len(s3_keys)} shards → merging …"
                )

            # Download each shard in-memory and merge
            frames: list[pl.DataFrame] = []
            downloaded_keys: list[str] = []

            for s3_key in s3_keys:
                df = s3_read_parquet(s3_bucket, s3_key, endpoint_url)
                if df is None:
                    if verbose:
                        print(f"    WARNING: could not read {s3_key}")
                    continue
                frames.append(df)
                downloaded_keys.append(s3_key)

            if not frames:
                results[kind] = False
                continue

            # Merge
            index_col = _INDEX_COL.get(kind)
            if kind in FIXED_LENGTH_KINDS:
                merged = _hconcat_fixed(frames, index_col)
            else:
                merged = pl.concat(frames, how="diagonal_relaxed")
                if kind == "frequency" and "geohash7" in merged.columns:
                    merged = merged.filter(pl.col("geohash7").is_not_null())
                if kind == "gonzalez" and "x_norm" in merged.columns:
                    merged = merged.filter(pl.col("x_norm").is_not_null())

            # Write merged result locally (in the normal shard dir)
            cp = self._consolidated_path(period, kind)
            merged.write_parquet(cp, compression="snappy")

            # Upload consolidated.parquet to S3
            ok = self.upload_file_to_s3(
                cp, s3_bucket, s3_consolidated_key, endpoint_url,
                delete_after=True,
            )
            results[kind] = ok

            if verbose:
                print(f"    {'OK' if ok else 'FAILED (local copy kept)'}")

            # Delete per-shard S3 files
            if delete_shards_after and ok:
                for s3_key in downloaded_keys:
                    s3_delete(s3_bucket, s3_key, endpoint_url)

        return results

    def upload_file_to_s3(
        self,
        local_path: Path,
        s3_bucket: str,
        s3_key: str,
        endpoint_url: str,
        delete_after: bool = True,
    ) -> bool:
        """
        Upload a single local file to an S3-compatible store.

        Uses the ``boto3`` library — no AWS CLI required.

        Parameters
        ----------
        local_path : Path
            The file to upload.  Must exist.
        s3_bucket : str
            Destination bucket (e.g. ``"chub-datalake"``).
        s3_key : str
            Full key inside the bucket (e.g.
            ``"final_pipeline/CA/all_scalars_.../consolidated.parquet"``).
        endpoint_url : str
            S3-compatible endpoint (e.g. ``"https://s3.atlas.fbk.eu"``).
        delete_after : bool
            If True *and* the upload succeeded, delete the local file.

        Returns
        -------
        bool
            True on success, False on failure (local file is kept on failure).
        """
        from .s3_io import s3_upload
        return s3_upload(local_path, s3_bucket, s3_key, endpoint_url, delete_after=delete_after)

    def upload_period_to_s3(
        self,
        period: str,
        s3_bucket: str,
        s3_prefix: str,
        endpoint_url: str,
        delete_after: bool = True,
        consolidate_first: bool = True,
        verbose: bool = True,
    ) -> dict[str, bool]:
        """
        Upload all consolidated parquet files for ``period`` to S3.

        The remote key layout mirrors the local shard-directory names::

            {s3_prefix}/{kind}_period_{period}_np_{np_}_t_{t}/consolidated.parquet

        Parameters
        ----------
        period : str
            Period name (e.g. ``"15 jan - 15 march"``).
        s3_bucket : str
            Destination bucket.
        s3_prefix : str
            Prefix inside the bucket
            (e.g. ``"final_pipeline/CA"``).
        endpoint_url : str
            S3-compatible endpoint URL.
        delete_after : bool
            Delete the local file after a successful upload.
        consolidate_first : bool
            Run ``consolidate_all(period)`` before uploading.
        verbose : bool
            Print progress messages.

        Returns
        -------
        dict[str, bool]
            ``{kind: success}`` mapping.
        """
        if consolidate_first:
            self.consolidate_all(period)

        results: dict[str, bool] = {}
        for kind in list(FIXED_LENGTH_KINDS) + list(LONG_FORMAT_KINDS):
            cp = self._consolidated_path(period, kind)
            if not cp.exists():
                continue

            # Mirror the local directory name on S3
            shard_dir_name = cp.parent.name  # e.g. all_scalars_period_…
            s3_key = f"{s3_prefix}/{shard_dir_name}/{CONSOLIDATED_FNAME}"

            if verbose:
                print(f"  Uploading [{kind}] {period!r} → s3://{s3_bucket}/{s3_key} …")

            ok = self.upload_file_to_s3(
                cp, s3_bucket, s3_key, endpoint_url,
                delete_after=delete_after,
            )
            results[kind] = ok
            if verbose:
                status = "OK" if ok else "FAILED (local copy kept)"
                print(f"    {status}")

        return results

    def upload_all_to_s3(
        self,
        period_names: list[str],
        s3_bucket: str,
        s3_prefix: str,
        endpoint_url: str,
        delete_after: bool = True,
        consolidate_first: bool = True,
        verbose: bool = True,
    ) -> dict[str, dict[str, bool]]:
        """
        Upload all periods to S3.

        Parameters
        ----------
        period_names : list[str]
            Periods to upload (typically ``PERIOD_NAMES``).
        s3_bucket : str
            Destination bucket.
        s3_prefix : str
            Prefix inside the bucket.
        endpoint_url : str
            S3-compatible endpoint URL.
        delete_after : bool
            Delete local files after successful upload.
        consolidate_first : bool
            Consolidate shards before uploading.
        verbose : bool

        Returns
        -------
        dict[str, dict[str, bool]]
            ``{period: {kind: success}}`` nested mapping.
        """
        all_results: dict[str, dict[str, bool]] = {}
        for period in period_names:
            if verbose:
                print(f"\nUploading period: {period!r} …")
            all_results[period] = self.upload_period_to_s3(
                period=period,
                s3_bucket=s3_bucket,
                s3_prefix=s3_prefix,
                endpoint_url=endpoint_url,
                delete_after=delete_after,
                consolidate_first=consolidate_first,
                verbose=verbose,
            )
        if verbose:
            print("\nS3 upload complete.")
        return all_results

    def list_s3_computed_periods(
        self,
        s3_bucket: str,
        s3_prefix: str,
        endpoint_url: str,
    ) -> list[str]:
        """
        List period names that already have consolidated data on S3.

        Uses the ``boto3`` library to list the S3 prefix.

        Returns
        -------
        list[str]
            Period names found on S3 (subset of ``PERIOD_NAMES``).
        """
        from .s3_io import s3_list
        import re as _re

        keys = s3_list(s3_bucket, s3_prefix, endpoint_url)
        period_pat = _re.compile(r"all_scalars_period_(.+?)_np_\d+_t_\d+")
        found: set[str] = set()
        for key in keys:
            m = period_pat.search(key)
            if m:
                period_safe = m.group(1)
                found.add(period_safe.replace("_", " "))
        return sorted(found)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract_uid(fname: str, prefix: str, suffix: str) -> str | None:
    """
    Extract user ID from ``{prefix}{uid}_period_{…}{suffix}``.
    Returns ``None`` if the pattern does not match.
    """
    if not (fname.startswith(prefix) and fname.endswith(suffix)):
        return None
    inner = fname[len(prefix): len(fname) - len(suffix)]
    if "_period_" not in inner:
        return None
    return inner.split("_period_")[0]


def _hconcat_fixed(
    frames: list[pl.DataFrame],
    index_col: str,
) -> pl.DataFrame:
    """
    Horizontally concatenate fixed-length frames that all share the same
    ``index_col`` (same rows in the same order).

    If two frames contain the same user column, the later one overwrites
    (idempotent for crash-recovery scenarios).
    """
    if not frames:
        return pl.DataFrame()

    result = frames[0]
    seen: set[str] = set(c for c in result.columns if c != index_col)

    for f in frames[1:]:
        new_cols  = [c for c in f.columns if c != index_col]
        unique    = [c for c in new_cols if c not in seen]
        duplicate = [c for c in new_cols if c in seen]

        if unique:
            result = result.with_columns([f[c] for c in unique])
            seen.update(unique)
        if duplicate:
            result = result.with_columns([f[c] for c in duplicate])

    return result


def _transpose_metrics_to_users(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transpose a ``[n_metrics × (1 + n_users)]`` DataFrame into
    ``[n_users × (n_metrics + 1)]`` with a ``"user_id"`` column.

    The ``"metric"`` column values become the new column names; the old
    column names (user IDs) become values in a ``"user_id"`` column.

    User columns are stored as Utf8; numeric metrics are cast to Float64,
    string metrics are kept as Utf8.
    """
    import pandas as _pd

    metric_names = df["metric"].to_list()
    user_cols    = [c for c in df.columns if c != "metric"]

    if not user_cols:
        return pl.DataFrame()

    rows: list[dict] = []
    for uid in user_cols:
        vals = df[uid].to_list()
        row: dict[str, Any] = {"user_id": uid}
        for m, v in zip(metric_names, vals):
            row[m] = v
        rows.append(row)

    result = pl.from_pandas(_pd.DataFrame(rows))

    # Cast numeric columns from Utf8 to Float64
    casts = []
    for col in result.columns:
        if col != "user_id" and col not in _STR_SCALAR_METRICS:
            casts.append(pl.col(col).replace("", None).cast(pl.Float64, strict=False))
    if casts:
        result = result.with_columns(casts)

    return result
