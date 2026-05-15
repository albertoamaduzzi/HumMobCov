"""
transition_pipeline.py
======================
Geohash-grid transition matrix and presence matrix computation for the
HumMobCov *final_pipeline*.

Design goals
------------
* **No AWS CLI** — all S3 operations use ``boto3`` via ``src.s3_io``.
* **S3-progressive** — raw trajectory shards are downloaded from S3 one at a
  time, counts are accumulated in memory / on disk, then the final matrices
  are uploaded and local temp files deleted.  No need for a large local disk.
* **Resume-safe** — a per-period JSON shard checkpoint (stored locally in
  ``temp_dir``) records which raw shards have been processed.  Re-running the
  pipeline continues from the first unprocessed shard.  Completed periods are
  skipped via a JSON cache index on S3.
* **Incremental accumulation** — because shards are partitioned by user
  (each user appears in exactly one shard), raw count tables are additive
  across shards.  Probabilities are computed once at the end after merging all
  shard contributions.
* **Local fallback** — when no ``raw_bucket`` / ``raw_s3_prefix`` are
  provided, the pipeline falls back to loading from local files defined in
  ``dataset.dir_files``.
* **Polars throughout** — all DataFrame operations use Polars.

Output schema
-------------
**presence_matrix** — ``pl.DataFrame``

| column               | dtype   | description                                              |
|----------------------|---------|----------------------------------------------------------|
| geohash              | Utf8    | geohash cell at the chosen precision                     |
| time_int             | Int64   | 0-based index of the time bin                            |
| datetime             | Utf8    | ISO-8601 string of the bin start (UTC)                   |
| count_birth          | Int64   | users whose first stop in any cell is in this bin        |
| count_death          | Int64   | users whose last stop in any cell is in this bin         |
| count_transit        | Int64   | users present in this bin AND in the next bin            |
| count                | Int64   | count_birth + count_death + count_transit                |
| probability          | Float64 | count_transit / Σ(count_transit) over all cells, this bin|

**transition_matrix** — ``pl.DataFrame``

| column                  | dtype   | description                                      |
|-------------------------|---------|--------------------------------------------------|
| geohash_start           | Utf8    | cell at time bin T                               |
| geohash_end             | Utf8    | cell at time bin T+1 (same user)                 |
| time_int                | Int64   | index of time bin T                              |
| datetime                | Utf8    | ISO-8601 of bin T start (UTC)                    |
| transitions             | Int64   | number of distinct users making this move        |
| transition_probability  | Float64 | transitions / Σ(transitions) leaving geohash_start|
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Literal

import polars as pl

from ..s3_io import (
    s3_upload,
    s3_download,
    s3_list,
    s3_read_parquet,
    s3_read_json,
    s3_write_json,
)


# ---------------------------------------------------------------------------
# Module-level helpers (standalone, testable functions)
# ---------------------------------------------------------------------------

def build_time_bins(
    df: pl.DataFrame,
    *,
    datetime_col: str = "datetime",
    delta_h: float = 1.0,
) -> pl.DataFrame:
    """
    Assign each row a discrete time bin index and bin-start datetime.

    The bin index is computed as::

        time_int = floor((unix_timestamp - global_min) / delta_seconds)

    so bin 0 starts at the earliest timestamp in the data.
    """
    delta_s = int(delta_h * 3600)

    df = df.with_columns(
        pl.col(datetime_col).cast(pl.Datetime("us")).alias(datetime_col)
    )
    df = df.with_columns(
        (pl.col(datetime_col).dt.epoch(time_unit="s")).alias("_ts_s")
    )

    t_min = df["_ts_s"].min()

    df = df.with_columns(
        ((pl.col("_ts_s") - t_min) // delta_s).cast(pl.Int64).alias("time_int")
    )
    df = df.with_columns(
        (
            pl.from_epoch(t_min + pl.col("time_int") * delta_s, time_unit="s")
            .cast(pl.Datetime("us"))
            .dt.to_string("%Y-%m-%dT%H:%M:%S")
        ).alias("bin_datetime")
    )

    return df.drop("_ts_s")


def _coarsen_geohash_col(df: pl.DataFrame, col: str, precision: int) -> pl.DataFrame:
    """Return *df* with *col* truncated to *precision* characters."""
    return df.with_columns(pl.col(col).str.slice(0, precision).alias(col))


# ---------------------------------------------------------------------------
# Raw-count variants (no probability columns — additive across user-shards)
# ---------------------------------------------------------------------------

def compute_presence_counts(
    df: pl.DataFrame,
    *,
    uid_col: str = "userId",
    geohash_col: str = "geohash",
    time_int_col: str = "time_int",
    bin_datetime_col: str = "bin_datetime",
    geohash_precision: int | None = None,
) -> pl.DataFrame:
    """
    Compute raw presence counts from a time-binned trajectory DataFrame.

    Unlike :func:`compute_presence_matrix`, this function does **not** add a
    ``probability`` column.  The returned table is additive across user-shards
    so that counts from multiple shards can be summed before finalisation.

    Returns columns: ``geohash, time_int, datetime,
    count_birth, count_death, count_transit``
    """
    if geohash_precision is not None:
        df = _coarsen_geohash_col(df, geohash_col, geohash_precision)

    user_range = (
        df
        .group_by(uid_col)
        .agg([
            pl.col(time_int_col).min().alias("first_bin"),
            pl.col(time_int_col).max().alias("last_bin"),
        ])
    )

    presence = (
        df
        .select([uid_col, geohash_col, time_int_col, bin_datetime_col])
        .unique()
    )

    presence = presence.join(user_range, on=uid_col, how="left")

    uid_bins = presence.select([uid_col, time_int_col]).unique()

    births = (
        presence
        .filter(pl.col(time_int_col) == pl.col("first_bin"))
        .group_by([geohash_col, time_int_col])
        .agg(pl.col(uid_col).n_unique().alias("count_birth"))
    )

    deaths = (
        presence
        .filter(pl.col(time_int_col) == pl.col("last_bin"))
        .group_by([geohash_col, time_int_col])
        .agg(pl.col(uid_col).n_unique().alias("count_death"))
    )

    # count_transit: users in bin T who also appear in bin T+1
    next_bin_users = (
        uid_bins
        .with_columns((pl.col(time_int_col) - 1).alias("current_bin"))
        .rename({uid_col: uid_col + "_next"})
        .select([uid_col + "_next", "current_bin"])
    )

    transit_flags = presence.join(
        next_bin_users.rename({"current_bin": time_int_col}),
        left_on=[uid_col, time_int_col],
        right_on=[uid_col + "_next", time_int_col],
        how="inner",
    )

    transits = (
        transit_flags
        .group_by([geohash_col, time_int_col])
        .agg(pl.col(uid_col).n_unique().alias("count_transit"))
    )

    base = (
        presence
        .group_by([geohash_col, time_int_col])
        .agg(pl.col(bin_datetime_col).first().alias("datetime"))
    )

    result = (
        base
        .join(births,   on=[geohash_col, time_int_col], how="left")
        .join(deaths,   on=[geohash_col, time_int_col], how="left")
        .join(transits, on=[geohash_col, time_int_col], how="left")
        .with_columns([
            pl.col("count_birth").fill_null(0),
            pl.col("count_death").fill_null(0),
            pl.col("count_transit").fill_null(0),
        ])
    )

    return result.rename({geohash_col: "geohash", time_int_col: "time_int"})[
        ["geohash", "time_int", "datetime",
         "count_birth", "count_death", "count_transit"]
    ].sort(["time_int", "geohash"])


def compute_transition_counts(
    df: pl.DataFrame,
    *,
    uid_col: str = "userId",
    geohash_col: str = "geohash",
    time_int_col: str = "time_int",
    bin_datetime_col: str = "bin_datetime",
    geohash_precision: int | None = None,
) -> pl.DataFrame:
    """
    Compute raw transition counts from time-binned trajectories.

    Unlike :func:`compute_transition_matrix`, this function does **not** add a
    ``transition_probability`` column.  The returned table is additive across
    user-shards.

    Returns columns: ``geohash_start, geohash_end, time_int, datetime,
    transitions``
    """
    if geohash_precision is not None:
        df = _coarsen_geohash_col(df, geohash_col, geohash_precision)

    dt_sort_col = "datetime" if "datetime" in df.columns else bin_datetime_col
    representative = (
        df
        .sort(dt_sort_col)
        .group_by([uid_col, time_int_col])
        .agg([
            pl.col(geohash_col).last().alias("geohash_start"),
            pl.col(bin_datetime_col).first().alias("datetime"),
        ])
    )

    next_bin = (
        representative
        .with_columns((pl.col(time_int_col) - 1).alias("current_bin"))
        .rename({
            "geohash_start": "geohash_end",
            uid_col:         uid_col + "_r",
            time_int_col:    time_int_col + "_next",
            "datetime":      "datetime_end",
        })
    )

    transitions_raw = representative.join(
        next_bin.select([uid_col + "_r", "current_bin", "geohash_end"]),
        left_on=[uid_col, time_int_col],
        right_on=[uid_col + "_r", "current_bin"],
        how="inner",
    )

    result = (
        transitions_raw
        .group_by(["geohash_start", "geohash_end", time_int_col])
        .agg([
            pl.col(uid_col).n_unique().alias("transitions"),
            pl.col("datetime").first().alias("datetime"),
        ])
    )

    return result.rename({time_int_col: "time_int"})[
        ["geohash_start", "geohash_end", "time_int", "datetime", "transitions"]
    ].sort(["time_int", "geohash_start", "geohash_end"])


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge_presence_counts(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Merge per-shard presence count tables by summing counts."""
    merged = pl.concat(frames, how="diagonal_relaxed")
    return (
        merged
        .group_by(["geohash", "time_int"])
        .agg([
            pl.col("datetime").first(),
            pl.col("count_birth").sum(),
            pl.col("count_death").sum(),
            pl.col("count_transit").sum(),
        ])
        .sort(["time_int", "geohash"])
    )


def merge_transition_counts(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Merge per-shard transition count tables by summing transitions."""
    merged = pl.concat(frames, how="diagonal_relaxed")
    return (
        merged
        .group_by(["geohash_start", "geohash_end", "time_int"])
        .agg([
            pl.col("datetime").first(),
            pl.col("transitions").sum(),
        ])
        .sort(["time_int", "geohash_start", "geohash_end"])
    )


# ---------------------------------------------------------------------------
# Finalise helpers (add probability columns)
# ---------------------------------------------------------------------------

def finalise_presence(counts: pl.DataFrame) -> pl.DataFrame:
    """Add ``count`` and ``probability`` columns to merged presence counts."""
    result = counts.with_columns(
        (pl.col("count_birth") + pl.col("count_death") + pl.col("count_transit"))
        .alias("count")
    )
    bin_totals = (
        result
        .group_by("time_int")
        .agg(pl.col("count_transit").sum().alias("total_transit"))
    )
    return (
        result
        .join(bin_totals, on="time_int", how="left")
        .with_columns(
            (pl.col("count_transit") / pl.col("total_transit"))
            .fill_nan(0.0)
            .alias("probability")
        )
        .drop("total_transit")
        .select(["geohash", "time_int", "datetime",
                 "count_birth", "count_death", "count_transit",
                 "count", "probability"])
        .sort(["time_int", "geohash"])
    )


def finalise_transition(counts: pl.DataFrame) -> pl.DataFrame:
    """Add ``transition_probability`` to merged transition counts."""
    start_totals = (
        counts
        .group_by(["geohash_start", "time_int"])
        .agg(pl.col("transitions").sum().alias("total_from_start"))
    )
    return (
        counts
        .join(start_totals, on=["geohash_start", "time_int"], how="left")
        .with_columns(
            (pl.col("transitions") / pl.col("total_from_start"))
            .alias("transition_probability")
        )
        .drop("total_from_start")
        .select(["geohash_start", "geohash_end", "time_int", "datetime",
                 "transitions", "transition_probability"])
        .sort(["time_int", "geohash_start", "geohash_end"])
    )


# ---------------------------------------------------------------------------
# Full-matrix wrappers (backward-compatible one-shot versions)
# ---------------------------------------------------------------------------

def compute_presence_matrix(
    df: pl.DataFrame,
    *,
    uid_col: str = "userId",
    geohash_col: str = "geohash",
    time_int_col: str = "time_int",
    bin_datetime_col: str = "bin_datetime",
    geohash_precision: int | None = None,
) -> pl.DataFrame:
    """Compute the full presence matrix (counts + probability) in one step."""
    counts = compute_presence_counts(
        df,
        uid_col=uid_col,
        geohash_col=geohash_col,
        time_int_col=time_int_col,
        bin_datetime_col=bin_datetime_col,
        geohash_precision=geohash_precision,
    )
    return finalise_presence(counts)


def compute_transition_matrix(
    df: pl.DataFrame,
    *,
    uid_col: str = "userId",
    geohash_col: str = "geohash",
    time_int_col: str = "time_int",
    bin_datetime_col: str = "bin_datetime",
    geohash_precision: int | None = None,
) -> pl.DataFrame:
    """Compute the full transition matrix (counts + probability) in one step."""
    counts = compute_transition_counts(
        df,
        uid_col=uid_col,
        geohash_col=geohash_col,
        time_int_col=time_int_col,
        bin_datetime_col=bin_datetime_col,
        geohash_precision=geohash_precision,
    )
    return finalise_transition(counts)


# ---------------------------------------------------------------------------
# Compression helper
# ---------------------------------------------------------------------------

def save_compressed(
    df: pl.DataFrame,
    path: Path,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
) -> Path:
    """Save a Polars DataFrame to a compressed parquet file."""
    path = Path(path)
    df.write_parquet(path, compression=compression)
    return path


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class TransitionPipeline:
    """
    End-to-end pipeline: raw trajectories → transition / presence matrices
    → S3 upload.  No AWS CLI required — all S3 operations use ``boto3``.

    S3-progressive mode (recommended)
    ----------------------------------
    Pass ``raw_bucket`` and ``raw_s3_prefix`` to load raw trajectory shards
    directly from S3 one at a time.  For each shard the pipeline:

    1. Downloads the shard to ``temp_dir``.
    2. Filters to the period date range.
    3. Coarsens geohash and bins time.
    4. Computes raw presence and transition counts.
    5. Accumulates counts into intermediate parquet files in ``temp_dir``.
    6. Deletes the raw shard temp file.

    After all shards are processed the counts are finalised (probabilities
    added), uploaded to S3, and intermediate files deleted.

    **Assumption**: shards are partitioned by user — each user appears in
    exactly one shard.  This makes counts additive across shards.

    Resume safety
    -------------
    * A per-period JSON checkpoint in ``temp_dir`` records which raw shards
      have been processed.  Restarting the pipeline continues from the next
      unprocessed shard and picks up intermediate parquet files.
    * Completed periods are recorded in a JSON cache index on S3
      (``s3_prefix/cache_index.json``) and are skipped on re-run.

    Local fallback
    --------------
    If ``raw_bucket`` / ``raw_s3_prefix`` are not provided the pipeline loads
    from ``dataset.dir_files`` (all files for the full period at once).

    Parameters
    ----------
    dataset : _BaseDataset
        Initialised dataset object providing period definitions.
    geohash_precision : int
        Geohash precision for the grid.  Default 5.
    delta_time_h : float
        Time-bin width in hours.  Default 1.0.
    endpoint_url : str
        S3 endpoint for the *output* store.
    bucket : str
        Output S3 bucket.
    s3_prefix : str, optional
        Key prefix for output files inside the bucket.
    temp_dir : Path or str, optional
        Local directory for temp files.
        Defaults to ``/tmp/humobcov_transitions``.
    keep_local : bool
        Keep temp files after upload (default False).
    geohash_col, uid_col, datetime_col : str
        Column names in the raw trajectory data.
    raw_bucket : str, optional
        S3 bucket containing raw trajectory shards.
        If None, falls back to loading from local files.
    raw_s3_prefix : str, optional
        S3 key prefix for raw trajectory shards.
    raw_endpoint_url : str, optional
        S3 endpoint for the raw data bucket.  Defaults to ``endpoint_url``.
    """

    _CACHE_KEY = "cache_index.json"

    def __init__(
        self,
        dataset: Any,
        geohash_precision: int = 5,
        delta_time_h: float = 1.0,
        endpoint_url: str = "https://s3.atlas.fbk.eu",
        bucket: str = "chub-datalake",
        s3_prefix: str | None = None,
        temp_dir: Path | str | None = None,
        keep_local: bool = False,
        geohash_col: str = "geohash",
        uid_col: str = "userId",
        datetime_col: str = "datetime",
        raw_bucket: str | None = None,
        raw_s3_prefix: str | None = None,
        raw_endpoint_url: str | None = None,
    ) -> None:
        self.dataset           = dataset
        self.geohash_precision = geohash_precision
        self.delta_time_h      = delta_time_h
        self.endpoint_url      = endpoint_url
        self.bucket            = bucket
        self.geohash_col       = geohash_col
        self.uid_col           = uid_col
        self.datetime_col      = datetime_col
        self.keep_local        = keep_local
        self.raw_bucket        = raw_bucket
        self.raw_s3_prefix     = raw_s3_prefix
        self.raw_endpoint_url  = raw_endpoint_url or endpoint_url

        region: str = getattr(dataset, "id_", "CA")
        if s3_prefix is None:
            s3_prefix = f"final_pipeline/{region}/transition_matrices"
        self.s3_prefix = s3_prefix

        self.temp_dir = (
            Path(temp_dir) if temp_dir
            else Path(tempfile.gettempdir()) / "humobcov_transitions"
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache index (on S3 — marks completed periods)
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict:
        data = s3_read_json(
            self.bucket,
            f"{self.s3_prefix}/{self._CACHE_KEY}",
            self.endpoint_url,
        )
        return data if isinstance(data, dict) else {}

    def _save_cache(self, cache: dict) -> None:
        s3_write_json(
            cache,
            self.bucket,
            f"{self.s3_prefix}/{self._CACHE_KEY}",
            self.endpoint_url,
        )

    def _cache_key(self, period: str, kind: str) -> str:
        safe = period.replace(" ", "_").replace("/", "-")
        return f"{safe}_prec{self.geohash_precision}_dh{self.delta_time_h}_{kind}"

    def is_computed(self, period: str, kind: str) -> bool:
        """Return True if *(period, kind)* is in the S3 cache."""
        return self._cache_key(period, kind) in self._load_cache()

    # ------------------------------------------------------------------
    # Per-period shard checkpoint (local)
    # ------------------------------------------------------------------

    def _checkpoint_path(self, period: str) -> Path:
        safe = period.replace(" ", "_").replace("/", "-")
        return self.temp_dir / f"{safe}_checkpoint.json"

    def _load_shard_checkpoint(self, period: str) -> set[str]:
        p = self._checkpoint_path(period)
        if not p.exists():
            return set()
        try:
            with open(p) as f:
                return set(json.load(f).get("completed_shards", []))
        except Exception:
            return set()

    def _save_shard_checkpoint(self, period: str, completed: set[str]) -> None:
        with open(self._checkpoint_path(period), "w") as f:
            json.dump({"completed_shards": sorted(completed)}, f, indent=2)

    # ------------------------------------------------------------------
    # Intermediate accumulated-counts (local temp files)
    # ------------------------------------------------------------------

    def _partial_path(self, period: str, kind: str) -> Path:
        safe = period.replace(" ", "_").replace("/", "-")
        return self.temp_dir / f"partial_{kind}_counts_{safe}.parquet"

    def _load_partial(self, period: str, kind: str) -> pl.DataFrame | None:
        p = self._partial_path(period, kind)
        if not p.exists():
            return None
        try:
            return pl.read_parquet(p)
        except Exception:
            return None

    def _save_partial(self, period: str, kind: str, df: pl.DataFrame) -> None:
        df.write_parquet(self._partial_path(period, kind), compression="zstd")

    def _clear_partial(self, period: str) -> None:
        for kind in ("presence", "transition"):
            self._partial_path(period, kind).unlink(missing_ok=True)
        self._checkpoint_path(period).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Output S3 key
    # ------------------------------------------------------------------

    def _period_s3_key(self, period: str, kind: str) -> str:
        safe = period.replace(" ", "_").replace("/", "-")
        fname = (
            f"{kind}_prec{self.geohash_precision}_dh{self.delta_time_h}"
            f"_{safe}.parquet"
        )
        return f"{self.s3_prefix}/{fname}"

    # ------------------------------------------------------------------
    # Upload helper
    # ------------------------------------------------------------------

    def _upload(self, local_path: Path, s3_key: str) -> bool:
        ok = s3_upload(local_path, self.bucket, s3_key, self.endpoint_url)
        if ok and not self.keep_local:
            local_path.unlink(missing_ok=True)
        return ok

    # ------------------------------------------------------------------
    # Shard data processing (filter + coarsen + bin + count)
    # ------------------------------------------------------------------

    def _filter_to_period(self, df: pl.DataFrame, period: str) -> pl.DataFrame:
        from ..constants import PERIOD_NAMES_TO_DIVISION

        bounds = PERIOD_NAMES_TO_DIVISION.get(period)
        if bounds is None:
            raise ValueError(f"Unknown period: {period!r}")
        start_dt, end_dt = bounds

        dt_col = self.datetime_col
        if dt_col not in df.columns:
            for c in ("timestamp", "time", "stop_time", "start_time", "begin", "end"):
                if c in df.columns:
                    df = df.rename({c: dt_col})
                    break

        gh_col = self.geohash_col
        if gh_col not in df.columns:
            for c in ("geohash7", "geohash6", "geohash5", "geohash4"):
                if c in df.columns:
                    df = df.rename({c: gh_col})
                    break

        df = df.with_columns(
            pl.col(dt_col).cast(pl.Datetime("us")).alias(dt_col)
        )
        return df.filter(
            (pl.col(dt_col) >= pl.lit(start_dt).cast(pl.Datetime("us")))
            & (pl.col(dt_col) < pl.lit(end_dt).cast(pl.Datetime("us")))
        ).select(
            [c for c in [self.uid_col, self.geohash_col, dt_col]
             if c in df.columns]
        )

    def _accumulate_shard(
        self,
        df: pl.DataFrame,
        period: str,
        acc_presence: pl.DataFrame | None,
        acc_transition: pl.DataFrame | None,
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
        """Filter *df* to *period*, compute counts, and merge into accumulators."""
        filtered = self._filter_to_period(df, period)
        if filtered.height == 0:
            return acc_presence, acc_transition

        filtered = _coarsen_geohash_col(
            filtered, self.geohash_col, self.geohash_precision
        )
        filtered = build_time_bins(
            filtered,
            datetime_col=self.datetime_col,
            delta_h=self.delta_time_h,
        )

        pres = compute_presence_counts(
            filtered,
            uid_col=self.uid_col,
            geohash_col=self.geohash_col,
            time_int_col="time_int",
            bin_datetime_col="bin_datetime",
        )
        trans = compute_transition_counts(
            filtered,
            uid_col=self.uid_col,
            geohash_col=self.geohash_col,
            time_int_col="time_int",
            bin_datetime_col="bin_datetime",
        )

        acc_presence  = pres  if acc_presence  is None else merge_presence_counts(  [acc_presence,  pres])
        acc_transition = trans if acc_transition is None else merge_transition_counts([acc_transition, trans])

        return acc_presence, acc_transition

    # ------------------------------------------------------------------
    # Local fallback: load all period data from dataset.dir_files
    # ------------------------------------------------------------------

    def _load_local(self, period: str) -> pl.DataFrame | None:
        from ..constants import PERIOD_NAMES_TO_DIVISION

        bounds = PERIOD_NAMES_TO_DIVISION.get(period)
        if bounds is None:
            raise ValueError(f"Unknown period: {period!r}")
        start_dt, end_dt = bounds

        frames: list[pl.DataFrame] = []
        for fpath in getattr(self.dataset, "dir_files", []):
            fp = Path(fpath)
            if not fp.exists():
                continue
            try:
                lf = pl.scan_parquet(fp)
                schema = lf.schema
                dt_col = self.datetime_col
                if dt_col not in schema:
                    for c in ("timestamp", "time", "stop_time", "start_time", "begin", "end"):
                        if c in schema:
                            dt_col = c
                            break
                gh_col = self.geohash_col
                if gh_col not in schema:
                    for c in ("geohash7", "geohash6", "geohash5", "geohash4"):
                        if c in schema:
                            gh_col = c
                            break
                chunk = (
                    lf
                    .with_columns(
                        pl.col(dt_col).cast(pl.Datetime("us")).alias(dt_col)
                    )
                    .filter(
                        (pl.col(dt_col) >= pl.lit(start_dt).cast(pl.Datetime("us")))
                        & (pl.col(dt_col) < pl.lit(end_dt).cast(pl.Datetime("us")))
                    )
                    .select(
                        [c for c in [self.uid_col, gh_col, dt_col]
                         if c in schema]
                    )
                    .collect()
                )
                if chunk.height > 0:
                    if dt_col != self.datetime_col:
                        chunk = chunk.rename({dt_col: self.datetime_col})
                    if gh_col != self.geohash_col:
                        chunk = chunk.rename({gh_col: self.geohash_col})
                    frames.append(chunk)
            except Exception as exc:
                print(f"  WARNING: could not read {fp.name}: {exc}")

        if not frames:
            return None
        return pl.concat(frames, how="vertical_relaxed")

    # ------------------------------------------------------------------
    # Main run: per-period
    # ------------------------------------------------------------------

    def run_period(
        self,
        period: str,
        force: bool = False,
        verbose: bool = True,
    ) -> dict[str, bool]:
        """
        Run the full pipeline for a single *period*.

        When ``raw_bucket`` / ``raw_s3_prefix`` are set, shards are downloaded
        one at a time from S3 and counts accumulated progressively.
        Otherwise, all data for the period is loaded from local files at once.

        Parameters
        ----------
        period : str
            Period name (e.g. ``"15 jan - 15 march"``).
        force : bool
            Re-compute even if already recorded in the S3 cache.
        verbose : bool
            Print progress messages.

        Returns
        -------
        dict[str, bool]
            ``{"presence": bool, "transition": bool}``
        """
        results = {"presence": False, "transition": False}
        cache = self._load_cache()

        skip_presence   = (not force) and (self._cache_key(period, "presence")   in cache)
        skip_transition = (not force) and (self._cache_key(period, "transition") in cache)

        if skip_presence and skip_transition:
            if verbose:
                print(f"  [skip] {period!r} — both matrices already on S3.")
            return {"presence": True, "transition": True}

        # ── Accumulate raw counts ────────────────────────────────────────
        if self.raw_bucket and self.raw_s3_prefix:
            acc_pres, acc_trans = self._run_s3_progressive(
                period, force=force, verbose=verbose
            )
        else:
            if verbose:
                print(f"\n[{period}] Loading from local files …")
            df = self._load_local(period)
            if df is None or df.height == 0:
                print(f"  WARNING: no local data for {period!r}. Skipping.")
                return results
            if verbose:
                print(
                    f"  {df.height:,} stops, "
                    f"{df[self.uid_col].n_unique():,} users."
                )
            df = _coarsen_geohash_col(df, self.geohash_col, self.geohash_precision)
            df = build_time_bins(
                df, datetime_col=self.datetime_col, delta_h=self.delta_time_h
            )
            acc_pres  = compute_presence_counts(
                df, uid_col=self.uid_col, geohash_col=self.geohash_col,
                time_int_col="time_int", bin_datetime_col="bin_datetime",
            )
            acc_trans = compute_transition_counts(
                df, uid_col=self.uid_col, geohash_col=self.geohash_col,
                time_int_col="time_int", bin_datetime_col="bin_datetime",
            )

        if acc_pres is None or acc_trans is None:
            print(f"  WARNING: no data produced for {period!r}.")
            return results

        # ── Finalise and upload ──────────────────────────────────────────
        for kind, skip, counts, finalise_fn in [
            ("presence",   skip_presence,   acc_pres,  finalise_presence),
            ("transition", skip_transition, acc_trans, finalise_transition),
        ]:
            if skip:
                if verbose:
                    print(f"  [skip] {kind} already on S3.")
                results[kind] = True
                continue

            if verbose:
                print(f"  Finalising {kind} matrix …")

            matrix = finalise_fn(counts)

            safe = period.replace(" ", "_").replace("/", "-")
            tmp_fname = (
                f"{kind}_prec{self.geohash_precision}"
                f"_dh{self.delta_time_h}_{safe}.parquet"
            )
            tmp_path = self.temp_dir / tmp_fname
            save_compressed(matrix, tmp_path)

            if verbose:
                size_mb = tmp_path.stat().st_size / 1e6
                print(f"  {tmp_path.name} ({size_mb:.1f} MB)")

            s3_key = self._period_s3_key(period, kind)
            if verbose:
                print(f"  → s3://{self.bucket}/{s3_key} …", end=" ", flush=True)

            ok = self._upload(tmp_path, s3_key)
            results[kind] = ok

            if ok:
                cache[self._cache_key(period, kind)] = f"s3://{self.bucket}/{s3_key}"
                if verbose:
                    print("OK")
            else:
                print(f"FAILED (temp file kept at {tmp_path})")

        self._save_cache(cache)
        self._clear_partial(period)   # remove checkpoint + partial counts
        return results

    # ------------------------------------------------------------------
    # S3-progressive inner loop
    # ------------------------------------------------------------------

    def _run_s3_progressive(
        self,
        period: str,
        force: bool = False,
        verbose: bool = True,
    ) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
        """
        Iterate over raw shards from S3 and accumulate counts for *period*.

        Returns (acc_presence_counts, acc_transition_counts).
        """
        if verbose:
            print(
                f"\n[{period}] S3-progressive: "
                f"s3://{self.raw_bucket}/{self.raw_s3_prefix}"
            )

        shard_keys = s3_list(
            self.raw_bucket,
            self.raw_s3_prefix,
            self.raw_endpoint_url,
            suffix=".parquet",
        )
        if not shard_keys:
            print(
                f"  WARNING: no .parquet shards at "
                f"s3://{self.raw_bucket}/{self.raw_s3_prefix}"
            )
            return None, None

        if verbose:
            print(f"  {len(shard_keys)} raw shards found.")

        completed = set() if force else self._load_shard_checkpoint(period)
        pending   = [k for k in shard_keys if k not in completed]

        if verbose:
            print(
                f"  Checkpoint: {len(completed)} done, {len(pending)} pending."
            )

        acc_pres  = None if force else self._load_partial(period, "presence")
        acc_trans = None if force else self._load_partial(period, "transition")

        if (acc_pres is not None or acc_trans is not None) and verbose:
            print("  Resuming from partial accumulated counts.")

        for idx, shard_key in enumerate(pending):
            shard_name = Path(shard_key).name
            tmp_shard  = self.temp_dir / f"_raw_{shard_name}"

            if verbose:
                print(
                    f"  [{idx + 1}/{len(pending)}] {shard_name} …",
                    end=" ", flush=True,
                )

            ok = s3_download(
                self.raw_bucket, shard_key, tmp_shard, self.raw_endpoint_url
            )
            if not ok:
                print("download FAILED — skipping.")
                continue

            try:
                df = pl.read_parquet(tmp_shard)
                acc_pres, acc_trans = self._accumulate_shard(
                    df, period, acc_pres, acc_trans
                )
                if verbose:
                    print("done.")
            except Exception as exc:
                print(f"processing FAILED: {exc} — skipping.")
            finally:
                tmp_shard.unlink(missing_ok=True)

            # Persist progress after every shard
            completed.add(shard_key)
            self._save_shard_checkpoint(period, completed)
            if acc_pres is not None:
                self._save_partial(period, "presence", acc_pres)
            if acc_trans is not None:
                self._save_partial(period, "transition", acc_trans)

        if verbose:
            n_p = acc_pres.height  if acc_pres  is not None else 0
            n_t = acc_trans.height if acc_trans is not None else 0
            print(
                f"  Accumulated {n_p:,} presence rows "
                f"and {n_t:,} transition rows."
            )

        return acc_pres, acc_trans

    # ------------------------------------------------------------------
    # Multi-period runner
    # ------------------------------------------------------------------

    def run_all_periods(
        self,
        force: bool = False,
        verbose: bool = True,
    ) -> dict[str, dict[str, bool]]:
        """Run the pipeline for all periods in ``dataset.period_names``."""
        all_results: dict[str, dict[str, bool]] = {}
        for period in self.dataset.period_names:
            all_results[period] = self.run_period(
                period, force=force, verbose=verbose
            )
        return all_results

    # ------------------------------------------------------------------
    # Read back from S3
    # ------------------------------------------------------------------

    def read_from_s3(self, period: str, kind: str) -> pl.DataFrame:
        """Download a computed matrix from S3 into a Polars DataFrame."""
        s3_key = self._period_s3_key(period, kind)
        df = s3_read_parquet(self.bucket, s3_key, self.endpoint_url)
        if df is None:
            print(f"  [read_from_s3] Not found: s3://{self.bucket}/{s3_key}")
            return pl.DataFrame()
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print the pipeline configuration and S3 cache status."""
        cache = self._load_cache()
        print("TransitionPipeline")
        print(f"  Region          : {getattr(self.dataset, 'id_', 'unknown')}")
        print(f"  Geohash prec.   : {self.geohash_precision}")
        print(f"  Delta time (h)  : {self.delta_time_h}")
        print(f"  Output S3       : s3://{self.bucket}/{self.s3_prefix}")
        if self.raw_bucket:
            print(
                f"  Raw S3 source   : "
                f"s3://{self.raw_bucket}/{self.raw_s3_prefix}"
            )
        else:
            print("  Raw source      : local files (dataset.dir_files)")
        print(f"  Computed entries: {len(cache)}")
        for k in cache:
            print(f"    ✓  {k}")
        if not cache:
            print("    (none — no computations recorded yet)")
