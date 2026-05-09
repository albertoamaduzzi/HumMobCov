"""
transition_pipeline.py
======================
Geohash-grid transition matrix and presence matrix computation for the
HumMobCov *final_pipeline*.

Design goals
------------
* **No local storage by default** — results are computed in memory and
  uploaded directly to S3.  A minimal temp file is written and deleted
  after a confirmed upload.
* **Modular** — each conceptual step is a standalone function so it can be
  tested, profiled and reused independently.
* **Resume-safe** — a lightweight JSON cache index on S3 records what
  (region, period, precision) combinations are already computed.  Re-running
  the pipeline is a no-op for completed combinations.
* **Polars throughout** — all DataFrame operations use Polars for zero-copy,
  parallel group-by and joins.

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

Usage
-----
>>> from src.datasets import DataSet_California
>>> from src.transition_matrices import TransitionPipeline
>>>
>>> dataset   = DataSet_California()
>>> pipeline  = TransitionPipeline(
...     dataset          = dataset,
...     geohash_precision= 5,
...     delta_time_h     = 1,
...     endpoint_url     = "https://s3.atlas.fbk.eu",
...     bucket           = "chub-datalake",
...     s3_prefix        = "shared/cuebiq/MOBS/final_pipeline/CA/transition_matrices",
...     temp_dir         = None,    # uses /tmp by default
...     keep_local       = False,   # delete after upload (default)
... )
>>> pipeline.run_period("15 jan - 15 march")
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

import polars as pl


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

    Parameters
    ----------
    df : pl.DataFrame
        Must contain a column *datetime_col* that is parseable as
        ``pl.Datetime`` (or already is one).
    datetime_col : str
        Name of the datetime column.  Default ``"datetime"``.
    delta_h : float
        Bin width in hours.  Default 1.0.

    Returns
    -------
    pl.DataFrame
        Original columns plus:

        * ``time_int``    — ``Int64`` bin index (0-based)
        * ``bin_datetime``— ``Utf8`` ISO-8601 string of the bin start
    """
    delta_s = int(delta_h * 3600)

    # Ensure datetime dtype
    df = df.with_columns(
        pl.col(datetime_col).cast(pl.Datetime("us")).alias(datetime_col)
    )

    # Unix timestamps in seconds
    df = df.with_columns(
        (pl.col(datetime_col).dt.epoch(time_unit="s")).alias("_ts_s")
    )

    t_min = df["_ts_s"].min()

    df = df.with_columns(
        ((pl.col("_ts_s") - t_min) // delta_s).cast(pl.Int64).alias("time_int")
    )

    # Bin-start datetime: t_min + time_int * delta_s
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


def compute_presence_matrix(
    df: pl.DataFrame,
    *,
    uid_col: str = "userId",
    geohash_col: str = "geohash",
    time_int_col: str = "time_int",
    bin_datetime_col: str = "bin_datetime",
    geohash_precision: int | None = None,
) -> pl.DataFrame:
    """
    Compute the presence matrix from a time-binned trajectory DataFrame.

    For each (geohash, time_bin) pair the function counts:

    * **count_birth**: users whose *first stop in any cell* occurs in
      this time bin (they "appear" for the first time).
    * **count_death**: users whose *last stop in any cell* occurs in
      this time bin (they "disappear").
    * **count_transit**: users present in this bin who *also* have a
      stop in the *following* bin (anywhere in the dataset).
    * **count** = count_birth + count_death + count_transit
    * **probability** = count_transit / Σ(count_transit) over all cells
      for this bin.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns *uid_col*, *geohash_col*, *time_int_col*,
        *bin_datetime_col*.  Must already be time-binned (see
        :func:`build_time_bins`).
    uid_col : str
        User identifier column.
    geohash_col : str
        Geohash column at the target precision (or coarser — see
        *geohash_precision*).
    time_int_col : str
        Integer time-bin index column.
    bin_datetime_col : str
        ISO-8601 bin-start datetime column.
    geohash_precision : int, optional
        If provided, truncate *geohash_col* to this many characters before
        aggregation (coarse-graining).

    Returns
    -------
    pl.DataFrame
        Presence matrix with columns:
        geohash, time_int, datetime, count_birth, count_death,
        count_transit, count, probability.
    """
    if geohash_precision is not None:
        df = _coarsen_geohash_col(df, geohash_col, geohash_precision)

    # ── per-user global first / last bin ─────────────────────────────────────
    user_range = (
        df
        .group_by(uid_col)
        .agg([
            pl.col(time_int_col).min().alias("first_bin"),
            pl.col(time_int_col).max().alias("last_bin"),
        ])
    )

    # ── base: unique (uid, geohash, time_int) presence ───────────────────────
    presence = (
        df
        .select([uid_col, geohash_col, time_int_col, bin_datetime_col])
        .unique()
    )

    # ── join user range onto presence ─────────────────────────────────────────
    presence = presence.join(user_range, on=uid_col, how="left")

    # ── per-uid, per-bin: is the user present in the *next* bin (anywhere)? ──
    # Build a (uid, time_int) table of ALL presence for the "next" bin lookup.
    uid_bins = presence.select([uid_col, time_int_col]).unique()
    uid_bins_next = uid_bins.with_columns(
        (pl.col(time_int_col) - 1).alias("prev_bin")
    ).rename({uid_col: uid_col + "_r", time_int_col: time_int_col + "_r"})

    # ── count_birth: users whose first_bin == current bin ────────────────────
    births = (
        presence
        .filter(pl.col(time_int_col) == pl.col("first_bin"))
        .group_by([geohash_col, time_int_col])
        .agg(pl.col(uid_col).n_unique().alias("count_birth"))
    )

    # ── count_death: users whose last_bin == current bin ─────────────────────
    deaths = (
        presence
        .filter(pl.col(time_int_col) == pl.col("last_bin"))
        .group_by([geohash_col, time_int_col])
        .agg(pl.col(uid_col).n_unique().alias("count_death"))
    )

    # ── count_transit: users present in bin T who also appear in bin T+1 ─────
    # Join presence with uid_bins shifted by -1 to find "next bin" users.
    transit_flags = (
        presence
        .join(
            uid_bins.rename({time_int_col: "next_bin"}),
            left_on=[uid_col, time_int_col],
            right_on=[uid_col + "", "next_bin"],  # workaround: join on uid + (t+1 in the other)
            how="left",
        )
    )
    # A simpler approach: for each uid in a bin, check if uid is in uid_bins at bin+1
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
        how="inner",  # only rows where uid appears in next bin too
    )

    transits = (
        transit_flags
        .group_by([geohash_col, time_int_col])
        .agg(pl.col(uid_col).n_unique().alias("count_transit"))
    )

    # ── base aggregation (for datetime and total users per bin-geohash) ───────
    base = (
        presence
        .group_by([geohash_col, time_int_col])
        .agg(pl.col(bin_datetime_col).first().alias("datetime"))
    )

    # ── assemble ──────────────────────────────────────────────────────────────
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
        .with_columns(
            (pl.col("count_birth") + pl.col("count_death") + pl.col("count_transit"))
            .alias("count")
        )
    )

    # ── probability = count_transit / Σ(count_transit) per bin ───────────────
    bin_totals = (
        result
        .group_by(time_int_col)
        .agg(pl.col("count_transit").sum().alias("total_transit"))
    )
    result = (
        result
        .join(bin_totals, on=time_int_col, how="left")
        .with_columns(
            (pl.col("count_transit") / pl.col("total_transit"))
            .fill_nan(0.0)
            .alias("probability")
        )
        .drop("total_transit")
    )

    return result.rename({geohash_col: "geohash", time_int_col: "time_int"})[
        ["geohash", "time_int", "datetime",
         "count_birth", "count_death", "count_transit", "count", "probability"]
    ].sort(["time_int", "geohash"])


def compute_transition_matrix(
    df: pl.DataFrame,
    *,
    uid_col: str = "userId",
    geohash_col: str = "geohash",
    time_int_col: str = "time_int",
    bin_datetime_col: str = "bin_datetime",
    geohash_precision: int | None = None,
) -> pl.DataFrame:
    """
    Compute the transition matrix from time-binned trajectories.

    A *transition* is the move of a user from cell A at time bin T to cell B
    at time bin T+1.  Multiple stops of the same user in the same cell within
    one bin are collapsed (only the last geohash of the bin is used as the
    departure cell, and the first of the next bin as the arrival cell).

    Parameters
    ----------
    df : pl.DataFrame
        Time-binned trajectory data.  Required columns: *uid_col*,
        *geohash_col*, *time_int_col*, *bin_datetime_col*.
    uid_col : str
        User identifier column.
    geohash_col : str
        Geohash column.
    time_int_col : str
        Integer time-bin index column.
    bin_datetime_col : str
        ISO-8601 bin-start datetime column.
    geohash_precision : int, optional
        Coarsen geohash to this precision before counting.

    Returns
    -------
    pl.DataFrame
        Transition matrix with columns:
        geohash_start, geohash_end, time_int, datetime,
        transitions, transition_probability.
    """
    if geohash_precision is not None:
        df = _coarsen_geohash_col(df, geohash_col, geohash_precision)

    # ── representative geohash per (uid, bin): take the *last* stop ──────────
    # Sort by datetime to get temporal order.
    representative = (
        df
        .sort(["datetime" if "datetime" in df.columns else bin_datetime_col])
        .group_by([uid_col, time_int_col])
        .agg([
            pl.col(geohash_col).last().alias("geohash_start"),
            pl.col(bin_datetime_col).first().alias("datetime"),
        ])
    )

    # ── self-join on uid, next bin → get geohash_end ──────────────────────────
    next_bin = representative.with_columns(
        (pl.col(time_int_col) - 1).alias("prev_bin")
    ).rename({
        "geohash_start": "geohash_end",
        uid_col:         uid_col + "_r",
        time_int_col:    time_int_col + "_next",
        "datetime":      "datetime_end",
        "prev_bin":      "current_bin",
    })

    transitions_raw = representative.join(
        next_bin.select([uid_col + "_r", "current_bin", "geohash_end"]),
        left_on=[uid_col, time_int_col],
        right_on=[uid_col + "_r", "current_bin"],
        how="inner",
    )

    # ── aggregate: count distinct users per (start, end, bin) ────────────────
    agg = (
        transitions_raw
        .group_by(["geohash_start", "geohash_end", time_int_col])
        .agg([
            pl.col(uid_col).n_unique().alias("transitions"),
            pl.col("datetime").first().alias("datetime"),
        ])
    )

    # ── transition probability: transitions / Σ(transitions) per (start, bin) -
    start_totals = (
        agg
        .group_by(["geohash_start", time_int_col])
        .agg(pl.col("transitions").sum().alias("total_from_start"))
    )
    result = (
        agg
        .join(start_totals, on=["geohash_start", time_int_col], how="left")
        .with_columns(
            (pl.col("transitions") / pl.col("total_from_start"))
            .alias("transition_probability")
        )
        .drop("total_from_start")
    )

    return result[
        ["geohash_start", "geohash_end", "time_int", "datetime",
         "transitions", "transition_probability"]
    ].sort(["time_int", "geohash_start", "geohash_end"])


# ---------------------------------------------------------------------------
# Compression helper
# ---------------------------------------------------------------------------

def save_compressed(
    df: pl.DataFrame,
    path: Path,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
) -> Path:
    """
    Save a Polars DataFrame to a compressed parquet file.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to save.
    path : Path
        Output file path.  Parent directory must exist.
    compression : str
        Parquet compression codec.  Default ``"zstd"`` (good ratio + speed).

    Returns
    -------
    Path
        The written file path.
    """
    path = Path(path)
    df.write_parquet(path, compression=compression)
    return path


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class TransitionPipeline:
    """
    End-to-end pipeline: raw trajectories → transition / presence matrices
    → S3 upload.

    The pipeline is **resume-safe** via a JSON cache index that is read from
    (and written to) S3 before each run.  Completed (region, period,
    precision) triples are skipped.

    Data flow
    ---------
    1. Load raw parquet shard for the period from the dataset.
    2. Coarsen geohash column to *geohash_precision* characters.
    3. Bin timestamps into *delta_time_h*-hour bins (:func:`build_time_bins`).
    4. Compute presence matrix (:func:`compute_presence_matrix`).
    5. Compute transition matrix (:func:`compute_transition_matrix`).
    6. Write both to a temp file (compressed parquet).
    7. Upload to S3 → delete temp file on success.
    8. Update the cache index on S3.

    Parameters
    ----------
    dataset : _BaseDataset
        Initialised ``DataSet_California`` or ``DataSet_Massachusets``
        instance.  Provides the raw file list, period definitions and
        preprocessing parameters.
    geohash_precision : int
        Geohash precision for the grid.  Default 5.
        Use 5 for CA/MA with 32 GB RAM.
    delta_time_h : float
        Time-bin width in hours.  Default 1.0.
    endpoint_url : str
        S3-compatible endpoint URL.
        Default ``"https://s3.atlas.fbk.eu"``.
    bucket : str
        S3 bucket.  Default ``"chub-datalake"``.
    s3_prefix : str
        Key prefix for this pipeline's output inside the bucket.
        Default ``"shared/cuebiq/MOBS/final_pipeline/{region}/transition_matrices"``.
    temp_dir : Path or str, optional
        Local directory for transient temp files.  Defaults to
        ``/tmp/humobcov_transitions``.
    keep_local : bool
        If True, keep the local temp file after upload (useful when local
        space is available).  Default False.
    geohash_col : str
        Name of the geohash column in the raw trajectory data.
        Default ``"geohash"``.
    uid_col : str
        Name of the user-id column.  Default ``"userId"``.
    datetime_col : str
        Name of the datetime column.  Default ``"datetime"``.
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

        region: str = getattr(dataset, "id_", "CA")
        if s3_prefix is None:
            s3_prefix = (
                f"shared/cuebiq/MOBS/final_pipeline/{region}/transition_matrices"
            )
        self.s3_prefix = s3_prefix

        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "humobcov_transitions"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache index (on S3)
    # ------------------------------------------------------------------

    def _cache_s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.s3_prefix}/{self._CACHE_KEY}"

    def _load_cache(self) -> dict:
        """Download the JSON cache index from S3, or return empty dict."""
        local_tmp = self.temp_dir / "_cache_index_tmp.json"
        result = subprocess.run(
            [
                "aws", "s3", "cp",
                self._cache_s3_uri(),
                str(local_tmp),
                "--endpoint-url", self.endpoint_url,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not local_tmp.exists():
            return {}
        try:
            with open(local_tmp) as f:
                data = json.load(f)
        except Exception:
            data = {}
        local_tmp.unlink(missing_ok=True)
        return data

    def _save_cache(self, cache: dict) -> None:
        """Upload the JSON cache index to S3."""
        local_tmp = self.temp_dir / "_cache_index_tmp.json"
        with open(local_tmp, "w") as f:
            json.dump(cache, f, indent=2)
        subprocess.run(
            [
                "aws", "s3", "cp",
                str(local_tmp),
                self._cache_s3_uri(),
                "--endpoint-url", self.endpoint_url,
            ],
            capture_output=True, text=True,
        )
        local_tmp.unlink(missing_ok=True)

    def _cache_key(self, period: str, kind: str) -> str:
        safe = period.replace(" ", "_").replace("/", "-")
        return f"{safe}_prec{self.geohash_precision}_dh{self.delta_time_h}_{kind}"

    def get_cache_status(self) -> dict[str, list[str]]:
        """
        Return a dict of already-computed (period, kind) pairs from the
        S3 cache index.

        Returns
        -------
        dict[str, list[str]]
            ``{"15_jan_-_15_march_prec5_dh1.0_presence": "s3://...", ...}``
        """
        return self._load_cache()

    def is_computed(self, period: str, kind: str) -> bool:
        """Check whether *(period, kind)* is already in the S3 cache."""
        cache = self._load_cache()
        return self._cache_key(period, kind) in cache

    # ------------------------------------------------------------------
    # S3 upload helper
    # ------------------------------------------------------------------

    def _upload(self, local_path: Path, s3_key: str) -> bool:
        """Upload a local file to S3.  Return True on success."""
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        result = subprocess.run(
            [
                "aws", "s3", "cp",
                str(local_path),
                s3_uri,
                "--endpoint-url", self.endpoint_url,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(
                f"  [S3 upload] FAILED {local_path.name} → {s3_uri}\n"
                f"  stderr: {result.stderr.strip()}"
            )
            return False
        if not self.keep_local:
            local_path.unlink(missing_ok=True)
        return True

    # ------------------------------------------------------------------
    # Per-period processing
    # ------------------------------------------------------------------

    def _load_period_trajectories(
        self,
        period: str,
    ) -> pl.DataFrame | None:
        """
        Load raw trajectory stops for *period* from the dataset's raw files.

        Reads all raw parquet shards, filters to the period's date range,
        and returns a concatenated Polars DataFrame.

        Returns None if no data is found.
        """
        from ..constants import PERIOD_NAMES_TO_DIVISION

        period_bounds = PERIOD_NAMES_TO_DIVISION.get(period)
        if period_bounds is None:
            raise ValueError(f"Unknown period: {period!r}")
        start_dt, end_dt = period_bounds

        frames: list[pl.DataFrame] = []
        for fpath in getattr(self.dataset, "dir_files", []):
            fp = Path(fpath)
            if not fp.exists():
                continue
            try:
                lf = pl.scan_parquet(fp)
                schema = lf.schema

                # Identify datetime column
                dt_col = self.datetime_col
                if dt_col not in schema:
                    # Try common alternatives
                    for c in ["timestamp", "time", "stop_time", "start_time"]:
                        if c in schema:
                            dt_col = c
                            break

                df_chunk = (
                    lf
                    .with_columns(
                        pl.col(dt_col).cast(pl.Datetime("us")).alias(dt_col)
                    )
                    .filter(
                        (pl.col(dt_col) >= pl.lit(start_dt).cast(pl.Datetime("us")))
                        & (pl.col(dt_col) < pl.lit(end_dt).cast(pl.Datetime("us")))
                    )
                    .select(
                        [c for c in [self.uid_col, self.geohash_col, dt_col]
                         if c in schema]
                    )
                    .collect()
                )
                if df_chunk.height > 0:
                    if dt_col != self.datetime_col:
                        df_chunk = df_chunk.rename({dt_col: self.datetime_col})
                    frames.append(df_chunk)
            except Exception as exc:
                print(f"  WARNING: could not read {fp.name}: {exc}")
                continue

        if not frames:
            return None
        return pl.concat(frames, how="vertical_relaxed")

    def _period_s3_key(self, period: str, kind: str) -> str:
        """S3 key for the output file of *(period, kind)*."""
        safe = period.replace(" ", "_").replace("/", "-")
        fname = (
            f"{kind}_prec{self.geohash_precision}_dh{self.delta_time_h}"
            f"_{safe}.parquet"
        )
        return f"{self.s3_prefix}/{fname}"

    def run_period(
        self,
        period: str,
        force: bool = False,
        verbose: bool = True,
    ) -> dict[str, bool]:
        """
        Run the full pipeline for a single *period*.

        Steps:
        1. Check cache — skip if already computed (unless *force=True*).
        2. Load raw trajectory data for the period.
        3. Coarsen geohash to *geohash_precision*.
        4. Build time bins.
        5. Compute presence matrix.
        6. Compute transition matrix.
        7. Write both to compressed temp parquet files.
        8. Upload to S3 and delete temp files.
        9. Update cache index.

        Parameters
        ----------
        period : str
            Period name (e.g. ``"15 jan - 15 march"``).
        force : bool
            If True, re-compute even if the cache says it is done.
        verbose : bool
            Print progress messages.

        Returns
        -------
        dict[str, bool]
            ``{"presence": True/False, "transition": True/False}``
        """
        results = {"presence": False, "transition": False}
        cache = self._load_cache()

        # ── 1. Resume check ──────────────────────────────────────────────
        skip_presence   = (not force) and (self._cache_key(period, "presence") in cache)
        skip_transition = (not force) and (self._cache_key(period, "transition") in cache)

        if skip_presence and skip_transition:
            if verbose:
                print(f"  [skip] {period!r} — both matrices already on S3.")
            return {"presence": True, "transition": True}

        # ── 2. Load raw data ─────────────────────────────────────────────
        if verbose:
            print(f"\n[{period}] Loading raw trajectories …")
        df = self._load_period_trajectories(period)
        if df is None or df.height == 0:
            print(f"  WARNING: no data found for period {period!r}. Skipping.")
            return results

        if verbose:
            print(f"  Loaded {df.height:,} stops for {df[self.uid_col].n_unique():,} users.")

        # ── 3. Coarsen geohash ───────────────────────────────────────────
        if verbose:
            print(f"  Coarsening geohash to precision {self.geohash_precision} …")
        df = _coarsen_geohash_col(df, self.geohash_col, self.geohash_precision)

        # ── 4. Time binning ──────────────────────────────────────────────
        if verbose:
            print(f"  Binning into {self.delta_time_h}h bins …")
        df = build_time_bins(df, datetime_col=self.datetime_col, delta_h=self.delta_time_h)

        # ── 5 & 6. Compute matrices ──────────────────────────────────────
        for kind, skip, compute_fn in [
            ("presence",   skip_presence,   compute_presence_matrix),
            ("transition", skip_transition, compute_transition_matrix),
        ]:
            if skip:
                if verbose:
                    print(f"  [skip] {kind} matrix already on S3.")
                results[kind] = True
                continue

            if verbose:
                print(f"  Computing {kind} matrix …")

            matrix = compute_fn(
                df,
                uid_col=self.uid_col,
                geohash_col=self.geohash_col,
                time_int_col="time_int",
                bin_datetime_col="bin_datetime",
                geohash_precision=None,   # already coarsened above
            )

            # ── 7. Write temp file ────────────────────────────────────────
            safe_period = period.replace(" ", "_").replace("/", "-")
            tmp_fname = (
                f"{kind}_prec{self.geohash_precision}_dh{self.delta_time_h}"
                f"_{safe_period}.parquet"
            )
            tmp_path = self.temp_dir / tmp_fname
            save_compressed(matrix, tmp_path)
            if verbose:
                size_mb = tmp_path.stat().st_size / 1e6
                print(f"  Written temp file: {tmp_path.name} ({size_mb:.1f} MB)")

            # ── 8. Upload to S3 ───────────────────────────────────────────
            s3_key = self._period_s3_key(period, kind)
            if verbose:
                print(f"  Uploading to s3://{self.bucket}/{s3_key} …")

            ok = self._upload(tmp_path, s3_key)
            results[kind] = ok

            if ok:
                cache[self._cache_key(period, kind)] = (
                    f"s3://{self.bucket}/{s3_key}"
                )
                if verbose:
                    print(f"  Upload OK.")
            else:
                # Keep temp file on failure so data is not lost
                print(
                    f"  WARNING: upload failed — temp file kept at {tmp_path}."
                )

        # ── 9. Update cache ───────────────────────────────────────────────
        self._save_cache(cache)
        return results

    def run_all_periods(
        self,
        force: bool = False,
        verbose: bool = True,
    ) -> dict[str, dict[str, bool]]:
        """
        Run the pipeline for all periods in ``dataset.period_names``.

        Parameters
        ----------
        force : bool
            Re-compute even if cached.
        verbose : bool

        Returns
        -------
        dict[str, dict[str, bool]]
            ``{period: {"presence": bool, "transition": bool}}``
        """
        all_results = {}
        for period in self.dataset.period_names:
            all_results[period] = self.run_period(
                period, force=force, verbose=verbose
            )
        return all_results

    def read_from_s3(
        self,
        period: str,
        kind: str,
    ) -> pl.DataFrame:
        """
        Download a computed matrix from S3 into a Polars DataFrame.

        Parameters
        ----------
        period : str
            Period name.
        kind : str
            ``"presence"`` or ``"transition"``.

        Returns
        -------
        pl.DataFrame
            The requested matrix, or an empty DataFrame if not found.
        """
        s3_key = self._period_s3_key(period, kind)
        safe_period = period.replace(" ", "_").replace("/", "-")
        tmp_path = self.temp_dir / f"_read_{kind}_{safe_period}.parquet"

        result = subprocess.run(
            [
                "aws", "s3", "cp",
                f"s3://{self.bucket}/{s3_key}",
                str(tmp_path),
                "--endpoint-url", self.endpoint_url,
            ],
            capture_output=True, text=True,
        )

        if result.returncode != 0 or not tmp_path.exists():
            print(f"  [read_from_s3] Not found: s3://{self.bucket}/{s3_key}")
            return pl.DataFrame()

        df = pl.read_parquet(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return df

    # ------------------------------------------------------------------
    # Info / diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print the cache status and S3 prefix."""
        cache = self._load_cache()
        print(f"TransitionPipeline")
        print(f"  Region          : {getattr(self.dataset, 'id_', 'unknown')}")
        print(f"  Geohash prec.   : {self.geohash_precision}")
        print(f"  Delta time (h)  : {self.delta_time_h}")
        print(f"  S3 prefix       : s3://{self.bucket}/{self.s3_prefix}")
        print(f"  Computed entries: {len(cache)}")
        for k, v in cache.items():
            print(f"    ✓  {k}")
        if not cache:
            print("    (none — no computations recorded yet)")
