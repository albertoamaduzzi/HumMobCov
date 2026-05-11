"""
pipeline.py
===========
High-level processing pipeline: per-user metric computation and the
top-level orchestrators.

Execution modes
---------------
``analyze_from_dataset``
    Process raw parquet shards already present on the local filesystem.
    Resume-safe at the user level via parquet footer metadata (O(1)).

``analyze_from_s3_progressive``
    Download one raw shard at a time from an S3-compatible store, compute
    all metrics, flush to the parquet store, then delete the local copy.
    Resume-safe at both the *shard* level (``shard_checkpoint_*.json``) and
    the *user* level (parquet footer metadata).  Designed to be re-run many
    times — each run picks up exactly where the previous one stopped.

Parquet-store mode
------------------
Pass a ``ParquetStore`` instance (``store=``) to ``compute_all`` or either
orchestrator to use the new columnar parquet backend.

* Resume check is done via parquet footer metadata (O(1), no data load).
* Per-user results are batched in memory and written as shard parquet files
  rather than individual ``*.csv.gz`` files — much lower I/O overhead.
* The old per-file path is still available when ``store=None`` (default).
"""

import json
import os
import subprocess
import time
from pathlib import Path
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from .User      import User
from .datasets  import dataset_info, _BaseDataset
from .utils     import (
    get_already_saved_user_per_period,
    ifnotexistsmkdir,
)
from .constants import DIR_OUTPUT, DIR_CONFIG, DIR_MILESTONES_SERVER, DIR_SHARD_TEMP

if TYPE_CHECKING:
    from .store import ParquetStore


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def get_config(region: str, config_dir: Path | str | None = None) -> dict:
    """
    Load the algorithm-flow configuration JSON for ``region``.

    Parameters
    ----------
    region : str
        ``"CA"`` or ``"MA"``.
    config_dir : Path or str, optional
        Directory that contains ``config_<region>.json``.
        Defaults to ``DIR_CONFIG`` (``data/config/``).

    Returns
    -------
    dict
        Mapping of feature flags, e.g. ``{"is_radius_gyration": true, ...}``.
    """
    base = Path(config_dir) if config_dir else DIR_CONFIG
    path = base / f"config_{region}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Create it in data/config/ or pass config_dir explicitly."
        )
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-batch computation
# ---------------------------------------------------------------------------

def compute_all(
    cfg: dict,
    dataset: _BaseDataset,
    list_users,
    period: str,
    df: dataset_info | None = None,
    output_dir: Path | str | None = None,
    store: "ParquetStore | None" = None,
    batch_size: int = 500,
    use_vectorized: bool = False,
) -> dict:
    """
    Compute all requested metrics for every user in ``list_users``.

    Parameters
    ----------
    cfg : dict
        Algorithm-flow config loaded by ``get_config()``.
    dataset : _BaseDataset
        Region dataset object (carries census data and parameters).
    list_users : iterable
        User IDs to process.
    period : str
        Period name being processed.
    df : dataset_info, optional
        Required when ``cfg["raw_trajectories"]`` is ``True``.
    output_dir : Path or str, optional
        Override the per-user output directory (legacy mode only).
    store : ParquetStore, optional
        When supplied, results are written to the columnar parquet store
        in batches of ``batch_size`` users instead of individual files.
        Resume check is performed via parquet footer metadata (O(1)).
    batch_size : int
        Number of users to accumulate before flushing to the store.
        Ignored when ``store`` is None.
    use_vectorized : bool
        When ``True``, bypass the per-user loop entirely and delegate to
        :func:`~src.vectorized_pipeline.compute_all_polars`.  Requires
        ``df`` to be a ``dataset_info`` object (raw trajectories mode).
        The polars-based path replaces all skmob calls with direct polars/
        numpy/numba implementations and can be 20–100× faster on wide
        DataFrames.  Default ``False`` keeps existing behaviour.

    Returns
    -------
    dict
        ``{week_datetime: [point_counts]}``
    """
    # ── fast vectorized path ────────────────────────────────────────────
    if use_vectorized and df is not None and cfg.get("raw_trajectories"):
        from .vectorized_pipeline import compute_all_polars as _vec_all
        import polars as _pl

        # Convert pandas period DataFrame to polars
        pandas_df = df.period2df.get(period)
        if pandas_df is None or len(pandas_df) == 0:
            return {}
        polars_df = _pl.from_pandas(pandas_df.reset_index(drop=True))

        # Add dur_min if missing
        if "dur_min" not in polars_df.columns:
            polars_df = polars_df.with_columns(
                (_pl.col("end") - _pl.col("begin"))
                .dt.total_minutes()
                .cast(_pl.Float64)
                .alias("dur_min")
            )
        # Rename skmob columns if needed
        rename_map = {}
        if "lat" not in polars_df.columns and "clusterLatitude" in polars_df.columns:
            rename_map["clusterLatitude"] = "lat"
        if "lon" not in polars_df.columns and "clusterLongitude" in polars_df.columns:
            rename_map["clusterLongitude"] = "lon"
        if rename_map:
            polars_df = polars_df.rename(rename_map)

        # Resume sets
        if store is not None:
            ad_scalars  = store.get_computed_users(period, "all_scalars")
            ad_gonzalez = store.get_computed_users_long(period, "gonzalez")
            ad_st       = store.get_computed_users(period, "S")
            ad_freq     = store.get_computed_users_long(period, "frequency")
            ad_wrg      = store.get_computed_users(period, "weekly_rg")
        else:
            ad_scalars = ad_gonzalez = ad_st = ad_freq = ad_wrg = set()

        return _vec_all(
            cfg, dataset, polars_df, period,
            ad_scalars, ad_gonzalez, ad_st, ad_freq, ad_wrg,
            store, batch_size,
        )

    # ── existing per-user path (unchanged) ──────────────────────────────
    overall_count = 0
    dictweek2npeople = None
    week2points: dict = defaultdict(list)

    base_out = (
        Path(output_dir)
        if output_dir
        else DIR_MILESTONES_SERVER / dataset.id_ / "dataxuser"
    )
    ifnotexistsmkdir(base_out)

    # ── resume logic ──────────────────────────────────────────────────────
    # Determine which users are already done so we can skip them.
    # Parquet-store mode: O(1) footer reads, no data loaded.
    # Legacy mode: scan file names in dataxuser/.
    if store is not None:
        already_done_scalars  = store.get_computed_users(period, "all_scalars")
        already_done_gonzalez = store.get_computed_users_long(period, "gonzalez")
        already_done_st       = store.get_computed_users(period, "S")
        already_done_freq     = store.get_computed_users_long(period, "frequency")
        already_done_wrg      = store.get_computed_users(period, "weekly_rg")
    else:
        # Legacy: derive from existing file names
        checkpoint = get_already_saved_user_per_period(str(base_out))
        already_done_scalars  = set(checkpoint[period].get("all_scalars", []))
        already_done_gonzalez = set(checkpoint[period].get("gonzalez", []))
        already_done_st       = set(checkpoint[period].get("S", []))
        already_done_freq     = set(checkpoint[period].get("frequency", []))
        already_done_wrg      = set(checkpoint[period].get("weekly_rg", []))

    # ── per-batch accumulators (parquet-store mode only) ──────────────────
    scalar_batch:   dict = {}
    gonzalez_batch: dict = {}
    st_batch:       dict = {}
    freq_batch:     dict = {}
    wrg_batch:      dict = {}
    all_weeks_for_wrg: list = []

    def _flush_batches(force: bool = False) -> None:
        """Write accumulated batches to the store."""
        if store is None:
            return
        if force or len(scalar_batch) >= batch_size:
            store.write_scalars_batch(period, scalar_batch)
            scalar_batch.clear()
        if force or len(gonzalez_batch) >= batch_size:
            store.write_gonzalez_batch(period, gonzalez_batch)
            gonzalez_batch.clear()
        if force or len(st_batch) >= batch_size:
            store.write_st_batch(period, st_batch)
            st_batch.clear()
        if force or len(freq_batch) >= batch_size:
            store.write_frequency_batch(period, freq_batch)
            freq_batch.clear()

    raw = cfg.get("raw_trajectories", False)

    # ── pre-group period DataFrame once ────────────────────────────────────
    # Calling groupby inside the user loop would be O(N*M) total (N users,
    # M rows).  Building the dict here is O(M) and every subsequent lookup
    # is O(1), typically giving 10–100x speedup on large shards.
    user_groups: dict | None = None
    if df is not None and raw:
        user_groups = {uid: grp for uid, grp in df.period2df[period].groupby("userId")}

    # ── serial user loop ────────────────────────────────────────────────────
    for user in list_users:
        uid_str = str(user)

        # ── skip already-computed (resume logic) ──────────────────────────
        if uid_str in already_done_scalars and store is not None:
            continue

        # ----------------------------------------------------------------
        # Instantiate User
        # ----------------------------------------------------------------
        if raw:
            user_df = user_groups[user] if (user_groups is not None and user in user_groups) else \
                      df.period2df[period].groupby("userId").get_group(user)
            ciccio  = User(
                user_df, period, dataset.id_,
                dataset.np_, dataset.t_threshold,
                dataset.period_names2period_division,
                output_dir=base_out,
            )
            ciccio.time_filtering_traj_per_person(df.t_threshold)
            dataset.period2totalpoints[period] += len(ciccio.df)
        else:
            ciccio = User(
                None, period, dataset.id_,
                dataset.np_, dataset.t_threshold,
                dataset.period_names2period_division,
                uname=user,
                output_dir=base_out,
            )

        # ----------------------------------------------------------------
        # Weekly point count
        # ----------------------------------------------------------------
        if cfg.get("is_week2points"):
            weeks = ciccio.divide_weeks(
                df.period_division, period, df.perodname2idx
            )
            for week in weeks:
                week2points.setdefault(week, [])
            w2p = ciccio.number_points_week(
                period, dictweek2npeople, df.period_division, df.perodname2idx
            )
            for idx_w, week in enumerate(list(w2p.keys())):
                week2points[weeks[idx_w]].append(list(w2p.keys())[idx_w])

        # ----------------------------------------------------------------
        # Weekly radius of gyration
        # ----------------------------------------------------------------
        if cfg.get("is_weekly_radius_gyration"):
            if dictweek2npeople is None:
                weeks = ciccio.divide_weeks(
                    df.period_division, period, df.perodname2idx
                )
                dictweek2npeople = {str(w): 0 for w in weeks}
                all_weeks_for_wrg = [str(w) for w in weeks]
            else:
                weeks = ciccio.divide_weeks(
                    df.period_division, period, df.perodname2idx
                )
            ciccio.compute_weekly_radius_gyration(
                period, dictweek2npeople, df.period_division, df.perodname2idx
            )
            if store is not None and uid_str not in already_done_wrg:
                wrg_batch[uid_str] = dict(ciccio.week2rg)
                if len(wrg_batch) >= batch_size:
                    store.write_weekly_rg_batch(period, wrg_batch, all_weeks_for_wrg)
                    wrg_batch.clear()
            else:
                ciccio._save_weekly_rg(period)

        # ----------------------------------------------------------------
        # Scalar metrics
        # ----------------------------------------------------------------
        def _run_if(flag, already_flag, method_name):
            if cfg.get(flag):
                if not cfg.get(already_flag) and raw:
                    getattr(ciccio, method_name)()

        _run_if("is_radius_gyration",       "already_computed_rg",                  "compute_radius_of_gyration")
        _run_if("is_random_entropy",        "already_computed_random_entropy",       "compute_random_entropy")
        _run_if("is_uncorrelated_entropy",  "already_computed_uncorrelated_entropy", "compute_uncorrelated_entropy")
        _run_if("is_real_entropy",          "already_computed_real_entropy",         "compute_real_entropy")
        _run_if("is_distance",              "already_computed_distance",             "compute_straight_line_distance")
        _run_if("is_home",                  "already_computed_home",                 "compute_home")
        _run_if("is_krg",                   "already_computed_krg",                  "compute_krg")
        _run_if("is_fraction_time",         "already_computed_fraction_time",        "compute_fraction_time_user_is_present")

        if cfg.get("is_county_rural") and not cfg.get("already_computed_county_rural"):
            ciccio._get_county(dataset.geojson, dataset.county2party, dataset.county2rural)

        # Gonzalez (writes its own file in legacy mode; batched in store mode)
        if cfg.get("is_gonzalez") and not cfg.get("already_computed_gonzalez") and raw:
            ciccio.compute_gonzalez()
            if store is not None and uid_str not in already_done_gonzalez:
                import pandas as _pd
                gdf = getattr(ciccio, "df2save_gonzalez", None)
                _GON_COLS = {"x_norm", "y_norm", "sigmax", "sigmay"}
                if (
                    isinstance(gdf, _pd.DataFrame)
                    and not gdf.empty
                    and _GON_COLS.issubset(gdf.columns)
                ):
                    gonzalez_batch[uid_str] = gdf

        # S(t) (writes its own file in legacy mode; batched in store mode)
        if cfg.get("is_St") and not cfg.get("already_computed_St") and raw:
            ciccio.compute_St()
            if store is not None and uid_str not in already_done_st:
                if hasattr(ciccio, "df_St"):
                    st_batch[uid_str] = ciccio.df_St["visited_places"].tolist()

        # Frequency (writes its own file in legacy mode; batched in store mode)
        if cfg.get("is_frequency") and not cfg.get("already_computed_frequency") and raw:
            ciccio.compute_frequency_location()
            if store is not None and uid_str not in already_done_freq:
                if hasattr(ciccio, "df2frequencyrank"):
                    freq_batch[uid_str] = ciccio.df2frequencyrank

        # ----------------------------------------------------------------
        # Save
        # ----------------------------------------------------------------
        if store is not None:
            # Accumulate scalar results for batch write
            if uid_str not in already_done_scalars:
                scalar_batch[uid_str] = dict(ciccio.df2save)
            _flush_batches()
        else:
            # Legacy: write one CSV.gz per user
            if cfg.get("save_results", True):
                ciccio._save_df()

        overall_count += 1
        if overall_count % 10_000 == 0:
            print(f"Processed {overall_count} users in period '{period}'")

    # Final flush of any remaining batch data
    _flush_batches(force=True)
    if store is not None and wrg_batch:
        store.write_weekly_rg_batch(period, wrg_batch, all_weeks_for_wrg)
        wrg_batch.clear()

    # Save weekly people-count summary
    npeople_path = DIR_OUTPUT / dataset.id_ / f"number_users_period_{period}.json"
    with open(npeople_path, "w") as f:
        json.dump(dictweek2npeople, f, indent=2)

    return week2points


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def analyze_from_dataset(
    dataset: _BaseDataset,
    region: str,
    config_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    store: "ParquetStore | None" = None,
    batch_size: int = 500,
    use_vectorized: bool = False,
) -> None:
    """
    Run the full analysis pipeline for ``dataset``.

    Parameters
    ----------
    dataset : _BaseDataset
        Initialised dataset object.
    region : str
        ``"CA"`` or ``"MA"``.
    config_dir : Path or str, optional
        Directory containing the ``config_<region>.json`` file.
    output_dir : Path or str, optional
        Override output directory for per-user files (legacy mode).
    store : ParquetStore, optional
        When supplied, results are written to the columnar parquet store
        (high-throughput mode).  Pass ``None`` to use the legacy
        per-file path.
    batch_size : int
        Users per shard write in parquet-store mode.
    use_vectorized : bool
        When ``True``, use the polars/numba vectorized path instead of
        the per-user loop.  Much faster on large shards.  Default ``False``.
    """
    cfg = get_config(region, config_dir)
    base_out = (
        Path(output_dir)
        if output_dir
        else DIR_MILESTONES_SERVER / dataset.id_ / "dataxuser"
    )
    ifnotexistsmkdir(base_out)

    if cfg.get("raw_trajectories"):
        for file in dataset.dir_files:
            df = dataset_info(
                file,
                dataset.period_division,
                dataset.period_names,
                dataset.np_,
                dataset.t_threshold,
                dataset.bounding_box,
            )
            print(f"Loaded {len(df.df):,} rows from {file}")
            df.preprocess()

            for period in df.period_names:
                n_users = len(df.period2listusers[period])
                dataset.period2totalusers[period] += n_users
                print(f"  Period '{period}': {dataset.period2totalusers[period]:,} users")
                week2point = compute_all(
                    cfg, dataset, df.period2listusers[period],
                    period, df,
                    output_dir=base_out,
                    store=store,
                    batch_size=batch_size,
                    use_vectorized=use_vectorized,
                )
                print(
                    f"  Total users: {dataset.period2totalusers[period]:,}  "
                    f"Total points: {dataset.period2totalpoints[period]:,}"
                )
                for week, counts in week2point.items():
                    plt.hist(counts)
                    plt.xlabel(f"Period: {period}  Week: {week}")
                    plt.ylabel("Point count")
                    out = base_out.parent / "plots"
                    ifnotexistsmkdir(out)
                    plt.savefig(out / f"count_points_{period}_{week}.png", dpi=200)
                    plt.close()

            # Consolidate shards after each file to keep the shard count low
            if store is not None:
                for period in df.period_names:
                    store.consolidate_all(period)
    else:
        directory = base_out
        user2period = get_already_saved_user_per_period(directory)
        for period in user2period:
            list_users = user2period[period]["all_scalars"]
            week2point = compute_all(
                cfg, dataset, list_users, period,
                output_dir=base_out,
                store=store,
                batch_size=batch_size,
                use_vectorized=use_vectorized,
            )
            print(
                f"  Period '{period}' — users: {dataset.period2totalusers[period]:,}  "
                f"points: {dataset.period2totalpoints[period]:,}"
            )


# ---------------------------------------------------------------------------
# S3 progressive orchestrator
# ---------------------------------------------------------------------------

def _shard_label(shard_key: str) -> str:
    """Return a stable 12-hex label from an S3 shard key (used as per-shard output key)."""
    import hashlib
    return hashlib.md5(shard_key.encode()).hexdigest()[:12]


def analyze_from_s3_progressive(
    dataset: _BaseDataset,
    region: str,
    cfg: dict,
    store: "ParquetStore",
    *,
    endpoint_url: str,
    bucket: str,
    s3_prefix: str,
    temp_dir: Path | None = None,
    batch_size: int = 500,
    use_vectorized: bool = False,
    output_endpoint_url: str | None = None,
    output_bucket: str | None = None,
    output_s3_prefix: str | None = None,
    delete_local_after_upload: bool = True,
) -> None:
    """
    Download raw Cuebiq parquet shards from S3 one at a time, compute all
    enabled metrics, save results to *store*, then delete the local copy.

    **Output to S3 (recommended)**

    When *output_bucket* and *output_s3_prefix* are supplied, each shard's
    output is uploaded to S3 immediately after processing and the local
    files are deleted — no output accumulates on local disk.  A final
    ``consolidate_s3_shards()`` call merges all per-shard S3 files into the
    canonical ``consolidated.parquet`` at the end.

    This function is designed to be interrupted and restarted safely:

    * **Shard-level resume** — a JSON checkpoint file
      ``milestones_analysis/{region}/shard_checkpoint_np_{np_}_t_{t}.json``
      records every shard that has been fully processed.  On restart, those
      shards are skipped entirely.

    * **User-level resume** — within each shard, users already present in
      *store* are detected via parquet footer metadata (O(1)) and skipped,
      so partial progress inside an interrupted shard is preserved.

    Parameters
    ----------
    dataset : _BaseDataset
        Initialised region dataset object.
    region : str
        ``"CA"`` or ``"MA"``.
    cfg : dict
        Algorithm-flow config from ``get_config()``.
    store : ParquetStore
        Target parquet store where results will be written.
    endpoint_url : str
        S3-compatible endpoint URL (e.g. ``"https://s3.atlas.fbk.eu"``).
    bucket : str
        Bucket name (e.g. ``"chub-datalake"``).
    s3_prefix : str
        Key prefix for raw shard files inside the bucket (no leading slash,
        no trailing slash).
    temp_dir : Path, optional
        Local directory for downloading shards before processing.
        Defaults to ``DIR_SHARD_TEMP``.
    batch_size : int
        Users to accumulate per parquet shard write.
    use_vectorized : bool
        When ``True``, delegate each shard's computation to the polars/numba
        vectorized path (``compute_all_polars``).  Recommended for large shards.
        Default ``False``.
    output_endpoint_url : str, optional
        S3 endpoint for the output store.  Defaults to ``endpoint_url``.
    output_bucket : str, optional
        Bucket to write output parquet files.  When set, each shard's output
        is uploaded immediately and local files are deleted.
    output_s3_prefix : str, optional
        Key prefix for output files (e.g.
        ``"final_pipeline/CA"``).
    delete_local_after_upload : bool
        Delete local output files after a successful S3 upload (default True).
    """
    _temp_dir = Path(temp_dir) if temp_dir else DIR_SHARD_TEMP
    _temp_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. List available shards on S3 ─────────────────────────────────────
    s3_uri = f"s3://{bucket}/{s3_prefix}"
    print(f"Listing shards at {s3_uri} (recursive) …")
    result = subprocess.run(
        [
            "aws", "s3", "ls", s3_uri + "/",
            "--endpoint-url", endpoint_url,
            "--recursive",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"aws s3 ls failed:\n  stdout: {result.stdout}\n  stderr: {result.stderr}"
        )

    # With --recursive the last column is the full S3 key relative to the bucket
    # root (e.g. "shared/cuebiq/MOBS/.../part-00000.parquet").  We use the full
    # key as the stable checkpoint identifier and derive the local filename from
    # the basename so that a flat temp-directory is always used.
    # Entries whose last column is a directory prefix ("PRE …") never end in
    # ".parquet" and are naturally excluded.
    shard_full_keys: list[str] = [
        line.split()[-1]
        for line in result.stdout.strip().splitlines()
        if line.strip() and line.split()[-1].endswith(".parquet")
    ]
    if not shard_full_keys:
        print(f"No .parquet shards found at {s3_uri}. Nothing to do.")
        # Print the raw listing to aid debugging
        if result.stdout.strip():
            preview = "\n  ".join(result.stdout.strip().splitlines()[:10])
            print(f"  Raw listing (first 10 lines):\n  {preview}")
        return

    # Deduplicate basenames: if two keys share the same basename, append a
    # short hash to avoid local collisions.
    _seen_names: dict[str, int] = {}
    shard_local_names: list[str] = []
    for key in shard_full_keys:
        base = Path(key).name
        if base in _seen_names:
            _seen_names[base] += 1
            stem, _, ext = base.rpartition(".")
            base = f"{stem}_{_seen_names[base]}.{ext}"
        else:
            _seen_names[base] = 0
        shard_local_names.append(base)

    # checkpoint keys are the full S3 keys (stable across re-runs)

    # ── 2. Load shard-level checkpoint ──────────────────────────────────────
    checkpoint_path = (
        DIR_MILESTONES_SERVER
        / region
        / f"shard_checkpoint_np_{dataset.np_}_t_{dataset.t_threshold}.json"
    )
    if checkpoint_path.exists():
        with open(checkpoint_path) as _f:
            completed_shards: set[str] = set(json.load(_f).get("completed", []))
    else:
        completed_shards = set()

    pending_keys:   list[str] = [k for k in shard_full_keys if k not in completed_shards]
    pending_locals: list[str] = [
        local
        for k, local in zip(shard_full_keys, shard_local_names)
        if k not in completed_shards
    ]
    print(
        f"Shards — total: {len(shard_full_keys)}  "
        f"completed: {len(completed_shards)}  "
        f"pending: {len(pending_keys)}"
    )

    if not pending_keys:
        print("All shards already processed. Nothing to do.")
        return

    # ── 3. Process shards one by one ────────────────────────────────────────
    for shard_idx, (shard_key, local_name) in enumerate(
        zip(pending_keys, pending_locals)
    ):
        local_path = _temp_dir / local_name
        s3_key     = f"s3://{bucket}/{shard_key}"

        print(f"\n[{shard_idx + 1}/{len(pending_keys)}] Processing {shard_key} …")

        # 3a. Download
        print(f"  Downloading from {s3_key} …")
        dl = subprocess.run(
            ["aws", "s3", "cp", s3_key, str(local_path),
             "--endpoint-url", endpoint_url],
            capture_output=True, text=True,
        )
        if dl.returncode != 0:
            print(f"  WARNING: download failed — skipping shard.\n  {dl.stderr.strip()}")
            continue

        # 3b. Preprocess
        try:
            if use_vectorized:
                # ── Fast path: polars preprocessing (no pandas, no skmob) ──
                from .vectorized_pipeline import (
                    preprocess_shard_polars as _prep_polars,
                    compute_all_polars as _vec_all,
                )
                period2pldf = _prep_polars(local_path, dataset)
                print(f"  [vectorized] Preprocessed {len(period2pldf)} periods.")

                for period in dataset.period_names:
                    pldf = period2pldf.get(period)
                    if pldf is None or pldf.is_empty():
                        continue
                    n_users = pldf["userId"].n_unique()
                    dataset.period2totalusers[period] += n_users
                    dataset.period2totalpoints[period] += len(pldf)
                    print(f"  Period '{period}': {n_users:,} users …")

                    ad_scalars  = store.get_computed_users(period, "all_scalars")
                    ad_gonzalez = store.get_computed_users_long(period, "gonzalez")
                    ad_st       = store.get_computed_users(period, "S")
                    ad_freq     = store.get_computed_users_long(period, "frequency")
                    ad_wrg      = store.get_computed_users(period, "weekly_rg")

                    _vec_all(
                        cfg, dataset, pldf, period,
                        ad_scalars, ad_gonzalez, ad_st, ad_freq, ad_wrg,
                        store, batch_size,
                    )

                local_path.unlink(missing_ok=True)
                print(f"  Local copy deleted.")
                completed_shards.add(shard_key)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w") as _f:
                    json.dump({"completed": sorted(completed_shards)}, _f, indent=2)

            # ── Legacy path: pandas / skmob preprocessing ──────────────────
            else:
                df_info = dataset_info(
                    local_path,
                    dataset.period_division,
                    dataset.period_names,
                    dataset.np_,
                    dataset.t_threshold,
                    dataset.bounding_box,
                )
                print(f"  Loaded {len(df_info.df):,} rows. Preprocessing …")
                df_info.preprocess()
        except Exception as exc:
            print(f"  WARNING: preprocessing error — skipping shard.\n  {exc}")
            local_path.unlink(missing_ok=True)
            continue

        # ── Consolidate / upload — runs for BOTH vectorized and legacy paths,
        #    OUTSIDE the preprocessing try/except so upload errors do not
        #    masquerade as preprocessing failures.
        if use_vectorized:
            print(f"  Consolidating store …")
            _label = _shard_label(shard_key)
            _out_ep = output_endpoint_url or endpoint_url
            for period in dataset.period_names:
                try:
                    if output_bucket and output_s3_prefix:
                        store.upload_shard_to_s3_unique(
                            period, _label,
                            output_bucket, output_s3_prefix, _out_ep,
                            delete_after=delete_local_after_upload,
                        )
                    else:
                        store.consolidate_all(period)
                except Exception as _upload_exc:
                    print(
                        f"  WARNING: upload/consolidation failed for period {period!r}"
                        f" — local results preserved, shard marked done.\n  {_upload_exc}"
                    )
            print(
                f"  Cumulative users: "
                + "  ".join(
                    f"{p!r}: {dataset.period2totalusers[p]:,}"
                    for p in dataset.period_names
                )
            )
            continue  # → next shard (skip legacy block below)

        # 3c. Compute metrics per period — per-period try/except so one
        #     failing period does NOT skip the remaining periods in the shard.
        shard_had_error = False
        for period in df_info.period_names:
            try:
                n_users = len(df_info.period2listusers[period])
                dataset.period2totalusers[period] += n_users
                if n_users == 0:
                    continue
                print(f"  Period '{period}': {n_users:,} users …")
                compute_all(
                    cfg,
                    dataset,
                    df_info.period2listusers[period],
                    period,
                    df_info,
                    store=store,
                    batch_size=batch_size,
                    use_vectorized=use_vectorized,
                )
            except Exception as exc:
                import traceback
                print(
                    f"  WARNING: compute error for period '{period}' — skipping period.\n"
                    f"  {exc}\n"
                    f"  {traceback.format_exc()}"
                )
                shard_had_error = True
                continue

        # 3d. Delete local copy immediately to save disk space
        local_path.unlink(missing_ok=True)
        if shard_had_error:
            print(f"  Local copy deleted (shard had partial errors — NOT marking as completed).")
            # Do not add to completed_shards so the shard will be retried on the
            # next run, potentially recovering users from periods that errored.
        else:
            print(f"  Local copy deleted.")

        # 3e. Update shard checkpoint (only for fully-clean shards)
        if not shard_had_error:
            completed_shards.add(shard_key)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as _f:
                json.dump({"completed": sorted(completed_shards)}, _f, indent=2)

        # 3f. Consolidate and optionally upload output to S3, freeing local disk
        if not shard_had_error:
            print(f"  Consolidating store …")
            _label = _shard_label(shard_key)
            _out_ep = output_endpoint_url or endpoint_url
            for period in dataset.period_names:
                try:
                    if output_bucket and output_s3_prefix:
                        store.upload_shard_to_s3_unique(
                            period, _label,
                            output_bucket, output_s3_prefix, _out_ep,
                            delete_after=delete_local_after_upload,
                        )
                    else:
                        store.consolidate_all(period)
                except Exception as _upload_exc:
                    print(
                        f"  WARNING: upload/consolidation failed for period {period!r}"
                        f" — local results preserved, shard marked done.\n  {_upload_exc}"
                    )

        print(
            f"  Cumulative users: "
            + "  ".join(
                f"{p!r}: {dataset.period2totalusers[p]:,}"
                for p in dataset.period_names
            )
        )

    # ── 4. Final step ────────────────────────────────────────────────────────
    if output_bucket and output_s3_prefix:
        _out_ep = output_endpoint_url or endpoint_url
        print(
            "\nAll shards processed.  Merging per-shard S3 files into"
            " final consolidated.parquet …"
        )
        for period in dataset.period_names:
            store.consolidate_s3_shards(
                period, output_bucket, output_s3_prefix, _out_ep,
                delete_shards_after=True,
            )
    else:
        print("\nFinal store consolidation …")
        for period in dataset.period_names:
            store.consolidate_all(period)
    print("Done.")

