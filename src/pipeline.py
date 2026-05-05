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

    Returns
    -------
    dict
        ``{week_datetime: [point_counts]}``
    """
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

    for user in list_users:
        uid_str = str(user)

        # ── skip already-computed (resume logic) ──────────────────────────
        if uid_str in already_done_scalars and store is not None:
            continue

        # ----------------------------------------------------------------
        # Instantiate User
        # ----------------------------------------------------------------
        if raw:
            user_df = df.period2df[period].groupby("userId").get_group(user)
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
                if hasattr(ciccio, "df2save_gonzalez") and ciccio.df2save_gonzalez is not None:
                    import pandas as _pd
                    gdf = ciccio.df2save_gonzalez
                    if isinstance(gdf, dict):
                        gdf = _pd.DataFrame(gdf)
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
            )
            print(
                f"  Period '{period}' — users: {dataset.period2totalusers[period]:,}  "
                f"points: {dataset.period2totalpoints[period]:,}"
            )


# ---------------------------------------------------------------------------
# S3 progressive orchestrator
# ---------------------------------------------------------------------------

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
    consolidate_every: int = 1,
) -> None:
    """
    Download raw Cuebiq parquet shards from S3 one at a time, compute all
    enabled metrics, save results to *store*, then delete the local copy.

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
    consolidate_every : int
        Consolidate the parquet store after every *N* processed shards
        (default 1 = after every shard).  Larger values reduce I/O at the
        cost of larger individual shard files.
    """
    _temp_dir = Path(temp_dir) if temp_dir else DIR_SHARD_TEMP
    _temp_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. List available shards on S3 ─────────────────────────────────────
    s3_uri = f"s3://{bucket}/{s3_prefix}"
    print(f"Listing shards at {s3_uri} …")
    result = subprocess.run(
        ["aws", "s3", "ls", s3_uri + "/", "--endpoint-url", endpoint_url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"aws s3 ls failed:\n  stdout: {result.stdout}\n  stderr: {result.stderr}"
        )

    shard_names: list[str] = [
        line.split()[-1]
        for line in result.stdout.strip().splitlines()
        if line.strip() and line.split()[-1].endswith(".parquet")
    ]
    if not shard_names:
        print(f"No .parquet shards found at {s3_uri}. Nothing to do.")
        return

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

    pending = [s for s in shard_names if s not in completed_shards]
    print(
        f"Shards — total: {len(shard_names)}  "
        f"completed: {len(completed_shards)}  "
        f"pending: {len(pending)}"
    )

    if not pending:
        print("All shards already processed. Nothing to do.")
        return

    # ── 3. Process shards one by one ────────────────────────────────────────
    for shard_idx, shard_name in enumerate(pending):
        local_path = _temp_dir / shard_name
        s3_key     = f"s3://{bucket}/{s3_prefix}/{shard_name}"

        print(f"\n[{shard_idx + 1}/{len(pending)}] Processing {shard_name} …")

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

        # 3c. Compute metrics per period
        try:
            for period in df_info.period_names:
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
                )
        except Exception as exc:
            print(f"  WARNING: compute error — skipping shard.\n  {exc}")
            local_path.unlink(missing_ok=True)
            continue

        # 3d. Delete local copy immediately to save disk space
        local_path.unlink(missing_ok=True)
        print(f"  Local copy deleted.")

        # 3e. Update shard checkpoint
        completed_shards.add(shard_name)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as _f:
            json.dump({"completed": sorted(completed_shards)}, _f, indent=2)

        # 3f. Periodic consolidation
        if (shard_idx + 1) % consolidate_every == 0:
            print(f"  Consolidating store …")
            for period in dataset.period_names:
                store.consolidate_all(period)

        print(
            f"  Cumulative users: "
            + "  ".join(
                f"{p!r}: {dataset.period2totalusers[p]:,}"
                for p in dataset.period_names
            )
        )

    # ── 4. Final consolidation ───────────────────────────────────────────────
    print("\nFinal store consolidation …")
    for period in dataset.period_names:
        store.consolidate_all(period)
    print("Done.")

