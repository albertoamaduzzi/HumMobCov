"""
pipeline.py
===========
High-level processing pipeline: per-user metric computation and the
top-level ``analyze_from_dataset`` orchestrator.

Parquet-store mode
------------------
Pass a ``ParquetStore`` instance (``store=``) to ``compute_all`` or
``analyze_from_dataset`` to use the new columnar parquet backend.

When ``store`` is supplied:

* Resume check is done via parquet footer metadata (O(1), no data load).
* Per-user results are batched in memory and written as shard parquet files
  rather than individual ``*.csv.gz`` files — much lower I/O overhead.
* The old per-file path is still available when ``store=None`` (default).
"""

import json
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
from .constants import DIR_OUTPUT, DIR_CONFIG, DIR_MILESTONES_SERVER

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
