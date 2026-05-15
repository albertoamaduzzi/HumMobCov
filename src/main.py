"""
HumMobCov — Main Analysis Script
=================================
Command-line equivalent of src/main.ipynb.

Usage
-----
  python src/main.py --region CA
  python src/main.py --region MA --np 30 --t-threshold 2
  python src/main.py --region CA --skip-pipeline --visualize
  python src/main.py --region CA --visualize --output-dir /my/output

Execution modes (selected automatically in order):
  A  Local raw parquet files found         → analyze_from_dataset()
  B  raw_trajectories=true, files on S3    → analyze_from_s3_progressive()
  C  CA legacy dataxuser/ CSV.gz files     → store.migrate_all_periods()
  C  MA legacy per-metric shard dirs       → store.migrate_all_periods_MA()

Resume safety:
  - S3 mode: shard_checkpoint_np_{np_}_t_{t}.json tracks completed shards.
  - All modes: user-level resume via parquet footer metadata (O(1)).

Sections
--------
  1. INPUT       — choose region, load dataset, inspect config
  2. MAIN        — run the full processing / migration pipeline
  3. VISUALIZATION — produce all plots from saved per-user results
"""

import argparse
import gc
import os as _os_pre
import sys
from pathlib import Path

# ─── Make the src package importable regardless of working directory ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # HumMobCov/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Cap Polars thread pool to the real cgroup CPU quota ─────────────────────
# Must happen BEFORE polars is imported (happens transitively via src below).
# os.cpu_count() / sched_getaffinity both return host CPUs in a container;
# the cgroup v2 cpu.max file gives the actual quota.
try:
    with open("/sys/fs/cgroup/cpu.max") as _f:
        _quota, _period = _f.read().split()
    if _quota != "max":
        _n_real = max(1, int(float(_quota) / float(_period)))
        _os_pre.environ.setdefault("POLARS_NUM_THREADS", str(_n_real))
except Exception:
    pass

from src import (
    # constants
    PROJECT_ROOT, DIR_SRC, DIR_OUTPUT, DIR_DATA, DIR_CONFIG,
    PERIOD_NAMES, PERIOD_DIVISION, PERIOD_NAMES_TO_DIVISION,
    MIN_POINTS_PER_USER, TIME_THRESHOLD_HOURS, US_BOUNDING_BOX,
    RURALITY_LEVELS, PARTY_NAMES, K_RADIUS_VALUES, LIST_REGIONS,
    # dataset classes
    DataSet_California, DataSet_Massachusets, dataset_info,
    # pipeline
    analyze_from_dataset, analyze_from_s3_progressive, compute_all, get_config,
    # user
    User,
    # plotter
    plotter,
    # storage
    ParquetStore,
    # utilities
    get_already_saved_user_per_period, ifnotexistsmkdir,
)
from src.constants import (
    DIR_MILESTONES_SERVER,
    S3_ENDPOINT_URL, S3_BUCKET, S3_RAW_PREFIX, DIR_SHARD_TEMP,
    S3_OUTPUT_BUCKET, S3_OUTPUT_PREFIX, S3_TRANSITION_PREFIX,
)
from src.tile_counties_via_geohash import tile_counties_via_geohash
from src.transition_matrices import TransitionPipeline
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for shell execution
import matplotlib.pyplot as plt
import json


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="HumMobCov main analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--region", choices=LIST_REGIONS, default="CA",
        help="Region to analyse.",
    )
    parser.add_argument(
        "--np", dest="np_", type=int, default=20,
        help="Override minimum stops per user per period (MIN_POINTS_PER_USER).",
    )
    parser.add_argument(
        "--t-threshold", dest="t_threshold", type=int, default=1,
        help="Override minimum hours between stops (TIME_THRESHOLD_HOURS).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Override config directory.",
    )
    parser.add_argument(
        "--s3-endpoint", type=str, default=None,
        help="Override S3 endpoint URL (default from S3_ENDPOINT_URL env / constants).",
    )
    parser.add_argument(
        "--s3-bucket", type=str, default=None,
        help="Override S3 bucket name (default from S3_BUCKET env / constants).",
    )
    parser.add_argument(
        "--s3-prefix", type=str, default=None,
        help="Override S3 key prefix for raw shards (default from S3_RAW_PREFIX[region]).",
    )
    parser.add_argument(
        "--temp-dir", type=Path, default=None,
        help="Override temp directory for S3 shard downloads (default DIR_SHARD_TEMP).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20000,
        help="Users per parquet shard write (S3 / local raw modes).",
    )
    parser.add_argument(
        "--no-vectorized", dest="use_vectorized", action="store_false",
        help="Disable the polars/numpy vectorized path and use the legacy skmob per-user loop.",
    )
    parser.set_defaults(use_vectorized=True)
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Skip section 2 (pipeline / migration) and go straight to visualisation.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Run section 3 (visualisation) after the pipeline.",
    )
    parser.add_argument(
        "--gap-analysis", action="store_true",
        help="Run section 4 gap-analysis plots after the pipeline.",
    )
    parser.add_argument(
        "--consolidate-s3", action="store_true",
        help=(
            "Run section 4 S3 consolidation: merge per-shard S3 files into "
            "consolidated.parquet for every period (MODE B only)."
        ),
    )
    parser.add_argument(
        "--upload-to-s3", action="store_true",
        help=(
            "Run section 4 S3 upload: push local parquet results to S3 "
            "(MODE A / MODE C)."
        ),
    )
    parser.add_argument(
        "--transition-matrices", action="store_true",
        help="Run section 5 transition matrices pipeline.",
    )
    parser.add_argument(
        "--geohash-precision", type=int, default=4,
        help="Geohash precision for transition matrices (4 ≈ 39×20 km, 5 ≈ 5×5 km).",
    )
    parser.add_argument(
        "--delta-time-h", type=float, default=1.0,
        help="Time-bin width in hours for transition matrices.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — INPUT
# ─────────────────────────────────────────────────────────────────────────────

def section_input(args):
    print("─" * 60)
    print("SECTION 1 · INPUT")
    print("─" * 60)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir:   {DIR_OUTPUT}")
    print(f"Data dir:     {DIR_DATA}")
    print(f"S3 endpoint:  {S3_ENDPOINT_URL}")
    print(f"S3 bucket:    {S3_BUCKET}")
    print(f"Out bucket:   {S3_OUTPUT_BUCKET}")

    # Initialise dataset
    if args.region == "CA":
        dataset = DataSet_California()
    elif args.region == "MA":
        dataset = DataSet_Massachusets()
    else:
        raise ValueError(f"Unknown region '{args.region}'. Choose from {LIST_REGIONS}")

    # Apply overrides
    if args.np_ is not None:
        dataset.np_ = args.np_
    if args.t_threshold is not None:
        dataset.t_threshold = args.t_threshold

    print(f"\nRegion:              {dataset.id_}")
    print(f"Min points (np_):    {dataset.np_}")
    print(f"Time threshold (h):  {dataset.t_threshold}")
    print(f"Output directory:    {dataset.dir_output}")
    print(f"Number of raw files: {len(dataset.dir_files)}")

    # Preview config
    cfg = get_config(args.region, config_dir=args.config_dir)
    print("\nActive feature flags:")
    for key, val in cfg.items():
        if not key.startswith("_"):
            flag = "✓" if val else "✗"
            print(f"  {flag}  {key}")

    # Preview time periods
    print("\nAnalysis periods:")
    for name, (start, end) in PERIOD_NAMES_TO_DIVISION.items():
        print(f"  {name:25s}  {start.date()}  →  {end.date()}")

    return dataset, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — MAIN (pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def section_main(args, dataset, cfg):
    print("\n" + "─" * 60)
    print("SECTION 2 · MAIN")
    print("─" * 60)

    # ── Resolve S3 settings (CLI flags take precedence over constants) ────────
    s3_endpoint = args.s3_endpoint or S3_ENDPOINT_URL
    s3_bucket   = args.s3_bucket   or S3_BUCKET
    s3_prefix   = args.s3_prefix   or S3_RAW_PREFIX[args.region]
    temp_dir    = args.temp_dir    or DIR_SHARD_TEMP

    store = ParquetStore(
        base_dir    = DIR_MILESTONES_SERVER / args.region,
        np_         = dataset.np_,
        t_threshold = dataset.t_threshold,
    )

    # ── Check what is already computed ───────────────────────────────────────
    # For S3 / local-raw modes the real completion signal is the shard
    # checkpoint: a period is "done" only when every raw shard has been
    # processed.  We cannot infer that from store contents alone, because a
    # partial run (1-of-N shards) produces a fully-consistent store that
    # would wrongly appear complete.
    #
    # Strategy:
    #   - S3 / local-raw modes: delegate entirely to analyze_from_s3_progressive
    #     / analyze_from_dataset — they manage their own resume logic.
    #     periods_todo = all periods (the inner functions skip already-done users).
    #   - Legacy migration modes (no raw_trajectories): a period is done when
    #     all_scalars has any users (migration is user-granular and resume-safe).
    use_shard_resume = bool(cfg.get("raw_trajectories"))

    if use_shard_resume:
        # Always enter the pipeline; inner resume logic handles skipping.
        periods_done = []
        periods_todo = list(PERIOD_NAMES)
    else:
        periods_done = [
            p for p in PERIOD_NAMES
            if len(store.get_computed_users(p, "all_scalars")) > 0
        ]
        periods_todo = [p for p in PERIOD_NAMES if p not in periods_done]

    use_vectorized = args.use_vectorized

    print(f"Periods already in store  : {periods_done or 'none'}")
    print(f"Periods still to compute  : {periods_todo or 'none'}")
    print(f"Compute path              : {'vectorized (polars)' if use_vectorized else 'legacy (skmob)'}")

    if not periods_todo:
        print("Parquet store already populated for all periods. Nothing to do.")

    else:
        # ── CASE A: local raw parquet files ───────────────────────────────────
        local_raw_files = [f for f in dataset.dir_files if Path(f).exists()]

        if cfg.get("raw_trajectories") and local_raw_files:
            print(f"\nMODE A — local raw data ({len(local_raw_files)} shards found).")
            analyze_from_dataset(
                dataset,
                region          = args.region,
                config_dir      = args.config_dir,
                output_dir      = args.output_dir,
                store           = store,
                batch_size      = args.batch_size,
                use_vectorized  = use_vectorized,
            )
            print("Computation complete. Consolidating shards…")
            for p in dataset.period_names:
                store.consolidate_all(p)
            print("Uploading results to S3…")
            store.upload_all_to_s3(
                period_names  = dataset.period_names,
                s3_bucket     = S3_OUTPUT_BUCKET,
                s3_prefix     = S3_OUTPUT_PREFIX[args.region],
                endpoint_url  = s3_endpoint,
                delete_after  = False,
                consolidate_first = False,
            )
            print("Done.")

        # ── CASE B: S3 progressive (download one shard → compute → delete) ───
        elif cfg.get("raw_trajectories"):
            print(
                f"\nMODE B — S3 progressive download.\n"
                f"  Input endpoint  : {s3_endpoint}\n"
                f"  Input bucket    : {s3_bucket}\n"
                f"  Input prefix    : {s3_prefix}\n"
                f"  Temp dir        : {temp_dir}\n"
                f"  Output bucket   : {S3_OUTPUT_BUCKET}\n"
                f"  Output prefix   : {S3_OUTPUT_PREFIX[args.region]}\n"
                f"  Periods         : {periods_todo}"
            )
            analyze_from_s3_progressive(
                dataset,
                region            = args.region,
                cfg               = cfg,
                store             = store,
                endpoint_url      = s3_endpoint,
                bucket            = s3_bucket,
                s3_prefix         = s3_prefix,
                temp_dir          = temp_dir,
                batch_size        = args.batch_size,
                use_vectorized    = use_vectorized,
                output_endpoint_url        = s3_endpoint,
                output_bucket              = S3_OUTPUT_BUCKET,
                output_s3_prefix           = S3_OUTPUT_PREFIX[args.region],
                delete_local_after_upload  = True,
            )

        # ── CASE C-CA: migrate legacy dataxuser per-user CSV.gz files ─────────
        elif args.region == "CA":
            legacy_dir = DIR_MILESTONES_SERVER / args.region / "dataxuser"
            if legacy_dir.exists() and any(legacy_dir.iterdir()):
                print(
                    f"\nMODE C-CA — migrating legacy dataxuser/ CSV.gz files.\n"
                    f"  Periods missing: {periods_todo}\n"
                    f"  (Already-migrated users are skipped — safe to re-run.)"
                )
                store.migrate_all_periods(
                    legacy_dir   = legacy_dir,
                    period_names = periods_todo,
                    np_          = dataset.np_,
                    t            = dataset.t_threshold,
                    batch_size   = args.batch_size,
                    consolidate  = True,
                )
                print("Uploading results to S3…")
                store.upload_all_to_s3(
                    period_names  = dataset.period_names,
                    s3_bucket     = S3_OUTPUT_BUCKET,
                    s3_prefix     = S3_OUTPUT_PREFIX[args.region],
                    endpoint_url  = s3_endpoint,
                    delete_after  = False,
                    consolidate_first = False,
                )
            else:
                print(
                    f"\nCA: no dataxuser/ directory found at {legacy_dir}.\n"
                    f"Set raw_trajectories=true in config_CA.json to compute from S3."
                )

        # ── CASE C-MA: migrate legacy per-metric shard directories ────────────
        elif args.region == "MA":
            ma_legacy_base = DIR_MILESTONES_SERVER / "MA"
            print(
                f"\nMODE C-MA — migrating legacy metric-folder shard files.\n"
                f"  Periods missing: {periods_todo}\n"
                f"  np_={dataset.np_}, t={dataset.t_threshold}\n"
                f"  (Already-migrated users are skipped — safe to re-run.)"
            )
            store.migrate_all_periods_MA(
                ma_legacy_base = ma_legacy_base,
                period_names   = periods_todo,
                np_            = dataset.np_,
                t              = dataset.t_threshold,
                batch_size     = args.batch_size,
                consolidate    = True,
            )
            print("Uploading results to S3…")
            store.upload_all_to_s3(
                period_names  = dataset.period_names,
                s3_bucket     = S3_OUTPUT_BUCKET,
                s3_prefix     = S3_OUTPUT_PREFIX[args.region],
                endpoint_url  = s3_endpoint,
                delete_after  = False,
                consolidate_first = False,
            )

        else:
            print(f"Unknown region {args.region!r}. No action taken.")

    # ── Pipeline summary ──────────────────────────────────────────────────────
    print("\nUsers available in parquet store:")
    for period in PERIOD_NAMES:
        scalars  = store.get_computed_users(period, "all_scalars")
        gonzalez = store.get_computed_users_long(period, "gonzalez")
        st       = store.get_computed_users(period, "S")
        wrg      = store.get_computed_users(period, "weekly_rg")
        freq     = store.get_computed_users_long(period, "frequency")
        print(
            f"  {period:25s}  "
            f"scalars={len(scalars):>7,}  "
            f"gonzalez={len(gonzalez):>7,}  "
            f"S(t)={len(st):>7,}  "
            f"weekly_rg={len(wrg):>7,}  "
            f"freq={len(freq):>7,}"
        )

    return store


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def section_visualization(args, dataset, store):
    print("\n" + "─" * 60)
    print("SECTION 3 · VISUALIZATION")
    print("─" * 60)

    plt_obj = plotter(
        np_             = dataset.np_,
        period_division = dataset.period_division,
        period_names    = dataset.period_names,
        t_threshold     = dataset.t_threshold,
        region          = args.region,
        county2party    = dataset.county2party,
        df_rurality     = dataset.df_rurality,
        output_dir      = args.output_dir,
        store           = store,
    )

    print("Users available per period (from parquet store):")
    for period in plt_obj.period_names:
        n = len(store.get_computed_users(period, "all_scalars"))
        print(f"  {period:25s}  {n:>8,} users")

    def _plot_one(label: str, fn) -> None:
        """
        Run a single plot function, then immediately close all open
        matplotlib figures and force a full garbage-collection cycle.

        Processing one plot at a time keeps peak RSS bounded: the large
        arrays loaded by each ``_load_*`` helper (especially the S(t)
        matrix, which can exceed several GB) are freed before the next
        plot starts, preventing accumulation that would otherwise exhaust
        RAM and drop the SSH connection.
        """
        print(f"  {label}")
        try:
            fn()
        except Exception as exc:
            print(f"    WARNING — {label} failed: {exc}")
        finally:
            plt.close("all")   # release all figure objects and their arrays
            gc.collect()       # reclaim memory immediately before next load

    print("\nGenerating plots…")

    _plot_one("3.1 Radius of Gyration",                 plt_obj.plot_rg)
    _plot_one("3.1 Radius of Gyration (party)",         plt_obj.plot_rg_party_per_period)
    _plot_one("3.1 Radius of Gyration (rurality)",      plt_obj.plot_rg_rurality_per_period)

    _plot_one("3.2 Weekly Radius of Gyration",          plt_obj.plot_weekly_rg)
    _plot_one("3.2 Weekly RG (rurality)",               plt_obj.plot_rg_rurality_weekly)
    _plot_one("3.2 Weekly RG (party)",                  plt_obj.plot_rg_party_weekly)

    _plot_one("3.3 k-Radius of Gyration",               plt_obj.plot_krg)

    _plot_one("3.4 Distance",                           plt_obj.plot_distance)

    _plot_one("3.5 Entropy",                            plt_obj.plot_entropy)

    # S(t) is the most RAM-intensive plot: it loads a [n_steps × n_users]
    # matrix per period.  Running it in isolation ensures all prior data
    # has already been freed before this load, and the S matrix is fully
    # released before any subsequent plot starts.
    _plot_one("3.6 Exploration Curve S(t)",             plt_obj.plot_St)

    _plot_one("3.7 Location Frequency",                 plt_obj.plot_frequency)

    _plot_one("3.8 Gonzalez Trajectory Shape",          plt_obj.plot_gonzalez)
    _plot_one("3.8 Gonzalez σ_x σ_y",                  plt_obj.plot_sigmaxy)

    print("All plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4a — S3 CONSOLIDATION  (MODE B finalisation)
# ─────────────────────────────────────────────────────────────────────────────

def section_s3_consolidate(args, store):
    """Merge per-shard S3 files into consolidated.parquet for every period."""
    print("\n" + "─" * 60)
    print("SECTION 4a · S3 CONSOLIDATION")
    print("─" * 60)

    upload_endpoint = args.s3_endpoint or S3_ENDPOINT_URL
    upload_bucket   = S3_OUTPUT_BUCKET
    upload_prefix   = S3_OUTPUT_PREFIX[args.region]

    print(f"S3 target : s3://{upload_bucket}/{upload_prefix}")
    print(f"Endpoint  : {upload_endpoint}")
    print(f"Region    : {args.region}")
    print("\nRunning final S3 consolidation for all periods …")

    for period in PERIOD_NAMES:
        print(f"\n  Period: {period!r}")
        store.consolidate_s3_shards(
            period,
            s3_bucket           = upload_bucket,
            s3_prefix           = upload_prefix,
            endpoint_url        = upload_endpoint,
            delete_shards_after = True,
            verbose             = True,
        )

    print("\nFinal consolidation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4b — S3 UPLOAD  (MODE A / MODE C)
# ─────────────────────────────────────────────────────────────────────────────

def section_s3_upload(args, store):
    """Push local parquet results to S3 (MODE A / MODE C)."""
    print("\n" + "─" * 60)
    print("SECTION 4b · S3 UPLOAD (local → S3)")
    print("─" * 60)

    upload_endpoint = args.s3_endpoint or S3_ENDPOINT_URL
    upload_bucket   = S3_OUTPUT_BUCKET
    upload_prefix   = S3_OUTPUT_PREFIX[args.region]

    periods_on_s3 = store.list_s3_computed_periods(
        s3_bucket    = upload_bucket,
        s3_prefix    = upload_prefix,
        endpoint_url = upload_endpoint,
    )
    periods_to_upload = [p for p in PERIOD_NAMES if p not in periods_on_s3]

    print(f"Periods already on S3 : {periods_on_s3 or 'none'}")
    print(f"Periods to upload     : {periods_to_upload or 'none'}")

    if not periods_to_upload:
        print("All periods already on S3. Nothing to upload.")
        return

    upload_results = store.upload_all_to_s3(
        period_names      = periods_to_upload,
        s3_bucket         = upload_bucket,
        s3_prefix         = upload_prefix,
        endpoint_url      = upload_endpoint,
        delete_after      = True,
        consolidate_first = True,
        verbose           = True,
    )

    print("\nUpload summary:")
    for period, kinds in upload_results.items():
        for kind, ok in kinds.items():
            status = "✓" if ok else "✗"
            print(f"  {status}  {period:25s}  {kind}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4c — GAP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def section_gap_analysis(args, dataset, store):
    """Produce the four methodological gap-analysis figures."""
    print("\n" + "─" * 60)
    print("SECTION 4c · GAP ANALYSIS")
    print("─" * 60)

    plt_obj = plotter(
        np_             = dataset.np_,
        period_division = dataset.period_division,
        period_names    = dataset.period_names,
        t_threshold     = dataset.t_threshold,
        region          = args.region,
        county2party    = dataset.county2party,
        df_rurality     = dataset.df_rurality,
        output_dir      = args.output_dir,
        store           = store,
    )

    print("  Gap 1: NPI event timeline")
    fig_gap1 = plt_obj.plot_gap1_npi_timeline(save=True)
    plt.close(fig_gap1)

    print("  Gap 2: Sampling bias")
    fig_gap2 = plt_obj.plot_gap2_sampling_bias(save=True)
    plt.close(fig_gap2)

    print("  Gap 3: Party / rurality conflation")
    fig_gap3 = plt_obj.plot_gap3_party_rurality(metric="radius_gyration", save=True)
    plt.close(fig_gap3)

    print("  Gap 4: Post-lockdown asymmetry")
    fig_gap4 = plt_obj.plot_gap4_post_lockdown_asymmetry(metric="radius_gyration", save=True)
    plt.close(fig_gap4)

    print("All gap-analysis plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — TRANSITION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def section_transition_matrices(args, dataset):
    """Compute presence and transition matrices on a geohash grid and upload to S3."""
    print("\n" + "─" * 60)
    print("SECTION 5 · TRANSITION MATRICES")
    print("─" * 60)

    s3_endpoint = args.s3_endpoint or S3_ENDPOINT_URL

    print(f"Building geohash grid (precision={args.geohash_precision}) for {args.region} …")
    grid = tile_counties_via_geohash(dataset.geojson, precision=args.geohash_precision)
    print(f"  Grid cells: {len(grid):,}")

    tm_pipeline = TransitionPipeline(
        dataset           = dataset,
        geohash_precision = args.geohash_precision,
        delta_time_h      = args.delta_time_h,
        endpoint_url      = s3_endpoint,
        bucket            = S3_OUTPUT_BUCKET,
        s3_prefix         = S3_TRANSITION_PREFIX[args.region],
        temp_dir          = None,     # uses /tmp/humobcov_transitions by default
        keep_local        = False,
    )

    tm_pipeline.summary()

    results = tm_pipeline.run_all_periods(force=False, verbose=True)

    print("\nTransition matrix run summary:")
    for period, kinds in results.items():
        for kind, ok in kinds.items():
            status = "✓" if ok else "✗ (temp file kept locally)"
            print(f"  {status}  {period:25s}  {kind}")




def main():
    args = parse_args()

    dataset, cfg = section_input(args)

    store = None
    if not args.skip_pipeline:
        store = section_main(args, dataset, cfg)

    # Ensure store is available for downstream sections
    if store is None:
        store = ParquetStore(
            base_dir    = DIR_MILESTONES_SERVER / args.region,
            np_         = dataset.np_,
            t_threshold = dataset.t_threshold,
        )

    if args.consolidate_s3:
        section_s3_consolidate(args, store)

    if args.upload_to_s3:
        section_s3_upload(args, store)

    if args.visualize:
        section_visualization(args, dataset, store)

    if args.gap_analysis:
        section_gap_analysis(args, dataset, store)

    if args.transition_matrices:
        section_transition_matrices(args, dataset)


if __name__ == "__main__":
    main()
