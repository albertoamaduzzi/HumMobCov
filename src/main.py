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
import sys
from pathlib import Path

# ─── Make the src package importable regardless of working directory ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # HumMobCov/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
)
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
        "--batch-size", type=int, default=500,
        help="Users per parquet shard write (S3 / local raw modes).",
    )
    parser.add_argument(
        "--consolidate-every", type=int, default=1,
        help="Consolidate store every N shards in S3 mode.",
    )
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Skip section 2 (pipeline / migration) and go straight to visualisation.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Run section 3 (visualisation) after the pipeline.",
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
    periods_done = [
        p for p in PERIOD_NAMES
        if len(store.get_computed_users(p, "all_scalars")) > 0
    ]
    periods_todo = [p for p in PERIOD_NAMES if p not in periods_done]

    print(f"Periods already in store : {periods_done or 'none'}")
    print(f"Periods still to compute : {periods_todo or 'none'}")

    if not periods_todo:
        print("Parquet store already populated for all periods. Nothing to do.")

    else:
        # ── CASE A: local raw parquet files ───────────────────────────────────
        local_raw_files = [f for f in dataset.dir_files if Path(f).exists()]

        if cfg.get("raw_trajectories") and local_raw_files:
            print(f"\nMODE A — local raw data ({len(local_raw_files)} shards found).")
            analyze_from_dataset(
                dataset,
                region     = args.region,
                config_dir = args.config_dir,
                output_dir = args.output_dir,
                store      = store,
                batch_size = args.batch_size,
            )
            print("Computation complete. Consolidating shards…")
            for p in dataset.period_names:
                store.consolidate_all(p)
            print("Done.")

        # ── CASE B: S3 progressive (download one shard → compute → delete) ───
        elif cfg.get("raw_trajectories"):
            print(
                f"\nMODE B — S3 progressive download.\n"
                f"  Endpoint : {s3_endpoint}\n"
                f"  Bucket   : {s3_bucket}\n"
                f"  Prefix   : {s3_prefix}\n"
                f"  Temp dir : {temp_dir}\n"
                f"  Periods to compute: {periods_todo}"
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
                consolidate_every = args.consolidate_every,
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
                    batch_size   = 5000,
                    consolidate  = True,
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
                batch_size     = 2000,
                consolidate    = True,
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

    print("\nGenerating plots…")

    print("  3.1 Radius of Gyration")
    plt_obj.plot_rg()
    plt_obj.plot_rg_party_per_period()
    plt_obj.plot_rg_rurality_per_period()

    print("  3.2 Weekly Radius of Gyration")
    plt_obj.plot_weekly_rg()
    plt_obj.plot_rg_rurality_weekly()
    plt_obj.plot_rg_party_weekly()

    print("  3.3 k-Radius of Gyration")
    plt_obj.plot_krg()

    print("  3.4 Distance")
    plt_obj.plot_distance()

    print("  3.5 Entropy")
    plt_obj.plot_entropy()

    print("  3.6 Exploration Curve S(t)")
    plt_obj.plot_St()

    print("  3.7 Location Frequency")
    plt_obj.plot_frequency()

    print("  3.8 Gonzalez Trajectory Shape")
    plt_obj.plot_gonzalez()
    plt_obj.plot_sigmaxy()

    print("All plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    dataset, cfg = section_input(args)

    store = None
    if not args.skip_pipeline:
        store = section_main(args, dataset, cfg)

    if args.visualize:
        if store is None:
            # Build a store object even when the pipeline was skipped
            store = ParquetStore(
                base_dir    = DIR_MILESTONES_SERVER / args.region,
                np_         = dataset.np_,
                t_threshold = dataset.t_threshold,
            )
        section_visualization(args, dataset, store)


if __name__ == "__main__":
    main()
