# Skill: New Project Structure (HumMobCov / src)

## Purpose

This skill documents the architecture of the **new portable pipeline** in
`src/`.  Consult it whenever extending, debugging, or connecting any part
of the refactored codebase.

---

## Project root layout

```
HumMobCov/
├── src/                    # production code
│   ├── main.ipynb          # main notebook entry-point
│   ├── main.py             # script entry-point (mirrors notebook)
│   ├── constants.py        # all paths, period defs, parameter values
│   ├── datasets.py         # DataSet_California / DataSet_Massachusets
│   ├── pipeline.py         # per-user computation orchestrators
│   ├── User.py             # individual mobility metric computation
│   ├── store.py            # columnar parquet storage layer (ParquetStore)
│   ├── plotter.py          # all statistical visualisations
│   ├── utils.py            # shared helpers (filter_, xy, t_stop, …)
│   ├── rg_fits.py          # power-law / fit utilities
│   ├── rg_histograms.py    # histogram helpers
│   ├── rg_mobility_maps.py # geo-map plotting
│   └── set_mpl.py          # matplotlib style configuration
├── data/
│   └── config/             # feature-flag JSON files per region
│       ├── config_CA.json
│       └── config_MA.json
├── census_data/            # shapefiles, density CSVs, party affiliation
│   ├── California/
│   └── Massachusets/
├── milestones_analysis/    # computed results (authoritative output root)
│   ├── CA/
│   │   ├── dataxuser/            # legacy per-user files (old format)
│   │   ├── all_scalars_period_*/ # new columnar parquet shards
│   │   ├── S_period_*/
│   │   ├── weekly_rg_period_*/
│   │   ├── gonzalez_period_*/
│   │   └── frequency_period_*/
│   └── MA/
│       ├── radius_gyration_measures_new_threshold/ (legacy)
│       ├── distance_measures_new_threshold/        (legacy)
│       ├── k_radius_gyration_measures_new_threshold/
│       ├── entropic_measures_new_threshold/
│       ├── gonzalez_new_threshold/
│       ├── st_new_threshold/
│       ├── location_frequency_new_threshold/
│       ├── home_new_threshold/
│       └── plot/
├── .github/
│   └── skills/             # documentation skills
├── pyproject.toml          # uv-managed dependencies
└── README.md
```

> `output/CA/` (old placeholder directory) has been removed.
> The canonical output root is always `milestones_analysis/`.

---

## Three periods analysed (COVID phases, year 2020)

| Period string | Dates | Phase |
|---|---|---|
| `15 jan - 15 march` | Jan 15 → Mar 15 | Pre-lockdown |
| `15 march - 15 may` | Mar 15 → May 15 | Lockdown |
| `15 may - sept` | May 15 → Sep 30 | Post-lockdown |

Defined in `constants.PERIOD_NAMES` and `constants.PERIOD_DIVISION`.

---

## Regions and parameter sets

| Region | np_ | t_threshold | Notes |
|--------|-----|------------|-------|
| CA | 20 | 1 h | fixed |
| MA | 20, 100 | 1, 8, 24 h | np_=20,t=1 is primary |

---

## Module responsibilities

### `constants.py`

Single source of truth for:
- all `Path` constants anchored to `PROJECT_ROOT`
- `PERIOD_NAMES`, `PERIOD_DIVISION`, `PERIOD_NAMES_TO_DIVISION`
- preprocessing params (`MIN_POINTS_PER_USER`, `TIME_THRESHOLD_HOURS`)
- metric name lists (`ALL_SCALAR_METRICS`, `GONZALEZ_COLUMNS`, etc.)
- file name templates (`FNAME_SCALARS`, `FNAME_GONZALEZ`, etc.)
- `MA_LEGACY_METRIC_DIRS` — mapping of MA legacy folder names
- `get_legacy_metric_dir(region, metric, np_, t)` — returns path to legacy folder
- S3 settings (`S3_ENDPOINT_URL`, `S3_BUCKET`, `S3_RAW_PREFIX`)

Override the output root via env variable:
```
MILESTONES_DIR=/mnt/server/milestones_analysis
```

### `datasets.py`

**`_BaseDataset`** — abstract base with `_init_common()`:
- loads census CSV → `county2rural`, `county2party`
- loads geojson shapefile
- sets `period_names`, `period_division`, `np_`, `t_threshold`
- sets `dir_output = milestones_analysis/{region}/dataxuser/`

**`DataSet_California()`** — CA-specific file list (all parquets in `DIR_RAW_DATA_CA`).

**`DataSet_Massachusets()`** — MA-specific fixed list of 15 named parquets.

**`dataset_info(file, ...)`** — wraps one parquet shard:
- `preprocess()` → `period2df`, `period2listusers`
- `spatial_filtering_per_country()` — clips to US bounding box

### `User.py`

Identical algorithm to the old `User` class.  Key differences:
- No hard-coded save paths — `base_dir` is passed in or derived from
  `constants.DIR_MILESTONES_SERVER`.
- Metric results are **returned** (or written to the `ParquetStore`)
  rather than always saved to individual files.

Metric methods (all preserve original algorithms):
`compute_radius_of_gyration`, `compute_random_entropy`,
`compute_uncorrelated_entropy`, `compute_real_entropy`,
`compute_home`, `compute_krg`, `compute_straight_line_distance`,
`compute_fraction_time_user_is_present`, `_get_county`,
`compute_gonzalez`, `compute_St`, `compute_frequency_location`,
`compute_weekly_radius_gyration`

### `store.py` — `ParquetStore`

Columnar parquet storage, replacing per-user `.csv.gz` files.

**Storage layout per `(region, kind, period, np_, t)`:**

```
milestones_analysis/{REGION}/
    {kind}_period_{period}_np_{np_}_t_{t}/
        shard_<timestamp>.parquet    ← write-once during computation
        consolidated.parquet         ← merged after consolidation
```

**Fixed-length kinds** (`all_scalars`, `S`, `weekly_rg`):
- Users as **columns**; index column = `metric` / `time` / `week`
- Resume check: `pl.read_parquet_schema()` → column names = computed users

**Variable-length kinds** (`gonzalez`, `frequency`):
- Long format with `user_id` column
- Resume check: read only `user_id` column

**Key methods:**

| method | purpose |
|--------|---------|
| `write_batch(kind, period, data)` | append shard parquet for a batch of users |
| `get_computed_users(kind, period)` | O(1) resume check via footer metadata |
| `consolidate(kind, period)` | merge all shards into `consolidated.parquet` |
| `read_scalars(period)` | return wide DataFrame `[users × metrics]` for plotting |
| `read_st_matrix(period)` | return `[time × users]` matrix |
| `read_weekly_rg_matrix(period)` | return `[weeks × users]` matrix |
| `migrate_from_legacy(kind, period, np_, t)` | migrate old per-user files → parquet store |
| `migrate_all_periods(np_, t)` | migrate all periods for a region |

### `pipeline.py`

**`get_config(region, config_dir)`** — loads `data/config/config_{region}.json`.

**`compute_all(cfg, dataset, list_users, period, df, output_dir, store, batch_size)`**:
1. Filters users with `< np_` stop-points.
2. Skips users already in the store (parquet footer check).
3. Creates a `User` per user, calls requested metric methods.
4. Accumulates in-memory batch dicts.
5. Flushes to `store` every `batch_size` (default 500) users.

**`analyze_from_dataset(dataset, name, store)`** — processes local raw parquets.

**`analyze_from_s3_progressive(dataset, name, store, s3_client)`**:
- Downloads one shard at a time from S3.
- Checkpoint file `shard_checkpoint_{region}.json` tracks completed shards.
- Deletes local shard copy after processing.
- Fully resume-safe.

### `utils.py`

Fast implementations using **Numba JIT** (falls back to NumPy):
- `filter_(x_arr, t_threshold)` — time-gap filter
- `xy(lat, lon, lat0, lon0)` — tangent-plane projection
- `t_stop(df)` — stop duration in minutes
- `get_already_saved_user_per_period(directory)` — threaded directory scan
- `ifnotexistsmkdir(dir_)` — mkdir helper

### `plotter.py`

Reads results from `ParquetStore` (not individual files).  Plot methods
match the old pipeline's capabilities plus enhancements:
`plot_rg`, `plot_rg_party`, `plot_rg_party_per_period`,
`plot_rg_urban_rural`, `plot_rg_rurality_per_period`,
`plot_rg_county`, `plot_weekly_rg`, `plot_distance`, `plot_entropy`,
`plot_gonzalez`, `plot_sigmaxy`, `plot_conditional_gonzalez`,
`plot_St`, `plot_frequency`, `plot_krg`

---

## Configuration flags (`data/config/config_*.json`)

```jsonc
{
  "raw_trajectories":            false,   // true = process raw parquets
  "is_gonzalez":                 true,
  "already_computed_gonzalez":   false,
  "is_St":                       true,
  "already_computed_St":         false,
  "is_frequency":                true,
  "already_computed_frequency":  false,
  "is_radius_gyration":          true,
  "already_computed_rg":         false,
  "is_random_entropy":           true,
  "is_uncorrelated_entropy":     true,
  "is_real_entropy":             true,
  "is_distance":                 true,
  "is_home":                     true,
  "is_krg":                      true,
  "is_fraction_time":            true,
  "is_county_rural":             true,
  "is_weekly_radius_gyration":   true
}
```

---

## Three execution modes (selected automatically)

| Situation | Action |
|-----------|--------|
| Raw Cuebiq parquet files accessible | `analyze_from_dataset()` or `analyze_from_s3_progressive()` |
| Only legacy per-user CSV.gz files exist | `store.migrate_all_periods()` one-time migration |
| Parquet store already populated | skip to visualisation |

---

## Desired future output format (per README)

Results should ultimately live in:

```
milestones_analysis/{REGION}/
    all_scalars_period_{period}_np_{np_}_t_{t}/
        shard_*.parquet         # write-once during computation
        consolidated.parquet    # merged result
    S_period_{period}_np_{np_}_t_{t}/
    weekly_rg_period_{period}_np_{np_}_t_{t}/
    gonzalez_period_{period}_np_{np_}_t_{t}/
    frequency_period_{period}_np_{np_}_t_{t}/
```

A single consolidated parquet contains **one row per metric, one column
per user** (fixed-length kinds) or long format with `user_id` column
(variable-length kinds).  This replaces both the CA flat `dataxuser/` and
the MA per-metric subdirectory layouts.

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MILESTONES_DIR` | `<project>/milestones_analysis` | Override output root |
| `SHARD_TEMP_DIR` | `<project>/.shard_tmp` | Temp dir for S3 downloads |
| `S3_ENDPOINT_URL` | `https://s3.atlas.fbk.eu` | S3-compatible endpoint |
| `S3_BUCKET` | `chub-datalake` | S3 bucket name |
| `S3_PREFIX_CA` | `shared/cuebiq/MOBS/urban_rural_flow_stops_cali_urban_rural_v3` | CA raw shard prefix |
| `S3_PREFIX_MA` | `shared/cuebiq/MOBS/20220330_stops_hq_users_MA` | MA raw shard prefix |

---

## Installation

```bash
# Requires Python 3.10, managed with uv
uv sync            # core deps
uv sync --group dev   # + JupyterLab, ipykernel, nbdime
source .venv/bin/activate
```

---

## Key design decisions

1. **No re-computation of already-stored users** — the parquet footer
   provides the user list with zero row reads.
2. **Portable paths** — nothing is hard-coded; all paths derive from
   `PROJECT_ROOT` or overridable env vars.
3. **Batch writes** — accumulate 500 users in RAM before writing to avoid
   millions of tiny I/O operations.
4. **One-time migration** — `migrate_from_legacy()` converts old per-user
   files to the new store; after migration the legacy files can be archived.
5. **MA scales np_** — MA analysis is run for multiple `np_` values
   (sensitivity analysis); CA uses only `np_=20`.
