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
│   ├── vectorized_pipeline.py        # polars/numba vectorized replacement
│   ├── User.py             # individual mobility metric computation (legacy)
│   ├── store.py            # columnar parquet storage layer (ParquetStore)
│   ├── plotter.py          # all statistical visualisations
│   ├── utils.py            # shared helpers (filter_, xy, t_stop, …)
│   ├── tile_counties_via_geohash.py  # geohash grid tiling for counties
│   ├── transition_matrices/          # transition & presence matrix pipeline
│   │   ├── __init__.py
│   │   └── transition_pipeline.py   # TransitionPipeline class
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
├── milestones_analysis/    # LOCAL computed results (transient / legacy)
│   ├── CA/
│   └── MA/
├── .github/
│   └── skills/             # documentation skills
│       ├── new_project_structure/SKILL.md  ← this file
│       ├── parallelization/SKILL.md        ← numba/polars patterns
│       ├── data-handling/SKILL.md
│       ├── old_structurefile_system/SKILL.md
│       ├── structure_old_process/SKILL.md
│       └── old_output_structure/SKILL.md
├── pyproject.toml          # uv-managed dependencies
└── README.md
```

> **Authoritative output location**: `chub-datalake/shared/cuebiq/MOBS/final_pipeline/`
> on the S3-compatible store at `https://s3.atlas.fbk.eu`.
> Local `milestones_analysis/` is a legacy / transitional cache only.

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
| `upload_file_to_s3(local_path, bucket, key, endpoint_url)` | upload single file to S3 |
| `upload_period_to_s3(period, bucket, prefix, endpoint_url)` | upload all kinds for a period |
| `upload_all_to_s3(period_names, bucket, prefix, endpoint_url)` | upload all periods |
| `list_s3_computed_periods(bucket, prefix, endpoint_url)` | check what is already on S3 |

**Default upload behaviour** — controlled by `S3_UPLOAD_DEFAULT` env var (default `"1"` = upload).
After consolidation, call `upload_all_to_s3()` to push data to `final_pipeline/`.

### `vectorized_pipeline.py`

Drop-in vectorized replacement for the per-user `User`-object loop.  Eliminates
all skmob overhead and scales to 100k+ users without Python per-user objects.

**Public API:**

```python
from src.vectorized_pipeline import preprocess_shard_polars, compute_all_polars
# or simply:
from src import preprocess_shard_polars, compute_all_polars
```

**`preprocess_shard_polars(file, dataset) -> dict[str, pl.DataFrame]`**
- Replaces `dataset_info.__init__()` + `preprocess()`.
- Uses `pl.scan_parquet()` with predicate-pushdown bbox filter (no Shapely per row).
- Applies numba-backed temporal filter via `map_groups`.
- Returns `{period_name: pl.DataFrame}` with columns
  `[userId, lat, lon, begin, end, geohash7, dur_min]`.

**`compute_all_polars(cfg, dataset, period_df, period, already_done_scalars,
already_done_gonzalez, already_done_st, already_done_freq, already_done_wrg,
store, batch_size=5000)`**
- Replaces `compute_all()` (legacy `User`-loop path).
- All scalar metrics computed in a **single** `group_by().agg()` pass.
- County / rurality assigned via GeoPandas `sjoin` (vectorized O(n log k)).
- S(t) and Gonzalez PCA via `map_groups` (no skmob).
- Writes results to `ParquetStore` in chunks of `batch_size`.

**Metrics implemented (vectorized, no skmob):**

| Metric | Approach |
|--------|----------|
| `radius_of_gyration` | Polars 2-pass join (center-of-mass → projection → weighted RG) |
| `random_entropy` | `log2(n_distinct)` via group_by agg |
| `uncorrelated_entropy` | `-Σ p·log2(p)` via group_by agg |
| `real_entropy` | LZ78 in pure Python via `map_groups` |
| `distance` | Haversine via polars `shift().over()` expressions |
| `home_location` | Most-visited geohash7 by `sum(dur_min)` |
| `k_radius_of_gyration` | Top-k locations by visit count + RG on subset |
| `q` (fraction time) | `sum(dur_min) / period_duration` via group_by |
| `S(t)` | `map_groups` + compiled per-user time-step loop |
| `Gonzalez PCA` | NumPy PCA on projected coordinates via `map_groups` |
| `frequency/rank` | Polars group_by count + sort |
| County assignment | GeoPandas `sjoin` (replaces Shapely per-row loop) |

**Bugs fixed vs `User.py`:**
- `radius_of_gyration`: old code used `xy()` on the full DataFrame inside a
  per-geohash loop (so `x[0]` always referred to the first row of the *full* frame).
- `Gonzalez`: old code had lat/lon variable labels swapped for the reference point
  (`mean_lon = np.array(self.df.lat).mean()`).

---

### `tile_counties_via_geohash.py`

Tiles a county shapefile / GeoDataFrame with a geohash grid.

```python
from src.tile_counties_via_geohash import tile_counties_via_geohash, coarsen_geohash_series

grid = tile_counties_via_geohash(county_gdf, precision=5)
# → GeoDataFrame with columns: geohash, geometry (Polygon, EPSG:4326), area_km2
```

Geohash is hierarchical — coarsen trajectory geohashes with:

```python
df["geohash5"] = coarsen_geohash_series(df["geohash7"], precision=5)
# or in Polars: pl.col("geohash7").str.slice(0, 5)
```

Recommended precision for 32 GB RAM: **5** (CA: ~17 000 cells, MA: ~4 500 cells).

### `transition_matrices/transition_pipeline.py` — `TransitionPipeline`

End-to-end pipeline from raw trajectories to transition / presence matrices.
Results go directly to S3 (`final_pipeline/{region}/transition_matrices/`).

```python
from src.transition_matrices import TransitionPipeline
pipeline = TransitionPipeline(dataset, geohash_precision=5, delta_time_h=1)
pipeline.run_all_periods()
```

Output tables:

* **presence_matrix** — `(geohash, time_int, datetime, count_birth, count_death, count_transit, count, probability)`
* **transition_matrix** — `(geohash_start, geohash_end, time_int, datetime, transitions, transition_probability)`

Cache index (`cache_index.json`) lives on S3 alongside the output files.
Resume is O(1) — the cache is downloaded once per run.

### `pipeline.py`

**`get_config(region, config_dir)`** — loads `data/config/config_{region}.json`.

**`compute_all(cfg, dataset, list_users, period, df, output_dir, store, batch_size, n_workers=1, use_vectorized=False)`**:
- When `use_vectorized=True`: converts `df` pandas → polars and delegates
  immediately to `compute_all_polars()` (no `User` objects created).
- Otherwise (legacy path):
  1. Filters users with `< np_` stop-points.
  2. Skips users already in the store (parquet footer check).
  3. Pre-groups DataFrame into `{uid: sub_df}` dict to avoid repeated pandas groupby.
  4. Creates a `User` per user (serial when `n_workers=1`, ProcessPoolExecutor otherwise).
  5. Flushes to `store` every `batch_size` (default 500) users.

**`analyze_from_dataset(dataset, name, store, n_workers=1, use_vectorized=False)`**
— processes local raw parquets.

**`analyze_from_s3_progressive(dataset, name, store, ..., n_workers=1, use_vectorized=False, output_bucket=None, output_s3_prefix=None, ...)`**:
- Downloads one shard at a time from S3 to a temp dir.
- When `use_vectorized=True`: calls `preprocess_shard_polars()` + `compute_all_polars()` — pandas/skmob never touched.
- After each successful shard: consolidates local output, uploads to
  `{output_s3_prefix}/{kind_dir}/shards/{shard_label}.parquet` on S3,
  deletes local output files → **no output accumulates on local disk**.
- After all shards: calls `store.consolidate_s3_shards()` to download
  per-shard S3 files, merge, write final `consolidated.parquet` on S3.
- Checkpoint file `shard_checkpoint_np_{np_}_t_{t}.json` tracks completed shards.
- Deletes local raw shard copy after processing.
- Fully resume-safe at both shard level and within-shard user level.

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
| Raw Cuebiq parquet files accessible (fast, recommended) | `analyze_from_s3_progressive(..., use_vectorized=True)` |
| Raw Cuebiq parquet files accessible (legacy skmob path) | `analyze_from_dataset()` or `analyze_from_s3_progressive()` |
| Only legacy per-user CSV.gz files exist | `store.migrate_all_periods()` one-time migration |
| Parquet store already populated | skip to visualisation |

---

## Completion and resume semantics (CRITICAL)

The pipeline must be able to answer at any time: **"which computations are done and which are not?"**
The answer depends on the execution mode — do NOT conflate them.

### Mode A/B — raw trajectories (`raw_trajectories=true`)

**Unit of work = raw input SHARD** (one `.parquet` file from S3 or local disk).

**Completion signal = shard checkpoint file**, NOT store contents:
```
milestones_analysis/{REGION}/shard_checkpoint_np_{np_}_t_{t}.json
{"completed": ["shared/cuebiq/.../shard_001.parquet", ...]}
```

A period is only fully done when **every** S3 key under the region prefix
appears in `completed`.  Checking store contents alone is WRONG — after
processing 1-of-N shards, all metric kinds (scalars, gonzalez, S(t), freq)
will be internally consistent with each other but represent only a fraction
of users.

**Resume rule in `main.py` / `main.ipynb`:**
```python
use_shard_resume = bool(cfg.get("raw_trajectories"))
if use_shard_resume:
    periods_done = []          # always enter the pipeline
    periods_todo = list(PERIOD_NAMES)
```
The inner functions `analyze_from_s3_progressive` and `analyze_from_dataset`
manage their own shard-level and user-level resume — they skip already-done
shards/users automatically.

### Mode C — legacy migration (`raw_trajectories=false`)

**Unit of work = individual USER** (read from legacy CSV.gz / JSON files).

**Completion signal = store contents**: a period is "done" if
`store.get_computed_users(period, "all_scalars")` is non-empty.
Migration methods (`migrate_all_periods`, `migrate_all_periods_MA`) skip
already-migrated users by reading parquet footer metadata (O(1)).

**Resume rule:**
```python
periods_done = [p for p in PERIOD_NAMES
                if len(store.get_computed_users(p, "all_scalars")) > 0]
periods_todo = [p for p in PERIOD_NAMES if p not in periods_done]
```

### Progress inspection at any time

```bash
# How many shards completed vs total (S3 mode)
python3 -c "
import json
with open('milestones_analysis/CA/shard_checkpoint_np_20_t_1.json') as f:
    d = json.load(f)
print('done shards:', len(d['completed']))
"

# How many users per kind per period (all modes)
python3 -c "
import sys; sys.path.insert(0, '.')
from src.store import ParquetStore
from src.constants import DIR_MILESTONES_SERVER, PERIOD_NAMES
store = ParquetStore(DIR_MILESTONES_SERVER / 'CA', np_=20, t_threshold=1)
for p in PERIOD_NAMES:
    sc = len(store.get_computed_users(p, 'all_scalars'))
    go = len(store.get_computed_users_long(p, 'gonzalez'))
    st = len(store.get_computed_users(p, 'S'))
    fr = len(store.get_computed_users_long(p, 'frequency'))
    print(f'{p}: scalars={sc}, gonzalez={go}, S(t)={st}, freq={fr}')
"
```

**Recommended call for production:**

```python
analyze_from_s3_progressive(
    dataset, "CA", cfg, store,
    endpoint_url=S3_ENDPOINT_URL, bucket=S3_BUCKET, s3_prefix=S3_RAW_PREFIX["CA"],
    batch_size=500,
    n_workers=os.cpu_count(),
    use_vectorized=True,           # polars+numba; skips skmob entirely
    output_bucket=S3_OUTPUT_BUCKET,
    output_s3_prefix=S3_OUTPUT_PREFIX["CA"],
    delete_local_after_upload=True,  # free local disk immediately
)
```

### Output storage architecture (CRITICAL)

Raw input data and computed output data are kept on S3.
**Local disk is only a temporary buffer.**

```
S3 input (raw trajectories):
  s3://chub-datalake/shared/cuebiq/MOBS/urban_rural_flow.../shard_001.parquet
     ↓ download → process → delete
Local temp (~temp_dir):
  only one shard at a time while being computed
     ↓ write output locally during computation
Local store (~milestones_analysis/):
  shard_TIMESTAMP.parquet  (one shard's worth of output, one metric at a time)
     ↓ consolidate + upload + delete after each shard completes
S3 output (per-shard):
  s3://chub-datalake/shared/cuebiq/MOBS/final_pipeline/CA/
    {kind}_period_{period}_np_{np_}_t_{t}/shards/{shard_label}.parquet
     ↓ consolidate_s3_shards() at the end (downloads, merges, re-uploads)
S3 output (final):
  s3://chub-datalake/shared/cuebiq/MOBS/final_pipeline/CA/
    {kind}_period_{period}_np_{np_}_t_{t}/consolidated.parquet
```

The `get_computed_users` / `get_computed_users_long` check reads the **local**
parquet footer only.  This is correct because:
- Completed shards have their local output already deleted → returns {} → correct
  (we skip those shards entirely via the checkpoint, so we never call this for them)
- The currently-processing shard may have partial local output (from an interrupted
  previous run) → returns those users → correctly skips them within the shard

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
| `MILESTONES_DIR` | `<project>/milestones_analysis` | Override local output root (legacy / transitional) |
| `SHARD_TEMP_DIR` | `<project>/.shard_tmp` | Temp dir for S3 downloads |
| `S3_ENDPOINT_URL` | `https://s3.atlas.fbk.eu` | S3-compatible endpoint |
| `S3_BUCKET` | `chub-datalake` | S3 bucket for raw input data |
| `S3_PREFIX_CA` | `shared/cuebiq/MOBS/urban_rural_flow_stops_cali_urban_rural_v3` | CA raw shard prefix |
| `S3_PREFIX_MA` | `shared/cuebiq/MOBS/20220330_stops_hq_users_MA` | MA raw shard prefix |
| `S3_OUTPUT_BUCKET` | `chub-datalake` | Bucket for final_pipeline output |
| `S3_OUTPUT_PREFIX_CA` | `shared/cuebiq/MOBS/final_pipeline/CA` | CA output prefix |
| `S3_OUTPUT_PREFIX_MA` | `shared/cuebiq/MOBS/final_pipeline/MA` | MA output prefix |
| `S3_UPLOAD_DEFAULT` | `1` | Set to `0` to keep data local only |

---

## S3 output layout (final_pipeline)

All computed results from the **new** pipeline live under:

```
chub-datalake/
  shared/cuebiq/MOBS/
    final_pipeline/               ← clean, versioned output root
      CA/
        all_scalars_period_*/consolidated.parquet
        S_period_*/consolidated.parquet
        weekly_rg_period_*/consolidated.parquet
        gonzalez_period_*/consolidated.parquet
        frequency_period_*/consolidated.parquet
        transition_matrices/
          presence_prec5_dh1.0_15_jan_-_15_march.parquet
          transition_prec5_dh1.0_15_jan_-_15_march.parquet
          cache_index.json          ← resume index
      MA/
        …same structure…
```

Raw input data stays in:

```
    urban_rural_flow_stops_cali_urban_rural_v3/  ← CA raw shards (read-only)
    20220330_stops_hq_users_MA/                  ← MA raw shards (read-only)
```

**Never write into the raw-data prefixes.**

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
6. **Vectorized path is the preferred production path** — `use_vectorized=True`
   eliminates all per-user Python object overhead and skmob calls.  The legacy
   `User`-based path is kept for reproducibility and regression testing only.
7. **Bug fixes in vectorized path** — RG and Gonzalez metrics have corrected
   implementations; results will differ slightly from the legacy path.
