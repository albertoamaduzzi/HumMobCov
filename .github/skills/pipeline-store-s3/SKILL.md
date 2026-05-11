# Skill: Pipeline, ParquetStore & S3 Architecture (HumMobCov)

## Purpose

This skill documents the **end-to-end pipeline and storage layer** as it
currently stands after the vectorized refactor and S3-progressive mode.
Consult it whenever debugging, extending, or resuming the pipeline in
`src/pipeline.py`, `src/store.py`, `src/vectorized_pipeline.py`.

---

## Execution modes

| Mode | Trigger | Entry point | Output target |
|------|---------|-------------|---------------|
| **A — local raw** | `raw_trajectories=true` in config AND shards on disk | `analyze_from_dataset()` | local ParquetStore, then `store.upload_all_to_s3()` |
| **B — S3 progressive** | `raw_trajectories=true` AND shards NOT on disk | `analyze_from_s3_progressive()` | per-shard upload via `store.upload_shard_to_s3_unique()` |
| **C-CA — legacy** | `raw_trajectories=false` AND `REGION=="CA"` | `store.migrate_all_periods()` | local ParquetStore |
| **C-MA — legacy** | `raw_trajectories=false` AND `REGION=="MA"` | `store.migrate_all_periods_MA()` | local ParquetStore |

**MODE B is the primary path for CA (54 raw shards on S3).**

---

## MODE B — S3 Progressive detailed flow

```
for each unprocessed shard in S3:
    1. Download shard → DIR_SHARD_TEMP
    2. try:
           preprocess (polars / legacy)
           compute all metrics → write to local ParquetStore (shard_*.parquet files)
           delete temp local file
           mark shard complete in checkpoint JSON
       except Exception:
           print "WARNING: preprocessing error — skipping shard"
           delete temp local file
           continue

    3. OUTSIDE the preprocessing try/except:
       for each period:
           try:
               store.upload_shard_to_s3_unique(period, shard_label, …)
               # this consolidates local shard_*.parquet → consolidated.parquet,
               # uploads as shards/{shard_label}.parquet on S3, deletes local
           except Exception:
               print traceback + WARNING (local results preserved)

4. FINAL STEP (after all shards):
   for each period:
       store.consolidate_s3_shards(period, …)
       # downloads all per-shard S3 files, merges into consolidated.parquet,
       # uploads consolidated.parquet, deletes per-shard S3 files
```

### Checkpoint file

`{base_dir}/shard_checkpoint_np_{np_}_t_{t}.json` — maps shard S3 key → `true`.
Shards in this file are always skipped regardless of store contents.

---

## ParquetStore architecture (`src/store.py`)

### Kinds and their merge strategy

```python
FIXED_LENGTH_KINDS = frozenset({"all_scalars", "weekly_rg", "S"})
LONG_FORMAT_KINDS  = frozenset({"gonzalez", "frequency"})
```

| Kind | Row semantics | Merge strategy |
|------|--------------|----------------|
| `all_scalars` | rows = metric names, cols = user_ids | `_hconcat_fixed(frames, "metric")` |
| `weekly_rg` | rows = week keys, cols = user_ids | `_hconcat_fixed(frames, "week")` |
| `S` | rows = time steps 0..1418, cols = user_ids | `_hconcat_fixed(frames, "time")` |
| `gonzalez` | rows = one row per user per trajectory point | `pl.concat(frames, how="vertical_relaxed")` |
| `frequency` | rows = one row per user per location rank | `pl.concat(frames, how="vertical_relaxed")` |

**IMPORTANT**: `S` has a fixed number of rows (1419 time steps) regardless of
users — two shards with different user sets CAN differ in row length only if
one is corrupt or a user has no data. `_hconcat_fixed` handles this by
aligning on the index column.

### Directory layout (local)

```
{base_dir}/{REGION}/
└── {kind}_period_{period_safe}_np_{np_}_t_{t}/
    ├── shard_{monotonic_ns}.parquet   ← per-batch write-once files
    └── consolidated.parquet           ← merged result
```

`period_safe` = period name with spaces/dashes → underscores.

### S3 output layout

```
{s3_prefix}/{kind_dir}/
    shards/
        {shard_label}.parquet          ← one per raw shard, uploaded by MODE B
    consolidated.parquet               ← final merged result (after consolidate_s3_shards)
```

`shard_label` = MD5 of the raw S3 shard key (8 hex chars).

---

## Key methods

### `store.consolidate(period, kind) → Path`

Merges all local `shard_*.parquet` into `consolidated.parquet`.
- Each `pl.read_parquet` is wrapped in try/except → corrupt shards are
  deleted and skipped (not propagated).
- Returns the consolidated path even if no new shards exist (existing
  `consolidated.parquet` is reused).
- Returns a path that `.exists()` is False if there is nothing at all.

### `store.upload_shard_to_s3_unique(period, shard_label, …)`

Iterates `FIXED_LENGTH_KINDS + LONG_FORMAT_KINDS`.
Each kind is wrapped in its own `try/except` → one failing kind does **not**
abort the remaining kinds. Prints a full traceback on failure.

### `store.consolidate_s3_shards(period, …)`

Called once at the end (final step 4). Downloads all per-shard S3 files,
merges them locally, uploads `consolidated.parquet`, deletes per-shard S3 files.
Handles corrupt/missing downloads with warnings (not crashes).

---

## Vectorized pipeline (`src/vectorized_pipeline.py`)

All metric functions are **serial** — NO ProcessPoolExecutor, no n_workers.
Polars' internal thread pool handles CPU parallelism automatically.

| Function | Output schema |
|----------|---------------|
| `_compute_radius_of_gyration_polars` | scalar → `all_scalars` |
| `_compute_distance_polars` | scalar → `all_scalars` |
| `_compute_entropies_polars` | scalars → `all_scalars` |
| `_compute_krg_polars` | scalars → `all_scalars` |
| `_compute_gonzalez_polars` | `x_norm, y_norm, sigmax, sigmay` per user row → `gonzalez` |
| `_compute_frequency_polars` | `frequency, rank, geohash7, geohash6` per user row → `frequency` |
| `_compute_st_polars` | 1419-element vector per user → `S` |
| `_compute_real_entropy_polars` | scalar → `all_scalars` |
| `_gonzalez_pca` | returns `(x_norm, y_norm, sigma_x, sigma_y)` — x_norm/y_norm are numpy arrays, sigma_x/sigma_y are float scalars; pandas broadcasts scalars correctly → all columns are float64 |

---

## Bugs fixed during this session (May 2026)

### 1. `n_workers` / ProcessPoolExecutor removal
`analyze_from_s3_progressive()` and `compute_all_polars()` no longer accept
or use `n_workers`. Removed from:
- `src/vectorized_pipeline.py`
- `src/pipeline.py`
- `src/main.py`
- `src/main.ipynb` (cell `MAIN`)

### 2. Exception scope bug in `analyze_from_s3_progressive`
Upload/consolidation code was **inside** the preprocessing `try/except`.
Upload failures were silently printed as "WARNING: preprocessing error" —
data was written to local store but never sent to S3.

**Fix**: preprocessing try/except exits early on error; upload/consolidation
runs OUTSIDE in its own per-period try/except block.

### 3. Corrupt shard crash in `consolidate()`
`pl.read_parquet(f)` without try/except crashed when a leftover empty/truncated
shard file was present (`parquet: File out of specification: The file must end with PAR1`).

**Fix**: each `pl.read_parquet` in `consolidate()` wrapped in try/except;
corrupt files are deleted and skipped.

### 4. One failing kind aborting all kinds in `upload_shard_to_s3_unique`
If `gonzalez` consolidation threw, the exception propagated out and aborted
`frequency` upload too.

**Fix**: each kind's processing wrapped in its own `try/except` inside
`upload_shard_to_s3_unique`; full traceback printed on failure.

### 5. Missing tracebacks in WARNING handlers
`pipeline.py` WARNING handlers only printed the exception message, not the
full traceback, making it impossible to identify the failing line.

**Fix**: both WARNING handlers in `pipeline.py` now call `traceback.format_exc()`.

### 6. "schema lengths differ" in `pl.concat` for `frequency` / `gonzalez`
**Symptom**: `consolidate()` raised `schema lengths differ` when concatenating
`frequency` shard files. The per-kind traceback (fix #4) revealed the exact line:
`pl.concat(frames, how="vertical_relaxed")` in `consolidate()`.

**Root cause**: `vertical_relaxed` requires the same column count across all frames
(it only allows type casting). The existing `consolidated.parquet` (written by an
earlier run with an older code version) had a **different number of columns** than the
new shard files — causing "schema lengths differ" at the Rust level.

**Fix** (store.py): changed all three `vertical_relaxed` occurrences used for
LONG_FORMAT_KINDS to `diagonal_relaxed`, which fills missing columns with `null`
instead of failing. Affected call sites:
1. `consolidate()` — merges local shards into consolidated.parquet
2. `read_long_format()` — reads shards for plotter use
3. `consolidate_s3_shards()` — final S3 merge step

**Note**: `diagonal_relaxed` is the correct mode for heterogeneous-schema long-format
concat in Polars. `vertical_relaxed` only handles type mismatches, not column count
differences.

**Old migration data contamination** (CA-specific): The `dataxuser/` migration wrote
frequency data with only [`frequency`, `rank`, `user_id`] (3 cols, no geohash). When
merged with new 5-col vectorized pipeline data via `diagonal_relaxed`, null rows
appear for `geohash7`/`geohash6`. These rows must be filtered out.
- Detection: `df.filter(pl.col("geohash7").is_not_null())`
- `consolidate_s3_shards` now applies this filter automatically after the merge
- One-time cleanup was applied manually to the two contaminated S3 shard files
  (`c734c7192436.parquet` for jan-march and march-may frequency)

---

## S3 / environment config

| Constant | Value |
|----------|-------|
| `S3_ENDPOINT_URL` | `https://s3.atlas.fbk.eu:443` |
| `S3_BUCKET` | `chub-datalake` |
| `S3_RAW_PREFIX["CA"]` | `shared/cuebiq/MOBS/urban_rural_flow_stops_cali_urban_rural_v3` |
| `S3_OUTPUT_BUCKET` | `chub-datalake` |
| `S3_OUTPUT_PREFIX["CA"]` | `final_pipeline/CA` |
| `DIR_SHARD_TEMP` | `HumMobCov/.shard_tmp` |

Raw CA shards: **54 files** on S3. All use `analyze_from_s3_progressive`.

---

## `main.ipynb` / `main.py` — entry-point conventions

- `n_workers` parameter: **removed everywhere** — do not add it back
- `USE_VECTORIZED = True` is the default (polars path); legacy skmob path is
  kept for reference only
- `ParquetStore` base dir: `DIR_MILESTONES_SERVER / REGION`
- Both `analyze_from_dataset` and `analyze_from_s3_progressive` calls omit `n_workers`
