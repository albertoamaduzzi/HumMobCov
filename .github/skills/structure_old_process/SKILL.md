# Skill: Old Processing Pipeline — Reference for New Pipeline Refactoring

## Purpose

This skill bridges the **old pipeline** (`Organized_processing.ipynb`) and the
**new pipeline** (`src/`).  Use it whenever:

- implementing or extending a feature in `src/` to verify correctness against
  the original algorithm,
- deciding whether an already-existing old result can substitute a new
  computation (i.e. cache-hit logic),
- checking what the new pipeline **must** reproduce and what can be improved.

---

## High-level pipeline comparison

| Aspect | Old pipeline | New pipeline (`src/`) |
|--------|-------------|----------------------|
| Entry point | Notebook cell: `analyze_from_dataset(dc, dc.id_)` | `src/main.ipynb` → `pipeline.analyze_from_dataset()` |
| Configuration | Hard-coded JSON at `/home/aamaduzzi/config/config_{region}.json` | `data/config/config_{region}.json` (portable, project-relative) |
| Output unit | One `.csv.gz` / `.json` file **per user** | One parquet **shard** per batch of 500 users, columnar |
| Resume detection | `os.path.isfile(per_user_path)` inside each save method | Parquet footer metadata scan (`pl.read_parquet_schema()`), O(1) |
| Data loading | `pd.read_parquet(file)` | Same |
| User filtering | `groupby('userId').size() > np_` per period | Same (`dataset_info.preprocess()`) |
| Spatial filter | `Polygon(bounding_box).within(point)` | Same |
| Server paths | Hard-coded `/data/aamaduzzi/...` | `PROJECT_ROOT/milestones_analysis/` via `DIR_MILESTONES_SERVER`; overridable via `MILESTONES_DIR` env var |
| Census files | Hard-coded absolute paths | `constants.CENSUS_FILES[region]` relative to `census_data/` |

---

## What must be preserved (algorithm equivalence)

All metric computations in `User.py` must produce the same numerical
results as the original `User` class in the notebook.  Key formulas:

### Radius of gyration

```python
# weight = time_spent_per_place / total_time_stops
rg = sqrt( sum( weight_i * (x_i^2 + y_i^2) ) )
```

where `x, y` are tangent-plane projections via `xy(lat, lon, avg_lat, avg_lon)`.

### Gonzalez trajectory shape

1. Project to tangent plane → `shifted_lat`, `shifted_lng`
2. Compute inertia tensor `(Ixx, Iyy, Ixy)`, rotation angle `theta`
3. Rotate coordinates: `rotated_lat`, `rotated_lng`
4. Normalise by `sigma_lat`, `sigma_lng`:
   `x_norm = rotated_lng / sigma_lng`, `y_norm = rotated_lat / sigma_lat`

Output columns: `x_norm`, `y_norm`, `sigmax`, `sigmay`

### S(t) exploration curve

Vector of length up to 1419 (time steps in hours × `t_threshold`).
`visited_places[t]` = number of distinct geohash7 cells visited up to
hour `t`.  Computed by `fill_dict(s_jan, t_jan)` which interpolates
between sparse trajectory events.

### Weekly radius of gyration

Computed only for weeks whose start falls within the period bounds.
Minimum 3 points required for a week to be valid; else `NaN`.

### Time filtering

`time_filtering_traj_per_person(t_threshold)`: removes rows where the
time gap to the next stop is less than `t_threshold` hours (cumulative
merging — short consecutive gaps are merged).

---

## Migration path: old → new for CA

CA currently has **1 438 077 files** in `milestones_analysis/CA/dataxuser/`
covering **only** the `15 jan - 15 march` period.

Files present per user:
- `all_scalars_{uid}_period_...csv.gz` → maps to `ParquetStore` kind `all_scalars`
- `gonzalez_{uid}_period_...csv.gz` → maps to `ParquetStore` kind `gonzalez`

Files **absent** (never computed for CA in old pipeline):
- `S_t_*` → kind `S` — **must be computed from raw trajectories**
- `frequnecy_rank_*` → kind `frequency` — **must be computed from raw trajectories**
- `weekly_rg_*` → kind `weekly_rg` — **must be computed from raw trajectories**
- All metrics for periods `15 march - 15 may` and `15 may - sept` — **must be computed**

Migration is handled by `store.migrate_from_legacy(kind, period, np_, t)`.
A consolidated parquet already exists:
`milestones_analysis/CA/all_scalars_period_15_jan_-_15_march_np_20_t_1/consolidated.parquet`

---

## Migration path: old → new for MA

MA has **all three periods** and **two np_ variants** (20 and 100) in
per-metric subdirectories.  Format is shard-level (many users per file),
not per-user.

Mapping of MA legacy dirs → new `ParquetStore` kinds:

| Legacy folder | New kind |
|---------------|----------|
| `radius_gyration_measures_new_threshold/` | `all_scalars` (column `radius_gyration`) |
| `distance_measures_new_threshold/` | `all_scalars` (column `distance`) |
| `k_radius_gyration_measures_new_threshold/` | `all_scalars` (columns `rg_3`, `rg_6`, `rg_10`) |
| `entropic_measures_new_threshold/` | `all_scalars` (columns `random_entropy`, `uncorrelated_entropy`, `real_entropy`) |
| `home_new_threshold/` | `all_scalars` (columns `home`, `home_geohash7`) |
| `gonzalez_new_threshold/` | `gonzalez` |
| `st_new_threshold/` | `S` |
| `location_frequency_new_threshold/` | `frequency` |

Migration is handled by `store.migrate_from_legacy_ma(np_, t)`.

> **Note:** For MA, `county_home`, `party_government`, `rurality_level`
> and `q` do not appear in the legacy metric folders — they were apparently
> not computed/saved in the MA old run.  These must be recomputed.

---

## np_ / t parameter differences between CA and MA

The old code had:
- **CA**: only `np_ = 20`, `t = 1` — fixed, no variation
- **MA**: `np_ ∈ {20, 100}`, `t ∈ {1, 8, 24}` — multiple parameter sets

The new pipeline should support this via `constants.MIN_POINTS_PER_USER`
and `TIME_THRESHOLD_HOURS`, but for CA only `np_=20, t=1` was ever used.
For MA, the `np_=20, t=1` set is the primary result; the others are
sensitivity analyses.

---

## Known differences and issues in the old pipeline

1. **"frequnecy" typo** in filename `frequnecy_rank_{uid}_...csv.gz` —
   carried into `constants.FNAME_FREQ_RANK` in the new code for migration
   compatibility.

2. **`already_saved_users_per_period.csv.gz` is unreliable** — it stores
   all users in one flat list regardless of period.  The new store uses
   parquet footer metadata instead.

3. **`number_users_period_*.json` is always `null` for CA** — was never
   populated correctly.

4. **`_get_county()` logic** — tries to read `all_scalars_*` file if it
   already exists, then assigns county.  This is a two-pass approach that
   the new pipeline consolidates into a single pass.

5. **Gonzalez `no_scaled` flag** — in the old code `no_scaled = False` is
   hard-coded inside `compute_gonzalez()`, meaning the normalised version
   is always computed.  Filename still contains `no_scaled` in some MA
   files due to an older code version.

6. **MA file naming inconsistency** — some MA gonzalez files have
   `no_scaled_t_threshold` in the name while others don't; this reflects
   different code versions used for different runs.

---

## Caching decision logic (for new pipeline)

When the new pipeline starts for a `(region, period, np_, t)` combination:

```
1. Check ParquetStore footer for each kind → lists already-computed users
2. Identify users in current parquet shard NOT in the store → must compute
3. If old per-user legacy files exist AND migration not done yet:
       call store.migrate_from_legacy() for this period/kind
       then re-check step 1
4. Compute remaining users → flush to store in batches of 500
```

The intermediate step (legacy migration) is a **one-time operation** per
`(period, kind, np_, t)`.  After migration the old per-user files can be
archived or deleted.

---

## What the new pipeline adds beyond the old

- **Parquet columnar store** — fast bulk reads for plotting (one file
  instead of millions).
- **S3 progressive download** — `analyze_from_s3_progressive()` downloads
  one shard at a time, processes it, then deletes the local copy.
- **Portable paths** — no hard-coded `/data/aamaduzzi/` references.
- **New output layout for CA** — aggregated per-period parquet dirs
  replace the flat `dataxuser/` layout (consistent with MA).
- **Unified scalar table** — all scalar metrics for a user are in a
  single columnar store, not scattered across many files.
