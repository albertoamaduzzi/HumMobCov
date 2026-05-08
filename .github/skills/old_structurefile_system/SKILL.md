# Skill: Old Pipeline File-System Structure

## Purpose

This skill documents the **internal structure of the old pipeline** as
coded in `most_updated_scripts/scripts/Organized_processing.ipynb`.
Consult this whenever you need to:

- understand what a legacy output file contains,
- map a legacy file to its equivalent in the new pipeline,
- plan migration of old results into the new parquet store.

---

## Source notebook

`most_updated_scripts/scripts/Organized_processing.ipynb`

The notebook is **one large file** with the following logical sections:

| Section | Lines | Content |
|---------|-------|---------|
| Imports | 5–37 | Python imports; hard-coded server path `/data/rgallotti/libraries/PythonScripts/` for `rg_histograms`, `rg_fits` |
| `User` class | 43–562 | Core per-user metric computation; saves results to `dataxuser/` |
| `simulation` class | 568–579 | Stub; not used in production |
| `DataSet_Massachusets` / `DataSet_California` | 585–802 | Configuration objects: file lists, census paths, time periods |
| `dataset_info` | 585–802 | Wraps a single parquet shard; handles filtering and user selection |
| `plotter` class | 811–1806 | All statistical visualisations |
| `Global functions` | 1812–1980 | `filter_`, `xy`, `update_already_saved_users`, `get_already_saved_user_per_period`, `get_config` |
| `compute_all` / `analyze_from_dataset` | 1986–2253 | Top-level orchestration |
| Execution cell | 2259–2265 | Entry point: `dc = DataSet_California(); analyze_from_dataset(dc, dc.id_)` |

---

## Class inventory

### `User`

Represents a single user within a single time period.

**Constructor parameters:**

| param | type | meaning |
|-------|------|---------|
| `df` | DataFrame or None | raw stop-point trajectory (None when loading from saved files) |
| `period` | str | period name e.g. `'15 jan - 15 march'` |
| `county` | str | `'CA'` or `'MA'` |
| `np_` | int | minimum-points threshold |
| `t_threshold` | int | hours between stops |
| `period_names2period_division` | dict | maps period name → `[start_dt, end_dt]` |
| `uname` | str | user identifier |

**Key attributes:**

- `self.base_dir` = `/data/aamaduzzi/milestones_analysis/{county}/dataxuser/`
- `self.df2save` = defaultdict accumulating scalar results
- `self.df2save_gonzalez` = dict for Gonzalez results
- `self.df_St` = DataFrame `{time, visited_places}` for S(t)
- `self.df2frequencyrank` = DataFrame `{frequency, rank}`
- `self.week2rg` = dict `{week_idx: rg_value}`

**Metric methods:**

| method | output stored in | saved file name |
|--------|-----------------|-----------------|
| `compute_radius_of_gyration()` | `df2save['radius_gyration']` | part of `all_scalars_*.csv.gz` |
| `compute_random_entropy()` | `df2save['random_entropy']` | part of `all_scalars_*.csv.gz` |
| `compute_uncorrelated_entropy()` | `df2save['uncorrelated_entropy']` | part of `all_scalars_*.csv.gz` |
| `compute_real_entropy()` | `df2save['real_entropy']` | part of `all_scalars_*.csv.gz` |
| `compute_home()` | `df2save['home']`, `df2save['home_geohash7']` | part of `all_scalars_*.csv.gz` |
| `compute_krg()` | `df2save['rg_3']`, `rg_6`, `rg_10` | part of `all_scalars_*.csv.gz` |
| `compute_straight_line_distance()` | `df2save['distance']` | part of `all_scalars_*.csv.gz` |
| `compute_fraction_time_user_is_present()` | `df2save['q']` | part of `all_scalars_*.csv.gz` |
| `_get_county()` | `df2save['county_home']`, `party_government`, `rurality_level` | part of `all_scalars_*.csv.gz` |
| `compute_gonzalez()` | `df2save_gonzalez` | `gonzalez_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz` |
| `compute_St()` | `self.df_St` | `S_t_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz` |
| `compute_frequency_location()` | `self.df2frequencyrank` | `frequnecy_rank_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz` |
| `compute_weekly_radius_gyration()` | `self.week2rg` | `weekly_rg_{uid}_period_{p}_np_{np_}_t_{t}.json` |

**Save methods:**

| method | file template |
|--------|---------------|
| `_save_df()` | `all_scalars_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz` |
| `_savedf_gonzalez()` | `gonzalez_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz` |
| `_save_df_St()` | `S_t_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz` |
| `_save_df2frequencyrank()` | `frequnecy_rank_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz` *(note: typo "frequnecy" is in original)* |
| `_save_weekly_rg()` | `weekly_rg_{uid}_period_{p}_np_{np_}_t_{t}.json` |

All files land in `self.base_dir` = `milestones_analysis/{county}/dataxuser/`.

**Resume guard:** Each save method checks `os.path.isfile(...)` before
writing — if the file exists it is skipped silently.

---

### `DataSet_California` / `DataSet_Massachusets`

Configuration objects (not data containers).

**Common attributes:**

| attribute | value (CA) | value (MA) |
|-----------|-----------|-----------|
| `id_` | `'CA'` | `'MA'` |
| `dir` | `/data/shared/cuebiq/MOBS/urban_rural_flow_stops_cali_urban_rural_v3/` | `/data/shared/cuebiq/MOBS/20220330_stops_hq_users_MA` |
| `list_files` | all `.parquet` files in `dir` | `['subset_1.snappy.parquet',...,'subset_f.snappy.parquet']` (15 files) |
| `np_` | 20 | 20 |
| `t_threshold` | 1 | 1 |
| `period_names` | `['15 jan - 15 march', '15 march - 15 may', '15 may - sept']` | same |
| `period_division` | `[2020-01-15, 2020-03-15, 2020-05-15, 2020-09-30]` | same |
| `dir_output` | `milestones_analysis/CA/dataxuser/` | `milestones_analysis/MA/dataxuser/` |

**Difference between CA and MA datasets:**
- CA data is a single directory of many parquet shards (Spark output, split files with UUID names).
- MA data is a fixed list of 15 named subset parquet files.
- Census / rurality CSVs use `,` separator for CA and `;` for MA.

---

### `dataset_info`

Wraps a **single parquet shard file**.

**Key methods:**

- `spatial_filtering_per_country()` — clips to US bounding box
- `preprocess()` — calls spatial filter, then for each period slices
  the DF by date and selects users with `> np_` stop-points.
  Populates `self.period2df` and `self.period2listusers`.

---

### `plotter`

Reads results from `dataxuser/` (per-user files) and generates plots.

**Key helpers used:**

- `get_already_saved_user_per_period(directory)` — scans `dataxuser/`
  by walking all files, parsing the period from the filename (`jan - `,
  `march - `, `may - `), and populating a dict
  `{period: {metric_kind: [uid, ...]}}`.  This is slow for millions of files.

- `AllScalarsDict()`, `GonzalezDict()`, `StDict()`, `FreqDict()`,
  `WeekRgDict()` — build lists of file paths per period for each metric.

**Plot methods (all call the above helpers lazily):**

`plot_rg`, `plot_rg_party`, `plot_rg_party_per_period`,
`plot_rg_urban_rural`, `plot_rg_rurality_per_period`,
`plot_rg_county`, `plot_weekly_rg`, `plot_distance`, `plot_entropy`,
`plot_gonzalez`, `plot_sigmaxy`, `plot_conditional_gonzalez`,
`plot_St`, `plot_frequency`, `plot_krg`

---

## Key global functions

### `get_already_saved_user_per_period(directory)`

Walks `dataxuser/`, classifies each file by period keyword in filename,
extracts `uid` from split position (position 2 for non-gonzalez,
position 1 for gonzalez).

Returns `{period: {metric_kind: [uid, ...]}}` for the five kinds:
`all_scalars`, `gonzalez`, `S`, `frequency`, `weekly_rg`.

**Known limitation:** O(n_files) scan every time; with 1.4 M files this
is very slow.

### `update_already_saved_users(already_saved, period, user, dataset)`

Appends `user` to the in-memory list, deduplicates, and writes the list
to a gzip CSV `milestones_analysis/{dataset}/already_saved_users_per_period.csv.gz`.
Note: the CSV has no period column — it is a flat list of all users ever saved.

### `upload_already_saved_users(period_names, dataset)`

Reads the gzip CSV and returns `{period: [users]}` (all periods get the
same user list).

### `compute_all(cfg, dataset, list_users, period, df)`

Iterates `list_users`, creates a `User` per user, calls requested metric
methods based on boolean flags in `cfg` (`dict_algorithm_flow`), saves
results.  Saves `dictweek2npeople` to `number_users_period_{period}.json`.

### `analyze_from_dataset(dataset, name)`

Two execution paths:

1. **raw_trajectories = True** — iterates over `dataset.dir_files`,
   creates a `dataset_info` per file, preprocesses, calls `compute_all`
   per period.
2. **raw_trajectories = False** — reads `dir_output`, calls
   `get_already_saved_user_per_period`, calls `compute_all` per period
   (i.e. post-processing on already-saved data).

---

## Configuration JSON

Located at `/home/aamaduzzi/config/config_{CA,MA}.json` (hard-coded path).

Boolean flags control which metrics are computed:

```jsonc
{
  "raw_trajectories":            false,
  "is_weekly_radius_gyration":   true,
  "already_computed_rg":         false,
  "is_radius_gyration":          true,
  "is_gonzalez":                 true,
  "already_computed_gonzalez":   false,
  "is_random_entropy":           true,
  "already_computed_random_entropy": false,
  "is_uncorrelated_entropy":     true,
  "is_real_entropy":             true,
  "is_distance":                 true,
  "is_home":                     true,
  "is_krg":                      true,
  "is_St":                       true,
  "is_fraction_time":            true,
  "is_county_rural":             true,
  "is_frequency":                true,
  "is_week2points":              false
}
```

---

## MA vs CA structural differences in old pipeline

| Aspect | CA | MA |
|--------|----|----|
| Output layout | all metrics → `dataxuser/` flat | each metric has its own subdirectory |
| Periods computed | only `15 jan - 15 march` completed | all three periods |
| `np_` variants | only 20 | 20 and 100 |
| `t_threshold` variants | only 1 | 1, 8, 24 |
| Parquet shard naming | UUID-based (Spark output) | sequential `subset_N.snappy.parquet` |
| User list file | `dataxuser/*.csv.gz` scan | `already_saved_users_per_period.csv.gz` flat list |

---

## Data format of legacy files

### `all_scalars_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz`

Single-row CSV (index 0) with columns:

```
radius_gyration, random_entropy, uncorrelated_entropy, real_entropy,
distance, q, home, home_geohash7, county_home, party_government,
rurality_level, rg_3, rg_6, rg_10
```

`home` is stored as a WKT `POINT(lon lat)` string.

### `gonzalez_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz`

Multi-row CSV (one row per visited location):

```
x_norm, y_norm, sigmax, sigmay
```

### `S_t_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz`

Time-series CSV of the S(t) exploration curve:

```
time, visited_places
```

`time` runs from 0 to 1419 in steps of `t_threshold` hours.

### `frequnecy_rank_{uid}_period_{p}_np_{np_}_t_{t}.csv.gz`

*(Note: "frequnecy" is a typo carried from the original code.)*

```
frequency, rank
```

### `weekly_rg_{uid}_period_{p}_np_{np_}_t_{t}.json`

```json
{"0": 12345.6, "1": null, "2": 9876.5, ...}
```

Keys are week indices (int as string); values are radius of gyration
in metres or `null` if fewer than 3 points in that week.
