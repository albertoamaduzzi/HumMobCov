# Organized_processing.ipynb — Summary

## Purpose

This notebook analyses **human mobility patterns** during the COVID-19 pandemic (January 15 – September 30, 2020) using GPS stop-point data from the **Cuebiq** dataset for two US states: **California (CA)** and **Massachusetts (MA)**.  
It computes a suite of individual mobility metrics per user per time period, saves the results to disk, and produces publication-quality plots.

---

## Time Periods

The study window is divided into three pandemic phases:

| Period name             | Date range                  | Context              |
|-------------------------|-----------------------------|----------------------|
| `15 jan - 15 march`     | 2020-01-15 → 2020-03-15     | Pre-lockdown         |
| `15 march - 15 may`     | 2020-03-15 → 2020-05-15     | Lockdown             |
| `15 may - sept`         | 2020-05-15 → 2020-09-30     | Post-lockdown        |

---

## Key Classes

### `DataSet_California` / `DataSet_Massachusets`
Dataset configuration objects. They hold:
- Paths to raw parquet files (Cuebiq stops).
- Reference files: county GeoJSON shapes, rurality classification, political party per county.
- Preprocessing parameters: `np_ = 20` (minimum stops per user per period), `t_threshold = 1` (minimum hours between successive stops).
- Time-period definitions.

### `dataset_info`
Wraps a single parquet file. Responsibilities:
1. Load the parquet file into a DataFrame.
2. **Spatial filtering** — keep only stops inside the US bounding box using Shapely.
3. **Time-period splitting** — build `period2df` and `period2listusers`, filtering to users with ≥ `np_` stops in each period.

### `User`
Core per-user object. Constructed from a slice of a `dataset_info` DataFrame.  
After applying **time filtering** (`time_filtering_traj_per_person`), which discards rows where the gap to the next stop is below `t_threshold`, it exposes methods to compute each mobility metric and save results to compressed CSV / JSON files.

### `plotter`
Reads the per-user saved files from disk and produces plots for each metric. It never re-runs user-level computation — it only loads and aggregates already-saved results.

### `time_analysis`
Utility class for defining and validating time-period partitions.

---

## Processing Pipeline

```
analyze_from_dataset(dataset, name)
│
├── load config JSON  (config_CA.json / config_MA.json)
│     └── dict_algorithm_flow: flags controlling which metrics to compute
│
└── for each parquet file in dataset.dir_files:
      │
      ├── dataset_info(file, ...)
      │     ├── spatial_filtering_per_country()   # keep US points
      │     └── preprocess()                      # split by period, filter by np_
      │
      └── for each period in ['15 jan - 15 march', '15 march - 15 may', '15 may - sept']:
            │
            └── compute_all(dict_algorithm_flow, dataset, list_users, period, df)
                  │
                  └── for each user:
                        ├── User(df_user, period, ...)
                        ├── time_filtering_traj_per_person(t_threshold)
                        └── [conditional on config flags]:
                              compute_radius_of_gyration()
                              compute_weekly_radius_gyration()
                              compute_gonzalez()
                              compute_random_entropy()
                              compute_uncorrelated_entropy()
                              compute_real_entropy()
                              compute_straight_line_distance()
                              compute_home()
                              compute_krg()
                              compute_St()
                              compute_fraction_time_user_is_present()
                              compute_frequency_location()
                              _get_county()
                              _save_df()  /  _save_weekly_rg()  / ...
```

The config file supports two modes:
- **`raw_trajectories: true`** — compute from scratch using the raw parquet files.
- **`raw_trajectories: false`** — skip re-computation; aggregate already-saved per-user files.

Each metric also has an `already_computed_X` flag to skip recomputation if the output file already exists.

---

## Mobility Metrics (computed per user per period)

| Metric | Method | Description |
|--------|--------|-------------|
| **Radius of gyration** | `compute_radius_of_gyration()` | Weighted RMS distance from centroid; weights = time spent at each stop. |
| **k-Radius of gyration** | `compute_krg()` | RG restricted to the k most visited locations (k = 3, 6, 10). |
| **Weekly radius of gyration** | `compute_weekly_radius_gyration()` | Per-week RG computed within each 7-day sliding window. |
| **Gonzalez shape** | `compute_gonzalez()` | PCA rotation of the trajectory; outputs normalised coordinates (x/σ_x, y/σ_y) and the two principal variances σ_x, σ_y. |
| **Random entropy** | `compute_random_entropy()` | log₂(N) where N = number of distinct locations. |
| **Uncorrelated entropy** | `compute_uncorrelated_entropy()` | Entropy of the empirical location-frequency distribution. |
| **Real entropy** | `compute_real_entropy()` | Lempel–Ziv compression-based entropy. |
| **Straight-line distance** | `compute_straight_line_distance()` | Total straight-line distance travelled (via `skmob`). |
| **Home location** | `compute_home()` | Most frequently visited location during night hours (via `skmob`). |
| **Location frequency & rank** | `compute_frequency_location()` | Fraction of time spent at each location and the corresponding rank. |
| **S(t) — exploration curve** | `compute_St()` | Number of distinct places visited as a function of elapsed time. |
| **Fraction of time present (q)** | `compute_fraction_time_user_is_present()` | Fraction of the period for which the user's location is known. |
| **County / rurality / party** | `_get_county()` | Associates the home location with the county, rurality level (urban/rural), and governing political party. |

---

## Output Files (per user, per period)

All files are saved under `/data/aamaduzzi/milestones_analysis/<region>/dataxuser/`.  
The naming convention encodes the user ID, period, `np_` threshold, and `t_threshold`.

| File pattern | Content |
|---|---|
| `all_scalars_<user>_period_<p>_np_<n>_t_<t>.csv.gz` | Single row: all scalar metrics (RG, distance, entropies, k-RG, home, county, rurality, party, q). |
| `gonzalez_<user>_period_<p>_np_<n>_t_<t>.csv.gz` | Columns: `x_norm`, `y_norm`, `sigmax`, `sigmay`. |
| `S_t_<user>_period_<p>_np_<n>_t_<t>.csv.gz` | Columns: `time` (hours), `visited_places`. |
| `frequnecy_rank_<user>_period_<p>_np_<n>_t_<t>.csv.gz` | Columns: `frequency`, `rank`. |
| `weekly_rg_<user>_period_<p>_np_<n>_t_<t>.json` | Dict `{week_index: radius_of_gyration}`. |

---

## Plots (produced by `plotter`)

| Method | Plot |
|---|---|
| `plot_rg` | Log-log PDF of RG across the three periods. |
| `plot_rg_party` / `plot_rg_party_per_period` | RG distributions split by Democratic / Republican county. |
| `plot_rg_urban_rural` / `plot_rg_rurality_per_period` | RG distributions split by urban / rural county. |
| `plot_rg_county` | Separate RG plot for each county. |
| `plot_weekly_rg` | Time series of weekly average RG. |
| `plot_rg_rurality_weekly` / `plot_rg_party_weekly` | Weekly avg RG with error bars, stratified by rurality or party. |
| `plot_krg` | 2D histogram of RG vs k-RG; separate log-log PDFs of k-RG. |
| `plot_distance` | Log-log PDF of total distance with power-law fit (exponent α ± σ). |
| `plot_entropy` | PDFs of random, uncorrelated, and real entropy. |
| `plot_compression` | PDF of the compression ratio (real entropy / random entropy). |
| `plot_St` | S(t) scatter with power-law fit (exponent μ ± σ) per period. |
| `plot_frequency` | Bar chart of average location frequency by rank (top 9 ranks). |
| `plot_gonzalez` | 2D log-colour-map of normalised trajectory shape (x/σ_x vs y/σ_y). |
| `plot_conditional_gonzalez` | Semi-log slice Φ(x/σ_x \| y/σ_y = 0) of the Gonzalez density. |
| `plot_sigmaxy` | Log-log PDFs of σ_x and σ_y per period. |
| `plot_sigma_per_period` | σ_y (and σ_x) PDF overlaid across periods. |

---

## Supporting Utilities

| Function | Role |
|---|---|
| `filter_(x, t_threshold)` | Keeps rows in a trajectory where cumulative inter-stop time ≥ `t_threshold`. |
| `xy(lat, lon, lat0, lon0)` | Projects geographic coordinates onto a local tangent plane (metres). |
| `t_stop(df)` | Returns stop durations in minutes for each row. |
| `time_difference(start, end)` | Returns the difference in hours between two timestamps. |
| `ifnotexistsmkdir(dir_)` | Creates a directory if it does not already exist. |
| `get_already_saved_user_per_period(dir)` | Scans the output directory to build the `{period → {metric → [users]}}` checkpoint dict. |
| `update_already_saved_users(...)` | Appends a user to the checkpoint CSV so that the pipeline can resume without re-computing. |
| `get_config(name)` | Loads the JSON config file for the given region (`CA` or `MA`). |
| `init_compare_periods_dict(...)` | Builds a comparison dict for pairwise period statistics. |

---

## Notes & Caveats

- The **DEPRECATED** cell at the bottom contains an older, monolithic version of the pipeline that was replaced by `analyze_from_dataset` + `compute_all`.
- The `simulation`, `POI`, and `county_net` classes are stubs and are not used in the current pipeline.
- The `saving_` flag in `compute_all` is hardcoded to `False`, so `_save_df()` is never called from the refactored pipeline — saving must be triggered explicitly or re-enabled.
- Plotting code is guarded by `if 0 == 1:` blocks in the execution cell, so plots are only generated when the condition is changed manually.
