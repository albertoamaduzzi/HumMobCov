---
name: transition-distribution
description: >
  Reference for HumMobCov transition/presence distribution objects.
  Use when working with compute_presence_distribution,
  compute_transition_distribution, plot_period_aggregate_distribution,
  plot_all_columns_for_period, animate_distribution_evolution,
  or whenever you need the schema of the distribution DataFrames produced
  by transition_matrices/distribution_transitions_and_presences.py.
---

# Transition & Presence Distribution — Objects and Column Reference

## Source modules

| Module | Location |
|--------|----------|
| `distribution_transitions_and_presences` | `src/transition_matrices/distribution_transitions_and_presences.py` |
| `visualization_distribution_transition_counts` | `src/visualization/visualization_distribution_transition_counts.py` |
| Math reference | `src/transition_matrices/readme_transition_distribution_evolution.md` |

---

## 1. Input DataFrames (TransitionPipeline outputs)

### Presence matrix

| Column              | dtype   | Description                                           |
|---------------------|---------|-------------------------------------------------------|
| `geohash`           | Utf8    | Geohash cell at chosen precision                      |
| `time_int`          | Int64   | 0-based time bin index                                |
| `datetime`          | Utf8    | ISO-8601 string of the bin start (UTC)                |
| `count_birth`       | Int64   | Users whose first stop is in this (cell, bin)         |
| `count_death`       | Int64   | Users whose last stop is in this (cell, bin)          |
| `count_transit`     | Int64   | Users present in this bin AND the next one            |
| `count`             | Int64   | count_birth + count_death + count_transit             |
| `probability`       | Float64 | count_transit / Σ(count_transit) across all cells     |

### Transition matrix

| Column                  | dtype   | Description                                           |
|-------------------------|---------|-------------------------------------------------------|
| `geohash_start`         | Utf8    | Origin cell at time bin T                             |
| `geohash_end`           | Utf8    | Destination cell at time bin T+1                      |
| `time_int`              | Int64   | 0-based time bin index (T)                            |
| `datetime`              | Utf8    | ISO-8601 of bin T start (UTC)                         |
| `transitions`           | Int64   | Distinct users making this move                       |
| `transition_probability`| Float64 | transitions / Σ(transitions) leaving geohash_start   |

---

## 2. Distribution DataFrames

### `compute_presence_distribution(presence_df, period_name, *, bins=100)`

Returns a **wide** DataFrame.  Shape: `len(unique time_int) × bins` rows.

| Column                      | dtype   | Description                                              |
|-----------------------------|---------|----------------------------------------------------------|
| `time_bin`                  | Int64   | Time bin index (same as `time_int` in the source)        |
| `period_observation`        | Utf8    | Period label, e.g. `"15 jan - 15 march"`                 |
| `bin_count_birth`           | Float64 | Left edge of the histogram bin for `count_birth`         |
| `count_count_birth`         | Int64   | Number of cells falling in this bin                      |
| `bin_count_death`           | Float64 | Left edge for `count_death`                              |
| `count_count_death`         | Int64   | Number of cells in this bin                              |
| `bin_count_transit`         | Float64 | Left edge for `count_transit`                            |
| `count_count_transit`       | Int64   | Number of cells in this bin                              |
| `bin_count`                 | Float64 | Left edge for `count`                                    |
| `count_count`               | Int64   | Number of cells in this bin                              |
| `bin_probability`           | Float64 | Left edge for `probability`                              |
| `count_probability`         | Int64   | Number of cells in this bin                              |

**Source columns covered** (`PRESENCE_COLS`):
`count_birth`, `count_death`, `count_transit`, `count`, `probability`

**Log-spaced bins by default**: `count`, `count_birth`, `count_death`, `count_transit`

---

### `compute_transition_distribution(transition_df, period_name, *, bins=100)`

Returns a **wide** DataFrame.  Shape: `len(unique time_int) × bins` rows.

| Column                            | dtype   | Description                                              |
|-----------------------------------|---------|----------------------------------------------------------|
| `time_bin`                        | Int64   | Time bin index                                           |
| `period_observation`              | Utf8    | Period label                                             |
| `bin_transitions`                 | Float64 | Left edge for `transitions`                              |
| `count_transitions`               | Int64   | Number of edges in this bin                              |
| `bin_transition_probability`      | Float64 | Left edge for `transition_probability`                   |
| `count_transition_probability`    | Int64   | Number of edges in this bin                              |

**Source columns covered** (`TRANSITION_COLS`):
`transitions`, `transition_probability`

**Log-spaced bins by default**: `transitions`

---

## 3. Naming convention

For each source column `col`:

* `bin_{col}` — left edge of the histogram bin (Float64)
* `count_{col}` — number of cells / edges in that bin (Int64)

---

## 4. Visualization functions

### `plot_period_aggregate_distribution(dist_df, col, *, figsize, log_x, log_y, colors, ax)`

- **Input**: distribution DataFrame (presence or transition), source column name
- **What it shows**: mean `count_{col}` over all time bins, one step-curve per period
- **Returns**: `(fig, ax)`

### `plot_all_columns_for_period(dist_df, period, cols, *, figsize, log_x, log_y, color)`

- **Input**: distribution DataFrame, single period name, list of source columns
- **What it shows**: grid of period-aggregate distribution plots, one sub-plot per column
- **Returns**: `(fig, axes)`

### `animate_distribution_evolution(dist_df, col, *, fps, figsize, window, log_x, log_y, colors, title_prefix)`

- **Input**: distribution DataFrame, source column name
- **What it shows**: frame-by-frame animation over time bins; bars = instantaneous distribution,
  solid line = centred moving average (width `window` bins); one sub-plot per period
- **Returns**: `matplotlib.animation.FuncAnimation`

---

## 5. Typical usage

```python
from src.transition_matrices.distribution_transitions_and_presences import (
    compute_presence_distribution,
    compute_transition_distribution,
)
from src.visualization.visualization_distribution_transition_counts import (
    plot_period_aggregate_distribution,
    animate_distribution_evolution,
)

# Build distribution for one period
pres_dist  = compute_presence_distribution(pres_df,  "15 jan - 15 march")
trans_dist = compute_transition_distribution(trans_df, "15 jan - 15 march")

# Concatenate periods
all_pres = pl.concat([
    compute_presence_distribution(p1, "15 jan - 15 march"),
    compute_presence_distribution(p2, "15 march - 15 may"),
    compute_presence_distribution(p3, "15 may - sept"),
])

# Static: period-aggregate view
fig, ax = plot_period_aggregate_distribution(all_pres, "count", log_x=True, log_y=True)

# Animation: evolution of 'transitions' distribution over time
anim = animate_distribution_evolution(trans_dist, "transitions", fps=4, window=5)
from IPython.display import HTML
HTML(anim.to_jshtml())
```

---

## 6. Period names (standard)

| Label               | Date range                  |
|---------------------|-----------------------------|
| `"15 jan - 15 march"` | 2020-01-15 → 2020-03-15   |
| `"15 march - 15 may"` | 2020-03-15 → 2020-05-15   |
| `"15 may - sept"`     | 2020-05-15 → 2020-09-30   |
