# Skill: Old Output Structure

## Purpose
This skill describes the **already-computed data** from the old pipeline
(run on the original server).  Consult it whenever you need to know which
users/metrics have already been computed so you can avoid re-doing work
or can feed old results into the new pipeline.

---

## Root output directory

```
milestones_analysis/
├── CA/
└── MA/
```

`output/CA/dataxuser/` existed as an empty placeholder and has been removed.
The authoritative output root is always `milestones_analysis/`.

---

## CA (California) — what exists

### `milestones_analysis/CA/dataxuser/`  (1 438 077 files)

One `.csv.gz` per user per metric, naming convention:

```
{metric}_{uid}_period_{period}_np_{np_}_t_{t}.csv.gz
```

| segment | meaning |
|---------|---------|
| `{metric}` | `all_scalars` or `gonzalez` |
| `{uid}` | SHA-256-like hex user identifier |
| `{period}` | time period string, see below |
| `{np_}` | minimum-points threshold (always **20** in CA) |
| `{t}` | time-threshold in hours (always **1** in CA) |

**Periods present in CA/dataxuser:**

| Period string | Dates | COVID phase |
|---|---|---|
| `15 jan - 15 march` | Jan 15 → Mar 15 2020 | Pre-lockdown |

> **Only `15 jan - 15 march` files are present in `dataxuser/`.**
> The JSON files `number_users_period_*.json` for the other two periods
> exist but are empty (`null`), indicating those periods were not
> fully processed in the old run.

### `milestones_analysis/CA/all_scalars_period_15_jan_-_15_march_np_20_t_1/`

Contains a single `consolidated.parquet` — a merged columnar store
(new format) for the `15 jan - 15 march` period.  This was produced
by migrating the per-user `all_scalars_*.csv.gz` files from `dataxuser/`.

### `milestones_analysis/CA/already_saved_users_per_period.csv.gz`

A gzip CSV tracking which users have been saved.  The current file
only contains a header row — it reflects the state at the moment of the
old run and is not a reliable complete list.

### `milestones_analysis/CA/number_users_period_*.json`

Three JSON files (one per period).  All contain `null` — they were never
properly populated on the old server.

---

## MA (Massachusetts) — what exists

The MA output uses a **different folder layout** (one folder per metric,
sub-divided by `np_` and `t_threshold`).

```
milestones_analysis/MA/
├── already_saved_users_per_period.csv.gz   # one user per line (single-period dump)
├── distance_measures_new_threshold/
│   ├── 20/    (np_ = 20)
│   │   ├── 1/   (t = 1 h)  — 7 843 shard CSV files
│   │   ├── 8/   (t = 8 h)  — ~46 files
│   │   └── 24/  (t = 24 h) — 46 files
│   └── 100/   (np_ = 100)
│       └── 1/   — 1 file
├── entropic_measures_new_threshold/    (same np/t sub-structure)
├── gonzalez_new_threshold/             (same np/t sub-structure)
├── home_new_threshold/
│   └── 20/1/
├── k_radius_gyration_measures_new_threshold/  (same np/t sub-structure)
├── location_frequency_new_threshold/
│   ├── 20/
│   └── (1, 8, 24 sub-dirs)
├── radius_gyration_measures_new_threshold/    (same np/t sub-structure)
├── st_new_threshold/                          (same np/t sub-structure)
└── plot/
```

**All three periods are present** for MA:

| Period | files |
|---|---|
| `15 jan - 15 march` | ✓ |
| `15 march - 15 may` | ✓ |
| `15 may - sept` | ✓ |

### np_ values present in MA

| np_ | t values present |
|-----|-----------------|
| 20  | 1, 8, 24 |
| 100 | 1 |

### File naming inside MA metric folders

```
# radius of gyration
rg_{period}_{np_}_threshold_{t}_hour_{shard_id}_CA.csv

# distance
dist_{period}_{np_}_threshold_{t}_hour_{shard_id}_CA.csv

# Gonzalez
gonzalez_{period}_{np_}_no_scaled_t_threshold_{t}_hour_{shard_id}_CA.json

# S(t) exploration curve
dict_s_{period}_{np_}_{shard_id}_hour_{t}_CA.csv

# Home location
home_{period}_{np_}_threshold_{t}_hour_subset_N.csv

# Entropic measures — several sub-files per period:
#   real_entropy_{period}_{np_}_threshold_{t}_hour.csv  (;values;uid)
#   uncorr_entropy_{period}_{np_}_threshold_{t}_hour.json
#   rdm_entropy_{period}_{np_}_threshold_{t}_hour.json
```

### `milestones_analysis/MA/already_saved_users_per_period.csv.gz`

Contains a flat list of user IDs (one per row, no period column).
This reflects the legacy single-period tracking and is not
period-specific.

---

## Summary: what is and is not computed

| Region | Period | Metrics | Status |
|--------|--------|---------|--------|
| CA | 15 jan - 15 march | all_scalars, gonzalez | **computed** (1.4 M files) |
| CA | 15 march - 15 may | all | **not computed** |
| CA | 15 may - sept | all | **not computed** |
| MA | 15 jan - 15 march | rg, dist, k_rg, entropy, gonzalez, S(t), home, freq | **computed** |
| MA | 15 march - 15 may | rg, dist, k_rg, entropy, gonzalez, S(t) | **computed** |
| MA | 15 may - sept | rg, dist, k_rg, entropy, gonzalez, S(t) | **computed** |

---

## Why the per-user file approach is being replaced

Each metric for each user was saved as a separate `.csv.gz` file.
This produced millions of tiny files, making bulk reads for plotting
extremely slow (scan all files one by one) and resume-checking
expensive (list entire directory).

The new pipeline uses a **columnar parquet store** (see `src/store.py`)
where each `(period, metric_kind)` pair maps to a shard directory of
parquet files, each shard containing a batch of users as columns.
Resume detection is O(1) via parquet footer metadata.

Migration from the old per-user files to the new store is handled by
`store.migrate_from_legacy()` / `store.migrate_all_periods()`.
