# HumMobCov

Analysis of how human mobility changed across three COVID-19 phases in **California** and **Massachusetts**, using anonymised GPS stop-point data (Cuebiq).  
Results are stratified by **rurality** (urban/rural) and **political affiliation** (Democratic/Republican counties).

---

## Goal

The project characterises mobility behaviour — before, during, and after the spring 2020 lockdowns — through a rich set of individual-level mobility metrics:

| Metric | Description |
|---|---|
| Radius of gyration | Typical spatial range of a user's movements |
| *k*-radius of gyration | Range covered by the *k* most-visited locations |
| Weekly radius of gyration | Radius of gyration binned by ISO week |
| Entropy (random / uncorrelated / real) | Predictability of the visited-location sequence |
| Distance | Total distance travelled per period |
| *S*(*t*) exploration curve | Number of distinct locations visited as a function of time |
| Location frequency / rank | Zipf-like distribution of visit frequencies |
| Gonzalez trajectory shape | PCA shape of normalised visit clouds |

The three analysis periods are:

| Period | Dates | COVID phase |
|---|---|---|
| `15 jan - 15 march` | 15 Jan 2020 → 15 Mar 2020 | Pre-lockdown |
| `15 march - 15 may` | 15 Mar 2020 → 15 May 2020 | Lockdown |
| `15 may - sept` | 15 May 2020 → 30 Sep 2020 | Post-lockdown |

---

## Installation

> Requires **Python 3.10**.  
> The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# 1. Install uv (once, system-wide)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter the project
git clone <repo-url>
cd HumMobCov

# 3. Create the virtual environment and install all dependencies
uv sync               # core dependencies
uv sync --group dev   # also installs JupyterLab, ipykernel, nbdime

# 4. Activate
source .venv/bin/activate
```

---

## Usage

Open and run `src/main.ipynb`.  The notebook is divided into three sections:

### 1 · INPUT
Choose the region (`CA` or `MA`), load the matching dataset object, and review the active feature flags from `data/config/config_<REGION>.json`.

### 2 · MAIN — computation pipeline

The pipeline supports three execution modes selected automatically at runtime:

| Situation | Action |
|---|---|
| Raw Cuebiq parquet files are accessible | Run the full per-user computation (`analyze_from_dataset`) |
| Only legacy per-user CSV.gz files exist | One-time migration into the parquet store (`store.migrate_all_periods`) |
| Parquet store already populated | Skip computation, go straight to visualisation |

The pipeline is **resume-safe**: already-computed users are detected by reading only the parquet footer metadata (no data is loaded), so interrupted runs can be continued without re-doing work.

### 3 · VISUALIZATION
Instantiate the `plotter` object and call any of the `plot_*` methods independently of Section 2, provided results are already on disk.

---

## Project structure

```
HumMobCov/
├── src/
│   ├── main.ipynb          # main entry-point notebook
│   ├── constants.py        # all paths, period definitions, parameters
│   ├── datasets.py         # DataSet_California / DataSet_Massachusets
│   ├── pipeline.py         # per-user computation orchestrator
│   ├── User.py             # individual mobility metric computation
│   ├── store.py            # columnar parquet storage layer (ParquetStore)
│   ├── plotter.py          # all visualisations
│   └── utils.py            # shared helpers
├── data/
│   └── config/             # feature-flag JSON files per region
├── census_data/            # shapefiles, population density, party affiliation
├── milestones_analysis/    # computed results (default location)
│   ├── CA/
│   │   ├── all_scalars_period_*/   # columnar parquet, one dir per period
│   │   ├── S_period_*/
│   │   ├── weekly_rg_period_*/
│   │   ├── gonzalez_period_*/
│   │   └── frequency_period_*/
│   └── MA/
└── pyproject.toml
```

---

## Architecture details

### Parquet store (`src/store.py`)

Results are stored as **columnar parquet files** where each user is a column, partitioned by *(period, metric kind)*. This gives:

- **Fast bulk reads** for plotting — one file read instead of millions of per-user files.
- **O(1) resume checks** — user IDs are stored as column names in the parquet footer; `pl.read_parquet_schema()` reads only metadata, no row data.
- **Low RAM usage** — data is written in batches (default `batch_size=500` users); each batch creates a small *shard* file. Shards are merged with `consolidate()` once a period is complete.

Directory layout per (period, kind):

```
milestones_analysis/{REGION}/
    {kind}_period_{period}_np_{np_}_t_{t}/
        shard_<timestamp>.parquet   ← append-only during computation
        consolidated.parquet        ← merged file after consolidation
```

Fixed-length kinds (`all_scalars`, `S`, `weekly_rg`) store users as columns with an index column (`metric`, `time`, or `week`).  
Variable-length kinds (`gonzalez`, `frequency`) use long format with a `user_id` column.

### Computation pipeline (`src/pipeline.py`)

`analyze_from_dataset()` iterates over raw parquet shards; for each shard it calls `compute_all()` which:

1. Filters users with fewer than `np_` stop-points in the period.
2. Skips users already present in the store (resume logic).
3. Instantiates a `User` object and computes all enabled metrics.
4. Accumulates results in in-memory batch dicts.
5. Flushes batches to the store every `batch_size` users.

### Configuration flags (`data/config/config_<REGION>.json`)

Each boolean flag enables or disables a computation step:

```jsonc
{
  "raw_trajectories": false,   // true → run from raw Cuebiq files
  "is_gonzalez":      true,    // compute Gonzalez trajectory shape
  "is_St":            true,    // compute S(t) exploration curve
  "is_frequency":     true,    // compute location frequency/rank
  "is_county_rural":  true,    // assign rurality and party from census
  ...
}
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MILESTONES_DIR` | `<project>/milestones_analysis` | Override the output root (e.g. a mounted server volume) |

---

## Data sources

- **Mobility data**: Cuebiq anonymised GPS stop-points (not publicly available; access via institutional agreement).
- **Census / boundary data**: U.S. Census Bureau TIGER shapefiles (`census_data/`).
- **Political affiliation**: County-level presidential vote share (`political_government_per_county.csv`).
- **Rurality**: Urban/rural classification derived from population density (`urban_info_threshold_urbanity_500.csv`).

---

## Overleaf synchronization

The manuscript lives in `.github/paper/` and is automatically pushed to the
shared [Overleaf project](https://www.overleaf.com/project/63aaac5d854f4a2fb2ce1c2b)
by the workflow at `.github/workflows/sync-overleaf.yml`.

### Source-of-truth rule

> **GitHub is the single source of truth.**  
> Always edit `.github/paper/` in this repository.  
> Do **not** edit files directly on Overleaf — those changes will be
> overwritten the next time the workflow runs.

### How the sync works

| Trigger | What happens |
|---|---|
| Push to `main` touching `.github/paper/**` | Files are pushed to Overleaf immediately |
| Daily at 06:00 UTC (schedule) | Baseline sync even with no new commits |
| Manual (`workflow_dispatch`) | Run any time; tick *Dry run* to preview without pushing |

The workflow clones the Overleaf project, replaces every file with the
contents of `.github/paper/`, commits with a timestamp, and pushes.
If nothing changed, the push is skipped.

### One-time setup — repository secrets

Go to **Settings → Secrets and variables → Actions** in this repository and
add the two secrets below.

| Secret name | Value |
|---|---|
| `OVERLEAF_USERNAME` | Your Overleaf account e-mail address |
| `OVERLEAF_TOKEN` | Your Overleaf account password. If your account uses Google / SSO sign-in, first set an Overleaf password via *Account → Password*. |

> **Tip:** Overleaf uses standard HTTP Basic Auth over HTTPS — there is no
> separate API token; your account password is the credential.

### Recovery from Overleaf conflicts

If someone has edited files directly on Overleaf and the push is rejected:

1. **Discard Overleaf changes** (recommended — GitHub is the source of truth):
   ```bash
   # Clone the Overleaf project locally
   git clone https://YOUR_EMAIL:YOUR_PASSWORD@git.overleaf.com/63aaac5d854f4a2fb2ce1c2b overleaf-local
   cd overleaf-local
   # Hard-reset to the last GitHub-synced commit, then force-push
   git fetch origin
   git reset --hard origin/master
   git push --force origin master
   ```
   Then re-run the GitHub Actions workflow.

2. **Preserve Overleaf changes** (manual merge):
   ```bash
   git clone https://YOUR_EMAIL:YOUR_PASSWORD@git.overleaf.com/63aaac5d854f4a2fb2ce1c2b overleaf-local
   cd overleaf-local
   # Identify changes, copy them into .github/paper/ in this repo, commit and push
   ```
   The next workflow run will then push the merged result back to Overleaf.
