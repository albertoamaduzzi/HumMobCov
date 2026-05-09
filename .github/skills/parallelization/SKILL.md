# Skill: Parallelization — Numba & Polars (HumMobCov)

## Purpose

This skill documents the preferred patterns for **parallelising CPU-bound
computations on NumPy arrays and DataFrames** within HumMobCov.
Apply these patterns whenever adding or refactoring hot-path code in
`src/` (User.py, pipeline.py, transition_matrices/, utils.py, …).

---

## Rule 1 — Use `@numba.jit` for NumPy / loop-heavy kernels

```python
import numba
import numpy as np

@numba.jit(nopython=True, parallel=True, cache=True)
def haversine_distances(lat_arr: np.ndarray, lon_arr: np.ndarray) -> np.ndarray:
    """Vectorised haversine over trajectory arrays."""
    n = lat_arr.shape[0]
    out = np.empty(n - 1, dtype=np.float64)
    for i in numba.prange(n - 1):   # ← parallel loop
        dlat = np.radians(lat_arr[i + 1] - lat_arr[i])
        dlon = np.radians(lon_arr[i + 1] - lon_arr[i])
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(lat_arr[i])) * np.cos(np.radians(lat_arr[i + 1]))
             * np.sin(dlon / 2) ** 2)
        out[i] = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return out
```

### When to use `@numba.jit`

| Scenario | Advice |
|----------|--------|
| Loops over millions of rows / trajectory points | ✓ Ideal — use `nopython=True, parallel=True` |
| Custom entropy / entropy-rate kernels | ✓ Good fit |
| NumPy array operations where `parallel=True` already helps | Prefer `numba.prange` over `range` |
| Pure pandas / string ops | ✗ Use Polars instead |

### Rules

- Always set `nopython=True` (raises error rather than falling back to Python).
- Set `cache=True` to avoid recompilation across kernel restarts.
- Use `numba.prange` instead of `range` inside `@jit(parallel=True)` loops.
- Numba does **not** support arbitrary Python objects — pre-extract arrays
  from DataFrames before entering jit'd code.
- First call triggers JIT compilation; warm-up with a small array if needed.

---

## Rule 2 — Use Polars for DataFrame operations

`ParquetStore` already uses Polars. Extend this pattern to all new modules.

```python
import polars as pl

# ── Groupby + aggregation — zero-copy, parallel under the hood ──────────────
presence = (
    df_traj                           # pl.DataFrame with uid, geohash, time_bin
    .group_by(["geohash", "time_bin"])
    .agg([
        pl.col("uid").n_unique().alias("count_transit"),
    ])
)

# ── Lazy API for large datasets (streaming, predicate push-down) ────────────
result = (
    pl.scan_parquet("/path/to/shards/*.parquet")
    .filter(pl.col("lat").is_between(32.5, 42.0))
    .select(["uid", "lat", "lon", "datetime"])
    .collect(streaming=True)            # streaming=True keeps RAM usage low
)

# ── Efficient joins ──────────────────────────────────────────────────────────
merged = df_left.join(df_right, on="uid", how="left")

# ── Fast string ops (geohash coarsening) ────────────────────────────────────
df = df.with_columns(
    pl.col("geohash7").str.slice(0, 5).alias("geohash5")
)
```

### When to use Polars vs. Pandas

| Operation | Prefer |
|-----------|--------|
| Read / write parquet shards | **Polars** (`pl.scan_parquet`, `write_parquet`) |
| Group-by aggregations on millions of rows | **Polars** |
| Lazy evaluation / streaming large files | **Polars** |
| Existing skmob `TrajDataFrame` API | **Pandas** (skmob requires it) |
| Census / small lookup tables | Either |

---

## Rule 3 — Numba + Polars bridging

Extract arrays from Polars before calling jit'd code, then wrap results back:

```python
import numba
import numpy as np
import polars as pl

@numba.jit(nopython=True, parallel=True, cache=True)
def _count_transitions_kernel(
    start_geohash: np.ndarray,  # int32 encoded geohash ids
    end_geohash:   np.ndarray,
    n_cells:       int,
) -> np.ndarray:
    """Returns (n_cells × n_cells) transition count matrix."""
    mat = np.zeros((n_cells, n_cells), dtype=np.int64)
    for i in numba.prange(start_geohash.shape[0]):
        mat[start_geohash[i], end_geohash[i]] += 1
    return mat

def count_transitions(df: pl.DataFrame) -> np.ndarray:
    s = df["geohash_start_id"].to_numpy()   # zero-copy if contiguous
    e = df["geohash_end_id"].to_numpy()
    n = int(df["geohash_start_id"].max()) + 1
    return _count_transitions_kernel(s, e, n)
```

---

## Rule 4 — Chunked / streaming processing to stay within 32 GB RAM

The project targets 32 GB RAM (CA + MA combined). Always use chunked reads:

```python
# Polars streaming (lazy, constant memory)
for batch in pl.scan_parquet(shard_path).collect(streaming=True).iter_slices(10_000):
    process_batch(batch)

# Or with polars LazyFrame
lf = pl.scan_parquet(shard_path)
lf.filter(...).collect(streaming=True)
```

For numpy-heavy pipelines, prefer processing one raw shard at a time
(already the pattern in `analyze_from_s3_progressive`).

---

## Project-specific hot paths (apply these rules here first)

| File | Hot path | Recommended fix |
|------|----------|-----------------|
| `User.py` — `compute_real_entropy` | Shannon entropy over trajectory sequence | `@numba.jit` |
| `User.py` — `compute_St` | S(t) curve over time steps | `@numba.jit` with `parallel=True` |
| `utils.py` — `filter_` | Time-gap filter over stop arrays | Already candidate for `@numba.jit` |
| `transition_matrices/transition_pipeline.py` | Time binning, transition counting | Polars group-by + `@numba.jit` for matrix fill |
| `tile_counties_via_geohash.py` | Neighbour BFS over geohash grid | Pure Python + `set` — acceptable at precision ≤ 6 |

---

## Installation

```bash
uv add numba        # adds numba to pyproject.toml
uv add polars       # already present
```

Or in the virtual environment:

```bash
pip install numba
```

Numba requires LLVM and is CPU-only by default.  GPU (CUDA) is available
via `numba.cuda` but is not required by this project.
