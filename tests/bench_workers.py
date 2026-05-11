"""
bench_workers.py
================
Module-level picklable worker functions for benchmark_time_functions.ipynb.

Must be a proper module (not a notebook cell) so that ProcessPoolExecutor
can pickle the callables when spawning worker processes.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root is on sys.path when executed from the tests/ dir
_TESTS_DIR = Path(__file__).parent
_PROJECT_ROOT = _TESTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vectorized_pipeline import (  # noqa: E402
    _compute_radius_of_gyration_polars,
    _compute_krg_polars,
    _compute_entropies_polars,
    _compute_home_polars,
    _compute_distance_polars,
    _compute_weekly_rg_polars,
    # Re-export existing workers so the notebook can import everything from one place
    _re_worker,
    _st_worker,
    _gonz_worker,
    _freq_worker,
)

# ---------------------------------------------------------------------------
# Constants shared between workers and the notebook
# ---------------------------------------------------------------------------

K_VALUES: list[int] = [3, 6, 10]
T_THRESHOLD: int = 1

# Period info for weekly_rg: a single calendar month (≈ 4 weeks)
PERIOD_DIVISION: list[datetime] = [datetime(2020, 1, 1), datetime(2020, 2, 1)]
PERIOD_NAME: str = "2020-01"
PERODNAME2IDX: dict[str, int] = {PERIOD_NAME: 0}


# ---------------------------------------------------------------------------
# Picklable wrappers for Polars-native functions
# ---------------------------------------------------------------------------

def rg_worker(chunk_df):
    """Radius of gyration — worker for ProcessPoolExecutor."""
    return _compute_radius_of_gyration_polars(chunk_df)


def krg_worker(chunk_df):
    """k-Radius of gyration (k=3,6,10) — worker for ProcessPoolExecutor."""
    return _compute_krg_polars(chunk_df, K_VALUES)


def entropies_worker(chunk_df):
    """Random + uncorrelated entropy — worker for ProcessPoolExecutor."""
    return _compute_entropies_polars(chunk_df)


def home_worker(chunk_df):
    """Home location — worker for ProcessPoolExecutor."""
    return _compute_home_polars(chunk_df)


def distance_worker(chunk_df):
    """Mean inter-stop distance — worker for ProcessPoolExecutor."""
    return _compute_distance_polars(chunk_df)


def weekly_rg_worker(chunk_df):
    """Weekly radius of gyration — worker for ProcessPoolExecutor."""
    return _compute_weekly_rg_polars(
        chunk_df,
        PERIOD_DIVISION,
        PERIOD_NAME,
        PERODNAME2IDX,
    )
