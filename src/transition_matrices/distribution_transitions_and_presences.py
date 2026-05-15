"""
distribution_transitions_and_presences.py
==========================================
Compute per-time-bin histogram distributions from TransitionPipeline outputs.

Each function produces a **wide** DataFrame where every row represents one
histogram bin for one time bin.  With ``bins=100`` (the default) and ``T``
distinct time bins, the output contains ``T × 100`` rows.

Column naming convention
------------------------
For each source column ``col``:

* ``bin_{col}``   — left edge of the histogram bin (Float64)
* ``count_{col}`` — number of cells / edges that fall in this bin (Int64)

Additional key columns
----------------------
* ``time_bin``          — time bin index (Int64, same as ``time_int`` in the source)
* ``period_observation``— observation period name (Utf8)

See ``readme_transition_distribution_evolution.md`` for the full mathematical
definition of the distribution vectors.

Functions
---------
compute_presence_distribution(presence_df, period_name, *, bins, log_cols)
    Per-time-bin histograms of all five presence columns.

compute_transition_distribution(transition_df, period_name, *, bins, log_cols)
    Per-time-bin histograms of both transition columns.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

PRESENCE_COLS: list[str] = [
    "count_birth",
    "count_death",
    "count_transit",
    "count",
    "probability",
]

TRANSITION_COLS: list[str] = [
    "transitions",
    "transition_probability",
]

# Columns where log-spaced bins are more informative by default
_DEFAULT_LOG_COLS: set[str] = {
    "count",
    "count_birth",
    "count_death",
    "count_transit",
    "transitions",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _global_edges(
    values: np.ndarray,
    bins: int,
    log_scale: bool,
) -> np.ndarray:
    """Compute ``bins + 1`` global bin edges from a 1-D array of values."""
    v_min = float(values.min())
    v_max = float(values.max())
    if v_min == v_max:
        return np.array([v_min - 0.5, v_max + 0.5] if not log_scale
                        else [max(v_min * 0.5, 1e-12), v_max * 1.5])
    if log_scale:
        pos = values[values > 0]
        if pos.size == 0:
            return np.linspace(v_min, v_max, bins + 1)
        return np.logspace(np.log10(pos.min()), np.log10(v_max + 1e-12), bins + 1)
    return np.linspace(v_min, v_max, bins + 1)


def _build_distribution(
    df: pl.DataFrame,
    group_col: str,
    value_cols: list[str],
    period_name: str,
    bins: int,
    log_cols: set[str],
) -> pl.DataFrame:
    """
    Core engine: for each unique value of *group_col* (i.e. each time bin)
    compute a histogram over *bins* bins for every column in *value_cols*.

    Returns a DataFrame with ``n_time_bins × bins`` rows.
    """
    # --- compute global bin edges once per column ---
    edges: dict[str, np.ndarray] = {}
    for col in value_cols:
        vals = df[col].drop_nulls().to_numpy().astype(float)
        if vals.size == 0:
            edges[col] = np.linspace(0, 1, bins + 1)
        else:
            edges[col] = _global_edges(vals, bins, col in log_cols)

    time_bins_sorted = sorted(df[group_col].unique().to_list())

    rows: list[dict] = []
    for t in time_bins_sorted:
        chunk = df.filter(pl.col(group_col) == t)
        row_base = {"time_bin": t, "period_observation": period_name}

        # Build histogram for each column; they share the same bin-index row
        col_hists: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for col in value_cols:
            vals = chunk[col].drop_nulls().to_numpy().astype(float)
            if vals.size == 0:
                counts = np.zeros(bins, dtype=np.int64)
            else:
                counts, _ = np.histogram(vals, bins=edges[col])
            col_hists[col] = (edges[col][:-1], counts.astype(np.int64))

        # Emit one output row per bin index
        for i in range(bins):
            row = dict(row_base)
            for col in value_cols:
                bin_edges_arr, counts_arr = col_hists[col]
                row[f"bin_{col}"]   = float(bin_edges_arr[i])
                row[f"count_{col}"] = int(counts_arr[i])
            rows.append(row)

    # Build schema explicitly so Polars picks the right dtypes
    schema: dict[str, pl.DataType] = {
        "time_bin": pl.Int64,
        "period_observation": pl.Utf8,
    }
    for col in value_cols:
        schema[f"bin_{col}"]   = pl.Float64
        schema[f"count_{col}"] = pl.Int64

    return pl.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_presence_distribution(
    presence_df: pl.DataFrame,
    period_name: str,
    *,
    bins: int = 100,
    log_cols: set[str] | None = None,
) -> pl.DataFrame:
    """
    Compute per-time-bin histogram distributions for all presence columns.

    Parameters
    ----------
    presence_df : pl.DataFrame
        Presence matrix produced by ``TransitionPipeline``.
        Required columns: ``time_int`` + all of :data:`PRESENCE_COLS`.
    period_name : str
        Label stored in ``period_observation``, e.g. ``"15 jan - 15 march"``.
    bins : int
        Number of histogram bins (default 100).
    log_cols : set of str, optional
        Columns for which log-spaced bin edges are used.
        Defaults to ``{"count", "count_birth", "count_death", "count_transit"}``.

    Returns
    -------
    pl.DataFrame
        Schema: ``time_bin``, ``period_observation``,
        then for each col in :data:`PRESENCE_COLS`:
        ``bin_{col}`` (Float64), ``count_{col}`` (Int64).

        Shape: ``len(unique time_int) × bins`` rows.

    Examples
    --------
    >>> dist = compute_presence_distribution(pres_df, "15 jan - 15 march")
    >>> dist.shape
    (T * 100, 12)
    """
    if log_cols is None:
        log_cols = _DEFAULT_LOG_COLS

    available = [c for c in PRESENCE_COLS if c in presence_df.columns]
    missing   = [c for c in PRESENCE_COLS if c not in presence_df.columns]
    if missing:
        import warnings
        warnings.warn(
            f"compute_presence_distribution: columns not found and skipped: {missing}",
            stacklevel=2,
        )

    return _build_distribution(
        presence_df,
        group_col="time_int",
        value_cols=available,
        period_name=period_name,
        bins=bins,
        log_cols=log_cols,
    )


def compute_transition_distribution(
    transition_df: pl.DataFrame,
    period_name: str,
    *,
    bins: int = 100,
    log_cols: set[str] | None = None,
) -> pl.DataFrame:
    """
    Compute per-time-bin histogram distributions for all transition columns.

    Parameters
    ----------
    transition_df : pl.DataFrame
        Transition matrix produced by ``TransitionPipeline``.
        Required columns: ``time_int`` + all of :data:`TRANSITION_COLS`.
    period_name : str
        Label stored in ``period_observation``, e.g. ``"15 jan - 15 march"``.
    bins : int
        Number of histogram bins (default 100).
    log_cols : set of str, optional
        Columns for which log-spaced bin edges are used.
        Defaults to ``{"transitions"}``.

    Returns
    -------
    pl.DataFrame
        Schema: ``time_bin``, ``period_observation``,
        then for each col in :data:`TRANSITION_COLS`:
        ``bin_{col}`` (Float64), ``count_{col}`` (Int64).

        Shape: ``len(unique time_int) × bins`` rows.

    Examples
    --------
    >>> dist = compute_transition_distribution(trans_df, "15 march - 15 may")
    >>> dist.columns
    ['time_bin', 'period_observation', 'bin_transitions', 'count_transitions',
     'bin_transition_probability', 'count_transition_probability']
    """
    if log_cols is None:
        log_cols = _DEFAULT_LOG_COLS

    available = [c for c in TRANSITION_COLS if c in transition_df.columns]
    missing   = [c for c in TRANSITION_COLS if c not in transition_df.columns]
    if missing:
        import warnings
        warnings.warn(
            f"compute_transition_distribution: columns not found and skipped: {missing}",
            stacklevel=2,
        )

    return _build_distribution(
        transition_df,
        group_col="time_int",
        value_cols=available,
        period_name=period_name,
        bins=bins,
        log_cols=log_cols,
    )
