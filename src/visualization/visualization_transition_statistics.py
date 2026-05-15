"""
visualization_transition_statistics.py
========================================
Static statistical summaries of TransitionPipeline outputs.

Functions
---------
plot_population_over_time(presence_df, col, *, figsize, color, ax)
    Total *col* (summed over all cells) per time bin — line/bar chart.

plot_population_distribution(presence_df, col, *, bins, figsize, color,
                              log_scale, ax)
    Histogram of per-cell *col* values aggregated across all bins.

plot_mean_transitions_over_time(transition_df, col, *, figsize, color, ax)
    Mean (or total) *col* per time bin, aggregated across all edges.

plot_transition_distribution(transition_df, col, *, bins, figsize,
                              color, log_scale, ax)
    Histogram of per-edge *col* values (all time bins pooled).

plot_top_transitions(transition_df, *, n, time_int, col, figsize, ax)
    Horizontal bar chart of the top-N (start → end) pairs by *col*.

Expected DataFrames
-------------------
**presence_matrix**
    Columns: ``geohash``, ``time_int``, ``datetime``,
    ``count_birth``, ``count_death``, ``count_transit``,
    ``count``, ``probability``

**transition_matrix**
    Columns: ``geohash_start``, ``geohash_end``, ``time_int``,
    ``datetime``, ``transitions``, ``transition_probability``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Presence / population plots
# ---------------------------------------------------------------------------

def plot_population_over_time(
    presence_df: pl.DataFrame,
    col: str = "count",
    *,
    agg: str = "sum",
    figsize: tuple[float, float] = (11, 4),
    color: str = "steelblue",
    xlabel: str = "Time bin",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Total (or mean) population column per time bin.

    Parameters
    ----------
    presence_df : pl.DataFrame
        Presence matrix.
    col : str
        Column to aggregate, e.g. ``"count"``, ``"count_transit"``,
        ``"probability"``.
    agg : {"sum", "mean"}
        Aggregation function.
    figsize : tuple
    color : str
    xlabel : str
    ax : plt.Axes, optional

    Returns
    -------
    (fig, ax)
    """
    agg_fn  = pl.col(col).sum() if agg == "sum" else pl.col(col).mean()
    series  = (
        presence_df
        .group_by("time_int")
        .agg(agg_fn.alias(col))
        .sort("time_int")
    )

    t   = series["time_int"].to_numpy()
    y   = series[col].to_numpy().astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.bar(t, y, color=color, alpha=0.8, width=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{agg}({col})")
    ax.set_title(f"{agg.capitalize()} of '{col}' per time bin")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return fig, ax


def plot_population_distribution(
    presence_df: pl.DataFrame,
    col: str = "count",
    *,
    bins: int = 40,
    figsize: tuple[float, float] = (8, 4),
    color: str = "steelblue",
    log_scale: bool = False,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Histogram of per-cell *col* values averaged across all time bins.

    Each geohash cell contributes its mean value across all bins.
    This gives the *average distribution of population* across grid cells.

    Parameters
    ----------
    presence_df : pl.DataFrame
    col : str
    bins : int
        Number of histogram bins.
    figsize : tuple
    color : str
    log_scale : bool
        Use a log scale on the x-axis.
    ax : plt.Axes, optional

    Returns
    -------
    (fig, ax)
    """
    per_cell = (
        presence_df
        .group_by("geohash")
        .agg(pl.col(col).mean().alias(col))
    )[col].to_numpy().astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.hist(per_cell, bins=bins, color=color, alpha=0.8, edgecolor="white")
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(f"Mean {col} per cell")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Distribution of average '{col}' across geohash cells")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Transition plots
# ---------------------------------------------------------------------------

def plot_mean_transitions_over_time(
    transition_df: pl.DataFrame,
    col: str = "transitions",
    *,
    agg: str = "mean",
    figsize: tuple[float, float] = (11, 4),
    color: str = "darkorange",
    xlabel: str = "Time bin",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Mean (or total) transition *col* per time bin, summed over all edges.

    This shows how the overall mobility volume (or weight) evolves.

    Parameters
    ----------
    transition_df : pl.DataFrame
        Transition matrix.
    col : str
        Column to aggregate, e.g. ``"transitions"`` or
        ``"transition_probability"``.
    agg : {"sum", "mean"}
    figsize : tuple
    color : str
    xlabel : str
    ax : plt.Axes, optional

    Returns
    -------
    (fig, ax)
    """
    agg_fn = pl.col(col).sum() if agg == "sum" else pl.col(col).mean()
    series = (
        transition_df
        .group_by("time_int")
        .agg(agg_fn.alias(col))
        .sort("time_int")
    )

    t = series["time_int"].to_numpy()
    y = series[col].to_numpy().astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.bar(t, y, color=color, alpha=0.8, width=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{agg}({col})")
    ax.set_title(f"{agg.capitalize()} '{col}' per time bin (all edges)")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return fig, ax


def plot_transition_distribution(
    transition_df: pl.DataFrame,
    col: str = "transition_probability",
    *,
    bins: int = 40,
    figsize: tuple[float, float] = (8, 4),
    color: str = "darkorange",
    log_scale: bool = False,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of per-edge *col* values averaged across all time bins.

    Each (start, end) edge contributes its mean value.  Values are first
    binned into *bins* equally-spaced (or log-spaced) intervals; the scatter
    then shows (bin_centre, count_in_bin) — one point per non-empty bin.
    This reveals the shape of the weight distribution without the visual
    density bias of a bar chart.

    Parameters
    ----------
    transition_df : pl.DataFrame
    col : str
    bins : int
        Number of bins used to compute the frequency of each value range.
    figsize : tuple
    color : str
    log_scale : bool
        Apply a log scale on both axes.
    ax : plt.Axes, optional

    Returns
    -------
    (fig, ax)
    """
    per_edge = (
        transition_df
        .group_by(["geohash_start", "geohash_end"])
        .agg(pl.col(col).mean().alias(col))
    )[col].to_numpy().astype(float)

    if log_scale:
        pos = per_edge[per_edge > 0]
        if pos.size > 0:
            edges = np.logspace(np.log10(pos.min()), np.log10(pos.max() + 1e-12), bins + 1)
        else:
            edges = np.linspace(per_edge.min(), per_edge.max(), bins + 1)
    else:
        edges = np.linspace(per_edge.min(), per_edge.max(), bins + 1)

    counts, _ = np.histogram(per_edge, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    mask = counts > 0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.scatter(centres[mask], counts[mask], color=color, alpha=0.8, s=30)
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(f"Mean {col} per edge")
    ax.set_ylabel("Number of edges")
    ax.set_title(f"Distribution of average '{col}' across transitions")
    ax.grid(linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return fig, ax


def plot_top_transitions(
    transition_df: pl.DataFrame,
    *,
    n: int = 20,
    time_int: int | None = None,
    col: str = "transitions",
    figsize: tuple[float, float] = (8, 6),
    color: str = "darkorange",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart of the top-N (start → end) pairs by *col*.

    Parameters
    ----------
    transition_df : pl.DataFrame
    n : int
        Number of top edges to show.
    time_int : int, optional
        If None, edges are summed across all time bins.
    col : str
    figsize : tuple
    color : str
    ax : plt.Axes, optional

    Returns
    -------
    (fig, ax)
    """
    if time_int is not None:
        df = transition_df.filter(pl.col("time_int") == time_int)
    else:
        df = (
            transition_df
            .group_by(["geohash_start", "geohash_end"])
            .agg(pl.col(col).sum().alias(col))
        )

    top = (
        df
        .sort(col, descending=True)
        .head(n)
        .with_columns(
            (pl.col("geohash_start") + " → " + pl.col("geohash_end"))
            .alias("edge")
        )
    )

    labels = top["edge"].to_list()
    values = top[col].to_numpy().astype(float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values[::-1], color=color, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel(col)
    bin_label = f"bin {time_int}" if time_int is not None else "all bins"
    ax.set_title(f"Top {n} transitions by '{col}'  [{bin_label}]")
    ax.grid(axis="x", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    return fig, ax


def plot_top_populated_stochastic(
    presence_df: pl.DataFrame,
    col: str = "count",
    *,
    n: int = 10,
    figsize: tuple[float, float] = (14, 5),
    cmap: str = "tab10",
    alpha: float = 0.85,
    datetime_col: str = "datetime",
    show_mean: bool = True,
    periods: list[tuple[str, "pl.DataFrame"]] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Time-series of *col* for the top-N most-populated geohash cells.

    Each cell is treated as a stochastic process: its value at each time
    bin is plotted as a separate coloured line.  This reveals both the
    level and the temporal variability of the busiest locations.

    The optional *mean* line (dashed) shows the per-bin mean across **all**
    cells for comparison.

    Pass ``periods`` to span **all three observation periods** on a single
    continuous x-axis.  Vertical dashed lines mark the period boundaries.

    Parameters
    ----------
    presence_df : pl.DataFrame
        Single-period presence matrix (used when *periods* is None).
    col : str
        Column to plot, e.g. ``"count"``, ``"count_transit"``,
        ``"probability"``.
    n : int
        Number of top cells to show.
    figsize : tuple
    cmap : str
        Matplotlib colormap for the cell lines.
    alpha : float
        Line opacity.
    datetime_col : str
        Name of the datetime string column (used for x-axis labels).
        Falls back to ``time_int`` if not present.
    show_mean : bool
        Overlay the per-bin mean across all cells as a thick dashed line.
    periods : list of (period_name, presence_df) tuples, optional
        When provided, the three period DataFrames are concatenated in the
        order given and displayed on a single x-axis that spans January to
        September.  Period boundaries are marked with vertical lines.
        *presence_df* is ignored in this mode.
    ax : plt.Axes, optional

    Returns
    -------
    (fig, ax)
    """
    # ── Multi-period mode: stitch all period DataFrames in chronological order
    if periods is not None:
        # Assign a continuous x-index that grows monotonically across periods.
        # Within each period, time_int values restart at 0, so we offset each
        # period by the cumulative number of bins from previous periods.
        combined_frames: list[pl.DataFrame] = []
        boundary_x: list[int] = []      # x positions of period starts (for vlines)
        period_names: list[str] = []
        x_offset = 0

        for period_name, pdf in periods:
            t_sorted = sorted(pdf["time_int"].unique().to_list())
            n_bins_this = len(t_sorted)
            t2local = {t: i for i, t in enumerate(t_sorted)}
            frame = (
                pdf
                .with_columns(
                    (pl.col("time_int").map_elements(
                        lambda t, m=t2local: m.get(t, 0), return_dtype=pl.Int64
                    ) + x_offset).alias("_x")
                )
            )
            combined_frames.append(frame)
            boundary_x.append(x_offset)
            period_names.append(period_name)
            x_offset += n_bins_this

        combined_df = pl.concat(combined_frames, how="diagonal_relaxed")
        source_df   = combined_df          # used for top-N selection & mean
        x_col_name  = "_x"
        total_x     = x_offset
        boundary_labels = period_names
    else:
        combined_df   = presence_df
        source_df     = presence_df
        x_col_name    = None              # will use t2x dict below
        boundary_x    = []
        boundary_labels = []
        total_x       = None

    # ── Select top-N cells by mean col value (over the full combined data)
    top_hashes = (
        source_df
        .group_by("geohash")
        .agg(pl.col(col).mean().alias("_mean"))
        .sort("_mean", descending=True)
        .head(n)["geohash"]
        .to_list()
    )

    if x_col_name is None:
        # Single-period: build x mapping from time_int
        time_sorted = (
            source_df
            .select(["time_int", datetime_col] if datetime_col in source_df.columns
                    else ["time_int"])
            .unique("time_int")
            .sort("time_int")
        )
        time_ints = time_sorted["time_int"].to_list()
        if datetime_col in time_sorted.columns:
            x_labels  = time_sorted[datetime_col].to_list()
            x_numeric = list(range(len(time_ints)))
        else:
            x_numeric = time_ints
            x_labels  = [str(t) for t in time_ints]
        t2x = {t: x for x, t in zip(x_numeric, time_ints)}
    else:
        # Multi-period: x values already embedded in _x column
        x_numeric = list(range(total_x))
        x_labels  = [str(i) for i in x_numeric]
        t2x       = {}   # not used in multi-period path

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = (
        plt.get_cmap(cmap).colors
        if hasattr(plt.get_cmap(cmap), "colors")
        else [plt.get_cmap(cmap)(i / max(n - 1, 1)) for i in range(n)]
    )

    for idx, gh in enumerate(top_hashes):
        ts = (
            combined_df
            .filter(pl.col("geohash") == gh)
            .sort(x_col_name if x_col_name else "time_int")
        )
        if x_col_name:
            xs = ts[x_col_name].to_list()
        else:
            xs = [t2x[t] for t in ts["time_int"].to_list()]
        ys = ts[col].to_numpy().astype(float)
        ax.plot(
            xs, ys,
            color=colors[idx % len(colors)],
            alpha=alpha,
            linewidth=1.4,
            label=gh,
        )

    if show_mean:
        mean_df = (
            combined_df
            .group_by(x_col_name if x_col_name else "time_int")
            .agg(pl.col(col).mean().alias("_mean"))
            .sort(x_col_name if x_col_name else "time_int")
        )
        xm_col = x_col_name if x_col_name else "time_int"
        xm = [t2x.get(t, t) if x_col_name is None else t
              for t in mean_df[xm_col].to_list()]
        ym = mean_df["_mean"].to_numpy().astype(float)
        ax.plot(
            xm, ym,
            color="black", linewidth=2.0, linestyle="--",
            alpha=0.6, label="all-cell mean", zorder=10,
        )

    # ── Period boundary markers (multi-period only)
    for bx, bname in zip(boundary_x, boundary_labels):
        if bx > 0:
            ax.axvline(bx, color="grey", linewidth=1.2,
                       linestyle=":", alpha=0.7, zorder=5)
        ax.text(
            bx + 0.5, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0.98,
            bname, fontsize=7, color="grey", va="top", ha="left",
            transform=ax.get_xaxis_transform(),
        )

    # ── x-axis tick labels
    max_ticks = 20
    step = max(1, len(x_numeric) // max_ticks)
    tick_pos    = x_numeric[::step]
    tick_labels = x_labels[::step]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Time bin")
    ax.set_ylabel(col)
    span_label = "Jan → Sept (all periods)" if periods is not None else ""
    ax.set_title(
        f"Top-{n} most-populated cells — '{col}' as stochastic processes"
        + (f"  [{span_label}]" if span_label else "")
    )
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.7)
    ax.grid(linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    return fig, ax
