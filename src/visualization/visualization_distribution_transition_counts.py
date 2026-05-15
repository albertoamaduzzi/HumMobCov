"""
visualization_distribution_transition_counts.py
================================================
Static and animated visualisations of the distribution DataFrames produced by
:mod:`transition_matrices.distribution_transitions_and_presences`.

Static plots
------------
plot_period_aggregate_distribution(dist_df, col, *, bins, figsize, ax)
    For a given source column, plot the mean ``count_{col}`` over all time
    bins, one line per period.  X-axis = bin left edge, Y-axis = mean count.

Animated plots
--------------
animate_distribution_evolution(dist_df, col, *, fps, figsize, window,
                                log_scale, title_prefix)
    Sweeps over time bins.  At each frame shows the distribution bar chart
    and its moving-average overlay.  One sub-plot per period.

Expected DataFrame schema
--------------------------
Output of ``compute_presence_distribution`` or
``compute_transition_distribution``::

    time_bin            Int64
    period_observation  Utf8
    bin_{col}           Float64   ← left edge of each histogram bin
    count_{col}         Int64     ← number of elements in that bin

See ``readme_transition_distribution_evolution.md`` for the mathematics.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge constant-padding."""
    if window <= 1:
        return arr.copy()
    half = window // 2
    padded = np.pad(arr, (half, half), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def _sorted_periods(periods: list[str]) -> list[str]:
    """
    Return *periods* in chronological order if they match the standard
    HumMobCov period names, otherwise return in the received order.
    """
    order = ["15 jan - 15 march", "15 march - 15 may", "15 may - sept"]
    known = [p for p in order if p in periods]
    unknown = [p for p in periods if p not in order]
    return known + unknown


# ---------------------------------------------------------------------------
# Static: period-aggregate distribution
# ---------------------------------------------------------------------------

def plot_period_aggregate_distribution(
    dist_df: pl.DataFrame,
    col: str,
    *,
    figsize: tuple[float, float] = (10, 4),
    log_x: bool = False,
    log_y: bool = False,
    colors: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the mean ``count_{col}`` (averaged over all time bins) per period.

    Each period produces one line/step curve.  The x-axis uses the bin left
    edges stored in ``bin_{col}``.

    This corresponds to $\\bar{X}^{(c)}_P$ in the README mathematics:

    .. math::

        \\bar{X}^{(c)}_P = \\frac{1}{|P|} \\sum_{t \\in P} X^{(c)}_{t}

    Parameters
    ----------
    dist_df : pl.DataFrame
        Output of ``compute_presence_distribution`` or
        ``compute_transition_distribution``.  Must contain
        ``bin_{col}`` and ``count_{col}`` columns.
    col : str
        Source column name, e.g. ``"count"``, ``"transitions"``,
        ``"transition_probability"``.
    figsize : tuple
    log_x : bool
        Log scale on x-axis.
    log_y : bool
        Log scale on y-axis.
    colors : list of str, optional
        One colour per period (cycled if fewer than periods).
    ax : plt.Axes, optional

    Returns
    -------
    (fig, ax)
    """
    bin_col   = f"bin_{col}"
    count_col = f"count_{col}"

    if bin_col not in dist_df.columns or count_col not in dist_df.columns:
        raise ValueError(
            f"Columns '{bin_col}' and/or '{count_col}' not found in dist_df. "
            f"Available: {dist_df.columns}"
        )

    periods = _sorted_periods(dist_df["period_observation"].unique().to_list())
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    if colors is None:
        colors = default_colors

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for i, period in enumerate(periods):
        sub = dist_df.filter(pl.col("period_observation") == period)

        # Average count_{col} over all time bins, grouped by bin edge
        agg = (
            sub
            .group_by(bin_col)
            .agg(pl.col(count_col).mean().alias(count_col))
            .sort(bin_col)
        )

        x = agg[bin_col].to_numpy().astype(float)
        y = agg[count_col].to_numpy().astype(float)

        color = colors[i % len(colors)]
        ax.step(x, y, where="post", color=color, linewidth=1.8,
                label=period, alpha=0.85)
        ax.fill_between(x, y, step="post", color=color, alpha=0.18)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(f"Bin left edge  ({col})")
    ax.set_ylabel(f"Mean count per bin")
    ax.set_title(f"Period-aggregate distribution of '{col}'")
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Static: all columns for a single period
# ---------------------------------------------------------------------------

def plot_all_columns_for_period(
    dist_df: pl.DataFrame,
    period: str,
    cols: list[str],
    *,
    figsize: tuple[float, float] | None = None,
    log_x: bool = False,
    log_y: bool = False,
    color: str = "steelblue",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Grid of period-aggregate distribution plots — one sub-plot per column.

    Parameters
    ----------
    dist_df : pl.DataFrame
    period : str
        Which period to plot.
    cols : list of str
        Source column names to include.
    figsize : tuple, optional
        Defaults to ``(5 * len(cols), 4)``.
    log_x, log_y : bool
    color : str

    Returns
    -------
    (fig, axes)
    """
    n = len(cols)
    if figsize is None:
        figsize = (5 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    sub = dist_df.filter(pl.col("period_observation") == period)

    for ax, col in zip(axes, cols):
        bin_col   = f"bin_{col}"
        count_col = f"count_{col}"
        if bin_col not in sub.columns:
            ax.set_title(f"'{col}' not available")
            continue

        agg = (
            sub
            .group_by(bin_col)
            .agg(pl.col(count_col).mean().alias(count_col))
            .sort(bin_col)
        )
        x = agg[bin_col].to_numpy().astype(float)
        y = agg[count_col].to_numpy().astype(float)

        ax.step(x, y, where="post", color=color, linewidth=1.6)
        ax.fill_between(x, y, step="post", color=color, alpha=0.2)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("Mean count")
        ax.set_title(f"{col}\n({period})")
        ax.grid(linewidth=0.35, alpha=0.3)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Animation: distribution evolution across time bins
# ---------------------------------------------------------------------------

def animate_distribution_evolution(
    dist_df: pl.DataFrame,
    col: str,
    *,
    fps: float = 3.0,
    figsize: tuple[float, float] | None = None,
    window: int = 5,
    log_x: bool = False,
    log_y: bool = False,
    colors: list[str] | None = None,
    title_prefix: str = "",
    max_frames: int | None = 100,
) -> FuncAnimation:
    """
    Animate the evolution of the distribution of ``col`` across time bins.

    At each frame (one time bin) the plot shows:

    * **bars** — the instantaneous distribution $X^{(c)}_t$
    * **solid line** — the centred moving average $\\widetilde{X}^{(c)}_t$
      (window width *window*)

    One sub-plot is created per period present in *dist_df*, so you can
    compare the three COVID phases side-by-side.

    Parameters
    ----------
    dist_df : pl.DataFrame
        Output of ``compute_presence_distribution`` or
        ``compute_transition_distribution``.
    col : str
        Source column, e.g. ``"count"``, ``"transitions"``,
        ``"transition_probability"``.
    fps : float
        Animation frame rate.
    figsize : tuple, optional
        Defaults to ``(6 * n_periods, 4)``.
    window : int
        Moving-average window width in time bins.
    log_x : bool
    log_y : bool
    colors : list of str, optional
        One colour per period.
    title_prefix : str
    max_frames : int or None
        Maximum number of animation frames *per period*.  When the number of
        unique time bins exceeds *max_frames*, the bins are evenly subsampled
        so the total frame count equals *max_frames*.  Set to ``None`` to use
        all time bins (may be very slow for large datasets).  Default: 100.

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Notes
    -----
    The moving average is computed once up-front over the full time series
    for each bin index, then indexed per frame — O(T × B) pre-computation.
    """
    bin_col   = f"bin_{col}"
    count_col = f"count_{col}"

    if bin_col not in dist_df.columns or count_col not in dist_df.columns:
        raise ValueError(
            f"Columns '{bin_col}' / '{count_col}' not found. "
            f"Available: {dist_df.columns}"
        )

    periods = _sorted_periods(dist_df["period_observation"].unique().to_list())
    n_periods = len(periods)
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    if colors is None:
        colors = default_colors

    if figsize is None:
        figsize = (6 * n_periods, 4)

    # --- Pre-build per-period arrays (shape: T × B) ---
    # All periods share the same set of time bins (outer join)
    all_time_bins: list[int] = sorted(dist_df["time_bin"].unique().to_list())
    bins_count = dist_df.filter(pl.col("period_observation") == periods[0])["time_bin"].n_unique()
    # actual number of histogram bins
    n_bins_hist = dist_df.filter(
        (pl.col("period_observation") == periods[0]) &
        (pl.col("time_bin") == all_time_bins[0])
    ).height

    per_period: dict[str, dict] = {}
    for period in periods:
        sub = dist_df.filter(pl.col("period_observation") == period)
        t_vals = sorted(sub["time_bin"].unique().to_list())

        # Subsample if max_frames is set
        if max_frames is not None and len(t_vals) > max_frames:
            step = max(1, len(t_vals) // max_frames)
            t_vals = t_vals[::step][:max_frames]

        # Matrix: rows = time bins (in order), cols = bin indices
        mat = np.zeros((len(t_vals), n_bins_hist), dtype=float)
        bin_edges_arr = np.zeros(n_bins_hist, dtype=float)

        for ti, t in enumerate(t_vals):
            row = sub.filter(pl.col("time_bin") == t).sort(bin_col)
            if row.height == 0:
                continue
            mat[ti] = row[count_col].to_numpy().astype(float)
            if ti == 0:
                bin_edges_arr = row[bin_col].to_numpy().astype(float)

        # Pre-compute moving average for each bin dimension
        ma_mat = np.stack(
            [_moving_average(mat[:, b], window) for b in range(n_bins_hist)],
            axis=1,
        )

        per_period[period] = {
            "time_bins": t_vals,
            "mat": mat,
            "ma_mat": ma_mat,
            "bin_edges": bin_edges_arr,
        }

    # --- Figure setup ---
    fig, axes = plt.subplots(1, n_periods, figsize=figsize, sharey=False)
    if n_periods == 1:
        axes = [axes]

    # Compute global y-max for sharey option
    global_ymax = max(
        per_period[p]["mat"].max() for p in periods
        if per_period[p]["mat"].size > 0
    ) * 1.05

    bar_containers: list = []
    ma_lines: list = []
    time_texts: list = []

    for ax, period, color in zip(axes, periods, colors):
        data = per_period[period]
        edges = data["bin_edges"]
        if edges.size == 0:
            bar_containers.append(None)
            ma_lines.append(None)
            time_texts.append(None)
            continue

        widths = np.diff(
            np.append(edges, edges[-1] + (edges[-1] - edges[-2]) if len(edges) > 1 else edges[-1] + 1)
        )
        first_counts = data["mat"][0] if data["mat"].shape[0] > 0 else np.zeros(len(edges))

        bars = ax.bar(edges, first_counts, width=widths, align="edge",
                      color=color, alpha=0.65, edgecolor="none")
        (line,) = ax.plot(edges, data["ma_mat"][0] if data["ma_mat"].shape[0] > 0
                          else np.zeros(len(edges)),
                          color="black", linewidth=2, label=f"MA(w={window})")

        ax.set_xlim(edges[0], edges[-1] + widths[-1])
        ax.set_ylim(0, global_ymax)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(period)
        ax.legend(fontsize=8)
        ax.grid(linewidth=0.35, alpha=0.3)

        txt = ax.text(
            0.02, 0.96, f"t=0",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

        bar_containers.append(bars)
        ma_lines.append(line)
        time_texts.append(txt)

    title_str = f"{title_prefix}  |  '{col}'" if title_prefix else f"'{col}'"
    fig.suptitle(title_str, fontsize=11)
    fig.tight_layout()

    # --- Animation update ---
    def _update(frame_idx: int):
        artists = []
        for period, bars, line, txt in zip(periods, bar_containers, ma_lines, time_texts):
            if bars is None:
                continue
            data = per_period[period]
            t_list = data["time_bins"]
            # map global frame index to per-period frame index
            p_idx = min(frame_idx, len(t_list) - 1)
            counts = data["mat"][p_idx]
            ma     = data["ma_mat"][p_idx]
            t_val  = t_list[p_idx]

            for bar, h in zip(bars, counts):
                bar.set_height(h)
            line.set_ydata(ma)
            txt.set_text(f"t={t_val}")
            artists.extend(list(bars) + [line, txt])
        return artists

    n_frames = max(len(per_period[p]["time_bins"]) for p in periods)
    interval_ms = int(1000 / fps)

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=interval_ms,
        blit=True,
    )
    return anim
