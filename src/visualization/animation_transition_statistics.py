"""
animation_transition_statistics.py
=====================================
Animated statistical distributions from TransitionPipeline outputs.

Each function returns a :class:`matplotlib.animation.FuncAnimation` that
loops over time bins, showing how the distribution changes over time.

Display in a notebook::

    from IPython.display import HTML
    HTML(anim.to_jshtml())

Save as video / GIF::

    anim.save("transitions.gif", writer="pillow", fps=4)

Functions
---------
animate_transition_distribution(transition_df, col, *, bins, fps, figsize,
                                  color, log_scale, fixed_xlim, fixed_ylim)
    Animated histogram of per-edge *col* distribution across time bins.

animate_weight_evolution(transition_df, *, bins, fps, figsize, color,
                          log_scale)
    Animated histogram of ``transition_probability`` per time bin.

animate_population_distribution(presence_df, col, *, bins, fps, figsize,
                                  color, log_scale, fixed_xlim, fixed_ylim)
    Animated histogram of per-cell *col* distribution across time bins.

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
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bin_edges(values: np.ndarray, bins: int, log_scale: bool) -> np.ndarray:
    """Compute histogram bin edges, optionally on a log scale."""
    v_min = values.min()
    v_max = values.max()
    if v_min <= 0 and log_scale:
        v_min = values[values > 0].min() if (values > 0).any() else 1e-9
    if log_scale:
        return np.logspace(np.log10(v_min), np.log10(v_max + 1e-12), bins + 1)
    return np.linspace(v_min, v_max, bins + 1)


# ---------------------------------------------------------------------------
# Transition distribution animation
# ---------------------------------------------------------------------------

def animate_transition_distribution(
    transition_df: pl.DataFrame,
    col: str = "transitions",
    *,
    bins: int = 40,
    fps: float = 2.0,
    figsize: tuple[float, float] = (8, 4),
    color: str = "darkorange",
    log_scale: bool = False,
    fixed_xlim: tuple[float, float] | None = None,
    fixed_ylim: tuple[float, float] | None = None,
    title_prefix: str = "",
) -> FuncAnimation:
    """
    Animated histogram of per-edge *col* values across time bins.

    Shows how the distribution of transition counts (or probabilities)
    evolves over time.

    Parameters
    ----------
    transition_df : pl.DataFrame
        Transition matrix.
    col : str
        Column to histogram per time bin, e.g. ``"transitions"`` or
        ``"transition_probability"``.
    bins : int
        Number of histogram bins.
    fps : float
        Frames per second.
    figsize : tuple
    color : str
    log_scale : bool
        Apply a log scale to the x-axis.
    fixed_xlim : (float, float), optional
        Fixed x-axis limits for all frames (recommended for comparability).
    fixed_ylim : (float, float), optional
        Fixed y-axis limits.
    title_prefix : str

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    time_bins = sorted(transition_df["time_int"].unique().to_list())
    if not time_bins:
        raise ValueError("DataFrame has no time_int values.")

    # Pre-compute global bin edges
    all_vals = transition_df[col].to_numpy().astype(float)
    edges = _bin_edges(all_vals, bins, log_scale)

    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)

    def _draw_frame(t: int) -> None:
        ax.cla()
        vals = (
            transition_df
            .filter(pl.col("time_int") == t)[col]
            .to_numpy()
            .astype(float)
        )
        ax.hist(
            vals, bins=edges, color=color, alpha=0.8, edgecolor="white"
        )
        if log_scale:
            ax.set_xscale("log")
        if fixed_xlim is not None:
            ax.set_xlim(*fixed_xlim)
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)

        if "datetime" in transition_df.columns:
            dt_vals = transition_df.filter(pl.col("time_int") == t)["datetime"]
            dt_str  = dt_vals[0] if dt_vals.len() > 0 else str(t)
        else:
            dt_str = str(t)

        prefix = f"{title_prefix}  " if title_prefix else ""
        ax.set_title(
            f"{prefix}Distribution of '{col}'  [bin {t}  |  {dt_str}]",
            fontsize=11,
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Number of edges")
        ax.grid(axis="y", linewidth=0.5, alpha=0.4)
        ax.text(
            0.98, 0.95,
            f"n = {len(vals):,}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
        )
        fig.tight_layout()

    interval_ms = int(1000.0 / fps)
    return FuncAnimation(
        fig,
        _draw_frame,
        frames=time_bins,
        interval=interval_ms,
        blit=False,
    )


# ---------------------------------------------------------------------------
# Weight (transition_probability) distribution animation
# ---------------------------------------------------------------------------

def animate_weight_evolution(
    transition_df: pl.DataFrame,
    *,
    bins: int = 40,
    fps: float = 2.0,
    figsize: tuple[float, float] = (8, 4),
    color: str = "mediumpurple",
    log_scale: bool = False,
    title_prefix: str = "",
) -> FuncAnimation:
    """
    Animated histogram of ``transition_probability`` per time bin.

    Equivalent to :func:`animate_transition_distribution` called with
    ``col="transition_probability"``, provided for convenience with
    sensible defaults (log scale off, purple colour).

    Parameters
    ----------
    transition_df : pl.DataFrame
    bins : int
    fps : float
    figsize : tuple
    color : str
    log_scale : bool
    title_prefix : str

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    return animate_transition_distribution(
        transition_df,
        col="transition_probability",
        bins=bins,
        fps=fps,
        figsize=figsize,
        color=color,
        log_scale=log_scale,
        fixed_xlim=(0.0, 1.0),
        title_prefix=title_prefix,
    )


# ---------------------------------------------------------------------------
# Population distribution animation
# ---------------------------------------------------------------------------

def animate_population_distribution(
    presence_df: pl.DataFrame,
    col: str = "count",
    *,
    bins: int = 40,
    fps: float = 2.0,
    figsize: tuple[float, float] = (8, 4),
    color: str = "steelblue",
    log_scale: bool = False,
    fixed_xlim: tuple[float, float] | None = None,
    fixed_ylim: tuple[float, float] | None = None,
    title_prefix: str = "",
) -> FuncAnimation:
    """
    Animated histogram of per-cell *col* values across time bins.

    Shows how the distribution of population (or presence probability)
    across geohash cells evolves over time.

    Parameters
    ----------
    presence_df : pl.DataFrame
        Presence matrix.
    col : str
        Column to histogram per time bin, e.g. ``"count"``,
        ``"count_transit"``, ``"probability"``.
    bins : int
    fps : float
    figsize : tuple
    color : str
    log_scale : bool
    fixed_xlim, fixed_ylim : (float, float), optional
        Fixed axis limits (recommended for frame-to-frame comparability).
    title_prefix : str

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    time_bins = sorted(presence_df["time_int"].unique().to_list())
    if not time_bins:
        raise ValueError("DataFrame has no time_int values.")

    all_vals = presence_df[col].to_numpy().astype(float)
    edges = _bin_edges(all_vals, bins, log_scale)

    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)

    def _draw_frame(t: int) -> None:
        ax.cla()
        vals = (
            presence_df
            .filter(pl.col("time_int") == t)[col]
            .to_numpy()
            .astype(float)
        )
        ax.hist(
            vals, bins=edges, color=color, alpha=0.8, edgecolor="white"
        )
        if log_scale:
            ax.set_xscale("log")
        if fixed_xlim is not None:
            ax.set_xlim(*fixed_xlim)
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)

        if "datetime" in presence_df.columns:
            dt_vals = presence_df.filter(pl.col("time_int") == t)["datetime"]
            dt_str  = dt_vals[0] if dt_vals.len() > 0 else str(t)
        else:
            dt_str = str(t)

        prefix = f"{title_prefix}  " if title_prefix else ""
        ax.set_title(
            f"{prefix}Cell distribution of '{col}'  [bin {t}  |  {dt_str}]",
            fontsize=11,
        )
        ax.set_xlabel(f"{col} per cell")
        ax.set_ylabel("Number of cells")
        ax.grid(axis="y", linewidth=0.5, alpha=0.4)
        ax.text(
            0.98, 0.95,
            f"n = {len(vals):,} cells",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
        )
        fig.tight_layout()

    interval_ms = int(1000.0 / fps)
    return FuncAnimation(
        fig,
        _draw_frame,
        frames=time_bins,
        interval=interval_ms,
        blit=False,
    )
