"""
animation_grid_transitions.py
==============================
Animated visualizations of TransitionPipeline outputs on the geohash grid.

Each function returns a :class:`matplotlib.animation.FuncAnimation` object
that loops over time bins.  You can display it in a notebook with::

    from IPython.display import HTML
    HTML(anim.to_jshtml())

or save it as a video / GIF::

    anim.save("presence.gif", writer="pillow", fps=4)

Functions
---------
animate_grid_heatmap(df, col, *, fps, figsize, cmap, vmin, vmax,
                     background_gdf)
    Animated choropleth: *col* on the geohash grid evolving over time bins.

animate_grid_network(transition_df, col, *, min_value, top_n, fps, figsize,
                     edge_cmap, edge_width_scale, node_size, background_gdf)
    Animated directed edge network evolving over time bins.

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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import polars as pl
from matplotlib.animation import FuncAnimation

from .visualization_grid_transitions import geohash_to_gdf


# ---------------------------------------------------------------------------
# Animated heatmap
# ---------------------------------------------------------------------------

def animate_grid_heatmap(
    df: pl.DataFrame,
    col: str,
    *,
    fps: float = 2.0,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    background_gdf: gpd.GeoDataFrame | None = None,
    title_prefix: str = "",
) -> FuncAnimation:
    """
    Animated choropleth of *col* on the geohash grid over time bins.

    Parameters
    ----------
    df : pl.DataFrame
        Presence matrix (columns: ``geohash``, ``time_int``, ``datetime``,
        and *col*).
    col : str
        Column to visualise, e.g. ``"count"``, ``"probability"``,
        ``"count_transit"``.
    fps : float
        Frames per second (controls animation speed).
    figsize : tuple
    cmap : str
    vmin, vmax : float, optional
        Fixed colour-scale limits.  If None, each frame is normalised
        independently (use fixed limits for comparability across frames).
    background_gdf : gpd.GeoDataFrame, optional
        Background polygon layer drawn behind the grid.
    title_prefix : str
        Optional prefix prepended to the per-frame title.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    time_bins = sorted(df["time_int"].unique().to_list())
    if not time_bins:
        raise ValueError("DataFrame has no time_int values.")

    # Pre-build GeoDataFrame for all geohash cells
    all_hashes = df["geohash"].unique().to_list()
    base_gdf   = geohash_to_gdf(all_hashes)

    # Determine global colour scale
    _vmin = vmin if vmin is not None else float(df[col].min())
    _vmax = vmax if vmax is not None else float(df[col].max())

    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)   # prevent display before animation is ready

    def _draw_frame(t: int) -> None:
        ax.cla()
        if background_gdf is not None:
            background_gdf.plot(
                ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3
            )
        frame_data = (
            df.filter(pl.col("time_int") == t)
            .select(["geohash", col])
            .to_pandas()
        )
        gdf = base_gdf.merge(frame_data, on="geohash", how="left")
        gdf.plot(
            column=col,
            ax=ax,
            cmap=cmap,
            vmin=_vmin,
            vmax=_vmax,
            legend=True,
            legend_kwds={"label": col, "orientation": "vertical"},
            edgecolor="none",
            alpha=0.85,
            missing_kwds={"color": "#f0f0f0"},
        )
        # Try to show datetime label
        if "datetime" in df.columns:
            dt_vals = df.filter(pl.col("time_int") == t)["datetime"]
            dt_str  = dt_vals[0] if dt_vals.len() > 0 else str(t)
        else:
            dt_str = str(t)
        prefix = f"{title_prefix}  " if title_prefix else ""
        ax.set_title(f"{prefix}{col}  [bin {t}  |  {dt_str}]", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    interval_ms = int(1000.0 / fps)
    return FuncAnimation(
        fig,
        _draw_frame,
        frames=time_bins,
        interval=interval_ms,
        blit=False,
    )


# ---------------------------------------------------------------------------
# Animated network
# ---------------------------------------------------------------------------

def animate_grid_network(
    transition_df: pl.DataFrame,
    col: str = "transitions",
    *,
    min_value: float = 0.0,
    top_n: int | None = 200,
    fps: float = 2.0,
    figsize: tuple[float, float] = (10, 8),
    edge_cmap: str = "plasma",
    edge_width_scale: float = 3.0,
    node_size: float = 8.0,
    background_gdf: gpd.GeoDataFrame | None = None,
    title_prefix: str = "",
) -> FuncAnimation:
    """
    Animated directed edge network on the geohash grid over time bins.

    Parameters
    ----------
    transition_df : pl.DataFrame
        Transition matrix with columns ``geohash_start``, ``geohash_end``,
        ``time_int``, and *col*.
    col : str
        Column to encode as edge colour / width.
    min_value : float
        Discard edges where *col* ≤ *min_value*.
    top_n : int, optional
        Per-frame cap on the number of edges drawn (highest *col* first).
    fps : float
    figsize : tuple
    edge_cmap : str
    edge_width_scale : float
    node_size : float
    background_gdf : gpd.GeoDataFrame, optional
    title_prefix : str

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    time_bins = sorted(transition_df["time_int"].unique().to_list())
    if not time_bins:
        raise ValueError("DataFrame has no time_int values.")

    # Pre-build centroid lookup for all cells
    all_hashes = list(set(
        transition_df["geohash_start"].to_list()
        + transition_df["geohash_end"].to_list()
    ))
    gdf_nodes = geohash_to_gdf(all_hashes)
    centroid  = {
        row["geohash"]: (row["centroid_lon"], row["centroid_lat"])
        for _, row in gdf_nodes.iterrows()
    }

    # Global colour scale
    v_global_min = float(transition_df[col].min())
    v_global_max = float(transition_df[col].max())
    if v_global_max <= v_global_min:
        v_global_max = v_global_min + 1.0
    norm  = mcolors.Normalize(vmin=v_global_min, vmax=v_global_max)
    cmap_ = plt.get_cmap(edge_cmap)

    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)

    def _draw_frame(t: int) -> None:
        ax.cla()
        if background_gdf is not None:
            background_gdf.plot(
                ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3
            )
        else:
            gdf_nodes.plot(
                ax=ax, facecolor="none", edgecolor="#dddddd", linewidth=0.4
            )

        edges = transition_df.filter(
            (pl.col("time_int") == t) & (pl.col(col) > min_value)
        )
        if top_n is not None:
            edges = edges.sort(col, descending=True).head(top_n)

        for row in edges.iter_rows(named=True):
            s = centroid.get(row["geohash_start"])
            e = centroid.get(row["geohash_end"])
            if s is None or e is None:
                continue
            v     = float(row[col])
            ratio = (v - v_global_min) / (v_global_max - v_global_min + 1e-12)
            alpha = 0.25 + 0.75 * ratio
            width = max(0.4, edge_width_scale * ratio)
            color = cmap_(norm(v))
            ax.annotate(
                "",
                xy=e,
                xytext=s,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=width,
                    alpha=alpha,
                ),
            )

        xs = [c[0] for c in centroid.values()]
        ys = [c[1] for c in centroid.values()]
        ax.scatter(xs, ys, s=node_size, c="steelblue", zorder=5, linewidths=0)

        # Colour bar (recreated each frame so it stays accurate)
        sm = plt.cm.ScalarMappable(cmap=cmap_, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=col, shrink=0.6)

        if "datetime" in transition_df.columns:
            dt_vals = transition_df.filter(pl.col("time_int") == t)["datetime"]
            dt_str  = dt_vals[0] if dt_vals.len() > 0 else str(t)
        else:
            dt_str = str(t)
        prefix = f"{title_prefix}  " if title_prefix else ""
        ax.set_title(
            f"{prefix}{col} network  [bin {t}  |  {dt_str}]", fontsize=11
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    interval_ms = int(1000.0 / fps)
    return FuncAnimation(
        fig,
        _draw_frame,
        frames=time_bins,
        interval=interval_ms,
        blit=False,
    )
