"""
visualization_grid_transitions.py
==================================
Static visualizations of TransitionPipeline outputs on the geohash grid.

Functions
---------
geohash_to_gdf(geohashes, crs="EPSG:4326")
    Convert geohash strings to a GeoDataFrame of bounding-box polygons.

plot_grid_heatmap(df, col, *, time_int, title, figsize, cmap,
                  vmin, vmax, legend, background_gdf, ax)
    Choropleth heatmap of *col* on the geohash grid.
    When *time_int* is None the mean across all bins is shown.

plot_grid_network(transition_df, col, *, time_int, min_value, top_n,
                  title, figsize, edge_cmap, edge_width_scale,
                  node_size, background_gdf, ax)
    Draw directed transition edges on the geohash grid.  Edge colour and
    width are proportional to *col* (e.g. ``"transitions"`` or
    ``"transition_probability"``).

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

from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import polars as pl
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Geohash → GeoDataFrame helper (shared across this package)
# ---------------------------------------------------------------------------

def geohash_to_gdf(
    geohashes: list[str],
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Convert a list of geohash strings to a GeoDataFrame of bounding-box
    polygons.

    Parameters
    ----------
    geohashes : list[str]
        Geohash cell identifiers (any precision).
    crs : str
        Output coordinate reference system.  Default ``"EPSG:4326"``.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: ``geohash``, ``geometry`` (Polygon),
        ``centroid_lon``, ``centroid_lat``.
    """
    try:
        import geohash as gh
    except ImportError:
        raise ImportError(
            "python-geohash is required.  "
            "Install with:  pip install python-geohash"
        )

    records = []
    for h in geohashes:
        bb = gh.bbox(h)
        geom = box(bb["w"], bb["s"], bb["e"], bb["n"])
        cx = (bb["w"] + bb["e"]) / 2.0
        cy = (bb["s"] + bb["n"]) / 2.0
        records.append(
            {"geohash": h, "geometry": geom,
             "centroid_lon": cx, "centroid_lat": cy}
        )
    return gpd.GeoDataFrame(records, crs=crs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _agg_presence(
    df: pl.DataFrame,
    col: str,
    time_int: int | None,
) -> pl.DataFrame:
    """Return a ``geohash`` × ``col`` frame for one bin or the mean."""
    if time_int is not None:
        return df.filter(pl.col("time_int") == time_int).select(["geohash", col])
    return df.group_by("geohash").agg(pl.col(col).mean().alias(col))


def _agg_transitions(
    df: pl.DataFrame,
    col: str,
    time_int: int | None,
) -> pl.DataFrame:
    """Return edges for one bin or summed across all bins."""
    if time_int is not None:
        return df.filter(pl.col("time_int") == time_int)
    return (
        df
        .group_by(["geohash_start", "geohash_end"])
        .agg(pl.col(col).sum().alias(col))
    )


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_grid_heatmap(
    df: pl.DataFrame,
    col: str,
    *,
    time_int: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
    legend: bool = True,
    background_gdf: gpd.GeoDataFrame | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Choropleth heatmap of *col* on the geohash grid.

    Parameters
    ----------
    df : pl.DataFrame
        Presence matrix (or any DataFrame with ``geohash``, ``time_int``,
        and *col* columns).
    col : str
        Column to visualise, e.g. ``"count"``, ``"probability"``,
        ``"count_transit"``.
    time_int : int, optional
        Time-bin index.  If None, the mean across all bins is plotted.
    title : str, optional
        Figure title.
    figsize, cmap, vmin, vmax, legend
        Passed to :func:`geopandas.GeoDataFrame.plot`.
    background_gdf : gpd.GeoDataFrame, optional
        Background polygon layer (e.g. county/state boundaries) drawn
        in light grey behind the grid.
    ax : plt.Axes, optional
        Existing axes.  A new figure is created if None.

    Returns
    -------
    (fig, ax)
    """
    data = _agg_presence(df, col, time_int)
    gdf  = geohash_to_gdf(data["geohash"].to_list())
    gdf  = gdf.merge(
        data.to_pandas()[["geohash", col]],
        on="geohash",
        how="left",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if background_gdf is not None:
        background_gdf.plot(
            ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3
        )

    gdf.plot(
        column=col,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        legend=legend,
        legend_kwds={"label": col, "orientation": "vertical"},
        edgecolor="none",
        alpha=0.85,
        missing_kwds={"color": "#f0f0f0"},
    )

    bin_label = f"bin {time_int}" if time_int is not None else "mean across bins"
    ax.set_title(title or f"{col}  [{bin_label}]", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Network (edge) plot
# ---------------------------------------------------------------------------

def plot_grid_network(
    transition_df: pl.DataFrame,
    col: str = "transitions",
    *,
    time_int: int | None = None,
    min_value: float = 0.0,
    top_n: int | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    edge_cmap: str = "plasma",
    edge_width_scale: float = 3.0,
    node_size: float = 8.0,
    background_gdf: gpd.GeoDataFrame | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draw directed transition edges on the geohash-grid map.

    Each edge connects the centroids of ``geohash_start`` and
    ``geohash_end`` cells.  Colour and width are scaled by *col*.

    Parameters
    ----------
    transition_df : pl.DataFrame
        Transition matrix with columns ``geohash_start``, ``geohash_end``,
        ``time_int``, and *col*.
    col : str
        Column controlling edge colour and width.
        Use ``"transitions"`` (raw counts) or
        ``"transition_probability"`` (normalised).
    time_int : int, optional
        Time-bin index to show.  If None, *col* is summed across all bins.
    min_value : float
        Discard edges where *col* ≤ *min_value*.
    top_n : int, optional
        Keep only the *top_n* edges with the highest *col* value.
        Useful for dense grids where drawing all edges is unreadable.
    title : str, optional
    figsize : tuple
    edge_cmap : str
        Matplotlib colormap name for edge colour.
    edge_width_scale : float
        Maximum drawn line width in points.
    node_size : float
        Scatter marker size for cell centroids.
    background_gdf : gpd.GeoDataFrame, optional
        Background polygon layer.
    ax : plt.Axes, optional

    Returns
    -------
    (fig, ax)
    """
    edges = _agg_transitions(transition_df, col, time_int)
    edges = edges.filter(pl.col(col) > min_value)
    if top_n is not None:
        edges = edges.sort(col, descending=True).head(top_n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bin_label = f"bin {time_int}" if time_int is not None else "all bins summed"
    default_title = f"{col} network  [{bin_label}]"

    if edges.height == 0:
        ax.set_title(title or default_title)
        ax.text(0.5, 0.5, "No edges to display",
                ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    # Build geohash → centroid lookup
    all_hashes = list(set(
        edges["geohash_start"].to_list() + edges["geohash_end"].to_list()
    ))
    gdf      = geohash_to_gdf(all_hashes)
    centroid = {
        row["geohash"]: (row["centroid_lon"], row["centroid_lat"])
        for _, row in gdf.iterrows()
    }

    vals  = edges[col].to_numpy().astype(float)
    v_min = float(vals.min())
    v_max = float(vals.max()) if vals.max() > v_min else v_min + 1e-9
    norm  = mcolors.Normalize(vmin=v_min, vmax=v_max)
    cmap_ = plt.get_cmap(edge_cmap)

    if background_gdf is not None:
        background_gdf.plot(
            ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3
        )
    else:
        gdf.plot(ax=ax, facecolor="none", edgecolor="#cccccc", linewidth=0.4)

    # Draw directed edges
    for row in edges.iter_rows(named=True):
        s = centroid.get(row["geohash_start"])
        e = centroid.get(row["geohash_end"])
        if s is None or e is None:
            continue
        v     = float(row[col])
        ratio = (v - v_min) / (v_max - v_min + 1e-12)
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

    # Node markers
    xs = [c[0] for c in centroid.values()]
    ys = [c[1] for c in centroid.values()]
    ax.scatter(xs, ys, s=node_size, c="steelblue", zorder=5, linewidths=0)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap_, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=col, shrink=0.6)

    ax.set_title(title or default_title, fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax
