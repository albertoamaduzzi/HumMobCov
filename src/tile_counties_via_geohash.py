"""
tile_counties_via_geohash.py
============================
Build a geohash grid covering a county (or any polygon) at a given precision.

The geohash encoding is **hierarchical**: a geohash of length N is always a
sub-cell of the geohash of length N-1 formed by its first N-1 characters.
This means that trajectory geohashes at a finer precision (e.g. 7) can be
coarse-grained to any lower precision by simply truncating the string.

Recommended precision for CA / MA with 32 GB RAM
-------------------------------------------------
| Precision | Cell size (approx.) | # cells over CA (~424k km²) |
|-----------|--------------------|-----------------------------|
| 4         | 39 km × 20 km      | ~540                        |
| 5         | 5 km × 5 km        | ~17 000                     |
| 6         | 1.2 km × 0.6 km    | ~590 000                    |

For transition matrices that need to be manageable in memory,
**precision 5** is the recommended default for both CA and MA.

Usage
-----
>>> import geopandas as gpd
>>> from src.tile_counties_via_geohash import tile_counties_via_geohash
>>> gdf_county = gpd.read_file("census_data/California/geometry_census_new.geojson")
>>> grid = tile_counties_via_geohash(gdf_county, precision=5)
>>> grid.head()
   geohash                                           geometry  area_km2
0    9q5c3  POLYGON ((-122.34375 37.265625, -122.34375 37....      ...
"""

from __future__ import annotations

import geohash as _gh
from shapely.geometry import Polygon, Point, box as _box
import geopandas as gpd
import pandas as pd
from typing import Iterable


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def geohash_to_polygon(gh: str) -> Polygon:
    """
    Return a Shapely Polygon for the bounding box of *gh*.

    Parameters
    ----------
    gh : str
        A geohash string of any precision.

    Returns
    -------
    Polygon
        Rectangular cell in geographic coordinates (lon/lat, EPSG:4326).
    """
    b = _gh.bbox(gh)
    return Polygon([
        (b["w"], b["s"]),
        (b["w"], b["n"]),
        (b["e"], b["n"]),
        (b["e"], b["s"]),
        (b["w"], b["s"]),
    ])


def coarsen_geohash(gh: str, precision: int) -> str:
    """
    Coarsen a fine-grained geohash to *precision* characters.

    Because geohash is hierarchical, this is simply ``gh[:precision]``.

    Parameters
    ----------
    gh : str
        Source geohash (length ≥ precision).
    precision : int
        Target precision (number of characters to keep).

    Returns
    -------
    str
        Coarsened geohash.

    Examples
    --------
    >>> coarsen_geohash("9q8yy96", 5)
    '9q8yy'
    """
    if len(gh) < precision:
        raise ValueError(
            f"Geohash '{gh}' is shorter than target precision {precision}."
        )
    return gh[:precision]


def coarsen_geohash_series(series: pd.Series, precision: int) -> pd.Series:
    """
    Vectorised coarsening of a pandas Series of geohash strings.

    Parameters
    ----------
    series : pd.Series[str]
        Column of geohash strings.
    precision : int
        Target precision.

    Returns
    -------
    pd.Series[str]
        Coarsened geohashes.
    """
    return series.str[:precision]


# ---------------------------------------------------------------------------
# Core tiling function
# ---------------------------------------------------------------------------

def tile_counties_via_geohash(
    county_gdf: gpd.GeoDataFrame,
    precision: int = 5,
    crs_metric: str = "EPSG:3857",
    include_partial: bool = True,
) -> gpd.GeoDataFrame:
    """
    Tile a county (or any collection of polygons) with a geohash grid.

    The function uses a BFS expansion starting from the centroid of the
    union of all geometries in *county_gdf*.  It enumerates all geohash
    cells whose *centre point* falls inside the union polygon (when
    ``include_partial=False``) or whose bounding box intersects it (when
    ``include_partial=True``).

    Parameters
    ----------
    county_gdf : GeoDataFrame
        Shapefile / GeoJSON of the county or counties to tile.
        Must be readable by GeoPandas (any CRS — it is reprojected
        internally).
    precision : int
        Geohash precision (number of characters).  Default 5.
        Recommended: 5 for CA/MA with 32 GB RAM.
    crs_metric : str
        EPSG code of a metric CRS used for area computation.
        Default ``"EPSG:3857"`` (Web Mercator, metres).
    include_partial : bool
        If True (default), include cells that partially overlap the
        county boundary.  If False, only include cells whose centre
        point is strictly inside.

    Returns
    -------
    GeoDataFrame
        One row per geohash cell with columns:

        * ``geohash``  — geohash string (length == *precision*)
        * ``geometry`` — Shapely Polygon of the cell in EPSG:4326
        * ``area_km2`` — area of the cell in km² (computed in *crs_metric*)

    Notes
    -----
    The BFS explores all eight neighbours of each valid cell.  Cells
    already visited or that fall outside the county bounding box are
    discarded immediately to keep the frontier small.
    """
    # ── 1. Reproject to geographic CRS for geohash operations ───────────────
    county_wgs84 = county_gdf.to_crs("EPSG:4326")
    union_poly   = county_wgs84.union_all()  # single polygon covering all counties

    # Bounding box of the union polygon (for fast rejection)
    minx, miny, maxx, maxy = union_poly.bounds

    # ── 2. BFS starting from centroid ────────────────────────────────────────
    centroid  = union_poly.centroid
    seed_gh   = _gh.encode(centroid.y, centroid.x, precision=precision)

    visited:  set[str] = set()
    frontier: list[str] = [seed_gh]
    valid_cells: list[str] = []

    def _is_in_county(gh: str) -> bool:
        """Return True if the cell overlaps or is inside the county."""
        b = _gh.bbox(gh)
        # Fast bounding-box pre-rejection
        if b["e"] < minx or b["w"] > maxx or b["n"] < miny or b["s"] > maxy:
            return False
        cell_box = _box(b["w"], b["s"], b["e"], b["n"])
        if include_partial:
            return union_poly.intersects(cell_box)
        # Strict: centre must be inside the union polygon
        centre = Point((b["w"] + b["e"]) / 2, (b["s"] + b["n"]) / 2)
        return union_poly.contains(centre)

    while frontier:
        current = frontier.pop()
        if current in visited:
            continue
        visited.add(current)

        if _is_in_county(current):
            valid_cells.append(current)
            for neighbour in _gh.neighbors(current):
                if neighbour not in visited:
                    frontier.append(neighbour)

    # ── 3. Build GeoDataFrame ─────────────────────────────────────────────────
    if not valid_cells:
        return gpd.GeoDataFrame(
            columns=["geohash", "geometry", "area_km2"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    polygons = [geohash_to_polygon(gh) for gh in valid_cells]
    grid_gdf = gpd.GeoDataFrame(
        {"geohash": valid_cells, "geometry": polygons},
        crs="EPSG:4326",
    )

    # ── 4. Compute cell area in km² ──────────────────────────────────────────
    grid_metric   = grid_gdf.to_crs(crs_metric)
    grid_gdf["area_km2"] = grid_metric.geometry.area / 1e6   # m² → km²

    return grid_gdf[["geohash", "geometry", "area_km2"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: build grids for the whole state (CA or MA)
# ---------------------------------------------------------------------------

def tile_region_via_geohash(
    geojson_path: str,
    precision: int = 5,
    include_partial: bool = True,
) -> gpd.GeoDataFrame:
    """
    Build a geohash grid for an entire region (state) from a GeoJSON / shapefile.

    Parameters
    ----------
    geojson_path : str or Path
        Path to the region shapefile or GeoJSON
        (e.g. ``census_data/California/geometry_census_new.geojson``).
    precision : int
        Geohash precision.  Default 5.
    include_partial : bool
        Include cells that partially overlap the boundary.

    Returns
    -------
    GeoDataFrame
        Geohash grid for the full region.  Same schema as
        :func:`tile_counties_via_geohash`.
    """
    gdf = gpd.read_file(str(geojson_path))
    return tile_counties_via_geohash(
        gdf,
        precision=precision,
        include_partial=include_partial,
    )
