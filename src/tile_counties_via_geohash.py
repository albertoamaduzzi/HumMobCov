"""
tile_counties_via_geohash.py  (compatibility shim)
===================================================
The implementation has moved to ``src/geometry/tile_counties_via_geohash.py``.
This module re-exports everything so that existing imports continue to work.
"""
# ruff: noqa: F401,F403
from src.geometry.tile_counties_via_geohash import *  # noqa: F401,F403
from src.geometry.tile_counties_via_geohash import tile_counties_via_geohash  # noqa: F401
