"""
__init__.py
===========
Central import hub for the HumMobCov `src` package.

Third-party libraries used across the project are imported here with their
canonical aliases so every module can simply do::

    from src import pd, np, gpd, ...

or::

    from src import User, plotter, DataSet_California, ...

rather than repeating the same ``import`` block everywhere.
"""

# ------------------------------------------------------------------
# Standard library
# ------------------------------------------------------------------
import os
import sys
import json
import math
import time
import datetime
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations, chain, groupby

# ------------------------------------------------------------------
# Scientific / numeric
# ------------------------------------------------------------------
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Geospatial
# ------------------------------------------------------------------
import geopandas as gpd
import geohash                         # python-geohash
from shapely import *                  # noqa: F401,F403  (wildcard used in original)
from shapely.geometry import (
    Polygon, LineString, Point, MultiPolygon,
)
from shapely.wkt import loads as wkt_loads

# ------------------------------------------------------------------
# Mobility / trajectory analysis
# ------------------------------------------------------------------
import skmob
from skmob.measures.individual import (
    waiting_times,
    random_entropy,
    uncorrelated_entropy,
    real_entropy,
    home_location,
    k_radius_of_gyration,
    distance_straight_line,
    location_frequency,
    frequency_rank,
)

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
import matplotlib as mtl
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import seaborn as sns

# ------------------------------------------------------------------
# Stats / fitting
# ------------------------------------------------------------------
import powerlaw

# ------------------------------------------------------------------
# Networking
# ------------------------------------------------------------------
import networkx as nx

# ------------------------------------------------------------------
# Local path helpers (add external library directory to sys.path)
# ------------------------------------------------------------------
from .constants import DIR_LIBRARIES

if str(DIR_LIBRARIES) not in sys.path and DIR_LIBRARIES.exists():
    sys.path.insert(0, str(DIR_LIBRARIES))

try:
    from rg_histograms import cumulative, histInt, histFloat, logHist  # noqa: F401
    from rg_fits import power_fit                                        # noqa: F401
except ImportError:
    pass   # external library not available in this environment

# ------------------------------------------------------------------
# Project modules  (lazy — avoids circular import issues at load time)
# ------------------------------------------------------------------
from .constants import *          # noqa: F401,F403  expose all constants
from .utils import (              # noqa: F401
    filter_,
    xy,
    t_stop,
    time_difference,
    ifnotexistsmkdir,
    generate_pth,
    get_already_saved_user_per_period,
    init_compare_periods_dict,
    extract_dataxuser_from_shards,
)
from .User      import User       # noqa: F401
from .datasets  import (          # noqa: F401
    DataSet_California,
    DataSet_Massachusets,
    dataset_info,
)
from .pipeline  import (          # noqa: F401
    compute_all,
    analyze_from_dataset,
    analyze_from_s3_progressive,
    get_config,
)
from .vectorized_pipeline import (  # noqa: F401
    preprocess_shard_polars,
    compute_all_polars,
)
from .plotter   import plotter    # noqa: F401
from .store     import (          # noqa: F401
    ParquetStore,
)
from .gap_analysis_plots import (  # noqa: F401
    # standalone plot functions (Gap 1-4)
    plot_npi_timeline,
    plot_sampling_bias_coverage,
    plot_sampling_bias_quintiles,
    plot_party_rurality_regression,
    plot_post_lockdown_asymmetry,
    # data helper
    compute_users_per_county,
    # default NPI event dates
    NPI_EVENTS,
)

warnings.filterwarnings("ignore")
