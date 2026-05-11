"""
datasets.py
===========
Dataset configuration classes and the per-file preprocessing wrapper.

``DataSet_California`` and ``DataSet_Massachusets`` are configuration objects
that bundle all paths, parameters, and metadata for a given region.
``dataset_info`` wraps a single parquet file, handling spatial/temporal
filtering and user selection per period.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from collections import defaultdict
from shapely.geometry import Point, Polygon

from .constants import (
    LIST_FILES_MA,
    CENSUS_FILES,
    DIR_RAW_DATA_CA,
    DIR_RAW_DATA_MA,
    DIR_OUTPUT,
    DIR_MILESTONES_SERVER,
    PERIOD_NAMES,
    PERIOD_DIVISION,
    MIN_POINTS_PER_USER,
    TIME_THRESHOLD_HOURS,
    US_BOUNDING_BOX,
)
from .utils import ifnotexistsmkdir


# ---------------------------------------------------------------------------
# Base dataset class
# ---------------------------------------------------------------------------

class _BaseDataset:
    """
    Shared initialisation logic for region-specific dataset classes.

    Subclasses set ``self.id_``, ``self.dir``, and ``self.list_files``
    before calling ``_init_common()``.
    """

    def _init_common(self):
        """Load census data, initialise counters and period mappings."""
        self._init_county2rural()
        self.county2party = self._init_county2party()
        self.geojson      = gpd.read_file(str(CENSUS_FILES[self.id_]["geojson"]))

        # Time periods
        self.period_names    = PERIOD_NAMES
        self.period_division = PERIOD_DIVISION
        self.period_names2period_division = {
            PERIOD_NAMES[p]: [PERIOD_DIVISION[p], PERIOD_DIVISION[p + 1]]
            for p in range(len(PERIOD_NAMES))
        }
        self.perodname2idx = {p: i for i, p in enumerate(PERIOD_NAMES)}

        # Preprocessing parameters
        self.np_         = MIN_POINTS_PER_USER
        self.t_threshold = TIME_THRESHOLD_HOURS

        # Bounding box
        self.bounding_box = US_BOUNDING_BOX

        # Counters (updated by the pipeline)
        self.period2totalusers  = {p: 0 for p in PERIOD_NAMES}
        self.period2totalpoints = {p: 0 for p in PERIOD_NAMES}

        # Output directory — mirrors the old pipeline's convention:
        #   milestones_analysis/<region>/dataxuser/
        self.dir_output = ifnotexistsmkdir(DIR_MILESTONES_SERVER / self.id_ / "dataxuser")

    # ------------------------------------------------------------------
    # Census helpers
    # ------------------------------------------------------------------

    def _init_county2rural(self) -> None:
        """Build ``self.county2rural = {county_name: rurality_type}``."""
        path = CENSUS_FILES[self.id_]["urban_info"]
        sep  = "," if self.id_ == "CA" else ";"
        df   = pd.read_csv(str(path), sep=sep, index_col=False)
        # Column names differ by region: CA uses 'name'/'type_marta', MA uses 'NAMELSAD'/'RURALITY'
        if "name" in df.columns:
            col_name, col_type = "name", "type_marta"
        else:
            col_name, col_type = "NAMELSAD", "RURALITY"
        self.county2rural  = {df.iloc[i][col_name]: df.iloc[i][col_type] for i in range(len(df))}
        self.df_rurality   = df

    def _init_county2party(self) -> dict:
        """Build ``{county_name: party}`` from the political CSV."""
        path = CENSUS_FILES[self.id_]["party_county"]
        if not Path(str(path)).exists():
            self.df_party = pd.DataFrame()
            return {}
        df   = pd.read_csv(str(path), sep=";", index_col=False)
        self.df_party = df
        return {
            col: df[col].iloc[0]
            for col in df.columns
            if col != "Unnamed: 0"
        }


# ---------------------------------------------------------------------------
# Region-specific dataset classes
# ---------------------------------------------------------------------------

class DataSet_California(_BaseDataset):
    """
    Configuration and metadata for the California Cuebiq dataset.

    Attributes
    ----------
    id_ : str
        ``"CA"``
    dir : Path
        Path to the directory containing raw parquet shards.
    dir_files : list[str]
        Full paths to every parquet file in ``dir``.
    """

    def __init__(self):
        self.id_  = "CA"
        self.dir  = DIR_RAW_DATA_CA
        self.list_files = [
            str(self.dir / f)
            for f in os.listdir(str(self.dir))
            if f.endswith(".parquet")
        ] if self.dir.exists() else []
        self.dir_files = self.list_files
        self._init_common()


class DataSet_Massachusets(_BaseDataset):
    """
    Configuration and metadata for the Massachusetts Cuebiq dataset.

    Attributes
    ----------
    id_ : str
        ``"MA"``
    dir : Path
        Path to the directory containing raw parquet shards.
    dir_files : list[str]
        Full paths to the fixed-name parquet shards.
    """

    def __init__(self):
        self.id_  = "MA"
        self.dir  = DIR_RAW_DATA_MA
        self.list_files = LIST_FILES_MA
        self.dir_files  = [str(self.dir / f) for f in LIST_FILES_MA]
        self._init_common()


# ---------------------------------------------------------------------------
# Per-file preprocessing
# ---------------------------------------------------------------------------

class dataset_info:
    """
    Wraps a single parquet file and applies spatial + temporal preprocessing.

    Parameters
    ----------
    file : str or Path
        Path to the parquet file.
    period_division : list[datetime.datetime]
        Sorted list of period boundary datetimes.
    period_names : list[str]
        Names for each period (len == len(period_division) - 1).
    np_ : int
        Minimum number of stop points per user per period.
    t_threshold : int
        Minimum inter-stop gap in hours.
    bounding_box : list[tuple]
        Four corner coordinates ``(lat, lon)`` defining the spatial ROI.
    """

    def __init__(
        self,
        file: str | Path,
        period_division: list,
        period_names: list,
        np_: int,
        t_threshold: int,
        bounding_box: list,
    ):
        self.df              = pd.read_parquet(str(file))
        self.period_division = period_division
        self.period_names    = period_names
        self.np_             = np_
        self.t_threshold     = t_threshold
        self.bounding_box    = bounding_box

        if len(period_names) != len(period_division) - 1:
            raise ValueError(
                "period_names length must equal len(period_division) - 1"
            )

        self.period_names2period_division = {
            period_names[p]: [period_division[p], period_division[p + 1]]
            for p in range(len(period_names))
        }
        self.perodname2idx = {p: i for i, p in enumerate(period_names)}

        self.total_number_points           = len(self.df)
        self.total_number_users            = len(self.df.groupby("userId"))
        self.total_numbers_of_points_after_filter = 0
        self.total_number_users_after_filter      = 0

        self.period2info      = {p: defaultdict() for p in period_names}
        self.period2df        = defaultdict()
        self.period2listusers = defaultdict()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def spatial_filtering_per_country(self) -> None:
        """
        Keep only rows whose (clusterLatitude, clusterLongitude) falls
        inside ``self.bounding_box``.
        """
        coords   = list(zip(self.df["clusterLatitude"], self.df["clusterLongitude"]))
        points   = [Point(ll) for ll in coords]
        polygon  = Polygon(self.bounding_box)
        mask     = [p.within(polygon) for p in points]
        self.df  = self.df.loc[mask]

    def preprocess(self) -> None:
        """
        Apply spatial filtering, then split the DataFrame by period and
        build ``period2listusers`` containing only users with >= np_ stops.

        Sets
        ----
        self.period2df : dict
        self.period2listusers : dict
        """
        self.spatial_filtering_per_country()
        for p_idx in range(len(self.period_names)):
            period = self.period_names[p_idx]
            tmp = self.df.loc[
                (self.df["begin"] > self.period_division[p_idx]) &
                (self.df["begin"] < self.period_division[p_idx + 1])
            ]
            self.period2df[period] = tmp
            size_series = tmp.groupby("userId").size().sort_values()
            self.period2listusers[period] = size_series[size_series > self.np_].index

    # ------------------------------------------------------------------
    # Comparison utility
    # ------------------------------------------------------------------

    def compare_userlist_among_periods(self) -> dict:
        """
        Print and return pairwise user overlap statistics between periods.
        """
        from .utils import init_compare_periods_dict
        tmp = init_compare_periods_dict(
            self.period2listusers, ["intersection", "difference1", "difference2"], "scalar"
        )
        for k in self.period2listusers:
            for k1 in self.period2listusers:
                if k1 > k:
                    key = f"{k}-{k1}"
                    s_k  = set(self.period2listusers[k])
                    s_k1 = set(self.period2listusers[k1])
                    tmp[key]["intersection"] = len(s_k & s_k1)
                    tmp[key]["difference1"]  = len(s_k - s_k1)
                    tmp[key]["difference2"]  = len(s_k1 - s_k)
                    print(f"Users in both {k} // {k1}:  {tmp[key]['intersection']}")
                    print(f"Users only in {k}:           {tmp[key]['difference1']}")
                    print(f"Users only in {k1}:          {tmp[key]['difference2']}")
        return tmp
