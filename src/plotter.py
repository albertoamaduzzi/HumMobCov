"""
plotter.py
==========
Aggregation and visualisation of per-user mobility metrics.

The ``plotter`` class reads the compressed per-user files produced by the
pipeline, aggregates them, and produces publication-quality figures.
All plots are saved under ``output/<region>/plots/``.

Parquet-store mode
------------------
Pass a ``ParquetStore`` instance (``store=``) to the constructor to use the
new columnar parquet backend.  When a store is supplied each ``_load_*``
helper reads one parquet file per period instead of iterating over thousands
of individual ``*.csv.gz`` files — dramatically faster for large datasets.
The plotting logic (histogram / scatter / bar code) is unchanged; only the
data-loading path switches.

Plot-data caching
-----------------
Every plot method serialises the data it plots to a uniquely-named parquet
file under ``<base>/plot_data/``.  On subsequent calls the method checks
whether the cache file already exists; if so the expensive data-loading step
is skipped entirely and the figure is produced straight from the cached
dataframe.  Cache files have fully-descriptive names so they can also be
inspected or reused outside of this class.
"""

import os
import json
import time
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib as mtl
import matplotlib.colors as _mcolors
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from .constants import (
    DIR_OUTPUT,
    DIR_MILESTONES_SERVER,
    PERIOD_NAMES,
    METRIC_FILE_KINDS,
    RURALITY_LEVELS,
    PARTY_NAMES,
    K_RADIUS_VALUES,
    FNAME_SCALARS,
    FNAME_GONZALEZ,
    FNAME_ST,
    FNAME_FREQ_RANK,
    FNAME_WEEKLY_RG,
)
from .utils import ifnotexistsmkdir, get_already_saved_user_per_period
from . import set_mpl

if TYPE_CHECKING:
    from .store import ParquetStore

try:
    from rg_histograms import logHist  # noqa: F401
    from rg_fits import power_fit       # noqa: F401
except ImportError:
    logHist    = None
    power_fit  = None

try:
    import powerlaw  # noqa: F401
except ImportError:
    powerlaw = None


class plotter:
    """
    Load per-user result files and produce plots for all mobility metrics.

    Parameters
    ----------
    np_ : int
        Minimum points threshold used during computation.
    period_division : list[datetime.datetime]
        Ordered list of period boundary datetimes.
    period_names : list[str]
        Names for each analysis period.
    t_threshold : int
        Time threshold (hours) used during computation.
    region : str
        Region identifier, e.g. ``"CA"`` or ``"MA"``.
    county2party : dict
        Mapping ``{county_name: party}``.
    df_rurality : pd.DataFrame
        Rurality info DataFrame loaded from census data.
    output_dir : Path or str, optional
        Override the base output directory.
    """

    def __init__(
        self,
        np_: int,
        period_division: list,
        period_names: list,
        t_threshold: int,
        region: str,
        county2party: dict,
        df_rurality,
        output_dir: Path | str | None = None,
        store: "ParquetStore | None" = None,
    ):
        self.np_             = np_
        self.t_threshold     = t_threshold
        self.period_division = period_division
        self.period_names    = period_names
        self.region          = region
        self.county2party    = county2party
        self.df_rurality     = df_rurality
        self.store           = store

        set_mpl.setup()

        base = Path(output_dir) if output_dir else DIR_MILESTONES_SERVER / region
        self.dir_users     = ifnotexistsmkdir(base / "dataxuser")
        self.dir_plot      = ifnotexistsmkdir(base / "plots")
        self.dir_plot_data = ifnotexistsmkdir(base / "plot_data")

        # Checkpoint index: {period: {kind: [user_ids]}}
        # (only used in legacy mode — store mode uses parquet footer metadata)
        self.period2users = get_already_saved_user_per_period(str(self.dir_users))

        # Lazy file lists — populated on first access (legacy mode only)
        self.period2scalarusers    = None
        self.period2gonzalezusers  = None
        self.period2wrgusers_files = None
        self.period2frequencyusers = None
        self.period2Stusers        = None

    # ------------------------------------------------------------------
    # Plot-data cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, name: str) -> Path:
        """Return the full path for a plot-data parquet cache file."""
        return (
            self.dir_plot_data
            / f"{name}_np{self.np_}_t{self.t_threshold}_{self.region}.parquet"
        )

    def _load_cache(self, name: str) -> "pd.DataFrame | None":
        """Return the cached dataframe, or *None* if the cache does not exist."""
        p = self._cache_path(name)
        if p.exists():
            return pd.read_parquet(p)
        return None

    def _save_cache(self, df: pd.DataFrame, name: str) -> None:
        """Write *df* to the plot-data cache for *name*."""
        df.to_parquet(self._cache_path(name), index=False)

    # ------------------------------------------------------------------
    # File-list builders
    # ------------------------------------------------------------------

    def _scalar_path(self, user, period):
        return self.dir_users / FNAME_SCALARS.format(
            user=user, period=period, np_=self.np_, t=self.t_threshold
        )

    def _gonzalez_path(self, user, period):
        return self.dir_users / FNAME_GONZALEZ.format(
            user=user, period=period, np_=self.np_, t=self.t_threshold
        )

    def _st_path(self, user, period):
        return self.dir_users / FNAME_ST.format(
            user=user, period=period, np_=self.np_, t=self.t_threshold
        )

    def _freq_path(self, user, period):
        return self.dir_users / FNAME_FREQ_RANK.format(
            user=user, period=period, np_=self.np_, t=self.t_threshold
        )

    def _wrg_path(self, user, period):
        return self.dir_users / FNAME_WEEKLY_RG.format(
            user=user, period=period, np_=self.np_, t=self.t_threshold
        )

    def AllScalarsDict(self) -> None:
        t0 = time.time()
        self.period2scalarusers = {
            p: [str(self._scalar_path(u, p)) for u in self.period2users[p]["all_scalars"]]
            for p in self.period_names
        }
        print(f"Scalar file index built in {time.time()-t0:.1f}s")

    def WeekRgDict(self) -> None:
        t0 = time.time()
        self.period2wrgusers_files = {
            p: [str(self._wrg_path(u, p)) for u in self.period2users[p]["weekly_rg"]]
            for p in self.period_names
        }
        print(f"Weekly-RG file index built in {time.time()-t0:.1f}s")

    def GonzalezDict(self) -> None:
        t0 = time.time()
        self.period2gonzalezusers = {
            p: [str(self._gonzalez_path(u, p)) for u in self.period2users[p]["all_scalars"]]
            for p in self.period_names
        }
        print(f"Gonzalez file index built in {time.time()-t0:.1f}s")

    def StDict(self) -> None:
        t0 = time.time()
        self.period2Stusers = {
            p: [str(self._st_path(u, p)) for u in self.period2users[p]["all_scalars"]]
            for p in self.period_names
        }
        print(f"S(t) file index built in {time.time()-t0:.1f}s")

    def FreqDict(self) -> None:
        t0 = time.time()
        self.period2frequencyusers = {
            p: [
                str(self.dir_users / f)
                for f in os.listdir(self.dir_users)
                if "freq" in f and p in f
            ]
            for p in self.period_names
        }
        print(f"Frequency file index built in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Helper: ensure file list is ready  (legacy mode)
    # ------------------------------------------------------------------

    def _ensure(self, kind: str) -> None:
        if self.store is not None:
            return   # parquet store mode — no file-list needed
        builders = {
            "scalar":   self.AllScalarsDict,
            "gonzalez": self.GonzalezDict,
            "weekly_rg":self.WeekRgDict,
            "St":       self.StDict,
            "freq":     self.FreqDict,
        }
        attr = {
            "scalar":   "period2scalarusers",
            "gonzalez": "period2gonzalezusers",
            "weekly_rg":"period2wrgusers_files",
            "St":       "period2Stusers",
            "freq":     "period2frequencyusers",
        }
        if getattr(self, attr[kind]) is None:
            builders[kind]()

    # ------------------------------------------------------------------
    # Data-loading abstraction (store mode vs. legacy file mode)
    # ------------------------------------------------------------------

    def _load_scalars(self, period: str) -> pd.DataFrame:
        """
        Return a pandas DataFrame of scalar metrics, one row per user.

        Parquet-store mode: reads one parquet file (fast).
        Legacy mode: iterates over individual CSV.gz files.
        """
        if self.store is not None:
            pl_df = self.store.read_scalars(period)
            if pl_df.is_empty():
                return pd.DataFrame()
            return pl_df.to_pandas()
        # Legacy
        self._ensure("scalar")
        rows: list[dict] = []
        for f in self.period2scalarusers.get(period, []):
            try:
                rows.append(pd.read_csv(f, compression="gzip").iloc[0].to_dict())
            except Exception:
                continue
        return pd.DataFrame(rows)

    def _load_gonzalez(self, period: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (x_norm, y_norm, sigmax, sigmay) arrays for ``period``.
        """
        if self.store is not None:
            gon = self.store.read_gonzalez(period)
            if gon.is_empty():
                return np.array([]), np.array([]), np.array([]), np.array([])
            g = gon.to_pandas()
            return (
                g["x_norm"].to_numpy(),
                g["y_norm"].to_numpy(),
                g["sigmax"].to_numpy(),
                g["sigmay"].to_numpy(),
            )
        # Legacy
        self._ensure("gonzalez")
        x = np.array([])
        y = np.array([])
        sx = np.array([])
        sy = np.array([])
        for f in self.period2gonzalezusers.get(period, []):
            try:
                df = pd.read_csv(f, compression="gzip")
                x  = np.append(x,  df["x_norm"].to_numpy())
                y  = np.append(y,  df["y_norm"].to_numpy())
                sx = np.append(sx, df["sigmax"].to_numpy())
                sy = np.append(sy, df["sigmay"].to_numpy())
            except Exception:
                continue
        return x, y, sx, sy

    def _load_st_dict(self, period: str, max_people: int = 400_000) -> dict:
        """
        Return ``{time_step: [visited_places, ...]}`` for ``period``.
        """
        period2St: dict = {}
        if self.store is not None:
            mat = self.store.read_st_matrix(period)
            if mat.is_empty():
                return period2St
            user_cols = [c for c in mat.columns if c != "time"][:max_people]
            time_vals = mat["time"].to_list()
            # Each column is one user's S(t) curve; take row-wise list
            arr = mat.select(user_cols).to_numpy()   # shape: [n_steps, n_users]
            for i, t_val in enumerate(time_vals):
                period2St[t_val] = arr[i].tolist()
            return period2St
        # Legacy
        self._ensure("St")
        count = 0
        for f in self.period2Stusers.get(period, []):
            if count >= max_people:
                break
            try:
                df = pd.read_csv(f, compression="gzip")
                new = df.groupby("time")["visited_places"].apply(list).to_dict()
                for t_val, visited in new.items():
                    period2St[t_val] = period2St.get(t_val, []) + visited
                count += 1
            except Exception:
                continue
        return period2St

    def _load_weekly_rg_stratified(self, period: str) -> tuple[dict, dict]:
        """
        Return two nested dicts for stratified weekly-RG analysis::

            period2rg_rural  = {week: {rurality: [rg_values]}}
            period2rg_party  = {week: {party:    [rg_values]}}

        Parquet-store mode: one parquet read + join.
        Legacy mode: file-pair iteration.
        """
        rg_rural: dict = {}
        rg_party: dict = {}

        if self.store is not None:
            import polars as _pl
            scalars_pl = self.store.read_scalars(period)
            wrg_pl     = self.store.read_weekly_rg_matrix(period)
            if scalars_pl.is_empty() or wrg_pl.is_empty():
                return rg_rural, rg_party

            # Unpivot weekly-RG matrix to long format
            user_cols = [c for c in wrg_pl.columns if c != "week"]
            wrg_long  = wrg_pl.unpivot(
                on=user_cols, index="week",
                variable_name="user_id", value_name="rg_value",
            )
            # Join with scalar metadata
            meta = scalars_pl.select(["user_id", "rurality_level", "party_government"])
            joined = wrg_long.join(meta, on="user_id", how="left")

            for row in joined.iter_rows(named=True):
                week = row["week"]
                rg   = row["rg_value"]
                rur  = row["rurality_level"]
                pty  = row["party_government"]
                if rg is None or (isinstance(rg, float) and (np.isnan(rg) or rg >= 20_037.0)):
                    continue
                if rur in RURALITY_LEVELS:
                    rg_rural.setdefault(week, {r: [] for r in RURALITY_LEVELS})
                    rg_rural[week][rur].append(rg)
                if pty in PARTY_NAMES:
                    rg_party.setdefault(week, {p: [] for p in PARTY_NAMES})
                    rg_party[week][pty].append(rg)
            return rg_rural, rg_party

        # Legacy
        self._ensure("weekly_rg")
        self._ensure("scalar")
        for f_wrg in self.period2wrgusers_files.get(period, []):
            f_sc = f_wrg.replace("weekly_rg", "all_scalars").replace(".json", ".csv.gz")
            if not (os.path.isfile(f_wrg) and os.path.isfile(f_sc)):
                continue
            try:
                df_sc = pd.read_csv(f_sc, compression="gzip")
                rur   = df_sc["rurality_level"].iloc[0]
                pty   = df_sc["party_government"].iloc[0]
                with open(f_wrg) as fh:
                    wrg = json.load(fh)
                for week, val in wrg.items():
                    if isinstance(val, float) and (np.isnan(val) or val >= 20_037.0):
                        continue
                    if rur in RURALITY_LEVELS:
                        rg_rural.setdefault(week, {r: [] for r in RURALITY_LEVELS})
                        rg_rural[week][rur].append(val)
                    if pty in PARTY_NAMES:
                        rg_party.setdefault(week, {p: [] for p in PARTY_NAMES})
                        rg_party[week][pty].append(val)
            except Exception:
                continue
        return rg_rural, rg_party

    def _load_frequency_by_rank(self, period: str, max_rank: int = 9) -> dict:
        """Return ``{rank: [frequency_values]}`` for ``period``."""
        if self.store is not None:
            freq_df = self.store.read_frequency(period)
            if freq_df.is_empty():
                return {r: [] for r in range(1, max_rank + 1)}
            result = {r: [] for r in range(1, max_rank + 1)}
            for uid_group in freq_df.to_pandas().groupby("user_id")["frequency"]:
                vals = list(uid_group[1])
                for r in range(1, max_rank + 1):
                    if r - 1 < len(vals):
                        result[r].append(vals[r - 1])
            return result
        # Legacy
        self._ensure("freq")
        result = {r: [] for r in range(1, max_rank + 1)}
        for f in self.period2frequencyusers.get(period, []):
            try:
                df = pd.read_csv(f, compression="gzip")
                for r in range(1, max_rank + 1):
                    if r - 1 < len(df["frequency"]):
                        result[r].append(df["frequency"].iloc[r - 1])
            except Exception:
                continue
        return result

    # ------------------------------------------------------------------
    # Radius of gyration
    # ------------------------------------------------------------------

    def plot_rg(self) -> None:
        """Log-log PDF of overall RG across the three periods."""
        assert logHist is not None, "rg_histograms library not available"
        _MAX_RG_KM = 20_037.0

        CACHE = "radius_gyration_distribution"
        df = self._load_cache(CACHE)
        if df is None:
            rows = []
            for p in self.period_names:
                df_s = self._load_scalars(p)
                if not df_s.empty:
                    vals = pd.to_numeric(df_s["radius_gyration"], errors="coerce").dropna()
                    vals = vals[(vals > 0.1) & (vals < _MAX_RG_KM)].tolist()
                    if vals:
                        q, a, _ = logHist(vals, 200)
                        for bc, pv in zip(a, q):
                            rows.append({"period": p, "bin_center": float(bc), "pdf": float(pv)})
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        fig, ax = plt.subplots(figsize=(8, 5))
        for p in self.period_names:
            sub = df[df["period"] == p].sort_values("bin_center")
            if sub.empty:
                continue
            ax.loglog(sub["bin_center"], sub["pdf"], linewidth=set_mpl.LINEWIDTH)
        ax.set_xlabel(r"$R_g$ (km)", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.set_ylabel("PDF", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.legend(self.period_names, fontsize=set_mpl.FONTSIZE_LEGEND)
        plt.tight_layout()
        set_mpl.save(fig, self.dir_plot / f"rg_{self.np_}_hour_{self.t_threshold}_{self.region}.png")
        plt.show()

    def plot_rg_party_per_period(self) -> None:
        """RG distribution split by Democratic / Republican county, one subplot per period."""
        assert logHist is not None, "rg_histograms library not available"
        _MAX_RG_KM = 20_037.0
        party2color = {"Democratic": "blue", "Republican": "red"}

        CACHE = "radius_gyration_by_party"
        df = self._load_cache(CACHE)
        if df is None:
            rows = []
            for p in self.period_names:
                df_s = self._load_scalars(p)
                if not df_s.empty and "party_government" in df_s.columns:
                    for party in PARTY_NAMES:
                        vals = pd.to_numeric(
                            df_s.loc[df_s["party_government"] == party, "radius_gyration"],
                            errors="coerce"
                        ).dropna()
                        vals = vals[(vals > 0.1) & (vals < _MAX_RG_KM)].tolist()
                        if vals:
                            q, a, _ = logHist(vals, 200)
                            for bc, pv in zip(a, q):
                                rows.append({"period": p, "party": party,
                                             "bin_center": float(bc), "pdf": float(pv)})
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        for p in self.period_names:
            fig, ax = plt.subplots(figsize=(8, 5))
            for party in PARTY_NAMES:
                sub = df[(df["period"] == p) & (df["party"] == party)].sort_values("bin_center")
                if not sub.empty:
                    ax.loglog(sub["bin_center"], sub["pdf"], linewidth=set_mpl.LINEWIDTH,
                              color=party2color[party], label=party)
            ax.set_xlabel(r"$R_g$ (km)", fontsize=set_mpl.FONTSIZE_LABEL)
            ax.set_ylabel("PDF", fontsize=set_mpl.FONTSIZE_LABEL)
#            ax.set_title(p, fontsize=set_mpl.FONTSIZE_TITLE)
            ax.legend(fontsize=set_mpl.FONTSIZE_LEGEND)
            plt.tight_layout()
            set_mpl.save(fig,
                self.dir_plot / f"rg_party_{self.np_}_t_{self.t_threshold}_{self.region}_{p}.png")
            plt.show()

    def plot_rg_rurality_per_period(self) -> None:
        """RG distribution split by urban / rural county, one subplot per period."""
        assert logHist is not None, "rg_histograms library not available"
        _MAX_RG_KM = 20_037.0
        rural2color = {"rural": "blue", "urban": "red"}

        CACHE = "radius_gyration_by_rurality"
        df = self._load_cache(CACHE)
        if df is None:
            rows = []
            for p in self.period_names:
                df_s = self._load_scalars(p)
                if not df_s.empty and "rurality_level" in df_s.columns:
                    for rur in RURALITY_LEVELS:
                        vals = pd.to_numeric(
                            df_s.loc[df_s["rurality_level"] == rur, "radius_gyration"],
                            errors="coerce"
                        ).dropna()
                        vals = vals[(vals > 0.1) & (vals < _MAX_RG_KM)].tolist()
                        if vals:
                            q, a, _ = logHist(vals, 200)
                            for bc, pv in zip(a, q):
                                rows.append({"period": p, "rurality": rur,
                                             "bin_center": float(bc), "pdf": float(pv)})
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        for p in self.period_names:
            fig, ax = plt.subplots(figsize=(8, 5))
            for rur in RURALITY_LEVELS:
                sub = df[(df["period"] == p) & (df["rurality"] == rur)].sort_values("bin_center")
                if not sub.empty:
                    ax.loglog(sub["bin_center"], sub["pdf"], linewidth=set_mpl.LINEWIDTH,
                              color=rural2color[rur], label=rur)
            ax.set_xlabel(r"$R_g$ (km)", fontsize=set_mpl.FONTSIZE_LABEL)
            ax.set_ylabel("PDF", fontsize=set_mpl.FONTSIZE_LABEL)
#            ax.set_title(p, fontsize=set_mpl.FONTSIZE_TITLE)
            ax.legend(fontsize=set_mpl.FONTSIZE_LEGEND)
            plt.tight_layout()
            set_mpl.save(fig,
                self.dir_plot / f"rg_rurality_{self.np_}_t_{self.t_threshold}_{self.region}_{p}.png")
            plt.show()

    # ------------------------------------------------------------------
    # Weekly radius of gyration
    # ------------------------------------------------------------------

    def _build_week2rg(self) -> dict:
        """Aggregate weekly RG values across all periods."""
        week2rg: dict = defaultdict(list)
        for p in self.period_names:
            if self.store is not None:
                mat = self.store.read_weekly_rg_matrix(p)
                if mat.is_empty():
                    continue
                user_cols = [c for c in mat.columns if c != "week"]
                for row in mat.iter_rows(named=True):
                    week = row["week"]
                    for uid in user_cols:
                        v = row[uid]
                        if v is not None and not (isinstance(v, float) and np.isnan(v)):
                            week2rg[week].append(v)
            else:
                self._ensure("weekly_rg")
                for f in self.period2wrgusers_files.get(p, []):
                    if not os.path.isfile(f):
                        continue
                    try:
                        with open(f) as fh:
                            d = json.load(fh)
                        for k, v in d.items():
                            week2rg[k].append(v)
                    except Exception:
                        continue
        return week2rg

    def plot_weekly_rg(self) -> None:
        """Time series of average weekly RG with period boundary lines."""
        import matplotlib.dates as _mdates
        from datetime import datetime as _dt

        def _parse_week(w: str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return _dt.strptime(w, fmt)
                except ValueError:
                    pass
            return w

        CACHE = "weekly_rg_timeseries"
        df = self._load_cache(CACHE)
        if df is None:
            week2rg = self._build_week2rg()
            if not week2rg:
                print("No weekly-RG data available.")
                return
            rows = [
                {"week": w, "mean_rg": float(np.nanmean(vals)),
                 "std_rg": float(np.nanstd(vals))}
                for w, vals in week2rg.items()
            ]
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        if df.empty:
            print("No weekly-RG data available.")
            return

        df = df.copy()
        df["week_dt"] = df["week"].apply(_parse_week)
        df = df.sort_values("week_dt")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["week_dt"], df["mean_rg"], color="steelblue",
                linewidth=set_mpl.LINEWIDTH, marker="o", ms=4)
        for boundary in self.period_division[1:-1]:
            ax.axvline(boundary, color="black", linestyle="--",
                       linewidth=set_mpl.LINEWIDTH_THIN, alpha=0.7)
        ax.xaxis.set_major_formatter(_mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(_mdates.WeekdayLocator(interval=4))
        plt.xticks(rotation=45, ha="right", fontsize=set_mpl.FONTSIZE_TICK)
        ax.set_xlabel("Week", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.set_ylabel(r"$\langle R_g \rangle$ (km)", fontsize=set_mpl.FONTSIZE_LABEL)
#        ax.set_title("Weekly average radius of gyration", fontsize=set_mpl.FONTSIZE_TITLE)
        plt.tight_layout()
        set_mpl.save(fig,
            self.dir_plot / f"weekly_rg_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    def plot_rg_rurality_weekly(self) -> None:
        """Weekly avg RG stratified by rurality, with period boundary lines."""
        import matplotlib.dates as _mdates
        from datetime import datetime as _dt

        rural2color = {"rural": "steelblue", "urban": "tomato"}

        def _parse_week(w: str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return _dt.strptime(w, fmt)
                except ValueError:
                    pass
            return w

        CACHE = "weekly_rg_by_rurality"
        df = self._load_cache(CACHE)
        if df is None:
            all_weeks: dict = {}
            for p in self.period_names:
                rg_rural, _ = self._load_weekly_rg_stratified(p)
                for week, strata in rg_rural.items():
                    all_weeks.setdefault(week, {r: [] for r in RURALITY_LEVELS})
                    for rur in RURALITY_LEVELS:
                        all_weeks[week][rur].extend(strata.get(rur, []))
            if not all_weeks:
                print("No weekly rurality-RG data available.")
                return
            rows = [
                {"week": w, "rurality": rur,
                 "mean_rg": float(np.nanmean(all_weeks[w].get(rur, [float("nan")]))),
                 "std_rg":  float(np.nanstd(all_weeks[w].get(rur, [0.0])))}
                for w in all_weeks
                for rur in RURALITY_LEVELS
            ]
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        if df.empty:
            print("No weekly rurality-RG data available.")
            return

        sorted_weeks = sorted(df["week"].unique(), key=_parse_week)
        fig, ax = plt.subplots(figsize=(12, 5))
        for rur in RURALITY_LEVELS:
            sub = df[df["rurality"] == rur].set_index("week").reindex(sorted_weeks)
            week_dts = [_parse_week(w) for w in sorted_weeks]
            valid = np.isfinite(sub["mean_rg"].to_numpy())
            ax.plot([d for d, v in zip(week_dts, valid) if v],
                    sub["mean_rg"].to_numpy()[valid],
                    color=rural2color[rur], linewidth=set_mpl.LINEWIDTH, label=rur)
        for boundary in self.period_division[1:-1]:
            ax.axvline(boundary, color="black", linestyle="--",
                       linewidth=set_mpl.LINEWIDTH_THIN, alpha=0.7)
        ax.xaxis.set_major_formatter(_mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(_mdates.WeekdayLocator(interval=4))
        plt.xticks(rotation=45, ha="right", fontsize=set_mpl.FONTSIZE_TICK)
        ax.set_xlabel("Week", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.set_ylabel(r"$\langle R_g \rangle$ (km)", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.legend(fontsize=set_mpl.FONTSIZE_LEGEND)
        plt.tight_layout()
        set_mpl.save(fig,
            self.dir_plot / f"weekly_rg_rurality_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    def plot_rg_party_weekly(self) -> None:
        """Weekly avg RG stratified by political party, with period boundary lines."""
        import matplotlib.dates as _mdates
        from datetime import datetime as _dt

        party2color = {"Democratic": "royalblue", "Republican": "firebrick"}

        def _parse_week(w: str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return _dt.strptime(w, fmt)
                except ValueError:
                    pass
            return w

        CACHE = "weekly_rg_by_party"
        df = self._load_cache(CACHE)
        if df is None:
            all_weeks: dict = {}
            for p in self.period_names:
                _, rg_party = self._load_weekly_rg_stratified(p)
                for week, strata in rg_party.items():
                    all_weeks.setdefault(week, {pt: [] for pt in PARTY_NAMES})
                    for pt in PARTY_NAMES:
                        all_weeks[week][pt].extend(strata.get(pt, []))
            if not all_weeks:
                print("No weekly party-RG data available.")
                return
            rows = [
                {"week": w, "party": pt,
                 "mean_rg": float(np.nanmean(all_weeks[w].get(pt, [float("nan")]))),
                 "std_rg":  float(np.nanstd(all_weeks[w].get(pt, [0.0])))}
                for w in all_weeks
                for pt in PARTY_NAMES
            ]
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        if df.empty:
            print("No weekly party-RG data available.")
            return

        sorted_weeks = sorted(df["week"].unique(), key=_parse_week)
        fig, ax = plt.subplots(figsize=(12, 5))
        for party in PARTY_NAMES:
            sub = df[df["party"] == party].set_index("week").reindex(sorted_weeks)
            week_dts = [_parse_week(w) for w in sorted_weeks]
            valid = np.isfinite(sub["mean_rg"].to_numpy())
            ax.plot([d for d, v in zip(week_dts, valid) if v],
                    sub["mean_rg"].to_numpy()[valid],
                    color=party2color[party], linewidth=set_mpl.LINEWIDTH, label=party)
        for boundary in self.period_division[1:-1]:
            ax.axvline(boundary, color="black", linestyle="--",
                       linewidth=set_mpl.LINEWIDTH_THIN, alpha=0.7)
        ax.xaxis.set_major_formatter(_mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(_mdates.WeekdayLocator(interval=4))
        plt.xticks(rotation=45, ha="right", fontsize=set_mpl.FONTSIZE_TICK)
        ax.set_xlabel("Week", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.set_ylabel(r"$\langle R_g \rangle$ (km)", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.legend(fontsize=set_mpl.FONTSIZE_LEGEND)
        plt.tight_layout()
        set_mpl.save(fig,
            self.dir_plot / f"weekly_rg_party_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    def plot_rg_weekly_combined(self) -> None:
        """Rurality (top) and party (bottom) weekly avg RG in one figure, shared x-axis."""
        import matplotlib.dates as _mdates
        from datetime import datetime as _dt

        rural2color = {"rural": "steelblue", "urban": "tomato"}
        party2color = {"Democratic": "royalblue", "Republican": "firebrick"}

        def _parse_week(w: str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return _dt.strptime(w, fmt)
                except ValueError:
                    pass
            return w

        # ── load / build rurality cache ──────────────────────────────
        df_rur = self._load_cache("weekly_rg_by_rurality")
        if df_rur is None:
            all_weeks: dict = {}
            for p in self.period_names:
                rg_rural, _ = self._load_weekly_rg_stratified(p)
                for week, strata in rg_rural.items():
                    all_weeks.setdefault(week, {r: [] for r in RURALITY_LEVELS})
                    for rur in RURALITY_LEVELS:
                        all_weeks[week][rur].extend(strata.get(rur, []))
            if not all_weeks:
                print("No weekly rurality-RG data available.")
                return
            df_rur = pd.DataFrame([
                {"week": w, "rurality": rur,
                 "mean_rg": float(np.nanmean(all_weeks[w].get(rur, [float("nan")]))),
                 "std_rg":  float(np.nanstd(all_weeks[w].get(rur, [0.0])))}
                for w in all_weeks for rur in RURALITY_LEVELS
            ])
            self._save_cache(df_rur, "weekly_rg_by_rurality")

        # ── load / build party cache ─────────────────────────────────
        df_party = self._load_cache("weekly_rg_by_party")
        if df_party is None:
            all_weeks = {}
            for p in self.period_names:
                _, rg_party = self._load_weekly_rg_stratified(p)
                for week, strata in rg_party.items():
                    all_weeks.setdefault(week, {pt: [] for pt in PARTY_NAMES})
                    for pt in PARTY_NAMES:
                        all_weeks[week][pt].extend(strata.get(pt, []))
            if not all_weeks:
                print("No weekly party-RG data available.")
                return
            df_party = pd.DataFrame([
                {"week": w, "party": pt,
                 "mean_rg": float(np.nanmean(all_weeks[w].get(pt, [float("nan")]))),
                 "std_rg":  float(np.nanstd(all_weeks[w].get(pt, [0.0])))}
                for w in all_weeks for pt in PARTY_NAMES
            ])
            self._save_cache(df_party, "weekly_rg_by_party")

        if df_rur.empty and df_party.empty:
            print("No weekly RG data available.")
            return

        sorted_weeks_rur   = sorted(df_rur["week"].unique(),   key=_parse_week)
        sorted_weeks_party = sorted(df_party["week"].unique(), key=_parse_week)

        fig, (ax_rur, ax_party) = plt.subplots(
            2, 1, figsize=(12, 9), sharex=True,
            constrained_layout=True,
        )

        # Top panel — rurality
        for rur in RURALITY_LEVELS:
            sub = df_rur[df_rur["rurality"] == rur].set_index("week").reindex(sorted_weeks_rur)
            week_dts = [_parse_week(w) for w in sorted_weeks_rur]
            valid = np.isfinite(sub["mean_rg"].to_numpy())
            ax_rur.plot([d for d, v in zip(week_dts, valid) if v],
                        sub["mean_rg"].to_numpy()[valid],
                        color=rural2color[rur], linewidth=set_mpl.LINEWIDTH, label=rur)
        for boundary in self.period_division[1:-1]:
            ax_rur.axvline(boundary, color="black", linestyle="--",
                           linewidth=set_mpl.LINEWIDTH_THIN, alpha=0.7)
        ax_rur.set_ylabel(r"$\langle R_g \rangle$ (km)", fontsize=set_mpl.FONTSIZE_LABEL)
        ax_rur.legend(fontsize=set_mpl.FONTSIZE_LEGEND)

        # Bottom panel — party
        for party in PARTY_NAMES:
            sub = df_party[df_party["party"] == party].set_index("week").reindex(sorted_weeks_party)
            week_dts = [_parse_week(w) for w in sorted_weeks_party]
            valid = np.isfinite(sub["mean_rg"].to_numpy())
            ax_party.plot([d for d, v in zip(week_dts, valid) if v],
                          sub["mean_rg"].to_numpy()[valid],
                          color=party2color[party], linewidth=set_mpl.LINEWIDTH, label=party)
        for boundary in self.period_division[1:-1]:
            ax_party.axvline(boundary, color="black", linestyle="--",
                             linewidth=set_mpl.LINEWIDTH_THIN, alpha=0.7)
        ax_party.set_ylabel(r"$\langle R_g \rangle$ (km)", fontsize=set_mpl.FONTSIZE_LABEL)
        ax_party.set_xlabel("Week", fontsize=set_mpl.FONTSIZE_LABEL)
        ax_party.legend(fontsize=set_mpl.FONTSIZE_LEGEND)

        ax_party.xaxis.set_major_formatter(_mdates.DateFormatter("%b %Y"))
        ax_party.xaxis.set_major_locator(_mdates.WeekdayLocator(interval=4))
        plt.setp(ax_party.get_xticklabels(), rotation=45, ha="right",
                 fontsize=set_mpl.FONTSIZE_TICK)

        set_mpl.save(fig,
            self.dir_plot / f"weekly_rg_combined_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    # ------------------------------------------------------------------
    # k-Radius of gyration
    # ------------------------------------------------------------------

    def plot_krg(self) -> None:
        """Grid of 2-D histograms: rows = periods (lockdown + post), columns = k values."""
        assert logHist is not None, "rg_histograms library not available"
        nbins    = 40
        list_k   = K_RADIUS_VALUES
        rg_range = [0.1, 1500.0]

        CACHE = "k_radius_gyration_distribution"
        df = self._load_cache(CACHE)
        if df is None:
            edges   = np.linspace(rg_range[0], rg_range[1], nbins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            rows = []
            for p in self.period_names[1:]:
                df_all = self._load_scalars(p)
                if df_all.empty:
                    continue
                rg_col = pd.to_numeric(df_all["radius_gyration"], errors="coerce").to_numpy()
                for k in list_k:
                    col = f"rg_{k}"
                    if col not in df_all.columns:
                        continue
                    rgk_col = pd.to_numeric(df_all[col], errors="coerce").to_numpy()
                    valid   = (np.isfinite(rg_col) & np.isfinite(rgk_col) &
                               (rg_col > rg_range[0]) & (rgk_col > rg_range[0]))
                    if not valid.any():
                        continue
                    H, _, _ = np.histogram2d(
                        rg_col[valid], rgk_col[valid],
                        bins=(nbins, nbins),
                        range=[rg_range, rg_range],
                        density=True,
                    )
                    H = H.T   # shape (nbins, nbins): row = y (k-Rg), col = x (Rg)
                    for i, yc in enumerate(centers):
                        for j, xc in enumerate(centers):
                            rows.append({"period": p, "k": int(k),
                                         "x_center": float(xc), "y_center": float(yc),
                                         "density": float(H[i, j])})
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        periods  = self.period_names[1:]   # skip pre-lockdown baseline (no data yet)
        n_rows  = len(periods)
        n_cols  = len(list_k)
        edges   = np.linspace(rg_range[0], rg_range[1], nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        Xm, Ym  = np.meshgrid(centers, centers)

        # Reconstruct 2D arrays from the long-format cache via pivot
        Hs: dict = {}
        for p in periods:
            for k in list_k:
                sub = df[(df["period"] == p) & (df["k"] == k)]
                if sub.empty:
                    Hs[(p, k)] = None
                    continue
                H = (sub.sort_values(["y_center", "x_center"])["density"]
                     .to_numpy()
                     .reshape(nbins, nbins))
                Hs[(p, k)] = H

        all_pos = [H[H > 0].ravel() for H in Hs.values() if H is not None]
        if not all_pos:
            return
        vmin_g  = min(v.min() for v in all_pos)
        vmax_g  = max(v.max() for v in all_pos)
        norm_sh = _mcolors.LogNorm(vmin=vmin_g, vmax=vmax_g)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(7 * n_cols, 6 * n_rows + 1),
                                 sharex=True, sharey=True,
                                 squeeze=False,
                                 constrained_layout=True)

        pcm_ref = None
        for row_idx, p in enumerate(periods):
            for col_idx, k in enumerate(list_k):
                ax = axes[row_idx][col_idx]
                H  = Hs.get((p, k))
                if H is None:
                    ax.set_visible(False)
                    continue
                pcm     = ax.pcolormesh(Xm, Ym, H, norm=norm_sh,
                                        cmap=plt.cm.jet, shading="auto")
                pcm_ref = pcm
                if row_idx == 0:
                    ax.set_title(rf"$k={k}$", fontsize=set_mpl.FONTSIZE_TITLE)
                if col_idx == 0:
                    ax.set_ylabel(p, fontsize=set_mpl.FONTSIZE_LABEL)
                if row_idx == n_rows - 1:
                    ax.set_xlabel(r"$R_g$ (km)", fontsize=set_mpl.FONTSIZE_LABEL)

        if pcm_ref is not None:
            cbar = fig.colorbar(pcm_ref, ax=axes.ravel().tolist(),
                                orientation="horizontal", location="bottom",
                                fraction=0.03, pad=0.05)
            cbar.set_label("Density", fontsize=set_mpl.FONTSIZE_LABEL)

        set_mpl.save(fig,
            self.dir_plot / f"rg_krg_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def plot_distance(self) -> None:
        """Log-log PDF of total haversine path length with power-law fit."""
        assert logHist is not None, "rg_histograms library required"
        _ALPHA_FALLBACK = {
            self.period_names[0]: 3.4,
            self.period_names[1]: 2.6,
            self.period_names[2]: 3.3,
        } if len(self.period_names) >= 3 else {}
        _FILTER_KM = 0.01     # exclude zeros / near-zeros
        _MAX_KM    = 20_037.0

        CACHE = "distance_distribution_v2"
        df = self._load_cache(CACHE)
        if df is None:
            rows = []
            for p in self.period_names:
                df_s = self._load_scalars(p)
                if df_s.empty:
                    continue
                vals_all = pd.to_numeric(df_s["distance"], errors="coerce").dropna()
                vals_all = vals_all[(vals_all > _FILTER_KM) & (vals_all < _MAX_KM)]
                vals = vals_all.tolist()
                if not vals:
                    continue
                q, a, _ = logHist(vals, 200)
                alpha_v: float = _ALPHA_FALLBACK.get(p, 3.0)
                sigma_v: float = float("nan")
                xmin_fit: float = float(np.percentile(vals, 50))  # fallback: median
                fraction: float = 1.0
                if powerlaw is not None:
                    try:
                        fit      = powerlaw.Fit(vals)          # auto-select xmin
                        alpha_v  = float(fit.power_law.alpha)
                        sigma_v  = float(fit.power_law.sigma)
                        xmin_fit = float(fit.power_law.xmin)
                        fraction = float(np.sum(vals_all >= xmin_fit) / len(vals_all))
                    except Exception:
                        pass
                for bc, pv in zip(a, q):
                    rows.append({"period": p, "bin_center": float(bc), "pdf": float(pv),
                                 "fit_alpha": alpha_v, "fit_sigma": sigma_v,
                                 "fit_xmin": xmin_fit, "fit_fraction": fraction})
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, p in enumerate(self.period_names):
            sub = df[df["period"] == p].sort_values("bin_center")
            if sub.empty:
                continue
            color    = colors[idx % len(colors)]
            alpha_v  = float(sub["fit_alpha"].iloc[0])
            sigma_v  = float(sub["fit_sigma"].iloc[0])
            xmin     = float(sub["fit_xmin"].iloc[0])
            fraction = float(sub["fit_fraction"].iloc[0]) if "fit_fraction" in sub.columns else 1.0
            ax.scatter(sub["bin_center"], sub["pdf"], s=20, color=color)
            # Scale the power-law line so it integrates to `fraction` over [xmin, ∞),
            # matching the empirical histogram normalized over all data.
            x_fit = np.logspace(np.log10(xmin), np.log10(sub["bin_center"].max()), 200)
            C     = fraction * (alpha_v - 1) * xmin ** (alpha_v - 1)
            if np.isfinite(sigma_v):
                lbl = rf"$\alpha$ = {alpha_v:.3f} $\pm$ {sigma_v:.3f}"
            else:
                lbl = rf"$\alpha$ ≈ {alpha_v:.3f} (preset)"
            ax.loglog(x_fit, C * x_fit ** (-alpha_v),
                      color=color, linestyle="--", linewidth=set_mpl.LINEWIDTH,
                      label=lbl)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Distance (km)", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.set_ylabel("PDF", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.legend(fontsize=set_mpl.FONTSIZE_LEGEND)
        plt.tight_layout()
        set_mpl.save(fig,
            self.dir_plot / f"distance_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------

    def plot_entropy(self) -> None:
        """Single figure with 4 panels: random, uncorrelated, real, and scaled (real/random) entropy."""
        assert logHist is not None, "rg_histograms library not available"

        ETYPES = ["random_entropy", "uncorrelated_entropy", "real_entropy", "scaled_entropy"]
        CACHE  = "entropy_distributions"

        df = self._load_cache(CACHE)
        # Recompute if cache is absent or doesn't yet contain the scaled entropy rows
        if df is None or "scaled_entropy" not in df["entropy_type"].values:
            etype_bins = {
                "random_entropy": 12,
                "uncorrelated_entropy": 200,
                "real_entropy": 200,
                "scaled_entropy": 200,
            }
            rows = []
            for p in self.period_names:
                df_s = self._load_scalars(p)
                if df_s.empty:
                    continue
                raw: dict = {et: [] for et in etype_bins}
                for etype in ["random_entropy", "uncorrelated_entropy", "real_entropy"]:
                    if etype in df_s.columns:
                        raw[etype] = df_s[etype].dropna().tolist()
                if "real_entropy" in df_s.columns and "random_entropy" in df_s.columns:
                    paired = df_s[["real_entropy", "random_entropy"]].dropna()
                    paired = paired[paired["random_entropy"] > 0]
                    raw["scaled_entropy"] = [
                        float(r["real_entropy"] / r["random_entropy"])
                        for _, r in paired.iterrows()
                    ]
                for etype, vals in raw.items():
                    if not vals:
                        continue
                    q, a, _ = logHist(vals, etype_bins[etype])
                    for bc, pv in zip(a, q):
                        rows.append({"period": p, "entropy_type": etype,
                                     "bin_center": float(bc), "pdf": float(pv)})
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        etype_xlabel = {
            "random_entropy":       "Random entropy",
            "uncorrelated_entropy": "Uncorrelated entropy",
            "real_entropy":         "Real entropy",
            "scaled_entropy":       r"Real / Random entropy",
        }

        fig, axes = plt.subplots(2, 2,
                                 figsize=(16, 10),
                                 squeeze=False)
        for idx, etype in enumerate(ETYPES):
            ax = axes[idx // 2][idx % 2]
            for p in self.period_names:
                sub = df[
                    (df["period"] == p) & (df["entropy_type"] == etype)
                ].sort_values("bin_center")
                if sub.empty:
                    continue
                ax.plot(sub["bin_center"], sub["pdf"], linewidth=set_mpl.LINEWIDTH)
            ax.set_xlabel(etype_xlabel[etype], fontsize=set_mpl.FONTSIZE_LABEL)
            if idx % 2 == 0:
                ax.set_ylabel("PDF", fontsize=set_mpl.FONTSIZE_LABEL)
            ax.legend(self.period_names, fontsize=12)
        plt.tight_layout()
        set_mpl.save(fig,
            self.dir_plot / f"entropy_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    # ------------------------------------------------------------------
    # S(t) exploration curve
    # ------------------------------------------------------------------

    def plot_St(self) -> None:
        """S(t) scatter with power-law fit per period (no error band)."""
        # Approximate exploration exponents (mu) per period as fallback estimates.
        _MU_FALLBACK = {
            self.period_names[0]: 0.668,
            self.period_names[1]: 0.523,
            self.period_names[2]: 0.668,
        } if len(self.period_names) >= 3 else {}

        CACHE = "st_exploration_curve"
        df = self._load_cache(CACHE)
        if df is None:
            MAX_PEOPLE = 400_000
            rows = []
            for p in self.period_names:
                period2St = self._load_st_dict(p, max_people=MAX_PEOPLE)
                if not period2St:
                    continue
                for t_step in sorted(period2St):
                    vals = period2St[t_step]
                    rows.append({
                        "period":   p,
                        "time_step": int(t_step),
                        "mean_st":  float(np.nanmean(vals)),
                        "std_st":   float(np.nanstd(vals)),
                    })
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, p in enumerate(self.period_names):
            sub = df[df["period"] == p].sort_values("time_step")
            if sub.empty:
                continue
            l_mean = sub["mean_st"].to_numpy()
            t_arr  = np.arange(len(l_mean))
            mask   = np.isfinite(l_mean)
            l_mean = l_mean[mask]
            t_arr  = t_arr[mask]

            color = colors[idx % len(colors)]
            ax.scatter(t_arr, l_mean, s=20, alpha=set_mpl.ALPHA_SIM, color=color)

            # Power-law fit
            if power_fit is not None:
                try:
                    slope, std_err, _r, _i = power_fit(t_arr, l_mean)
                    u_spacing = t_arr ** slope
                    ax.loglog(t_arr[10:], u_spacing[10:],
                              linestyle="dashed", color=color,
                              linewidth=set_mpl.LINEWIDTH,
                              label=rf"$\mu$ = {slope:.3f} $\pm$ {std_err:.3f}")
                except Exception:
                    mu_fb = _MU_FALLBACK.get(p, 0.6)
                    u_spacing = t_arr ** mu_fb
                    ax.loglog(t_arr[10:], u_spacing[10:],
                              linestyle="dashed", color=color,
                              linewidth=set_mpl.LINEWIDTH,
                              label=rf"$\mu$ ≈ {mu_fb:.3f} (fallback)")
            else:
                mu_fb = _MU_FALLBACK.get(p, 0.6)
                u_spacing = t_arr ** mu_fb
                ax.loglog(t_arr[10:], u_spacing[10:],
                          linestyle="dashed", color=color,
                          linewidth=set_mpl.LINEWIDTH,
                          label=rf"$\mu$ ≈ {mu_fb:.3f} (preset)")

        ax.set_xlabel("t (h)", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.set_ylabel("S(t)", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=set_mpl.FONTSIZE_LEGEND)
        plt.tight_layout()
        set_mpl.save(fig,
            self.dir_plot / f"St_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    # ------------------------------------------------------------------
    # Location frequency
    # ------------------------------------------------------------------

    def plot_frequency(self) -> None:
        """Bar chart of average location frequency by rank (top 9)."""
        max_rank = 9

        CACHE = "location_frequency_by_rank"
        df = self._load_cache(CACHE)
        if df is None:
            rows = []
            for p in self.period_names:
                freq = self._load_frequency_by_rank(p, max_rank=max_rank)
                for r in range(1, max_rank + 1):
                    vals = freq.get(r, [])
                    if vals:
                        rows.append({"period": p, "rank": r,
                                     "mean_frequency": float(np.mean(vals))})
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        x   = np.arange(max_rank)
        w   = 0.3
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, p in enumerate(self.period_names):
            sub = df[df["period"] == p].sort_values("rank")
            vals = sub["mean_frequency"].tolist()
            ax.bar(x[:len(vals)] + idx * w, vals, width=w, label=p, edgecolor="black")
        ax.set_xlabel("Rank", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.set_ylabel(r"$\langle k \rangle$", fontsize=set_mpl.FONTSIZE_LABEL)
        ax.legend(fontsize=set_mpl.FONTSIZE_LEGEND)
        plt.tight_layout()
        set_mpl.save(fig,
            self.dir_plot / f"frequency_rank_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    # ------------------------------------------------------------------
    # Gonzalez trajectory shape
    # ------------------------------------------------------------------

    def _load_gonzalez_all_raw(self) -> pd.DataFrame:
        """Load raw per-user gonzalez data for all periods (used to build plot caches)."""
        rows = []
        for p in self.period_names:
            x, y, sx, sy = self._load_gonzalez(p)
            for xv, yv, sxv, syv in zip(x, y, sx, sy):
                rows.append({"period": p,
                             "x_norm": float(xv), "y_norm": float(yv),
                             "sigmax": float(sxv), "sigmay": float(syv)})
        return pd.DataFrame(rows)

    def plot_gonzalez(
        self,
        xmin: float = -1.5, xmax: float = 1.5,
        ymin: float = -2.2, ymax: float = 2.2,
        nbins: int = 40,
    ) -> None:
        """2-row grid per period: top row = 2-D density map, bottom row = profile at y/σ_y = 0."""
        CACHE = "gonzalez_2d"
        df = self._load_cache(CACHE)
        if df is None:
            raw  = self._load_gonzalez_all_raw()
            x_edges   = np.linspace(xmin, xmax, nbins + 1)
            y_edges   = np.linspace(ymin, ymax, nbins + 1)
            x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
            y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
            rows = []
            for p in self.period_names:
                sub  = raw[raw["period"] == p]
                x    = sub["x_norm"].to_numpy()
                y    = sub["y_norm"].to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                H, _, _ = np.histogram2d(x[mask], y[mask], bins=(nbins, nbins),
                                         range=[[xmin, xmax], [ymin, ymax]])
                H = H.T   # shape (nbins_y, nbins_x)
                for i, yc in enumerate(y_centers):
                    for j, xc in enumerate(x_centers):
                        rows.append({"period": p,
                                     "x_center": float(xc), "y_center": float(yc),
                                     "density": float(H[i, j])})
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        n_periods = len(self.period_names)
        x_edges   = np.linspace(xmin, xmax, nbins + 1)
        y_edges   = np.linspace(ymin, ymax, nbins + 1)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_centers, y_centers)

        # Rebuild 2D arrays from cache and determine shared colour scale
        Hs = []
        for p in self.period_names:
            sub = df[df["period"] == p].sort_values(["y_center", "x_center"])
            H   = sub["density"].to_numpy().reshape(nbins, nbins)
            Hs.append(H)

        pos_vals = [H[H > 0] for H in Hs]
        vmin_g   = min(v.min() for v in pos_vals if len(v)) if pos_vals else 1
        vmax_g   = max(H.max() for H in Hs if H.max() > 0)
        norm_sh  = _mcolors.LogNorm(vmin=vmin_g, vmax=vmax_g)

        fig, axes = plt.subplots(
            2, n_periods,
            figsize=(8 * n_periods, 12),
            sharex=True, sharey="row",
            squeeze=False,
        )

        pcm_ref = None
        for idx, (p, H) in enumerate(zip(self.period_names, Hs)):
            ax_2d      = axes[0][idx]
            ax_profile = axes[1][idx]

            # ── 2D density map ──
            pcm     = ax_2d.pcolormesh(X, Y, H, norm=norm_sh,
                                       cmap=plt.cm.jet, shading="auto")
            pcm_ref = pcm
            ax_2d.set_title(p, fontsize=set_mpl.FONTSIZE_TITLE)
            if idx == 0:
                ax_2d.set_ylabel(r"$y / \sigma_y$", fontsize=set_mpl.FONTSIZE_LABEL)

            # ── Profile at y/σ_y ≈ 0 (middle y-bin) ──
            profile             = H[nbins // 2].astype(float)
            profile[profile == 0] = np.nan
            ax_profile.semilogy(x_centers, profile, linewidth=set_mpl.LINEWIDTH)
            ax_profile.set_xlabel(r"$x / \sigma_x$", fontsize=set_mpl.FONTSIZE_LABEL)
            if idx == 0:
                ax_profile.set_ylabel(
                    r"$\Phi\!\left(\frac{x}{\sigma_x}\,\middle|\,\frac{y}{\sigma_y}=0\right)$",
                    fontsize=set_mpl.FONTSIZE_LABEL,
                )

        # Single shared colorbar for all 2-D panels
        cbar = fig.colorbar(pcm_ref, ax=axes[0].ravel().tolist(),
                            fraction=0.02, pad=0.04)
        cbar.set_label("Count", fontsize=set_mpl.FONTSIZE_LABEL)

        plt.tight_layout()
        set_mpl.save(fig,
            self.dir_plot / f"gonzalez_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    def plot_sigmaxy(self) -> None:
        """Grid of σ_x / σ_y PDFs (one panel per period)."""
        assert logHist is not None, "rg_histograms library not available"

        CACHE = "sigmaxy_distributions"
        df = self._load_cache(CACHE)
        if df is None:
            raw  = self._load_gonzalez_all_raw()
            rows = []
            for p in self.period_names:
                sub = raw[raw["period"] == p]
                for axis, col in [("sigma_x", "sigmax"), ("sigma_y", "sigmay")]:
                    vals = [v for v in sub[col].tolist() if v > 100]
                    if not vals:
                        continue
                    q, a, _ = logHist(vals, 150)
                    for bc, pv in zip(a, q):
                        rows.append({"period": p, "axis": axis,
                                     "bin_center": float(bc), "pdf": float(pv)})
            df = pd.DataFrame(rows)
            self._save_cache(df, CACHE)

        n_periods = len(self.period_names)
        axis2color = {"sigma_x": "m", "sigma_y": "c"}
        axis2label = {"sigma_x": r"$\sigma_x$", "sigma_y": r"$\sigma_y$"}
        fig, axes = plt.subplots(1, n_periods,
                                 figsize=(8 * n_periods, 5),
                                 sharex=True, sharey=True,
                                 squeeze=False)
        for idx, p in enumerate(self.period_names):
            ax = axes[0][idx]
            for axis in ["sigma_x", "sigma_y"]:
                sub = df[(df["period"] == p) & (df["axis"] == axis)].sort_values("bin_center")
                if not sub.empty:
                    ax.loglog(sub["bin_center"], sub["pdf"],
                              linewidth=set_mpl.LINEWIDTH,
                              color=axis2color[axis], label=axis2label[axis])
            ax.set_xlabel(r"$\sigma$ (m)", fontsize=set_mpl.FONTSIZE_LABEL)
            if idx == 0:
                ax.set_ylabel("PDF", fontsize=set_mpl.FONTSIZE_LABEL)
            ax.set_title(p, fontsize=set_mpl.FONTSIZE_TITLE)
            ax.legend(fontsize=set_mpl.FONTSIZE_LEGEND)
        plt.tight_layout()
        set_mpl.save(fig,
            self.dir_plot / f"sigmaxy_{self.np_}_t_{self.t_threshold}_{self.region}.png")
        plt.show()

    # ------------------------------------------------------------------
    # Gap-analysis bridge methods (delegate to gap_analysis_plots.py)
    # ------------------------------------------------------------------

    def _load_scalars_all_periods(self) -> "dict[str, pd.DataFrame]":
        """Return ``{period: scalar_df}`` for all periods (convenience helper)."""
        return {p: self._load_scalars(p) for p in self.period_names}

    def _build_week2rg_by_party(self) -> "dict[str, dict[str, list[float]]]":
        """
        Aggregate weekly RG values per week and per party across all periods.

        Returns
        -------
        ``{iso_week_str: {party: [rg_values_metres]}}``
        """
        from .constants import PARTY_NAMES as _PARTY_NAMES
        import polars as _pl

        week2party: dict = {}

        for p in self.period_names:
            if self.store is not None:
                scalars_pl = self.store.read_scalars(p)
                wrg_pl     = self.store.read_weekly_rg_matrix(p)
                if scalars_pl.is_empty() or wrg_pl.is_empty():
                    continue
                user_cols = [c for c in wrg_pl.columns if c != "week"]
                wrg_long  = wrg_pl.unpivot(
                    on=user_cols, index="week",
                    variable_name="user_id", value_name="rg_value",
                )
                meta   = scalars_pl.select(["user_id", "party_government"])
                joined = wrg_long.join(meta, on="user_id", how="left")
                for row in joined.iter_rows(named=True):
                    week = row["week"]
                    rg   = row["rg_value"]
                    pty  = row["party_government"]
                    if rg is None or (isinstance(rg, float) and np.isnan(rg)):
                        continue
                    if pty not in _PARTY_NAMES:
                        continue
                    if week not in week2party:
                        week2party[week] = {party: [] for party in _PARTY_NAMES}
                    week2party[week][pty].append(rg)
            else:
                # Legacy file mode: pair weekly_rg JSON with scalar CSV
                self._ensure("weekly_rg")
                self._ensure("scalar")
                for f_wrg in self.period2wrgusers_files.get(p, []):
                    import os, json
                    f_sc = f_wrg.replace("weekly_rg", "all_scalars").replace(".json", ".csv.gz")
                    if not (os.path.isfile(f_wrg) and os.path.isfile(f_sc)):
                        continue
                    try:
                        df_sc = pd.read_csv(f_sc, compression="gzip")
                        pty   = df_sc["party_government"].iloc[0]
                        with open(f_wrg) as fh:
                            wrg = json.load(fh)
                        for week, val in wrg.items():
                            if isinstance(val, float) and np.isnan(val):
                                continue
                            if pty not in _PARTY_NAMES:
                                continue
                            if week not in week2party:
                                week2party[week] = {party: [] for party in _PARTY_NAMES}
                            week2party[week][pty].append(val)
                    except Exception:
                        continue

        return week2party

    def plot_gap1_npi_timeline(
        self,
        npi_events: "dict[str, object] | None" = None,
        save: bool = True,
    ) -> "plt.Figure":
        """
        Gap 1 – Causal framing.

        Weekly mobility by party with NPI event dates overlaid, so the reader
        can judge whether behavioural change preceded formal lockdown orders.

        Parameters
        ----------
        npi_events:
            Override the default event-date dict.  Keys are labels, values
            are ``datetime.date`` objects.  Defaults to
            ``gap_analysis_plots.NPI_EVENTS[self.region]``.
        save:
            If ``True`` the figure is written to the ``plots/`` directory.
        """
        from .gap_analysis_plots import plot_npi_timeline

        week2party = self._build_week2rg_by_party()
        out = (
            self.dir_plot
            / f"gap1_npi_timeline_{self.np_}_t_{self.t_threshold}_{self.region}.png"
        ) if save else None

        return plot_npi_timeline(
            week2rg_by_party=week2party,
            npi_events=npi_events,
            region=self.region,
            period_names=self.period_names,
            period_division=self.period_division,
            output_path=out,
        )

    def plot_gap2_sampling_bias(
        self,
        df_census: "pd.DataFrame | None" = None,
        income_col: "str | None" = None,
        save: bool = True,
    ) -> "plt.Figure":
        """
        Gap 2 – Sampling bias.

        Compare per-county user counts to census population to quantify
        which counties and socioeconomic groups are over/under-represented.

        Parameters
        ----------
        df_census:
            DataFrame with ``county``, ``pop2023``, and optionally an income
            or density column.  If ``None``, uses the project's built-in
            ``df_rurality`` (which has ``pop2023`` and ``area`` columns)
            merged with party information.
        income_col:
            Column name in ``df_census`` for median household income.
        save:
            If ``True`` the figure is written to the ``plots/`` directory.
        """
        from .gap_analysis_plots import (
            plot_sampling_bias_coverage,
            compute_users_per_county,
        )

        dfs = self._load_scalars_all_periods()
        df_users = compute_users_per_county(dfs)

        if df_census is None:
            # Build a minimal census proxy from the rurality table
            df_cen = self.df_rurality.copy()
            # Normalise county column name
            if "name" in df_cen.columns:
                df_cen = df_cen.rename(columns={"name": "county"})
            # Attach party info
            county2party_s = pd.Series(self.county2party, name="party_government")
            df_cen = df_cen.merge(
                county2party_s.reset_index().rename(columns={"index": "county"}),
                on="county", how="left",
            )
        else:
            df_cen = df_census.copy()

        out = (
            self.dir_plot
            / f"gap2_sampling_bias_{self.np_}_t_{self.t_threshold}_{self.region}.png"
        ) if save else None

        return plot_sampling_bias_coverage(
            df_users_per_county=df_users,
            df_census=df_cen,
            income_col=income_col,
            output_path=out,
        )

    def plot_gap3_party_rurality(
        self,
        metric: str = "radius_gyration",
        save: bool = True,
    ) -> "plt.Figure":
        """
        Gap 3 – Party / rurality conflation.

        OLS regression and partial-correlation analysis that disentangles
        the independent effects of party and rurality on ``metric``.

        Parameters
        ----------
        metric:
            Mobility metric column name (must be in the scalar parquet table).
        save:
            If ``True`` the figure is written to the ``plots/`` directory.
        """
        from .gap_analysis_plots import plot_party_rurality_regression

        dfs = self._load_scalars_all_periods()
        out = (
            self.dir_plot
            / f"gap3_party_rurality_{metric}_{self.np_}_t_{self.t_threshold}_{self.region}.png"
        ) if save else None

        return plot_party_rurality_regression(
            dfs_by_period=dfs,
            metric=metric,
            output_path=out,
        )

    def plot_gap4_post_lockdown_asymmetry(
        self,
        metric: str = "radius_gyration",
        save: bool = True,
    ) -> "plt.Figure":
        """
        Gap 4 – Post-lockdown asymmetry.

        Explicitly shows in Results how mobility change relative to the
        pre-lockdown baseline differs between Democratic and Republican
        counties across all three pandemic phases.

        Parameters
        ----------
        metric:
            Mobility metric column name (must be in the scalar parquet table).
        save:
            If ``True`` the figure is written to the ``plots/`` directory.
        """
        from .gap_analysis_plots import plot_post_lockdown_asymmetry

        dfs = self._load_scalars_all_periods()
        out = (
            self.dir_plot
            / f"gap4_post_lockdown_{metric}_{self.np_}_t_{self.t_threshold}_{self.region}.png"
        ) if save else None

        return plot_post_lockdown_asymmetry(
            dfs_by_period=dfs,
            metric=metric,
            period_names=self.period_names,
            output_path=out,
        )
