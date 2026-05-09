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
"""

import os
import json
import time
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import matplotlib as mtl
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

        base = Path(output_dir) if output_dir else DIR_MILESTONES_SERVER / region
        self.dir_users  = ifnotexistsmkdir(base / "dataxuser")
        self.dir_plot   = ifnotexistsmkdir(base / "plots")

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
                if rg is None or (isinstance(rg, float) and np.isnan(rg)):
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
                    if isinstance(val, float) and np.isnan(val):
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
        period2rg = {p: [] for p in self.period_names}
        for p in self.period_names:
            df = self._load_scalars(p)
            if not df.empty:
                period2rg[p] = df["radius_gyration"].dropna().tolist()
        fig, ax = plt.subplots(figsize=(8, 5))
        for p in self.period_names:
            vals = [v for v in period2rg[p] if v > 100]
            q, a, _ = logHist(vals, 200)
            ax.loglog(a, q, linewidth=3)
        ax.set_xlabel("Radius of Gyration (m)", fontsize=15)
        ax.set_ylabel("PDF", fontsize=15)
        ax.legend(self.period_names)
        plt.tight_layout()
        plt.savefig(self.dir_plot / f"rg_{self.np_}_hour_{self.t_threshold}_{self.region}.png", dpi=200)
        plt.show()

    def plot_rg_party_per_period(self) -> None:
        """RG distribution split by Democratic / Republican county, one subplot per period."""
        assert logHist is not None, "rg_histograms library not available"
        parties = PARTY_NAMES
        party2color = {"Democratic": "blue", "Republican": "red"}
        period2rg = {p: {party: [] for party in parties} for p in self.period_names}
        for p in self.period_names:
            df = self._load_scalars(p)
            if not df.empty and "party_government" in df.columns:
                for party in parties:
                    sub = df[df["party_government"] == party]
                    period2rg[p][party] = sub["radius_gyration"].dropna().tolist()
        for p in self.period_names:
            fig, ax = plt.subplots(figsize=(8, 5))
            for party in parties:
                vals = [v for v in period2rg[p][party] if v > 100]
                if vals:
                    q, a, _ = logHist(vals, 200)
                    ax.loglog(a, q, linewidth=3, color=party2color[party], label=party)
            ax.set_xlabel("Radius of Gyration (m)", fontsize=15)
            ax.set_ylabel("PDF", fontsize=15)
            ax.set_title(p)
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                self.dir_plot / f"rg_party_{self.np_}_t_{self.t_threshold}_{self.region}_{p}.png",
                dpi=200,
            )
            plt.show()

    def plot_rg_rurality_per_period(self) -> None:
        """RG distribution split by urban / rural county, one subplot per period."""
        assert logHist is not None, "rg_histograms library not available"
        rural2color = {"rural": "blue", "urban": "red"}
        period2rg = {p: {r: [] for r in RURALITY_LEVELS} for p in self.period_names}
        for p in self.period_names:
            df = self._load_scalars(p)
            if not df.empty and "rurality_level" in df.columns:
                for rur in RURALITY_LEVELS:
                    sub = df[df["rurality_level"] == rur]
                    period2rg[p][rur] = sub["radius_gyration"].dropna().tolist()
        for p in self.period_names:
            fig, ax = plt.subplots(figsize=(8, 5))
            for rur in RURALITY_LEVELS:
                vals = [v for v in period2rg[p][rur] if v > 100]
                if vals:
                    q, a, _ = logHist(vals, 200)
                    ax.loglog(a, q, linewidth=3, color=rural2color[rur], label=rur)
            ax.set_xlabel("Radius of Gyration (m)", fontsize=15)
            ax.set_ylabel("PDF", fontsize=15)
            ax.set_title(p)
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                self.dir_plot / f"rg_rurality_{self.np_}_t_{self.t_threshold}_{self.region}_{p}.png",
                dpi=200,
            )
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
        """Time series of average weekly RG."""
        week2rg = self._build_week2rg()
        fig, ax = plt.subplots(figsize=(8, 5))
        weeks   = list(week2rg.keys())
        avg_rg  = [np.nanmean(week2rg[w]) for w in weeks]
        ax.plot(range(len(weeks)), avg_rg, marker="o")
        ax.set_xlabel("Week", fontsize=15)
        ax.set_ylabel("Average Radius of Gyration (m)", fontsize=15)
        ax.set_title("Weekly average radius of gyration")
        plt.tight_layout()
        plt.savefig(
            self.dir_plot / f"weekly_rg_{self.np_}_t_{self.t_threshold}_{self.region}.png",
            dpi=200,
        )
        plt.show()

    def plot_rg_rurality_weekly(self) -> None:
        """Weekly avg RG with error bars, stratified by rurality."""
        rural2color = {"rural": "blue", "urban": "red"}
        period2rg_rural: dict = defaultdict(dict)

        for p in self.period_names:
            rg_rural, _ = self._load_weekly_rg_stratified(p)
            for week, strata in rg_rural.items():
                period2rg_rural[p][week] = strata

        fig, ax = plt.subplots(figsize=(8, 5))
        for rur in RURALITY_LEVELS:
            weeks_n, avg, err = [], [], []
            week_n = 0
            for p in self.period_names:
                for week in period2rg_rural[p]:
                    vals = period2rg_rural[p][week].get(rur, [])
                    avg.append(np.nanmean(vals) if vals else float("nan"))
                    err.append(np.nanstd(vals) / np.sqrt(len(vals)) if vals else 0)
                    weeks_n.append(week_n)
                    week_n += 1
            ax.scatter(weeks_n, avg, color=rural2color[rur], label=rur)
            ax.errorbar(weeks_n, avg, yerr=err, ecolor=rural2color[rur])
        ax.set_xlabel("Week", fontsize=15)
        ax.set_ylabel("Radius of Gyration (m)", fontsize=15)
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            self.dir_plot / f"weekly_rg_rurality_{self.np_}_t_{self.t_threshold}_{self.region}.png",
            dpi=200,
        )
        plt.show()

    def plot_rg_party_weekly(self) -> None:
        """Weekly avg RG with error bars, stratified by political party."""
        party2color = {"Democratic": "blue", "Republican": "red"}
        period2rg_party: dict = defaultdict(dict)

        for p in self.period_names:
            _, rg_party = self._load_weekly_rg_stratified(p)
            for week, strata in rg_party.items():
                period2rg_party[p][week] = strata

        fig, ax = plt.subplots(figsize=(8, 5))
        for party in PARTY_NAMES:
            weeks_n, avg, err = [], [], []
            week_n = 0
            for p in self.period_names:
                for week in period2rg_party[p]:
                    vals = period2rg_party[p][week].get(party, [])
                    avg.append(np.nanmean(vals) if vals else float("nan"))
                    err.append(
                        np.nanstd(vals) / np.sqrt(len(vals)) if vals else 0
                    )
                    weeks_n.append(week_n)
                    week_n += 1
            ax.scatter(weeks_n, avg, color=party2color[party], label=party)
            ax.errorbar(weeks_n, avg, yerr=err, ecolor=party2color[party])
        ax.set_xlabel("Week", fontsize=15)
        ax.set_ylabel("Radius of Gyration (m)", fontsize=15)
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            self.dir_plot / f"weekly_rg_party_{self.np_}_t_{self.t_threshold}_{self.region}.png",
            dpi=200,
        )
        plt.show()

    # ------------------------------------------------------------------
    # k-Radius of gyration
    # ------------------------------------------------------------------

    def plot_krg(self) -> None:
        """2-D histograms of RG vs k-RG for each k and period."""
        assert logHist is not None, "rg_histograms library not available"
        nbins = 40
        list_k = K_RADIUS_VALUES
        period2krg = {
            p: {f"rg_{k}": [] for k in list_k}
            for p in self.period_names
        }
        for p in self.period_names[1:]:          # skip pre-lockdown baseline
            df_all = self._load_scalars(p)
            if df_all.empty:
                continue
            period2krg[p]["rg_1"] = df_all["radius_gyration"].dropna().tolist()
            for k in list_k:
                col = f"rg_{k}"
                if col in df_all.columns:
                    period2krg[p][col] = (df_all[col].dropna() * 1000).tolist()

        for p in self.period_names[1:]:
            rg1 = np.array(period2krg[p]["rg_1"])
            for k in list_k:
                rgk  = np.array(period2krg[p][f"rg_{k}"])
                cond = (rg1 > 100) & (rgk > 100)
                rg1f = rg1[cond]
                rgkf = rgk[cond]
                if len(rg1f) == 0:
                    continue
                fig, ax = plt.subplots(figsize=(8, 5))
                plt.hist2d(
                    rg1f, rgkf,
                    bins=(nbins, nbins),
                    range=[[100, 1_500_000], [100, 1_500_000]],
                    density=True,
                    cmap=plt.cm.jet,
                    norm=mtl.colors.LogNorm(),
                )
                plt.colorbar()
                ax.set_xlabel("Radius of Gyration (m)", fontsize=15)
                ax.set_ylabel(f"{k}-Radius of Gyration (m)", fontsize=15)
                ax.set_title(p)
                plt.tight_layout()
                plt.savefig(
                    self.dir_plot / f"rg_krg_{p}_{self.np_}_t_{self.t_threshold}_k{k}_{self.region}.png",
                    dpi=200,
                )
                plt.show()

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def plot_distance(self) -> None:
        """Log-log PDF of total straight-line distance with power-law fit."""
        assert logHist is not None and powerlaw is not None, (
            "rg_histograms and powerlaw libraries required"
        )
        period2dist = {p: [] for p in self.period_names}
        for p in self.period_names:
            df = self._load_scalars(p)
            if not df.empty:
                period2dist[p] = df["distance"].dropna().tolist()

        fig, ax = plt.subplots(figsize=(8, 5))
        legend_items = []
        for idx, p in enumerate(self.period_names):
            vals = [v for v in period2dist[p] if v > 100]
            q, a, _ = logHist(vals, 200)
            ax.scatter(a, q, linewidth=3)
            ax.set_yscale("log")
            ax.set_xscale("log")
            fit   = powerlaw.Fit(vals)
            alpha = fit.power_law.alpha
            sigma = fit.power_law.sigma
            fit.power_law.plot_pdf(
                color=plt.rcParams["axes.prop_cycle"].by_key()["color"][idx],
                linestyle="--", linewidth=3, ax=ax,
            )
            legend_items += [rf"$\alpha$ = {alpha:.3f} $\pm$ {sigma:.3f}", p]
        ax.set_xlabel("Distance (m)", fontsize=15)
        ax.set_ylabel("PDF", fontsize=15)
        ax.legend(legend_items)
        plt.tight_layout()
        plt.savefig(
            self.dir_plot / f"distance_{self.np_}_t_{self.t_threshold}_{self.region}.png",
            dpi=200,
        )
        plt.show()

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------

    def plot_entropy(self) -> None:
        """PDF for each of the three entropy types across periods."""
        assert logHist is not None, "rg_histograms library not available"
        for etype in ["random_entropy", "uncorrelated_entropy", "real_entropy"]:
            period2ent = {p: [] for p in self.period_names}
            for p in self.period_names:
                df = self._load_scalars(p)
                if not df.empty and etype in df.columns:
                    period2ent[p] = df[etype].dropna().tolist()
            fig, ax = plt.subplots(figsize=(8, 5))
            for p in self.period_names:
                bins = 12 if etype == "random_entropy" else 200
                q, a, _ = logHist(period2ent[p], bins)
                ax.plot(a, q, linewidth=4)
            ax.set_xlabel(etype.replace("_", " ").title(), fontsize=15)
            ax.set_ylabel("PDF", fontsize=15)
            ax.legend(self.period_names)
            plt.tight_layout()
            plt.savefig(
                self.dir_plot / f"{etype}_{self.region}.png", dpi=200
            )
            plt.show()

    # ------------------------------------------------------------------
    # S(t) exploration curve
    # ------------------------------------------------------------------

    def plot_St(self) -> None:
        """S(t) scatter with power-law fit per period."""
        assert power_fit is not None, "rg_fits library required"
        fig, ax = plt.subplots(figsize=(8, 5))
        legend_items = []
        MAX_PEOPLE = 400_000

        for p in self.period_names:
            period2St = self._load_st_dict(p, max_people=MAX_PEOPLE)

            l_mean = np.array([np.mean(period2St[h]) for h in sorted(period2St)])
            mask   = np.isfinite(l_mean)
            l_mean = l_mean[mask]
            t_arr  = np.arange(len(l_mean))

            slope, std_err, _r, _i = power_fit(t_arr, l_mean)
            u_spacing = t_arr ** slope

            ax.scatter(t_arr, l_mean, s=60, alpha=0.7)
            ax.loglog(t_arr[10:], u_spacing[10:], linestyle="dashed")
            legend_items += [rf"$\mu$ = {slope:.3f} $\pm$ {std_err:.3f}", p]

        ax.set_xlabel("t (h)", fontsize=15)
        ax.set_ylabel("S(t)", fontsize=15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(legend_items)
        plt.tight_layout()
        plt.savefig(
            self.dir_plot / f"St_{self.np_}_t_{self.t_threshold}_{self.region}.png",
            dpi=200,
        )
        plt.show()

    # ------------------------------------------------------------------
    # Location frequency
    # ------------------------------------------------------------------

    def plot_frequency(self) -> None:
        """Bar chart of average location frequency by rank (top 9)."""
        max_rank = 9
        Period2Rank = {
            p: self._load_frequency_by_rank(p, max_rank=max_rank)
            for p in self.period_names
        }

        df_avg = pd.DataFrame(
            {
                p: [np.mean(Period2Rank[p][r]) for r in range(1, max_rank + 1)]
                for p in self.period_names
            }
        )
        x   = np.arange(max_rank)
        w   = 0.3
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, p in enumerate(self.period_names):
            ax.bar(x + idx * w, df_avg[p], width=w, label=p, edgecolor="black")
        ax.set_xlabel("Rank", fontsize=15)
        ax.set_ylabel(r"$\langle k \rangle$", fontsize=15)
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            self.dir_plot / f"frequency_rank_{self.np_}_t_{self.t_threshold}_{self.region}.png",
            dpi=200,
        )
        plt.show()

    # ------------------------------------------------------------------
    # Gonzalez trajectory shape
    # ------------------------------------------------------------------

    def plot_gonzalez(
        self,
        xmin: float = -1.5, xmax: float = 1.5,
        ymin: float = -2.2, ymax: float = 2.2,
        nbins: int = 40,
    ) -> None:
        """2-D log-colourmap of normalised trajectory shape."""
        for p in self.period_names:
            x, y, _, _ = self._load_gonzalez(p)
            X, Y = np.meshgrid(
                np.linspace(xmin, xmax, nbins),
                np.linspace(ymin, ymax, nbins),
            )
            H, _, _ = np.histogram2d(x, y, bins=(nbins, nbins),
                                      range=[[xmin, xmax], [ymin, ymax]])
            H = H.T
            fig, ax = plt.subplots(figsize=(8, 5))
            pcm = ax.pcolormesh(
                X, Y, H, norm=mtl.colors.LogNorm(),
                cmap=plt.cm.jet, shading="auto",
            )
            fig.colorbar(pcm).set_label("Number of points")
            ax.set_xlabel(r"$x / \sigma_x$", fontsize=15)
            ax.set_ylabel(r"$y / \sigma_y$", fontsize=15)
            ax.set_title(p)
            plt.tight_layout()
            plt.savefig(
                self.dir_plot / f"gonzalez_{p}_{self.np_}_t_{self.t_threshold}_{self.region}.png",
                dpi=200,
            )
            plt.show()

    def plot_sigmaxy(self) -> None:
        """Log-log PDF of σ_x and σ_y per period."""
        assert logHist is not None, "rg_histograms library not available"
        for p in self.period_names:
            _, _, sx, sy = self._load_gonzalez(p)
            fig, ax = plt.subplots(figsize=(8, 5))
            q, a, _ = logHist([v for v in sx if v > 100], 150)
            ax.loglog(a, q, linewidth=3, color="m", label=r"$\sigma_x$")
            q1, a1, _ = logHist([v for v in sy if v > 100], 150)
            ax.loglog(a1, q1, linewidth=3, color="c", label=r"$\sigma_y$")
            ax.set_xlabel(r"$\sigma$ (m)", fontsize=15)
            ax.set_ylabel("PDF", fontsize=15)
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                self.dir_plot / f"sigmaxy_{p}_{self.np_}_t_{self.t_threshold}_{self.region}.png",
                dpi=200,
            )
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
