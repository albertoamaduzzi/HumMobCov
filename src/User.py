"""
User.py
=======
Defines the ``User`` class, which encapsulates all per-user mobility
computation and serialisation for the HumMobCov analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import skmob
from datetime import timedelta
from collections import defaultdict
from pathlib import Path
from shapely.wkt import loads
from shapely.geometry import Point

from .constants import (
    DIR_OUTPUT,
    K_RADIUS_VALUES,
    TIME_INTERVAL_S_MAX,
    FNAME_SCALARS,
    FNAME_GONZALEZ,
    FNAME_ST,
    FNAME_FREQ_RANK,
    FNAME_WEEKLY_RG,
)
from .utils import xy, t_stop, time_difference, ifnotexistsmkdir


class User:
    """
    Represents a single user's trajectory for one time period.

    Parameters
    ----------
    df : pd.DataFrame or None
        Raw stops DataFrame (columns: userId, clusterLatitude,
        clusterLongitude, begin, end, geohash7).  Pass ``None`` when
        loading pre-computed results.
    period : str
        Period name, e.g. ``"15 jan - 15 march"``.
    region : str
        Region identifier, e.g. ``"CA"`` or ``"MA"``.
    np_ : int
        Minimum number of points threshold used during preprocessing.
    t_threshold : int
        Time threshold in hours between successive stops.
    period_names2period_division : dict
        Mapping ``{period_name: [start_dt, end_dt]}``.
    uname : str or None
        User identifier.  Inferred from ``df`` when ``df`` is not None.
    output_dir : Path or str, optional
        Override per-user output directory.  Defaults to
        ``DIR_OUTPUT / region / "dataxuser"``.
    """

    def __init__(
        self,
        df,
        period: str,
        region: str,
        np_: int,
        t_threshold: int,
        period_names2period_division: dict,
        uname: str | None = None,
        output_dir: Path | str | None = None,
    ):
        self.np_         = np_
        self.t_threshold = t_threshold
        self.region      = region
        self.period      = period

        # Time axis for S(t) computation
        self.time_interval_s = np.arange(0, TIME_INTERVAL_S_MAX, self.t_threshold).astype(int)

        self.period_names2period_division = period_names2period_division
        self.df2save            = defaultdict(list)
        self.df2save_gonzalez   = defaultdict()
        self.week2rg            = defaultdict()

        # Output directory
        base = Path(output_dir) if output_dir else DIR_OUTPUT / region / "dataxuser"
        self.base_dir = ifnotexistsmkdir(base)

        if df is None:
            self.uname    = uname
            self.dict_df  = defaultdict()
        else:
            self.df   = skmob.TrajDataFrame(
                df,
                latitude="clusterLatitude",
                longitude="clusterLongitude",
                datetime="begin",
                user_id="userId",
            )
            self.uname = df.userId.iloc[0]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def time_filtering_traj_per_person(self, t_threshold: int) -> None:
        """
        Remove rows where the inter-stop gap is below ``t_threshold`` hours,
        accumulating partial gaps so no data is silently discarded.
        """
        from .utils import filter_

        df = self.df.sort_values(by="end")
        list_time_diff = [0]
        list_time_diff.extend(
            (
                (
                    np.roll(df.datetime.to_numpy(dtype="datetime64[s]"), -1)
                    - df.datetime.to_numpy(dtype="datetime64[s]")
                ) / 3600
            ).astype(int).tolist()
        )
        list_time_diff.pop()
        df["time_diff"] = list_time_diff
        list_loc = filter_(df["time_diff"], t_threshold)
        self.df = df.loc[list_loc]

    # ------------------------------------------------------------------
    # Mobility metrics
    # ------------------------------------------------------------------

    def _t_stop(self) -> list:
        """Return stop durations (minutes) for each row of ``self.df``."""
        return t_stop(self.df)

    def compute_radius_of_gyration(self) -> None:
        """
        Compute the time-weighted radius of gyration and store in
        ``self.df2save['radius_gyration']``.
        """
        avg_lat = np.mean(self.df.lat)
        avg_lon = np.mean(self.df.lng)
        total_time = sum(self._t_stop())
        summa_2 = 0.0
        for _id, grp in self.df.groupby("geohash7"):
            time_spent = sum(t_stop(grp))
            x, y = xy(self.df.lat, self.df.lng, avg_lat, avg_lon)
            summa_2 += (x[0] ** 2 + y[0] ** 2) * time_spent
        self.radius_gyration = np.sqrt(summa_2 / total_time)
        self.df2save["radius_gyration"] = self.radius_gyration

    def compute_weekly_radius_gyration(
        self, period: str, dictweek2npeople: dict,
        period_division: list, perodname2idx: dict,
    ) -> None:
        """Compute per-week radius of gyration for the given period."""
        if len(self.df) == 0:
            return
        for week in range(len(self.weeks) - 1):
            w_start = period_division[perodname2idx[period]]
            w_end   = period_division[perodname2idx[period] + 1]
            if not (w_start <= self.weeks[week] <= w_end):
                continue
            mask = [
                self.weeks[week] < self.df.iloc[i]["datetime"] < self.weeks[week + 1]
                for i in range(len(self.df))
            ]
            tmp = self.df.loc[mask]
            if len(tmp) <= 3:
                self.week2rg[week] = np.nan
                continue
            avg_lat = np.mean(tmp.lat)
            avg_lon = np.mean(tmp.lng)
            total_time = sum(t_stop(tmp))
            if total_time == 0:
                self.week2rg[week] = np.nan
                continue
            summa_2 = 0.0
            for _id, grp in tmp.groupby("geohash7"):
                time_spent = sum(t_stop(grp))
                x, y = xy(tmp.lat, tmp.lng, avg_lat, avg_lon)
                summa_2 += (x[0] ** 2 + y[0] ** 2) * time_spent
            self.week2rg[week] = np.sqrt(summa_2 / total_time)
            dictweek2npeople[str(self.weeks[week])] += 1

    def number_points_week(
        self, period: str, dictweek2npeople,
        period_division: list, perodname2idx: dict,
    ) -> dict:
        """Return ``{week_index: point_count}`` for the current period."""
        self.week2point = defaultdict()
        if len(self.df) == 0:
            return self.week2point
        for week in range(len(self.weeks) - 1):
            w_start = period_division[perodname2idx[period]]
            w_end   = period_division[perodname2idx[period] + 1]
            if not (w_start <= self.weeks[week] <= w_end):
                continue
            mask = [
                self.weeks[week] < self.df.iloc[i]["datetime"] < self.weeks[week + 1]
                for i in range(len(self.df))
            ]
            self.week2point[week] = len(self.df.loc[mask])
        df = pd.DataFrame([self.week2point])
        fname = FNAME_SCALARS.format(  # reuse naming pattern for week2point
            user=self.uname, period=self.period,
            np_=self.np_, t=self.t_threshold,
        ).replace("all_scalars", "week2point")
        df.to_csv(self.base_dir / fname, sep=",", compression="gzip")
        return self.week2point

    def compute_gonzalez(self) -> None:
        """
        Compute PCA rotation of trajectory (Gonzalez et al.) and save
        normalised coordinates + principal variances.
        """
        out_path = self.base_dir / FNAME_GONZALEZ.format(
            user=self.uname, period=self.period,
            np_=self.np_, t=self.t_threshold,
        )
        if out_path.exists():
            return

        mean_lon = np.array(self.df.lat).mean()
        mean_lat = np.array(self.df.lng).mean()
        proj_x, proj_y = xy(
            np.array(self.df.lat), np.array(self.df.lng),
            mean_lat, mean_lon,
        )
        shifted_lat = proj_y - proj_y.mean()
        shifted_lng = proj_x - proj_x.mean()

        all_zero = (shifted_lat < 1).all() and (shifted_lng < 1).all()
        if all_zero:
            shifted_lat = np.zeros(len(shifted_lat))
            shifted_lng = np.zeros(len(shifted_lat))

        Ixx = np.sum(shifted_lng ** 2)
        Iyy = np.sum(shifted_lat ** 2)
        Ixy = np.sum(shifted_lat * shifted_lng)
        mu  = np.sqrt(4 * Ixy ** 2 + Ixx ** 2 - 2 * Ixx * Iyy + Iyy ** 2)

        if all_zero:
            cos_theta = 0.0
        else:
            denom = (0.5 * Ixx - 0.5 * Iyy + 0.5 * mu)
            cos_theta = -Ixy / (denom * np.sqrt(1 + Ixy ** 2 / denom ** 2))

        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        if all_zero:
            rotated_lat = np.zeros(len(shifted_lat))
            rotated_lng = np.zeros(len(shifted_lat))
        else:
            rotated_lat = -cos_theta * shifted_lat + sin_theta * shifted_lng
            rotated_lng = -cos_theta * shifted_lng - sin_theta * shifted_lat

        valid = np.isfinite(rotated_lat) & np.isfinite(rotated_lng)
        rotated_lat = rotated_lat[valid]
        rotated_lng = rotated_lng[valid]

        sigma_lat = np.sqrt((rotated_lat ** 2).mean()) if len(rotated_lat) > 0 else 0.0
        sigma_lng = np.sqrt((rotated_lng ** 2).mean()) if len(rotated_lng) > 0 else 0.0
        if sigma_lat < 1e-5:
            sigma_lat = 0.0
        if sigma_lng < 1e-5:
            sigma_lng = 0.0

        if sigma_lat == 0 and sigma_lng != 0:
            norm_y = np.zeros(len(rotated_lat))
            norm_x = rotated_lng / sigma_lng
        elif sigma_lat != 0 and sigma_lng == 0:
            norm_y = rotated_lat / sigma_lat
            norm_x = np.zeros(len(rotated_lat))
        elif sigma_lat == 0 and sigma_lng == 0:
            norm_y = np.zeros(len(rotated_lat))
            norm_x = np.zeros(len(rotated_lng))
        else:
            norm_y = rotated_lat / sigma_lat
            norm_x = rotated_lng / sigma_lng

        self.df2save_gonzalez = pd.DataFrame({
            "x_norm": norm_x,
            "y_norm": norm_y,
            "sigmax": sigma_lng,
            "sigmay": sigma_lat,
        })
        self._savedf_gonzalez()

    def compute_random_entropy(self) -> None:
        self.df2save["random_entropy"] = (
            skmob.measures.individual
            .random_entropy(self.df, show_progress=False)
            .iloc[0]["random_entropy"]
        )

    def compute_uncorrelated_entropy(self) -> None:
        self.df2save["uncorrelated_entropy"] = (
            skmob.measures.individual
            .uncorrelated_entropy(self.df, show_progress=False)
            .iloc[0]["uncorrelated_entropy"]
        )

    def compute_real_entropy(self) -> None:
        self.df2save["real_entropy"] = (
            skmob.measures.individual
            .real_entropy(self.df, show_progress=False)
            .iloc[0]["real_entropy"]
        )

    def compute_home(self) -> None:
        """Compute home location and its geohash."""
        home = skmob.measures.individual.home_location(self.df, show_progress=False)
        mask_lat = self.df.lat == home.iloc[0]["lat"]
        mask_lng = self.df.lng == home.iloc[0]["lng"]
        self.geohash_home = self.df.loc[mask_lat & mask_lng]["geohash7"].unique()[0]
        self.df2save["home"]          = Point([home.iloc[0]["lat"], home.iloc[0]["lng"]])
        self.df2save["home_geohash7"] = self.geohash_home

    def compute_krg(self) -> None:
        """Compute k-radius of gyration for all k in K_RADIUS_VALUES."""
        for k in K_RADIUS_VALUES:
            rg = (
                skmob.measures.individual
                .k_radius_of_gyration(self.df, k, show_progress=False)
                .iloc[0][f"{k}k_radius_of_gyration"]
            )
            self.df2save[f"rg_{k}"] = rg

    def compute_straight_line_distance(self) -> None:
        self.df2save["distance"] = (
            skmob.measures.individual
            .distance_straight_line(self.df, show_progress=False)
            .iloc[0]["distance_straight_line"]
        )

    def compute_frequency_location(self) -> None:
        """Compute location frequency and rank, then save."""
        freq = (
            skmob.measures.individual
            .location_frequency(self.df, show_progress=False)["datetime"]
            .tolist()
        )
        rank = (
            skmob.measures.individual
            .frequency_rank(self.df, show_progress=False)["frequency_rank"]
            .tolist()
        )
        self.df2frequencyrank = pd.DataFrame({"frequency": freq, "rank": rank})
        self._save_df2frequencyrank()

    def compute_St(self) -> None:
        """Compute the exploration curve S(t) — distinct places vs. time."""
        t_jan = self.df["datetime"].apply(
            lambda x: int(
                np.timedelta64(
                    x.to_datetime64() - self.df.datetime.iloc[0], "h"
                ).astype(int)
            )
        )
        self.list_visited = []
        self.COUNT_NEW    = 0
        s_jan = self.df["geohash7"].apply(lambda x: self._add_append(x))
        self._fill_dict(s_jan, t_jan)

    def _add_append(self, loc) -> int:
        if loc not in self.list_visited:
            self.COUNT_NEW += 1
            self.list_visited.append(loc)
        return self.COUNT_NEW

    def _fill_dict(self, s_jan, t_jan) -> bool:
        list_time   = []
        list_visits = []
        t_arr = t_jan.to_numpy()
        s_arr = s_jan.to_numpy()
        for e in range(len(t_arr)):
            h = t_arr[e]
            if e < len(t_arr) - 1 and t_arr[e + 1] - t_arr[e] > 0:
                for _i in range(t_arr[e + 1] - t_arr[e]):
                    if h == t_arr[e] or (t_arr[e] < h < t_arr[e + 1]):
                        if h < max(self.time_interval_s):
                            list_time.append(int(h))
                            list_visits.append(int(s_arr[e]))
                            h += self.t_threshold
                    elif h < t_arr[e]:
                        pass
                    else:
                        break
            elif e == len(t_arr) - 1:
                while h < max(self.time_interval_s):
                    list_time.append(int(h))
                    list_visits.append(int(s_arr[e]))
                    h += self.t_threshold
        if len(list_time) == TIME_INTERVAL_S_MAX - 1:
            list_time.append(TIME_INTERVAL_S_MAX - 1)
            list_visits.append(int(s_arr[-1]))
        self.df_St = pd.DataFrame({"time": list_time, "visited_places": list_visits})
        self._save_df_St()
        return True

    def compute_fraction_time_user_is_present(self) -> None:
        """Compute q: fraction of period for which user's location is known."""
        start_dt, end_dt = self.period_names2period_division[self.period]
        total_hours = time_difference(start_dt, end_dt)
        df = self.df.sort_values(by="end")
        length_stop = df.apply(
            lambda row: time_difference(row["datetime"], row["end"]), axis=1
        )
        self.df2save["q"] = sum(length_stop) / total_hours

    # ------------------------------------------------------------------
    # County / geographic association
    # ------------------------------------------------------------------

    def _get_county(
        self,
        county_geojson,
        county2party: dict,
        county2rural: dict,
        debug: bool = False,
    ) -> bool:
        """
        Associate the user's home location with a county, rurality level,
        and governing political party.

        Tries to load a pre-existing scalar file first; falls back to the
        in-memory ``df2save`` if none exists.

        Returns ``True`` on success, ``False`` otherwise.
        """
        scalar_path = self.base_dir / FNAME_SCALARS.format(
            user=self.uname, period=self.period,
            np_=self.np_, t=self.t_threshold,
        )

        if scalar_path.exists():
            try:
                self.df2save = pd.read_csv(
                    scalar_path, sep=",", index_col=False, compression="gzip"
                )
            except Exception:
                scalar_path.unlink(missing_ok=True)
                return False

            if "home" not in self.df2save.keys() or "radius_gyration" not in self.df2save.keys():
                scalar_path.unlink(missing_ok=True)
                return False

            point_wkt = self.df2save["home"].tolist()[0]
            y_vals, x_vals = loads(point_wkt).xy
            point = Point([x_vals[0], y_vals[0]])
        else:
            point = self.df2save.get("home")
            if point is None:
                return False

        for idx, row in county_geojson.iterrows():
            try:
                if point.within(row["geometry"]):
                    name = county_geojson.at[idx, "name"]
                    self.df2save["county_home"]      = name
                    self.df2save["party_government"] = county2party.get(name, float("nan"))
                    self.df2save["rurality_level"]   = county2rural.get(name, float("nan"))
                    if debug:
                        print(f"Assigned county: {name}")
                    break
            except Exception:
                continue
        return True

    # ------------------------------------------------------------------
    # Weekly helpers
    # ------------------------------------------------------------------

    def divide_weeks(
        self,
        period_division: list,
        period: str,
        perodname2idx: dict,
        time_window: int = 7,
    ) -> list:
        """
        Build a list of weekly datetime boundaries within the current period.

        Returns the list and stores it as ``self.weeks``.
        """
        self.weeks = []
        count = 0
        start = period_division[perodname2idx[period]]
        end   = period_division[perodname2idx[period] + 1]
        while start + timedelta(days=count * time_window) < end:
            self.weeks.append(start + timedelta(days=count * time_window))
            count += 1
        fmt = "%Y-%m-%d %H:%M:%S"
        self.weeks_str = [w.strftime(fmt) for w in self.weeks]
        self.strweeks2dtweeks = dict(zip(self.weeks_str, self.weeks))
        self.dtweeks2strweeks = dict(zip(self.weeks, self.weeks_str))
        return self.weeks

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _savedf_gonzalez(self) -> None:
        out_path = self.base_dir / FNAME_GONZALEZ.format(
            user=self.uname, period=self.period,
            np_=self.np_, t=self.t_threshold,
        )
        if not out_path.exists():
            if isinstance(self.df2save_gonzalez, dict):
                self.df2save_gonzalez = pd.DataFrame(self.df2save_gonzalez)
            self.df2save_gonzalez.to_csv(out_path, index=False, compression="gzip")

    def _save_df_St(self) -> None:
        out_path = self.base_dir / FNAME_ST.format(
            user=self.uname, period=self.period,
            np_=self.np_, t=self.t_threshold,
        )
        self.df_St.to_csv(out_path, index=False, compression="gzip")

    def _save_df2frequencyrank(self) -> None:
        out_path = self.base_dir / FNAME_FREQ_RANK.format(
            user=self.uname, period=self.period,
            np_=self.np_, t=self.t_threshold,
        )
        self.df2frequencyrank.to_csv(out_path, index=False, compression="gzip")

    def _save_weekly_rg(self, period: str) -> None:
        out_path = self.base_dir / FNAME_WEEKLY_RG.format(
            user=self.uname, period=period,
            np_=self.np_, t=self.t_threshold,
        )
        with open(out_path, "w") as f:
            json.dump(self.week2rg, f, indent=2)

    def _save_df(self) -> None:
        """
        Save all scalar metrics to a compressed CSV (one row per user).
        Skips if the file already exists or if 'home' is not yet computed.
        """
        out_path = self.base_dir / FNAME_SCALARS.format(
            user=self.uname, period=self.period,
            np_=self.np_, t=self.t_threshold,
        )
        if out_path.exists() or "home" not in self.df2save:
            return
        pd.DataFrame(self.df2save, index=[0]).to_csv(
            out_path, index=False, compression="gzip"
        )
