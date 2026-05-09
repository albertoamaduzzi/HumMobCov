"""
gap_analysis_plots.py
=====================
Standalone plotting functions addressing four methodological gaps in the paper.

Gap 1 – Causal framing
    Annotate the weekly-RG timeline with NPI event dates to show whether
    mobility drops preceded formal lockdown orders (voluntary vs. mandated).

Gap 2 – Sampling bias
    Quantify over/under-representation of each county by comparing observed
    user counts to census population figures; expose income-skew patterns.

Gap 3 – Party/rurality conflation
    OLS regression and partial-correlation analysis to disentangle the
    independent contributions of political affiliation and rurality to a
    mobility metric.

Gap 4 – Post-lockdown asymmetry
    Explicitly plot the change in mobility between pandemic phases by party,
    making the claim in the paper's conclusion verifiable in the Results.

All functions are *standalone*: they accept plain DataFrames / dicts and
return ``matplotlib.figure.Figure`` objects.  No side effects; saving is
optional via the ``output_path`` argument.

Typical usage (inside the notebook)
------------------------------------
>>> from src.gap_analysis_plots import (
...     plot_npi_timeline,
...     plot_sampling_bias_coverage,
...     plot_party_rurality_regression,
...     plot_post_lockdown_asymmetry,
...     compute_users_per_county,
...     NPI_EVENTS,
... )
"""

from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from .constants import PARTY_NAMES, RURALITY_LEVELS, PERIOD_NAMES

# ---------------------------------------------------------------------------
# Optional heavyweight deps — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import statsmodels.formula.api as _smf  # type: ignore
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

# ---------------------------------------------------------------------------
# Default NPI event dates (override per-call via ``npi_events`` kwarg)
# Sources:
#   CA – https://www.gov.ca.gov/2020/03/19/governor-gavin-newsom-issues-stay-at-home-order/
#   MA – https://www.mass.gov/info-details/covid-19-state-of-emergency
# ---------------------------------------------------------------------------
NPI_EVENTS: dict[str, dict[str, datetime.date]] = {
    "CA": {
        "CA emergency declared":   datetime.date(2020, 3, 4),
        "SFBA shelter-in-place":   datetime.date(2020, 3, 16),
        "CA stay-at-home order":   datetime.date(2020, 3, 19),
        "CA phase-2 reopening":    datetime.date(2020, 5, 8),
    },
    "MA": {
        "MA emergency declared":   datetime.date(2020, 3, 10),
        "MA non-essential closure": datetime.date(2020, 3, 24),
        "MA phase-1 reopening":    datetime.date(2020, 5, 18),
    },
}

_PARTY_COLOR: dict[str, str] = {
    "Democratic": "#3366CC",
    "Republican": "#CC3333",
}
_RURAL_COLOR: dict[str, str] = {
    "rural": "#228B22",
    "urban": "#FF8C00",
}
_PERIOD_SHADE_COLORS = ["#E8E8E8", "#D8D8FF", "#FFD8D8"]


# ===========================================================================
# Gap 1 – Causal framing
# ===========================================================================

def plot_npi_timeline(
    week2rg_by_party: dict[str, dict[str, list[float]]],
    npi_events: dict[str, datetime.date] | None = None,
    region: str = "CA",
    period_names: list[str] | None = None,
    period_division: list[datetime.date | datetime.datetime] | None = None,
    metric_label: str = "Median radius of gyration (km)",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Weekly mobility time-series annotated with NPI event dates.

    Addresses **Gap 1** (causal vs. correlational framing): by overlaying the
    exact dates of formal NPIs, the reader can judge whether behavioural
    change preceded governmental action.

    Parameters
    ----------
    week2rg_by_party:
        ``{iso_week_str: {party: [rg_values_in_metres]}}``.
        ISO week strings should be in the form ``"2020-W03"``.
    npi_events:
        ``{event_label: date}``.  Defaults to ``NPI_EVENTS[region]``.
    region:
        ``"CA"`` or ``"MA"`` — used only to choose the default events dict.
    period_names:
        Optional period names for shading background bands.
    period_division:
        Ordered list of period boundary ``date``/``datetime`` objects matching
        ``period_names`` (one more element than ``period_names``).
    metric_label:
        Y-axis label.
    output_path:
        If provided the figure is saved to this path (PNG, dpi=200).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if npi_events is None:
        npi_events = NPI_EVENTS.get(region, {})

    weeks_sorted = sorted(week2rg_by_party.keys())
    if not weeks_sorted:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No weekly data available", ha="center", va="center")
        return fig

    # Try to parse ISO-week strings to real dates for a meaningful x-axis.
    def _iso_to_date(w: str) -> datetime.date | None:
        try:
            # "2020-W03" → first day (Monday) of that ISO week
            return datetime.date.fromisoformat(f"{w}-1")
        except ValueError:
            return None

    dates = [_iso_to_date(w) for w in weeks_sorted]
    has_dates = all(d is not None for d in dates)
    x_vals: list[Any] = dates if has_dates else list(range(len(weeks_sorted)))

    fig, ax = plt.subplots(figsize=(13, 5))

    for party in PARTY_NAMES:
        medians, q25, q75 = [], [], []
        for w in weeks_sorted:
            arr = np.asarray(week2rg_by_party[w].get(party, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                medians.append(float(np.median(arr)) / 1_000)   # m → km
                q25.append(float(np.percentile(arr, 25)) / 1_000)
                q75.append(float(np.percentile(arr, 75)) / 1_000)
            else:
                medians.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)

        color = _PARTY_COLOR.get(party, "gray")
        ax.plot(x_vals, medians, color=color, linewidth=2.5, label=party)
        ax.fill_between(x_vals, q25, q75, color=color, alpha=0.15)

    # Optional period shading
    if has_dates and period_division and period_names:
        for i, (pname, pcolor) in enumerate(
            zip(period_names, _PERIOD_SHADE_COLORS)
        ):
            start = _to_date(period_division[i])
            end   = _to_date(period_division[i + 1])
            ax.axvspan(start, end, alpha=0.18, color=pcolor, label=pname, zorder=0)

    # NPI event vertical lines
    if has_dates and npi_events:
        linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1))]
        for j, (label, event_date) in enumerate(npi_events.items()):
            ls = linestyles[j % len(linestyles)]
            ax.axvline(
                event_date, color="black", linestyle=ls,
                linewidth=1.5, alpha=0.85, label=label,
            )

    if has_dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
        fig.autofmt_xdate()
    else:
        ax.set_xlabel("Week index")

    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(
        "Weekly mobility by party with NPI event dates\n"
        "(Gap 1: causal vs. voluntary behavioural change)",
        fontsize=11,
    )
    ax.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    return fig


# ===========================================================================
# Gap 2 – Sampling bias
# ===========================================================================

def compute_users_per_county(
    dfs_by_period: dict[str, pd.DataFrame],
    county_col: str = "county_home",
    party_col: str = "party_government",
    rurality_col: str = "rurality_level",
) -> pd.DataFrame:
    """Aggregate the number of unique users per county across all periods.

    Parameters
    ----------
    dfs_by_period:
        ``{period_name: scalar_df}`` as returned by
        ``plotter._load_scalars``.

    Returns
    -------
    DataFrame with columns ``county``, ``n_users``, and optionally
    ``party_government``, ``rurality_level``.
    """
    all_frames: list[pd.DataFrame] = []
    for df in dfs_by_period.values():
        if df.empty:
            continue
        keep = [c for c in [county_col, party_col, rurality_col, "user_id"] if c in df.columns]
        all_frames.append(df[keep].copy())

    if not all_frames:
        return pd.DataFrame(columns=["county", "n_users"])

    combined = pd.concat(all_frames, ignore_index=True)
    if "user_id" in combined.columns:
        combined = combined.drop_duplicates(subset="user_id")

    agg_cols = {c: (c, "first") for c in [party_col, rurality_col] if c in combined.columns}
    grouped = (
        combined.groupby(county_col)
        .agg(n_users=(county_col, "count"), **agg_cols)
        .reset_index()
        .rename(columns={county_col: "county"})
    )
    return grouped


def plot_sampling_bias_coverage(
    df_users_per_county: pd.DataFrame,
    df_census: pd.DataFrame,
    county_col: str = "county",
    users_col: str = "n_users",
    pop_col: str = "pop2023",
    income_col: str | None = None,
    density_col: str | None = None,
    party_col: str | None = "party_government",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Quantify sampling bias by comparing per-county user counts to census population.

    Addresses **Gap 2**: makes the opt-in bias *measurable* rather than
    merely acknowledged.

    Produces two panels:

    * **Left** — n_users vs. county population (log-log).  A slope of 1.0
      would mean perfect proportional representation; slope < 1 indicates
      large counties are over-represented relative to small ones.
    * **Right** — coverage rate (users per 1 000 residents) vs. median
      household income or population density, revealing socioeconomic skew.

    Parameters
    ----------
    df_users_per_county:
        DataFrame with at least ``county`` and ``n_users`` columns.
        Optionally ``party_government``, ``rurality_level``.
    df_census:
        DataFrame with at least the same ``county`` column and ``pop2023``
        (county population).  Optionally ``income_col`` or ``density_col``.
    county_col:
        Column name used to join the two DataFrames.
    pop_col:
        Column in ``df_census`` with county population counts.
    income_col:
        Optional column in ``df_census`` with median household income.
    density_col:
        Optional column in ``df_census`` with population density (pop/km²).
    output_path:
        If provided the figure is saved to this path (PNG, dpi=200).

    Returns
    -------
    matplotlib.figure.Figure
    """
    merged = df_users_per_county.merge(df_census, on=county_col, how="inner")
    merged = merged.dropna(subset=[users_col, pop_col])
    merged["coverage_per_1000"] = (
        merged[users_col].astype(float) / merged[pop_col].astype(float) * 1_000
    )

    # Determine secondary axis variable
    secondary_col: str | None = None
    secondary_label: str = ""
    if income_col and income_col in merged.columns:
        secondary_col   = income_col
        secondary_label = "Median household income ($)"
    elif density_col and density_col in merged.columns:
        secondary_col   = density_col
        secondary_label = "Population density (pop / km²)"
    else:
        # Fall back to population density derived from pop2023 / area if available
        if "area" in merged.columns:
            merged["_pop_density"] = merged[pop_col] / merged["area"].clip(lower=1)
            secondary_col   = "_pop_density"
            secondary_label = "Population density (pop / km²; derived)"

    n_panels = 2 if secondary_col else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    scatter_kw: dict[str, Any] = dict(s=70, alpha=0.75, edgecolors="white", linewidths=0.5)

    # ---- Panel 1: n_users vs population ----
    ax1 = axes[0]
    if party_col and party_col in merged.columns:
        for party in PARTY_NAMES:
            sub = merged[merged[party_col] == party]
            ax1.scatter(
                sub[pop_col], sub[users_col],
                color=_PARTY_COLOR.get(party, "gray"), label=party, **scatter_kw,
            )
    else:
        ax1.scatter(merged[pop_col], merged[users_col], color="steelblue", **scatter_kw)

    # OLS reference line in log space
    log_x = np.log10(np.clip(merged[pop_col].to_numpy(dtype=float), 1, None))
    log_y = np.log10(np.clip(merged[users_col].to_numpy(dtype=float), 1, None))
    valid  = np.isfinite(log_x) & np.isfinite(log_y)
    if valid.sum() > 1:
        m, b = np.polyfit(log_x[valid], log_y[valid], 1)
        xs = np.logspace(log_x[valid].min(), log_x[valid].max(), 100)
        ax1.loglog(xs, 10 ** (m * np.log10(xs) + b), "k--",
                   linewidth=1.5, label=f"OLS fit (slope={m:.2f})")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("County population (census)", fontsize=12)
    ax1.set_ylabel("Users in sample", fontsize=12)
    ax1.set_title(
        "User count vs. county population\n(proportional representation = slope 1.0)",
        fontsize=10,
    )
    ax1.legend(fontsize=9)

    # ---- Panel 2: coverage rate vs. income / density ----
    if n_panels == 2 and secondary_col:
        ax2 = axes[1]
        if party_col and party_col in merged.columns:
            for party in PARTY_NAMES:
                sub = merged[merged[party_col] == party]
                ax2.scatter(
                    sub[secondary_col], sub["coverage_per_1000"],
                    color=_PARTY_COLOR.get(party, "gray"), label=party, **scatter_kw,
                )
        else:
            ax2.scatter(
                merged[secondary_col], merged["coverage_per_1000"],
                color="steelblue", **scatter_kw,
            )
        ax2.set_xlabel(secondary_label, fontsize=12)
        ax2.set_ylabel("Coverage (users per 1 000 residents)", fontsize=12)
        ax2.set_title(
            "Sampling coverage vs. socioeconomic variable\n(Gap 2: sampling bias quantification)",
            fontsize=10,
        )
        ax2.legend(fontsize=9)

    plt.suptitle("Sampling bias quantification (Gap 2)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    return fig


def plot_sampling_bias_quintiles(
    df_users_per_county: pd.DataFrame,
    df_census: pd.DataFrame,
    county_col: str = "county",
    users_col: str = "n_users",
    pop_col: str = "pop2023",
    stratify_col: str | None = None,
    stratify_label: str = "Stratification variable",
    n_quintiles: int = 5,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Coverage rate by quintile of a stratification variable (e.g. income).

    Reveals monotonic under/over-representation across the distribution of
    a socioeconomic variable.

    Parameters
    ----------
    stratify_col:
        Column in ``df_census`` to stratify by (e.g. income, density).
        If ``None`` the function uses ``pop2023`` as the stratifying variable.

    Returns
    -------
    matplotlib.figure.Figure
    """
    merged = df_users_per_county.merge(df_census, on=county_col, how="inner")
    merged = merged.dropna(subset=[users_col, pop_col])
    merged["coverage_per_1000"] = (
        merged[users_col].astype(float) / merged[pop_col].astype(float) * 1_000
    )

    if stratify_col is None or stratify_col not in merged.columns:
        stratify_col  = pop_col
        stratify_label = "County population (quintiles)"

    merged = merged.dropna(subset=[stratify_col])
    merged["quintile"] = pd.qcut(
        merged[stratify_col], q=n_quintiles, labels=[f"Q{i+1}" for i in range(n_quintiles)],
    )

    agg = (
        merged.groupby("quintile", observed=True)["coverage_per_1000"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        agg["quintile"].astype(str), agg["mean"],
        yerr=agg["sem"], capsize=5,
        color="#4477AA", alpha=0.8, edgecolor="white",
        error_kw={"ecolor": "black", "elinewidth": 1.5},
    )
    ax.set_xlabel(f"{stratify_label} (quintile)", fontsize=12)
    ax.set_ylabel("Coverage (users per 1 000 residents)", fontsize=12)
    ax.set_title(
        f"Sampling coverage by {stratify_label} quintile\n"
        "(Gap 2: are richer / denser counties over-represented?)",
        fontsize=10,
    )
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    return fig


# ===========================================================================
# Gap 3 – Party / rurality conflation
# ===========================================================================

def plot_party_rurality_regression(
    dfs_by_period: dict[str, pd.DataFrame],
    metric: str = "radius_gyration",
    metric_label: str | None = None,
    party_col: str = "party_government",
    rurality_col: str = "rurality_level",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """OLS regression and partial-correlation analysis for party vs. rurality.

    Addresses **Gap 3**: disentangles the independent contributions of
    political affiliation and county rurality to a mobility metric.

    Three panels
    ------------
    1. Box-plots of the metric by party × rurality combination.
    2. OLS coefficient plot (party, rurality, interaction term) with 95% CIs.
    3. Partial-correlation bar chart (each predictor controlling for the other).

    Parameters
    ----------
    dfs_by_period:
        ``{period_name: scalar_df}`` — one DataFrame per period, each
        containing at least ``metric``, ``party_government``,
        ``rurality_level`` columns.
    metric:
        Column name of the mobility metric to analyse.
    metric_label:
        Y-axis / axis label; derived from ``metric`` if ``None``.
    output_path:
        If provided the figure is saved to this path (PNG, dpi=200).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metric_label is None:
        metric_label = metric.replace("_", " ").title()

    # Pool all periods
    frames: list[pd.DataFrame] = []
    for pname, df in dfs_by_period.items():
        if df.empty:
            continue
        cols = [c for c in [metric, party_col, rurality_col] if c in df.columns]
        if len(cols) < 3:
            continue
        sub = df[cols].dropna().copy()
        sub["period"] = pname
        frames.append(sub)

    if not frames:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    data = pd.concat(frames, ignore_index=True)

    # Scale m → km heuristically (median > 500 ≈ metres)
    if data[metric].median() > 500:
        data[metric] = data[metric] / 1_000
        metric_label += " (km)"

    data["is_republican"] = (data[party_col] == "Republican").astype(float)
    data["is_rural"]      = (data[rurality_col] == "rural").astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ---- Panel 1: box-plots per group ----
    ax1 = axes[0]
    groups = [
        ("Democratic", "urban"),
        ("Democratic", "rural"),
        ("Republican", "urban"),
        ("Republican", "rural"),
    ]
    box_data:   list[np.ndarray] = []
    box_labels: list[str]        = []
    box_colors: list[str]        = []
    for party, rurality in groups:
        sub = data[(data[party_col] == party) & (data[rurality_col] == rurality)][metric].dropna()
        box_data.append(sub.to_numpy())
        box_labels.append(f"{party[:3]}\n{rurality}")
        box_colors.append(_PARTY_COLOR.get(party, "gray"))

    bplot = ax1.boxplot(
        box_data, patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_xticklabels(box_labels, fontsize=9)
    ax1.set_ylabel(metric_label, fontsize=11)
    ax1.set_title("Mobility by party × rurality\n(median + IQR, no fliers)", fontsize=10)

    # ---- Panel 2: OLS coefficient plot ----
    ax2 = axes[1]
    if _HAS_STATSMODELS:
        formula = f"{metric} ~ is_republican + is_rural + is_republican:is_rural"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _smf.ols(formula, data=data).fit()

        term_map = {
            "is_republican":           "Republican\n(vs. Democrat)",
            "is_rural":                "Rural\n(vs. urban)",
            "is_republican:is_rural":  "Republican × Rural\n(interaction)",
        }
        coef_colors = ["#CC3333", "#228B22", "#AA22AA"]
        ci = model.conf_int()

        for i, (raw_term, pretty_term) in enumerate(term_map.items()):
            coef  = model.params[raw_term]
            lo    = ci.loc[raw_term, 0]
            hi    = ci.loc[raw_term, 1]
            ax2.barh(
                pretty_term, coef, color=coef_colors[i], alpha=0.75,
                xerr=[[coef - lo], [hi - coef]], capsize=5,
                error_kw={"ecolor": "black"},
            )

        ax2.axvline(0, color="black", linewidth=1)
        ax2.set_xlabel(f"OLS coefficient (Δ {metric_label})", fontsize=11)
        ax2.set_title(
            "OLS regression coefficients\n(95% CI; baseline = Dem, urban)",
            fontsize=10,
        )
        ax2.text(
            0.98, 0.02,
            f"Adj. R²={model.rsquared_adj:.3f}  N={len(data):,}",
            transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
        )
    else:
        ax2.text(
            0.5, 0.5,
            "Install statsmodels\nfor regression coefficients\n(pip install statsmodels)",
            ha="center", va="center", transform=ax2.transAxes,
        )
        ax2.set_title("OLS regression (statsmodels required)", fontsize=10)

    # ---- Panel 3: partial correlations ----
    ax3 = axes[2]
    if _HAS_SCIPY:
        pairs = [
            ("is_republican", "is_rural",      "Party effect\n| controlling rurality"),
            ("is_rural",      "is_republican", "Rurality effect\n| controlling party"),
        ]
        bar_colors = ["#993333", "#336633"]
        r_vals, sig_labels, labels = [], [], []
        for predictor, control, label in pairs:
            y_res = _ols_residuals(data[metric].to_numpy(), data[control].to_numpy())
            x_res = _ols_residuals(data[predictor].to_numpy(), data[control].to_numpy())
            r, p  = _scipy_stats.pearsonr(x_res, y_res)
            sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            r_vals.append(r)
            sig_labels.append(sig)
            labels.append(label)

        bars = ax3.barh(labels, r_vals, color=bar_colors, alpha=0.75)
        for bar, sig in zip(bars, sig_labels):
            x_pos = bar.get_width()
            offset = 0.005 if x_pos >= 0 else -0.005
            ha = "left" if x_pos >= 0 else "right"
            ax3.text(
                x_pos + offset, bar.get_y() + bar.get_height() / 2,
                sig, va="center", ha=ha, fontsize=11,
            )

        ax3.axvline(0, color="black", linewidth=1)
        ax3.set_xlabel("Partial Pearson r", fontsize=11)
        ax3.set_title(
            "Partial correlations with mobility\n(residualisation; each controls the other)",
            fontsize=10,
        )
    else:
        ax3.text(
            0.5, 0.5,
            "Install scipy\nfor partial correlations\n(pip install scipy)",
            ha="center", va="center", transform=ax3.transAxes,
        )
        ax3.set_title("Partial correlations (scipy required)", fontsize=10)

    plt.suptitle(
        "Gap 3: Party vs. rurality — disentangling independent effects",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    return fig


# ===========================================================================
# Gap 4 – Post-lockdown asymmetry
# ===========================================================================

def plot_post_lockdown_asymmetry(
    dfs_by_period: dict[str, pd.DataFrame],
    metric: str = "radius_gyration",
    metric_label: str | None = None,
    party_col: str = "party_government",
    period_names: list[str] | None = None,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Show how mobility changes across pandemic phases differ by party.

    Addresses **Gap 4**: the paper's conclusion states that post-lockdown
    willingness to resume mobility differs between Democratic and Republican
    counties, but this is not shown in the Results section.  This function
    makes that claim explicit and quantified with two panels:

    1. Median metric per period by party (line + IQR shading).
    2. Δ metric (period_i − period_baseline) by party as a grouped bar chart.

    Parameters
    ----------
    dfs_by_period:
        ``{period_name: scalar_df}`` with at least ``metric`` and
        ``party_government`` columns.
    metric:
        Column name of the mobility metric.
    metric_label:
        Axis label; derived from ``metric`` if ``None``.
    period_names:
        Ordered list of period names; defaults to ``constants.PERIOD_NAMES``.
    output_path:
        If provided the figure is saved to this path (PNG, dpi=200).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metric_label is None:
        metric_label = metric.replace("_", " ").title()
    if period_names is None:
        period_names = PERIOD_NAMES

    # Build per-period-per-party summary statistics
    rows: list[dict[str, Any]] = []
    scale_to_km = False
    for pname in period_names:
        df = dfs_by_period.get(pname, pd.DataFrame())
        if df.empty or metric not in df.columns:
            for party in PARTY_NAMES:
                rows.append({"period": pname, "party": party,
                             "median": np.nan, "q25": np.nan, "q75": np.nan})
            continue
        for party in PARTY_NAMES:
            arr = df[df[party_col] == party][metric].dropna().to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if not arr.size:
                rows.append({"period": pname, "party": party,
                             "median": np.nan, "q25": np.nan, "q75": np.nan})
                continue
            if np.nanmedian(arr) > 500:
                arr = arr / 1_000
                scale_to_km = True
            rows.append({
                "period": pname, "party": party,
                "median": float(np.nanmedian(arr)),
                "q25":    float(np.nanpercentile(arr, 25)),
                "q75":    float(np.nanpercentile(arr, 75)),
            })

    if scale_to_km and "(km)" not in metric_label:
        metric_label += " (km)"

    summary = pd.DataFrame(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    period_idx = {p: i for i, p in enumerate(period_names)}

    # ---- Panel 1: time-series per party ----
    for party in PARTY_NAMES:
        sub = summary[summary["party"] == party]
        x   = [period_idx[p] for p in sub["period"]]
        ax1.plot(
            x, sub["median"], marker="o", linewidth=2.5,
            color=_PARTY_COLOR.get(party, "gray"), label=party,
        )
        ax1.fill_between(
            x, sub["q25"], sub["q75"],
            color=_PARTY_COLOR.get(party, "gray"), alpha=0.15,
        )

    ax1.set_xticks(list(range(len(period_names))))
    ax1.set_xticklabels([_short_period(p) for p in period_names], fontsize=9)
    ax1.set_ylabel(metric_label, fontsize=11)
    ax1.set_title("Median mobility per period\n(IQR shaded)", fontsize=10)
    ax1.legend()

    # ---- Panel 2: Δ metric vs. pre-lockdown baseline ----
    baseline_period = period_names[0]
    x_bar = np.arange(len(period_names) - 1)
    width = 0.35

    for i, party in enumerate(PARTY_NAMES):
        base_rows = summary[
            (summary["period"] == baseline_period) & (summary["party"] == party)
        ]["median"].to_numpy()
        base_val = float(base_rows[0]) if len(base_rows) else np.nan

        deltas: list[float] = []
        for pname in period_names[1:]:
            med_rows = summary[
                (summary["period"] == pname) & (summary["party"] == party)
            ]["median"].to_numpy()
            delta = float(med_rows[0]) - base_val if len(med_rows) else np.nan
            deltas.append(delta)

        ax2.bar(
            x_bar + i * width, deltas, width,
            color=_PARTY_COLOR.get(party, "gray"),
            alpha=0.80, edgecolor="white", label=party,
        )

    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(x_bar + width / 2)
    ax2.set_xticklabels(
        [f"Δ vs baseline\n({_short_period(p)})" for p in period_names[1:]],
        fontsize=9,
    )
    ax2.set_ylabel(f"Δ {metric_label}", fontsize=11)
    ax2.set_title(
        "Change relative to pre-lockdown baseline\n"
        "(Gap 4: post-lockdown asymmetry by party — explicit Results figure)",
        fontsize=10,
    )
    ax2.legend()

    plt.suptitle(
        f"Post-lockdown behavioural asymmetry by party — {metric_label}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    return fig


# ===========================================================================
# Internal helpers
# ===========================================================================

def _ols_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return OLS residuals of y ~ 1 + x."""
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return np.zeros_like(y)
    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    residuals = np.full_like(y, np.nan)
    residuals[mask] = y[mask] - X @ beta
    return residuals


def _to_date(d: Any) -> datetime.date:
    """Convert datetime/date/string to ``datetime.date``."""
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    return datetime.date.fromisoformat(str(d))


def _short_period(p: str) -> str:
    """Abbreviate a period name for compact axis labels."""
    return p.replace("15 ", "")
