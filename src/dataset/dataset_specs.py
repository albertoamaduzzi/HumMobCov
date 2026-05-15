"""
dataset_specs.py
================
Collect and save a priori characteristics of the HumMobCov input and output
datasets without running the full analysis pipeline.

Two main entry-points:

* ``DatasetSpecs(region, np_, t_threshold)``
    Inspect object that lazily probes files on demand.

* ``collect_and_save(region, np_, t_threshold, output_path)``
    Convenience function: run all inspections, print a human-readable
    summary, and persist the results to ``output_path`` as JSON.

Metrics collected
-----------------
Input (raw parquet shards)
    file_size_gb        Disk size of each shard file.
    n_rows              Total stop-point records in the file (may skip
                        if the file is very large; controlled by
                        ``sample_input``).
    n_users             Unique user IDs in the file.
    date_range          [min_datetime, max_datetime] of the timestamp
                        column (if present).

Output (parquet store — per period × kind)
    store_dir_size_gb   Total size of all parquet files in the shard dir.
    n_users             Number of users (read from parquet footer only —
                        O(1), no row data loaded).
    memory_estimate_gb  Estimated RAM needed to load the full matrix into
                        a NumPy array (fixed-length kinds only).

Derived / cross-cutting
    users_per_period    {period: n_users} from the all_scalars kind.
    users_overlap       Number of users present in all three periods.
    s_matrix_ram_gb     Peak RAM estimate for loading all S matrices
                        (worst case: Python-list representation used by
                        the plotter).  Printed as a warning if > 4 GB.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Project constants (imported lazily to avoid circular dependency at package
# load time — dataset_specs may be called before the full src package is
# initialised).
# ---------------------------------------------------------------------------

def _load_constants():
    """Return the constants module, adding project root to sys.path if needed."""
    import sys
    _here = Path(__file__).resolve()
    # Walk up until we find the project root (has pyproject.toml or src/)
    project_root = _here.parent
    for _ in range(5):
        project_root = project_root.parent
        if (project_root / "src").is_dir():
            break
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src import constants  # noqa: PLC0415
    return constants


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dir_size_gb(path: Path) -> float:
    """Total size of all files under ``path`` (non-recursive), in GB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    return total / 1e9


def _file_size_gb(path: Path) -> float:
    try:
        return path.stat().st_size / 1e9
    except FileNotFoundError:
        return 0.0


def _safe_period(period: str) -> str:
    return period.replace(" ", "_").replace("/", "-")


def _store_shard_dir(base_dir: Path, period: str, kind: str, np_: int, t: int) -> Path:
    name = f"{kind}_period_{_safe_period(period)}_np_{np_}_t_{t}"
    return base_dir / name


def _users_from_store_fixed(shard_dir: Path) -> int:
    """
    Count users stored in a fixed-length parquet store directory.

    Reads only parquet *footer metadata* (O(1) per file) — never loads row
    data.  The index column (``"metric"``, ``"time"``, or ``"week"``) is
    excluded from the count.
    """
    INDEX_COLS = {"metric", "time", "week"}
    users: set[str] = set()
    if not shard_dir.exists():
        return 0
    for f in shard_dir.glob("*.parquet"):
        try:
            schema = pl.read_parquet_schema(f)
            users.update(c for c in schema if c not in INDEX_COLS)
        except Exception:
            continue
    return len(users)


def _users_from_store_long(shard_dir: Path) -> int:
    """
    Count users stored in a long-format parquet store directory.

    Scans only the ``user_id`` column — avoids loading the full table.
    """
    users: set[str] = set()
    if not shard_dir.exists():
        return 0
    for f in shard_dir.glob("*.parquet"):
        try:
            users.update(
                pl.scan_parquet(f)
                .select("user_id")
                .collect()["user_id"]
                .cast(pl.Utf8)
                .to_list()
            )
        except Exception:
            continue
    return len(users)


def _n_rows_store_long(shard_dir: Path) -> int:
    """Row count for a long-format shard directory (gonzalez, frequency)."""
    total = 0
    if not shard_dir.exists():
        return 0
    for f in shard_dir.glob("*.parquet"):
        try:
            total += pl.scan_parquet(f).select(pl.len()).collect().item()
        except Exception:
            continue
    return total


def _memory_estimate_gb_fixed(shard_dir: Path, kind: str) -> float:
    """
    Estimate RAM (GB) needed to load the full fixed-length matrix as float32.

    Layout:
        all_scalars : [14 metrics × n_users]  → 14 * n_users * 4 B
        S           : [1418 steps × n_users]  → 1418 * n_users * 4 B
        weekly_rg   : [n_weeks × n_users]     → n_weeks * n_users * 4 B
    """
    ROW_COUNTS = {"all_scalars": 14, "S": 1418, "weekly_rg": 40}
    n_users = _users_from_store_fixed(shard_dir)
    if n_users == 0:
        return 0.0
    n_rows = ROW_COUNTS.get(kind, 50)
    return n_users * n_rows * 4 / 1e9  # float32


def _s_matrix_python_list_gb(n_users: int, n_steps: int = 1418) -> float:
    """
    Worst-case RAM for the S(t) matrix when stored as a Python dict of lists
    (as done by ``plotter._load_st_dict``).

    Python float: ~24 B; list overhead: ~56 B + 8 B/elem; dict: ~200 B/entry.
    For large n: dominant cost ≈ n_steps * n_users * 32 B (empirically).
    """
    return n_steps * n_users * 32 / 1e9


# ---------------------------------------------------------------------------
# Input-file inspector
# ---------------------------------------------------------------------------

def _inspect_input_file(
    path: Path,
    timestamp_col: str = "datetime",
    user_col: str = "user_id",
    sample_rows: bool = True,
) -> dict[str, Any]:
    """
    Inspect a single raw parquet shard.

    Parameters
    ----------
    sample_rows : bool
        If True, count users and rows via lazy scanning (cheap).
        Always True — the parameter is kept for API clarity.
    """
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "file_size_gb": _file_size_gb(path),
        "n_rows": None,
        "n_users": None,
        "date_range": [None, None],
        "columns": [],
    }

    if not path.exists():
        return result

    try:
        schema = pl.read_parquet_schema(path)
        result["columns"] = list(schema.keys())
    except Exception as exc:
        result["schema_error"] = str(exc)
        return result

    cols = result["columns"]

    # Row count (lazy, no data loaded)
    try:
        result["n_rows"] = (
            pl.scan_parquet(path).select(pl.len()).collect().item()
        )
    except Exception:
        pass

    # Unique user count (scan user_id column only)
    if user_col in cols:
        try:
            result["n_users"] = (
                pl.scan_parquet(path)
                .select(pl.col(user_col).n_unique())
                .collect()
                .item()
            )
        except Exception:
            pass

    # Date range (scan timestamp column only)
    if timestamp_col in cols:
        try:
            agg = (
                pl.scan_parquet(path)
                .select([
                    pl.col(timestamp_col).min().alias("dt_min"),
                    pl.col(timestamp_col).max().alias("dt_max"),
                ])
                .collect()
            )
            dt_min = agg["dt_min"][0]
            dt_max = agg["dt_max"][0]
            result["date_range"] = [
                str(dt_min) if dt_min is not None else None,
                str(dt_max) if dt_max is not None else None,
            ]
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class DatasetSpecs:
    """
    Inspect and summarise the HumMobCov dataset characteristics.

    Parameters
    ----------
    region : str
        ``"CA"`` or ``"MA"``.
    np_ : int
        Min-points threshold used in the parquet store (default 20).
    t_threshold : int
        Time threshold in hours used in the parquet store (default 1).
    milestones_dir : Path, optional
        Override the computed-results root directory (default: value from
        ``constants.DIR_MILESTONES_SERVER``).
    raw_data_dir : Path, optional
        Override the raw input data directory.
    """

    FIXED_KINDS = ("all_scalars", "S", "weekly_rg")
    LONG_KINDS  = ("gonzalez", "frequency")
    ALL_KINDS   = FIXED_KINDS + LONG_KINDS

    def __init__(
        self,
        region: str,
        np_: int = 20,
        t_threshold: int = 1,
        milestones_dir: Path | None = None,
        raw_data_dir: Path | None = None,
    ):
        self.region      = region
        self.np_         = np_
        self.t_threshold = t_threshold

        constants = _load_constants()

        self._milestones_dir = Path(milestones_dir) if milestones_dir else (
            constants.DIR_MILESTONES_SERVER / region
        )
        self._raw_data_dir = Path(raw_data_dir) if raw_data_dir else (
            constants.DIR_RAW_DATA.get(region, Path("/dev/null"))
        )
        self._period_names = constants.PERIOD_NAMES
        self._period_names_to_division = constants.PERIOD_NAMES_TO_DIVISION

        # Discover raw input files
        if region == "CA":
            self._raw_files = sorted(self._raw_data_dir.glob("*.parquet")) \
                if self._raw_data_dir.exists() else []
        else:  # MA
            self._raw_files = [
                self._raw_data_dir / f
                for f in constants.LIST_FILES_MA
            ]

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def inspect_input(
        self,
        timestamp_col: str = "datetime",
        user_col: str = "user_id",
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Inspect all raw input parquet shards.

        Returns
        -------
        list of per-file dicts with keys:
            path, exists, file_size_gb, n_rows, n_users, date_range, columns
        """
        results = []
        n = len(self._raw_files)
        for i, f in enumerate(self._raw_files, 1):
            if verbose:
                print(f"  [{i}/{n}] {f.name} …", end="\r", flush=True)
            info = _inspect_input_file(
                Path(f),
                timestamp_col=timestamp_col,
                user_col=user_col,
            )
            results.append(info)
        if verbose:
            print()
        return results

    # ------------------------------------------------------------------
    # Output store
    # ------------------------------------------------------------------

    def inspect_store(self, verbose: bool = True) -> dict[str, dict[str, Any]]:
        """
        Inspect the computed-results parquet store.

        Returns
        -------
        Nested dict ``{period: {kind: {...stats...}}}``.
        """
        out: dict[str, dict[str, Any]] = {}

        for period in self._period_names:
            out[period] = {}
            for kind in self.ALL_KINDS:
                shard_dir = _store_shard_dir(
                    self._milestones_dir, period, kind,
                    self.np_, self.t_threshold,
                )
                if verbose:
                    print(f"  {period!r}  {kind} …", end="\r", flush=True)

                size_gb = _dir_size_gb(shard_dir)

                if kind in self.FIXED_KINDS:
                    n_users = _users_from_store_fixed(shard_dir)
                    mem_gb  = _memory_estimate_gb_fixed(shard_dir, kind)
                    entry: dict[str, Any] = {
                        "shard_dir": str(shard_dir),
                        "exists": shard_dir.exists(),
                        "store_dir_size_gb": round(size_gb, 4),
                        "n_users": n_users,
                        "memory_estimate_gb": round(mem_gb, 4),
                    }
                else:
                    n_users = _users_from_store_long(shard_dir)
                    n_rows  = _n_rows_store_long(shard_dir)
                    entry = {
                        "shard_dir": str(shard_dir),
                        "exists": shard_dir.exists(),
                        "store_dir_size_gb": round(size_gb, 4),
                        "n_users": n_users,
                        "n_rows": n_rows,
                    }
                out[period][kind] = entry

        if verbose:
            print()
        return out

    # ------------------------------------------------------------------
    # Cross-period user statistics
    # ------------------------------------------------------------------

    def users_per_period(self) -> dict[str, int]:
        """Return ``{period: n_users}`` from the ``all_scalars`` kind."""
        result = {}
        for period in self._period_names:
            shard_dir = _store_shard_dir(
                self._milestones_dir, period, "all_scalars",
                self.np_, self.t_threshold,
            )
            result[period] = _users_from_store_fixed(shard_dir)
        return result

    def user_sets_per_period(self) -> dict[str, set[str]]:
        """
        Return ``{period: set_of_user_ids}`` from the ``all_scalars`` store.

        Uses parquet footer only — no row data loaded.
        """
        INDEX_COLS = {"metric"}
        sets: dict[str, set[str]] = {}
        for period in self._period_names:
            shard_dir = _store_shard_dir(
                self._milestones_dir, period, "all_scalars",
                self.np_, self.t_threshold,
            )
            users: set[str] = set()
            if shard_dir.exists():
                for f in shard_dir.glob("*.parquet"):
                    try:
                        schema = pl.read_parquet_schema(f)
                        users.update(c for c in schema if c not in INDEX_COLS)
                    except Exception:
                        continue
            sets[period] = users
        return sets

    def user_overlap(self) -> dict[str, Any]:
        """
        Compute cross-period user overlap statistics.

        Returns
        -------
        dict with keys:
            per_period          {period: n_users}
            in_all_periods      n users present in every period
            in_exactly_two      n users present in exactly 2 periods
            in_only_one         n users present in exactly 1 period
            pairwise_overlap    {"{p1} ∩ {p2}": n}
        """
        sets = self.user_sets_per_period()
        periods = self._period_names

        result: dict[str, Any] = {
            "per_period": {p: len(sets[p]) for p in periods},
        }

        all_users = set().union(*sets.values())
        counts = {u: sum(1 for p in periods if u in sets[p]) for u in all_users}
        result["in_all_periods"] = sum(1 for c in counts.values() if c == len(periods))
        result["in_exactly_two"] = sum(1 for c in counts.values() if c == 2)
        result["in_only_one"]    = sum(1 for c in counts.values() if c == 1)

        pairwise: dict[str, int] = {}
        for i, p1 in enumerate(periods):
            for p2 in periods[i + 1:]:
                key = f"{p1} ∩ {p2}"
                pairwise[key] = len(sets[p1] & sets[p2])
        result["pairwise_overlap"] = pairwise

        return result

    # ------------------------------------------------------------------
    # Memory warnings
    # ------------------------------------------------------------------

    def memory_warnings(self, warn_threshold_gb: float = 4.0) -> list[str]:
        """
        Return a list of human-readable warnings for structures that may
        exhaust RAM when loaded during visualisation.

        Checks:
        * S(t) matrix per period (Python-list representation used by
          ``plotter._load_st_dict``, which is far larger than the numpy array).
        * Total across all periods.
        """
        warnings: list[str] = []
        period_users = self.users_per_period()
        total_s_gb = 0.0

        for period, n_users in period_users.items():
            numpy_gb  = _memory_estimate_gb_fixed(
                _store_shard_dir(
                    self._milestones_dir, period, "S",
                    self.np_, self.t_threshold,
                ),
                "S",
            )
            pylist_gb = _s_matrix_python_list_gb(n_users)
            total_s_gb += pylist_gb

            if pylist_gb > warn_threshold_gb:
                warnings.append(
                    f"S({period!r}): loading as Python lists ≈ {pylist_gb:.1f} GB "
                    f"({n_users:,} users × 1418 steps × ~32 B/float). "
                    f"NumPy array would be {numpy_gb:.1f} GB."
                )

        if total_s_gb > warn_threshold_gb:
            warnings.append(
                f"Total S(t) across all periods if kept simultaneously in RAM: "
                f"≈ {total_s_gb:.1f} GB — run plot_St() in isolation."
            )

        return warnings

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def collect(self, verbose: bool = True) -> dict[str, Any]:
        """
        Run all inspections and return a single report dict.

        The report is safe to serialise with ``json.dumps``.
        """
        report: dict[str, Any] = {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "region": self.region,
            "np_": self.np_,
            "t_threshold": self.t_threshold,
            "milestones_dir": str(self._milestones_dir),
            "raw_data_dir": str(self._raw_data_dir),
        }

        # ── Input files ───────────────────────────────────────────────
        if verbose:
            print("\n── Input files ──────────────────────────────────")
        input_data = self.inspect_input(verbose=verbose)
        report["input_files"] = input_data
        report["input_summary"] = {
            "n_files":           len(input_data),
            "total_size_gb":     round(sum(f["file_size_gb"] for f in input_data), 4),
            "total_rows":        sum(
                f["n_rows"] for f in input_data if f["n_rows"] is not None
            ),
            "total_users_rough": sum(
                f["n_users"] for f in input_data if f["n_users"] is not None
            ),
            "files_found":       sum(1 for f in input_data if f["exists"]),
        }

        # ── Output store ─────────────────────────────────────────────
        if verbose:
            print("\n── Output parquet store ─────────────────────────")
        store_data = self.inspect_store(verbose=verbose)
        report["store"] = store_data

        # Per-period user summary
        report["users_per_period"] = {
            p: store_data[p]["all_scalars"]["n_users"]
            for p in self._period_names
            if "all_scalars" in store_data.get(p, {})
        }

        # ── Cross-period overlap ─────────────────────────────────────
        if verbose:
            print("\n── Cross-period user overlap ─────────────────────")
        try:
            report["user_overlap"] = self.user_overlap()
        except Exception as exc:
            report["user_overlap"] = {"error": str(exc)}

        # ── Memory warnings ──────────────────────────────────────────
        warnings = self.memory_warnings()
        report["memory_warnings"] = warnings
        if verbose and warnings:
            print("\n── Memory warnings ───────────────────────────────")
            for w in warnings:
                print(f"  ⚠  {w}")

        # ── Period date ranges (from constants) ───────────────────────
        report["period_date_ranges"] = {
            p: [str(start.date()), str(end.date())]
            for p, (start, end) in self._period_names_to_division.items()
        }

        return report

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_summary(self, report: dict[str, Any]) -> None:
        """Print a concise human-readable summary of ``report``."""
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"Dataset specs  ·  region={self.region}  np_={self.np_}  t={self.t_threshold}")
        print(sep)

        s = report.get("input_summary", {})
        print(
            f"\nInput raw files : {s.get('files_found', '?')}/{s.get('n_files', '?')} found"
            f"   {s.get('total_size_gb', '?'):.3f} GB"
        )
        if s.get("total_rows"):
            print(f"  Total rows    : {s['total_rows']:>12,}")
        if s.get("total_users_rough"):
            print(f"  ~Users (sum)  : {s['total_users_rough']:>12,}  (double-counts cross-file users)")

        print("\nUsers per analysis period (parquet store):")
        for p, n in report.get("users_per_period", {}).items():
            start, end = report.get("period_date_ranges", {}).get(p, ["?", "?"])
            print(f"  {p:25s}  {n:>8,}  ({start} → {end})")

        overlap = report.get("user_overlap", {})
        if overlap and "in_all_periods" in overlap:
            print(f"\nUsers present in all {len(self._period_names)} periods : "
                  f"{overlap['in_all_periods']:>8,}")
            print(f"Users present in exactly 2 periods   : {overlap['in_exactly_two']:>8,}")
            print(f"Users present in exactly 1 period    : {overlap['in_only_one']:>8,}")
            for pair, n in overlap.get("pairwise_overlap", {}).items():
                print(f"  {pair:45s}  {n:>8,}")

        print("\nParquet store sizes (GB) and user counts:")
        col_w = max(len(p) for p in self._period_names) + 2
        print(f"  {'Period':{col_w}}  {'Kind':<14}  {'Size GB':>8}  {'Users':>9}  {'RAM est. GB':>11}")
        for period in self._period_names:
            for kind in self.ALL_KINDS:
                entry = report.get("store", {}).get(period, {}).get(kind, {})
                size  = entry.get("store_dir_size_gb", 0.0)
                users = entry.get("n_users", 0)
                mem   = entry.get("memory_estimate_gb", "—")
                mem_s = f"{mem:>11.3f}" if isinstance(mem, float) else f"{'—':>11}"
                print(
                    f"  {period:{col_w}}  {kind:<14}  {size:>8.4f}  {users:>9,}  {mem_s}"
                )

        warnings = report.get("memory_warnings", [])
        if warnings:
            print("\nMemory warnings:")
            for w in warnings:
                print(f"  ⚠  {w}")

        print(sep)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def collect_and_save(
    region: str,
    np_: int = 20,
    t_threshold: int = 1,
    output_path: Path | str | None = None,
    verbose: bool = True,
    milestones_dir: Path | None = None,
    raw_data_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Collect all dataset specs, print a summary, and save to JSON.

    Parameters
    ----------
    region : str
        ``"CA"`` or ``"MA"``.
    np_ : int
        Min-points threshold used for the parquet store.
    t_threshold : int
        Time threshold in hours used for the parquet store.
    output_path : Path or str, optional
        Where to save the JSON report.  Defaults to
        ``<project_root>/output/<region>/dataset_specs.json``.
    verbose : bool
        Print progress and summary (default True).
    milestones_dir, raw_data_dir : Path, optional
        Override default directories.

    Returns
    -------
    dict
        The full report dictionary.
    """
    specs = DatasetSpecs(
        region        = region,
        np_           = np_,
        t_threshold   = t_threshold,
        milestones_dir = milestones_dir,
        raw_data_dir   = raw_data_dir,
    )

    report = specs.collect(verbose=verbose)

    if verbose:
        specs.print_summary(report)

    # ── Persist to JSON ───────────────────────────────────────────────────
    if output_path is None:
        constants = _load_constants()
        out_dir = constants.DIR_OUTPUT / region
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "dataset_specs.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    if verbose:
        print(f"\nReport saved → {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Collect and save HumMobCov dataset specs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--region", choices=["CA", "MA"], default="CA")
    parser.add_argument("--np",    dest="np_",         type=int, default=20)
    parser.add_argument("--t",     dest="t_threshold", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None,
                        help="Path for the JSON output file.")
    parser.add_argument("--milestones-dir", type=Path, default=None)
    parser.add_argument("--raw-data-dir",   type=Path, default=None)
    args = parser.parse_args()

    collect_and_save(
        region        = args.region,
        np_           = args.np_,
        t_threshold   = args.t_threshold,
        output_path   = args.output,
        milestones_dir = args.milestones_dir,
        raw_data_dir   = args.raw_data_dir,
    )
