# Skill: Paper ↔ Pipeline Synchronisation (HumMobCov)

## Purpose

The **final goal of the HumMobCov project is to produce the paper**:

```
.github/paper/
    main.tex                   ← root document (authors, abstract, sections)
    introduction.tex
    results.tex                ← all figures for California (main analysis)
    discussion.tex
    methods.tex                ← notation, dataset description, metric formulas
    supplementary_informations.tex  ← Massachusetts + stability analysis (np_=20, t=8 h)
```

This skill governs how the pipeline and the paper co-evolve:

1. **Paper → Pipeline**: every red-coloured `{\color{red} …}` section is an
   open question that must be answered by a new or updated pipeline computation.
2. **Pipeline → Paper**: every new plot or metric added to the pipeline that is
   relevant to the paper's narrative must be saved under the **exact** filename
   cited in the paper's `\includegraphics{…}` commands.
3. **New sections**: if a new pipeline capability (e.g. transition matrices,
   geohash grid analysis) does not yet have a place in the paper, a new `.tex`
   file must be created in `.github/paper/` and `\subfile{…}` added to
   `main.tex`.

---

## Section-to-pipeline mapping

### `methods.tex` ↔ preprocessing + metrics

| Paper sub-section | Pipeline component |
|-------------------|--------------------|
| Notations / stop points | `dataset_info.preprocess()` + `preprocess_shard_polars()` |
| Burst filter ($t^{burst}=1\,\text{h}$) | `utils.filter_()` (`TIME_THRESHOLD_HOURS`) |
| Period split (Jan15–Mar15 / Mar15–May15 / May15–Sep30) | `constants.PERIOD_DIVISION`, `PERIOD_NAMES` |
| User inclusion per period | `MIN_POINTS_PER_USER = 20` filter in `compute_all()` |
| Radius of gyration $R_g^\alpha$ (eq. rg) | `User.compute_radius_of_gyration` (legacy) / `_compute_radius_of_gyration_polars` (vectorized) |
| Gonzalez PCA / intrinsic frame (alg. phi) | `User.compute_gonzalez` / `_compute_gonzalez_polars` |
| S(t) exploration curve | `User.compute_St` / `_compute_st_polars` |
| Rank frequency $\langle k \rangle^{r,p}$ (eq. k_avg) | `User.compute_frequency_location` / `_compute_frequency_polars` |
| Random entropy $H_{random}$ (eq. rdm_entropy) | `User.compute_random_entropy` / `_compute_entropies_polars` |
| Uncorrelated entropy $H_{unc}$ (eq. unc_entropy) | `User.compute_uncorrelated_entropy` / `_compute_entropies_polars` |
| Real entropy $H_{real}$ (eq. real_entropy, LZ78) | `User.compute_real_entropy` / `_compute_real_entropy_polars` |
| k-Radius of gyration | `User.compute_krg` / `_compute_krg_polars` |
| Geodesic distance | `User.compute_straight_line_distance` / `_compute_distance_polars` |

### `results.tex` ↔ `plotter.py` (California, np_=20, t=1 h)

| Figure label | File cited in paper | Pipeline method | Current plotter filename |
|---|---|---|---|
| `\ref{radg_and_dist_california}` | `figures/main/rg_20_hour_1_CA.png` | `plotter.plot_rg()` | `rg_20_hour_1_CA.png` ✅ |
| `\ref{radg_and_dist_california}` | `figures/main/distance_20_hour_1_CA.png` | `plotter.plot_distance()` | `distance_20_t_1_CA.png` ⚠️ name mismatch |
| `\ref{returner_explorers_california}` | `rg_krg_15 jan - 15 march_20_hour_1_k_3_CA.png` | `plotter.plot_krg()` | `rg_krg_15 jan - 15 march_20_t_1_k3_CA.png` ⚠️ |
| `\ref{returner_explorers_california}` | `rg_krg_…_k_6_CA.png` **MISSING** | `plotter.plot_krg()` | k=6 not generated ❌ |
| `\ref{returner_explorers_california}` | `rg_krg_…_k_10.png` (no _CA suffix!) | `plotter.plot_krg()` | inconsistent suffix ⚠️ |
| `\ref{Gonzalez_california}` | `gonzalez_15 jan - 15 march_20_hour_1_CA.png` | `plotter.plot_gonzalez()` | `gonzalez_15 jan - 15 march_20_t_1_CA.png` ⚠️ |
| `\ref{Gonzalez_california}` | `conditional_gonzalez_…_CA.png` | **not yet implemented** ❌ | — |
| `\ref{sigma}` | `sigmaxy_15 jan - 15 march_20_hour_1_CA.png` | `plotter.plot_sigmaxy()` | `sigmaxy_15 jan - 15 march_20_t_1_CA.png` ⚠️ |
| `\ref{visitation_increase-frequency_california}` | `St_different_periods_fit_20_hour_1_CA.png` | `plotter.plot_St()` | `St_20_t_1_CA.png` ⚠️ |
| `\ref{visitation_increase-frequency_california}` | `rank_plot_20_hour_1_CA.png` | `plotter.plot_frequency()` | `frequency_rank_20_t_1_CA.png` ⚠️ |
| `\ref{weekly_county_rural}` | `avg_rg_all_week_parties_20_hour_1_CA.png` | `plotter.plot_rg_party_weekly()` | `weekly_rg_party_20_t_1_CA.png` ⚠️ |
| `\ref{weekly_county_rural}` | `avg_rg_all_week_rurals_20_hour_1_CA.png` | `plotter.plot_rg_rurality_weekly()` | `weekly_rg_rurality_20_t_1_CA.png` ⚠️ |
| `\ref{entropic_measures_california}` | `random_entropy_three_periods_CA.png` | `plotter.plot_entropy()` | `random_entropy_CA.png` ⚠️ |
| `\ref{entropic_measures_california}` | `uncorrelated_entropy_three_periods_CA.png` | `plotter.plot_entropy()` | `uncorrelated_entropy_CA.png` ⚠️ |
| `\ref{entropic_measures_california}` | `real_entropy_three_periods_CA` (no .png ext) | `plotter.plot_entropy()` | `real_entropy_CA.png` ⚠️ |
| `\ref{entropic_measures_california}` | `compression_rate_15 may - sept (3).png` | **not yet implemented** ❌ | — |

### `supplementary_informations.tex` ↔ `plotter.py` (Massachusetts, np_=20, t=1 h and t=8 h)

All MA figures go in `figures/Supplementary_figures/`. The naming convention
there is `<metric>_<np_>_hour_<t>.png` (no region suffix).

| Figure label | File cited | Status |
|---|---|---|
| `\ref{radg_and_dist}` | `rg_20_hour_1.png`, `distance_20_hour_1.png` | ⚠️ name mismatch |
| `\ref{returner_explorers_massachusetts}` | `rg_krg_…_k_3.png` | ⚠️ name mismatch |
| `\ref{Gonzalez_massachusetts}` | `gonzalez_…_20_hour_1.png` + `conditional_gonzalez_…` | conditional ❌ |
| `\ref{sigma_massachusetts}` | `sigmaxy_…_20_hour_1.png` | ⚠️ name mismatch |
| `\ref{visitation_increase-frequency_massachusetts}` | `St_different_periods_fit_20_hour_1.png`, `rank_plot_20_hour_1.png` | ⚠️ |
| `\ref{entropic_measures_massachusetts}` | `rdm_entropy_three_periods.png`, `uncorr_entropy_three_periods.png`, `real_entropy_three_periods.png` | ⚠️ |
| Stability t=8 h (all figures) | same but `_hour_8.png` | ⚠️ |

---

## Open problems (red sections) and pipeline answers

Each `{\color{red} …}` block in the paper is an explicit gap. Below is a
description of each gap, where it lives, and what pipeline code must produce
to resolve it.

---

### RED 1 — `results.tex` line 6: "Is there any difference in the results [between CA and MA]?"

**Location**: opening sentence of Results, before Fig. 1.

**What to produce**:
A direct quantitative comparison table or supplementary figure showing, for
each metric, the mean / median shift between lockdown and pre-lockdown periods
for both CA and MA.

**Pipeline action**:
Add a `plotter.plot_rg_ca_vs_ma()` method (or a notebook cell) that loads
`ParquetStore` for both regions and produces a **side-by-side** panel.  
Suggested filename (to be added to `results.tex` discussion paragraph):

```
figures/main/rg_ca_vs_ma_comparison_20_hour_1.png
```

---

### RED 2 — `results.tex` line 8: "Is there a consistent decrease across the entire distribution (a simple rescaling) or a change in shape?"

**Location**: Fig. 1 caption / paragraph.

**What to produce**:
Normalised RG distributions (divide by the mode or by the total count so all
three periods overlap) overlaid on the same axes.  The shape change (not just
rescaling) must be visible.  Quantify with a KS test or distribution overlap
coefficient across periods.

**Pipeline action**:
Add `plotter.plot_rg_normalised()` that plots the PDF divided by the period
mode, with a log-log KS-test annotation.

Suggested filename:

```
figures/main/rg_normalised_20_hour_1_CA.png
```

Also add to `results.tex`:
```latex
\includegraphics[width=0.6\textwidth]{figures/main/rg_normalised_20_hour_1_CA.png}
```

---

### RED 3 — `results.tex`: "From top to bottom k = 3, 6 **missing**, 10"

**Location**: Fig. 2 caption (`\ref{returner_explorers_california}`).

**What to produce**:
k=6 panels for all three periods.  K_RADIUS_VALUES already contains 6
(`constants.K_RADIUS_VALUES = [3, 6, 10]`), so the computation already
happens — only the plotter is not saving the file with the name expected by
the paper.

**Pipeline action**:
`plotter.plot_krg()` must save files named:

```
figures/main/rg_krg_{period}_20_hour_1_k_6_CA.png
```

i.e. the template must be `rg_krg_{period}_{np_}_hour_{t}_k_{k}_{region}.png`.
Fix `plotter.plot_krg()` `savefig` call to use `hour` not `t` in the token.

---

### RED 4 — `results.tex`: "What does it mean [peaks closer to one another]? How much closer?"

**Location**: Gonzalez paragraph, Fig. 3 (`\ref{Gonzalez_california}`).

**What to produce**:
1. Quantify the distance between the two dominant peaks of the 2-D Gonzalez
   distribution per period.
2. A `conditional_gonzalez` plot (cross-section at y=0) — **not yet
   implemented** in `plotter.py` — showing how the peak positions shift.

**Pipeline action**:
Add `plotter.plot_conditional_gonzalez()` that computes the marginal
$P(x/\sigma_x \mid y/\sigma_y \approx 0)$ and overlays the three periods.

Expected filenames (paper already cites them):

```
figures/main/conditional_gonzalez_{period}_20_hour_1_CA.png
```

---

### RED 5 — `results.tex` last figure caption: weekly RG unclear motivation

**Location**: Fig. 7 caption / `\ref{weekly_county_rural}`.

The red text asks: why was this analysis conducted? What do previous works
predict? Are urban/rural differences just a spatial-scale artefact?

**Pipeline action**:
No new computation needed, but the **discussion** paragraph in `results.tex`
needs expanding.  Add lockdown-period shading (vertical bands) to
`plot_rg_rurality_weekly()` and `plot_rg_party_weekly()`.

Expected filenames (paper cites):

```
figures/main/avg_rg_all_week_parties_20_hour_1_CA.png
figures/main/avg_rg_all_week_rurals_20_hour_1_CA.png
```

Fix `plotter.plot_rg_party_weekly()` and `plotter.plot_rg_rurality_weekly()`
to save with these exact names.

---

### RED 6 — Missing: `compression_rate` plot

**Location**: `\ref{entropic_measures_california}`, bottom-right panel.

File cited: `figures/main/compression_rate_15 may - sept (3).png`

**What is this**: the ratio $H_{real} / H_{random}$ per user, plotted as a
PDF per period.  This shows how much of the "missing entropy" is due to
structure (small ratio = more structured than random).

**Pipeline action**:
Add `plotter.plot_compression_rate()` that computes
`real_entropy / random_entropy` from the scalars parquet and plots its PDF
for all three periods.

Expected filenames (one per period, or all three on one figure):

```
figures/main/compression_rate_three_periods_20_hour_1_CA.png
```

Update the `\includegraphics` in `results.tex` to this exact name.

---

## MANDATORY RULES for plot filenames

**Every time a new plot is added to `plotter.py` that corresponds to a figure
in the paper, its `plt.savefig(…)` path MUST match the filename used in the
corresponding `\includegraphics{…}` command.**

### Canonical filename templates

| Figure family | Template |
|---|---|
| RG distribution | `rg_{np_}_hour_{t}_{region}.png` |
| Distance distribution | `distance_{np_}_hour_{t}_{region}.png` |
| RG vs k-RG 2-D | `rg_krg_{period}_{np_}_hour_{t}_k_{k}_{region}.png` |
| Gonzalez 2-D | `gonzalez_{period}_{np_}_hour_{t}_{region}.png` |
| Conditional Gonzalez | `conditional_gonzalez_{period}_{np_}_hour_{t}_{region}.png` |
| σ_x, σ_y | `sigmaxy_{period}_{np_}_hour_{t}_{region}.png` |
| S(t) exploration | `St_different_periods_fit_{np_}_hour_{t}_{region}.png` |
| Rank frequency | `rank_plot_{np_}_hour_{t}_{region}.png` |
| Weekly RG by party | `avg_rg_all_week_parties_{np_}_hour_{t}_{region}.png` |
| Weekly RG by rurality | `avg_rg_all_week_rurals_{np_}_hour_{t}_{region}.png` |
| Random entropy | `random_entropy_three_periods_{region}.png` |
| Uncorrelated entropy | `uncorrelated_entropy_three_periods_{region}.png` |
| Real entropy | `real_entropy_three_periods_{region}.png` |
| Compression rate | `compression_rate_three_periods_{np_}_hour_{t}_{region}.png` |
| RG normalised | `rg_normalised_{np_}_hour_{t}_{region}.png` |
| CA vs MA comparison | `rg_ca_vs_ma_comparison_{np_}_hour_{t}.png` |

For **Massachusetts main** figures (Supplementary): same templates but without
`_{region}` suffix (the paper cites them without `_MA`).

For **stability analysis** (t=8 h, t=24 h): append `_hour_8` or `_hour_24`.

**Region codes**: `CA` for California, no suffix for Massachusetts.

---

## Procedure when adding a new pipeline capability

1. **Compute**: add / update the metric in `vectorized_pipeline.py` or `User.py`.
2. **Store**: ensure it is written via `store.write_scalars_batch()` or the
   appropriate long-format writer.
3. **Plot**: add / update the method in `plotter.py`.
4. **Filename**: set `plt.savefig(self.dir_plot / "<canonical_name>")` matching
   the table above.
5. **Paper**: if the figure is already cited, verify the `\includegraphics`
   path matches exactly.  If it is a new result, add the `\includegraphics`
   command in the appropriate `.tex` file.
6. **New section**: if the capability represents a completely new analysis (not
   covered by existing sections), create a new `.tex` file and add
   `\subfile{<new_section>}` to `main.tex`.

---

## Currently missing `.tex` files

The following pipeline capabilities exist but have no corresponding paper
section yet.  A `.tex` file should be created when the analysis matures:

| Pipeline module | Suggested new .tex file |
|-----------------|------------------------|
| `transition_matrices/transition_pipeline.py` | `.github/paper/mobility_networks.tex` |
| `tile_counties_via_geohash.py` (geohash tiling) | `.github/paper/spatial_resolution.tex` |

---

## Quick reference: figure directories

```
.github/paper/
    figures/
        main/                    ← California figures (results.tex)
        Supplementary_figures/   ← Massachusetts + stability (supplementary)
```

All `plt.savefig(self.dir_plot / …)` calls in `plotter.py` must resolve to
one of these two directories depending on the region:

```python
# in plotter.__init__:
if self.region == "CA":
    self.dir_plot = paper_root / "figures" / "main"
else:  # MA
    self.dir_plot = paper_root / "figures" / "Supplementary_figures"
```
