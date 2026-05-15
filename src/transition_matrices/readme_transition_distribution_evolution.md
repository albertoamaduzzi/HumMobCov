# Distribution Evolution of Transition Counts and Presence — Mathematics

## 1. Objects and notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{G}$ | Set of geohash cells |
| $\mathcal{E}$ | Set of directed edges $(g_s, g_e)$ between cells |
| $T$ | Total number of time bins |
| $t \in \{0, \ldots, T-1\}$ | Time bin index |
| $B$ | Number of histogram bins (default 100) |
| $P$ | One of the three observation periods |

---

## 2. Presence distribution at a single time bin

At time bin $t$, each cell $g \in \mathcal{G}$ has a scalar value for each
presence column $c$ (e.g.\ $c = \texttt{count}$, $\texttt{probability}$, …):

$$
\mathbf{x}^{(c)}_t = \bigl(x^{(c)}_{t,g}\bigr)_{g \in \mathcal{G}_t}
$$

where $\mathcal{G}_t \subseteq \mathcal{G}$ is the set of cells that are
non-zero at time $t$.

### 2.1 Global bin edges

Bin edges are fixed **globally** across all time bins and periods, so that
distributions at different times are directly comparable:

$$
e^{(c)}_i = \min_g x^{(c)}_{\cdot,g}
  + i \cdot \frac{\max_g x^{(c)}_{\cdot,g} - \min_g x^{(c)}_{\cdot,g}}{B},
\quad i = 0, \ldots, B
$$

(Log-spaced edges are used when the column spans multiple orders of magnitude.)

### 2.2 Per-time-bin histogram vector

$$
X^{(c)}_t \in \mathbb{Z}_{\ge 0}^B, \qquad
X^{(c)}_{t,i}
  = \bigl|\{ g \in \mathcal{G}_t : e^{(c)}_i \le x^{(c)}_{t,g} < e^{(c)}_{i+1} \}\bigr|
$$

This vector is stored in the distribution DataFrame as columns
`bin_{c}` (the left edge $e^{(c)}_i$) and `count_{c}` ($X^{(c)}_{t,i}$),
one row per $(t, i)$ pair.

---

## 3. Transition distribution at a single time bin

Identically defined for edges: each edge $(g_s, g_e) \in \mathcal{E}$ carries
a value for each transition column $c$ (e.g.\ $c = \texttt{transitions}$,
$\texttt{transition\_probability}$):

$$
\mathbf{y}^{(c)}_t = \bigl(y^{(c)}_{t,e}\bigr)_{e \in \mathcal{E}_t}
$$

$$
Y^{(c)}_t \in \mathbb{Z}_{\ge 0}^B, \qquad
Y^{(c)}_{t,i}
  = \bigl|\{ e \in \mathcal{E}_t : e^{(c)}_i \le y^{(c)}_{t,e} < e^{(c)}_{i+1} \}\bigr|
$$

---

## 4. Period-aggregate distribution

Over a period $P = \{t_1, t_1+1, \ldots, t_2\}$ of $|P| = t_2 - t_1 + 1$
time bins, the **mean distribution** is:

$$
\bar{X}^{(c)}_P = \frac{1}{|P|} \sum_{t \in P} X^{(c)}_t \;\in \mathbb{R}^B
$$

This is what `visualization_distribution_transition_counts.py` shows in its
static plots: a single histogram-shaped curve that summarises the
*average shape* of the distribution over the whole period.

---

## 5. Moving average of the distribution

To smooth temporal fluctuations in the animated view, a **centred moving
average** of window width $w$ is defined:

$$
\widetilde{X}^{(c)}_{t,i}
  = \frac{1}{w} \sum_{k = -\lfloor w/2 \rfloor}^{\lfloor w/2 \rfloor}
    X^{(c)}_{t+k,i}
$$

with edge bins filled by the boundary value (constant padding).

The animation shows, at each frame $t$:
- the **instantaneous** distribution $X^{(c)}_t$ (bar or step chart),
- the **moving-average** distribution $\widetilde{X}^{(c)}_t$ (solid line).

---

## 6. Normalised probability distribution

When comparing across periods of different lengths it is useful to normalise:

$$
p^{(c)}_{t,i} = \frac{X^{(c)}_{t,i}}{\displaystyle\sum_{j=1}^{B} X^{(c)}_{t,j}}
$$

The sum is over all $B$ bins so $\sum_i p^{(c)}_{t,i} = 1$ for every $t$
where at least one entity is present.

---

## 7. DataFrame schema summary

### `distribution_df_presence`

| Column | Dtype | Description |
|--------|-------|-------------|
| `time_bin` | Int64 | Time bin index ($t$) |
| `period_observation` | Utf8 | Period name (e.g. `"15 jan - 15 march"`) |
| `bin_count_birth` | Float64 | Left edge $e_i$ for `count_birth` |
| `count_count_birth` | Int64 | $X^{(\texttt{count\_birth})}_{t,i}$ |
| `bin_count_death` | Float64 | Left edge $e_i$ for `count_death` |
| `count_count_death` | Int64 | $X^{(\texttt{count\_death})}_{t,i}$ |
| `bin_count_transit` | Float64 | Left edge $e_i$ for `count_transit` |
| `count_count_transit` | Int64 | $X^{(\texttt{count\_transit})}_{t,i}$ |
| `bin_count` | Float64 | Left edge $e_i$ for `count` |
| `count_count` | Int64 | $X^{(\texttt{count})}_{t,i}$ |
| `bin_probability` | Float64 | Left edge $e_i$ for `probability` |
| `count_probability` | Int64 | $X^{(\texttt{probability})}_{t,i}$ |

Rows: $T \times B$ (one per time-bin × bin-index combination).

### `distribution_df_transition`

| Column | Dtype | Description |
|--------|-------|-------------|
| `time_bin` | Int64 | Time bin index ($t$) |
| `period_observation` | Utf8 | Period name |
| `bin_transitions` | Float64 | Left edge $e_i$ for `transitions` |
| `count_transitions` | Int64 | $Y^{(\texttt{transitions})}_{t,i}$ |
| `bin_transition_probability` | Float64 | Left edge $e_i$ for `transition_probability` |
| `count_transition_probability` | Int64 | $Y^{(\texttt{transition\_probability})}_{t,i}$ |

Rows: $T \times B$.

---

## 8. Visualisation summary

| Plot | X-axis | Y-axis | Grouping |
|------|--------|--------|----------|
| Static (period aggregate) | bin left edge | $\bar{X}^{(c)}_{P,i}$ | one line/bar per period |
| Animation frame at $t$ | bin left edge | $X^{(c)}_{t,i}$ (bars) + $\widetilde{X}^{(c)}_t$ (line) | one panel per column |
