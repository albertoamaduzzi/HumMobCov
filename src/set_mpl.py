"""
set_mpl.py — Shared matplotlib/seaborn configuration for HPQC plots.

Usage:
    import set_mpl
    set_mpl.setup()           # call once per session/notebook
    # then use the constants below in your plot code
"""

import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Global setup
# ---------------------------------------------------------------------------

def setup():
    """Apply project-wide matplotlib/seaborn defaults."""
    sns.set_style("whitegrid")
    plt.rcParams["axes.grid"] = True
    pd.set_option("display.max_rows", 100)
    warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# DPI / figure
# ---------------------------------------------------------------------------

DPI = 200


# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------

FONTSIZE_TITLE  = 25   # ax.set_title(...)
FONTSIZE_LABEL  = 25   # ax.set_xlabel / set_ylabel
FONTSIZE_TICK   = 20   # set_xticklabels / set_yticklabels
FONTSIZE_LEGEND = 20   # ax.legend(fontsize=...)
FONTSIZE_CBAR   = 25   # colorbar tick / label


# ---------------------------------------------------------------------------
# Line / marker parameters
# ---------------------------------------------------------------------------

LINEWIDTH       = 3      # theory curves
LINEWIDTH_THIN  = 1.5    # secondary / thin curves

SCATTER_SIZE_LG = 250    # large scatter (single-panel figures)
SCATTER_SIZE_MD = 150    # medium scatter (multi-panel figures)
SCATTER_SIZE_SM = 50     # small scatter (crowded panels)

ALPHA_THEORY    = 0.7    # line opacity — theory / mean-field data
ALPHA_SIM       = 0.4    # scatter opacity — stochastic simulations


# ---------------------------------------------------------------------------
# SIRV compartment colors
# ---------------------------------------------------------------------------

COLOR_SUSCEPTIBLE = "darkgreen"
COLOR_INFECTED    = "firebrick"
COLOR_RECOVERED   = "grey"
COLOR_VACCINATED  = "darkblue"
COLOR_BASELINE    = "darkorange"   # random / no-strategy reference curve


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

def palette_teal(n=3):
    """Light teal palette with *n* shades (n <= 3, darkest last)."""
    pp = sns.light_palette("teal", n_colors=6)
    return [pp[3], pp[5], pp[5]][:n]


def palette_teal_slate(n=3):
    """Teal + darkslategrey blend used for radius = 3, 2, 1 curves."""
    pp  = sns.light_palette("teal", n_colors=6)
    ppp = sns.light_palette("darkslategrey", n_colors=6)
    return [pp[3], pp[5], ppp[5]][:n]


def palette_PuBuGn(n=3):
    """PuBuGn colormap sampled at high-value positions."""
    cmap = plt.cm.get_cmap("PuBuGn")
    return [cmap(s) for s in [0.65, 0.80, 0.95][:n]]


# Heatmap colormap
CMAP_HEATMAP = "viridis"


# ---------------------------------------------------------------------------
# Convenience: save figure
# ---------------------------------------------------------------------------

def save(fig, filename, fmt="png", transparent=True):
    """Save *fig* with project-wide defaults (DPI, bbox, transparency).

    Parameters
    ----------
    fig        : matplotlib Figure
    filename   : output path (str or Path)
    fmt        : 'png' (default) or 'pdf'
    transparent: whether the background is transparent (default True)
    """
    fig.savefig(
        filename,
        format=fmt,
        dpi=DPI,
        transparent=transparent,
        bbox_inches="tight",
    )
