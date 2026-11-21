"""Shared plotting style for all DM1 notebooks (one visual system for the report).

Colors come from the validated reference palette (dataviz skill):
- Target classes are ordinal (low < medium < high) -> one-hue blue ramp,
  light -> dark (validated with --ordinal).
- Unordered groups (clusters, categories) -> categorical slots in fixed order;
  at most 3 slots when all pairs share the plot (scatter), otherwise fold
  into "Other" or facet.
- Correlations -> diverging blue <-> red with a gray midpoint.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"

CLASS_ORDER = ["low", "medium", "high"]
CLASS_COLORS = {"low": "#86b6ef", "medium": "#2a78d6", "high": "#104281"}

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8984"          # "Other" / de-emphasised marks
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#9b1c1c"])

# Two-column LaTeX article: column ~3.35 in, text width ~7.0 in.
COL_W = 3.35
FULL_W = 7.0


def setup():
    """Apply the shared rcParams; call once per notebook."""
    FIG_DIR.mkdir(exist_ok=True)
    TAB_DIR.mkdir(exist_ok=True)
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 2,
        "pdf.fonttype": 42,  # embed TrueType -> selectable text in the PDF
    })


def save(fig, name):
    """Save a figure as vector PDF (for LaTeX) and PNG (for quick viewing)."""
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.close(fig)
