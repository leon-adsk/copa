"""plot_style.py - one consistent, colour-blind-checked chart style for all figures."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules and "google.colab" not in sys.modules:
    matplotlib.use("Agg")          # headless when run as a script; notebooks keep their inline backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Ink and chrome
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
# Categorical slots (fixed order, validated palette): identity of anomaly types A/B/C
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
TYPE_COLOR = {"A": BLUE, "B": ORANGE, "C": AQUA}
# Status colours (reserved for meaning, always with a text label)
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
NORMAL = "#b9b8b0"   # neutral grey for "normal" data behind flagged points


def _register_office_fonts() -> None:
    for d in [Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts"),
              Path("C:/Windows/Fonts")]:
        if d.exists():
            for f in d.glob("[Cc]alibri*.tt*"):
                try:
                    font_manager.fontManager.addfont(str(f))
                except Exception:
                    pass


def apply() -> None:
    _register_office_fonts()
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Calibri missing on Colab/Linux
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 11, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 10,
        "lines.linewidth": 2.0, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, path: Path) -> None:
    from copa_data import atomic_write
    atomic_write(path, lambda tmp: fig.savefig(tmp, format=path.suffix.lstrip(".")))
    plt.close(fig)


def pct_formatter():
    """Plain percentage tick labels for log axes (Calibri has no superscript minus)."""
    from matplotlib.ticker import FuncFormatter
    import math
    return FuncFormatter(lambda v, _: f"{v:.{max(0, -math.floor(math.log10(v)))}f} %" if v > 0 else "")
