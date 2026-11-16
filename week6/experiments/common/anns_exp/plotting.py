"""Static matplotlib figures with the project's validated light palette.

Palette (reference instance, light mode):
  surface #fcfcfb, ink #0b0b0b, secondary #52514e, muted #898781,
  grid #e1e0d9, series in fixed order blue/orange/aqua/yellow/magenta/...
Series identity is always carried by marker shape + direct label as well as
colour; sequential heatmaps use the single blue ramp.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
       "#0d366b"]


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": SECOND,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "figure.dpi": 150,
    })


def clean(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED)


def save(fig, path, note=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=MUTED, ha="left")
        fig.tight_layout(rect=(0, 0.025, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def ecdf(ax, x, label, color, marker=None, ls="-"):
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    y = np.arange(1, x.size + 1) / x.size
    ax.plot(x, y, color=color, lw=2, ls=ls, label=label,
            marker=marker, markevery=max(1, len(x) // 12), ms=4)


def seq_color(v, vmin=0.0, vmax=1.0):
    i = int(round((v - vmin) / max(vmax - vmin, 1e-12) * (len(SEQ) - 1)))
    return SEQ[min(max(i, 0), len(SEQ) - 1)]
