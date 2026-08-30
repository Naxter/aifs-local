"""Shared look of the figures in this repository."""

SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e5e4e0"
SURFACE = "#fcfcfb"


def style_axis(ax, title=None):
    """Recessive grid and axes, title left-aligned above the panel."""
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
