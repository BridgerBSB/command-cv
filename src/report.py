"""Strike-Zone View report renderer.

Recreates the reference report: strike-zone box, crosshair = intended target
(catcher's glove), filled circle = actual pitch (Trackman), dashed line = miss.
Pure matplotlib (Agg), cv2-free. Coordinates are Trackman feet, catcher view
(x increases to the right).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .geometry import MissResult, StrikeZone

# Astros palette
NAVY = "#002D62"
ORANGE = "#EB6E1F"
GOLD = "#C9A23A"
INK = "#1B2330"
PANEL = "#F7F8FB"


def render_strike_zone_view(
    miss: MissResult,
    zone: StrikeZone,
    out_path: str,
    pitch_label: str = "",
    subtitle: str = "",
    approximate: bool = True,
    dpi: int = 150,
) -> str:
    """Render the Strike-Zone View to `out_path` (PNG). Returns the path."""
    fig = plt.figure(figsize=(7.6, 7.2), facecolor="white")
    ax = fig.add_axes([0.06, 0.07, 0.88, 0.80])
    ax.set_facecolor(PANEL)

    # Header
    fig.text(0.06, 0.945, "Strike-Zone View", fontsize=26, fontweight="bold",
             color=INK, ha="left", va="center")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.905, 0.905], color=GOLD, lw=2.5))
    if subtitle:
        fig.text(0.94, 0.945, subtitle, fontsize=11, color="#6B7280",
                 ha="right", va="center")

    # View bounds: pad ~1  width/height around the zone so big misses still show
    pad_x = max(zone.width_ft, 0.9)
    pad_z = max(zone.height_ft, 0.9)
    ax.set_xlim(zone.x_left - pad_x, zone.x_right + pad_x)
    ax.set_ylim(zone.z_bot - pad_z, zone.z_top + pad_z)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Plate-center vertical reference (dashed gold)
    ax.axvline(0.0, color=GOLD, lw=1.4, ls=(0, (4, 4)), zorder=1)

    # Strike-zone box
    ax.add_patch(Rectangle(
        (zone.x_left, zone.z_bot), zone.width_ft, zone.height_ft,
        fill=False, edgecolor=NAVY, lw=2.6, zorder=2))

    ix, iz = miss.intended_x, miss.intended_z
    ax_, az = miss.actual_x, miss.actual_z

    # Miss line (dashed) intended -> actual
    ax.plot([ix, ax_], [iz, az], color=GOLD, lw=2.4, ls=(0, (6, 5)), zorder=3)

    # Intended target = navy crosshair-in-circle
    ax.scatter([ix], [iz], s=320, facecolors="none", edgecolors=NAVY,
               linewidths=2.6, zorder=5)
    ax.plot([ix - 0.12, ix + 0.12], [iz, iz], color=NAVY, lw=2.0, zorder=6)
    ax.plot([ix, ix], [iz - 0.12, iz + 0.12], color=NAVY, lw=2.0, zorder=6)
    if pitch_label:
        ax.annotate(pitch_label, (ix, iz), textcoords="offset points",
                    xytext=(12, 10), fontsize=11, color=INK, fontweight="bold")

    # Actual pitch = orange filled circle with white crosshair
    ax.scatter([ax_], [az], s=300, facecolors=ORANGE, edgecolors="white",
               linewidths=1.6, zorder=5)
    ax.plot([ax_ - 0.10, ax_ + 0.10], [az, az], color="white", lw=1.6, zorder=6)
    ax.plot([ax_, ax_], [az - 0.10, az + 0.10], color="white", lw=1.6, zorder=6)

    # Miss readout (top-left inside panel)
    ax.text(0.03, 0.97, f"Miss: {miss.miss_in:.1f} in", transform=ax.transAxes,
            ha="left", va="top", fontsize=15, fontweight="bold", color=NAVY)
    where = []
    if miss.horiz_label != "on-line":
        where.append(miss.horiz_label)
    if miss.vert_label != "level":
        where.append(miss.vert_label)
    ax.text(0.03, 0.915, (" / ".join(where) or "on target"),
            transform=ax.transAxes, ha="left", va="top", fontsize=12, color=ORANGE)

    # Caption (legend line, like the reference)
    cap = "Crosshair = intended target  |  Circle = actual pitch  |  Dashed line = miss"
    fig.text(0.5, 0.045, cap, fontsize=9.5, color="#6B7280", ha="center")
    if approximate:
        fig.text(0.5, 0.018, "glove location approximate", fontsize=8.5,
                 style="italic", color="#9AA1AC", ha="center")

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return out_path
