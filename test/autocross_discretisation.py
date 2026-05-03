"""
Autocross track layout showing the QSS solver discretisation.

Shows the discrete points at which the solver evaluates forces
and velocities, before and after track refinement.

Usage:
    python test/autocross_discretisation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.autocross_generator import build_standard_autocross
from solver.qss_speed import refine_track

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def _draw_track_surface(ax, x, y, half_width=1.5):
    """Draw track boundaries and shaded surface."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    length = np.maximum(np.sqrt(dx**2 + dy**2), 1e-9)
    nx = -dy / length
    ny = dx / length

    x_inner = x + half_width * nx
    y_inner = y + half_width * ny
    x_outer = x - half_width * nx
    y_outer = y - half_width * ny

    # Shaded track surface
    ax.fill(
        np.concatenate([x_outer, x_inner[::-1]]),
        np.concatenate([y_outer, y_inner[::-1]]),
        color="#D0D0D0", alpha=0.5, zorder=0,
    )

    # Track boundaries
    ax.plot(x_inner, y_inner, color="black", linewidth=1, alpha=0.5, zorder=1)
    ax.plot(x_outer, y_outer, color="black", linewidth=1, alpha=0.5, zorder=1)


def plot_discretisation():
    track, _ = build_standard_autocross()

    x = track.x
    y = track.y
    n = len(track.s)
    avg_ds = np.mean(track.ds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: full track with track surface and discrete points as driving line
    _draw_track_surface(ax1, x, y)
    ax1.scatter(x, y, color="#2070B0", s=3, zorder=5, label="Solver segments")
    ax1.plot(x[0], y[0], "o", color="#30A050", markersize=10, zorder=6, label="Start")
    ax1.plot(x[-1], y[-1], "o", color="#D04040", markersize=10, zorder=6, label="Finish")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m]", fontsize=12)
    ax1.set_ylabel("y [m]", fontsize=12)
    ax1.set_title("Full Track", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="lower left")
    ax1.grid(True, alpha=0.3)

    # Highlight the zoomed region on the full track
    i_start = 200
    i_end = 350
    ax1.plot(x[i_start:i_end], y[i_start:i_end], color="#D04040", linewidth=1.5, alpha=0.6, zorder=7)
    # Bounding box
    pad = 5
    x_min, x_max = x[i_start:i_end].min() - pad, x[i_start:i_end].max() + pad
    y_min, y_max = y[i_start:i_end].min() - pad, y[i_start:i_end].max() + pad
    from matplotlib.patches import Rectangle
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                      linewidth=1.5, edgecolor="#D04040", facecolor="none",
                      linestyle="solid")
    ax1.add_patch(rect)

    # Right: zoomed into corner with track surface and individual nodes
    x_zoom = x[i_start:i_end]
    y_zoom = y[i_start:i_end]
    _draw_track_surface(ax2, x_zoom, y_zoom)
    ax2.scatter(x_zoom, y_zoom, color="#2070B0", s=25, zorder=5,
                label="Solver segments")

    # Draw segments between consecutive nodes
    for i in range(len(x_zoom) - 1):
        ax2.plot([x_zoom[i], x_zoom[i + 1]], [y_zoom[i], y_zoom[i + 1]],
                 color="#2070B0", linewidth=0.8, alpha=0.4, zorder=4)

    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]", fontsize=12)
    ax2.set_ylabel("y [m]", fontsize=12)
    ax2.set_title("Zoomed Corner Section", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Autocross Track Discretisation", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()

    path = os.path.join(SAVE_DIR, "autocross_discretisation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Total points: {n}, avg segment: {avg_ds:.2f} m")
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_discretisation()
