"""
Side-by-side autocross and full skidpad track layouts.

Usage:
    python test/event_layouts.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.autocross_generator import build_standard_autocross
from events.skidpad import (
    SKIDPAD_CENTRE_RADIUS, SKIDPAD_INNER_RADIUS,
    SKIDPAD_OUTER_RADIUS, TRACK_WIDTH,
)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_event_layouts():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left: Autocross ---
    track, _ = build_standard_autocross()
    x, y = track.x, track.y

    dx = np.gradient(x)
    dy = np.gradient(y)
    length = np.maximum(np.sqrt(dx**2 + dy**2), 1e-9)
    nx = -dy / length
    ny = dx / length
    offset = 1.5  # half track width

    ax1.fill(
        np.concatenate([x + offset * nx, (x - offset * nx)[::-1]]),
        np.concatenate([y + offset * ny, (y - offset * ny)[::-1]]),
        color="#D0D0D0", alpha=0.5,
    )
    ax1.plot(x + offset * nx, y + offset * ny, color="black", linewidth=1, alpha=0.5)
    ax1.plot(x - offset * nx, y - offset * ny, color="black", linewidth=1, alpha=0.5)
    ax1.plot(x, y, color="#2070B0", linewidth=2, label="Driving line")
    ax1.plot(x[0], y[0], "o", color="#30A050", markersize=10, zorder=5, label="Start")
    ax1.plot(x[-1], y[-1], "o", color="#D04040", markersize=10, zorder=5, label="Finish")

    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m]", fontsize=12)
    ax1.set_ylabel("y [m]", fontsize=12)
    ax1.set_title("Autocross Track Layout", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="lower left")
    ax1.grid(True, alpha=0.3)

    # --- Right: Full Skidpad ---
    left_centre = (0.0, 0.0)
    right_centre = (2 * SKIDPAD_CENTRE_RADIUS, 0.0)

    theta = np.linspace(0, 2 * np.pi, 200)

    # Lane boundaries
    lane_left_x = SKIDPAD_INNER_RADIUS
    lane_right_x = SKIDPAD_OUTER_RADIUS
    lane_centre_x = SKIDPAD_CENTRE_RADIUS
    lane_extension = 3.0
    lane_bottom_y = -SKIDPAD_OUTER_RADIUS - lane_extension
    lane_top_y = SKIDPAD_OUTER_RADIUS + lane_extension

    # Shade circle track areas
    for centre in [left_centre, right_centre]:
        x_outer = centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta)
        y_outer = centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta)
        x_inner = centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta[::-1])
        y_inner = centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta[::-1])
        ax2.fill(np.concatenate([x_outer, x_inner]),
                 np.concatenate([y_outer, y_inner]),
                 color="#D0D0D0", alpha=0.5)

    # Shade the lane
    ax2.fill([lane_left_x, lane_right_x, lane_right_x, lane_left_x],
             [lane_bottom_y, lane_bottom_y, lane_top_y, lane_top_y],
             color="#D0D0D0", alpha=0.5)

    # Circle boundaries
    for centre in [left_centre, right_centre]:
        ax2.plot(centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta),
                 centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta),
                 color="black", linewidth=2)
        ax2.plot(centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta),
                 centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta),
                 color="black", linewidth=2)

    # Lane boundaries
    ax2.plot([lane_left_x, lane_left_x], [lane_bottom_y, lane_top_y],
             color="black", linewidth=2)
    ax2.plot([lane_right_x, lane_right_x], [lane_bottom_y, lane_top_y],
             color="black", linewidth=2)

    # Driving lines (centre lines)
    for centre in [left_centre, right_centre]:
        ax2.plot(centre[0] + SKIDPAD_CENTRE_RADIUS * np.cos(theta),
                 centre[1] + SKIDPAD_CENTRE_RADIUS * np.sin(theta),
                 color="#2070B0", linewidth=2)
    ax2.plot([lane_centre_x, lane_centre_x], [lane_bottom_y, lane_top_y],
             color="#2070B0", linewidth=2, label="Driving line")

    # Circle centres
    ax2.plot(*left_centre, "+", color="#D04040", markersize=12, markeredgewidth=2,
             label="Circle centres")
    ax2.plot(*right_centre, "+", color="#D04040", markersize=12, markeredgewidth=2)

    # Entry and exit
    ax2.plot(lane_centre_x, lane_bottom_y, "o", color="#30A050", markersize=10,
             zorder=5, label="Entry")
    ax2.plot(lane_centre_x, lane_top_y, "o", color="#D04040", markersize=10,
             zorder=5, label="Exit")

    ax2.set_aspect("equal")
    ax2.set_xlabel("x [m]", fontsize=12)
    ax2.set_ylabel("y [m]", fontsize=12)
    ax2.set_title("Skidpad Track Layout", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "event_layouts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_event_layouts()
