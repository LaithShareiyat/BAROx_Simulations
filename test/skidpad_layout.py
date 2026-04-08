"""
Simplified skidpad track layout (single circle).

Usage:
    python test/skidpad_layout.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.skidpad import (
    SKIDPAD_CENTRE_RADIUS, SKIDPAD_INNER_RADIUS,
    SKIDPAD_OUTER_RADIUS, TRACK_WIDTH,
)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_simplified_skidpad():
    theta = np.linspace(0, 2 * np.pi, 200)

    x_centre = SKIDPAD_CENTRE_RADIUS * np.cos(theta)
    y_centre = SKIDPAD_CENTRE_RADIUS * np.sin(theta)
    x_inner = SKIDPAD_INNER_RADIUS * np.cos(theta)
    y_inner = SKIDPAD_INNER_RADIUS * np.sin(theta)
    x_outer = SKIDPAD_OUTER_RADIUS * np.cos(theta)
    y_outer = SKIDPAD_OUTER_RADIUS * np.sin(theta)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Shade the track surface
    ax.fill(
        np.concatenate([x_outer, x_inner[::-1]]),
        np.concatenate([y_outer, y_inner[::-1]]),
        color="#D0D0D0", alpha=0.5, label="Track surface",
    )

    # Track boundaries
    ax.plot(x_inner, y_inner, color="black", linewidth=2, label="Track boundaries")
    ax.plot(x_outer, y_outer, color="black", linewidth=2)

    # Driving line (centre)
    ax.plot(x_centre, y_centre, color="#2070B0", linewidth=2,
            label=f"Driving line (R = {SKIDPAD_CENTRE_RADIUS} m)")

    # Centre point
    ax.plot(0, 0, "+", color="#D04040", markersize=12, markeredgewidth=2,
            label="Circle centre")

    # Radius annotation
    ax.annotate(
        "", xy=(SKIDPAD_CENTRE_RADIUS, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="<->", color="#D04040", lw=1.5),
    )
    ax.text(SKIDPAD_CENTRE_RADIUS / 2, 0.6,
            f"R = {SKIDPAD_CENTRE_RADIUS} m", fontsize=11,
            color="#D04040", ha="center")

    # Start/finish marker
    ax.plot(SKIDPAD_CENTRE_RADIUS, 0, "o", color="#30A050",
            markersize=10, zorder=5, label="Start / Finish")

    # Track info
    circumference = 2 * np.pi * SKIDPAD_CENTRE_RADIUS
    info_text = (
        f"Circumference: {circumference:.1f} m\n"
        f"Track width: {TRACK_WIDTH:.1f} m"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title("Simplified Skidpad Track Layout", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "skidpad_layout.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_simplified_skidpad()
