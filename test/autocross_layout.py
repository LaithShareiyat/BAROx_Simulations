"""
Autocross track layout.

Usage:
    python test/autocross_layout.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.autocross_generator import build_standard_autocross

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_autocross_layout():
    track, _ = build_standard_autocross()

    x = track.x
    y = track.y

    fig, ax = plt.subplots(figsize=(12, 8))

    # Track centre line
    ax.plot(x, y, color="#2070B0", linewidth=2, label="Driving line")

    # Track boundaries
    dx = np.gradient(x)
    dy = np.gradient(y)
    length = np.sqrt(dx ** 2 + dy ** 2)
    length = np.maximum(length, 1e-9)
    nx = -dy / length
    ny = dx / length

    offset = 1.5  # half track width (3m / 2)
    ax.plot(x + offset * nx, y + offset * ny, color="black", linewidth=1, alpha=0.5)
    ax.plot(x - offset * nx, y - offset * ny, color="black", linewidth=1, alpha=0.5)

    # Start and finish markers
    ax.plot(x[0], y[0], "o", color="#30A050", markersize=10, zorder=5,
            label="Start")
    ax.plot(x[-1], y[-1], "o", color="#D04040", markersize=10, zorder=5,
            label="Finish")

    # Track info
    track_length = track.s[-1]
    info_text = (
        f"Track length: {track_length:.0f} m\n"
        f"Track width: 3.0 m"
    )
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title("Autocross Track Layout", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "autocross_layout.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_autocross_layout()
