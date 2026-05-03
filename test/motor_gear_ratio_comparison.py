"""
Autocross lap time vs gear ratio for different motors.

Usage:
    python test/motor_gear_ratio_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Motor data: {name: (gear_ratios, autocross_laptimes)}
MOTORS = {
    "EMRAX 208": (
        [0.5, 1, 2, 3, 4, 4.5, 5],
        [105.3, 96.2, 89.114, 85.688, 84.153, 84.081, 84.38],
    ),
    "YASA P400R": (
        [0.5, 1, 2, 3, 4, 4.25, 4.5, 5],
        [103.934, 95.957, 89.291, 86.459, 86.058, 86.058, 86.058, 86.13],
    ),
    "YASA 750R": (
        [0.5, 1, 1.5, 1.75, 2, 3],
        [94.975, 88.587, 86.291, 86.141, 86.22, 89.237],
    ),
    "Plettenberg 30-50-B10": (
        [0.5, 1, 2, 3, 3.5, 4],
        [104.094, 95.812, 88.794, 85.528, 85.03, 85.597],
    ),
    "Retorq200-HS": (
        [0.5, 1, 2, 3, 4, 5, 5.5, 6, 7],
        [99.536, 91.981, 86.436, 85.642, 85.487, 85.432, 85.423, 85.415, 85.456],
    ),
}

COLOURS = ["#2070B0", "#D04040", "#30A050", "#9040B0", "#E08020"]
MARKERS = ["o", "s", "D", "^", "v"]


def plot_autocross_vs_gear_ratio():
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, (gr, lt)) in enumerate(MOTORS.items()):
        gr, lt = np.array(gr), np.array(lt)
        ax.plot(gr, lt, marker=MARKERS[i], color=COLOURS[i], linewidth=1.8,
                markersize=7, label=name)

    ax.set_xlabel("Gear Ratio", fontsize=12)
    ax.set_ylabel("Autocross Lap Time [s]", fontsize=12)
    ax.set_title("Autocross Lap Time vs Gear Ratio — Motor Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "motor_gear_ratio_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_autocross_vs_gear_ratio()
