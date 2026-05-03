"""
Combined velocity profile: solver pass colouring with braking zones shaded.

Single plot showing the speed trace coloured by which solver pass sets the
speed (lateral, forward, backward) with braking zones shaded in the background.

Usage:
    python test/velocity_profile_combined.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import (
    VehicleParams, EVPowertrainParams, VehicleGeometry,
    build_tyre_from_config, build_aero_from_config,
)
from events.autocross_generator import build_standard_autocross
from solver.qss_speed import solve_qss

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_vehicle():
    config_path = os.path.join(os.path.dirname(SAVE_DIR), "config", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tyre = build_tyre_from_config(config["tyre"])
    aero = build_aero_from_config(config["aero"])
    pt = config["powertrain"]
    powertrain = EVPowertrainParams(
        drivetrain=pt["drivetrain"], motor_power_kW=pt["motor_power_kW"],
        motor_torque_Nm=pt["motor_torque_Nm"], motor_rpm_max=pt["motor_rpm_max"],
        gear_ratio=pt["gear_ratio"], wheel_radius_m=pt["wheel_radius_m"],
    )
    geometry = VehicleGeometry(**config["geometry"])

    return VehicleParams(
        m=config["vehicle"]["mass_kg"], g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"], aero=aero, tyre=tyre,
        powertrain=powertrain, geometry=geometry,
    )


def plot_combined():
    vehicle = build_vehicle()
    track, _ = build_standard_autocross()

    result, lap_time = solve_qss(
        track, vehicle, refine=True, min_points=500, use_bicycle_model=True,
    )

    v = result["v"]
    v_lat = result["v_lat"]
    v_fwd = result["v_fwd"]
    v_bwd = result["v_bwd"]
    s = track.s

    # Classify each point by which solver pass sets the speed
    solver_pass = np.zeros(len(v), dtype=int)
    for i in range(len(v)):
        candidates = [v_lat[i], v_fwd[i], v_bwd[i]]
        solver_pass[i] = int(np.argmin(candidates))

    is_braking = solver_pass == 2

    colours = {
        0: "#2070B0",   # lateral limit: blue
        1: "#30A050",   # forward pass: green
        2: "#D04040",   # backward pass: red
    }
    labels = {
        0: "Lateral limit",
        1: "Forward pass",
        2: "Backward pass",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    # Shade braking zones in background first
    i = 0
    first_braking = True
    while i < len(s) - 1:
        if is_braking[i]:
            start = s[i]
            while i < len(s) - 1 and is_braking[i]:
                i += 1
            end = s[i]
            if first_braking:
                ax.axvspan(start, end, color="#D04040", alpha=0.15,
                           label="Braking zone")
                first_braking = False
            else:
                ax.axvspan(start, end, color="#D04040", alpha=0.15)
        else:
            i += 1

    # Speed profile coloured by solver pass
    for i in range(len(s) - 1):
        ax.plot(
            s[i:i + 2], v[i:i + 2],
            color=colours[solver_pass[i]], linewidth=1.5, solid_capstyle="round",
        )

    # Legend combining solver passes and braking zone
    patches = [
        mpatches.Patch(color=colours[k], label=labels[k])
        for k in sorted(colours.keys())
    ]
    patches.append(mpatches.Patch(color="#D04040", alpha=0.15, label="Braking zone"))
    ax.legend(handles=patches, fontsize=10, loc="upper right")

    ax.set_xlabel("Distance [m]", fontsize=12)
    ax.set_ylabel("Speed [m/s]", fontsize=12)
    ax.set_title("Velocity Profile vs Solver Pass (and braking regions)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(s[0], s[-1])

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "velocity_profile_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    n_total = len(solver_pass)
    print()
    for k in sorted(colours.keys()):
        pct = np.sum(solver_pass == k) / n_total * 100
        print(f"  {labels[k]:30s}: {pct:5.1f}% of lap")
    print(f"\n  Lap time: {lap_time:.2f} s")
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("Combined Velocity Profile Figure")
    print("=" * 50)
    plot_combined()
