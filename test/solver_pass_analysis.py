"""
Velocity profile coloured by which solver pass determines the speed.

Shows whether each point on the lap is set by the lateral limit,
forward pass (acceleration), or backward pass (braking).

Usage:
    python test/solver_pass_analysis.py
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


def plot_solver_pass():
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
    v_kmh = v * 3.6

    # Classify by which solver pass sets the speed
    solver_pass = np.zeros(len(v), dtype=int)
    for i in range(len(v)):
        candidates = [v_lat[i], v_fwd[i], v_bwd[i]]
        solver_pass[i] = int(np.argmin(candidates))

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

    # Speed profile coloured by solver pass
    for i in range(len(s) - 1):
        ax.plot(
            s[i:i + 2], v_kmh[i:i + 2],
            color=colours[solver_pass[i]], linewidth=1.5, solid_capstyle="round",
        )

    # Legend
    patches = [
        mpatches.Patch(color=colours[k], label=labels[k])
        for k in sorted(colours.keys())
    ]
    ax.legend(handles=patches, fontsize=10, loc="upper right")
    ax.set_xlabel("Distance [m]", fontsize=12)
    ax.set_ylabel("Speed [km/h]", fontsize=12)
    ax.set_title("Velocity Profile by Solver Pass",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(s[0], s[-1])

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "solver_pass_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    n_total = len(solver_pass)
    print()
    for k in sorted(colours.keys()):
        pct = np.sum(solver_pass == k) / n_total * 100
        print(f"  {labels[k]:30s}: {pct:5.1f}% of lap")
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    print("Solver Pass Analysis")
    print("=" * 50)
    plot_solver_pass()
