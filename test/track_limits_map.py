"""
Autocross track map coloured by performance limit.

Usage:
    python test/track_limits_map.py
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
from physics.aero import downforce
from physics.tyre import a_max_from_tyre, ax_available, ax_traction_axle_aware
from physics.powertrain import max_tractive_force_extended

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


def classify_limits(track, vehicle, v, v_lat, v_fwd, v_bwd):
    n = len(v)
    limits = np.zeros(n, dtype=int)
    m = vehicle.m
    g = vehicle.g
    mu = vehicle.tyre.mu
    use_bicycle = vehicle.has_extended_powertrain

    for i in range(n):
        envelope_speeds = [v_lat[i], v_fwd[i], v_bwd[i]]
        active = int(np.argmin(envelope_speeds))

        if active == 0:
            limits[i] = 0  # Traction
        elif active == 2:
            limits[i] = 2  # Braking
        else:
            v_i = v[i]
            Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v_i)

            if use_bicycle:
                ax_grip = ax_traction_axle_aware(
                    mu, vehicle, v_i, abs(track.kappa[i]), Fdown, 0.0,
                )
                Fx_grip = ax_grip * m
            else:
                amax = a_max_from_tyre(vehicle.tyre, g, m, Fdown)
                Fx_grip = ax_available(amax, v_i ** 2 * abs(track.kappa[i])) * m

            Fx_motor = max_tractive_force_extended(vehicle.powertrain, v_i)

            if Fx_motor < Fx_grip:
                limits[i] = 1  # Power
            else:
                limits[i] = 0  # Traction

    return limits


def plot_track_limits_map():
    vehicle = build_vehicle()
    track, _ = build_standard_autocross()

    result, lap_time = solve_qss(
        track, vehicle, refine=True, min_points=500, use_bicycle_model=True,
    )

    v = result["v"]
    v_lat = result["v_lat"]
    v_fwd = result["v_fwd"]
    v_bwd = result["v_bwd"]
    x = track.x
    y = track.y

    limits = classify_limits(track, vehicle, v, v_lat, v_fwd, v_bwd)

    colours_map = {
        0: "#30A050",   # traction: green
        1: "#2070B0",   # power: blue
        2: "#D04040",   # braking: red
    }
    labels = {
        0: "Traction limited",
        1: "Power limited",
        2: "Braking limited",
    }

    # Percentages
    n_total = len(limits)
    pcts = {k: np.sum(limits == k) / n_total * 100 for k in colours_map}

    # Map limits to colours array
    colour_array = [colours_map[l] for l in limits]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot track segments coloured by limit
    for i in range(len(x) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], color=colour_array[i],
                linewidth=3, solid_capstyle="round")

    # Start and finish markers
    ax.plot(x[0], y[0], "o", color="black", markersize=10, zorder=5, label="Start")
    ax.plot(x[-1], y[-1], "s", color="black", markersize=10, zorder=5, label="Finish")

    # Legend with percentages
    patches = [
        mpatches.Patch(color=colours_map[k], label=f"{labels[k]} ({pcts[k]:.1f}%)")
        for k in sorted(colours_map.keys())
    ]
    ax.legend(handles=patches, fontsize=11, loc="upper right")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    ax.set_title("Autocross Track Map by Performance Limit", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "track_limits_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print()
    for k in sorted(colours_map.keys()):
        print(f"  {labels[k]:20s}: {pcts[k]:5.1f}%")
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    print("Track Map by Performance Limit")
    print("=" * 50)
    plot_track_limits_map()
