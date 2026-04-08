"""
Velocity profile with traction, power, and braking limited regions.

Usage:
    python test/velocity_limits.py
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
    """Classify each point as traction, power, or braking limited.

    Returns an array of integers:
        0 = traction limited  (cornering grip or longitudinal grip is bottleneck)
        1 = power limited     (motor power is the bottleneck)
        2 = braking limited   (decelerating into a corner)
    """
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
            # Lateral limit = traction limited (grip)
            limits[i] = 0
        elif active == 2:
            # Backward pass = braking limited
            limits[i] = 2
        else:
            # Forward pass — check if grip or power is the bottleneck
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
                limits[i] = 1  # Power limited
            else:
                limits[i] = 0  # Traction limited

    return limits


def plot_velocity_limits():
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

    limits = classify_limits(track, vehicle, v, v_lat, v_fwd, v_bwd)

    colours = {
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
    pcts = {}
    for k in sorted(colours.keys()):
        pcts[k] = np.sum(limits == k) / n_total * 100

    fig, ax = plt.subplots(figsize=(14, 6))

    # Speed profile coloured by limit
    for i in range(len(s) - 1):
        ax.plot(
            s[i:i + 2], v_kmh[i:i + 2],
            color=colours[limits[i]], linewidth=1.5, solid_capstyle="round",
        )

    # Legend with percentages
    patches = [
        mpatches.Patch(color=colours[k], label=f"{labels[k]} ({pcts[k]:.1f}%)")
        for k in sorted(colours.keys())
    ]
    ax.legend(handles=patches, fontsize=11, loc="upper right")

    ax.set_xlabel("Distance [m]", fontsize=12)
    ax.set_ylabel("Speed [km/h]", fontsize=12)
    ax.set_title("Velocity Profile by Performance Limit", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "velocity_limits.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print()
    for k in sorted(colours.keys()):
        print(f"  {labels[k]:20s}: {pcts[k]:5.1f}%")
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    print("Velocity Profile by Performance Limit")
    print("=" * 50)
    plot_velocity_limits()
