"""
Maximum cornering velocity as a function of corner radius.

Computes v_max = sqrt(a_y_max * R) iteratively to account for
speed-dependent aerodynamic downforce, and marks the autocross
and skidpad operating ranges.

Usage:
    python test/cornering_velocity_vs_radius.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import (
    VehicleParams, EVPowertrainParams, VehicleGeometry,
    build_tyre_from_config, build_aero_from_config,
)
from physics.aero import downforce
from physics.tyre import a_max_from_tyre

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# FSAE event radius ranges
SKIDPAD_RADIUS = 9.125          # Skidpad centre-line radius [m]
AUTOCROSS_MIN_RADIUS = 4.5      # Tightest hairpin centre-line (9m outer - half track)
AUTOCROSS_MAX_RADIUS = 50.0     # Fastest sweepers


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


def max_cornering_speed(R, vehicle, tol=0.01, max_iter=50):
    """Iteratively solve for maximum cornering speed at radius R.

    v = sqrt(a_y_max * R), but a_y_max depends on v through downforce.
    """
    m = vehicle.m
    g = vehicle.g
    v = np.sqrt(vehicle.tyre.mu * g * R)  # Initial guess (no aero)

    for _ in range(max_iter):
        Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v)
        a_max_val = a_max_from_tyre(vehicle.tyre, g, m, Fdown)
        v_new = np.sqrt(a_max_val * R)
        if abs(v_new - v) < tol:
            return v_new
        v = v_new

    return v


def plot_cornering_velocity():
    vehicle = build_vehicle()

    radii = np.linspace(2.0, 60.0, 300)
    v_max = np.array([max_cornering_speed(R, vehicle) for R in radii])
    v_max_kmh = v_max * 3.6

    fig, ax = plt.subplots(figsize=(10, 6))

    # Theoretical sqrt(R) line (no aero: v = sqrt(μg·R))
    mu = vehicle.tyre.mu
    g = vehicle.g
    v_no_aero_kmh = np.sqrt(mu * g * radii) * 3.6
    ax.plot(radii, v_no_aero_kmh, color="#888888", linewidth=1.5, linestyle="dashed",
            label=r"$v = \sqrt{\mu g R}$ (no aero)")

    # Main curve
    ax.plot(radii, v_max_kmh, color="#2070B0", linewidth=2.5, label="Max cornering speed")

    # Skidpad radius
    v_skidpad = max_cornering_speed(SKIDPAD_RADIUS, vehicle)
    ax.axvline(SKIDPAD_RADIUS, color="#D04040", linewidth=1.5, linestyle="solid",
               label=f"Skidpad (R = {SKIDPAD_RADIUS} m)")
    ax.plot(SKIDPAD_RADIUS, v_skidpad * 3.6, "o", color="#D04040",
            markersize=8, zorder=5,
            label=f"Maximum skidpad cornering speed: {v_skidpad * 3.6:.1f} km/h")

    ax.set_xlabel("Corner Radius [m]", fontsize=12)
    ax.set_ylabel("Maximum Cornering Speed [km/h]", fontsize=12)
    ax.set_title("Maximum Cornering Velocity vs Corner Radius", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 62)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "cornering_velocity_vs_radius.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    print(f"\nSkidpad:  R = {SKIDPAD_RADIUS:.1f} m  ->  v = {v_skidpad*3.6:.1f} km/h")


if __name__ == "__main__":
    plot_cornering_velocity()
