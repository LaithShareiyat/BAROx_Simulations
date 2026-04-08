"""
Longitudinal acceleration vs vehicle speed.

Shows the traction limit (tyre grip) and power limit (motor) with the
actual acceleration as the minimum of the two.

Usage:
    python test/longitudinal_accel_vs_speed.py
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
from physics.aero import downforce, drag
from physics.tyre import a_max_from_tyre
from physics.powertrain import max_tractive_force_extended
from physics.resistive import rolling_resistance

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


def plot_accel_vs_speed():
    vehicle = build_vehicle()
    m = vehicle.m
    g = vehicle.g

    speeds = np.linspace(1.0, 40.0, 500)  # m/s
    speeds_kmh = speeds * 3.6

    ax_traction = np.zeros_like(speeds)
    ax_power = np.zeros_like(speeds)
    ax_net = np.zeros_like(speeds)

    for i, v in enumerate(speeds):
        Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v)
        Fdrag = drag(vehicle.aero.rho, vehicle.aero.CD_A, v)
        Frr = rolling_resistance(vehicle.Crr, m, g)

        # Traction limit (tyre grip on straight, no lateral load)
        a_grip = a_max_from_tyre(vehicle.tyre, g, m, Fdown)
        Fx_grip = a_grip * m

        # Power limit (motor torque/power/RPM)
        Fx_motor = max_tractive_force_extended(vehicle.powertrain, v)

        # Net acceleration after drag and rolling resistance
        ax_traction[i] = (Fx_grip - Fdrag - Frr) / m
        ax_power[i] = (Fx_motor - Fdrag - Frr) / m

        Fx = min(Fx_grip, Fx_motor)
        ax_net[i] = max((Fx - Fdrag - Frr) / m, 0.0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(speeds_kmh, ax_traction, color="#30A050", linewidth=2, label="Traction limit (tyre grip)")
    ax.plot(speeds_kmh, ax_power, color="#2070B0", linewidth=2, label="Power limit (motor)")
    ax.plot(speeds_kmh, ax_net, color="black", linewidth=2.5, linestyle="dashed", label="Actual acceleration")

    # Find crossover point
    diff = ax_traction - ax_power
    crossover_idx = np.where(np.diff(np.sign(diff)))[0]
    if len(crossover_idx) > 0:
        idx = crossover_idx[0]
        v_cross = speeds_kmh[idx]
        a_cross = ax_net[idx]
        ax.plot(v_cross, a_cross, "o", color="#D04040", markersize=8, zorder=5,
                label=f"Crossover speed: {v_cross:.0f} km/h")

    ax.set_xlabel("Vehicle Speed [km/h]", fontsize=12)
    ax.set_ylabel("Longitudinal Acceleration [m/s²]", fontsize=12)
    ax.set_title("Longitudinal Acceleration vs Vehicle Speed", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, speeds_kmh[-1])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "longitudinal_accel_vs_speed.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    if len(crossover_idx) > 0:
        print(f"Crossover at {v_cross:.1f} km/h ({speeds[crossover_idx[0]]:.1f} m/s)")


if __name__ == "__main__":
    plot_accel_vs_speed()
