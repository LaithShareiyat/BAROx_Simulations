"""
Velocity profile with braking zones highlighted.

Shows the speed trace around the autocross lap with braking zones
(where the car is decelerating into corners) shaded in red.

Usage:
    python test/braking_zones.py
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


def plot_braking_zones():
    vehicle = build_vehicle()
    track, _ = build_standard_autocross()

    result, lap_time = solve_qss(
        track, vehicle, refine=True, min_points=500, use_bicycle_model=True,
    )

    v = result["v"]
    v_bwd = result["v_bwd"]
    v_lat = result["v_lat"]
    v_fwd = result["v_fwd"]
    s = track.s
    v_kmh = v * 3.6

    # Identify braking zones: where the backward pass sets the speed
    is_braking = np.zeros(len(v), dtype=bool)
    for i in range(len(v)):
        candidates = [v_lat[i], v_fwd[i], v_bwd[i]]
        is_braking[i] = int(np.argmin(candidates)) == 2

    fig, ax = plt.subplots(figsize=(12, 5))

    # Shade braking zones
    i = 0
    first_braking = True
    while i < len(s) - 1:
        if is_braking[i]:
            start = s[i]
            while i < len(s) - 1 and is_braking[i]:
                i += 1
            end = s[i]
            if first_braking:
                ax.axvspan(start, end, color="#D04040", alpha=0.2, label="Braking zone")
                first_braking = False
            else:
                ax.axvspan(start, end, color="#D04040", alpha=0.2)
        else:
            i += 1

    # Speed trace
    ax.plot(s, v_kmh, color="#2070B0", linewidth=2, label="Vehicle speed")

    ax.set_xlabel("Distance [m]", fontsize=12)
    ax.set_ylabel("Speed [km/h]", fontsize=12)
    ax.set_title("Velocity Profile with Braking Zones",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "braking_zones.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_braking_zones()
