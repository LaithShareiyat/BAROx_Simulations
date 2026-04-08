"""
State of Charge vs track distance for the autocross event.

Usage:
    python test/soc_vs_distance.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import (
    VehicleParams, EVPowertrainParams, VehicleGeometry, BatteryParams,
    build_tyre_from_config, build_aero_from_config,
)
from events.autocross_generator import build_standard_autocross
from solver.qss_speed import solve_qss
from solver.battery import simulate_battery

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
    bat = config["battery"]
    battery = BatteryParams(
        capacity_kWh=bat["capacity_kWh"],
        initial_soc=bat["initial_soc"],
        min_soc=bat["min_soc"],
        max_discharge_kW=bat["max_discharge_kW"],
        eta_discharge=bat["eta_discharge"],
        nominal_voltage_V=bat["nominal_voltage_V"],
        max_current_A=bat["max_current_A"],
        regen_enabled=bat.get("regen_enabled", False),
        eta_regen=bat.get("eta_regen", 0.85),
        max_regen_kW=bat.get("max_regen_kW", 50),
        regen_capture_percent=bat.get("regen_capture_percent", 100),
    )

    return VehicleParams(
        m=config["vehicle"]["mass_kg"], g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"], aero=aero, tyre=tyre,
        powertrain=powertrain, geometry=geometry, battery=battery,
    )


def plot_soc_vs_distance():
    vehicle = build_vehicle()
    track, _ = build_standard_autocross()

    result, lap_time = solve_qss(
        track, vehicle, refine=True, min_points=500, use_bicycle_model=True,
    )
    v = result["v"]

    battery_state = simulate_battery(track, v, vehicle)
    soc_pct = battery_state.soc * 100.0
    s = track.s

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(s, soc_pct, color="#2070B0", linewidth=2, label="State of Charge")
    ax.axhline(vehicle.battery.min_soc * 100, color="#D04040", linewidth=1.5,
               linestyle="solid", label=f"Minimum SoC ({vehicle.battery.min_soc*100:.0f}%)")

    ax.set_xlabel("Distance [m]", fontsize=12)
    ax.set_ylabel("State of Charge [%]", fontsize=12)
    ax.set_title("Battery State of Charge vs Distance (142S5P)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(bottom=0, top=105)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "soc_vs_distance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Initial SoC: {soc_pct[0]:.1f}%")
    print(f"Final SoC:   {soc_pct[-1]:.1f}%")
    print(f"Energy used: {battery_state.energy_used_kWh[-1]:.3f} kWh")
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_soc_vs_distance()
