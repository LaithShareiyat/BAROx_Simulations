"""
Power draw vs track distance for the autocross event.

Shows instantaneous electrical power demand and average power draw.

Usage:
    python test/power_draw_vs_distance.py
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
from solver.qss_speed import solve_qss, refine_track
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


def plot_power_draw():
    vehicle = build_vehicle()
    track, _ = build_standard_autocross()

    # Refine track before solving so track and v share the same grid
    track = refine_track(track, min_points=500)

    result, lap_time = solve_qss(
        track, vehicle, refine=False, use_bicycle_model=True,
    )
    v = result["v"]

    battery_state = simulate_battery(track, v, vehicle)
    power_kW = battery_state.power_kW
    s = track.s

    # Average power (discharge only, i.e. positive power)
    discharging = power_kW[power_kW > 0]
    avg_power = np.mean(discharging) if len(discharging) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(s, power_kW, alpha=0.25, color="#2070B0")
    ax.plot(s, power_kW, color="#2070B0", linewidth=1.5, label="Instantaneous Power")
    ax.axhline(avg_power, color="#D04040", linewidth=2, linestyle="--",
               label=f"Average Power = {avg_power:.1f} kW")
    ax.set_xlabel("Distance [m]", fontsize=12)
    ax.set_ylabel("Power Draw [kW]", fontsize=12)
    ax.set_title("Electrical Power Draw vs Distance", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(s[0], s[-1])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "power_draw_vs_distance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Peak power:    {np.max(power_kW):.1f} kW")
    print(f"Average power: {avg_power:.1f} kW")
    print(f"Lap time:      {lap_time:.2f} s")
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_power_draw()
