"""
Tyre thermal model justification: shows that temperature rise across
a single autocross lap is small and the grip effect is negligible.

Produces two plots:
  1. Tyre temperature and grip multiplier vs distance
  2. Velocity profile comparison: isothermal vs thermal

Usage:
    python test/thermal_justification.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import (
    VehicleParams, EVPowertrainParams, VehicleGeometry, TyreThermalParams,
    build_tyre_from_config, build_aero_from_config,
)
from events.autocross_generator import build_standard_autocross
from solver.qss_speed import solve_qss, refine_track
from solver.tyre_thermal import integrate_tyre_temperature, solve_qss_thermal

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

    th = config["tyre_thermal"]
    tyre_thermal = TyreThermalParams(
        enabled=True,
        T_ambient=th["T_ambient"],
        T_initial=th["T_initial"],
        T_opt=th["T_opt"],
        T_width=th["T_width"],
        C_thermal=th["C_thermal"],
        k_heating=th["k_heating"],
        h_static=th["h_static"],
        h_speed=th["h_speed"],
        max_thermal_iter=th["max_thermal_iter"],
        thermal_tol=th["thermal_tol"],
        relaxation=th["relaxation"],
    )

    return VehicleParams(
        m=config["vehicle"]["mass_kg"], g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"], aero=aero, tyre=tyre,
        powertrain=powertrain, geometry=geometry,
        tyre_thermal=tyre_thermal,
    )


def run_comparison():
    vehicle = build_vehicle()
    track_raw, _ = build_standard_autocross()
    track = refine_track(track_raw, min_points=500)

    # Isothermal (no thermal model)
    result_iso, lt_iso = solve_qss(
        track, vehicle, refine=False, use_bicycle_model=True,
    )

    # With thermal coupling
    result_therm, lt_therm = solve_qss_thermal(
        track, vehicle, refine=False, min_points=500, use_bicycle_model=True,
    )

    # Also compute temperature profile from the isothermal solution
    # to show what the temperature does without feedback
    thermal_state_iso = integrate_tyre_temperature(track, result_iso["v"], vehicle)

    return track, result_iso, lt_iso, result_therm, lt_therm, thermal_state_iso, vehicle


def plot_thermal_justification(track, result_iso, lt_iso, result_therm, lt_therm,
                                thermal_state_iso, vehicle):
    s = track.s
    thermal = vehicle.tyre_thermal

    T_iso = thermal_state_iso.temperature
    grip_iso = thermal_state_iso.grip_multiplier

    # --- Plot 1: Temperature and grip vs distance ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    colour_temp = "#D04040"
    colour_grip = "#2070B0"

    ax1.plot(s, T_iso, color=colour_temp, linewidth=2, label="Tyre temperature")
    ax1.axhline(thermal.T_opt, color=colour_temp, linewidth=1, linestyle="solid",
                alpha=0.4, label=f"Optimal temperature ({thermal.T_opt:.0f} °C)")
    ax1.set_xlabel("Distance [m]", fontsize=12)
    ax1.set_ylabel("Temperature [°C]", fontsize=12, color=colour_temp)
    ax1.tick_params(axis="y", labelcolor=colour_temp)
    ax1.set_xlim(s[0], s[-1])

    ax2 = ax1.twinx()
    ax2.plot(s, grip_iso * 100, color=colour_grip, linewidth=2, label="Grip multiplier")
    ax2.set_ylabel("Grip Multiplier [%]", fontsize=12, color=colour_grip)
    ax2.tick_params(axis="y", labelcolor=colour_grip)
    ax2.set_ylim(90, 102)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="lower right")

    ax1.set_title("Tyre Temperature and Grip Across Autocross Lap",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Annotate key values
    T_start = T_iso[0]
    T_end = T_iso[-1]
    T_max = np.max(T_iso)
    grip_min = np.min(grip_iso) * 100
    grip_mean = np.mean(grip_iso) * 100

    info_text = (f"T start: {T_start:.1f} °C\n"
                 f"T end: {T_end:.1f} °C\n"
                 f"T peak: {T_max:.1f} °C\n"
                 f"Grip range: {grip_min:.1f}% \u2013 100%\n"
                 f"Mean grip: {grip_mean:.1f}%")
    ax1.text(0.02, 0.97, info_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig1.tight_layout()
    path1 = os.path.join(SAVE_DIR, "thermal_justification_temperature.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {path1}")

    # --- Plot 2: Velocity comparison ---
    fig2, ax3 = plt.subplots(figsize=(10, 6))

    v_iso_kmh = result_iso["v"] * 3.6
    v_therm_kmh = result_therm["v"] * 3.6

    ax3.plot(s, v_iso_kmh, color="#2070B0", linewidth=2, label=f"Isothermal: {lt_iso:.2f} s")
    ax3.plot(s, v_therm_kmh, color="#D04040", linewidth=2, label=f"Thermal model: {lt_therm:.2f} s")

    ax3.set_xlabel("Distance [m]", fontsize=12)
    ax3.set_ylabel("Speed [km/h]", fontsize=12)
    ax3.set_title("Velocity Profile: Isothermal vs Thermal Tyre Model",
                  fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11, loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(s[0], s[-1])
    ax3.set_ylim(bottom=0)

    # Lap time difference annotation
    delta = lt_therm - lt_iso
    pct = delta / lt_iso * 100
    ax3.text(0.02, 0.02,
             f"Lap time difference: {delta:+.3f} s ({pct:+.2f}%)",
             transform=ax3.transAxes, fontsize=11, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig2.tight_layout()
    path2 = os.path.join(SAVE_DIR, "thermal_justification_velocity.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")

    # Summary
    print(f"\nIsothermal lap time:  {lt_iso:.3f} s")
    print(f"Thermal lap time:     {lt_therm:.3f} s")
    print(f"Difference:           {delta:+.3f} s ({pct:+.2f}%)")
    print(f"\nTemperature: {T_start:.1f} -> {T_end:.1f} °C (peak {T_max:.1f} °C)")
    print(f"Grip: {grip_min:.1f}% - 100% (mean {grip_mean:.1f}%)")


if __name__ == "__main__":
    print("Tyre Thermal Model Justification")
    print("=" * 50)
    track, result_iso, lt_iso, result_therm, lt_therm, thermal_state_iso, vehicle = run_comparison()
    plot_thermal_justification(track, result_iso, lt_iso, result_therm, lt_therm,
                                thermal_state_iso, vehicle)
