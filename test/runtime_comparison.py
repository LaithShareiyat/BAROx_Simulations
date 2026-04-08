"""
LTS runtime comparison under different configurations.

Measures execution time for increasingly complex simulation setups,
all using the default vehicle configuration.

Usage:
    python test/runtime_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import (
    VehicleParams, EVPowertrainParams, VehicleGeometry, BatteryParams,
    TorqueVectoringParams, build_tyre_from_config, build_aero_from_config,
)
from events.autocross_generator import build_standard_autocross
from events.skidpad import build_skidpad_track
from solver.qss_speed import solve_qss
from solver.battery import simulate_battery, validate_battery_capacity

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config():
    config_path = os.path.join(os.path.dirname(SAVE_DIR), "config", "default.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_base_vehicle(config):
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


def build_battery(config):
    bat = config["battery"]
    return BatteryParams(
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


def run_events(vehicle, track_ac, track_sk, battery=False):
    """Run autocross + skidpad, optionally with battery analysis."""
    from solver.qss_speed import refine_track
    track_ac_r = refine_track(track_ac, min_points=500)
    track_sk_r = refine_track(track_sk, min_points=500)

    result_ac, _ = solve_qss(track_ac_r, vehicle, refine=False, use_bicycle_model=True)
    result_sk, _ = solve_qss(track_sk_r, vehicle, refine=False, use_bicycle_model=True)

    if battery and vehicle.battery is not None:
        simulate_battery(track_ac_r, result_ac["v"], vehicle)
        validate_battery_capacity(track_ac_r, result_ac["v"], vehicle)
        simulate_battery(track_sk_r, result_sk["v"], vehicle)
        validate_battery_capacity(track_sk_r, result_sk["v"], vehicle)

    return track_ac_r


def run_config_1(vehicle, track_ac, track_sk):
    """Autocross + Skidpad only."""
    run_events(vehicle, track_ac, track_sk, battery=False)


def run_config_2(vehicle, track_ac, track_sk):
    """Autocross + Skidpad + Battery Analysis."""
    run_events(vehicle, track_ac, track_sk, battery=True)


def run_config_3(vehicle, track_ac, track_sk):
    """Autocross + Skidpad + Battery Analysis + Torque Vectoring."""
    run_events(vehicle, track_ac, track_sk, battery=True)


def run_config_4(vehicle, track_ac, track_sk):
    """Autocross + Skidpad + Battery + TV + Gear Ratio Sweep."""
    track_ac_r = run_events(vehicle, track_ac, track_sk, battery=True)

    # Gear ratio sweep (21 points)
    current_gr = vehicle.powertrain.gear_ratio
    gr_lo = max(1.0, current_gr * 0.5)
    gr_hi = min(10.0, current_gr * 1.5)
    gr_values = np.linspace(gr_lo, gr_hi, 21)
    for gr_val in gr_values:
        pt_mod = replace(vehicle.powertrain, gear_ratio=gr_val)
        v_mod = replace(vehicle, powertrain=pt_mod)
        solve_qss(track_ac_r, v_mod, refine=False, use_bicycle_model=True)


def time_function(func, *args, n_runs=3):
    """Run function multiple times and return average time."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times)


def plot_runtime_comparison():
    config = load_config()

    # Build tracks (outside timing)
    track_ac, _ = build_standard_autocross()
    track_sk = build_skidpad_track()

    # Config 1: Base vehicle (no battery, no TV)
    vehicle_base = build_base_vehicle(config)

    # Config 2: Add battery
    battery = build_battery(config)
    vehicle_bat = replace(vehicle_base, battery=battery)

    # Config 3: Add torque vectoring
    tv = TorqueVectoringParams(enabled=True, effectiveness=1.0,
                                max_torque_transfer=0.5,
                                strategy="load_proportional")
    vehicle_tv = replace(vehicle_bat, torque_vectoring=tv)

    # Config 4: Same as 3 but with gear sweep
    vehicle_sweep = vehicle_tv

    # Time each configuration
    n_runs = 3
    labels = [
        "Autocross\n+ Skidpad",
        "  + Battery\nAnalysis",
        "  + Torque\nVectoring",
        "  + Parameter\nSweeps",
    ]

    print("Timing each configuration (average of 3 runs)...")
    t1 = time_function(run_config_1, vehicle_base, track_ac, track_sk, n_runs=n_runs)
    print(f"  Config 1: {t1:.3f}s")

    t2 = time_function(run_config_2, vehicle_bat, track_ac, track_sk, n_runs=n_runs)
    print(f"  Config 2: {t2:.3f}s")

    t3 = time_function(run_config_3, vehicle_tv, track_ac, track_sk, n_runs=n_runs)
    print(f"  Config 3: {t3:.3f}s")

    t4 = time_function(run_config_4, vehicle_sweep, track_ac, track_sk, n_runs=n_runs)
    print(f"  Config 4: {t4:.3f}s")

    times = [t1, t2, t3, t4]

    # Cumulative times
    cum_times = np.cumsum(times)

    # Plot as stacked bars — each bar shows the cumulative total,
    # with the delta (new feature cost) in a distinct colour
    colours = ["#2070B0", "#30A050", "#E8A020", "#D04040"]
    delta_labels = [
        "Autocross + Skidpad",
        "Battery Analysis",
        "Torque Vectoring",
        "Parameter Sweeps",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(cum_times)):
        bottom = cum_times[i - 1] if i > 0 else 0
        delta = times[i]
        ax.bar(range(i, len(cum_times)), [delta] * (len(cum_times) - i),
               bottom=[bottom] * (len(cum_times) - i),
               color=colours[i], width=0.6, edgecolor="black", linewidth=0.8,
               label=f"{delta_labels[i]} (+{delta:.2f}s)")

    # Add cumulative time labels on top of each bar
    for i, ct in enumerate(cum_times):
        ax.text(i, ct + 0.3, f"{ct:.2f}s", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Cumulative Runtime [s]", fontsize=12)
    ax.set_title("LTS Runtime by Configuration", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0, top=max(cum_times) * 1.2)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "runtime_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    plot_runtime_comparison()
