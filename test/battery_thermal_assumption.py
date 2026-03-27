"""
Battery Thermal Model Assumption Validation.

The simulator has NO thermal model. It assumes constant cell performance
regardless of temperature. This script investigates whether that is valid
by estimating I²R heat generation and temperature rise over single and
multi-lap events.

Cell: 2600 mAh / 3.6 V / 12 mOhm ESR / 45 g (Samsung 25R class 18650)
Pack: 142S5P (710 cells)

Usage:
    python test/battery_thermal_assumption.py
    python test/battery_thermal_assumption.py --save test/
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import (
    VehicleParams, BatteryParams, EVPowertrainParams,
    build_tyre_from_config, build_aero_from_config, VehicleGeometry,
)
from solver.qss_speed import solve_qss
from solver.battery import calculate_power_profile
from events.autocross_generator import build_standard_autocross


# ── Cell and pack constants ─────────────────────────────────────────────────
CELL_ESR_MOHM = 12.0         # Internal resistance per cell [mOhm] at 25 C
N_SERIES = 142
N_PARALLEL = 5
V_NOMINAL_CELL = 3.6          # [V]
CELL_MASS_G = 45.0            # [g]
MAX_CELL_CURRENT_A = 35.0     # [A]

# Thermal properties (typical 18650 NMC/NCA)
CELL_SPECIFIC_HEAT = 1.0      # [J/(g K)]  literature: 0.8 to 1.1

# Pack-level
R_PACK_OHM = (N_SERIES * CELL_ESR_MOHM / 1000.0) / N_PARALLEL
V_NOMINAL_PACK = N_SERIES * V_NOMINAL_CELL
I_MAX_PACK = N_PARALLEL * MAX_CELL_CURRENT_A

# Temperature thresholds
T_AMBIENT = 25.0               # [C] typical ambient
T_OPTIMAL_MAX = 40.0           # [C] upper bound of optimal range
T_SAFE_MAX = 60.0              # [C] manufacturer safe limit
T_DERATE_START = 45.0          # [C] where BMS typically starts derating

# Convective cooling
H_FORCED = 35.0               # [W/(m2 K)] forced air cooling coefficient
A_COOLING = 0.5               # [m2] effective cooling surface area


def esr_vs_temperature(T_celsius: float) -> float:
    """Cell ESR as a function of temperature [mOhm].

    U-shaped curve for 18650 NMC based on Samsung 25R published data.
    """
    if T_celsius < 25:
        factor = 1.0 + 0.015 * ((25 - T_celsius) ** 2) / 100.0
        return CELL_ESR_MOHM * factor
    else:
        factor = 1.0 + 0.005 * (T_celsius - 25)
        return CELL_ESR_MOHM * factor


def current_from_power(power_kW: float, soc: float = 0.5,
                       T: float = 25.0) -> float:
    """Solve for pack discharge current given terminal power [kW]."""
    v_ocv = N_SERIES * (3.0 + 1.2 * soc - 0.35 * soc**2 + 0.35 * soc**3)
    R = (N_SERIES * esr_vs_temperature(T) / 1000.0) / N_PARALLEL
    P = power_kW * 1000.0
    discriminant = v_ocv**2 - 4 * R * P
    if discriminant < 0:
        return v_ocv / (2 * R)
    return (v_ocv - np.sqrt(discriminant)) / (2 * R)


def build_vehicle(config):
    """Build vehicle from config dictionary."""
    aero = build_aero_from_config(config["aero"])
    geo = VehicleGeometry(**config["geometry"])
    tyre = build_tyre_from_config(config["tyre"])
    pt_cfg = config["powertrain"]
    pt = EVPowertrainParams(
        drivetrain=pt_cfg["drivetrain"],
        motor_power_kW=pt_cfg["motor_power_kW"],
        motor_torque_Nm=pt_cfg["motor_torque_Nm"],
        motor_rpm_max=pt_cfg["motor_rpm_max"],
        motor_efficiency=pt_cfg.get("motor_efficiency", 0.96),
        gear_ratio=pt_cfg["gear_ratio"],
        wheel_radius_m=pt_cfg["wheel_radius_m"],
        motor_weight_kg=pt_cfg.get("motor_weight_kg", 9.4),
        inverter_weight_kg=pt_cfg.get("inverter_weight_kg", 6.9),
        powertrain_overhead_kg=pt_cfg.get("powertrain_overhead_kg", 10.0),
    )
    bat_cfg = config["battery"]
    battery = BatteryParams(
        capacity_kWh=bat_cfg["capacity_kWh"],
        initial_soc=bat_cfg.get("initial_soc", 1.0),
        min_soc=bat_cfg.get("min_soc", 0.1),
        max_discharge_kW=bat_cfg.get("max_discharge_kW", 80),
        eta_discharge=bat_cfg.get("eta_discharge", 0.95),
        nominal_voltage_V=bat_cfg.get("nominal_voltage_V", 511),
        max_current_A=bat_cfg.get("max_current_A", 175),
    )
    return VehicleParams(
        m=config["vehicle"]["mass_kg"],
        g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"],
        aero=aero, tyre=tyre, powertrain=pt,
        battery=battery, geometry=geo,
    )


def run_lap_simulation(config):
    """Run a QSS lap and return power profile + timing."""
    track, _ = build_standard_autocross()
    vehicle = build_vehicle(config)
    result, lap_time = solve_qss(track, vehicle)
    v = result["v"]
    power_kW = calculate_power_profile(track, v, vehicle)
    v_avg = np.maximum(0.5 * (v[:-1] + v[1:]), 0.1)
    dt = track.ds / v_avg
    return track, power_kW, dt, lap_time


# ── Figure 1: Cell ESR vs Temperature ──────────────────────────────────────
def fig1_esr_vs_temperature(save_dir=None):
    """Show how cell internal resistance varies with temperature."""
    temps = np.linspace(-20, 80, 500)
    esr_values = [esr_vs_temperature(T) for T in temps]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cell Internal Resistance vs Temperature",
                 fontsize=13, fontweight="bold")

    ax.plot(temps, esr_values, "#2196F3", linewidth=2.5)
    ax.axhline(CELL_ESR_MOHM, color="#FF5722", linewidth=2,
               label=f"Simulator assumption ({CELL_ESR_MOHM} mOhm, constant)")

    ax.set_xlabel("Cell temperature [C]")
    ax.set_ylabel("Cell internal resistance [mOhm]")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig1_esr_vs_temperature.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()

    print("\n  Internal resistance at key temperatures:")
    for T in [-10, 0, 10, 25, 40, 60]:
        r = esr_vs_temperature(T)
        print(f"    T = {T:+3d} C  |  R_int = {r:.1f} mOhm  |  "
              f"ratio vs 25C = {r/CELL_ESR_MOHM:.2f}x")


# ── Figure 2: Heat per cell over a single lap ─────────────────────────────
def fig2_heat_per_cell(config, save_dir=None):
    """Instantaneous I²R heat dissipation per cell over a lap."""
    track, power_kW, dt, lap_time = run_lap_simulation(config)
    print(f"\n  Lap time: {lap_time:.3f} s")

    n_seg = len(power_kW)
    distance = track.s[:-1]

    Q_cell_W = np.zeros(n_seg)
    for i in range(n_seg):
        if power_kW[i] > 0:
            I_pack = current_from_power(power_kW[i])
            I_cell = I_pack / N_PARALLEL
            Q_cell_W[i] = I_cell**2 * (CELL_ESR_MOHM / 1000.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Heat Dissipation Per Cell Over an Autocross Lap",
                 fontsize=13, fontweight="bold")

    ax.plot(distance, Q_cell_W, "#FF5722", linewidth=1.2)
    ax.fill_between(distance, Q_cell_W, alpha=0.15, color="#FF5722")

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Heat per cell [W]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig2_heat_per_cell.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()

    total_J = np.sum(Q_cell_W * dt)
    print(f"\n  Single lap heat summary:")
    print(f"    Total heat per cell: {total_J:.2f} J")
    print(f"    Peak heat per cell:  {Q_cell_W.max():.2f} W")
    print(f"    Mean heat (driving): {Q_cell_W[Q_cell_W > 0].mean():.2f} W")

    return power_kW, dt, lap_time


# ── Figure 3: Temperature rise over a single lap ──────────────────────────
def fig3_single_lap_temperature(power_kW, dt, lap_time, save_dir=None):
    """Adiabatic temperature rise per cell over one autocross lap."""
    n_seg = len(power_kW)
    cell_heat_cap = CELL_MASS_G * CELL_SPECIFIC_HEAT  # [J/K]

    # Cumulative heat and temperature
    Q_cum = 0.0
    T_profile = np.zeros(n_seg)
    for i in range(n_seg):
        if power_kW[i] > 0:
            I_pack = current_from_power(power_kW[i])
            I_cell = I_pack / N_PARALLEL
            Q_cum += I_cell**2 * (CELL_ESR_MOHM / 1000.0) * dt[i]
        T_profile[i] = T_AMBIENT + Q_cum / cell_heat_cap

    time_s = np.cumsum(dt)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cell Temperature Over a Single Autocross Lap (No Cooling)",
                 fontsize=13, fontweight="bold")

    ax.plot(time_s, T_profile, "#FF5722", linewidth=2.5)
    ax.axhline(T_AMBIENT, color="grey", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cell temperature [C]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig3_single_lap_temperature.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()

    dT = T_profile[-1] - T_AMBIENT
    print(f"\n  Single lap temperature rise (adiabatic):")
    print(f"    Start:  {T_AMBIENT:.1f} C")
    print(f"    End:    {T_profile[-1]:.1f} C")
    print(f"    Rise:   {dT:.2f} C")


# ── Figure 4: Endurance temperature (forced air cooling) ──────────────────
def fig4_endurance_temperature(power_kW, dt, lap_time, save_dir=None):
    """Cell temperature over a 20-lap endurance with forced air cooling."""
    n_laps = 20
    n_seg = len(power_kW)
    total_segments = n_laps * n_seg

    cell_heat_cap = CELL_MASS_G * CELL_SPECIFIC_HEAT
    A_cell_cool = A_COOLING / (N_SERIES * N_PARALLEL)

    time_s = np.zeros(total_segments)
    T_profile = np.zeros(total_segments)

    T = T_AMBIENT
    t_cum = 0.0

    for lap in range(n_laps):
        for i in range(n_seg):
            idx = lap * n_seg + i
            t_cum += dt[i]
            time_s[idx] = t_cum

            # Heat in
            if power_kW[i] > 0:
                I_pack = current_from_power(power_kW[i])
                I_cell = I_pack / N_PARALLEL
                Q_in = I_cell**2 * (CELL_ESR_MOHM / 1000.0) * dt[i]
            else:
                Q_in = 0.0

            # Cooling out
            Q_cool = H_FORCED * A_cell_cool * (T - T_AMBIENT) * dt[i]

            T += (Q_in - Q_cool) / cell_heat_cap
            T_profile[idx] = T

    time_min = time_s / 60.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Cell Temperature Over {n_laps} Lap Endurance (Forced Air Cooling)",
                 fontsize=13, fontweight="bold")

    ax.plot(time_min, T_profile, "#FF5722", linewidth=2.5)
    ax.axhline(T_DERATE_START, color="#FF9800", linewidth=1.5,
               label=f"BMS derate threshold ({T_DERATE_START:.0f} C)")
    ax.axhline(T_SAFE_MAX, color="#F44336", linewidth=1.5,
               label=f"Manufacturer safe limit ({T_SAFE_MAX:.0f} C)")

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Cell temperature [C]")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig4_endurance_temperature.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()

    # Find derate crossing
    derate_idx = np.argmax(T_profile >= T_DERATE_START)
    safe_idx = np.argmax(T_profile >= T_SAFE_MAX)

    print(f"\n  Endurance thermal summary ({n_laps} laps, {time_s[-1]/60:.1f} min):")
    print(f"    Final temperature:     {T_profile[-1]:.1f} C")
    if T_profile[derate_idx] >= T_DERATE_START:
        print(f"    Reaches derate at:     {time_s[derate_idx]/60:.1f} min "
              f"(lap {derate_idx // n_seg + 1})")
    if T_profile[safe_idx] >= T_SAFE_MAX:
        print(f"    Reaches safe limit at: {time_s[safe_idx]/60:.1f} min "
              f"(lap {safe_idx // n_seg + 1})")

    return T_profile, time_s


# ── Figure 5: ESR growth due to temperature feedback ──────────────────────
def fig5_esr_feedback(power_kW, dt, lap_time, save_dir=None):
    """Show how ESR grows as cells heat up during endurance."""
    n_laps = 20
    n_seg = len(power_kW)
    total_segments = n_laps * n_seg

    cell_heat_cap = CELL_MASS_G * CELL_SPECIFIC_HEAT
    A_cell_cool = A_COOLING / (N_SERIES * N_PARALLEL)

    time_s = np.zeros(total_segments)
    esr_ratio = np.zeros(total_segments)

    T = T_AMBIENT
    t_cum = 0.0

    for lap in range(n_laps):
        for i in range(n_seg):
            idx = lap * n_seg + i
            t_cum += dt[i]
            time_s[idx] = t_cum

            # Dynamic ESR at current temperature
            R_dyn = esr_vs_temperature(T) / 1000.0
            esr_ratio[idx] = esr_vs_temperature(T) / CELL_ESR_MOHM

            if power_kW[i] > 0:
                I_pack = current_from_power(power_kW[i])
                I_cell = I_pack / N_PARALLEL
                Q_in = I_cell**2 * R_dyn * dt[i]
            else:
                Q_in = 0.0

            Q_cool = H_FORCED * A_cell_cool * (T - T_AMBIENT) * dt[i]
            T += (Q_in - Q_cool) / cell_heat_cap

    time_min = time_s / 60.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Cell Internal Resistance Growth Over {n_laps} Lap Endurance",
                 fontsize=13, fontweight="bold")

    ax.plot(time_min, esr_ratio, "#9C27B0", linewidth=2.5)
    ax.axhline(1.0, color="grey", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Internal resistance relative to 25 C nominal [-]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig5_esr_feedback.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()

    print(f"\n  Internal resistance feedback summary:")
    print(f"    Starting R_int: {CELL_ESR_MOHM:.1f} mOhm (1.00x)")
    print(f"    Final R_int:    {esr_ratio[-1] * CELL_ESR_MOHM:.1f} mOhm "
          f"({esr_ratio[-1]:.2f}x)")
    print(f"    R_int increase: {(esr_ratio[-1] - 1) * 100:.0f}%")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Battery thermal model assumption validation")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save figures")
    args = parser.parse_args()
    save_dir = args.save

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "config", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("  BATTERY THERMAL MODEL ASSUMPTION VALIDATION")
    print("=" * 70)
    print(f"\n  Pack: {N_SERIES}S{N_PARALLEL}P ({N_SERIES * N_PARALLEL} cells)")
    print(f"  Cell R_int at 25 C: {CELL_ESR_MOHM} mOhm")
    print(f"  Pack R_int at 25 C: {R_PACK_OHM*1000:.1f} mOhm")
    print(f"  Cell mass:          {CELL_MASS_G} g")
    print(f"  Cell Cp:            {CELL_SPECIFIC_HEAT} J/(g K)")

    print("\n" + "=" * 70)
    print("  STEP 1: Cell Internal Resistance vs Temperature")
    print("=" * 70)
    fig1_esr_vs_temperature(save_dir)

    print("\n" + "=" * 70)
    print("  STEP 2: Heat Per Cell Over a Lap")
    print("=" * 70)
    power_kW, dt, lap_time = fig2_heat_per_cell(config, save_dir)

    print("\n" + "=" * 70)
    print("  STEP 3: Single Lap Temperature Rise")
    print("=" * 70)
    fig3_single_lap_temperature(power_kW, dt, lap_time, save_dir)

    print("\n" + "=" * 70)
    print("  STEP 4: Endurance Temperature (Forced Air Cooling)")
    print("=" * 70)
    fig4_endurance_temperature(power_kW, dt, lap_time, save_dir)

    print("\n" + "=" * 70)
    print("  STEP 5: Internal Resistance Growth From Thermal Feedback")
    print("=" * 70)
    fig5_esr_feedback(power_kW, dt, lap_time, save_dir)

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
