"""
Battery Discharge Efficiency Assumption Validation.

Investigates whether the constant eta_discharge = 0.95 assumption is valid
by comparing it against a physics-based I²R internal resistance model.

Cell: 2600 mAh / 3.6 V / 12 mOhm ESR (Samsung 25R class 18650)
Pack: 142S5P  ->  R_pack = (142 x 12 mOhm) / 5 = 340.8 mOhm

Usage:
    python test/battery_efficiency_assumption.py
    python test/battery_efficiency_assumption.py --save test/
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
CELL_ESR_MOHM = 12.0         # Internal resistance per cell [mOhm]
N_SERIES = 142
N_PARALLEL = 5
V_NOMINAL_CELL = 3.6          # [V]
V_MAX_CELL = 4.2              # [V]
CAPACITY_MAH = 2600           # [mAh]
MAX_CELL_CURRENT_A = 35.0     # [A]

# Pack-level derived values
R_PACK_OHM = (N_SERIES * CELL_ESR_MOHM / 1000.0) / N_PARALLEL
V_NOMINAL_PACK = N_SERIES * V_NOMINAL_CELL
I_MAX_PACK = N_PARALLEL * MAX_CELL_CURRENT_A
CAPACITY_AH = (CAPACITY_MAH / 1000.0) * N_PARALLEL
CAPACITY_KWH = CAPACITY_AH * V_NOMINAL_PACK / 1000.0

# Simulator assumption
ETA_FLAT = 0.93


def ocv_from_soc(soc: float) -> float:
    """Approximate open-circuit voltage per cell from SoC.

    Polynomial fit typical of NMC/NCA 18650 cells:
        OCV = 3.0 + 1.2 * SoC - 0.35 * SoC^2 + 0.35 * SoC^3
    Gives ~3.0 V at 0% SoC and ~4.2 V at 100% SoC.
    """
    soc = np.clip(soc, 0.0, 1.0)
    return 3.0 + 1.2 * soc - 0.35 * soc**2 + 0.35 * soc**3


def pack_ocv(soc: float) -> float:
    """Pack open-circuit voltage from SoC [V]."""
    return N_SERIES * ocv_from_soc(soc)


def real_efficiency(current_A: float, soc: float = 0.5) -> float:
    """True battery discharge efficiency including I^2 R losses.

    eta = 1 - (I * R_pack) / V_ocv
    """
    v_ocv = pack_ocv(soc)
    v_drop = current_A * R_PACK_OHM
    if v_ocv <= 0:
        return 0.0
    return max(1.0 - v_drop / v_ocv, 0.0)


def power_from_current(current_A: float, soc: float = 0.5) -> float:
    """Terminal power delivered to the motor [kW]."""
    v_ocv = pack_ocv(soc)
    v_term = v_ocv - current_A * R_PACK_OHM
    return max(v_term * current_A / 1000.0, 0.0)


def current_from_power(power_kW: float, soc: float = 0.5) -> float:
    """Solve for discharge current given desired terminal power [kW]."""
    v_ocv = pack_ocv(soc)
    P = power_kW * 1000.0
    R = R_PACK_OHM
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


# ── Figure 1: Efficiency vs Discharge Current ──────────────────────────────
def fig1_efficiency_vs_current(save_dir=None):
    """Discharge efficiency vs pack current at mid SoC."""
    currents = np.linspace(0, I_MAX_PACK, 500)
    etas = [real_efficiency(I, 0.5) for I in currents]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Discharge Efficiency vs Pack Current (SoC = 50%)",
                 fontsize=13, fontweight="bold")

    ax.plot(currents, etas, "#2196F3", linewidth=2.5, label="I²R model")
    ax.axhline(ETA_FLAT, color="#FF5722", linewidth=2,
               label=f"Simulator assumption (η = {ETA_FLAT})")

    # Crossover point
    crossover_idx = np.argmin(np.abs(np.array(etas) - ETA_FLAT))
    I_cross = currents[crossover_idx]
    P_cross = power_from_current(I_cross, 0.5)
    ax.plot(I_cross, ETA_FLAT, "ko", markersize=8, zorder=5)
    ax.annotate(f"Crossover: {I_cross:.0f} A ({P_cross:.1f} kW)",
                xy=(I_cross, ETA_FLAT), xytext=(I_cross + 12, ETA_FLAT + 0.012),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=10)

    ax.set_xlabel("Pack discharge current [A]")
    ax.set_ylabel("Discharge efficiency [-]")
    ax.set_ylim(0.85, 1.01)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig1_efficiency_vs_current.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()

    print("\n  Efficiency at key operating points (SoC = 50%):")
    for I in [25, 50, 75, 100, 125, 150, 175]:
        eta = real_efficiency(I, 0.5)
        P = power_from_current(I, 0.5)
        P_loss = I**2 * R_PACK_OHM
        print(f"    I = {I:4d} A  |  P = {P:5.1f} kW  |  "
              f"eta = {eta:.4f}  |  I²R loss = {P_loss:.0f} W  |  "
              f"error vs 0.95 = {(eta - ETA_FLAT)*100:+.2f} pp")


# ── Figure 2: I²R Power Loss vs Current ────────────────────────────────────
def fig2_power_loss_vs_current(save_dir=None):
    """Quadratic growth of resistive power loss with current."""
    currents = np.linspace(0, I_MAX_PACK, 500)
    P_loss_kW = currents**2 * R_PACK_OHM / 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Resistive Power Loss vs Pack Current",
                 fontsize=13, fontweight="bold")

    ax.plot(currents, P_loss_kW, "#FF5722", linewidth=2.5)
    ax.fill_between(currents, P_loss_kW, alpha=0.15, color="#FF5722")

    # Annotate max
    ax.annotate(f"{P_loss_kW[-1]:.1f} kW at {I_MAX_PACK:.0f} A",
                xy=(I_MAX_PACK, P_loss_kW[-1]),
                xytext=(I_MAX_PACK - 50, P_loss_kW[-1] - 1.5),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=11)

    ax.set_xlabel("Pack discharge current [A]")
    ax.set_ylabel("I²R power loss [kW]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig2_power_loss_vs_current.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()


# ── Figure 3: Efficiency vs Delivered Power ─────────────────────────────────
def fig3_efficiency_vs_power(save_dir=None):
    """Efficiency mapped to the power domain at mid SoC."""
    powers_kW = np.linspace(0.5, 80, 500)
    etas = []
    for P in powers_kW:
        I = current_from_power(P, 0.5)
        etas.append(real_efficiency(I, 0.5))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Discharge Efficiency vs Delivered Power (SoC = 50%)",
                 fontsize=13, fontweight="bold")

    ax.plot(powers_kW, etas, "#2196F3", linewidth=2.5, label="I²R model")
    ax.axhline(ETA_FLAT, color="#FF5722", linewidth=2,
               label=f"Simulator assumption (η = {ETA_FLAT})")

    ax.set_xlabel("Terminal power delivered to motor [kW]")
    ax.set_ylabel("Discharge efficiency [-]")
    ax.set_ylim(0.88, 1.01)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig3_efficiency_vs_power.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()


# ── Figure 4: Efficiency Over a Lap ────────────────────────────────────────
def fig4_lap_efficiency(config, save_dir=None):
    """Real efficiency at each discharge segment over a lap."""
    track, _ = build_standard_autocross()
    vehicle = build_vehicle(config)
    result, lap_time = solve_qss(track, vehicle)
    v = result["v"]
    print(f"\n  Lap time: {lap_time:.3f} s")

    power_kW = calculate_power_profile(track, v, vehicle)
    n_seg = len(power_kW)

    v_avg = 0.5 * (v[:-1] + v[1:])
    v_avg = np.maximum(v_avg, 0.1)
    dt = track.ds / v_avg

    # Calculate real efficiency at each segment
    eta_real = np.ones(n_seg)
    cumulative_energy_kWh = 0.0
    for i in range(n_seg):
        soc_i = max(1.0 - cumulative_energy_kWh / CAPACITY_KWH, 0.0)
        if power_kW[i] > 0:
            I = current_from_power(power_kW[i], soc_i)
            eta_real[i] = real_efficiency(I, soc_i)
            cumulative_energy_kWh += (power_kW[i] / eta_real[i]) * dt[i] / 3600.0

    distance = track.s[:-1]
    discharge_mask = power_kW > 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Real Discharge Efficiency Over an Autocross Lap",
                 fontsize=13, fontweight="bold")

    ax.scatter(distance[discharge_mask], eta_real[discharge_mask],
               s=4, c="#2196F3", alpha=0.7, label="I²R model at each segment")
    ax.axhline(ETA_FLAT, color="#FF5722", linewidth=2,
               label=f"Simulator assumption (η = {ETA_FLAT})")

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Discharge efficiency [-]")
    ax.set_ylim(0.89, 1.01)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig4_lap_efficiency.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()

    # Summary
    eta_discharge_only = eta_real[discharge_mask]
    weighted_eta = np.average(eta_discharge_only,
                              weights=power_kW[discharge_mask] * dt[discharge_mask])
    print(f"\n  Lap efficiency summary:")
    print(f"    Power-weighted mean eta: {weighted_eta:.4f}")
    print(f"    Simulator assumption:    {ETA_FLAT}")
    print(f"    Error:                   {abs(weighted_eta - ETA_FLAT)*100:.2f} pp")
    print(f"    Min eta (peak power):    {eta_discharge_only.min():.4f}")
    print(f"    Max eta (low power):     {eta_discharge_only.max():.4f}")

    return power_kW, dt, eta_real


# ── Figure 5: Cumulative Energy Error Over a Lap ───────────────────────────
def fig5_energy_error(power_kW, dt, eta_real, config, save_dir=None):
    """Cumulative energy error from using a flat efficiency assumption."""
    n_seg = len(power_kW)
    track, _ = build_standard_autocross()
    distance = track.s[:-1]

    energy_flat = np.cumsum(np.where(power_kW > 0,
                                     power_kW / ETA_FLAT * dt / 3600.0, 0.0))
    energy_real = np.zeros(n_seg)
    cum = 0.0
    for i in range(n_seg):
        if power_kW[i] > 0:
            cum += (power_kW[i] / eta_real[i]) * dt[i] / 3600.0
        energy_real[i] = cum

    error_pct = (energy_flat - energy_real) / np.maximum(energy_real, 1e-9) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cumulative Energy Error from Constant Efficiency Assumption",
                 fontsize=13, fontweight="bold")

    ax.plot(distance, error_pct, "#9C27B0", linewidth=2)
    ax.axhline(0, color="grey", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("Energy error [%]")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "fig5_energy_error.png")
        fig.savefig(path, dpi=150)
        print(f"  Saved: {path}")
    plt.show()

    print(f"\n  Energy totals:")
    print(f"    Flat assumption: {energy_flat[-1]*1000:.1f} Wh")
    print(f"    I²R model:       {energy_real[-1]*1000:.1f} Wh")
    print(f"    Difference:      {(energy_flat[-1] - energy_real[-1])*1000:.1f} Wh "
          f"({error_pct[-1]:+.2f}%)")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Battery discharge efficiency assumption validation")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save figures")
    args = parser.parse_args()
    save_dir = args.save

    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "config", "default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("  BATTERY DISCHARGE EFFICIENCY ASSUMPTION VALIDATION")
    print("=" * 70)
    print(f"\n  Pack: {N_SERIES}S{N_PARALLEL}P")
    print(f"  Cell ESR:        {CELL_ESR_MOHM} mOhm")
    print(f"  Pack resistance: {R_PACK_OHM*1000:.1f} mOhm ({R_PACK_OHM:.4f} Ohm)")
    print(f"  Nominal voltage: {V_NOMINAL_PACK:.1f} V")
    print(f"  Max current:     {I_MAX_PACK:.0f} A")
    print(f"  Simulator eta:   {ETA_FLAT}")

    print("\n" + "=" * 70)
    print("  STEP 1: Efficiency vs Discharge Current")
    print("=" * 70)
    fig1_efficiency_vs_current(save_dir)

    print("\n" + "=" * 70)
    print("  STEP 2: Resistive Power Loss vs Current")
    print("=" * 70)
    fig2_power_loss_vs_current(save_dir)

    print("\n" + "=" * 70)
    print("  STEP 3: Efficiency vs Delivered Power")
    print("=" * 70)
    fig3_efficiency_vs_power(save_dir)

    print("\n" + "=" * 70)
    print("  STEP 4: Efficiency Over a Simulated Lap")
    print("=" * 70)
    power_kW, dt, eta_real = fig4_lap_efficiency(config, save_dir)

    print("\n" + "=" * 70)
    print("  STEP 5: Cumulative Energy Error")
    print("=" * 70)
    fig5_energy_error(power_kW, dt, eta_real, config, save_dir)

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
