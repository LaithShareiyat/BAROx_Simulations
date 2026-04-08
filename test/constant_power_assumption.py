"""
Constant power assumption validation for the LTS.

The LTS models the powertrain as:
  - Constant torque below crossover speed: F_x = F_x,max
  - Constant power above crossover speed:  F_x = P_max / v
  - Zero force above RPM limit

This script compares the idealised envelope against the real EMRAX 208 HV
torque-speed characteristics and shows that the FS rules 80 kW power cap
makes the constant power assumption valid.

Usage:
    python test/constant_power_assumption.py
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
from events.autocross_generator import build_standard_autocross
from solver.qss_speed import solve_qss
from physics.powertrain import max_tractive_force_extended
from physics.aero import drag
from physics.resistive import rolling_resistance

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── EMRAX 208 HV Air Cooled Specifications ────────────────────────────────
EMRAX_PEAK_TORQUE_NM = 150.0       # Peak torque (flat region) [Nm]
EMRAX_CORNER_RPM = 5500.0          # RPM where field weakening begins
EMRAX_PEAK_POWER_KW = 86.0         # Peak motor power [kW]
EMRAX_MAX_RPM = 7000.0             # Maximum motor RPM


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


def emrax_real_torque(rpm):
    """Real EMRAX 208 HV torque-speed envelope (per motor).

    - 0 to corner_rpm: flat peak torque (150 Nm)
    - corner_rpm to max_rpm: field weakening (constant power at corner)
    - Above max_rpm: zero

    Power in the field weakening region is set to T_peak × ω_corner
    to ensure a smooth transition (no discontinuity at corner speed).
    """
    rpm = np.atleast_1d(np.asarray(rpm, dtype=float))
    T = np.zeros_like(rpm)

    omega_corner = EMRAX_CORNER_RPM * 2 * np.pi / 60
    P_corner = EMRAX_PEAK_TORQUE_NM * omega_corner  # Smooth join

    for i in range(len(rpm)):
        if rpm[i] <= 0:
            T[i] = EMRAX_PEAK_TORQUE_NM
        elif rpm[i] <= EMRAX_CORNER_RPM:
            T[i] = EMRAX_PEAK_TORQUE_NM
        elif rpm[i] <= EMRAX_MAX_RPM:
            omega = rpm[i] * 2 * np.pi / 60
            T[i] = P_corner / omega
        else:
            T[i] = 0.0

    return T


def lts_torque_per_motor(rpm, pt):
    """LTS idealised torque per motor at given RPM.

    Applies FS rules power cap (P_max includes the cap).
    """
    rpm = np.atleast_1d(np.asarray(rpm, dtype=float))
    T = np.zeros_like(rpm)

    P_per_motor = pt.P_max / pt.n_motors  # FS-capped power per motor
    T_max = pt.motor_torque_Nm

    for i in range(len(rpm)):
        if rpm[i] <= 0:
            T[i] = T_max
        elif rpm[i] > pt.motor_rpm_max:
            T[i] = 0.0
        else:
            omega = rpm[i] * 2 * np.pi / 60
            T_power = P_per_motor / max(omega, 0.1)
            T[i] = min(T_max, T_power)

    return T


# ── Plot 1: Motor Torque Envelope ─────────────────────────────────────────

def plot_torque_envelope(vehicle):
    """Real EMRAX 208 vs LTS idealised torque envelope (per motor)."""
    pt = vehicle.powertrain
    rpm = np.linspace(0.1, EMRAX_MAX_RPM * 1.05, 1000)

    T_real = emrax_real_torque(rpm)
    T_lts = lts_torque_per_motor(rpm, pt)

    # Crossover RPM (where FS power cap starts limiting)
    P_per_motor = pt.P_max / pt.n_motors
    omega_cross = P_per_motor / pt.motor_torque_Nm
    rpm_cross = omega_cross * 60 / (2 * np.pi)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rpm, T_real, color="#888888", linewidth=2.5,
            label="EMRAX 208 capability")
    ax.plot(rpm, T_lts, color="#2070B0", linewidth=2.5,
            label="LTS model (FS 80 kW cap)")

    # Shade the torque margin (motor headroom)
    ax.fill_between(rpm, T_lts, T_real,
                    where=T_real > T_lts,
                    color="#30A050", alpha=0.15, label="Motor torque headroom")

    # Mark crossover speed
    ax.plot(rpm_cross, pt.motor_torque_Nm, "o", color="#D04040",
            markersize=8, zorder=5,
            label=f"FS power cap onset ({rpm_cross:.0f} RPM)")

    # Mark field weakening onset
    ax.plot(EMRAX_CORNER_RPM, EMRAX_PEAK_TORQUE_NM, "s", color="#E8A020",
            markersize=8, zorder=5,
            label=f"Field weakening onset ({EMRAX_CORNER_RPM:.0f} RPM)")

    ax.set_xlabel("Motor Speed [RPM]", fontsize=12)
    ax.set_ylabel("Torque per Motor [Nm]", fontsize=12)
    ax.set_title("EMRAX 208 Torque Envelope vs LTS Model",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, EMRAX_MAX_RPM * 1.05)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "powertrain_torque_envelope.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Torque margin at key speeds
    print(f"\n  FS power cap crossover: {rpm_cross:.0f} RPM "
          f"({rpm_cross / 60 * 2 * np.pi * pt.wheel_radius_m / pt.gear_ratio * 3.6:.0f} km/h)")
    print(f"  Field weakening onset:  {EMRAX_CORNER_RPM:.0f} RPM "
          f"({EMRAX_CORNER_RPM / 60 * 2 * np.pi * pt.wheel_radius_m / pt.gear_ratio * 3.6:.0f} km/h)")


# ── Plot 2: Power Envelope ────────────────────────────────────────────────

def plot_power_envelope(vehicle):
    """Real EMRAX 208 vs LTS power envelope with autocross operating points."""
    pt = vehicle.powertrain
    rpm = np.linspace(0.1, EMRAX_MAX_RPM * 1.05, 1000)
    omega = rpm * 2 * np.pi / 60

    T_real = emrax_real_torque(rpm)
    T_lts = lts_torque_per_motor(rpm, pt)

    # Total powertrain power (n_motors)
    P_real_kW = T_real * omega * pt.n_motors / 1000.0
    P_lts_kW = T_lts * omega * pt.n_motors / 1000.0

    # Autocross operating points
    track, _ = build_standard_autocross()
    result, _ = solve_qss(track, vehicle, refine=True, min_points=500,
                           use_bicycle_model=True)
    v = result["v"]

    motor_rpm_op = v * 60 * pt.gear_ratio / (2 * np.pi * pt.wheel_radius_m)

    a_x = np.zeros(len(v))
    for i in range(len(v) - 1):
        if track.ds[i] > 0.01:
            a_x[i] = (v[i + 1] ** 2 - v[i] ** 2) / (2 * track.ds[i])

    F_wheel = np.zeros(len(v))
    for i in range(len(v)):
        Fdrag = drag(vehicle.aero.rho, vehicle.aero.CD_A, v[i])
        Frr = rolling_resistance(vehicle.Crr, vehicle.m, vehicle.g)
        F_wheel[i] = vehicle.m * a_x[i] + Fdrag + Frr

    # Total power demand at wheel
    P_demand_kW = F_wheel * v / 1000.0
    drive_mask = P_demand_kW > 0.5

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rpm, P_real_kW, color="#888888", linewidth=2.5,
            label=f"EMRAX 208 capability ({EMRAX_PEAK_POWER_KW * pt.n_motors:.0f} kW peak)")
    ax.plot(rpm, P_lts_kW, color="#2070B0", linewidth=2.5,
            label=f"LTS model ({pt.P_max/1000:.0f} kW FS cap)")

    # FS rules cap line
    ax.axhline(pt.P_max / 1000, color="#D04040", linewidth=1.5, alpha=0.5,
               label=f"FS rules limit ({pt.P_max/1000:.0f} kW)")

    # Operating points
    ax.scatter(motor_rpm_op[drive_mask], P_demand_kW[drive_mask],
               s=8, color="#D04040", alpha=0.5, zorder=3,
               label="Autocross operating points")

    ax.set_xlabel("Motor Speed [RPM]", fontsize=12)
    ax.set_ylabel("Total Power [kW]", fontsize=12)
    ax.set_title("Power Envelope: EMRAX 208 Capability vs FS Rules Cap",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, EMRAX_MAX_RPM * 1.05)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "powertrain_power_envelope.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 3: Operating Points on Torque Envelope ──────────────────────────

def plot_operating_points(vehicle):
    """Autocross operating points overlaid on the torque envelope."""
    pt = vehicle.powertrain
    track, _ = build_standard_autocross()
    result, _ = solve_qss(track, vehicle, refine=True, min_points=500,
                           use_bicycle_model=True)
    v = result["v"]

    # Motor RPM from vehicle speed
    motor_rpm = v * 60 * pt.gear_ratio / (2 * np.pi * pt.wheel_radius_m)

    # Compute motor torque demand at each point
    a_x = np.zeros(len(v))
    for i in range(len(v) - 1):
        if track.ds[i] > 0.01:
            a_x[i] = (v[i + 1] ** 2 - v[i] ** 2) / (2 * track.ds[i])

    F_wheel = np.zeros(len(v))
    for i in range(len(v)):
        Fdrag = drag(vehicle.aero.rho, vehicle.aero.CD_A, v[i])
        Frr = rolling_resistance(vehicle.Crr, vehicle.m, vehicle.g)
        F_wheel[i] = vehicle.m * a_x[i] + Fdrag + Frr

    motor_torque = F_wheel * pt.wheel_radius_m / (pt.gear_ratio * pt.n_motors)
    drive_mask = motor_torque > 1.0

    # Envelopes
    rpm_env = np.linspace(0.1, EMRAX_MAX_RPM * 1.05, 1000)
    T_real = emrax_real_torque(rpm_env)
    T_lts = lts_torque_per_motor(rpm_env, pt)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rpm_env, T_real, color="#888888", linewidth=2,
            label="EMRAX 208 capability")
    ax.plot(rpm_env, T_lts, color="#2070B0", linewidth=2,
            label="LTS model (FS 80 kW cap)")

    ax.scatter(motor_rpm[drive_mask], motor_torque[drive_mask],
               s=8, color="#D04040", alpha=0.5, zorder=3,
               label="Autocross operating points")

    ax.set_xlabel("Motor Speed [RPM]", fontsize=12)
    ax.set_ylabel("Torque per Motor [Nm]", fontsize=12)
    ax.set_title("Autocross Operating Points on Torque Envelope",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, EMRAX_MAX_RPM * 1.05)
    ax.set_ylim(bottom=0, top=EMRAX_PEAK_TORQUE_NM * 1.1)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "powertrain_operating_envelope.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Statistics
    rpm_drive = motor_rpm[drive_mask]
    P_per_motor = pt.P_max / pt.n_motors
    omega_cross = P_per_motor / pt.motor_torque_Nm
    rpm_cross = omega_cross * 60 / (2 * np.pi)
    pct_torque_region = np.sum(rpm_drive < rpm_cross) / len(rpm_drive) * 100
    pct_power_region = np.sum(rpm_drive >= rpm_cross) / len(rpm_drive) * 100
    pct_above_fw = np.sum(rpm_drive > EMRAX_CORNER_RPM) / len(rpm_drive) * 100

    print(f"\n  Operating point breakdown:")
    print(f"    In torque-limited region:     {pct_torque_region:.1f}%")
    print(f"    In power-limited region:      {pct_power_region:.1f}%")
    print(f"    Above field weakening onset:  {pct_above_fw:.1f}%")
    print(f"    Max motor RPM reached:        {np.max(rpm_drive):.0f} RPM")


if __name__ == "__main__":
    print("Constant Power Assumption Validation")
    print("=" * 50)
    vehicle = build_vehicle()

    print("\n1. Torque Envelope: Real vs LTS")
    print("-" * 40)
    plot_torque_envelope(vehicle)

    print("\n2. Power Envelope: Capability vs FS Cap")
    print("-" * 40)
    plot_power_envelope(vehicle)

    print("\n3. Operating Points on Envelope")
    print("-" * 40)
    plot_operating_points(vehicle)
