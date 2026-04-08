"""
Powertrain assumption validation for the LTS.

Analyses two key assumptions:
  1. Constant power: flat efficiency produces an idealised torque-power envelope
  2. Instantaneous torque response: motor delivers torque with zero delay

Usage:
    python test/powertrain_assumptions.py
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


# ── Assumption 1: Constant Power (Idealised Torque–Power Envelope) ─────────

def plot_torque_power_envelope(vehicle):
    """Plot the idealised F_x vs speed envelope with torque, power, and RPM regions."""
    pt = vehicle.powertrain
    v_max_rpm = pt.v_max_rpm
    v_cross = pt.v_crossover

    v = np.linspace(0.1, v_max_rpm * 1.15, 1000)
    Fx = np.array([max_tractive_force_extended(pt, vi) for vi in v])
    power_kW = Fx * v / 1000.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Top: Tractive force vs speed ---
    ax1.plot(v * 3.6, Fx, color="#2070B0", linewidth=2.5)

    # Shade regions
    v_torque = v[v <= v_cross]
    v_power = v[(v > v_cross) & (v <= v_max_rpm)]
    v_rpm = v[v > v_max_rpm]

    ax1.fill_between(v_torque * 3.6, 0,
                     [max_tractive_force_extended(pt, vi) for vi in v_torque],
                     color="#30A050", alpha=0.15, label="Torque limited")
    ax1.fill_between(v_power * 3.6, 0,
                     [max_tractive_force_extended(pt, vi) for vi in v_power],
                     color="#2070B0", alpha=0.15, label="Power limited")
    if len(v_rpm) > 0:
        ax1.fill_between(v_rpm * 3.6, 0,
                         [max_tractive_force_extended(pt, vi) for vi in v_rpm],
                         color="#D04040", alpha=0.15, label="RPM limited")

    # Crossover and RPM limit markers
    ax1.axvline(v_cross * 3.6, color="#888888", linewidth=1, linestyle="solid", alpha=0.5)
    ax1.axvline(v_max_rpm * 3.6, color="#888888", linewidth=1, linestyle="solid", alpha=0.5)

    ax1.set_ylabel("Tractive Force [N]", fontsize=12)
    ax1.set_title("Idealised Powertrain Envelope", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Annotate key values
    ax1.text(v_cross * 3.6 / 2, pt.Fx_max * 0.85,
             f"$F_{{x,max}}$ = {pt.Fx_max:.0f} N", fontsize=10, ha="center",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # --- Bottom: Power vs speed ---
    ax2.plot(v * 3.6, power_kW, color="#D04040", linewidth=2.5)
    ax2.axhline(pt.P_max / 1000, color="#888888", linewidth=1, linestyle="solid", alpha=0.5)

    ax2.fill_between(v_torque * 3.6, 0,
                     [max_tractive_force_extended(pt, vi) * vi / 1000 for vi in v_torque],
                     color="#30A050", alpha=0.15)
    ax2.fill_between(v_power * 3.6, 0,
                     [max_tractive_force_extended(pt, vi) * vi / 1000 for vi in v_power],
                     color="#2070B0", alpha=0.15)

    ax2.text(v_max_rpm * 3.6 * 0.6, pt.P_max / 1000 + 2,
             f"$P_{{max}}$ = {pt.P_max/1000:.0f} kW (FS rules cap)",
             fontsize=10, ha="center",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax2.set_xlabel("Speed [km/h]", fontsize=12)
    ax2.set_ylabel("Tractive Power [kW]", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(0, v[-1] * 3.6)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "powertrain_envelope.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_operating_points(vehicle):
    """Show where the vehicle operates on the motor map during an autocross lap."""
    pt = vehicle.powertrain
    track, _ = build_standard_autocross()
    result, lap_time = solve_qss(track, vehicle, refine=True, min_points=500,
                                  use_bicycle_model=True)
    v = result["v"]

    # Compute actual tractive force at each point
    n = len(v)
    Fx_available = np.array([max_tractive_force_extended(pt, vi) for vi in v])
    power_available_kW = Fx_available * v / 1000.0

    # Actual power demand: F_net * v (from kinematics)
    v_avg = 0.5 * (v[:-1] + v[1:])
    a_x = np.diff(v ** 2) / (2 * track.ds)
    power_demand_kW = vehicle.m * a_x * v_avg / 1000.0

    # Only acceleration segments matter for powertrain
    accel_mask = power_demand_kW > 0

    fig, ax = plt.subplots(figsize=(10, 6))

    # Envelope curve
    v_env = np.linspace(0.1, pt.v_max_rpm, 500)
    Fx_env = np.array([max_tractive_force_extended(pt, vi) for vi in v_env])
    P_env = Fx_env * v_env / 1000.0
    ax.plot(v_env * 3.6, P_env, color="#2070B0", linewidth=2.5,
            label="Available power")

    # Operating points (acceleration only)
    ax.scatter(v_avg[accel_mask] * 3.6, power_demand_kW[accel_mask],
               s=6, color="#D04040", alpha=0.5, zorder=3,
               label="Acceleration operating points")

    # FS rules cap
    ax.axhline(pt.P_max / 1000, color="#888888", linewidth=1, linestyle="solid",
               alpha=0.5, label=f"FS rules cap ({pt.P_max/1000:.0f} kW)")

    ax.set_xlabel("Speed [km/h]", fontsize=12)
    ax.set_ylabel("Power [kW]", fontsize=12)
    ax.set_title("Powertrain Operating Points During Autocross",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, pt.v_max_rpm * 3.6 * 1.05)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "powertrain_operating_points.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Statistics
    P_accel = power_demand_kW[accel_mask]
    print(f"\n  Operating point statistics (acceleration only):")
    print(f"    Mean power demand:  {np.mean(P_accel):.1f} kW")
    print(f"    Median power demand: {np.median(P_accel):.1f} kW")
    print(f"    Max power demand:   {np.max(P_accel):.1f} kW")
    print(f"    % time at P_max:    {np.sum(P_accel > pt.P_max/1000 * 0.95) / len(P_accel) * 100:.1f}%")


# ── Assumption 2: Instantaneous Torque Response ───────────────────────────

def plot_torque_response_justification(vehicle):
    """Compare motor electrical time constant to solver segment transit times."""
    track, _ = build_standard_autocross()
    result, lap_time = solve_qss(track, vehicle, refine=True, min_points=500,
                                  use_bicycle_model=True)
    v = result["v"]

    # Segment transit times
    v_avg = 0.5 * (v[:-1] + v[1:])
    v_avg = np.maximum(v_avg, 0.1)
    dt_segment = track.ds / v_avg

    # EMRAX 208 electrical time constant
    # L/R time constant for PMSM is typically 1-5 ms
    # EMRAX 208: L ≈ 0.1 mH, R_phase ≈ 8 mΩ → τ_e = L/R ≈ 12.5 ms
    # However, with field-oriented control, torque rise time is ~5-10 ms
    # Conservative estimate: 10 ms (includes inverter + controller delay)
    tau_motor_ms = 10.0  # Conservative torque rise time [ms]

    # Torque reaches 95% (3τ) and 99% (5τ) of commanded value
    t_95 = 3 * tau_motor_ms  # 30 ms
    t_99 = 5 * tau_motor_ms  # 50 ms

    dt_ms = dt_segment * 1000.0
    s_mid = 0.5 * (track.s[:-1] + track.s[1:])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(s_mid, dt_ms, color="#2070B0", linewidth=1.5,
            label="Segment transit time")

    # Motor response time bands
    ax.axhline(tau_motor_ms, color="#30A050", linewidth=2, linestyle="solid",
               label=f"Motor time constant ({tau_motor_ms:.0f} ms)")
    ax.axhline(t_95, color="#E8A020", linewidth=1.5, linestyle="solid",
               alpha=0.7, label=f"95% response ({t_95:.0f} ms)")

    ax.set_xlabel("Distance [m]", fontsize=12)
    ax.set_ylabel("Time [ms]", fontsize=12)
    ax.set_title("Segment Transit Time vs Motor Torque Response",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(s_mid[0], s_mid[-1])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "powertrain_torque_response.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Statistics
    print(f"\n  Segment transit time statistics:")
    print(f"    Min:    {np.min(dt_ms):.1f} ms")
    print(f"    Mean:   {np.mean(dt_ms):.1f} ms")
    print(f"    Median: {np.median(dt_ms):.1f} ms")
    print(f"    Max:    {np.max(dt_ms):.1f} ms")
    print(f"    Motor time constant: {tau_motor_ms:.0f} ms")
    print(f"    Segments faster than τ: {np.sum(dt_ms < tau_motor_ms)}/{len(dt_ms)} "
          f"({np.sum(dt_ms < tau_motor_ms)/len(dt_ms)*100:.1f}%)")
    print(f"    Segments faster than 3τ: {np.sum(dt_ms < t_95)}/{len(dt_ms)} "
          f"({np.sum(dt_ms < t_95)/len(dt_ms)*100:.1f}%)")

    # Ratio: how many times larger is segment time vs motor response
    ratio = dt_ms / tau_motor_ms
    print(f"    Mean ratio (dt/τ): {np.mean(ratio):.1f}x")
    print(f"    Min ratio (dt/τ):  {np.min(ratio):.1f}x")


if __name__ == "__main__":
    print("Powertrain Assumption Validation")
    print("=" * 50)
    vehicle = build_vehicle()

    print("\n1. Constant Power Envelope")
    print("-" * 40)
    plot_torque_power_envelope(vehicle)

    print("\n2. Operating Points on Power Curve")
    print("-" * 40)
    plot_operating_points(vehicle)

    print("\n3. Instantaneous Torque Response")
    print("-" * 40)
    plot_torque_response_justification(vehicle)
