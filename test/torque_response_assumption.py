"""
Instantaneous torque response assumption validation.

The LTS assumes the motor delivers requested torque immediately.
In reality, the EMRAX 208 with field-oriented control (FOC) has
a torque rise time of ~5-10 ms (electrical time constant).

This script shows that the solver segment transit times are
significantly longer than the motor response time, making the
instantaneous torque assumption valid.

Usage:
    python test/torque_response_assumption.py
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

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# EMRAX 208 with FOC torque response
# L_phase ≈ 0.1 mH, R_phase ≈ 8 mΩ → τ_e = L/R ≈ 12.5 ms
# With FOC current loop bandwidth ~500 Hz → rise time ≈ 2 ms
# Conservative system-level estimate (inverter + controller + CAN): 5 ms
TAU_MOTOR_MS = 5.0


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


# ── Plot 1: Torque Step Response ──────────────────────────────────────────

def plot_step_response():
    """First-order torque step response showing settling time."""
    t_ms = np.linspace(0, 40, 500)
    tau = TAU_MOTOR_MS

    # First-order response: T(t) = T_cmd * (1 - e^(-t/τ))
    response = 1.0 - np.exp(-t_ms / tau)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t_ms, response * 100, color="#2070B0", linewidth=2.5,
            label="Motor torque response")

    # Mark key thresholds
    t_63 = tau  # 63% at 1τ
    t_95 = 3 * tau  # 95% at 3τ
    t_99 = 5 * tau  # 99% at 5τ

    ax.plot(t_63, 63.2, "o", color="#30A050", markersize=8, zorder=5,
            label=f"63% response (τ = {tau:.0f} ms)")
    ax.plot(t_95, 95.0, "o", color="#E8A020", markersize=8, zorder=5,
            label=f"95% response (3τ = {t_95:.0f} ms)")
    ax.plot(t_99, 99.3, "o", color="#D04040", markersize=8, zorder=5,
            label=f"99% response (5τ = {t_99:.0f} ms)")

    # Commanded torque level
    ax.axhline(100, color="#888888", linewidth=1, alpha=0.4)

    ax.set_xlabel("Time [ms]", fontsize=12)
    ax.set_ylabel("Torque Response [%]", fontsize=12)
    ax.set_title("EMRAX 208 Torque Step Response (FOC)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="center right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 110)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "torque_step_response.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot 2: Segment Time vs Motor Response ────────────────────────────────

def plot_segment_vs_response(vehicle):
    """Histogram of segment transit times with motor response time marked."""
    track, _ = build_standard_autocross()
    result, _ = solve_qss(track, vehicle, refine=True, min_points=500,
                           use_bicycle_model=True)
    v = result["v"]

    v_avg = 0.5 * (v[:-1] + v[1:])
    v_avg = np.maximum(v_avg, 0.1)
    dt_ms = (track.ds / v_avg) * 1000.0

    # Remove near-zero segments (numerical artefacts)
    dt_ms = dt_ms[dt_ms > 0.1]

    tau = TAU_MOTOR_MS
    t_95 = 3 * tau
    t_99 = 5 * tau

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, max(120, np.percentile(dt_ms, 99)), 60)
    ax.hist(dt_ms, bins=bins, color="#2070B0", edgecolor="white",
            linewidth=0.5, alpha=0.85)

    # Motor response markers
    ax.axvline(tau, color="#30A050", linewidth=2.5,
               label=f"Motor time constant (τ = {tau:.0f} ms)")
    ax.axvline(t_95, color="#E8A020", linewidth=2.5,
               label=f"95% torque response (3τ = {t_95:.0f} ms)")

    ax.set_xlabel("Segment Transit Time [ms]", fontsize=12)
    ax.set_ylabel("Number of Segments", fontsize=12)
    ax.set_title("Solver Segment Duration vs Motor Torque Response Time",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "torque_response_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Statistics
    pct_below_tau = np.sum(dt_ms < tau) / len(dt_ms) * 100
    pct_below_3tau = np.sum(dt_ms < t_95) / len(dt_ms) * 100
    ratio = np.median(dt_ms) / tau

    print(f"\n  Segment transit time statistics:")
    print(f"    Median:  {np.median(dt_ms):.1f} ms")
    print(f"    Mean:    {np.mean(dt_ms):.1f} ms")
    print(f"    Min:     {np.min(dt_ms):.1f} ms")
    print(f"    Max:     {np.max(dt_ms):.1f} ms")
    print(f"    Motor τ: {tau:.0f} ms")
    print(f"    Segments shorter than τ:   {pct_below_tau:.1f}%")
    print(f"    Segments shorter than 3τ:  {pct_below_3tau:.1f}%")
    print(f"    Median segment / τ ratio:  {ratio:.1f}x")


if __name__ == "__main__":
    print("Instantaneous Torque Response Assumption")
    print("=" * 50)
    vehicle = build_vehicle()

    print("\n1. Torque Step Response")
    print("-" * 40)
    plot_step_response()

    print("\n2. Segment Duration vs Motor Response")
    print("-" * 40)
    plot_segment_vs_response(vehicle)
