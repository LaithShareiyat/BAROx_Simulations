"""
Discretisation convergence study: lap time accuracy and runtime vs segment count.

Sweeps the number of solver segments from coarse to fine and measures
how the computed lap time converges and how runtime scales.

Usage:
    python test/discretisation_convergence.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
import time
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import (
    VehicleParams, EVPowertrainParams, VehicleGeometry,
    build_tyre_from_config, build_aero_from_config,
)
from models.track import from_xy
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


def resample_track(track, n_points):
    """Resample track to exactly n_points using cubic interpolation."""
    s = track.s
    # Remove duplicate s values for interpolation
    unique_mask = np.concatenate(([True], np.diff(s) > 1e-10))
    s_unique = s[unique_mask]
    x_unique = track.x[unique_mask]
    y_unique = track.y[unique_mask]

    s_new = np.linspace(s_unique[0], s_unique[-1], n_points)

    kind = "cubic" if len(s_unique) >= 4 else "linear"
    x_new = interp1d(s_unique, x_unique, kind=kind)(s_new)
    y_new = interp1d(s_unique, y_unique, kind=kind)(s_new)

    return from_xy(x_new, y_new, closed=track.closed)


def run_convergence_study():
    vehicle = build_vehicle()
    track_raw, _ = build_standard_autocross()

    n_points_list = [25, 50, 75, 100, 150, 200, 300, 500,
                     750, 1000, 1500, 2000, 3000, 5000,
                     7500, 10000, 15000, 20000, 30000, 50000,
                     75000, 100000, 150000, 200000]

    lap_times = []
    runtimes = []
    actual_points = []

    track_length = track_raw.s[-1]

    print("Running convergence study...")
    for n_pts in n_points_list:
        track = resample_track(track_raw, n_pts)
        actual_n = len(track.s)
        avg_ds = track_length / max(actual_n - 1, 1)

        # Fewer runs for large segment counts to keep total time reasonable
        n_runs = 1 if n_pts >= 50000 else 3

        times = []
        lt = None
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _, lt = solve_qss(track, vehicle, refine=False, use_bicycle_model=True)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_runtime = np.mean(times)
        lap_times.append(lt)
        runtimes.append(avg_runtime)
        actual_points.append(actual_n)

        print(f"  N = {actual_n:5d}  |  ds = {avg_ds:.3f} m  |  "
              f"Lap = {lt:.3f} s  |  Runtime = {avg_runtime:.4f} s")

    return n_points_list, lap_times, runtimes, actual_points


def plot_convergence(n_points, lap_times, runtimes, actual_points):
    lt_ref = lap_times[-1]

    # --- Plot 1: Lap Time Convergence ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(n_points, lap_times, color="#2070B0", linewidth=2, marker="o",
             markersize=5)

    ax1.set_xlabel("Number of Segments", fontsize=12)
    ax1.set_ylabel("Lap Time [s]", fontsize=12)
    ax1.set_title("Lap Time Convergence vs Discretisation",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xlim(n_points[0] * 0.8, n_points[-1] * 1.2)

    fig1.tight_layout()
    path1 = os.path.join(SAVE_DIR, "discretisation_laptime.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {path1}")

    # --- Plot 2: Runtime ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(n_points, [r * 1000 for r in runtimes], color="#D04040", linewidth=2,
             marker="o", markersize=5)

    ax2.set_xlabel("Number of Segments", fontsize=12)
    ax2.set_ylabel("Runtime [ms]", fontsize=12)
    ax2.set_title("Solver Runtime vs Discretisation",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_xlim(n_points[0] * 0.8, n_points[-1] * 1.2)
    ax2.set_ylim(bottom=0)

    fig2.tight_layout()
    path2 = os.path.join(SAVE_DIR, "discretisation_runtime.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    print("Discretisation Convergence Study")
    print("=" * 50)
    n_points, lap_times, runtimes, actual_points = run_convergence_study()
    plot_convergence(n_points, lap_times, runtimes, actual_points)
