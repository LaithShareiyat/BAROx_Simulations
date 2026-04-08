"""
Adaptive vs uniform discretisation study.

Compares lap time accuracy and runtime for:
  - Uniform segment spacing
  - Curvature-adaptive spacing (finer in corners, coarser on straights)

Usage:
    python test/adaptive_discretisation.py
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


def _clean_track_for_interp(track):
    """Return unique-s arrays suitable for interpolation."""
    s = track.s
    unique_mask = np.concatenate(([True], np.diff(s) > 1e-10))
    return s[unique_mask], track.x[unique_mask], track.y[unique_mask]


def resample_uniform(track, n_points):
    """Resample track with uniform arc-length spacing."""
    s_u, x_u, y_u = _clean_track_for_interp(track)
    kind = "cubic" if len(s_u) >= 4 else "linear"
    s_new = np.linspace(s_u[0], s_u[-1], n_points)
    x_new = interp1d(s_u, x_u, kind=kind)(s_new)
    y_new = interp1d(s_u, y_u, kind=kind)(s_new)
    return from_xy(x_new, y_new, closed=track.closed)


def resample_adaptive(track, n_points, curvature_weight=0.8):
    """Resample track with curvature-weighted spacing.

    Points are distributed so that regions of high curvature receive
    proportionally more segments.  ``curvature_weight`` controls the
    blend between uniform (0.0) and fully curvature-proportional (1.0).
    """
    s_u, x_u, y_u = _clean_track_for_interp(track)
    kind = "cubic" if len(s_u) >= 4 else "linear"

    # Build a curvature density function along the track
    kappa_interp = interp1d(s_u, np.abs(track.kappa[np.concatenate(([True], np.diff(track.s) > 1e-10))]),
                            kind="linear", fill_value="extrapolate")

    # Sample curvature at many points to build the density
    n_sample = max(10000, n_points * 10)
    s_dense = np.linspace(s_u[0], s_u[-1], n_sample)
    kappa_dense = np.abs(kappa_interp(s_dense))

    # Density = blend of uniform + curvature-proportional
    uniform_density = np.ones_like(s_dense)
    curvature_density = kappa_dense + 1e-6  # small floor so straights still get points
    density = (1.0 - curvature_weight) * uniform_density + curvature_weight * curvature_density

    # Normalised cumulative density -> maps [0, s_max] to [0, 1]
    cum_density = np.cumsum(density)
    cum_density = cum_density / cum_density[-1]

    # Invert: for n_points equally spaced in cumulative-density space,
    # find corresponding arc-length positions
    target = np.linspace(0, 1, n_points)
    s_new = np.interp(target, cum_density, s_dense)

    # Ensure endpoints are exact
    s_new[0] = s_u[0]
    s_new[-1] = s_u[-1]

    x_new = interp1d(s_u, x_u, kind=kind)(s_new)
    y_new = interp1d(s_u, y_u, kind=kind)(s_new)
    return from_xy(x_new, y_new, closed=track.closed)


def run_study():
    vehicle = build_vehicle()
    track_raw, _ = build_standard_autocross()

    # Reference lap time at very high resolution
    print("Computing reference lap time (200 000 segments)...")
    track_ref = resample_uniform(track_raw, 200000)
    _, lt_ref = solve_qss(track_ref, vehicle, refine=False, use_bicycle_model=True)
    print(f"  Reference: {lt_ref:.3f} s\n")

    n_points_list = [50, 75, 100, 150, 200, 300, 500, 750,
                     1000, 1500, 2000, 3000, 5000, 7500, 10000]

    results = {"uniform": {"lt": [], "rt": []},
               "adaptive": {"lt": [], "rt": []}}

    n_runs = 3

    print("Running comparison...")
    for n_pts in n_points_list:
        for method, resample_fn in [("uniform", resample_uniform),
                                     ("adaptive", lambda t, n: resample_adaptive(t, n))]:
            track = resample_fn(track_raw, n_pts)
            times = []
            lt = None
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _, lt = solve_qss(track, vehicle, refine=False, use_bicycle_model=True)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            avg_rt = np.mean(times)
            results[method]["lt"].append(lt)
            results[method]["rt"].append(avg_rt)

        err_u = abs(results["uniform"]["lt"][-1] - lt_ref) / lt_ref * 100
        err_a = abs(results["adaptive"]["lt"][-1] - lt_ref) / lt_ref * 100
        print(f"  N = {n_pts:5d}  |  Uniform: {results['uniform']['lt'][-1]:.2f} s "
              f"({err_u:.2f}%)  |  Adaptive: {results['adaptive']['lt'][-1]:.2f} s "
              f"({err_a:.2f}%)")

    return n_points_list, results, lt_ref, track_raw


def plot_study(n_points, results, lt_ref, track_raw):
    lt_u = np.array(results["uniform"]["lt"])
    lt_a = np.array(results["adaptive"]["lt"])
    rt_u = np.array(results["uniform"]["rt"])
    rt_a = np.array(results["adaptive"]["rt"])

    err_u = np.abs(lt_u - lt_ref) / lt_ref * 100
    err_a = np.abs(lt_a - lt_ref) / lt_ref * 100

    # --- Plot 1: Lap Time Error vs Segments ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(n_points, err_u, color="#D04040", linewidth=2, marker="o",
             markersize=5, label="Uniform spacing")
    ax1.plot(n_points, err_a, color="#2070B0", linewidth=2, marker="s",
             markersize=5, label="Adaptive spacing")

    ax1.set_xlabel("Number of Segments", fontsize=12)
    ax1.set_ylabel("Lap Time Error [%]", fontsize=12)
    ax1.set_title("Lap Time Error: Uniform vs Adaptive Discretisation",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xlim(n_points[0] * 0.8, n_points[-1] * 1.2)
    ax1.set_ylim(bottom=0)

    fig1.tight_layout()
    path1 = os.path.join(SAVE_DIR, "adaptive_vs_uniform_error.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved: {path1}")

    # --- Plot 2: Lap Time Error vs Runtime ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(rt_u * 1000, err_u, color="#D04040", linewidth=2, marker="o",
             markersize=5, label="Uniform spacing")
    ax2.plot(rt_a * 1000, err_a, color="#2070B0", linewidth=2, marker="s",
             markersize=5, label="Adaptive spacing")

    ax2.set_xlabel("Runtime [ms]", fontsize=12)
    ax2.set_ylabel("Lap Time Error [%]", fontsize=12)
    ax2.set_title("Accuracy vs Runtime: Uniform vs Adaptive Discretisation",
                  fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_ylim(bottom=0)

    fig2.tight_layout()
    path2 = os.path.join(SAVE_DIR, "adaptive_vs_uniform_efficiency.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")

    # --- Plot 3: Segment distribution visualisation ---
    n_demo = 300
    track_uni = resample_uniform(track_raw, n_demo)
    track_adp = resample_adaptive(track_raw, n_demo)

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

    # Uniform
    ax3a.plot(track_uni.x, track_uni.y, color="#888888", linewidth=1, zorder=1)
    ax3a.scatter(track_uni.x, track_uni.y, s=8, color="#D04040", zorder=2)
    ax3a.set_aspect("equal")
    ax3a.set_title(f"Uniform ({n_demo} segments)", fontsize=13, fontweight="bold")
    ax3a.set_xlabel("x [m]", fontsize=11)
    ax3a.set_ylabel("y [m]", fontsize=11)
    ax3a.grid(True, alpha=0.3)

    # Adaptive
    ax3b.plot(track_adp.x, track_adp.y, color="#888888", linewidth=1, zorder=1)
    ax3b.scatter(track_adp.x, track_adp.y, s=8, color="#2070B0", zorder=2)
    ax3b.set_aspect("equal")
    ax3b.set_title(f"Adaptive ({n_demo} segments)", fontsize=13, fontweight="bold")
    ax3b.set_xlabel("x [m]", fontsize=11)
    ax3b.set_ylabel("y [m]", fontsize=11)
    ax3b.grid(True, alpha=0.3)

    fig3.tight_layout()
    path3 = os.path.join(SAVE_DIR, "adaptive_vs_uniform_segments.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {path3}")


if __name__ == "__main__":
    print("Adaptive vs Uniform Discretisation Study")
    print("=" * 50)
    n_points, results, lt_ref, track_raw = run_study()
    plot_study(n_points, results, lt_ref, track_raw)
