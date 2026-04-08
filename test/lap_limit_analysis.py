"""
Autocross lap limit analysis.

Produces a speed vs distance plot colour-coded by the active performance
limit at each point: cornering, traction, power, or braking.

Usage:
    python test/lap_limit_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
from physics.aero import downforce
from physics.tyre import a_max_from_tyre, ax_available, ax_traction_axle_aware
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
    ), config


def classify_limits(track, vehicle, v, v_lat, v_fwd, v_bwd):
    """Classify each point by the active performance limit.

    Returns an array of integers:
        0 = cornering limited   (v set by lateral grip envelope)
        1 = traction limited    (accelerating, grip is the bottleneck)
        2 = power limited       (accelerating, motor power is the bottleneck)
        3 = braking limited     (decelerating into a corner)
    """
    n = len(v)
    limits = np.zeros(n, dtype=int)
    m = vehicle.m
    g = vehicle.g
    mu = vehicle.tyre.mu
    use_bicycle = vehicle.has_extended_powertrain

    for i in range(n):
        # Which envelope sets the speed?
        envelope_speeds = [v_lat[i], v_fwd[i], v_bwd[i]]
        active = int(np.argmin(envelope_speeds))

        if active == 0:
            # Cornering limited
            limits[i] = 0
        elif active == 2:
            # Braking limited
            limits[i] = 3
        else:
            # Forward pass is active — determine if traction or power limited
            v_i = v[i]
            Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v_i)
            ay = v_i ** 2 * abs(track.kappa[i])

            if use_bicycle:
                ax_grip = ax_traction_axle_aware(
                    mu, vehicle, v_i, abs(track.kappa[i]), Fdown, 0.0,
                )
                Fx_grip = ax_grip * m
            else:
                amax = a_max_from_tyre(vehicle.tyre, g, m, Fdown)
                Fx_grip = ax_available(amax, ay) * m

            Fx_motor = max_tractive_force_extended(vehicle.powertrain, v_i)

            if Fx_motor < Fx_grip:
                limits[i] = 2  # Power limited
            else:
                limits[i] = 1  # Traction limited

    return limits


def plot_limit_analysis():
    vehicle, config = build_vehicle()
    track, _ = build_standard_autocross()

    result, lap_time = solve_qss(
        track, vehicle, refine=True, min_points=500, use_bicycle_model=True,
    )

    v = result["v"]
    v_lat = result["v_lat"]
    v_fwd = result["v_fwd"]
    v_bwd = result["v_bwd"]
    s = track.s

    limits = classify_limits(track, vehicle, v, v_lat, v_fwd, v_bwd)

    # Colours and labels
    colours = {
        0: "#E8A020",   # cornering: amber
        1: "#30A050",   # traction: green
        2: "#2070B0",   # power: blue
        3: "#D04040",   # braking: red
    }
    labels = {
        0: "Cornering limited",
        1: "Traction limited",
        2: "Power limited",
        3: "Braking",
    }

    v_kmh = v * 3.6

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # Top: speed profile coloured by limit
    for i in range(len(s) - 1):
        ax1.plot(
            s[i:i + 2], v_kmh[i:i + 2],
            color=colours[limits[i]], linewidth=1.5, solid_capstyle="round",
        )

    # Legend
    patches = [
        mpatches.Patch(color=colours[k], label=labels[k])
        for k in sorted(colours.keys())
    ]
    ax1.legend(handles=patches, fontsize=10, loc="upper right")
    ax1.set_ylabel("Speed [km/h]", fontsize=12)
    ax1.set_title(
        f"Autocross Lap Limit Analysis (lap time: {lap_time:.3f}s)",
        fontsize=14, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Bottom: limit region bar
    for i in range(len(s) - 1):
        ax2.axvspan(s[i], s[i + 1], color=colours[limits[i]], alpha=0.7)

    ax2.set_yticks([])
    ax2.set_xlabel("Distance [m]", fontsize=12)
    ax2.set_ylabel("Active\nLimit", fontsize=10, rotation=0, labelpad=40, va="center")
    ax2.set_xlim(s[0], s[-1])

    # Summary statistics
    n_total = len(limits)
    print()
    for k in sorted(colours.keys()):
        pct = np.sum(limits == k) / n_total * 100
        print(f"  {labels[k]:30s}: {pct:5.1f}% of lap")

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "lap_limit_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")


def plot_limit_analysis_simple():
    """Simplified version: 3 categories (traction, power, braking) as
    shaded regions behind the speed trace, no lap time in title."""
    vehicle, config = build_vehicle()
    track, _ = build_standard_autocross()

    result, lap_time = solve_qss(
        track, vehicle, refine=True, min_points=500, use_bicycle_model=True,
    )

    v = result["v"]
    v_lat = result["v_lat"]
    v_fwd = result["v_fwd"]
    v_bwd = result["v_bwd"]
    s = track.s

    limits = classify_limits(track, vehicle, v, v_lat, v_fwd, v_bwd)

    # Merge cornering (0) and traction (1) into "Acceleration constraint"
    # 0 = acceleration constraint (was cornering + traction)
    # 1 = power constraint (was power)
    # 2 = braking constraint (was braking)
    simple_limits = np.zeros(len(limits), dtype=int)
    for i in range(len(limits)):
        if limits[i] in (0, 1):
            simple_limits[i] = 0  # Acceleration constraint
        elif limits[i] == 2:
            simple_limits[i] = 1  # Power constraint
        else:
            simple_limits[i] = 2  # Braking constraint

    colours = {
        0: "#30A050",   # acceleration: green
        1: "#2070B0",   # power: blue
        2: "#D04040",   # braking: red
    }
    labels = {
        0: "Acceleration constraint",
        1: "Power constraint",
        2: "Braking constraint",
    }

    v_kmh = v * 3.6

    # Determine which solver pass sets the speed at each point
    # 0 = lateral (cornering), 1 = forward pass, 2 = backward pass
    solver_pass = np.zeros(len(v), dtype=int)
    for i in range(len(v)):
        candidates = [v_lat[i], v_fwd[i], v_bwd[i]]
        solver_pass[i] = int(np.argmin(candidates))

    pass_colours = {
        0: "#9B59B6",   # lateral limit: purple
        1: "#E8A020",   # forward pass: amber
        2: "#2C3E50",   # backward pass: dark grey
    }
    pass_labels = {
        0: "Lateral limit",
        1: "Forward pass",
        2: "Backward pass",
    }

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [5, 1, 1], "hspace": 0.08},
    )

    # Top: speed profile coloured by constraint
    for i in range(len(s) - 1):
        ax1.plot(
            s[i:i + 2], v_kmh[i:i + 2],
            color=colours[simple_limits[i]], linewidth=1.5, solid_capstyle="round",
        )

    # Legend
    patches = [
        mpatches.Patch(color=colours[k], label=labels[k])
        for k in sorted(colours.keys())
    ]
    ax1.legend(handles=patches, fontsize=10, loc="upper right")
    ax1.set_ylabel("Speed [km/h]", fontsize=12)
    ax1.set_title("Velocity Profile with Performance Constraints",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Middle: constraint region bar
    for i in range(len(s) - 1):
        ax2.axvspan(s[i], s[i + 1], color=colours[simple_limits[i]], alpha=0.7)

    ax2.set_yticks([])
    ax2.set_ylabel("Active\nConstraint", fontsize=10, rotation=0, labelpad=50, va="center")
    ax2.set_xlim(s[0], s[-1])

    # Bottom: solver pass bar
    for i in range(len(s) - 1):
        ax3.axvspan(s[i], s[i + 1], color=pass_colours[solver_pass[i]], alpha=0.7)

    ax3.set_yticks([])
    ax3.set_xlabel("Distance [m]", fontsize=12)
    ax3.set_ylabel("Solver\nPass", fontsize=10, rotation=0, labelpad=50, va="center")
    ax3.set_xlim(s[0], s[-1])

    # Add solver pass legend to the bottom bar
    pass_patches = [
        mpatches.Patch(color=pass_colours[k], label=pass_labels[k])
        for k in sorted(pass_colours.keys())
    ]
    ax3.legend(handles=pass_patches, fontsize=9, loc="upper right", ncol=3)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "lap_limit_analysis_simple.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary
    n_total = len(simple_limits)
    print()
    for k in sorted(colours.keys()):
        pct = np.sum(simple_limits == k) / n_total * 100
        print(f"  {labels[k]:30s}: {pct:5.1f}% of lap")
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    print("Lap Limit Analysis")
    print("=" * 50)
    plot_limit_analysis()
    plot_limit_analysis_simple()
