"""
Skidpad track layout (full figure-of-8) with performance limit analysis.

Shows both circles, entry/exit lanes, track edges, and confirms 100%
traction-limited performance (constant curvature => lateral grip is
the sole constraint).

Usage:
    python test/skidpad_layout.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import (
    VehicleParams, EVPowertrainParams, VehicleGeometry, BatteryParams,
    build_tyre_from_config, build_aero_from_config,
)
from events.skidpad import (
    SKIDPAD_CENTRE_RADIUS, SKIDPAD_INNER_RADIUS,
    SKIDPAD_OUTER_RADIUS, TRACK_WIDTH, build_skidpad_track,
)
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
    bat = config["battery"]
    battery = BatteryParams(
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

    return VehicleParams(
        m=config["vehicle"]["mass_kg"], g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"], aero=aero, tyre=tyre,
        powertrain=powertrain, geometry=geometry, battery=battery,
    )


def plot_skidpad_layout():
    vehicle = build_vehicle()

    # Solve on a single circle (the timed portion)
    track = build_skidpad_track(n_points=200)
    result, lap_time = solve_qss(track, vehicle, use_bicycle_model=True)
    v = result["v"]
    v_lat = result["v_lat"]

    traction_pct = np.mean(v / v_lat) * 100.0
    v_mean = np.mean(v)
    v_kmh = v_mean * 3.6
    a_lat = v_mean**2 / SKIDPAD_CENTRE_RADIUS
    a_lat_g = a_lat / 9.81

    # --- Figure ---
    fig, (ax_track, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 7),
        gridspec_kw={"width_ratios": [3.5, 1], "wspace": 0.25},
    )

    theta = np.linspace(0, 2 * np.pi, 300)

    # Circle centres (FS layout: two circles side-by-side)
    left_cx, left_cy = 0.0, 0.0
    right_cx, right_cy = 2 * SKIDPAD_CENTRE_RADIUS, 0.0

    # --- Draw track surface (shaded annuli) ---
    for cx, cy in [(left_cx, left_cy), (right_cx, right_cy)]:
        x_out = cx + SKIDPAD_OUTER_RADIUS * np.cos(theta)
        y_out = cy + SKIDPAD_OUTER_RADIUS * np.sin(theta)
        x_inn = cx + SKIDPAD_INNER_RADIUS * np.cos(theta)
        y_inn = cy + SKIDPAD_INNER_RADIUS * np.sin(theta)
        ax_track.fill(
            np.concatenate([x_out, x_inn[::-1]]),
            np.concatenate([y_out, y_inn[::-1]]),
            color="#D0D0D0", alpha=0.45,
        )

    # Entry/exit lane (vertical strip between circles)
    lane_x_left = SKIDPAD_INNER_RADIUS
    lane_x_right = SKIDPAD_OUTER_RADIUS
    lane_y_lo = -SKIDPAD_OUTER_RADIUS - 3.0
    lane_y_hi = SKIDPAD_OUTER_RADIUS + 3.0
    ax_track.fill(
        [lane_x_left, lane_x_right, lane_x_right, lane_x_left],
        [lane_y_lo, lane_y_lo, lane_y_hi, lane_y_hi],
        color="#D0D0D0", alpha=0.45,
    )

    # --- Track edges ---
    for cx, cy, lbl in [(left_cx, left_cy, "Track edges"),
                         (right_cx, right_cy, None)]:
        x_inn = cx + SKIDPAD_INNER_RADIUS * np.cos(theta)
        y_inn = cy + SKIDPAD_INNER_RADIUS * np.sin(theta)
        x_out = cx + SKIDPAD_OUTER_RADIUS * np.cos(theta)
        y_out = cy + SKIDPAD_OUTER_RADIUS * np.sin(theta)
        ax_track.plot(x_inn, y_inn, color="black", lw=2,
                      label=lbl if lbl else None)
        ax_track.plot(x_out, y_out, color="black", lw=2)

    # Lane edges
    ax_track.plot([lane_x_left, lane_x_left], [lane_y_lo, lane_y_hi], "k-", lw=2)
    ax_track.plot([lane_x_right, lane_x_right], [lane_y_lo, lane_y_hi], "k-", lw=2)

    # --- Centre lines (driving line) ---
    for cx, cy in [(left_cx, left_cy), (right_cx, right_cy)]:
        x_cl = cx + SKIDPAD_CENTRE_RADIUS * np.cos(theta)
        y_cl = cy + SKIDPAD_CENTRE_RADIUS * np.sin(theta)
        ax_track.plot(x_cl, y_cl, color="#2070B0", lw=2, linestyle="--", alpha=0.7)
    # Lane centre line
    ax_track.plot([SKIDPAD_CENTRE_RADIUS, SKIDPAD_CENTRE_RADIUS],
                  [lane_y_lo, lane_y_hi], color="#2070B0", lw=1.5,
                  linestyle="--", alpha=0.7, label="Centre line")

    # --- Circle centres ---
    for cx, cy in [(left_cx, left_cy), (right_cx, right_cy)]:
        ax_track.plot(cx, cy, "+", color="#D04040", markersize=10, markeredgewidth=2)

    # --- Radius annotation on left circle ---
    ax_track.annotate(
        "", xy=(SKIDPAD_CENTRE_RADIUS - 0.3, left_cy),
        xytext=(left_cx, left_cy),
        arrowprops=dict(arrowstyle="<->", color="#D04040", lw=1.5),
    )
    ax_track.text(
        SKIDPAD_CENTRE_RADIUS / 2, 0.8,
        f"R = {SKIDPAD_CENTRE_RADIUS} m", fontsize=10,
        color="#D04040", ha="center",
    )

    # --- Entry / exit markers ---
    ax_track.annotate(
        "ENTRY", xy=(SKIDPAD_CENTRE_RADIUS, lane_y_lo),
        fontsize=10, fontweight="bold", color="green", ha="center",
        xytext=(SKIDPAD_CENTRE_RADIUS, lane_y_lo - 1.2),
    )
    ax_track.plot(SKIDPAD_CENTRE_RADIUS, lane_y_lo, "^",
                  color="green", markersize=10, zorder=5)
    ax_track.annotate(
        "EXIT", xy=(SKIDPAD_CENTRE_RADIUS, lane_y_hi),
        fontsize=10, fontweight="bold", color="red", ha="center",
        xytext=(SKIDPAD_CENTRE_RADIUS, lane_y_hi + 1.0),
    )
    ax_track.plot(SKIDPAD_CENTRE_RADIUS, lane_y_hi, "s",
                  color="red", markersize=10, zorder=5)

    # --- Info box ---
    info = (
        f"Lap time (1 circle): {lap_time:.2f} s\n"
        f"Speed: {v_kmh:.1f} km/h\n"
        f"Lat accel: {a_lat_g:.2f} g\n"
        f"Track width: {TRACK_WIDTH:.1f} m"
    )
    ax_track.text(
        0.02, 0.02, info, transform=ax_track.transAxes, fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax_track.set_aspect("equal")
    ax_track.set_xlabel("x [m]", fontsize=11)
    ax_track.set_ylabel("y [m]", fontsize=11)
    ax_track.set_title("Formula Student Skidpad Layout", fontsize=13, fontweight="bold")
    ax_track.legend(fontsize=9, loc="upper right")
    ax_track.grid(True, alpha=0.3)

    # ---- Right: performance-limit bar ----
    bar_labels = ["Traction\n(lateral grip)"]
    bar_vals = [traction_pct]

    bars = ax_bar.barh(bar_labels, bar_vals, height=0.45, color="#2070B0",
                       edgecolor="black", linewidth=0.8)
    ax_bar.barh(bar_labels, [100], height=0.45, color="none",
                edgecolor="black", linewidth=1.2)
    ax_bar.text(
        bar_vals[0] / 2, 0, f"{bar_vals[0]:.0f}%",
        ha="center", va="center", fontsize=14, fontweight="bold", color="white",
    )

    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Contribution to limit [%]", fontsize=11)
    ax_bar.set_title("Performance Limit", fontsize=13, fontweight="bold")
    ax_bar.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "skidpad_layout.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Lap time:      {lap_time:.2f} s")
    print(f"Avg speed:     {v_kmh:.1f} km/h")
    print(f"Lat accel:     {a_lat_g:.2f} g")
    print(f"Traction use:  {traction_pct:.1f}%")
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_skidpad_layout()
