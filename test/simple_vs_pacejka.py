"""
Simple vs Pacejka A/B Comparison.

Runs the same vehicle on the same track with both tyre models and
produces overlaid plots to highlight where and why they differ.

Outputs:
1. Lap time comparison table
2. Speed vs distance overlay
3. Delta speed (v_pacejka - v_simple) vs distance
4. Peak cornering speed comparison at each corner

Usage:
    python test/simple_vs_pacejka.py
    python test/simple_vs_pacejka.py --save test/simple_vs_pacejka.png
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import (
    VehicleParams, AeroParams,
    EVPowertrainParams, VehicleGeometry, build_tyre_from_config,
)
from solver.qss_speed import solve_qss
from events.autocross_generator import build_standard_autocross


def build_vehicles(config: dict) -> tuple:
    """Build Simple and Pacejka vehicles from the same base config.

    Returns:
        (vehicle_simple, vehicle_pacejka)
    """
    aero = AeroParams(
        rho=config["aero"]["rho"], Cd=config["aero"]["Cd"],
        Cl=config["aero"]["Cl"], A=config["aero"]["A"],
    )
    geo = VehicleGeometry(**config["geometry"])

    pt_cfg = config["powertrain"]
    pt = EVPowertrainParams(
        drivetrain=pt_cfg["drivetrain"],
        motor_power_kW=pt_cfg["motor_power_kW"],
        motor_torque_Nm=pt_cfg["motor_torque_Nm"],
        motor_rpm_max=pt_cfg["motor_rpm_max"],
        motor_efficiency=pt_cfg["motor_efficiency"],
        gear_ratio=pt_cfg["gear_ratio"],
        wheel_radius_m=pt_cfg["wheel_radius_m"],
        motor_weight_kg=pt_cfg.get("motor_weight_kg", 9.4),
        inverter_weight_kg=pt_cfg.get("inverter_weight_kg", 6.9),
        powertrain_overhead_kg=pt_cfg.get("powertrain_overhead_kg", 10.0),
        inverter_peak_power_kW=pt_cfg.get("inverter_peak_power_kW", 320.0),
        inverter_peak_current_A=pt_cfg.get("inverter_peak_current_A", 600.0),
    )

    common = dict(
        m=config["vehicle"]["mass_kg"],
        g=config["vehicle"]["g"],
        aero=aero,
        powertrain=pt,
        geometry=geo,
        Crr=config["vehicle"]["Crr"],
    )

    # Simple: constant-mu with bicycle model (same solver path as Pacejka)
    tyre_cfg = config["tyre"]
    simple_cfg = dict(tyre_cfg)
    simple_cfg["model"] = "simple"
    simple_tyre = build_tyre_from_config(simple_cfg)
    v_simple = VehicleParams(tyre=simple_tyre, **common)

    # Pacejka
    pac_cfg = dict(tyre_cfg)
    pac_cfg["model"] = "pacejka"
    pac_tyre = build_tyre_from_config(pac_cfg)
    v_pac = VehicleParams(tyre=pac_tyre, **common)

    return v_simple, v_pac


def find_corners(kappa: np.ndarray, s: np.ndarray,
                 threshold: float = 0.01) -> list:
    """Find corner apex indices where |kappa| exceeds threshold.

    Returns list of (index, s_position, kappa_value) for each apex.
    """
    in_corner = np.abs(kappa) > threshold
    corners = []

    i = 0
    while i < len(kappa):
        if in_corner[i]:
            # Find the apex (max |kappa|) within this corner
            start = i
            while i < len(kappa) and in_corner[i]:
                i += 1
            end = i
            segment = np.abs(kappa[start:end])
            apex_local = np.argmax(segment)
            apex = start + apex_local
            corners.append((apex, s[apex], kappa[apex]))
        else:
            i += 1

    return corners


def run_comparison(config: dict, save_path: str = None):
    """Run the A/B comparison and produce plots."""
    # Build track and vehicles
    track, metadata = build_standard_autocross()
    v_simple, v_pac = build_vehicles(config)

    # Solve
    result_s, lap_s = solve_qss(track, v_simple)
    result_p, lap_p = solve_qss(track, v_pac)

    s = track.s
    v_s = result_s["v"]
    v_p = result_p["v"]
    delta_v = v_p - v_s

    # Find corners
    corners = find_corners(track.kappa, s)

    # ---- Print summary ----
    print("=" * 60)
    print("SIMPLE vs PACEJKA — A/B COMPARISON")
    print("=" * 60)
    print(f"Track length:  {s[-1]:.1f} m")
    print(f"Vehicle mass:  {config['vehicle']['mass_kg']} kg")
    print(f"Simple mu:     {config['tyre']['mu']}")
    pac = config["tyre"].get("pacejka", {})
    print(f"Pacejka mu_peak: {pac.get('mu_peak', '?')}")
    print()
    print(f"{'Model':<12} {'Lap Time':>10} {'V_max':>10} {'V_min':>10}")
    print("-" * 44)
    print(f"{'Simple':<12} {lap_s:>9.3f}s {np.max(v_s):>9.2f} {np.min(v_s[v_s > 0]):>9.2f}")
    print(f"{'Pacejka':<12} {lap_p:>9.3f}s {np.max(v_p):>9.2f} {np.min(v_p[v_p > 0]):>9.2f}")
    print(f"{'Delta':<12} {lap_p - lap_s:>+9.3f}s")
    print()

    # Corner-by-corner comparison
    if corners:
        print(f"{'Corner':>6} {'Distance':>10} {'V_simple':>10} {'V_pacejka':>10} {'Delta':>8}")
        print("-" * 48)
        for idx, (apex, s_pos, kap) in enumerate(corners):
            dv = v_p[apex] - v_s[apex]
            print(f"  {idx + 1:>4} {s_pos:>9.1f}m {v_s[apex]:>9.2f} {v_p[apex]:>10.2f} {dv:>+7.2f}")
        print()

    # ---- Plots ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle("Simple vs Pacejka — A/B Comparison", fontsize=14, fontweight="bold")

    # Plot 1: Speed vs distance overlay
    ax = axes[0]
    ax.plot(s, v_s * 3.6, color="#2196F3", linewidth=1.5, label=f"Simple (mu={config['tyre']['mu']})")
    ax.plot(s, v_p * 3.6, color="#F44336", linewidth=1.5, label=f"Pacejka (mu_peak={pac.get('mu_peak', '?')})")
    ax.set_ylabel("Speed [km/h]")
    ax.set_title(f"Speed Profile — Simple: {lap_s:.3f}s | Pacejka: {lap_p:.3f}s | Delta: {lap_p - lap_s:+.3f}s")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Delta speed
    ax = axes[1]
    pos = np.where(delta_v >= 0, delta_v * 3.6, 0)
    neg = np.where(delta_v < 0, delta_v * 3.6, 0)
    ax.fill_between(s, pos, 0, color="#4CAF50", alpha=0.6, label="Pacejka faster")
    ax.fill_between(s, neg, 0, color="#F44336", alpha=0.6, label="Simple faster")
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.set_ylabel("Delta Speed [km/h]")
    ax.set_title("Speed Difference (Pacejka - Simple)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Curvature (to correlate with speed differences)
    ax = axes[2]
    with np.errstate(divide="ignore"):
        radius = np.where(np.abs(track.kappa) > 1e-4, 1.0 / np.abs(track.kappa), 500)
    radius = np.clip(radius, 0, 500)
    ax.fill_between(s, radius, 500, color="#9E9E9E", alpha=0.3)
    ax.plot(s, radius, color="#616161", linewidth=0.8)
    ax.set_ylabel("Corner Radius [m]")
    ax.set_title("Track Curvature (lower = tighter corner)")
    ax.set_ylim(0, 100)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # Mark corner apices
    for idx, (apex, s_pos, kap) in enumerate(corners):
        r = 1.0 / abs(kap) if abs(kap) > 1e-4 else 500
        if r < 100:
            ax.annotate(
                f"{idx + 1}", (s_pos, min(r, 100)),
                fontsize=7, ha="center", va="bottom", color="#D32F2F",
            )

    # Plot 4: Lateral limit comparison
    ax = axes[3]
    ax.plot(s, result_s["v_lat"] * 3.6, color="#2196F3", linewidth=1, alpha=0.7, label="Simple v_lat")
    ax.plot(s, result_p["v_lat"] * 3.6, color="#F44336", linewidth=1, alpha=0.7, label="Pacejka v_lat")
    ax.set_ylabel("Lateral Speed Limit [km/h]")
    ax.set_xlabel("Distance [m]")
    ax.set_title("Cornering Speed Limit (grip-limited ceiling)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Cap y-axis for readability (straights have very high v_lat)
    v_lat_cap = max(np.max(v_s), np.max(v_p)) * 3.6 * 1.3
    ax.set_ylim(0, min(v_lat_cap, 200))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple vs Pacejka A/B comparison")
    parser.add_argument(
        "--save", default=None,
        help="Save path for the figure. Shows interactively if omitted.",
    )
    args = parser.parse_args()

    # Load default config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "default.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_comparison(config, save_path=args.save)
