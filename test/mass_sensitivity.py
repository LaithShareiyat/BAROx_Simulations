"""
Mass Sensitivity — Simple vs Pacejka.

Sweeps vehicle mass and plots lap time for both tyre models.
If the Pacejka curve is steeper at high mass, load sensitivity is
working correctly (heavier car pushes tyres into lower mu_eff region).

Usage:
    python test/mass_sensitivity.py
    python test/mass_sensitivity.py --save test/mass_sensitivity.png
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


def run_mass_sweep(config: dict, masses: np.ndarray, save_path: str = None):
    """Sweep mass for both tyre models and plot results."""
    track, _ = build_standard_autocross()

    # Build shared components once
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

    # Build tyre objects
    tyre_cfg = config["tyre"]

    simple_cfg = dict(tyre_cfg)
    simple_cfg["model"] = "simple"
    simple_tyre = build_tyre_from_config(simple_cfg)

    pac_cfg = dict(tyre_cfg)
    pac_cfg["model"] = "pacejka"
    pac_tyre = build_tyre_from_config(pac_cfg)

    laps_simple = []
    laps_pacejka = []

    print(f"{'Mass [kg]':>10} {'Simple [s]':>12} {'Pacejka [s]':>12} {'Delta [s]':>10} {'mu_eff':>8}")
    print("-" * 56)

    for m in masses:
        common = dict(
            m=float(m), g=config["vehicle"]["g"],
            aero=aero, powertrain=pt, geometry=geo,
            Crr=config["vehicle"]["Crr"],
        )

        v_s = VehicleParams(tyre=simple_tyre, **common)
        v_p = VehicleParams(tyre=pac_tyre, **common)

        _, lap_s = solve_qss(track, v_s)
        _, lap_p = solve_qss(track, v_p)

        laps_simple.append(lap_s)
        laps_pacejka.append(lap_p)

        # Effective mu at per-tyre static load
        Fz_per = float(m) * 9.81 / 4
        lat = pac_tyre.lateral
        mu_eff = lat.a1 * Fz_per + lat.a2

        print(f"{m:>10.0f} {lap_s:>11.3f}s {lap_p:>11.3f}s {lap_p - lap_s:>+9.3f}s {mu_eff:>7.3f}")

    laps_simple = np.array(laps_simple)
    laps_pacejka = np.array(laps_pacejka)
    delta = laps_pacejka - laps_simple

    # ---- Plots ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle("Mass Sensitivity", fontsize=14, fontweight="bold")

    # Plot 1: Lap time vs mass
    ax = axes[0]
    ax.plot(masses, laps_simple, "o-", color="#2196F3", linewidth=2, markersize=5,
            label=f"Simple (mu={config['tyre']['mu']})")
    ax.plot(masses, laps_pacejka, "s-", color="#F44336", linewidth=2, markersize=5,
            label=f"Pacejka (mu_peak={pac_cfg.get('pacejka', {}).get('mu_peak', '?')})")
    ax.set_ylabel("Lap Time [s]")
    ax.set_title("Lap Time vs Vehicle Mass")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Delta lap time (Pacejka - Simple)
    ax = axes[1]
    ax.plot(masses, delta, "D-", color="#9C27B0", linewidth=2, markersize=5)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.fill_between(masses, delta, 0,
                     where=(delta < 0), color="#4CAF50", alpha=0.3, label="Pacejka faster")
    ax.fill_between(masses, delta, 0,
                     where=(delta >= 0), color="#F44336", alpha=0.3, label="Simple faster")
    ax.set_ylabel("Delta Lap Time [s]")
    ax.set_title("Lap Time Difference (Pacejka - Simple)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Effective mu vs mass
    ax = axes[2]
    Fz_per_tyre = masses * 9.81 / 4
    lat = pac_tyre.lateral
    mu_eff = lat.a1 * Fz_per_tyre + lat.a2
    ax.plot(masses, mu_eff, "o-", color="#F44336", linewidth=2, markersize=5,
            label="Pacejka mu_eff (per-tyre)")
    ax.axhline(y=config["tyre"]["mu"], color="#2196F3", linewidth=2,
               linestyle="--", label=f"Simple mu = {config['tyre']['mu']}")
    ax.axhline(y=pac_tyre.mu_peak, color="grey", linewidth=1,
               linestyle=":", label=f"mu_peak = {pac_tyre.mu_peak}")

    # Mark where mu_eff crosses constant mu
    cross_Fz = (config["tyre"]["mu"] - lat.a2) / lat.a1
    cross_mass = cross_Fz * 4 / 9.81
    if masses[0] <= cross_mass <= masses[-1]:
        ax.axvline(x=cross_mass, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.annotate(f"Crossover: {cross_mass:.0f} kg",
                    xy=(cross_mass, config["tyre"]["mu"]),
                    xytext=(cross_mass + 20, config["tyre"]["mu"] + 0.03),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))

    ax.set_xlabel("Vehicle Mass [kg]")
    ax.set_ylabel("Effective Friction Coefficient [-]")
    ax.set_title("Pacejka Effective mu vs Mass (load sensitivity)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mass sensitivity: Simple vs Pacejka")
    parser.add_argument(
        "--save", default=None,
        help="Save path for the figure. Shows interactively if omitted.",
    )
    parser.add_argument(
        "--min-mass", type=float, default=180,
        help="Minimum mass [kg] (default: 180)",
    )
    parser.add_argument(
        "--max-mass", type=float, default=400,
        help="Maximum mass [kg] (default: 400)",
    )
    parser.add_argument(
        "--steps", type=int, default=12,
        help="Number of mass points (default: 12)",
    )
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "default.yaml",
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    masses = np.linspace(args.min_mass, args.max_mass, args.steps)
    run_mass_sweep(config, masses, save_path=args.save)
