"""
Pacejka Magic Formula Tyre Model Verification.

Produces five individual verification plots, each validating a specific
aspect of the implementation against known analytical properties of the
Magic Formula.

Usage:
    python test/pacejka_verification.py
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import PacejkaCoefficients, PacejkaParams
from physics.tyre import (
    _pacejka_BCDE, pacejka_Fy0, pacejka_Fx0,
    pacejka_combined_Fx, pacejka_combined_Fy,
)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_tyre():
    """Load tyre preset from config/tyres.yaml."""
    tyres_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "tyres.yaml",
    )
    with open(tyres_path, "r") as f:
        db = yaml.safe_load(f)

    t = db["tyres"]["hoosier_r25b_18x6"]
    lat = PacejkaCoefficients(**t["lateral"])
    lon = PacejkaCoefficients(**t["longitudinal"])
    pac = PacejkaParams(
        lateral=lat, longitudinal=lon,
        **t["combined"],
        mu_peak=t["mu_peak"],
        Fz_nominal=t["Fz_nominal"],
        C_alpha_f=t["C_alpha_f"],
        C_alpha_r=t["C_alpha_r"],
    )
    return pac, lat, lon


def plot_lateral_force(lat, loads, colours):
    """Fig 1: Lateral force vs slip angle at multiple vertical loads."""
    alphas_deg = np.linspace(-15, 15, 500)
    alphas_rad = np.deg2rad(alphas_deg)

    fig, ax = plt.subplots(figsize=(8, 5))
    for Fz, c in zip(loads, colours):
        Fy = [pacejka_Fy0(a, Fz, lat) for a in alphas_rad]
        ax.plot(alphas_deg, Fy, linewidth=2, color=c, label=f"Fz = {Fz} N")

    ax.set_xlabel("Slip Angle [deg]", fontsize=12)
    ax.set_ylabel("Lateral Force Fy [N]", fontsize=12)
    ax.set_title("Lateral Force vs Slip Angle", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "pacejka_fig1_lateral_force.png"), dpi=150)
    plt.close(fig)
    print("  Fig 1 saved: pacejka_fig1_lateral_force.png")


def plot_longitudinal_force(lon, loads, colours):
    """Fig 2: Longitudinal force vs slip ratio at multiple vertical loads."""
    kappas = np.linspace(-0.3, 0.3, 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    for Fz, c in zip(loads, colours):
        Fx = [pacejka_Fx0(k, Fz, lon) for k in kappas]
        ax.plot(kappas * 100, Fx, linewidth=2, color=c, label=f"Fz = {Fz} N")

    ax.set_xlabel("Slip Ratio [%]", fontsize=12)
    ax.set_ylabel("Longitudinal Force Fx [N]", fontsize=12)
    ax.set_title("Longitudinal Force vs Slip Ratio", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "pacejka_fig2_longitudinal_force.png"), dpi=150)
    plt.close(fig)
    print("  Fig 2 saved: pacejka_fig2_longitudinal_force.png")


def plot_load_sensitivity(lat):
    """Fig 3: Peak force and effective friction coefficient vs vertical load."""
    Fz_range = np.linspace(100, 3000, 200)

    # Analytical peak: D = (a1*Fz + a2)*Fz
    D_analytical = np.array([(lat.a1 * Fz + lat.a2) * Fz for Fz in Fz_range])
    mu_eff = np.array([lat.a1 * Fz + lat.a2 for Fz in Fz_range])

    # Numerical peak: sweep slip angle and find max force
    D_numerical = []
    for Fz in Fz_range:
        alphas = np.linspace(0, 0.5, 500)
        forces = [abs(pacejka_Fy0(a, Fz, lat)) for a in alphas]
        D_numerical.append(max(forces))
    D_numerical = np.array(D_numerical)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    colour_force = "#2070B0"
    colour_mu = "#D04040"

    ax1.plot(Fz_range, D_analytical, linewidth=2, color=colour_force,
             label="Analytical D = (a1 Fz + a2) Fz")
    ax1.plot(Fz_range, D_numerical, linewidth=2.5, linestyle=":",
             color="black", label="Numerical peak (sweep)")
    ax1.set_xlabel("Vertical Load Fz [N]", fontsize=12)
    ax1.set_ylabel("Peak Lateral Force [N]", fontsize=12, color=colour_force)
    ax1.tick_params(axis="y", labelcolor=colour_force)

    ax2 = ax1.twinx()
    ax2.plot(Fz_range, mu_eff, linewidth=2, color=colour_mu,
             label="Effective mu = a1 Fz + a2")
    ax2.set_ylabel("Effective Friction Coefficient", fontsize=12, color=colour_mu)
    ax2.tick_params(axis="y", labelcolor=colour_mu)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center left")
    ax1.set_title("Load Sensitivity: Peak Force and Effective Mu",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "pacejka_fig3_load_sensitivity.png"), dpi=150)
    plt.close(fig)

    max_err = np.max(np.abs(D_analytical - D_numerical))
    print(f"  Fig 3 saved: pacejka_fig3_load_sensitivity.png  "
          f"(analytical vs numerical peak error: {max_err:.1f} N)")


def plot_friction_ellipse(pac, lat, lon, loads, colours):
    """Fig 4: Combined slip friction ellipse at multiple loads."""
    fig, ax = plt.subplots(figsize=(7, 7))

    for Fz, c in zip(loads, colours):
        Fx_pts = []
        Fy_pts = []
        for a in np.linspace(0, 0.2, 120):
            Fy_val = abs(pacejka_Fy0(a, Fz, lat))
            best_Fx = 0
            for k in np.linspace(0, 0.3, 60):
                Fx_val = pacejka_combined_Fx(k, a, Fz, pac)
                if Fx_val > best_Fx:
                    best_Fx = Fx_val
            Fx_pts.append(best_Fx)
            Fy_pts.append(Fy_val)

        Fx_arr = np.array(Fx_pts)
        Fy_arr = np.array(Fy_pts)
        # Draw all four quadrants
        ax.plot(Fy_arr, Fx_arr, color=c, linewidth=2, label=f"Fz = {Fz} N")
        ax.plot(-Fy_arr, Fx_arr, color=c, linewidth=2)
        ax.plot(Fy_arr, -Fx_arr, color=c, linewidth=2)
        ax.plot(-Fy_arr, -Fx_arr, color=c, linewidth=2)

    ax.set_xlabel("Lateral Force Fy [N]", fontsize=12)
    ax.set_ylabel("Longitudinal Force Fx [N]", fontsize=12)
    ax.set_title("Combined Slip Friction Ellipse", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "pacejka_fig4_friction_ellipse.png"), dpi=150)
    plt.close(fig)
    print("  Fig 4 saved: pacejka_fig4_friction_ellipse.png")


def plot_cornering_stiffness(lat):
    """Fig 5: Cornering stiffness vs vertical load."""
    Fz_range = np.linspace(100, 3000, 200)

    # Analytical: BCD = a3 * sin(a4 * atan(a5 * Fz))
    BCD_analytical = np.array([
        lat.a3 * math.sin(lat.a4 * math.atan(lat.a5 * Fz))
        for Fz in Fz_range
    ])

    # Numerical: finite difference dFy/dalpha at alpha = 0
    da = 1e-5
    BCD_numerical = np.array([
        abs(pacejka_Fy0(da, Fz, lat)) / da
        for Fz in Fz_range
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Fz_range, BCD_analytical, linewidth=2, color="#2070B0",
            label="Analytical: a3 sin(a4 atan(a5 Fz))")
    ax.plot(Fz_range, BCD_numerical, linewidth=2.5, linestyle=":",
            color="black", label="Numerical: dFy/d(alpha) at alpha = 0")
    ax.set_xlabel("Vertical Load Fz [N]", fontsize=12)
    ax.set_ylabel("Cornering Stiffness [N/rad]", fontsize=12)
    ax.set_title("Cornering Stiffness vs Vertical Load",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "pacejka_fig5_cornering_stiffness.png"), dpi=150)
    plt.close(fig)

    max_err = np.max(np.abs(BCD_analytical - BCD_numerical))
    pct_err = max_err / np.max(BCD_analytical) * 100
    print(f"  Fig 5 saved: pacejka_fig5_cornering_stiffness.png  "
          f"(analytical vs numerical error: {pct_err:.2f}%)")


if __name__ == "__main__":
    pac, lat, lon = load_tyre()
    loads = [500, 1000, 1500, 2000, 2500]
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

    print("Pacejka Magic Formula Verification")
    print("=" * 50)
    plot_lateral_force(lat, loads, colours)
    plot_longitudinal_force(lon, loads, colours)
    plot_load_sensitivity(lat)
    plot_friction_ellipse(pac, lat, lon, loads, colours)
    plot_cornering_stiffness(lat)
    print("=" * 50)
    print("All verification plots generated.")
