"""
Pacejka Magic Formula verification plots.

Generates 6 plots to visually verify the tyre model:
1. Pure lateral force (Fy) vs slip angle at multiple loads
2. Pure longitudinal force (Fx) vs slip ratio at multiple loads
3. Load sensitivity — effective mu vs vertical load
4. Combined slip — Fx reduction with increasing slip angle
5. Combined slip — Fy reduction with increasing slip ratio
6. Friction ellipse — combined slip force envelope

Usage:
    python scripts/pacejka_verification.py
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vehicle import PacejkaCoefficients, PacejkaParams
from physics.tyre import pacejka_Fy0, pacejka_Fx0, pacejka_combined_Fx, pacejka_combined_Fy


def load_tyre_preset(preset_id: str = "hoosier_r25b_18x6") -> tuple:
    """Load a tyre preset from the database.

    Returns:
        (PacejkaParams, preset_dict)
    """
    tyres_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "tyres.yaml",
    )
    with open(tyres_path, "r") as f:
        db = yaml.safe_load(f)

    t = db["tyres"][preset_id]
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
    return pac, t


def plot_verification(preset_id: str = "hoosier_r25b_18x6",
                      save_path: str = None):
    """Generate the 6-panel Pacejka verification figure."""
    pac, t = load_tyre_preset(preset_id)
    lat = pac.lateral
    lon = pac.longitudinal

    loads = [400, 800, 1200, 1500]
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Pacejka Validation", fontsize=14, fontweight="bold"
    )

    # ---- Plot 1: Fy vs alpha ----
    ax = axes[0, 0]
    alpha = np.linspace(-0.25, 0.25, 500)
    for Fz, c in zip(loads, colours):
        Fy = [pacejka_Fy0(a, Fz, lat) for a in alpha]
        ax.plot(np.degrees(alpha), Fy, color=c, linewidth=2, label=f"Fz = {Fz} N")
    ax.set_xlabel("Slip Angle [deg]")
    ax.set_ylabel("Lateral Force Fy [N]")
    ax.set_title("Pure Lateral Force vs Slip Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

    # ---- Plot 2: Fx vs kappa ----
    ax = axes[0, 1]
    kappa = np.linspace(-0.3, 0.3, 500)
    for Fz, c in zip(loads, colours):
        Fx = [pacejka_Fx0(k, Fz, lon) for k in kappa]
        ax.plot(kappa, Fx, color=c, linewidth=2, label=f"Fz = {Fz} N")
    ax.set_xlabel("Slip Ratio [-]")
    ax.set_ylabel("Longitudinal Force Fx [N]")
    ax.set_title("Pure Longitudinal Force vs Slip Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

    # ---- Plot 3: Effective mu vs Fz (load sensitivity) ----
    ax = axes[0, 2]
    Fz_range = np.linspace(100, 3000, 200)
    mu_lat = [(lat.a1 * Fz + lat.a2) for Fz in Fz_range]
    mu_lon = [(lon.a1 * Fz + lon.a2) for Fz in Fz_range]
    ax.plot(Fz_range, mu_lat, color="#2196F3", linewidth=2, label="Lateral mu_eff")
    ax.plot(Fz_range, mu_lon, color="#F44336", linewidth=2, label="Longitudinal mu_eff")
    ax.axhline(
        y=t["mu_peak"], color="grey", linestyle="--", linewidth=1,
        label=f"mu_peak = {t['mu_peak']}",
    )
    ax.axvline(
        x=t["Fz_nominal"], color="grey", linestyle=":", linewidth=1,
        label=f"Fz_nom = {t['Fz_nominal']} N",
    )
    ax.axvspan(400, 900, alpha=0.1, color="green", label="Typical FSAE per-tyre Fz")
    ax.set_xlabel("Vertical Load Fz [N]")
    ax.set_ylabel("Effective Friction Coefficient [-]")
    ax.set_title("Load Sensitivity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Plot 4: Combined slip — Fx vs kappa at different alpha ----
    ax = axes[1, 0]
    kappa_pos = np.linspace(0, 0.3, 300)
    Fz_test = 1500
    alphas = [0, 0.02, 0.05, 0.1, 0.15]
    alpha_colours = ["#1a237e", "#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for a, c in zip(alphas, alpha_colours):
        Fx = [pacejka_combined_Fx(k, a, Fz_test, pac) for k in kappa_pos]
        ax.plot(
            kappa_pos, Fx, color=c, linewidth=2,
            label=f"alpha = {np.degrees(a):.1f} deg",
        )
    ax.set_xlabel("Slip Ratio [-]")
    ax.set_ylabel("Longitudinal Force Fx [N]")
    ax.set_title(f"Combined Slip: Fx (Fz = {Fz_test} N)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Plot 5: Combined slip — Fy vs alpha at different kappa ----
    ax = axes[1, 1]
    alpha_pos = np.linspace(0, 0.25, 300)
    kappas = [0, 0.02, 0.05, 0.1, 0.15]
    kappa_colours = ["#1a237e", "#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for k, c in zip(kappas, kappa_colours):
        Fy = [pacejka_combined_Fy(k, a, Fz_test, pac) for a in alpha_pos]
        ax.plot(
            np.degrees(alpha_pos), Fy, color=c, linewidth=2,
            label=f"kappa = {k:.2f}",
        )
    ax.set_xlabel("Slip Angle [deg]")
    ax.set_ylabel("Lateral Force Fy [N]")
    ax.set_title(f"Combined Slip: Fy (Fz = {Fz_test} N)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Plot 6: Friction ellipse ----
    ax = axes[1, 2]
    for Fz, c in zip(loads, colours):
        Fx_pts = []
        Fy_pts = []
        for a in np.linspace(0, 0.2, 100):
            Fy_val = abs(pacejka_Fy0(a, Fz, lat))
            best_Fx = 0
            for k in np.linspace(0, 0.3, 50):
                Fx_val = pacejka_combined_Fx(k, a, Fz, pac)
                if Fx_val > best_Fx:
                    best_Fx = Fx_val
            Fx_pts.append(best_Fx)
            Fy_pts.append(Fy_val)

        Fx_arr = np.array(Fx_pts)
        Fy_arr = np.array(Fy_pts)
        ax.plot(Fy_arr, Fx_arr, color=c, linewidth=2, label=f"Fz = {Fz} N")
        ax.plot(-Fy_arr, Fx_arr, color=c, linewidth=2)
        ax.plot(Fy_arr, -Fx_arr, color=c, linewidth=2)
        ax.plot(-Fy_arr, -Fx_arr, color=c, linewidth=2)

    ax.set_xlabel("Lateral Force Fy [N]")
    ax.set_ylabel("Longitudinal Force Fx [N]")
    ax.set_title("Friction Ellipse (Combined Slip Envelope)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pacejka verification plots")
    parser.add_argument(
        "--tyre", default="hoosier_r25b_18x6",
        help="Tyre preset ID from config/tyres.yaml",
    )
    parser.add_argument(
        "--save", default=None,
        help="Save path for the figure (e.g. plots/pacejka.png). Shows interactively if omitted.",
    )
    args = parser.parse_args()

    plot_verification(preset_id=args.tyre, save_path=args.save)
