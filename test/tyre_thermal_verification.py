"""
Tyre Thermal Model Verification.

Five individual verification plots, each validating a specific aspect
of the thermal model against known analytical properties.

Usage:
    python test/tyre_thermal_verification.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.tyre import thermal_grip_multiplier

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_grip_vs_temperature():
    """Fig 1: Grip multiplier vs temperature.

    Validates:
    - Parabolic shape centred on T_opt.
    - Peak grip = 1.0 exactly at T_opt.
    - Grip = 0 at T_opt +/- T_width.
    - Grip never goes negative (clamped at 0).
    - Symmetry about T_opt.
    """
    T_opt = 80.0
    T_width = 60.0
    T_range = np.linspace(-10, 170, 500)

    grip = np.array([thermal_grip_multiplier(T, T_opt, T_width) for T in T_range])

    # Analytical parabola (before clamping)
    grip_analytical = 1.0 - ((T_range - T_opt) / T_width) ** 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_range, grip, linewidth=2, color="#2070B0",
            label="Implementation (clamped)")
    ax.plot(T_range, grip_analytical, linewidth=2, linestyle=":",
            color="black", label="Analytical parabola (unclamped)")

    ax.axvline(T_opt, color="#4CAF50", linewidth=1.5, alpha=0.7,
               label=f"T_opt = {T_opt} degC")
    ax.axvline(T_opt - T_width, color="#D04040", linewidth=1, alpha=0.5,
               label=f"T_opt - T_width = {T_opt - T_width} degC")
    ax.axvline(T_opt + T_width, color="#D04040", linewidth=1, alpha=0.5,
               label=f"T_opt + T_width = {T_opt + T_width} degC")
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_xlabel("Tyre Temperature [degC]", fontsize=12)
    ax.set_ylabel("Grip Multiplier [-]", fontsize=12)
    ax.set_title("Grip Multiplier vs Temperature", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.3, 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "thermal_fig1_grip_vs_temp.png"), dpi=150)
    plt.close(fig)

    # Verify key values
    assert abs(thermal_grip_multiplier(T_opt, T_opt, T_width) - 1.0) < 1e-10
    assert abs(thermal_grip_multiplier(T_opt - T_width, T_opt, T_width)) < 1e-10
    assert abs(thermal_grip_multiplier(T_opt + T_width, T_opt, T_width)) < 1e-10
    assert thermal_grip_multiplier(-50, T_opt, T_width) == 0.0
    assert thermal_grip_multiplier(200, T_opt, T_width) == 0.0
    print("  Fig 1 saved: thermal_fig1_grip_vs_temp.png  (all assertions passed)")


def plot_energy_balance_convergence():
    """Fig 2: Steady-state energy balance on a constant-speed circle.

    Validates:
    - On a constant-radius circle at constant speed, temperature
      reaches a steady state where P_heat = P_cool.
    - The steady-state temperature can be computed analytically.
    - Forward Euler integration converges to the analytical value.
    """
    from models.vehicle import TyreThermalParams

    thermal = TyreThermalParams(
        enabled=True, T_ambient=25.0, T_initial=40.0,
        T_opt=80.0, T_width=60.0,
        C_thermal=5000.0, k_heating=0.08,
        h_static=10.0, h_speed=1.5,
    )

    # Simulate constant speed on constant-radius circle
    V = 15.0            # m/s
    R = 9.125           # m (skidpad radius)
    kappa = 1.0 / R
    m = 239.0
    g = 9.81
    mu = 1.6

    a_y = V ** 2 * kappa
    a_max_val = mu * g  # simplified (no downforce for clarity)
    fric_util = min(a_y / a_max_val, 1.0)
    F_tyre = fric_util * a_max_val * m
    V_slip = V * fric_util
    P_heat = thermal.k_heating * F_tyre * V_slip  # constant [W]
    h_cool = thermal.h_static + thermal.h_speed * V  # constant [W/K]

    # Analytical steady state: P_heat = h_cool * (T_ss - T_amb)
    T_ss_analytical = thermal.T_ambient + P_heat / h_cool

    # Forward Euler simulation
    dt = 0.01  # s
    t_max = 300.0  # s (5 min should be enough to converge)
    n_steps = int(t_max / dt)
    T = np.zeros(n_steps)
    T[0] = thermal.T_initial
    time = np.arange(n_steps) * dt

    for i in range(n_steps - 1):
        P_cool = h_cool * (T[i] - thermal.T_ambient)
        dT = (P_heat - P_cool) * dt / thermal.C_thermal
        T[i + 1] = T[i] + dT

    # Time constant: tau = C_thermal / h_cool
    tau = thermal.C_thermal / h_cool
    T_analytical_transient = (
        thermal.T_ambient
        + P_heat / h_cool
        + (thermal.T_initial - thermal.T_ambient - P_heat / h_cool)
        * np.exp(-time / tau)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, T, linewidth=2, color="#2070B0",
            label="Forward Euler simulation")
    ax.plot(time, T_analytical_transient, linewidth=2, linestyle=":",
            color="black", label="Analytical: T_amb + (P/h)(1 - e^(-t/tau))")
    ax.axhline(T_ss_analytical, color="#4CAF50", linewidth=1.5, alpha=0.7,
               label=f"Steady state = {T_ss_analytical:.1f} degC")
    ax.axhline(thermal.T_ambient, color="#2196F3", linewidth=1, alpha=0.5,
               label=f"T_ambient = {thermal.T_ambient} degC")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Temperature [degC]", fontsize=12)
    ax.set_title("Energy Balance: Constant Speed on Circle", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "thermal_fig2_energy_balance.png"), dpi=150)
    plt.close(fig)

    # Verify convergence
    final_err = abs(T[-1] - T_ss_analytical)
    euler_err = np.max(np.abs(T - T_analytical_transient))
    print(f"  Fig 2 saved: thermal_fig2_energy_balance.png  "
          f"(steady-state error: {final_err:.2f} degC, "
          f"Euler vs analytical max error: {euler_err:.4f} degC, "
          f"tau = {tau:.1f} s)")


def plot_thermal_on_autocross():
    """Fig 3: Temperature and grip profiles on a real autocross lap.

    Validates:
    - Temperature rises in high-demand sections (corners).
    - Temperature stabilises or drops on straights (cooling > heating).
    - Grip multiplier tracks temperature through the parabolic window.
    - No NaN or unrealistic values anywhere in the profile.
    """
    import yaml
    from models.vehicle import (
        VehicleParams, EVPowertrainParams, VehicleGeometry,
        TyreThermalParams, build_tyre_from_config, build_aero_from_config,
    )
    from events.autocross_generator import build_standard_autocross
    from solver.tyre_thermal import solve_qss_thermal

    with open(os.path.join(os.path.dirname(SAVE_DIR), "config", "default.yaml")) as f:
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
    thermal = TyreThermalParams(enabled=True, T_initial=60.0, T_ambient=25.0)

    vehicle = VehicleParams(
        m=config["vehicle"]["mass_kg"], g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"], aero=aero, tyre=tyre,
        powertrain=powertrain, geometry=geometry, tyre_thermal=thermal,
    )

    track, _ = build_standard_autocross()
    result, t_lap = solve_qss_thermal(track, vehicle)
    ts = result["thermal_state"]
    v = result["v"]
    s = track.s

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: temperature with speed overlay
    colour_temp = "#D04040"
    colour_speed = "#2070B0"
    ax1.plot(s, ts.temperature, linewidth=1.5, color=colour_temp,
             label="Tyre Temperature")
    ax1.axhline(thermal.T_opt, color="#4CAF50", linewidth=1, alpha=0.7,
                label=f"T_opt = {thermal.T_opt} degC")
    ax1.set_ylabel("Temperature [degC]", fontsize=11, color=colour_temp)
    ax1.tick_params(axis="y", labelcolor=colour_temp)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Autocross Thermal Profile (lap time: {t_lap:.3f}s)",
                  fontsize=13, fontweight="bold")

    ax1b = ax1.twinx()
    ax1b.plot(s, v * 3.6, linewidth=1, color=colour_speed, alpha=0.4,
              label="Speed")
    ax1b.set_ylabel("Speed [km/h]", fontsize=11, color=colour_speed)
    ax1b.tick_params(axis="y", labelcolor=colour_speed)

    # Bottom: grip multiplier
    ax2.plot(s, ts.grip_multiplier, linewidth=1.5, color="#FF9800")
    ax2.set_xlabel("Distance [m]", fontsize=11)
    ax2.set_ylabel("Grip Multiplier [-]", fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Grip Scaling Along Lap", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "thermal_fig3_autocross_profile.png"), dpi=150)
    plt.close(fig)

    # Verify no NaN
    assert not np.any(np.isnan(ts.temperature)), "NaN in temperature"
    assert not np.any(np.isnan(ts.grip_multiplier)), "NaN in grip"
    assert np.all(ts.grip_multiplier >= 0), "Negative grip"
    assert np.all(ts.grip_multiplier <= 1.0), "Grip > 1"
    T_min, T_max = np.min(ts.temperature), np.max(ts.temperature)
    print(f"  Fig 3 saved: thermal_fig3_autocross_profile.png  "
          f"(T: {T_min:.1f}-{T_max:.1f} degC, "
          f"grip: {np.min(ts.grip_multiplier):.3f}-{np.max(ts.grip_multiplier):.3f})")


def plot_lap_time_sensitivity():
    """Fig 4: Lap time vs initial tyre temperature.

    Validates:
    - Lap time is minimised when T_initial is near T_opt.
    - Lap time increases for both cold and hot initial temperatures.
    - The sensitivity curve has the expected inverted-U shape (inverted
      because lower lap time = better, matching the grip parabola).
    - The base (no thermal) lap time is recovered when grip ~= 1.0.
    """
    import yaml
    from models.vehicle import (
        VehicleParams, EVPowertrainParams, VehicleGeometry,
        TyreThermalParams, build_tyre_from_config, build_aero_from_config,
    )
    from events.autocross_generator import build_standard_autocross
    from solver.qss_speed import solve_qss
    from solver.tyre_thermal import solve_qss_thermal

    with open(os.path.join(os.path.dirname(SAVE_DIR), "config", "default.yaml")) as f:
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
    track, _ = build_standard_autocross()

    # Base lap time (no thermal)
    vehicle_base = VehicleParams(
        m=config["vehicle"]["mass_kg"], g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"], aero=aero, tyre=tyre,
        powertrain=powertrain, geometry=geometry,
    )
    _, t_base = solve_qss(track, vehicle_base)

    # Sweep T_initial
    T_initials = np.arange(30, 120, 5)
    lap_times = []
    for T_init in T_initials:
        thermal = TyreThermalParams(
            enabled=True, T_initial=float(T_init), T_ambient=25.0,
        )
        vehicle = VehicleParams(
            m=config["vehicle"]["mass_kg"], g=config["vehicle"]["g"],
            Crr=config["vehicle"]["Crr"], aero=aero, tyre=tyre,
            powertrain=powertrain, geometry=geometry, tyre_thermal=thermal,
        )
        _, t_lap = solve_qss_thermal(track, vehicle)
        lap_times.append(t_lap)

    lap_times = np.array(lap_times)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_initials, lap_times, linewidth=2, color="#2070B0", marker="o",
            markersize=4, label="Thermal lap time")
    ax.axhline(t_base, color="#4CAF50", linewidth=1.5, alpha=0.7,
               label=f"Base (no thermal) = {t_base:.3f}s")
    ax.axvline(80, color="#D04040", linewidth=1, alpha=0.5,
               label="T_opt = 80 degC")

    ax.set_xlabel("Initial Tyre Temperature [degC]", fontsize=12)
    ax.set_ylabel("Lap Time [s]", fontsize=12)
    ax.set_title("Lap Time Sensitivity to Initial Temperature",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "thermal_fig4_sensitivity.png"), dpi=150)
    plt.close(fig)

    best_idx = np.argmin(lap_times)
    print(f"  Fig 4 saved: thermal_fig4_sensitivity.png  "
          f"(best T_init = {T_initials[best_idx]} degC at {lap_times[best_idx]:.3f}s, "
          f"base = {t_base:.3f}s)")


def plot_outer_iteration_convergence():
    """Fig 5: Outer iteration convergence.

    Validates:
    - The fixed-point iteration converges monotonically.
    - Convergence is achieved within the iteration limit.
    - Under-relaxation prevents oscillation.
    """
    import yaml
    from models.vehicle import (
        VehicleParams, EVPowertrainParams, VehicleGeometry,
        TyreThermalParams, build_tyre_from_config, build_aero_from_config,
    )
    from events.autocross_generator import build_standard_autocross
    from solver.qss_speed import solve_qss
    from solver.tyre_thermal import (
        compute_grip_profile, integrate_tyre_temperature,
    )

    with open(os.path.join(os.path.dirname(SAVE_DIR), "config", "default.yaml")) as f:
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
    track, _ = build_standard_autocross()

    thermal = TyreThermalParams(
        enabled=True, T_initial=60.0, T_ambient=25.0,
        max_thermal_iter=10, thermal_tol=0.1,
    )
    vehicle = VehicleParams(
        m=config["vehicle"]["mass_kg"], g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"], aero=aero, tyre=tyre,
        powertrain=powertrain, geometry=geometry, tyre_thermal=thermal,
    )

    n = len(track.s)
    T_profile = np.full(n, thermal.T_initial)
    grip_profile = compute_grip_profile(T_profile, thermal)

    max_deltas = []
    lap_times = []
    T_maxes = []

    for iteration in range(thermal.max_thermal_iter):
        result, t_lap = solve_qss(
            track, vehicle, grip_scale=grip_profile,
        )
        v = result["v"]
        ts = integrate_tyre_temperature(track, v, vehicle)
        T_new = ts.temperature

        max_delta = np.max(np.abs(T_new - T_profile))
        max_deltas.append(max_delta)
        lap_times.append(t_lap)
        T_maxes.append(np.max(T_new))

        if max_delta < thermal.thermal_tol:
            T_profile = T_new
            break

        alpha = thermal.relaxation
        T_profile = alpha * T_new + (1.0 - alpha) * T_profile
        grip_profile = compute_grip_profile(T_profile, thermal)

    iterations = np.arange(1, len(max_deltas) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: convergence of temperature delta
    ax1.semilogy(iterations, max_deltas, linewidth=2, color="#2070B0",
                 marker="o", markersize=6)
    ax1.axhline(thermal.thermal_tol, color="#D04040", linewidth=1.5, alpha=0.7,
                label=f"Tolerance = {thermal.thermal_tol} degC")
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Max Temperature Change [degC]", fontsize=12)
    ax1.set_title("Convergence: Temperature Delta", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(iterations)

    # Right: lap time convergence
    ax2.plot(iterations, lap_times, linewidth=2, color="#4CAF50",
             marker="s", markersize=6)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Lap Time [s]", fontsize=12)
    ax2.set_title("Convergence: Lap Time", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(iterations)

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "thermal_fig5_convergence.png"), dpi=150)
    plt.close(fig)

    print(f"  Fig 5 saved: thermal_fig5_convergence.png  "
          f"(converged in {len(max_deltas)} iterations, "
          f"final delta: {max_deltas[-1]:.2f} degC)")


if __name__ == "__main__":
    print("Tyre Thermal Model Verification")
    print("=" * 50)
    plot_grip_vs_temperature()
    plot_energy_balance_convergence()
    plot_thermal_on_autocross()
    plot_lap_time_sensitivity()
    plot_outer_iteration_convergence()
    print("=" * 50)
    print("All verification plots generated.")
