"""
Tyre thermal model with coupled grip feedback.

Single-node lumped thermal model integrated with the QSS solver via
a fixed-point outer iteration.  Temperature affects the Pacejka D
coefficient (peak grip) through a parabolic scaling window.

Energy balance per segment:
    C_thermal * dT/dt = P_heat - P_cool
where:
    P_heat = k_heating * F_tyre * V_slip
    P_cool = (h_static + h_speed * V) * (T - T_ambient)
    V_slip = V * friction_utilisation
    friction_utilisation = sqrt(a_x^2 + a_y^2) / a_max

Grip scaling:
    grip_mult = max(0, 1 - ((T - T_opt) / T_width)^2)
"""

from dataclasses import dataclass
import numpy as np

from models.track import Track
from models.vehicle import VehicleParams, TyreThermalParams
from physics.aero import downforce
from physics.tyre import a_max_from_tyre, thermal_grip_multiplier
from solver.qss_speed import solve_qss


@dataclass
class TyreThermalState:
    """Tyre thermal state at each track point."""
    temperature: np.ndarray       # [degC] tyre temperature at each point
    grip_multiplier: np.ndarray   # [-] grip scaling factor at each point
    heat_input_kW: np.ndarray     # [kW] heat generation at each point
    cooling_kW: np.ndarray        # [kW] heat rejection at each point


def compute_grip_profile(T_profile: np.ndarray,
                         thermal: TyreThermalParams) -> np.ndarray:
    """Convert temperature profile to grip multiplier array."""
    return np.array([
        thermal_grip_multiplier(T, thermal.T_opt, thermal.T_width)
        for T in T_profile
    ])


def integrate_tyre_temperature(track: Track, v: np.ndarray,
                                vehicle: VehicleParams) -> TyreThermalState:
    """Forward-integrate tyre temperature along the driven path.

    Args:
        track: Track geometry
        v: Solved velocity profile [m/s]
        vehicle: Vehicle parameters (must have tyre_thermal)

    Returns:
        TyreThermalState with per-point arrays
    """
    thermal = vehicle.tyre_thermal
    n = len(track.s)
    m = vehicle.m
    g = vehicle.g

    T = np.zeros(n)
    T[0] = thermal.T_initial
    heat_in = np.zeros(n)
    cool_out = np.zeros(n)

    for i in range(n - 1):
        v_i = max(v[i], 0.1)
        ds_i = track.ds[i]
        dt = ds_i / v_i

        # Compute accelerations at this point
        kappa_i = abs(track.kappa[i])
        a_y = v_i ** 2 * kappa_i

        Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v_i)
        a_max_val = a_max_from_tyre(vehicle.tyre, g, m, Fdown)

        # Longitudinal acceleration from kinematic equation
        v_next = max(v[i + 1], 0.1)
        if ds_i > 0.01:
            a_x = (v_next ** 2 - v_i ** 2) / (2 * ds_i)
        else:
            a_x = 0.0

        # Friction utilisation [0, 1]
        a_total = np.sqrt(a_x ** 2 + a_y ** 2)
        fric_util = min(a_total / max(a_max_val, 0.1), 1.0)

        # Tyre force magnitude and slip velocity
        F_tyre = fric_util * a_max_val * m
        V_slip = v_i * fric_util

        # Heat generation [W]
        P_heat = thermal.k_heating * F_tyre * V_slip
        heat_in[i] = P_heat / 1000.0  # kW

        # Convective cooling [W]
        h_cool = thermal.h_static + thermal.h_speed * v_i
        P_cool = h_cool * (T[i] - thermal.T_ambient)
        cool_out[i] = P_cool / 1000.0  # kW

        # Forward Euler temperature update
        dT = (P_heat - P_cool) * dt / thermal.C_thermal
        T[i + 1] = np.clip(T[i] + dT, -40.0, 200.0)

    # Fill last point for heat/cooling arrays
    heat_in[-1] = heat_in[-2] if n > 1 else 0.0
    cool_out[-1] = cool_out[-2] if n > 1 else 0.0

    grip = compute_grip_profile(T, thermal)

    return TyreThermalState(
        temperature=T,
        grip_multiplier=grip,
        heat_input_kW=heat_in,
        cooling_kW=cool_out,
    )


def solve_qss_thermal(track: Track, vehicle: VehicleParams,
                       refine: bool = False, min_points: int = 500,
                       use_bicycle_model: bool = True) -> tuple[dict, float]:
    """QSS solver with coupled tyre thermal model.

    Wraps solve_qss in a fixed-point outer iteration:
    1. Initialise T = T_initial everywhere
    2. Compute grip_scale from T
    3. Run solve_qss with grip_scale
    4. Integrate T from solved velocity
    5. Check convergence, under-relax, repeat

    Args:
        track: Track geometry
        vehicle: Vehicle parameters (must have tyre_thermal enabled)
        refine: Track refinement flag
        min_points: Minimum track points
        use_bicycle_model: Bicycle model flag

    Returns:
        (result_dict, lap_time) — result_dict includes 'thermal_state'
    """
    thermal = vehicle.tyre_thermal
    n = len(track.s)

    # Initial temperature: uniform at T_initial
    T_profile = np.full(n, thermal.T_initial)
    grip_profile = compute_grip_profile(T_profile, thermal)

    result = None
    lap_time = 0.0
    thermal_state = None

    for iteration in range(thermal.max_thermal_iter):
        # Solve QSS with current grip profile
        result, lap_time = solve_qss(
            track, vehicle,
            refine=refine, min_points=min_points,
            use_bicycle_model=use_bicycle_model,
            grip_scale=grip_profile,
        )
        v = result["v"]

        # Integrate temperature from solved velocity
        thermal_state = integrate_tyre_temperature(track, v, vehicle)
        T_new = thermal_state.temperature

        # Check convergence
        max_delta = np.max(np.abs(T_new - T_profile))
        if max_delta < thermal.thermal_tol:
            T_profile = T_new
            break

        # Under-relax
        alpha = thermal.relaxation
        T_profile = alpha * T_new + (1.0 - alpha) * T_profile

        # Recompute grip from relaxed temperature
        grip_profile = compute_grip_profile(T_profile, thermal)

    # Final thermal state from converged temperature
    thermal_state = integrate_tyre_temperature(track, result["v"], vehicle)
    result["thermal_state"] = thermal_state

    return result, lap_time
