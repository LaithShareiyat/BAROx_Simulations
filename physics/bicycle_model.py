"""
Quasi-Steady-State Bicycle Model for Vehicle Dynamics.

This module implements a linear bicycle model solved in quasi-steady-state
for lap time simulation. At each track point, the model calculates the
steady-state equilibrium of forces and moments.

The Bicycle Model:
    - Combines left/right wheels into single front and rear axles
    - Captures slip angles, sideslip, and yaw moment balance
    - Includes torque vectoring as an external yaw moment

Coordinate System:
    - x: forward (along vehicle centreline)
    - y: leftward (perpendicular to centreline)
    - Positive yaw rate: counter-clockwise (turning left)
    - Positive lateral acceleration: leftward (turning left)

Steady-State Equations:
    1. Yaw rate from path:        r = V × κ
    2. Lateral acceleration:       a_y = V × r = V² × κ
    3. Force equilibrium:          F_yf + F_yr = m × a_y
    4. Moment equilibrium:         F_yf × L_f - F_yr × L_r + M_z = 0
    5. Tire forces (linear):       F_y = C_α × α

Slip Angles:
    Front: α_f = δ - β - (L_f × r) / V
    Rear:  α_r = -β + (L_r × r) / V

Where:
    V = velocity [m/s]
    κ = path curvature [1/m]
    r = yaw rate [rad/s]
    β = body sideslip angle [rad]
    δ = steering angle [rad]
    α = slip angle [rad]
    L_f, L_r = distances from CoG to front/rear axle [m]
    C_α = cornering stiffness [N/rad]
    M_z = external yaw moment (from torque vectoring) [Nm]
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from models.vehicle import VehicleParams
from physics.weight_transfer import calculate_axle_loads


@dataclass
class BicycleModelState:
    """State of the bicycle model at a track point."""
    # Inputs
    V: float              # Velocity [m/s]
    kappa: float          # Path curvature [1/m]
    a_x: float            # Longitudinal acceleration [m/s²]

    # Kinematic states
    r: float              # Yaw rate [rad/s]
    a_y: float            # Lateral acceleration [m/s²]

    # Tire states
    alpha_f: float        # Front slip angle [rad]
    alpha_r: float        # Rear slip angle [rad]
    beta: float           # Body sideslip angle [rad]
    delta: float          # Steering angle [rad]

    # Forces
    F_yf: float           # Front lateral force [N]
    F_yr: float           # Rear lateral force [N]
    F_zf: float           # Front vertical load [N]
    F_zr: float           # Rear vertical load [N]

    # External moments
    M_z_tv: float         # Torque vectoring yaw moment [Nm]

    # Grip utilisation
    grip_front: float     # Front grip utilisation [0-1]
    grip_rear: float      # Rear grip utilisation [0-1]
    saturated: bool       # True if either axle is saturated


def solve_qss_bicycle(vehicle: VehicleParams,
                      V: float,
                      kappa: float,
                      a_x: float = 0.0,
                      M_z_tv: float = 0.0) -> BicycleModelState:
    """
    Solve the quasi-steady-state bicycle model.

    Given velocity, curvature, and longitudinal acceleration, this function
    solves for the steady-state slip angles, forces, and sideslip angle.

    Args:
        vehicle: Vehicle parameters (must have geometry and extended tyre)
        V: Velocity [m/s]
        kappa: Path curvature [1/m] (positive = turning left)
        a_x: Longitudinal acceleration [m/s²]
        M_z_tv: Torque vectoring yaw moment [Nm] (positive = helps turn left)

    Returns:
        BicycleModelState with all computed values
    """
    # Get vehicle parameters
    m = vehicle.m
    g = vehicle.g
    mu = vehicle.mu

    # Get geometry
    if vehicle.geometry is not None:
        L_f = vehicle.geometry.L_f_m
        L_r = vehicle.geometry.L_r_m
        L = vehicle.geometry.L
    else:
        L_f = 0.75
        L_r = 0.80
        L = L_f + L_r

    # Get cornering stiffness
    C_f, C_r = vehicle.get_cornering_stiffness()

    # Calculate axle loads with longitudinal weight transfer
    F_zf, F_zr = calculate_axle_loads(vehicle, a_x)

    # Kinematic relationships
    r = V * kappa                    # Yaw rate from path curvature
    a_y = V * r                      # Lateral acceleration = V² × κ

    # Ensure minimum velocity to avoid division by zero
    V_safe = max(V, 0.1)

    # Solve force and moment equilibrium
    # From: F_yf + F_yr = m × a_y
    #       F_yf × L_f - F_yr × L_r + M_z_tv = 0
    #
    # Solving for F_yr:
    # F_yf = m × a_y - F_yr
    # (m × a_y - F_yr) × L_f - F_yr × L_r + M_z_tv = 0
    # m × a_y × L_f - F_yr × L_f - F_yr × L_r + M_z_tv = 0
    # m × a_y × L_f + M_z_tv = F_yr × (L_f + L_r)
    # F_yr = (m × a_y × L_f + M_z_tv) / L

    F_yr = (m * a_y * L_f + M_z_tv) / L
    F_yf = m * a_y - F_yr

    # Slip angles from linear tire model (inverted)
    # F_y = C_α × α  →  α = F_y / C_α
    alpha_f = F_yf / C_f if C_f > 0 else 0.0
    alpha_r = F_yr / C_r if C_r > 0 else 0.0

    # Sideslip angle from rear slip angle geometry
    # α_r = -β + (L_r × r) / V  →  β = -α_r + (L_r × r) / V
    beta = -alpha_r + (L_r * r) / V_safe

    # Steering angle from front slip angle geometry
    # α_f = δ - β - (L_f × r) / V  →  δ = α_f + β + (L_f × r) / V
    delta = alpha_f + beta + (L_f * r) / V_safe

    # Calculate grip utilisation
    # Maximum lateral force limited by friction circle
    # For combined slip, reserve some grip for longitudinal force
    F_yf_max = mu * F_zf
    F_yr_max = mu * F_zr

    grip_front = abs(F_yf) / F_yf_max if F_yf_max > 0 else 0.0
    grip_rear = abs(F_yr) / F_yr_max if F_yr_max > 0 else 0.0

    saturated = grip_front > 1.0 or grip_rear > 1.0

    return BicycleModelState(
        V=V,
        kappa=kappa,
        a_x=a_x,
        r=r,
        a_y=a_y,
        alpha_f=alpha_f,
        alpha_r=alpha_r,
        beta=beta,
        delta=delta,
        F_yf=F_yf,
        F_yr=F_yr,
        F_zf=F_zf,
        F_zr=F_zr,
        M_z_tv=M_z_tv,
        grip_front=grip_front,
        grip_rear=grip_rear,
        saturated=saturated,
    )


def calculate_max_lateral_accel(vehicle: VehicleParams,
                                V: float,
                                a_x: float = 0.0,
                                M_z_tv: float = 0.0,
                                max_iterations: int = 20) -> float:
    """
    Find maximum lateral acceleration before tire saturation.

    Uses iterative search to find the lateral acceleration at which
    either the front or rear axle reaches its grip limit.

    Args:
        vehicle: Vehicle parameters
        V: Velocity [m/s]
        a_x: Longitudinal acceleration [m/s²]
        M_z_tv: Torque vectoring yaw moment [Nm]
        max_iterations: Maximum binary search iterations

    Returns:
        Maximum lateral acceleration [m/s²]
    """
    mu = vehicle.mu
    g = vehicle.g

    # Upper bound: simple friction limit
    a_y_max_simple = mu * g

    # Binary search for actual limit considering weight transfer
    a_y_low = 0.0
    a_y_high = a_y_max_simple * 1.5  # Allow some margin

    for _ in range(max_iterations):
        a_y_mid = (a_y_low + a_y_high) / 2

        # Calculate curvature that would give this lateral acceleration
        if V > 0.1:
            kappa = a_y_mid / (V * V)
        else:
            kappa = 0.0

        # Solve bicycle model
        state = solve_qss_bicycle(vehicle, V, kappa, a_x, M_z_tv)

        if state.saturated:
            a_y_high = a_y_mid
        else:
            a_y_low = a_y_mid

        # Convergence check
        if a_y_high - a_y_low < 0.01:
            break

    return a_y_low


def calculate_understeer_gradient(vehicle: VehicleParams) -> float:
    """
    Calculate the understeer gradient (K) of the vehicle.

    The understeer gradient determines vehicle handling balance:
        K > 0: Understeer (front saturates first)
        K < 0: Oversteer (rear saturates first)
        K = 0: Neutral

    Formula:
        K = (W_f / C_f) - (W_r / C_r)

    Args:
        vehicle: Vehicle parameters

    Returns:
        Understeer gradient [rad / (m/s²)]
    """
    m = vehicle.m
    g = vehicle.g

    # Get geometry
    if vehicle.geometry is not None:
        L = vehicle.geometry.L
        L_r = vehicle.geometry.L_r_m
        L_f = vehicle.geometry.L_f_m
    else:
        L = 1.55
        L_f = 0.75
        L_r = 0.80

    # Static axle weights
    W_f = m * g * (L_r / L)
    W_r = m * g * (L_f / L)

    # Cornering stiffness
    C_f, C_r = vehicle.get_cornering_stiffness()

    # Understeer gradient
    K = (W_f / C_f) - (W_r / C_r)

    return K


def get_handling_characteristic(K: float) -> str:
    """
    Get handling characteristic description from understeer gradient.

    Args:
        K: Understeer gradient [rad / (m/s²)]

    Returns:
        String description of handling characteristic
    """
    if K > 0.002:
        return "Understeer"
    elif K < -0.002:
        return "Oversteer"
    else:
        return "Neutral"


def calculate_critical_speed(vehicle: VehicleParams) -> Optional[float]:
    """
    Calculate critical speed for oversteer vehicles.

    Critical speed is the speed at which an oversteer vehicle becomes
    unstable. Only applies when K < 0 (oversteer).

    Formula:
        V_critical = sqrt(-L × g / K)

    Args:
        vehicle: Vehicle parameters

    Returns:
        Critical speed [m/s] or None if vehicle understeers
    """
    K = calculate_understeer_gradient(vehicle)

    if K >= 0:
        return None  # Understeer - no critical speed

    if vehicle.geometry is not None:
        L = vehicle.geometry.L
    else:
        L = 1.55

    g = vehicle.g

    V_critical = np.sqrt(-L * g / K)

    return V_critical


def calculate_characteristic_speed(vehicle: VehicleParams) -> Optional[float]:
    """
    Calculate characteristic speed for understeer vehicles.

    Characteristic speed is the speed at which steering angle sensitivity
    is halved compared to low speed. Only applies when K > 0 (understeer).

    Formula:
        V_char = sqrt(L × g / K)

    Args:
        vehicle: Vehicle parameters

    Returns:
        Characteristic speed [m/s] or None if vehicle oversteers
    """
    K = calculate_understeer_gradient(vehicle)

    if K <= 0:
        return None  # Oversteer - no characteristic speed

    if vehicle.geometry is not None:
        L = vehicle.geometry.L
    else:
        L = 1.55

    g = vehicle.g

    V_char = np.sqrt(L * g / K)

    return V_char
