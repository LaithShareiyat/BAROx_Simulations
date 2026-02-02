import math
import numpy as np

def normal_load(m: float, g: float, downforce: float) -> float:
    """
    Total vertical load on tyres [N].
    
    F_z = m*g + F_downforce
    """
    return m * g + downforce

def max_friction_force(mu: float, Fz: float) -> float:
    """
    Maximum friction force available [N].
    
    F_max = μ * F_z
    """
    return mu * Fz

def a_max(mu: float, g: float, m: float, downforce: float = 0.0) -> float:
    """
    Maximum total acceleration (friction circle radius) [m/s²].
    
    a_max = μ * (g + F_downforce/m)
    
    The friction circle constraint is:
        a_x² + a_y² ≤ a_max²
    
    Args:
        mu: Tyre friction coefficient [-]
        g: Gravitational acceleration [m/s²]
        m: Vehicle mass [kg]
        downforce: Aerodynamic downforce [N]
    
    Returns:
        Maximum acceleration [m/s²]
    """
    Fz = normal_load(m, g, downforce)
    return mu * Fz / m

def ax_available(a_max_val: float, ay: float) -> float:
    """
    Available longitudinal acceleration given lateral acceleration [m/s²].
    
    From friction circle:
        a_x = √(a_max² - a_y²)
    
    Returns 0 if |a_y| > a_max (fully saturated in lateral).
    
    Args:
        a_max_val: Maximum total acceleration [m/s²]
        ay: Current lateral acceleration [m/s²]
    
    Returns:
        Available longitudinal acceleration [m/s²]
    """
    if abs(ay) >= a_max_val:
        return 0.0
    return np.sqrt(a_max_val**2 - ay**2)


def ax_traction_axle_aware(mu, vehicle, V, kappa, Fdown, M_z_tv=0.0):
    """
    Maximum traction acceleration with axle-aware grip [m/s²].

    Accounts for:
    - Drivetrain (which axle is driven)
    - Longitudinal weight transfer under acceleration
    - Per-axle friction circles with lateral force reservation
    - Downforce distribution proportional to weight distribution

    Uses iterative solver for the a_x <-> weight transfer coupling.

    Args:
        mu: Tyre friction coefficient [-]
        vehicle: VehicleParams with geometry and extended powertrain
        V: Current velocity [m/s]
        kappa: Absolute path curvature [1/m]
        Fdown: Total aerodynamic downforce [N]
        M_z_tv: Torque vectoring yaw moment [Nm]

    Returns:
        Maximum traction acceleration [m/s²]
    """
    m = vehicle.m
    g = vehicle.g
    geo = vehicle.geometry
    L = geo.L
    L_f = geo.L_f_m
    L_r = geo.L_r_m
    h = geo.h_cg_m

    driven = vehicle.powertrain.driven_axle  # 'front', 'rear', 'both'

    a_y = V ** 2 * kappa

    # Lateral force split from yaw moment balance
    F_yr = (m * a_y * L_f + M_z_tv) / L
    F_yf = m * a_y - F_yr

    # Split downforce proportional to weight distribution
    wd_front = geo.weight_distribution_front
    Fdown_f = Fdown * wd_front
    Fdown_r = Fdown * (1 - wd_front)

    # Static axle weights + downforce
    W = m * g
    W_f_static = W * (L_r / L) + Fdown_f
    W_r_static = W * (L_f / L) + Fdown_r

    # Iterative solve: a_x -> weight transfer -> grip -> a_x
    a_x = 0.0
    for _ in range(6):
        dW = (m * a_x * h) / L
        F_zf = max(0.0, W_f_static - dW)  # Front unloads under accel
        F_zr = max(0.0, W_r_static + dW)  # Rear loads up under accel

        if driven == 'rear':
            cap = (mu * F_zr) ** 2 - F_yr ** 2
            F_x_max = np.sqrt(max(cap, 0.0))
        elif driven == 'front':
            cap = (mu * F_zf) ** 2 - F_yf ** 2
            F_x_max = np.sqrt(max(cap, 0.0))
        else:  # 'both' (AWD)
            cap_f = (mu * F_zf) ** 2 - F_yf ** 2
            cap_r = (mu * F_zr) ** 2 - F_yr ** 2
            F_x_max = np.sqrt(max(cap_f, 0.0)) + np.sqrt(max(cap_r, 0.0))

        a_x = F_x_max / m

    return a_x


def ax_braking_axle_aware(mu, vehicle, V, kappa, Fdown, M_z_tv=0.0):
    """
    Maximum braking deceleration with axle-aware grip [m/s²].

    Both axles contribute to braking with optimal brake bias (proportional
    to vertical load). Weight transfers forward under braking.

    Args:
        mu: Tyre friction coefficient [-]
        vehicle: VehicleParams with geometry and extended powertrain
        V: Current velocity [m/s]
        kappa: Absolute path curvature [1/m]
        Fdown: Total aerodynamic downforce [N]
        M_z_tv: Torque vectoring yaw moment [Nm]

    Returns:
        Maximum braking deceleration [m/s²] (positive value)
    """
    m = vehicle.m
    g = vehicle.g
    geo = vehicle.geometry
    L = geo.L
    L_f = geo.L_f_m
    L_r = geo.L_r_m
    h = geo.h_cg_m

    a_y = V ** 2 * kappa

    # Lateral force split from yaw moment balance
    F_yr = (m * a_y * L_f + M_z_tv) / L
    F_yf = m * a_y - F_yr

    # Split downforce proportional to weight distribution
    wd_front = geo.weight_distribution_front
    Fdown_f = Fdown * wd_front
    Fdown_r = Fdown * (1 - wd_front)

    # Static axle weights + downforce
    W = m * g
    W_f_static = W * (L_r / L) + Fdown_f
    W_r_static = W * (L_f / L) + Fdown_r

    # Iterative solve: a_x_brake -> weight transfer -> grip -> a_x_brake
    a_x_brake = 0.0
    for _ in range(6):
        dW = (m * a_x_brake * h) / L
        F_zf = max(0.0, W_f_static + dW)  # Front loads up under braking
        F_zr = max(0.0, W_r_static - dW)  # Rear unloads under braking

        cap_f = (mu * F_zf) ** 2 - F_yf ** 2
        cap_r = (mu * F_zr) ** 2 - F_yr ** 2
        F_x_brake = np.sqrt(max(cap_f, 0.0)) + np.sqrt(max(cap_r, 0.0))

        a_x_brake = F_x_brake / m

    return a_x_brake