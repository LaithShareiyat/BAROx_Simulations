import math
import numpy as np

from models.vehicle import PacejkaParams, PacejkaCoefficients


# ═══════════════════════════════════════════════════════════════════
# Constant-mu tyre model (original)
# ═══════════════════════════════════════════════════════════════════

def normal_load(m: float, g: float, downforce: float) -> float:
    """
    Total vertical load on tyres [N].

    F_z = m*g + F_downforce
    """
    return m * g + downforce

def max_friction_force(mu: float, Fz: float) -> float:
    """
    Maximum friction force available [N].

    F_max = mu * F_z
    """
    return mu * Fz

def a_max(mu: float, g: float, m: float, downforce: float = 0.0) -> float:
    """
    Maximum total acceleration (friction circle radius) [m/s²].

    a_max = mu * (g + F_downforce/m)

    The friction circle constraint is:
        a_x² + a_y² <= a_max²

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
        a_x = sqrt(a_max² - a_y²)

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


# ═══════════════════════════════════════════════════════════════════
# Pacejka Magic Formula
# ═══════════════════════════════════════════════════════════════════

def _pacejka_BCDE(Fz: float, coeffs: PacejkaCoefficients) -> tuple:
    """Compute (B, C, D, E) at a given vertical load.

    D (peak)    = (a1 * Fz + a2) * Fz
    BCD (stiff) = a3 * sin(a4 * arctan(a5 * Fz))
    B           = BCD / (C * D)
    E (curve)   = a6 * Fz + a7

    Args:
        Fz: Vertical load [N] (positive downward)
        coeffs: Pacejka coefficient set

    Returns:
        (B, C, D, E) tuple
    """
    C = coeffs.C
    D = (coeffs.a1 * Fz + coeffs.a2) * Fz
    BCD = coeffs.a3 * math.sin(coeffs.a4 * math.atan(coeffs.a5 * Fz))
    CD = C * D
    B = BCD / CD if abs(CD) > 1e-6 else 0.0
    E = coeffs.a6 * Fz + coeffs.a7
    # Clamp E < 1 for numerical stability
    E = min(E, 0.99)
    return B, C, D, E


def _pacejka_formula(x: float, B: float, C: float, D: float, E: float) -> float:
    """Evaluate Pacejka Magic Formula.

    F = D * sin(C * arctan(B*x - E*(B*x - arctan(B*x))))

    Args:
        x: Slip input (slip angle [rad] or slip ratio [-])
        B, C, D, E: Pacejka coefficients at current load

    Returns:
        Force [N]
    """
    Bx = B * x
    return D * math.sin(C * math.atan(Bx - E * (Bx - math.atan(Bx))))


def pacejka_Fy0(alpha: float, Fz: float, coeffs: PacejkaCoefficients) -> float:
    """Pure lateral force from Pacejka Magic Formula [N].

    F_y0 = D * sin(C * arctan(B*alpha - E*(B*alpha - arctan(B*alpha))))

    Args:
        alpha: Slip angle [rad]
        Fz: Vertical load [N] (positive downward)
        coeffs: Lateral Pacejka coefficients

    Returns:
        Lateral force [N] (positive opposes slip)
    """
    if Fz < 1.0:
        return 0.0
    B, C, D, E = _pacejka_BCDE(Fz, coeffs)
    return _pacejka_formula(alpha, B, C, D, E)


def pacejka_Fx0(kappa: float, Fz: float, coeffs: PacejkaCoefficients) -> float:
    """Pure longitudinal force from Pacejka Magic Formula [N].

    F_x0 = D * sin(C * arctan(B*kappa - E*(B*kappa - arctan(B*kappa))))

    Args:
        kappa: Slip ratio [-] (positive = traction, negative = braking)
        Fz: Vertical load [N]
        coeffs: Longitudinal Pacejka coefficients

    Returns:
        Longitudinal force [N]
    """
    if Fz < 1.0:
        return 0.0
    B, C, D, E = _pacejka_BCDE(Fz, coeffs)
    return _pacejka_formula(kappa, B, C, D, E)


def pacejka_combined_Fx(kappa: float, alpha: float, Fz: float,
                         tyre: PacejkaParams) -> float:
    """Combined-slip longitudinal force [N].

    F_x = G_xa(alpha) * F_x0(kappa)
    G_xa = cos(C_xa * arctan(B_xa * alpha))

    The weighting function G_xa reduces longitudinal force when
    lateral slip is present.

    Args:
        kappa: Slip ratio [-]
        alpha: Slip angle [rad]
        Fz: Vertical load [N]
        tyre: PacejkaParams

    Returns:
        Longitudinal force reduced by lateral slip [N]
    """
    Fx0 = pacejka_Fx0(kappa, Fz, tyre.longitudinal)
    G_xa = math.cos(tyre.C_xa * math.atan(tyre.B_xa * alpha))
    return G_xa * Fx0


def pacejka_combined_Fy(kappa: float, alpha: float, Fz: float,
                         tyre: PacejkaParams) -> float:
    """Combined-slip lateral force [N].

    F_y = G_yk(kappa) * F_y0(alpha)
    G_yk = cos(C_yk * arctan(B_yk * kappa))

    The weighting function G_yk reduces lateral force when
    longitudinal slip is present.

    Args:
        kappa: Slip ratio [-]
        alpha: Slip angle [rad]
        Fz: Vertical load [N]
        tyre: PacejkaParams

    Returns:
        Lateral force reduced by longitudinal slip [N]
    """
    Fy0 = pacejka_Fy0(alpha, Fz, tyre.lateral)
    G_yk = math.cos(tyre.C_yk * math.atan(tyre.B_yk * kappa))
    return G_yk * Fy0


def _peak_slip_angle(Fz: float, coeffs: PacejkaCoefficients) -> float:
    """Estimate slip angle at which lateral force peaks [rad].

    The peak occurs approximately at x_peak = 1/B for the basic formula.

    Args:
        Fz: Vertical load [N]
        coeffs: Lateral Pacejka coefficients

    Returns:
        Peak slip angle [rad], clamped to [0.01, 0.5]
    """
    B, C, D, E = _pacejka_BCDE(Fz, coeffs)
    if abs(B) < 1e-6:
        return 0.2  # fallback
    alpha_peak = 1.0 / abs(B)
    return max(0.01, min(alpha_peak, 0.5))


def _find_alpha_for_Fy(Fy_target: float, Fz: float,
                        coeffs: PacejkaCoefficients,
                        tol: float = 1.0,
                        max_iter: int = 20) -> float:
    """Find slip angle that produces the required lateral force [rad].

    Uses bisection on the monotonic rising portion of F_y(alpha)
    from 0 to alpha_peak.  If Fy_target exceeds the peak, returns
    the peak slip angle.

    Args:
        Fy_target: Required lateral force [N] (positive)
        Fz: Vertical load [N]
        coeffs: Lateral Pacejka coefficients
        tol: Force tolerance [N]
        max_iter: Maximum bisection iterations

    Returns:
        Slip angle [rad]
    """
    if Fz < 1.0 or Fy_target <= 0.0:
        return 0.0

    alpha_peak = _peak_slip_angle(Fz, coeffs)
    Fy_peak = abs(pacejka_Fy0(alpha_peak, Fz, coeffs))

    # If required force exceeds peak, tyre is saturated
    if Fy_target >= Fy_peak:
        return alpha_peak

    # Bisection on [0, alpha_peak]
    lo, hi = 0.0, alpha_peak
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        Fy_mid = abs(pacejka_Fy0(mid, Fz, coeffs))
        if Fy_mid < Fy_target:
            lo = mid
        else:
            hi = mid
        if abs(Fy_mid - Fy_target) < tol:
            break
    return 0.5 * (lo + hi)


# ═══════════════════════════════════════════════════════════════════
# Tyre model abstraction layer
# ═══════════════════════════════════════════════════════════════════

def max_lateral_force(tyre, Fz: float, n_tyres: int = 1) -> float:
    """Maximum lateral force at given vertical load [N].

    For constant-mu: F_y_max = mu * Fz
    For Pacejka: peak of F_y0(alpha) curve (= |D|), computed per-tyre
    then summed. This correctly captures load sensitivity — splitting
    load across multiple tyres gives more total force than one tyre
    at the combined load.

    Args:
        tyre: TyreParamsMVP, TyreParams, or PacejkaParams
        Fz: Total vertical load on this group of tyres [N]
        n_tyres: Number of tyres sharing this load (1=per-tyre,
                 2=per-axle, 4=whole vehicle)

    Returns:
        Maximum lateral force [N] (sum across all n_tyres)
    """
    if isinstance(tyre, PacejkaParams):
        Fz_per = Fz / max(n_tyres, 1)
        D_per = (tyre.lateral.a1 * Fz_per + tyre.lateral.a2) * Fz_per
        return n_tyres * abs(D_per)
    return tyre.mu * Fz


def max_longitudinal_force(tyre, Fz: float, n_tyres: int = 1) -> float:
    """Maximum longitudinal force at given vertical load [N].

    For constant-mu: F_x_max = mu * Fz
    For Pacejka: peak of F_x0(kappa) curve (= |D|), computed per-tyre
    then summed.

    Args:
        tyre: TyreParamsMVP, TyreParams, or PacejkaParams
        Fz: Total vertical load on this group of tyres [N]
        n_tyres: Number of tyres sharing this load (1=per-tyre,
                 2=per-axle, 4=whole vehicle)

    Returns:
        Maximum longitudinal force [N] (sum across all n_tyres)
    """
    if isinstance(tyre, PacejkaParams):
        Fz_per = Fz / max(n_tyres, 1)
        D_per = (tyre.longitudinal.a1 * Fz_per + tyre.longitudinal.a2) * Fz_per
        return n_tyres * abs(D_per)
    return tyre.mu * Fz


def available_longitudinal_force(tyre, Fz: float, Fy_required: float,
                                  n_tyres: int = 1) -> float:
    """Available longitudinal force given a required lateral force [N].

    For constant-mu: friction circle sqrt((mu*Fz)^2 - Fy^2)
    For Pacejka: split loads per-tyre, find slip angle producing
    Fy_required/n, then compute max Fx at that slip angle via the
    combined slip model, and sum across tyres.

    Args:
        tyre: Tyre parameters (any model)
        Fz: Total vertical load on this group of tyres [N]
        Fy_required: Total required lateral force [N] (absolute value used)
        n_tyres: Number of tyres sharing this load (1=per-tyre,
                 2=per-axle, 4=whole vehicle)

    Returns:
        Maximum available longitudinal force [N] (sum across all n_tyres)
    """
    Fy_abs = abs(Fy_required)

    if isinstance(tyre, PacejkaParams):
        Fz_per = Fz / max(n_tyres, 1)
        Fy_per = Fy_abs / max(n_tyres, 1)

        if Fz_per < 1.0:
            return 0.0

        # Find slip angle needed for the per-tyre lateral force
        alpha = _find_alpha_for_Fy(Fy_per, Fz_per, tyre.lateral)

        # Check if laterally saturated (per-tyre)
        Fy_peak_per = abs((tyre.lateral.a1 * Fz_per + tyre.lateral.a2) * Fz_per)
        if Fy_per >= Fy_peak_per:
            return 0.0

        # Peak longitudinal force per tyre at this Fz
        D_lon_per = (tyre.longitudinal.a1 * Fz_per + tyre.longitudinal.a2) * Fz_per
        Fx_peak_per = abs(D_lon_per)

        # Combined slip reduction: G_xa reduces Fx when alpha is nonzero
        G_xa = math.cos(tyre.C_xa * math.atan(tyre.B_xa * alpha))
        Fx_per = Fx_peak_per * max(G_xa, 0.0)

        return n_tyres * Fx_per
    else:
        # Constant-mu friction circle (linear — n_tyres doesn't affect result)
        F_max = tyre.mu * Fz
        if Fy_abs >= F_max:
            return 0.0
        return math.sqrt(F_max**2 - Fy_abs**2)


def a_max_from_tyre(tyre, g: float, m: float, downforce: float = 0.0) -> float:
    """Maximum acceleration (friction circle radius) [m/s²].

    Drop-in replacement for a_max(mu, g, m, Fdown).

    For constant-mu: mu * (g + downforce/m)
    For Pacejka: sum of per-tyre max lateral forces / m

    Args:
        tyre: Tyre parameters (any model)
        g: Gravitational acceleration [m/s²]
        m: Vehicle mass [kg]
        downforce: Aerodynamic downforce [N]

    Returns:
        Maximum total acceleration [m/s²]
    """
    Fz = m * g + downforce
    return max_lateral_force(tyre, Fz, n_tyres=4) / m


# ═══════════════════════════════════════════════════════════════════
# Axle-aware grip (updated for Pacejka abstraction)
# ═══════════════════════════════════════════════════════════════════

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
        mu: Tyre friction coefficient [-] (used for constant-mu fallback)
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
    tyre = vehicle.tyre

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
    for iteration in range(10):
        a_x_prev = a_x
        dW = (m * a_x * h) / L
        F_zf = max(0.0, W_f_static - dW)  # Front unloads under accel
        F_zr = max(0.0, W_r_static + dW)  # Rear loads up under accel

        if driven == 'rear':
            F_x_max = available_longitudinal_force(tyre, F_zr, F_yr, n_tyres=2)
        elif driven == 'front':
            F_x_max = available_longitudinal_force(tyre, F_zf, F_yf, n_tyres=2)
        else:  # 'both' (AWD)
            F_x_f = available_longitudinal_force(tyre, F_zf, F_yf, n_tyres=2)
            F_x_r = available_longitudinal_force(tyre, F_zr, F_yr, n_tyres=2)
            F_x_max = F_x_f + F_x_r

        a_x = F_x_max / m

        # Early convergence check
        if iteration > 0 and abs(a_x - a_x_prev) < 0.01:
            break

    return a_x


def ax_braking_axle_aware(mu, vehicle, V, kappa, Fdown, M_z_tv=0.0):
    """
    Maximum braking deceleration with axle-aware grip [m/s²].

    Both axles contribute to braking with optimal brake bias (proportional
    to vertical load). Weight transfers forward under braking.

    Args:
        mu: Tyre friction coefficient [-] (used for constant-mu fallback)
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
    tyre = vehicle.tyre

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
    for iteration in range(10):
        a_x_prev = a_x_brake
        dW = (m * a_x_brake * h) / L
        F_zf = max(0.0, W_f_static + dW)  # Front loads up under braking
        F_zr = max(0.0, W_r_static - dW)  # Rear unloads under braking

        F_x_brake_f = available_longitudinal_force(tyre, F_zf, F_yf, n_tyres=2)
        F_x_brake_r = available_longitudinal_force(tyre, F_zr, F_yr, n_tyres=2)
        F_x_brake = F_x_brake_f + F_x_brake_r

        a_x_brake = F_x_brake / m

        # Early convergence check
        if iteration > 0 and abs(a_x_brake - a_x_prev) < 0.01:
            break

    return a_x_brake
