from typing import Union


def max_tractive_force(P_max: float, Fx_max: float, v: float) -> float:
    """
    Maximum tractive force from EV motor [N] (legacy function).

    Limited by:
    1. Torque limit at low speed: F ≤ Fx_max
    2. Power limit at high speed: F ≤ P_max / v

    The crossover speed is v_cross = P_max / Fx_max

    Args:
        P_max: Maximum power [W]
        Fx_max: Maximum force (torque-limited) [N]
        v: Current velocity [m/s]

    Returns:
        Maximum tractive force [N]
    """
    if v < 0.1:  # Avoid division by zero
        return Fx_max

    F_power_limited = P_max / v
    return min(Fx_max, F_power_limited)


def max_tractive_force_extended(powertrain, v: float) -> float:
    """
    Maximum tractive force with RPM limit consideration [N].

    For use with EVPowertrainParams (extended powertrain model).

    Limited by:
    1. Torque limit at low speed: F ≤ Fx_max
    2. Power limit at high speed: F ≤ P_max / v
    3. RPM limit at very high speed: F = 0 if v > v_max_rpm

    Args:
        powertrain: EVPowertrainParams or EVPowertrainMVP object
        v: Current velocity [m/s]

    Returns:
        Maximum tractive force [N]
    """
    # Check if this is the extended powertrain with RPM limit
    if hasattr(powertrain, 'v_max_rpm'):
        # Beyond motor RPM capability - no tractive force available
        if v > powertrain.v_max_rpm:
            return 0.0

    # Standard torque/power limiting
    if v < 0.1:  # Avoid division by zero
        return powertrain.Fx_max

    F_power_limited = powertrain.P_max / v
    return min(powertrain.Fx_max, F_power_limited)


def get_powertrain_limits(powertrain) -> dict:
    """
    Get summary of powertrain performance limits.

    Args:
        powertrain: EVPowertrainParams or EVPowertrainMVP object

    Returns:
        Dictionary with key performance values
    """
    limits = {
        'P_max_kW': powertrain.P_max / 1000,
        'Fx_max_N': powertrain.Fx_max,
        'v_crossover_ms': powertrain.P_max / powertrain.Fx_max if powertrain.Fx_max > 0 else 0,
    }

    # Add extended powertrain info if available
    if hasattr(powertrain, 'v_max_rpm'):
        limits['v_max_rpm_ms'] = powertrain.v_max_rpm
        limits['v_max_rpm_kmh'] = powertrain.v_max_rpm * 3.6
        limits['n_motors'] = powertrain.n_motors
        limits['drivetrain'] = powertrain.drivetrain

    return limits