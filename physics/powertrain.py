def max_tractive_force(P_max: float, Fx_max: float, v: float) -> float:
    """
    Maximum tractive force from EV motor [N].
    
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