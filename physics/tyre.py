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