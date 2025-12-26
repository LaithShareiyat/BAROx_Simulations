import numpy as np
from models.track import Track
from models.vehicle import VehicleParams
from physics.aero import drag, downforce
from physics.resistive import rolling_resistance

def lap_time(track, v, v_eps=0.5):
    vi = np.maximum(v[:-1], v_eps)
    return float(np.sum(track.ds / vi))

def channels(track: Track, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate acceleration channels from velocity profile.
    
    Returns:
        ax: Longitudinal acceleration [m/s²]
        ay: Lateral acceleration [m/s²]
    """
    # Lateral: a_y = v² * κ
    ay = v**2 * track.kappa
    
    # Longitudinal: a_x = v * dv/ds
    # Ensure monotonically increasing s to avoid divide-by-zero in gradient
    s_safe = np.maximum.accumulate(track.s + np.arange(len(track.s)) * 1e-10)
    dv_ds = np.gradient(v, s_safe)
    ax = v * dv_ds
    
    return ax[:-1], ay[:-1]

def energy_consumption(track: Track, v: np.ndarray, vehicle: VehicleParams,
                       eta_motor: float = 0.85, eta_regen: float = 0.7) -> dict:
    """
    Calculate energy consumption and regeneration.
    
    Args:
        track: Track object
        v: Velocity profile [m/s]
        vehicle: Vehicle parameters
        eta_motor: Motor efficiency (electrical → mechanical)
        eta_regen: Regeneration efficiency (mechanical → electrical)
    
    Returns:
        Dictionary with energy values in J and kWh
    """
    ax, ay = channels(track, v)
    m = vehicle.m
    
    E_consumed = 0.0
    E_regen = 0.0
    
    for i in range(len(track.ds)):
        vi = v[i]
        
        # Resistance forces
        Fdrag = drag(vehicle.aero.rho, vehicle.aero.CD_A, vi)
        Frr = rolling_resistance(vehicle.Crr, m, vehicle.g)
        
        # Required tractive force
        Fx_req = m * ax[i] + Fdrag + Frr
        ds = track.ds[i]
        
        if Fx_req > 0:
            # Accelerating: motor consumes energy
            E_consumed += Fx_req * ds / eta_motor
        else:
            # Braking: potential regeneration
            E_regen += abs(Fx_req) * ds * eta_regen
    
    E_net = E_consumed - E_regen
    
    return {
        "E_consumed_J": E_consumed,
        "E_regen_J": E_regen,
        "E_net_J": E_net,
        "E_net_kWh": E_net / 3.6e6
    }