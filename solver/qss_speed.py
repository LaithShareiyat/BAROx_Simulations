from dataclasses import dataclass

@dataclass(frozen=True)
class SolverSettings:
    v0: float = 0.0
    v_min: float = 0.1
    max_iter: int = 30
    tol: float = 1e-3     # m/s

# solver/qss_speed_profile.py
import numpy as np
from models.track import Track
from models.vehicle import VehicleParams
from physics.aero import drag, downforce
from physics.tyre import a_max, ax_available
from physics.powertrain import max_tractive_force
from physics.resistive import rolling_resistance

def solve_qss(track: Track, vehicle: VehicleParams) -> tuple[dict, float]:
    """
    Quasi-Steady-State lap time solver.
    
    Algorithm:
    1. Calculate lateral velocity limit at each point (cornering)
    2. Forward pass: accelerate from start, limited by grip & power
    3. Backward pass: decelerate from end, limited by grip
    4. Take minimum of all three limits
    5. Integrate time
    
    Returns:
        result: {"v": velocity array, "v_lat": lateral limit, ...}
        lap_time: Total time [s]
    """
    n = len(track.s)
    m = vehicle.m
    g = vehicle.g
    mu = vehicle.tyre.mu
    
    # =========================================
    # STEP 1: Lateral (cornering) speed limit
    # =========================================
    # At each point: a_y = v² * κ ≤ a_y_max
    # Therefore: v ≤ √(a_y_max / |κ|)
    
    v_lat = np.zeros(n)
    for i in range(n):
        v_guess = 50.0  # Initial guess [m/s]
        
        # Iterate because downforce depends on speed
        for _ in range(10):
            Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v_guess)
            amax = a_max(mu, g, m, Fdown)
            
            kappa_i = abs(track.kappa[i])
            if kappa_i < 1e-6:  # Straight section
                v_lat[i] = 100.0  # High limit
                break
            else:
                v_lat[i] = np.sqrt(amax / kappa_i)
            
            if abs(v_lat[i] - v_guess) < 0.01:
                break
            v_guess = v_lat[i]
    
    # =========================================
    # STEP 2: Forward pass (acceleration limit)
    # =========================================
    # Starting from rest (or entry speed), accelerate as hard as possible
    
    v_fwd = np.zeros(n)
    v_fwd[0] = min(v_lat[0], 10.0)  # Start speed
    
    for i in range(n - 1):
        v_i = v_fwd[i]
        ds = track.ds[i]
        
        # Current forces
        Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v_i)
        Fdrag = drag(vehicle.aero.rho, vehicle.aero.CD_A, v_i)
        Frr = rolling_resistance(vehicle.Crr, m, g)
        
        # Maximum grip-limited acceleration
        amax = a_max(mu, g, m, Fdown)
        ay = v_i**2 * abs(track.kappa[i])
        ax_grip = ax_available(amax, ay)
        
        # Maximum power-limited acceleration
        Fx_motor = max_tractive_force(
            vehicle.powertrain.P_max,
            vehicle.powertrain.Fx_max,
            v_i
        )
        ax_power = (Fx_motor - Fdrag - Frr) / m
        
        # Take minimum (most limiting)
        ax = min(ax_grip, ax_power)
        ax = max(ax, 0)  # Can't accelerate backwards
        
        # Kinematic equation: v² = v₀² + 2*a*ds
        v_next_sq = v_i**2 + 2 * ax * ds
        v_fwd[i + 1] = min(np.sqrt(max(v_next_sq, 0)), v_lat[i + 1])
    
    # =========================================
    # STEP 3: Backward pass (braking limit)
    # =========================================
    # From end, work backwards: what speed could we brake from?
    
    v_bwd = np.zeros(n)
    v_bwd[-1] = v_fwd[-1]
    
    for i in range(n - 2, -1, -1):
        v_i = v_bwd[i + 1]
        ds = track.ds[i]
        
        # Braking forces (drag helps slow down)
        Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v_i)
        Fdrag = drag(vehicle.aero.rho, vehicle.aero.CD_A, v_i)
        Frr = rolling_resistance(vehicle.Crr, m, g)
        
        # Maximum grip-limited deceleration
        amax = a_max(mu, g, m, Fdown)
        ay = v_i**2 * abs(track.kappa[i + 1])
        ax_grip = ax_available(amax, ay)
        
        # Deceleration (braking adds to drag/rr)
        ax_brake = ax_grip + (Fdrag + Frr) / m
        
        # Kinematic equation (going backwards)
        v_prev_sq = v_i**2 + 2 * ax_brake * ds
        v_bwd[i] = min(np.sqrt(max(v_prev_sq, 0)), v_lat[i])
    
    # =========================================
    # STEP 4: Combine all limits
    # =========================================
    v = np.minimum(np.minimum(v_fwd, v_bwd), v_lat)
    
    # =========================================
    # STEP 5: Calculate lap time
    # =========================================
    # t = ∫ ds/v
    v_avg = 0.5 * (v[:-1] + v[1:])  # Average speed per segment
    dt = track.ds / np.maximum(v_avg, 0.1)  # Time per segment
    lap_time = np.sum(dt)
    
    return {
        "v": v,
        "v_lat": v_lat,
        "v_fwd": v_fwd,
        "v_bwd": v_bwd,
    }, lap_time