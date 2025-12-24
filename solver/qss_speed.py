from dataclasses import dataclass

@dataclass(frozen=True)
class SolverSettings:
    v0: float = 0.0
    v_min: float = 0.1
    max_iter: int = 30
    tol: float = 1e-3     # m/s
    min_points: int = 500  # Minimum points for track refinement
    refine: bool = False   # Whether to refine track resolution

# solver/qss_speed_profile.py
import numpy as np
from models.track import Track
from models.vehicle import VehicleParams
from physics.aero import drag, downforce
from physics.tyre import a_max, ax_available
from physics.powertrain import max_tractive_force, max_tractive_force_extended
from physics.resistive import rolling_resistance


def refine_track(track: Track, min_points: int = 500) -> Track:
    """
    Refine track to minimum number of points using cubic interpolation.
    
    This improves solver accuracy by:
    - Better curvature estimation at corners
    - Smoother speed transitions
    - More accurate lap time integration
    
    Args:
        track: Original track object
        min_points: Minimum number of points (default 500)
    
    Returns:
        Refined Track object (or original if already fine enough)
    """
    if len(track.s) >= min_points:
        return track
    
    from scipy.interpolate import interp1d
    from models.track import from_xy
    
    # Create finer arc-length spacing
    s_fine = np.linspace(0, track.s[-1], min_points)
    
    # Interpolate x and y coordinates
    x_interp = interp1d(track.s, track.x, kind='cubic', fill_value='extrapolate')
    y_interp = interp1d(track.s, track.y, kind='cubic', fill_value='extrapolate')
    
    x_fine = x_interp(s_fine)
    y_fine = y_interp(s_fine)
    
    # Rebuild track with finer resolution
    return from_xy(x_fine, y_fine, closed=track.closed)


def solve_qss(track: Track, vehicle: VehicleParams, 
              refine: bool = False, min_points: int = 500) -> tuple[dict, float]:
    """
    Quasi-Steady-State lap time solver.
    
    Algorithm:
    1. Calculate lateral velocity limit at each point (cornering)
    2. Forward pass: accelerate from start, limited by grip & power
    3. Backward pass: decelerate from end, limited by grip
    4. Take minimum of all three limits
    5. Integrate time
    
    Args:
        track: Track object with geometry
        vehicle: Vehicle parameters
        refine: If True, interpolate track to finer resolution for accuracy
        min_points: Minimum points when refining (default 500)
    
    Returns:
        result: {"v": velocity array, "v_lat": lateral limit, ...}
        lap_time: Total time [s]
    
    Accuracy Notes:
        - Discretization error scales as O(ds²)
        - For best accuracy, use ds ≈ 0.5-1.0 m (500-1500 points per km)
        - Enable refine=True for coarse tracks
        - For closed tracks (skidpad), enforces periodicity: v_start = v_end
    """
    # Optionally refine track for better accuracy
    if refine:
        track = refine_track(track, min_points)
    
    n = len(track.s)
    m = vehicle.m
    g = vehicle.g
    mu = vehicle.tyre.mu
    
    # =========================================
    # STEP 1: Lateral (cornering) speed limit
    # =========================================
    # At each point: a_y = v² * κ ≤ a_y_max
    # Therefore: v ≤ √(a_y_max / |κ|)
    # Also limited by: motor RPM limit if using extended powertrain

    # Get RPM-limited top speed (if using extended powertrain)
    v_max_rpm = 100.0  # Default high limit
    if hasattr(vehicle.powertrain, 'v_max_rpm'):
        v_max_rpm = vehicle.powertrain.v_max_rpm

    v_lat = np.zeros(n)
    for i in range(n):
        v_guess = 50.0  # Initial guess [m/s]

        # Iterate because downforce depends on speed
        for _ in range(10):
            Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v_guess)
            amax = a_max(mu, g, m, Fdown)

            kappa_i = abs(track.kappa[i])
            if kappa_i < 1e-6:  # Straight section
                v_lat[i] = v_max_rpm  # Limited by RPM, not grip
                break
            else:
                v_lat[i] = min(np.sqrt(amax / kappa_i), v_max_rpm)

            if abs(v_lat[i] - v_guess) < 0.01:
                break
            v_guess = v_lat[i]
    
    # =========================================
    # STEP 2: Forward pass (acceleration limit)
    # =========================================
    # Starting from rest (or entry speed), accelerate as hard as possible
    # With LOOK-AHEAD: anticipate upcoming lateral limits to prevent overshoots
    
    # For closed tracks: start at lateral limit (already cornering)
    # For open tracks: start from low speed (standing start)
    if track.closed:
        v_start = v_lat[0]  # Already at cornering speed
    else:
        v_start = min(v_lat[0], 10.0)  # Standing start
    
    v_fwd = np.zeros(n)
    v_fwd[0] = v_start
    
    # Pre-compute maximum braking capability for look-ahead
    # This helps determine if we can reach a speed and still brake to v_lat
    def max_entry_speed(v_exit: float, ds: float, v_for_aero: float) -> float:
        """Calculate maximum entry speed to brake down to v_exit over distance ds."""
        Fdown = downforce(vehicle.aero.rho, vehicle.aero.CL_A, v_for_aero)
        Fdrag = drag(vehicle.aero.rho, vehicle.aero.CD_A, v_for_aero)
        Frr = rolling_resistance(vehicle.Crr, m, g)
        amax = a_max(mu, g, m, Fdown)
        # Assume we use full grip for braking (no lateral accel during pure braking)
        ax_brake = amax + (Fdrag + Frr) / m
        # v_entry² = v_exit² + 2 * a_brake * ds
        v_entry_sq = v_exit**2 + 2 * ax_brake * ds
        return np.sqrt(max(v_entry_sq, 0))
    
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
        
        # Maximum power-limited acceleration (uses extended function for RPM limit)
        Fx_motor = max_tractive_force_extended(vehicle.powertrain, v_i)
        ax_power = (Fx_motor - Fdrag - Frr) / m
        
        # Take minimum (most limiting)
        ax = min(ax_grip, ax_power)
        ax = max(ax, 0)  # Can't accelerate backwards
        
        # Kinematic equation: v² = v₀² + 2*a*ds
        v_next_sq = v_i**2 + 2 * ax * ds
        v_kinematic = np.sqrt(max(v_next_sq, 0))
        
        # LOOK-AHEAD: Check if we can brake to upcoming lateral limits
        # Look ahead several segments to find the most restrictive constraint
        v_max_lookahead = v_kinematic
        lookahead_distance = 0.0
        
        for j in range(i + 1, min(i + 20, n)):  # Look ahead up to 20 segments
            lookahead_distance += track.ds[j - 1] if j > i + 1 else ds
            v_lat_ahead = v_lat[j]
            
            # What's the max speed we can have NOW and still brake to v_lat_ahead?
            v_max_entry = max_entry_speed(v_lat_ahead, lookahead_distance, v_i)
            v_max_lookahead = min(v_max_lookahead, v_max_entry)
        
        # Final speed is minimum of kinematic result, lateral limit, and look-ahead
        v_fwd[i + 1] = min(v_kinematic, v_lat[i + 1], v_max_lookahead)
    
    # =========================================
    # STEP 3: Backward pass (braking limit)
    # =========================================
    # From end, work backwards: what speed could we brake from?
    # For closed tracks: end speed must equal start speed (periodicity)
    
    v_bwd = np.zeros(n)
    if track.closed:
        # For closed tracks, end speed = start speed (we return to same point)
        v_bwd[-1] = min(v_fwd[-1], v_lat[-1], v_start)
    else:
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