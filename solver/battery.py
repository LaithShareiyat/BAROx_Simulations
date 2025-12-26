import numpy as np
from dataclasses import dataclass
from models.track import Track
from models.vehicle import VehicleParams, BatteryParams
from physics.aero import drag
from physics.resistive import rolling_resistance


@dataclass
class BatteryState:
    """Battery state at each track point (supports regen)."""
    soc: np.ndarray             # State of charge [0-1] at each point
    energy_used_kWh: np.ndarray # Cumulative net energy used [kWh] (can decrease with regen)
    power_kW: np.ndarray        # Instantaneous power [kW] (positive = discharge, negative = regen charge)


@dataclass
class BatteryValidation:
    """Results of battery capacity validation (supports regen)."""
    sufficient: bool            # True if battery has enough capacity
    final_soc: float            # Final state of charge
    min_soc: float              # Minimum SoC during lap
    min_soc_index: int          # Track index where min SoC occurs
    min_soc_distance: float     # Distance [m] where min SoC occurs
    total_energy_kWh: float     # Net energy consumed [kWh] (may be reduced by regen)
    peak_power_kW: float        # Peak discharge power [kW]
    avg_power_kW: float         # Average power during discharge [kW]
    warnings: list              # List of warning messages
    errors: list                # List of error messages


def calculate_power_profile(track: Track, v: np.ndarray, vehicle: VehicleParams,
                            eta_motor: float = None) -> np.ndarray:
    """
    Calculate instantaneous electrical power at each segment.

    Supports regenerative braking if enabled in battery parameters.
    Positive power = discharge (accelerating/cruising)
    Negative power = charge (regenerative braking)

    Args:
        track: Track object
        v: Velocity profile [m/s]
        vehicle: Vehicle parameters
        eta_motor: Motor efficiency (if None, uses powertrain.motor_efficiency or 0.85)

    Returns:
        power_kW: Power at each segment [kW] (positive = discharge, negative = regen)
    """
    m = vehicle.m
    g = vehicle.g
    battery = vehicle.battery

    # Get motor efficiency from powertrain if available, otherwise use parameter or default
    if eta_motor is None:
        if hasattr(vehicle.powertrain, 'motor_efficiency'):
            eta_motor = vehicle.powertrain.motor_efficiency
        else:
            eta_motor = 0.85

    # Check if regen is enabled
    regen_enabled = battery is not None and battery.regen_enabled

    n_segments = len(track.ds)
    power_kW = np.zeros(n_segments)

    for i in range(n_segments):
        # Average velocity for segment
        v_avg = 0.5 * (v[i] + v[i + 1])
        v_avg = max(v_avg, 0.1)  # Avoid division by zero

        # Acceleration for this segment
        dv = v[i + 1] - v[i]
        ds = track.ds[i]
        dt = ds / v_avg
        ax = dv / dt if dt > 0 else 0

        # Resistance forces at segment average speed
        Fdrag = drag(vehicle.aero.rho, vehicle.aero.CD_A, v_avg)
        Frr = rolling_resistance(vehicle.Crr, m, g)

        # Required tractive force at wheels
        Fx_wheel = m * ax + Fdrag + Frr

        if Fx_wheel > 0:
            # Accelerating/cruising - discharge battery
            P_mech = Fx_wheel * v_avg
            P_elec = P_mech / eta_motor
            power_kW[i] = P_elec / 1000.0  # Convert to kW
        elif regen_enabled and Fx_wheel < 0:
            # Braking with regen enabled - recover energy
            # Braking force available for regen (negative Fx_wheel means braking)
            F_brake = abs(Fx_wheel)

            # Apply capture percentage (how much braking uses regen vs friction)
            F_regen = F_brake * (battery.regen_capture_percent / 100.0)

            # Mechanical power available from braking [W]
            P_mech_brake = F_regen * v_avg

            # Apply regen efficiency (motor generating + battery charging losses)
            P_regen = P_mech_brake * battery.eta_regen

            # Cap at maximum regen power
            P_regen = min(P_regen, battery.max_regen_kW * 1000.0)

            # Negative power = charging the battery
            power_kW[i] = -P_regen / 1000.0
        else:
            # Braking without regen - no power flow
            power_kW[i] = 0.0

    return power_kW


def simulate_battery(track: Track, v: np.ndarray, vehicle: VehicleParams,
                     eta_motor: float = None) -> BatteryState:
    """
    Simulate battery state throughout the lap (supports regen).

    Args:
        track: Track object
        v: Velocity profile [m/s]
        vehicle: Vehicle parameters
        eta_motor: Motor efficiency (if None, uses powertrain.motor_efficiency or 0.85)

    Returns:
        BatteryState with SoC and energy arrays
    """
    if vehicle.battery is None:
        raise ValueError("Vehicle has no battery parameters defined")

    battery = vehicle.battery
    n = len(track.s)
    n_segments = len(track.ds)

    # Get motor efficiency from powertrain if not specified
    if eta_motor is None:
        if hasattr(vehicle.powertrain, 'motor_efficiency'):
            eta_motor = vehicle.powertrain.motor_efficiency
        else:
            eta_motor = 0.85

    # Calculate power profile (positive = discharge, negative = regen charge)
    power_kW = calculate_power_profile(track, v, vehicle, eta_motor)

    # Calculate current-limited power (P = V × I, FS rules: 500A max)
    max_power_from_current = battery.max_power_from_current_kW

    # Apply power limits and efficiency
    power_limited = np.zeros(n_segments)
    power_battery = np.zeros(n_segments)

    for i in range(n_segments):
        if power_kW[i] > 0:
            # Discharging - limit by BOTH power limit AND current limit
            # Current limit: P_max = V × I_max (e.g., 400V × 500A = 200kW)
            discharge_limit = min(battery.max_discharge_kW, max_power_from_current)
            power_limited[i] = min(power_kW[i], discharge_limit)
            # Apply discharge efficiency (more power drawn from battery than delivered)
            power_battery[i] = power_limited[i] / battery.eta_discharge
        elif power_kW[i] < 0:
            # Charging (regen) - already limited by max_regen_kW in calculate_power_profile
            power_limited[i] = power_kW[i]  # Negative value
            # For charging, efficiency is already applied in calculate_power_profile
            # The negative power represents energy going into the battery
            power_battery[i] = power_limited[i]
        else:
            power_limited[i] = 0.0
            power_battery[i] = 0.0

    # Calculate time for each segment
    v_avg = 0.5 * (v[:-1] + v[1:])
    v_avg = np.maximum(v_avg, 0.1)
    dt = track.ds / v_avg  # Time per segment [s]

    # Energy per segment [kWh] (positive = used, negative = recovered)
    energy_segment_kWh = power_battery * dt / 3600.0

    # Cumulative energy tracking
    energy_used_kWh = np.zeros(n)
    soc = np.zeros(n)

    soc[0] = battery.initial_soc

    for i in range(n_segments):
        # Accumulate net energy (can decrease with regen)
        energy_used_kWh[i + 1] = energy_used_kWh[i] + energy_segment_kWh[i]

        # Update SoC (increases when energy_used decreases due to regen)
        soc[i + 1] = battery.initial_soc - (energy_used_kWh[i + 1] / battery.capacity_kWh)

        # Clamp SoC to valid range [0, 1] - can't overcharge or go negative
        soc[i + 1] = np.clip(soc[i + 1], 0.0, 1.0)

    # Extend power array to match track points (pad last value)
    power_kW_full = np.zeros(n)
    power_kW_full[:-1] = power_limited
    power_kW_full[-1] = power_limited[-1]

    return BatteryState(
        soc=soc,
        energy_used_kWh=energy_used_kWh,
        power_kW=power_kW_full,
    )


def validate_battery_capacity(track: Track, v: np.ndarray, vehicle: VehicleParams,
                              eta_motor: float = None) -> BatteryValidation:
    """
    Validate if battery capacity is sufficient for the lap (supports regen).

    Args:
        track: Track object
        v: Velocity profile [m/s]
        vehicle: Vehicle parameters
        eta_motor: Motor efficiency (if None, uses powertrain.motor_efficiency or 0.85)

    Returns:
        BatteryValidation with detailed results
    """
    if vehicle.battery is None:
        return BatteryValidation(
            sufficient=False,
            final_soc=0.0,
            min_soc=0.0,
            min_soc_index=0,
            min_soc_distance=0.0,
            total_energy_kWh=0.0,
            peak_power_kW=0.0,
            avg_power_kW=0.0,
            warnings=[],
            errors=["No battery parameters defined"]
        )
    
    battery = vehicle.battery
    warnings = []
    errors = []
    
    # Simulate battery
    state = simulate_battery(track, v, vehicle, eta_motor)

    # Find minimum SoC (may occur mid-lap with regen, or at end without regen)
    min_soc_index = np.argmin(state.soc)
    min_soc = state.soc[min_soc_index]
    min_soc_distance = track.s[min_soc_index]
    
    # Final state
    final_soc = state.soc[-1]
    
    # Energy totals
    total_energy = state.energy_used_kWh[-1]
    
    # Power statistics
    peak_power = np.max(state.power_kW)
    # Average power when actually discharging (power > 0)
    discharging_power = state.power_kW[state.power_kW > 0]
    avg_power = np.mean(discharging_power) if len(discharging_power) > 0 else 0.0
    
    # Check if sufficient
    sufficient = min_soc >= battery.min_soc
    
    # Generate warnings and errors
    if min_soc < battery.min_soc:
        errors.append(
            f"Battery depleted below minimum SoC ({min_soc:.1%} < {battery.min_soc:.1%}) "
            f"at {min_soc_distance:.1f}m"
        )
    
    if min_soc < 0.2:
        warnings.append(f"Low minimum SoC: {min_soc:.1%}")
    
    if peak_power > battery.max_discharge_kW * 0.95:
        warnings.append(f"Peak power near limit: {peak_power:.1f} kW")
    
    # Check usable capacity
    usable_capacity = battery.capacity_kWh * (battery.initial_soc - battery.min_soc)
    if total_energy > usable_capacity * 0.9:
        warnings.append("Energy consumption approaching available capacity")
    
    if total_energy > usable_capacity:
        errors.append(
            f"Required energy ({total_energy:.3f} kWh) exceeds usable capacity "
            f"({usable_capacity:.3f} kWh)"
        )
    
    return BatteryValidation(
        sufficient=sufficient,
        final_soc=final_soc,
        min_soc=min_soc,
        min_soc_index=min_soc_index,
        min_soc_distance=min_soc_distance,
        total_energy_kWh=total_energy,
        peak_power_kW=peak_power,
        avg_power_kW=avg_power,
        warnings=warnings,
        errors=errors
    )


def required_battery_capacity(track: Track, v: np.ndarray, vehicle: VehicleParams,
                              safety_margin: float = 0.2,
                              eta_motor: float = None) -> float:
    """
    Calculate minimum required battery capacity for the lap (supports regen).

    Args:
        track: Track object
        v: Velocity profile [m/s]
        vehicle: Vehicle parameters
        safety_margin: Additional capacity margin [0-1]
        eta_motor: Motor efficiency (if None, uses powertrain.motor_efficiency or 0.85)

    Returns:
        Required battery capacity [kWh]
    """
    # Calculate power profile (positive = discharge, negative = regen)
    power_kW = calculate_power_profile(track, v, vehicle, eta_motor)

    # Time per segment
    v_avg = 0.5 * (v[:-1] + v[1:])
    v_avg = np.maximum(v_avg, 0.1)
    dt = track.ds / v_avg

    # Energy per segment [kWh] (positive = used, negative = recovered)
    energy_kWh = power_kW * dt / 3600.0

    # Net total energy (reduced by regen if enabled)
    total_energy = np.sum(energy_kWh)
    
    # Add safety margin
    required = total_energy * (1 + safety_margin)
    
    # Account for min_soc if battery params exist
    if vehicle.battery is not None:
        usable_fraction = 1.0 - vehicle.battery.min_soc
        required = required / usable_fraction

    return max(required, 0.1)  # Minimum 0.1 kWh