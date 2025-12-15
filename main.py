import inquirer
import yaml
import os
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vehicle import VehicleParams, AeroParams, TyreParamsMVP, EVPowertrainMVP, BatteryParams
from solver.qss_speed import solve_qss
from solver.metrics import lap_time, channels, energy_consumption
from solver.battery import simulate_battery, validate_battery_capacity, required_battery_capacity


def load_standard_vehicle(config_path: str = "config/default.yaml") -> dict:
    """Load standard vehicle parameters from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_vehicle_from_config(config: dict) -> VehicleParams:
    """Create a VehicleParams object from config dictionary."""
    aero = AeroParams(
        rho=config['aero']['rho'],
        CD_A=config['aero']['CD_A'],
        CL_A=config['aero']['CL_A'],
    )
    tyre = TyreParamsMVP(
        mu=config['tyre']['mu'],
    )
    powertrain = EVPowertrainMVP(
        P_max=config['powertrain']['P_max_kW'] * 1000,  # Convert kW to W
        Fx_max=config['powertrain']['Fx_max_N'],
    )
    
    # Create battery params if present in config (no regen)
    battery = None
    if 'battery' in config:
        battery = BatteryParams(
            capacity_kWh=config['battery']['capacity_kWh'],
            initial_soc=config['battery'].get('initial_soc', 1.0),
            min_soc=config['battery'].get('min_soc', 0.1),
            max_discharge_kW=config['battery'].get('max_discharge_kW', 80.0),
            eta_discharge=config['battery'].get('eta_discharge', 0.95),
        )
    
    return VehicleParams(
        m=config['vehicle']['mass_kg'],
        g=config['vehicle']['g'],
        Crr=config['vehicle']['Crr'],
        aero=aero,
        tyre=tyre,
        powertrain=powertrain,
        battery=battery,
    )


def get_custom_vehicle_params(defaults: dict) -> dict:
    """Prompt user for custom vehicle parameters."""
    
    print("\n" + "=" * 50)
    print("CUSTOM VEHICLE PARAMETERS")
    print("Press Enter to keep default value shown in brackets")
    print("=" * 50)
    
    # Vehicle parameters
    print("\n--- Vehicle ---")
    questions = [
        inquirer.Text('mass_kg', 
                      message=f"Mass [kg] ({defaults['vehicle']['mass_kg']})",
                      default=str(defaults['vehicle']['mass_kg'])),
        inquirer.Text('g', 
                      message=f"Gravity [m/s²] ({defaults['vehicle']['g']})",
                      default=str(defaults['vehicle']['g'])),
        inquirer.Text('Crr', 
                      message=f"Rolling resistance coefficient ({defaults['vehicle']['Crr']})",
                      default=str(defaults['vehicle']['Crr'])),
    ]
    vehicle_answers = inquirer.prompt(questions)
    
    # Aero parameters
    print("\n--- Aerodynamics ---")
    questions = [
        inquirer.Text('rho', 
                      message=f"Air density [kg/m³] ({defaults['aero']['rho']})",
                      default=str(defaults['aero']['rho'])),
        inquirer.Text('CD_A', 
                      message=f"Drag coefficient × Area (CD×A) [m²] ({defaults['aero']['CD_A']})",
                      default=str(defaults['aero']['CD_A'])),
        inquirer.Text('CL_A', 
                      message=f"Lift coefficient × Area (CL×A) [m²] ({defaults['aero']['CL_A']})",
                      default=str(defaults['aero']['CL_A'])),
    ]
    aero_answers = inquirer.prompt(questions)
    
    # Tyre parameters
    print("\n--- Tyre ---")
    questions = [
        inquirer.Text('mu', 
                      message=f"Tyre friction coefficient (μ) ({defaults['tyre']['mu']})",
                      default=str(defaults['tyre']['mu'])),
    ]
    tyre_answers = inquirer.prompt(questions)
    
    # Powertrain parameters
    print("\n--- Powertrain ---")
    questions = [
        inquirer.Text('P_max_kW', 
                      message=f"Max power [kW] ({defaults['powertrain']['P_max_kW']})",
                      default=str(defaults['powertrain']['P_max_kW'])),
        inquirer.Text('Fx_max_N', 
                      message=f"Max tractive force [N] ({defaults['powertrain']['Fx_max_N']})",
                      default=str(defaults['powertrain']['Fx_max_N'])),
    ]
    powertrain_answers = inquirer.prompt(questions)
    
    # Battery parameters (no regen)
    print("\n--- Battery (No Regen) ---")
    battery_defaults = defaults.get('battery', {
        'capacity_kWh': 6.5,
        'initial_soc': 1.0,
        'min_soc': 0.1,
        'max_discharge_kW': 80.0,
        'eta_discharge': 0.95,
    })
    questions = [
        inquirer.Text('capacity_kWh', 
                      message=f"Battery capacity [kWh] ({battery_defaults['capacity_kWh']})",
                      default=str(battery_defaults['capacity_kWh'])),
        inquirer.Text('initial_soc', 
                      message=f"Initial SoC [0-1] ({battery_defaults['initial_soc']})",
                      default=str(battery_defaults['initial_soc'])),
        inquirer.Text('min_soc', 
                      message=f"Minimum SoC [0-1] ({battery_defaults['min_soc']})",
                      default=str(battery_defaults['min_soc'])),
        inquirer.Text('max_discharge_kW', 
                      message=f"Max discharge power [kW] ({battery_defaults['max_discharge_kW']})",
                      default=str(battery_defaults['max_discharge_kW'])),
    ]
    battery_answers = inquirer.prompt(questions)
    
    # Build config dict
    config = {
        'vehicle': {
            'mass_kg': float(vehicle_answers['mass_kg']),
            'g': float(vehicle_answers['g']),
            'Crr': float(vehicle_answers['Crr']),
        },
        'aero': {
            'rho': float(aero_answers['rho']),
            'CD_A': float(aero_answers['CD_A']),
            'CL_A': float(aero_answers['CL_A']),
        },
        'tyre': {
            'mu': float(tyre_answers['mu']),
        },
        'powertrain': {
            'P_max_kW': float(powertrain_answers['P_max_kW']),
            'Fx_max_N': float(powertrain_answers['Fx_max_N']),
        },
        'battery': {
            'capacity_kWh': float(battery_answers['capacity_kWh']),
            'initial_soc': float(battery_answers['initial_soc']),
            'min_soc': float(battery_answers['min_soc']),
            'max_discharge_kW': float(battery_answers['max_discharge_kW']),
            'eta_discharge': battery_defaults.get('eta_discharge', 0.95),
        }
    }
    
    return config


def print_vehicle_params(config: dict):
    """Print vehicle parameters in a formatted table."""
    print("\n" + "=" * 50)
    print("VEHICLE PARAMETERS")
    print("=" * 50)
    
    print("\n--- Vehicle ---")
    print(f"  Mass:                  {config['vehicle']['mass_kg']} kg")
    print(f"  Gravity:               {config['vehicle']['g']} m/s²")
    print(f"  Rolling resistance:    {config['vehicle']['Crr']}")
    
    print("\n--- Aerodynamics ---")
    print(f"  Air density:           {config['aero']['rho']} kg/m³")
    print(f"  CD × A:                {config['aero']['CD_A']} m²")
    print(f"  CL × A:                {config['aero']['CL_A']} m²")
    
    print("\n--- Tyre ---")
    print(f"  Friction coeff (μ):    {config['tyre']['mu']}")
    
    print("\n--- Powertrain ---")
    print(f"  Max power:             {config['powertrain']['P_max_kW']} kW")
    print(f"  Max tractive force:    {config['powertrain']['Fx_max_N']} N")
    
    if 'battery' in config:
        print("\n--- Battery (No Regen) ---")
        print(f"  Capacity:              {config['battery']['capacity_kWh']} kWh")
        print(f"  Initial SoC:           {config['battery']['initial_soc']:.0%}")
        print(f"  Minimum SoC:           {config['battery']['min_soc']:.0%}")
        print(f"  Max discharge:         {config['battery']['max_discharge_kW']} kW")
        print(f"  Discharge efficiency:  {config['battery'].get('eta_discharge', 0.95):.0%}")
    
    print("=" * 50 + "\n")


def compute_metrics(track, v: np.ndarray, vehicle: VehicleParams) -> dict:
    """Compute performance metrics from velocity profile."""
    t = lap_time(track, v)
    ax, ay = channels(track, v)
    
    # Handle energy consumption - may return NaN if issues
    try:
        energy = energy_consumption(track, v, vehicle)
        energy_kwh = energy.get('E_net_kWh', np.nan)
    except Exception:
        energy_kwh = np.nan
    
    # Handle edge cases in velocity
    v_positive = v[v > 0.1]
    min_speed = np.min(v_positive) if len(v_positive) > 0 else 0.0
    
    # Handle NaN in acceleration arrays
    ax_clean = ax[~np.isnan(ax)]
    ay_clean = ay[~np.isnan(ay)]
    
    return {
        'lap_time': t,
        'avg_speed': np.nanmean(v),
        'max_speed': np.nanmax(v),
        'min_speed': min_speed,
        'max_ax': np.max(ax_clean) if len(ax_clean) > 0 else 0.0,
        'min_ax': np.min(ax_clean) if len(ax_clean) > 0 else 0.0,
        'max_ay': np.max(np.abs(ay_clean)) if len(ay_clean) > 0 else 0.0,
        'energy_consumed_kWh': energy_kwh if not np.isnan(energy_kwh) else 0.0,
    }


def run_skidpad_simulation(config: dict):
    """Run the skidpad simulation with lap time calculation."""
    from events.skidpad import (
        build_skidpad_track, 
        plot_skidpad, 
        skidpad_time_from_single_circle,
        SKIDPAD_CENTRE_RADIUS,
        TRACK_WIDTH
    )
    
    print("\n" + "=" * 50)
    print("SKIDPAD SIMULATION")
    print("=" * 50)
    
    # Build track (single circle for timing)
    track = build_skidpad_track()
    circle_length = 2 * np.pi * SKIDPAD_CENTRE_RADIUS
    
    print(f"\nTrack Configuration:")
    print(f"  Centre-line radius:    {SKIDPAD_CENTRE_RADIUS:.3f} m")
    print(f"  Circle circumference:  {circle_length:.2f} m")
    print(f"  Track width:           {TRACK_WIDTH} m")
    
    # Create vehicle
    vehicle = create_vehicle_from_config(config)
    
    # Solve for velocity profile
    print("\nSolving for velocity profile...")
    result, t_lap = solve_qss(track, vehicle)
    v = result['v']
    
    # Get timing breakdown
    timing = skidpad_time_from_single_circle(t_lap)
    
    # Compute additional metrics
    metrics = compute_metrics(track, v, vehicle)
    
    # Print results
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"\n  Timing:")
    print(f"    Single circle time:  {timing['t_official']:.3f} s")
    print(f"    Full run time:       {timing['t_full_run']:.3f} s (4 circles)")
    print(f"\n  Performance:")
    print(f"    Average speed:       {timing['avg_speed']:.2f} m/s ({timing['avg_speed']*3.6:.1f} km/h)")
    print(f"    Max lateral accel:   {metrics['max_ay']:.2f} m/s² ({metrics['max_ay']/9.81:.2f} g)")
    print(f"    Energy (1 circle):   {metrics['energy_consumed_kWh']*1000:.1f} Wh")
    print("-" * 50)
    
    # Add timing to metrics for return
    metrics['t_official'] = timing['t_official']
    metrics['t_full_run'] = timing['t_full_run']
    
    # Plot with velocity colouring
    plot_skidpad(track, v=v)
    
    return metrics


def run_autocross_simulation(config: dict):
    """Run the autocross simulation with lap time calculation and battery tracking."""
    from events.autocross_generator import (
        build_standard_autocross, 
        plot_autocross, 
        validate_autocross,
        MAX_TRACK_LENGTH
    )
    from plots.plots import plot_battery_state, plot_soc_on_track
    
    print("\n" + "=" * 50)
    print("AUTOCROSS SIMULATION")
    print("=" * 50)
    
    # Build track
    print("\nBuilding track...")
    track, metadata = build_standard_autocross()
    
    print(f"Track length: {track.s[-1]:.1f} m")
    print(f"Number of points: {len(track.x)}")
    
    max_kappa = np.max(np.abs(track.kappa[track.kappa != 0]))
    print(f"Min turn radius: {1/max_kappa:.2f} m")
    print(f"Number of slalom cones: {len(metadata['slalom_cones_x'])}")
    
    # Validate track
    validation = validate_autocross(track, metadata)
    if validation["valid"]:
        print("✓ Track passes all validation checks")
    else:
        print("✗ Track validation FAILED:")
        for err in validation["errors"]:
            print(f"  - {err}")
    
    for warn in validation.get("warnings", []):
        print(f"  Warning: {warn}")
    
    if track.s[-1] <= MAX_TRACK_LENGTH:
        print(f"✓ Track length {track.s[-1]:.1f}m ≤ {MAX_TRACK_LENGTH}m maximum")
    else:
        print(f"✗ Track length {track.s[-1]:.1f}m exceeds {MAX_TRACK_LENGTH}m maximum")
    
    # Create vehicle
    vehicle = create_vehicle_from_config(config)
    
    # Solve for velocity profile
    print("\nSolving for velocity profile...")
    result, t_lap = solve_qss(track, vehicle)
    v = result['v']
    
    # Compute metrics
    metrics = compute_metrics(track, v, vehicle)
    
    # Print results
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"  Lap time:              {metrics['lap_time']:.3f} s")
    print(f"  Average speed:         {metrics['avg_speed']:.2f} m/s ({metrics['avg_speed']*3.6:.1f} km/h)")
    print(f"  Max speed:             {metrics['max_speed']:.2f} m/s ({metrics['max_speed']*3.6:.1f} km/h)")
    print(f"  Min speed:             {metrics['min_speed']:.2f} m/s ({metrics['min_speed']*3.6:.1f} km/h)")
    print(f"  Max longitudinal accel:{metrics['max_ax']:.2f} m/s² ({metrics['max_ax']/9.81:.2f} g)")
    print(f"  Max braking decel:     {abs(metrics['min_ax']):.2f} m/s² ({abs(metrics['min_ax'])/9.81:.2f} g)")
    print(f"  Max lateral accel:     {metrics['max_ay']:.2f} m/s² ({metrics['max_ay']/9.81:.2f} g)")
    print(f"  Energy consumed:       {metrics['energy_consumed_kWh']*1000:.1f} Wh")
    print("-" * 50)
    
    # =========================================
    # BATTERY VALIDATION (No Regen)
    # =========================================
    if vehicle.battery is not None:
        print("\n" + "-" * 50)
        print("BATTERY ANALYSIS (No Regen)")
        print("-" * 50)
        
        # Validate battery capacity
        battery_validation = validate_battery_capacity(track, v, vehicle)
        
        # Print battery results
        if battery_validation.sufficient:
            print(f"  ✓ Battery capacity SUFFICIENT")
        else:
            print(f"  ✗ Battery capacity INSUFFICIENT")
        
        print(f"\n  Battery Status:")
        print(f"    Initial SoC:         {vehicle.battery.initial_soc:.1%}")
        print(f"    Final SoC:           {battery_validation.final_soc:.1%}")
        print(f"    Minimum SoC:         {battery_validation.min_soc:.1%} (at {battery_validation.min_soc_distance:.1f}m)")
        print(f"    Min allowed SoC:     {vehicle.battery.min_soc:.1%}")
        
        print(f"\n  Energy:")
        print(f"    Capacity:            {vehicle.battery.capacity_kWh:.2f} kWh ({vehicle.battery.capacity_kWh*1000:.0f} Wh)")
        usable = vehicle.battery.capacity_kWh * (vehicle.battery.initial_soc - vehicle.battery.min_soc)
        print(f"    Usable capacity:     {usable:.2f} kWh ({usable*1000:.0f} Wh)")
        print(f"    Energy consumed:     {battery_validation.total_energy_kWh*1000:.1f} Wh")
        print(f"    Remaining usable:    {(usable - battery_validation.total_energy_kWh)*1000:.1f} Wh")
        
        print(f"\n  Power:")
        print(f"    Peak discharge:      {battery_validation.peak_power_kW:.1f} kW")
        print(f"    Average discharge:   {battery_validation.avg_power_kW:.1f} kW")
        
        # Print warnings and errors
        if battery_validation.warnings:
            print(f"\n  Warnings:")
            for warn in battery_validation.warnings:
                print(f"    ⚠ {warn}")
        
        if battery_validation.errors:
            print(f"\n  Errors:")
            for err in battery_validation.errors:
                print(f"    ✗ {err}")
        
        # Calculate required capacity if insufficient
        if not battery_validation.sufficient:
            req_capacity = required_battery_capacity(track, v, vehicle, safety_margin=0.2)
            print(f"\n  Recommended minimum capacity: {req_capacity:.2f} kWh")
        
        print("-" * 50)
        
        # Store battery results in metrics
        metrics['battery_validation'] = battery_validation
        metrics['battery_sufficient'] = battery_validation.sufficient
        metrics['final_soc'] = battery_validation.final_soc
        metrics['min_soc'] = battery_validation.min_soc
        
        # Simulate battery for plotting
        battery_state = simulate_battery(track, v, vehicle)
        
        # Plot battery state
        plot_battery_state(track, battery_state, vehicle, battery_validation,
                          title="Autocross Battery Analysis (No Regen)")
        
        # Plot SoC on track
        plot_soc_on_track(track, battery_state, vehicle, title="Autocross - SoC Map")
    
    # Plot with velocity colouring
    plot_autocross(track, v=v, metadata=metadata)
    
    return metrics


def main():
    """Main entry point with interactive menu."""
    
    print("\n" + "=" * 50)
    print("    BAROx FORMULA STUDENT SIMULATION")
    print("=" * 50 + "\n")
    
    # =========================================
    # Question 1: Select simulation type
    # =========================================
    simulation_question = [
        inquirer.List('simulation',
                      message="What simulation would you like to run?",
                      choices=[
                          ('Both (Autocross + Skidpad)', 'both'),
                          ('Autocross (Sprint)', 'autocross'),
                          ('Skidpad', 'skidpad'),
                      ],
                      ),
    ]
    simulation_answer = inquirer.prompt(simulation_question)
    
    if simulation_answer is None:
        print("Cancelled.")
        return
    
    simulation_type = simulation_answer['simulation']
    
    # =========================================
    # Question 2: Select vehicle parameters
    # =========================================
    vehicle_question = [
        inquirer.List('vehicle_params',
                      message="Vehicle Parameters:",
                      choices=[
                          ('Standard (default.yaml)', 'standard'),
                          ('Custom', 'custom'),
                      ],
                      ),
    ]
    vehicle_answer = inquirer.prompt(vehicle_question)
    
    if vehicle_answer is None:
        print("Cancelled.")
        return
    
    # Load or get vehicle parameters
    defaults = load_standard_vehicle()
    
    if vehicle_answer['vehicle_params'] == 'standard':
        config = defaults
        print("\nUsing standard vehicle parameters from default.yaml")
    else:
        config = get_custom_vehicle_params(defaults)
    
    # Print the vehicle parameters being used
    print_vehicle_params(config)
    
    # =========================================
    # Run selected simulation(s)
    # =========================================
    results = {}
    
    if simulation_type == 'both':
        results['autocross'] = run_autocross_simulation(config)
        results['skidpad'] = run_skidpad_simulation(config)
        
        # Print combined summary
        print("\n" + "=" * 50)
        print("    COMBINED RESULTS SUMMARY")
        print("=" * 50)
        print(f"  Autocross lap time:    {results['autocross']['lap_time']:.3f} s")
        print(f"  Skidpad lap time:      {results['skidpad']['lap_time']:.3f} s")
        if 'battery_sufficient' in results['autocross']:
            status = "✓ Yes" if results['autocross']['battery_sufficient'] else "✗ No"
            print(f"  Battery sufficient:    {status}")
        print("=" * 50)
        
    elif simulation_type == 'autocross':
        results['autocross'] = run_autocross_simulation(config)
    elif simulation_type == 'skidpad':
        results['skidpad'] = run_skidpad_simulation(config)
    
    print("\n" + "=" * 50)
    print("    SIMULATION COMPLETE")
    print("=" * 50 + "\n")
    
    return results


if __name__ == "__main__":
    main()