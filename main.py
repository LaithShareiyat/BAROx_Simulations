import inquirer
import yaml
import os

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_standard_vehicle(config_path: str = "config/vehicle_2025.yaml") -> dict:
    """Load standard vehicle parameters from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
    print("=" * 50 + "\n")


def run_skidpad_simulation(config: dict):
    """Run the skidpad simulation."""
    from events.skidpad import build_skidpad_track, plot_skidpad
    
    print("\n" + "=" * 50)
    print("SKIDPAD SIMULATION")
    print("=" * 50)
    
    track = build_skidpad_track()
    print(f"Track length: {track.s[-1]:.2f} m")
    print(f"Curvature: {track.kappa[0]:.4f} m⁻¹")
    print(f"Turn radius: {1/abs(track.kappa[0]):.2f} m")
    
    # TODO: Add lap time simulation here when solver is implemented
    # result, lap_time = run(track, vehicle)
    # print(f"Lap time: {lap_time:.3f} s")
    
    plot_skidpad(track)


def run_autocross_simulation(config: dict):
    """Run the autocross simulation."""
    from events.autocross_generator import (
        build_standard_autocross, 
        plot_autocross, 
        validate_autocross,
        MAX_TRACK_LENGTH
    )
    import numpy as np
    
    print("\n" + "=" * 50)
    print("AUTOCROSS SIMULATION")
    print("=" * 50)
    
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
    
    # TODO: Add lap time simulation here when solver is implemented
    # result, lap_time = run(track, vehicle)
    # print(f"Lap time: {lap_time:.3f} s")
    
    plot_autocross(track, metadata=metadata)


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
                          ('Standard (vehicle_2025.yaml)', 'standard'),
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
        print("\nUsing standard vehicle parameters from vehicle_2025.yaml")
    else:
        config = get_custom_vehicle_params(defaults)
    
    # Print the vehicle parameters being used
    print_vehicle_params(config)
    
    # =========================================
    # Run selected simulation(s)
    # =========================================
    if simulation_type == 'both':
        run_autocross_simulation(config)
        run_skidpad_simulation(config)
    elif simulation_type == 'autocross':
        run_autocross_simulation(config)
    elif simulation_type == 'skidpad':
        run_skidpad_simulation(config)
    
    print("\n" + "=" * 50)
    print("    SIMULATION COMPLETE")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()