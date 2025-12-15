import yaml
from pathlib import Path

from models.vehicle import VehicleParams, AeroParams, TyreParamsMVP, EVPowertrainMVP
from events.skidpad import build_skidpad_track
from events.autocross_generator import build_autocross_from_segments
from events.run_event import run
from solver.metrics import energy_consumption
from plots.plots import plot_track, plot_speed_profile, plot_gg_diagram

def load_vehicle_from_yaml(path: str) -> VehicleParams:
    """Load vehicle parameters from YAML config."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    aero = AeroParams(
        rho=cfg['aero']['rho'],
        CD_A=cfg['aero']['CD_A'],
        CL_A=cfg['aero']['CL_A']
    )
    tyre = TyreParamsMVP(mu=cfg['tyre']['mu'])
    powertrain = EVPowertrainMVP(
        P_max=cfg['powertrain']['P_max_kW'] * 1000,  # Convert to W
        Fx_max=cfg['powertrain']['Fx_max_N']
    )
    
    return VehicleParams(
        m=cfg['vehicle']['mass_kg'],
        g=cfg['vehicle']['g'],
        Crr=cfg['vehicle']['Crr'],
        aero=aero,
        tyre=tyre,
        powertrain=powertrain
    )

def main():
    # Load vehicle config
    config_path = Path(__file__).parent / "config" / "vehicle_2025.yaml"
    vehicle = load_vehicle_from_yaml(config_path)
    
    print("=" * 50)
    print("BAROx Lap Time Simulation - MVP")
    print("=" * 50)
    print(f"Vehicle mass: {vehicle.m} kg")
    print(f"Tyre mu: {vehicle.tyre.mu}")
    print(f"Max power: {vehicle.powertrain.P_max/1000:.1f} kW")
    print()
    
    # Run skidpad event
    print("Skidpad Event Details")
    skidpad_track = build_skidpad_track(radius=7.625)  # FS rules: 15.25m diameter
    result, lap_time = run(skidpad_track, vehicle)
    
    # Calculate metrics (kept for later use)
    avg_speed = result['v'].mean()
    max_lat_g = result['v'].max()**2 / (7.625 * 9.81)
    
    print(f"Skidpad lap time: {lap_time:.3f} s")
    print()

    # Example autocross
    print("Autocross (Sprint) Event Details")
    autocross_segments = [
        {"type": "straight", "length": 75},
        {"type": "arc", "radius": 15, "angle_deg": 90},
        {"type": "straight", "length": 30},
        {"type": "arc", "radius": 9, "angle_deg": -180},
        {"type": "straight", "length": 50},
        {"type": "arc", "radius": 12, "angle_deg": 120},
    ]
    autocross_track = build_autocross_from_segments(autocross_segments)

    out, T = run(autocross_track, vehicle)
    
    # Calculate energy (kept for later use)
    energy = energy_consumption(autocross_track, out["v"], vehicle)

    print(f"Autocross lap time: {T:.3f} s")
    print()

    # Visualize
    plot_track(autocross_track, out["v"], "Autocross - Speed Map")
    plot_speed_profile(autocross_track, out["v"], out["v_lat"])
    plot_gg_diagram(autocross_track, out["v"], vehicle)

if __name__ == "__main__":
    main()