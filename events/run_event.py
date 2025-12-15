from models.track import Track
from models.vehicle import VehicleParams
from solver.qss_speed import solve_qss

def run(track: Track, vehicle: VehicleParams) -> tuple[dict, float]:
    """
    Run a complete event simulation.
    
    Args:
        track: Track object
        vehicle: Vehicle parameters
    
    Returns:
        result: Dictionary with velocity profile and intermediate results
        lap_time: Total time [s]
    """
    return solve_qss(track, vehicle)