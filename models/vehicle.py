from dataclasses import dataclass

@dataclass(frozen=True)
class AeroParams:
    rho: float      # kg/m^3
    CD_A: float     # m^2
    CL_A: float     # m^2  

@dataclass(frozen=True)
class TyreParamsMVP:
    mu: float       # [-] constant friction coefficient

@dataclass(frozen=True)
class EVPowertrainMVP:
    P_max: float    # W
    Fx_max: float   # N  (low-speed tractive force cap)

@dataclass(frozen=True)
class BatteryParams:
    """Battery parameters for energy tracking (no regen)."""
    capacity_kWh: float             # Total battery capacity [kWh]
    initial_soc: float = 1.0        # Initial state of charge [0-1]
    min_soc: float = 0.1            # Minimum allowed SoC [0-1] (safety margin)
    max_discharge_kW: float = 80.0  # Maximum discharge power [kW]
    eta_discharge: float = 0.95     # Battery discharge efficiency

@dataclass(frozen=True)
class VehicleParams:
    m: float        # kg
    g: float        # m/s^2
    Crr: float      # [-]
    aero: AeroParams
    tyre: TyreParamsMVP
    powertrain: EVPowertrainMVP
    battery: BatteryParams = None  # Optional battery params