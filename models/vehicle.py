from dataclasses import dataclass

@dataclass(frozen=True)
class AeroParams:
    rho: float      # kg/m^3 - air density
    Cd: float       # [-] - drag coefficient
    Cl: float       # [-] - lift coefficient (downforce is positive)
    A: float        # m^2 - frontal area

    @property
    def CD_A(self) -> float:
        """Drag coefficient × Area [m²]"""
        return self.Cd * self.A

    @property
    def CL_A(self) -> float:
        """Lift coefficient × Area [m²]"""
        return self.Cl * self.A

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