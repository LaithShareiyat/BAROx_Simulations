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
    """Battery parameters for energy tracking with optional regenerative braking."""
    capacity_kWh: float             # Total battery capacity [kWh]
    initial_soc: float = 1.0        # Initial state of charge [0-1]
    min_soc: float = 0.1            # Minimum allowed SoC [0-1] (safety margin)
    max_discharge_kW: float = 80.0  # Maximum discharge power [kW]
    eta_discharge: float = 0.95     # Battery discharge efficiency
    # Current limiting (FS 2025 rules: 500A max)
    nominal_voltage_V: float = 400.0  # Nominal pack voltage [V]
    max_current_A: float = 500.0      # Maximum discharge current [A] (FS rules limit)
    # Regenerative braking parameters
    regen_enabled: bool = False     # Enable regenerative braking
    eta_regen: float = 0.85         # Regen efficiency (motor + battery charging)
    max_regen_kW: float = 50.0      # Maximum regen charging power [kW]
    regen_capture_percent: float = 100.0  # % of braking that uses regen vs friction [0-100]

    @property
    def max_power_from_current_kW(self) -> float:
        """Maximum power limited by current: P = V × I [kW]"""
        return (self.nominal_voltage_V * self.max_current_A) / 1000.0

@dataclass(frozen=True)
class VehicleParams:
    m: float        # kg
    g: float        # m/s^2
    Crr: float      # [-]
    aero: AeroParams
    tyre: TyreParamsMVP
    powertrain: EVPowertrainMVP
    battery: BatteryParams = None  # Optional battery params