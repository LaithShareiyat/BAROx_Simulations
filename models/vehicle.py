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
class VehicleParams:
    m: float        # kg
    g: float        # m/s^2
    Crr: float      # [-]
    aero: AeroParams
    tyre: TyreParamsMVP
    powertrain: EVPowertrainMVP