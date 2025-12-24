from dataclasses import dataclass
import numpy as np
from typing import Union

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
    """Legacy powertrain model - kept for backward compatibility."""
    P_max: float    # W
    Fx_max: float   # N  (low-speed tractive force cap)


@dataclass(frozen=True)
class EVPowertrainParams:
    """
    Extended powertrain parameters with drivetrain configuration.

    Drivetrain types:
    - 'FWD': Front-wheel drive (2 motors on front axle)
    - 'RWD': Rear-wheel drive (2 motors on rear axle)
    - 'AWD': All-wheel drive (4 motors, 2 front + 2 rear)
    """
    drivetrain: str           # 'FWD', 'RWD', or 'AWD'
    motor_power_kW: float     # Power per motor [kW]
    motor_torque_Nm: float    # Peak torque per motor [Nm]
    motor_rpm_max: float      # Maximum motor RPM
    gear_ratio: float         # Final drive gear ratio (motor:wheel)
    wheel_radius_m: float     # Wheel radius [m]
    motor_efficiency: float = 0.80  # Motor efficiency [0-1]

    @property
    def n_motors(self) -> int:
        """Number of motors based on drivetrain configuration."""
        return 4 if self.drivetrain == 'AWD' else 2

    @property
    def P_max(self) -> float:
        """Total maximum power [W]."""
        return self.motor_power_kW * 1000 * self.n_motors

    @property
    def Fx_max(self) -> float:
        """Maximum tractive force at low speed [N].

        Calculated from motor torque, number of motors, gear ratio, and wheel radius.
        Fx = (torque × n_motors × gear_ratio) / wheel_radius
        """
        return (self.motor_torque_Nm * self.n_motors * self.gear_ratio) / self.wheel_radius_m

    @property
    def v_max_rpm(self) -> float:
        """Maximum vehicle speed limited by motor RPM [m/s].

        v = (wheel_rpm × 2π × wheel_radius) / 60
        where wheel_rpm = motor_rpm / gear_ratio
        """
        wheel_rpm = self.motor_rpm_max / self.gear_ratio
        return (wheel_rpm * 2 * np.pi * self.wheel_radius_m) / 60

    @property
    def v_crossover(self) -> float:
        """Speed at which power limit takes over from torque limit [m/s]."""
        return self.P_max / self.Fx_max


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
    powertrain: Union[EVPowertrainMVP, EVPowertrainParams]  # Either legacy or extended
    battery: BatteryParams = None  # Optional battery params

    @property
    def has_extended_powertrain(self) -> bool:
        """Check if using extended powertrain with drivetrain config."""
        return isinstance(self.powertrain, EVPowertrainParams)