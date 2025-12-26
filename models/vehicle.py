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
class TyreParams:
    """Extended tyre parameters with cornering stiffness for bicycle model."""
    mu: float                    # [-] friction coefficient
    C_alpha_f: float = 45000.0   # [N/rad] front axle cornering stiffness
    C_alpha_r: float = 50000.0   # [N/rad] rear axle cornering stiffness

    @property
    def as_mvp(self) -> TyreParamsMVP:
        """Convert to legacy MVP format."""
        return TyreParamsMVP(mu=self.mu)


@dataclass(frozen=True)
class VehicleGeometry:
    """Vehicle geometry parameters for bicycle model and weight transfer."""
    wheelbase_m: float = 1.55       # [m] total wheelbase (L_f + L_r)
    L_f_m: float = 0.75             # [m] CoG to front axle
    L_r_m: float = 0.80             # [m] CoG to rear axle
    track_front_m: float = 1.20     # [m] front track width
    track_rear_m: float = 1.20      # [m] rear track width
    h_cg_m: float = 0.28            # [m] CoG height above ground

    def __post_init__(self):
        # Validate wheelbase consistency
        if abs(self.wheelbase_m - (self.L_f_m + self.L_r_m)) > 0.01:
            object.__setattr__(self, 'wheelbase_m', self.L_f_m + self.L_r_m)

    @property
    def L(self) -> float:
        """Wheelbase [m]."""
        return self.L_f_m + self.L_r_m

    @property
    def weight_distribution_front(self) -> float:
        """Front weight distribution [0-1]."""
        return self.L_r_m / self.L

    @property
    def weight_distribution_rear(self) -> float:
        """Rear weight distribution [0-1]."""
        return self.L_f_m / self.L


@dataclass(frozen=True)
class TorqueVectoringParams:
    """Torque vectoring system parameters."""
    enabled: bool = False           # Enable torque vectoring
    effectiveness: float = 1.0      # TV system effectiveness [0-1]
    max_torque_transfer: float = 0.5  # Max transfer ratio (0.5 = can shift 50% to one side)
    strategy: str = 'load_proportional'  # 'load_proportional', 'fixed_bias', 'yaw_rate'

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

    Weight calculation:
        total_powertrain_mass = (motor_weight_kg × n_motors) + powertrain_overhead_kg
    """
    drivetrain: str           # 'FWD', 'RWD', or 'AWD'
    motor_power_kW: float     # Power per motor [kW]
    motor_torque_Nm: float    # Peak torque per motor [Nm]
    motor_rpm_max: float      # Maximum motor RPM
    gear_ratio: float         # Final drive gear ratio (motor:wheel)
    wheel_radius_m: float     # Wheel radius [m]
    motor_efficiency: float = 0.85  # Motor efficiency [0-1]
    # Motor identification and weight
    motor_name: str = "Default Motor"  # Motor name/model
    motor_weight_kg: float = 10.0      # Weight per motor [kg]
    motor_constant_Nm_A: float = 0.5   # Torque constant Km [Nm/A]
    peak_current_A: float = 200.0      # Peak motor current [A]
    # Powertrain overhead (inverters, wiring, cooling, mounts)
    powertrain_overhead_kg: float = 25.0  # Additional powertrain mass [kg]

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

    @property
    def total_motor_mass_kg(self) -> float:
        """Total mass of all motors [kg]."""
        return self.motor_weight_kg * self.n_motors

    @property
    def total_powertrain_mass_kg(self) -> float:
        """Total powertrain mass including motors and overhead [kg].

        total = (motor_weight × n_motors) + powertrain_overhead
        """
        return self.total_motor_mass_kg + self.powertrain_overhead_kg


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
    tyre: Union[TyreParamsMVP, TyreParams]  # Either legacy or extended tyre model
    powertrain: Union[EVPowertrainMVP, EVPowertrainParams]  # Either legacy or extended
    battery: BatteryParams = None  # Optional battery params
    geometry: VehicleGeometry = None  # Optional geometry for bicycle model
    torque_vectoring: TorqueVectoringParams = None  # Optional TV params

    @property
    def has_extended_powertrain(self) -> bool:
        """Check if using extended powertrain with drivetrain config."""
        return isinstance(self.powertrain, EVPowertrainParams)

    @property
    def has_bicycle_model(self) -> bool:
        """Check if vehicle has parameters for bicycle model."""
        return self.geometry is not None and isinstance(self.tyre, TyreParams)

    @property
    def has_torque_vectoring(self) -> bool:
        """Check if torque vectoring is enabled."""
        return (self.torque_vectoring is not None and
                self.torque_vectoring.enabled and
                self.has_extended_powertrain and
                self.powertrain.drivetrain in ('RWD', 'AWD'))

    @property
    def mu(self) -> float:
        """Get friction coefficient from either tyre model."""
        return self.tyre.mu

    def get_cornering_stiffness(self) -> tuple:
        """Get front and rear cornering stiffness [N/rad]."""
        if isinstance(self.tyre, TyreParams):
            return self.tyre.C_alpha_f, self.tyre.C_alpha_r
        else:
            # Estimate from friction coefficient and typical values
            # C_alpha ≈ 20 × W (typical for racing tires)
            W_f = self.m * self.g * 0.5  # Approximate front weight
            W_r = self.m * self.g * 0.5  # Approximate rear weight
            return 20.0 * W_f, 20.0 * W_r