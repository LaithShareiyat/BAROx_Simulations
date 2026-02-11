import inquirer
import yaml
import os
import numpy as np

# Add project root to path
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vehicle import (
    VehicleParams,
    AeroParams,
    TyreParamsMVP,
    TyreParams,
    EVPowertrainMVP,
    EVPowertrainParams,
    BatteryParams,
    VehicleGeometry,
    TorqueVectoringParams,
)
from solver.qss_speed import solve_qss
from solver.metrics import lap_time, channels, energy_consumption
from solver.battery import (
    simulate_battery,
    validate_battery_capacity,
    required_battery_capacity,
)


def load_standard_vehicle(config_path: str = "config/default.yaml") -> dict:
    """Load standard vehicle parameters from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_motor_database(motors_path: str = "config/motors.yaml") -> dict:
    """Load motor database from YAML file.

    Returns:
        Dictionary of motors keyed by motor ID
    """
    try:
        with open(motors_path, "r") as f:
            data = yaml.safe_load(f)
            return data.get("motors", {})
    except FileNotFoundError:
        print(f"Warning: Motor database not found at {motors_path}")
        return {}


def get_motor_params(motor_id: str, motors_db: dict = None) -> dict:
    """Get motor parameters from database by ID.

    Args:
        motor_id: Motor identifier (e.g., 'amk_dd5', 'emrax_228')
        motors_db: Optional pre-loaded motor database

    Returns:
        Dictionary with motor parameters, or None if not found
    """
    if motors_db is None:
        motors_db = load_motor_database()

    return motors_db.get(motor_id)


def create_vehicle_from_config(config: dict) -> VehicleParams:
    """Create a VehicleParams object from config dictionary."""
    aero = AeroParams(
        rho=config["aero"]["rho"],
        Cd=config["aero"]["Cd"],
        Cl=config["aero"]["Cl"],
        A=config["aero"]["A"],
    )

    # Check if using extended tyre model with cornering stiffness
    tyre_config = config["tyre"]
    if "C_alpha_f" in tyre_config:
        tyre = TyreParams(
            mu=tyre_config["mu"],
            C_alpha_f=tyre_config.get("C_alpha_f", 45000.0),
            C_alpha_r=tyre_config.get("C_alpha_r", 50000.0),
        )
    else:
        tyre = TyreParamsMVP(
            mu=tyre_config["mu"],
        )

    # Check if using new extended powertrain format or legacy format
    pt_config = config["powertrain"]
    if "drivetrain" in pt_config:
        # Check if referencing a motor from the database
        motor_params = None
        if "motor" in pt_config:
            motor_id = pt_config["motor"]
            motor_params = get_motor_params(motor_id)
            if motor_params is None:
                print(
                    f"Warning: Motor '{motor_id}' not found in database, using inline params"
                )

        # Get motor parameters (from database or inline config)
        if motor_params:
            motor_name = motor_params.get("name", motor_id)
            motor_power_kW = motor_params.get(
                "peak_power_kW", pt_config.get("motor_power_kW", 40)
            )
            motor_torque_Nm = motor_params.get(
                "peak_torque_Nm", pt_config.get("motor_torque_Nm", 100)
            )
            motor_rpm_max = motor_params.get(
                "max_rpm", pt_config.get("motor_rpm_max", 6000)
            )
            motor_efficiency = motor_params.get(
                "peak_efficiency", pt_config.get("motor_efficiency", 0.96)
            )
            motor_weight_kg = motor_params.get(
                "weight_kg", pt_config.get("motor_weight_kg", 9.4)
            )
            motor_constant = motor_params.get(
                "motor_constant_Nm_A", pt_config.get("motor_constant_Nm_A", 0.5)
            )
            peak_current = motor_params.get(
                "peak_current_A", pt_config.get("peak_current_A", 200.0)
            )
        else:
            motor_name = pt_config.get("motor_name", "EMRAX 208")
            motor_power_kW = pt_config["motor_power_kW"]
            motor_torque_Nm = pt_config["motor_torque_Nm"]
            motor_rpm_max = pt_config["motor_rpm_max"]
            motor_efficiency = pt_config.get("motor_efficiency", 0.96)
            motor_weight_kg = pt_config.get("motor_weight_kg", 9.4)
            motor_constant = pt_config.get("motor_constant_Nm_A", 0.62)
            peak_current = pt_config.get("peak_current_A", 240.0)

        # New extended powertrain format with motor weight
        powertrain = EVPowertrainParams(
            drivetrain=pt_config["drivetrain"],
            motor_power_kW=motor_power_kW,
            motor_torque_Nm=motor_torque_Nm,
            motor_rpm_max=motor_rpm_max,
            gear_ratio=pt_config["gear_ratio"],
            wheel_radius_m=pt_config["wheel_radius_m"],
            motor_efficiency=motor_efficiency,
            motor_name=motor_name,
            motor_weight_kg=motor_weight_kg,
            motor_constant_Nm_A=motor_constant,
            peak_current_A=peak_current,
            powertrain_overhead_kg=pt_config.get("powertrain_overhead_kg", 10.0),
            inverter_peak_power_kW=pt_config.get("inverter_peak_power_kW", 320.0),
            inverter_peak_current_A=pt_config.get("inverter_peak_current_A", 600.0),
            inverter_weight_kg=pt_config.get("inverter_weight_kg", 6.9),
        )
    else:
        # Legacy format with P_max_kW and Fx_max_N
        powertrain = EVPowertrainMVP(
            P_max=pt_config["P_max_kW"] * 1000,  # Convert kW to W
            Fx_max=pt_config["Fx_max_N"],
        )

    # Create battery params if present in config
    battery = None
    if "battery" in config:
        battery = BatteryParams(
            capacity_kWh=config["battery"]["capacity_kWh"],
            initial_soc=config["battery"].get("initial_soc", 1.0),
            min_soc=config["battery"].get("min_soc", 0.1),
            max_discharge_kW=config["battery"].get("max_discharge_kW", 80.0),
            eta_discharge=config["battery"].get("eta_discharge", 0.95),
            # Current limiting (142S5P pack default)
            nominal_voltage_V=config["battery"].get("nominal_voltage_V", 511.0),
            max_current_A=config["battery"].get("max_current_A", 175.0),
            # Regenerative braking parameters
            regen_enabled=config["battery"].get("regen_enabled", False),
            eta_regen=config["battery"].get("eta_regen", 0.85),
            max_regen_kW=config["battery"].get("max_regen_kW", 50.0),
            regen_capture_percent=config["battery"].get("regen_capture_percent", 100.0),
        )

    # Create geometry params if present in config (for bicycle model)
    geometry = None
    if "geometry" in config:
        geo_config = config["geometry"]
        geometry = VehicleGeometry(
            wheelbase_m=geo_config.get("wheelbase_m", 1.55),
            L_f_m=geo_config.get("L_f_m", 0.75),
            L_r_m=geo_config.get("L_r_m", 0.80),
            track_front_m=geo_config.get("track_front_m", 1.20),
            track_rear_m=geo_config.get("track_rear_m", 1.20),
            h_cg_m=geo_config.get("h_cg_m", 0.28),
        )

    # Create torque vectoring params if present in config
    torque_vectoring = None
    if "torque_vectoring" in config:
        tv_config = config["torque_vectoring"]
        torque_vectoring = TorqueVectoringParams(
            enabled=tv_config.get("enabled", False),
            effectiveness=tv_config.get("effectiveness", 1.0),
            max_torque_transfer=tv_config.get("max_torque_transfer", 0.5),
            strategy=tv_config.get("strategy", "load_proportional"),
        )

    # Calculate total vehicle mass
    # If mass breakdown is provided, use calculated powertrain mass from motors
    vehicle_config = config["vehicle"]
    if "mass_chassis_kg" in vehicle_config:
        # Mass breakdown mode: use calculated powertrain mass
        mass_chassis = vehicle_config.get("mass_chassis_kg", 55)
        mass_aero = vehicle_config.get("mass_aero_kg", 20)
        mass_suspension_tyres = vehicle_config.get("mass_suspension_tyres_kg", 50)
        mass_battery = vehicle_config.get("mass_battery_kg", 45)
        mass_electronics = vehicle_config.get("mass_electronics_kg", 25)

        # Use calculated powertrain mass if available (from motor weight × n_motors + overhead)
        if isinstance(powertrain, EVPowertrainParams):
            mass_powertrain = powertrain.total_powertrain_mass_kg
        else:
            mass_powertrain = vehicle_config.get("mass_powertrain_kg", 44)

        total_mass = (
            mass_chassis
            + mass_aero
            + mass_suspension_tyres
            + mass_powertrain
            + mass_battery
            + mass_electronics
        )
    else:
        # Simple mode: use mass_kg directly
        total_mass = vehicle_config["mass_kg"]

    return VehicleParams(
        m=total_mass,
        g=config["vehicle"]["g"],
        Crr=config["vehicle"]["Crr"],
        aero=aero,
        tyre=tyre,
        powertrain=powertrain,
        battery=battery,
        geometry=geometry,
        torque_vectoring=torque_vectoring,
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
        inquirer.Text(
            "mass_kg",
            message=f"Mass [kg] ({defaults['vehicle']['mass_kg']})",
            default=str(defaults["vehicle"]["mass_kg"]),
        ),
        inquirer.Text(
            "g",
            message=f"Gravity [m/s²] ({defaults['vehicle']['g']})",
            default=str(defaults["vehicle"]["g"]),
        ),
        inquirer.Text(
            "Crr",
            message=f"Rolling resistance coefficient ({defaults['vehicle']['Crr']})",
            default=str(defaults["vehicle"]["Crr"]),
        ),
    ]
    vehicle_answers = inquirer.prompt(questions)

    # Aero parameters
    print("\n--- Aerodynamics ---")
    questions = [
        inquirer.Text(
            "rho",
            message=f"Air density [kg/m³] ({defaults['aero']['rho']})",
            default=str(defaults["aero"]["rho"]),
        ),
        inquirer.Text(
            "Cd",
            message=f"Drag coefficient (Cd) [-] ({defaults['aero']['Cd']})",
            default=str(defaults["aero"]["Cd"]),
        ),
        inquirer.Text(
            "Cl",
            message=f"Lift coefficient (Cl) [-] ({defaults['aero']['Cl']})",
            default=str(defaults["aero"]["Cl"]),
        ),
        inquirer.Text(
            "A",
            message=f"Frontal area [m²] ({defaults['aero']['A']})",
            default=str(defaults["aero"]["A"]),
        ),
    ]
    aero_answers = inquirer.prompt(questions)

    # Tyre parameters
    print("\n--- Tyre ---")
    questions = [
        inquirer.Text(
            "mu",
            message=f"Tyre friction coefficient (μ) ({defaults['tyre']['mu']})",
            default=str(defaults["tyre"]["mu"]),
        ),
    ]
    tyre_answers = inquirer.prompt(questions)

    # Powertrain parameters - Drivetrain selection
    print("\n--- Powertrain ---")
    drivetrain_question = [
        inquirer.List(
            "drivetrain",
            message="Select drivetrain configuration",
            choices=[
                ("1 Motor Rear (1RWD) - with differential", "1RWD"),
                ("1 Motor Front (1FWD) - with differential", "1FWD"),
                ("2 Motor Rear (2RWD)", "2RWD"),
                ("2 Motor Front (2FWD)", "2FWD"),
                ("4 Motor All-Wheel (AWD)", "AWD"),
            ],
            default="2RWD",
        ),
    ]
    drivetrain_answer = inquirer.prompt(drivetrain_question)
    drivetrain = drivetrain_answer["drivetrain"]
    # Calculate number of motors from drivetrain
    if drivetrain == "AWD":
        n_motors = 4
    elif drivetrain.startswith("1"):
        n_motors = 1
    else:
        n_motors = 2

    # Motor parameters
    pt_defaults = defaults.get(
        "powertrain",
        {
            "motor_power_kW": 40,  # 40 kW per motor (FS 80 kW total limit)
            "motor_torque_Nm": 150,  # EMRAX 208
            "motor_rpm_max": 7000,  # EMRAX 208
            "gear_ratio": 3.0,
            "wheel_radius_m": 0.203,
        },
    )

    print(f"\n  Selected: {drivetrain} ({n_motors} motors)")
    print("\n--- Motor Parameters (per motor) ---")
    questions = [
        inquirer.Text(
            "motor_power_kW",
            message=f"Motor power [kW] ({pt_defaults.get('motor_power_kW', 40)})",
            default=str(pt_defaults.get("motor_power_kW", 40)),
        ),
        inquirer.Text(
            "motor_torque_Nm",
            message=f"Motor peak torque [Nm] ({pt_defaults.get('motor_torque_Nm', 150)})",
            default=str(pt_defaults.get("motor_torque_Nm", 150)),
        ),
        inquirer.Text(
            "motor_rpm_max",
            message=f"Motor max RPM ({pt_defaults.get('motor_rpm_max', 7000)})",
            default=str(pt_defaults.get("motor_rpm_max", 7000)),
        ),
        inquirer.Text(
            "motor_efficiency",
            message=f"Motor efficiency [0-1] ({pt_defaults.get('motor_efficiency', 0.96)})",
            default=str(pt_defaults.get("motor_efficiency", 0.96)),
        ),
    ]
    motor_answers = inquirer.prompt(questions)

    print("\n--- Transmission ---")
    questions = [
        inquirer.Text(
            "gear_ratio",
            message=f"Gear ratio (motor:wheel) ({pt_defaults.get('gear_ratio', 3.0)})",
            default=str(pt_defaults.get("gear_ratio", 3.5)),
        ),
        inquirer.Text(
            "wheel_radius_m",
            message=f"Wheel radius [m] ({pt_defaults.get('wheel_radius_m', 0.203)})",
            default=str(pt_defaults.get("wheel_radius_m", 0.203)),
        ),
    ]
    transmission_answers = inquirer.prompt(questions)

    # Calculate and display derived values
    motor_power = float(motor_answers["motor_power_kW"])
    motor_torque = float(motor_answers["motor_torque_Nm"])
    motor_rpm = float(motor_answers["motor_rpm_max"])
    motor_efficiency = float(motor_answers["motor_efficiency"])
    gear_ratio = float(transmission_answers["gear_ratio"])
    wheel_radius = float(transmission_answers["wheel_radius_m"])

    total_power = motor_power * n_motors
    fx_max = (motor_torque * n_motors * gear_ratio) / wheel_radius
    wheel_rpm = motor_rpm / gear_ratio
    v_max_rpm = (wheel_rpm * 2 * 3.14159 * wheel_radius) / 60

    print(f"\n  Calculated values:")
    print(f"    Total power:     {total_power:.0f} kW")
    print(f"    Max force:       {fx_max:.0f} N")
    print(f"    Max speed (RPM): {v_max_rpm:.1f} m/s ({v_max_rpm * 3.6:.0f} km/h)")
    print(f"    Motor efficiency: {motor_efficiency * 100:.0f}%")

    # Battery parameters
    print("\n--- Battery ---")
    battery_defaults = defaults.get(
        "battery",
        {
            "capacity_kWh": 6.65,  # 142S5P pack
            "initial_soc": 1.0,
            "min_soc": 0.1,
            "max_discharge_kW": 80.0,  # Match FS 80 kW total power limit
            "eta_discharge": 0.95,
            "nominal_voltage_V": 511.0,  # 142S × 3.6V
            "max_current_A": 175.0,  # 5P × 35A cell limit
            "regen_enabled": False,
            "eta_regen": 0.85,
            "max_regen_kW": 50.0,
            "regen_capture_percent": 100.0,
        },
    )
    questions = [
        inquirer.Text(
            "capacity_kWh",
            message=f"Battery capacity [kWh] ({battery_defaults['capacity_kWh']})",
            default=str(battery_defaults["capacity_kWh"]),
        ),
        inquirer.Text(
            "initial_soc",
            message=f"Initial SoC [0-1] ({battery_defaults['initial_soc']})",
            default=str(battery_defaults["initial_soc"]),
        ),
        inquirer.Text(
            "min_soc",
            message=f"Minimum SoC [0-1] ({battery_defaults['min_soc']})",
            default=str(battery_defaults["min_soc"]),
        ),
        inquirer.Text(
            "max_discharge_kW",
            message=f"Max discharge power [kW] ({battery_defaults['max_discharge_kW']})",
            default=str(battery_defaults["max_discharge_kW"]),
        ),
        inquirer.Text(
            "nominal_voltage_V",
            message=f"Nominal pack voltage [V] ({battery_defaults.get('nominal_voltage_V', 511)})",
            default=str(battery_defaults.get("nominal_voltage_V", 511)),
        ),
        inquirer.Text(
            "max_current_A",
            message=f"Max current [A] (pack limit) ({battery_defaults.get('max_current_A', 175)})",
            default=str(battery_defaults.get("max_current_A", 175)),
        ),
    ]
    battery_answers = inquirer.prompt(questions)

    # Regenerative braking option
    print("\n--- Regenerative Braking ---")
    regen_question = [
        inquirer.Confirm(
            "regen_enabled",
            message="Enable regenerative braking?",
            default=battery_defaults.get("regen_enabled", False),
        ),
    ]
    regen_answer = inquirer.prompt(regen_question)
    regen_enabled = regen_answer["regen_enabled"]

    # If regen enabled, prompt for regen parameters
    regen_answers = {}
    if regen_enabled:
        regen_questions = [
            inquirer.Text(
                "eta_regen",
                message=f"Regen efficiency [0-1] ({battery_defaults.get('eta_regen', 0.85)})",
                default=str(battery_defaults.get("eta_regen", 0.85)),
            ),
            inquirer.Text(
                "max_regen_kW",
                message=f"Max regen power [kW] ({battery_defaults.get('max_regen_kW', 50.0)})",
                default=str(battery_defaults.get("max_regen_kW", 50.0)),
            ),
            inquirer.Text(
                "regen_capture_percent",
                message=f"Braking capture [%] ({battery_defaults.get('regen_capture_percent', 100.0)})",
                default=str(battery_defaults.get("regen_capture_percent", 100.0)),
            ),
        ]
        regen_answers = inquirer.prompt(regen_questions)

    # Build config dict
    config = {
        "vehicle": {
            "mass_kg": float(vehicle_answers["mass_kg"]),
            "g": float(vehicle_answers["g"]),
            "Crr": float(vehicle_answers["Crr"]),
        },
        "aero": {
            "rho": float(aero_answers["rho"]),
            "Cd": float(aero_answers["Cd"]),
            "Cl": float(aero_answers["Cl"]),
            "A": float(aero_answers["A"]),
        },
        "tyre": {
            "mu": float(tyre_answers["mu"]),
        },
        "powertrain": {
            "drivetrain": drivetrain,
            "motor_power_kW": motor_power,
            "motor_torque_Nm": motor_torque,
            "motor_rpm_max": motor_rpm,
            "motor_efficiency": motor_efficiency,
            "gear_ratio": gear_ratio,
            "wheel_radius_m": wheel_radius,
        },
        "battery": {
            "capacity_kWh": float(battery_answers["capacity_kWh"]),
            "initial_soc": float(battery_answers["initial_soc"]),
            "min_soc": float(battery_answers["min_soc"]),
            "max_discharge_kW": float(battery_answers["max_discharge_kW"]),
            "eta_discharge": battery_defaults.get("eta_discharge", 0.95),
            # Current limiting (FS 2025 rules)
            "nominal_voltage_V": float(battery_answers["nominal_voltage_V"]),
            "max_current_A": float(battery_answers["max_current_A"]),
            # Regenerative braking
            "regen_enabled": regen_enabled,
            "eta_regen": float(
                regen_answers.get("eta_regen", battery_defaults.get("eta_regen", 0.85))
            )
            if regen_enabled
            else 0.85,
            "max_regen_kW": float(
                regen_answers.get(
                    "max_regen_kW", battery_defaults.get("max_regen_kW", 50.0)
                )
            )
            if regen_enabled
            else 50.0,
            "regen_capture_percent": float(
                regen_answers.get(
                    "regen_capture_percent",
                    battery_defaults.get("regen_capture_percent", 100.0),
                )
            )
            if regen_enabled
            else 100.0,
        },
    }

    return config


def print_header(title: str, width: int = 60):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 60):
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


def print_vehicle_params(config: dict):
    """Print vehicle parameters in a formatted table."""
    print_header("VEHICLE CONFIGURATION")

    cd_a = config["aero"]["Cd"] * config["aero"]["A"]
    cl_a = config["aero"]["Cl"] * config["aero"]["A"]

    print("\n  VEHICLE")
    print(f"    Mass:              {config['vehicle']['mass_kg']} kg")
    print(f"    Gravity:           {config['vehicle']['g']} m/s2")
    print(f"    Rolling resist:    {config['vehicle']['Crr']}")

    print("\n  AERODYNAMICS")
    print(f"    Air density:       {config['aero']['rho']} kg/m3")
    print(f"    Cd:                {config['aero']['Cd']}")
    print(f"    Cl:                {config['aero']['Cl']}")
    print(f"    Frontal area:      {config['aero']['A']} m2")
    print(f"    Cd x A:            {cd_a:.3f} m2")
    print(f"    Cl x A:            {cl_a:.3f} m2")

    print("\n  TYRE")
    print(f"    Friction (mu):     {config['tyre']['mu']}")

    print("\n  POWERTRAIN")
    pt = config["powertrain"]
    if "drivetrain" in pt:
        # New extended format
        drivetrain = pt["drivetrain"]
        if drivetrain == "AWD":
            n_motors = 4
        elif drivetrain.startswith("1"):
            n_motors = 1
        else:
            n_motors = 2
        motor_power_total = pt["motor_power_kW"] * n_motors
        rules_cap = pt.get("P_max_rules_kW", 80.0)
        total_power = min(motor_power_total, rules_cap)
        fx_max = (pt["motor_torque_Nm"] * n_motors * pt["gear_ratio"]) / pt[
            "wheel_radius_m"
        ]
        wheel_rpm = pt["motor_rpm_max"] / pt["gear_ratio"]
        v_max = (wheel_rpm * 2 * 3.14159 * pt["wheel_radius_m"]) / 60

        print(f"    Drivetrain:        {pt['drivetrain']} ({n_motors} motors)")
        print(f"    Motor power:       {pt['motor_power_kW']} kW (per motor)")
        print(f"    Motor torque:      {pt['motor_torque_Nm']} Nm (per motor)")
        print(f"    Motor max RPM:     {pt['motor_rpm_max']}")
        print(f"    Motor efficiency:  {pt.get('motor_efficiency', 0.85):.0%}")
        print(f"    Gear ratio:        {pt['gear_ratio']}:1")
        print(f"    Wheel radius:      {pt['wheel_radius_m']} m")
        inv_weight = pt.get('inverter_weight_kg', 6.9)
        print(f"    Inverter:          {pt.get('inverter_name', 'DTI HV-850')} ({n_motors}x, {inv_weight:.1f} kg each)")
        print(f"    --- Calculated ---")
        print(f"    Total power:       {total_power:.0f} kW (FS cap: {rules_cap:.0f} kW)")
        print(f"    Max force:         {fx_max:.0f} N")
        print(f"    Max speed (RPM):   {v_max:.1f} m/s ({v_max * 3.6:.0f} km/h)")
    else:
        # Legacy format
        print(f"    Max power:         {pt['P_max_kW']} kW")
        print(f"    Max force:         {pt['Fx_max_N']} N")

    if "battery" in config:
        print("\n  BATTERY")
        print(f"    Capacity:          {config['battery']['capacity_kWh']} kWh")
        print(f"    Initial SoC:       {config['battery']['initial_soc']:.0%}")
        print(f"    Min SoC:           {config['battery']['min_soc']:.0%}")
        print(f"    Max discharge:     {config['battery']['max_discharge_kW']} kW")
        print(
            f"    Efficiency:        {config['battery'].get('eta_discharge', 0.95):.0%}"
        )

        # Current limiting (FS 2025 rules)
        nominal_v = config["battery"].get("nominal_voltage_V", 400)
        max_current = config["battery"].get("max_current_A", 500)
        power_from_current = (nominal_v * max_current) / 1000
        print(f"\n  CURRENT LIMIT (FS Rules)")
        print(f"    Nominal voltage:   {nominal_v} V")
        print(f"    Max current:       {max_current} A")
        print(f"    Power limit (V×I): {power_from_current:.1f} kW")

        # Regenerative braking
        regen_enabled = config["battery"].get("regen_enabled", False)
        regen_status = "ENABLED" if regen_enabled else "DISABLED"
        print(f"\n  REGENERATIVE BRAKING: {regen_status}")
        if regen_enabled:
            print(
                f"    Regen efficiency:  {config['battery'].get('eta_regen', 0.85):.0%}"
            )
            print(
                f"    Max regen power:   {config['battery'].get('max_regen_kW', 50.0)} kW"
            )
            print(
                f"    Braking capture:   {config['battery'].get('regen_capture_percent', 100.0):.0f}%"
            )


def compute_metrics(track, v: np.ndarray, vehicle: VehicleParams) -> dict:
    """Compute performance metrics from velocity profile."""
    t = lap_time(track, v)
    ax, ay = channels(track, v)

    # Handle energy consumption
    try:
        energy = energy_consumption(track, v, vehicle)
        energy_kwh = energy.get("E_net_kWh", np.nan)
    except Exception:
        energy_kwh = np.nan

    # Handle edge cases in velocity
    v_positive = v[v > 0.1]
    min_speed = np.min(v_positive) if len(v_positive) > 0 else 0.0

    # Handle NaN in acceleration arrays
    ax_clean = ax[~np.isnan(ax)]
    ay_clean = ay[~np.isnan(ay)]

    return {
        "lap_time": t,
        "avg_speed": np.nanmean(v),
        "max_speed": np.nanmax(v),
        "min_speed": min_speed,
        "max_ax": np.max(ax_clean) if len(ax_clean) > 0 else 0.0,
        "min_ax": np.min(ax_clean) if len(ax_clean) > 0 else 0.0,
        "max_ay": np.max(np.abs(ay_clean)) if len(ay_clean) > 0 else 0.0,
        "energy_consumed_kWh": energy_kwh if not np.isnan(energy_kwh) else 0.0,
    }


def run_skidpad_simulation(config: dict):
    """Run the skidpad simulation with lap time calculation."""
    from events.skidpad import (
        build_skidpad_track,
        plot_skidpad,
        skidpad_time_from_single_circle,
        SKIDPAD_CENTRE_RADIUS,
        TRACK_WIDTH,
    )

    print_header("SKIDPAD EVENT")

    # Build track (single circle for timing)
    track = build_skidpad_track()
    circle_length = 2 * np.pi * SKIDPAD_CENTRE_RADIUS

    # Create vehicle
    vehicle = create_vehicle_from_config(config)

    # Solve for velocity profile
    result, t_lap = solve_qss(track, vehicle)
    v = result["v"]

    # Get timing breakdown
    timing = skidpad_time_from_single_circle(t_lap)

    # Compute additional metrics
    metrics = compute_metrics(track, v, vehicle)

    # Print track info
    print("\n  TRACK")
    print(f"    Radius:          {SKIDPAD_CENTRE_RADIUS:.3f} m")
    print(f"    Circumference:   {circle_length:.2f} m")
    print(f"    Track width:     {TRACK_WIDTH} m")

    # Print results
    print_subheader("SKIDPAD RESULTS")

    print("\n  TIMING")
    print(f"    Single circle:   {timing['t_official']:.3f} s")
    print(f"    Full run (4x):   {timing['t_full_run']:.3f} s")

    print("\n  PERFORMANCE")
    print(
        f"    Avg speed:       {timing['avg_speed']:.2f} m/s  ({timing['avg_speed'] * 3.6:.1f} km/h)"
    )
    print(
        f"    Lat accel:       {metrics['max_ay']:.2f} m/s2  ({metrics['max_ay'] / 9.81:.2f} g)"
    )
    print(f"    Energy:          {metrics['energy_consumed_kWh'] * 1000:.1f} Wh")

    # Add timing to metrics for return
    metrics["t_official"] = timing["t_official"]
    metrics["t_full_run"] = timing["t_full_run"]

    # Plot with velocity colouring
    plot_skidpad(track, v=v)

    return metrics


def run_autocross_simulation(config: dict):
    """Run the autocross simulation with lap time calculation and battery tracking."""
    from events.autocross_generator import (
        build_standard_autocross,
        plot_autocross,
        validate_autocross,
        MAX_TRACK_LENGTH,
    )

    print_header("AUTOCROSS EVENT")

    # Build track
    track, metadata = build_standard_autocross()

    max_kappa = np.max(np.abs(track.kappa[track.kappa != 0]))
    min_radius = 1 / max_kappa
    n_cones = len(metadata["slalom_cones_x"])

    # Validate track
    validation = validate_autocross(track, metadata)
    track_valid = validation["valid"]
    valid_symbol = "[OK]" if track_valid else "[FAIL]"

    print("\n  TRACK CONFIGURATION")
    print(f"    Length:          {track.s[-1]:.1f} m  (max {MAX_TRACK_LENGTH:.0f} m)")
    print(f"    Min radius:      {min_radius:.2f} m")
    print(f"    Slalom cones:    {n_cones}")
    print(f"    Status:          {valid_symbol}")

    # Print validation warnings/errors if any
    for warn in validation.get("warnings", []):
        print(f"    Warning: {warn}")
    if not validation["valid"]:
        for err in validation["errors"]:
            print(f"    Error: {err}")

    # Create vehicle
    vehicle = create_vehicle_from_config(config)

    # Solve for velocity profile
    result, _ = solve_qss(track, vehicle)
    v = result["v"]

    # Compute metrics
    metrics = compute_metrics(track, v, vehicle)

    # Print results
    print_subheader("AUTOCROSS RESULTS")

    print(f"\n  LAP TIME:          {metrics['lap_time']:.3f} s")

    print("\n  SPEED                    m/s        km/h")
    print(
        f"    Average:             {metrics['avg_speed']:>6.2f}      {metrics['avg_speed'] * 3.6:>6.1f}"
    )
    print(
        f"    Maximum:             {metrics['max_speed']:>6.2f}      {metrics['max_speed'] * 3.6:>6.1f}"
    )
    print(
        f"    Minimum:             {metrics['min_speed']:>6.2f}      {metrics['min_speed'] * 3.6:>6.1f}"
    )

    print("\n  ACCELERATION            m/s2          g")
    print(
        f"    Max longitudinal:    {metrics['max_ax']:>6.2f}      {metrics['max_ax'] / 9.81:>6.2f}"
    )
    print(
        f"    Max braking:         {abs(metrics['min_ax']):>6.2f}      {abs(metrics['min_ax']) / 9.81:>6.2f}"
    )
    print(
        f"    Max lateral:         {metrics['max_ay']:>6.2f}      {metrics['max_ay'] / 9.81:>6.2f}"
    )

    print(f"\n  ENERGY:            {metrics['energy_consumed_kWh'] * 1000:.1f} Wh")

    # Store data for battery analysis (done later)
    if vehicle.battery is not None:
        battery_validation = validate_battery_capacity(track, v, vehicle)
        metrics["battery_validation"] = battery_validation
        metrics["battery_sufficient"] = battery_validation.sufficient
        metrics["final_soc"] = battery_validation.final_soc
        metrics["min_soc"] = battery_validation.min_soc
        metrics["track"] = track
        metrics["v"] = v
        metrics["vehicle"] = vehicle

    # Plot with velocity colouring
    plot_autocross(track, v=v, metadata=metadata)

    return metrics


def print_battery_analysis(metrics: dict, config: dict):
    """Print battery analysis results and show plots."""
    from plots.plots import plot_battery_state

    if "battery_validation" not in metrics:
        return

    battery_validation = metrics["battery_validation"]
    vehicle = metrics["vehicle"]
    track = metrics["track"]
    v = metrics["v"]

    usable = vehicle.battery.capacity_kWh * (
        vehicle.battery.initial_soc - vehicle.battery.min_soc
    )
    status_text = (
        "[OK] SUFFICIENT" if battery_validation.sufficient else "[FAIL] INSUFFICIENT"
    )

    print_subheader("BATTERY ANALYSIS")
    print(f"\n  STATUS: {status_text}")

    print("\n  STATE OF CHARGE")
    print(f"    Initial:         {vehicle.battery.initial_soc:.1%}")
    print(f"    Final:           {battery_validation.final_soc:.1%}")
    print(
        f"    Minimum:         {battery_validation.min_soc:.1%}  (at {battery_validation.min_soc_distance:.1f} m)"
    )
    print(f"    Min allowed:     {vehicle.battery.min_soc:.1%}")

    print("\n  ENERGY                   kWh          Wh")
    print(
        f"    Capacity:            {vehicle.battery.capacity_kWh:>6.2f}      {vehicle.battery.capacity_kWh * 1000:>6.0f}"
    )
    print(f"    Usable:              {usable:>6.2f}      {usable * 1000:>6.0f}")
    print(
        f"    Consumed:            {battery_validation.total_energy_kWh:>6.4f}      {battery_validation.total_energy_kWh * 1000:>6.1f}"
    )
    print(
        f"    Remaining:           {usable - battery_validation.total_energy_kWh:>6.2f}      {(usable - battery_validation.total_energy_kWh) * 1000:>6.1f}"
    )

    print("\n  POWER")
    print(f"    Peak discharge:      {battery_validation.peak_power_kW:.1f} kW")
    print(f"    Average discharge:   {battery_validation.avg_power_kW:.1f} kW")

    # Print warnings and errors
    for warn in battery_validation.warnings:
        print(f"\n    Warning: {warn}")
    for err in battery_validation.errors:
        print(f"\n    Error: {err}")

    if not battery_validation.sufficient:
        req_capacity = required_battery_capacity(track, v, vehicle, safety_margin=0.2)
        print(f"\n  Recommended minimum capacity: {req_capacity:.2f} kWh")

    # Simulate battery for plotting
    battery_state = simulate_battery(track, v, vehicle)

    # Plot battery state with appropriate title
    regen_status = "With Regen" if vehicle.battery.regen_enabled else "No Regen"
    plot_battery_state(
        track,
        battery_state,
        vehicle,
        battery_validation,
        title=f"Autocross Battery Analysis ({regen_status})",
    )

    # SoC track map disabled - redundant since min SoC is at track end
    # plot_soc_on_track(track, battery_state, vehicle, title="Autocross - SoC Map")


def main():
    """Main entry point with interactive menu."""

    # Banner
    print("\n" + "=" * 60)
    print("  BAROx - Formula Student Lap Time Simulator")
    print("  QSS Point-Mass Model")
    print("=" * 60)

    # Question 1: Select simulation type
    simulation_question = [
        inquirer.List(
            "simulation",
            message="Select simulation",
            choices=[
                ("Both (Autocross + Skidpad)", "both"),
                ("Autocross only", "autocross"),
                ("Skidpad only", "skidpad"),
            ],
        ),
    ]
    simulation_answer = inquirer.prompt(simulation_question)

    if simulation_answer is None:
        print("\n  Cancelled.\n")
        return

    simulation_type = simulation_answer["simulation"]

    # Question 2: Select vehicle parameters
    vehicle_question = [
        inquirer.List(
            "vehicle_params",
            message="Vehicle parameters",
            choices=[
                ("Standard (default.yaml)", "standard"),
                ("Custom (enter values)", "custom"),
            ],
        ),
    ]
    vehicle_answer = inquirer.prompt(vehicle_question)

    if vehicle_answer is None:
        print("\n  Cancelled.\n")
        return

    # Load or get vehicle parameters
    defaults = load_standard_vehicle()

    if vehicle_answer["vehicle_params"] == "standard":
        config = defaults
        print("\n  Using standard vehicle parameters from default.yaml")
    else:
        config = get_custom_vehicle_params(defaults)

    # Print the vehicle parameters being used
    print_vehicle_params(config)

    # Run selected simulation(s)
    # Order: Autocross -> Skidpad -> Battery Analysis -> Summary
    results = {}

    if simulation_type == "both":
        # 1. Autocross results
        results["autocross"] = run_autocross_simulation(config)

        # 2. Skidpad results
        results["skidpad"] = run_skidpad_simulation(config)

        # 3. Battery analysis (after both events)
        if "battery_validation" in results["autocross"]:
            print_battery_analysis(results["autocross"], config)

        # 4. Summary
        print_header("SUMMARY")
        print(f"\n  Autocross lap time:    {results['autocross']['lap_time']:.3f} s")
        print(f"  Skidpad lap time:      {results['skidpad']['t_official']:.3f} s")
        if "battery_sufficient" in results["autocross"]:
            status = "[OK]" if results["autocross"]["battery_sufficient"] else "[FAIL]"
            print(f"  Battery sufficient:    {status}")

    elif simulation_type == "autocross":
        # 1. Autocross results
        results["autocross"] = run_autocross_simulation(config)

        # 2. Battery analysis
        if "battery_validation" in results["autocross"]:
            print_battery_analysis(results["autocross"], config)

    elif simulation_type == "skidpad":
        results["skidpad"] = run_skidpad_simulation(config)

    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    main()
