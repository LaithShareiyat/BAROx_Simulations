<div align="center">

# BAROx Formula Student Lap Time Simulator

**Blue Arrows Racing Oxford (BAROx)**

![BAROx Logo](resources/BAROx_Logo.png)

University of Oxford

MEng Engineering Science

3rd Year Design Project

*Author: Laith Shareiyat*
</div>

---

## Overview

A **Quasi-Steady-State (QSS) point-mass lap time simulator** designed specifically for Formula Student electric vehicles. This tool simulates the **Autocross** and **Skidpad** events, calculating lap times, energy consumption, and battery state-of-charge throughout each run.

### Key Features

- **QSS Lap Time Simulation** — Three-pass algorithm (lateral limit, forward acceleration, backward braking) with friction circle constraint
- **Full GUI Application** — Interactive graphical interface with dark/light theme, 11 visualisation tabs, and one-click results export
- **Config Comparison** — Save multiple vehicle configurations, run A/B comparisons on autocross, and rank them by lap time
- **Battery Pack Optimiser** — Sweep series/parallel configurations and discharge power limits to find optimal battery pack sizing
- **Parameter Sweeps** — Gear ratio sensitivity analysis with automatic optimum detection
- **Regenerative Braking** — Optional regen modelling with configurable capture percentage
- **Torque Vectoring** — Yaw moment generation from differential motor torque (RWD/AWD)
- **Inverter Modelling** — Current-limited effective torque with per-inverter peak power and current constraints
- **Bicycle Model** — Linear tyre model with slip angle calculations for cornering validation
- **Weight Transfer** — Longitudinal and lateral load transfer during acceleration and cornering
- **Motor Database** — YAML-based motor library with GUI dropdown selection
- **Configurable Vehicle** — YAML-based configuration with Standard/Custom modes and detailed mass breakdown
- **Track Validation** — Checks against Formula Student 2025 rules

---

## GUI Application

### Control Panel (Left Side)

The control panel offers two configuration modes:

| Mode | Description |
|------|-------------|
| **Standard** | All fields locked to `default.yaml` values — fast iteration with validated defaults |
| **Custom** | All fields editable — full control over every parameter including motor selection, drivetrain, inverter specs, and mass breakdown |

**Parameter sections:**
- **Vehicle** — Total mass (standard) or itemised mass breakdown: chassis, aero, suspension/tyres, powertrain, battery, electronics (custom)
- **Geometry** — Wheelbase, CoG position, track widths, CoG height
- **Aerodynamics** — Air density, Cd, Cl, frontal area
- **Tyre** — Friction coefficient, cornering stiffness (front/rear)
- **Powertrain** — Drivetrain selection, motor dropdown (from `motors.yaml`), motor specs, inverter specs (peak power, peak current, weight), gear ratio, wheel radius
- **Torque Vectoring** — Enable/disable, effectiveness, max transfer ratio, strategy
- **Battery** — Capacity, SoC bounds, max discharge power, voltage, current, regen settings

**Action buttons:**
- **Run Simulation** — Runs selected events and parameter sweeps
- **Save Results** — Exports all plots, config, and results text to a timestamped folder
- **Theme Toggle** — Dark/light mode switch (☀/☾ button in banner)

### Visualisation Tabs (Right Side)

| # | Tab | Description |
|---|-----|-------------|
| 1 | **Results** | Summary text: lap time, speed statistics, peak accelerations, energy consumption, battery state |
| 2 | **Track Layout** | Track geometry with start/finish markers |
| 3 | **Speed Track Map** | Velocity-coloured track overlay |
| 4 | **Speed vs Distance** | Speed profile with lateral grip limit curve |
| 5 | **Accel vs Distance** | Longitudinal acceleration profile using kinematic model |
| 6 | **RPM vs Distance** | Motor RPM throughout the lap |
| 7 | **Power Demand** | Instantaneous power consumption vs distance |
| 8 | **Gear Ratio Sweep** | Lap time sensitivity to gear ratio (autocross) |
| 9 | **Battery** | SoC, power draw, and cumulative energy vs distance |
| 10 | **Pack Optimiser** | Battery pack configuration sweep (series/parallel + power limit) |
| 11 | **Config Comparison** | Multi-config A/B comparison with bar chart and delta table |

### Config Comparison Workflow

1. Set up a vehicle configuration in the control panel (any parameters — motor, drivetrain, aero, mass, etc.)
2. Navigate to the **Config Comparison** tab and click **Add Config**
3. Name the configuration in the dialog
4. Return to the control panel, modify parameters, and click **Add Config** again
5. Repeat for as many configs as desired
6. Click **Run Comparison** — all configs are simulated on autocross
7. Results display as a horizontal bar chart (fastest in green) and a ranked delta table

---

## Physics Model & Assumptions

This simulator uses a **Quasi-Steady-State (QSS) point-mass model**. Understanding the assumptions is critical for interpreting results correctly.

### Solver Algorithm

The QSS solver uses a three-pass algorithm:

1. **Lateral Velocity Limit** — At each track point, calculate maximum cornering speed:
   ```
   v_lat = sqrt(a_max / |kappa|)
   ```
   where `kappa` is track curvature and `a_max` is grip-limited acceleration.

2. **Forward Pass** — Starting from rest, accelerate as hard as possible, limited by:
   - Tyre grip (friction circle)
   - Motor power (80 kW FS limit)
   - Inverter current limit (effective torque capping)
   - Motor RPM (top speed limit)
   - Look-ahead braking anticipation

3. **Backward Pass** — From finish, work backwards to find braking limits.

4. **Combine** — Final velocity: `v = min(v_forward, v_backward, v_lateral)`

5. **Integrate** — Lap time: `t = sum(ds / v_avg)` using average velocity per segment.

### Tyre Model Assumptions

| Assumption | Description | Impact |
|------------|-------------|--------|
| **Constant friction coefficient** | `mu = 1.6` (typical racing slick) | No load sensitivity — grip doesn't decrease at high loads |
| **Symmetric friction circle** | `ax^2 + ay^2 <= a_max^2` | Same grip for acceleration and braking |
| **No tyre temperature** | Grip is constant regardless of thermal state | Optimistic for cold tyres, pessimistic for optimal temps |
| **No slip ratio modelling** | Instantaneous grip available | No traction control effects |

**Friction Circle Equation:**
```
a_max = mu * (g + F_downforce / m)
```

### Aerodynamic Assumptions

| Assumption | Description |
|------------|-------------|
| **Constant coefficients** | Cd = 1.1, Cl = 1.5 (high downforce FS car) |
| **No pitch/ride height effects** | Aero loads don't change with vehicle attitude |
| **Symmetric flow** | No crosswind or yaw angle effects |

**Equations:**
```
F_downforce = 0.5 * rho * Cl * A * v^2
F_drag = 0.5 * rho * Cd * A * v^2
```

### Powertrain Assumptions

| Assumption | Description |
|------------|-------------|
| **Constant motor efficiency** | eta = 0.96 (EMRAX 208 peak) across all operating points |
| **No thermal derating** | Motor power doesn't reduce with temperature |
| **No drivetrain losses** | Gear efficiency = 100% (real: ~97%) |
| **Instant torque response** | No motor dynamics or inverter delays |
| **FS 80 kW cap** | Total power capped at 80 kW regardless of motor capability |

**Torque/Power Limiting:**
```
At low speed:  F = min(motor_torque, inverter_limited_torque) * n_motors * gear_ratio / wheel_radius
At high speed: F = P_max / v
P_max = min(motor_power * n_motors, 80 kW)
```

### Battery Assumptions

| Assumption | Description |
|------------|-------------|
| **Constant cell voltage** | Nominal 3.6V per cell (actual: 3.0–4.2V depending on SoC) |
| **No internal resistance losses** | Voltage sag under load not modelled in lap time |
| **Current limit not enforced in solver** | Power limit (80 kW) is binding, not current (175 A) |
| **Linear SoC depletion** | No Peukert effect or non-linear capacity |

### What Is NOT Modelled

| Feature | Status |
|---------|--------|
| Suspension dynamics | Not modelled |
| Roll/pitch angles | Not modelled |
| Tyre load sensitivity | Not modelled |
| Tyre temperature | Not modelled |
| Driver behaviour | Perfect driver assumed |
| Transient dynamics | Quasi-steady-state only |
| Variable weather | Fixed air density |

---

## Default Vehicle Configuration

The simulator is configured for a BAROx-style FS electric vehicle:

### Powertrain: 2× EMRAX 208 (Rear-Wheel Drive)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Configuration | 2RWD | Dual motor, rear axle |
| Motor | EMRAX 208 | 9.4 kg each |
| Peak Torque | 150 Nm | Per motor |
| Power Limit | 40 kW | Per motor (80 kW total, FS rules) |
| Max RPM | 7,000 | Top speed ~153 km/h |
| Efficiency | 96% | Peak efficiency |
| Gear Ratio | 3.0:1 | Single-stage reduction |
| Wheel Radius | 0.203 m | 16" wheel |

### Inverter: DTI HV-850

| Parameter | Value | Notes |
|-----------|-------|-------|
| Peak Power | 320 kW | Per inverter |
| Peak Current | 600 A | Per inverter |
| Weight | 6.9 kg | Per inverter |

### Battery: 142S5P Configuration

| Parameter | Value | Calculation |
|-----------|-------|-------------|
| Configuration | 142S5P | 710 cells total |
| Capacity | 6.65 kWh | (2.6 Ah × 5P) × (142 × 3.6V) |
| Nominal Voltage | 511 V | 142S × 3.6V |
| Max Current | 175 A | 5P × 35A per cell |
| Pack Mass | 45 kg | 710 × 45g × 1.4 casing factor |

### Vehicle

| Parameter | Value |
|-----------|-------|
| Total Mass | 239 kg |
| Wheelbase | 1.55 m |
| Track Width | 1.20 m |
| CoG Height | 0.28 m |
| Tyre Friction | 1.6 |
| Cd × A | 1.1 m² |
| Cl × A | 1.5 m² |

### Mass Breakdown (Custom Mode)

| Component | Mass (kg) |
|-----------|-----------|
| Chassis | 55 |
| Aero Package | 20 |
| Suspension & Tyres | 50 |
| Powertrain | 42.6 |
| Battery Pack | 45 |
| Electronics | 25 |
| **Total** | **237.6** |

*Note: In standard mode, total mass is 239 kg. In custom mode, mass is the sum of individual components plus powertrain-calculated masses.*

---

## Motor Database

Motors available for selection in `config/motors.yaml`:

| Motor | Weight | Peak Torque | Peak Power | Max RPM | Efficiency |
|-------|--------|-------------|------------|---------|------------|
| **EMRAX 188** | 7.1 kg | 100 Nm | 60 kW | 8,000 | 96% |
| **EMRAX 208** (default) | 9.4 kg | 150 Nm | 86 kW | 7,000 | 96% |
| **Plettenberg 15 WK** | 3.2 kg | 40 Nm | 20 kW | 8,000 | 90% |
| **Plettenberg 30-50 B10** | 6.2 kg | 80 Nm | 30 kW | 5,000 | 92% |

Add new motors by editing `config/motors.yaml`:
```yaml
your_motor:
  name: "Motor Name"
  weight_kg: 10.0
  peak_torque_Nm: 100
  peak_power_kW: 50
  max_rpm: 8000
  peak_efficiency: 0.95
```

### Drivetrain Options

| Config | Description | Motors |
|--------|-------------|--------|
| 1FWD | Single motor + differential, front | 1 |
| 1RWD | Single motor + differential, rear | 1 |
| 2FWD | Dual motor, front axle | 2 |
| 2RWD | Dual motor, rear axle | 2 |
| AWD | Four motors, all wheels | 4 |

---

## Project Structure

```
BAROx_Simulations/
├── main.py                     # CLI entry point
├── gui_main.py                 # GUI entry point
├── requirements.txt
├── config/
│   ├── default.yaml            # Default vehicle configuration
│   └── motors.yaml             # Motor database
├── solver/
│   ├── qss_speed.py            # QSS lap time solver
│   ├── battery.py              # Battery state simulation
│   └── metrics.py              # Lap time & energy calculations
├── physics/
│   ├── aero.py                 # Drag and downforce
│   ├── tyre.py                 # Friction circle model
│   ├── powertrain.py           # Motor torque/power limits
│   ├── resistive.py            # Rolling resistance
│   ├── weight_transfer.py      # Load transfer calculations
│   ├── bicycle_model.py        # Linear bicycle model
│   └── torque_vectoring.py     # TV yaw moment generation
├── models/
│   ├── vehicle.py              # Vehicle parameter dataclasses
│   └── track.py                # Track geometry
├── events/
│   ├── autocross_generator.py  # Autocross track builder
│   └── skidpad.py              # Skidpad geometry
├── gui/
│   ├── app.py                  # Main GUI application
│   ├── panels/
│   │   ├── control_panel.py            # Parameter input panel
│   │   ├── results_panel.py            # Visualisation tabs
│   │   ├── battery_optimiser_panel.py  # Pack optimisation
│   │   └── config_comparison_panel.py  # Multi-config comparison
│   └── widgets/
│       ├── parameter_group.py          # Parameter input helpers
│       └── plot_canvas.py              # Matplotlib canvas wrapper
├── plots/
│   └── plots.py                # Plotting utilities
└── resources/
    ├── BAROx_Logo.png          # Application banner logo
    └── BAROx_Icon.png          # Application icon
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LaithShareiyat/BAROx_Simulations.git
   cd BAROx_Simulations
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### GUI Application (Recommended)

```bash
python gui_main.py
```

The GUI provides:
- Standard/Custom configuration modes with motor database selection
- 11 visualisation tabs (results, track maps, speed/accel/RPM profiles, power demand, sweeps, battery analysis)
- Config comparison tool for A/B testing different vehicle setups
- Battery pack optimiser with series/parallel and power limit sweeps
- Dark/light theme toggle
- One-click results export (plots + config + summary text)

### Command Line Interface

```bash
python main.py
```

Interactive menu options:
| Option | Description |
|--------|-------------|
| Both | Runs Autocross followed by Skidpad simulation |
| Autocross | Sprint event simulation only |
| Skidpad | Skidpad event simulation only |

Vehicle parameters:
| Option | Description |
|--------|-------------|
| Standard | Uses parameters from `config/default.yaml` |
| Custom | Interactive prompts for custom values |

---

## Output

### Terminal Results

The simulator outputs detailed performance metrics:
- Lap time
- Average/max/min speed
- Peak accelerations (longitudinal and lateral)
- Energy consumption
- Battery state validation

### Plots

- **Track Layout** — Geometry with start/finish markers
- **Speed Track Map** — Velocity-coloured track overlay
- **Speed vs Distance** — Speed profile with lateral limit
- **Acceleration vs Distance** — Longitudinal acceleration profile
- **RPM vs Distance** — Motor RPM throughout the lap
- **Power Demand vs Distance** — Instantaneous power consumption
- **Gear Ratio Sweep** — Lap time sensitivity analysis
- **Battery Analysis** — State of Charge, Power, Cumulative Energy vs Distance
- **Config Comparison** — Bar chart ranking multiple configurations by lap time

### Save Results

Click **Save Results** in the GUI to export a timestamped folder containing:
- All plot images (PNG)
- Configuration snapshot (YAML)
- Results summary (text)

---

## Configuration

### Modifying Vehicle Parameters

Edit `config/default.yaml` to change:
- Vehicle mass and geometry
- Aerodynamic coefficients
- Tyre friction coefficient
- Powertrain configuration (drivetrain, motor, inverter, gearing)
- Battery pack specification
- Torque vectoring settings

---

## Validation & Accuracy

### Segment Resolution

The solver discretises the track into segments. Current defaults:
- Autocross: ~0.7 m mean segment length (2100+ points)
- Skidpad: ~0.57 m mean segment length (100 points per circle)

Convergence testing shows <1 ms lap time variation with segment sizes below 1 m.

### Known Limitations

1. **Optimistic acceleration** — No traction control or wheelspin modelling
2. **Optimistic braking** — Perfect brake bias assumed
3. **Steady-state only** — No transient vehicle dynamics
4. **No driver model** — Perfect racing line and inputs assumed

---

## Formula Student Rules Compliance

The simulator validates against FS 2025 rules:

| Rule | Limit | Enforced |
|------|-------|----------|
| Max powertrain power | 80 kW | Yes |
| Max track length (Autocross) | 1500 m | Yes |
| Max straight length | 80 m | Yes |
| Min hairpin outer radius | 9 m | Yes |
| Skidpad centre radius | 9.125 m | Yes |

---

## Future Development

Planned improvements:
- Save/load named configurations
- Run history table with logged metrics
- A/B overlay on plots (pin a baseline, overlay subsequent runs)
- Tyre load sensitivity (Pacejka or similar)
- Front/rear aero balance with axle-aware grip model
- Thermal modelling (motor and battery)
- Transient vehicle dynamics
- Driver model / racing line optimisation
- Endurance event simulation

---

## License

This project is developed for the University of Oxford Formula Student team (BAROx).

---

## Acknowledgements

- Blue Arrows Racing Oxford team members
- University of Oxford Engineering Science department
- EMRAX for motor specifications
