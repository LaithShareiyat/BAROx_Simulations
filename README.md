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

- **QSS Lap Time Simulation** - Three-pass algorithm (lateral limit, forward acceleration, backward braking) with friction circle constraint
- **Battery Pack Optimiser** - Sweep series/parallel configurations to find optimal battery pack sizing
- **Regenerative Braking** - Optional regen modelling with configurable capture percentage
- **Torque Vectoring** - Yaw moment generation from differential motor torque (RWD/AWD)
- **Bicycle Model** - Linear tyre model with slip angle calculations for cornering validation
- **Weight Transfer** - Longitudinal and lateral load transfer during acceleration and cornering
- **GUI Application** - Full graphical interface with real-time visualisation
- **Configurable Vehicle** - YAML-based configuration with motor database
- **Track Validation** - Checks against Formula Student 2025 rules

---

## Physics Model & Assumptions

This simulator uses a **Quasi-Steady-State (QSS) point-mass model**. Understanding the assumptions is critical for interpreting results correctly.

### Solver Algorithm

The QSS solver uses a three-pass algorithm:

1. **Lateral Velocity Limit** - At each track point, calculate maximum cornering speed:
   ```
   v_lat = sqrt(a_max / |kappa|)
   ```
   where `kappa` is track curvature and `a_max` is grip-limited acceleration.

2. **Forward Pass** - Starting from rest, accelerate as hard as possible, limited by:
   - Tyre grip (friction circle)
   - Motor power (80 kW FS limit)
   - Motor RPM (top speed limit)
   - Look-ahead braking anticipation

3. **Backward Pass** - From finish, work backwards to find braking limits.

4. **Combine** - Final velocity: `v = min(v_forward, v_backward, v_lateral)`

5. **Integrate** - Lap time: `t = sum(ds / v_avg)` using average velocity per segment.

### Tyre Model Assumptions

| Assumption | Description | Impact |
|------------|-------------|--------|
| **Constant friction coefficient** | `mu = 1.6` (typical racing slick) | No load sensitivity - grip doesn't decrease at high loads |
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

**Torque/Power Limiting:**
```
At low speed:  F = (torque * n_motors * gear_ratio) / wheel_radius
At high speed: F = P_max / v
Crossover at:  v_cross = P_max / F_max
```

### Battery Assumptions

| Assumption | Description |
|------------|-------------|
| **Constant cell voltage** | Nominal 3.6V per cell (actual: 3.0-4.2V depending on SoC) |
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

### Powertrain: 2x EMRAX 208 (Rear-Wheel Drive)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Configuration | 2RWD | Dual motor, rear axle |
| Motor | EMRAX 208 | 9.4 kg each |
| Peak Torque | 150 Nm | Per motor |
| Power Limit | 40 kW | Per motor (80 kW total, FS rules) |
| Max RPM | 7000 | Top speed ~153 km/h |
| Efficiency | 96% | Peak efficiency |
| Gear Ratio | 3.5:1 | Single-stage reduction |
| Wheel Radius | 0.203 m | 16" wheel |

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

---

## Project Structure

```
BAROx_Simulations/
├── main.py                 # CLI entry point
├── gui_main.py             # GUI entry point
├── config/
│   ├── default.yaml        # Default vehicle configuration
│   └── motors.yaml         # Motor database
├── solver/
│   ├── qss_speed.py        # QSS lap time solver
│   ├── battery.py          # Battery state simulation
│   └── metrics.py          # Lap time & energy calculations
├── physics/
│   ├── aero.py             # Drag and downforce
│   ├── tyre.py             # Friction circle model
│   ├── powertrain.py       # Motor torque/power limits
│   ├── resistive.py        # Rolling resistance
│   ├── weight_transfer.py  # Load transfer calculations
│   ├── bicycle_model.py    # Linear bicycle model
│   └── torque_vectoring.py # TV yaw moment generation
├── models/
│   ├── vehicle.py          # Vehicle parameter dataclasses
│   └── track.py            # Track geometry
├── events/
│   ├── autocross_generator.py  # Autocross track builder
│   └── skidpad.py              # Skidpad geometry
├── gui/
│   ├── app.py              # Main GUI application
│   └── panels/
│       ├── control_panel.py        # Parameter input
│       ├── results_panel.py        # Visualisation
│       └── battery_optimiser_panel.py  # Pack optimisation
└── plots/
    └── plots.py            # Plotting utilities
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
- Interactive parameter adjustment
- Real-time simulation results
- Track layout and speed map visualisation
- Battery state-of-charge plots
- Battery pack optimiser tool

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

- **Track Layout** - Geometry with start/finish markers
- **Speed Track Map** - Velocity-coloured track overlay
- **Speed Profile** - Speed vs distance with lateral limit
- **Battery Analysis**:
  - State of Charge vs Distance
  - Power vs Distance
  - Cumulative Energy vs Distance

---

## Configuration

### Modifying Vehicle Parameters

Edit `config/default.yaml` to change:
- Vehicle mass and geometry
- Aerodynamic coefficients
- Tyre friction coefficient
- Powertrain configuration
- Battery pack specification

### Adding Motors

Add new motor definitions to `config/motors.yaml`:
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

## Validation & Accuracy

### Segment Resolution

The solver discretises the track into segments. Current defaults:
- Autocross: ~0.7 m mean segment length (2100+ points)
- Skidpad: ~0.57 m mean segment length (100 points per circle)

Convergence testing shows <1 ms lap time variation with segment sizes below 1 m.

### Known Limitations

1. **Optimistic acceleration** - No traction control or wheelspin modelling
2. **Optimistic braking** - Perfect brake bias assumed
3. **Steady-state only** - No transient vehicle dynamics
4. **No driver model** - Perfect racing line and inputs assumed

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
- Tyre load sensitivity (Pacejka or similar)
- Thermal modelling (motor and battery)
- Transient vehicle dynamics
- Driver model / racing line optimisation
- Endurance event simulation
- Monte Carlo sensitivity analysis

---

## License

This project is developed for the University of Oxford Formula Student team (BAROx).

---

## Acknowledgements

- Blue Arrows Racing Oxford team members
- University of Oxford Engineering Science department
- EMRAX for motor specifications
