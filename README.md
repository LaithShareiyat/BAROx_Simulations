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

## Lap Time Simulation — Final Formulation

This section documents the complete mathematical formulation of the Quasi-Steady-State (QSS) lap time simulator as implemented. The solver treats the vehicle as a point mass traversing a discretised track, solving for the maximum achievable velocity at each point subject to grip, powertrain, and geometric constraints.

### 1. Track Discretisation

The track is represented as a sequence of $n$ points with position $(x_i, y_i)$, arc-length $s_i$, segment length $\Delta s_i = s_{i+1} - s_i$, and signed curvature $\kappa_i$.

**Adaptive refinement** (optional): when the track has fewer than a minimum number of points, it is interpolated using curvature-weighted spacing — more points are placed in corners, fewer on straights:

```
density(s) = (1 - w) × uniform + w × (|κ(s)| + ε)
```

where $w$ is the curvature weight (default 0.8) and $\varepsilon$ prevents straights from having zero density.

### 2. Solver Algorithm — Five Steps

#### Step 1: Lateral (Cornering) Velocity Limit

At each track point, the maximum speed is limited by the grip available for cornering. Since aerodynamic downforce depends on speed, this requires an iterative solution:

```
Iterate until converged:
    F_down   = ½ ρ CL_A v²
    a_max    = F_grip(F_z) / m        where F_z = mg + F_down
    v_lat[i] = min(√(a_max / |κ_i|), v_max_rpm)
```

On straight sections ($|\kappa| < 10^{-6}$), the lateral limit is the RPM-limited top speed:

```
v_max_rpm = (RPM_max / G) × 2π r_wheel / 60
```

**With bicycle model**: after computing $v_\text{lat}$ from the friction circle, the bicycle model checks for tyre saturation. If either axle exceeds its grip limit (slip angle beyond peak), the speed is progressively reduced (down to 50%) until the yaw moment balance is satisfied:

```
F_yf + F_yr = m a_y
F_yf L_f − F_yr L_r + M_z,TV = 0
α_f = F_yf / C_αf,   α_r = F_yr / C_αr
```

**With torque vectoring**: the effective lateral grip is augmented by a TV multiplier (capped at 1.15×):

```
a_max,TV = a_max × min(1 + ΔF_y,eq / F_z,rear × η_TV, 1.15)
```

#### Step 2: Forward Pass (Acceleration Limit)

Starting from rest (open track) or the lateral limit (closed track), the forward pass accelerates as hard as possible at each step.

**Traction force** — the minimum of grip-limited and powertrain-limited force:

```
F_x = min(F_x,grip, F_x,motor)
```

*Grip-limited force* has two formulations:

**Point-mass (simple):**
```
a_max  = μ (g + F_down / m)
a_x    = √(a_max² − a_y²)           (friction circle)
F_x,grip = m × a_x
```

**Axle-aware (with bicycle model):**
```
Static loads:     W_f = mg(L_r/L) + F_down,f     W_r = mg(L_f/L) + F_down,r
Weight transfer:  ΔW = m a_x h / L
Dynamic loads:    F_zf = W_f − ΔW               F_zr = W_r + ΔW
Lateral split:    F_yr = (m a_y L_f + M_z,TV) / L     F_yf = m a_y − F_yr

For driven axle:  F_x,grip = F_available(F_z, F_y_required)
```

The axle-aware model solves iteratively because $a_x$ and $\Delta W$ are coupled (weight transfer depends on acceleration, which depends on grip, which depends on load).

*Powertrain-limited force:*
```
Low speed (torque-limited):   F_x,motor = T_eff × n_motors × G / r_wheel
High speed (power-limited):   F_x,motor = P_max / v
Beyond RPM limit:             F_x,motor = 0

where:
    T_eff = min(T_motor, K_m × I_inverter)      (inverter current limit)
    P_max = min(P_motor × n_motors, P_inverter × n_inv, P_rules)
    P_rules = 80 kW                              (FS rules cap)
```

**Net acceleration:**
```
a_x = max((F_x − F_drag − F_rr) / m, 0)
v[i+1]² = v[i]² + 2 a_x Δs
```

**Look-ahead braking anticipation** — at each step, the solver checks whether the vehicle can brake to upcoming lateral limits within the next 20 segments:

```
v_max,entry = √(v_lat[j]² + 2 a_brake Δs_total)
v_fwd[i+1] = min(v_kinematic, v_lat[i+1], v_max,lookahead)
```

#### Step 3: Backward Pass (Braking Limit)

Working backwards from the end, the solver finds the maximum entry speed at each point. A predictor-corrector (2 iterations) re-evaluates aerodynamic forces at the higher entry speed:

```
a_x,brake = a_x,grip + (F_drag + F_rr) / m       (drag assists braking)
v[i]² = v[i+1]² + 2 a_x,brake Δs
v_bwd[i] = min(√(v_prev²), v_lat[i])
```

For the axle-aware model, both axles contribute to braking with optimal brake bias (proportional to vertical load). Weight transfers forward under braking ($F_{zf}$ increases, $F_{zr}$ decreases).

**Closed tracks**: the backward pass enforces periodicity by constraining $v_\text{end} = v_\text{start}$.

#### Step 4: Combine All Limits

```
v[i] = min(v_fwd[i], v_bwd[i], v_lat[i])
```

#### Step 5: Integrate Lap Time

```
v_avg[i] = (v[i] + v[i+1]) / 2
Δt[i]    = Δs[i] / max(v_avg[i], 0.1)
t_lap    = Σ Δt[i]
```

### 3. Tyre Models

The simulator supports three tyre models of increasing fidelity:

#### 3a. Constant-μ (Default)

```
F_max = μ × F_z
a_max = μ × (g + F_down / m)
Friction circle: a_x² + a_y² ≤ a_max²
```

#### 3b. Pacejka Magic Formula

Load-sensitive tyre model with combined slip:

**Pure slip:**
```
F = D sin(C arctan(Bx − E(Bx − arctan(Bx))))

where:
    D = (a₁ F_z + a₂) F_z            (peak force — load sensitive)
    BCD = a₃ sin(a₄ arctan(a₅ F_z))  (cornering stiffness)
    B = BCD / (CD)
    E = a₆ F_z + a₇                  (curvature factor, clamped < 1)
```

**Combined slip (similarity method):**
```
F_x = G_xα(α) × F_x0(κ)       G_xα = cos(C_xα arctan(B_xα α))
F_y = G_yκ(κ) × F_y0(α)       G_yκ = cos(C_yκ arctan(B_yκ κ))
```

The Pacejka model captures load sensitivity: splitting load across multiple tyres yields more total force than one tyre at the combined load, which matters for the axle-aware grip calculation.

#### 3c. Thermal Grip Model (Optional)

Temperature-dependent grip scaling with a parabolic window:

```
grip(T) = max(0, 1 − ((T − T_opt) / T_width)²)

Thermal energy balance:
    C_th dT/dt = k_heat |F| V_slip − h_cool(V)(T − T_amb)
    h_cool(V) = h_static + h_speed × V
```

### 4. Aerodynamics

```
F_drag     = ½ ρ (Cd × A) v²
F_downforce = ½ ρ (CL_A) v²

where:
    CL_A = CL_A_f + CL_A_r
    CL_A_f = Cl_f × A_f    (front effective downforce area)
    CL_A_r = Cl_r × A_r    (rear effective downforce area)
    CD_A = Cd × A           (drag reference area)
```

Downforce is distributed front/rear by the aero balance:

```
aero_balance_front = CL_A_f / (CL_A_f + CL_A_r)
F_down,f = F_down × aero_balance_front
F_down,r = F_down × (1 − aero_balance_front)
```

### 5. Powertrain

The extended powertrain model has three operating regimes:

```
                  ┌─ Torque-limited:  F = T_eff × n × G / r       (v < v_cross)
F_x,motor(v) =   ├─ Power-limited:   F = P_max / v               (v_cross ≤ v ≤ v_RPM)
                  └─ RPM-limited:     F = 0                        (v > v_RPM)

v_cross = P_max / F_x,max
v_RPM   = (RPM_max / G) × 2π r / 60

Effective torque (inverter current limit):
    T_eff = min(T_motor, K_m × I_inverter)

Total power (rules-capped):
    P_max = min(P_motor × n, P_inverter × n, P_rules)
```

### 6. Weight Transfer

**Longitudinal:**
```
ΔW_long = m a_x h_cg / L
F_zf = W_f,static − ΔW_long      (front unloads under acceleration)
F_zr = W_r,static + ΔW_long      (rear loads up under acceleration)
```

**Lateral (per axle, load-proportional roll model):**
```
ΔW_lat,f = m |a_y| h_cg × (W_f / W) / t_f
ΔW_lat,r = m |a_y| h_cg × (W_r / W) / t_r
```

Wheel loads are clamped to non-negative (wheel lift protection).

### 7. Bicycle Model

The linear bicycle model solves for steady-state force and moment equilibrium at each track point:

```
Kinematics:
    r = V κ                          (yaw rate from path curvature)
    a_y = V² κ                       (lateral acceleration)

Force equilibrium:
    F_yf + F_yr = m a_y

Moment equilibrium:
    F_yf L_f − F_yr L_r + M_z,TV = 0

    → F_yr = (m a_y L_f + M_z,TV) / L
    → F_yf = m a_y − F_yr

Linear tyre model:
    α_f = F_yf / C_αf               (front slip angle)
    α_r = F_yr / C_αr               (rear slip angle)

Sideslip:   β = −α_r + L_r r / V
Steering:   δ = α_f + β + L_f r / V

Grip utilisation:
    u_f = |F_yf| / F_y,max,f        u_r = |F_yr| / F_y,max,r
    Saturated if max(u_f, u_r) > 1

Understeer gradient:
    K = W_f/C_αf − W_r/C_αr
    K > 0: understeer    K < 0: oversteer    K = 0: neutral
```

### 8. Torque Vectoring

Torque vectoring generates a yaw moment by applying differential torque to left and right wheels:

```
M_z = ΔF_x × (t / 2)
```

**Strategies:**

| Strategy | Torque Split |
|----------|-------------|
| Load Proportional | $\text{split}_w = F_{z,w} / F_{z,\text{driven}}$, deviation limited by `max_torque_transfer` |
| Fixed Bias | Equal split ± `bias/2` to outer/inner wheels |

**TV benefits:**
- Lateral: equivalent lateral force $F_{y,\text{eq}} = M_z / L_r$ augments cornering grip (capped at 15%)
- Traction: load-proportional split uses all available grip vs equal-split being limited by least-loaded wheel

Only available for multi-motor drivetrains (2FWD, 2RWD, AWD). Single-motor configs use a mechanical differential.

### 9. Battery & Energy Model

**Instantaneous power demand:**
```
F_x,wheel = m a_x + F_drag + F_rr

If F_x > 0 (accelerating):   P_elec = F_x × v / η_motor
If F_x < 0 (braking, regen):  P_regen = −|F_x| × v × (capture% / 100) × η_regen
                                P_regen = min(P_regen, P_regen,max)
```

**State of charge integration:**
```
E_segment = P_battery × Δt / 3600          [kWh]
SoC[i+1] = SoC_initial − E_cumulative / C_battery
```

Power is capped by both the discharge power limit and the current limit ($P = V \times I$).

### 10. Resistive Forces

```
F_rr = C_rr × m × g            (rolling resistance — constant, speed-independent)
F_drag = ½ ρ CD_A v²            (aerodynamic drag — quadratic with speed)
```

### Assumptions & Limitations

| Category | Assumption | Impact |
|----------|------------|--------|
| **Tyre** | Constant μ = 1.6 (default) | No load sensitivity unless Pacejka model selected |
| **Tyre** | Symmetric friction circle | Same grip for acceleration and braking |
| **Tyre** | No slip ratio modelling | Instantaneous grip, no traction control effects |
| **Aero** | Constant coefficients | No pitch/ride-height/yaw effects |
| **Aero** | Front/rear balance is area-weighted | No dynamic aero balance shift |
| **Powertrain** | Constant motor efficiency (η = 0.96) | No efficiency map across RPM/torque |
| **Powertrain** | No thermal derating | Motor power doesn't reduce with temperature |
| **Powertrain** | No drivetrain losses | Gear efficiency = 100% (real: ~97%) |
| **Powertrain** | Instant torque response | No motor dynamics or inverter delays |
| **Battery** | Constant cell voltage (3.6 V) | No voltage sag under load (actual: 3.0–4.2 V) |
| **Battery** | No internal resistance losses | IR drop not modelled |
| **Battery** | Linear SoC depletion | No Peukert effect |
| **Vehicle** | No suspension dynamics | No roll, pitch, or ride-height changes |
| **Vehicle** | Perfect driver | Optimal racing line and instantaneous inputs assumed |
| **Vehicle** | Quasi-steady-state only | No transient vehicle dynamics |
| **Environment** | Fixed air density | No variable weather or altitude effects |

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

**Planned improvements:**
- Save/load named configurations
- Run history table with logged metrics
- A/B overlay on plots (pin a baseline, overlay subsequent runs)
- Motor thermal derating
- Transient vehicle dynamics
- Driver model / racing line optimisation
- Endurance event simulation (if the LTS is used by the OURS)

---

## Acknowledgements

- Blue Arrows Racing Oxford team members
- University of Oxford Engineering Science department
