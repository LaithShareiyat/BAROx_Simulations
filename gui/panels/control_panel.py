"""Left control panel with event selection and parameter inputs."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os
from typing import Callable, Dict, Any, Optional, List


class ControlPanel(ttk.Frame):
    """Left panel containing event selection, parameters, and controls."""

    def __init__(
        self,
        parent,
        on_run: Callable,
        on_config_change: Callable = None,
        on_save_results: Callable = None,
        **kwargs,
    ):
        """
        Create the control panel.

        Args:
            parent: Parent widget
            on_run: Callback function when Run button is clicked
            on_config_change: Callback when configuration changes
            on_save_results: Callback for saving results
        """
        super().__init__(parent, **kwargs)

        self.on_run = on_run
        self.on_config_change = on_config_change
        self.on_save_results = on_save_results
        self.default_config = self._load_default_config()
        self.motor_database = self._load_motor_database()

        # Create scrollable frame
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Enable mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Build UI components
        self._create_event_selection()
        self._create_config_selection()
        self._create_parameter_groups()
        self._create_action_buttons()

        # Initialise state
        self._update_parameter_state()

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        if event.num == 4:  # Linux scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # Linux scroll down
            self.canvas.yview_scroll(1, "units")
        else:  # Windows/Mac
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _load_default_config(self) -> dict:
        """Load default configuration from YAML."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "default.yaml",
        )
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load default config: {e}")
            return self._get_fallback_defaults()

    def _load_motor_database(self) -> Dict[str, dict]:
        """Load motor database from YAML."""
        motors_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "motors.yaml",
        )
        try:
            with open(motors_path, "r") as f:
                data = yaml.safe_load(f)
                return data.get("motors", {})
        except Exception as e:
            print(f"Warning: Could not load motor database: {e}")
            return {
                "default_motor": {
                    "name": "Default: 2x 40kW motors",
                    "allowed_drivetrains": ["2RWD", "2FWD"],
                    "weight_kg": 10.0,
                    "peak_power_kW": 80,
                    "peak_torque_Nm": 100,
                    "max_rpm": 6000,
                    "peak_efficiency": 0.85,
                    "motor_constant_Nm_A": 0.5,
                    "peak_current_A": 200,
                }
            }

    def _get_motor_list(self) -> List[tuple]:
        """Get list of (display_name, motor_id) tuples for dropdown."""
        motors = []
        for motor_id, params in self.motor_database.items():
            display_name = params.get("name", motor_id)
            motors.append((display_name, motor_id))
        return sorted(motors, key=lambda x: x[0])

    def _get_fallback_defaults(self) -> dict:
        """Fallback default values if config file not found."""
        return {
            "vehicle": {
                "mass_kg": 239,
                "g": 9.81,
                "Crr": 0.015,
                # Mass breakdown
                "mass_chassis_aero_kg": 75,
                "mass_suspension_tyres_kg": 50,
                "mass_powertrain_kg": 44,  # 2× EMRAX 208 (9.4kg) + 25kg overhead
                "mass_battery_kg": 45,  # 142S5P pack
                "mass_electronics_kg": 25,
            },
            "aero": {"rho": 1.225, "Cd": 1.1, "Cl": 1.5, "A": 1.0},
            "tyre": {"mu": 1.6},
            "powertrain": {
                "drivetrain": "2RWD",
                "motor": "emrax_208",  # EMRAX 208 motor
                "motor_power_kW": 40,  # FS rules: 80 kW total (2 × 40 kW)
                "motor_torque_Nm": 150,  # EMRAX 208
                "motor_rpm_max": 7000,  # EMRAX 208
                "motor_efficiency": 0.96,  # EMRAX 208
                "motor_weight_kg": 9.4,  # EMRAX 208 weight [kg]
                "gear_ratio": 3.5,
                "wheel_radius_m": 0.203,
                "powertrain_overhead_kg": 25.0,  # Inverters, wiring, cooling [kg]
            },
            "battery": {
                "capacity_kWh": 6.65,
                "initial_soc": 1.0,
                "min_soc": 0.1,  # 142S5P
                "max_discharge_kW": 80,
                "eta_discharge": 0.95,  # FS 80 kW limit
                "nominal_voltage_V": 511,
                "max_current_A": 175,  # 142S5P pack limits
                "regen_enabled": False,
                "eta_regen": 0.85,
                "max_regen_kW": 50,
                "regen_capture_percent": 100,
            },
            "torque_vectoring": {
                "enabled": False,
                "effectiveness": 1.0,
                "max_torque_transfer": 0.5,
                "strategy": "load_proportional",
            },
        }

    def _create_event_selection(self):
        """Create event selection radio buttons."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Event Selection", padding=(10, 5)
        )
        frame.pack(fill="x", padx=10, pady=5)

        self.event_var = tk.StringVar(value="both")

        events = [("Autocross", "autocross"), ("Skidpad", "skidpad"), ("Both", "both")]

        for text, value in events:
            rb = ttk.Radiobutton(frame, text=text, variable=self.event_var, value=value)
            rb.pack(anchor="w", padx=5, pady=2)

    def _create_config_selection(self):
        """Create configuration type selection."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="Configuration", padding=(10, 5)
        )
        frame.pack(fill="x", padx=10, pady=5)

        self.config_var = tk.StringVar(value="standard")

        ttk.Radiobutton(
            frame,
            text="Standard (default.yaml)",
            variable=self.config_var,
            value="standard",
            command=self._update_parameter_state,
        ).pack(anchor="w", padx=5, pady=2)

        ttk.Radiobutton(
            frame,
            text="Custom",
            variable=self.config_var,
            value="custom",
            command=self._update_parameter_state,
        ).pack(anchor="w", padx=5, pady=2)

    def _create_parameter_groups(self):
        """Create all parameter input groups."""
        self.param_entries: Dict[str, Dict[str, tk.StringVar]] = {}
        self.param_widgets: Dict[
            str, Dict[str, ttk.Entry]
        ] = {}  # Store entry widgets for enabling/disabling

        # Vehicle parameters (with mass breakdown for custom mode)
        self._create_vehicle_section()

        # Aero parameters
        self._create_param_section(
            "AERODYNAMICS",
            "aero",
            [
                ("rho", "Air Density", "kg/m³"),
                ("Cd", "Drag Coeff (Cd)", ""),
                ("Cl", "Lift Coeff (Cl)", ""),
                ("A", "Frontal Area", "m²"),
            ],
        )

        # Tyre parameters
        self._create_param_section(
            "TYRE",
            "tyre",
            [
                ("mu", "Friction (μ)", ""),
            ],
        )

        # Powertrain parameters with drivetrain selection
        self._create_powertrain_section()

        # Torque vectoring section (visible in custom mode)
        self._create_torque_vectoring_section()

        # Battery parameters with enable checkbox
        self._create_battery_section()

    def _create_vehicle_section(self):
        """Create vehicle parameters section with mass breakdown for custom mode."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="VEHICLE", padding=(10, 5))
        frame.pack(fill="x", padx=10, pady=5)
        self.vehicle_frame = frame

        self.param_entries["vehicle"] = {}
        self.param_widgets["vehicle"] = {}
        defaults = self.default_config.get("vehicle", {})

        # Standard mass field (shown in standard mode)
        self.mass_standard_widgets = []
        row = 0

        lbl = ttk.Label(frame, text="Total Mass", width=16, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.mass_standard_widgets.append(lbl)

        var = tk.StringVar(value=str(defaults.get("mass_kg", 250)))
        entry = ttk.Entry(frame, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        self.mass_standard_widgets.append(entry)

        unit_lbl = ttk.Label(frame, text="kg", foreground="gray", width=6)
        unit_lbl.grid(row=row, column=2, sticky="w", pady=2)
        self.mass_standard_widgets.append(unit_lbl)

        self.param_entries["vehicle"]["mass_kg"] = var
        self.param_widgets["vehicle"]["mass_kg"] = entry
        self.mass_standard_entry = entry
        self.mass_standard_var = var

        # Mass breakdown fields (shown in custom mode)
        self.mass_breakdown_widgets = []
        self.mass_breakdown_vars = {}
        self.mass_breakdown_entries = {}

        mass_components = [
            ("mass_chassis_aero_kg", "Chassis & Aero", 75),
            ("mass_suspension_tyres_kg", "Suspension & Tyres", 50),
            ("mass_powertrain_kg", "Powertrain", 45),
            ("mass_battery_kg", "Battery Systems", 55),
            ("mass_electronics_kg", "Electronics & Other", 25),
        ]

        for i, (key, label, default) in enumerate(mass_components):
            comp_row = row + 1 + i
            default_val = defaults.get(key, default)

            lbl = ttk.Label(frame, text=label, width=16, anchor="w")
            lbl.grid(row=comp_row, column=0, sticky="w", padx=5, pady=1)
            self.mass_breakdown_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=comp_row, column=1, sticky="w", padx=5, pady=1)
            entry.bind("<KeyRelease>", self._update_total_mass)
            self.mass_breakdown_widgets.append(entry)

            unit_lbl = ttk.Label(frame, text="kg", foreground="gray", width=6)
            unit_lbl.grid(row=comp_row, column=2, sticky="w", pady=1)
            self.mass_breakdown_widgets.append(unit_lbl)

            self.param_entries["vehicle"][key] = var
            self.param_widgets["vehicle"][key] = entry
            self.mass_breakdown_vars[key] = var
            self.mass_breakdown_entries[key] = entry

        # Total mass display (read-only, shown in custom mode)
        total_row = row + len(mass_components) + 1

        sep = ttk.Separator(frame, orient="horizontal")
        sep.grid(row=total_row, column=0, columnspan=3, sticky="ew", padx=5, pady=3)
        self.mass_breakdown_widgets.append(sep)

        total_row += 1
        total_lbl = ttk.Label(
            frame,
            text="Total Mass",
            width=16,
            anchor="w",
            font=("TkDefaultFont", 9, "bold"),
        )
        total_lbl.grid(row=total_row, column=0, sticky="w", padx=5, pady=2)
        self.mass_breakdown_widgets.append(total_lbl)

        self.total_mass_var = tk.StringVar(value="250")
        total_entry = ttk.Entry(
            frame, textvariable=self.total_mass_var, width=10, state="readonly"
        )
        total_entry.grid(row=total_row, column=1, sticky="w", padx=5, pady=2)
        self.mass_breakdown_widgets.append(total_entry)

        total_unit = ttk.Label(frame, text="kg", foreground="gray", width=6)
        total_unit.grid(row=total_row, column=2, sticky="w", pady=2)
        self.mass_breakdown_widgets.append(total_unit)

        # Other vehicle parameters (g, Crr)
        other_params = [
            ("g", "Gravity", "m/s²"),
            ("Crr", "Rolling Resistance", ""),
        ]

        self.vehicle_other_widgets = []
        other_start_row = total_row + 1

        for i, (key, label, unit) in enumerate(other_params):
            param_row = other_start_row + i
            default_val = defaults.get(key, 0)

            lbl = ttk.Label(frame, text=label, width=16, anchor="w")
            lbl.grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
            self.vehicle_other_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=param_row, column=1, sticky="w", padx=5, pady=2)
            self.vehicle_other_widgets.append(entry)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground="gray", width=6)
                unit_lbl.grid(row=param_row, column=2, sticky="w", pady=2)
                self.vehicle_other_widgets.append(unit_lbl)

            self.param_entries["vehicle"][key] = var
            self.param_widgets["vehicle"][key] = entry

        # Initially hide mass breakdown (standard mode)
        self._toggle_mass_breakdown(show_breakdown=False)

    def _toggle_mass_breakdown(self, show_breakdown: bool):
        """Show/hide mass breakdown fields based on mode."""
        if show_breakdown:
            # Hide standard mass field
            for widget in self.mass_standard_widgets:
                widget.grid_remove()
            # Show breakdown fields
            for widget in self.mass_breakdown_widgets:
                widget.grid()
            # Update total
            self._update_total_mass()
        else:
            # Show standard mass field
            for widget in self.mass_standard_widgets:
                widget.grid()
            # Hide breakdown fields
            for widget in self.mass_breakdown_widgets:
                widget.grid_remove()

    def _update_total_mass(self, event=None):
        """Calculate and display total mass from components."""
        total = 0
        for key, var in self.mass_breakdown_vars.items():
            try:
                total += float(var.get())
            except ValueError:
                pass
        self.total_mass_var.set(f"{total:.1f}")
        # Also update the main mass_kg variable
        self.mass_standard_var.set(f"{total:.1f}")

    def _create_powertrain_section(self):
        """Create powertrain section with motor selection, drivetrain, and parameters."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="POWERTRAIN", padding=(10, 5)
        )
        frame.pack(fill="x", padx=10, pady=5)
        self.powertrain_frame = frame

        self.param_entries["powertrain"] = {}
        self.param_widgets["powertrain"] = {}
        defaults = self.default_config.get("powertrain", {})

        row = 0

        # Motor selection dropdown
        motor_label = ttk.Label(frame, text="Motor", width=16, anchor="w")
        motor_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)

        motor_list = self._get_motor_list()
        motor_names = [name for name, _ in motor_list]
        self.motor_id_map = {name: motor_id for name, motor_id in motor_list}

        self.motor_var = tk.StringVar()
        default_motor = defaults.get("motor", "default_motor")
        # Find display name for default motor
        for name, motor_id in motor_list:
            if motor_id == default_motor:
                self.motor_var.set(name)
                break
        else:
            if motor_names:
                self.motor_var.set(motor_names[0])

        self.motor_combo = ttk.Combobox(
            frame,
            textvariable=self.motor_var,
            values=motor_names,
            state="readonly",
            width=25,
        )
        self.motor_combo.grid(
            row=row, column=1, columnspan=2, sticky="w", padx=5, pady=2
        )
        self.motor_combo.bind("<<ComboboxSelected>>", self._on_motor_change)
        self.motor_widgets = [motor_label, self.motor_combo]
        row += 1

        # Drivetrain selection (dropdown)
        dt_label = ttk.Label(frame, text="Drivetrain", width=16, anchor="w")
        dt_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.motor_widgets.append(dt_label)

        self.drivetrain_var = tk.StringVar(value=defaults.get("drivetrain", "2RWD"))
        # Map legacy values to new format
        if self.drivetrain_var.get() == "RWD":
            self.drivetrain_var.set("2RWD")
        elif self.drivetrain_var.get() == "FWD":
            self.drivetrain_var.set("2FWD")

        drivetrain_options = ["1FWD", "1RWD", "2FWD", "2RWD", "AWD"]
        self.drivetrain_combo = ttk.Combobox(
            frame,
            textvariable=self.drivetrain_var,
            values=drivetrain_options,
            state="readonly",
            width=8,
        )
        self.drivetrain_combo.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        self.drivetrain_combo.bind(
            "<<ComboboxSelected>>", lambda e: self._on_drivetrain_change()
        )
        self.motor_widgets.append(self.drivetrain_combo)

        # Description label
        self.drivetrain_desc_var = tk.StringVar(value="")
        dt_desc = ttk.Label(
            frame, textvariable=self.drivetrain_desc_var, foreground="gray", width=20
        )
        dt_desc.grid(row=row, column=2, sticky="w", pady=2)
        self.motor_widgets.append(dt_desc)
        self._update_drivetrain_description()

        self.param_entries["powertrain"]["drivetrain"] = self.drivetrain_var
        row += 1

        # Motor weight (per motor) - read-only, comes from database
        lbl = ttk.Label(frame, text="Motor Weight", width=16, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.motor_widgets.append(lbl)

        motor_weight_var = tk.StringVar(
            value=str(defaults.get("motor_weight_kg", 10.0))
        )
        motor_weight_entry = ttk.Entry(
            frame, textvariable=motor_weight_var, width=10, state="readonly"
        )
        motor_weight_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        self.motor_widgets.append(motor_weight_entry)
        self.motor_spec_entries_weight = motor_weight_entry  # Track for state updates

        unit_lbl = ttk.Label(frame, text="kg/motor", foreground="gray", width=8)
        unit_lbl.grid(row=row, column=2, sticky="w", pady=2)
        self.motor_widgets.append(unit_lbl)

        self.param_entries["powertrain"]["motor_weight_kg"] = motor_weight_var
        self.param_widgets["powertrain"]["motor_weight_kg"] = motor_weight_entry
        row += 1

        # Powertrain overhead - editable in custom mode
        lbl = ttk.Label(frame, text="PT Overhead", width=16, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.motor_widgets.append(lbl)

        overhead_var = tk.StringVar(
            value=str(defaults.get("powertrain_overhead_kg", 25.0))
        )
        overhead_entry = ttk.Entry(frame, textvariable=overhead_var, width=10)
        overhead_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        overhead_entry.bind("<KeyRelease>", self._update_powertrain_display)
        self.motor_widgets.append(overhead_entry)
        self.overhead_entry = overhead_entry  # Track for state updates

        unit_lbl = ttk.Label(frame, text="kg", foreground="gray", width=8)
        unit_lbl.grid(row=row, column=2, sticky="w", pady=2)
        self.motor_widgets.append(unit_lbl)

        self.param_entries["powertrain"]["powertrain_overhead_kg"] = overhead_var
        self.param_widgets["powertrain"]["powertrain_overhead_kg"] = overhead_entry
        row += 1

        # Differential mass - only shown for single-motor configs
        self.diff_label = ttk.Label(frame, text="Differential", width=16, anchor="w")
        self.diff_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)

        self.diff_mass_var = tk.StringVar(value="5.0")
        self.diff_entry = ttk.Entry(frame, textvariable=self.diff_mass_var, width=10)
        self.diff_entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
        self.diff_entry.bind("<KeyRelease>", self._update_powertrain_display)

        self.diff_unit_lbl = ttk.Label(frame, text="kg", foreground="gray", width=8)
        self.diff_unit_lbl.grid(row=row, column=2, sticky="w", pady=2)

        # Store references for show/hide
        self.diff_widgets = [self.diff_label, self.diff_entry, self.diff_unit_lbl]
        self.param_entries["powertrain"]["differential_mass_kg"] = self.diff_mass_var
        row += 1

        # Initially hide differential field (shown when 1-motor selected)
        self._update_differential_visibility()

        # Separator before motor specs
        sep1 = ttk.Separator(frame, orient="horizontal")
        sep1.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=3)
        self.motor_widgets.append(sep1)
        row += 1

        # Motor spec header
        spec_label = ttk.Label(
            frame,
            text="Motor Specs:",
            width=16,
            anchor="w",
            font=("TkDefaultFont", 9, "italic"),
        )
        spec_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.motor_widgets.append(spec_label)
        row += 1

        # Motor parameters (from database, read-only - specs come from motors.yaml)
        motor_params = [
            ("motor_power_kW", "Peak Power", "kW"),
            ("motor_torque_Nm", "Peak Torque", "Nm"),
            ("motor_rpm_max", "Max RPM", "rpm"),
            ("motor_efficiency", "Efficiency", ""),
        ]

        self.motor_spec_entries = []  # Track motor spec entries for read-only state
        for key, label, unit in motor_params:
            default_val = defaults.get(key, 0)

            lbl = ttk.Label(frame, text=label, width=16, anchor="w")
            lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            self.motor_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10, state="readonly")
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            self.motor_widgets.append(entry)
            self.motor_spec_entries.append(entry)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground="gray", width=6)
                unit_lbl.grid(row=row, column=2, sticky="w", pady=2)
                self.motor_widgets.append(unit_lbl)

            self.param_entries["powertrain"][key] = var
            self.param_widgets["powertrain"][key] = entry
            row += 1

        # Transmission parameters
        trans_params = [
            ("gear_ratio", "Gear Ratio", ":1"),
            ("wheel_radius_m", "Wheel Radius", "m"),
        ]

        for key, label, unit in trans_params:
            default_val = defaults.get(key, 0)

            lbl = ttk.Label(frame, text=label, width=16, anchor="w")
            lbl.grid(row=row, column=0, sticky="w", padx=5, pady=2)
            self.motor_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            entry.bind("<KeyRelease>", self._update_powertrain_display)
            self.motor_widgets.append(entry)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground="gray", width=6)
                unit_lbl.grid(row=row, column=2, sticky="w", pady=2)
                self.motor_widgets.append(unit_lbl)

            self.param_entries["powertrain"][key] = var
            self.param_widgets["powertrain"][key] = entry
            row += 1

        # Separator before calculated values
        sep = ttk.Separator(frame, orient="horizontal")
        sep.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.motor_widgets.append(sep)
        row += 1

        # Calculated values display (read-only)
        calc_label = ttk.Label(
            frame,
            text="Calculated:",
            width=16,
            anchor="w",
            font=("TkDefaultFont", 9, "italic"),
        )
        calc_label.grid(row=row, column=0, sticky="w", padx=5, pady=2)
        self.motor_widgets.append(calc_label)
        row += 1

        # Number of motors display
        lbl = ttk.Label(frame, text="Motors", width=16, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(lbl)
        self.n_motors_var = tk.StringVar(value="2")
        n_entry = ttk.Entry(
            frame, textvariable=self.n_motors_var, width=10, state="readonly"
        )
        n_entry.grid(row=row, column=1, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(n_entry)
        row += 1

        # Motor mass display
        lbl = ttk.Label(frame, text="Motor Mass", width=16, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(lbl)
        self.motor_mass_var = tk.StringVar(value="20.0")
        mm_entry = ttk.Entry(
            frame, textvariable=self.motor_mass_var, width=10, state="readonly"
        )
        mm_entry.grid(row=row, column=1, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(mm_entry)
        unit_lbl = ttk.Label(frame, text="kg", foreground="gray", width=6)
        unit_lbl.grid(row=row, column=2, sticky="w", pady=1)
        self.motor_widgets.append(unit_lbl)
        row += 1

        # Total powertrain mass display
        lbl = ttk.Label(frame, text="PT Total Mass", width=16, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(lbl)
        self.pt_total_mass_var = tk.StringVar(value="45.0")
        pt_entry = ttk.Entry(
            frame, textvariable=self.pt_total_mass_var, width=10, state="readonly"
        )
        pt_entry.grid(row=row, column=1, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(pt_entry)
        unit_lbl = ttk.Label(frame, text="kg", foreground="gray", width=6)
        unit_lbl.grid(row=row, column=2, sticky="w", pady=1)
        self.motor_widgets.append(unit_lbl)
        row += 1

        # Total power display
        lbl = ttk.Label(frame, text="Total Power", width=16, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(lbl)
        self.total_power_var = tk.StringVar(value="160")
        p_entry = ttk.Entry(
            frame, textvariable=self.total_power_var, width=10, state="readonly"
        )
        p_entry.grid(row=row, column=1, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(p_entry)
        unit_lbl = ttk.Label(frame, text="kW", foreground="gray", width=6)
        unit_lbl.grid(row=row, column=2, sticky="w", pady=1)
        self.motor_widgets.append(unit_lbl)
        row += 1

        # Max force display
        lbl = ttk.Label(frame, text="Max Force", width=16, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(lbl)
        self.max_force_var = tk.StringVar(value="3111")
        f_entry = ttk.Entry(
            frame, textvariable=self.max_force_var, width=10, state="readonly"
        )
        f_entry.grid(row=row, column=1, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(f_entry)
        unit_lbl = ttk.Label(frame, text="N", foreground="gray", width=6)
        unit_lbl.grid(row=row, column=2, sticky="w", pady=1)
        self.motor_widgets.append(unit_lbl)
        row += 1

        # Max speed display
        lbl = ttk.Label(frame, text="Max Speed", width=16, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(lbl)
        self.max_speed_var = tk.StringVar(value="40.4 m/s (145 km/h)")
        s_entry = ttk.Entry(
            frame, textvariable=self.max_speed_var, width=18, state="readonly"
        )
        s_entry.grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=1)
        self.motor_widgets.append(s_entry)

        # Initialise display
        self._on_motor_change()
        self._update_powertrain_display()

    def _on_motor_change(self, event=None):
        """Handle motor selection change - update specs from database."""
        motor_name = self.motor_var.get()
        motor_id = self.motor_id_map.get(motor_name)

        if motor_id and motor_id in self.motor_database:
            motor = self.motor_database[motor_id]
            # Update motor specs from database
            if "motor_power_kW" in self.param_entries["powertrain"]:
                self.param_entries["powertrain"]["motor_power_kW"].set(
                    str(motor.get("peak_power_kW", 80))
                )
            if "motor_torque_Nm" in self.param_entries["powertrain"]:
                self.param_entries["powertrain"]["motor_torque_Nm"].set(
                    str(motor.get("peak_torque_Nm", 100))
                )
            if "motor_rpm_max" in self.param_entries["powertrain"]:
                self.param_entries["powertrain"]["motor_rpm_max"].set(
                    str(motor.get("max_rpm", 6000))
                )
            if "motor_efficiency" in self.param_entries["powertrain"]:
                self.param_entries["powertrain"]["motor_efficiency"].set(
                    str(motor.get("peak_efficiency", 0.85))
                )
            if "motor_weight_kg" in self.param_entries["powertrain"]:
                self.param_entries["powertrain"]["motor_weight_kg"].set(
                    str(motor.get("weight_kg", 10.0))
                )

            # Update allowed drivetrains based on motor
            self._update_allowed_drivetrains(motor)

        self._update_powertrain_display()

    def _update_allowed_drivetrains(self, motor: dict):
        """Update drivetrain dropdown based on motor's allowed configurations."""
        if not hasattr(self, "drivetrain_combo"):
            return

        # Get allowed drivetrains from motor config (default: all)
        all_drivetrains = ["1FWD", "1RWD", "2FWD", "2RWD", "AWD"]
        allowed = motor.get("allowed_drivetrains", all_drivetrains)

        # Update combobox values
        self.drivetrain_combo["values"] = allowed

        # If current selection is not in allowed list, switch to first allowed
        current = self.drivetrain_var.get()
        if current not in allowed:
            self.drivetrain_var.set(allowed[0])
            self._on_drivetrain_change()

    def _create_torque_vectoring_section(self):
        """Create torque vectoring section (visible in custom mode only)."""
        frame = ttk.LabelFrame(
            self.scrollable_frame, text="TORQUE VECTORING", padding=(10, 5)
        )
        frame.pack(fill="x", padx=10, pady=5)
        self.tv_frame = frame

        defaults = self.default_config.get("torque_vectoring", {})

        # Enable checkbox
        self.tv_enabled = tk.BooleanVar(value=defaults.get("enabled", False))
        self.tv_checkbox = ttk.Checkbutton(
            frame,
            text="Enable Torque Vectoring",
            variable=self.tv_enabled,
            command=self._update_tv_state,
        )
        self.tv_checkbox.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5)

        # TV parameters
        self.param_entries["torque_vectoring"] = {}
        self.tv_widgets = []  # Store widgets for enabling/disabling

        tv_params = [
            ("effectiveness", "Effectiveness", ""),
            ("max_torque_transfer", "Max Transfer", ""),
        ]

        for i, (key, label, unit) in enumerate(tv_params):
            default_val = defaults.get(key, 1.0 if key == "effectiveness" else 0.5)
            param_row = i + 1

            lbl = ttk.Label(frame, text=label, width=16, anchor="w")
            lbl.grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
            self.tv_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=param_row, column=1, sticky="w", padx=5, pady=2)
            self.tv_widgets.append(entry)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground="gray", width=6)
                unit_lbl.grid(row=param_row, column=2, sticky="w", pady=2)
                self.tv_widgets.append(unit_lbl)

            self.param_entries["torque_vectoring"][key] = var

        # Strategy selection (radio buttons)
        strategy_row = len(tv_params) + 1
        strategy_lbl = ttk.Label(frame, text="Strategy", width=16, anchor="w")
        strategy_lbl.grid(row=strategy_row, column=0, sticky="w", padx=5, pady=2)
        self.tv_widgets.append(strategy_lbl)

        self.tv_strategy_var = tk.StringVar(
            value=defaults.get("strategy", "load_proportional")
        )
        self.param_entries["torque_vectoring"]["strategy"] = self.tv_strategy_var

        strategy_frame = ttk.Frame(frame)
        strategy_frame.grid(
            row=strategy_row, column=1, columnspan=2, sticky="w", padx=5, pady=2
        )
        self.tv_widgets.append(strategy_frame)

        strategies = [("Load Prop.", "load_proportional"), ("Fixed Bias", "fixed_bias")]
        self.tv_strategy_widgets = []
        for text, value in strategies:
            rb = ttk.Radiobutton(
                strategy_frame, text=text, variable=self.tv_strategy_var, value=value
            )
            rb.pack(side="left", padx=5)
            self.tv_strategy_widgets.append(rb)

        # Note about drivetrain requirement
        note_row = strategy_row + 1
        note_lbl = ttk.Label(
            frame,
            text="  Not available for FWD",
            foreground="gray",
            font=("TkDefaultFont", 8, "italic"),
        )
        note_lbl.grid(
            row=note_row, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 2)
        )
        self.tv_widgets.append(note_lbl)

        # Initially hide section and disable widgets (only shown in custom mode)
        self._update_tv_state()

    def _update_tv_state(self):
        """Enable/disable TV parameter entries based on TV checkbox and drivetrain."""
        is_custom = self.config_var.get() == "custom"
        drivetrain = (
            self.drivetrain_var.get() if hasattr(self, "drivetrain_var") else "2RWD"
        )
        n_motors = self._get_n_motors(drivetrain)

        # TV not available for single-motor configs (mechanical differential)
        # or front-wheel drive (steering/drive conflict)
        is_single_motor = n_motors == 1
        is_fwd = drivetrain in ("FWD", "2FWD", "1FWD")
        tv_not_available = is_single_motor or is_fwd
        tv_enabled = self.tv_enabled.get()

        # If TV not available, force TV off and disable the checkbox
        if tv_not_available and tv_enabled:
            self.tv_enabled.set(False)
            tv_enabled = False

        # TV checkbox: disabled if not custom mode OR if TV not available
        checkbox_state = (
            "normal" if (is_custom and not tv_not_available) else "disabled"
        )
        if hasattr(self, "tv_checkbox"):
            self.tv_checkbox.configure(state=checkbox_state)

        # Only enable TV params if: custom mode AND TV enabled AND TV available
        state = (
            "normal"
            if (is_custom and tv_enabled and not tv_not_available)
            else "disabled"
        )

        for widget in self.tv_widgets:
            if isinstance(widget, ttk.Entry):
                widget.configure(state=state)

        # Enable/disable strategy radio buttons
        if hasattr(self, "tv_strategy_widgets"):
            for rb in self.tv_strategy_widgets:
                rb.configure(state=state)

    def _update_drivetrain_description(self):
        """Update the drivetrain description label."""
        if not hasattr(self, "drivetrain_desc_var"):
            return
        drivetrain = self.drivetrain_var.get()
        descriptions = {
            "1FWD": "1 motor + diff, front",
            "1RWD": "1 motor + diff, rear",
            "2FWD": "2 motors, front axle",
            "2RWD": "2 motors, rear axle",
            "AWD": "4 motors, all wheels",
            # Legacy support
            "FWD": "2 motors, front axle",
            "RWD": "2 motors, rear axle",
        }
        self.drivetrain_desc_var.set(descriptions.get(drivetrain, ""))

    def _update_differential_visibility(self):
        """Show/hide differential mass field based on drivetrain selection."""
        if not hasattr(self, "diff_widgets"):
            return
        drivetrain = (
            self.drivetrain_var.get() if hasattr(self, "drivetrain_var") else "2RWD"
        )
        is_single_motor = drivetrain.startswith("1")

        for widget in self.diff_widgets:
            if is_single_motor:
                widget.grid()  # Show
            else:
                widget.grid_remove()  # Hide but keep grid position

    def _get_n_motors(self, drivetrain: str) -> int:
        """Get number of motors for a drivetrain configuration."""
        if drivetrain == "AWD":
            return 4
        elif drivetrain.startswith("1"):
            return 1
        else:
            # '2FWD', '2RWD', 'FWD', 'RWD' all have 2 motors
            return 2

    def _on_drivetrain_change(self):
        """Handle drivetrain selection change."""
        self._update_drivetrain_description()
        self._update_differential_visibility()
        self._update_powertrain_display()
        # Update TV state (disable if FWD or single-motor selected)
        if hasattr(self, "tv_enabled"):
            self._update_tv_state()

    def _update_powertrain_display(self, event=None):
        """Update calculated powertrain values based on inputs."""
        try:
            drivetrain = self.drivetrain_var.get()
            n_motors = self._get_n_motors(drivetrain)

            motor_power = float(
                self.param_entries["powertrain"]["motor_power_kW"].get()
            )
            motor_torque = float(
                self.param_entries["powertrain"]["motor_torque_Nm"].get()
            )
            motor_rpm = float(self.param_entries["powertrain"]["motor_rpm_max"].get())
            gear_ratio = float(self.param_entries["powertrain"]["gear_ratio"].get())
            wheel_radius = float(
                self.param_entries["powertrain"]["wheel_radius_m"].get()
            )

            # Motor weight calculations
            motor_weight = float(
                self.param_entries["powertrain"]["motor_weight_kg"].get()
            )
            overhead = float(
                self.param_entries["powertrain"]["powertrain_overhead_kg"].get()
            )
            motor_mass_total = motor_weight * n_motors

            # Add differential mass for single-motor configs
            diff_mass = 0.0
            if drivetrain.startswith("1") and hasattr(self, "diff_mass_var"):
                try:
                    diff_mass = float(self.diff_mass_var.get())
                except ValueError:
                    diff_mass = 0.0

            pt_mass_total = motor_mass_total + overhead + diff_mass

            total_power = motor_power * n_motors
            fx_max = (
                (motor_torque * n_motors * gear_ratio) / wheel_radius
                if wheel_radius > 0
                else 0
            )
            wheel_rpm = motor_rpm / gear_ratio if gear_ratio > 0 else 0
            v_max = (wheel_rpm * 2 * 3.14159 * wheel_radius) / 60

            self.n_motors_var.set(str(n_motors))
            self.motor_mass_var.set(f"{motor_mass_total:.1f}")
            self.pt_total_mass_var.set(f"{pt_mass_total:.1f}")
            self.total_power_var.set(f"{total_power:.0f}")
            self.max_force_var.set(f"{fx_max:.0f}")
            self.max_speed_var.set(f"{v_max:.1f} m/s ({v_max * 3.6:.0f} km/h)")

            # Update mass breakdown powertrain field if in custom mode
            if (
                hasattr(self, "mass_breakdown_vars")
                and "mass_powertrain_kg" in self.mass_breakdown_vars
            ):
                is_custom = self.config_var.get() == "custom"
                if is_custom:
                    self.mass_breakdown_vars["mass_powertrain_kg"].set(
                        f"{pt_mass_total:.1f}"
                    )
                    self._update_total_mass()
        except (ValueError, ZeroDivisionError, KeyError):
            # Keep current values if inputs are invalid
            pass

    def _create_param_section(self, title: str, section: str, params: list):
        """Create a parameter section with labelled entries."""
        frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding=(10, 5))
        frame.pack(fill="x", padx=10, pady=5)

        self.param_entries[section] = {}
        self.param_widgets[section] = {}  # Store entry widgets
        defaults = self.default_config.get(section, {})

        for i, (key, label, unit) in enumerate(params):
            default_val = defaults.get(key, 0)

            lbl = ttk.Label(frame, text=label, width=16, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", padx=5, pady=2)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground="gray", width=6)
                unit_lbl.grid(row=i, column=2, sticky="w", pady=2)

            self.param_entries[section][key] = var
            self.param_widgets[section][key] = entry  # Store reference to entry widget

    def _create_battery_section(self):
        """Create battery parameters section with enable checkbox and regen options."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="BATTERY", padding=(10, 5))
        frame.pack(fill="x", padx=10, pady=5)

        self.battery_enabled = tk.BooleanVar(value=True)
        self.battery_checkbox = ttk.Checkbutton(
            frame,
            text="Enable Battery Analysis",
            variable=self.battery_enabled,
            command=self._update_battery_state,
        )
        self.battery_checkbox.grid(
            row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )

        self.param_entries["battery"] = {}
        self.battery_widgets = []  # Store entry widgets for battery section
        defaults = self.default_config.get("battery", {})

        params = [
            ("capacity_kWh", "Capacity", "kWh"),
            ("initial_soc", "Initial SoC", ""),
            ("min_soc", "Min SoC", ""),
            ("max_discharge_kW", "Max Discharge", "kW"),
            ("eta_discharge", "Efficiency", ""),
            ("nominal_voltage_V", "Pack Voltage", "V"),
            ("max_current_A", "Max Current", "A"),
        ]

        for i, (key, label, unit) in enumerate(params):
            default_val = defaults.get(key, 0)

            lbl = ttk.Label(frame, text=label, width=16, anchor="w")
            lbl.grid(row=i + 1, column=0, sticky="w", padx=5, pady=2)
            self.battery_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=i + 1, column=1, sticky="w", padx=5, pady=2)
            self.battery_widgets.append(entry)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground="gray", width=6)
                unit_lbl.grid(row=i + 1, column=2, sticky="w", pady=2)
                self.battery_widgets.append(unit_lbl)

            self.param_entries["battery"][key] = var

        # Regenerative braking section
        regen_row = len(params) + 1

        # Separator
        sep = ttk.Separator(frame, orient="horizontal")
        sep.grid(row=regen_row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.battery_widgets.append(sep)

        # Regen enable checkbox
        regen_row += 1
        self.regen_enabled = tk.BooleanVar(value=defaults.get("regen_enabled", False))
        self.regen_checkbox = ttk.Checkbutton(
            frame,
            text="Enable Regenerative Braking",
            variable=self.regen_enabled,
            command=self._update_regen_state,
        )
        self.regen_checkbox.grid(
            row=regen_row, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )
        self.battery_widgets.append(self.regen_checkbox)

        # Regen parameters
        self.regen_widgets = []  # Store regen entry widgets for enabling/disabling
        regen_params = [
            ("eta_regen", "Regen Efficiency", ""),
            ("max_regen_kW", "Max Regen Power", "kW"),
            ("regen_capture_percent", "Braking Capture", "%"),
        ]

        for i, (key, label, unit) in enumerate(regen_params):
            default_val = defaults.get(key, 0)
            param_row = regen_row + 1 + i

            lbl = ttk.Label(frame, text=label, width=16, anchor="w")
            lbl.grid(row=param_row, column=0, sticky="w", padx=5, pady=2)
            self.regen_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=param_row, column=1, sticky="w", padx=5, pady=2)
            self.regen_widgets.append(entry)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground="gray", width=6)
                unit_lbl.grid(row=param_row, column=2, sticky="w", pady=2)
                self.regen_widgets.append(unit_lbl)

            self.param_entries["battery"][key] = var

        # Initially update regen state
        self._update_regen_state()

    def _update_battery_state(self):
        """Enable/disable battery parameter entries."""
        is_custom = self.config_var.get() == "custom"
        # Only enable if both custom mode AND battery is enabled
        state = "normal" if (is_custom and self.battery_enabled.get()) else "disabled"
        for widget in self.battery_widgets:
            if isinstance(widget, ttk.Entry):
                widget.configure(state=state)
            elif (
                isinstance(widget, ttk.Checkbutton) and widget != self.battery_checkbox
            ):
                # Enable regen checkbox only if battery is enabled
                widget.configure(state=state)
        # Also update regen state
        self._update_regen_state()

    def _update_regen_state(self):
        """Enable/disable regen parameter entries based on regen checkbox."""
        is_custom = self.config_var.get() == "custom"
        battery_enabled = self.battery_enabled.get()
        regen_enabled = self.regen_enabled.get()
        # Only enable regen params if: custom mode AND battery enabled AND regen enabled
        state = (
            "normal"
            if (is_custom and battery_enabled and regen_enabled)
            else "disabled"
        )
        for widget in self.regen_widgets:
            if isinstance(widget, ttk.Entry):
                widget.configure(state=state)

    def _create_action_buttons(self):
        """Create load, save, and run buttons."""
        # Config buttons frame - use grid for alignment with input fields
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.pack(fill="x", padx=10, pady=10)

        # Configure columns to expand equally
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        btn_frame.columnconfigure(2, weight=1)

        ttk.Button(btn_frame, text="Load Config", command=self._load_config).grid(
            row=0, column=0, sticky="ew", padx=(0, 3)
        )
        ttk.Button(btn_frame, text="Save Config", command=self._save_config).grid(
            row=0, column=1, sticky="ew", padx=3
        )
        ttk.Button(btn_frame, text="Reset", command=self._reset_to_defaults).grid(
            row=0, column=2, sticky="ew", padx=(3, 0)
        )

        # Run button (prominent)
        run_frame = ttk.Frame(self.scrollable_frame)
        run_frame.pack(fill="x", padx=10, pady=(5, 10))

        self.run_button = ttk.Button(
            run_frame,
            text="RUN SIMULATION",
            command=self._on_run_clicked,
            style="Accent.TButton",
        )
        self.run_button.pack(fill="x", ipady=10)

        # Save results button
        save_frame = ttk.Frame(self.scrollable_frame)
        save_frame.pack(fill="x", padx=10, pady=(0, 15))

        self.save_button = ttk.Button(
            save_frame,
            text="SAVE RESULTS",
            command=self._on_save_clicked,
            style="Accent.TButton",
        )
        self.save_button.pack(fill="x", ipady=10)

    def _on_save_clicked(self):
        """Handle save results button click."""
        if self.on_save_results:
            self.on_save_results()

    def _update_parameter_state(self):
        """Enable/disable parameter entries based on config selection."""
        is_custom = self.config_var.get() == "custom"
        state = "normal" if is_custom else "disabled"

        # Toggle mass breakdown view
        self._toggle_mass_breakdown(show_breakdown=is_custom)

        # Update all parameter entry widgets
        for section, widgets in self.param_widgets.items():
            for key, entry in widgets.items():
                # Skip mass breakdown entries in standard mode (they're hidden)
                if not is_custom and key.startswith("mass_") and key != "mass_kg":
                    continue
                # Motor specs are always readonly (come from database)
                if key in (
                    "motor_power_kW",
                    "motor_torque_Nm",
                    "motor_rpm_max",
                    "motor_efficiency",
                    "motor_weight_kg",
                ):
                    entry.configure(state="readonly")
                else:
                    entry.configure(state=state)

        # Update motor dropdown - disabled in standard mode
        if hasattr(self, "motor_combo"):
            self.motor_combo.configure(state="readonly" if is_custom else "disabled")

        # Update drivetrain radio buttons
        if hasattr(self, "drivetrain_widgets"):
            for rb in self.drivetrain_widgets:
                rb.configure(state=state)

        # Update powertrain overhead entry
        if hasattr(self, "overhead_entry"):
            self.overhead_entry.configure(state=state)

        # Update battery widgets (entries only)
        for widget in self.battery_widgets:
            if isinstance(widget, ttk.Entry):
                # Only enable if custom mode AND battery is enabled
                battery_state = (
                    "normal"
                    if (is_custom and self.battery_enabled.get())
                    else "disabled"
                )
                widget.configure(state=battery_state)

        # Update battery checkbox
        if hasattr(self, "battery_checkbox"):
            self.battery_checkbox.configure(state=state)

        # Update regen checkbox and entries
        if hasattr(self, "regen_checkbox"):
            # Regen checkbox enabled only if battery is enabled
            regen_cb_state = (
                "normal" if (is_custom and self.battery_enabled.get()) else "disabled"
            )
            self.regen_checkbox.configure(state=regen_cb_state)
            self._update_regen_state()

        # Update torque vectoring checkbox and entries
        if hasattr(self, "tv_checkbox"):
            self.tv_checkbox.configure(state=state)
            self._update_tv_state()

        # Reset to defaults when switching to standard
        if not is_custom:
            self._reset_to_defaults()

    def _load_config(self):
        """Load configuration from file."""
        filepath = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config"
            ),
        )

        if filepath:
            try:
                with open(filepath, "r") as f:
                    config = yaml.safe_load(f)
                self._set_config(config)
                self.config_var.set("custom")
                messagebox.showinfo(
                    "Success",
                    f"Configuration loaded from:\n{os.path.basename(filepath)}",
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{e}")

    def _save_config(self):
        """Save current configuration to file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config"
            ),
        )

        if filepath:
            try:
                config = self.get_config()
                with open(filepath, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                messagebox.showinfo(
                    "Success", f"Configuration saved to:\n{os.path.basename(filepath)}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration:\n{e}")

    def _reset_to_defaults(self):
        """Reset all parameters to default values."""
        self._set_config(self.default_config)

    def _set_config(self, config: dict):
        """Set all parameter values from a config dictionary."""
        for section, params in self.param_entries.items():
            if section in config:
                for key, var in params.items():
                    if key in config[section]:
                        var.set(str(config[section][key]))

        # If mass breakdown values aren't in config, calculate from total mass
        if "vehicle" in config:
            total_mass = config["vehicle"].get("mass_kg", 250)
            if "mass_chassis_aero_kg" not in config["vehicle"]:
                # Apply default ratios
                self.mass_breakdown_vars["mass_chassis_aero_kg"].set(
                    str(round(total_mass * 0.30))
                )
                self.mass_breakdown_vars["mass_suspension_tyres_kg"].set(
                    str(round(total_mass * 0.20))
                )
                self.mass_breakdown_vars["mass_powertrain_kg"].set(
                    str(round(total_mass * 0.18))
                )
                self.mass_breakdown_vars["mass_battery_kg"].set(
                    str(round(total_mass * 0.22))
                )
                self.mass_breakdown_vars["mass_electronics_kg"].set(
                    str(round(total_mass * 0.10))
                )
            self._update_total_mass()

        # Handle motor and drivetrain selection if present
        if "powertrain" in config:
            # Set motor selection
            if hasattr(self, "motor_var") and "motor" in config["powertrain"]:
                motor_id = config["powertrain"]["motor"]
                # Find display name for this motor ID
                for name, mid in self.motor_id_map.items():
                    if mid == motor_id:
                        self.motor_var.set(name)
                        self._on_motor_change()
                        break

            # Set drivetrain (map legacy values to new format)
            if hasattr(self, "drivetrain_var"):
                drivetrain = config["powertrain"].get("drivetrain", "2RWD")
                # Map legacy values
                if drivetrain == "RWD":
                    drivetrain = "2RWD"
                elif drivetrain == "FWD":
                    drivetrain = "2FWD"
                self.drivetrain_var.set(drivetrain)
                self._update_drivetrain_description()

            self._update_powertrain_display()

        # Handle battery enabled state
        if "battery" in config:
            self.battery_enabled.set(config["battery"].get("enabled", True))
            # Handle regen enabled state
            if hasattr(self, "regen_enabled"):
                self.regen_enabled.set(config["battery"].get("regen_enabled", False))
            self._update_battery_state()

        # Handle torque vectoring enabled state
        if "torque_vectoring" in config and hasattr(self, "tv_enabled"):
            self.tv_enabled.set(config["torque_vectoring"].get("enabled", False))
            if hasattr(self, "tv_strategy_var"):
                self.tv_strategy_var.set(
                    config["torque_vectoring"].get("strategy", "load_proportional")
                )
            self._update_tv_state()

    def _on_run_clicked(self):
        """Handle run button click."""
        if self.on_run:
            self.on_run()

    def get_config(self) -> dict:
        """Get current configuration as a dictionary."""
        config = {}
        is_custom = self.config_var.get() == "custom"

        for section, params in self.param_entries.items():
            config[section] = {}
            for key, var in params.items():
                try:
                    val = var.get()
                    # Convert to appropriate type
                    if "." in val:
                        config[section][key] = float(val)
                    else:
                        config[section][key] = int(val) if val.isdigit() else float(val)
                except ValueError:
                    config[section][key] = var.get()

        # In custom mode, calculate total mass from components
        if is_custom:
            total_mass = 0
            for key in self.mass_breakdown_vars:
                total_mass += config["vehicle"].get(key, 0)
            config["vehicle"]["mass_kg"] = total_mass

        # Add motor ID to powertrain config
        if hasattr(self, "motor_var") and hasattr(self, "motor_id_map"):
            motor_name = self.motor_var.get()
            motor_id = self.motor_id_map.get(motor_name)
            if motor_id:
                config["powertrain"]["motor"] = motor_id

        # Add battery enabled flag
        config["battery"]["enabled"] = self.battery_enabled.get()

        # Add regen enabled flag
        if hasattr(self, "regen_enabled"):
            config["battery"]["regen_enabled"] = self.regen_enabled.get()

        # Add torque vectoring enabled flag
        if hasattr(self, "tv_enabled"):
            if "torque_vectoring" not in config:
                config["torque_vectoring"] = {}
            config["torque_vectoring"]["enabled"] = self.tv_enabled.get()

        return config

    def get_event(self) -> str:
        """Get selected event type."""
        return self.event_var.get()

    def set_running(self, running: bool):
        """Set the running state of the panel."""
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.save_button.configure(state=state)
        if running:
            self.run_button.configure(text="RUNNING...")
        else:
            self.run_button.configure(text="RUN SIMULATION")
