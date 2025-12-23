"""Left control panel with event selection and parameter inputs."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os
from typing import Callable, Dict, Any, Optional


class ControlPanel(ttk.Frame):
    """Left panel containing event selection, parameters, and controls."""

    def __init__(self, parent, on_run: Callable, on_config_change: Callable = None,
                 on_save_results: Callable = None, **kwargs):
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

        # Create scrollable frame
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            '<Configure>',
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Enable mousewheel scrolling
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind_all('<Button-4>', self._on_mousewheel)
        self.canvas.bind_all('<Button-5>', self._on_mousewheel)

        self.scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)

        # Build UI components
        self._create_event_selection()
        self._create_config_selection()
        self._create_parameter_groups()
        self._create_action_buttons()

        # Initialize state
        self._update_parameter_state()

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        if event.num == 4:  # Linux scroll up
            self.canvas.yview_scroll(-1, 'units')
        elif event.num == 5:  # Linux scroll down
            self.canvas.yview_scroll(1, 'units')
        else:  # Windows/Mac
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    def _load_default_config(self) -> dict:
        """Load default configuration from YAML."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'config', 'default.yaml'
        )
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load default config: {e}")
            return self._get_fallback_defaults()

    def _get_fallback_defaults(self) -> dict:
        """Fallback default values if config file not found."""
        return {
            'vehicle': {
                'mass_kg': 250, 'g': 9.81, 'Crr': 0.015,
                # Mass breakdown (percentages of total mass)
                'mass_chassis_aero_kg': 75,      # 30%
                'mass_suspension_tyres_kg': 50,  # 20%
                'mass_powertrain_kg': 45,        # 18%
                'mass_battery_kg': 55,           # 22%
                'mass_electronics_kg': 25,       # 10%
            },
            'aero': {'rho': 1.225, 'Cd': 1.1, 'Cl': 1.5, 'A': 1.0},
            'tyre': {'mu': 1.6},
            'powertrain': {'P_max_kW': 160, 'Fx_max_N': 3900},  # 2 × 80kW RWD
            'battery': {
                'capacity_kWh': 6, 'initial_soc': 1.0, 'min_soc': 0.1,
                'max_discharge_kW': 160, 'eta_discharge': 0.95,  # Match 2 × 80kW
                'nominal_voltage_V': 400, 'max_current_A': 500,
                'regen_enabled': False, 'eta_regen': 0.85,
                'max_regen_kW': 50, 'regen_capture_percent': 100
            }
        }

    def _create_event_selection(self):
        """Create event selection radio buttons."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Event Selection", padding=(10, 5))
        frame.pack(fill='x', padx=10, pady=5)

        self.event_var = tk.StringVar(value='both')

        events = [
            ('Autocross', 'autocross'),
            ('Skidpad', 'skidpad'),
            ('Both', 'both')
        ]

        for text, value in events:
            rb = ttk.Radiobutton(frame, text=text, variable=self.event_var, value=value)
            rb.pack(anchor='w', padx=5, pady=2)

    def _create_config_selection(self):
        """Create configuration type selection."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="Configuration", padding=(10, 5))
        frame.pack(fill='x', padx=10, pady=5)

        self.config_var = tk.StringVar(value='standard')

        ttk.Radiobutton(
            frame, text="Standard (default.yaml)",
            variable=self.config_var, value='standard',
            command=self._update_parameter_state
        ).pack(anchor='w', padx=5, pady=2)

        ttk.Radiobutton(
            frame, text="Custom",
            variable=self.config_var, value='custom',
            command=self._update_parameter_state
        ).pack(anchor='w', padx=5, pady=2)

    def _create_parameter_groups(self):
        """Create all parameter input groups."""
        self.param_entries: Dict[str, Dict[str, tk.StringVar]] = {}
        self.param_widgets: Dict[str, Dict[str, ttk.Entry]] = {}  # Store entry widgets for enabling/disabling

        # Vehicle parameters (with mass breakdown for custom mode)
        self._create_vehicle_section()

        # Aero parameters
        self._create_param_section('AERODYNAMICS', 'aero', [
            ('rho', 'Air Density', 'kg/m³'),
            ('Cd', 'Drag Coeff (Cd)', ''),
            ('Cl', 'Lift Coeff (Cl)', ''),
            ('A', 'Frontal Area', 'm²'),
        ])

        # Tyre parameters
        self._create_param_section('TYRE', 'tyre', [
            ('mu', 'Friction (μ)', ''),
        ])

        # Powertrain parameters
        self._create_param_section('POWERTRAIN', 'powertrain', [
            ('P_max_kW', 'Max Power', 'kW'),
            ('Fx_max_N', 'Max Force', 'N'),
        ])

        # Battery parameters with enable checkbox
        self._create_battery_section()

    def _create_vehicle_section(self):
        """Create vehicle parameters section with mass breakdown for custom mode."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="VEHICLE", padding=(10, 5))
        frame.pack(fill='x', padx=10, pady=5)
        self.vehicle_frame = frame

        self.param_entries['vehicle'] = {}
        self.param_widgets['vehicle'] = {}
        defaults = self.default_config.get('vehicle', {})

        # Standard mass field (shown in standard mode)
        self.mass_standard_widgets = []
        row = 0
        
        lbl = ttk.Label(frame, text='Total Mass', width=16, anchor='w')
        lbl.grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.mass_standard_widgets.append(lbl)

        var = tk.StringVar(value=str(defaults.get('mass_kg', 250)))
        entry = ttk.Entry(frame, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky='w', padx=5, pady=2)
        self.mass_standard_widgets.append(entry)

        unit_lbl = ttk.Label(frame, text='kg', foreground='gray', width=6)
        unit_lbl.grid(row=row, column=2, sticky='w', pady=2)
        self.mass_standard_widgets.append(unit_lbl)

        self.param_entries['vehicle']['mass_kg'] = var
        self.param_widgets['vehicle']['mass_kg'] = entry
        self.mass_standard_entry = entry
        self.mass_standard_var = var

        # Mass breakdown fields (shown in custom mode)
        self.mass_breakdown_widgets = []
        self.mass_breakdown_vars = {}
        self.mass_breakdown_entries = {}
        
        mass_components = [
            ('mass_chassis_aero_kg', 'Chassis & Aero', 75),
            ('mass_suspension_tyres_kg', 'Suspension & Tyres', 50),
            ('mass_powertrain_kg', 'Powertrain', 45),
            ('mass_battery_kg', 'Battery Systems', 55),
            ('mass_electronics_kg', 'Electronics & Other', 25),
        ]

        for i, (key, label, default) in enumerate(mass_components):
            comp_row = row + 1 + i
            default_val = defaults.get(key, default)

            lbl = ttk.Label(frame, text=f'  {label}', width=16, anchor='w')
            lbl.grid(row=comp_row, column=0, sticky='w', padx=5, pady=1)
            self.mass_breakdown_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=comp_row, column=1, sticky='w', padx=5, pady=1)
            entry.bind('<KeyRelease>', self._update_total_mass)
            self.mass_breakdown_widgets.append(entry)

            unit_lbl = ttk.Label(frame, text='kg', foreground='gray', width=6)
            unit_lbl.grid(row=comp_row, column=2, sticky='w', pady=1)
            self.mass_breakdown_widgets.append(unit_lbl)

            self.param_entries['vehicle'][key] = var
            self.param_widgets['vehicle'][key] = entry
            self.mass_breakdown_vars[key] = var
            self.mass_breakdown_entries[key] = entry

        # Total mass display (read-only, shown in custom mode)
        total_row = row + len(mass_components) + 1
        
        sep = ttk.Separator(frame, orient='horizontal')
        sep.grid(row=total_row, column=0, columnspan=3, sticky='ew', padx=5, pady=3)
        self.mass_breakdown_widgets.append(sep)
        
        total_row += 1
        total_lbl = ttk.Label(frame, text='Total Mass', width=16, anchor='w', font=('TkDefaultFont', 9, 'bold'))
        total_lbl.grid(row=total_row, column=0, sticky='w', padx=5, pady=2)
        self.mass_breakdown_widgets.append(total_lbl)

        self.total_mass_var = tk.StringVar(value='250')
        total_entry = ttk.Entry(frame, textvariable=self.total_mass_var, width=10, state='readonly')
        total_entry.grid(row=total_row, column=1, sticky='w', padx=5, pady=2)
        self.mass_breakdown_widgets.append(total_entry)

        total_unit = ttk.Label(frame, text='kg', foreground='gray', width=6)
        total_unit.grid(row=total_row, column=2, sticky='w', pady=2)
        self.mass_breakdown_widgets.append(total_unit)

        # Other vehicle parameters (g, Crr)
        other_params = [
            ('g', 'Gravity', 'm/s²'),
            ('Crr', 'Rolling Resistance', ''),
        ]
        
        self.vehicle_other_widgets = []
        other_start_row = total_row + 1
        
        for i, (key, label, unit) in enumerate(other_params):
            param_row = other_start_row + i
            default_val = defaults.get(key, 0)

            lbl = ttk.Label(frame, text=label, width=16, anchor='w')
            lbl.grid(row=param_row, column=0, sticky='w', padx=5, pady=2)
            self.vehicle_other_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=param_row, column=1, sticky='w', padx=5, pady=2)
            self.vehicle_other_widgets.append(entry)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground='gray', width=6)
                unit_lbl.grid(row=param_row, column=2, sticky='w', pady=2)
                self.vehicle_other_widgets.append(unit_lbl)

            self.param_entries['vehicle'][key] = var
            self.param_widgets['vehicle'][key] = entry

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
        self.total_mass_var.set(f'{total:.1f}')
        # Also update the main mass_kg variable
        self.mass_standard_var.set(f'{total:.1f}')

    def _create_param_section(self, title: str, section: str, params: list):
        """Create a parameter section with labeled entries."""
        frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding=(10, 5))
        frame.pack(fill='x', padx=10, pady=5)

        self.param_entries[section] = {}
        self.param_widgets[section] = {}  # Store entry widgets
        defaults = self.default_config.get(section, {})

        for i, (key, label, unit) in enumerate(params):
            default_val = defaults.get(key, 0)

            lbl = ttk.Label(frame, text=label, width=16, anchor='w')
            lbl.grid(row=i, column=0, sticky='w', padx=5, pady=2)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky='w', padx=5, pady=2)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground='gray', width=6)
                unit_lbl.grid(row=i, column=2, sticky='w', pady=2)

            self.param_entries[section][key] = var
            self.param_widgets[section][key] = entry  # Store reference to entry widget

    def _create_battery_section(self):
        """Create battery parameters section with enable checkbox and regen options."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="BATTERY", padding=(10, 5))
        frame.pack(fill='x', padx=10, pady=5)

        self.battery_enabled = tk.BooleanVar(value=True)
        self.battery_checkbox = ttk.Checkbutton(
            frame, text="Enable Battery Analysis",
            variable=self.battery_enabled,
            command=self._update_battery_state
        )
        self.battery_checkbox.grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=5)

        self.param_entries['battery'] = {}
        self.battery_widgets = []  # Store entry widgets for battery section
        defaults = self.default_config.get('battery', {})

        params = [
            ('capacity_kWh', 'Capacity', 'kWh'),
            ('initial_soc', 'Initial SoC', ''),
            ('min_soc', 'Min SoC', ''),
            ('max_discharge_kW', 'Max Discharge', 'kW'),
            ('eta_discharge', 'Efficiency', ''),
            ('nominal_voltage_V', 'Pack Voltage', 'V'),
            ('max_current_A', 'Max Current', 'A'),
        ]

        for i, (key, label, unit) in enumerate(params):
            default_val = defaults.get(key, 0)

            lbl = ttk.Label(frame, text=label, width=16, anchor='w')
            lbl.grid(row=i + 1, column=0, sticky='w', padx=5, pady=2)
            self.battery_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=i + 1, column=1, sticky='w', padx=5, pady=2)
            self.battery_widgets.append(entry)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground='gray', width=6)
                unit_lbl.grid(row=i + 1, column=2, sticky='w', pady=2)
                self.battery_widgets.append(unit_lbl)

            self.param_entries['battery'][key] = var

        # Regenerative braking section
        regen_row = len(params) + 1

        # Separator
        sep = ttk.Separator(frame, orient='horizontal')
        sep.grid(row=regen_row, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        self.battery_widgets.append(sep)

        # Regen enable checkbox
        regen_row += 1
        self.regen_enabled = tk.BooleanVar(value=defaults.get('regen_enabled', False))
        self.regen_checkbox = ttk.Checkbutton(
            frame, text="Enable Regenerative Braking",
            variable=self.regen_enabled,
            command=self._update_regen_state
        )
        self.regen_checkbox.grid(row=regen_row, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        self.battery_widgets.append(self.regen_checkbox)

        # Regen parameters
        self.regen_widgets = []  # Store regen entry widgets for enabling/disabling
        regen_params = [
            ('eta_regen', 'Regen Efficiency', ''),
            ('max_regen_kW', 'Max Regen Power', 'kW'),
            ('regen_capture_percent', 'Braking Capture', '%'),
        ]

        for i, (key, label, unit) in enumerate(regen_params):
            default_val = defaults.get(key, 0)
            param_row = regen_row + 1 + i

            lbl = ttk.Label(frame, text=f'  {label}', width=16, anchor='w')
            lbl.grid(row=param_row, column=0, sticky='w', padx=5, pady=2)
            self.regen_widgets.append(lbl)

            var = tk.StringVar(value=str(default_val))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=param_row, column=1, sticky='w', padx=5, pady=2)
            self.regen_widgets.append(entry)

            if unit:
                unit_lbl = ttk.Label(frame, text=unit, foreground='gray', width=6)
                unit_lbl.grid(row=param_row, column=2, sticky='w', pady=2)
                self.regen_widgets.append(unit_lbl)

            self.param_entries['battery'][key] = var

        # Initially update regen state
        self._update_regen_state()

    def _update_battery_state(self):
        """Enable/disable battery parameter entries."""
        is_custom = self.config_var.get() == 'custom'
        # Only enable if both custom mode AND battery is enabled
        state = 'normal' if (is_custom and self.battery_enabled.get()) else 'disabled'
        for widget in self.battery_widgets:
            if isinstance(widget, ttk.Entry):
                widget.configure(state=state)
            elif isinstance(widget, ttk.Checkbutton) and widget != self.battery_checkbox:
                # Enable regen checkbox only if battery is enabled
                widget.configure(state=state)
        # Also update regen state
        self._update_regen_state()

    def _update_regen_state(self):
        """Enable/disable regen parameter entries based on regen checkbox."""
        is_custom = self.config_var.get() == 'custom'
        battery_enabled = self.battery_enabled.get()
        regen_enabled = self.regen_enabled.get()
        # Only enable regen params if: custom mode AND battery enabled AND regen enabled
        state = 'normal' if (is_custom and battery_enabled and regen_enabled) else 'disabled'
        for widget in self.regen_widgets:
            if isinstance(widget, ttk.Entry):
                widget.configure(state=state)

    def _create_action_buttons(self):
        """Create load, save, and run buttons."""
        # Config buttons frame
        btn_frame = ttk.Frame(self.scrollable_frame)
        btn_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(btn_frame, text="Load Config", command=self._load_config).pack(
            side='left', padx=5
        )
        ttk.Button(btn_frame, text="Save Config", command=self._save_config).pack(
            side='left', padx=5
        )
        ttk.Button(btn_frame, text="Reset", command=self._reset_to_defaults).pack(
            side='left', padx=5
        )

        # Run button (prominent)
        run_frame = ttk.Frame(self.scrollable_frame)
        run_frame.pack(fill='x', padx=10, pady=(5, 10))

        self.run_button = ttk.Button(
            run_frame, text="RUN SIMULATION",
            command=self._on_run_clicked,
            style='Accent.TButton'
        )
        self.run_button.pack(fill='x', ipady=10)

        # Save results button
        save_frame = ttk.Frame(self.scrollable_frame)
        save_frame.pack(fill='x', padx=10, pady=(0, 15))

        self.save_button = ttk.Button(
            save_frame, text="SAVE RESULTS",
            command=self._on_save_clicked,
            style='Accent.TButton'
        )
        self.save_button.pack(fill='x', ipady=10)

    def _on_save_clicked(self):
        """Handle save results button click."""
        if self.on_save_results:
            self.on_save_results()

    def _update_parameter_state(self):
        """Enable/disable parameter entries based on config selection."""
        is_custom = self.config_var.get() == 'custom'
        state = 'normal' if is_custom else 'disabled'

        # Toggle mass breakdown view
        self._toggle_mass_breakdown(show_breakdown=is_custom)

        # Update all parameter entry widgets
        for section, widgets in self.param_widgets.items():
            for key, entry in widgets.items():
                # Skip mass breakdown entries in standard mode (they're hidden)
                if not is_custom and key.startswith('mass_') and key != 'mass_kg':
                    continue
                entry.configure(state=state)

        # Update battery widgets (entries only)
        for widget in self.battery_widgets:
            if isinstance(widget, ttk.Entry):
                # Only enable if custom mode AND battery is enabled
                battery_state = 'normal' if (is_custom and self.battery_enabled.get()) else 'disabled'
                widget.configure(state=battery_state)

        # Update battery checkbox
        if hasattr(self, 'battery_checkbox'):
            self.battery_checkbox.configure(state=state)

        # Update regen checkbox and entries
        if hasattr(self, 'regen_checkbox'):
            # Regen checkbox enabled only if battery is enabled
            regen_cb_state = 'normal' if (is_custom and self.battery_enabled.get()) else 'disabled'
            self.regen_checkbox.configure(state=regen_cb_state)
            self._update_regen_state()

        # Reset to defaults when switching to standard
        if not is_custom:
            self._reset_to_defaults()

    def _load_config(self):
        """Load configuration from file."""
        filepath = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config'
            )
        )

        if filepath:
            try:
                with open(filepath, 'r') as f:
                    config = yaml.safe_load(f)
                self._set_config(config)
                self.config_var.set('custom')
                messagebox.showinfo("Success", f"Configuration loaded from:\n{os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{e}")

    def _save_config(self):
        """Save current configuration to file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config'
            )
        )

        if filepath:
            try:
                config = self.get_config()
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                messagebox.showinfo("Success", f"Configuration saved to:\n{os.path.basename(filepath)}")
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
        if 'vehicle' in config:
            total_mass = config['vehicle'].get('mass_kg', 250)
            if 'mass_chassis_aero_kg' not in config['vehicle']:
                # Apply default ratios
                self.mass_breakdown_vars['mass_chassis_aero_kg'].set(str(round(total_mass * 0.30)))
                self.mass_breakdown_vars['mass_suspension_tyres_kg'].set(str(round(total_mass * 0.20)))
                self.mass_breakdown_vars['mass_powertrain_kg'].set(str(round(total_mass * 0.18)))
                self.mass_breakdown_vars['mass_battery_kg'].set(str(round(total_mass * 0.22)))
                self.mass_breakdown_vars['mass_electronics_kg'].set(str(round(total_mass * 0.10)))
            self._update_total_mass()

        # Handle battery enabled state
        if 'battery' in config:
            self.battery_enabled.set(config['battery'].get('enabled', True))
            # Handle regen enabled state
            if hasattr(self, 'regen_enabled'):
                self.regen_enabled.set(config['battery'].get('regen_enabled', False))
            self._update_battery_state()

    def _on_run_clicked(self):
        """Handle run button click."""
        if self.on_run:
            self.on_run()

    def get_config(self) -> dict:
        """Get current configuration as a dictionary."""
        config = {}
        is_custom = self.config_var.get() == 'custom'

        for section, params in self.param_entries.items():
            config[section] = {}
            for key, var in params.items():
                try:
                    val = var.get()
                    # Convert to appropriate type
                    if '.' in val:
                        config[section][key] = float(val)
                    else:
                        config[section][key] = int(val) if val.isdigit() else float(val)
                except ValueError:
                    config[section][key] = var.get()

        # In custom mode, calculate total mass from components
        if is_custom:
            total_mass = 0
            for key in self.mass_breakdown_vars:
                total_mass += config['vehicle'].get(key, 0)
            config['vehicle']['mass_kg'] = total_mass

        # Add battery enabled flag
        config['battery']['enabled'] = self.battery_enabled.get()

        # Add regen enabled flag
        if hasattr(self, 'regen_enabled'):
            config['battery']['regen_enabled'] = self.regen_enabled.get()

        return config

    def get_event(self) -> str:
        """Get selected event type."""
        return self.event_var.get()

    def set_running(self, running: bool):
        """Set the running state of the panel."""
        state = 'disabled' if running else 'normal'
        self.run_button.configure(state=state)
        self.save_button.configure(state=state)
        if running:
            self.run_button.configure(text="RUNNING...")
        else:
            self.run_button.configure(text="RUN SIMULATION")
