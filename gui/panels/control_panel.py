"""Left control panel with event selection and parameter inputs."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
import os
from typing import Callable, Dict, Any, Optional


class ControlPanel(ttk.Frame):
    """Left panel containing event selection, parameters, and controls."""

    def __init__(self, parent, on_run: Callable, on_config_change: Callable = None, **kwargs):
        """
        Create the control panel.

        Args:
            parent: Parent widget
            on_run: Callback function when Run button is clicked
            on_config_change: Callback when configuration changes
        """
        super().__init__(parent, **kwargs)

        self.on_run = on_run
        self.on_config_change = on_config_change
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
            'vehicle': {'mass_kg': 250, 'g': 9.81, 'Crr': 0.015},
            'aero': {'rho': 1.225, 'Cd': 1.1, 'Cl': 1.5, 'A': 1.0},
            'tyre': {'mu': 1.6},
            'powertrain': {'P_max_kW': 80, 'Fx_max_N': 3900},
            'battery': {
                'capacity_kWh': 6, 'initial_soc': 1.0, 'min_soc': 0.1,
                'max_discharge_kW': 80, 'eta_discharge': 0.95
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

        # Vehicle parameters
        self._create_param_section('VEHICLE', 'vehicle', [
            ('mass_kg', 'Mass', 'kg'),
            ('g', 'Gravity', 'm/s²'),
            ('Crr', 'Rolling Resistance', ''),
        ])

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

    def _create_param_section(self, title: str, section: str, params: list):
        """Create a parameter section with labeled entries."""
        frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding=(10, 5))
        frame.pack(fill='x', padx=10, pady=5)

        self.param_entries[section] = {}
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

    def _create_battery_section(self):
        """Create battery parameters section with enable checkbox."""
        frame = ttk.LabelFrame(self.scrollable_frame, text="BATTERY", padding=(10, 5))
        frame.pack(fill='x', padx=10, pady=5)

        self.battery_enabled = tk.BooleanVar(value=True)
        cb = ttk.Checkbutton(
            frame, text="Enable Battery Analysis",
            variable=self.battery_enabled,
            command=self._update_battery_state
        )
        cb.grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=5)

        self.param_entries['battery'] = {}
        self.battery_widgets = []
        defaults = self.default_config.get('battery', {})

        params = [
            ('capacity_kWh', 'Capacity', 'kWh'),
            ('initial_soc', 'Initial SoC', ''),
            ('min_soc', 'Min SoC', ''),
            ('max_discharge_kW', 'Max Discharge', 'kW'),
            ('eta_discharge', 'Efficiency', ''),
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

    def _update_battery_state(self):
        """Enable/disable battery parameter entries."""
        state = 'normal' if self.battery_enabled.get() else 'disabled'
        for widget in self.battery_widgets:
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
        run_frame.pack(fill='x', padx=10, pady=(5, 15))

        self.run_button = ttk.Button(
            run_frame, text="RUN SIMULATION",
            command=self._on_run_clicked,
            style='Accent.TButton'
        )
        self.run_button.pack(fill='x', ipady=10)

    def _update_parameter_state(self):
        """Enable/disable parameter entries based on config selection."""
        state = 'normal' if self.config_var.get() == 'custom' else 'disabled'

        for section in self.param_entries.values():
            for var_name, var in section.items():
                # Find the entry widget
                pass  # Entries are always editable, just reset to defaults when standard

        if self.config_var.get() == 'standard':
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

        # Handle battery enabled state
        if 'battery' in config:
            self.battery_enabled.set(config['battery'].get('enabled', True))
            self._update_battery_state()

    def _on_run_clicked(self):
        """Handle run button click."""
        if self.on_run:
            self.on_run()

    def get_config(self) -> dict:
        """Get current configuration as a dictionary."""
        config = {}

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

        # Add battery enabled flag
        config['battery']['enabled'] = self.battery_enabled.get()

        return config

    def get_event(self) -> str:
        """Get selected event type."""
        return self.event_var.get()

    def set_running(self, running: bool):
        """Set the running state of the panel."""
        state = 'disabled' if running else 'normal'
        self.run_button.configure(state=state)
        if running:
            self.run_button.configure(text="RUNNING...")
        else:
            self.run_button.configure(text="RUN SIMULATION")
