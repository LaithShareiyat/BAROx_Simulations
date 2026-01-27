"""Battery Pack Optimiser Panel for finding optimal cell configuration."""

import tkinter as tk
from tkinter import ttk
import threading
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass
class CellSpec:
    """Lithium-ion cell specifications."""

    capacity_mAh: float = 2600  # Cell capacity [mAh]
    voltage_nominal: float = 3.6  # Nominal voltage [V]
    voltage_max: float = 4.2  # Max voltage at full charge [V]
    max_current_A: float = 35  # Max discharge current [A]
    internal_resistance_mohm: float = 12  # Internal resistance [mΩ]
    weight_g: float = 45  # Cell weight [g]


@dataclass
class PackConfig:
    """Battery pack configuration."""

    series: int  # Number of cells in series (m)
    parallel: int  # Number of parallel strings (n)
    cell: CellSpec
    casing_factor: float = 1.4  # Mass multiplier for casing/BMS/wiring

    @property
    def total_cells(self) -> int:
        return self.series * self.parallel

    @property
    def voltage_nominal(self) -> float:
        """Pack nominal voltage [V]."""
        return self.series * self.cell.voltage_nominal

    @property
    def voltage_max(self) -> float:
        """Pack max voltage [V] at full charge."""
        return self.series * self.cell.voltage_max

    @property
    def max_current(self) -> float:
        """Pack max continuous current [A]."""
        return self.parallel * self.cell.max_current_A

    @property
    def internal_resistance(self) -> float:
        """Pack internal resistance [mΩ]."""
        if self.parallel < 1:
            return float("inf")  # Invalid config
        return (self.series * self.cell.internal_resistance_mohm) / self.parallel

    @property
    def internal_resistance_ohm(self) -> float:
        """Pack internal resistance [Ω]."""
        if self.parallel < 1:
            return float("inf")  # Invalid config
        return self.internal_resistance / 1000.0

    @property
    def cell_mass_kg(self) -> float:
        """Total cell mass [kg]."""
        return (self.series * self.parallel * self.cell.weight_g) / 1000

    @property
    def pack_mass_kg(self) -> float:
        """Total pack mass including casing [kg]."""
        return self.cell_mass_kg * self.casing_factor

    @property
    def capacity_kWh(self) -> float:
        """Pack capacity [kWh]."""
        capacity_Ah = (self.cell.capacity_mAh / 1000) * self.parallel
        return (capacity_Ah * self.voltage_nominal) / 1000

    @property
    def max_power_kW(self) -> float:
        """Max discharge power [kW]."""
        return (self.voltage_nominal * self.max_current) / 1000

    @property
    def config_name(self) -> str:
        """Configuration name (e.g., '142S4P')."""
        return f"{self.series}S{self.parallel}P"

    def calculate_current_at_power(self, power_kW: float) -> float:
        """Calculate actual current needed to deliver power, accounting for voltage loss.

        Solves the coupled equation: P = V_actual × I, where V_actual = V_nom - I×R

        Using quadratic formula:
            P = (V_nom - I×R) × I
            I×R×I - V_nom×I + P = 0
            R×I² - V_nom×I + P = 0

            I = (V_nom - sqrt(V_nom² - 4×R×P)) / (2×R)

        We use the minus solution because it gives the lower (physical) current.
        """
        V = self.voltage_nominal
        R = self.internal_resistance_ohm
        P = power_kW * 1000  # Convert to W

        # Handle edge cases
        if V <= 0 or self.parallel < 1:
            return 0.0

        if R == 0 or R < 1e-9:
            # No resistance - simple I = P/V
            return P / V if V > 0 else 0.0

        discriminant = V * V - 4 * R * P

        if discriminant < 0:
            # Power demand exceeds what pack can physically deliver
            # Return the current at max power point (V/2R)
            return V / (2 * R)

        # Use minus solution for lower (physical) current
        I = (V - np.sqrt(discriminant)) / (2 * R)
        return max(0.0, I)  # Ensure non-negative


@dataclass
class OptimisationResult:
    """Result of a single configuration simulation."""

    config: PackConfig
    lap_time: float
    energy_used_kWh: float
    final_soc: float
    sufficient: bool
    total_vehicle_mass: float
    peak_power_kW: float  # Peak DEMANDED power (before any limiting)
    power_limited: bool  # True if peak demand > pack max power
    peak_current_A: float  # Peak current (based on effective power)
    # Voltage and power loss fields
    voltage_loss_V: float = 0.0  # Voltage drop under peak current [V]
    voltage_under_load_V: float = 0.0  # Actual voltage at peak current [V]
    i2r_loss_kW: float = 0.0  # Peak I²R power loss [kW]
    # Constraint violation flags
    exceeds_fs_current: bool = False  # True if peak current > 500A (FS rules)
    exceeds_inverter_limit: bool = False  # True if peak current > inverter limit
    exceeds_max_voltage: bool = False  # True if V_max > max voltage constraint
    below_min_voltage: bool = False  # True if V_nom < min voltage constraint
    exceeds_max_mass: bool = False  # True if pack mass > max mass constraint
    # Imposed power limit
    imposed_limit_kW: float = None  # The user-imposed power limit (None if no limit)
    effective_power_kW: float = (
        None  # Effective power after user-imposed limit (None if no limit)
    )
    power_limit_active: bool = False  # True if user-imposed limit is capping the power


class BatteryOptimiserPanel(ttk.Frame):
    """Panel for battery pack optimisation."""

    def __init__(self, parent, get_base_config: Callable, **kwargs):
        """
        Initialise the battery optimiser panel.

        Args:
            parent: Parent widget
            get_base_config: Callback to get base vehicle configuration
        """
        super().__init__(parent, **kwargs)
        self.get_base_config = get_base_config
        self.running = False
        self.results: List[OptimisationResult] = []
        self.best_result: Optional[OptimisationResult] = None

        self._create_widgets()

    def _create_widgets(self):
        """Create all widgets for the optimiser panel."""
        # Main container with two columns
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left column - Parameters
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="y", padx=(0, 10))

        # Right column - Results
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side="left", fill="both", expand=True)

        # === LEFT COLUMN: Parameters ===
        self._create_cell_params(left_frame)
        self._create_sweep_params(left_frame)
        self._create_constraints(left_frame)
        self._create_run_button(left_frame)

        # === RIGHT COLUMN: Results ===
        self._create_results_display(right_frame)

        # Initial config count update (after all widgets created)
        self._update_config_count()

    def _create_cell_params(self, parent):
        """Create cell specification inputs."""
        frame = ttk.LabelFrame(parent, text="CELL SPECIFICATIONS", padding=(10, 5))
        frame.pack(fill="x", pady=5)

        self.cell_vars = {}
        params = [
            ("capacity_mAh", "Capacity", "mAh", 2600),
            ("voltage_nominal", "Voltage (nom)", "V", 3.6),
            ("voltage_max", "Voltage (max)", "V", 4.2),
            ("max_current_A", "Max Current", "A", 35),
            ("resistance_mohm", "Int. Resistance", "mΩ", 12),
            ("weight_g", "Weight", "g", 45),
        ]

        for i, (key, label, unit, default) in enumerate(params):
            lbl = ttk.Label(frame, text=label, width=14, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", padx=5, pady=2)

            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)

            unit_lbl = ttk.Label(frame, text=unit, foreground="gray", width=6)
            unit_lbl.grid(row=i, column=2, sticky="w", pady=2)

            self.cell_vars[key] = var

    def _create_sweep_params(self, parent):
        """Create sweep range inputs."""
        frame = ttk.LabelFrame(parent, text="SWEEP RANGE", padding=(10, 5))
        frame.pack(fill="x", pady=5)

        self.sweep_vars = {}

        # Series cells range
        row = 0
        ttk.Label(frame, text="Series (m)", width=14, anchor="w").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )

        series_frame = ttk.Frame(frame)
        series_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=2)

        self.sweep_vars["series_min"] = tk.StringVar(value="0")
        self.sweep_vars["series_max"] = tk.StringVar(value="150")
        self.sweep_vars["series_step"] = tk.StringVar(value="1")

        ttk.Entry(
            series_frame, textvariable=self.sweep_vars["series_min"], width=5
        ).pack(side="left")
        ttk.Label(series_frame, text=" to ").pack(side="left")
        ttk.Entry(
            series_frame, textvariable=self.sweep_vars["series_max"], width=5
        ).pack(side="left")
        ttk.Label(series_frame, text=" step ").pack(side="left")
        ttk.Entry(
            series_frame, textvariable=self.sweep_vars["series_step"], width=4
        ).pack(side="left")

        # Parallel strings range
        row = 1
        ttk.Label(frame, text="Parallel (n)", width=14, anchor="w").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )

        parallel_frame = ttk.Frame(frame)
        parallel_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=2)

        self.sweep_vars["parallel_min"] = tk.StringVar(value="1")
        self.sweep_vars["parallel_max"] = tk.StringVar(value="15")
        self.sweep_vars["parallel_step"] = tk.StringVar(value="1")

        ttk.Entry(
            parallel_frame, textvariable=self.sweep_vars["parallel_min"], width=5
        ).pack(side="left")
        ttk.Label(parallel_frame, text=" to ").pack(side="left")
        ttk.Entry(
            parallel_frame, textvariable=self.sweep_vars["parallel_max"], width=5
        ).pack(side="left")
        ttk.Label(parallel_frame, text=" step ").pack(side="left")
        ttk.Entry(
            parallel_frame, textvariable=self.sweep_vars["parallel_step"], width=4
        ).pack(side="left")

        # Estimated configurations
        row = 2
        ttk.Label(frame, text="Configurations", width=14, anchor="w").grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        self.config_count_var = tk.StringVar(value="0")
        ttk.Label(frame, textvariable=self.config_count_var, width=16, anchor="w").grid(
            row=row, column=1, columnspan=2, sticky="w", padx=5, pady=2
        )

        # Bind updates (actual update called after all widgets created)
        for var in self.sweep_vars.values():
            var.trace_add("write", self._update_config_count)

    def _create_constraints(self, parent):
        """Create constraint inputs."""
        frame = ttk.LabelFrame(parent, text="CONSTRAINTS", padding=(10, 5))
        frame.pack(fill="x", pady=5)

        self.constraint_vars = {}
        constraints = [
            ("max_voltage", "Max Voltage", "V", 600),
            ("min_voltage", "Min Voltage", "V", 300),
            ("fs_max_current", "FS Rules I max", "A", 500),
            ("inverter_current", "Inverter I max", "A", 500),
            ("max_pack_mass", "Max Pack Mass", "kg", 80),
            ("min_voltage_load", "Min V under load", "V", 250),
            ("casing_factor", "Casing Factor", "×", 1.4),
        ]

        for i, (key, label, unit, default) in enumerate(constraints):
            lbl = ttk.Label(frame, text=label, width=14, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", padx=5, pady=2)

            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)

            unit_lbl = ttk.Label(frame, text=unit, foreground="gray", width=6)
            unit_lbl.grid(row=i, column=2, sticky="w", pady=2)

            self.constraint_vars[key] = var

        # Checkbox to enable/disable mass constraint
        row = len(constraints)
        self.enforce_mass_constraint = tk.BooleanVar(value=True)
        mass_check = ttk.Checkbutton(
            frame, text="Enforce mass limit", variable=self.enforce_mass_constraint
        )
        mass_check.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(8, 2)
        )

        # Power limit section
        row += 1
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(10, 5)
        )

        row += 1
        self.enforce_power_limit = tk.BooleanVar(value=False)
        power_check = ttk.Checkbutton(
            frame,
            text="Impose power draw limit",
            variable=self.enforce_power_limit,
            command=self._on_power_limit_toggle,
        )
        power_check.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=(5, 2)
        )

        # Power limit sweep checkbox
        row += 1
        self.power_limit_sweep_enabled = tk.BooleanVar(value=False)
        self.power_sweep_check = ttk.Checkbutton(
            frame,
            text="Sweep power limits",
            variable=self.power_limit_sweep_enabled,
            command=self._on_power_limit_toggle,
        )
        self.power_sweep_check.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=20, pady=(2, 2)
        )

        # Single power limit value (shown when sweep is disabled)
        row += 1
        self.power_limit_single_frame = ttk.Frame(frame)
        self.power_limit_single_frame.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2
        )

        self.power_limit_label = ttk.Label(
            self.power_limit_single_frame, text="Max Power Draw", width=14, anchor="w"
        )
        self.power_limit_label.pack(side="left")

        self.constraint_vars["power_limit"] = tk.StringVar(value="50")
        self.power_limit_entry = ttk.Entry(
            self.power_limit_single_frame,
            textvariable=self.constraint_vars["power_limit"],
            width=10,
        )
        self.power_limit_entry.pack(side="left", padx=5)

        self.power_limit_unit = ttk.Label(
            self.power_limit_single_frame, text="kW", foreground="gray", width=6
        )
        self.power_limit_unit.pack(side="left")

        # Power limit sweep range (shown when sweep is enabled)
        row += 1
        self.power_limit_sweep_frame = ttk.Frame(frame)
        self.power_limit_sweep_frame.grid(
            row=row, column=0, columnspan=3, sticky="w", padx=5, pady=2
        )

        ttk.Label(
            self.power_limit_sweep_frame, text="Power Range", width=14, anchor="w"
        ).pack(side="left")

        self.power_sweep_vars = {}
        self.power_sweep_vars["min"] = tk.StringVar(value="30")
        self.power_sweep_vars["max"] = tk.StringVar(value="80")
        self.power_sweep_vars["step"] = tk.StringVar(value="10")

        ttk.Entry(
            self.power_limit_sweep_frame,
            textvariable=self.power_sweep_vars["min"],
            width=5,
        ).pack(side="left")
        ttk.Label(self.power_limit_sweep_frame, text=" to ").pack(side="left")
        ttk.Entry(
            self.power_limit_sweep_frame,
            textvariable=self.power_sweep_vars["max"],
            width=5,
        ).pack(side="left")
        ttk.Label(self.power_limit_sweep_frame, text=" step ").pack(side="left")
        ttk.Entry(
            self.power_limit_sweep_frame,
            textvariable=self.power_sweep_vars["step"],
            width=4,
        ).pack(side="left")
        ttk.Label(self.power_limit_sweep_frame, text=" kW", foreground="gray").pack(
            side="left"
        )

        # Bind updates for config count
        for var in self.power_sweep_vars.values():
            var.trace_add("write", self._update_config_count)

        # Initially disable power limit controls (checkbox is off by default)
        self._on_power_limit_toggle()

    def _create_run_button(self, parent):
        """Create run optimisation button."""
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=10)

        self.run_button = ttk.Button(
            btn_frame,
            text="RUN OPTIMISATION",
            command=self._run_optimisation,
            style="Accent.TButton",
        )
        self.run_button.pack(fill="x", ipady=8)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            btn_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill="x", pady=(5, 0))

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            btn_frame, textvariable=self.status_var, foreground="gray"
        )
        self.status_label.pack(pady=(5, 0))

    def _create_results_display(self, parent):
        """Create results display area."""
        # Best result summary
        summary_frame = ttk.LabelFrame(
            parent, text="OPTIMAL CONFIGURATION", padding=(10, 5)
        )
        summary_frame.pack(fill="x", pady=5)

        self.summary_text = tk.Text(
            summary_frame,
            height=16,  # Increased height for new metrics
            wrap="word",
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            state="disabled",
        )
        self.summary_text.pack(fill="x", padx=5, pady=5)

        # All results table
        table_frame = ttk.LabelFrame(parent, text="ALL CONFIGURATIONS", padding=(10, 5))
        table_frame.pack(fill="both", expand=True, pady=5)

        # Treeview for results - updated columns
        columns = (
            "config",
            "cells",
            "P_lim",
            "V_nom",
            "V_load",
            "I_peak",
            "I2R",
            "kWh",
            "mass",
            "lap",
            "status",
        )
        self.results_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=15
        )

        # Column headings
        self.results_tree.heading("config", text="Config")
        self.results_tree.heading("cells", text="Cells")
        self.results_tree.heading("P_lim", text="P lim")
        self.results_tree.heading("V_nom", text="V nom")
        self.results_tree.heading("V_load", text="V load")
        self.results_tree.heading("I_peak", text="I peak")
        self.results_tree.heading("I2R", text="I²R kW")
        self.results_tree.heading("kWh", text="kWh")
        self.results_tree.heading("mass", text="kg")
        self.results_tree.heading("lap", text="Lap (s)")
        self.results_tree.heading("status", text="Status")

        # Column widths
        self.results_tree.column("config", width=65, anchor="center")
        self.results_tree.column("cells", width=45, anchor="center")
        self.results_tree.column("P_lim", width=45, anchor="center")
        self.results_tree.column("V_nom", width=50, anchor="center")
        self.results_tree.column("V_load", width=50, anchor="center")
        self.results_tree.column("I_peak", width=50, anchor="center")
        self.results_tree.column("I2R", width=50, anchor="center")
        self.results_tree.column("kWh", width=45, anchor="center")
        self.results_tree.column("mass", width=45, anchor="center")
        self.results_tree.column("lap", width=55, anchor="center")
        self.results_tree.column("status", width=85, anchor="center")

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.results_tree.yview
        )
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _update_config_count(self, *args):
        """Update the configuration count."""
        # Guard against calls before all widgets are created
        if not hasattr(self, "sweep_vars"):
            return
        try:
            # Get sweep parameters
            s_min = max(1, int(self.sweep_vars["series_min"].get()))
            s_max = int(self.sweep_vars["series_max"].get())
            s_step = max(1, int(self.sweep_vars["series_step"].get()))
            p_min = max(1, int(self.sweep_vars["parallel_min"].get()))
            p_max = int(self.sweep_vars["parallel_max"].get())
            p_step = max(1, int(self.sweep_vars["parallel_step"].get()))

            # Get power sweep count (if enabled)
            power_count = 1
            if (
                hasattr(self, "enforce_power_limit")
                and hasattr(self, "power_limit_sweep_enabled")
                and self.enforce_power_limit.get()
                and self.power_limit_sweep_enabled.get()
            ):
                pwr_min = int(self.power_sweep_vars["min"].get())
                pwr_max = int(self.power_sweep_vars["max"].get())
                pwr_step = max(1, int(self.power_sweep_vars["step"].get()))
                power_count = len(range(pwr_min, pwr_max + 1, pwr_step))

            # Count all configurations
            total_count = 0
            for series in range(s_min, s_max + 1, s_step):
                for parallel in range(p_min, p_max + 1, p_step):
                    total_count += 1

            # Multiply by power sweep count
            total_count *= power_count

            self.config_count_var.set(f"{total_count}")
        except (ValueError, KeyError):
            self.config_count_var.set("—")

    def _on_power_limit_toggle(self):
        """Enable/disable power limit controls based on checkbox states."""
        power_enabled = self.enforce_power_limit.get()
        sweep_enabled = self.power_limit_sweep_enabled.get()

        if power_enabled:
            # Enable the sweep checkbox
            self.power_sweep_check.configure(state="normal")

            if sweep_enabled:
                # Show sweep controls, hide single value
                self.power_limit_single_frame.grid_remove()
                self.power_limit_sweep_frame.grid()
                # Enable sweep entries
                for child in self.power_limit_sweep_frame.winfo_children():
                    if isinstance(child, ttk.Entry):
                        child.configure(state="normal")
            else:
                # Show single value, hide sweep controls
                self.power_limit_sweep_frame.grid_remove()
                self.power_limit_single_frame.grid()
                self.power_limit_entry.configure(state="normal")
                self.power_limit_label.configure(foreground="")
                self.power_limit_unit.configure(foreground="")
        else:
            # Disable everything
            self.power_sweep_check.configure(state="disabled")
            self.power_limit_sweep_enabled.set(False)

            # Show single value (disabled), hide sweep
            self.power_limit_sweep_frame.grid_remove()
            self.power_limit_single_frame.grid()
            self.power_limit_entry.configure(state="disabled")
            self.power_limit_label.configure(foreground="gray")
            self.power_limit_unit.configure(foreground="gray")

        # Update config count when toggling
        self._update_config_count()

    def _get_cell_spec(self) -> CellSpec:
        """Get cell specifications from inputs."""
        return CellSpec(
            capacity_mAh=float(self.cell_vars["capacity_mAh"].get()),
            voltage_nominal=float(self.cell_vars["voltage_nominal"].get()),
            voltage_max=float(self.cell_vars["voltage_max"].get()),
            max_current_A=float(self.cell_vars["max_current_A"].get()),
            internal_resistance_mohm=float(self.cell_vars["resistance_mohm"].get()),
            weight_g=float(self.cell_vars["weight_g"].get()),
        )

    def _get_configurations(self) -> List[PackConfig]:
        """Generate all pack configurations to test.

        All configurations are tested; constraint violations are flagged during simulation.
        """
        cell = self._get_cell_spec()
        casing_factor = float(self.constraint_vars["casing_factor"].get())

        s_min = int(self.sweep_vars["series_min"].get())
        s_max = int(self.sweep_vars["series_max"].get())
        s_step = max(1, int(self.sweep_vars["series_step"].get()))
        p_min = int(self.sweep_vars["parallel_min"].get())
        p_max = int(self.sweep_vars["parallel_max"].get())
        p_step = max(1, int(self.sweep_vars["parallel_step"].get()))

        # Ensure minimum values are at least 1 (can't have 0 cells)
        s_min = max(1, s_min)
        p_min = max(1, p_min)

        configs = []
        for series in range(s_min, s_max + 1, s_step):
            for parallel in range(p_min, p_max + 1, p_step):
                config = PackConfig(
                    series=series,
                    parallel=parallel,
                    cell=cell,
                    casing_factor=casing_factor,
                )
                configs.append(config)

        return configs

    def _run_optimisation(self):
        """Start the optimisation sweep."""
        if self.running:
            return

        self.running = True
        self.run_button.configure(state="disabled", text="RUNNING...")
        self.progress_var.set(0)
        self.status_var.set("Starting optimisation...")
        self.results = []
        self.best_result = None
        self.best_is_valid = False

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        # Run in background thread
        thread = threading.Thread(target=self._optimisation_thread)
        thread.daemon = True
        thread.start()

    def _optimisation_thread(self):
        """Execute optimisation in background thread."""
        try:
            configs = self._get_configurations()
            total = len(configs)

            if total == 0:
                self.winfo_toplevel().after(
                    0, lambda: self._show_error("No valid configurations found")
                )
                return

            base_config = self.get_base_config()
            casing_factor = float(self.constraint_vars["casing_factor"].get())
            fs_max_current = float(self.constraint_vars["fs_max_current"].get())
            inverter_max_current = float(self.constraint_vars["inverter_current"].get())
            min_voltage_under_load = float(
                self.constraint_vars["min_voltage_load"].get()
            )
            max_voltage = float(self.constraint_vars["max_voltage"].get())
            min_voltage = float(self.constraint_vars["min_voltage"].get())
            max_pack_mass = float(self.constraint_vars["max_pack_mass"].get())
            enforce_mass_constraint = self.enforce_mass_constraint.get()

            # Power limit settings
            enforce_power_limit = self.enforce_power_limit.get()
            power_sweep_enabled = (
                self.power_limit_sweep_enabled.get() if enforce_power_limit else False
            )

            # Build list of power limits to test
            if enforce_power_limit:
                if power_sweep_enabled:
                    pwr_min = int(self.power_sweep_vars["min"].get())
                    pwr_max = int(self.power_sweep_vars["max"].get())
                    pwr_step = max(1, int(self.power_sweep_vars["step"].get()))
                    power_limits = list(range(pwr_min, pwr_max + 1, pwr_step))
                else:
                    power_limits = [float(self.constraint_vars["power_limit"].get())]
            else:
                power_limits = [None]  # No power limit

            # Import simulation modules
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
            from events.autocross_generator import build_standard_autocross
            from solver.qss_speed import solve_qss
            from solver.battery import simulate_battery, validate_battery_capacity

            # Build track once
            track, _ = build_standard_autocross()

            # Calculate total iterations
            total_iterations = len(configs) * len(power_limits)

            results = []
            iteration = 0
            for pack_config in configs:
                for power_limit in power_limits:
                    iteration += 1
                    # Update progress
                    progress = (iteration / total_iterations) * 100
                    if power_limit is not None:
                        status_msg = (
                            f"Testing {pack_config.config_name} @ {power_limit}kW..."
                        )
                    else:
                        status_msg = f"Testing {pack_config.config_name}..."
                    self.winfo_toplevel().after(
                        0, lambda p=progress, s=status_msg: self._update_progress(p, s)
                    )

                    # Create modified vehicle config
                    result = self._simulate_config(
                        pack_config,
                        base_config,
                        track,
                        casing_factor,
                        fs_max_current,
                        inverter_max_current,
                        min_voltage_under_load,
                        max_voltage,
                        min_voltage,
                        max_pack_mass,
                        enforce_mass_constraint,
                        power_limit,
                    )
                    if result:
                        results.append(result)
                        self.winfo_toplevel().after(
                            0, lambda r=result: self._add_result_row(r)
                        )

            self.results = results

            # Find best (lowest lap time among sufficient configs)
            sufficient_results = [r for r in results if r.sufficient]

            # Debug: print counts
            print(
                f"[Pack Optimiser] Total results: {len(results)}, Valid: {len(sufficient_results)}"
            )

            if sufficient_results:
                # Among valid results, prefer: lowest lap time, then lowest mass, then highest power limit
                self.best_result = min(
                    sufficient_results,
                    key=lambda r: (
                        r.lap_time,
                        r.config.pack_mass_kg,
                        -(r.imposed_limit_kW or 0),
                    ),
                )
                self.best_is_valid = True
                print(
                    f"[Pack Optimiser] Selected VALID: {self.best_result.config.config_name} "
                    f"@ {self.best_result.imposed_limit_kW}kW, sufficient={self.best_result.sufficient}"
                )
            elif results:
                # If none sufficient, find one with best lap time anyway (for reference)
                self.best_result = min(results, key=lambda r: r.lap_time)
                self.best_is_valid = False
                print(
                    f"[Pack Optimiser] Selected INVALID (fallback): {self.best_result.config.config_name}, "
                    f"sufficient={self.best_result.sufficient}"
                )
            else:
                self.best_is_valid = False
                print("[Pack Optimiser] No results at all!")

            # Defensive check: verify selection
            if (
                self.best_result
                and sufficient_results
                and not self.best_result.sufficient
            ):
                print(
                    f"[Pack Optimiser] BUG DETECTED! Selected invalid result when valid ones exist!"
                )
                # Force select a valid one
                self.best_result = sufficient_results[0]
                self.best_is_valid = True

            self.winfo_toplevel().after(0, self._show_results)

        except Exception as e:
            import traceback

            error_msg = f"Optimisation error: {str(e)}\n{traceback.format_exc()}"
            self.winfo_toplevel().after(0, lambda: self._show_error(error_msg))

        finally:
            self.winfo_toplevel().after(0, self._optimisation_complete)

    def _simulate_config(
        self,
        pack_config: PackConfig,
        base_config: dict,
        track,
        casing_factor: float,
        fs_max_current: float,
        inverter_max_current: float,
        min_voltage_under_load: float,
        max_voltage: float,
        min_voltage: float,
        max_pack_mass: float,
        enforce_mass_constraint: bool = True,
        imposed_power_limit_kW: float = None,
    ) -> Optional[OptimisationResult]:
        """Simulate a single pack configuration with comprehensive validation.

        Args:
            pack_config: Battery pack configuration to test
            base_config: Base vehicle configuration dict
            track: Track object for simulation
            casing_factor: Pack casing mass multiplier
            fs_max_current: FS rules max current limit [A]
            inverter_max_current: Inverter DC bus current limit [A]
            min_voltage_under_load: Minimum acceptable voltage under load [V]
            max_voltage: Maximum allowed voltage [V] (FS rules: 600V)
            min_voltage: Minimum nominal voltage [V]
            max_pack_mass: Maximum allowed pack mass [kg]
            enforce_mass_constraint: If False, mass limit won't reject configurations
            imposed_power_limit_kW: If set, caps the power draw at this value [kW]
        """
        try:
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
            from solver.battery import validate_battery_capacity

            # Calculate pack mass
            pack_mass = pack_config.pack_mass_kg

            # Get base vehicle mass without battery
            base_mass = base_config["vehicle"]["mass_kg"]
            base_battery_mass = base_config["vehicle"].get("mass_battery_kg", 45)
            mass_without_battery = base_mass - base_battery_mass

            # New total mass with this pack
            total_mass = mass_without_battery + pack_mass

            # Create vehicle
            aero = AeroParams(
                rho=base_config["aero"]["rho"],
                Cd=base_config["aero"]["Cd"],
                Cl=base_config["aero"]["Cl"],
                A=base_config["aero"]["A"],
            )

            tyre = TyreParamsMVP(mu=base_config["tyre"]["mu"])

            pt = base_config["powertrain"]
            if "drivetrain" in pt:
                # Get base motor power
                base_motor_power_kW = pt["motor_power_kW"]

                # Calculate number of motors for this drivetrain
                drivetrain = pt["drivetrain"]
                if drivetrain == "AWD":
                    n_motors = 4
                elif drivetrain.startswith("1"):
                    n_motors = 1
                else:
                    n_motors = 2

                # Apply power limit if set - limit per-motor power
                if imposed_power_limit_kW is not None and imposed_power_limit_kW > 0:
                    # Total power limit divided by number of motors
                    limited_motor_power_kW = imposed_power_limit_kW / n_motors
                    # Use the minimum of base power and limited power
                    effective_motor_power_kW = min(
                        base_motor_power_kW, limited_motor_power_kW
                    )
                else:
                    effective_motor_power_kW = base_motor_power_kW

                powertrain = EVPowertrainParams(
                    drivetrain=drivetrain,
                    motor_power_kW=effective_motor_power_kW,
                    motor_torque_Nm=pt["motor_torque_Nm"],
                    motor_rpm_max=pt["motor_rpm_max"],
                    gear_ratio=pt["gear_ratio"],
                    wheel_radius_m=pt["wheel_radius_m"],
                    motor_efficiency=pt.get("motor_efficiency", 0.96),
                )
            else:
                # Legacy powertrain - apply power limit directly
                base_power_W = pt["P_max_kW"] * 1000
                if imposed_power_limit_kW is not None and imposed_power_limit_kW > 0:
                    effective_power_W = min(base_power_W, imposed_power_limit_kW * 1000)
                else:
                    effective_power_W = base_power_W

                powertrain = EVPowertrainMVP(
                    P_max=effective_power_W,
                    Fx_max=pt["Fx_max_N"],
                )

            battery = BatteryParams(
                capacity_kWh=pack_config.capacity_kWh,
                initial_soc=1.0,
                min_soc=0.1,
                max_discharge_kW=pack_config.max_power_kW,
                eta_discharge=0.95,
                nominal_voltage_V=pack_config.voltage_nominal,
                max_current_A=pack_config.max_current,
            )

            vehicle = VehicleParams(
                m=total_mass,
                g=base_config["vehicle"]["g"],
                Crr=base_config["vehicle"]["Crr"],
                aero=aero,
                tyre=tyre,
                powertrain=powertrain,
                battery=battery,
            )

            # Run simulation
            result, lap_time = solve_qss(track, vehicle)
            v = result["v"]

            # Battery analysis
            bv = validate_battery_capacity(track, v, vehicle)

            # CRITICAL: Get the DEMANDED power, not the LIMITED power
            # bv.peak_power_kW is already capped by battery limits, so we must
            # calculate the raw demanded power from the velocity profile
            from solver.battery import calculate_power_profile

            raw_power_kW = calculate_power_profile(track, v, vehicle)
            peak_demanded_power_kW = np.max(raw_power_kW)  # Raw demand before limiting

            # ============================================
            # APPLY IMPOSED POWER LIMIT (if set)
            # ============================================
            # If user has imposed a power limit, cap the effective power demand
            # This simulates software-limiting the power draw from the battery
            effective_peak_power_kW = peak_demanded_power_kW
            if imposed_power_limit_kW is not None and imposed_power_limit_kW > 0:
                effective_peak_power_kW = min(
                    peak_demanded_power_kW, imposed_power_limit_kW
                )

            # ============================================
            # CURRENT CALCULATION (accounting for voltage loss)
            # ============================================
            # When voltage drops, MORE current is needed to deliver the same power
            # P = V_actual × I = (V_nom - I×R) × I
            # This is solved using the quadratic formula in calculate_current_at_power()
            # Use effective power (which may be limited by user-imposed limit)
            peak_current_A = pack_config.calculate_current_at_power(
                effective_peak_power_kW
            )

            # ============================================
            # VOLTAGE LOSS CALCULATION (I×R drop)
            # ============================================
            # V_loss = I × R (now using the correctly calculated current)
            voltage_loss_V = peak_current_A * pack_config.internal_resistance_ohm

            # Actual voltage under load
            voltage_under_load_V = pack_config.voltage_nominal - voltage_loss_V

            # ============================================
            # I²R POWER LOSS CALCULATION
            # ============================================
            # Power dissipated as heat in pack: P_loss = I² × R
            i2r_loss_kW = (
                (peak_current_A**2) * pack_config.internal_resistance_ohm / 1000.0
            )

            # ============================================
            # CONSTRAINT CHECKS
            # ============================================
            # 1. Cell current limit: I_pack / n <= I_cell_max
            current_per_string = peak_current_A / pack_config.parallel
            cell_current_exceeded = current_per_string > pack_config.cell.max_current_A

            # 2. FS Rules current limit (500A max from accumulator)
            exceeds_fs_current = peak_current_A > fs_max_current

            # 3. Inverter DC bus current limit
            exceeds_inverter_limit = peak_current_A > inverter_max_current

            # 4. Minimum voltage under load check
            voltage_too_low = voltage_under_load_V < min_voltage_under_load

            # 5. Max voltage check (V_max at full charge vs FS 600V limit)
            exceeds_max_voltage = pack_config.voltage_max > max_voltage

            # 6. Min voltage check (V_nom below minimum operating voltage)
            below_min_voltage = pack_config.voltage_nominal < min_voltage

            # 7. Pack mass check
            exceeds_max_mass = pack_config.pack_mass_kg > max_pack_mass

            # Power limited if ANY current/voltage constraint is violated
            power_limited = (
                cell_current_exceeded
                or exceeds_fs_current
                or exceeds_inverter_limit
                or voltage_too_low
            )

            # Pack is valid if: all constraints met
            # Mass constraint is only applied if enforce_mass_constraint is True
            mass_ok = not exceeds_max_mass if enforce_mass_constraint else True
            valid = (
                bv.sufficient
                and not power_limited
                and not exceeds_max_voltage
                and not below_min_voltage
                and mass_ok
            )

            # Determine if user-imposed power limit is actively capping the power
            power_limit_active = (
                imposed_power_limit_kW is not None
                and effective_peak_power_kW < peak_demanded_power_kW
            )

            return OptimisationResult(
                config=pack_config,
                lap_time=lap_time,
                energy_used_kWh=bv.total_energy_kWh,
                final_soc=bv.final_soc,
                sufficient=valid,
                total_vehicle_mass=total_mass,
                peak_power_kW=peak_demanded_power_kW,
                power_limited=power_limited,
                peak_current_A=peak_current_A,
                voltage_loss_V=voltage_loss_V,
                voltage_under_load_V=voltage_under_load_V,
                i2r_loss_kW=i2r_loss_kW,
                exceeds_fs_current=exceeds_fs_current,
                exceeds_inverter_limit=exceeds_inverter_limit,
                exceeds_max_voltage=exceeds_max_voltage,
                below_min_voltage=below_min_voltage,
                exceeds_max_mass=exceeds_max_mass,
                imposed_limit_kW=imposed_power_limit_kW,
                effective_power_kW=effective_peak_power_kW
                if power_limit_active
                else None,
                power_limit_active=power_limit_active,
            )

        except Exception as e:
            print(f"Error simulating {pack_config.config_name}: {e}")
            return None

    def _update_progress(self, progress: float, status: str):
        """Update progress display."""
        self.progress_var.set(progress)
        self.status_var.set(status)

    def _add_result_row(self, result: OptimisationResult):
        """Add a result row to the treeview."""
        config = result.config

        # Determine status with specific failure reason (check in priority order)
        if result.sufficient:
            status = "✓ OK"
        elif result.exceeds_max_voltage:
            status = "✗ V > max"
        elif result.below_min_voltage:
            status = "✗ V < min"
        elif result.exceeds_max_mass:
            status = "✗ Mass"
        elif result.exceeds_fs_current:
            status = "✗ FS 500A"
        elif result.exceeds_inverter_limit:
            status = "✗ Inverter"
        elif result.power_limited:
            # Cell current or voltage loss issue
            current_per_string = result.peak_current_A / config.parallel
            if current_per_string > config.cell.max_current_A:
                status = "✗ Cell I"
            else:
                status = "✗ V loss"
        else:
            status = "✗ LOW SoC"

        # Format power limit column
        p_lim_str = (
            f"{result.imposed_limit_kW:.0f}"
            if result.imposed_limit_kW is not None
            else "—"
        )

        self.results_tree.insert(
            "",
            "end",
            values=(
                config.config_name,
                config.total_cells,
                p_lim_str,
                f"{config.voltage_nominal:.0f}",
                f"{result.voltage_under_load_V:.0f}",
                f"{result.peak_current_A:.0f}",
                f"{result.i2r_loss_kW:.1f}",
                f"{config.capacity_kWh:.2f}",
                f"{config.pack_mass_kg:.1f}",
                f"{result.lap_time:.3f}",
                status,
            ),
        )

    def _show_results(self):
        """Display final results summary."""
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")

        if self.best_result:
            r = self.best_result
            c = r.config

            # Current margin calculation
            current_per_string = r.peak_current_A / c.parallel
            current_margin = (
                (c.cell.max_current_A - current_per_string) / c.cell.max_current_A
            ) * 100

            # Power margin calculation
            max_pack_power_kW = c.max_power_kW
            power_margin = (
                ((max_pack_power_kW - r.peak_power_kW) / max_pack_power_kW) * 100
                if max_pack_power_kW > 0
                else 0
            )

            # Voltage margin
            voltage_margin_pct = (r.voltage_under_load_V / c.voltage_nominal) * 100

            # Efficiency (power delivered vs power from pack including I²R)
            efficiency = (
                (r.peak_power_kW / (r.peak_power_kW + r.i2r_loss_kW)) * 100
                if r.peak_power_kW > 0
                else 100
            )

            # Check if this is a valid result or a fallback
            is_valid = getattr(self, "best_is_valid", r.sufficient)

            lines = []

            # Show warning if no valid configs found
            if not is_valid:
                lines.extend(
                    [
                        f"{'!' * 50}",
                        f"  ⚠️  NO VALID CONFIGURATIONS FOUND  ⚠️",
                        f"{'!' * 50}",
                        f"",
                        f"  All tested configurations failed at least one",
                        f"  constraint. Showing best FAILING config below",
                        f"  for reference only.",
                        f"",
                        f"  Check your constraints or expand search range.",
                        f"",
                    ]
                )

            lines.extend(
                [
                    f"{'═' * 50}",
                    f"  {'BEST FAILING CONFIG (reference)' if not is_valid else 'OPTIMAL BATTERY CONFIGURATION'}",
                    f"{'═' * 50}",
                    f"",
                    f"  Configuration:    {c.config_name}",
                    f"  Total Cells:      {c.total_cells}",
                    f"  Capacity:         {c.capacity_kWh:.2f} kWh",
                    f"  Cell Mass:        {c.cell_mass_kg:.2f} kg",
                    f"  Pack Mass:        {c.pack_mass_kg:.1f} kg (with casing)",
                    f"  Int. Resistance:  {c.internal_resistance:.1f} mΩ",
                    f"",
                    f"{'─' * 50}",
                    f"  VOLTAGE (under peak load)",
                    f"{'─' * 50}",
                    f"  Nominal Voltage:   {c.voltage_nominal:.0f} V",
                    f"  Voltage Loss:      {r.voltage_loss_V:.1f} V  (I×R drop)",
                    f"  Voltage @ Load:    {r.voltage_under_load_V:.0f} V  ({voltage_margin_pct:.1f}%)",
                    f"",
                    f"{'─' * 50}",
                    f"  CURRENT LIMITS",
                    f"{'─' * 50}",
                    f"  Peak Current:      {r.peak_current_A:.0f} A",
                    f"  Per String:        {current_per_string:.1f} A  (limit: {c.cell.max_current_A}A)",
                    f"  Cell I Margin:     {current_margin:.1f}%",
                    f"  FS Rules (500A):   {'✓ OK' if not r.exceeds_fs_current else '✗ EXCEEDED'}",
                    f"  Inverter Limit:    {'✓ OK' if not r.exceeds_inverter_limit else '✗ EXCEEDED'}",
                    f"",
                    f"{'─' * 50}",
                    f"  POWER & EFFICIENCY",
                    f"{'─' * 50}",
                    f"  Peak Power Demand: {r.peak_power_kW:.1f} kW",
                ]
            )

            # Add power limit line if set
            if r.imposed_limit_kW is not None:
                if r.power_limit_active:
                    lines.append(
                        f"  Imposed Limit:     {r.imposed_limit_kW:.0f} kW  ⚡ ACTIVE"
                    )
                else:
                    lines.append(
                        f"  Imposed Limit:     {r.imposed_limit_kW:.0f} kW  (not limiting)"
                    )

            lines.extend(
                [
                    f"  I²R Loss (peak):   {r.i2r_loss_kW:.2f} kW",
                    f"  Pack Efficiency:   {efficiency:.1f}%",
                    f"",
                    f"{'─' * 50}",
                    f"  PERFORMANCE",
                    f"{'─' * 50}",
                    f"  Lap Time:          {r.lap_time:.3f} s",
                    f"  Energy Used:       {r.energy_used_kWh:.3f} kWh",
                    f"  Final SoC:         {r.final_soc * 100:.1f}%",
                    f"  Vehicle Mass:      {r.total_vehicle_mass:.1f} kg",
                    f"  Status:            {'VALID ✓' if r.sufficient else 'INVALID ✗'}",
                    f"",
                    f"{'═' * 50}",
                ]
            )
            self.summary_text.insert("1.0", "\n".join(lines))
        else:
            self.summary_text.insert("1.0", "No valid configurations found.")

        self.summary_text.configure(state="disabled")

        total_tested = len(self.results)
        sufficient = len([r for r in self.results if r.sufficient])
        self.status_var.set(
            f"Complete - {total_tested} configs tested, {sufficient} valid"
        )

    def _show_error(self, message: str):
        """Display error message."""
        self.status_var.set("Error occurred")
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", f"ERROR:\n\n{message}")
        self.summary_text.configure(state="disabled")

    def _optimisation_complete(self):
        """Clean up after optimisation completes."""
        self.running = False
        self.run_button.configure(state="normal", text="RUN OPTIMISATION")
        self.progress_var.set(100)
