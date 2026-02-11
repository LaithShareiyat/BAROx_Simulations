"""Main BAROx GUI Application."""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sys
from dataclasses import replace
import numpy as np
from PIL import Image, ImageTk

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.panels.control_panel import ControlPanel
from gui.panels.results_panel import ResultsPanel


class BAROxGUI:
    """Main application class for BAROx GUI."""

    def __init__(self, root: tk.Tk):
        """Initialise the GUI application."""
        self.root = root
        self.root.title("BAROx - Formula Student Lap Time Simulator")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)

        # Configure styles
        self._configure_styles()

        # Create main layout
        self._create_layout()

        # Simulation state
        self.running = False
        self.results = {}

    # Current theme state
    _dark_mode = False

    def _configure_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        # Use clam as base — lightweight and cross-platform
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self._apply_theme(style)

    def _apply_theme(self, style: ttk.Style = None):
        """Apply dark or light colour scheme on top of the clam theme."""
        if style is None:
            style = ttk.Style()
        is_dark = self._dark_mode

        if is_dark:
            bg = "#2b2b2b"
            fg = "#e0e0e0"
            field_bg = "#3c3f41"
            select_bg = "#4a6984"
            border = "#555555"
        else:
            bg = "#dcdad5"
            fg = "#000000"
            field_bg = "#ffffff"
            select_bg = "#4a6984"
            border = "#9e9a91"

        # Override clam base colours
        style.configure(".", background=bg, foreground=fg,
                        fieldbackground=field_bg, bordercolor=border,
                        selectbackground=select_bg, selectforeground="#ffffff",
                        insertcolor=fg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TLabelFrame", background=bg, padding=5)
        style.configure("TLabelFrame.Label", background=bg, foreground=fg,
                        font=("Segoe UI", 9, "bold"))
        style.configure("TNotebook", background=bg)
        style.configure("TNotebook.Tab", background=bg, foreground=fg, padding=[8, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", field_bg), ("!selected", bg)],
                  foreground=[("selected", fg), ("!selected", fg)])
        style.configure("TButton", background=bg, foreground=fg)
        style.map("TButton", background=[("active", select_bg)])
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=10)
        style.configure("TEntry", fieldbackground=field_bg, foreground=fg)
        style.configure("TCombobox", fieldbackground=field_bg, foreground=fg)
        style.map("TCombobox", fieldbackground=[("readonly", field_bg)])
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.configure("TRadiobutton", background=bg, foreground=fg)
        style.configure("TPanedwindow", background=bg)
        style.configure("Horizontal.TScrollbar", background=bg, troughcolor=bg)
        style.configure("Vertical.TScrollbar", background=bg, troughcolor=bg)

        # Invalid entry
        style.configure("Invalid.TEntry", fieldbackground="#ffcccc")

        # Unit label style (for control_panel kg, kW, etc.)
        unit_fg = "#888888" if is_dark else "gray"
        style.configure("Unit.TLabel", background=bg, foreground=unit_fg)

        # Theme-aware text colours for results cards
        text_primary = "#e0e0e0" if is_dark else "#1a1a2e"
        text_secondary = "#aaaaaa" if is_dark else "#555555"
        text_data = "#cccccc" if is_dark else "#444444"
        text_header = "#dddddd" if is_dark else "#333333"
        text_cell = "#cccccc" if is_dark else "#222222"
        text_card_title = "#e0e0e0" if is_dark else "#1a1a2e"
        text_section = "#b0bec5" if is_dark else "#37474f"

        style.configure("Headline.TLabel", font=("Segoe UI", 28, "bold"),
                        background=bg, foreground=text_primary)
        style.configure("HeadlineUnit.TLabel", font=("Segoe UI", 14),
                        background=bg, foreground=text_secondary)
        style.configure("DataLabel.TLabel", font=("Segoe UI", 13),
                        background=bg, foreground=text_data)
        style.configure("DataValue.TLabel", font=("Segoe UI", 13, "bold"),
                        background=bg, foreground=text_primary)
        style.configure("Pass.TLabel", font=("Segoe UI", 16, "bold"),
                        background=bg, foreground="#4caf50" if is_dark else "#2e7d32")
        style.configure("Fail.TLabel", font=("Segoe UI", 16, "bold"),
                        background=bg, foreground="#ef5350" if is_dark else "#c62828")
        style.configure("Warning.TLabel", font=("Segoe UI", 12),
                        background=bg, foreground="#ffb74d" if is_dark else "#e65100")
        style.configure("TableHeader.TLabel", font=("Segoe UI", 12, "bold"),
                        background=bg, foreground=text_header)
        style.configure("TableCell.TLabel", font=("Segoe UI", 12),
                        background=bg, foreground=text_cell)
        style.configure("CardTitle.TLabel", font=("Segoe UI", 14, "bold"),
                        background=bg, foreground=text_card_title)
        style.configure("SectionTitle.TLabel", font=("Segoe UI", 12, "bold"),
                        background=bg, foreground=text_section)

    def _create_logo_banner(self):
        """Create the logo banner at the top of the window."""
        # Get path to logo
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "resources", "BAROx_Logo.png"
        )

        # Create banner frame with Oxford Blue background
        self.banner_frame = tk.Frame(self.root, bg="#12243e", height=70)
        self.banner_frame.pack(fill="x", side="top", padx=5, pady=(5, 0))
        self.banner_frame.pack_propagate(False)  # Maintain fixed height

        if os.path.exists(logo_path):
            try:
                # Load and resize logo
                original_image = Image.open(logo_path)

                # Calculate new dimensions maintaining aspect ratio
                banner_height = 60
                aspect_ratio = original_image.width / original_image.height
                new_width = int(banner_height * aspect_ratio)

                # Resize with high-quality resampling
                resized_image = original_image.resize(
                    (new_width, banner_height), Image.Resampling.LANCZOS
                )

                # Convert to PhotoImage and store reference
                self.logo_image = ImageTk.PhotoImage(resized_image)

                # Create label with logo
                logo_label = tk.Label(
                    self.banner_frame, image=self.logo_image, bg="#12243e"
                )
                logo_label.pack(side="left", padx=10, pady=5)

                # Add application title on the right side
                title_label = tk.Label(
                    self.banner_frame,
                    text="Lap Time Simulator",
                    font=("Segoe UI", 16, "bold"),
                    fg="white",
                    bg="#12243e",
                )
                title_label.pack(side="right", padx=20, pady=5)

                # Dark/light mode toggle
                self.theme_toggle_btn = tk.Button(
                    self.banner_frame,
                    text="\u2600",  # Sun symbol (currently light mode)
                    font=("Segoe UI", 16),
                    fg="#f0c040", bg="#12243e",
                    bd=0, highlightthickness=0,
                    activebackground="#12243e", activeforeground="#d4a017",
                    command=self._toggle_theme,
                )
                self.theme_toggle_btn.pack(side="right", padx=(0, 5))

            except Exception as e:
                # Fallback to text-only banner if image fails
                self._create_text_banner()
        else:
            self._create_text_banner()

    def _create_text_banner(self):
        """Create a text-only banner as fallback."""
        title_label = tk.Label(
            self.banner_frame,
            text="BAROx - Formula Student Lap Time Simulator",
            font=("Segoe UI", 14, "bold"),
            fg="white",
            bg="#12243e",
        )
        title_label.pack(expand=True)

        # Dark/light mode toggle
        self.theme_toggle_btn = tk.Button(
            self.banner_frame,
            text="\u2600",
            font=("Segoe UI", 16),
            fg="#f0c040", bg="#12243e",
            bd=0, highlightthickness=0,
            activebackground="#12243e", activeforeground="#d4a017",
            command=self._toggle_theme,
        )
        self.theme_toggle_btn.pack(side="right", padx=(0, 10))

    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        self._dark_mode = not self._dark_mode
        is_dark = self._dark_mode

        # Update toggle button symbol and colour
        if is_dark:
            self.theme_toggle_btn.config(text="\u263e", fg="#c0c0e0")  # Silver moon
        else:
            self.theme_toggle_btn.config(text="\u2600", fg="#f0c040")  # Gold sun

        # Re-apply theme colours
        self._apply_theme()

        # Update results panel theme-dependent colours
        if hasattr(self, "results_panel"):
            self.results_panel.update_theme_colours()

        # Re-render plots if results exist
        if self.results:
            event_type = (
                "both" if "autocross" in self.results and "skidpad" in self.results
                else "autocross" if "autocross" in self.results
                else "skidpad"
            )
            self._update_display(self.results, event_type)

    def _create_layout(self):
        """Create the main application layout."""
        # Create logo banner at the top
        self._create_logo_banner()

        # Main paned window for resizable panels
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill="both", expand=True, padx=5, pady=5)

        # Left panel - Controls (fixed width)
        self.control_frame = ttk.Frame(self.paned, width=350)
        self.control_panel = ControlPanel(
            self.control_frame,
            on_run=self.run_simulation,
            on_save_results=self.save_results,
        )
        self.control_panel.pack(fill="both", expand=True)

        # Right panel - Results (with config callback for battery optimiser)
        self.results_frame = ttk.Frame(self.paned)
        self.results_panel = ResultsPanel(
            self.results_frame, get_config_callback=self.control_panel.get_config
        )
        self.results_panel.pack(fill="both", expand=True)

        # Add panels to paned window
        self.paned.add(self.control_frame, weight=0)
        self.paned.add(self.results_frame, weight=1)

    def run_simulation(self):
        """Run the simulation in a background thread."""
        if self.running:
            return

        self.running = True
        self.control_panel.set_running(True)
        self.results_panel.set_status("Running simulation...")
        self.results_panel.clear_all()

        # Run in background thread
        thread = threading.Thread(target=self._run_simulation_thread)
        thread.daemon = True
        thread.start()

    def _run_simulation_thread(self):
        """Execute simulation in background thread."""
        try:
            config = self.control_panel.get_config()
            event_type = self.control_panel.get_event()

            # Import simulation modules
            from models.vehicle import (
                VehicleParams,
                AeroParams,
                TyreParamsMVP,
                EVPowertrainMVP,
                EVPowertrainParams,
                BatteryParams,
            )
            from solver.qss_speed import solve_qss
            from solver.metrics import lap_time, channels, energy_consumption
            from solver.battery import simulate_battery, validate_battery_capacity

            # Create vehicle from config
            vehicle = self._create_vehicle(config)

            results = {}

            # Run Autocross
            if event_type in ("autocross", "both"):
                self.root.after(
                    0,
                    lambda: self.results_panel.set_status(
                        "Running Autocross simulation..."
                    ),
                )
                results["autocross"] = self._run_autocross(vehicle, config)

            # Run Skidpad
            if event_type in ("skidpad", "both"):
                self.root.after(
                    0,
                    lambda: self.results_panel.set_status(
                        "Running Skidpad simulation..."
                    ),
                )
                results["skidpad"] = self._run_skidpad(vehicle)

            # Run parameter sweeps (autocross only)
            if event_type in ("autocross", "both"):
                self.root.after(
                    0,
                    lambda: self.results_panel.set_status(
                        "Running parameter sweeps..."
                    ),
                )
                results["sweeps"] = self._run_parameter_sweeps(vehicle, config)

            self.results = results

            # Update GUI in main thread
            self.root.after(0, lambda: self._update_display(results, event_type))

        except Exception as e:
            import traceback

            error_msg = f"Simulation error: {str(e)}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._show_error(error_msg))

        finally:
            self.root.after(0, self._simulation_complete)

    def _create_vehicle(self, config: dict):
        """Create VehicleParams from config dictionary."""
        from models.vehicle import (
            VehicleParams,
            AeroParams,
            EVPowertrainMVP,
            EVPowertrainParams,
            BatteryParams,
            VehicleGeometry,
            TorqueVectoringParams,
            build_tyre_from_config,
        )

        aero = AeroParams(
            rho=config["aero"]["rho"],
            Cd=config["aero"]["Cd"],
            Cl=config["aero"]["Cl"],
            A=config["aero"]["A"],
        )

        tyre = build_tyre_from_config(config["tyre"])

        # Check if using new extended powertrain format or legacy format
        pt_config = config["powertrain"]
        if "drivetrain" in pt_config:
            # New extended powertrain format
            powertrain = EVPowertrainParams(
                drivetrain=pt_config["drivetrain"],
                motor_power_kW=pt_config["motor_power_kW"],
                motor_torque_Nm=pt_config["motor_torque_Nm"],
                motor_rpm_max=pt_config["motor_rpm_max"],
                gear_ratio=pt_config["gear_ratio"],
                wheel_radius_m=pt_config["wheel_radius_m"],
                motor_efficiency=pt_config.get("motor_efficiency", 0.96),
                powertrain_overhead_kg=pt_config.get("powertrain_overhead_kg", 10.0),
                inverter_peak_power_kW=pt_config.get("inverter_peak_power_kW", 320.0),
                inverter_peak_current_A=pt_config.get("inverter_peak_current_A", 600.0),
                inverter_weight_kg=pt_config.get("inverter_weight_kg", 6.9),
            )
        else:
            # Legacy format with P_max_kW and Fx_max_N
            powertrain = EVPowertrainMVP(
                P_max=pt_config["P_max_kW"] * 1000,
                Fx_max=pt_config["Fx_max_N"],
            )

        battery = None
        if config["battery"].get("enabled", True):
            battery = BatteryParams(
                capacity_kWh=config["battery"]["capacity_kWh"],
                initial_soc=config["battery"]["initial_soc"],
                min_soc=config["battery"]["min_soc"],
                max_discharge_kW=config["battery"]["max_discharge_kW"],
                eta_discharge=config["battery"].get("eta_discharge", 0.95),
                # Current limiting (142S5P pack default)
                nominal_voltage_V=config["battery"].get("nominal_voltage_V", 511.0),
                max_current_A=config["battery"].get("max_current_A", 175.0),
                # Regenerative braking parameters
                regen_enabled=config["battery"].get("regen_enabled", False),
                eta_regen=config["battery"].get("eta_regen", 0.85),
                max_regen_kW=config["battery"].get("max_regen_kW", 50.0),
                regen_capture_percent=config["battery"].get(
                    "regen_capture_percent", 100.0
                ),
            )

        # Create geometry params if present (for bicycle model)
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

        # Create torque vectoring params if present
        torque_vectoring = None
        if "torque_vectoring" in config:
            tv_config = config["torque_vectoring"]
            torque_vectoring = TorqueVectoringParams(
                enabled=tv_config.get("enabled", False),
                effectiveness=tv_config.get("effectiveness", 1.0),
                max_torque_transfer=tv_config.get("max_torque_transfer", 0.5),
                strategy=tv_config.get("strategy", "load_proportional"),
            )

        return VehicleParams(
            m=config["vehicle"]["mass_kg"],
            g=config["vehicle"]["g"],
            Crr=config["vehicle"]["Crr"],
            aero=aero,
            tyre=tyre,
            powertrain=powertrain,
            battery=battery,
            geometry=geometry,
            torque_vectoring=torque_vectoring,
        )

    def _run_autocross(self, vehicle, config: dict) -> dict:
        """Run autocross simulation and return metrics."""
        from events.autocross_generator import (
            build_standard_autocross,
            validate_autocross,
        )
        from solver.qss_speed import solve_qss
        from solver.battery import simulate_battery, validate_battery_capacity

        # Build track
        track, metadata = build_standard_autocross()

        # Solve for velocity profile
        result, t_lap = solve_qss(track, vehicle)
        v = result["v"]

        # Compute metrics
        metrics = self._compute_metrics(track, v, vehicle)
        metrics["metadata"] = metadata
        metrics["track"] = track
        metrics["v"] = v
        metrics["v_lat"] = result["v_lat"]
        metrics["v_fwd"] = result["v_fwd"]
        metrics["v_bwd"] = result["v_bwd"]
        metrics["max_speed_analysis"] = self._analyse_max_speed(
            track, v, result["v_fwd"], result["v_bwd"], result["v_lat"], vehicle
        )

        # Compute RPM profile if using extended powertrain
        if hasattr(vehicle.powertrain, "gear_ratio"):
            wheel_rpm = (v / (2 * np.pi * vehicle.powertrain.wheel_radius_m)) * 60
            motor_rpm = wheel_rpm * vehicle.powertrain.gear_ratio
            metrics["motor_rpm_profile"] = motor_rpm
            metrics["motor_rpm_limit"] = vehicle.powertrain.motor_rpm_max

        # Always include vehicle for power demand and sweep plots
        metrics["vehicle"] = vehicle

        # Battery analysis
        if vehicle.battery is not None:
            battery_validation = validate_battery_capacity(track, v, vehicle)
            battery_state = simulate_battery(track, v, vehicle)
            metrics["battery_validation"] = battery_validation
            metrics["battery_sufficient"] = battery_validation.sufficient
            metrics["battery_state"] = battery_state

        return metrics

    def _run_skidpad(self, vehicle) -> dict:
        """Run skidpad simulation and return metrics."""
        from events.skidpad import (
            build_skidpad_track,
            skidpad_time_from_single_circle,
            SKIDPAD_CENTRE_RADIUS,
        )
        from solver.qss_speed import solve_qss
        from solver.battery import simulate_battery, validate_battery_capacity

        # Build track
        track = build_skidpad_track()

        # Solve
        result, t_lap = solve_qss(track, vehicle)
        v = result["v"]

        # Get timing
        timing = skidpad_time_from_single_circle(t_lap)

        # Compute metrics
        metrics = self._compute_metrics(track, v, vehicle)
        metrics["t_official"] = timing["t_official"]
        metrics["t_full_run"] = timing["t_full_run"]
        metrics["track"] = track
        metrics["v"] = v
        metrics["v_lat"] = result["v_lat"]

        # Compute RPM profile if using extended powertrain
        if hasattr(vehicle.powertrain, "gear_ratio"):
            wheel_rpm = (v / (2 * np.pi * vehicle.powertrain.wheel_radius_m)) * 60
            motor_rpm = wheel_rpm * vehicle.powertrain.gear_ratio
            metrics["motor_rpm_profile"] = motor_rpm
            metrics["motor_rpm_limit"] = vehicle.powertrain.motor_rpm_max

        # Always include vehicle for power demand plots
        metrics["vehicle"] = vehicle

        # Battery analysis (same as autocross)
        if vehicle.battery is not None:
            battery_validation = validate_battery_capacity(track, v, vehicle)
            battery_state = simulate_battery(track, v, vehicle)
            metrics["battery_validation"] = battery_validation
            metrics["battery_sufficient"] = battery_validation.sufficient
            metrics["battery_state"] = battery_state

        return metrics

    def _run_parameter_sweeps(self, vehicle, config: dict) -> dict:
        """Run gear ratio sensitivity sweep on autocross.

        Returns a dict with key 'gear_ratio_sweep'.
        Value is a dict with 'values', 'lap_times', and 'current_value'.
        """
        from events.autocross_generator import build_standard_autocross
        from solver.qss_speed import solve_qss

        track, _ = build_standard_autocross()
        n_points = 21
        sweeps = {}

        # --- Gear ratio sweep (only for extended powertrain) ---
        if hasattr(vehicle.powertrain, "gear_ratio"):
            current_gr = vehicle.powertrain.gear_ratio
            gr_lo = max(1.0, current_gr * 0.5)
            gr_hi = min(10.0, current_gr * 1.5)
            gr_values = np.linspace(gr_lo, gr_hi, n_points)
            gr_lap_times = np.full(n_points, np.nan)

            for i, gr_val in enumerate(gr_values):
                try:
                    pt_mod = replace(vehicle.powertrain, gear_ratio=gr_val)
                    v_mod = replace(vehicle, powertrain=pt_mod)
                    _, t_lap = solve_qss(track, v_mod)
                    gr_lap_times[i] = t_lap
                except Exception:
                    pass

            sweeps["gear_ratio_sweep"] = {
                "values": gr_values,
                "lap_times": gr_lap_times,
                "current_value": current_gr,
            }
        else:
            sweeps["gear_ratio_sweep"] = None

        return sweeps

    def _compute_metrics(self, track, v: np.ndarray, vehicle) -> dict:
        """Compute performance metrics from velocity profile."""
        from solver.metrics import lap_time, channels, energy_consumption

        t = lap_time(track, v)
        ax, ay = channels(track, v)

        try:
            energy = energy_consumption(track, v, vehicle)
            energy_kwh = energy.get("E_net_kWh", 0)
        except Exception:
            energy_kwh = 0

        v_positive = v[v > 0.1]
        min_speed = np.min(v_positive) if len(v_positive) > 0 else 0.0

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
            "energy_consumed_kWh": energy_kwh,
        }

    def _analyse_max_speed(self, track, v, v_fwd, v_bwd, v_lat, vehicle):
        """Analyse max speed: when/where, how many times, and what limits it."""
        from physics.powertrain import max_tractive_force_extended
        from physics.aero import drag
        from physics.resistive import rolling_resistance

        max_speed = np.nanmax(v)

        # Within 0.5% of max speed counts as "at max speed"
        threshold = max_speed * 0.995

        # Group consecutive points at max speed into distinct occurrences
        at_max = v >= threshold
        padded = np.concatenate([[False], at_max, [False]])
        transitions = np.diff(padded.astype(int))
        starts = np.where(transitions == 1)[0]
        n_times = len(starts)

        # Peak index (absolute maximum)
        peak_idx = np.argmax(v)
        peak_distance = track.s[peak_idx]

        # Compute cumulative time up to peak
        v_avg = 0.5 * (v[:-1] + v[1:])
        dt = track.ds / np.maximum(v_avg, 0.1)
        cum_time = np.concatenate([[0], np.cumsum(dt)])
        peak_time = cum_time[peak_idx]

        # Determine what limits the max speed
        v_max_rpm = getattr(vehicle.powertrain, "v_max_rpm", 1000.0)

        if abs(max_speed - v_max_rpm) / max(v_max_rpm, 0.1) < 0.02:
            limiter = "RPM limit"
            limiter_detail = (
                f"Motor tops out at {v_max_rpm:.1f} m/s "
                f"({v_max_rpm * 3.6:.0f} km/h)"
            )
        elif (
            v_lat[peak_idx] <= v_fwd[peak_idx]
            and v_lat[peak_idx] <= v_bwd[peak_idx]
        ):
            limiter = "Cornering grip"
            limiter_detail = "Tyre grip limits speed through corner"
        else:
            F_motor = max_tractive_force_extended(vehicle.powertrain, max_speed)
            F_drag_val = drag(vehicle.aero.rho, vehicle.aero.CD_A, max_speed)
            F_rr = rolling_resistance(vehicle.Crr, vehicle.m, vehicle.g)
            F_net = F_motor - F_drag_val - F_rr

            if F_motor < 0.01:
                limiter = "RPM limit"
                limiter_detail = "Beyond motor RPM range"
            elif F_net / max(F_motor, 1) < 0.10:
                limiter = "Power limit (terminal velocity)"
                limiter_detail = (
                    f"Motor force ({F_motor:.0f} N) ~ "
                    f"Drag + RR ({F_drag_val + F_rr:.0f} N)"
                )
            else:
                limiter = "Track layout (braking for corner)"
                limiter_detail = (
                    f"Net force: {F_net:.0f} N available — "
                    f"must brake for upcoming corner"
                )

        # Compute motor RPM at max speed
        if hasattr(vehicle.powertrain, "gear_ratio"):
            wheel_rpm = (max_speed / (2 * np.pi * vehicle.powertrain.wheel_radius_m)) * 60
            motor_rpm = wheel_rpm * vehicle.powertrain.gear_ratio
            motor_rpm_max = vehicle.powertrain.motor_rpm_max
        else:
            motor_rpm = None
            motor_rpm_max = None

        return {
            "max_speed_mps": max_speed,
            "max_speed_kmh": max_speed * 3.6,
            "peak_distance_m": peak_distance,
            "peak_time_s": peak_time,
            "n_times": n_times,
            "motor_rpm": motor_rpm,
            "motor_rpm_max": motor_rpm_max,
            "limiter": limiter,
            "limiter_detail": limiter_detail,
        }

    def _update_display(self, results: dict, event_type: str):
        """Update the display with simulation results."""
        # Update text results
        self.results_panel.display_results(results, event_type)

        # Prepare data for plots
        autocross_data = results.get("autocross")
        skidpad_data = results.get("skidpad")

        # Update track layout plot (geometry only)
        self.results_panel.update_layout_plot(autocross_data, skidpad_data)

        # Update speed track map (velocity coloured)
        self.results_panel.update_speed_track_plot(autocross_data, skidpad_data)

        # Update independent plot tabs
        self.results_panel.update_speed_distance_plot(autocross_data, skidpad_data)
        self.results_panel.update_accel_distance_plot(autocross_data, skidpad_data)
        self.results_panel.update_rpm_distance_plot(autocross_data, skidpad_data)
        self.results_panel.update_power_demand_plot(autocross_data, skidpad_data)

        # Update sensitivity sweep plot
        sweeps = results.get("sweeps", {})
        self.results_panel.update_gear_ratio_sweep_plot(sweeps.get("gear_ratio_sweep"))

        # Update battery plot (for both events)
        self.results_panel.update_battery_combined_plot(autocross_data, skidpad_data)

        # Show results tab
        self.results_panel.show_tab("results")

        # Update status
        if "autocross" in results and "skidpad" in results:
            ac_time = results["autocross"]["lap_time"]
            sp_time = results["skidpad"]["t_official"]
            self.results_panel.set_status(
                f"Complete - Autocross: {ac_time:.3f}s | Skidpad: {sp_time:.3f}s"
            )
        elif "autocross" in results:
            lap_time = results["autocross"]["lap_time"]
            self.results_panel.set_status(f"Complete - Autocross: {lap_time:.3f}s")
        elif "skidpad" in results:
            lap_time = results["skidpad"]["t_official"]
            self.results_panel.set_status(f"Complete - Skidpad: {lap_time:.3f}s")

    def _show_error(self, message: str):
        """Display error message."""
        self.results_panel.set_status("Error occurred", "error")
        self.results_panel.set_results_text(f"ERROR:\n\n{message}")
        messagebox.showerror("Simulation Error", message[:500])

    def _simulation_complete(self):
        """Clean up after simulation completes."""
        self.running = False
        self.control_panel.set_running(False)

    def save_results(self):
        """Save all plots and results to a user-selected folder."""
        from tkinter import filedialog
        from datetime import datetime

        # Check if there are results to save
        if not self.results:
            messagebox.showwarning(
                "No Results", "No simulation results to save. Run a simulation first."
            )
            return

        # Ask user to select a folder
        folder_path = filedialog.askdirectory(
            title="Select Folder to Save Results", initialdir=os.path.expanduser("~")
        )

        if not folder_path:
            return  # User cancelled

        try:
            # Create timestamped subfolder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_folder = os.path.join(folder_path, f"BAROx_Results_{timestamp}")
            os.makedirs(save_folder, exist_ok=True)

            saved_files = []

            # Save plots
            plots_folder = os.path.join(save_folder, "plots")
            os.makedirs(plots_folder, exist_ok=True)

            # Save each plot canvas
            plot_configs = [
                (self.results_panel.layout_canvas, "track_layout.png"),
                (self.results_panel.track_canvas, "speed_track_map.png"),
                (self.results_panel.speed_distance_canvas, "speed_vs_distance.png"),
                (self.results_panel.rpm_distance_canvas, "rpm_vs_distance.png"),
                (self.results_panel.power_demand_canvas, "power_demand.png"),
                (self.results_panel.gear_ratio_sweep_canvas, "gear_ratio_sweep.png"),
                (self.results_panel.battery_canvas, "battery_analysis.png"),
            ]

            # Add config comparison chart if it exists
            if hasattr(self.results_panel, "config_comparison"):
                plot_configs.append(
                    (self.results_panel.config_comparison.chart_canvas, "config_comparison.png"),
                )

            for canvas, filename in plot_configs:
                try:
                    fig = canvas.get_figure()
                    filepath = os.path.join(plots_folder, filename)
                    fig.savefig(
                        filepath, dpi=150, bbox_inches="tight", facecolor="white"
                    )
                    saved_files.append(f"plots/{filename}")
                except Exception as e:
                    print(f"Failed to save {filename}: {e}")

            # Save results text
            results_text = self._generate_results_text()
            results_filepath = os.path.join(save_folder, "results.txt")
            with open(results_filepath, "w") as f:
                f.write(results_text)
            saved_files.append("results.txt")

            # Save configuration
            config = self.control_panel.get_config()
            config_filepath = os.path.join(save_folder, "config.yaml")
            import yaml

            with open(config_filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            saved_files.append("config.yaml")

            # Show success message
            files_list = "\n".join(f"  • {f}" for f in saved_files)
            messagebox.showinfo(
                "Results Saved",
                f"Results saved to:\n{save_folder}\n\nFiles saved:\n{files_list}",
            )
            self.results_panel.set_status(f"Results saved to {save_folder}")

        except Exception as e:
            import traceback

            error_msg = f"Failed to save results: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Save Error", error_msg[:500])

    def _generate_results_text(self) -> str:
        """Generate comprehensive results text for saving."""
        from datetime import datetime

        lines = []
        lines.append("=" * 70)
        lines.append("BAROx SIMULATION RESULTS")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Configuration summary
        config = self.control_panel.get_config()
        lines.append("CONFIGURATION")
        lines.append("-" * 50)
        lines.append(f"Vehicle Mass: {config['vehicle']['mass_kg']} kg")
        lines.append(f"Rolling Resistance (Crr): {config['vehicle']['Crr']}")
        lines.append(f"Air Density: {config['aero']['rho']} kg/m³")
        lines.append(f"Drag Coefficient (Cd): {config['aero']['Cd']}")
        lines.append(f"Lift Coefficient (Cl): {config['aero']['Cl']}")
        lines.append(f"Frontal Area: {config['aero']['A']} m²")
        tyre_model = config['tyre'].get('model', 'simple')
        if tyre_model == 'pacejka':
            pac = config['tyre'].get('pacejka', {})
            mu_peak = pac.get('mu_peak', config['tyre'].get('mu', 1.6))
            lines.append(f"Tyre Model: Pacejka (μ_peak: {mu_peak})")
        else:
            lines.append(f"Tyre Friction (μ): {config['tyre']['mu']}")

        # Powertrain info - handle both new and legacy formats
        pt = config["powertrain"]
        if "drivetrain" in pt:
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
            lines.append(f"Drivetrain: {pt['drivetrain']} ({n_motors} motors)")
            lines.append(f"Motor Power: {pt['motor_power_kW']} kW (per motor)")
            lines.append(f"Motor Torque: {pt['motor_torque_Nm']} Nm (per motor)")
            lines.append(f"Motor Max RPM: {pt['motor_rpm_max']}")
            motor_eff = pt.get("motor_efficiency", 0.85)
            lines.append(f"Motor Efficiency: {motor_eff * 100:.0f}%")
            lines.append(f"Gear Ratio: {pt['gear_ratio']}:1")
            lines.append(f"Wheel Radius: {pt['wheel_radius_m']} m")
            inv_weight = pt.get("inverter_weight_kg", 6.9)
            lines.append(
                f"Inverter: {pt.get('inverter_name', 'DTI HV-850')} "
                f"({n_motors}x, {inv_weight:.1f} kg each)"
            )
            lines.append(f"Total Power: {total_power:.0f} kW (FS cap: {rules_cap:.0f} kW)")
            lines.append(f"Max Tractive Force: {fx_max:.0f} N")
            lines.append(
                f"Max Speed (RPM limit): {v_max:.1f} m/s ({v_max * 3.6:.0f} km/h)"
            )
        else:
            lines.append(f"Max Power: {pt['P_max_kW']} kW")
            lines.append(f"Max Tractive Force: {pt['Fx_max_N']} N")

        if config["battery"].get("enabled", True):
            lines.append(f"Battery Capacity: {config['battery']['capacity_kWh']} kWh")
            lines.append(f"Initial SoC: {config['battery']['initial_soc'] * 100:.0f}%")
            lines.append(f"Min SoC: {config['battery']['min_soc'] * 100:.0f}%")
            regen_enabled = config["battery"].get("regen_enabled", False)
            lines.append(
                f"Regenerative Braking: {'ENABLED' if regen_enabled else 'DISABLED'}"
            )
            if regen_enabled:
                lines.append(
                    f"  Regen Efficiency: {config['battery'].get('eta_regen', 0.85) * 100:.0f}%"
                )
                lines.append(
                    f"  Max Regen Power: {config['battery'].get('max_regen_kW', 50)} kW"
                )
                lines.append(
                    f"  Braking Capture: {config['battery'].get('regen_capture_percent', 100):.0f}%"
                )
        lines.append("")

        # Autocross results
        if "autocross" in self.results:
            ac = self.results["autocross"]
            lines.append("AUTOCROSS RESULTS")
            lines.append("-" * 50)
            lines.append(f"Lap Time: {ac['lap_time']:.3f} s")
            lines.append(
                f"Average Speed: {ac['avg_speed']:.2f} m/s ({ac['avg_speed'] * 3.6:.1f} km/h)"
            )
            lines.append(
                f"Maximum Speed: {ac['max_speed']:.2f} m/s ({ac['max_speed'] * 3.6:.1f} km/h)"
            )
            lines.append(
                f"Minimum Speed: {ac['min_speed']:.2f} m/s ({ac['min_speed'] * 3.6:.1f} km/h)"
            )
            lines.append(f"Max Longitudinal Accel: {ac['max_ax']:.2f} m/s²")
            lines.append(f"Max Braking Decel: {ac['min_ax']:.2f} m/s²")
            lines.append(f"Max Lateral Accel: {ac['max_ay']:.2f} m/s²")

            if ac.get("track"):
                lines.append(f"Track Length: {ac['track'].s[-1]:.1f} m")

            if ac.get("battery_validation"):
                bv = ac["battery_validation"]
                lines.append("")
                lines.append("Battery Analysis:")
                lines.append(
                    f"  Status: {'SUFFICIENT' if bv.sufficient else 'INSUFFICIENT'}"
                )
                lines.append(f"  Final SoC: {bv.final_soc * 100:.1f}%")
                lines.append(f"  Minimum SoC: {bv.min_soc * 100:.1f}%")
                lines.append(f"  Total Energy: {bv.total_energy_kWh:.3f} kWh")
                lines.append(f"  Peak Power: {bv.peak_power_kW:.1f} kW")
                lines.append(f"  Average Power: {bv.avg_power_kW:.1f} kW")

            lines.append("")

        # Skidpad results
        if "skidpad" in self.results:
            sp = self.results["skidpad"]
            lines.append("SKIDPAD RESULTS")
            lines.append("-" * 50)
            lines.append(f"Official Time (2 laps avg): {sp['t_official']:.3f} s")
            lines.append(f"Full Run Time: {sp['t_full_run']:.3f} s")
            lines.append(f"Single Circle Lap Time: {sp['lap_time']:.3f} s")
            lines.append(
                f"Cornering Speed: {sp['avg_speed']:.2f} m/s ({sp['avg_speed'] * 3.6:.1f} km/h)"
            )
            lines.append(f"Lateral Acceleration: {sp['max_ay']:.2f} m/s²")

            if sp.get("battery_validation"):
                bv = sp["battery_validation"]
                lines.append("")
                lines.append("Battery Analysis (single circle):")
                lines.append(
                    f"  Energy per circle: {bv.total_energy_kWh * 1000:.1f} Wh"
                )
                lines.append(f"  Average Power: {bv.avg_power_kW:.2f} kW")

            lines.append("")

        lines.append("=" * 70)
        lines.append("End of Report")
        lines.append("=" * 70)

        return "\n".join(lines)


def main():
    """Main entry point for the GUI."""
    root = tk.Tk()

    # Set window icon using the dedicated icon file
    try:
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "resources", "BAROx Icon.png"
        )
        if os.path.exists(icon_path):
            # Load icon image
            icon_image = Image.open(icon_path)
            # Resize to standard icon size
            icon_size = 64
            icon_image.thumbnail((icon_size, icon_size), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_image)
            root.iconphoto(True, icon_photo)
            # Keep reference to prevent garbage collection
            root._icon_photo = icon_photo
    except Exception:
        pass

    app = BAROxGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
