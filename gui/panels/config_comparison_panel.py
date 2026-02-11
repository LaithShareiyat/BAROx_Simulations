"""Config Comparison Panel for comparing multiple vehicle configurations.

Workflow:
1. User sets up a vehicle config via the control panel (left side).
2. User clicks "Add Config" — the full config dict is captured and named.
3. User modifies the control panel for a different config and adds again.
4. "Run Comparison" simulates all saved configs on autocross and shows results.
"""

import copy
import tkinter as tk
from tkinter import ttk, simpledialog
import threading
from typing import Callable, List
import numpy as np


def _summarise_config(config: dict) -> dict:
    """Extract key display values from a full config dict."""
    v = config.get("vehicle", {})
    a = config.get("aero", {})
    t = config.get("tyre", {})
    pt = config.get("powertrain", {})

    # Drivetrain description
    drivetrain = pt.get("drivetrain", "—")

    # Motor name from the config (if present)
    motor = pt.get("motor", "—")

    tyre_model = t.get("model", "simple")
    mu_display = t.get("mu", "—")
    if tyre_model == "pacejka":
        pac = t.get("pacejka", {})
        mu_display = pac.get("mu_peak", mu_display)

    return {
        "mass_kg": v.get("mass_kg", "—"),
        "Cl": a.get("Cl", "—"),
        "Cd": a.get("Cd", "—"),
        "mu": mu_display,
        "tyre_model": tyre_model.capitalize(),
        "gear_ratio": pt.get("gear_ratio", "—"),
        "drivetrain": drivetrain,
        "motor": motor,
    }


class ConfigComparisonPanel(ttk.Frame):
    """Panel for comparing multiple vehicle configurations on autocross."""

    def __init__(self, parent, get_base_config: Callable, **kwargs):
        """
        Initialise the config comparison panel.

        Args:
            parent: Parent widget
            get_base_config: Callback returning current full config dict
                             from the control panel
        """
        super().__init__(parent, **kwargs)
        self.get_base_config = get_base_config
        self.running = False

        # Saved configs: list of (name, full_config_dict)
        self.saved_configs: List[tuple] = []

        self._create_widgets()

    # ------------------------------------------------------------------
    # Widget creation
    # ------------------------------------------------------------------

    def _create_widgets(self):
        """Build the full panel layout."""
        # Top: saved configs list
        self._create_config_list()

        # Middle: buttons + progress
        self._create_controls()

        # Bottom: results (bar chart + table)
        self._create_results_area()

    def _create_config_list(self):
        """Create the Treeview showing saved configurations."""
        list_frame = ttk.LabelFrame(
            self, text="SAVED CONFIGURATIONS", padding=(8, 4),
        )
        list_frame.pack(fill="x", padx=8, pady=(8, 4))

        columns = ("name", "mass", "Cl", "Cd", "mu", "tyre", "gr", "drivetrain", "motor")
        self.config_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=5,
        )

        self.config_tree.heading("name", text="Name")
        self.config_tree.heading("mass", text="Mass (kg)")
        self.config_tree.heading("Cl", text="Cl")
        self.config_tree.heading("Cd", text="Cd")
        self.config_tree.heading("mu", text="mu")
        self.config_tree.heading("tyre", text="Tyre")
        self.config_tree.heading("gr", text="GR")
        self.config_tree.heading("drivetrain", text="Drive")
        self.config_tree.heading("motor", text="Motor")

        self.config_tree.column("name", width=100, anchor="w")
        self.config_tree.column("mass", width=60, anchor="center")
        self.config_tree.column("Cl", width=40, anchor="center")
        self.config_tree.column("Cd", width=40, anchor="center")
        self.config_tree.column("mu", width=40, anchor="center")
        self.config_tree.column("tyre", width=60, anchor="center")
        self.config_tree.column("gr", width=40, anchor="center")
        self.config_tree.column("drivetrain", width=50, anchor="center")
        self.config_tree.column("motor", width=90, anchor="w")

        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.config_tree.yview,
        )
        self.config_tree.configure(yscrollcommand=scrollbar.set)
        self.config_tree.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Instruction label
        ttk.Label(
            self,
            text="Set up a config in the control panel, then click Add Config to save it.",
            style="Unit.TLabel",
        ).pack(padx=8, anchor="w")

    def _create_controls(self):
        """Create buttons, run button, progress bar, status."""
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=4)

        ttk.Button(
            btn_frame, text="Add Config", command=self._add_config,
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            btn_frame, text="Remove Selected", command=self._remove_selected,
        ).pack(side="left", padx=4)
        ttk.Button(
            btn_frame, text="Clear All", command=self._clear_all_configs,
        ).pack(side="left", padx=4)

        # Note
        ttk.Label(
            btn_frame, text="(Autocross only)", style="Unit.TLabel",
        ).pack(side="right", padx=4)

        # Run button
        run_frame = ttk.Frame(self)
        run_frame.pack(fill="x", padx=8, pady=(4, 2))

        self.run_button = ttk.Button(
            run_frame, text="RUN COMPARISON",
            command=self._run_comparison, style="Accent.TButton",
        )
        self.run_button.pack(fill="x", ipady=6)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            run_frame, variable=self.progress_var, maximum=100,
        )
        self.progress_bar.pack(fill="x", pady=(4, 0))

        # Status
        self.status_var = tk.StringVar(value="Ready — add configs to compare")
        ttk.Label(
            run_frame, textvariable=self.status_var, style="Unit.TLabel",
        ).pack(pady=(2, 0))

    def _create_results_area(self):
        """Create the bar chart and results table."""
        from ..widgets.plot_canvas import PlotCanvas

        results_frame = ttk.Frame(self)
        results_frame.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        # Bar chart
        self.chart_canvas = PlotCanvas(results_frame, figsize=(10, 3.5), dpi=100)
        self.chart_canvas.pack(fill="both", expand=True, pady=(0, 4))

        # Results table (Treeview)
        table_lf = ttk.LabelFrame(results_frame, text="RESULTS", padding=(4, 2))
        table_lf.pack(fill="x", pady=(0, 4))

        columns = ("rank", "name", "lap_time", "delta", "pct_diff")
        self.results_tree = ttk.Treeview(
            table_lf, columns=columns, show="headings", height=6,
        )

        self.results_tree.heading("rank", text="#")
        self.results_tree.heading("name", text="Config")
        self.results_tree.heading("lap_time", text="Lap Time (s)")
        self.results_tree.heading("delta", text="Delta (s)")
        self.results_tree.heading("pct_diff", text="% Diff")

        self.results_tree.column("rank", width=30, anchor="center")
        self.results_tree.column("name", width=120, anchor="w")
        self.results_tree.column("lap_time", width=90, anchor="center")
        self.results_tree.column("delta", width=80, anchor="center")
        self.results_tree.column("pct_diff", width=70, anchor="center")

        scrollbar = ttk.Scrollbar(
            table_lf, orient="vertical", command=self.results_tree.yview,
        )
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        self.results_tree.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    def _add_config(self):
        """Capture the current control panel config and save it."""
        config = copy.deepcopy(self.get_base_config())

        # Ask for a name
        default_name = f"Config {len(self.saved_configs) + 1}"
        name = simpledialog.askstring(
            "Config Name",
            "Enter a name for this configuration:",
            initialvalue=default_name,
            parent=self,
        )
        if not name:
            return  # User cancelled

        self.saved_configs.append((name, config))
        self._insert_config_row(name, config)
        self.status_var.set(f"{len(self.saved_configs)} config(s) saved")

    def _insert_config_row(self, name: str, config: dict):
        """Insert a summary row into the config Treeview."""
        s = _summarise_config(config)
        self.config_tree.insert("", "end", values=(
            name,
            f"{s['mass_kg']:.1f}" if isinstance(s["mass_kg"], (int, float)) else s["mass_kg"],
            f"{s['Cl']}" if isinstance(s["Cl"], (int, float)) else s["Cl"],
            f"{s['Cd']}" if isinstance(s["Cd"], (int, float)) else s["Cd"],
            f"{s['mu']}" if isinstance(s["mu"], (int, float)) else s["mu"],
            s.get("tyre_model", "Simple"),
            f"{s['gear_ratio']}" if isinstance(s["gear_ratio"], (int, float)) else s["gear_ratio"],
            s["drivetrain"],
            s["motor"],
        ))

    def _remove_selected(self):
        """Remove the selected config from the list."""
        selection = self.config_tree.selection()
        if not selection:
            return

        # Find index of the selected item
        all_items = self.config_tree.get_children()
        for item in selection:
            idx = list(all_items).index(item)
            if 0 <= idx < len(self.saved_configs):
                self.saved_configs.pop(idx)
            self.config_tree.delete(item)

        self.status_var.set(f"{len(self.saved_configs)} config(s) saved")

    def _clear_all_configs(self):
        """Remove all saved configs."""
        self.saved_configs.clear()
        for item in self.config_tree.get_children():
            self.config_tree.delete(item)
        self.status_var.set("Ready — add configs to compare")

    # ------------------------------------------------------------------
    # Vehicle construction (mirrors app.py:_create_vehicle)
    # ------------------------------------------------------------------

    @staticmethod
    def _create_vehicle(config: dict):
        """Create VehicleParams from a config dict.

        Same logic as BAROxGUI._create_vehicle() — duplicated here to
        avoid circular imports (same pattern as BatteryOptimiserPanel).
        """
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

        pt_config = config["powertrain"]
        if "drivetrain" in pt_config:
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
            powertrain = EVPowertrainMVP(
                P_max=pt_config["P_max_kW"] * 1000,
                Fx_max=pt_config["Fx_max_N"],
            )

        battery = None
        if config.get("battery", {}).get("enabled", True):
            bat = config["battery"]
            battery = BatteryParams(
                capacity_kWh=bat["capacity_kWh"],
                initial_soc=bat["initial_soc"],
                min_soc=bat["min_soc"],
                max_discharge_kW=bat["max_discharge_kW"],
                eta_discharge=bat.get("eta_discharge", 0.95),
                nominal_voltage_V=bat.get("nominal_voltage_V", 511.0),
                max_current_A=bat.get("max_current_A", 175.0),
                regen_enabled=bat.get("regen_enabled", False),
                eta_regen=bat.get("eta_regen", 0.85),
                max_regen_kW=bat.get("max_regen_kW", 50.0),
                regen_capture_percent=bat.get("regen_capture_percent", 100.0),
            )

        geometry = None
        if "geometry" in config:
            geo = config["geometry"]
            geometry = VehicleGeometry(
                wheelbase_m=geo.get("wheelbase_m", 1.55),
                L_f_m=geo.get("L_f_m", 0.75),
                L_r_m=geo.get("L_r_m", 0.80),
                track_front_m=geo.get("track_front_m", 1.20),
                track_rear_m=geo.get("track_rear_m", 1.20),
                h_cg_m=geo.get("h_cg_m", 0.28),
            )

        torque_vectoring = None
        if "torque_vectoring" in config:
            tv = config["torque_vectoring"]
            torque_vectoring = TorqueVectoringParams(
                enabled=tv.get("enabled", False),
                effectiveness=tv.get("effectiveness", 1.0),
                max_torque_transfer=tv.get("max_torque_transfer", 0.5),
                strategy=tv.get("strategy", "load_proportional"),
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

    # ------------------------------------------------------------------
    # Run comparison
    # ------------------------------------------------------------------

    def _run_comparison(self):
        """Start the comparison in a background thread."""
        if self.running:
            return

        if len(self.saved_configs) < 2:
            self.status_var.set("Add at least 2 configs to compare")
            return

        self.running = True
        self.run_button.configure(state="disabled", text="RUNNING...")
        self.progress_var.set(0)
        self.status_var.set("Starting comparison...")

        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        thread = threading.Thread(target=self._comparison_thread, daemon=True)
        thread.start()

    def _comparison_thread(self):
        """Execute comparisons in a background thread."""
        try:
            from solver.qss_speed import solve_qss
            from events.autocross_generator import build_standard_autocross

            track, _ = build_standard_autocross()

            total = len(self.saved_configs)
            results = []

            for i, (name, cfg) in enumerate(self.saved_configs):
                progress = ((i + 1) / total) * 100
                status = f"Simulating '{name}' ({i + 1}/{total})..."
                self.winfo_toplevel().after(
                    0, lambda p=progress, s=status: self._update_progress(p, s),
                )

                try:
                    vehicle = self._create_vehicle(cfg)
                    _result, t_lap = solve_qss(track, vehicle)
                    results.append({
                        "name": name,
                        "lap_time": t_lap,
                        "mass_kg": cfg["vehicle"]["mass_kg"],
                    })
                except Exception as e:
                    results.append({
                        "name": name,
                        "lap_time": float("nan"),
                        "mass_kg": cfg["vehicle"]["mass_kg"],
                        "error": str(e),
                    })

            # Sort by lap time (NaN at end)
            results.sort(
                key=lambda r: r["lap_time"] if not np.isnan(r["lap_time"]) else 1e9,
            )

            self.winfo_toplevel().after(0, lambda: self._display_results(results))

        except Exception as e:
            import traceback
            msg = f"Comparison error: {e}\n{traceback.format_exc()}"
            self.winfo_toplevel().after(0, lambda: self._show_error(msg))

        finally:
            self.winfo_toplevel().after(0, self._comparison_complete)

    def _update_progress(self, progress: float, status: str):
        self.progress_var.set(progress)
        self.status_var.set(status)

    def _comparison_complete(self):
        self.running = False
        self.run_button.configure(state="normal", text="RUN COMPARISON")
        self.progress_var.set(100)

    def _show_error(self, message: str):
        self.status_var.set("Error occurred")
        fig = self.chart_canvas.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(
            0.5, 0.5, message[:200], ha="center", va="center",
            fontsize=10, wrap=True, transform=ax.transAxes,
        )
        self.chart_canvas.draw()

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------

    def _get_plot_colours(self) -> dict:
        """Get matplotlib colours matching the current ttk theme."""
        bg_hex = ttk.Style().lookup("TFrame", "background") or "#dcdad5"
        try:
            r = int(bg_hex[1:3], 16)
            g = int(bg_hex[3:5], 16)
            b = int(bg_hex[5:7], 16)
            is_dark = (r + g + b) / 3 < 128
        except (ValueError, IndexError):
            is_dark = False

        if is_dark:
            return {"bg": "#1c1c1c", "text": "#e0e0e0", "grid_alpha": 0.2, "edge": "white"}
        return {"bg": "white", "text": "black", "grid_alpha": 0.3, "edge": "black"}

    def _style_axes(self, ax, colours: dict):
        """Apply theme colours to a matplotlib axes."""
        ax.set_facecolor(colours["bg"])
        ax.tick_params(colors=colours["text"])
        for spine in ax.spines.values():
            spine.set_color(colours["text"])
        ax.xaxis.label.set_color(colours["text"])
        ax.yaxis.label.set_color(colours["text"])
        ax.title.set_color(colours["text"])

    def _display_results(self, results: list):
        """Render bar chart and populate results table."""
        valid = [r for r in results if not np.isnan(r["lap_time"])]
        failed = [r for r in results if np.isnan(r["lap_time"])]

        if not valid:
            self.status_var.set("All simulations failed")
            return

        best_time = valid[0]["lap_time"]

        # --- Bar chart ---
        colours = self._get_plot_colours()
        fig = self.chart_canvas.get_figure()
        fig.clear()
        fig.patch.set_facecolor(colours["bg"])

        ax = fig.add_subplot(111)
        self._style_axes(ax, colours)

        names = [r["name"] for r in valid]
        times = [r["lap_time"] for r in valid]

        y_pos = np.arange(len(names))
        bar_colours = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(times))]

        bars = ax.barh(
            y_pos, times, color=bar_colours,
            edgecolor=colours["edge"], linewidth=0.5, height=0.6,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # Fastest at top
        ax.set_xlabel("Lap Time [s]")
        ax.set_title("Config Comparison — Autocross Lap Time")

        # Annotate bars with lap time values
        for bar, t in zip(bars, times):
            ax.text(
                bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{t:.3f}s", va="center", ha="left",
                fontsize=9, color=colours["text"],
            )

        # x-axis limits with padding for annotations
        if times:
            ax.set_xlim(min(times) * 0.98, max(times) * 1.04)

        ax.grid(True, axis="x", alpha=colours["grid_alpha"])
        fig.tight_layout()
        self.chart_canvas.draw()

        # --- Results table ---
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        for rank, r in enumerate(valid, start=1):
            delta = r["lap_time"] - best_time
            pct = (delta / best_time) * 100 if best_time > 0 else 0.0

            delta_str = f"+{delta:.3f}" if delta > 0 else "0.000"
            pct_str = f"+{pct:.2f}%" if pct > 0 else "0.00%"

            self.results_tree.insert("", "end", values=(
                rank, r["name"], f"{r['lap_time']:.3f}", delta_str, pct_str,
            ))

        for r in failed:
            err = r.get("error", "Unknown error")
            self.results_tree.insert("", "end", values=(
                "—", r["name"], "FAILED", "—", err[:30],
            ))

        self.status_var.set(
            f"Complete — {len(valid)}/{len(results)} configs, "
            f"best: {valid[0]['name']} ({best_time:.3f}s)"
        )
