"""Right results panel with text output and embedded plots."""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any, Callable
import numpy as np
from matplotlib.figure import Figure

# Track width constant (same as in events modules)
TRACK_WIDTH = 3.0


def calculate_track_boundaries(
    track_x: np.ndarray, track_y: np.ndarray, track_width: float = TRACK_WIDTH
):
    """
    Calculate left and right track boundaries from centre line.

    Args:
        track_x: X coordinates of track centre line
        track_y: Y coordinates of track centre line
        track_width: Total track width in meters

    Returns:
        tuple: (x_left, y_left, x_right, y_right) boundary coordinates
    """
    dx = np.gradient(track_x)
    dy = np.gradient(track_y)
    length = np.sqrt(dx**2 + dy**2)
    length = np.maximum(length, 1e-9)  # Avoid division by zero

    # Normal vectors (perpendicular to track direction)
    nx = -dy / length
    ny = dx / length

    offset = track_width / 2
    x_left = track_x + offset * nx
    y_left = track_y + offset * ny
    x_right = track_x - offset * nx
    y_right = track_y - offset * ny

    return x_left, y_left, x_right, y_right


class ResultsPanel(ttk.Frame):
    """Right panel containing results text and embedded plots."""

    def __init__(self, parent, get_config_callback: Callable = None, **kwargs):
        """Create the results panel with tabbed interface.

        Args:
            parent: Parent widget
            get_config_callback: Optional callback to get current config for battery optimiser
        """
        super().__init__(parent, **kwargs)

        self.get_config_callback = get_config_callback

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Create tabs
        self._create_results_tab()
        self._create_layout_tab()
        self._create_speed_track_tab()
        self._create_speed_distance_tab()
        self._create_accel_distance_tab()
        self._create_rpm_distance_tab()
        self._create_power_demand_tab()
        self._create_gear_ratio_sweep_tab()
        self._create_battery_tab()
        self._create_battery_optimiser_tab()
        self._create_config_comparison_tab()

        # Status bar at bottom
        self._create_status_bar()

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------

    def _get_plot_colours(self) -> dict:
        """Get matplotlib colours matching the current ttk theme."""
        # Detect dark mode from ttk background colour (no sv_ttk dependency)
        bg_hex = ttk.Style().lookup("TFrame", "background") or "#dcdad5"
        # Dark themes have low luminance backgrounds
        try:
            r = int(bg_hex[1:3], 16)
            g = int(bg_hex[3:5], 16)
            b = int(bg_hex[5:7], 16)
            is_dark = (r + g + b) / 3 < 128
        except (ValueError, IndexError):
            is_dark = False

        if is_dark:
            return {
                "bg": "#1c1c1c",
                "text": "#e0e0e0",
                "grid_alpha": 0.2,
                "edge": "white",
            }
        return {
            "bg": "white",
            "text": "black",
            "grid_alpha": 0.3,
            "edge": "black",
        }

    def _style_axes(self, ax, colours: dict = None):
        """Apply theme colours to a matplotlib axes."""
        if colours is None:
            colours = self._get_plot_colours()
        ax.set_facecolor(colours["bg"])
        ax.tick_params(colors=colours["text"])
        for spine in ax.spines.values():
            spine.set_color(colours["text"])
        ax.xaxis.label.set_color(colours["text"])
        ax.yaxis.label.set_color(colours["text"])
        ax.title.set_color(colours["text"])

    def _style_figure(self, fig, colours: dict = None):
        """Apply theme background to a matplotlib figure."""
        if colours is None:
            colours = self._get_plot_colours()
        fig.patch.set_facecolor(colours["bg"])

    def update_theme_colours(self):
        """Update colours that don't automatically follow the ttk theme."""
        bg = ttk.Style().lookup("TFrame", "background") or "white"
        self._results_canvas.configure(bg=bg)

    def _create_results_tab(self):
        """Create the card-based results tab with scrollable canvas."""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")

        # Scrollable canvas (same pattern as control_panel.py)
        # Use ttk theme background so the canvas doesn't show as black
        bg = ttk.Style().lookup("TFrame", "background") or "white"
        self._results_canvas = tk.Canvas(
            self.results_frame, highlightthickness=0, bg=bg
        )
        self._results_scrollbar = ttk.Scrollbar(
            self.results_frame,
            orient="vertical",
            command=self._results_canvas.yview,
        )
        self._results_inner = ttk.Frame(self._results_canvas)

        self._results_inner.bind(
            "<Configure>",
            lambda e: self._results_canvas.configure(
                scrollregion=self._results_canvas.bbox("all")
            ),
        )

        self._results_canvas_window = self._results_canvas.create_window(
            (0, 0), window=self._results_inner, anchor="nw"
        )
        self._results_canvas.configure(
            yscrollcommand=self._results_scrollbar.set
        )

        # Keep inner frame width in sync with canvas width
        self._results_canvas.bind(
            "<Configure>",
            lambda e: self._results_canvas.itemconfigure(
                self._results_canvas_window, width=e.width
            ),
        )

        # Mousewheel scrolling
        self._results_canvas.bind(
            "<Enter>",
            lambda e: self._results_canvas.bind_all(
                "<MouseWheel>", self._on_results_mousewheel
            ),
        )
        self._results_canvas.bind(
            "<Leave>",
            lambda e: self._results_canvas.unbind_all("<MouseWheel>"),
        )

        self._results_scrollbar.pack(side="right", fill="y")
        self._results_canvas.pack(side="left", fill="both", expand=True)

    def _on_results_mousewheel(self, event):
        """Handle mousewheel scrolling on the results canvas."""
        if event.num == 4:
            self._results_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._results_canvas.yview_scroll(1, "units")
        else:
            self._results_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units"
            )

    def _create_layout_tab(self):
        """Create the track layout tab (geometry only, no speed colouring)."""
        self.layout_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.layout_frame, text="Track Layout")

        from ..widgets.plot_canvas import PlotCanvas

        self.layout_canvas = PlotCanvas(self.layout_frame, figsize=(12, 6))
        self.layout_canvas.pack(fill="both", expand=True)

    def _create_speed_track_tab(self):
        """Create the speed track map tab (velocity coloured)."""
        self.track_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.track_frame, text="Speed Track Map")

        from ..widgets.plot_canvas import PlotCanvas

        self.track_canvas = PlotCanvas(self.track_frame, figsize=(12, 6))
        self.track_canvas.pack(fill="both", expand=True)

    def _create_speed_distance_tab(self):
        """Create the speed vs distance tab."""
        from ..widgets.plot_canvas import PlotCanvas

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Speed vs Distance")
        self.speed_distance_canvas = PlotCanvas(frame, figsize=(10, 6))
        self.speed_distance_canvas.pack(fill="both", expand=True)

    def _create_accel_distance_tab(self):
        """Create the acceleration vs distance tab."""
        from ..widgets.plot_canvas import PlotCanvas

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Accel vs Distance")
        self.accel_distance_canvas = PlotCanvas(frame, figsize=(10, 6))
        self.accel_distance_canvas.pack(fill="both", expand=True)

    def _create_rpm_distance_tab(self):
        """Create the motor RPM vs distance tab."""
        from ..widgets.plot_canvas import PlotCanvas

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="RPM vs Distance")
        self.rpm_distance_canvas = PlotCanvas(frame, figsize=(10, 6))
        self.rpm_distance_canvas.pack(fill="both", expand=True)

    def _create_power_demand_tab(self):
        """Create the power demand vs distance tab."""
        from ..widgets.plot_canvas import PlotCanvas

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Power Demand")
        self.power_demand_canvas = PlotCanvas(frame, figsize=(10, 6))
        self.power_demand_canvas.pack(fill="both", expand=True)

    def _create_gear_ratio_sweep_tab(self):
        """Create the lap time vs gear ratio sweep tab."""
        from ..widgets.plot_canvas import PlotCanvas

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Gear Ratio Sweep")
        self.gear_ratio_sweep_canvas = PlotCanvas(frame, figsize=(10, 6))
        self.gear_ratio_sweep_canvas.pack(fill="both", expand=True)

    def _create_battery_tab(self):
        """Create the battery analysis tab."""
        self.battery_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.battery_frame, text="Battery")

        from ..widgets.plot_canvas import PlotCanvas

        self.battery_canvas = PlotCanvas(self.battery_frame, figsize=(10, 8))
        self.battery_canvas.pack(fill="both", expand=True)

    def _create_battery_optimiser_tab(self):
        """Create the battery pack optimiser tab."""
        self.optimiser_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.optimiser_frame, text="Pack Optimiser")

        from .battery_optimiser_panel import BatteryOptimiserPanel

        # Create optimiser panel with config callback
        self.battery_optimiser = BatteryOptimiserPanel(
            self.optimiser_frame,
            get_base_config=self.get_config_callback or self._get_default_config,
        )
        self.battery_optimiser.pack(fill="both", expand=True)

    def _create_config_comparison_tab(self):
        """Create the config comparison tab."""
        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="Config Comparison")

        from .config_comparison_panel import ConfigComparisonPanel

        self.config_comparison = ConfigComparisonPanel(
            self.comparison_frame,
            get_base_config=self.get_config_callback or self._get_default_config,
        )
        self.config_comparison.pack(fill="both", expand=True)

    def _get_default_config(self) -> dict:
        """Fallback default config if no callback provided."""
        return {
            "vehicle": {
                "mass_kg": 239,
                "g": 9.81,
                "Crr": 0.015,
                "mass_battery_kg": 45,  # 142S5P pack
            },
            "aero": {"rho": 1.225, "Cd": 1.1, "Cl": 1.5, "A": 1.0},
            "tyre": {"mu": 1.6},
            "powertrain": {
                "drivetrain": "2RWD",
                "motor_power_kW": 40,  # FS rules: 80 kW total (2 × 40 kW)
                "motor_torque_Nm": 150,  # EMRAX 208
                "motor_rpm_max": 7000,  # EMRAX 208
                "motor_efficiency": 0.96,  # EMRAX 208
                "gear_ratio": 3.0,
                "wheel_radius_m": 0.203,
                "powertrain_overhead_kg": 10.0,
                "inverter_peak_power_kW": 320.0,
                "inverter_peak_current_A": 600.0,
                "inverter_weight_kg": 6.9,
            },
        }

    def _create_status_bar(self):
        """Create status bar at bottom."""
        self.status_frame = ttk.Frame(self)
        self.status_frame.pack(fill="x", side="bottom", pady=(5, 0))

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            self.status_frame, textvariable=self.status_var, anchor="w", padding=(10, 5)
        )
        self.status_label.pack(fill="x")

    # ------------------------------------------------------------------
    # Card helper methods
    # ------------------------------------------------------------------

    def _clear_results_inner(self):
        """Destroy all children of the results inner frame."""
        for child in self._results_inner.winfo_children():
            child.destroy()

    def _build_data_table(self, parent, title, headers, rows):
        """Build a labelled grid table inside *parent*.

        Args:
            parent: Parent widget
            title: Section title string (or None to skip)
            headers: list of column header strings
            rows: list of tuples, one per row, matching len(headers)
        """
        if title:
            ttk.Label(
                parent, text=title, style="SectionTitle.TLabel"
            ).pack(anchor="w", padx=8, pady=(8, 2))

        table = ttk.Frame(parent)
        table.pack(fill="x", padx=12, pady=(0, 4))

        for col, hdr in enumerate(headers):
            lbl = ttk.Label(table, text=hdr, style="TableHeader.TLabel")
            lbl.grid(row=0, column=col, sticky="w", padx=(0, 16), pady=2)

        for r, row_data in enumerate(rows, start=1):
            for col, cell in enumerate(row_data):
                lbl = ttk.Label(
                    table, text=str(cell), style="TableCell.TLabel"
                )
                lbl.grid(row=r, column=col, sticky="w", padx=(0, 16), pady=1)

        # Separator
        ttk.Separator(parent, orient="horizontal").pack(
            fill="x", padx=8, pady=4
        )

    def _build_kv_section(self, parent, title, pairs):
        """Build a key-value section inside *parent*.

        Args:
            parent: Parent widget
            title: Section title (or None)
            pairs: list of (label, value) or (label, value, unit)
        """
        if title:
            ttk.Label(
                parent, text=title, style="SectionTitle.TLabel"
            ).pack(anchor="w", padx=8, pady=(8, 2))

        grid = ttk.Frame(parent)
        grid.pack(fill="x", padx=12, pady=(0, 4))

        for r, item in enumerate(pairs):
            label_text = item[0]
            value_text = str(item[1])
            unit_text = item[2] if len(item) > 2 else ""

            ttk.Label(
                grid, text=label_text, style="DataLabel.TLabel"
            ).grid(row=r, column=0, sticky="w", padx=(0, 12), pady=1)
            ttk.Label(
                grid, text=value_text, style="DataValue.TLabel"
            ).grid(row=r, column=1, sticky="e", padx=(0, 6), pady=1)
            if unit_text:
                ttk.Label(
                    grid, text=unit_text, style="DataLabel.TLabel"
                ).grid(row=r, column=2, sticky="w", pady=1)

        ttk.Separator(parent, orient="horizontal").pack(
            fill="x", padx=8, pady=4
        )

    # ------------------------------------------------------------------
    # Headline banner
    # ------------------------------------------------------------------

    def _build_headline_row(self, results):
        """Build the top headline row with large numbers."""
        row = ttk.Frame(self._results_inner)
        row.pack(fill="x", padx=8, pady=(8, 4))

        cards = []

        if "autocross" in results:
            cards.append(
                ("Autocross", f"{results['autocross']['lap_time']:.3f}", "s")
            )

        if "skidpad" in results:
            cards.append(
                ("Skidpad", f"{results['skidpad']['t_official']:.3f}", "s")
            )

        if "autocross" in results and "battery_sufficient" in results["autocross"]:
            is_pass = results["autocross"]["battery_sufficient"]
            cards.append(
                ("Battery", "PASS" if is_pass else "FAIL", "pass" if is_pass else "fail")
            )

        for i, (label, value, unit_or_tag) in enumerate(cards):
            card = ttk.LabelFrame(row, text=label, padding=8)
            card.grid(row=0, column=i, sticky="nsew", padx=4)
            row.grid_columnconfigure(i, weight=1)

            if unit_or_tag in ("pass", "fail"):
                style = "Pass.TLabel" if unit_or_tag == "pass" else "Fail.TLabel"
                ttk.Label(card, text=value, style=style).pack(anchor="center")
            else:
                val_frame = ttk.Frame(card)
                val_frame.pack(anchor="center")
                ttk.Label(
                    val_frame, text=value, style="Headline.TLabel"
                ).pack(side="left")
                ttk.Label(
                    val_frame, text=f" {unit_or_tag}", style="HeadlineUnit.TLabel"
                ).pack(side="left", anchor="s", pady=(0, 6))

    # ------------------------------------------------------------------
    # Card builders
    # ------------------------------------------------------------------

    def _build_autocross_card(self, metrics):
        """Build the Autocross Results card."""
        card = ttk.LabelFrame(
            self._results_inner, text="Autocross Results", padding=6
        )
        card.pack(fill="x", padx=8, pady=4)

        # Speed table
        self._build_data_table(
            card,
            "Speed",
            ["Parameter", "m/s", "km/h"],
            [
                ("Average", f"{metrics['avg_speed']:.2f}", f"{metrics['avg_speed'] * 3.6:.1f}"),
                ("Maximum", f"{metrics['max_speed']:.2f}", f"{metrics['max_speed'] * 3.6:.1f}"),
                ("Minimum", f"{metrics['min_speed']:.2f}", f"{metrics['min_speed'] * 3.6:.1f}"),
            ],
        )

        # Max speed analysis
        msa = metrics.get("max_speed_analysis")
        if msa:
            pairs = [
                ("Reached At", f"{msa['peak_distance_m']:.1f} m  /  {msa['peak_time_s']:.2f} s"),
            ]
            if msa.get("motor_rpm") is not None:
                pairs.append(
                    ("Motor RPM at Max", f"{msa['motor_rpm']:.0f}  /  {msa['motor_rpm_max']:.0f} max")
                )
            pairs.append(("Times at Max Speed", str(msa["n_times"])))
            pairs.append(("Limiter", msa["limiter"]))
            self._build_kv_section(card, "Max Speed Analysis", pairs)

        # RPM profile table
        rpm_profile = metrics.get("motor_rpm_profile")
        if rpm_profile is not None:
            rpm_limit = metrics.get("motor_rpm_limit", 1)
            rpm_avg = np.nanmean(rpm_profile)
            rpm_max = np.nanmax(rpm_profile)
            rpm_min_arr = rpm_profile[rpm_profile > 1]
            rpm_min = np.min(rpm_min_arr) if len(rpm_min_arr) > 0 else 0.0

            rows = [
                ("Average", f"{rpm_avg:.0f}", f"{rpm_avg / rpm_limit * 100:.1f}%"),
                ("Maximum", f"{rpm_max:.0f}", f"{rpm_max / rpm_limit * 100:.1f}%"),
                ("Minimum", f"{rpm_min:.0f}", f"{rpm_min / rpm_limit * 100:.1f}%"),
                ("RPM Limit", f"{rpm_limit:.0f}", ""),
            ]
            self._build_data_table(
                card, "RPM Profile", ["Parameter", "RPM", "% of Limit"], rows
            )

        # Acceleration table
        self._build_data_table(
            card,
            "Acceleration",
            ["Parameter", "m/s\u00b2", "g"],
            [
                ("Max Longitudinal", f"{metrics['max_ax']:.2f}", f"{metrics['max_ax'] / 9.81:.2f}"),
                ("Max Braking", f"{abs(metrics['min_ax']):.2f}", f"{abs(metrics['min_ax']) / 9.81:.2f}"),
                ("Max Lateral", f"{metrics['max_ay']:.2f}", f"{metrics['max_ay'] / 9.81:.2f}"),
            ],
        )

        # Energy consumed
        self._build_kv_section(
            card,
            None,
            [("Energy Consumed", f"{metrics['energy_consumed_kWh'] * 1000:.1f}", "Wh")],
        )

    def _build_skidpad_card(self, metrics):
        """Build the Skidpad Results card."""
        card = ttk.LabelFrame(
            self._results_inner, text="Skidpad Results", padding=6
        )
        card.pack(fill="x", padx=8, pady=4)

        # Timing
        self._build_kv_section(
            card,
            "Timing",
            [
                ("Official Time (2-lap avg)", f"{metrics['t_official']:.3f}", "s"),
                ("Single Circle", f"{metrics['lap_time']:.3f}", "s"),
                ("Full Run (4 circles)", f"{metrics['t_full_run']:.3f}", "s"),
            ],
        )

        # Performance
        avg_speed = metrics.get("avg_speed", 0)
        if avg_speed == 0 and "t_official" in metrics:
            from events.skidpad import SKIDPAD_CENTRE_RADIUS
            circumference = 2 * np.pi * SKIDPAD_CENTRE_RADIUS
            avg_speed = circumference / metrics["t_official"]

        self._build_kv_section(
            card,
            "Performance",
            [
                ("Cornering Speed", f"{avg_speed:.2f} m/s  ({avg_speed * 3.6:.1f} km/h)"),
                ("Lateral Acceleration", f"{metrics['max_ay']:.2f} m/s\u00b2  ({metrics['max_ay'] / 9.81:.2f} g)"),
                ("Energy Consumed", f"{metrics['energy_consumed_kWh'] * 1000:.1f}", "Wh"),
            ],
        )

    def _build_battery_card(self, metrics):
        """Build the Battery Analysis card."""
        bv = metrics.get("battery_validation")
        if not bv:
            return

        card = ttk.LabelFrame(
            self._results_inner, text="Battery Analysis", padding=6
        )
        card.pack(fill="x", padx=8, pady=4)

        # Status indicator
        status_text = "SUFFICIENT" if bv.sufficient else "INSUFFICIENT"
        style = "Pass.TLabel" if bv.sufficient else "Fail.TLabel"
        status_frame = ttk.Frame(card)
        status_frame.pack(fill="x", padx=8, pady=(4, 2))
        ttk.Label(status_frame, text="Status:", style="DataLabel.TLabel").pack(side="left")
        ttk.Label(status_frame, text=f"  {status_text}", style=style).pack(side="left")

        ttk.Separator(card, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # State of Charge
        self._build_kv_section(
            card,
            "State of Charge",
            [
                ("Final SoC", f"{bv.final_soc * 100:.1f}", "%"),
                ("Minimum SoC", f"{bv.min_soc * 100:.1f} %  (at {bv.min_soc_distance:.0f} m)"),
            ],
        )

        # Energy & Power
        self._build_kv_section(
            card,
            "Energy & Power",
            [
                ("Energy Consumed", f"{bv.total_energy_kWh * 1000:.1f}", "Wh"),
                ("Peak Power", f"{bv.peak_power_kW:.1f}", "kW"),
                ("Average Power", f"{bv.avg_power_kW:.1f}", "kW"),
            ],
        )

        # Warnings / errors
        if bv.warnings or bv.errors:
            warn_frame = ttk.Frame(card)
            warn_frame.pack(fill="x", padx=8, pady=(0, 4))
            for warn in bv.warnings:
                ttk.Label(
                    warn_frame, text=f"\u26a0  {warn}", style="Warning.TLabel"
                ).pack(anchor="w")
            for err in bv.errors:
                ttk.Label(
                    warn_frame, text=f"\u2716  {err}", style="Fail.TLabel"
                ).pack(anchor="w")

    def _build_summary_card(self, results):
        """Build the Summary card at the bottom."""
        card = ttk.LabelFrame(
            self._results_inner, text="Summary", padding=6
        )
        card.pack(fill="x", padx=8, pady=4)

        pairs = []
        if "autocross" in results:
            pairs.append(
                ("Autocross Lap Time", f"{results['autocross']['lap_time']:.3f}", "s")
            )
        if "skidpad" in results:
            pairs.append(
                ("Skidpad Official Time", f"{results['skidpad']['t_official']:.3f}", "s")
            )
        if "autocross" in results and "battery_sufficient" in results["autocross"]:
            is_pass = results["autocross"]["battery_sufficient"]
            pairs.append(
                ("Battery Status", "PASS" if is_pass else "FAIL")
            )

        self._build_kv_section(card, None, pairs)

        # Simulation complete label
        ttk.Label(
            card,
            text="Simulation Complete",
            style="CardTitle.TLabel",
            anchor="center",
        ).pack(fill="x", pady=(4, 2))

    # ------------------------------------------------------------------
    # Public API — display / clear
    # ------------------------------------------------------------------

    def clear_all(self):
        """Clear all results and plots."""
        self._clear_results_inner()

        self.layout_canvas.clear()
        self.track_canvas.clear()
        self.speed_distance_canvas.clear()
        self.rpm_distance_canvas.clear()
        self.power_demand_canvas.clear()
        self.gear_ratio_sweep_canvas.clear()
        self.battery_canvas.clear()

    def set_status(self, message: str, status_type: str = "normal"):
        """Update status bar message."""
        self.status_var.set(message)

    def set_results_text(self, text: str):
        """Show a plain-text message (e.g. error) in a card."""
        self._clear_results_inner()
        card = ttk.LabelFrame(
            self._results_inner, text="Output", padding=10
        )
        card.pack(fill="x", padx=8, pady=8)
        ttk.Label(
            card,
            text=text,
            style="DataLabel.TLabel",
            wraplength=600,
            justify="left",
        ).pack(anchor="w")

    def display_results(self, results: Dict[str, Any], event_type: str):
        """Display simulation results as structured cards."""
        self._clear_results_inner()

        # Headline banner
        self._build_headline_row(results)

        # Event cards
        if "autocross" in results:
            self._build_autocross_card(results["autocross"])

        if "skidpad" in results:
            self._build_skidpad_card(results["skidpad"])

        if "autocross" in results and "battery_validation" in results["autocross"]:
            self._build_battery_card(results["autocross"])

        # Summary
        self._build_summary_card(results)

        self.notebook.select(0)  # Switch to results tab

    def update_layout_plot(
        self, autocross_data: dict = None, skidpad_data: dict = None
    ):
        """Update the track layout plot (geometry only, no speed colouring)."""
        fig = self.layout_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)

        # Determine layout based on available data
        has_autocross = autocross_data is not None
        has_skidpad = skidpad_data is not None

        if has_autocross and has_skidpad:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        elif has_autocross:
            ax1 = fig.add_subplot(111)
            ax2 = None
        elif has_skidpad:
            ax1 = None
            ax2 = fig.add_subplot(111)
        else:
            return

        # Apply theme to axes
        if ax1 is not None:
            self._style_axes(ax1, colours)
        if ax2 is not None:
            self._style_axes(ax2, colours)

        # Plot autocross layout
        if has_autocross and ax1 is not None:
            track = autocross_data["track"]
            metadata = autocross_data.get("metadata")

            # Calculate and plot track boundaries
            x_left, y_left, x_right, y_right = calculate_track_boundaries(
                track.x, track.y
            )
            ax1.plot(x_left, y_left, "k-", lw=1, alpha=0.5)
            ax1.plot(x_right, y_right, "k-", lw=1, alpha=0.5)

            # Shade track area
            ax1.fill(
                np.concatenate([x_left, x_right[::-1]]),
                np.concatenate([y_left, y_right[::-1]]),
                color="gray",
                alpha=0.15,
            )

            # Plot centre line
            ax1.plot(track.x, track.y, "b-", lw=2, label="Centre line", zorder=3)

            # Mark start and end
            ax1.plot(
                track.x[0], track.y[0], "go", markersize=12, label="Start", zorder=10
            )
            ax1.plot(
                track.x[-1], track.y[-1], "rs", markersize=12, label="End", zorder=10
            )

            if metadata and "slalom_cones_x" in metadata:
                ax1.scatter(
                    metadata["slalom_cones_x"],
                    metadata["slalom_cones_y"],
                    c="orange",
                    marker="^",
                    s=80,
                    label="Slalom cones",
                    zorder=15,
                    edgecolors=colours["edge"],
                    linewidths=0.5,
                )

            ax1.set_aspect("equal")
            ax1.set_xlabel("x [m]")
            ax1.set_ylabel("y [m]")
            ax1.set_title("Autocross Track Layout")
            ax1.legend(loc="lower right", fontsize=8, labelcolor=colours["text"])
            ax1.grid(True, alpha=colours["grid_alpha"])

        # Plot skidpad layout (using same approach as plot_skidpad_full)
        if has_skidpad and ax2 is not None:
            from events.skidpad import (
                SKIDPAD_CENTRE_RADIUS,
                SKIDPAD_INNER_RADIUS,
                SKIDPAD_OUTER_RADIUS,
            )

            # Circle centres
            left_centre = (0.0, 0.0)
            right_centre = (2 * SKIDPAD_CENTRE_RADIUS, 0.0)

            # Lane boundaries
            lane_left_x = SKIDPAD_INNER_RADIUS
            lane_right_x = SKIDPAD_OUTER_RADIUS
            lane_centre_x = SKIDPAD_CENTRE_RADIUS

            # Lane extends beyond circles for entry/exit
            lane_extension = 3.0
            lane_bottom_y = -SKIDPAD_OUTER_RADIUS - lane_extension
            lane_top_y = SKIDPAD_OUTER_RADIUS + lane_extension

            theta = np.linspace(0, 2 * np.pi, 100)

            # Shade circle track areas (annulus between inner and outer radius)
            for centre in [left_centre, right_centre]:
                x_outer = centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta)
                y_outer = centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta)
                x_inner = centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta[::-1])
                y_inner = centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta[::-1])
                ax2.fill(
                    np.concatenate([x_outer, x_inner]),
                    np.concatenate([y_outer, y_inner]),
                    color="gray",
                    alpha=0.2,
                )

            # Shade the continuous lane (from bottom to top through middle)
            ax2.fill(
                [lane_left_x, lane_right_x, lane_right_x, lane_left_x],
                [lane_bottom_y, lane_bottom_y, lane_top_y, lane_top_y],
                color="gray",
                alpha=0.2,
            )

            # Left circle boundaries and centre line
            ax2.plot(
                left_centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta),
                left_centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta),
                "k-",
                lw=2,
                label="Track boundaries",
            )
            ax2.plot(
                left_centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta),
                left_centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta),
                "k-",
                lw=2,
            )
            ax2.plot(
                left_centre[0] + SKIDPAD_CENTRE_RADIUS * np.cos(theta),
                left_centre[1] + SKIDPAD_CENTRE_RADIUS * np.sin(theta),
                "b--",
                lw=1.5,
                alpha=0.7,
                label="Centre line",
            )

            # Right circle boundaries and centre line
            ax2.plot(
                right_centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta),
                right_centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta),
                "k-",
                lw=2,
            )
            ax2.plot(
                right_centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta),
                right_centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta),
                "k-",
                lw=2,
            )
            ax2.plot(
                right_centre[0] + SKIDPAD_CENTRE_RADIUS * np.cos(theta),
                right_centre[1] + SKIDPAD_CENTRE_RADIUS * np.sin(theta),
                "b--",
                lw=1.5,
                alpha=0.7,
            )

            # Continuous lane boundaries through the middle
            ax2.plot(
                [lane_left_x, lane_left_x], [lane_bottom_y, lane_top_y], "k-", lw=2
            )
            ax2.plot(
                [lane_right_x, lane_right_x], [lane_bottom_y, lane_top_y], "k-", lw=2
            )
            ax2.plot(
                [lane_centre_x, lane_centre_x],
                [lane_bottom_y, lane_top_y],
                "b--",
                lw=1.5,
                alpha=0.7,
            )

            # Mark entry and exit
            ax2.plot(
                lane_centre_x,
                lane_bottom_y,
                "g^",
                markersize=12,
                label="Entry",
                zorder=10,
            )
            ax2.plot(
                lane_centre_x, lane_top_y, "rs", markersize=12, label="Exit", zorder=10
            )

            ax2.set_aspect("equal")
            ax2.set_xlabel("x [m]")
            ax2.set_ylabel("y [m]")
            ax2.set_title("Skidpad Track Layout")
            ax2.legend(loc="upper right", fontsize=8, labelcolor=colours["text"])
            ax2.grid(True, alpha=colours["grid_alpha"])

        fig.tight_layout()
        self.layout_canvas.draw()

    def update_speed_track_plot(
        self, autocross_data: dict = None, skidpad_data: dict = None
    ):
        """Update the speed track map with velocity colouring for both events."""
        fig = self.track_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)

        # Determine layout based on available data
        has_autocross = autocross_data is not None
        has_skidpad = skidpad_data is not None

        if has_autocross and has_skidpad:
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
        elif has_autocross:
            ax1 = fig.add_subplot(111)
            ax2 = None
        elif has_skidpad:
            ax1 = None
            ax2 = fig.add_subplot(111)
        else:
            return

        # Apply theme to axes
        if ax1 is not None:
            self._style_axes(ax1, colours)
        if ax2 is not None:
            self._style_axes(ax2, colours)

        # Plot autocross with speed colouring
        if has_autocross and ax1 is not None:
            track = autocross_data["track"]
            v = autocross_data["v"]
            metadata = autocross_data.get("metadata")

            # Calculate and plot track boundaries
            x_left, y_left, x_right, y_right = calculate_track_boundaries(
                track.x, track.y
            )
            ax1.plot(x_left, y_left, "k-", lw=1, alpha=0.5)
            ax1.plot(x_right, y_right, "k-", lw=1, alpha=0.5)

            # Shade track area
            ax1.fill(
                np.concatenate([x_left, x_right[::-1]]),
                np.concatenate([y_left, y_right[::-1]]),
                color="gray",
                alpha=0.15,
            )

            # Speed-coloured scatter plot
            sc = ax1.scatter(track.x, track.y, c=v, cmap="RdYlGn", s=8, zorder=5)
            fig.colorbar(sc, ax=ax1, label="Speed [m/s]", shrink=0.8)

            ax1.plot(
                track.x[0], track.y[0], "go", markersize=12, label="Start", zorder=10
            )
            ax1.plot(
                track.x[-1], track.y[-1], "rs", markersize=12, label="End", zorder=10
            )

            if metadata and "slalom_cones_x" in metadata:
                ax1.scatter(
                    metadata["slalom_cones_x"],
                    metadata["slalom_cones_y"],
                    c="orange",
                    marker="^",
                    s=60,
                    label="Slaloms",
                    zorder=15,
                    edgecolors=colours["edge"],
                    linewidths=0.5,
                )

            ax1.set_aspect("equal")
            ax1.set_xlabel("x [m]")
            ax1.set_ylabel("y [m]")
            ax1.set_title("Autocross - Speed Map")
            ax1.legend(loc="lower right", fontsize=8, labelcolor=colours["text"])
            ax1.grid(True, alpha=colours["grid_alpha"])

        # Plot skidpad with speed colouring
        if has_skidpad and ax2 is not None:
            from matplotlib.ticker import ScalarFormatter

            track = skidpad_data["track"]
            v = skidpad_data["v"]

            # Calculate and plot track boundaries
            x_left, y_left, x_right, y_right = calculate_track_boundaries(
                track.x, track.y
            )
            ax2.plot(x_left, y_left, "k-", lw=1, alpha=0.5)
            ax2.plot(x_right, y_right, "k-", lw=1, alpha=0.5)

            # For skidpad (constant speed), set explicit colour limits to avoid
            # scientific notation in colourbar
            v_mean = np.mean(v)
            v_range = np.max(v) - np.min(v)
            if v_range < 0.1:  # Nearly constant speed
                # Set colour limits with a small margin around the constant speed
                vmin = v_mean - 0.5
                vmax = v_mean + 0.5
                sc = ax2.scatter(
                    track.x,
                    track.y,
                    c=v,
                    cmap="RdYlGn",
                    s=10,
                    zorder=5,
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                sc = ax2.scatter(track.x, track.y, c=v, cmap="RdYlGn", s=10, zorder=5)

            cbar = fig.colorbar(sc, ax=ax2, label="Speed [m/s]", shrink=0.8)
            # Disable scientific notation on colourbar
            cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            cbar.ax.ticklabel_format(style="plain", axis="y")

            ax2.plot(
                track.x[0], track.y[0], "go", markersize=12, label="Start", zorder=10
            )
            ax2.plot(
                track.x[-1], track.y[-1], "rs", markersize=12, label="End", zorder=10
            )

            ax2.set_aspect("equal")
            ax2.set_xlabel("x [m]")
            ax2.set_ylabel("y [m]")
            # Show the actual speed value in title for constant speed
            if v_range < 0.1:
                ax2.set_title(f"Skidpad - Steady State: {v_mean:.2f} m/s")
            else:
                ax2.set_title("Skidpad - Speed Map")
            ax2.legend(loc="upper right", fontsize=8, labelcolor=colours["text"])
            ax2.grid(True, alpha=colours["grid_alpha"])

        fig.tight_layout()
        self.track_canvas.draw()

    def update_track_plot(
        self,
        track,
        v: np.ndarray,
        metadata: dict = None,
        title: str = "Speed Track Map",
    ):
        """Update the track plot with velocity colouring (legacy single-track method)."""
        fig = self.track_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)
        ax = fig.add_subplot(111)
        self._style_axes(ax, colours)

        # Calculate and plot track boundaries
        x_left, y_left, x_right, y_right = calculate_track_boundaries(track.x, track.y)
        ax.plot(x_left, y_left, "k-", lw=1, alpha=0.5)
        ax.plot(x_right, y_right, "k-", lw=1, alpha=0.5)

        # Shade track area
        ax.fill(
            np.concatenate([x_left, x_right[::-1]]),
            np.concatenate([y_left, y_right[::-1]]),
            color="gray",
            alpha=0.15,
        )

        # Scatter plot coloured by speed
        sc = ax.scatter(track.x, track.y, c=v, cmap="RdYlGn", s=10, zorder=5)
        fig.colorbar(sc, ax=ax, label="Speed [m/s]", shrink=0.8)

        # Mark start and end
        ax.plot(track.x[0], track.y[0], "go", markersize=15, label="Start", zorder=10)
        ax.plot(track.x[-1], track.y[-1], "rs", markersize=15, label="End", zorder=10)

        # Plot slalom cones if metadata available
        if metadata and "slalom_cones_x" in metadata:
            ax.scatter(
                metadata["slalom_cones_x"],
                metadata["slalom_cones_y"],
                c="orange",
                marker="^",
                s=80,
                label="Slalom cones",
                zorder=15,
                edgecolors=colours["edge"],
                linewidths=0.5,
            )

        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.legend(loc="lower right", labelcolor=colours["text"])
        ax.grid(True, alpha=colours["grid_alpha"])

        fig.tight_layout()
        self.track_canvas.draw()

    # ------------------------------------------------------------------
    # Speed vs Distance
    # ------------------------------------------------------------------

    def update_speed_distance_plot(
        self, autocross_data: dict = None, skidpad_data: dict = None
    ):
        """Update the speed vs distance plot."""
        fig = self.speed_distance_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)

        has_autocross = autocross_data is not None
        has_skidpad = skidpad_data is not None
        if not has_autocross and not has_skidpad:
            return

        axes = self._make_event_subplots(fig, has_autocross, has_skidpad)

        if has_autocross and axes[0] is not None:
            self._style_axes(axes[0], colours)
            self._plot_speed_distance(axes[0], autocross_data, "Autocross", colours)

        if has_skidpad and axes[1] is not None:
            self._style_axes(axes[1], colours)
            self._plot_speed_distance(axes[1], skidpad_data, "Skidpad", colours)

        fig.tight_layout()
        self.speed_distance_canvas.draw()

    def _plot_speed_distance(self, ax, data: dict, event_name: str, colours: dict = None):
        """Plot speed vs distance on a single axes."""
        from matplotlib.ticker import ScalarFormatter

        if colours is None:
            colours = self._get_plot_colours()

        track = data["track"]
        v = data["v"]
        avg_speed = np.nanmean(v)

        ax.plot(track.s, v, "b-", lw=1.5, label="Speed")
        ax.axhline(
            avg_speed, color="grey", ls="--", lw=1,
            label=f"Average: {avg_speed:.1f} m/s",
        )
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Speed [m/s]")
        ax.set_title(f"{event_name} — Speed vs Distance")
        ax.legend(loc="upper right", fontsize=8, labelcolor=colours["text"])
        ax.grid(True, alpha=colours["grid_alpha"])

        # Skidpad steady-state formatting
        v_range = np.max(v) - np.min(v)
        if v_range < 0.1:
            margin = max(0.5, avg_speed * 0.05)
            ax.set_ylim(avg_speed - margin, avg_speed + margin)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="y")

    # ------------------------------------------------------------------
    # Acceleration vs Distance
    # ------------------------------------------------------------------

    def update_accel_distance_plot(
        self, autocross_data: dict = None, skidpad_data: dict = None
    ):
        """Update the acceleration vs distance plot."""
        fig = self.accel_distance_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)

        has_autocross = autocross_data is not None
        has_skidpad = skidpad_data is not None
        if not has_autocross and not has_skidpad:
            return

        axes = self._make_event_subplots(fig, has_autocross, has_skidpad)

        if has_autocross and axes[0] is not None:
            self._style_axes(axes[0], colours)
            self._plot_accel_distance(axes[0], autocross_data, "Autocross", colours)

        if has_skidpad and axes[1] is not None:
            self._style_axes(axes[1], colours)
            self._plot_accel_distance(axes[1], skidpad_data, "Skidpad", colours)

        fig.tight_layout()
        self.accel_distance_canvas.draw()

    def _plot_accel_distance(self, ax, data: dict, event_name: str, colours: dict = None):
        """Plot longitudinal and lateral acceleration vs distance."""
        if colours is None:
            colours = self._get_plot_colours()

        track = data["track"]
        v = data["v"]
        g = 9.81

        # Compute acceleration channels
        # Lateral: a_y = v² × κ
        ay = v**2 * np.abs(track.kappa)

        # Longitudinal: a_x = (v²[i+1] - v²[i]) / (2 × ds)
        ax_vals = np.zeros(len(v) - 1)
        for i in range(len(v) - 1):
            ds = track.ds[i]
            if ds > 0:
                ax_vals[i] = (v[i + 1]**2 - v[i]**2) / (2 * ds)

        s_seg = track.s[:-1]  # Segment start points

        # Plot longitudinal acceleration
        ax.fill_between(
            s_seg, 0, ax_vals / g,
            where=(ax_vals >= 0), color="#2ecc71", alpha=0.4, label="Driving"
        )
        ax.fill_between(
            s_seg, 0, ax_vals / g,
            where=(ax_vals < 0), color="#e74c3c", alpha=0.4, label="Braking"
        )
        ax.plot(s_seg, ax_vals / g, color="#333333" if colours["bg"] == "white" else "#cccccc",
                lw=0.5, alpha=0.6)

        # Plot lateral acceleration
        ax.plot(track.s, ay / g, color="#3498db", lw=1.2, alpha=0.8, label="Lateral")

        ax.axhline(0, color=colours["text"], lw=0.5, alpha=0.3)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Acceleration [g]")
        ax.set_title(f"{event_name} — Acceleration vs Distance")
        ax.legend(loc="upper right", fontsize=8, labelcolor=colours["text"])
        ax.grid(True, alpha=colours["grid_alpha"])

    # ------------------------------------------------------------------
    # Motor RPM vs Distance
    # ------------------------------------------------------------------

    def update_rpm_distance_plot(
        self, autocross_data: dict = None, skidpad_data: dict = None
    ):
        """Update the motor RPM vs distance plot."""
        fig = self.rpm_distance_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)

        has_autocross = autocross_data is not None
        has_skidpad = skidpad_data is not None
        if not has_autocross and not has_skidpad:
            return

        axes = self._make_event_subplots(fig, has_autocross, has_skidpad)

        if has_autocross and axes[0] is not None:
            self._style_axes(axes[0], colours)
            self._plot_rpm_distance(axes[0], autocross_data, "Autocross", colours)

        if has_skidpad and axes[1] is not None:
            self._style_axes(axes[1], colours)
            self._plot_rpm_distance(axes[1], skidpad_data, "Skidpad", colours)

        fig.tight_layout()
        self.rpm_distance_canvas.draw()

    def _plot_rpm_distance(self, ax, data: dict, event_name: str, colours: dict = None):
        """Plot motor RPM vs distance on a single axes."""
        if colours is None:
            colours = self._get_plot_colours()

        rpm_profile = data.get("motor_rpm_profile")
        if rpm_profile is None:
            ax.text(
                0.5, 0.5,
                "RPM data not available\n(Extended powertrain required)",
                ha="center", va="center", fontsize=12, color="grey",
                transform=ax.transAxes,
            )
            ax.set_title(f"{event_name} — Motor RPM vs Distance")
            return

        track = data["track"]
        rpm_limit = data.get("motor_rpm_limit", 0)
        avg_rpm = np.nanmean(rpm_profile)

        ax.plot(track.s, rpm_profile, color="darkorange", lw=1.5, label="Motor RPM")
        ax.axhline(
            avg_rpm, color="grey", ls="--", lw=1,
            label=f"Average: {avg_rpm:.0f} RPM",
        )
        if rpm_limit > 0:
            ax.axhline(
                rpm_limit, color="red", ls=":", lw=1,
                label=f"Limit: {rpm_limit:.0f} RPM",
            )
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Motor RPM")
        ax.set_title(f"{event_name} — Motor RPM vs Distance")
        ax.legend(loc="upper right", fontsize=8, labelcolor=colours["text"])
        ax.grid(True, alpha=colours["grid_alpha"])

    # ------------------------------------------------------------------
    # Power Demand vs Distance
    # ------------------------------------------------------------------

    def update_power_demand_plot(
        self, autocross_data: dict = None, skidpad_data: dict = None
    ):
        """Update the mechanical power demand vs distance plot."""
        fig = self.power_demand_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)

        has_autocross = autocross_data is not None
        has_skidpad = skidpad_data is not None
        if not has_autocross and not has_skidpad:
            return

        axes = self._make_event_subplots(fig, has_autocross, has_skidpad)

        if has_autocross and axes[0] is not None:
            self._style_axes(axes[0], colours)
            self._plot_power_demand(axes[0], autocross_data, "Autocross", colours)

        if has_skidpad and axes[1] is not None:
            self._style_axes(axes[1], colours)
            self._plot_power_demand(axes[1], skidpad_data, "Skidpad", colours)

        fig.tight_layout()
        self.power_demand_canvas.draw()

    def _plot_power_demand(self, ax, data: dict, event_name: str, colours: dict = None):
        """Plot mechanical power demand vs distance on a single axes."""
        if colours is None:
            colours = self._get_plot_colours()

        vehicle = data.get("vehicle")
        if vehicle is None:
            ax.text(
                0.5, 0.5, "Vehicle data not available",
                ha="center", va="center", fontsize=12, color="grey",
                transform=ax.transAxes,
            )
            return

        track = data["track"]
        v = data["v"]
        power_kW = self._compute_mechanical_power(track, v, vehicle)
        s_seg = track.s[:-1]

        # Filled regions: positive (driving) and negative (braking)
        ax.fill_between(
            s_seg, power_kW, 0,
            where=(power_kW >= 0), color="salmon", alpha=0.4, label="Driving",
        )
        ax.fill_between(
            s_seg, power_kW, 0,
            where=(power_kW < 0), color="lightskyblue", alpha=0.4, label="Braking",
        )
        ax.plot(s_seg, power_kW, color="darkred", lw=0.8)

        # Powertrain power limit reference line
        if hasattr(vehicle.powertrain, "P_max"):
            p_max_kW = vehicle.powertrain.P_max / 1000.0
            ax.axhline(
                p_max_kW, color="red", ls=":", lw=1,
                label=f"P_max: {p_max_kW:.0f} kW",
            )

        # Average of positive (driving) power only
        positive_power = power_kW[power_kW > 0]
        if len(positive_power) > 0:
            avg_power = np.mean(positive_power)
            ax.axhline(
                avg_power, color="grey", ls="--", lw=1,
                label=f"Avg driving: {avg_power:.1f} kW",
            )

        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Mechanical Power [kW]")
        ax.set_title(f"{event_name} — Power Demand vs Distance")
        ax.legend(loc="upper right", fontsize=8, labelcolor=colours["text"])
        ax.grid(True, alpha=colours["grid_alpha"])

    @staticmethod
    def _compute_mechanical_power(track, v, vehicle):
        """Compute actual motor power output [kW] per segment.

        Driving power is capped to the motor's capability at each speed
        (torque limit, power limit, RPM limit) using the same powertrain
        model the solver uses.  Braking power is shown as-is (negative).
        """
        from physics.aero import drag
        from physics.resistive import rolling_resistance
        from physics.powertrain import max_tractive_force_extended

        n_segments = len(track.ds)
        power_kW = np.zeros(n_segments)
        m = vehicle.m

        for i in range(n_segments):
            v_i = max(v[i], 0.1)
            ds = track.ds[i]

            # Kinematic acceleration (same equation the solver uses)
            a_x = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds) if ds > 0 else 0.0

            # Forces at v[i] (same evaluation point as the solver)
            F_drag = drag(vehicle.aero.rho, vehicle.aero.CD_A, v_i)
            F_rr = rolling_resistance(vehicle.Crr, vehicle.m, vehicle.g)
            F_required = m * a_x + F_drag + F_rr

            if F_required > 0:
                # Driving — cap to actual motor capability
                F_motor_max = max_tractive_force_extended(
                    vehicle.powertrain, v_i
                )
                F_actual = min(F_required, F_motor_max)
                power_kW[i] = (F_actual * v_i) / 1000.0
            else:
                # Braking — show full braking power (negative)
                power_kW[i] = (F_required * v_i) / 1000.0

        # Hard clamp driving power to P_max (catches float rounding)
        p_max_kW = vehicle.powertrain.P_max / 1000.0
        power_kW = np.clip(power_kW, None, p_max_kW)

        return power_kW

    # ------------------------------------------------------------------
    # Sensitivity sweep plots
    # ------------------------------------------------------------------

    def update_gear_ratio_sweep_plot(self, sweep_data: dict = None):
        """Update the lap time vs gear ratio sweep plot."""
        fig = self.gear_ratio_sweep_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)
        ax = fig.add_subplot(111)
        self._style_axes(ax, colours)

        self._plot_sweep_on_ax(
            ax, sweep_data,
            xlabel="Gear Ratio [-]",
            title="Autocross Lap Time vs Gear Ratio",
            value_fmt=".2f",
            colours=colours,
        )

        fig.tight_layout()
        self.gear_ratio_sweep_canvas.draw()

    @staticmethod
    def _plot_sweep_on_ax(ax, sweep_data, xlabel, title, value_fmt, colours=None):
        """Plot a single sensitivity sweep on the given axes."""
        if colours is None:
            colours = {"bg": "white", "text": "black", "grid_alpha": 0.3, "edge": "black"}

        if sweep_data is None:
            ax.text(
                0.5, 0.5,
                "Sweep data not available\n(Autocross event required)",
                ha="center", va="center", fontsize=12, color="grey",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            return

        values = sweep_data["values"]
        lap_times = sweep_data["lap_times"]
        current_val = sweep_data["current_value"]

        # Filter out NaN entries from failed simulations
        valid = ~np.isnan(lap_times)
        if not np.any(valid):
            ax.text(
                0.5, 0.5, "All sweep simulations failed",
                ha="center", va="center", fontsize=12, color="grey",
                transform=ax.transAxes,
            )
            return

        ax.plot(
            values[valid], lap_times[valid], "b-o",
            lw=1.5, markersize=4, label="Lap time",
        )

        # Mark the current configuration value
        current_lap = np.interp(current_val, values[valid], lap_times[valid])
        ax.axvline(current_val, color="green", ls="--", lw=1, alpha=0.7)
        ax.plot(
            current_val, current_lap, "gs", markersize=10, zorder=10,
            label=f"Current: {current_val:{value_fmt}}",
        )

        # Mark the optimum (minimum lap time)
        best_idx = np.nanargmin(lap_times)
        ax.plot(
            values[best_idx], lap_times[best_idx], "r*", markersize=15, zorder=10,
            label=f"Optimum: {values[best_idx]:{value_fmt}}",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Lap Time [s]")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8, labelcolor=colours["text"])
        ax.grid(True, alpha=colours["grid_alpha"])

    # ------------------------------------------------------------------
    # Subplot layout helper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_event_subplots(fig, has_autocross: bool, has_skidpad: bool):
        """Create 1 or 2 subplots depending on which events are present.

        Returns (ax_autocross, ax_skidpad) — either may be None.
        """
        if has_autocross and has_skidpad:
            return fig.add_subplot(211), fig.add_subplot(212)
        elif has_autocross:
            return fig.add_subplot(111), None
        else:
            return None, fig.add_subplot(111)

    def update_battery_combined_plot(
        self, autocross_data: dict = None, skidpad_data: dict = None
    ):
        """Update the battery analysis plot for both events."""
        fig = self.battery_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)

        has_autocross = autocross_data is not None and "battery_state" in autocross_data
        has_skidpad = skidpad_data is not None and "battery_state" in skidpad_data

        if not has_autocross and not has_skidpad:
            # No battery data available
            ax = fig.add_subplot(111)
            self._style_axes(ax, colours)
            ax.text(
                0.5,
                0.5,
                "No battery data available\n(Battery analysis disabled)",
                ha="center",
                va="center",
                fontsize=12,
                color="gray",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            self.battery_canvas.draw()
            return

        # Determine layout
        if has_autocross and has_skidpad:
            # 3 rows x 2 columns
            ax1 = fig.add_subplot(321)  # Autocross SoC
            ax2 = fig.add_subplot(322)  # Skidpad SoC
            ax3 = fig.add_subplot(323)  # Autocross Power
            ax4 = fig.add_subplot(324)  # Skidpad Power
            ax5 = fig.add_subplot(325)  # Autocross Energy
            ax6 = fig.add_subplot(326)  # Skidpad Energy
        elif has_autocross:
            ax1 = fig.add_subplot(311)
            ax3 = fig.add_subplot(312)
            ax5 = fig.add_subplot(313)
            ax2 = ax4 = ax6 = None
        else:
            ax2 = fig.add_subplot(311)
            ax4 = fig.add_subplot(312)
            ax6 = fig.add_subplot(313)
            ax1 = ax3 = ax5 = None

        # Apply theme to all axes
        for _ax in (ax1, ax2, ax3, ax4, ax5, ax6):
            if _ax is not None:
                self._style_axes(_ax, colours)

        # Plot Autocross battery
        if has_autocross:
            track = autocross_data["track"]
            battery_state = autocross_data["battery_state"]
            vehicle = autocross_data["vehicle"]
            validation = autocross_data.get("battery_validation")

            # SoC
            ax1.plot(track.s, battery_state.soc * 100, "b-", lw=2)
            ax1.axhline(
                y=vehicle.battery.min_soc * 100,
                color="r",
                linestyle="--",
                label=f"Min SoC ({vehicle.battery.min_soc:.0%})",
            )
            ax1.fill_between(
                track.s, 0, vehicle.battery.min_soc * 100, color="red", alpha=0.1
            )
            ax1.set_ylabel("SoC [%]")
            ax1.set_ylim(0, 105)
            ax1.legend(loc="upper right", fontsize=7, labelcolor=colours["text"])
            ax1.grid(True, alpha=colours["grid_alpha"])
            ax1.set_title("Autocross - State of Charge")

            # Power
            ax3.fill_between(track.s, 0, battery_state.power_kW, color="red", alpha=0.5)
            ax3.plot(track.s, battery_state.power_kW, "darkred", lw=1)
            ax3.axhline(
                y=vehicle.battery.max_discharge_kW,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Max ({vehicle.battery.max_discharge_kW:.0f} kW)",
            )
            ax3.set_ylabel("Power [kW]")
            ax3.legend(loc="upper right", fontsize=7, labelcolor=colours["text"])
            ax3.grid(True, alpha=colours["grid_alpha"])
            ax3.set_title("Autocross - Discharge Power")

            # Energy
            ax5.plot(track.s, battery_state.energy_used_kWh * 1000, "r-", lw=2)
            usable = (
                vehicle.battery.capacity_kWh
                * (vehicle.battery.initial_soc - vehicle.battery.min_soc)
                * 1000
            )
            ax5.axhline(
                y=usable,
                color="orange",
                linestyle="--",
                label=f"Usable ({usable:.0f} Wh)",
            )
            ax5.set_xlabel("Distance [m]")
            ax5.set_ylabel("Energy [Wh]")
            ax5.legend(loc="upper left", fontsize=7, labelcolor=colours["text"])
            ax5.grid(True, alpha=colours["grid_alpha"])
            ax5.set_title("Autocross - Cumulative Energy")

        # Plot Skidpad battery
        if has_skidpad:
            from matplotlib.ticker import ScalarFormatter

            track = skidpad_data["track"]
            battery_state = skidpad_data["battery_state"]
            vehicle = skidpad_data["vehicle"]
            validation = skidpad_data.get("battery_validation")

            # SoC
            ax2.plot(track.s, battery_state.soc * 100, "b-", lw=2)
            ax2.axhline(
                y=vehicle.battery.min_soc * 100,
                color="r",
                linestyle="--",
                label=f"Min SoC ({vehicle.battery.min_soc:.0%})",
            )
            ax2.fill_between(
                track.s, 0, vehicle.battery.min_soc * 100, color="red", alpha=0.1
            )
            ax2.set_ylabel("SoC [%]")
            ax2.set_ylim(0, 105)
            ax2.legend(loc="upper right", fontsize=7, labelcolor=colours["text"])
            ax2.grid(True, alpha=colours["grid_alpha"])

            # For skidpad, SoC drop is very small - show it meaningfully
            soc_drop = (battery_state.soc[0] - battery_state.soc[-1]) * 100
            ax2.set_title(f"Skidpad - SoC (Δ = {soc_drop:.3f}%)")

            # Power - for skidpad it's constant
            power_kW = battery_state.power_kW
            ax4.fill_between(track.s, 0, power_kW, color="red", alpha=0.5)
            ax4.plot(track.s, power_kW, "darkred", lw=1)
            ax4.axhline(
                y=vehicle.battery.max_discharge_kW,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Max ({vehicle.battery.max_discharge_kW:.0f} kW)",
            )
            ax4.set_ylabel("Power [kW]")
            ax4.legend(loc="upper right", fontsize=7, labelcolor=colours["text"])
            ax4.grid(True, alpha=colours["grid_alpha"])

            # For constant power, set sensible y-axis and show the value in title
            power_mean = np.mean(power_kW)
            power_range = (
                np.max(power_kW) - np.min(power_kW) if len(power_kW) > 0 else 0
            )
            if power_range < 0.01:  # Constant power (skidpad steady-state)
                ax4.set_ylim(
                    0, max(power_mean * 2, 5)
                )  # Show up to 2x the actual or 5kW
                ax4.set_title(f"Skidpad - Constant Power: {power_mean:.2f} kW")
            else:
                ax4.set_title("Skidpad - Discharge Power")
            ax4.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

            # Energy
            ax6.plot(track.s, battery_state.energy_used_kWh * 1000, "r-", lw=2)
            usable = (
                vehicle.battery.capacity_kWh
                * (vehicle.battery.initial_soc - vehicle.battery.min_soc)
                * 1000
            )
            ax6.axhline(
                y=usable,
                color="orange",
                linestyle="--",
                label=f"Usable ({usable:.0f} Wh)",
            )
            ax6.set_xlabel("Distance [m]")
            ax6.set_ylabel("Energy [Wh]")
            ax6.legend(loc="upper left", fontsize=7, labelcolor=colours["text"])
            ax6.grid(True, alpha=colours["grid_alpha"])

            # Show actual energy used in title
            energy_used_Wh = (
                battery_state.energy_used_kWh[-1] * 1000
                if len(battery_state.energy_used_kWh) > 0
                else 0
            )
            ax6.set_title(f"Skidpad - Energy: {energy_used_Wh:.1f} Wh")

        fig.tight_layout()
        self.battery_canvas.draw()

    def update_battery_plot(self, track, battery_state, vehicle, validation=None):
        """Update the battery analysis plot (legacy single-event method)."""
        fig = self.battery_canvas.get_figure()
        fig.clear()
        colours = self._get_plot_colours()
        self._style_figure(fig, colours)

        # Three subplots: SoC, Power, Energy
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax3 = fig.add_subplot(313, sharex=ax1)
        self._style_axes(ax1, colours)
        self._style_axes(ax2, colours)
        self._style_axes(ax3, colours)

        # SoC plot
        ax1.plot(track.s, battery_state.soc * 100, "b-", lw=2)
        ax1.axhline(
            y=vehicle.battery.min_soc * 100,
            color="r",
            linestyle="--",
            label=f"Min SoC ({vehicle.battery.min_soc:.0%})",
        )
        ax1.fill_between(
            track.s, 0, vehicle.battery.min_soc * 100, color="red", alpha=0.1
        )
        ax1.set_ylabel("State of Charge [%]")
        ax1.set_ylim(0, 105)
        ax1.legend(loc="upper right", labelcolor=colours["text"])
        ax1.grid(True, alpha=colours["grid_alpha"])
        ax1.set_title("Battery State of Charge")

        # Power plot
        ax2.fill_between(track.s, 0, battery_state.power_kW, color="red", alpha=0.5)
        ax2.plot(track.s, battery_state.power_kW, "darkred", lw=1)
        ax2.axhline(
            y=vehicle.battery.max_discharge_kW,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Max ({vehicle.battery.max_discharge_kW:.0f} kW)",
        )
        ax2.set_ylabel("Power [kW]")
        ax2.legend(loc="upper right", labelcolor=colours["text"])
        ax2.grid(True, alpha=colours["grid_alpha"])
        ax2.set_title("Discharge Power")

        # Energy plot
        ax3.plot(track.s, battery_state.energy_used_kWh * 1000, "r-", lw=2)
        usable = (
            vehicle.battery.capacity_kWh
            * (vehicle.battery.initial_soc - vehicle.battery.min_soc)
            * 1000
        )
        ax3.axhline(
            y=usable, color="orange", linestyle="--", label=f"Usable ({usable:.0f} Wh)"
        )
        ax3.set_xlabel("Distance [m]")
        ax3.set_ylabel("Energy [Wh]")
        ax3.legend(loc="upper left", labelcolor=colours["text"])
        ax3.grid(True, alpha=colours["grid_alpha"])
        ax3.set_title("Cumulative Energy")

        # Status text
        if validation:
            status = "SUFFICIENT" if validation.sufficient else "INSUFFICIENT"
            color = "green" if validation.sufficient else "red"
            fig.text(
                0.02,
                0.98,
                f"Status: {status}",
                fontsize=10,
                fontweight="bold",
                color=color,
                verticalalignment="top",
                transform=fig.transFigure,
            )

        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        self.battery_canvas.draw()

    def show_tab(self, tab_name: str):
        """Switch to a specific tab by name."""
        tab_map = {
            "results": 0,
            "layout": 1,
            "track": 2,
            "speed_distance": 3,
            "accel_distance": 4,
            "rpm_distance": 5,
            "power_demand": 6,
            "gear_ratio_sweep": 7,
            "battery": 8,
            "optimiser": 9,
            "comparison": 10,
        }
        if tab_name in tab_map:
            self.notebook.select(tab_map[tab_name])
