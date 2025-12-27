"""Right results panel with text output and embedded plots."""
import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any, Callable
import numpy as np
from matplotlib.figure import Figure

# Track width constant (same as in events modules)
TRACK_WIDTH = 3.0


def calculate_track_boundaries(track_x: np.ndarray, track_y: np.ndarray,
                                track_width: float = TRACK_WIDTH):
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
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Create tabs
        self._create_results_tab()
        self._create_layout_tab()
        self._create_speed_track_tab()
        self._create_speed_tab()
        self._create_battery_tab()
        self._create_battery_optimizer_tab()

        # Status bar at bottom
        self._create_status_bar()

    def _create_results_tab(self):
        """Create the text results tab."""
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")

        # Scrollable text widget
        text_frame = ttk.Frame(self.results_frame)
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.results_text = tk.Text(
            text_frame,
            wrap='none',
            font=('Consolas', 10),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='white',
            state='disabled'
        )

        # Scrollbars
        y_scroll = ttk.Scrollbar(text_frame, orient='vertical', command=self.results_text.yview)
        x_scroll = ttk.Scrollbar(text_frame, orient='horizontal', command=self.results_text.xview)
        self.results_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        # Grid layout for text and scrollbars
        self.results_text.grid(row=0, column=0, sticky='nsew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')

        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        # Configure text tags for colored output
        self.results_text.tag_configure('header', foreground='#569cd6', font=('Consolas', 11, 'bold'))
        self.results_text.tag_configure('subheader', foreground='#4ec9b0')
        self.results_text.tag_configure('success', foreground='#6a9955')
        self.results_text.tag_configure('error', foreground='#f14c4c')
        self.results_text.tag_configure('warning', foreground='#cca700')
        self.results_text.tag_configure('value', foreground='#ce9178')
        self.results_text.tag_configure('label', foreground='#9cdcfe')

    def _create_layout_tab(self):
        """Create the track layout tab (geometry only, no speed coloring)."""
        self.layout_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.layout_frame, text="Track Layout")

        from ..widgets.plot_canvas import PlotCanvas
        self.layout_canvas = PlotCanvas(self.layout_frame, figsize=(12, 6))
        self.layout_canvas.pack(fill='both', expand=True)

    def _create_speed_track_tab(self):
        """Create the speed track map tab (velocity colored)."""
        self.track_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.track_frame, text="Speed Track Map")

        from ..widgets.plot_canvas import PlotCanvas
        self.track_canvas = PlotCanvas(self.track_frame, figsize=(12, 6))
        self.track_canvas.pack(fill='both', expand=True)

    def _create_speed_tab(self):
        """Create the speed profile tab."""
        self.speed_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.speed_frame, text="Speed Profile")

        from ..widgets.plot_canvas import PlotCanvas
        self.speed_canvas = PlotCanvas(self.speed_frame, figsize=(10, 6))
        self.speed_canvas.pack(fill='both', expand=True)

    def _create_battery_tab(self):
        """Create the battery analysis tab."""
        self.battery_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.battery_frame, text="Battery")

        from ..widgets.plot_canvas import PlotCanvas
        self.battery_canvas = PlotCanvas(self.battery_frame, figsize=(10, 8))
        self.battery_canvas.pack(fill='both', expand=True)

    def _create_battery_optimizer_tab(self):
        """Create the battery pack optimiser tab."""
        self.optimizer_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.optimizer_frame, text="Pack Optimiser")

        from .battery_optimizer_panel import BatteryOptimizerPanel

        # Create optimiser panel with config callback
        self.battery_optimizer = BatteryOptimizerPanel(
            self.optimizer_frame,
            get_base_config=self.get_config_callback or self._get_default_config
        )
        self.battery_optimizer.pack(fill='both', expand=True)

    def _get_default_config(self) -> dict:
        """Fallback default config if no callback provided."""
        return {
            'vehicle': {
                'mass_kg': 250, 'g': 9.81, 'Crr': 0.015,
                'mass_battery_kg': 55,
            },
            'aero': {'rho': 1.225, 'Cd': 1.1, 'Cl': 1.5, 'A': 1.0},
            'tyre': {'mu': 1.6},
            'powertrain': {
                'drivetrain': 'RWD',
                'motor_power_kW': 40,  # FS rules: 80 kW total (2 × 40 kW)
                'motor_torque_Nm': 100,
                'motor_rpm_max': 6000,
                'motor_efficiency': 0.85,
                'gear_ratio': 3.5,
                'wheel_radius_m': 0.203,
            },
        }

    def _create_status_bar(self):
        """Create status bar at bottom."""
        self.status_frame = ttk.Frame(self)
        self.status_frame.pack(fill='x', side='bottom', pady=(5, 0))

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            self.status_frame,
            textvariable=self.status_var,
            anchor='w',
            padding=(10, 5)
        )
        self.status_label.pack(fill='x')

    def clear_all(self):
        """Clear all results and plots."""
        self.results_text.configure(state='normal')
        self.results_text.delete('1.0', 'end')
        self.results_text.configure(state='disabled')

        self.layout_canvas.clear()
        self.track_canvas.clear()
        self.speed_canvas.clear()
        self.battery_canvas.clear()

    def set_status(self, message: str, status_type: str = 'normal'):
        """Update status bar message."""
        self.status_var.set(message)

    def append_text(self, text: str, tag: str = None):
        """Append text to results with optional formatting tag."""
        self.results_text.configure(state='normal')
        if tag:
            self.results_text.insert('end', text, tag)
        else:
            self.results_text.insert('end', text)
        self.results_text.see('end')
        self.results_text.configure(state='disabled')

    def set_results_text(self, text: str):
        """Set the entire results text content."""
        self.results_text.configure(state='normal')
        self.results_text.delete('1.0', 'end')
        self.results_text.insert('1.0', text)
        self.results_text.configure(state='disabled')

    def display_results(self, results: Dict[str, Any], event_type: str):
        """Display simulation results in formatted text."""
        self.results_text.configure(state='normal')
        self.results_text.delete('1.0', 'end')

        # Header
        self.append_text("=" * 70 + "\n", 'header')
        self.append_text("                      SIMULATION RESULTS\n", 'header')
        self.append_text("=" * 70 + "\n\n", 'header')

        if 'autocross' in results:
            self._display_autocross_results(results['autocross'])

        if 'skidpad' in results:
            self._display_skidpad_results(results['skidpad'])

        if 'autocross' in results and 'battery_validation' in results['autocross']:
            self._display_battery_results(results['autocross'])

        # Summary
        self._display_summary(results)

        self.results_text.configure(state='disabled')
        self.notebook.select(0)  # Switch to results tab

    def _display_autocross_results(self, metrics: Dict[str, Any]):
        """Display autocross results with aligned columns."""
        self.append_text("-" * 70 + "\n", 'subheader')
        self.append_text("  AUTOCROSS RESULTS\n", 'subheader')
        self.append_text("-" * 70 + "\n\n", 'subheader')

        # Lap time
        self.append_text("  Lap Time:                    ", 'label')
        self.append_text(f"{metrics['lap_time']:.3f} s\n\n", 'value')

        # Speed section
        self.append_text("  SPEED\n", 'label')
        self.append_text("  " + "-" * 50 + "\n")
        self.append_text(f"  {'Parameter':<24} {'m/s':>10} {'km/h':>12}\n")
        self.append_text("  " + "-" * 50 + "\n")
        self.append_text(f"  {'Average':<24} {metrics['avg_speed']:>10.2f} {metrics['avg_speed']*3.6:>12.1f}\n")
        self.append_text(f"  {'Maximum':<24} {metrics['max_speed']:>10.2f} {metrics['max_speed']*3.6:>12.1f}\n")
        self.append_text(f"  {'Minimum':<24} {metrics['min_speed']:>10.2f} {metrics['min_speed']*3.6:>12.1f}\n\n")

        # Acceleration section
        self.append_text("  ACCELERATION\n", 'label')
        self.append_text("  " + "-" * 50 + "\n")
        self.append_text(f"  {'Parameter':<24} {'m/s²':>10} {'g':>12}\n")
        self.append_text("  " + "-" * 50 + "\n")
        self.append_text(f"  {'Max Longitudinal':<24} {metrics['max_ax']:>10.2f} {metrics['max_ax']/9.81:>12.2f}\n")
        self.append_text(f"  {'Max Braking':<24} {abs(metrics['min_ax']):>10.2f} {abs(metrics['min_ax'])/9.81:>12.2f}\n")
        self.append_text(f"  {'Max Lateral':<24} {metrics['max_ay']:>10.2f} {metrics['max_ay']/9.81:>12.2f}\n\n")

        # Energy
        self.append_text("  Energy Consumed:             ", 'label')
        self.append_text(f"{metrics['energy_consumed_kWh']*1000:.1f} Wh\n\n", 'value')

    def _display_skidpad_results(self, metrics: Dict[str, Any]):
        """Display skidpad results with aligned columns."""
        self.append_text("-" * 70 + "\n", 'subheader')
        self.append_text("  SKIDPAD RESULTS\n", 'subheader')
        self.append_text("-" * 70 + "\n\n", 'subheader')

        # Timing section
        self.append_text("  TIMING\n", 'label')
        self.append_text("  " + "-" * 50 + "\n")
        self.append_text(f"  {'Official Time (2-lap avg)':<30}", 'label')
        self.append_text(f"{metrics['t_official']:.3f} s\n", 'value')
        self.append_text(f"  {'Single Circle':<30} {metrics['lap_time']:.3f} s\n")
        self.append_text(f"  {'Full Run (4 circles)':<30} {metrics['t_full_run']:.3f} s\n\n")

        # Performance section
        avg_speed = metrics.get('avg_speed', 0)
        if avg_speed == 0 and 't_official' in metrics:
            from events.skidpad import SKIDPAD_CENTRE_RADIUS
            circumference = 2 * np.pi * SKIDPAD_CENTRE_RADIUS
            avg_speed = circumference / metrics['t_official']

        self.append_text("  PERFORMANCE\n", 'label')
        self.append_text("  " + "-" * 50 + "\n")
        self.append_text(f"  {'Cornering Speed':<24} {avg_speed:>10.2f} m/s  ({avg_speed*3.6:.1f} km/h)\n")
        self.append_text(f"  {'Lateral Acceleration':<24} {metrics['max_ay']:>10.2f} m/s² ({metrics['max_ay']/9.81:.2f} g)\n")
        self.append_text(f"  {'Energy Consumed':<24} {metrics['energy_consumed_kWh']*1000:>10.1f} Wh\n\n")

    def _display_battery_results(self, metrics: Dict[str, Any]):
        """Display battery analysis results with aligned columns."""
        bv = metrics.get('battery_validation')
        if not bv:
            return

        self.append_text("-" * 70 + "\n", 'subheader')
        self.append_text("  BATTERY ANALYSIS\n", 'subheader')
        self.append_text("-" * 70 + "\n\n", 'subheader')

        # Status
        status_tag = 'success' if bv.sufficient else 'error'
        status_text = "SUFFICIENT" if bv.sufficient else "INSUFFICIENT"
        self.append_text("  Status:                      ", 'label')
        self.append_text(f"{status_text}\n\n", status_tag)

        # State of Charge section
        self.append_text("  STATE OF CHARGE\n", 'label')
        self.append_text("  " + "-" * 50 + "\n")
        self.append_text(f"  {'Final SoC':<30} {bv.final_soc*100:>10.1f} %\n")
        self.append_text(f"  {'Minimum SoC':<30} {bv.min_soc*100:>10.1f} %  (at {bv.min_soc_distance:.0f} m)\n\n")

        # Energy & Power section
        self.append_text("  ENERGY & POWER\n", 'label')
        self.append_text("  " + "-" * 50 + "\n")
        self.append_text(f"  {'Energy Consumed':<30} {bv.total_energy_kWh*1000:>10.1f} Wh\n")
        self.append_text(f"  {'Peak Power':<30} {bv.peak_power_kW:>10.1f} kW\n")
        self.append_text(f"  {'Average Power':<30} {bv.avg_power_kW:>10.1f} kW\n\n")

        # Warnings and errors
        for warn in bv.warnings:
            self.append_text(f"  Warning: {warn}\n", 'warning')
        for err in bv.errors:
            self.append_text(f"  Error: {err}\n", 'error')
        if bv.warnings or bv.errors:
            self.append_text("\n")

    def _display_summary(self, results: Dict[str, Any]):
        """Display summary section with aligned columns."""
        self.append_text("=" * 70 + "\n", 'header')
        self.append_text("                           SUMMARY\n", 'header')
        self.append_text("=" * 70 + "\n\n", 'header')

        if 'autocross' in results:
            self.append_text("  Autocross Lap Time:          ", 'label')
            self.append_text(f"{results['autocross']['lap_time']:.3f} s\n", 'value')

        if 'skidpad' in results:
            self.append_text("  Skidpad Official Time:       ", 'label')
            self.append_text(f"{results['skidpad']['t_official']:.3f} s\n", 'value')

        if 'autocross' in results and 'battery_sufficient' in results['autocross']:
            status = "PASS" if results['autocross']['battery_sufficient'] else "FAIL"
            tag = 'success' if results['autocross']['battery_sufficient'] else 'error'
            self.append_text("  Battery Status:              ", 'label')
            self.append_text(f"{status}\n", tag)

        self.append_text("\n" + "=" * 70 + "\n", 'header')
        self.append_text("                      SIMULATION COMPLETE\n", 'success')
        self.append_text("=" * 70 + "\n", 'header')

    def update_layout_plot(self, autocross_data: dict = None, skidpad_data: dict = None):
        """Update the track layout plot (geometry only, no speed coloring)."""
        fig = self.layout_canvas.get_figure()
        fig.clear()

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

        # Plot autocross layout
        if has_autocross and ax1 is not None:
            track = autocross_data['track']
            metadata = autocross_data.get('metadata')

            # Calculate and plot track boundaries
            x_left, y_left, x_right, y_right = calculate_track_boundaries(track.x, track.y)
            ax1.plot(x_left, y_left, 'k-', lw=1, alpha=0.5)
            ax1.plot(x_right, y_right, 'k-', lw=1, alpha=0.5)

            # Shade track area
            ax1.fill(np.concatenate([x_left, x_right[::-1]]),
                     np.concatenate([y_left, y_right[::-1]]),
                     color='gray', alpha=0.15)

            # Plot centre line
            ax1.plot(track.x, track.y, 'b-', lw=2, label='Centre line', zorder=3)

            # Mark start and end
            ax1.plot(track.x[0], track.y[0], 'go', markersize=12, label='Start', zorder=10)
            ax1.plot(track.x[-1], track.y[-1], 'rs', markersize=12, label='End', zorder=10)

            if metadata and 'slalom_cones_x' in metadata:
                ax1.scatter(metadata['slalom_cones_x'], metadata['slalom_cones_y'],
                           c='orange', marker='^', s=80, label='Slalom cones',
                           zorder=15, edgecolors='black', linewidths=0.5)

            ax1.set_aspect('equal')
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('y [m]')
            ax1.set_title('Autocross Track Layout')
            ax1.legend(loc='lower right', fontsize=8)
            ax1.grid(True, alpha=0.3)

        # Plot skidpad layout (using same approach as plot_skidpad_full)
        if has_skidpad and ax2 is not None:
            from events.skidpad import (
                SKIDPAD_CENTRE_RADIUS, SKIDPAD_INNER_RADIUS, SKIDPAD_OUTER_RADIUS
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
                ax2.fill(np.concatenate([x_outer, x_inner]),
                         np.concatenate([y_outer, y_inner]),
                         color='gray', alpha=0.2)

            # Shade the continuous lane (from bottom to top through middle)
            ax2.fill([lane_left_x, lane_right_x, lane_right_x, lane_left_x],
                     [lane_bottom_y, lane_bottom_y, lane_top_y, lane_top_y],
                     color='gray', alpha=0.2)

            # Left circle boundaries and centre line
            ax2.plot(left_centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta),
                     left_centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta),
                     'k-', lw=2, label='Track boundaries')
            ax2.plot(left_centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta),
                     left_centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta), 'k-', lw=2)
            ax2.plot(left_centre[0] + SKIDPAD_CENTRE_RADIUS * np.cos(theta),
                     left_centre[1] + SKIDPAD_CENTRE_RADIUS * np.sin(theta),
                     'b--', lw=1.5, alpha=0.7, label='Centre line')

            # Right circle boundaries and centre line
            ax2.plot(right_centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta),
                     right_centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta), 'k-', lw=2)
            ax2.plot(right_centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta),
                     right_centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta), 'k-', lw=2)
            ax2.plot(right_centre[0] + SKIDPAD_CENTRE_RADIUS * np.cos(theta),
                     right_centre[1] + SKIDPAD_CENTRE_RADIUS * np.sin(theta),
                     'b--', lw=1.5, alpha=0.7)

            # Continuous lane boundaries through the middle
            ax2.plot([lane_left_x, lane_left_x], [lane_bottom_y, lane_top_y], 'k-', lw=2)
            ax2.plot([lane_right_x, lane_right_x], [lane_bottom_y, lane_top_y], 'k-', lw=2)
            ax2.plot([lane_centre_x, lane_centre_x], [lane_bottom_y, lane_top_y],
                     'b--', lw=1.5, alpha=0.7)

            # Mark entry and exit
            ax2.plot(lane_centre_x, lane_bottom_y, 'g^', markersize=12, label='Entry', zorder=10)
            ax2.plot(lane_centre_x, lane_top_y, 'rs', markersize=12, label='Exit', zorder=10)

            ax2.set_aspect('equal')
            ax2.set_xlabel('x [m]')
            ax2.set_ylabel('y [m]')
            ax2.set_title('Skidpad Track Layout')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self.layout_canvas.draw()

    def update_speed_track_plot(self, autocross_data: dict = None, skidpad_data: dict = None):
        """Update the speed track map with velocity coloring for both events."""
        fig = self.track_canvas.get_figure()
        fig.clear()

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

        # Plot autocross with speed coloring
        if has_autocross and ax1 is not None:
            track = autocross_data['track']
            v = autocross_data['v']
            metadata = autocross_data.get('metadata')

            # Calculate and plot track boundaries
            x_left, y_left, x_right, y_right = calculate_track_boundaries(track.x, track.y)
            ax1.plot(x_left, y_left, 'k-', lw=1, alpha=0.5)
            ax1.plot(x_right, y_right, 'k-', lw=1, alpha=0.5)

            # Shade track area
            ax1.fill(np.concatenate([x_left, x_right[::-1]]),
                     np.concatenate([y_left, y_right[::-1]]),
                     color='gray', alpha=0.15)

            # Speed-colored scatter plot
            sc = ax1.scatter(track.x, track.y, c=v, cmap='RdYlGn', s=8, zorder=5)
            fig.colorbar(sc, ax=ax1, label='Speed [m/s]', shrink=0.8)

            ax1.plot(track.x[0], track.y[0], 'go', markersize=12, label='Start', zorder=10)
            ax1.plot(track.x[-1], track.y[-1], 'rs', markersize=12, label='End', zorder=10)

            if metadata and 'slalom_cones_x' in metadata:
                ax1.scatter(metadata['slalom_cones_x'], metadata['slalom_cones_y'],
                           c='orange', marker='^', s=60, label='Slaloms',
                           zorder=15, edgecolors='black', linewidths=0.5)

            ax1.set_aspect('equal')
            ax1.set_xlabel('x [m]')
            ax1.set_ylabel('y [m]')
            ax1.set_title('Autocross - Speed Map')
            ax1.legend(loc='lower right', fontsize=8)
            ax1.grid(True, alpha=0.3)

        # Plot skidpad with speed coloring
        if has_skidpad and ax2 is not None:
            from matplotlib.ticker import ScalarFormatter
            
            track = skidpad_data['track']
            v = skidpad_data['v']

            # Calculate and plot track boundaries
            x_left, y_left, x_right, y_right = calculate_track_boundaries(track.x, track.y)
            ax2.plot(x_left, y_left, 'k-', lw=1, alpha=0.5)
            ax2.plot(x_right, y_right, 'k-', lw=1, alpha=0.5)

            # For skidpad (constant speed), set explicit colour limits to avoid
            # scientific notation in colourbar
            v_mean = np.mean(v)
            v_range = np.max(v) - np.min(v)
            if v_range < 0.1:  # Nearly constant speed
                # Set colour limits with a small margin around the constant speed
                vmin = v_mean - 0.5
                vmax = v_mean + 0.5
                sc = ax2.scatter(track.x, track.y, c=v, cmap='RdYlGn', s=10, 
                                zorder=5, vmin=vmin, vmax=vmax)
            else:
                sc = ax2.scatter(track.x, track.y, c=v, cmap='RdYlGn', s=10, zorder=5)
            
            cbar = fig.colorbar(sc, ax=ax2, label='Speed [m/s]', shrink=0.8)
            # Disable scientific notation on colorbar
            cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            cbar.ax.ticklabel_format(style='plain', axis='y')

            ax2.plot(track.x[0], track.y[0], 'go', markersize=12, label='Start', zorder=10)
            ax2.plot(track.x[-1], track.y[-1], 'rs', markersize=12, label='End', zorder=10)

            ax2.set_aspect('equal')
            ax2.set_xlabel('x [m]')
            ax2.set_ylabel('y [m]')
            # Show the actual speed value in title for constant speed
            if v_range < 0.1:
                ax2.set_title(f'Skidpad - Steady State: {v_mean:.2f} m/s')
            else:
                ax2.set_title('Skidpad - Speed Map')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self.track_canvas.draw()

    def update_track_plot(self, track, v: np.ndarray, metadata: dict = None, title: str = "Speed Track Map"):
        """Update the track plot with velocity coloring (legacy single-track method)."""
        fig = self.track_canvas.get_figure()
        fig.clear()
        ax = fig.add_subplot(111)

        # Calculate and plot track boundaries
        x_left, y_left, x_right, y_right = calculate_track_boundaries(track.x, track.y)
        ax.plot(x_left, y_left, 'k-', lw=1, alpha=0.5)
        ax.plot(x_right, y_right, 'k-', lw=1, alpha=0.5)

        # Shade track area
        ax.fill(np.concatenate([x_left, x_right[::-1]]),
                np.concatenate([y_left, y_right[::-1]]),
                color='gray', alpha=0.15)

        # Scatter plot colored by speed
        sc = ax.scatter(track.x, track.y, c=v, cmap='RdYlGn', s=10, zorder=5)
        fig.colorbar(sc, ax=ax, label='Speed [m/s]', shrink=0.8)

        # Mark start and end
        ax.plot(track.x[0], track.y[0], 'go', markersize=15, label='Start', zorder=10)
        ax.plot(track.x[-1], track.y[-1], 'rs', markersize=15, label='End', zorder=10)

        # Plot slalom cones if metadata available
        if metadata and 'slalom_cones_x' in metadata:
            ax.scatter(metadata['slalom_cones_x'], metadata['slalom_cones_y'],
                       c='orange', marker='^', s=80, label='Slalom cones',
                       zorder=15, edgecolors='black', linewidths=0.5)

        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self.track_canvas.draw()

    def update_speed_profile_plot(self, autocross_data: dict = None, skidpad_data: dict = None):
        """Update the speed profile plot for both events."""
        fig = self.speed_canvas.get_figure()
        fig.clear()

        has_autocross = autocross_data is not None
        has_skidpad = skidpad_data is not None

        if not has_autocross and not has_skidpad:
            return

        from solver.metrics import channels

        # Determine layout: side by side if both, single column if one
        if has_autocross and has_skidpad:
            # 2x2 layout: top row speed, bottom row acceleration
            ax1 = fig.add_subplot(221)  # Autocross speed
            ax2 = fig.add_subplot(222)  # Skidpad speed
            ax3 = fig.add_subplot(223)  # Autocross acceleration
            ax4 = fig.add_subplot(224)  # Skidpad acceleration
        elif has_autocross:
            ax1 = fig.add_subplot(211)
            ax3 = fig.add_subplot(212)
            ax2 = ax4 = None
        else:
            ax2 = fig.add_subplot(211)
            ax4 = fig.add_subplot(212)
            ax1 = ax3 = None

        # Plot Autocross
        if has_autocross and ax1 is not None:
            track = autocross_data['track']
            v = autocross_data['v']
            v_lat = autocross_data.get('v_lat')

            ax1.plot(track.s, v, 'b-', lw=1.5, label='Speed')
            if v_lat is not None:
                ax1.plot(track.s, v_lat, 'r--', alpha=0.5, lw=1, label='Lateral limit')
            ax1.set_ylabel('Speed [m/s]')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Autocross - Speed Profile')

            ax_long, ay_lat = channels(track, v)
            ax3.plot(track.s[:-1], ax_long, 'g-', lw=1, label='Longitudinal')
            ax3.plot(track.s[:-1], ay_lat, 'm-', lw=1, label='Lateral')
            ax3.axhline(0, color='k', lw=0.5)
            ax3.set_xlabel('Distance [m]')
            ax3.set_ylabel('Acceleration [m/s²]')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)
            ax3.set_title('Autocross - Acceleration')

        # Plot Skidpad
        if has_skidpad and ax2 is not None:
            from matplotlib.ticker import ScalarFormatter
            
            track = skidpad_data['track']
            v = skidpad_data['v']
            v_lat = skidpad_data.get('v_lat')

            ax2.plot(track.s, v, 'b-', lw=1.5, label='Speed')
            if v_lat is not None:
                ax2.plot(track.s, v_lat, 'r--', alpha=0.5, lw=1, label='Lateral limit')
            ax2.set_ylabel('Speed [m/s]')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # For skidpad (constant speed), set sensible y-axis limits to avoid
            # scientific notation on a flat line
            v_mean = np.mean(v)
            v_range = np.max(v) - np.min(v)
            if v_range < 0.1:  # Nearly constant speed (skidpad steady-state)
                # Add ±5% margin around the constant speed
                margin = max(0.5, v_mean * 0.05)
                ax2.set_ylim(v_mean - margin, v_mean + margin)
                ax2.set_title(f'Skidpad - Steady State: {v_mean:.2f} m/s ({v_mean*3.6:.1f} km/h)')
            else:
                ax2.set_title('Skidpad - Speed Profile')
            
            # Disable scientific notation on y-axis
            ax2.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax2.ticklabel_format(style='plain', axis='y')

            ax_long, ay_lat = channels(track, v)
            ax4.plot(track.s[:-1], ax_long, 'g-', lw=1, label='Longitudinal')
            ax4.plot(track.s[:-1], ay_lat, 'm-', lw=1, label='Lateral')
            ax4.axhline(0, color='k', lw=0.5)
            ax4.set_xlabel('Distance [m]')
            ax4.set_ylabel('Acceleration [m/s²]')
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # For skidpad, longitudinal accel should be ~0 (steady state)
            # Set sensible limits so the plot is readable
            ax_range = np.max(np.abs(ax_long)) if len(ax_long) > 0 else 0
            if ax_range < 0.1:  # Nearly zero longitudinal accel (expected for skidpad)
                ax4.set_ylim(-1, np.max(ay_lat) * 1.1 if len(ay_lat) > 0 else 20)
            ax4.set_title(f'Skidpad - Lateral: {np.mean(ay_lat):.1f} m/s² ({np.mean(ay_lat)/9.81:.2f} g)')
            ax4.set_title('Skidpad - Acceleration')

        fig.tight_layout()
        self.speed_canvas.draw()

    def update_battery_combined_plot(self, autocross_data: dict = None, skidpad_data: dict = None):
        """Update the battery analysis plot for both events."""
        fig = self.battery_canvas.get_figure()
        fig.clear()

        has_autocross = autocross_data is not None and 'battery_state' in autocross_data
        has_skidpad = skidpad_data is not None and 'battery_state' in skidpad_data

        if not has_autocross and not has_skidpad:
            # No battery data available
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No battery data available\n(Battery analysis disabled)',
                    ha='center', va='center', fontsize=12, color='gray',
                    transform=ax.transAxes)
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

        # Plot Autocross battery
        if has_autocross:
            track = autocross_data['track']
            battery_state = autocross_data['battery_state']
            vehicle = autocross_data['vehicle']
            validation = autocross_data.get('battery_validation')

            # SoC
            ax1.plot(track.s, battery_state.soc * 100, 'b-', lw=2)
            ax1.axhline(y=vehicle.battery.min_soc * 100, color='r', linestyle='--',
                        label=f'Min SoC ({vehicle.battery.min_soc:.0%})')
            ax1.fill_between(track.s, 0, vehicle.battery.min_soc * 100, color='red', alpha=0.1)
            ax1.set_ylabel('SoC [%]')
            ax1.set_ylim(0, 105)
            ax1.legend(loc='upper right', fontsize=7)
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Autocross - State of Charge')

            # Power
            ax3.fill_between(track.s, 0, battery_state.power_kW, color='red', alpha=0.5)
            ax3.plot(track.s, battery_state.power_kW, 'darkred', lw=1)
            ax3.axhline(y=vehicle.battery.max_discharge_kW, color='red', linestyle='--',
                        alpha=0.5, label=f'Max ({vehicle.battery.max_discharge_kW:.0f} kW)')
            ax3.set_ylabel('Power [kW]')
            ax3.legend(loc='upper right', fontsize=7)
            ax3.grid(True, alpha=0.3)
            ax3.set_title('Autocross - Discharge Power')

            # Energy
            ax5.plot(track.s, battery_state.energy_used_kWh * 1000, 'r-', lw=2)
            usable = vehicle.battery.capacity_kWh * (vehicle.battery.initial_soc - vehicle.battery.min_soc) * 1000
            ax5.axhline(y=usable, color='orange', linestyle='--', label=f'Usable ({usable:.0f} Wh)')
            ax5.set_xlabel('Distance [m]')
            ax5.set_ylabel('Energy [Wh]')
            ax5.legend(loc='upper left', fontsize=7)
            ax5.grid(True, alpha=0.3)
            ax5.set_title('Autocross - Cumulative Energy')

        # Plot Skidpad battery
        if has_skidpad:
            from matplotlib.ticker import ScalarFormatter
            
            track = skidpad_data['track']
            battery_state = skidpad_data['battery_state']
            vehicle = skidpad_data['vehicle']
            validation = skidpad_data.get('battery_validation')

            # SoC
            ax2.plot(track.s, battery_state.soc * 100, 'b-', lw=2)
            ax2.axhline(y=vehicle.battery.min_soc * 100, color='r', linestyle='--',
                        label=f'Min SoC ({vehicle.battery.min_soc:.0%})')
            ax2.fill_between(track.s, 0, vehicle.battery.min_soc * 100, color='red', alpha=0.1)
            ax2.set_ylabel('SoC [%]')
            ax2.set_ylim(0, 105)
            ax2.legend(loc='upper right', fontsize=7)
            ax2.grid(True, alpha=0.3)
            
            # For skidpad, SoC drop is very small - show it meaningfully
            soc_drop = (battery_state.soc[0] - battery_state.soc[-1]) * 100
            ax2.set_title(f'Skidpad - SoC (Δ = {soc_drop:.3f}%)')

            # Power - for skidpad it's constant
            power_kW = battery_state.power_kW
            ax4.fill_between(track.s, 0, power_kW, color='red', alpha=0.5)
            ax4.plot(track.s, power_kW, 'darkred', lw=1)
            ax4.axhline(y=vehicle.battery.max_discharge_kW, color='red', linestyle='--',
                        alpha=0.5, label=f'Max ({vehicle.battery.max_discharge_kW:.0f} kW)')
            ax4.set_ylabel('Power [kW]')
            ax4.legend(loc='upper right', fontsize=7)
            ax4.grid(True, alpha=0.3)
            
            # For constant power, set sensible y-axis and show the value in title
            power_mean = np.mean(power_kW)
            power_range = np.max(power_kW) - np.min(power_kW) if len(power_kW) > 0 else 0
            if power_range < 0.01:  # Constant power (skidpad steady-state)
                ax4.set_ylim(0, max(power_mean * 2, 5))  # Show up to 2x the actual or 5kW
                ax4.set_title(f'Skidpad - Constant Power: {power_mean:.2f} kW')
            else:
                ax4.set_title('Skidpad - Discharge Power')
            ax4.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

            # Energy
            ax6.plot(track.s, battery_state.energy_used_kWh * 1000, 'r-', lw=2)
            usable = vehicle.battery.capacity_kWh * (vehicle.battery.initial_soc - vehicle.battery.min_soc) * 1000
            ax6.axhline(y=usable, color='orange', linestyle='--', label=f'Usable ({usable:.0f} Wh)')
            ax6.set_xlabel('Distance [m]')
            ax6.set_ylabel('Energy [Wh]')
            ax6.legend(loc='upper left', fontsize=7)
            ax6.grid(True, alpha=0.3)
            
            # Show actual energy used in title
            energy_used_Wh = battery_state.energy_used_kWh[-1] * 1000 if len(battery_state.energy_used_kWh) > 0 else 0
            ax6.set_title(f'Skidpad - Energy: {energy_used_Wh:.1f} Wh')

        fig.tight_layout()
        self.battery_canvas.draw()

    def update_speed_plot(self, track, v: np.ndarray, v_lat: np.ndarray = None):
        """Update the speed profile plot (legacy single-event method)."""
        fig = self.speed_canvas.get_figure()
        fig.clear()

        # Two subplots: speed and acceleration
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        # Speed plot
        ax1.plot(track.s, v, 'b-', lw=1.5, label='Actual speed')
        if v_lat is not None:
            ax1.plot(track.s, v_lat, 'r--', alpha=0.5, lw=1, label='Lateral limit')
        ax1.set_ylabel('Speed [m/s]')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Speed Profile')

        # Acceleration plot
        from solver.metrics import channels
        ax_long, ay_lat = channels(track, v)
        ax2.plot(track.s[:-1], ax_long, 'g-', lw=1, label='Longitudinal')
        ax2.plot(track.s[:-1], ay_lat, 'm-', lw=1, label='Lateral')
        ax2.axhline(0, color='k', lw=0.5)
        ax2.set_xlabel('Distance [m]')
        ax2.set_ylabel('Acceleration [m/s²]')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        self.speed_canvas.draw()

    def update_battery_plot(self, track, battery_state, vehicle, validation=None):
        """Update the battery analysis plot (legacy single-event method)."""
        fig = self.battery_canvas.get_figure()
        fig.clear()

        # Three subplots: SoC, Power, Energy
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax3 = fig.add_subplot(313, sharex=ax1)

        # SoC plot
        ax1.plot(track.s, battery_state.soc * 100, 'b-', lw=2)
        ax1.axhline(y=vehicle.battery.min_soc * 100, color='r', linestyle='--',
                    label=f'Min SoC ({vehicle.battery.min_soc:.0%})')
        ax1.fill_between(track.s, 0, vehicle.battery.min_soc * 100, color='red', alpha=0.1)
        ax1.set_ylabel('State of Charge [%]')
        ax1.set_ylim(0, 105)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Battery State of Charge')

        # Power plot
        ax2.fill_between(track.s, 0, battery_state.power_kW, color='red', alpha=0.5)
        ax2.plot(track.s, battery_state.power_kW, 'darkred', lw=1)
        ax2.axhline(y=vehicle.battery.max_discharge_kW, color='red', linestyle='--',
                    alpha=0.5, label=f'Max ({vehicle.battery.max_discharge_kW:.0f} kW)')
        ax2.set_ylabel('Power [kW]')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Discharge Power')

        # Energy plot
        ax3.plot(track.s, battery_state.energy_used_kWh * 1000, 'r-', lw=2)
        usable = vehicle.battery.capacity_kWh * (vehicle.battery.initial_soc - vehicle.battery.min_soc) * 1000
        ax3.axhline(y=usable, color='orange', linestyle='--', label=f'Usable ({usable:.0f} Wh)')
        ax3.set_xlabel('Distance [m]')
        ax3.set_ylabel('Energy [Wh]')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Cumulative Energy')

        # Status text
        if validation:
            status = "SUFFICIENT" if validation.sufficient else "INSUFFICIENT"
            color = 'green' if validation.sufficient else 'red'
            fig.text(0.02, 0.98, f"Status: {status}", fontsize=10, fontweight='bold',
                     color=color, verticalalignment='top', transform=fig.transFigure)

        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        self.battery_canvas.draw()

    def show_tab(self, tab_name: str):
        """Switch to a specific tab by name."""
        tab_map = {
            'results': 0,
            'layout': 1,
            'track': 2,
            'speed': 3,
            'battery': 4,
        }
        if tab_name in tab_map:
            self.notebook.select(tab_map[tab_name])
