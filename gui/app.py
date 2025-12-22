"""Main BAROx GUI Application."""
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.panels.control_panel import ControlPanel
from gui.panels.results_panel import ResultsPanel


class BAROxGUI:
    """Main application class for BAROx GUI."""

    def __init__(self, root: tk.Tk):
        """Initialize the GUI application."""
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

    def _configure_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()

        # Try to use a modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')

        # Custom styles
        style.configure('TLabelFrame', padding=5)
        style.configure('TLabelFrame.Label', font=('Segoe UI', 9, 'bold'))

        # Accent button style
        style.configure('Accent.TButton',
                        font=('Segoe UI', 10, 'bold'),
                        padding=10)

        # Invalid entry style
        style.configure('Invalid.TEntry', fieldbackground='#ffcccc')

    def _create_layout(self):
        """Create the main application layout."""
        # Main paned window for resizable panels
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill='both', expand=True, padx=5, pady=5)

        # Left panel - Controls (fixed width)
        self.control_frame = ttk.Frame(self.paned, width=350)
        self.control_panel = ControlPanel(
            self.control_frame,
            on_run=self.run_simulation
        )
        self.control_panel.pack(fill='both', expand=True)

        # Right panel - Results
        self.results_frame = ttk.Frame(self.paned)
        self.results_panel = ResultsPanel(self.results_frame)
        self.results_panel.pack(fill='both', expand=True)

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
                VehicleParams, AeroParams, TyreParamsMVP,
                EVPowertrainMVP, BatteryParams
            )
            from solver.qss_speed import solve_qss
            from solver.metrics import lap_time, channels, energy_consumption
            from solver.battery import simulate_battery, validate_battery_capacity

            # Create vehicle from config
            vehicle = self._create_vehicle(config)

            results = {}

            # Run Autocross
            if event_type in ('autocross', 'both'):
                self.root.after(0, lambda: self.results_panel.set_status("Running Autocross simulation..."))
                results['autocross'] = self._run_autocross(vehicle, config)

            # Run Skidpad
            if event_type in ('skidpad', 'both'):
                self.root.after(0, lambda: self.results_panel.set_status("Running Skidpad simulation..."))
                results['skidpad'] = self._run_skidpad(vehicle)

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
            VehicleParams, AeroParams, TyreParamsMVP,
            EVPowertrainMVP, BatteryParams
        )

        aero = AeroParams(
            rho=config['aero']['rho'],
            Cd=config['aero']['Cd'],
            Cl=config['aero']['Cl'],
            A=config['aero']['A'],
        )

        tyre = TyreParamsMVP(mu=config['tyre']['mu'])

        powertrain = EVPowertrainMVP(
            P_max=config['powertrain']['P_max_kW'] * 1000,
            Fx_max=config['powertrain']['Fx_max_N'],
        )

        battery = None
        if config['battery'].get('enabled', True):
            battery = BatteryParams(
                capacity_kWh=config['battery']['capacity_kWh'],
                initial_soc=config['battery']['initial_soc'],
                min_soc=config['battery']['min_soc'],
                max_discharge_kW=config['battery']['max_discharge_kW'],
                eta_discharge=config['battery'].get('eta_discharge', 0.95),
            )

        return VehicleParams(
            m=config['vehicle']['mass_kg'],
            g=config['vehicle']['g'],
            Crr=config['vehicle']['Crr'],
            aero=aero,
            tyre=tyre,
            powertrain=powertrain,
            battery=battery,
        )

    def _run_autocross(self, vehicle, config: dict) -> dict:
        """Run autocross simulation and return metrics."""
        from events.autocross_generator import build_standard_autocross, validate_autocross
        from solver.qss_speed import solve_qss
        from solver.battery import simulate_battery, validate_battery_capacity

        # Build track
        track, metadata = build_standard_autocross()

        # Solve for velocity profile
        result, t_lap = solve_qss(track, vehicle)
        v = result['v']

        # Compute metrics
        metrics = self._compute_metrics(track, v, vehicle)
        metrics['metadata'] = metadata
        metrics['track'] = track
        metrics['v'] = v
        metrics['v_lat'] = result['v_lat']

        # Battery analysis
        if vehicle.battery is not None:
            battery_validation = validate_battery_capacity(track, v, vehicle)
            battery_state = simulate_battery(track, v, vehicle)
            metrics['battery_validation'] = battery_validation
            metrics['battery_sufficient'] = battery_validation.sufficient
            metrics['battery_state'] = battery_state
            metrics['vehicle'] = vehicle

        return metrics

    def _run_skidpad(self, vehicle) -> dict:
        """Run skidpad simulation and return metrics."""
        from events.skidpad import (
            build_skidpad_track, skidpad_time_from_single_circle,
            SKIDPAD_CENTRE_RADIUS
        )
        from solver.qss_speed import solve_qss
        from solver.battery import simulate_battery, validate_battery_capacity

        # Build track
        track = build_skidpad_track()

        # Solve
        result, t_lap = solve_qss(track, vehicle)
        v = result['v']

        # Get timing
        timing = skidpad_time_from_single_circle(t_lap)

        # Compute metrics
        metrics = self._compute_metrics(track, v, vehicle)
        metrics['t_official'] = timing['t_official']
        metrics['t_full_run'] = timing['t_full_run']
        metrics['track'] = track
        metrics['v'] = v
        metrics['v_lat'] = result['v_lat']

        # Battery analysis (same as autocross)
        if vehicle.battery is not None:
            battery_validation = validate_battery_capacity(track, v, vehicle)
            battery_state = simulate_battery(track, v, vehicle)
            metrics['battery_validation'] = battery_validation
            metrics['battery_sufficient'] = battery_validation.sufficient
            metrics['battery_state'] = battery_state
            metrics['vehicle'] = vehicle

        return metrics

    def _compute_metrics(self, track, v: np.ndarray, vehicle) -> dict:
        """Compute performance metrics from velocity profile."""
        from solver.metrics import lap_time, channels, energy_consumption

        t = lap_time(track, v)
        ax, ay = channels(track, v)

        try:
            energy = energy_consumption(track, v, vehicle)
            energy_kwh = energy.get('E_net_kWh', 0)
        except Exception:
            energy_kwh = 0

        v_positive = v[v > 0.1]
        min_speed = np.min(v_positive) if len(v_positive) > 0 else 0.0

        ax_clean = ax[~np.isnan(ax)]
        ay_clean = ay[~np.isnan(ay)]

        return {
            'lap_time': t,
            'avg_speed': np.nanmean(v),
            'max_speed': np.nanmax(v),
            'min_speed': min_speed,
            'max_ax': np.max(ax_clean) if len(ax_clean) > 0 else 0.0,
            'min_ax': np.min(ax_clean) if len(ax_clean) > 0 else 0.0,
            'max_ay': np.max(np.abs(ay_clean)) if len(ay_clean) > 0 else 0.0,
            'energy_consumed_kWh': energy_kwh,
        }

    def _update_display(self, results: dict, event_type: str):
        """Update the display with simulation results."""
        # Update text results
        self.results_panel.display_results(results, event_type)

        # Prepare data for plots
        autocross_data = results.get('autocross')
        skidpad_data = results.get('skidpad')

        # Update track layout plot (geometry only)
        self.results_panel.update_layout_plot(autocross_data, skidpad_data)

        # Update speed track map (velocity colored)
        self.results_panel.update_speed_track_plot(autocross_data, skidpad_data)

        # Update speed profile (for both events)
        self.results_panel.update_speed_profile_plot(autocross_data, skidpad_data)

        # Update battery plot (for both events)
        self.results_panel.update_battery_combined_plot(autocross_data, skidpad_data)

        # Show results tab
        self.results_panel.show_tab('results')

        # Update status
        if 'autocross' in results and 'skidpad' in results:
            ac_time = results['autocross']['lap_time']
            sp_time = results['skidpad']['t_official']
            self.results_panel.set_status(f"Complete - Autocross: {ac_time:.3f}s | Skidpad: {sp_time:.3f}s")
        elif 'autocross' in results:
            lap_time = results['autocross']['lap_time']
            self.results_panel.set_status(f"Complete - Autocross: {lap_time:.3f}s")
        elif 'skidpad' in results:
            lap_time = results['skidpad']['t_official']
            self.results_panel.set_status(f"Complete - Skidpad: {lap_time:.3f}s")

    def _show_error(self, message: str):
        """Display error message."""
        self.results_panel.set_status("Error occurred", 'error')
        self.results_panel.set_results_text(f"ERROR:\n\n{message}")
        messagebox.showerror("Simulation Error", message[:500])

    def _simulation_complete(self):
        """Clean up after simulation completes."""
        self.running = False
        self.control_panel.set_running(False)


def main():
    """Main entry point for the GUI."""
    root = tk.Tk()

    # Set icon if available
    try:
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'assets', 'icon.ico'
        )
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except Exception:
        pass

    app = BAROxGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
