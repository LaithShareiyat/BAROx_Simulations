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
            on_run=self.run_simulation,
            on_save_results=self.save_results
        )
        self.control_panel.pack(fill='both', expand=True)

        # Right panel - Results (with config callback for battery optimiser)
        self.results_frame = ttk.Frame(self.paned)
        self.results_panel = ResultsPanel(
            self.results_frame,
            get_config_callback=self.control_panel.get_config
        )
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
                EVPowertrainMVP, EVPowertrainParams, BatteryParams
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
            VehicleParams, AeroParams, TyreParamsMVP, TyreParams,
            EVPowertrainMVP, EVPowertrainParams, BatteryParams,
            VehicleGeometry, TorqueVectoringParams
        )

        aero = AeroParams(
            rho=config['aero']['rho'],
            Cd=config['aero']['Cd'],
            Cl=config['aero']['Cl'],
            A=config['aero']['A'],
        )

        # Check if using extended tyre model with cornering stiffness
        tyre_config = config['tyre']
        if 'C_alpha_f' in tyre_config:
            tyre = TyreParams(
                mu=tyre_config['mu'],
                C_alpha_f=tyre_config.get('C_alpha_f', 45000.0),
                C_alpha_r=tyre_config.get('C_alpha_r', 50000.0),
            )
        else:
            tyre = TyreParamsMVP(mu=tyre_config['mu'])

        # Check if using new extended powertrain format or legacy format
        pt_config = config['powertrain']
        if 'drivetrain' in pt_config:
            # New extended powertrain format
            powertrain = EVPowertrainParams(
                drivetrain=pt_config['drivetrain'],
                motor_power_kW=pt_config['motor_power_kW'],
                motor_torque_Nm=pt_config['motor_torque_Nm'],
                motor_rpm_max=pt_config['motor_rpm_max'],
                gear_ratio=pt_config['gear_ratio'],
                wheel_radius_m=pt_config['wheel_radius_m'],
                motor_efficiency=pt_config.get('motor_efficiency', 0.85),
            )
        else:
            # Legacy format with P_max_kW and Fx_max_N
            powertrain = EVPowertrainMVP(
                P_max=pt_config['P_max_kW'] * 1000,
                Fx_max=pt_config['Fx_max_N'],
            )

        battery = None
        if config['battery'].get('enabled', True):
            battery = BatteryParams(
                capacity_kWh=config['battery']['capacity_kWh'],
                initial_soc=config['battery']['initial_soc'],
                min_soc=config['battery']['min_soc'],
                max_discharge_kW=config['battery']['max_discharge_kW'],
                eta_discharge=config['battery'].get('eta_discharge', 0.95),
                # Current limiting (FS 2025 rules)
                nominal_voltage_V=config['battery'].get('nominal_voltage_V', 400.0),
                max_current_A=config['battery'].get('max_current_A', 500.0),
                # Regenerative braking parameters
                regen_enabled=config['battery'].get('regen_enabled', False),
                eta_regen=config['battery'].get('eta_regen', 0.85),
                max_regen_kW=config['battery'].get('max_regen_kW', 50.0),
                regen_capture_percent=config['battery'].get('regen_capture_percent', 100.0),
            )

        # Create geometry params if present (for bicycle model)
        geometry = None
        if 'geometry' in config:
            geo_config = config['geometry']
            geometry = VehicleGeometry(
                wheelbase_m=geo_config.get('wheelbase_m', 1.55),
                L_f_m=geo_config.get('L_f_m', 0.75),
                L_r_m=geo_config.get('L_r_m', 0.80),
                track_front_m=geo_config.get('track_front_m', 1.20),
                track_rear_m=geo_config.get('track_rear_m', 1.20),
                h_cg_m=geo_config.get('h_cg_m', 0.28),
            )

        # Create torque vectoring params if present
        torque_vectoring = None
        if 'torque_vectoring' in config:
            tv_config = config['torque_vectoring']
            torque_vectoring = TorqueVectoringParams(
                enabled=tv_config.get('enabled', False),
                effectiveness=tv_config.get('effectiveness', 1.0),
                max_torque_transfer=tv_config.get('max_torque_transfer', 0.5),
                strategy=tv_config.get('strategy', 'load_proportional'),
            )

        return VehicleParams(
            m=config['vehicle']['mass_kg'],
            g=config['vehicle']['g'],
            Crr=config['vehicle']['Crr'],
            aero=aero,
            tyre=tyre,
            powertrain=powertrain,
            battery=battery,
            geometry=geometry,
            torque_vectoring=torque_vectoring,
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

    def save_results(self):
        """Save all plots and results to a user-selected folder."""
        from tkinter import filedialog
        from datetime import datetime
        
        # Check if there are results to save
        if not self.results:
            messagebox.showwarning("No Results", "No simulation results to save. Run a simulation first.")
            return
        
        # Ask user to select a folder
        folder_path = filedialog.askdirectory(
            title="Select Folder to Save Results",
            initialdir=os.path.expanduser("~")
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
                (self.results_panel.speed_canvas, "speed_profile.png"),
                (self.results_panel.battery_canvas, "battery_analysis.png"),
            ]
            
            for canvas, filename in plot_configs:
                try:
                    fig = canvas.get_figure()
                    filepath = os.path.join(plots_folder, filename)
                    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
                    saved_files.append(f"plots/{filename}")
                except Exception as e:
                    print(f"Failed to save {filename}: {e}")
            
            # Save results text
            results_text = self._generate_results_text()
            results_filepath = os.path.join(save_folder, "results.txt")
            with open(results_filepath, 'w') as f:
                f.write(results_text)
            saved_files.append("results.txt")
            
            # Save configuration
            config = self.control_panel.get_config()
            config_filepath = os.path.join(save_folder, "config.yaml")
            import yaml
            with open(config_filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            saved_files.append("config.yaml")
            
            # Show success message
            files_list = "\n".join(f"  • {f}" for f in saved_files)
            messagebox.showinfo(
                "Results Saved",
                f"Results saved to:\n{save_folder}\n\nFiles saved:\n{files_list}"
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
        lines.append(f"Tyre Friction (μ): {config['tyre']['mu']}")

        # Powertrain info - handle both new and legacy formats
        pt = config['powertrain']
        if 'drivetrain' in pt:
            n_motors = 4 if pt['drivetrain'] == 'AWD' else 2
            total_power = pt['motor_power_kW'] * n_motors
            fx_max = (pt['motor_torque_Nm'] * n_motors * pt['gear_ratio']) / pt['wheel_radius_m']
            wheel_rpm = pt['motor_rpm_max'] / pt['gear_ratio']
            v_max = (wheel_rpm * 2 * 3.14159 * pt['wheel_radius_m']) / 60
            lines.append(f"Drivetrain: {pt['drivetrain']} ({n_motors} motors)")
            lines.append(f"Motor Power: {pt['motor_power_kW']} kW (per motor)")
            lines.append(f"Motor Torque: {pt['motor_torque_Nm']} Nm (per motor)")
            lines.append(f"Motor Max RPM: {pt['motor_rpm_max']}")
            motor_eff = pt.get('motor_efficiency', 0.85)
            lines.append(f"Motor Efficiency: {motor_eff*100:.0f}%")
            lines.append(f"Gear Ratio: {pt['gear_ratio']}:1")
            lines.append(f"Wheel Radius: {pt['wheel_radius_m']} m")
            lines.append(f"Total Power: {total_power:.0f} kW")
            lines.append(f"Max Tractive Force: {fx_max:.0f} N")
            lines.append(f"Max Speed (RPM limit): {v_max:.1f} m/s ({v_max*3.6:.0f} km/h)")
        else:
            lines.append(f"Max Power: {pt['P_max_kW']} kW")
            lines.append(f"Max Tractive Force: {pt['Fx_max_N']} N")

        if config['battery'].get('enabled', True):
            lines.append(f"Battery Capacity: {config['battery']['capacity_kWh']} kWh")
            lines.append(f"Initial SoC: {config['battery']['initial_soc']*100:.0f}%")
            lines.append(f"Min SoC: {config['battery']['min_soc']*100:.0f}%")
            regen_enabled = config['battery'].get('regen_enabled', False)
            lines.append(f"Regenerative Braking: {'ENABLED' if regen_enabled else 'DISABLED'}")
            if regen_enabled:
                lines.append(f"  Regen Efficiency: {config['battery'].get('eta_regen', 0.85)*100:.0f}%")
                lines.append(f"  Max Regen Power: {config['battery'].get('max_regen_kW', 50)} kW")
                lines.append(f"  Braking Capture: {config['battery'].get('regen_capture_percent', 100):.0f}%")
        lines.append("")
        
        # Autocross results
        if 'autocross' in self.results:
            ac = self.results['autocross']
            lines.append("AUTOCROSS RESULTS")
            lines.append("-" * 50)
            lines.append(f"Lap Time: {ac['lap_time']:.3f} s")
            lines.append(f"Average Speed: {ac['avg_speed']:.2f} m/s ({ac['avg_speed']*3.6:.1f} km/h)")
            lines.append(f"Maximum Speed: {ac['max_speed']:.2f} m/s ({ac['max_speed']*3.6:.1f} km/h)")
            lines.append(f"Minimum Speed: {ac['min_speed']:.2f} m/s ({ac['min_speed']*3.6:.1f} km/h)")
            lines.append(f"Max Longitudinal Accel: {ac['max_ax']:.2f} m/s²")
            lines.append(f"Max Braking Decel: {ac['min_ax']:.2f} m/s²")
            lines.append(f"Max Lateral Accel: {ac['max_ay']:.2f} m/s²")
            
            if ac.get('track'):
                lines.append(f"Track Length: {ac['track'].s[-1]:.1f} m")
            
            if ac.get('battery_validation'):
                bv = ac['battery_validation']
                lines.append("")
                lines.append("Battery Analysis:")
                lines.append(f"  Status: {'SUFFICIENT' if bv.sufficient else 'INSUFFICIENT'}")
                lines.append(f"  Final SoC: {bv.final_soc*100:.1f}%")
                lines.append(f"  Minimum SoC: {bv.min_soc*100:.1f}%")
                lines.append(f"  Total Energy: {bv.total_energy_kWh:.3f} kWh")
                lines.append(f"  Peak Power: {bv.peak_power_kW:.1f} kW")
                lines.append(f"  Average Power: {bv.avg_power_kW:.1f} kW")

            lines.append("")
        
        # Skidpad results
        if 'skidpad' in self.results:
            sp = self.results['skidpad']
            lines.append("SKIDPAD RESULTS")
            lines.append("-" * 50)
            lines.append(f"Official Time (2 laps avg): {sp['t_official']:.3f} s")
            lines.append(f"Full Run Time: {sp['t_full_run']:.3f} s")
            lines.append(f"Single Circle Lap Time: {sp['lap_time']:.3f} s")
            lines.append(f"Cornering Speed: {sp['avg_speed']:.2f} m/s ({sp['avg_speed']*3.6:.1f} km/h)")
            lines.append(f"Lateral Acceleration: {sp['max_ay']:.2f} m/s²")
            
            if sp.get('battery_validation'):
                bv = sp['battery_validation']
                lines.append("")
                lines.append("Battery Analysis (single circle):")
                lines.append(f"  Energy per circle: {bv.total_energy_kWh*1000:.1f} Wh")
                lines.append(f"  Average Power: {bv.avg_power_kW:.2f} kW")
            
            lines.append("")
        
        lines.append("=" * 70)
        lines.append("End of Report")
        lines.append("=" * 70)
        
        return "\n".join(lines)


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
