import numpy as np
import matplotlib.pyplot as plt
from models.track import Track
from models.vehicle import VehicleParams
from solver.metrics import channels
from solver.battery import BatteryState, BatteryValidation


def plot_track(track: Track, v: np.ndarray = None, title: str = "Track Layout"):
    """Plot track layout, optionally coloured by speed."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if v is not None:
        sc = ax.scatter(track.x, track.y, c=v, cmap='RdYlGn', s=5)
        plt.colorbar(sc, label='Speed [m/s]')
    else:
        ax.plot(track.x, track.y, 'k-', lw=1)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    plt.show()
    return fig, ax


def plot_speed_profile(track: Track, v: np.ndarray, v_lat: np.ndarray):
    """Plot speed vs distance with lateral limit."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Speed plot
    axes[0].plot(track.s, v, 'b-', label='Actual speed')
    axes[0].plot(track.s, v_lat, 'r--', alpha=0.5, label='Lateral limit')
    axes[0].set_ylabel('Speed [m/s]')
    axes[0].legend()
    axes[0].grid(True)
    
    # Acceleration plot
    ax_long, ay_lat = channels(track, v)
    axes[1].plot(track.s[:-1], ax_long, 'g-', label='$a_x$ (longitudinal)')
    axes[1].plot(track.s[:-1], ay_lat, 'm-', label='$a_y$ (lateral)')
    axes[1].set_xlabel('Distance [m]')
    axes[1].set_ylabel('Acceleration [m/s²]')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_battery_state(track: Track, state: BatteryState, vehicle: VehicleParams,
                       validation: BatteryValidation = None,
                       title: str = "Battery State During Lap"):
    """
    Plot battery state of charge and power throughout the lap (supports regen).

    Args:
        track: Track object
        state: BatteryState from simulation
        vehicle: Vehicle parameters
        validation: Optional validation results
        title: Plot title

    Returns:
        fig, axes
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Check if regen is enabled
    regen_enabled = vehicle.battery.regen_enabled if vehicle.battery else False

    # Plot 1: State of Charge
    ax1 = axes[0]
    ax1.plot(track.s, state.soc * 100, 'b-', lw=2, label='SoC')
    ax1.axhline(y=vehicle.battery.min_soc * 100, color='r', linestyle='--',
                label=f'Min SoC ({vehicle.battery.min_soc:.0%})')
    ax1.axhline(y=vehicle.battery.initial_soc * 100, color='g', linestyle='--',
                alpha=0.5, label=f'Initial SoC ({vehicle.battery.initial_soc:.0%})')

    if validation is not None:
        ax1.axvline(x=validation.min_soc_distance, color='orange', linestyle=':',
                    alpha=0.7, label=f'Min SoC point')
        ax1.scatter([validation.min_soc_distance], [validation.min_soc * 100],
                    c='orange', s=100, zorder=5)

    ax1.fill_between(track.s, 0, vehicle.battery.min_soc * 100,
                     color='red', alpha=0.1, label='Depleted zone')
    ax1.set_ylabel('State of Charge [%]')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Battery State of Charge')

    # Plot 2: Power (discharge and regen)
    ax2 = axes[1]

    power = state.power_kW

    # Separate positive (discharge) and negative (regen) power
    power_discharge = np.maximum(power, 0)
    power_regen = np.minimum(power, 0)

    # Plot discharge (positive power) in red
    ax2.fill_between(track.s, 0, power_discharge, color='red', alpha=0.5, label='Discharge')

    # Plot regen (negative power) in green
    if regen_enabled and np.any(power_regen < 0):
        ax2.fill_between(track.s, 0, power_regen, color='green', alpha=0.5, label='Regen')

    ax2.plot(track.s, power, 'k-', lw=0.5, alpha=0.7)

    ax2.axhline(y=vehicle.battery.max_discharge_kW, color='red', linestyle='--',
                alpha=0.5, label=f'Max discharge ({vehicle.battery.max_discharge_kW:.0f} kW)')
    if regen_enabled:
        ax2.axhline(y=-vehicle.battery.max_regen_kW, color='green', linestyle='--',
                    alpha=0.5, label=f'Max regen ({vehicle.battery.max_regen_kW:.0f} kW)')
    ax2.axhline(y=0, color='black', linestyle='-', lw=0.5)

    ax2.set_ylabel('Power [kW]')
    # Set y-limits to show both discharge and regen
    if regen_enabled:
        min_power = min(np.min(power_regen), -vehicle.battery.max_regen_kW) * 1.1
        max_power = max(np.max(power_discharge), vehicle.battery.max_discharge_kW) * 1.1
        ax2.set_ylim(min_power, max_power)
    else:
        ax2.set_ylim(bottom=-5)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    power_title = 'Electrical Power (With Regen)' if regen_enabled else 'Electrical Power (No Regen)'
    ax2.set_title(power_title)

    # Plot 3: Cumulative Energy
    ax3 = axes[2]
    energy_Wh = state.energy_used_kWh * 1000
    ax3.plot(track.s, energy_Wh, 'b-', lw=2, label='Net energy')

    # Show usable capacity line
    usable_capacity_Wh = vehicle.battery.capacity_kWh * (vehicle.battery.initial_soc - vehicle.battery.min_soc) * 1000
    ax3.axhline(y=usable_capacity_Wh, color='orange', linestyle='--',
                alpha=0.7, label=f'Usable capacity ({usable_capacity_Wh:.0f} Wh)')
    ax3.axhline(y=0, color='black', linestyle='-', lw=0.5)

    ax3.set_xlabel('Distance [m]')
    ax3.set_ylabel('Energy [Wh]')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    energy_title = 'Net Energy (With Regen Recovery)' if regen_enabled else 'Cumulative Energy Consumption'
    ax3.set_title(energy_title)

    # Add summary text
    if validation is not None:
        regen_text = "Regen ON" if regen_enabled else "No regen"
        if regen_enabled:
            regen_text += f" (η={vehicle.battery.eta_regen:.0%})"

        summary = (
            f"Battery: {vehicle.battery.capacity_kWh:.2f} kWh\n"
            f"Net energy: {validation.total_energy_kWh*1000:.1f} Wh\n"
            f"Final SoC: {validation.final_soc:.1%}\n"
            f"Min SoC: {validation.min_soc:.1%}\n"
            f"Peak power: {validation.peak_power_kW:.1f} kW\n"
            f"Avg power: {validation.avg_power_kW:.1f} kW\n"
            f"{regen_text}"
        )
        status = "✓ SUFFICIENT" if validation.sufficient else "✗ INSUFFICIENT"
        color = 'green' if validation.sufficient else 'red'

        fig.text(0.02, 0.98, summary, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 transform=fig.transFigure)
        fig.text(0.02, 0.70, status, fontsize=12, fontweight='bold', color=color,
                 verticalalignment='top', transform=fig.transFigure)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

    return fig, axes


def plot_soc_on_track(track: Track, state: BatteryState, vehicle: VehicleParams,
                      title: str = "SoC Map"):
    """
    Plot track coloured by state of charge.
    
    Args:
        track: Track object
        state: BatteryState from simulation
        vehicle: Vehicle parameters
        title: Plot title
    
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sc = ax.scatter(track.x, track.y, c=state.soc * 100, cmap='RdYlGn', 
                    s=10, vmin=0, vmax=100)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('State of Charge [%]')
    
    # Mark start and finish
    ax.plot(track.x[0], track.y[0], 'go', markersize=15, label='Start', zorder=10)
    ax.plot(track.x[-1], track.y[-1], 'rs', markersize=15, label='Finish', zorder=10)
    
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax