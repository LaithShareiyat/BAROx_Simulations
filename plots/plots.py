import numpy as np
import matplotlib.pyplot as plt
from models.track import Track
from models.vehicle import VehicleParams
from solver.metrics import channels

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

def plot_gg_diagram(track: Track, v: np.ndarray, vehicle: VehicleParams):
    """Plot g-g diagram showing tyre utilisation."""
    ax_long, ay_lat = channels(track, v)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ay_lat / vehicle.g, ax_long / vehicle.g, s=1, alpha=0.5, label='Operating points')
    
    # Friction circle
    theta = np.linspace(0, 2 * np.pi, 100)
    mu = vehicle.tyre.mu
    ax.plot(mu * np.cos(theta), mu * np.sin(theta), 'r--', lw=2, label=f'Friction limit (μ={mu})')
    
    ax.set_xlabel('$a_y / g$ (lateral)')
    ax.set_ylabel('$a_x / g$ (longitudinal)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title('g-g Diagram')
    plt.show()
    return fig, ax