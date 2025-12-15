import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.track import Track, from_xy

# Formula Student Skidpad Rules
SKIDPAD_CENTRE_RADIUS = 9.125  # Centre-line radius [m]
SKIDPAD_INNER_RADIUS = 7.625   # Inner boundary radius [m]  
SKIDPAD_OUTER_RADIUS = 10.625  # Outer boundary radius [m]
TRACK_WIDTH = 3.0              # Track width [m]

# Skidpad run structure:
# - Enter from straight
# - Right circle: 2 laps (first to settle, second is timed)
# - Left circle: 2 laps (first to settle, second is timed)
# - Exit to straight
# Official time = average of second lap on each circle


def build_skidpad_track(n_points: int = 100) -> Track:
    """
    Build a skidpad track (single circle for timing).
    
    The official skidpad time is for ONE circle only.
    The full run is: 2 laps right + 2 laps left, but only the 
    second lap on each side is timed (first lap settles dynamics).
    
    This function returns a single circle for lap time calculation.
    
    Args:
        n_points: Number of points for the circle
    
    Returns:
        Track object representing one circle
    """
    # Single circle (clockwise, matching right-side circle)
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = SKIDPAD_CENTRE_RADIUS * np.cos(theta)
    y = SKIDPAD_CENTRE_RADIUS * np.sin(theta)
    
    return from_xy(x, y, closed=True)


def build_skidpad_full_run(n_points_per_circle: int = 100) -> Track:
    """
    Build the full skidpad run (entry + 2 right + 2 left + exit).
    
    This is for visualization purposes. For timing, use build_skidpad_track().
    
    Args:
        n_points_per_circle: Number of points per circle
    
    Returns:
        Track object representing full figure-8 run
    """
    # Entry straight (15m approach)
    entry_length = 15.0
    n_entry = 15
    x_entry = np.linspace(-entry_length, 0, n_entry)
    y_entry = np.zeros(n_entry)
    
    # Right circle - 2 laps (clockwise, so negative angle)
    # Circle centre is at (SKIDPAD_CENTRE_RADIUS, 0)
    theta_right = np.linspace(np.pi, np.pi - 4*np.pi, 2 * n_points_per_circle, endpoint=False)
    x_right = SKIDPAD_CENTRE_RADIUS * np.cos(theta_right) + SKIDPAD_CENTRE_RADIUS
    y_right = SKIDPAD_CENTRE_RADIUS * np.sin(theta_right)
    
    # Left circle - 2 laps (counter-clockwise, positive angle)
    # Circle centre is at (-SKIDPAD_CENTRE_RADIUS, 0)
    theta_left = np.linspace(0, 4*np.pi, 2 * n_points_per_circle, endpoint=False)
    x_left = SKIDPAD_CENTRE_RADIUS * np.cos(theta_left) - SKIDPAD_CENTRE_RADIUS
    y_left = SKIDPAD_CENTRE_RADIUS * np.sin(theta_left)
    
    # Exit straight (15m)
    exit_length = 15.0
    n_exit = 15
    x_exit = np.linspace(0, -exit_length, n_exit)
    y_exit = np.zeros(n_exit)
    
    # Combine all segments
    x = np.concatenate([x_entry, x_right, x_left, x_exit])
    y = np.concatenate([y_entry, y_right, y_left, y_exit])
    
    return from_xy(x, y, closed=False)


def skidpad_time_from_single_circle(t_circle: float) -> dict:
    """
    Calculate official skidpad time from single circle time.
    
    In competition:
    - Run right circle twice, second lap is timed (t_right)
    - Run left circle twice, second lap is timed (t_left)
    - Official time = average = (t_right + t_left) / 2
    
    For simulation with identical left/right performance:
    - t_right = t_left = t_circle
    - Official time = t_circle
    
    Args:
        t_circle: Time for one circle [s]
    
    Returns:
        Dictionary with timing breakdown
    """
    return {
        't_right_circle': t_circle,
        't_left_circle': t_circle,
        't_official': t_circle,  # Average of both (same in simulation)
        't_full_run': t_circle * 4,  # Total time for all 4 circles
        'circle_length': 2 * np.pi * SKIDPAD_CENTRE_RADIUS,
        'avg_speed': 2 * np.pi * SKIDPAD_CENTRE_RADIUS / t_circle,
    }


def plot_skidpad(track: Track, v: np.ndarray = None, 
                 title: str = "Formula Student Skidpad"):
    """
    Plot the skidpad track with velocity colouring.
    
    Args:
        track: Track object
        v: Velocity array for colouring (optional)
        title: Plot title
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot centre line with velocity colouring if provided
    if v is not None:
        # Ensure v matches track length
        if len(v) != len(track.x):
            # Interpolate v to match track points
            from scipy.interpolate import interp1d
            s_v = np.linspace(0, track.s[-1], len(v))
            f = interp1d(s_v, v, kind='linear', fill_value='extrapolate')
            v_interp = f(track.s)
        else:
            v_interp = v
        
        sc = ax.scatter(track.x, track.y, c=v_interp, cmap='RdYlGn', s=30, zorder=5)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Speed [m/s]')
    else:
        ax.plot(track.x, track.y, 'b-', lw=2, label='Centre line')
        v_interp = None
    
    # Mark start/finish
    ax.plot(track.x[0], track.y[0], 'go', markersize=12, label='Start/Finish', zorder=10)
    ax.annotate('START', (track.x[0], track.y[0]), textcoords="offset points",
                xytext=(10, 10), fontsize=10, fontweight='bold', color='green')
    
    # Plot track boundaries
    dx = np.gradient(track.x)
    dy = np.gradient(track.y)
    length = np.sqrt(dx**2 + dy**2)
    length = np.maximum(length, 1e-9)
    nx = -dy / length
    ny = dx / length
    
    offset = TRACK_WIDTH / 2
    x_left = track.x + offset * nx
    y_left = track.y + offset * ny
    x_right = track.x - offset * nx
    y_right = track.y - offset * ny
    
    ax.plot(x_left, y_left, 'k-', lw=1, alpha=0.5)
    ax.plot(x_right, y_right, 'k-', lw=1, alpha=0.5)
    
    # Draw reference circle
    theta_circle = np.linspace(0, 2*np.pi, 100)
    x_ref = SKIDPAD_CENTRE_RADIUS * np.cos(theta_circle)
    y_ref = SKIDPAD_CENTRE_RADIUS * np.sin(theta_circle)
    ax.plot(x_ref, y_ref, 'r--', lw=1, alpha=0.3, label='Reference circle')
    
    # Mark centre
    ax.plot(0, 0, 'rx', markersize=10, label='Circle centre')
    
    # Track info
    track_length = track.s[-1]
    circle_circumference = 2 * np.pi * SKIDPAD_CENTRE_RADIUS
    
    info_text = (f'Circle circumference: {circle_circumference:.1f} m\n'
                 f'Track width: {TRACK_WIDTH} m\n'
                 f'Turn radius: {SKIDPAD_CENTRE_RADIUS:.2f} m')
    
    if v_interp is not None:
        avg_speed = np.mean(v_interp)
        info_text += f'\nAvg speed: {avg_speed:.1f} m/s ({avg_speed*3.6:.1f} km/h)'
        info_text += f'\nLap time: {circle_circumference/avg_speed:.3f} s'
    
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    # Test the skidpad track
    track = build_skidpad_track()
    circle_length = 2 * np.pi * SKIDPAD_CENTRE_RADIUS
    
    print("=" * 50)
    print("SKIDPAD TRACK")
    print("=" * 50)
    print(f"Centre-line radius: {SKIDPAD_CENTRE_RADIUS:.3f} m")
    print(f"Circle circumference: {circle_length:.2f} m")
    print(f"Track width: {TRACK_WIDTH} m")
    print(f"Inner radius: {SKIDPAD_INNER_RADIUS:.3f} m")
    print(f"Outer radius: {SKIDPAD_OUTER_RADIUS:.3f} m")
    print("=" * 50)
    print("\nNote: Official time is for ONE circle (second lap on each side)")
    print("Full run: Entry → 2× Right circle → 2× Left circle → Exit")
    
    plot_skidpad(track)