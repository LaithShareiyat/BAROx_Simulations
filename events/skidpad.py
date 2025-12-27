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

    This is for visualisation purposes. For timing, use build_skidpad_track().

    Layout:
    - Left circle: centred at (0, 0)
    - Right circle: centred at (18.25, 0) = (2 * SKIDPAD_CENTRE_RADIUS, 0)
    - Circles touch at (9.125, 0)

    Args:
        n_points_per_circle: Number of points per circle

    Returns:
        Track object representing full figure-8 run
    """
    # Circle centres
    left_centre = (0.0, 0.0)
    right_centre = (2 * SKIDPAD_CENTRE_RADIUS, 0.0)  # (18.25, 0)

    # Entry straight (15m approach) - approaching from left side
    entry_length = 15.0
    n_entry = 15
    x_entry = np.linspace(-SKIDPAD_CENTRE_RADIUS - entry_length, -SKIDPAD_CENTRE_RADIUS, n_entry)
    y_entry = np.zeros(n_entry)

    # Left circle - 2 laps (counter-clockwise starting from left side)
    # Start at (-R, 0), go counter-clockwise
    theta_left = np.linspace(np.pi, np.pi + 4*np.pi, 2 * n_points_per_circle, endpoint=False)
    x_left = left_centre[0] + SKIDPAD_CENTRE_RADIUS * np.cos(theta_left)
    y_left = left_centre[1] + SKIDPAD_CENTRE_RADIUS * np.sin(theta_left)

    # Right circle - 2 laps (clockwise starting from the touching point)
    # Start at (R, 0) of left circle = touching point, go clockwise on right circle
    theta_right = np.linspace(np.pi, np.pi - 4*np.pi, 2 * n_points_per_circle, endpoint=False)
    x_right = right_centre[0] + SKIDPAD_CENTRE_RADIUS * np.cos(theta_right)
    y_right = right_centre[1] + SKIDPAD_CENTRE_RADIUS * np.sin(theta_right)

    # Exit straight (15m) - exiting from the touching point back to left
    exit_length = 15.0
    n_exit = 15
    x_exit = np.linspace(SKIDPAD_CENTRE_RADIUS, SKIDPAD_CENTRE_RADIUS + exit_length, n_exit)
    y_exit = np.zeros(n_exit)

    # Combine all segments
    x = np.concatenate([x_entry, x_left, x_right, x_exit])
    y = np.concatenate([y_entry, y_left, y_right, y_exit])

    return from_xy(x, y, closed=False)


def plot_skidpad_full(title: str = "Formula Student Skidpad - Full Layout"):
    """
    Plot the full skidpad layout showing both circles with entry/exit lanes.

    Layout:
    - Left circle: centred at (0, 0)
    - Right circle: centred at (18.25, 0)
    - Entry lane: from bottom, between x = 7.625 and x = 10.625
    - Exit lane: to top, between x = 7.625 and x = 10.625

    Run order: Enter from bottom -> 2 laps left circle -> 2 laps right circle -> Exit top

    Args:
        title: Plot title

    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    # Circle centres
    left_centre = (0.0, 0.0)
    right_centre = (2 * SKIDPAD_CENTRE_RADIUS, 0.0)  # (18.25, 0)

    # Lane boundaries (vertical lanes between circles)
    lane_left_x = SKIDPAD_INNER_RADIUS    # 7.625
    lane_right_x = SKIDPAD_OUTER_RADIUS   # 10.625
    lane_centre_x = SKIDPAD_CENTRE_RADIUS  # 9.125

    # Lane extends slightly beyond circles for entry/exit
    lane_extension = 3.0  # Short extension beyond circles
    lane_bottom_y = -SKIDPAD_OUTER_RADIUS - lane_extension
    lane_top_y = SKIDPAD_OUTER_RADIUS + lane_extension

    theta = np.linspace(0, 2*np.pi, 100)

    # Left circle boundaries and centre line
    ax.plot(left_centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta),
            left_centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta),
            'k-', lw=2, label='Track boundaries')
    ax.plot(left_centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta),
            left_centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta),
            'k-', lw=2)
    ax.plot(left_centre[0] + SKIDPAD_CENTRE_RADIUS * np.cos(theta),
            left_centre[1] + SKIDPAD_CENTRE_RADIUS * np.sin(theta),
            'b--', lw=1.5, alpha=0.7, label='Centre line (driving line)')

    # Right circle boundaries and centre line
    ax.plot(right_centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta),
            right_centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta),
            'k-', lw=2)
    ax.plot(right_centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta),
            right_centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta),
            'k-', lw=2)
    ax.plot(right_centre[0] + SKIDPAD_CENTRE_RADIUS * np.cos(theta),
            right_centre[1] + SKIDPAD_CENTRE_RADIUS * np.sin(theta),
            'b--', lw=1.5, alpha=0.7)

    # Continuous lane through the middle (from entry bottom to exit top)
    # Boundaries and centre line extend through where circles meet
    ax.plot([lane_left_x, lane_left_x], [lane_bottom_y, lane_top_y], 'k-', lw=2)
    ax.plot([lane_right_x, lane_right_x], [lane_bottom_y, lane_top_y], 'k-', lw=2)
    ax.plot([lane_centre_x, lane_centre_x], [lane_bottom_y, lane_top_y], 'b--', lw=1.5, alpha=0.7)

    # Shade circle track areas
    for centre in [left_centre, right_centre]:
        theta_fill = np.linspace(0, 2*np.pi, 100)
        x_outer = centre[0] + SKIDPAD_OUTER_RADIUS * np.cos(theta_fill)
        y_outer = centre[1] + SKIDPAD_OUTER_RADIUS * np.sin(theta_fill)
        x_inner = centre[0] + SKIDPAD_INNER_RADIUS * np.cos(theta_fill[::-1])
        y_inner = centre[1] + SKIDPAD_INNER_RADIUS * np.sin(theta_fill[::-1])
        ax.fill(np.concatenate([x_outer, x_inner]),
                np.concatenate([y_outer, y_inner]),
                color='gray', alpha=0.2)

    # Shade the continuous lane (from bottom to top through middle)
    ax.fill([lane_left_x, lane_right_x, lane_right_x, lane_left_x],
            [lane_bottom_y, lane_bottom_y, lane_top_y, lane_top_y],
            color='gray', alpha=0.2)

    # Mark circle centres
    ax.plot(*left_centre, 'rx', markersize=10, mew=2, label='Circle centres')
    ax.plot(*right_centre, 'rx', markersize=10, mew=2)
    ax.annotate('Left circle\n(0, 0)', left_centre, textcoords="offset points",
                xytext=(0, -25), ha='center', fontsize=9)
    ax.annotate('Right circle\n(18.25, 0)', right_centre, textcoords="offset points",
                xytext=(0, -25), ha='center', fontsize=9)

    # Mark entry point (bottom of lane)
    entry_point = (lane_centre_x, lane_bottom_y)
    ax.plot(*entry_point, 'g^', markersize=12, label='Entry', zorder=10)
    ax.annotate('ENTRY', entry_point, textcoords="offset points",
                xytext=(15, 0), fontsize=10, fontweight='bold', color='green')

    # Mark exit point (top of lane)
    exit_point = (lane_centre_x, lane_top_y)
    ax.plot(*exit_point, 'rs', markersize=12, label='Exit', zorder=10)
    ax.annotate('EXIT', exit_point, textcoords="offset points",
                xytext=(15, 0), fontsize=10, fontweight='bold', color='red')

    # Track info
    info_text = (f'Centre line radius: {SKIDPAD_CENTRE_RADIUS:.3f} m\n'
                 f'Inner boundary: {SKIDPAD_INNER_RADIUS:.3f} m\n'
                 f'Outer boundary: {SKIDPAD_OUTER_RADIUS:.3f} m\n'
                 f'Track width: {TRACK_WIDTH:.1f} m\n'
                 f'Circle separation: {2*SKIDPAD_CENTRE_RADIUS:.2f} m')

    ax.text(0.02, 0.02, info_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
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
    track = plot_skidpad_full()
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