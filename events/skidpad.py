import numpy as np
import matplotlib.pyplot as plt
from models.track import Track, from_xy

# Formula Student Skidpad Dimensions
INNER_CONE_RADIUS = 7.625    # Inner circle cone radius [m]
OUTER_CONE_RADIUS = 10.625   # Outer circle cone radius [m] (3m track width)
CENTRE_LINE_RADIUS = (INNER_CONE_RADIUS + OUTER_CONE_RADIUS) / 2  # 9.125 m
RIGHT_CIRCLE_CENTRE = (18.25, 0)  # Centre of right circle [m]
LEFT_CIRCLE_CENTRE = (0, 0)       # Centre of left circle [m]

# Entry/Exit Lane Dimensions
ENTRY_LANE_LEFT_X = 7.625     # Left curb of entry/exit lane [m]
ENTRY_LANE_RIGHT_X = 10.625   # Right curb of entry/exit lane [m]
ENTRY_LANE_CENTRE_X = (ENTRY_LANE_LEFT_X + ENTRY_LANE_RIGHT_X) / 2  # 9.125 m

def build_skidpad_track(radius: float = CENTRE_LINE_RADIUS, n_points: int = 100) -> Track:
    """
    Generate a circular skidpad track (single circle for lap time calculation).
    
    Formula Student skidpad:
    - Inner cone diameter: 15.25 m (radius 7.625 m)
    - Outer cone diameter: 21.25 m (radius 10.625 m)
    - Track width: 3 m
    - Driving line: Centre of track (radius 9.125 m)
    - Car drives figure-8 pattern
    - Timed on single circles
    
    Args:
        radius: Circle radius [m] (default 9.125 m for centre line)
        n_points: Number of discretisation points
    
    Returns:
        Circular Track object
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    return from_xy(x, y, closed=True)


def plot_skidpad(track: Track = None, v: np.ndarray = None, 
                 show_cones: bool = True, title: str = "Formula Student Skidpad"):
    """
    Plot the complete skidpad layout (figure-8) with optional velocity colouring.
    
    Args:
        track: Track object (if None, builds default track)
        v: Velocity array for colouring (optional)
        show_cones: Whether to show inner/outer cone boundaries
        title: Plot title
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    if track is None:
        track = build_skidpad_track()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    theta = np.linspace(0, 2 * np.pi, 100)
    
    # Plot cone boundaries for both circles
    if show_cones:
        for centre, label_prefix in [(LEFT_CIRCLE_CENTRE, 'Left'), 
                                      (RIGHT_CIRCLE_CENTRE, 'Right')]:
            cx, cy = centre
            
            # Inner cones (red)
            x_inner = cx + INNER_CONE_RADIUS * np.cos(theta)
            y_inner = cy + INNER_CONE_RADIUS * np.sin(theta)
            ax.plot(x_inner, y_inner, 'r-', lw=2, 
                    label=f'Inner cones (R={INNER_CONE_RADIUS}m)' if centre == LEFT_CIRCLE_CENTRE else None)
            
            # Outer cones (blue)
            x_outer = cx + OUTER_CONE_RADIUS * np.cos(theta)
            y_outer = cy + OUTER_CONE_RADIUS * np.sin(theta)
            ax.plot(x_outer, y_outer, 'b-', lw=2,
                    label=f'Outer cones (R={OUTER_CONE_RADIUS}m)' if centre == LEFT_CIRCLE_CENTRE else None)
            
            # Shade track area
            theta_fill = np.linspace(0, 2 * np.pi, 100)
            for i in range(len(theta_fill) - 1):
                ax.fill([cx + INNER_CONE_RADIUS * np.cos(theta_fill[i]), 
                         cx + OUTER_CONE_RADIUS * np.cos(theta_fill[i]),
                         cx + OUTER_CONE_RADIUS * np.cos(theta_fill[i+1]), 
                         cx + INNER_CONE_RADIUS * np.cos(theta_fill[i+1])],
                        [cy + INNER_CONE_RADIUS * np.sin(theta_fill[i]), 
                         cy + OUTER_CONE_RADIUS * np.sin(theta_fill[i]),
                         cy + OUTER_CONE_RADIUS * np.sin(theta_fill[i+1]), 
                         cy + INNER_CONE_RADIUS * np.sin(theta_fill[i+1])],
                        color='gray', alpha=0.2)
            
            # Centre point marker
            ax.plot(cx, cy, 'k+', markersize=15, mew=2)
            ax.annotate(f'{label_prefix} Centre', (cx, cy), textcoords="offset points",
                        xytext=(5, -15), fontsize=9, ha='center')
    
    # =========================================
    # Entry/Exit Lane
    # =========================================
    # Calculate lane extent (from bottom of plot to top)
    lane_y_min = -OUTER_CONE_RADIUS - 1
    lane_y_max = OUTER_CONE_RADIUS + 1
    
    # Left curb (grey solid line)
    ax.plot([ENTRY_LANE_LEFT_X, ENTRY_LANE_LEFT_X], [lane_y_min, lane_y_max], 
            color='grey', lw=2, linestyle='-', label='Entry/Exit lane curbs')
    
    # Right curb (grey solid line)
    ax.plot([ENTRY_LANE_RIGHT_X, ENTRY_LANE_RIGHT_X], [lane_y_min, lane_y_max], 
            color='grey', lw=2, linestyle='-')
    
    # Centre driving line (green dashed)
    ax.plot([ENTRY_LANE_CENTRE_X, ENTRY_LANE_CENTRE_X], [lane_y_min, lane_y_max], 
            color='green', lw=2, linestyle='--', alpha=0.7, label='Entry/Exit driving line')
    
    # Shade entry/exit lane area
    ax.fill_betweenx([lane_y_min, lane_y_max], 
                     ENTRY_LANE_LEFT_X, ENTRY_LANE_RIGHT_X,
                     color='gray', alpha=0.1)
    
    # =========================================
    # Plot driving line for left circle (from track object)
    # =========================================
    if v is not None:
        sc = ax.scatter(track.x, track.y, c=v, cmap='RdYlGn', s=20, zorder=5)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Speed [m/s]')
        ax.plot(track.x, track.y, 'k--', lw=1, alpha=0.3)
    else:
        ax.plot(track.x, track.y, 'g-', lw=3, label=f'Driving line (R={CENTRE_LINE_RADIUS}m)')
    
    # Plot driving line for right circle (mirrored)
    x_right = RIGHT_CIRCLE_CENTRE[0] + CENTRE_LINE_RADIUS * np.cos(theta)
    y_right = RIGHT_CIRCLE_CENTRE[1] + CENTRE_LINE_RADIUS * np.sin(theta)
    if v is not None:
        ax.scatter(x_right, y_right, c=v, cmap='RdYlGn', s=20, zorder=5)
        ax.plot(x_right, y_right, 'k--', lw=1, alpha=0.3)
    else:
        ax.plot(x_right, y_right, 'g-', lw=3)
    
    # Mark start/finish on left circle
    ax.plot(track.x[0], track.y[0], 'go', markersize=12, label='Start/Finish', zorder=10)
    ax.annotate('START', (track.x[0], track.y[0]), textcoords="offset points", 
                xytext=(10, 10), fontsize=10, fontweight='bold')
    
    # =========================================
    # Formatting
    # =========================================
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with padding
    x_min = LEFT_CIRCLE_CENTRE[0] - OUTER_CONE_RADIUS - 2
    x_max = RIGHT_CIRCLE_CENTRE[0] + OUTER_CONE_RADIUS + 2
    y_limit = OUTER_CONE_RADIUS + 2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_limit, y_limit)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    # Test the skidpad plotting
    track = build_skidpad_track()
    print(f"Track length: {track.s[-1]:.2f} m")
    print(f"Curvature: {track.kappa[0]:.4f} m⁻¹")
    print(f"Left circle centre: {LEFT_CIRCLE_CENTRE}")
    print(f"Right circle centre: {RIGHT_CIRCLE_CENTRE}")
    print(f"Entry lane: x = {ENTRY_LANE_LEFT_X}m to x = {ENTRY_LANE_RIGHT_X}m")
    plot_skidpad(track)