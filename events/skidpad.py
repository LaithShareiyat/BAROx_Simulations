import numpy as np
import matplotlib.pyplot as plt
from models.track import Track, from_xy

# Formula Student Skidpad Dimensions
INNER_CURB_RADIUS = 7.625    # Inner circle curb radius [m]
OUTER_CURB_RADIUS = 10.625   # Outer circle curb radius [m] (3m track width)
CENTRE_LINE_RADIUS = (INNER_CURB_RADIUS + OUTER_CURB_RADIUS) / 2  # 9.125 m

def build_skidpad_track(radius: float = CENTRE_LINE_RADIUS, n_points: int = 100) -> Track:
    """
    Generate a circular skidpad track.
    
    Formula Student skidpad:
    - Inner curb diameter: 15.25 m (radius 7.625 m)
    - Outer curb diameter: 21.25 m (radius 10.625 m)
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
    Plot the skidpad track with optional velocity colouring.
    
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
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot cone boundaries
    if show_cones:
        theta = np.linspace(0, 2 * np.pi, 100)
        
        # Inner cones (red)
        x_inner = INNER_CURB_RADIUS * np.cos(theta)
        y_inner = INNER_CURB_RADIUS * np.sin(theta)
        ax.plot(x_inner, y_inner, 'r-', lw=2, label=f'Inner cones (R={INNER_CURB_RADIUS}m)')
        
        # Outer cones (blue)
        x_outer = OUTER_CURB_RADIUS * np.cos(theta)
        y_outer = OUTER_CURB_RADIUS * np.sin(theta)
        ax.plot(x_outer, y_outer, 'b-', lw=2, label=f'Outer cones (R={OUTER_CURB_RADIUS}m)')
        
        # Shade track area
        ax.fill_between(theta, 0, 1, alpha=0.1)  # Placeholder
        theta_fill = np.linspace(0, 2 * np.pi, 100)
        for i in range(len(theta_fill) - 1):
            ax.fill([INNER_CURB_RADIUS * np.cos(theta_fill[i]), 
                     OUTER_CURB_RADIUS * np.cos(theta_fill[i]),
                     OUTER_CURB_RADIUS * np.cos(theta_fill[i+1]), 
                     INNER_CURB_RADIUS * np.cos(theta_fill[i+1])],
                    [INNER_CURB_RADIUS * np.sin(theta_fill[i]), 
                     OUTER_CURB_RADIUS * np.sin(theta_fill[i]),
                     OUTER_CURB_RADIUS * np.sin(theta_fill[i+1]), 
                     INNER_CURB_RADIUS * np.sin(theta_fill[i+1])],
                    color='gray', alpha=0.2)
    
    # Plot driving line
    if v is not None:
        sc = ax.scatter(track.x, track.y, c=v, cmap='RdYlGn', s=20, zorder=5)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Speed [m/s]')
        ax.plot(track.x, track.y, 'k--', lw=1, alpha=0.3, label=f'Driving line (R={CENTRE_LINE_RADIUS}m)')
    else:
        ax.plot(track.x, track.y, 'g-', lw=3, label=f'Driving line (R={CENTRE_LINE_RADIUS}m)')
    
    # Mark start/finish
    ax.plot(track.x[0], track.y[0], 'go', markersize=12, label='Start', zorder=10)
    ax.annotate('START', (track.x[0], track.y[0]), textcoords="offset points", 
                xytext=(10, 10), fontsize=10, fontweight='bold')
    
    # Centre point
    ax.plot(0, 0, 'k+', markersize=15, mew=2)
    
    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits with padding
    limit = OUTER_CURB_RADIUS * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


if __name__ == "__main__":
    # Test the skidpad plotting
    track = build_skidpad_track()
    print(f"Track length: {track.s[-1]:.2f} m")
    print(f"Curvature: {track.kappa[0]:.4f} m⁻¹")
    plot_skidpad(track)