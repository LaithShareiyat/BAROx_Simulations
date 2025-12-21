import numpy as np
import matplotlib.pyplot as plt
from models.track import Track, from_xy

# Formula Student Autocross Rules
MAX_TRACK_LENGTH = 1500.0               # Maximum track length [m]
MAX_STRAIGHT_LENGTH = 80.0              # Maximum single straight segment [m]
TRACK_WIDTH = 3.0                       # Track width [m]
MIN_HAIRPIN_OUTER_RADIUS = 9.0          # Minimum hairpin outside radius [m]
MAX_REGULAR_TURN_DIAMETER = 50.0        # Maximum regular turn diameter [m]
SLALOM_CONE_SEPARATION = (7.5, 12.0)    # Slalom cone separation range [m]


def build_straight(length: float, heading: float, start_x: float, start_y: float, 
                   ds: float = 1.0) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Generate a straight segment.
    
    Args:
        length: Segment length [m] (must be <= 80m per rules)
        heading: Current heading angle [rad]
        start_x, start_y: Starting coordinates [m]
        ds: Discretisation step [m]
    
    Returns:
        x, y arrays, end_x, end_y
    """
    if length > MAX_STRAIGHT_LENGTH:
        raise ValueError(f"Straight length {length}m exceeds maximum {MAX_STRAIGHT_LENGTH}m")
    
    n_pts = max(int(length / ds), 2)
    distances = np.linspace(0, length, n_pts)[1:]  # Exclude start point
    
    x = start_x + distances * np.cos(heading)
    y = start_y + distances * np.sin(heading)
    
    end_x = start_x + length * np.cos(heading)
    end_y = start_y + length * np.sin(heading)
    
    return x, y, end_x, end_y


def build_arc(radius: float, angle_deg: float, heading: float, 
              start_x: float, start_y: float, ds: float = 0.5) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Generate an arc segment.
    
    Args:
        radius: Arc radius [m]
        angle_deg: Arc angle [degrees] (positive = left turn, negative = right turn)
        heading: Current heading angle [rad]
        start_x, start_y: Starting coordinates [m]
        ds: Discretisation step [m]
    
    Returns:
        x, y arrays, end_x, end_y, new_heading
    """
    theta = np.radians(angle_deg)
    arc_length = abs(radius * theta)
    n_pts = max(int(arc_length / ds), 10)
    
    sign = np.sign(theta)  # +1 for left turn, -1 for right turn
    
    # Find centre of arc
    cx = start_x - sign * radius * np.sin(heading)
    cy = start_y + sign * radius * np.cos(heading)
    
    # Generate arc points
    angles = np.linspace(0, theta, n_pts + 1)[1:]  # Exclude start point
    x = cx + sign * radius * np.sin(heading + angles)
    y = cy - sign * radius * np.cos(heading + angles)
    
    new_heading = heading + theta
    end_x = x[-1] if len(x) > 0 else start_x
    end_y = y[-1] if len(y) > 0 else start_y
    
    return x, y, end_x, end_y, new_heading


def build_hairpin(outer_radius: float, direction: str, heading: float,
                  start_x: float, start_y: float, ds: float = 0.5) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Generate a hairpin turn (180-degree turn).
    
    Args:
        outer_radius: Outside radius [m] (must be ≥ 9m per rules)
        direction: 'left' or 'right'
        heading: Current heading angle [rad]
        start_x, start_y: Starting coordinates [m]
        ds: Discretisation step [m]
    
    Returns:
        x, y arrays, end_x, end_y, new_heading
    """
    if outer_radius < MIN_HAIRPIN_OUTER_RADIUS:
        raise ValueError(f"Hairpin outer radius {outer_radius}m is less than minimum {MIN_HAIRPIN_OUTER_RADIUS}m")
    
    # Driving line radius is outer_radius - half track width
    centre_radius = outer_radius - TRACK_WIDTH / 2
    
    angle_deg = 180.0 if direction == 'left' else -180.0
    
    return build_arc(centre_radius, angle_deg, heading, start_x, start_y, ds)


def build_regular_turn(diameter: float, angle_deg: float, heading: float,
                       start_x: float, start_y: float, ds: float = 0.5) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Generate a regular turn.
    
    Args:
        diameter: Turn diameter [m] (must be <= 50m per rules)
        angle_deg: Turn angle [degrees]
        heading: Current heading angle [rad]
        start_x, start_y: Starting coordinates [m]
        ds: Discretisation step [m]
    
    Returns:
        x, y arrays, end_x, end_y, new_heading
    """
    if diameter > MAX_REGULAR_TURN_DIAMETER:
        raise ValueError(f"Turn diameter {diameter}m exceeds maximum {MAX_REGULAR_TURN_DIAMETER}m")
    
    radius = diameter / 2
    
    return build_arc(radius, angle_deg, heading, start_x, start_y, ds)


def build_slalom(n_cones: int, cone_separation: float, heading: float,
                 start_x: float, start_y: float, ds: float = 0.5) -> tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray, np.ndarray]:
    """
    Generate a slalom section (weaving between cones placed on the centreline).
    
    The cones are placed along the centreline at equal intervals.
    The car weaves left-right-left-right around them.
    
    Args:
        n_cones: Number of cones (typically 4 per rules)
        cone_separation: Distance between cones [m] (7.5 - 12m per rules)
        heading: Current heading angle [rad]
        start_x, start_y: Starting coordinates [m]
        ds: Discretisation step [m]
    
    Returns:
        x, y arrays, end_x, end_y, new_heading, cone_x, cone_y
    """
    if not (SLALOM_CONE_SEPARATION[0] <= cone_separation <= SLALOM_CONE_SEPARATION[1]):
        raise ValueError(f"Cone separation {cone_separation}m outside allowed range {SLALOM_CONE_SEPARATION}")
    
    # Calculate cone positions along centreline
    cone_positions = np.arange(n_cones) * cone_separation
    cone_x = start_x + cone_positions * np.cos(heading)
    cone_y = start_y + cone_positions * np.sin(heading)
    
    # Generate slalom path using sinusoidal weave
    total_length = (n_cones - 1) * cone_separation
    n_pts = max(int(total_length / ds), 50)
    
    # Distance along centreline
    s = np.linspace(0, total_length, n_pts)[1:]  # Exclude start point
    
    # Lateral offset: sinusoidal weave with period = 2 * cone_separation
    # Amplitude is slightly less than half track width to stay within bounds
    # TODO The actual slalom section is more open than 3m and allows for larger amplitudes
    amplitude = TRACK_WIDTH / 2 * 0.7
    # Phase shift so we pass cones on alternating sides
    # First cone: pass on right (negative offset), second: left, etc.
    lateral_offset = amplitude * np.sin(np.pi * s / cone_separation - np.pi/2)
    
    # Convert to global coordinates
    perp_x = -np.sin(heading)
    perp_y = np.cos(heading)
    
    x = start_x + s * np.cos(heading) + lateral_offset * perp_x
    y = start_y + s * np.sin(heading) + lateral_offset * perp_y
    
    end_x = start_x + total_length * np.cos(heading)
    end_y = start_y + total_length * np.sin(heading)
    
    # Heading remains unchanged (straight slalom)
    return x, y, end_x, end_y, heading, cone_x, cone_y


def build_chicane(width: float, length: float, heading: float,
                  start_x: float, start_y: float, ds: float = 0.5) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Generate a chicane (quick left-right or right-left combination).
    
    Args:
        width: Lateral offset of chicane [m]
        length: Total chicane length [m]
        heading: Current heading angle [rad]
        start_x, start_y: Starting coordinates [m]
        ds: Discretisation step [m]
    
    Returns:
        x, y arrays, end_x, end_y, new_heading (same as input heading)
    """
    x_list, y_list = [], []
    current_x, current_y = start_x, start_y
    current_heading = heading
    
    # Chicane is two opposite arcs
    arc_length = length / 2
    arc_radius = (arc_length**2 + width**2) / (2 * width) if width > 0 else arc_length
    arc_angle_rad = 2 * np.arcsin(min(arc_length / (2 * arc_radius), 1.0)) if arc_radius > 0 else 0
    arc_angle_deg = np.degrees(arc_angle_rad)
    
    # First arc (e.g., left)
    x1, y1, current_x, current_y, current_heading = build_arc(
        arc_radius, arc_angle_deg, current_heading, current_x, current_y, ds
    )
    x_list.extend(x1)
    y_list.extend(y1)
    
    # Second arc (opposite direction, e.g., right)
    x2, y2, current_x, current_y, current_heading = build_arc(
        arc_radius, -arc_angle_deg, current_heading, current_x, current_y, ds
    )
    x_list.extend(x2)
    y_list.extend(y2)
    
    return np.array(x_list), np.array(y_list), current_x, current_y, current_heading


def build_autocross_from_segments(segments: list[dict]) -> tuple[Track, dict]:
    """
    Build track from segment definitions.
    
    Segment types:
    - {"type": "straight", "length": L}
    - {"type": "arc", "radius": R, "angle_deg": theta}
    - {"type": "hairpin", "outer_radius": R, "direction": "left"/"right"}
    - {"type": "regular_turn", "diameter": D, "angle_deg": theta}
    - {"type": "slalom", "n_cones": N, "cone_separation": S}
    - {"type": "chicane", "width": W, "length": L}
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        Track object, metadata dict (including slalom cone positions)
    """
    x_list, y_list = [0.0], [0.0]
    heading = 0.0
    current_x, current_y = 0.0, 0.0
    
    # Store slalom cone positions for plotting
    slalom_cones_x = []
    slalom_cones_y = []
    
    for seg in segments:
        seg_type = seg["type"]
        
        if seg_type == "straight":
            x, y, current_x, current_y = build_straight(
                seg["length"], heading, current_x, current_y
            )
            x_list.extend(x)
            y_list.extend(y)
            
        elif seg_type == "arc":
            x, y, current_x, current_y, heading = build_arc(
                seg["radius"], seg["angle_deg"], heading, current_x, current_y
            )
            x_list.extend(x)
            y_list.extend(y)
            
        elif seg_type == "hairpin":
            x, y, current_x, current_y, heading = build_hairpin(
                seg["outer_radius"], seg["direction"], heading, current_x, current_y
            )
            x_list.extend(x)
            y_list.extend(y)
            
        elif seg_type == "regular_turn":
            x, y, current_x, current_y, heading = build_regular_turn(
                seg["diameter"], seg["angle_deg"], heading, current_x, current_y
            )
            x_list.extend(x)
            y_list.extend(y)
            
        elif seg_type == "slalom":
            x, y, current_x, current_y, heading, cone_x, cone_y = build_slalom(
                seg["n_cones"], seg["cone_separation"], heading, current_x, current_y
            )
            x_list.extend(x)
            y_list.extend(y)
            slalom_cones_x.extend(cone_x)
            slalom_cones_y.extend(cone_y)
            
        elif seg_type == "chicane":
            x, y, current_x, current_y, heading = build_chicane(
                seg["width"], seg["length"], heading, current_x, current_y
            )
            x_list.extend(x)
            y_list.extend(y)
            
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")
    
    metadata = {
        "slalom_cones_x": np.array(slalom_cones_x),
        "slalom_cones_y": np.array(slalom_cones_y),
    }
    
    return from_xy(np.array(x_list), np.array(y_list), closed=False), metadata


def build_standard_autocross() -> tuple[Track, dict]:
    """
    Build a Formula Student autocross track based on specific segment definitions.
    
    Track layout defined segment by segment.
    
    Returns:
        Track object, metadata dict
    """
    
    # Default parameters for turns
    TURN_RADIUS = 15.0          # Default turn radius [m] (diameter 30m)
    HAIRPIN_OUTER_RADIUS = 10.0  # Hairpin outer radius [m]
    CHICANE_WIDTH = 3.0         # Chicane lateral offset [m]
    CHICANE_LENGTH = 20.0       # Chicane total length [m]
    
    segments = [
        {"type": "straight", "length": 80},
        
        {"type": "chicane", "width": CHICANE_WIDTH, "length": CHICANE_LENGTH},
        
        {"type": "straight", "length": 80},
        
        {"type": "arc", "radius": TURN_RADIUS, "angle_deg": -110},
        
        {"type": "straight", "length": 30},
        
        {"type": "hairpin", "outer_radius": HAIRPIN_OUTER_RADIUS, "direction": "right"},
        
        {"type": "straight", "length": 20},
        
        {"type": "hairpin", "outer_radius": HAIRPIN_OUTER_RADIUS, "direction": "left"},
        
        {"type": "straight", "length": 40},
        
        {"type": "hairpin", "outer_radius": HAIRPIN_OUTER_RADIUS, "direction": "right"},
        
        {"type": "straight", "length": 20},
        
        {"type": "hairpin", "outer_radius": HAIRPIN_OUTER_RADIUS, "direction": "left"},
        
        {"type": "straight", "length": 80},
        
        {"type": "arc", "radius": TURN_RADIUS, "angle_deg": -150},
        
        {"type": "straight", "length": 50},
        
        {"type": "arc", "radius": TURN_RADIUS, "angle_deg": 85},
        
        {"type": "straight", "length": 15},
        
        {"type": "arc", "radius": TURN_RADIUS, "angle_deg": 120}, 
        
        {"type": "straight", "length": 40},       
        
        {"type": "slalom", "n_cones": 4, "cone_separation": 12},
        
        {"type": "straight", "length": 60},
        
        {"type": "arc", "radius": TURN_RADIUS, "angle_deg": 160},
        
        {"type": "straight", "length": 50},    
        
        {"type": "arc", "radius": TURN_RADIUS, "angle_deg": -90},
        
        {"type": "chicane", "width": CHICANE_WIDTH, "length": CHICANE_LENGTH},
        
        {"type": "straight", "length": 60},
        
        {"type": "hairpin", "outer_radius": HAIRPIN_OUTER_RADIUS, "direction": "right"},
        
        {"type": "straight", "length": 60},
        
        {"type": "arc", "radius": TURN_RADIUS, "angle_deg": 90},
        
        {"type": "straight", "length": 20},
        
        {"type": "slalom", "n_cones": 4, "cone_separation": 12}, 
        
        {"type": "straight", "length": 20},
        
        {"type": "arc", "radius": TURN_RADIUS, "angle_deg": -110},
        
        {"type": "straight", "length": 20},
        
        {"type": "chicane", "width": CHICANE_WIDTH, "length": CHICANE_LENGTH},
        
        {"type": "chicane", "width": CHICANE_WIDTH, "length": CHICANE_LENGTH},
        
        {"type": "chicane", "width": CHICANE_WIDTH, "length": CHICANE_LENGTH},
        
        {"type": "straight", "length": 10},
        
        {"type": "hairpin", "outer_radius": HAIRPIN_OUTER_RADIUS, "direction": "left"},
        
        {"type": "straight", "length": 50},
        
        {"type": "arc", "radius": TURN_RADIUS, "angle_deg": 25},
        
        {"type": "chicane", "width": CHICANE_WIDTH, "length": CHICANE_LENGTH},
        
        {"type": "straight", "length": 50},
        
        {"type": "hairpin", "outer_radius": HAIRPIN_OUTER_RADIUS, "direction": "right"},
        
        {"type": "straight", "length": 10},
    ]
    
    return build_autocross_from_segments(segments)


def plot_autocross(track: Track, v: np.ndarray = None, metadata: dict = None,
                   title: str = "Formula Student Autocross"):
    """
    Plot the autocross track with velocity colouring.
    
    Args:
        track: Track object
        v: Velocity array for colouring (optional)
        metadata: Dictionary containing slalom cone positions etc.
        title: Plot title
    
    Returns:
        fig, ax: Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
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
        
        sc = ax.scatter(track.x, track.y, c=v_interp, cmap='RdYlGn', s=10, zorder=5)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Speed [m/s]')
    else:
        ax.plot(track.x, track.y, 'b-', lw=2, label='Centre line')
    
    # Mark start and finish
    ax.plot(track.x[0], track.y[0], 'go', markersize=15, label='Start', zorder=10)
    ax.plot(track.x[-1], track.y[-1], 'rs', markersize=15, label='Finish', zorder=10)
    ax.annotate('START', (track.x[0], track.y[0]), textcoords="offset points",
                xytext=(-20, -20), fontsize=12, fontweight='bold', color='green')
    ax.annotate('FINISH', (track.x[-1], track.y[-1]), textcoords="offset points",
                xytext=(-20, -20), fontsize=12, fontweight='bold', color='red')
    
    # Plot slalom cones
    if metadata is not None and len(metadata.get("slalom_cones_x", [])) > 0:
        ax.scatter(metadata["slalom_cones_x"], metadata["slalom_cones_y"], 
                   c='orange', s=100, marker='^', zorder=15, label='Slalom cones',
                   edgecolors='black', linewidths=1)
    
    # Plot track boundaries (offset by half track width)
    dx = np.gradient(track.x)
    dy = np.gradient(track.y)
    length = np.sqrt(dx**2 + dy**2)
    # Avoid division by zero
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
    
    # Shade track area
    ax.fill(np.concatenate([x_left, x_right[::-1]]),
            np.concatenate([y_left, y_right[::-1]]),
            color='gray', alpha=0.15)
    
    # Track info
    track_length = track.s[-1]
    if len(track.kappa) > 0 and np.any(track.kappa != 0):
        max_kappa = np.max(np.abs(track.kappa[track.kappa != 0]))
        min_radius = 1 / max_kappa if max_kappa > 0 else np.inf
    else:
        min_radius = np.inf
    
    info_text = (f'Track length: {track_length:.1f} m\n'
                 f'Track width: {TRACK_WIDTH} m\n')
    
    if v is not None:
        v_use = v_interp if 'v_interp' in dir() else v
        info_text += f'\nAvg speed: {np.mean(v_use):.1f} m/s'
        info_text += f'\nMax speed: {np.max(v_use):.1f} m/s'
    
    ax.text(0.02, 0.02, info_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax



def validate_autocross(track: Track, metadata: dict = None) -> dict:
    """
    Validate track against Formula Student rules.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check total length
    if track.s[-1] > MAX_TRACK_LENGTH:
        results["valid"] = False
        results["errors"].append(
            f"Track length {track.s[-1]:.1f}m exceeds maximum {MAX_TRACK_LENGTH}m"
        )
    
    # Check minimum turn radius
    max_kappa = np.max(np.abs(track.kappa[track.kappa != 0]))
    min_radius = 1 / max_kappa if max_kappa > 0 else np.inf
    if min_radius < (MIN_HAIRPIN_OUTER_RADIUS - TRACK_WIDTH / 2):
        results["warnings"].append(
            f"Minimum turn radius {min_radius:.2f}m may be too tight"
        )
    
    return results


if __name__ == "__main__":
    # Build and plot standard autocross
    print("BAROx Autocross Track Summary")
    track, metadata = build_standard_autocross()
    
    print("-" * 50)
    print(f"\nActual track length: {track.s[-1]:.1f} m")
    
    max_kappa = np.max(np.abs(track.kappa[track.kappa != 0]))
    print(f"Min turn radius: {1/max_kappa:.2f} m")
    print(f"Number of slalom cones: {len(metadata['slalom_cones_x'])}")
    
    # Validate track
    print("Validation:")
    validation = validate_autocross(track, metadata)
    if validation["valid"]:
        print("✓ Track passes all validation checks")
    else:
        print("✗ Track validation FAILED:")
        for err in validation["errors"]:
            print(f"  - {err}")
    
    for warn in validation.get("warnings", []):
        print(f"  Warning: {warn}")
    
    if track.s[-1] <= MAX_TRACK_LENGTH:
        print(f"✓ Track length {track.s[-1]:.1f}m ≤ {MAX_TRACK_LENGTH}m maximum")
    else:
        print(f"✗ Track length {track.s[-1]:.1f}m exceeds {MAX_TRACK_LENGTH}m maximum")
    
    plot_autocross(track, metadata=metadata)