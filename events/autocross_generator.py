import numpy as np
from models.track import Track, from_xy

def build_autocross_track(left: np.ndarray, right: np.ndarray) -> "Track":
    """
    Build track from left/right curb positions.
    Centerline is midpoint between curb pairs.
    """
    x_center = 0.5 * (left[:, 0] + right[:, 0])
    y_center = 0.5 * (left[:, 1] + right[:, 1])
    return from_xy(x_center, y_center, closed=False)

def build_autocross_from_segments(segments: list[dict]) -> Track:
    """
    Build track from segment definitions.
    
    Segment types:
    - {"type": "straight", "length": L}
    - {"type": "arc", "radius": R, "angle_deg": theta}
      (positive angle = left turn, negative = right turn)
    
    Args:
        segments: List of segment dictionaries
    
    Returns:
        Track object
    """
    x, y = [0.0], [0.0]
    heading = 0.0  # Current heading angle [rad]
    
    for seg in segments:
        if seg["type"] == "straight":
            L = seg["length"]
            # Discretise straight into ~1m segments
            n_pts = max(int(L), 2)
            for i in range(1, n_pts + 1):
                x.append(x[-1] + (L / n_pts) * np.cos(heading))
                y.append(y[-1] + (L / n_pts) * np.sin(heading))
                
        elif seg["type"] == "arc":
            R = seg["radius"]
            theta = np.radians(seg["angle_deg"])
            
            # Discretise arc into ~0.5m segments
            arc_length = abs(R * theta)
            n_pts = max(int(arc_length / 0.5), 10)
            
            # Find centre of arc
            sign = np.sign(theta)  # +1 for left turn, -1 for right
            cx = x[-1] - sign * R * np.sin(heading)
            cy = y[-1] + sign * R * np.cos(heading)
            
            # Generate arc points
            angles = np.linspace(0, theta, n_pts + 1)[1:]
            for a in angles:
                x.append(cx + sign * R * np.sin(heading + a))
                y.append(cy - sign * R * np.cos(heading + a))
            
            heading += theta
    
    return from_xy(np.array(x), np.array(y), closed=False)