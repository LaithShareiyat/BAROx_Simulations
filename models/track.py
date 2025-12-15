from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Track:
    s: np.ndarray       # (N,)
    ds: np.ndarray      # (N-1,)
    kappa: np.ndarray   # (N,)
    x: np.ndarray       # (N,)
    y: np.ndarray       # (N,)

def from_xy(x: np.ndarray, y: np.ndarray, closed: bool = True) -> Track:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if closed:
        # ensure closed loop
        if (x[0] != x[-1]) or (y[0] != y[-1]):
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx*dx + dy*dy)
    s = np.r_[0.0, np.cumsum(ds)]

    # heading and curvature
    psi = np.arctan2(dy, dx)
    psi = np.unwrap(psi)

    # psi is length N-1; map to N by padding endpoints
    dpsi = np.diff(psi)
    ds_mid = 0.5*(ds[:-1] + ds[1:])
    kappa_mid = dpsi / np.maximum(ds_mid, 1e-9)

    kappa = np.zeros_like(x)
    kappa[1:-1] = kappa_mid
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]

    return Track(s=s, ds=ds, kappa=kappa, x=x, y=y)