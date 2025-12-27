"""
Weight Transfer Calculations for Vehicle Dynamics.

This module calculates the vertical load (F_z) on each wheel during
acceleration, braking, and cornering manoeuvres.

Physics:
    Longitudinal weight transfer (acceleration/braking):
        ΔF_z,long = (m × a_x × h_cg) / L

    Lateral weight transfer (cornering):
        ΔF_z,lat = (m × a_y × h_cg) / t

    Where:
        m = vehicle mass [kg]
        a_x = longitudinal acceleration [m/s²] (positive = accelerating)
        a_y = lateral acceleration [m/s²] (positive = turning right)
        h_cg = centre of gravity height [m]
        L = wheelbase [m]
        t = track width [m]
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from models.vehicle import VehicleParams, VehicleGeometry


@dataclass
class WheelLoads:
    """Vertical loads on each wheel [N]."""
    FL: float  # Front Left
    FR: float  # Front Right
    RL: float  # Rear Left
    RR: float  # Rear Right

    @property
    def front_total(self) -> float:
        """Total front axle load [N]."""
        return self.FL + self.FR

    @property
    def rear_total(self) -> float:
        """Total rear axle load [N]."""
        return self.RL + self.RR

    @property
    def left_total(self) -> float:
        """Total left side load [N]."""
        return self.FL + self.RL

    @property
    def right_total(self) -> float:
        """Total right side load [N]."""
        return self.FR + self.RR

    @property
    def total(self) -> float:
        """Total vehicle weight [N]."""
        return self.FL + self.FR + self.RL + self.RR

    def as_dict(self) -> Dict[str, float]:
        """Return loads as dictionary."""
        return {'FL': self.FL, 'FR': self.FR, 'RL': self.RL, 'RR': self.RR}

    def as_array(self) -> np.ndarray:
        """Return loads as array [FL, FR, RL, RR]."""
        return np.array([self.FL, self.FR, self.RL, self.RR])


def calculate_static_loads(vehicle: VehicleParams) -> WheelLoads:
    """
    Calculate static wheel loads (vehicle at rest on level ground).

    Args:
        vehicle: Vehicle parameters

    Returns:
        WheelLoads with static vertical loads [N]
    """
    m = vehicle.m
    g = vehicle.g
    W = m * g  # Total weight

    if vehicle.geometry is not None:
        geo = vehicle.geometry
        # Load distribution based on CoG position
        W_front = W * geo.weight_distribution_front
        W_rear = W * geo.weight_distribution_rear
    else:
        # Default 50/50 distribution
        W_front = W * 0.5
        W_rear = W * 0.5

    return WheelLoads(
        FL=W_front / 2,
        FR=W_front / 2,
        RL=W_rear / 2,
        RR=W_rear / 2,
    )


def calculate_wheel_loads(vehicle: VehicleParams,
                          a_x: float = 0.0,
                          a_y: float = 0.0) -> WheelLoads:
    """
    Calculate wheel loads with weight transfer from acceleration.

    Sign conventions:
        a_x > 0: accelerating (load transfers rearward)
        a_x < 0: braking (load transfers forward)
        a_y > 0: turning right (load transfers to left wheels)
        a_y < 0: turning left (load transfers to right wheels)

    Args:
        vehicle: Vehicle parameters
        a_x: Longitudinal acceleration [m/s²]
        a_y: Lateral acceleration [m/s²]

    Returns:
        WheelLoads with dynamic vertical loads [N]
    """
    m = vehicle.m
    g = vehicle.g
    W = m * g

    # Get geometry (use defaults if not specified)
    if vehicle.geometry is not None:
        geo = vehicle.geometry
        L = geo.L
        L_f = geo.L_f_m
        L_r = geo.L_r_m
        t_f = geo.track_front_m
        t_r = geo.track_rear_m
        h = geo.h_cg_m
    else:
        # Default geometry for FSAE vehicle
        L = 1.55
        L_f = 0.75
        L_r = 0.80
        t_f = 1.20
        t_r = 1.20
        h = 0.28

    # Static load distribution
    W_front_static = W * (L_r / L)
    W_rear_static = W * (L_f / L)

    # Longitudinal weight transfer
    dW_longitudinal = (m * a_x * h) / L

    # Dynamic axle loads
    W_front = W_front_static - dW_longitudinal
    W_rear = W_rear_static + dW_longitudinal

    # Lateral weight transfer per axle
    # Distribute based on axle load (simplified roll stiffness model)
    front_load_ratio = W_front / W
    rear_load_ratio = W_rear / W

    dW_lat_front = (m * abs(a_y) * h * front_load_ratio) / t_f
    dW_lat_rear = (m * abs(a_y) * h * rear_load_ratio) / t_r

    # Apply lateral transfer based on turn direction
    # a_y > 0 means turning right, load goes to left wheels
    if a_y >= 0:
        # Right turn: left wheels get more load
        F_z_FL = W_front / 2 + dW_lat_front
        F_z_FR = W_front / 2 - dW_lat_front
        F_z_RL = W_rear / 2 + dW_lat_rear
        F_z_RR = W_rear / 2 - dW_lat_rear
    else:
        # Left turn: right wheels get more load
        F_z_FL = W_front / 2 - dW_lat_front
        F_z_FR = W_front / 2 + dW_lat_front
        F_z_RL = W_rear / 2 - dW_lat_rear
        F_z_RR = W_rear / 2 + dW_lat_rear

    # Clamp to non-negative (wheel lift)
    F_z_FL = max(0.0, F_z_FL)
    F_z_FR = max(0.0, F_z_FR)
    F_z_RL = max(0.0, F_z_RL)
    F_z_RR = max(0.0, F_z_RR)

    return WheelLoads(FL=F_z_FL, FR=F_z_FR, RL=F_z_RL, RR=F_z_RR)


def calculate_axle_loads(vehicle: VehicleParams,
                         a_x: float = 0.0) -> tuple:
    """
    Calculate front and rear axle loads (for bicycle model).

    Args:
        vehicle: Vehicle parameters
        a_x: Longitudinal acceleration [m/s²]

    Returns:
        (F_z_front, F_z_rear): Axle loads [N]
    """
    m = vehicle.m
    g = vehicle.g
    W = m * g

    if vehicle.geometry is not None:
        geo = vehicle.geometry
        L = geo.L
        L_f = geo.L_f_m
        L_r = geo.L_r_m
        h = geo.h_cg_m
    else:
        L = 1.55
        L_f = 0.75
        L_r = 0.80
        h = 0.28

    # Static distribution
    W_front_static = W * (L_r / L)
    W_rear_static = W * (L_f / L)

    # Longitudinal transfer
    dW = (m * a_x * h) / L

    F_z_front = max(0.0, W_front_static - dW)
    F_z_rear = max(0.0, W_rear_static + dW)

    return F_z_front, F_z_rear


def get_inner_outer_wheels(a_y: float) -> tuple:
    """
    Determine which wheels are inner/outer based on turn direction.

    Args:
        a_y: Lateral acceleration [m/s²] (positive = right turn)

    Returns:
        (inner_wheels, outer_wheels): Lists of wheel names
    """
    if a_y >= 0:
        # Right turn: right wheels are inner, left are outer
        return ['FR', 'RR'], ['FL', 'RL']
    else:
        # Left turn: left wheels are inner, right are outer
        return ['FL', 'RL'], ['FR', 'RR']
