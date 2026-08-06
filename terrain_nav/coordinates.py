"""Coordinate and angular math utilities for terrain navigation."""

import math
from typing import Tuple


def normalize_heading(heading_deg: float) -> float:
    """Normalize heading to [0.0, 360.0)."""
    return float(heading_deg % 360.0)


def circular_heading_error(estimated_deg: float, true_deg: float) -> float:
    """Calculate the shortest circular distance between two headings."""
    # e_psi = |((hat_psi - psi + 180) mod 360) - 180|
    diff = normalize_heading(estimated_deg) - normalize_heading(true_deg)
    return abs(((diff + 180.0) % 360.0) - 180.0)


class CoordinateTransform:
    """Handles conversions between world (x,y) and raster (row,col) coordinates.

    World coordinates:
    - x: East (meters)
    - y: North (meters)

    Raster coordinates:
    - col: Left-to-right (matches East)
    - row: Top-to-bottom (opposite to North)

    Heading (psi):
    - 0 deg: North
    - 90 deg: East
    - 180 deg: South
    - 270 deg: West
    """

    def __init__(self, dx: float, dy: float, origin_x: float = 0.0, origin_y: float = 0.0):
        if dx <= 0 or dy <= 0:
            raise ValueError("Cell sizes dx and dy must be positive.")
        self.dx = dx
        self.dy = dy
        self.origin_x = origin_x
        self.origin_y = origin_y

    def offset_meters(self, distance_m: float, heading_deg: float) -> Tuple[float, float]:
        """Calculate world offsets (delta_x, delta_y) for a given distance and heading."""
        angle_rad = math.radians(normalize_heading(heading_deg))
        delta_x = distance_m * math.sin(angle_rad)
        delta_y = distance_m * math.cos(angle_rad)
        return delta_x, delta_y

    def offset_pixels(self, distance_m: float, heading_deg: float) -> Tuple[float, float]:
        """Calculate raster offsets (delta_row, delta_col) for a given distance and heading."""
        delta_x, delta_y = self.offset_meters(distance_m, heading_deg)
        delta_col = delta_x / self.dx
        delta_row = -delta_y / self.dy  # row grows downwards, North is upwards
        return delta_row, delta_col

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world (x,y) to raster (row,col)."""
        # Assumes origin is at (row=0, col=0)
        col = (x - self.origin_x) / self.dx
        row = (self.origin_y - y) / self.dy
        return row, col

    def pixel_to_world(self, row: float, col: float) -> Tuple[float, float]:
        """Convert raster (row,col) to world (x,y)."""
        x = self.origin_x + (col * self.dx)
        y = self.origin_y - (row * self.dy)
        return x, y
