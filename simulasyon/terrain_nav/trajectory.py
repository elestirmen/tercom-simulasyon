"""Trajectory generation and routing."""

import math
from typing import List, Tuple
from terrain_nav.config import RouteConfig
from terrain_nav.coordinates import CoordinateTransform, normalize_heading

class RouteGenerator:
    """Generates truth trajectory points and relative offsets."""
    
    def __init__(self, config: RouteConfig, coord_transform: CoordinateTransform):
        self.config = config
        self.ct = coord_transform
        
    def _parse_sequence(self, sequence: str) -> List[Tuple[float, float]]:
        """Parse heading sequence string e.g., '0:500,45:300' to list of (heading, distance)."""
        if not sequence:
            return []
        parts = sequence.split(",")
        result = []
        for p in parts:
            h, d = p.split(":")
            result.append((float(h), float(d)))
        return result

    def generate_offsets(self) -> List[Tuple[float, float, float]]:
        """
        Generate relative offsets from origin.
        Returns a list of (dx_meters, dy_meters, current_heading_deg)
        """
        offsets = []
        
        if self.config.mode == "straight_heading":
            samples = int(self.config.route_length_m / self.config.sample_spacing_m)
            heading = normalize_heading(self.config.heading_deg)
            for i in range(samples):
                dist = i * self.config.sample_spacing_m
                dx, dy = self.ct.offset_meters(dist, heading)
                offsets.append((dx, dy, heading))
                
        elif self.config.mode == "heading_sequence":
            sequence = self._parse_sequence(self.config.heading_sequence)
            current_x, current_y = 0.0, 0.0
            total_dist_in_segment = 0.0
            
            for (heading, dist_m) in sequence:
                heading = normalize_heading(heading)
                samples = int(dist_m / self.config.sample_spacing_m)
                for _ in range(samples):
                    # We only sample every sample_spacing_m.
                    offsets.append((current_x, current_y, heading))
                    dx, dy = self.ct.offset_meters(self.config.sample_spacing_m, heading)
                    current_x += dx
                    current_y += dy

        elif self.config.mode == "waypoint_route":
            current_x, current_y = 0.0, 0.0 # Origin
            # config.waypoints contains (x, y) coordinates relative to origin or global
            # Assuming they are world coordinates. So we need to subtract start pos if it's absolute,
            # or treat them as relative. Let's assume they are relative to start for offset generation,
            # but wait, config.start_row/col defines start.
            
            # For simplicity in offsets, we assume waypoints are relative to start (0,0)
            wp_list = self.config.waypoints
            if not wp_list:
                return []
                
            for i in range(len(wp_list)):
                target_x, target_y = wp_list[i]
                diff_x = target_x - current_x
                diff_y = target_y - current_y
                dist = math.hypot(diff_x, diff_y)
                heading = math.degrees(math.atan2(diff_x, diff_y))
                heading = normalize_heading(heading)
                
                samples = int(dist / self.config.sample_spacing_m)
                for _ in range(samples):
                    offsets.append((current_x, current_y, heading))
                    dx, dy = self.ct.offset_meters(self.config.sample_spacing_m, heading)
                    current_x += dx
                    current_y += dy
                    
        return offsets
        
    def generate_true_path(self) -> List[Tuple[float, float, float]]:
        """Returns list of absolute world coordinates and heading: (x, y, heading)"""
        start_x, start_y = self.ct.pixel_to_world(self.config.start_row, self.config.start_col)
        offsets = self.generate_offsets()
        return [(start_x + dx, start_y + dy, h) for dx, dy, h in offsets]
