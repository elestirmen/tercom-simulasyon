"""End-to-end simulation engine with ground truth isolation."""

import math
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from terrain_nav.confidence import detect_ambiguity
from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import ProfileMatcher
from terrain_nav.metrics import EstimatedState
from terrain_nav.sensors import Measurement, SensorSimulator
from terrain_nav.terrain import TerrainManager
from terrain_nav.trajectory import RouteGenerator


class LocalizationEngine:
    """Isolated from ground truth. Only knows navigation DEM and incoming measurements."""
    
    def __init__(self, config: LocalizationConfig, nav_dem: np.ndarray, ct: CoordinateTransform):
        self.config = config
        self.nav_dem = nav_dem
        self.ct = ct
        self.matcher = ProfileMatcher(config, nav_dem, ct)
        
        self.measurements = [] # List of Measurement
        self.relative_offsets = [] # List of (dx, dy, heading) relative to start of window
        self.last_match_pixel: Optional[Tuple[float, float]] = None
        self.base_search_roi_size = max(0, int(config.algorithm.search_roi_size_px))
        self.current_search_roi_size = self.base_search_roi_size
        
    def add_measurement(self, m: Measurement):
        self.measurements.append(m)
        
        if len(self.measurements) > self.config.algorithm.profile_window_size:
            removed = self.measurements.pop(0)
            if self.last_match_pixel is not None:
                d_row, d_col = self.ct.offset_pixels(
                    removed.traveled_distance_m,
                    removed.sensor_heading_deg,
                )
                self.last_match_pixel = (
                    self.last_match_pixel[0] + d_row,
                    self.last_match_pixel[1] + d_col,
                )
            
        self._recompute_offsets()
        
    def _recompute_offsets(self):
        """Compute offsets relative to the *oldest* measurement in the window."""
        self.relative_offsets = []
        if not self.measurements:
            return
        
        curr_x, curr_y = 0.0, 0.0
        
        for _i, m in enumerate(self.measurements):
            # The heading is the heading the sensor reported for this step
            self.relative_offsets.append((curr_x, curr_y, m.sensor_heading_deg))
            
            # To get to the next point, move by traveled_distance in the direction of heading
            dx, dy = self.ct.offset_meters(m.traveled_distance_m, m.sensor_heading_deg)
            curr_x += dx
            curr_y += dy

    def _search_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        if self.last_match_pixel is None or self.current_search_roi_size <= 0:
            return None

        rows, cols = self.nav_dem.shape
        size = self.current_search_roi_size
        center_row, center_col = self.last_match_pixel

        def bounded_interval(center: float, length: int) -> Tuple[int, int]:
            if size >= length:
                return 0, length
            start = int(round(center)) - size // 2
            start = min(max(0, start), length - size)
            return start, start + size

        row_start, row_end = bounded_interval(center_row, rows)
        col_start, col_end = bounded_interval(center_col, cols)
        return row_start, row_end, col_start, col_end

    def _grow_search_roi(self) -> None:
        if self.base_search_roi_size <= 0:
            return
        map_size = max(self.nav_dem.shape)
        grown = max(
            self.current_search_roi_size + self.base_search_roi_size // 2,
            int(round(self.current_search_roi_size * 1.5)),
        )
        self.current_search_roi_size = min(map_size, grown)
            
    def localize(self, timestamp: float) -> Optional[EstimatedState]:
        if len(self.measurements) < self.config.algorithm.min_profile_length:
            return None
            
        laser = np.array([m.laser_agl_m for m in self.measurements])
        valid = np.array([m.laser_valid for m in self.measurements])
        baro = np.array([m.baro_msl_m for m in self.measurements])
        
        # Determine search headings
        if self.config.sensor.heading_mode == "known_heading":
            search_headings = [self.relative_offsets[0][2]] # Just search with reported heading
        elif self.config.sensor.heading_mode == "noisy_heading":
            # Search around reported heading
            base_h = self.relative_offsets[0][2]
            unc = self.config.sensor.heading_uncertainty_deg
            search_headings = np.arange(base_h - unc, base_h + unc + 0.1, 1.0).tolist()
        else: # unknown_heading
            search_headings = np.arange(0.0, 360.0, 5.0).tolist() # Every 5 degrees initially

        search_bounds = self._search_bounds()

        if self.config.algorithm.method == "exhaustive":
            cands = self.matcher.exhaustive_search(
                laser,
                valid,
                baro,
                self.relative_offsets,
                search_headings,
                search_bounds=search_bounds,
            )
        else:
            cands = self.matcher.coarse_to_fine_search(
                laser,
                valid,
                baro,
                self.relative_offsets,
                search_headings,
                search_bounds=search_bounds,
            )

        if not cands:
            if search_bounds is not None:
                self._grow_search_roi()
            return None

        max_jump_m = float(self.config.algorithm.max_match_jump_m)
        if self.last_match_pixel is not None and max_jump_m > 0:
            anchor_row, anchor_col = self.last_match_pixel
            plausible_cands = [
                candidate
                for candidate in cands
                if math.hypot(
                    (candidate.row - anchor_row) * self.ct.dy,
                    (candidate.col - anchor_col) * self.ct.dx,
                )
                <= max_jump_m
            ]
            if not plausible_cands:
                self._grow_search_roi()
                return None
            cands = plausible_cands

        is_ambiguous, margin, spread = detect_ambiguity(cands)
        best = cands[0]
        if search_bounds is not None:
            edge_margin = max(
                int(self.config.algorithm.coarse_stride),
                int(self.config.algorithm.refinement_radius_px),
            )
            near_roi_edge = (
                best.row - search_bounds[0] < edge_margin
                or search_bounds[1] - best.row <= edge_margin
                or best.col - search_bounds[2] < edge_margin
                or search_bounds[3] - best.col <= edge_margin
            )
            is_ambiguous = is_ambiguous or near_roi_edge

        if is_ambiguous:
            if self.last_match_pixel is None:
                self.last_match_pixel = (best.row, best.col)
            self._grow_search_roi()
        else:
            self.last_match_pixel = (best.row, best.col)
            self.current_search_roi_size = self.base_search_roi_size
        
        # `best.row`, `best.col` is the location of the *start* of the window (the oldest point).
        # We want the location of the *current* point (the newest point).
        # The newest point is at relative offset corresponding to the last measurement.
        last_offset = self.relative_offsets[-1]
        
        # Need to rotate last_offset by the matched heading difference.
        # Wait, the offset rotation logic is in `rotate_offsets` and is applied during extraction.
        # Let's do it manually for the last point.
        angle_diff_deg = best.heading_deg - self.relative_offsets[0][2]
        rad = math.radians(angle_diff_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        dx = last_offset[0]
        dy = last_offset[1]
        
        dx_new = dx * cos_a - dy * sin_a
        dy_new = dx * sin_a + dy * cos_a
        
        # convert to row/col
        d_col = dx_new / self.ct.dx
        d_row = -dy_new / self.ct.dy
        
        curr_row = best.row + d_row
        curr_col = best.col + d_col
        curr_h = (last_offset[2] + angle_diff_deg) % 360.0
        
        curr_x, curr_y = self.ct.pixel_to_world(curr_row, curr_col)
        
        return EstimatedState(
            timestamp_s=timestamp,
            is_ambiguous=is_ambiguous,
            estimated_x=curr_x,
            estimated_y=curr_y,
            estimated_heading_deg=curr_h,
            estimated_msl_m=best.estimated_msl_m,
            score=best.score,
            spatial_spread=spread,
            score_margin=margin
        )

class SimulationEngine:
    """Master engine. Knows truth. Feeds LocalizationEngine."""
    
    def __init__(self, config: LocalizationConfig):
        self.terrain = TerrainManager(config.terrain)
        if config.terrain.dem_path:
            minimum_safe_msl = float(
                np.ceil(
                    (
                        float(np.nanmax(self.terrain.nav_dem))
                        + config.sensor.min_safe_agl_m
                    )
                    / 10.0
                )
                * 10.0
            )
            if config.sensor.constant_msl_m < minimum_safe_msl:
                config = replace(
                    config,
                    sensor=replace(config.sensor, constant_msl_m=minimum_safe_msl),
                )
        self.config = config
        self.ct = CoordinateTransform(self.terrain.dx, self.terrain.dy)
        
        self.rg = RouteGenerator(config.route, self.ct)
        # We still generate true_path so total_steps is known, but we won't strictly follow it 
        # if the user manually steers.
        self.total_steps = int(config.route.route_length_m / config.route.sample_spacing_m)
        if config.route.mode == "heading_sequence":
            # Approximation for total steps
            self.total_steps = len(self.rg.generate_true_path())
            
        self.sensors = SensorSimulator(config.sensor, config.terrain.seed)
        
        # Isolated
        self.localization = LocalizationEngine(
            config,
            self.terrain.get_navigation_dem(copy=False),
            self.ct,
        )
        
        self.step_idx = 0
        self.dt_s = config.route.sample_spacing_m / config.route.speed_m_s
        
        # Dynamic State for Manual Steering
        start_row = config.route.start_row
        start_col = config.route.start_col
        if config.terrain.dem_path and config.terrain.external_auto_center_start:
            start_row, start_col = self.terrain.get_center_pixel()
            if config.route.mode == "straight_heading":
                route_d_row, route_d_col = self.ct.offset_pixels(
                    config.route.route_length_m,
                    config.route.heading_deg,
                )
                rows, cols = self.terrain.truth_dem.shape
                start_row = int(round(start_row - route_d_row / 2.0))
                start_col = int(round(start_col - route_d_col / 2.0))
                start_row = min(max(1, start_row), rows - 2)
                start_col = min(max(1, start_col), cols - 2)
        start_x, start_y = self.ct.pixel_to_world(start_row, start_col)
        self.dynamic_x = start_x
        self.dynamic_y = start_y
        self.dynamic_h = config.route.heading_deg
        
    def turn_vehicle(self, angle_deg: float):
        """Turn the vehicle's heading by angle_deg."""
        self.dynamic_h = (self.dynamic_h + angle_deg) % 360.0
        
    def get_total_steps(self) -> int:
        return self.total_steps
        
    def step(self) -> Tuple[Tuple[float, float, float], Optional[EstimatedState], Measurement]:
        """Run one step. Returns (true_state, estimated_state, measurement)."""
        if self.step_idx >= self.total_steps:
            return None, None, None
            
        true_x, true_y, true_h = self.dynamic_x, self.dynamic_y, self.dynamic_h
        true_r, true_c = self.ct.world_to_pixel(true_x, true_y)
        
        # Ground truth elevation (using exact DEM cell, or bilinear)
        r_int = int(round(true_r))
        c_int = int(round(true_c))
        rows, cols = self.terrain.truth_dem.shape
        
        terrain_elev = 0.0
        if 0 <= r_int < rows and 0 <= c_int < cols:
            terrain_elev = self.terrain.sample_truth_elevation(r_int, c_int)
            
        true_msl = self.config.sensor.constant_msl_m # Simplify, normally could change
        
        # Generate Measurement
        m = self.sensors.generate_measurement(
            true_msl_m=true_msl,
            terrain_elevation_m=terrain_elev,
            true_heading_deg=true_h,
            traveled_distance_m=self.config.route.sample_spacing_m,
            dt_s=self.dt_s
        )
        
        # Pass to localization
        self.localization.add_measurement(m)
        timestamp = self.step_idx * self.dt_s
        est = self.localization.localize(timestamp)
        
        self.step_idx += 1
        
        # Advance vehicle forward for the next step
        dx, dy = self.ct.offset_meters(self.config.route.sample_spacing_m, self.dynamic_h)
        self.dynamic_x += dx
        self.dynamic_y += dy
        
        return (true_x, true_y, true_h), est, m
