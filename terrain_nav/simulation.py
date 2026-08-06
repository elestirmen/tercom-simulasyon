"""End-to-end simulation engine with ground truth isolation."""

import math
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np

from terrain_nav.confidence import detect_ambiguity
from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import Candidate, ProfileMatcher
from terrain_nav.metrics import EstimatedState
from terrain_nav.sensors import Measurement, SensorSimulator
from terrain_nav.terrain import TerrainManager


class MotionOutOfBoundsError(ValueError):
    """Raised when a commanded motion would leave the complete source map."""


class LocalizationEngine:
    """Isolated from ground truth. Only knows navigation DEM and incoming measurements."""

    def __init__(self, config: LocalizationConfig, nav_dem: np.ndarray, ct: CoordinateTransform):
        self.config = config
        self.nav_dem = nav_dem
        self.ct = ct
        self.matcher = ProfileMatcher(config, nav_dem, ct)

        self.measurements = []  # List of Measurement
        self.relative_offsets = []  # List of (dx, dy, heading) relative to start of window
        self.last_match_pixel: Optional[Tuple[float, float]] = None
        self.base_search_roi_size = max(0, int(config.algorithm.search_roi_size_px))
        self.current_search_roi_size = self.base_search_roi_size
        self.recovery_active = False
        self.last_rejection_reason: Optional[str] = None
        self.last_rejected_score: Optional[float] = None

    def add_measurement(self, m: Measurement):
        self.measurements.append(m)

        if len(self.measurements) > self.config.algorithm.profile_window_size:
            self.measurements.pop(0)
            if self.last_match_pixel is not None:
                # Each measurement stores the motion that ended at that sample.
                # After removing the oldest sample, advance the anchor with the
                # motion attached to the new oldest sample.
                new_oldest = self.measurements[0]
                d_row, d_col = self.ct.offset_pixels(
                    new_oldest.traveled_distance_m,
                    new_oldest.sensor_heading_deg,
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

        for index, measurement in enumerate(self.measurements):
            # A measurement's odometry describes the movement from the previous
            # sample into this sample. The first item anchors the profile at zero.
            if index > 0:
                dx, dy = self.ct.offset_meters(
                    measurement.traveled_distance_m,
                    measurement.sensor_heading_deg,
                )
                curr_x += dx
                curr_y += dy
            self.relative_offsets.append((curr_x, curr_y, measurement.sensor_heading_deg))

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
        self.recovery_active = True

    def _bounds_cover_full_map(self, bounds: Tuple[int, int, int, int]) -> bool:
        rows, cols = self.nav_dem.shape
        return bounds == (0, rows, 0, cols)

    def _enter_global_recovery(self) -> None:
        """Drop a stale anchor while retaining the measured terrain profile."""
        self.last_match_pixel = None
        self.current_search_roi_size = self.base_search_roi_size
        self.recovery_active = True

    def _reject_search_result(
        self,
        search_bounds: Optional[Tuple[int, int, int, int]],
    ) -> None:
        """Broaden a tracked search without converting a poor candidate into a fix."""
        if search_bounds is not None:
            if self._bounds_cover_full_map(search_bounds):
                self._enter_global_recovery()
            else:
                self._grow_search_roi()
        elif self.last_match_pixel is not None:
            self._enter_global_recovery()

    def _candidate_passes_quality_gate(self, candidate: Candidate) -> bool:
        algorithm = self.config.algorithm
        quality_score = self._candidate_quality_score(candidate)
        correlation = self._candidate_quality_correlation(candidate)
        return (
            quality_score <= algorithm.max_match_inlier_rmse_m
            and correlation >= algorithm.min_match_inlier_correlation
            and candidate.valid_ratio >= algorithm.min_match_valid_ratio
        )

    @staticmethod
    def _candidate_quality_score(candidate: Candidate) -> float:
        metrics = candidate.metrics
        return float(metrics.get("inlier_rmse", metrics.get("rmse", candidate.score)))

    @staticmethod
    def _candidate_quality_correlation(candidate: Candidate) -> float:
        metrics = candidate.metrics
        return float(metrics.get("inlier_correlation", metrics.get("correlation", 1.0)))

    def get_search_status(self) -> dict:
        """Expose the exact search area used for the next localization call."""
        rows, cols = self.nav_dem.shape
        local_bounds = self._search_bounds()
        if local_bounds is None or self._bounds_cover_full_map(local_bounds):
            if self.recovery_active:
                phase = "recovery"
            elif self.last_match_pixel is None:
                phase = "initial"
            else:
                phase = "tracking"
            return {
                "mode": "global_search",
                "phase": phase,
                "bounds": (0, rows, 0, cols),
                "draw_bounds": None,
                "has_anchor": self.last_match_pixel is not None,
                "roi_size_px": max(rows, cols),
                "rejection_reason": self.last_rejection_reason,
                "rejected_score": self.last_rejected_score,
            }
        return {
            "mode": "local_roi",
            "phase": "recovery" if self.recovery_active else "tracking",
            "bounds": local_bounds,
            "draw_bounds": local_bounds,
            "has_anchor": True,
            "roi_size_px": self.current_search_roi_size,
            "rejection_reason": self.last_rejection_reason,
            "rejected_score": self.last_rejected_score,
        }

    def localize(self, timestamp: float) -> Optional[EstimatedState]:
        if len(self.measurements) < self.config.algorithm.min_profile_length:
            self.last_rejection_reason = "profile_incomplete"
            self.last_rejected_score = None
            return None

        laser = np.array([m.laser_agl_m for m in self.measurements])
        valid = np.array([m.laser_valid for m in self.measurements])
        baro = np.array([m.baro_msl_m for m in self.measurements])

        # Determine search headings
        if self.config.sensor.heading_mode == "known_heading":
            search_headings = [self.relative_offsets[0][2]]  # Just search with reported heading
        elif self.config.sensor.heading_mode == "noisy_heading":
            # Search around reported heading
            base_h = self.relative_offsets[0][2]
            unc = self.config.sensor.heading_uncertainty_deg
            search_headings = np.arange(base_h - unc, base_h + unc + 0.1, 1.0).tolist()
        else:  # unknown_heading
            search_headings = np.arange(0.0, 360.0, 5.0).tolist()  # Every 5 degrees initially

        search_bounds = self._search_bounds()

        cands = self.matcher.coarse_to_fine_search(
            laser,
            valid,
            baro,
            self.relative_offsets,
            search_headings,
            search_bounds=search_bounds,
        )

        if not cands:
            self.last_rejection_reason = "no_candidates"
            self.last_rejected_score = None
            self._reject_search_result(search_bounds)
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
                self.last_rejection_reason = "continuity"
                self.last_rejected_score = cands[0].score
                self._reject_search_result(search_bounds)
                return None
            cands = plausible_cands

        quality_cands = [
            candidate for candidate in cands if self._candidate_passes_quality_gate(candidate)
        ]
        if not quality_cands:
            self.last_rejection_reason = "quality"
            self.last_rejected_score = self._candidate_quality_score(cands[0])
            self._reject_search_result(search_bounds)
            return None
        cands = sorted(
            quality_cands,
            key=lambda candidate: (self._candidate_quality_score(candidate), candidate.score),
        )

        is_ambiguous, margin, spread = detect_ambiguity(
            cands,
            score_getter=self._candidate_quality_score,
        )
        best = cands[0]
        quality_score = self._candidate_quality_score(best)
        quality_correlation = self._candidate_quality_correlation(best)
        self.last_rejection_reason = None
        self.last_rejected_score = None
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
            # Never turn an ambiguous global candidate into a local lock. When
            # tracking an existing lock, broaden its ROI until global recovery.
            if search_bounds is not None:
                self._grow_search_roi()
        else:
            self.last_match_pixel = (best.row, best.col)
            self.current_search_roi_size = self.base_search_roi_size
            self.recovery_active = False

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
            score_margin=margin,
            quality_score=quality_score,
            quality_correlation=quality_correlation,
        )


class SimulationEngine:
    """Master engine. Knows truth. Feeds LocalizationEngine."""

    def __init__(self, config: LocalizationConfig, manual_control: bool = False):
        """Create the simulation engine.

        ``manual_control`` is intentionally opt-in so existing headless and
        benchmark runs retain their automatic route semantics. The desktop UI
        enables it and advances only when a keyboard command is received.
        """
        self.terrain = TerrainManager(config.terrain)
        if config.terrain.dem_path:
            minimum_safe_msl = float(
                np.ceil(
                    (self.terrain.get_source_max_elevation() + config.sensor.min_safe_agl_m) / 10.0
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

        self.manual_control = manual_control
        # In manual mode one command represents one sensor observation. Limit
        # the number of commands by the configured route length, using the
        # larger UI movement quantum rather than the automatic sample spacing.
        if manual_control:
            self.total_steps = int(
                math.ceil(config.route.route_length_m / config.route.manual_step_distance_m)
            )
        else:
            self.total_steps = int(config.route.route_length_m / config.route.sample_spacing_m)
        self.sensors = SensorSimulator(config.sensor, config.terrain.seed)

        # Isolated
        self.localization = LocalizationEngine(
            config,
            self.terrain.get_navigation_dem(copy=False),
            self.ct,
        )

        self.step_idx = 0
        self.dt_s = config.route.sample_spacing_m / config.route.speed_m_s
        self.elapsed_s = 0.0

        # Dynamic State for Manual Steering
        start_row = config.route.start_row
        start_col = config.route.start_col
        if config.terrain.dem_path and config.terrain.external_auto_center_start:
            start_row, start_col = self.terrain.get_center_pixel()
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

    def close(self) -> None:
        """Release external terrain resources owned by the simulation."""
        self.terrain.close()

    def get_current_state(self) -> Tuple[float, float, float]:
        """Return the current truth state without taking a measurement."""
        return self.dynamic_x, self.dynamic_y, self.dynamic_h

    def get_localization_status(self) -> dict:
        """Return UI-safe localization search state without exposing truth."""
        return self.localization.get_search_status()

    def _validate_world_position(self, x: float, y: float) -> None:
        if not self.terrain.is_inside_source_map(x, y):
            raise MotionOutOfBoundsError("Komut reddedildi: İHA kaynak haritanın dışına çıkamaz.")

    def _sample_current(
        self,
        traveled_distance_m: float,
        sensor_heading_deg: float,
        dt_s: float,
    ) -> Tuple[Tuple[float, float, float], Optional[EstimatedState], Measurement]:
        """Take one sensor/localization sample at the current truth position."""
        true_x, true_y, true_h = self.dynamic_x, self.dynamic_y, self.dynamic_h
        try:
            terrain_elev = self.terrain.sample_elevation_at_world(true_x, true_y)
        except ValueError as exc:
            raise MotionOutOfBoundsError(str(exc)) from exc

        true_msl = self.config.sensor.constant_msl_m
        m = self.sensors.generate_measurement(
            true_msl_m=true_msl,
            terrain_elevation_m=terrain_elev,
            true_heading_deg=sensor_heading_deg,
            traveled_distance_m=traveled_distance_m,
            dt_s=dt_s,
        )

        self.localization.add_measurement(m)
        est = self.localization.localize(self.elapsed_s)
        self.elapsed_s += dt_s
        self.step_idx += 1

        return (true_x, true_y, true_h), est, m

    def execute_motion(
        self,
        distance_m: float,
        relative_heading_deg: float = 0.0,
    ) -> Tuple[Tuple[float, float, float], Optional[EstimatedState], Measurement]:
        """Move once relative to the vehicle heading, then take a sample.

        The vehicle orientation stays unchanged for forward, reverse and
        lateral commands. ``relative_heading_deg`` describes the direction of
        travel: 0 forward, 180 reverse, -90 left and +90 right.
        """
        if not self.manual_control:
            raise RuntimeError("execute_motion requires manual_control=True")

        distance_m = abs(float(distance_m))
        if distance_m <= 0.0:
            raise ValueError("Manual movement distance must be positive")

        movement_heading = (self.dynamic_h + relative_heading_deg) % 360.0
        dx, dy = self.ct.offset_meters(distance_m, movement_heading)
        next_x = self.dynamic_x + dx
        next_y = self.dynamic_y + dy
        self._validate_world_position(next_x, next_y)
        self.dynamic_x = next_x
        self.dynamic_y = next_y

        motion_dt_s = distance_m / max(self.config.route.speed_m_s, 1e-9)
        return self._sample_current(distance_m, movement_heading, motion_dt_s)

    def get_total_steps(self) -> int:
        return self.total_steps

    def step(self) -> Tuple[Tuple[float, float, float], Optional[EstimatedState], Measurement]:
        """Run one step. Returns (true_state, estimated_state, measurement)."""
        if self.manual_control:
            raise RuntimeError("Manual simulations advance with execute_motion()")
        if self.step_idx >= self.total_steps:
            return None, None, None

        true_state, est, m = self._sample_current(
            self.config.route.sample_spacing_m,
            self.dynamic_h,
            self.dt_s,
        )

        # Advance vehicle forward for the next step
        dx, dy = self.ct.offset_meters(self.config.route.sample_spacing_m, self.dynamic_h)
        next_x = self.dynamic_x + dx
        next_y = self.dynamic_y + dy
        if self.step_idx < self.total_steps:
            self._validate_world_position(next_x, next_y)
        self.dynamic_x = next_x
        self.dynamic_y = next_y

        return true_state, est, m
