"""End-to-end simulation engine with ground truth isolation."""

import math
import time
from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np

from terrain_nav.confidence import SpeedConfidence, assess_speed_confidence, detect_ambiguity
from terrain_nav.config import (
    MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
    LocalizationConfig,
    LocalizationRuntimeConfig,
    localization_runtime_config,
)
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import Candidate, ProfileMatcher
from terrain_nav.metrics import EstimatedState, ProfileComparison
from terrain_nav.profile import (
    build_distances_for_candidate_speed,
    build_offsets_for_candidate_speed,
    extract_profile,
    rotate_offsets,
)
from terrain_nav.sensors import Measurement, SensorSimulator
from terrain_nav.terrain import TerrainManager


class MotionOutOfBoundsError(ValueError):
    """Raised when a commanded motion would leave the complete source map."""


class LocalizationEngine:
    """Isolated from ground truth. Only knows navigation DEM and incoming measurements."""

    def __init__(
        self,
        config: LocalizationConfig | LocalizationRuntimeConfig,
        nav_dem: np.ndarray,
        ct: CoordinateTransform,
    ):
        self.config = (
            localization_runtime_config(config)
            if isinstance(config, LocalizationConfig)
            else config
        )
        self.nav_dem = nav_dem
        self.ct = ct
        self.matcher = ProfileMatcher(self.config, nav_dem, ct)

        self.measurements = []  # List of Measurement
        self.relative_offsets = []  # List of (dx, dy, heading) relative to start of window
        self.last_match_pixel: Optional[Tuple[float, float]] = None
        self.base_search_roi_size = max(0, int(self.config.algorithm.search_roi_size_px))
        self.current_search_roi_size = self.base_search_roi_size
        self.recovery_active = False
        self.last_rejection_reason: Optional[str] = None
        self.last_rejected_score: Optional[float] = None
        self.last_profile_comparison: Optional[ProfileComparison] = None
        self.last_estimated_speed_m_s: Optional[float] = None
        self.last_profile_measurements: list[Measurement] = []
        self.last_runtime_profile: dict[str, float] = {}

    def _uses_unknown_speed(self) -> bool:
        return self.config.motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED

    def _drop_oldest_measurement(self) -> None:
        """Remove the oldest sample and keep the profile-start anchor aligned."""
        removed = self.measurements.pop(0)
        if self.last_match_pixel is not None and self.measurements:
            # Each measurement stores the motion that ended at that sample.
            # After removing the oldest sample, advance the anchor with the
            # motion attached to the new oldest sample.
            new_oldest = self.measurements[0]
            if self._uses_unknown_speed():
                if self.last_estimated_speed_m_s is None:
                    self._enter_global_recovery()
                    return
                dt_s = float(new_oldest.timestamp_s) - float(removed.timestamp_s)
                d_row, d_col = self.ct.offset_pixels(
                    self.last_estimated_speed_m_s * dt_s,
                    new_oldest.sensor_heading_deg,
                )
            else:
                if new_oldest.traveled_distance_m is None:
                    raise ValueError("Distance is required outside unknown-speed mode")
                d_row, d_col = self.ct.offset_pixels(
                    new_oldest.traveled_distance_m,
                    new_oldest.sensor_heading_deg,
                )
            self.last_match_pixel = (
                self.last_match_pixel[0] + d_row,
                self.last_match_pixel[1] + d_col,
            )

    def add_measurement(self, m: Measurement):
        if self._uses_unknown_speed():
            if m.traveled_distance_m is not None or m.measured_speed_m_s is not None:
                raise ValueError(
                    "unknown_constant_speed measurements must not contain speed or distance"
                )
            if self.measurements and m.timestamp_s <= self.measurements[-1].timestamp_s:
                raise ValueError("Measurement timestamps must be strictly increasing")
        elif m.traveled_distance_m is None:
            raise ValueError("Distance is required outside unknown-speed mode")

        self.measurements.append(m)

        algorithm = self.config.algorithm
        max_measurements = max(0, int(algorithm.profile_window_size))
        while len(self.measurements) > max_measurements:
            self._drop_oldest_measurement()

        if self._uses_unknown_speed():
            max_duration_s = float(algorithm.max_profile_duration_s)
            while (
                len(self.measurements) > algorithm.min_profile_length
                and self._profile_duration_s() > max_duration_s
            ):
                self._drop_oldest_measurement()
        else:
            max_profile_distance_m = float(algorithm.max_profile_distance_m)
            while (
                max_profile_distance_m > 0.0
                and len(self.measurements) > algorithm.min_profile_length
                and self._profile_distance_m() > max_profile_distance_m
            ):
                self._drop_oldest_measurement()

        self._recompute_offsets()

    def _recompute_offsets(self):
        """Compute offsets relative to the *oldest* measurement in the window."""
        self.relative_offsets = []
        if not self.measurements:
            return

        if self._uses_unknown_speed():
            if self.last_estimated_speed_m_s is not None:
                self.relative_offsets = build_offsets_for_candidate_speed(
                    self.measurements,
                    self.last_estimated_speed_m_s,
                    self.ct,
                )
            return

        curr_x, curr_y = 0.0, 0.0

        for index, measurement in enumerate(self.measurements):
            # A measurement's odometry describes the movement from the previous
            # sample into this sample. The first item anchors the profile at zero.
            if index > 0:
                if measurement.traveled_distance_m is None:
                    raise ValueError("Distance is required outside unknown-speed mode")
                dx, dy = self.ct.offset_meters(
                    measurement.traveled_distance_m,
                    measurement.sensor_heading_deg,
                )
                curr_x += dx
                curr_y += dy
            self.relative_offsets.append((curr_x, curr_y, measurement.sensor_heading_deg))

    def _profile_distance_m(self) -> float:
        """Return the measured horizontal span covered by the current profile."""
        distances = [m.traveled_distance_m for m in self.measurements[1:]]
        if any(distance is None for distance in distances):
            raise ValueError("Profile distance is unavailable in unknown-speed mode")
        return float(sum(float(distance) for distance in distances))

    def _profile_duration_s(self) -> float:
        if len(self.measurements) < 2:
            return 0.0
        return float(self.measurements[-1].timestamp_s - self.measurements[0].timestamp_s)

    def _measurements_for_localization(self) -> list[Measurement]:
        algorithm = self.config.algorithm
        mode = algorithm.profile_resampling_mode
        points = int(algorithm.profile_points)
        if mode == "raw" or points <= 0 or len(self.measurements) <= points:
            return list(self.measurements)
        points = max(2, points)
        if mode == "uniform":
            return self._uniform_measurement_subset(points)
        if mode == "interpolated":
            return self._interpolated_measurement_profile(points)
        raise ValueError(f"Unknown profile resampling mode: {mode}")

    def _uniform_measurement_subset(self, points: int) -> list[Measurement]:
        indices = np.rint(np.linspace(0, len(self.measurements) - 1, points)).astype(int)
        indices = np.unique(indices)
        if indices[-1] != len(self.measurements) - 1:
            indices = np.append(indices, len(self.measurements) - 1)
        selected = [self.measurements[int(index)] for index in indices.tolist()]
        return self._rebased_measurement_series(selected, indices.tolist())

    def _interpolated_measurement_profile(self, points: int) -> list[Measurement]:
        if len(self.measurements) < 2:
            return list(self.measurements)

        source_times = np.asarray(
            [measurement.timestamp_s for measurement in self.measurements],
            dtype=np.float64,
        )
        target_times = np.linspace(source_times[0], source_times[-1], points)
        baro = np.interp(
            target_times,
            source_times,
            [measurement.baro_msl_m for measurement in self.measurements],
        )
        headings = self._piecewise_headings(source_times, target_times)
        laser, laser_valid = self._interpolated_laser(source_times, target_times)

        if self._uses_unknown_speed():
            traveled = [None] * points
            speeds = [None] * points
        else:
            source_distances = np.asarray(
                self._profile_measurement_distances(self.measurements),
                dtype=np.float64,
            )
            target_distances = np.interp(target_times, source_times, source_distances)
            traveled_values = np.diff(target_distances, prepend=target_distances[0])
            dt_values = np.diff(target_times, prepend=target_times[0])
            traveled = [float(max(0.0, value)) for value in traveled_values]
            speeds = [
                float(distance / dt_s) if dt_s > 0.0 else None
                for distance, dt_s in zip(traveled, dt_values, strict=True)
            ]

        return [
            Measurement(
                laser_agl_m=float(laser[index]),
                laser_valid=bool(laser_valid[index]),
                baro_msl_m=float(baro[index]),
                sensor_heading_deg=float(headings[index]),
                traveled_distance_m=traveled[index],
                measured_speed_m_s=speeds[index],
                timestamp_s=float(target_times[index]),
            )
            for index in range(points)
        ]

    def _piecewise_headings(
        self,
        source_times: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        indices = np.searchsorted(source_times, target_times, side="right") - 1
        indices = np.clip(indices, 0, len(self.measurements) - 1)
        return np.asarray(
            [self.measurements[int(index)].sensor_heading_deg for index in indices],
            dtype=np.float64,
        )

    def _interpolated_laser(
        self,
        source_times: np.ndarray,
        target_times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        source_valid = np.asarray(
            [measurement.laser_valid for measurement in self.measurements],
            dtype=bool,
        )
        source_laser = np.asarray(
            [measurement.laser_agl_m for measurement in self.measurements],
            dtype=np.float64,
        )
        valid_times = source_times[source_valid]
        if len(valid_times) >= 2:
            laser = np.interp(target_times, valid_times, source_laser[source_valid])
            source_step = (
                float(np.median(np.diff(source_times))) if len(source_times) > 1 else 0.0
            )
            max_gap = max(source_step * 2.5, 1e-9)
            nearest_right = np.searchsorted(valid_times, target_times, side="left")
            nearest_left = np.maximum(nearest_right - 1, 0)
            nearest_right = np.minimum(nearest_right, len(valid_times) - 1)
            nearest_gap = np.minimum(
                np.abs(target_times - valid_times[nearest_left]),
                np.abs(target_times - valid_times[nearest_right]),
            )
            return laser, nearest_gap <= max_gap
        return (
            np.full(len(target_times), np.nan, dtype=np.float64),
            np.zeros(len(target_times), dtype=bool),
        )

    def _rebased_measurement_series(
        self,
        selected: Sequence[Measurement],
        selected_indices: Sequence[int],
    ) -> list[Measurement]:
        if self._uses_unknown_speed():
            return [
                Measurement(
                    laser_agl_m=measurement.laser_agl_m,
                    laser_valid=measurement.laser_valid,
                    baro_msl_m=measurement.baro_msl_m,
                    sensor_heading_deg=measurement.sensor_heading_deg,
                    traveled_distance_m=None,
                    measured_speed_m_s=None,
                    timestamp_s=measurement.timestamp_s,
                )
                for measurement in selected
            ]

        source_distances = self._profile_measurement_distances(self.measurements)
        selected_distances = [source_distances[int(index)] for index in selected_indices]
        rebased: list[Measurement] = []
        previous_distance = selected_distances[0]
        previous_timestamp = selected[0].timestamp_s
        for index, measurement in enumerate(selected):
            if index == 0:
                traveled_distance = 0.0
                speed = None
            else:
                traveled_distance = max(0.0, selected_distances[index] - previous_distance)
                dt_s = float(measurement.timestamp_s) - float(previous_timestamp)
                speed = float(traveled_distance / dt_s) if dt_s > 0.0 else None
            rebased.append(
                Measurement(
                    laser_agl_m=measurement.laser_agl_m,
                    laser_valid=measurement.laser_valid,
                    baro_msl_m=measurement.baro_msl_m,
                    sensor_heading_deg=measurement.sensor_heading_deg,
                    traveled_distance_m=float(traveled_distance),
                    measured_speed_m_s=speed,
                    timestamp_s=measurement.timestamp_s,
                )
            )
            previous_distance = selected_distances[index]
            previous_timestamp = measurement.timestamp_s
        return rebased

    def _profile_measurement_distances(
        self,
        measurements: Sequence[Measurement],
    ) -> list[float]:
        distances = []
        total = 0.0
        for index, measurement in enumerate(measurements):
            if index > 0:
                if measurement.traveled_distance_m is None:
                    raise ValueError("Distance is required outside unknown-speed mode")
                total += float(measurement.traveled_distance_m)
            distances.append(total)
        return distances

    def _offsets_from_measurements(
        self,
        measurements: Sequence[Measurement],
    ) -> list[tuple[float, float, float]]:
        offsets = []
        current_x = 0.0
        current_y = 0.0
        for index, measurement in enumerate(measurements):
            if index > 0:
                if measurement.traveled_distance_m is None:
                    raise ValueError("Distance is required outside unknown-speed mode")
                dx, dy = self.ct.offset_meters(
                    measurement.traveled_distance_m,
                    measurement.sensor_heading_deg,
                )
                current_x += dx
                current_y += dy
            offsets.append((current_x, current_y, measurement.sensor_heading_deg))
        return offsets

    def _profile_distances_m(self, candidate: Optional[Candidate] = None) -> list[float]:
        measurements = self.last_profile_measurements or self.measurements
        if self._uses_unknown_speed():
            speed = (
                candidate.estimated_speed_m_s
                if candidate is not None
                else self.last_estimated_speed_m_s
            )
            if speed is None:
                algorithm = self.config.algorithm
                speed = (
                    algorithm.speed_search_min_m_s + algorithm.speed_search_max_m_s
                ) / 2.0
            return build_distances_for_candidate_speed(measurements, speed)

        return self._profile_measurement_distances(measurements)

    def _offsets_for_candidate(self, candidate: Candidate) -> list[tuple[float, float, float]]:
        measurements = self.last_profile_measurements or self.measurements
        if not self._uses_unknown_speed():
            return self._offsets_from_measurements(measurements)
        if candidate.estimated_speed_m_s is None:
            raise ValueError("Unknown-speed candidate is missing its speed hypothesis")
        return build_offsets_for_candidate_speed(
            measurements,
            candidate.estimated_speed_m_s,
            self.ct,
        )

    def _measured_terrain_profile(
        self,
        laser_agl: np.ndarray,
        laser_valid: np.ndarray,
        baro_msl: np.ndarray,
        matched_dem: np.ndarray,
        candidate: Optional[Candidate],
    ) -> np.ndarray:
        measured = np.full(len(laser_agl), np.nan, dtype=np.float64)
        mode = self.config.sensor.altitude_mode
        if mode == "known_msl_altitude":
            measured = float(self.config.sensor.constant_msl_m) - laser_agl
        elif mode == "unknown_constant_msl_altitude":
            if candidate is not None and np.isfinite(candidate.estimated_msl_m):
                measured = float(candidate.estimated_msl_m) - laser_agl
        elif mode == "barometric_altitude":
            valid_for_bias = laser_valid & ~np.isnan(matched_dem)
            if np.any(valid_for_bias):
                bias = float(
                    np.median(
                        laser_agl[valid_for_bias]
                        + matched_dem[valid_for_bias]
                        - baro_msl[valid_for_bias]
                    )
                )
                measured = baro_msl + bias - laser_agl
            else:
                measured = baro_msl - laser_agl
        else:
            raise ValueError(f"Unknown altitude mode: {mode}")

        measured = measured.astype(np.float64, copy=False)
        measured[~laser_valid] = np.nan
        return measured

    def _build_profile_comparison(
        self,
        candidate: Optional[Candidate],
        status: str,
    ) -> ProfileComparison:
        measurements = self.last_profile_measurements or self.measurements
        laser = np.array([m.laser_agl_m for m in measurements], dtype=np.float64)
        valid = np.array([m.laser_valid for m in measurements], dtype=bool)
        baro = np.array([m.baro_msl_m for m in measurements], dtype=np.float64)
        matched_dem = np.full(len(laser), np.nan, dtype=np.float64)

        if candidate is not None:
            candidate_offsets = self._offsets_for_candidate(candidate)
        else:
            candidate_offsets = []
        if candidate is not None and candidate_offsets:
            angle_delta = candidate.heading_deg - candidate_offsets[0][2]
            matched_dem = extract_profile(
                self.nav_dem,
                candidate.row,
                candidate.col,
                rotate_offsets(candidate_offsets, angle_delta),
                self.ct,
            )

        measured = self._measured_terrain_profile(laser, valid, baro, matched_dem, candidate)
        quality_score = (
            self._candidate_quality_score(candidate) if candidate is not None else None
        )
        quality_correlation = (
            self._candidate_quality_correlation(candidate) if candidate is not None else None
        )
        return ProfileComparison(
            distances_m=self._profile_distances_m(candidate),
            measured_elevation_m=measured.tolist(),
            matched_elevation_m=matched_dem.tolist(),
            status=status,
            candidate_score=candidate.score if candidate is not None else None,
            quality_score=quality_score,
            quality_correlation=quality_correlation,
            estimated_speed_m_s=(
                candidate.estimated_speed_m_s if candidate is not None else None
            ),
        )

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

    @staticmethod
    def _inclusive_speed_values(minimum: float, maximum: float, step: float) -> list[float]:
        count = int(math.floor((maximum - minimum) / step + 1e-9))
        values = [round(minimum + index * step, 9) for index in range(count + 1)]
        if not values or values[-1] < maximum - 1e-9:
            values.append(round(maximum, 9))
        return values

    @staticmethod
    def _refined_speed_values(
        centers: list[float],
        *,
        radius: float,
        step: float,
        minimum: float,
        maximum: float,
    ) -> list[float]:
        values: set[float] = set()
        for center in centers:
            first_index = math.ceil((minimum - center) / step)
            last_index = math.floor((maximum - center) / step)
            radius_steps = int(math.ceil(radius / step))
            first_index = max(first_index, -radius_steps)
            last_index = min(last_index, radius_steps)
            for index in range(first_index, last_index + 1):
                value = center + index * step
                if minimum - 1e-9 <= value <= maximum + 1e-9:
                    values.add(round(min(maximum, max(minimum, value)), 9))
        return sorted(values)

    def _speed_seed_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        algorithm = self.config.algorithm
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                self._candidate_quality_score(candidate),
                candidate.score,
            ),
        )
        best_by_speed: dict[float, Candidate] = {}
        for candidate in ranked:
            if candidate.estimated_speed_m_s is None:
                continue
            key = round(candidate.estimated_speed_m_s, 9)
            best_by_speed.setdefault(key, candidate)
        selected_speeds = {
            round(float(candidate.estimated_speed_m_s), 9)
            for candidate in list(best_by_speed.values())[
                : algorithm.speed_search_keep_hypotheses
            ]
        }
        limit = algorithm.speed_search_keep_hypotheses * max(1, algorithm.top_k)
        return [
            candidate
            for candidate in ranked
            if candidate.estimated_speed_m_s is not None
            and round(candidate.estimated_speed_m_s, 9) in selected_speeds
        ][:limit]

    def _bounds_around_speed_seed(
        self,
        seed: Candidate,
        radius_px: int,
        parent_bounds: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        rows, cols = self.nav_dem.shape
        parent = parent_bounds or (0, rows, 0, cols)
        return (
            max(parent[0], 0, int(math.floor(seed.row - radius_px))),
            min(parent[1], rows, int(math.ceil(seed.row + radius_px + 1))),
            max(parent[2], 0, int(math.floor(seed.col - radius_px))),
            min(parent[3], cols, int(math.ceil(seed.col + radius_px + 1))),
        )

    @staticmethod
    def _merge_search_bounds(
        bounds: list[Tuple[int, int, int, int]],
    ) -> list[Tuple[int, int, int, int]]:
        merged: list[Tuple[int, int, int, int]] = []
        for candidate in bounds:
            current = candidate
            index = 0
            while index < len(merged):
                existing = merged[index]
                separated = (
                    current[1] < existing[0]
                    or existing[1] < current[0]
                    or current[3] < existing[2]
                    or existing[3] < current[2]
                )
                if separated:
                    index += 1
                    continue
                current = (
                    min(current[0], existing[0]),
                    max(current[1], existing[1]),
                    min(current[2], existing[2]),
                    max(current[3], existing[3]),
                )
                merged.pop(index)
                index = 0
            merged.append(current)
        return merged

    def _bounds_area_fraction(
        self,
        bounds: Optional[Tuple[int, int, int, int]],
    ) -> float:
        rows, cols = self.nav_dem.shape
        if bounds is None:
            return 1.0
        row_start, row_end, col_start, col_end = bounds
        area = max(0, row_end - row_start) * max(0, col_end - col_start)
        return float(area) / float(max(1, rows * cols))

    def _estimate_grid_candidates(
        self,
        bounds: Optional[Tuple[int, int, int, int]],
        stride: int,
        heading_count: int,
    ) -> int:
        rows, cols = self.nav_dem.shape
        if bounds is None:
            row_start, row_end, col_start, col_end = 0, rows, 0, cols
        else:
            row_start, row_end, col_start, col_end = bounds
        stride = max(1, int(stride))
        row_count = max(0, math.ceil(max(0, row_end - row_start) / stride))
        col_count = max(0, math.ceil(max(0, col_end - col_start) / stride))
        return int(row_count * col_count * max(1, heading_count))

    def _estimate_refinement_work(
        self,
        seeds: Sequence[Candidate],
        parent_bounds: Optional[Tuple[int, int, int, int]],
        radius_px: int,
        stride: int,
        heading_count: int,
    ) -> tuple[int, float]:
        bounds = [
            self._bounds_around_speed_seed(seed, radius_px, parent_bounds)
            for seed in seeds
        ]
        merged_bounds = self._merge_search_bounds(bounds)
        candidate_count = sum(
            self._estimate_grid_candidates(bound, stride, heading_count)
            for bound in bounds
        )
        area_fraction = sum(self._bounds_area_fraction(bound) for bound in merged_bounds)
        return candidate_count, area_fraction

    def _search_speed_stage(
        self,
        measurements: Sequence[Measurement],
        laser: np.ndarray,
        valid: np.ndarray,
        baro: np.ndarray,
        search_headings: list[float],
        speeds: list[float],
        parent_bounds: Optional[Tuple[int, int, int, int]],
        *,
        stage_name: str,
        stride: int,
        seeds: Optional[list[Candidate]] = None,
        seed_radius_px: int = 0,
        seed_speed_radius_m_s: float | None = None,
        seed_limit: int | None = None,
        fine_heading_refinement: bool = False,
    ) -> list[Candidate]:
        stage_started = time.perf_counter()
        unique_candidates: dict[tuple[float, float, float, float], Candidate] = {}
        dem_searches = 0
        spatial_candidates = 0
        searched_area = 0.0
        if seeds and fine_heading_refinement and len(search_headings) > 1:
            estimated_heading_count = 3
        elif seeds and len(search_headings) > 1:
            estimated_heading_count = min(5, len(search_headings))
        else:
            estimated_heading_count = max(1, len(search_headings))

        for speed in speeds:
            offsets = build_offsets_for_candidate_speed(measurements, speed, self.ct)
            if seeds:
                stage_seeds = self._stage_seeds_for_speed(
                    seeds,
                    speed,
                    speed_radius_m_s=seed_speed_radius_m_s,
                    limit=seed_limit,
                )
                if not stage_seeds:
                    continue
                stage_headings = search_headings
                if fine_heading_refinement and len(search_headings) > 1:
                    step = self.config.algorithm.fine_heading_step_deg
                    stage_headings = sorted(
                        {
                            (seed.heading_deg + delta) % 360.0
                            for seed in stage_seeds
                            for delta in (-step, 0.0, step)
                        }
                    )
                if fine_heading_refinement:
                    candidates = self.matcher.refine_around_candidates(
                        laser,
                        valid,
                        baro,
                        offsets,
                        stage_seeds,
                        search_headings,
                        stride=stride,
                        radius_px=seed_radius_px,
                        search_bounds=parent_bounds,
                        fine_heading_refinement=True,
                    )
                    dem_searches += 1
                    count, area = self._estimate_refinement_work(
                        stage_seeds,
                        parent_bounds,
                        seed_radius_px,
                        stride,
                        len(stage_headings),
                    )
                    spatial_candidates += count
                    searched_area += area
                else:
                    stage_bounds = self._merge_search_bounds(
                        [
                            self._bounds_around_speed_seed(
                                seed,
                                seed_radius_px,
                                parent_bounds,
                            )
                            for seed in stage_seeds
                        ]
                    )
                    candidates = []
                    for bounds in stage_bounds:
                        stage_candidates = self.matcher.exhaustive_search(
                            laser,
                            valid,
                            baro,
                            offsets,
                            stage_headings,
                            stride=stride,
                            search_bounds=bounds,
                        )
                        candidates.extend(stage_candidates)
                        dem_searches += 1
                        spatial_candidates += self._estimate_grid_candidates(
                            bounds,
                            stride,
                            len(stage_headings),
                        )
                        searched_area += self._bounds_area_fraction(bounds)
            else:
                candidates = self.matcher.coarse_search(
                    laser,
                    valid,
                    baro,
                    offsets,
                    search_headings,
                    stride=stride,
                    search_bounds=parent_bounds,
                )
                dem_searches += 1
                spatial_candidates += self._estimate_grid_candidates(
                    parent_bounds,
                    stride,
                    estimated_heading_count,
                )
                searched_area += self._bounds_area_fraction(parent_bounds)

            for candidate in candidates:
                tagged = replace(candidate, estimated_speed_m_s=float(speed))
                key = (
                    round(tagged.row, 3),
                    round(tagged.col, 3),
                    round(tagged.heading_deg, 3),
                    round(float(speed), 9),
                )
                previous = unique_candidates.get(key)
                if previous is None or tagged.score < previous.score:
                    unique_candidates[key] = tagged

        if hasattr(self, "_current_speed_search_profile"):
            profile = self._current_speed_search_profile
            profile[f"{stage_name}_speed_search_ms"] = (
                time.perf_counter() - stage_started
            ) * 1000.0
            profile["speed_hypotheses_evaluated"] = profile.get(
                "speed_hypotheses_evaluated",
                0.0,
            ) + float(len(speeds))
            profile["dem_searches"] = profile.get("dem_searches", 0.0) + float(dem_searches)
            profile["spatial_candidates_evaluated"] = profile.get(
                "spatial_candidates_evaluated",
                0.0,
            ) + float(spatial_candidates)
            profile["searched_dem_area_fraction_sum"] = profile.get(
                "searched_dem_area_fraction_sum",
                0.0,
            ) + float(searched_area)
        return sorted(
            unique_candidates.values(),
            key=lambda candidate: (
                self._candidate_quality_score(candidate),
                candidate.score,
            ),
        )

    def _stage_seeds_for_speed(
        self,
        seeds: Sequence[Candidate],
        speed: float,
        *,
        speed_radius_m_s: float | None,
        limit: int | None,
    ) -> list[Candidate]:
        usable = [
            seed
            for seed in seeds
            if seed.estimated_speed_m_s is not None
            and (
                speed_radius_m_s is None
                or abs(float(seed.estimated_speed_m_s) - float(speed)) <= speed_radius_m_s + 1e-9
            )
        ]
        usable.sort(
            key=lambda seed: (
                abs(float(seed.estimated_speed_m_s or 0.0) - float(speed)),
                self._candidate_quality_score(seed),
                seed.score,
            )
        )
        selected_limit = int(limit) if limit is not None else int(self.config.algorithm.top_k)
        return usable[: max(1, selected_limit)]

    def _unknown_speed_candidates(
        self,
        measurements: Sequence[Measurement],
        laser: np.ndarray,
        valid: np.ndarray,
        baro: np.ndarray,
        search_headings: list[float],
        search_bounds: Optional[Tuple[int, int, int, int]],
    ) -> list[Candidate]:
        """Jointly refine position and one constant speed over the profile window."""
        algorithm = self.config.algorithm
        minimum = float(algorithm.speed_search_min_m_s)
        maximum = float(algorithm.speed_search_max_m_s)
        coarse_step = float(algorithm.speed_search_coarse_step_m_s)
        medium_step = float(algorithm.speed_search_medium_step_m_s)
        fine_step = float(algorithm.speed_search_fine_step_m_s)
        tracking_speed = (
            self.last_estimated_speed_m_s
            if self.last_match_pixel is not None and not self.recovery_active
            else None
        )
        tracking_speed = (
            float(tracking_speed)
            if tracking_speed is not None
            and math.isfinite(float(tracking_speed))
            and minimum <= float(tracking_speed) <= maximum
            else None
        )
        if tracking_speed is not None:
            tracking_half_range = float(algorithm.speed_tracking_half_range_m_s)
            tracking_step = float(algorithm.speed_tracking_step_m_s)
            speed_minimum = max(minimum, tracking_speed - tracking_half_range)
            speed_maximum = min(maximum, tracking_speed + tracking_half_range)
            coarse_speeds = self._inclusive_speed_values(
                speed_minimum,
                speed_maximum,
                tracking_step,
            )
            medium_radius = max(medium_step, tracking_half_range)
        else:
            speed_minimum = minimum
            speed_maximum = maximum
            coarse_speeds = self._inclusive_speed_values(minimum, maximum, coarse_step)
            medium_radius = coarse_step
        self._current_speed_search_profile = {
            "coarse_speed_search_ms": 0.0,
            "medium_speed_search_ms": 0.0,
            "fine_speed_search_ms": 0.0,
            "speed_hypotheses_evaluated": 0.0,
            "dem_searches": 0.0,
            "spatial_candidates_evaluated": 0.0,
            "searched_dem_area_fraction_sum": 0.0,
        }

        coarse = self._search_speed_stage(
            measurements,
            laser,
            valid,
            baro,
            search_headings,
            coarse_speeds,
            search_bounds,
            stage_name="coarse",
            stride=algorithm.coarse_stride,
        )
        coarse_seeds = self._speed_seed_candidates(coarse)
        if not coarse_seeds:
            return []

        medium_centers = sorted(
            {float(candidate.estimated_speed_m_s) for candidate in coarse_seeds}
        )
        medium_speeds = self._refined_speed_values(
            medium_centers,
            radius=medium_radius,
            step=medium_step,
            minimum=speed_minimum,
            maximum=speed_maximum,
        )
        medium = self._search_speed_stage(
            measurements,
            laser,
            valid,
            baro,
            search_headings,
            medium_speeds,
            search_bounds,
            stage_name="medium",
            stride=algorithm.medium_stride,
            seeds=coarse_seeds,
            seed_radius_px=max(
                algorithm.refinement_radius_px * 2,
                algorithm.coarse_stride * 2,
            ),
            seed_speed_radius_m_s=medium_radius,
        )
        medium_seeds = self._speed_seed_candidates(medium or coarse)

        fine_centers = sorted(
            {float(candidate.estimated_speed_m_s) for candidate in medium_seeds}
        )
        fine_speeds = self._refined_speed_values(
            fine_centers,
            radius=medium_step,
            step=fine_step,
            minimum=speed_minimum,
            maximum=speed_maximum,
        )
        fine = self._search_speed_stage(
            measurements,
            laser,
            valid,
            baro,
            search_headings,
            fine_speeds,
            search_bounds,
            stage_name="fine",
            stride=algorithm.fine_stride,
            seeds=medium_seeds,
            seed_radius_px=max(
                algorithm.refinement_radius_px,
                algorithm.medium_stride * 2,
            ),
            seed_speed_radius_m_s=medium_step,
            seed_limit=min(
                int(algorithm.top_k),
                int(algorithm.speed_search_keep_hypotheses),
            ),
            fine_heading_refinement=True,
        )

        combined: dict[tuple[float, float, float, float], Candidate] = {}
        for candidate in [*coarse, *medium, *fine]:
            if candidate.estimated_speed_m_s is None:
                continue
            key = (
                round(candidate.row, 3),
                round(candidate.col, 3),
                round(candidate.heading_deg, 3),
                round(candidate.estimated_speed_m_s, 9),
            )
            previous = combined.get(key)
            if previous is None or candidate.score < previous.score:
                combined[key] = candidate
        return sorted(
            combined.values(),
            key=lambda candidate: (
                self._candidate_quality_score(candidate),
                candidate.score,
            ),
        )

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
                "motion_mode": self.config.motion_mode,
                "estimated_speed_m_s": self.last_estimated_speed_m_s,
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
            "motion_mode": self.config.motion_mode,
            "estimated_speed_m_s": self.last_estimated_speed_m_s,
        }

    def get_profile_comparison(self) -> Optional[ProfileComparison]:
        """Return the latest measured-vs-candidate terrain profile for UI display."""
        return self.last_profile_comparison

    def get_runtime_profile(self) -> dict[str, float]:
        """Return timing/counter data for the most recent localization call."""
        return dict(self.last_runtime_profile)

    def localize(self, timestamp: float) -> Optional[EstimatedState]:
        started = time.perf_counter()
        timing: dict[str, float] = {
            "coarse_speed_search_ms": 0.0,
            "medium_speed_search_ms": 0.0,
            "fine_speed_search_ms": 0.0,
            "dem_profile_matching_ms": 0.0,
            "quality_gate_ms": 0.0,
            "position_ambiguity_ms": 0.0,
            "speed_ambiguity_ms": 0.0,
            "total_localization_ms": 0.0,
            "speed_hypotheses_evaluated": 0.0,
            "dem_searches": 0.0,
            "spatial_candidates_evaluated": 0.0,
            "searched_dem_area_fraction_sum": 0.0,
            "profile_point_count": 0.0,
        }

        def finish() -> None:
            timing["total_localization_ms"] = (time.perf_counter() - started) * 1000.0
            timing["profile_point_count"] = float(len(self.last_profile_measurements))
            self.last_runtime_profile = dict(timing)

        algorithm = self.config.algorithm
        profile_incomplete = len(self.measurements) < algorithm.min_profile_length
        if self._uses_unknown_speed():
            profile_incomplete = (
                profile_incomplete
                or self._profile_duration_s() < algorithm.min_profile_duration_s
            )
        else:
            profile_incomplete = (
                profile_incomplete
                or self._profile_distance_m() < algorithm.min_profile_distance_m
            )
        if profile_incomplete:
            self.last_profile_measurements = list(self.measurements)
            self.last_rejection_reason = "profile_incomplete"
            self.last_rejected_score = None
            self.last_profile_comparison = self._build_profile_comparison(
                None,
                "profile_incomplete",
            )
            finish()
            return None

        active_measurements = self._measurements_for_localization()
        self.last_profile_measurements = active_measurements
        if len(active_measurements) < algorithm.min_profile_length:
            self.last_rejection_reason = "profile_incomplete"
            self.last_rejected_score = None
            self.last_profile_comparison = self._build_profile_comparison(
                None,
                "profile_incomplete",
            )
            finish()
            return None

        laser = np.array([m.laser_agl_m for m in active_measurements], dtype=np.float64)
        valid = np.array([m.laser_valid for m in active_measurements], dtype=bool)
        baro = np.array([m.baro_msl_m for m in active_measurements], dtype=np.float64)

        # Heading changes within the window remain in the per-sample offsets.
        base_heading = float(active_measurements[0].sensor_heading_deg)
        if self.config.sensor.heading_mode == "known_heading":
            search_headings = [base_heading]
        elif self.config.sensor.heading_mode == "noisy_heading":
            unc = self.config.sensor.heading_uncertainty_deg
            search_headings = np.arange(
                base_heading - unc,
                base_heading + unc + 0.1,
                1.0,
            ).tolist()
        else:  # unknown_heading
            search_headings = np.arange(0.0, 360.0, 5.0).tolist()  # Every 5 degrees initially

        search_bounds = self._search_bounds()
        search_started = time.perf_counter()
        if self._uses_unknown_speed():
            cands = self._unknown_speed_candidates(
                active_measurements,
                laser,
                valid,
                baro,
                search_headings,
                search_bounds,
            )
            timing.update(getattr(self, "_current_speed_search_profile", {}))
        else:
            active_offsets = self._offsets_from_measurements(active_measurements)
            cands = self.matcher.coarse_to_fine_search(
                laser,
                valid,
                baro,
                active_offsets,
                search_headings,
                search_bounds=search_bounds,
            )
            timing["dem_searches"] = 1.0
            timing["searched_dem_area_fraction_sum"] = self._bounds_area_fraction(search_bounds)
            timing["spatial_candidates_evaluated"] = float(
                self._estimate_grid_candidates(
                    search_bounds,
                    self.config.algorithm.coarse_stride,
                    len(search_headings),
                )
            )
        timing["dem_profile_matching_ms"] = (time.perf_counter() - search_started) * 1000.0

        if not cands:
            self.last_rejection_reason = "no_candidates"
            self.last_rejected_score = None
            self.last_profile_comparison = self._build_profile_comparison(None, "no_candidates")
            self._reject_search_result(search_bounds)
            finish()
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
                self.last_profile_comparison = self._build_profile_comparison(
                    cands[0],
                    "continuity_rejected",
                )
                self._reject_search_result(search_bounds)
                finish()
                return None
            cands = plausible_cands

        speed_confidence_candidates = cands
        quality_started = time.perf_counter()
        quality_cands = [
            candidate for candidate in cands if self._candidate_passes_quality_gate(candidate)
        ]
        timing["quality_gate_ms"] = (time.perf_counter() - quality_started) * 1000.0
        if not quality_cands:
            self.last_rejection_reason = "quality"
            self.last_rejected_score = self._candidate_quality_score(cands[0])
            self.last_profile_comparison = self._build_profile_comparison(
                cands[0],
                "quality_rejected",
            )
            self._reject_search_result(search_bounds)
            finish()
            return None
        cands = sorted(
            quality_cands,
            key=lambda candidate: (self._candidate_quality_score(candidate), candidate.score),
        )

        ambiguity_started = time.perf_counter()
        is_ambiguous, margin, spread = detect_ambiguity(
            cands,
            score_getter=self._candidate_quality_score,
        )
        timing["position_ambiguity_ms"] = (time.perf_counter() - ambiguity_started) * 1000.0
        speed_confidence = SpeedConfidence(False, None, None, None, None, "unavailable")
        if self._uses_unknown_speed():
            speed_started = time.perf_counter()
            speed_confidence = assess_speed_confidence(
                speed_confidence_candidates,
                score_margin_threshold=algorithm.speed_ambiguity_score_margin,
                speed_std_threshold_m_s=algorithm.speed_ambiguity_std_threshold_m_s,
                top_k=algorithm.speed_ambiguity_top_k,
                score_getter=self._candidate_quality_score,
            )
            timing["speed_ambiguity_ms"] = (time.perf_counter() - speed_started) * 1000.0
            is_ambiguous = is_ambiguous or speed_confidence.is_ambiguous
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
            if best.estimated_speed_m_s is not None:
                self.last_estimated_speed_m_s = best.estimated_speed_m_s
            self.current_search_roi_size = self.base_search_roi_size
            self.recovery_active = False

        profile_status = "ambiguous" if is_ambiguous else "fix"
        if speed_confidence.is_ambiguous:
            profile_status = "speed_ambiguous"
        self.last_profile_comparison = self._build_profile_comparison(
            best,
            profile_status,
        )

        # `best.row`, `best.col` is the location of the *start* of the window (the oldest point).
        # We want the location of the *current* point (the newest point).
        # The newest point is at relative offset corresponding to the last measurement.
        best_offsets = self._offsets_for_candidate(best)
        last_offset = best_offsets[-1]

        # Need to rotate last_offset by the matched heading difference.
        # Wait, the offset rotation logic is in `rotate_offsets` and is applied during extraction.
        # Let's do it manually for the last point.
        angle_diff_deg = best.heading_deg - best_offsets[0][2]
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
        # Compass headings increase clockwise, while the Cartesian offset
        # rotation above is counter-clockwise.
        curr_h = (last_offset[2] - angle_diff_deg) % 360.0

        curr_x, curr_y = self.ct.pixel_to_world(curr_row, curr_col)

        estimated = EstimatedState(
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
            quality_valid_ratio=best.valid_ratio,
            estimated_speed_m_s=best.estimated_speed_m_s,
            second_best_speed_m_s=speed_confidence.second_best_speed_m_s,
            speed_is_ambiguous=speed_confidence.is_ambiguous,
            speed_score_margin=speed_confidence.score_margin,
            speed_spread_m_s=speed_confidence.top_speed_std_m_s,
            speed_confidence=speed_confidence.indicator,
        )
        finish()
        return estimated


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
        # In manual mode this remains a command-count hint for the UI; each
        # command may still generate several sensor/localization samples.
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

    def get_profile_comparison(self) -> Optional[ProfileComparison]:
        """Return the latest measured-vs-matched terrain profile."""
        return self.localization.get_profile_comparison()

    def get_runtime_profile(self) -> dict[str, float]:
        """Return timing/counter data for the latest localization call."""
        return self.localization.get_runtime_profile()

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
            timestamp_s=self.elapsed_s,
            motion_mode=self.config.motion_mode,
        )

        self.localization.add_measurement(m)
        est = self.localization.localize(m.timestamp_s)
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
        sample_spacing_m = max(1e-9, float(self.config.route.manual_sample_spacing_m))
        segment_distances = []
        remaining_distance_m = distance_m
        while remaining_distance_m > 1e-9:
            segment_distance_m = min(sample_spacing_m, remaining_distance_m)
            segment_distances.append(segment_distance_m)
            remaining_distance_m -= segment_distance_m

        planned_segments = []
        cursor_x = self.dynamic_x
        cursor_y = self.dynamic_y
        for segment_distance_m in segment_distances:
            dx, dy = self.ct.offset_meters(segment_distance_m, movement_heading)
            cursor_x += dx
            cursor_y += dy
            self._validate_world_position(cursor_x, cursor_y)
            planned_segments.append((cursor_x, cursor_y))

        last_result = None
        for segment_distance_m, (next_x, next_y) in zip(
            segment_distances,
            planned_segments,
            strict=True,
        ):
            self.dynamic_x = next_x
            self.dynamic_y = next_y
            motion_dt_s = segment_distance_m / max(self.config.route.speed_m_s, 1e-9)
            last_result = self._sample_current(
                segment_distance_m,
                movement_heading,
                motion_dt_s,
            )

        if last_result is None:
            raise RuntimeError("Manual movement did not produce a sensor sample")
        return last_result

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
