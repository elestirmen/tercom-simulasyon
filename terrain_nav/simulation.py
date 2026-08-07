"""End-to-end simulation engine with ground truth isolation."""

import math
from dataclasses import replace
from typing import Optional, Tuple

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

    def _profile_distances_m(self, candidate: Optional[Candidate] = None) -> list[float]:
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
            return build_distances_for_candidate_speed(self.measurements, speed)

        distances = []
        total = 0.0
        for index, measurement in enumerate(self.measurements):
            if index > 0:
                if measurement.traveled_distance_m is None:
                    raise ValueError("Distance is required outside unknown-speed mode")
                total += float(measurement.traveled_distance_m)
            distances.append(total)
        return distances

    def _offsets_for_candidate(self, candidate: Candidate) -> list[tuple[float, float, float]]:
        if not self._uses_unknown_speed():
            return self.relative_offsets
        if candidate.estimated_speed_m_s is None:
            raise ValueError("Unknown-speed candidate is missing its speed hypothesis")
        return build_offsets_for_candidate_speed(
            self.measurements,
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
        laser = np.array([m.laser_agl_m for m in self.measurements], dtype=np.float64)
        valid = np.array([m.laser_valid for m in self.measurements], dtype=bool)
        baro = np.array([m.baro_msl_m for m in self.measurements], dtype=np.float64)
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

    def _search_speed_stage(
        self,
        laser: np.ndarray,
        valid: np.ndarray,
        baro: np.ndarray,
        search_headings: list[float],
        speeds: list[float],
        parent_bounds: Optional[Tuple[int, int, int, int]],
        *,
        seeds: Optional[list[Candidate]] = None,
        seed_radius_px: int = 0,
    ) -> list[Candidate]:
        unique_candidates: dict[tuple[float, float, float, float], Candidate] = {}
        for speed in speeds:
            offsets = build_offsets_for_candidate_speed(self.measurements, speed, self.ct)
            if seeds:
                stage_bounds = self._merge_search_bounds(
                    [
                        self._bounds_around_speed_seed(
                            seed,
                            seed_radius_px,
                            parent_bounds,
                        )
                        for seed in seeds
                    ]
                )
            else:
                stage_bounds = [parent_bounds]

            for bounds in stage_bounds:
                candidates = self.matcher.coarse_to_fine_search(
                    laser,
                    valid,
                    baro,
                    offsets,
                    search_headings,
                    search_bounds=bounds,
                )
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
        return sorted(
            unique_candidates.values(),
            key=lambda candidate: (
                self._candidate_quality_score(candidate),
                candidate.score,
            ),
        )

    def _unknown_speed_candidates(
        self,
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

        coarse_speeds = self._inclusive_speed_values(minimum, maximum, coarse_step)
        coarse = self._search_speed_stage(
            laser,
            valid,
            baro,
            search_headings,
            coarse_speeds,
            search_bounds,
        )
        coarse_seeds = self._speed_seed_candidates(coarse)
        if not coarse_seeds:
            return []

        medium_centers = sorted(
            {float(candidate.estimated_speed_m_s) for candidate in coarse_seeds}
        )
        medium_speeds = self._refined_speed_values(
            medium_centers,
            radius=coarse_step,
            step=medium_step,
            minimum=minimum,
            maximum=maximum,
        )
        medium = self._search_speed_stage(
            laser,
            valid,
            baro,
            search_headings,
            medium_speeds,
            search_bounds,
            seeds=coarse_seeds,
            seed_radius_px=max(
                algorithm.refinement_radius_px * 2,
                algorithm.coarse_stride * 2,
            ),
        )
        medium_seeds = self._speed_seed_candidates(medium or coarse)

        fine_centers = sorted(
            {float(candidate.estimated_speed_m_s) for candidate in medium_seeds}
        )
        fine_speeds = self._refined_speed_values(
            fine_centers,
            radius=medium_step,
            step=fine_step,
            minimum=minimum,
            maximum=maximum,
        )
        fine = self._search_speed_stage(
            laser,
            valid,
            baro,
            search_headings,
            fine_speeds,
            search_bounds,
            seeds=medium_seeds,
            seed_radius_px=max(
                algorithm.refinement_radius_px,
                algorithm.medium_stride * 2,
            ),
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

    def localize(self, timestamp: float) -> Optional[EstimatedState]:
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
            self.last_rejection_reason = "profile_incomplete"
            self.last_rejected_score = None
            self.last_profile_comparison = self._build_profile_comparison(
                None,
                "profile_incomplete",
            )
            return None

        laser = np.array([m.laser_agl_m for m in self.measurements])
        valid = np.array([m.laser_valid for m in self.measurements])
        baro = np.array([m.baro_msl_m for m in self.measurements])

        # Heading changes within the window remain in the per-sample offsets.
        base_heading = float(self.measurements[0].sensor_heading_deg)
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
        if self._uses_unknown_speed():
            cands = self._unknown_speed_candidates(
                laser,
                valid,
                baro,
                search_headings,
                search_bounds,
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
            self.last_rejection_reason = "no_candidates"
            self.last_rejected_score = None
            self.last_profile_comparison = self._build_profile_comparison(None, "no_candidates")
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
                self.last_profile_comparison = self._build_profile_comparison(
                    cands[0],
                    "continuity_rejected",
                )
                self._reject_search_result(search_bounds)
                return None
            cands = plausible_cands

        speed_confidence_candidates = cands
        quality_cands = [
            candidate for candidate in cands if self._candidate_passes_quality_gate(candidate)
        ]
        if not quality_cands:
            self.last_rejection_reason = "quality"
            self.last_rejected_score = self._candidate_quality_score(cands[0])
            self.last_profile_comparison = self._build_profile_comparison(
                cands[0],
                "quality_rejected",
            )
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
        speed_confidence = SpeedConfidence(False, None, None, None, None, "unavailable")
        if self._uses_unknown_speed():
            speed_confidence = assess_speed_confidence(
                speed_confidence_candidates,
                score_margin_threshold=algorithm.speed_ambiguity_score_margin,
                speed_std_threshold_m_s=algorithm.speed_ambiguity_std_threshold_m_s,
                top_k=algorithm.speed_ambiguity_top_k,
                score_getter=self._candidate_quality_score,
            )
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
            quality_valid_ratio=best.valid_ratio,
            estimated_speed_m_s=best.estimated_speed_m_s,
            second_best_speed_m_s=speed_confidence.second_best_speed_m_s,
            speed_is_ambiguous=speed_confidence.is_ambiguous,
            speed_score_margin=speed_confidence.score_margin,
            speed_spread_m_s=speed_confidence.top_speed_std_m_s,
            speed_confidence=speed_confidence.indicator,
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
