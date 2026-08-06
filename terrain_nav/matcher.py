"""Profile matching algorithms and metrics."""

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.profile import extract_profile_pixels, rotate_offsets


@dataclass
class Candidate:
    row: float
    col: float
    heading_deg: float
    estimated_msl_m: float
    score: float  # Lower is better (error)
    valid_ratio: float
    metrics: Dict[str, float]


def huber_loss(error: np.ndarray, delta: float) -> float:
    abs_e = np.abs(error)
    quadratic = np.minimum(abs_e, delta)
    linear = abs_e - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return float(np.mean(loss))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    a_m = a - np.mean(a)
    b_m = b - np.mean(b)
    denom = np.sqrt(np.sum(a_m**2) * np.sum(b_m**2))
    if denom == 0:
        return 0.0
    return float(np.sum(a_m * b_m) / denom)


def evaluate_candidate(
    candidate_row: float,
    candidate_col: float,
    candidate_heading_deg: float,
    dem: np.ndarray,
    laser_agl: np.ndarray,
    laser_valid: np.ndarray,
    baro_msl: np.ndarray,
    base_offsets: List[Tuple[float, float, float]],
    ct: CoordinateTransform,
    config: LocalizationConfig,
    pixel_offsets: Optional[np.ndarray] = None,
    compute_all_metrics: bool = True,
) -> Optional[Candidate]:
    # Check bounds coarsely before full extraction
    rows, cols = dem.shape
    if candidate_row < 0 or candidate_row >= rows or candidate_col < 0 or candidate_col >= cols:
        return None

    if pixel_offsets is None:
        angle_delta = candidate_heading_deg - base_offsets[0][2] if base_offsets else 0.0
        rotated_offsets = rotate_offsets(base_offsets, angle_delta)
        pixel_offsets = np.asarray(
            [(-dy / ct.dy, dx / ct.dx) for dx, dy, _ in rotated_offsets],
            dtype=np.float64,
        )
    dem_prof = extract_profile_pixels(dem, candidate_row, candidate_col, pixel_offsets)

    # Valid mask
    valid_mask = laser_valid & ~np.isnan(dem_prof)
    valid_count = np.sum(valid_mask)
    if valid_count < config.algorithm.min_profile_length:
        return None

    v_laser = laser_agl[valid_mask]
    v_dem = dem_prof[valid_mask]
    v_baro = baro_msl[valid_mask]

    estimated_msl = 0.0

    if config.sensor.altitude_mode == "known_msl_altitude":
        estimated_msl = config.sensor.constant_msl_m
        expected_laser = estimated_msl - v_dem
    elif config.sensor.altitude_mode == "unknown_constant_msl_altitude":
        # H = median(laser + DEM)
        estimated_msl = float(np.median(v_laser + v_dem))
        expected_laser = estimated_msl - v_dem
    elif config.sensor.altitude_mode == "barometric_altitude":
        # H is dynamic from barometer.
        # Baro gives estimated MSL, but with bias/drift.
        # For simplicity, we can do H_est = median(laser + DEM - baro) to find the bias
        bias = float(np.median(v_laser + v_dem - v_baro))
        estimated_msl = float(np.mean(v_baro + bias))  # mean over profile as a representative value
        expected_laser = (v_baro + bias) - v_dem
    else:
        raise ValueError(f"Unknown altitude mode: {config.sensor.altitude_mode}")

    error = v_laser - expected_laser

    loss_method = config.algorithm.loss_method
    rmse = (
        float(np.sqrt(np.mean(error**2))) if compute_all_metrics or loss_method == "rmse" else 0.0
    )
    mae = float(np.mean(np.abs(error))) if compute_all_metrics or loss_method == "mae" else 0.0
    huber = (
        huber_loss(error, config.algorithm.huber_delta)
        if compute_all_metrics or loss_method == "huber"
        else 0.0
    )
    score = {"rmse": rmse, "mae": mae, "huber": huber}.get(loss_method, rmse)

    metrics = {}
    if compute_all_metrics:
        median_abs_error = float(np.median(np.abs(error)))
        corr = pearson_corr(v_laser, expected_laser)
        diff_rmse = 0.0
        diff2_rmse = 0.0
        if valid_count > 1:
            d_laser = np.diff(v_laser)
            d_expected = np.diff(expected_laser)
            diff_rmse = float(np.sqrt(np.mean((d_laser - d_expected) ** 2)))
        if valid_count > 2:
            d2_laser = np.diff(v_laser, n=2)
            d2_expected = np.diff(expected_laser, n=2)
            diff2_rmse = float(np.sqrt(np.mean((d2_laser - d2_expected) ** 2)))

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "median_abs_error": median_abs_error,
            "huber": huber,
            "correlation": corr,
            "diff_rmse": diff_rmse,
            "diff2_rmse": diff2_rmse,
        }

    return Candidate(
        row=candidate_row,
        col=candidate_col,
        heading_deg=candidate_heading_deg,
        estimated_msl_m=estimated_msl,
        score=score,
        valid_ratio=float(valid_count) / len(laser_agl),
        metrics=metrics,
    )


class ProfileMatcher:
    def __init__(self, config: LocalizationConfig, dem: np.ndarray, ct: CoordinateTransform):
        self.config = config
        self.dem = dem
        self.ct = ct

    def _pixel_offsets_for_heading(
        self,
        cache: Dict[float, np.ndarray],
        base_offsets: List[Tuple[float, float, float]],
        heading_deg: float,
    ) -> np.ndarray:
        key = float(heading_deg)
        if key not in cache:
            angle_delta = key - base_offsets[0][2] if base_offsets else 0.0
            rotated = rotate_offsets(base_offsets, angle_delta)
            cache[key] = np.asarray(
                [(-dy / self.ct.dy, dx / self.ct.dx) for dx, dy, _ in rotated],
                dtype=np.float64,
            )
        return cache[key]

    @staticmethod
    def _retain_top_candidate(
        heap: List[Tuple[float, int, Candidate]],
        candidate: Candidate,
        sequence: int,
        limit: int,
    ) -> None:
        entry = (-candidate.score, -sequence, candidate)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif candidate.score < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    @staticmethod
    def _sorted_heap_candidates(
        heap: List[Tuple[float, int, Candidate]],
    ) -> List[Candidate]:
        ordered = sorted(heap, key=lambda entry: (entry[2].score, -entry[1]))
        return [entry[2] for entry in ordered]

    def _hydrate_candidates(
        self,
        candidates: List[Candidate],
        laser_agl: np.ndarray,
        laser_valid: np.ndarray,
        baro_msl: np.ndarray,
        base_offsets: List[Tuple[float, float, float]],
        pixel_offset_cache: Dict[float, np.ndarray],
    ) -> List[Candidate]:
        hydrated = []
        for candidate in candidates:
            full_candidate = evaluate_candidate(
                candidate.row,
                candidate.col,
                candidate.heading_deg,
                self.dem,
                laser_agl,
                laser_valid,
                baro_msl,
                base_offsets,
                self.ct,
                self.config,
                pixel_offsets=self._pixel_offsets_for_heading(
                    pixel_offset_cache, base_offsets, candidate.heading_deg
                ),
                compute_all_metrics=True,
            )
            if full_candidate is not None:
                hydrated.append(full_candidate)
        hydrated.sort(key=lambda candidate: candidate.score)
        return hydrated

    def _vectorized_known_altitude_search(
        self,
        laser_agl: np.ndarray,
        laser_valid: np.ndarray,
        baro_msl: np.ndarray,
        base_offsets: List[Tuple[float, float, float]],
        heading_deg: float,
        stride: int,
        search_bounds: Optional[Tuple[int, int, int, int]],
    ) -> List[Candidate]:
        """Score a known-heading coarse grid in bulk using bounded RAM arrays."""
        rows, cols = self.dem.shape
        stride = max(1, int(stride))
        if search_bounds is None:
            row_start, row_end, col_start, col_end = 0, rows, 0, cols
        else:
            row_start, row_end, col_start, col_end = search_bounds
            row_start = max(0, min(rows, int(row_start)))
            row_end = max(row_start, min(rows, int(row_end)))
            col_start = max(0, min(cols, int(col_start)))
            col_end = max(col_start, min(cols, int(col_end)))

        row_start = ((row_start + stride - 1) // stride) * stride
        col_start = ((col_start + stride - 1) // stride) * stride
        row_values = np.arange(row_start, row_end, stride, dtype=np.int32)
        col_values = np.arange(col_start, col_end, stride, dtype=np.int32)
        if row_values.size == 0 or col_values.size == 0:
            return []

        anchor_rows, anchor_cols = np.meshgrid(
            row_values,
            col_values,
            indexing="ij",
        )
        anchor_rows = anchor_rows.ravel()
        anchor_cols = anchor_cols.ravel()
        candidate_count = anchor_rows.size
        loss_sum = np.zeros(candidate_count, dtype=np.float64)
        valid_counts = np.zeros(candidate_count, dtype=np.uint16)
        pixel_offset_cache: Dict[float, np.ndarray] = {}
        pixel_offsets = self._pixel_offsets_for_heading(
            pixel_offset_cache,
            base_offsets,
            heading_deg,
        )
        altitude_msl = float(self.config.sensor.constant_msl_m)
        loss_method = self.config.algorithm.loss_method
        huber_delta = float(self.config.algorithm.huber_delta)

        for measurement_index, measurement_valid in enumerate(laser_valid):
            if not measurement_valid:
                continue
            d_row, d_col = pixel_offsets[measurement_index]
            row_base = math.floor(d_row)
            col_base = math.floor(d_col)
            row_fraction = d_row - row_base
            col_fraction = d_col - col_base
            sample_rows = anchor_rows + row_base
            sample_cols = anchor_cols + col_base
            in_bounds = (
                (sample_rows >= 0)
                & (sample_rows < rows - 1)
                & (sample_cols >= 0)
                & (sample_cols < cols - 1)
            )
            if not np.any(in_bounds):
                continue

            indices = np.flatnonzero(in_bounds)
            valid_rows = sample_rows[in_bounds]
            valid_cols = sample_cols[in_bounds]
            top = (
                self.dem[valid_rows, valid_cols] * (1.0 - col_fraction)
                + self.dem[valid_rows, valid_cols + 1] * col_fraction
            )
            bottom = (
                self.dem[valid_rows + 1, valid_cols] * (1.0 - col_fraction)
                + self.dem[valid_rows + 1, valid_cols + 1] * col_fraction
            )
            sampled_dem = top * (1.0 - row_fraction) + bottom * row_fraction
            error = laser_agl[measurement_index] - altitude_msl + sampled_dem

            if loss_method == "mae":
                loss = np.abs(error)
            elif loss_method == "huber":
                absolute_error = np.abs(error)
                quadratic = np.minimum(absolute_error, huber_delta)
                loss = 0.5 * quadratic**2 + huber_delta * (absolute_error - quadratic)
            else:
                loss = error**2
            loss_sum[indices] += loss
            valid_counts[indices] += 1

        eligible = valid_counts >= self.config.algorithm.min_profile_length
        if not np.any(eligible):
            return []

        scores = np.full(candidate_count, np.inf, dtype=np.float64)
        scores[eligible] = loss_sum[eligible] / valid_counts[eligible]
        if loss_method not in {"mae", "huber"}:
            scores[eligible] = np.sqrt(scores[eligible])

        eligible_indices = np.flatnonzero(eligible)
        order = np.lexsort((eligible_indices, scores[eligible_indices]))
        top_k = max(0, int(self.config.algorithm.top_k))
        best_indices = eligible_indices[order[:top_k]]
        candidates = [
            Candidate(
                row=float(anchor_rows[index]),
                col=float(anchor_cols[index]),
                heading_deg=float(heading_deg),
                estimated_msl_m=altitude_msl,
                score=float(scores[index]),
                valid_ratio=float(valid_counts[index]) / len(laser_agl),
                metrics={},
            )
            for index in best_indices
        ]
        return self._hydrate_candidates(
            candidates,
            laser_agl,
            laser_valid,
            baro_msl,
            base_offsets,
            pixel_offset_cache,
        )

    def exhaustive_search(
        self,
        laser_agl: np.ndarray,
        laser_valid: np.ndarray,
        baro_msl: np.ndarray,
        base_offsets: List[Tuple[float, float, float]],
        search_headings: List[float],
        stride: int = 1,
        search_bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Candidate]:
        """Search all valid pixels in DEM."""
        rows, cols = self.dem.shape
        stride = max(1, int(stride))
        top_k = max(0, int(self.config.algorithm.top_k))
        if top_k == 0:
            return []
        candidate_heap: List[Tuple[float, int, Candidate]] = []
        pixel_offset_cache: Dict[float, np.ndarray] = {}
        sequence = 0

        if search_bounds is None:
            row_start, row_end, col_start, col_end = 0, rows, 0, cols
        else:
            row_start, row_end, col_start, col_end = search_bounds
            row_start = max(0, min(rows, int(row_start)))
            row_end = max(row_start, min(rows, int(row_end)))
            col_start = max(0, min(cols, int(col_start)))
            col_end = max(col_start, min(cols, int(col_end)))

        # Keep the same global stride lattice inside an ROI. A shifted local
        # grid can otherwise miss the global coarse winner and reduce accuracy.
        row_start = ((row_start + stride - 1) // stride) * stride
        col_start = ((col_start + stride - 1) // stride) * stride

        for r in range(row_start, row_end, stride):
            for c in range(col_start, col_end, stride):
                for h in search_headings:
                    cand = evaluate_candidate(
                        r,
                        c,
                        h,
                        self.dem,
                        laser_agl,
                        laser_valid,
                        baro_msl,
                        base_offsets,
                        self.ct,
                        self.config,
                        pixel_offsets=self._pixel_offsets_for_heading(
                            pixel_offset_cache, base_offsets, h
                        ),
                        compute_all_metrics=False,
                    )
                    if cand is not None:
                        self._retain_top_candidate(candidate_heap, cand, sequence, top_k)
                        sequence += 1

        candidates = self._sorted_heap_candidates(candidate_heap)
        return self._hydrate_candidates(
            candidates,
            laser_agl,
            laser_valid,
            baro_msl,
            base_offsets,
            pixel_offset_cache,
        )

    def coarse_to_fine_search(
        self,
        laser_agl: np.ndarray,
        laser_valid: np.ndarray,
        baro_msl: np.ndarray,
        base_offsets: List[Tuple[float, float, float]],
        search_headings: List[float],
        search_bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Candidate]:
        # 1. Coarse search
        if self.config.sensor.altitude_mode == "known_msl_altitude" and len(search_headings) == 1:
            coarse_cands = self._vectorized_known_altitude_search(
                laser_agl,
                laser_valid,
                baro_msl,
                base_offsets,
                search_headings[0],
                self.config.algorithm.coarse_stride,
                search_bounds,
            )
        else:
            coarse_cands = self.exhaustive_search(
                laser_agl,
                laser_valid,
                baro_msl,
                base_offsets,
                search_headings,
                stride=self.config.algorithm.coarse_stride,
                search_bounds=search_bounds,
            )
        if not coarse_cands:
            return []

        # 2. Medium search around top K coarse
        top_k = max(0, int(self.config.algorithm.top_k))
        medium_heap: List[Tuple[float, int, Candidate]] = []
        medium_sequence = 0
        pixel_offset_cache: Dict[float, np.ndarray] = {}
        radius = self.config.algorithm.refinement_radius_px
        m_stride = self.config.algorithm.medium_stride
        rows, cols = self.dem.shape
        bounds = search_bounds or (0, rows, 0, cols)
        for c in coarse_cands:
            # We search around this c.row, c.col
            r_start = max(bounds[0], 0, int(c.row - radius))
            r_end = min(bounds[1], rows, int(c.row + radius + 1))
            c_start = max(bounds[2], 0, int(c.col - radius))
            c_end = min(bounds[3], cols, int(c.col + radius + 1))

            for r in range(r_start, r_end, m_stride):
                for col in range(c_start, c_end, m_stride):
                    cand = evaluate_candidate(
                        r,
                        col,
                        c.heading_deg,
                        self.dem,
                        laser_agl,
                        laser_valid,
                        baro_msl,
                        base_offsets,
                        self.ct,
                        self.config,
                        pixel_offsets=self._pixel_offsets_for_heading(
                            pixel_offset_cache, base_offsets, c.heading_deg
                        ),
                        compute_all_metrics=False,
                    )
                    if cand is not None:
                        self._retain_top_candidate(medium_heap, cand, medium_sequence, top_k)
                        medium_sequence += 1

        medium_cands = self._sorted_heap_candidates(medium_heap)
        if not medium_cands:
            return []

        # 3. Fine search around top K medium
        fine_candidates: Dict[Tuple[int, int, float], Candidate] = {}
        f_stride = self.config.algorithm.fine_stride
        f_radius = radius // 2
        for c in medium_cands:
            r_start = max(bounds[0], 0, int(c.row - f_radius))
            r_end = min(bounds[1], rows, int(c.row + f_radius + 1))
            c_start = max(bounds[2], 0, int(c.col - f_radius))
            c_end = min(bounds[3], cols, int(c.col + f_radius + 1))

            # Sub-degree headings?
            fine_headings = [c.heading_deg]
            if len(search_headings) > 1:  # implies we are searching heading
                step = self.config.algorithm.fine_heading_step_deg
                fine_headings = [c.heading_deg - step, c.heading_deg, c.heading_deg + step]

            for r in range(r_start, r_end, f_stride):
                for col in range(c_start, c_end, f_stride):
                    for h in fine_headings:
                        cand = evaluate_candidate(
                            r,
                            col,
                            h,
                            self.dem,
                            laser_agl,
                            laser_valid,
                            baro_msl,
                            base_offsets,
                            self.ct,
                            self.config,
                            pixel_offsets=self._pixel_offsets_for_heading(
                                pixel_offset_cache, base_offsets, h
                            ),
                            compute_all_metrics=False,
                        )
                        if cand is not None:
                            key = (round(cand.row), round(cand.col), round(cand.heading_deg, 1))
                            previous = fine_candidates.get(key)
                            if previous is None or cand.score < previous.score:
                                fine_candidates[key] = cand

        unique_cands = sorted(fine_candidates.values(), key=lambda candidate: candidate.score)
        unique_cands = unique_cands[:top_k]
        return self._hydrate_candidates(
            unique_cands,
            laser_agl,
            laser_valid,
            baro_msl,
            base_offsets,
            pixel_offset_cache,
        )
