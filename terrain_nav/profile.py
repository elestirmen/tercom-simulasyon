"""Profile extraction from DEM using bilinear interpolation."""

import math
from typing import List, Sequence, Tuple

import numpy as np

from terrain_nav.coordinates import CoordinateTransform, normalize_heading
from terrain_nav.sensors import Measurement


def build_offsets_for_candidate_speed(
    measurements: Sequence[Measurement],
    candidate_speed_m_s: float,
    ct: CoordinateTransform,
) -> List[Tuple[float, float, float]]:
    """Build a turn-aware relative route using timestamps and one speed hypothesis."""
    speed = float(candidate_speed_m_s)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("candidate_speed_m_s must be a finite positive value")
    if not measurements:
        return []

    offsets: List[Tuple[float, float, float]] = []
    current_x = 0.0
    current_y = 0.0
    previous_timestamp = float(measurements[0].timestamp_s)
    if not math.isfinite(previous_timestamp):
        raise ValueError("Measurement timestamps must be finite")

    for index, measurement in enumerate(measurements):
        timestamp = float(measurement.timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("Measurement timestamps must be finite")
        if index > 0:
            dt_s = timestamp - previous_timestamp
            if dt_s <= 0.0:
                raise ValueError("Measurement timestamps must be strictly increasing")
            distance_m = speed * dt_s
            dx, dy = ct.offset_meters(distance_m, measurement.sensor_heading_deg)
            current_x += dx
            current_y += dy
        offsets.append((current_x, current_y, measurement.sensor_heading_deg))
        previous_timestamp = timestamp
    return offsets


def build_distances_for_candidate_speed(
    measurements: Sequence[Measurement],
    candidate_speed_m_s: float,
) -> List[float]:
    """Return cumulative profile distances implied by timestamps and candidate speed."""
    if not measurements:
        return []
    start_timestamp = float(measurements[0].timestamp_s)
    return [
        max(0.0, (float(measurement.timestamp_s) - start_timestamp) * candidate_speed_m_s)
        for measurement in measurements
    ]


def rotate_offsets(
    offsets: List[Tuple[float, float, float]], angle_deg: float
) -> List[Tuple[float, float, float]]:
    """
    Rotate relative offsets (dx, dy) by an angle (e.g. for unknown heading).
    angle_deg is the candidate rotation.
    """
    if math.isclose(angle_deg, 0.0):
        return offsets

    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    rotated = []
    for dx, dy, h in offsets:
        # dx_new = dx*cos(a) - dy*sin(a)
        # dy_new = dx*sin(a) + dy*cos(a)
        # Wait, the spec says:
        # \Delta x_i'= \Delta x_i\cos\alpha - \Delta y_i\sin\alpha
        # \Delta y_i'= \Delta x_i\sin\alpha + \Delta y_i\cos\alpha
        # Let's apply this:
        dx_new = dx * cos_a - dy * sin_a
        dy_new = dx * sin_a + dy * cos_a
        rotated.append((dx_new, dy_new, normalize_heading(h + angle_deg)))
    return rotated


def extract_profile(
    dem: np.ndarray,
    start_row: float,
    start_col: float,
    offsets: List[Tuple[float, float, float]],
    ct: CoordinateTransform,
) -> np.ndarray:
    """
    Extract DEM profile along offsets from (start_row, start_col).
    Returns numpy array of elevations. np.nan for out-of-bounds.
    """
    pixel_offsets = [(-dy / ct.dy, dx / ct.dx) for dx, dy, _ in offsets]
    return extract_profile_pixels(dem, start_row, start_col, pixel_offsets)


def extract_profile_pixels(
    dem: np.ndarray,
    start_row: float,
    start_col: float,
    pixel_offsets: List[Tuple[float, float]] | np.ndarray,
) -> np.ndarray:
    """Extract a profile using precomputed ``(d_row, d_col)`` offsets."""
    rows, cols = dem.shape
    profile = np.empty(len(pixel_offsets), dtype=np.float64)

    # NumPy setup costs more for very short profiles, while vectorized raster
    # sampling is substantially faster once the online profile grows.
    if len(pixel_offsets) >= 24:
        offsets = np.asarray(pixel_offsets, dtype=np.float64)
        target_rows = start_row + offsets[:, 0]
        target_cols = start_col + offsets[:, 1]
        row_indices = np.floor(target_rows).astype(np.intp)
        col_indices = np.floor(target_cols).astype(np.intp)
        valid = (
            (row_indices >= 0)
            & (row_indices < rows - 1)
            & (col_indices >= 0)
            & (col_indices < cols - 1)
        )
        profile.fill(np.nan)
        if not np.any(valid):
            return profile

        valid_rows = row_indices[valid]
        valid_cols = col_indices[valid]
        row_fraction = target_rows[valid] - valid_rows
        col_fraction = target_cols[valid] - valid_cols
        top = (
            dem[valid_rows, valid_cols] * (1.0 - col_fraction)
            + dem[valid_rows, valid_cols + 1] * col_fraction
        )
        bottom = (
            dem[valid_rows + 1, valid_cols] * (1.0 - col_fraction)
            + dem[valid_rows + 1, valid_cols + 1] * col_fraction
        )
        profile[valid] = top * (1.0 - row_fraction) + bottom * row_fraction
        return profile

    for i, (d_row, d_col) in enumerate(pixel_offsets):
        target_r = start_row + d_row
        target_c = start_col + d_col

        # Bilinear interpolation
        r_int = int(math.floor(target_r))
        c_int = int(math.floor(target_c))

        if r_int < 0 or r_int >= rows - 1 or c_int < 0 or c_int >= cols - 1:
            profile[i] = np.nan
            continue

        r_frac = target_r - r_int
        c_frac = target_c - c_int

        v00 = dem[r_int, c_int]
        v10 = dem[r_int, c_int + 1]
        v01 = dem[r_int + 1, c_int]
        v11 = dem[r_int + 1, c_int + 1]

        v0 = v00 * (1 - c_frac) + v10 * c_frac
        v1 = v01 * (1 - c_frac) + v11 * c_frac
        val = v0 * (1 - r_frac) + v1 * r_frac

        profile[i] = val

    return profile
